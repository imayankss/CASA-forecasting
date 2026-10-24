"""
run_propensity.py — CASA-acquisition lead scoring on the Santander dataset.

Target: does the customer add a CASA-type account next month (see
src/propensity/santander.py). Split by month, never at random:
train on feature months 2015-06..2016-02, validate on 2016-03, and test on
2016-04 (labels from May 2016).

Usage
-----
    python run_propensity.py            # ingest + features if missing, then train and evaluate
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.evaluation.lift import bootstrap_lift_ci, lift_at_k
from src.evaluation.temporal import assert_temporal_split
from src.forecasting.registry import run_metadata
from src.propensity.santander import (
    FEATURES_PARQUET, TRAIN_PARQUET, build_features, ingest, load,
)
from src.utils import get_logger

logger = get_logger("run_propensity")

OUT_DIR = ROOT / "reports" / "propensity"
TRAIN_MONTHS = ("2015-06-28", "2016-02-28")
VAL_MONTH = "2016-03-28"
TEST_MONTH = "2016-04-28"
NEG_SAMPLE = 0.10          # keep 10% of training negatives, reweighted by 1/0.10
SEED = 42
DROP = ["ncodpers", "fecha_dato", "casa_add", "label_month"]
LGB_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, seed=SEED, num_threads=6,
)


def _split(feats: pl.LazyFrame, lo: str, hi: str) -> pd.DataFrame:
    d = pl.col("fecha_dato")
    return feats.filter((d >= pl.lit(lo).str.to_date()) & (d <= pl.lit(hi).str.to_date())).collect().to_pandas()


def _train_frame(feats: pl.LazyFrame) -> pd.DataFrame:
    d = pl.col("fecha_dato")
    in_window = (d >= pl.lit(TRAIN_MONTHS[0]).str.to_date()) & (d <= pl.lit(TRAIN_MONTHS[1]).str.to_date())
    keep = (pl.col("casa_add") == 1) | (pl.int_range(pl.len()).shuffle(seed=SEED) < pl.len() * NEG_SAMPLE)
    return feats.filter(in_window).filter(keep).collect().to_pandas()


def _metrics(y: np.ndarray, score: np.ndarray, prob: np.ndarray = None) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    lo, hi = bootstrap_lift_ci(y, score, k=0.1, n_boot=1000, seed=SEED)
    out = {
        "lift_at_10pct": round(lift_at_k(y, score, 0.1), 3),
        "lift_at_10pct_ci95": [round(lo, 3), round(hi, 3)],
        "lift_at_5pct": round(lift_at_k(y, score, 0.05), 3),
        "lift_at_1pct": round(lift_at_k(y, score, 0.01), 3),
        "roc_auc": round(roc_auc_score(y, score), 4),
        "pr_auc": round(average_precision_score(y, score), 4),
    }
    if prob is not None:
        out["brier"] = round(brier_score_loss(y, prob), 5)
    return out


def _gains(y: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-score, kind="stable")
    deciles = np.array_split(y[order], 10)
    cum = np.cumsum([d.sum() for d in deciles]) / y.sum()
    return pd.DataFrame({
        "decile": range(1, 11),
        "customers": [len(d) for d in deciles],
        "adds": [int(d.sum()) for d in deciles],
        "hit_rate": [d.mean() for d in deciles],
        "lift": [d.mean() / y.mean() for d in deciles],
        "cumulative_share_of_adds": cum,
    })


def main() -> None:
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if not TRAIN_PARQUET.exists():
        logger.info("Ingesting raw CSV …")
        ingest()
    if not FEATURES_PARQUET.exists():
        logger.info("Building features …")
        build_features(load())

    feats = pl.scan_parquet(FEATURES_PARQUET)
    train = _train_frame(feats)
    val = _split(feats, VAL_MONTH, VAL_MONTH)
    test = _split(feats, TEST_MONTH, TEST_MONTH)
    for a, b in [(train, val), (val, test), (train, test)]:
        assert_temporal_split(pd.Index(a["fecha_dato"].unique()), pd.Index(b["fecha_dato"].unique()))
    logger.info(f"train {len(train):,} rows (sampled), val {len(val):,}, test {len(test):,}")

    features = [c for c in train.columns if c not in DROP]
    w_train = np.where(train["casa_add"] == 1, 1.0, 1.0 / NEG_SAMPLE)
    y_test = test["casa_add"].to_numpy()
    results = {"base_rate_test": round(float(y_test.mean()), 5), "test_customers": int(len(test)),
               "test_adds": int(y_test.sum()), "models": {}}

    # Baseline 1: a rule a CRM team could write without a model
    rule = test["ind_actividad_cliente"].fillna(0).to_numpy() * 100 + test["n_products"].to_numpy()
    rule = rule + 1e-6 * np.random.default_rng(SEED).random(len(rule))   # break ties randomly
    results["models"]["rule_active_x_products"] = _metrics(y_test, rule)

    # Baseline 2: logistic regression on numeric features
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    lr.fit(train[numeric].fillna(0), train["casa_add"], logisticregression__sample_weight=w_train)
    p_lr = lr.predict_proba(test[numeric].fillna(0))[:, 1]
    results["models"]["logistic_regression"] = _metrics(y_test, p_lr, p_lr)

    # LightGBM
    dtrain = lgb.Dataset(train[features], train["casa_add"], weight=w_train)
    dval = lgb.Dataset(val[features], val["casa_add"], reference=dtrain)
    model = lgb.train(LGB_PARAMS | {"metric": "auc"}, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    p_lgb = model.predict(test[features], num_iteration=model.best_iteration)
    results["models"]["lightgbm"] = _metrics(y_test, p_lgb, p_lgb) | {"best_iteration": model.best_iteration}

    # Stricter view: exclude customers who held a CASA product in the last 5 months
    # but not now. Their "adds" are mostly flag flicker / re-activations, which are
    # easy to predict and worth little; the rest are genuinely new acquisitions.
    from src.propensity.santander import CASA_PRODUCTS, CASA_LAGS
    readd = np.zeros(len(test), dtype=bool)
    for c in CASA_PRODUCTS:
        for k in [1, *CASA_LAGS]:
            readd |= (test[f"{c}_l{k}"].fillna(0).to_numpy() == 1) & (test[c].to_numpy() == 0)
    new = ~readd
    results["new_adds_only"] = {
        "customers": int(new.sum()), "adds": int(y_test[new].sum()),
        "base_rate": round(float(y_test[new].mean()), 5),
        "share_of_adds_excluded": round(float(y_test[readd].sum() / y_test.sum()), 4),
        "models": {
            "rule_active_x_products": _metrics(y_test[new], rule[new]),
            "logistic_regression": _metrics(y_test[new], p_lr[new], p_lr[new]),
            "lightgbm": _metrics(y_test[new], p_lgb[new], p_lgb[new]),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _gains(y_test[new], p_lgb[new]).round(4).to_csv(OUT_DIR / "gains_lightgbm_new_adds.csv", index=False)
    _gains(y_test, p_lgb).round(4).to_csv(OUT_DIR / "gains_lightgbm.csv", index=False)
    seg = test.assign(score=p_lgb).groupby("segmento", observed=True)
    pd.DataFrame({
        "customers": seg.size(),
        "base_rate": seg["casa_add"].mean(),
        "lift_at_10pct": seg.apply(lambda g: lift_at_k(g["casa_add"], g["score"], 0.1)),
    }).round(4).to_csv(OUT_DIR / "lift_by_segment.csv")
    pd.Series(model.feature_importance("gain"), index=features).sort_values(ascending=False) \
        .round(1).to_csv(OUT_DIR / "feature_importance_gain.csv", header=["gain"])

    results["run"] = run_metadata(
        config={"train_months": TRAIN_MONTHS, "val_month": VAL_MONTH, "test_month": TEST_MONTH,
                "neg_sample": NEG_SAMPLE, "lgb_params": LGB_PARAMS, "n_features": len(features)},
        data_files=[str(FEATURES_PARQUET.relative_to(ROOT))],
    )
    (OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    logger.info(json.dumps({k: results[k] for k in ("models", "new_adds_only")}, indent=2))


if __name__ == "__main__":
    main()
