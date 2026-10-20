"""
run_panel.py — Deposit forecasting across thousands of US banks (FDIC Call Reports).

Target: CASA analog y = TRN + NTRSMMDA + NTRSOTH ($ thousands), forecast on log y.
Backtest: re-forecast from every quarter-end 2021Q4..2023Q4 (9 origins),
1-4 quarters ahead. Only banks reporting at the origin with >= MIN_HISTORY
quarters of history are scored. Metric: MASE (season length 4) per bank and origin.

Usage
-----
    python run_panel.py                      # baselines + global LightGBM
    python run_panel.py --models lgbm        # one family only
    python run_panel.py --chronos-only       # Chronos-2 zero-shot, then merged with the saved main run
    python run_panel.py --transfer [--chronos-only]   # BOI transfer (Task B.7)

Chronos runs in its own process: torch and LightGBM each load an OpenMP
runtime, and using both in one process deadlocked on macOS.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.evaluation.temporal import assert_temporal_split
from src.forecasting.registry import run_metadata
from src.utils import get_logger

logger = get_logger("run_panel")

PANEL = ROOT / "data" / "external" / "fdic" / "panel.parquet"
OUT_DIR = ROOT / "reports" / "panel"
ERRORS = {tag: ROOT / "data" / "external" / "fdic" / f"backtest_errors_{tag}.parquet"
          for tag in ("main", "chronos")}
KEYS = ["unique_id", "ds", "h", "origin", "actual", "scale"]
ORIGINS = pd.date_range("2021-12-31", "2023-12-31", freq="QE")
# Earlier origins used only to choose model settings, never for reported results
VALIDATION_ORIGINS = pd.to_datetime(["2019-12-31", "2020-12-31"])
H = 4
SEASON = 4
MIN_HISTORY = 16
SEED = 42


def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(PANEL)[["unique_id", "ds", "y"]]
    df["ds"] = pd.to_datetime(df["ds"])
    df = df[df["y"] > 0].sort_values(["unique_id", "ds"]).reset_index(drop=True)
    df["y"] = np.log(df["y"])
    return df


def split_at(df: pd.DataFrame, origin: pd.Timestamp):
    """Training data up to the origin for banks active at the origin; actuals for the next H quarters."""
    train = df[df["ds"] <= origin]
    stats = train.groupby("unique_id")["ds"].agg(["max", "size"])
    active = stats[(stats["max"] == origin) & (stats["size"] >= MIN_HISTORY)].index
    train = train[train["unique_id"].isin(active)]
    horizon_end = origin + pd.offsets.QuarterEnd(H)
    test = df[(df["ds"] > origin) & (df["ds"] <= horizon_end) & df["unique_id"].isin(active)]
    assert_temporal_split(pd.Index(train["ds"].unique()), pd.Index(test["ds"].unique()))
    return train, test


def mase_scale(train: pd.DataFrame) -> pd.Series:
    """In-sample seasonal-naive MAE per bank, on levels."""
    lv = train.assign(level=np.exp(train["y"]))
    diff = lv.groupby("unique_id")["level"].diff(SEASON).abs()
    return diff.groupby(lv["unique_id"]).mean()


def fit_stats(train: pd.DataFrame, families) -> pd.DataFrame:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, AutoTheta, Naive, SeasonalNaive
    models = [SeasonalNaive(season_length=SEASON), Naive()]
    if "ets" in families:
        models.append(AutoETS(season_length=SEASON))
    if "theta" in families:
        models.append(AutoTheta(season_length=SEASON))
    sf = StatsForecast(models=models, freq="QE", n_jobs=6)
    return sf.forecast(df=train, h=H)


def fit_lgbm(train: pd.DataFrame, objective: str = "l2") -> pd.DataFrame:
    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean
    from mlforecast.target_transforms import Differences
    mlf = MLForecast(
        models={"GlobalLGBM": lgb.LGBMRegressor(
            objective=objective, n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
            n_jobs=6, verbose=-1)},
        freq="QE",
        lags=list(range(1, 9)),
        lag_transforms={1: [RollingMean(window_size=4), RollingMean(window_size=8)]},
        date_features=["quarter"],
        target_transforms=[Differences([1])],
    )
    mlf.fit(train, static_features=[])
    return mlf.predict(H)


def aggregate_check(errors: pd.DataFrame) -> dict:
    """Sum of bank forecasts vs sum of actuals per target quarter (h=1): absolute % error."""
    one = errors[errors["h"] == 1]
    preds = [c for c in one.columns if c.startswith("pred_")]
    agg = one.groupby("ds")[["actual", *preds]].sum()
    ape = agg[preds].sub(agg["actual"], axis=0).abs().div(agg["actual"], axis=0) * 100
    return {c[5:]: round(float(v), 3) for c, v in ape.mean().items()}


BOI = ROOT / "data" / "real" / "boi_casa_quarterly.csv"
BOI_ORIGIN = pd.Timestamp("2023-12-31")


def transfer_to_boi(df: pd.DataFrame, chronos: bool) -> pd.DataFrame:
    """Train on the FDIC panel up to the BOI holdout origin, forecast BOI's real CASA series.

    The model works on log differences, so the currency and scale difference
    (US $ thousands vs Rs crore) does not matter.
    """
    boi = pd.read_csv(BOI, parse_dates=["quarter_end"])
    boi = pd.DataFrame({"unique_id": "BOI", "ds": boi["quarter_end"], "y": np.log(boi["casa_cr"])})
    hist, test = boi[boi["ds"] <= BOI_ORIGIN], boi[boi["ds"] > BOI_ORIGIN]
    assert_temporal_split(pd.Index(hist["ds"]), pd.Index(test["ds"]))
    train = df[df["ds"] <= BOI_ORIGIN]
    h = len(test)
    if chronos:
        fc = fit_chronos(hist, h)
        out = test.merge(fc, on=["unique_id", "ds"])
        return pd.DataFrame({"ds": out["ds"], "actual": np.exp(out["y"]), "Chronos2": np.exp(out["Chronos2"])})
    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean
    from mlforecast.target_transforms import Differences
    mlf = MLForecast(
        models={"GlobalLGBM_FDIC": lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
            n_jobs=6, verbose=-1)},
        freq="QE", lags=list(range(1, 9)),
        lag_transforms={1: [RollingMean(window_size=4), RollingMean(window_size=8)]},
        date_features=["quarter"], target_transforms=[Differences([1])],
    )
    mlf.fit(train, static_features=[])
    fc = mlf.predict(h, new_df=hist)
    out = test.merge(fc, on=["unique_id", "ds"])
    res = pd.DataFrame({"ds": out["ds"], "actual": np.exp(out["y"])})
    for c in [c for c in fc.columns if c not in ("unique_id", "ds")]:
        res[c] = np.exp(out[c])
    return res


def fit_chronos(train: pd.DataFrame, h: int = H) -> pd.DataFrame:
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    pred = pipe.predict_df(train, prediction_length=h, quantile_levels=[0.5],
                           id_column="unique_id", timestamp_column="ds", target="y")
    return pred.rename(columns={"predictions": "Chronos2"})[["unique_id", "ds", "Chronos2"]]


def score(fc: pd.DataFrame, test: pd.DataFrame, scale: pd.Series, origin) -> pd.DataFrame:
    models = [c for c in fc.columns if c not in ("unique_id", "ds")]
    m = test.merge(fc, on=["unique_id", "ds"], how="inner")
    m["h"] = ((m["ds"].dt.year - origin.year) * 4 + (m["ds"].dt.quarter - origin.quarter))
    actual = np.exp(m["y"])
    rows = m[["unique_id", "ds", "h"]].copy()
    rows["origin"] = origin
    rows["actual"] = actual
    for c in models:
        rows[f"pred_{c}"] = np.exp(m[c])
        rows[f"ae_{c}"] = (actual - rows[f"pred_{c}"]).abs()
    rows["scale"] = rows["unique_id"].map(scale)
    return rows


def summarise(errors: pd.DataFrame) -> pd.DataFrame:
    models = [c[3:] for c in errors.columns if c.startswith("ae_")]
    per = errors.groupby(["unique_id", "origin"]).agg(
        **{m: (f"ae_{m}", "mean") for m in models}, scale=("scale", "first"))
    per = per[per["scale"] > 0]
    mase = per[models].div(per["scale"], axis=0)
    return pd.DataFrame({
        "MASE_mean": mase.mean(), "MASE_median": mase.median(),
        "share_beats_SeasonalNaive": (mase.lt(mase["SeasonalNaive"], axis=0)).mean(),
        "series_origins": len(mase),
    }).round(4).sort_values("MASE_mean"), mase


def paired_bootstrap(mase: pd.DataFrame, a: str, b: str, n_boot: int = 2000) -> dict:
    """95% CI of the mean MASE difference (a - b), resampling banks."""
    diff = (mase[a] - mase[b]).groupby(level="unique_id").mean()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff.to_numpy()[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"a": a, "b": b, "mean_diff": round(float(diff.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)], "banks": int(len(diff))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ets", "theta", "lgbm"])
    ap.add_argument("--chronos-only", action="store_true")
    ap.add_argument("--origins", type=int, default=len(ORIGINS))
    ap.add_argument("--validation", action="store_true", help="score on validation origins only")
    ap.add_argument("--objective", default="l2")
    ap.add_argument("--chronos-sample", type=int, default=0,
                    help="score Chronos on a random sample of N banks per origin (cut rule)")
    ap.add_argument("--transfer", action="store_true", help="only run the BOI transfer (Task B.7)")
    args = ap.parse_args()

    df = load_panel()
    if args.transfer:
        res = transfer_to_boi(df, chronos=args.chronos_only)
        path = OUT_DIR / "boi_transfer_forecasts.csv"
        if path.exists():
            prev = pd.read_csv(path, parse_dates=["ds"])
            res = prev.drop(columns=[c for c in res.columns if c in prev.columns and c not in ("ds", "actual")]) \
                      .merge(res, on=["ds", "actual"], how="outer")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        res.round(1).to_csv(path, index=False)
        logger.info("\n" + res.round(0).to_string())
        return
    logger.info(f"panel: {len(df):,} rows, {df['unique_id'].nunique():,} banks")
    all_errors, timings = [], {}
    origins = VALIDATION_ORIGINS if args.validation else ORIGINS[: args.origins]
    for origin in origins:
        train, test = split_at(df, origin)
        if args.chronos_only:
            fc = None
        else:
            t0 = time.time()
            fc = fit_stats(train, args.models)
            timings.setdefault("stats", []).append(time.time() - t0)
        if "lgbm" in args.models and not args.chronos_only:
            t0 = time.time()
            fc = fc.merge(fit_lgbm(train, args.objective), on=["unique_id", "ds"])
            timings.setdefault("lgbm", []).append(time.time() - t0)
        if args.chronos_only:
            t0 = time.time()
            ctrain = train
            if args.chronos_sample:
                ids = train["unique_id"].drop_duplicates().sample(args.chronos_sample, random_state=SEED)
                ctrain = train[train["unique_id"].isin(ids)]
            fc = fit_chronos(ctrain)
            timings.setdefault("chronos", []).append(time.time() - t0)
        errs = score(fc, test, mase_scale(train), origin)
        all_errors.append(errs)
        logger.info(f"origin {origin.date()}: {train['unique_id'].nunique():,} banks, "
                    + ", ".join(f"{k} {v[-1]:.0f}s" for k, v in timings.items()))

    errors = pd.concat(all_errors, ignore_index=True)
    tag = "chronos" if args.chronos_only else "main"
    if not args.validation:
        errors.to_parquet(ERRORS[tag])
    if args.validation:
        pass
    elif ERRORS["main"].exists() and ERRORS["chronos"].exists():
        main_e = pd.read_parquet(ERRORS["main"])
        chron = pd.read_parquet(ERRORS["chronos"])
        errors = main_e.merge(chron.drop(columns=["actual", "scale"]), on=["unique_id", "ds", "h", "origin"])
        logger.info(f"merged with saved {'main' if args.chronos_only else 'chronos'} run: {len(errors):,} rows")
    elif args.chronos_only:
        logger.info("no saved main run to merge with")
    table, mase = summarise(errors)
    logger.info("\n" + table.to_string())

    # 2023Q1 deposit outflows: one-step-ahead errors for that quarter vs all other quarters
    one = errors[errors["h"] == 1]
    stress = {}
    for c in [c[3:] for c in errors.columns if c.startswith("ae_")]:
        r = (one[f"ae_{c}"] / one["scale"]).replace([np.inf], np.nan)
        stress[c] = {"MASE_h1_2023Q1": round(float(r[one["ds"] == "2023-03-31"].mean()), 4),
                     "MASE_h1_other": round(float(r[one["ds"] != "2023-03-31"].mean()), 4)}

    if args.validation:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "leaderboard.csv")
    mase.groupby(level="origin").mean().round(4).to_csv(OUT_DIR / "mase_by_origin.csv")
    tests = [paired_bootstrap(mase, m, "AutoETS") for m in mase.columns if m != "AutoETS"] \
        if "AutoETS" in mase.columns else []
    summary = {
        "panel_rows": int(len(df)), "banks": int(df["unique_id"].nunique()),
        "origins": [str(o.date()) for o in ORIGINS[: args.origins]], "horizon": H,
        "leaderboard": table.reset_index(names="model").to_dict("records"),
        "paired_bootstrap_vs_AutoETS": tests, "stress_2023Q1": stress,
        "aggregate_ape_pct_h1": aggregate_check(errors),
        "chronos_sample_banks_per_origin": args.chronos_sample or None,
        "seconds_per_origin": {k: round(float(np.mean(v)), 1) for k, v in timings.items()},
        "run": run_metadata({"origins": len(ORIGINS[: args.origins]), "h": H, "models": args.models,
                             "chronos": ERRORS["chronos"].exists(), "min_history": MIN_HISTORY,
                             "lgbm_objective": args.objective},
                            [str(PANEL.relative_to(ROOT))]),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
