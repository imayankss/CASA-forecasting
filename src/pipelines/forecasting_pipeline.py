"""
forecasting_pipeline.py — Master automated forecasting & evaluation pipeline.

Key changes from v1
-------------------
* ARIMA uses order (1,1,0) — fewer parameters, better for small samples.
* SARIMA uses (1,1,1)(0,1,1,4) — reduced seasonal params.
* SARIMAX uses REAL exogenous variables from macro data
  ['RBI_Repo_Rate', 'CPI_Inflation', 'GDP_Growth_Rate'] — NOT a time index.
* Holt-Winters (additive trend + additive seasonal, period=4) added as
  "HoltWinters" in FITTERS dict.
* TimeSeriesSplit(n_splits=5) replaces single train/test split for CV.
* CV reports mean ± std of MAPE across all 5 folds.
* run_cv flag uses proper sklearn TimeSeriesSplit with min_train_size=16.
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import (
    DATA_FILE, FORECAST_HORIZON, TRAIN_FRACTION,
    PROCESSED_DATA_DIR, REPORTS_DIR
)
from src.data_loader import load_casa_data, get_train_test, scale_exog_train_test
from src.evaluation.metrics import compute_full_metrics
from src.evaluation.diagnostics import run_full_diagnostics
from src.evaluation.comparison import ModelComparison
from src.forecasting.registry import ModelRegistry, ModelRecord
from src.utils import get_logger

warnings.filterwarnings("ignore")
logger = get_logger("pipeline")

# Default exogenous columns for SARIMAX
SARIMAX_EXOG_COLS = ["RBI_Repo_Rate", "CPI_Inflation", "GDP_Growth_Rate"]


# ─────────────────────────────────────────────────────────────
# INDIVIDUAL MODEL FITTERS
# ─────────────────────────────────────────────────────────────

def _fit_arima(train: pd.Series, test: pd.Series, df: Optional[pd.DataFrame] = None) -> Dict:
    """ARIMA(1,1,0) — parsimonious, fewer params than (1,1,1) on 30 points."""
    from statsmodels.tsa.arima.model import ARIMA
    train_log = np.log(train.replace(0, np.nan)).dropna()
    m = ARIMA(train_log, order=(1, 1, 0)).fit()
    fitted = np.exp(m.fittedvalues)
    fc_obj = m.get_forecast(steps=len(test))
    fc = np.exp(fc_obj.predicted_mean)
    fc.index = test.index
    ci_log = fc_obj.conf_int()
    ci_df = pd.DataFrame(
        {"lower": np.exp(ci_log.iloc[:, 0].values),
         "upper": np.exp(ci_log.iloc[:, 1].values)},
        index=test.index,
    )
    res = (train - fitted.reindex(train.index)).dropna().values
    return dict(
        forecast=fc, ci=ci_df, fitted=fitted, residuals=res,
        metrics=compute_full_metrics(test.values, fc.values, res),
        params={"order": "(1,1,0)"},
    )


def _fit_sarima(train: pd.Series, test: pd.Series, df: Optional[pd.DataFrame] = None) -> Dict:
    """SARIMA(1,1,1)(0,1,1,4) — reduced seasonal params to avoid overfitting."""
    import statsmodels.api as sm
    train_log = np.log(train.replace(0, np.nan)).dropna()
    m = sm.tsa.SARIMAX(
        train_log,
        order=(1, 1, 1),
        seasonal_order=(0, 1, 1, 4),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    fitted = np.exp(m.fittedvalues)
    fc_obj = m.get_forecast(steps=len(test))
    fc = np.exp(fc_obj.predicted_mean)
    fc.index = test.index
    ci_log = fc_obj.conf_int()
    ci_df = pd.DataFrame(
        {"lower": np.exp(ci_log.iloc[:, 0].values),
         "upper": np.exp(ci_log.iloc[:, 1].values)},
        index=test.index,
    )
    res = (train - fitted.reindex(train.index)).dropna().values
    return dict(
        forecast=fc, ci=ci_df, fitted=fitted, residuals=res,
        metrics=compute_full_metrics(test.values, fc.values, res),
        params={"order": "(1,1,1)", "seasonal_order": "(0,1,1,4)"},
    )


def _fit_sarimax(train: pd.Series, test: pd.Series, df: Optional[pd.DataFrame] = None) -> Dict:
    """SARIMAX(1,1,1)(0,1,1,4) with REAL macro exogenous variables.

    Exog: RBI_Repo_Rate, CPI_Inflation, GDP_Growth_Rate.
    Scaler is fit on train portion only (no data leakage).
    Falls back to SARIMA (no exog) if macro data is unavailable.
    """
    import statsmodels.api as sm

    train_log = np.log(train.replace(0, np.nan)).dropna()

    # ── Prepare real exogenous variables ──────────────────────────────────
    exog_train = None
    exog_test  = None
    exog_feature_count = 0

    if df is not None:
        train_exog, test_exog, _ = scale_exog_train_test(
            df=df,
            train_index=train.index,
            test_index=test.index,
            exog_cols=SARIMAX_EXOG_COLS,
        )
        if not train_exog.empty and not test_exog.empty:
            exog_feature_count = len(train_exog.columns)
            exog_train = train_exog.values
            exog_test = test_exog.values

    m = sm.tsa.SARIMAX(
        train_log,
        exog=exog_train,
        order=(1, 1, 1),
        seasonal_order=(0, 1, 1, 4),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    fitted = np.exp(m.fittedvalues)
    fc_obj = m.get_forecast(steps=len(test), exog=exog_test)
    fc = np.exp(fc_obj.predicted_mean)
    fc.index = test.index
    ci_log = fc_obj.conf_int()
    ci_df = pd.DataFrame(
        {"lower": np.exp(ci_log.iloc[:, 0].values),
         "upper": np.exp(ci_log.iloc[:, 1].values)},
        index=test.index,
    )
    res = (train - fitted.reindex(train.index)).dropna().values
    exog_label = SARIMAX_EXOG_COLS if exog_train is not None else "none"
    return dict(
        forecast=fc, ci=ci_df, fitted=fitted, residuals=res,
        metrics=compute_full_metrics(test.values, fc.values, res, n_features=exog_feature_count),
        params={"order": "(1,1,1)", "seasonal_order": "(0,1,1,4)", "exog": str(exog_label)},
    )


def _fit_auto_arima(train: pd.Series, test: pd.Series, df: Optional[pd.DataFrame] = None) -> Dict:
    import pmdarima as pm
    train_log = np.log(train.replace(0, np.nan)).dropna()
    model = pm.auto_arima(
        train_log, seasonal=True, m=4,
        max_p=3, max_q=3, stepwise=True,
        trace=False, error_action="ignore",
        suppress_warnings=True,
    )
    fitted_log = model.predict_in_sample()
    fitted = np.exp(pd.Series(fitted_log, index=train_log.index))
    fc_log, ci_log = model.predict(n_periods=len(test), return_conf_int=True)
    fc = pd.Series(np.exp(fc_log), index=test.index, name="AutoARIMA")
    ci_df = pd.DataFrame(
        {"lower": np.exp(ci_log[:, 0]), "upper": np.exp(ci_log[:, 1])},
        index=test.index,
    )
    res = (train - fitted.reindex(train.index)).dropna().values
    return dict(
        forecast=fc, ci=ci_df, fitted=fitted, residuals=res,
        metrics=compute_full_metrics(test.values, fc.values, res),
        params={"order": str(model.order), "seasonal_order": str(model.seasonal_order)},
    )


def _fit_holt_winters(train: pd.Series, test: pd.Series, df: Optional[pd.DataFrame] = None) -> Dict:
    """Holt-Winters with additive trend and additive seasonal (period=4)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=4,
        damped_trend=False,
        initialization_method="estimated",
    ).fit(optimized=True)

    fitted = model.fittedvalues
    fc = model.forecast(steps=len(test))
    fc.index = test.index
    fc.name = "HoltWinters"

    # Holt-Winters in statsmodels doesn't expose standard CIs natively;
    # build approximate 95% CI from RMSE on training residuals
    res = (train - fitted).values
    rmse_train = float(np.sqrt(np.mean(res**2)))
    ci_df = pd.DataFrame(
        {"lower": fc.values - 1.96 * rmse_train,
         "upper": fc.values + 1.96 * rmse_train},
        index=test.index,
    )

    return dict(
        forecast=fc, ci=ci_df, fitted=fitted, residuals=res,
        metrics=compute_full_metrics(test.values, fc.values, res),
        params={"trend": "add", "seasonal": "add", "seasonal_periods": 4},
    )


def _fit_prophet(train: pd.Series, test: pd.Series, df: Optional[pd.DataFrame] = None) -> Dict:
    from prophet import Prophet
    train_df = train.reset_index()
    train_df.columns = ["ds", "y"]
    m = Prophet(seasonality_mode="multiplicative", yearly_seasonality=True)
    m.fit(train_df)
    future_df = pd.DataFrame({"ds": test.index})
    fc_df = m.predict(future_df)
    fc = pd.Series(fc_df["yhat"].values, index=test.index, name="Prophet")
    ci_df = pd.DataFrame(
        {"lower": fc_df["yhat_lower"].values, "upper": fc_df["yhat_upper"].values},
        index=test.index,
    )
    train_pred = m.predict(train_df[["ds"]])
    fitted = pd.Series(train_pred["yhat"].values, index=train.index)
    res = (train - fitted).values
    return dict(
        forecast=fc, ci=ci_df, fitted=fitted, residuals=res,
        metrics=compute_full_metrics(test.values, fc.values, res),
        params={"seasonality_mode": "multiplicative", "yearly_seasonality": True},
    )


FITTERS = {
    "ARIMA":       _fit_arima,
    "SARIMA":      _fit_sarima,
    "SARIMAX":     _fit_sarimax,
    "AutoARIMA":   _fit_auto_arima,
    "HoltWinters": _fit_holt_winters,
    "Prophet":     _fit_prophet,
}


# ─────────────────────────────────────────────────────────────
# TimeSeriesSplit CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────

def timeseries_cv(
    series: pd.Series,
    fitter_fn,
    n_splits: int = 5,
    min_train_size: int = 16,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Proper walk-forward CV using sklearn TimeSeriesSplit.

    Parameters
    ----------
    series         : full target series (pre-final-test portion recommended)
    fitter_fn      : one of the FITTERS callables
    n_splits       : number of folds (default 5)
    min_train_size : minimum quarters in training window
    df             : full enriched DataFrame (passed to SARIMAX for real exog)

    Returns
    -------
    DataFrame of per-fold metrics.  Each row is one fold.
    Columns include MAPE, RMSE_%, R2, Direction_Accuracy, etc.

    Guarantee: all train indices < test indices (no leakage).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(series)):
        if len(train_idx) < min_train_size:
            logger.debug(f"  Fold {fold_idx + 1}: train too small ({len(train_idx)}), skipping.")
            continue

        # Leakage guard: all train indices strictly less than all test indices
        assert train_idx.max() < test_idx.min(), "Data leakage detected in TimeSeriesSplit!"

        fold_train = series.iloc[train_idx]
        fold_test  = series.iloc[test_idx]

        try:
            out = fitter_fn(fold_train, fold_test, df)
            row = {"fold": fold_idx + 1, **out["metrics"]}
            results.append(row)
        except Exception as e:
            logger.warning(f"  CV fold {fold_idx + 1} failed: {e}")

    return pd.DataFrame(results) if results else pd.DataFrame()


# Legacy alias kept for backward-compat with existing tests
def walk_forward_cv(
    series: pd.Series,
    fitter_fn,
    n_splits: int = 5,
    test_size: int = 4,
    min_train: int = 16,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Backward-compatible wrapper — delegates to timeseries_cv."""
    return timeseries_cv(
        series=series,
        fitter_fn=fitter_fn,
        n_splits=n_splits,
        min_train_size=min_train,
        df=df,
    )


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

class ForecastingPipeline:
    """
    End-to-end automated forecasting pipeline.

    Steps
    -----
    1. Load and validate data (with macro merge)
    2. Train/test split
    3. Fit all models
    4. Evaluate with full metrics (R², Adjusted R², Direction Accuracy)
    5. Run residual diagnostics
    6. TimeSeriesSplit CV (n_splits=5)
    7. Build comparison + leaderboard
    8. Generate insights
    9. Export CSV reports
    """

    def __init__(
        self,
        data_path: str = DATA_FILE,
        train_frac: float = TRAIN_FRACTION,
        models: Optional[List[str]] = None,
        run_cv: bool = True,
        cv_splits: int = 5,
    ) -> None:
        self.data_path   = data_path
        self.train_frac  = train_frac
        self.model_names = models or list(FITTERS.keys())
        self.run_cv      = run_cv
        self.cv_splits   = cv_splits

        self.df: Optional[pd.DataFrame]  = None
        self.scaler                       = None
        self.train: Optional[pd.Series]  = None
        self.test: Optional[pd.Series]   = None

        self.results:     Dict = {}
        self.cv_results:  Dict = {}
        self.diagnostics: Dict = {}

        self.comparison = ModelComparison()
        self.registry   = ModelRegistry()
        self._ran       = False

    # ── Step 1: Load ──────────────────────────────────────────────────────

    def load(self) -> "ForecastingPipeline":
        logger.info(f"Loading data: {self.data_path}")
        result = load_casa_data(self.data_path, fit_scaler=False, return_scaler=False)
        # Keep compatibility with older loader returns while defaulting to raw features.
        if isinstance(result, tuple):
            self.df, self.scaler = result
        else:
            self.df = result
            self.scaler = None
        self.train, self.test = get_train_test(self.df, train_frac=self.train_frac)
        logger.info(f"Train: {len(self.train)} | Test: {len(self.test)}")
        return self

    # ── Step 2-4: Train + Evaluate ────────────────────────────────────────

    def _train_model(self, name: str) -> None:
        fitter = FITTERS.get(name)
        if fitter is None:
            logger.warning(f"No fitter registered for '{name}' — skipping.")
            return
        logger.info(f"  Training {name} …")
        t0 = time.time()
        try:
            # Pass df so SARIMAX can access real exog columns
            out = fitter(self.train, self.test, self.df)
            out["train_time"] = time.time() - t0
            self.results[name] = out

            self.comparison.add(
                model_name=name,
                actual=self.test.values,
                predicted=out["forecast"].values,
                residuals=out["residuals"],
                train_time=out["train_time"],
            )

            self.registry.register(ModelRecord(
                name=name,
                params=out["params"],
                metrics=out["metrics"],
                forecast=out["forecast"],
                ci=out["ci"],
                residuals=out["residuals"],
                fitted=out["fitted"],
                train_time=out["train_time"],
            ))

            mape = out["metrics"]["MAPE"]
            r2   = out["metrics"]["R2"]
            logger.info(f"  {name} done — MAPE={mape:.2f}%  R²={r2:.4f}  ({out['train_time']:.1f}s)")
        except Exception as e:
            logger.error(f"  {name} FAILED: {e}")

    # ── Step 5: Diagnostics ───────────────────────────────────────────────

    def _run_diagnostics(self) -> None:
        for name, rec in self.results.items():
            logger.info(f"  Diagnostics: {name} …")
            diag = run_full_diagnostics(
                residuals=rec["residuals"],
                series=self.train,
                model_name=name,
            )
            self.diagnostics[name] = diag

    # ── Step 6: TimeSeriesSplit CV ────────────────────────────────────────

    def _run_cv(self) -> None:
        series = self.train
        for name in self.model_names:
            fitter = FITTERS.get(name)
            if fitter is None:
                continue
            logger.info(f"  TimeSeriesSplit CV ({self.cv_splits} folds): {name} …")
            cv = timeseries_cv(
                series=series,
                fitter_fn=fitter,
                n_splits=self.cv_splits,
                min_train_size=16,
                df=self.df,
            )
            self.cv_results[name] = cv
            if not cv.empty:
                mean_mape = cv["MAPE"].mean()
                std_mape  = cv["MAPE"].std()
                logger.info(f"  {name} CV MAPE: {mean_mape:.2f} ± {std_mape:.2f}%")

    # ── Master run ────────────────────────────────────────────────────────

    def run(self) -> "ForecastingPipeline":
        if self.df is None:
            self.load()

        logger.info("\n▶ Training models …")
        for name in self.model_names:
            self._train_model(name)

        logger.info("\n▶ Running diagnostics …")
        self._run_diagnostics()

        if self.run_cv:
            logger.info("\n▶ Walk-forward cross-validation (TimeSeriesSplit) …")
            self._run_cv()

        self._ran = True
        logger.info("\n✅ Pipeline complete.")
        return self

    # ── Results access ────────────────────────────────────────────────────

    def leaderboard(self) -> pd.DataFrame:
        return self.comparison.leaderboard()

    def insights(self) -> Dict:
        return self.comparison.generate_insights()

    def cv_summary(self) -> pd.DataFrame:
        """Return CV summary with mean ± std of MAPE across all folds."""
        rows = []
        for name, cv_df in self.cv_results.items():
            if cv_df.empty:
                continue
            mean = cv_df.mean(numeric_only=True)
            std  = cv_df.std(numeric_only=True)
            rows.append({
                "Model":             name,
                "CV_MAPE_Mean":      round(mean.get("MAPE", np.nan), 3),
                "CV_MAPE_Std":       round(std.get("MAPE", np.nan), 3),
                "CV_RMSE_%_Mean":    round(mean.get("RMSE_%", np.nan), 3),
                "CV_R2_Mean":        round(mean.get("R2", np.nan), 4),
                "CV_Dir_Acc_Mean":   round(mean.get("Direction_Accuracy", np.nan), 2),
                "CV_Folds":          len(cv_df),
            })
        return pd.DataFrame(rows).set_index("Model") if rows else pd.DataFrame()

    # ── Export ────────────────────────────────────────────────────────────

    def save_results(self, out_dir: Optional[str] = None) -> Dict[str, str]:
        """Export all results CSVs and the registry JSON."""
        from pathlib import Path as _Path
        out = _Path(out_dir or PROCESSED_DATA_DIR)
        out.mkdir(parents=True, exist_ok=True)
        saved = {}

        lb_path = str(out / "model_leaderboard.csv")
        self.leaderboard().to_csv(lb_path)
        saved["leaderboard"] = lb_path

        m_path = str(out / "model_metrics.csv")
        self.comparison.metrics_df().to_csv(m_path)
        saved["metrics"] = m_path

        fc_rows = {}
        for name, rec in self.results.items():
            fc_rows[name] = rec["forecast"]
        if fc_rows:
            fc_df = pd.DataFrame(fc_rows)
            fc_df["Actual_Test"] = self.test.values[:len(fc_df)]
            fc_path = str(out / "forecast_results.csv")
            fc_df.to_csv(fc_path)
            saved["forecasts"] = fc_path

        cv_df = self.cv_summary()
        if not cv_df.empty:
            cv_path = str(out / "cv_summary.csv")
            cv_df.to_csv(cv_path)
            saved["cv_summary"] = cv_path

        reg_path = str(_Path(REPORTS_DIR) / "model_registry.json")
        self.registry.save_index(reg_path)
        saved["registry"] = reg_path

        for label, path in saved.items():
            logger.info(f"  ✓ {label:12s} → {path}")

        return saved
