"""
run_visualizations.py
─────────────────────
Master script that:
  1. Loads the BOI CASA deposit data
  2. Fits ARIMA, SARIMA, SARIMAX, AutoARIMA, and Prophet models
  3. Collects forecasts, CIs, residuals, and metrics
  4. Calls generate_all_visualizations() to produce every chart

Run:
    python run_visualizations.py
or:
    python run_visualizations.py --horizon 6 --data data/raw/boi_casa_deposits.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config      import DATA_FILE, FORECAST_HORIZON, TRAIN_FRACTION, VISUALIZATIONS_DIR
from src.data_loader import load_casa_data, get_train_test
from src.utils       import compute_metrics, get_logger, train_test_split_ts
from visualizations.plots import generate_all_visualizations

logger = get_logger("run_visualizations")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _make_residuals(actual: pd.Series, fitted: pd.Series) -> np.ndarray:
    common = actual.index.intersection(fitted.index)
    return (actual.loc[common] - fitted.loc[common]).values


def _build_metrics_df(results: dict) -> pd.DataFrame:
    rows = {}
    for model, info in results.items():
        m = info["metrics"]
        rows[model] = {
            "MAE (%)":  round(m["MAE_pct"],  2),
            "RMSE (%)": round(m["RMSE_pct"], 2),
            "MAPE (%)": round(m["MAPE"],     2),
            "MSE (%)":  round(m["MSE_pct"],  2),
        }
    return pd.DataFrame(rows).T


# ─────────────────────────────────────────────
# FIT HELPERS (graceful fall-through on error)
# ─────────────────────────────────────────────

def _fit_arima(train, test):
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    train_log = np.log(train.replace(0, np.nan)).dropna()
    m = _ARIMA(train_log, order=(1, 1, 1)).fit()
    fitted     = np.exp(m.fittedvalues)
    fc_obj     = m.get_forecast(steps=len(test))
    fc_log     = fc_obj.predicted_mean
    ci_log     = fc_obj.conf_int()
    fc         = np.exp(fc_log)
    ci_lower   = np.exp(ci_log.iloc[:, 0])
    ci_upper   = np.exp(ci_log.iloc[:, 1])
    fc.index   = ci_lower.index = ci_upper.index = test.index
    ci_df      = pd.DataFrame({"lower": ci_lower, "upper": ci_upper})
    residuals  = _make_residuals(train, fitted)
    metrics    = compute_metrics(test.values, fc.values)
    return dict(forecast=fc, ci=ci_df, fitted=fitted, residuals=residuals, metrics=metrics)


def _fit_sarima(train, test):
    import statsmodels.api as sm
    train_log = np.log(train.replace(0, np.nan)).dropna()
    m = sm.tsa.SARIMAX(train_log, order=(1, 1, 1),
                       seasonal_order=(1, 1, 1, 4),
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False)
    fitted    = np.exp(m.fittedvalues)
    fc_obj    = m.get_forecast(steps=len(test))
    fc        = np.exp(fc_obj.predicted_mean);  fc.index = test.index
    ci_log    = fc_obj.conf_int()
    ci_df     = pd.DataFrame({
        "lower": np.exp(ci_log.iloc[:, 0].values),
        "upper": np.exp(ci_log.iloc[:, 1].values),
    }, index=test.index)
    residuals = _make_residuals(train, fitted)
    metrics   = compute_metrics(test.values, fc.values)
    return dict(forecast=fc, ci=ci_df, fitted=fitted, residuals=residuals, metrics=metrics)


def _fit_sarimax(train, test):
    import statsmodels.api as sm
    train_log = np.log(train.replace(0, np.nan)).dropna()
    # Use time index as a simple exogenous proxy
    exog_train = np.arange(len(train_log)).reshape(-1, 1)
    exog_test  = np.arange(len(train_log), len(train_log) + len(test)).reshape(-1, 1)
    m = sm.tsa.SARIMAX(train_log, exog=exog_train, order=(1, 1, 1),
                       seasonal_order=(1, 1, 1, 4),
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False)
    fitted    = np.exp(m.fittedvalues)
    fc_obj    = m.get_forecast(steps=len(test), exog=exog_test)
    fc        = np.exp(fc_obj.predicted_mean);  fc.index = test.index
    ci_log    = fc_obj.conf_int()
    ci_df     = pd.DataFrame({
        "lower": np.exp(ci_log.iloc[:, 0].values),
        "upper": np.exp(ci_log.iloc[:, 1].values),
    }, index=test.index)
    residuals = _make_residuals(train, fitted)
    metrics   = compute_metrics(test.values, fc.values)
    return dict(forecast=fc, ci=ci_df, fitted=fitted, residuals=residuals, metrics=metrics)


def _fit_auto_arima(train, test):
    import pmdarima as pm
    train_log = np.log(train.replace(0, np.nan)).dropna()
    model     = pm.auto_arima(train_log, seasonal=True, m=4,
                               max_p=4, max_q=4, stepwise=True,
                               trace=False, error_action="ignore",
                               suppress_warnings=True)
    fitted_log = model.predict_in_sample()
    fitted     = np.exp(pd.Series(fitted_log, index=train_log.index))
    fc_log, ci_log = model.predict(n_periods=len(test), return_conf_int=True)
    fc         = pd.Series(np.exp(fc_log), index=test.index, name="AutoARIMA")
    ci_df      = pd.DataFrame({
        "lower": np.exp(ci_log[:, 0]),
        "upper": np.exp(ci_log[:, 1]),
    }, index=test.index)
    residuals  = _make_residuals(train, fitted)
    metrics    = compute_metrics(test.values, fc.values)
    return dict(forecast=fc, ci=ci_df, fitted=fitted, residuals=residuals, metrics=metrics)


def _fit_prophet(train, test):
    from prophet import Prophet
    train_df = train.reset_index()
    train_df.columns = ["ds", "y"]
    m = Prophet(seasonality_mode="multiplicative", yearly_seasonality=True)
    m.fit(train_df)
    future_df = pd.DataFrame({"ds": test.index})
    fc_df     = m.predict(future_df)
    fc        = pd.Series(fc_df["yhat"].values, index=test.index, name="Prophet")
    ci_df     = pd.DataFrame({
        "lower": fc_df["yhat_lower"].values,
        "upper": fc_df["yhat_upper"].values,
    }, index=test.index)
    train_pred = m.predict(train_df[["ds"]])
    fitted     = pd.Series(train_pred["yhat"].values, index=train.index)
    residuals  = _make_residuals(train, fitted)
    metrics    = compute_metrics(test.values, fc.values)
    return dict(forecast=fc, ci=ci_df, fitted=fitted, residuals=residuals, metrics=metrics)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(data_path: str = DATA_FILE, horizon: int = FORECAST_HORIZON) -> None:
    logger.info(f"Loading data from {data_path}")
    df = load_casa_data(data_path)

    train, test = get_train_test(df, train_frac=TRAIN_FRACTION)
    logger.info(f"Train: {len(train)} quarters  |  Test: {len(test)} quarters")

    # ── Fit models ──────────────────────────────────────────────────────
    model_fitters = {
        "ARIMA":     _fit_arima,
        "SARIMA":    _fit_sarima,
        "SARIMAX":   _fit_sarimax,
        "AutoARIMA": _fit_auto_arima,
        "Prophet":   _fit_prophet,
    }

    results = {}
    for name, fitter in model_fitters.items():
        logger.info(f"  Fitting {name} …")
        try:
            results[name] = fitter(train, test)
            logger.info(f"  {name} MAPE = {results[name]['metrics']['MAPE']:.2f}%")
        except Exception as e:
            logger.warning(f"  {name} failed: {e}")

    # ── Assemble visualisation inputs ────────────────────────────────────
    forecasts     = {k: v["forecast"]  for k, v in results.items()}
    ci_map        = {k: v["ci"]        for k, v in results.items()}
    residuals_map = {k: v["residuals"] for k, v in results.items()}
    metrics_df    = _build_metrics_df(results)

    logger.info("\nMetrics Summary:\n" + metrics_df.to_string())

    # ── Save metrics CSV ─────────────────────────────────────────────────
    metrics_path = ROOT / "data" / "processed" / "model_comparison_metrics.csv"
    metrics_df.to_csv(metrics_path)
    logger.info(f"Metrics saved → {metrics_path}")

    # ── Generate all visualisations ──────────────────────────────────────
    out_dir = ROOT / "visualizations" / "output"
    generate_all_visualizations(
        df=df,
        actual_train=train,
        actual_test=test,
        forecasts=forecasts,
        ci_map=ci_map,
        residuals_map=residuals_map,
        metrics_df=metrics_df,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BOI CASA Forecasting — Visualisations")
    parser.add_argument("--data",    default=DATA_FILE,        help="Path to CSV data file")
    parser.add_argument("--horizon", default=FORECAST_HORIZON, type=int,
                        help="Number of future quarters to forecast")
    args = parser.parse_args()
    main(data_path=args.data, horizon=args.horizon)
