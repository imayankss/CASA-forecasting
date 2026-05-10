"""
metrics.py — Comprehensive evaluation metrics for time-series forecasting.

New in this version
-------------------
* compute_full_metrics: uses sklearn r2_score with aligned arrays; adds
  adjusted_r2 and direction_accuracy.
* adjusted_r2(r2, n_samples, n_features) — standalone helper.
* direction_accuracy(actual, predicted) — % correct up/down calls.
"""

import warnings
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────────────────────

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error — skips zero actuals."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean forecast bias (positive = overforecast)."""
    return float(np.mean(np.array(y_pred, dtype=float) - np.array(y_true, dtype=float)))


def forecast_bias_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Forecast bias as % of mean actual."""
    mean_actual = np.mean(np.array(y_true, dtype=float))
    if mean_actual == 0:
        return np.nan
    return float(forecast_bias(y_true, y_pred) / mean_actual * 100)


def stability_score(residuals: np.ndarray) -> float:
    """Residual stability score (0–100, higher = more stable)."""
    res = np.array(residuals, dtype=float)
    std = np.std(res)
    mean_abs = np.mean(np.abs(res)) + 1e-9
    cv = std / mean_abs
    score = max(0.0, 100.0 - cv * 50)
    return round(min(score, 100.0), 2)


def theil_u(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Theil's U statistic — measures forecast quality vs naive."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    naive  = y_true[:-1]
    actual = y_true[1:]
    pred   = y_pred[1:]
    num = np.sqrt(np.mean((actual - pred) ** 2))
    den = np.sqrt(np.mean((actual - naive) ** 2))
    if den == 0:
        return np.nan
    return float(num / den)


# ─────────────────────────────────────────────────────────────
# NEW HELPERS
# ─────────────────────────────────────────────────────────────

def adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
    """
    Adjusted R² — penalises for adding non-informative predictors.

    Parameters
    ----------
    r2         : raw R² value
    n_samples  : number of observations
    n_features : number of predictors (excluding intercept)

    Returns
    -------
    Adjusted R² (float). Returns np.nan when denominator ≤ 0.
    """
    if np.isnan(r2) or n_samples <= n_features + 1:
        return np.nan
    denom = n_samples - n_features - 1
    if denom <= 0:
        return np.nan
    return float(1 - (1 - r2) * (n_samples - 1) / denom)


def direction_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Directional accuracy — percentage of quarters where the forecast
    correctly predicted the sign (up / down / flat) of the period-over-period
    change.

    A "correct" call means sign(Δactual) == sign(Δpredicted).
    Flat (zero) changes on either side are excluded from the denominator.

    Returns
    -------
    Percentage in [0, 100], or np.nan when fewer than 2 points.
    """
    actual    = np.array(actual,    dtype=float)
    predicted = np.array(predicted, dtype=float)
    if len(actual) < 2:
        return np.nan

    delta_actual = np.diff(actual)
    delta_pred   = np.diff(predicted)

    # Exclude steps where either series is flat (ambiguous direction)
    mask = (delta_actual != 0) & (delta_pred != 0)
    if mask.sum() == 0:
        return np.nan

    correct = np.sign(delta_actual[mask]) == np.sign(delta_pred[mask])
    return float(correct.mean() * 100)


# ─────────────────────────────────────────────────────────────
# FULL METRICS SUITE
# ─────────────────────────────────────────────────────────────

def compute_full_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    residuals: Optional[np.ndarray] = None,
    n_features: int = 1,
) -> Dict[str, float]:
    """
    Compute the full suite of evaluation metrics.

    Parameters
    ----------
    actual     : array of true values
    predicted  : array of forecast values (must be same length as actual)
    residuals  : optional pre-computed residuals; computed as actual-predicted if None
    n_features : number of model parameters (for adjusted R²); default 1

    Returns
    -------
    Flat dict with all metrics including R2, Adjusted_R2, Direction_Accuracy.
    """
    actual    = np.array(actual,    dtype=float).ravel()
    predicted = np.array(predicted, dtype=float).ravel()

    # Align lengths — trim to shorter if mismatch (safety guard)
    min_len = min(len(actual), len(predicted))
    actual    = actual[:min_len]
    predicted = predicted[:min_len]

    mean_val = float(np.nanmean(actual)) or 1.0

    if residuals is None:
        residuals = actual - predicted
    residuals = np.array(residuals, dtype=float)

    mae  = float(mean_absolute_error(actual, predicted))
    mse  = float(mean_squared_error(actual, predicted))
    rmse = float(np.sqrt(mse))

    # ── R² using sklearn with aligned arrays ─────────────────────────────────
    try:
        r2 = float(r2_score(actual, predicted))
    except Exception:
        r2 = np.nan

    adj_r2 = adjusted_r2(r2, n_samples=len(actual), n_features=n_features)
    dir_acc = direction_accuracy(actual, predicted)

    return {
        # Absolute
        "MAE":               round(mae, 4),
        "RMSE":              round(rmse, 4),
        "MSE":               round(mse, 4),
        # Percentage
        "MAE_%":             round(mae  / mean_val * 100, 3),
        "RMSE_%":            round(rmse / mean_val * 100, 3),
        "MSE_%":             round(mse  / mean_val**2 * 100, 4),
        "MAPE":              round(safe_mape(actual, predicted), 3),
        "SMAPE":             round(smape(actual, predicted), 3),
        # Fit quality
        "R2":                round(r2, 4),
        "Adjusted_R2":       round(adj_r2, 4) if not np.isnan(adj_r2) else float("nan"),
        "Direction_Accuracy": round(dir_acc, 2) if not np.isnan(dir_acc) else float("nan"),
        "Bias_%":            round(forecast_bias_pct(actual, predicted), 3),
        "Stability":         stability_score(residuals),
        "Theil_U":           round(theil_u(actual, predicted), 4) if len(actual) > 2 else float("nan"),
    }


# ─────────────────────────────────────────────────────────────
# WEIGHTED COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────

SCORE_WEIGHTS: Dict[str, float] = {
    "MAPE":      0.35,
    "RMSE_%":    0.25,
    "MAE_%":     0.20,
    "Stability": 0.20,
}


def composite_score(metrics: Dict[str, float], all_metrics: pd.DataFrame) -> float:
    """
    Weighted composite score (0–100). Higher = better.
    Normalises each metric relative to all models before weighting.
    """
    score = 0.0
    for col, weight in SCORE_WEIGHTS.items():
        if col not in metrics or col not in all_metrics.columns:
            continue
        col_vals = all_metrics[col].dropna()
        if col_vals.empty:
            continue
        val = metrics[col]
        mn, mx = col_vals.min(), col_vals.max()
        if mx == mn:
            normalised = 1.0
        elif col == "Stability":
            normalised = (val - mn) / (mx - mn)       # higher is better
        else:
            normalised = 1 - (val - mn) / (mx - mn)   # lower is better
        score += weight * normalised * 100
    return round(score, 2)
