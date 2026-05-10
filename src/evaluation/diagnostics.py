"""
diagnostics.py — Statistical diagnostics engine for residual analysis.
Covers: ADF, Shapiro-Wilk, Jarque-Bera, Ljung-Box, ARCH test, ACF/PACF.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# INDIVIDUAL TESTS
# ─────────────────────────────────────────────────────────────

def run_adf_test(series: pd.Series, name: str = "series") -> Dict:
    """Augmented Dickey-Fuller stationarity test."""
    from statsmodels.tsa.stattools import adfuller
    s = series.dropna()
    result = adfuller(s)
    stationary = result[1] <= 0.05
    return {
        "test":        "ADF",
        "series":      name,
        "statistic":   round(result[0], 4),
        "p_value":     round(result[1], 4),
        "critical_1%": round(result[4]["1%"], 4),
        "critical_5%": round(result[4]["5%"], 4),
        "lags_used":   result[2],
        "nobs":        result[3],
        "stationary":  bool(stationary),
        "verdict":     "Stationary ✓" if stationary else "Non-stationary ✗",
        "interpretation": (
            "Series is stationary — suitable for ARMA/ARIMA modelling."
            if stationary else
            "Series is non-stationary — differencing or transformation required."
        ),
    }


def run_shapiro_test(residuals: np.ndarray) -> Dict:
    """Shapiro-Wilk normality test on residuals."""
    r = np.array(residuals, dtype=float)
    stat, p = stats.shapiro(r)
    normal = p > 0.05
    return {
        "test":        "Shapiro-Wilk",
        "statistic":   round(float(stat), 4),
        "p_value":     round(float(p), 4),
        "normal":      bool(normal),
        "verdict":     "Normal ✓" if normal else "Non-normal ✗",
        "skewness":    round(float(stats.skew(r)), 4),
        "kurtosis":    round(float(stats.kurtosis(r)), 4),
        "interpretation": (
            "Residuals are approximately normally distributed (good model fit)."
            if normal else
            "Residuals deviate from normality — check for outliers or model misspecification."
        ),
    }


def run_jarque_bera_test(residuals: np.ndarray) -> Dict:
    """Jarque-Bera normality test (skewness + kurtosis)."""
    r = np.array(residuals, dtype=float)
    stat, p = stats.jarque_bera(r)
    normal = p > 0.05
    return {
        "test":        "Jarque-Bera",
        "statistic":   round(float(stat), 4),
        "p_value":     round(float(p), 4),
        "normal":      bool(normal),
        "verdict":     "Normal ✓" if normal else "Non-normal ✗",
        "interpretation": (
            "Residuals pass JB normality test — skewness and kurtosis are acceptable."
            if normal else
            "Residuals fail JB normality test — skewness or excess kurtosis detected."
        ),
    }


def run_ljung_box_test(residuals: np.ndarray, lags: int = 10) -> Dict:
    """Ljung-Box test for residual autocorrelation."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    r = pd.Series(np.array(residuals, dtype=float))
    lb = acorr_ljungbox(r, lags=[lags], return_df=True)
    stat = float(lb["lb_stat"].iloc[-1])
    p    = float(lb["lb_pvalue"].iloc[-1])
    no_autocorr = p > 0.05
    return {
        "test":           "Ljung-Box",
        "lags":           lags,
        "statistic":      round(stat, 4),
        "p_value":        round(p, 4),
        "no_autocorr":    bool(no_autocorr),
        "verdict":        "No autocorrelation ✓" if no_autocorr else "Autocorrelation detected ✗",
        "interpretation": (
            "No significant residual autocorrelation — model captures the temporal structure well."
            if no_autocorr else
            f"Significant autocorrelation detected at lag {lags} — model may be under-specified."
        ),
    }


def run_arch_test(residuals: np.ndarray, lags: int = 4) -> Dict:
    """ARCH LM test for heteroscedasticity in residuals."""
    from statsmodels.stats.diagnostic import het_arch
    r = np.array(residuals, dtype=float)
    lm_stat, lm_p, f_stat, f_p = het_arch(r, nlags=lags)
    no_arch = lm_p > 0.05
    return {
        "test":        "ARCH-LM",
        "lags":        lags,
        "statistic":   round(float(lm_stat), 4),
        "p_value":     round(float(lm_p), 4),
        "no_arch":     bool(no_arch),
        "verdict":     "Homoscedastic ✓" if no_arch else "ARCH effects detected ✗",
        "interpretation": (
            "No ARCH effects — residual variance is stable over time."
            if no_arch else
            "ARCH effects present — residual variance is not constant (volatility clustering)."
        ),
    }


# ─────────────────────────────────────────────────────────────
# FULL DIAGNOSTIC SUITE
# ─────────────────────────────────────────────────────────────

def run_full_diagnostics(
    residuals:  np.ndarray,
    series:     Optional[pd.Series] = None,
    model_name: str = "Model",
) -> Dict:
    """
    Run all diagnostic tests for a model's residuals.

    Parameters
    ----------
    residuals   : in-sample residuals array
    series      : original series for ADF test (optional)
    model_name  : model label

    Returns
    -------
    nested dict with results per test + overall health score
    """
    r = np.array(residuals, dtype=float)
    r = r[~np.isnan(r)]

    results: Dict = {"model": model_name, "n_residuals": len(r)}

    results["shapiro"]     = run_shapiro_test(r)
    results["jarque_bera"] = run_jarque_bera_test(r)

    if len(r) > 10:
        results["ljung_box"] = run_ljung_box_test(r)
    if len(r) > 8:
        try:
            results["arch"] = run_arch_test(r)
        except Exception:
            pass

    if series is not None:
        results["adf"] = run_adf_test(series, name=model_name)

    # ── Summary statistics ───────────────────────────────────────────────
    results["stats"] = {
        "mean":     round(float(np.mean(r)), 4),
        "std":      round(float(np.std(r)),  4),
        "min":      round(float(np.min(r)),  4),
        "max":      round(float(np.max(r)),  4),
        "skewness": round(float(stats.skew(r)),     4),
        "kurtosis": round(float(stats.kurtosis(r)), 4),
        "q25":      round(float(np.percentile(r, 25)), 4),
        "q75":      round(float(np.percentile(r, 75)), 4),
    }

    # ── Health score (% of tests passed) ────────────────────────────────
    passed = sum([
        results.get("shapiro",     {}).get("normal",       False),
        results.get("jarque_bera", {}).get("normal",       False),
        results.get("ljung_box",   {}).get("no_autocorr",  False),
        results.get("arch",        {}).get("no_arch",      False),
    ])
    total = sum([
        "shapiro"     in results,
        "jarque_bera" in results,
        "ljung_box"   in results,
        "arch"        in results,
    ])
    results["health_score"] = round(passed / total * 100, 1) if total else 0.0
    results["health_label"] = (
        "Excellent" if results["health_score"] >= 75 else
        "Good"      if results["health_score"] >= 50 else
        "Fair"      if results["health_score"] >= 25 else
        "Poor"
    )

    return results


# ─────────────────────────────────────────────────────────────
# ACF / PACF VALUES
# ─────────────────────────────────────────────────────────────

def acf_pacf_values(
    residuals: np.ndarray,
    nlags: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (lags, acf_values, pacf_values) arrays for plotting.
    """
    from statsmodels.tsa.stattools import acf, pacf
    r    = np.array(residuals, dtype=float)
    lags = min(nlags, len(r) // 2 - 1)
    acf_vals  = acf(r,  nlags=lags, fft=True)
    pacf_vals = pacf(r, nlags=lags)
    lag_arr   = np.arange(len(acf_vals))
    return lag_arr, acf_vals, pacf_vals
