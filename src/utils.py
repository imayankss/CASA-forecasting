"""
utils.py — Shared utilities for BOI CASA Forecasting Platform.

Provides
--------
* get_logger()             — thin wrapper around src.logger.get_logger
* compute_metrics()        — thin wrapper around evaluation.metrics
* train_test_split_ts()    — chronological split helper
* adf_test()               — ADF stationarity test
* shapiro_test()           — Shapiro-Wilk test
* jarque_bera_test()       — Jarque-Bera test
* save_metrics_csv()       — persist metric dict to CSV
* format_inr()             — format float as Indian currency string
* retry()                  — decorator: retry on exception
* safe_execute()           — run a callable, catch & log all errors
* validate_series()        — guard-clause for Series inputs
* timer_context()          — context manager for elapsed-time logging
"""

from __future__ import annotations

import functools
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Internal imports (lazy to avoid circular imports) ─────────────────────────
from src.logger import get_logger as _get_logger      # noqa: E402
from src.exceptions import (                           # noqa: E402
    DataValidationError,
    InsufficientDataError,
)

# Re-export get_logger so callers can do: from src.utils import get_logger
get_logger = _get_logger

_log = _get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# SERIES VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def validate_series(
    series: pd.Series,
    name: str = "series",
    min_length: int = 8,
) -> pd.Series:
    """
    Guard clause: raise typed exceptions if *series* is unsuitable.

    Parameters
    ----------
    series     : the time series to validate
    name       : human-readable label for error messages
    min_length : minimum required non-NaN observations

    Returns
    -------
    pd.Series — cleaned (dropna) copy

    Raises
    ------
    DataValidationError   if the series is empty or too short
    InsufficientDataError if valid observations < min_length
    """
    if series is None or len(series) == 0:
        raise DataValidationError(f"'{name}' is empty", n_rows=0)

    clean = series.dropna()
    if len(clean) < min_length:
        raise InsufficientDataError(n_available=len(clean), n_required=min_length)

    return clean


# ═════════════════════════════════════════════════════════════════════════════
# METRICS  (thin wrapper — evaluation.metrics is the source of truth)
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, MAPE, MSE, and their % counterparts.

    Parameters
    ----------
    actual    : ground-truth values
    predicted : model-predicted values

    Returns
    -------
    dict with keys MAE, RMSE, MSE, MAPE, MAE_pct, RMSE_pct, MSE_pct
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    a = np.array(actual,    dtype=float)
    p = np.array(predicted, dtype=float)
    mean_val = float(np.mean(a)) or 1.0

    mae  = float(mean_absolute_error(a, p))
    mse  = float(mean_squared_error(a, p))
    rmse = float(np.sqrt(mse))

    mask = a != 0
    mape = float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.sum() else float("nan")

    return {
        "MAE":      round(mae,  4),
        "RMSE":     round(rmse, 4),
        "MSE":      round(mse,  4),
        "MAPE":     round(mape, 3),
        "MAE_pct":  round(mae  / mean_val * 100, 3),
        "RMSE_pct": round(rmse / mean_val * 100, 3),
        "MSE_pct":  round(mse  / mean_val ** 2 * 100, 4),
    }


# ═════════════════════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT
# ═════════════════════════════════════════════════════════════════════════════

def train_test_split_ts(
    series: pd.Series,
    train_frac: float = 0.80,
) -> Tuple[pd.Series, pd.Series]:
    """
    Chronological train / test split for a time series.

    Parameters
    ----------
    series     : time-indexed pd.Series
    train_frac : fraction for training (default 0.80)

    Returns
    -------
    (train, test)  — non-overlapping pd.Series
    """
    if not (0 < train_frac < 1):
        raise DataValidationError(
            f"train_frac must be in (0, 1). Got {train_frac}."
        )
    n     = len(series)
    split = int(n * train_frac)
    return series.iloc[:split].copy(), series.iloc[split:].copy()


# ═════════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═════════════════════════════════════════════════════════════════════════════

def adf_test(
    series: pd.Series,
    name: str = "series",
    logger=None,
) -> bool:
    """
    Augmented Dickey-Fuller stationarity test.

    Returns True if the series is stationary (p-value ≤ 0.05).
    """
    from statsmodels.tsa.stattools import adfuller

    log = logger or _log
    clean = series.dropna()
    result = adfuller(clean)
    stationary = bool(result[1] <= 0.05)
    log.debug(
        "ADF [%s] stat=%.4f  p=%.4f → %s",
        name, result[0], result[1],
        "Stationary" if stationary else "Non-stationary",
    )
    return stationary


def shapiro_test(
    residuals: np.ndarray,
    logger=None,
) -> Dict[str, Any]:
    """
    Shapiro-Wilk normality test.

    Returns dict with keys: statistic, p_value, normal (bool).
    """
    from scipy.stats import shapiro

    log  = logger or _log
    stat, p = shapiro(np.array(residuals, dtype=float))
    normal  = bool(p > 0.05)
    log.debug("Shapiro-Wilk  stat=%.4f  p=%.4f → %s",
              stat, p, "Normal" if normal else "Non-normal")
    return {"statistic": float(stat), "p_value": float(p), "normal": normal}


def jarque_bera_test(
    residuals: np.ndarray,
    logger=None,
) -> Dict[str, Any]:
    """
    Jarque-Bera normality test (skewness + kurtosis).

    Returns dict with keys: statistic, p_value, normal (bool).
    """
    from scipy.stats import jarque_bera

    log  = logger or _log
    stat, p = jarque_bera(np.array(residuals, dtype=float))
    normal  = bool(p > 0.05)
    log.debug("Jarque-Bera  stat=%.4f  p=%.4f → %s",
              stat, p, "Normal" if normal else "Non-normal")
    return {"statistic": float(stat), "p_value": float(p), "normal": normal}


# ═════════════════════════════════════════════════════════════════════════════
# I/O HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def save_metrics_csv(
    metrics_dict: Dict[str, Dict],
    path: Union[str, Path],
) -> None:
    """
    Persist a {model_name → metrics_dict} mapping as a CSV file.

    Parameters
    ----------
    metrics_dict : outer keys are model names; inner keys are metric names
    path         : destination file path (created if missing)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics_dict).T.round(4)
    df.to_csv(path)
    _log.info("Metrics saved → %s", path)


def format_inr(value: float) -> str:
    """
    Format a number as an Indian Rupee string with crore notation.

    Examples
    --------
    >>> format_inr(64500000)
    '₹6.45 Cr'
    >>> format_inr(-1000000)
    '−₹0.10 Cr'
    """
    sign = "−" if value < 0 else ""
    return f"{sign}₹{abs(value) / 1e7:.2f} Cr"


# ═════════════════════════════════════════════════════════════════════════════
# RESILIENCE UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,),
    logger=None,
) -> Callable:
    """
    Decorator: retry a function up to *max_attempts* times on failure.

    Parameters
    ----------
    max_attempts : maximum number of calls (including the first)
    delay        : seconds to wait between attempts
    exceptions   : exception types that trigger a retry
    logger       : optional logger; falls back to module-level _log

    Example
    -------
        @retry(max_attempts=3, delay=0.5, exceptions=(TimeoutError,))
        def fetch_data(url: str) -> dict: …
    """
    log = logger or _log

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        log.warning(
                            "retry %d/%d for %s — %s",
                            attempt, max_attempts, fn.__qualname__, exc,
                        )
                        time.sleep(delay)
                    else:
                        log.error(
                            "All %d attempts failed for %s — %s",
                            max_attempts, fn.__qualname__, exc,
                        )
            raise last_exc   # type: ignore[misc]
        return wrapper
    return decorator


def safe_execute(
    fn: Callable[..., Any],
    *args,
    fallback: Any = None,
    logger=None,
    label: str = "",
    **kwargs,
) -> Any:
    """
    Execute *fn(*args, **kwargs)* and return *fallback* on any exception.

    Logs the error at WARNING level so the pipeline continues gracefully.

    Parameters
    ----------
    fn       : callable to execute
    fallback : value to return on failure (default: None)
    logger   : optional logger
    label    : short description for log messages

    Example
    -------
        result = safe_execute(fit_arima, train, test, fallback={}, label="ARIMA")
    """
    log = logger or _log
    tag = label or getattr(fn, "__qualname__", str(fn))
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        log.warning("safe_execute [%s] caught %s: %s", tag, type(exc).__name__, exc)
        return fallback


# ═════════════════════════════════════════════════════════════════════════════
# CONTEXT MANAGER  — elapsed timing
# ═════════════════════════════════════════════════════════════════════════════

@contextmanager
def timer_context(label: str, logger=None) -> Generator[None, None, None]:
    """
    Context manager that logs elapsed time on exit.

    Example
    -------
        with timer_context("SARIMA training"):
            model.fit(train)
    """
    log = logger or _log
    t0  = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        log.debug("⏱  %s  →  %.3f s", label, elapsed)
