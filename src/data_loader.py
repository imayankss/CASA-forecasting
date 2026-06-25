"""
data_loader.py — Unified data loading, macro merging, and feature engineering
for BOI CASA deposits.

The default loader returns raw engineered features and does not fit a scaler.
Forecasting code should scale exogenous variables with training data only via
``get_train_test_with_exog`` or ``scale_exog_train_test``.
"""

import warnings
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Column lists ───────────────────────────────────────────────────────────────

MACRO_COLS = [
    "RBI_Repo_Rate",
    "CPI_Inflation",
    "GDP_Growth_Rate",
    "Nifty_Bank_Index",
    "Credit_Deposit_Ratio",
    "CASA_Ratio_Industry",
]

EXOG_COLS = [
    "RBI_Repo_Rate",
    "CPI_Inflation",
    "GDP_Growth_Rate",
    "Nifty_Bank_Index",
    "Credit_Deposit_Ratio",
    "CASA_Ratio_Industry",
    "Deposit_t1",
    "Deposit_t4",
    "Deposit_t8",
    "rate_x_growth",
    "rolling_vol_4q",
]

_SCALE_COLS = EXOG_COLS


# ── Internal helpers ───────────────────────────────────────────────────────────

def _resolve_macro_path(deposit_path: Path) -> Optional[Path]:
    candidates = [
        deposit_path.parent / "macro_indicators.csv",
        deposit_path.parents[1] / "raw" / "macro_indicators.csv",
        Path("data/raw/macro_indicators.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _parse_period_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Period"] = pd.PeriodIndex(df["Quarter_Year"], freq="Q").to_timestamp()
    df.set_index("Period", inplace=True)
    df.sort_index(inplace=True)
    return df


# ── Public API ─────────────────────────────────────────────────────────────────

def load_casa_data(
    filepath,
    macro_path=None,
    fit_scaler: bool = False,
    return_scaler: bool = False,
):
    """
    Load and enrich the BOI CASA deposit CSV.

    Parameters
    ----------
    filepath   : path to boi_casa_deposits.csv
    macro_path : explicit path to macro_indicators.csv (auto-resolved if None)
    fit_scaler : if True, fit a StandardScaler on engineered feature columns.
                 Keep this False for model training to avoid full-series leakage.
    return_scaler : if True, return (df, scaler). Defaults to False for
                    backward-compatible DataFrame-only callers.

    Returns
    -------
    df or (df, scaler)
        df     — enriched DataFrame with DatetimeIndex
        scaler — fitted StandardScaler when explicitly requested
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    df = _parse_period_index(df)
    df["Deposit_Amount"] = pd.to_numeric(df["Deposit_Amount"], errors="coerce")

    if "Log_Deposit" not in df.columns:
        df["Log_Deposit"] = np.log(df["Deposit_Amount"].replace(0, np.nan))

    if "YoY_Growth" not in df.columns:
        df["YoY_Growth"] = df["Deposit_Amount"].pct_change(4) * 100

    for q in [1, 2, 3, 4]:
        col = f"Q{q}"
        if col not in df.columns:
            df[col] = df.index.quarter.map(lambda x, _q=q: int(x == _q))

    if "Rolling_12M_Avg" not in df.columns:
        df["Rolling_12M_Avg"] = df["Deposit_Amount"].rolling(4).mean()

    # ── Merge macro indicators ────────────────────────────────────────────────
    mpath = Path(macro_path) if macro_path else _resolve_macro_path(filepath)
    if mpath and mpath.exists():
        macro = pd.read_csv(mpath)
        macro.columns = macro.columns.str.strip()
        macro = _parse_period_index(macro)
        macro = macro[[c for c in MACRO_COLS if c in macro.columns]]
        df = df.join(macro, how="left")
    else:
        for col in MACRO_COLS:
            df[col] = np.nan

    # ── Lag features ─────────────────────────────────────────────────────────
    dep = df["Deposit_Amount"]
    df["Deposit_t1"] = dep.shift(1)
    df["Deposit_t4"] = dep.shift(4)
    df["Deposit_t8"] = dep.shift(8)

    # ── Interaction feature ───────────────────────────────────────────────────
    df["rate_x_growth"] = df["RBI_Repo_Rate"] * df["YoY_Growth"]

    # ── Rolling volatility ────────────────────────────────────────────────────
    df["rolling_vol_4q"] = dep.rolling(4).std()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ── StandardScaler on engineered features ─────────────────────────────────
    scaler = None
    scale_cols = [c for c in _SCALE_COLS if c in df.columns]

    if fit_scaler and scale_cols:
        valid_mask = df[scale_cols].notna().all(axis=1)
        scaler = StandardScaler()
        # Cast to float to avoid dtype mismatch on integer columns (e.g. Nifty)
        for _c in scale_cols:
            df[_c] = df[_c].astype(float)
        scaler.fit(df.loc[valid_mask, scale_cols])
        df.loc[valid_mask, scale_cols] = scaler.transform(df.loc[valid_mask, scale_cols])

    return (df, scaler) if return_scaler else df


def get_train_test(
    df: pd.DataFrame,
    target: str = "Deposit_Amount",
    train_frac: float = 0.80,
):
    """Return chronological train / test splits of the target column."""
    series = df[target].dropna()
    n = len(series)
    split = int(n * train_frac)
    return series.iloc[:split], series.iloc[split:]


def get_train_test_with_exog(
    df: pd.DataFrame,
    target: str = "Deposit_Amount",
    train_frac: float = 0.80,
    exog_cols=None,
):
    """
    Return train/test splits for target + exogenous features.
    Scaler is fit on TRAIN only — no data leakage.

    Returns
    -------
    train_y, test_y, train_exog, test_exog
    """
    if exog_cols is None:
        exog_cols = ["RBI_Repo_Rate", "CPI_Inflation", "GDP_Growth_Rate"]

    series = df[target].dropna()
    n = len(series)
    split = int(n * train_frac)

    train_y = series.iloc[:split]
    test_y  = series.iloc[split:]

    train_exog, test_exog, _ = scale_exog_train_test(
        df=df,
        train_index=train_y.index,
        test_index=test_y.index,
        exog_cols=exog_cols,
    )

    return train_y, test_y, train_exog, test_exog


def scale_exog_train_test(
    df: pd.DataFrame,
    train_index: Sequence,
    test_index: Sequence,
    exog_cols=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[StandardScaler]]:
    """
    Scale exogenous variables with a scaler fitted only on the train window.

    The helper accepts explicit train/test indices so it is safe for both the
    final holdout split and walk-forward CV folds.
    """
    if exog_cols is None:
        exog_cols = ["RBI_Repo_Rate", "CPI_Inflation", "GDP_Growth_Rate"]

    available = [c for c in exog_cols if c in df.columns]
    train_index = pd.Index(train_index)
    test_index = pd.Index(test_index)

    if not available:
        return (
            pd.DataFrame(index=train_index),
            pd.DataFrame(index=test_index),
            None,
        )

    train_raw = df.reindex(train_index)[available].astype(float)
    test_raw = df.reindex(test_index)[available].astype(float)

    train_filled = train_raw.ffill().bfill()
    train_means = train_filled.mean()
    train_filled = train_filled.fillna(train_means).fillna(0.0)
    test_filled = test_raw.ffill().fillna(train_means).fillna(0.0)

    scaler = StandardScaler()
    scaler.fit(train_filled)

    train_exog = pd.DataFrame(
        scaler.transform(train_filled),
        index=train_index,
        columns=available,
    )
    test_exog = pd.DataFrame(
        scaler.transform(test_filled),
        index=test_index,
        columns=available,
    )

    return train_exog, test_exog, scaler


def get_log_series(df: pd.DataFrame) -> pd.Series:
    return df["Log_Deposit"].dropna()
