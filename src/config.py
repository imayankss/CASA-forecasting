"""
config.py — Central configuration for BOI CASA Deposit Forecasting System (v2.0)

Changes from v1.0
-----------------
* ARIMA order corrected to (1,1,0) — prevents overfitting on 32-point train set
* SARIMA seasonal order corrected to (0,1,1,4) — fewer parameters
* SARIMAX real exog columns defined (RBI_Repo_Rate, CPI_Inflation, GDP_Growth_Rate)
* HOLT_WINTERS_CONFIG added (additive trend + seasonal, period=4)
* CV_SPLITS set to 5 (TimeSeriesSplit)
* MACRO_DATA_FILE path added
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────
ROOT_DIR             = Path(__file__).resolve().parent.parent
DATA_DIR             = ROOT_DIR / "data"
RAW_DATA_DIR         = DATA_DIR / "raw"
PROCESSED_DATA_DIR   = DATA_DIR / "processed"
REPORTS_DIR          = ROOT_DIR / "reports"
VISUALIZATIONS_DIR   = ROOT_DIR / "visualizations" / "output"
MODELS_DIR           = ROOT_DIR / "models"
DASHBOARD_DIR        = ROOT_DIR / "dashboard"

# Create output directories on import
for _d in [VISUALIZATIONS_DIR, REPORTS_DIR, PROCESSED_DATA_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# DATA SETTINGS
# ─────────────────────────────────────────────
DATA_FILE        = os.getenv("BOI_DATA_FILE", str(RAW_DATA_DIR / "boi_casa_deposits.csv"))
MACRO_DATA_FILE  = str(RAW_DATA_DIR / "macro_indicators.csv")   # NEW in v2.0
DATE_COLUMN      = "Quarter_Year"
TARGET_COLUMN    = "Deposit_Amount"

# Original engineered features (kept for backward-compat)
EXOG_COLUMNS     = ["Log_Deposit", "YoY_Growth", "Q1", "Q2", "Q3", "Q4", "Rolling_12M_Avg"]

# NEW v2.0 — real macro exogenous variables for SARIMAX
SARIMAX_REAL_EXOG = ["RBI_Repo_Rate", "CPI_Inflation", "GDP_Growth_Rate"]

SEASONAL_PERIOD  = 4        # Quarterly data (4 quarters per year)
TRAIN_FRACTION   = 0.80
CV_SPLITS        = 5        # TimeSeriesSplit n_splits (v2.0: was 3)
CV_MIN_TRAIN     = 16       # Minimum quarters in each CV training window

# ─────────────────────────────────────────────
# FORECAST SETTINGS
# ─────────────────────────────────────────────
FORECAST_HORIZON  = int(os.getenv("FORECAST_HORIZON", 4))   # Next N quarters
CONFIDENCE_LEVEL  = 0.95

# ─────────────────────────────────────────────
# MODEL PARAMETERS  (v2.0 — FIXED)
# ─────────────────────────────────────────────

# ARIMA: reduced from (1,1,1) to (1,1,0) — prevents overfitting on small samples
ARIMA_ORDER              = (1, 1, 0)          # v2.0 fix: was (1,1,1)

# SARIMA: reduced seasonal params from (1,1,1,4) to (0,1,1,4)
SARIMA_ORDER             = (1, 1, 1)
SARIMA_SEASONAL_ORDER    = (0, 1, 1, SEASONAL_PERIOD)   # v2.0 fix: was (1,1,1,4)

# SARIMAX: same orders as SARIMA; real exog from SARIMAX_REAL_EXOG (not time index)
SARIMAX_ORDER            = (1, 1, 1)
SARIMAX_SEASONAL_ORDER   = (0, 1, 1, SEASONAL_PERIOD)   # v2.0 fix: was (1,1,1,4)

# AutoARIMA — unchanged, but max_P/max_Q reduced to avoid overfit
AUTO_ARIMA_CONFIG = {
    "seasonal":           True,
    "m":                  SEASONAL_PERIOD,
    "max_p":              3,
    "max_q":              3,
    "max_P":              1,
    "max_Q":              1,
    "stepwise":           True,
    "approximation":      False,
    "trace":              False,
    "error_action":       "ignore",
    "suppress_warnings":  True,
}

# Prophet — unchanged
PROPHET_CONFIG = {
    "seasonality_mode":   "multiplicative",
    "yearly_seasonality": True,
}

# HoltWinters — NEW in v2.0
HOLT_WINTERS_CONFIG = {
    "trend":              "add",    # additive trend
    "seasonal":           "add",    # additive seasonal (safer than multiplicative for deposits)
    "seasonal_periods":   SEASONAL_PERIOD,
    "damped_trend":       False,
    "initialization_method": "estimated",
    "optimized":          True,
}

# ─────────────────────────────────────────────
# VISUALIZATION SETTINGS
# ─────────────────────────────────────────────
PALETTE = {
    "primary":    "#2563EB",
    "secondary":  "#7C3AED",
    "accent":     "#059669",
    "warning":    "#D97706",
    "danger":     "#DC2626",
    "muted":      "#6B7280",
    "background": "#0F172A",
    "surface":    "#1E293B",
    "text":       "#F1F5F9",
    "grid":       "#334155",
}

MODEL_COLORS = {
    "ARIMA":       "#3B82F6",
    "SARIMA":      "#10B981",
    "SARIMAX":     "#F59E0B",
    "AutoARIMA":   "#EF4444",
    "HoltWinters": "#8B5CF6",    # NEW in v2.0
    "Prophet":     "#EC4899",
    "Actual":      "#F1F5F9",
}

FIGSIZE_WIDE   = (16, 6)
FIGSIZE_TALL   = (14, 10)
FIGSIZE_SQUARE = (10, 10)
DPI            = 150

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE  = ROOT_DIR / "reports" / "forecast.log"
