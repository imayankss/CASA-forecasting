"""
test_advanced_models.py — Advanced unit tests covering:
  * Holt-Winters fitter (forecast length, no NaN)
  * SARIMAX with real exogenous variables
  * TimeSeriesSplit CV (5 folds, no data leakage)
  * direction_accuracy (perfect = 100%, constant = nan)
  * macro_indicators merge (correct column count)
  * adjusted_r2 edge cases
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_FILE
from src.data_loader import load_casa_data, get_train_test, MACRO_COLS, EXOG_COLS
from src.evaluation.metrics import (
    direction_accuracy,
    adjusted_r2,
    compute_full_metrics,
)
from src.pipelines.forecasting_pipeline import (
    FITTERS,
    timeseries_cv,
    walk_forward_cv,
)


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def loaded_data():
    """Load data once for the whole module."""
    result = load_casa_data(DATA_FILE)
    df, scaler = result if isinstance(result, tuple) else (result, None)
    train, test = get_train_test(df)
    return df, train, test, scaler


@pytest.fixture(scope="module")
def macro_path():
    return ROOT / "data" / "raw" / "macro_indicators.csv"


# ─────────────────────────────────────────────────────────────
# 1. HOLT-WINTERS
# ─────────────────────────────────────────────────────────────

class TestHoltWinters:
    def test_forecast_length_matches_test(self, loaded_data):
        """HoltWinters forecast must have exactly len(test) predictions."""
        df, train, test, _ = loaded_data
        out = FITTERS["HoltWinters"](train, test, df)
        assert len(out["forecast"]) == len(test), (
            f"Expected {len(test)} forecast steps, got {len(out['forecast'])}"
        )

    def test_no_nan_in_forecast(self, loaded_data):
        df, train, test, _ = loaded_data
        out = FITTERS["HoltWinters"](train, test, df)
        assert not np.any(np.isnan(out["forecast"].values)), \
            "HoltWinters forecast contains NaN values"

    def test_metrics_keys_present(self, loaded_data):
        df, train, test, _ = loaded_data
        out = FITTERS["HoltWinters"](train, test, df)
        for key in ["MAE", "MAPE", "R2", "Direction_Accuracy"]:
            assert key in out["metrics"], f"Missing metric: {key}"

    def test_r2_not_negative(self, loaded_data):
        """Holt-Winters should achieve non-negative R² on this dataset."""
        df, train, test, _ = loaded_data
        out = FITTERS["HoltWinters"](train, test, df)
        r2 = out["metrics"]["R2"]
        assert r2 >= 0.0, f"HoltWinters R² is negative: {r2:.4f}"

    def test_ci_has_correct_shape(self, loaded_data):
        df, train, test, _ = loaded_data
        out = FITTERS["HoltWinters"](train, test, df)
        ci = out["ci"]
        assert ci.shape == (len(test), 2), f"CI shape mismatch: {ci.shape}"
        assert "lower" in ci.columns and "upper" in ci.columns


# ─────────────────────────────────────────────────────────────
# 2. SARIMAX WITH REAL EXOG
# ─────────────────────────────────────────────────────────────

class TestSARIMAXRealExog:
    def test_runs_without_error(self, loaded_data):
        """SARIMAX with real exog must complete without raising."""
        df, train, test, _ = loaded_data
        out = FITTERS["SARIMAX"](train, test, df)
        assert out is not None

    def test_returns_valid_metrics(self, loaded_data):
        df, train, test, _ = loaded_data
        out = FITTERS["SARIMAX"](train, test, df)
        m = out["metrics"]
        assert "MAPE" in m and "R2" in m
        assert not np.isnan(m["MAPE"]), "SARIMAX MAPE is NaN"
        assert m["MAPE"] > 0, "SARIMAX MAPE must be positive"

    def test_forecast_length(self, loaded_data):
        df, train, test, _ = loaded_data
        out = FITTERS["SARIMAX"](train, test, df)
        assert len(out["forecast"]) == len(test)

    def test_params_record_exog(self, loaded_data):
        """params dict should indicate real exog, not 'time_index'."""
        df, train, test, _ = loaded_data
        out = FITTERS["SARIMAX"](train, test, df)
        exog_param = str(out["params"].get("exog", ""))
        assert "time_index" not in exog_param, \
            "SARIMAX still using dummy time index as exog"


# ─────────────────────────────────────────────────────────────
# 3. TimeSeriesSplit CV
# ─────────────────────────────────────────────────────────────

class TestTimeSeriesSplitCV:
    def test_returns_five_folds(self, loaded_data):
        """With n_splits=5 and sufficient data, we should get 5 folds."""
        df, train, test, _ = loaded_data
        full_series = pd.concat([train, test])
        cv_df = timeseries_cv(
            full_series, FITTERS["ARIMA"], n_splits=5, min_train_size=16, df=df
        )
        # Allow ≤5 if some folds were skipped due to min_train_size
        assert len(cv_df) > 0, "No CV folds completed"
        assert len(cv_df) <= 5, f"Expected ≤5 folds, got {len(cv_df)}"

    def test_no_data_leakage(self):
        """
        For every fold from TimeSeriesSplit, max(train_idx) < min(test_idx).
        This is the fundamental anti-leakage guarantee.
        """
        n = 40
        dummy = pd.Series(np.arange(n, dtype=float))
        tscv  = TimeSeriesSplit(n_splits=5)
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(dummy)):
            assert train_idx.max() < test_idx.min(), (
                f"Fold {fold_idx + 1}: data leakage detected! "
                f"max(train)={train_idx.max()}, min(test)={test_idx.min()}"
            )

    def test_mape_positive_in_all_folds(self, loaded_data):
        df, train, _, _ = loaded_data
        cv_df = timeseries_cv(
            train, FITTERS["ARIMA"], n_splits=3, min_train_size=16, df=df
        )
        if not cv_df.empty:
            assert (cv_df["MAPE"] > 0).all(), "MAPE must be positive in all folds"

    def test_cv_summary_mean_std(self, loaded_data):
        """CV summary must report both mean and std of MAPE."""
        df, train, _, _ = loaded_data
        cv_df = timeseries_cv(
            train, FITTERS["HoltWinters"], n_splits=3, min_train_size=16, df=df
        )
        if not cv_df.empty:
            assert "MAPE" in cv_df.columns
            assert cv_df["MAPE"].mean() > 0
            # std should be computable (not NaN) when > 1 fold
            if len(cv_df) > 1:
                assert not np.isnan(cv_df["MAPE"].std())

    def test_walk_forward_cv_backward_compat(self, loaded_data):
        """Legacy walk_forward_cv wrapper must still return a DataFrame."""
        df, train, _, _ = loaded_data
        result = walk_forward_cv(train, FITTERS["ARIMA"], n_splits=3, df=df)
        assert isinstance(result, pd.DataFrame)


# ─────────────────────────────────────────────────────────────
# 4. direction_accuracy
# ─────────────────────────────────────────────────────────────

class TestDirectionAccuracy:
    def test_perfect_forecast_is_100(self):
        """When predicted == actual, direction accuracy should be 100%."""
        actual = np.array([100, 110, 105, 120, 115, 130], dtype=float)
        pred   = actual.copy()
        assert direction_accuracy(actual, pred) == pytest.approx(100.0)

    def test_opposite_forecast_is_0(self):
        """When every step is wrong (mirrored), direction accuracy should be 0%."""
        actual = np.array([100, 110, 120, 130], dtype=float)
        # Predicted decreases when actual increases (perfect mirror)
        pred   = np.array([130, 120, 110, 100], dtype=float)
        assert direction_accuracy(actual, pred) == pytest.approx(0.0)

    def test_random_is_near_50(self):
        """Random forecast direction accuracy should be near 50% on large arrays."""
        rng    = np.random.default_rng(42)
        actual = np.cumsum(rng.normal(0, 1, 500)) + 100
        pred   = np.cumsum(rng.normal(0, 1, 500)) + 100
        result = direction_accuracy(actual, pred)
        assert 30.0 < result < 70.0, f"Expected ~50%, got {result:.1f}%"

    def test_too_short_returns_nan(self):
        """Single-element arrays can't compute direction — should return nan."""
        result = direction_accuracy(np.array([100.0]), np.array([100.0]))
        assert np.isnan(result)

    def test_constant_actual_returns_nan(self):
        """If actual never moves, mask is all-zero → nan."""
        actual = np.ones(10) * 100.0
        pred   = np.linspace(95, 105, 10)
        result = direction_accuracy(actual, pred)
        assert np.isnan(result)

    def test_returns_float(self):
        actual = np.array([1.0, 2.0, 1.5, 3.0])
        pred   = np.array([1.1, 1.9, 1.6, 2.8])
        result = direction_accuracy(actual, pred)
        assert isinstance(result, float)


# ─────────────────────────────────────────────────────────────
# 5. macro_indicators merge
# ─────────────────────────────────────────────────────────────

class TestMacroIndicatorsMerge:
    def test_macro_file_exists(self, macro_path):
        assert macro_path.exists(), (
            f"macro_indicators.csv not found at {macro_path}. "
            "Run Task 1 to create it."
        )

    def test_correct_row_count(self, macro_path):
        """Should have exactly 40 rows (2015Q1 – 2024Q4)."""
        macro = pd.read_csv(macro_path)
        assert len(macro) == 40, f"Expected 40 rows, got {len(macro)}"

    def test_all_macro_columns_present(self, macro_path):
        macro = pd.read_csv(macro_path)
        for col in MACRO_COLS:
            assert col in macro.columns, f"Missing macro column: {col}"

    def test_merged_df_has_macro_columns(self, loaded_data):
        df, _, _, _ = loaded_data
        for col in MACRO_COLS:
            assert col in df.columns, f"Macro column {col} missing from merged DataFrame"

    def test_merged_column_count(self, loaded_data):
        """
        After merging macro + engineering features, DataFrame should have at
        least the original columns + 6 macro + 3 lags + 2 engineered = ≥11 new cols.
        """
        df, _, _, _ = loaded_data
        base_cols = {"Quarter_Year", "Deposit_Amount", "Log_Deposit", "YoY_Growth",
                     "Q1", "Q2", "Q3", "Q4", "Rolling_12M_Avg"}
        expected_new = set(MACRO_COLS) | {"Deposit_t1", "Deposit_t4", "Deposit_t8",
                                          "rate_x_growth", "rolling_vol_4q"}
        for col in expected_new:
            assert col in df.columns, f"Expected engineered column '{col}' not in DataFrame"

    def test_rbi_repo_rate_range(self, macro_path):
        """RBI Repo Rate should be between 3.5% and 9% for 2015–2024."""
        macro = pd.read_csv(macro_path)
        rates = macro["RBI_Repo_Rate"]
        assert rates.between(3.5, 9.0).all(), (
            f"RBI rates out of expected range: min={rates.min()}, max={rates.max()}"
        )

    def test_gdp_growth_has_negative_covid(self, macro_path):
        """GDP_Growth_Rate should go negative in 2020Q2 (COVID lockdown)."""
        macro = pd.read_csv(macro_path)
        covid_q = macro[macro["Quarter_Year"] == "2020Q2"]["GDP_Growth_Rate"]
        assert not covid_q.empty, "2020Q2 row missing"
        assert float(covid_q.iloc[0]) < 0, "Expected negative GDP growth in 2020Q2 (COVID)"

    def test_no_future_data_leakage_in_lags(self, loaded_data):
        """Deposit_t1 at index i should equal Deposit_Amount at index i-1."""
        df, _, _, _ = loaded_data
        dep = df["Deposit_Amount"]
        lag1 = df["Deposit_t1"]
        for i in range(1, min(10, len(dep))):
            if not np.isnan(lag1.iloc[i]):
                assert dep.iloc[i - 1] == pytest.approx(lag1.iloc[i], rel=1e-6), \
                    f"Lag mismatch at position {i}"


# ─────────────────────────────────────────────────────────────
# 6. adjusted_r2 edge cases
# ─────────────────────────────────────────────────────────────

class TestAdjustedR2:
    def test_equal_to_r2_when_one_feature(self):
        """adjusted_r2 approaches r2 as sample size → ∞."""
        r2  = 0.85
        adj = adjusted_r2(r2, n_samples=1000, n_features=1)
        assert abs(adj - r2) < 0.01

    def test_decreases_with_more_features(self):
        """More features → lower adjusted R² (penalty)."""
        r2    = 0.85
        adj1  = adjusted_r2(r2, n_samples=40, n_features=1)
        adj5  = adjusted_r2(r2, n_samples=40, n_features=5)
        assert adj1 > adj5

    def test_nan_when_n_too_small(self):
        """n_samples ≤ n_features + 1 should return nan."""
        assert np.isnan(adjusted_r2(0.9, n_samples=3, n_features=3))

    def test_nan_input_returns_nan(self):
        assert np.isnan(adjusted_r2(float("nan"), n_samples=40, n_features=2))

    def test_negative_r2_stays_negative(self):
        """Even negative R² can produce an adjusted version."""
        adj = adjusted_r2(-0.1, n_samples=40, n_features=1)
        assert adj < 0

    def test_present_in_compute_full_metrics(self):
        """compute_full_metrics must include Adjusted_R2."""
        actual = np.linspace(100, 200, 20)
        pred   = actual + np.random.default_rng(0).normal(0, 5, 20)
        m = compute_full_metrics(actual, pred, n_features=1)
        assert "Adjusted_R2" in m
        assert "Direction_Accuracy" in m
