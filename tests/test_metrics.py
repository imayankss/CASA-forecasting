"""
test_metrics.py — Unit tests for src/evaluation/metrics.py
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    safe_mape, smape, forecast_bias, forecast_bias_pct,
    stability_score, theil_u, compute_full_metrics
)


ACTUAL = np.array([100, 110, 105, 120, 115, 130, 125, 140], dtype=float)
PRED   = np.array([102, 108, 107, 118, 117, 128, 127, 138], dtype=float)


class TestSafeMAPE:
    def test_basic(self):
        result = safe_mape(ACTUAL, PRED)
        assert 0 < result < 10

    def test_zero_actuals_ignored(self):
        y_true = np.array([0, 100, 200], dtype=float)
        y_pred = np.array([10, 110, 210], dtype=float)
        result = safe_mape(y_true, y_pred)
        assert not np.isnan(result)

    def test_all_zeros_returns_nan(self):
        result = safe_mape(np.zeros(5), np.ones(5))
        assert np.isnan(result)

    def test_perfect_forecast(self):
        result = safe_mape(ACTUAL, ACTUAL)
        assert result == pytest.approx(0.0, abs=1e-6)


class TestSMAPE:
    def test_basic(self):
        result = smape(ACTUAL, PRED)
        assert 0 < result < 10

    def test_symmetric(self):
        s1 = smape(ACTUAL, PRED)
        s2 = smape(PRED, ACTUAL)
        assert s1 == pytest.approx(s2, rel=0.01)


class TestForecastBias:
    def test_positive_bias(self):
        actual = np.array([100.0, 100.0, 100.0])
        pred   = np.array([110.0, 110.0, 110.0])
        assert forecast_bias(actual, pred) == pytest.approx(10.0)

    def test_negative_bias(self):
        actual = np.array([100.0, 100.0])
        pred   = np.array([90.0, 90.0])
        assert forecast_bias(actual, pred) == pytest.approx(-10.0)

    def test_zero_bias(self):
        assert forecast_bias(ACTUAL, ACTUAL) == pytest.approx(0.0)

    def test_bias_pct(self):
        actual = np.array([100.0, 100.0])
        pred   = np.array([105.0, 105.0])
        assert forecast_bias_pct(actual, pred) == pytest.approx(5.0)


class TestStabilityScore:
    def test_perfect_residuals(self):
        score = stability_score(np.zeros(20))
        assert score == pytest.approx(100.0)

    def test_bounded(self):
        res = np.random.randn(50) * 1000
        score = stability_score(res)
        assert 0 <= score <= 100

    def test_noisy_lower_than_stable(self):
        stable = stability_score(np.full(50, 0.001))   # constant → CV=0 → 100
        noisy  = stability_score(np.linspace(-1000, 1000, 50))  # high CV
        assert stable > noisy


class TestTheilU:
    def test_perfect_less_than_one(self):
        u = theil_u(ACTUAL, ACTUAL)
        assert u == pytest.approx(0.0, abs=1e-6)

    def test_returns_float(self):
        u = theil_u(ACTUAL, PRED)
        assert isinstance(u, float)

    def test_short_series(self):
        u = theil_u(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert isinstance(u, float)


class TestComputeFullMetrics:
    def test_all_keys_present(self):
        m = compute_full_metrics(ACTUAL, PRED)
        required = ["MAE", "RMSE", "MSE", "MAE_%", "RMSE_%", "MAPE", "R2", "Stability"]
        for k in required:
            assert k in m, f"Missing key: {k}"

    def test_mae_positive(self):
        m = compute_full_metrics(ACTUAL, PRED)
        assert m["MAE"] > 0

    def test_perfect_r2(self):
        m = compute_full_metrics(ACTUAL, ACTUAL)
        assert m["MAE"] == pytest.approx(0.0, abs=1e-6)

    def test_mape_range(self):
        m = compute_full_metrics(ACTUAL, PRED)
        assert 0 <= m["MAPE"] < 100
