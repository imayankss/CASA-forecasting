"""
test_diagnostics.py — Unit tests for src/evaluation/diagnostics.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.diagnostics import (
    run_adf_test, run_shapiro_test, run_jarque_bera_test,
    run_ljung_box_test, run_full_diagnostics, acf_pacf_values,
)

np.random.seed(42)
NORMAL_RESIDUALS = np.random.normal(0, 1, 40)
STATIONARY_SERIES = pd.Series(np.random.normal(0, 1, 40))


class TestADFTest:
    def test_stationary_series(self):
        result = run_adf_test(STATIONARY_SERIES)
        assert "statistic" in result
        assert "p_value" in result
        assert "stationary" in result
        assert isinstance(result["stationary"], bool)

    def test_non_stationary_series(self):
        rng = np.random.default_rng(42)
        rw = pd.Series(np.cumsum(rng.normal(0, 1, 80)))
        result = run_adf_test(rw)
        assert result["stationary"] is False


class TestShapiroTest:
    def test_normal_residuals(self):
        result = run_shapiro_test(NORMAL_RESIDUALS)
        assert "statistic" in result
        assert "p_value" in result
        assert "normal" in result

    def test_all_keys(self):
        result = run_shapiro_test(NORMAL_RESIDUALS)
        for k in ["test", "statistic", "p_value", "normal", "verdict", "interpretation"]:
            assert k in result

    def test_non_normal_fails(self):
        bimodal = np.concatenate([np.random.normal(-10, 0.5, 20),
                                   np.random.normal(10, 0.5, 20)])
        result = run_shapiro_test(bimodal)
        assert result["normal"] is False


class TestJarqueBeraTest:
    def test_structure(self):
        result = run_jarque_bera_test(NORMAL_RESIDUALS)
        for k in ["test", "statistic", "p_value", "normal"]:
            assert k in result

    def test_normal_residuals_pass(self):
        rng = np.random.default_rng(123)
        result = run_jarque_bera_test(rng.normal(0, 1, 200))
        assert result["normal"] is True


class TestLjungBoxTest:
    def test_white_noise(self):
        result = run_ljung_box_test(np.random.randn(40))
        assert "no_autocorr" in result

    def test_autocorrelated_fails(self):
        ar = np.zeros(50)
        for t in range(1, 50):
            ar[t] = 0.9 * ar[t-1] + np.random.randn()
        result = run_ljung_box_test(ar)
        assert result["no_autocorr"] is False


class TestFullDiagnostics:
    def test_runs_without_error(self):
        diag = run_full_diagnostics(NORMAL_RESIDUALS, model_name="Test")
        assert diag is not None

    def test_health_score_bounded(self):
        diag = run_full_diagnostics(NORMAL_RESIDUALS)
        assert 0 <= diag["health_score"] <= 100

    def test_stats_present(self):
        diag = run_full_diagnostics(NORMAL_RESIDUALS)
        for k in ["mean", "std", "skewness", "kurtosis"]:
            assert k in diag["stats"]

    def test_handles_nan(self):
        r = NORMAL_RESIDUALS.copy().astype(float)
        r[0] = np.nan
        diag = run_full_diagnostics(r)
        assert diag["n_residuals"] == sum(~np.isnan(r))


class TestACFPACF:
    def test_returns_three_arrays(self):
        lags, acf, pacf = acf_pacf_values(NORMAL_RESIDUALS)
        assert len(lags) == len(acf) == len(pacf)

    def test_lag0_acf_is_one(self):
        _, acf, _ = acf_pacf_values(NORMAL_RESIDUALS)
        assert acf[0] == pytest.approx(1.0, abs=1e-6)
