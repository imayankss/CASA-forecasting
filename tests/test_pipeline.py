"""
test_pipeline.py — Integration tests for the forecasting pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_FILE
from src.pipelines.forecasting_pipeline import ForecastingPipeline, walk_forward_cv, FITTERS
from src.evaluation.comparison import ModelComparison
from src.forecasting.registry import ModelRegistry, ModelRecord


# ── Run a lightweight pipeline once for the whole test session ───────────────
@pytest.fixture(scope="session")
def pipeline():
    p = ForecastingPipeline(
        data_path=DATA_FILE,
        models=["ARIMA", "SARIMA"],   # Fast subset for tests
        run_cv=True,
        cv_splits=2,
    )
    p.run()
    return p


class TestPipelineRun:
    def test_results_not_empty(self, pipeline):
        assert len(pipeline.results) == 2

    def test_models_trained(self, pipeline):
        for name in ["ARIMA", "SARIMA"]:
            assert name in pipeline.results

    def test_forecast_correct_length(self, pipeline):
        test_len = len(pipeline.test)
        for name, rec in pipeline.results.items():
            assert len(rec["forecast"]) == test_len, f"{name} forecast length mismatch"

    def test_metrics_present(self, pipeline):
        for name, rec in pipeline.results.items():
            m = rec["metrics"]
            assert "MAE" in m and "MAPE" in m and "R2" in m

    def test_residuals_not_all_nan(self, pipeline):
        for name, rec in pipeline.results.items():
            res = rec["residuals"]
            assert not np.all(np.isnan(res)), f"{name} residuals are all NaN"


class TestLeaderboard:
    def test_leaderboard_has_rows(self, pipeline):
        lb = pipeline.leaderboard()
        assert len(lb) == 2

    def test_rank_column(self, pipeline):
        lb = pipeline.leaderboard()
        assert "Rank" in lb.columns
        assert lb["Rank"].iloc[0] == 1

    def test_composite_score_bounded(self, pipeline):
        lb = pipeline.leaderboard()
        assert (lb["Composite_Score"] >= 0).all()
        assert (lb["Composite_Score"] <= 100).all()


class TestInsights:
    def test_insights_not_empty(self, pipeline):
        ins = pipeline.insights()
        assert ins

    def test_best_model_key(self, pipeline):
        ins = pipeline.insights()
        assert "best_model" in ins
        assert ins["best_model"] in ["ARIMA", "SARIMA"]

    def test_executive_summary_string(self, pipeline):
        ins = pipeline.insights()
        assert isinstance(ins["executive_summary"], str)
        assert len(ins["executive_summary"]) > 50


class TestCVSummary:
    def test_cv_has_rows(self, pipeline):
        cv = pipeline.cv_summary()
        assert len(cv) > 0

    def test_cv_mape_positive(self, pipeline):
        cv = pipeline.cv_summary()
        assert (cv["CV_MAPE_Mean"] > 0).all()


class TestModelRegistry:
    def test_registry_populated(self, pipeline):
        reg = pipeline.registry
        assert len(reg.all_names()) == 2

    def test_best_by_mape(self, pipeline):
        best = pipeline.registry.best_by("MAPE")
        assert best in ["ARIMA", "SARIMA"]

    def test_metrics_df(self, pipeline):
        df = pipeline.registry.metrics_df()
        assert "MAPE" in df.columns


class TestSaveResults:
    def test_save_creates_files(self, pipeline, tmp_path):
        saved = pipeline.save_results(out_dir=str(tmp_path))
        assert "leaderboard" in saved
        assert Path(saved["leaderboard"]).exists()

    def test_forecasts_csv_exists(self, pipeline, tmp_path):
        saved = pipeline.save_results(out_dir=str(tmp_path))
        assert Path(saved["forecasts"]).exists()


class TestWalkForwardCV:
    def test_returns_dataframe(self):
        import warnings
        warnings.filterwarnings("ignore")
        from src.data_loader import load_casa_data, get_train_test
        df = load_casa_data(DATA_FILE)
        series, _ = get_train_test(df)
        cv_df = walk_forward_cv(series, FITTERS["ARIMA"], n_splits=2, test_size=3)
        assert isinstance(cv_df, pd.DataFrame)
        assert "MAPE" in cv_df.columns


class TestModelComparison:
    def test_add_and_leaderboard(self):
        actual = np.array([100, 110, 120, 130], dtype=float)
        pred1  = np.array([102, 108, 122, 128], dtype=float)
        pred2  = np.array([99,  112, 119, 132], dtype=float)
        res1   = actual - pred1
        res2   = actual - pred2

        comp = ModelComparison()
        comp.add("ModelA", actual, pred1, res1)
        comp.add("ModelB", actual, pred2, res2)

        lb = comp.leaderboard()
        assert len(lb) == 2
        assert "Composite_Score" in lb.columns
        assert lb["Rank"].tolist() == [1, 2]

    def test_best_model_returns_string(self):
        actual = np.ones(10) * 100
        pred   = actual + np.random.randn(10)
        comp   = ModelComparison()
        comp.add("OnlyModel", actual, pred, actual - pred)
        assert comp.best_model() == "OnlyModel"
