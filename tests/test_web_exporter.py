"""Tests for static web JSON export."""

import json
from pathlib import Path

import pandas as pd

from src.export.web_exporter import WebDataExporter


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_exporter_generates_required_json_with_missing_optional_outputs(tmp_path):
    _write_csv(
        tmp_path / "data" / "raw" / "boi_casa_deposits.csv",
        [
            {"Quarter_Year": "2024Q1", "Deposit_Amount": 100.0},
            {"Quarter_Year": "2024Q2", "Deposit_Amount": 110.0},
        ],
    )
    _write_csv(
        tmp_path / "data" / "processed" / "model_leaderboard.csv",
        [
            {
                "Model": "HoltWinters",
                "Rank": 1,
                "MAPE": 2.1,
                "RMSE_%": 2.3,
                "MAE_%": 1.8,
                "R2": 0.89,
                "Stability": 82.0,
                "Composite_Score": 100.0,
                "Recommended": True,
            }
        ],
    )
    _write_csv(
        tmp_path / "data" / "processed" / "forecast_results.csv",
        [
            {
                "Period": "2024-07-01",
                "HoltWinters": 115.0,
                "Actual_Test": 113.0,
            }
        ],
    )

    output_dir = tmp_path / "web" / "public" / "data"
    generated = WebDataExporter(root_dir=tmp_path, output_dir=output_dir).export_all()

    assert len(generated) == 10
    for filename in [
        "manifest.json",
        "kpis.json",
        "forecast.json",
        "leaderboard.json",
        "model_metrics.json",
        "cv_summary.json",
        "confidence_scores.json",
        "insights.json",
        "anomalies.json",
        "drift.json",
    ]:
        assert (output_dir / filename).exists()

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["model_count"] == 1
    assert manifest["best_model"] == "HoltWinters"

    drift = json.loads((output_dir / "drift.json").read_text())
    assert drift["drift_detected"] is False
    assert drift["drift_type"] == "Unavailable"


def test_exporter_maps_confidence_and_forecast_payloads(tmp_path):
    _write_csv(
        tmp_path / "data" / "raw" / "boi_casa_deposits.csv",
        [{"Quarter_Year": "2024Q4", "Deposit_Amount": 200.0}],
    )
    _write_csv(
        tmp_path / "data" / "processed" / "model_leaderboard.csv",
        [{"Model": "Prophet", "Rank": 1, "MAPE": 2.0, "R2": 0.7, "Recommended": True}],
    )
    _write_csv(
        tmp_path / "data" / "processed" / "forecast_results.csv",
        [{"Period": "2025-01-01", "Prophet": 220.0, "Actual_Test": 218.0}],
    )
    _write_csv(
        tmp_path / "data" / "processed" / "confidence_scores.csv",
        [
            {
                "period": "2025-01-01",
                "forecast": 220.0,
                "lower_95": 210.0,
                "upper_95": 230.0,
                "confidence_score": 72.0,
                "confidence_label": "High",
                "drift_penalty": 0.0,
            }
        ],
    )

    output_dir = tmp_path / "web" / "public" / "data"
    WebDataExporter(root_dir=tmp_path, output_dir=output_dir).export_all()

    forecast = json.loads((output_dir / "forecast.json").read_text())
    assert forecast[0]["period"] == "2025 Q1"
    assert forecast[0]["Prophet"] == 220.0
    assert forecast[0]["lower_ci"] == 210.0

    kpis = json.loads((output_dir / "kpis.json").read_text())
    assert kpis["best_model"] == "Prophet"
    assert kpis["confidence_label"] == "High"
