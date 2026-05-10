"""
run_advanced.py — Advanced features runner for BOI CASA Forecasting.

Runs after the pipeline and produces:
  • Ensemble forecast (inverse-MAPE weighted combination)
  • Anomaly detection report (IQR + Z-Score + Rolling)
  • Drift detection (KS + CUSUM + Residual)
  • Per-period confidence scores

Usage
-----
    python run_advanced.py
    python run_advanced.py --no-ensemble
    python run_advanced.py --out-dir data/processed
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.logger import get_logger, PipelineLogger
from src.pipelines.forecasting_pipeline import ForecastingPipeline
from src.advanced.ensemble import EnsembleForecaster
from src.advanced.anomaly_detection import detect_anomalies
from src.advanced.drift_detection import DriftDetector
from src.advanced.confidence_scoring import ForecastConfidenceScorer
from src.config import DATA_FILE, TRAIN_FRACTION

_log = get_logger("run_advanced")
SEP  = "═" * 64


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BOI CASA — Advanced Features (Point 11)"
    )
    parser.add_argument("--data",       default=DATA_FILE)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--out-dir",    default="data/processed")
    parser.add_argument("--no-ensemble",   action="store_true")
    parser.add_argument("--no-anomaly",    action="store_true")
    parser.add_argument("--no-drift",      action="store_true")
    parser.add_argument("--no-confidence", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Run full pipeline ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("  BOI CASA — ADVANCED FEATURES RUNNER")
    print(SEP)

    with PipelineLogger("Full forecasting pipeline", "run_advanced"):
        pipeline = ForecastingPipeline(
            data_path=args.data,
            train_frac=args.train_frac,
            models=["ARIMA", "SARIMA", "SARIMAX", "AutoARIMA", "Prophet"],
            run_cv=False,
        )
        pipeline.run()

    results = pipeline.results
    train   = pipeline.train
    test    = pipeline.test
    df      = pipeline.df
    series  = df["Deposit_Amount"]

    # ── Step 2: Ensemble ─────────────────────────────────────────────────
    if not args.no_ensemble:
        print(f"\n{'─'*64}")
        print("  ENSEMBLE FORECASTING")
        print(f"{'─'*64}")
        with PipelineLogger("Ensemble forecaster", "run_advanced"):
            ens = EnsembleForecaster(results, test, strategy="inverse_mape")
            ens_fc, ens_ci = ens.forecast()
        print(ens.summary())

        ens_path = out_dir / "ensemble_forecast.csv"
        ens_out  = ens_fc.to_frame("Ensemble_Forecast")
        ens_out["CI_Lower"] = ens_ci["lower"]
        ens_out["CI_Upper"] = ens_ci["upper"]
        ens_out["Actual"]   = test
        ens_out.to_csv(ens_path)
        _log.info("Ensemble forecast saved → %s", ens_path)

    # ── Step 3: Anomaly Detection ─────────────────────────────────────────
    if not args.no_anomaly:
        print(f"\n{'─'*64}")
        print("  ANOMALY DETECTION")
        print(f"{'─'*64}")
        with PipelineLogger("Anomaly detection", "run_advanced"):
            report = detect_anomalies(
                series,
                methods=["iqr", "zscore", "rolling"],
            )
        print(report.summary())

        flagged = report.flagged_periods(min_methods=1)
        if not flagged.empty:
            anom_path = out_dir / "anomaly_report.csv"
            flagged.to_csv(anom_path)
            _log.info("Anomaly report saved → %s", anom_path)

    # ── Step 4: Drift Detection ───────────────────────────────────────────
    if not args.no_drift:
        print(f"\n{'─'*64}")
        print("  DRIFT DETECTION")
        print(f"{'─'*64}")
        best_model  = pipeline.comparison.best_model()
        best_res    = results[best_model]["residuals"]
        live_series = test          # treat test set as "live" data

        with PipelineLogger("Drift detection", "run_advanced"):
            detector = DriftDetector(
                train_series=train,
                train_residuals=best_res,
            )
            drift_report = detector.check(
                live_series=live_series,
                live_residuals=best_res[-len(test):],
            )
        print(drift_report.summary())

    # ── Step 5: Confidence Scoring ────────────────────────────────────────
    if not args.no_confidence:
        print(f"\n{'─'*64}")
        print("  FORECAST CONFIDENCE SCORING")
        print(f"{'─'*64}")
        best_model = pipeline.comparison.best_model()
        rec        = results[best_model]

        with PipelineLogger("Confidence scoring", "run_advanced"):
            scorer = ForecastConfidenceScorer(
                metrics=rec["metrics"],
                residuals=rec["residuals"],
            )
            conf_df = scorer.score(
                forecast=rec["forecast"],
                ci_df=rec["ci"],
                drift_detected=drift_report.overall_drift if not args.no_drift else False,
            )
        print(scorer.summary(conf_df))

        conf_path = out_dir / "confidence_scores.csv"
        conf_df.to_csv(conf_path)
        _log.info("Confidence scores saved → %s", conf_path)

    print(f"\n{SEP}")
    print("  ✅  All advanced features complete.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
