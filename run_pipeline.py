"""
run_pipeline.py — Master CLI runner for the BOI CASA Forecasting Pipeline.

Trains all models, evaluates, cross-validates, generates reports, and exports CSVs.

Usage
-----
    python run_pipeline.py
    python run_pipeline.py --models ARIMA SARIMA SARIMAX HoltWinters --no-cv
    python run_pipeline.py --models SARIMAX HoltWinters AutoARIMA --no-cv
    python run_pipeline.py --data data/raw/boi_casa_deposits.csv --train-frac 0.75
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_FILE, FORECAST_HORIZON, TRAIN_FRACTION
from src.pipelines.forecasting_pipeline import ForecastingPipeline, FITTERS
from src.utils import get_logger

logger = get_logger("run_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# REPORT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _model_health(cv_mape_mean: float) -> str:
    """
    Map CV MAPE Mean to a human-readable health rating.

    Thresholds (empirical for quarterly banking deposits):
      < 2%   → Excellent
      < 5%   → Good
      < 10%  → Fair
      ≥ 10%  → Poor
    """
    if np.isnan(cv_mape_mean):
        return "N/A"
    if cv_mape_mean < 2.0:
        return "Excellent ⭐"
    if cv_mape_mean < 5.0:
        return "Good ✅"
    if cv_mape_mean < 10.0:
        return "Fair ⚠️"
    return "Poor ❌"


def print_full_report(pipeline: ForecastingPipeline) -> None:
    """Print a comprehensive terminal report after pipeline completes."""
    lb       = pipeline.leaderboard()
    cv       = pipeline.cv_summary()
    insights = pipeline.insights()
    diags    = pipeline.diagnostics

    SEP  = "═" * 72
    SEP2 = "─" * 72

    print(f"\n{SEP}")
    print("  BOI CASA DEPOSIT FORECASTING — PIPELINE REPORT")
    print(f"  Generated: {insights.get('timestamp', '—')}")
    print(SEP)

    # ── Model Leaderboard ─────────────────────────────────────────────────
    print("\n📊  MODEL LEADERBOARD")
    print(SEP2)
    display_cols = [c for c in ["Rank", "MAE_%", "RMSE_%", "MAPE",
                                "R2", "Stability", "Composite_Score"] if c in lb.columns]
    print(lb[display_cols].round(3).to_string())

    # ── CV Summary with Mean ± Std and Model Health ───────────────────────
    if not cv.empty:
        print(f"\n🔄  CROSS-VALIDATION SUMMARY  (TimeSeriesSplit, {pipeline.cv_splits} folds)")
        print(SEP2)
        print(f"  {'Model':<14} {'CV_MAPE_Mean':>13} {'CV_MAPE_Std':>12} "
              f"{'CV_R2_Mean':>11} {'Dir_Acc%':>9} {'Health':>14}")
        print("  " + "-" * 68)
        for model_name, row in cv.iterrows():
            mape_mean = row.get("CV_MAPE_Mean", np.nan)
            mape_std  = row.get("CV_MAPE_Std",  np.nan)
            r2_mean   = row.get("CV_R2_Mean",   np.nan)
            dir_acc   = row.get("CV_Dir_Acc_Mean", np.nan)
            health    = _model_health(mape_mean)

            mape_str  = f"{mape_mean:.3f} ± {mape_std:.3f}%" if not np.isnan(mape_mean) else "  N/A"
            r2_str    = f"{r2_mean:.4f}"  if not np.isnan(r2_mean)  else "  N/A"
            dir_str   = f"{dir_acc:.1f}%" if not np.isnan(dir_acc)  else "  N/A"

            print(f"  {model_name:<14} {mape_str:>25} {r2_str:>11} {dir_str:>9} {health:>14}")

    # ── Per-Model Detail (Adjusted R², Direction Accuracy) ─────────────────
    print(f"\n🔬  PER-MODEL DETAILED METRICS")
    print(SEP2)
    print(f"  {'Model':<14} {'MAPE':>8} {'R²':>8} {'Adj_R²':>9} {'Dir_Acc%':>9} {'Bias_%':>8}")
    print("  " + "-" * 60)
    for name, rec in pipeline.results.items():
        m      = rec["metrics"]
        mape   = m.get("MAPE", float("nan"))
        r2     = m.get("R2", float("nan"))
        adj_r2 = m.get("Adjusted_R2", float("nan"))
        dir_a  = m.get("Direction_Accuracy", float("nan"))
        bias   = m.get("Bias_%", float("nan"))

        def _fmt(v): return f"{v:.3f}" if not (isinstance(v, float) and np.isnan(v)) else " N/A"
        print(f"  {name:<14} {_fmt(mape):>8} {_fmt(r2):>8} {_fmt(adj_r2):>9} "
              f"{_fmt(dir_a):>9} {_fmt(bias):>8}")

    # ── Best Model highlight ──────────────────────────────────────────────
    best = insights["best_model"]
    bm   = insights["best_metrics"]
    print(f"\n{SEP2}")
    print(f"  🏆  BEST MODEL: {best}")
    print(f"      MAPE            : {bm['MAPE']:.3f}%")
    print(f"      RMSE %          : {bm['RMSE_%']:.3f}%")
    print(f"      R²              : {bm['R2']:.4f}")
    print(f"      Adjusted R²     : {bm.get('Adjusted_R2', float('nan')):.4f}")
    print(f"      Direction Acc.  : {bm.get('Direction_Accuracy', float('nan')):.1f}%")
    print(f"      Stability       : {bm['Stability']:.1f}/100")
    print(f"      Bias %          : {bm['Bias_%']:+.3f}%")
    print(f"  🔒  MOST STABLE     : {insights['stable_model']}")
    print(SEP2)

    # ── Diagnostics health ────────────────────────────────────────────────
    print(f"\n🧪  RESIDUAL DIAGNOSTIC HEALTH SCORES")
    print(SEP2)
    for name, diag in diags.items():
        h     = diag.get("health_score", 0)
        label = diag.get("health_label", "—")
        bar   = "█" * int(h / 10) + "░" * (10 - int(h / 10))
        print(f"  {name:<14}  [{bar}]  {h:5.1f}/100  {label}")

    # ── Executive summary ─────────────────────────────────────────────────
    print(f"\n💡  EXECUTIVE SUMMARY")
    print(SEP2)
    summary = insights.get("executive_summary", "—")
    words = summary.split()
    line, lines = "", []
    for w in words:
        if len(line) + len(w) + 1 > 68:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    for l in lines:
        print(f"  {l}")

    # ── Recommendations ───────────────────────────────────────────────────
    print(f"\n📋  RECOMMENDATIONS")
    print(SEP2)
    for rec in insights.get("model_recommendations", []):
        print(f"  {rec.replace('**', '')}")

    # ── Banking insights ──────────────────────────────────────────────────
    print(f"\n🏦  BANKING INSIGHTS")
    print(SEP2)
    for ins in insights.get("banking_insights", []):
        print(f"  {ins.replace('**', '')}")

    print(f"\n{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BOI CASA Deposit — Automated Forecasting Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data",      default=DATA_FILE,
                        help=f"Path to CSV data file (default: {DATA_FILE})")
    parser.add_argument("--models",    nargs="+", default=list(FITTERS.keys()),
                        choices=list(FITTERS.keys()), metavar="MODEL",
                        help=f"Models to train. Choices: {list(FITTERS.keys())}")
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRACTION,
                        help=f"Train/test split fraction (default: {TRAIN_FRACTION})")
    parser.add_argument("--no-cv",     action="store_true",
                        help="Skip cross-validation")
    parser.add_argument("--cv-splits", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--out-dir",   default=str(ROOT / "data" / "processed"),
                        help="Output directory for CSV results")
    parser.add_argument("--quiet",     action="store_true",
                        help="Suppress terminal report")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  BOI CASA FORECASTING PIPELINE — STARTING")
    logger.info("=" * 60)
    logger.info(f"  Data       : {args.data}")
    logger.info(f"  Models     : {args.models}")
    logger.info(f"  Train frac : {args.train_frac}")
    logger.info(f"  CV splits  : {0 if args.no_cv else args.cv_splits}")
    logger.info(f"  Output dir : {args.out_dir}")
    logger.info("=" * 60)

    pipeline = ForecastingPipeline(
        data_path=args.data,
        train_frac=args.train_frac,
        models=args.models,
        run_cv=not args.no_cv,
        cv_splits=args.cv_splits,
    )

    pipeline.run()
    saved = pipeline.save_results(out_dir=args.out_dir)

    if not args.quiet:
        print_full_report(pipeline)

    logger.info("Pipeline complete. Files saved:")
    for label, path in saved.items():
        logger.info(f"  {label}: {path}")


if __name__ == "__main__":
    main()
