"""
run_reports.py — CLI entry point for BOI CASA Forecasting Report Generation.

Trains all models, runs the full evaluation pipeline, then generates
PDF, HTML, and/or Markdown reports in the reports/ directory.

Usage
-----
    python run_reports.py
    python run_reports.py --formats pdf html
    python run_reports.py --formats markdown --models SARIMAX AutoARIMA --no-cv
    python run_reports.py --report-dir reports --author "Your Name"
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.cli import build_parser, parse_and_validate, print_banner
from src.logger import get_logger, PipelineLogger
from src.exceptions import ForecastingError, ReportError
from src.pipelines.forecasting_pipeline import ForecastingPipeline
from src.reporting.insights_engine import build_full_insights

_log = get_logger("run_reports")


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL REPORT GENERATORS (wrapped with exception handling)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_pdf(pipeline, full_insights: dict, report_dir: str) -> Path | None:
    from src.reporting.pdf_report import generate_pdf_report
    try:
        with PipelineLogger("PDF report", "run_reports") as log:
            path = generate_pdf_report(pipeline, full_insights, out_dir=report_dir)
        return path
    except ReportError as e:
        _log.error("PDF generation failed: %s", e)
        return None


def _generate_html(pipeline, full_insights: dict, report_dir: str) -> Path | None:
    from src.reporting.html_report import generate_html_report
    try:
        with PipelineLogger("HTML report", "run_reports") as log:
            path = generate_html_report(pipeline, full_insights, out_dir=report_dir)
        return path
    except ReportError as e:
        _log.error("HTML generation failed: %s", e)
        return None


def _generate_markdown(pipeline, full_insights: dict, report_dir: str) -> Path | None:
    from src.reporting.markdown_report import generate_markdown_report
    try:
        with PipelineLogger("Markdown report", "run_reports") as log:
            path = generate_markdown_report(pipeline, full_insights, out_dir=report_dir)
        return path
    except ReportError as e:
        _log.error("Markdown generation failed: %s", e)
        return None


GENERATORS = {
    "pdf":      _generate_pdf,
    "html":     _generate_html,
    "markdown": _generate_markdown,
}


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _print_report_summary(saved: dict[str, Path | None]) -> None:
    SEP = "═" * 62
    print(f"\n{SEP}")
    print("  REPORT GENERATION COMPLETE")
    print(SEP)
    for fmt, path in saved.items():
        if path and path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✅  {fmt.upper():<12}  {path.name}  ({size_kb:.1f} KB)")
            print(f"       → {path}")
        else:
            print(f"  ❌  {fmt.upper():<12}  Failed or skipped.")
    print(f"{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Parse arguments ───────────────────────────────────────────────────
    parser = build_parser(
        description="BOI CASA Deposit Forecasting — Report Generator",
        include_models=True,
        include_report=True,
    )
    try:
        args = parse_and_validate(parser)
    except ForecastingError as e:
        _log.error("Configuration error: %s", e)
        sys.exit(1)

    if not args.quiet:
        print_banner("REPORT GENERATOR", args)

    # ── Run pipeline ──────────────────────────────────────────────────────
    _log.info("Step 1/3 — Running forecasting pipeline …")
    try:
        pipeline = ForecastingPipeline(
            data_path=args.data,
            train_frac=args.train_frac,
            models=args.models,
            run_cv=not args.no_cv,
            cv_splits=args.cv_splits,
        )
        pipeline.run()
        pipeline.save_results(out_dir=args.out_dir)
    except ForecastingError as e:
        _log.error("Pipeline failed: %s", e)
        sys.exit(1)
    except Exception as e:
        _log.error("Unexpected pipeline error: %s", e)
        sys.exit(1)

    # ── Build insights ────────────────────────────────────────────────────
    _log.info("Step 2/3 — Building insights …")
    try:
        full_insights = build_full_insights(pipeline)
    except Exception as e:
        _log.error("Insights generation failed: %s", e)
        sys.exit(1)

    # ── Generate reports ──────────────────────────────────────────────────
    _log.info("Step 3/3 — Generating reports: %s", args.formats)
    report_dir = args.report_dir
    saved: dict[str, Path | None] = {}

    for fmt in args.formats:
        gen_fn = GENERATORS.get(fmt)
        if gen_fn is None:
            _log.warning("Unknown format '%s' — skipping.", fmt)
            saved[fmt] = None
            continue
        saved[fmt] = gen_fn(pipeline, full_insights, report_dir)

    # ── Print summary ─────────────────────────────────────────────────────
    if not args.quiet:
        _print_report_summary(saved)

    # Exit with error code if any report failed
    failed = [f for f, p in saved.items() if p is None]
    if failed:
        _log.warning("Some reports failed: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
