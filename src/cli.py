"""
cli.py — Reusable CLI argument-parser factory for BOI CASA Forecasting scripts.

Each entry-point script calls ``build_parser()``, optionally extends the
returned parser with script-specific arguments, then calls ``parse_and_validate()``.

Example
-------
    from src.cli import build_parser, parse_and_validate

    parser = build_parser(description="My script")
    parser.add_argument("--extra", default="value")
    args = parse_and_validate(parser)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from src.config import DATA_FILE, FORECAST_HORIZON, TRAIN_FRACTION
from src.exceptions import ConfigurationError


# ═════════════════════════════════════════════════════════════════════════════
# PARSER FACTORY
# ═════════════════════════════════════════════════════════════════════════════

def build_parser(
    description: str = "BOI CASA Deposit Forecasting",
    include_models: bool = True,
    include_report: bool = False,
) -> argparse.ArgumentParser:
    """
    Build and return a base argument parser shared by all CLI entry points.

    Parameters
    ----------
    description    : Help text shown at the top of ``--help`` output.
    include_models : Whether to include model-selection arguments.
    include_report : Whether to include report-format arguments.

    Returns
    -------
    argparse.ArgumentParser — extend before calling ``parse_and_validate()``.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment variables\n"
            "  BOI_DATA_FILE    override --data default\n"
            "  FORECAST_HORIZON override --horizon default\n"
            "  LOG_LEVEL        DEBUG | INFO | WARNING | ERROR\n"
        ),
    )

    # ── Data arguments ────────────────────────────────────────────────────
    data_grp = parser.add_argument_group("Data")
    data_grp.add_argument(
        "--data",
        metavar="PATH",
        default=DATA_FILE,
        help=f"Path to CASA deposit CSV (default: {DATA_FILE})",
    )
    data_grp.add_argument(
        "--target",
        metavar="COL",
        default="Deposit_Amount",
        help="Target column name (default: Deposit_Amount)",
    )
    data_grp.add_argument(
        "--date-col",
        metavar="COL",
        default="Quarter_Year",
        help="Date/period column name (default: Quarter_Year)",
    )

    # ── Forecast arguments ────────────────────────────────────────────────
    fc_grp = parser.add_argument_group("Forecasting")
    fc_grp.add_argument(
        "--horizon",
        metavar="N",
        type=int,
        default=FORECAST_HORIZON,
        help=f"Forecast horizon in periods (default: {FORECAST_HORIZON})",
    )
    fc_grp.add_argument(
        "--train-frac",
        metavar="F",
        type=float,
        default=TRAIN_FRACTION,
        help=f"Train/test split fraction (default: {TRAIN_FRACTION})",
    )
    fc_grp.add_argument(
        "--seasonal-period",
        metavar="P",
        type=int,
        default=4,
        help="Seasonal period (default: 4 for quarterly data)",
    )

    # ── Model arguments ───────────────────────────────────────────────────
    if include_models:
        _KNOWN_MODELS = ["ARIMA", "SARIMA", "SARIMAX", "AutoARIMA", "HoltWinters", "Prophet"]
        model_grp = parser.add_argument_group("Models")
        model_grp.add_argument(
            "--models",
            nargs="+",
            metavar="MODEL",
            default=_KNOWN_MODELS,
            choices=_KNOWN_MODELS,
            help=f"Models to train. Choices: {_KNOWN_MODELS}",
        )
        model_grp.add_argument(
            "--no-cv",
            action="store_true",
            help="Skip walk-forward cross-validation",
        )
        model_grp.add_argument(
            "--cv-splits",
            metavar="N",
            type=int,
            default=5,
            help="Number of CV folds (default: 5)",
        )

    # ── Output arguments ──────────────────────────────────────────────────
    out_grp = parser.add_argument_group("Output")
    out_grp.add_argument(
        "--out-dir",
        metavar="DIR",
        default="data/processed",
        help="Directory for CSV / report outputs (default: data/processed)",
    )
    out_grp.add_argument(
        "--log-level",
        metavar="LEVEL",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level (default: INFO)",
    )
    out_grp.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress terminal summary report",
    )

    # ── Report arguments ──────────────────────────────────────────────────
    if include_report:
        rep_grp = parser.add_argument_group("Reports")
        rep_grp.add_argument(
            "--formats",
            nargs="+",
            metavar="FMT",
            default=["pdf", "html", "markdown"],
            choices=["pdf", "html", "markdown"],
            help="Report formats to generate (default: pdf html markdown)",
        )
        rep_grp.add_argument(
            "--report-dir",
            metavar="DIR",
            default="reports",
            help="Directory for generated reports (default: reports)",
        )
        rep_grp.add_argument(
            "--author",
            metavar="NAME",
            default="Data Science Team",
            help="Author name for PDF / HTML reports",
        )

    return parser


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def parse_and_validate(
    parser: argparse.ArgumentParser,
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """
    Parse argv and run post-parse validation.

    Raises
    ------
    ConfigurationError  if any argument value is invalid.
    SystemExit          (via argparse) on ``--help`` or parse error.
    """
    args = parser.parse_args(argv)

    # Data file must exist
    if not Path(args.data).exists():
        raise ConfigurationError(
            "data",
            f"File not found: '{args.data}'. "
            "Use --data to specify a valid CSV path.",
        )

    # Train fraction in (0, 1)
    if not (0.0 < args.train_frac < 1.0):
        raise ConfigurationError(
            "train_frac",
            f"Must be in (0, 1). Got: {args.train_frac}",
        )

    # Horizon must be positive
    if args.horizon < 1:
        raise ConfigurationError(
            "horizon",
            f"Must be ≥ 1. Got: {args.horizon}",
        )

    # Seasonal period must be positive
    if args.seasonal_period < 1:
        raise ConfigurationError(
            "seasonal_period",
            f"Must be ≥ 1. Got: {args.seasonal_period}",
        )

    # Create output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    return args


# ═════════════════════════════════════════════════════════════════════════════
# HELPER  — print a clean startup banner
# ═════════════════════════════════════════════════════════════════════════════

def print_banner(script_name: str, args: argparse.Namespace) -> None:
    """Print a formatted startup banner to stdout."""
    SEP = "═" * 60
    print(f"\n{SEP}")
    print(f"  BOI CASA FORECASTING — {script_name.upper()}")
    print(SEP)
    for key, val in vars(args).items():
        print(f"  {key:<22} {val}")
    print(f"{SEP}\n")
