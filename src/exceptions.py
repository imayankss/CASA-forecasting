"""
exceptions.py — Custom exception hierarchy for BOI CASA Forecasting Platform.

All platform exceptions inherit from ``ForecastingError`` so callers can
catch either specific or broad errors in a single ``except`` clause.

Exception Tree
--------------
ForecastingError
├── DataError
│   ├── DataNotFoundError
│   ├── DataValidationError
│   └── DataFormatError
├── ModelError
│   ├── ModelFitError
│   ├── ModelForecastError
│   └── ModelNotRegisteredError
├── EvaluationError
│   └── InsufficientDataError
├── ReportError
│   ├── ReportGenerationError
│   └── ReportExportError
└── ConfigurationError
"""

from __future__ import annotations
from typing import Optional


# ═════════════════════════════════════════════════════════════════════════════
# BASE
# ═════════════════════════════════════════════════════════════════════════════

class ForecastingError(Exception):
    """Base exception for all BOI CASA Forecasting Platform errors."""

    def __init__(self, message: str, context: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        base = self.message
        if self.context:
            details = "  |  ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{base}  [{details}]"
        return base


# ═════════════════════════════════════════════════════════════════════════════
# DATA ERRORS
# ═════════════════════════════════════════════════════════════════════════════

class DataError(ForecastingError):
    """Base class for data-layer errors."""


class DataNotFoundError(DataError):
    """Raised when a required data file or column is missing."""

    def __init__(self, path_or_column: str) -> None:
        super().__init__(
            f"Data source not found: '{path_or_column}'",
            {"source": path_or_column},
        )


class DataValidationError(DataError):
    """Raised when loaded data fails validation checks."""

    def __init__(self, reason: str, n_rows: int = 0) -> None:
        super().__init__(
            f"Data validation failed: {reason}",
            {"reason": reason, "n_rows": n_rows},
        )


class DataFormatError(DataError):
    """Raised when the data file format cannot be parsed."""

    def __init__(self, filename: str, detail: str) -> None:
        super().__init__(
            f"Cannot parse '{filename}': {detail}",
            {"file": filename, "detail": detail},
        )


# ═════════════════════════════════════════════════════════════════════════════
# MODEL ERRORS
# ═════════════════════════════════════════════════════════════════════════════

class ModelError(ForecastingError):
    """Base class for model-layer errors."""


class ModelFitError(ModelError):
    """Raised when a model fails to fit the training data."""

    def __init__(self, model_name: str, cause: str) -> None:
        super().__init__(
            f"Model '{model_name}' failed to fit: {cause}",
            {"model": model_name, "cause": cause},
        )
        self.model_name = model_name


class ModelForecastError(ModelError):
    """Raised when a fitted model fails to generate a forecast."""

    def __init__(self, model_name: str, steps: int, cause: str) -> None:
        super().__init__(
            f"Model '{model_name}' forecast({steps} steps) failed: {cause}",
            {"model": model_name, "steps": steps, "cause": cause},
        )
        self.model_name = model_name


class ModelNotRegisteredError(ModelError):
    """Raised when an unknown model name is requested."""

    def __init__(self, model_name: str, available: list[str]) -> None:
        super().__init__(
            f"Model '{model_name}' is not registered.",
            {"requested": model_name, "available": ", ".join(available)},
        )


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION ERRORS
# ═════════════════════════════════════════════════════════════════════════════

class EvaluationError(ForecastingError):
    """Base class for evaluation-layer errors."""


class InsufficientDataError(EvaluationError):
    """Raised when there are too few observations for a meaningful evaluation."""

    def __init__(self, n_available: int, n_required: int) -> None:
        super().__init__(
            f"Insufficient data: {n_available} observations available, "
            f"{n_required} required.",
            {"n_available": n_available, "n_required": n_required},
        )


# ═════════════════════════════════════════════════════════════════════════════
# REPORT ERRORS
# ═════════════════════════════════════════════════════════════════════════════

class ReportError(ForecastingError):
    """Base class for report-layer errors."""


class ReportGenerationError(ReportError):
    """Raised when report content cannot be assembled."""

    def __init__(self, report_type: str, reason: str) -> None:
        super().__init__(
            f"{report_type} report generation failed: {reason}",
            {"type": report_type, "reason": reason},
        )


class ReportExportError(ReportError):
    """Raised when a report cannot be written to disk."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            f"Failed to export report to '{path}': {reason}",
            {"path": path, "reason": reason},
        )


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION ERROR
# ═════════════════════════════════════════════════════════════════════════════

class ConfigurationError(ForecastingError):
    """Raised when the project configuration is invalid or incomplete."""

    def __init__(self, param: str, detail: str) -> None:
        super().__init__(
            f"Configuration error — '{param}': {detail}",
            {"param": param, "detail": detail},
        )
