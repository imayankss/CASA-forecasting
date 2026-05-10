"""
logger.py — Centralised logging for the BOI CASA Forecasting Platform.

Features
--------
* Colour-coded console output (DEBUG→CRITICAL)
* Rotating file handler  (10 MB × 5 backups)
* Per-module log levels via LOG_LEVELS env-var
* Context-aware log records (model name, phase)
* Singleton registry so each name gets one handler set

Usage
-----
    from src.logger import get_logger
    log = get_logger(__name__)
    log.info("Training ARIMA …")

    # Context-enhanced logging
    from src.logger import PipelineLogger
    with PipelineLogger("SARIMAX") as log:
        log.info("Fitting model …")
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Generator, Optional

# ── ANSI colour codes ─────────────────────────────────────────────────────────
_COLOURS = {
    "DEBUG":    "\033[36m",    # cyan
    "INFO":     "\033[32m",    # green
    "WARNING":  "\033[33m",    # amber
    "ERROR":    "\033[31m",    # red
    "CRITICAL": "\033[35m",    # magenta
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"

# ── Default settings  ─────────────────────────────────────────────────────────
_ROOT_DIR   = Path(__file__).resolve().parents[1]
_LOG_DIR    = _ROOT_DIR / "reports" / "logs"
_LOG_FILE   = _LOG_DIR / "forecasting.log"
_DEFAULT_LVL = os.getenv("LOG_LEVEL", "INFO").upper()


# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM FORMATTER
# ═════════════════════════════════════════════════════════════════════════════

class _ColourFormatter(logging.Formatter):
    """Adds ANSI colours to level names when writing to a TTY."""

    _FMT = (
        "%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s"
    )
    _DATE = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        record.name = record.name.split(".")[-1][:28]   # shorten module path
        lvl    = record.levelname
        colour = _COLOURS.get(lvl, "")
        record.levelname = f"{colour}{_BOLD}{lvl:<8}{_RESET}"
        return logging.Formatter(self._FMT, datefmt=self._DATE).format(record)


class _PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no ANSI codes)."""

    _FMT  = "%(asctime)s  [%(levelname)-8s]  %(name)-28s  %(message)s"
    _DATE = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        record.name = record.name.split(".")[-1][:28]
        return logging.Formatter(self._FMT, datefmt=self._DATE).format(record)


# ═════════════════════════════════════════════════════════════════════════════
# REGISTRY  (singleton — one handler set per logger name)
# ═════════════════════════════════════════════════════════════════════════════

_REGISTRY: set[str] = set()


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Return a fully configured logger.

    Parameters
    ----------
    name      : logger name (use ``__name__``)
    level     : override log level ('DEBUG', 'INFO', …). Falls back to
                the LOG_LEVEL environment variable, then 'INFO'.
    log_file  : path to rotating log file.  Defaults to
                ``reports/logs/forecasting.log``.
    console   : whether to attach a streaming console handler.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if name in _REGISTRY:          # already configured
        return logger

    _REGISTRY.add(name)

    # Resolve level
    resolved_level = getattr(
        logging,
        (level or os.getenv("LOG_LEVEL", "INFO")).upper(),
        logging.INFO,
    )
    logger.setLevel(resolved_level)
    logger.propagate = False

    # ── Console handler ───────────────────────────────────────────────────
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(resolved_level)
        use_colour = sys.stdout.isatty() and os.name != "nt"
        ch.setFormatter(_ColourFormatter() if use_colour else _PlainFormatter())
        logger.addHandler(ch)

    # ── Rotating file handler ─────────────────────────────────────────────
    target = Path(log_file or _LOG_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        target,
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)       # file always captures everything
    fh.setFormatter(_PlainFormatter())
    logger.addHandler(fh)

    return logger


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE CONTEXT LOGGER
# ═════════════════════════════════════════════════════════════════════════════

class PipelineLogger:
    """
    Context manager that logs phase entry/exit with elapsed time.

    Usage
    -----
        with PipelineLogger("ARIMA training") as log:
            log.info("Fitting model …")
        # → logs "▶  ARIMA training" on enter
        # → logs "✓  ARIMA training  (2.31 s)" on exit
    """

    def __init__(self, phase: str, logger_name: str = "pipeline") -> None:
        self.phase  = phase
        self._log   = get_logger(logger_name)
        self._start = 0.0

    def __enter__(self) -> logging.Logger:
        self._start = time.perf_counter()
        self._log.info("▶  %s", self.phase)
        return self._log

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = time.perf_counter() - self._start
        if exc_type:
            self._log.error("✗  %s  FAILED after %.2f s  — %s", self.phase, elapsed, exc_val)
        else:
            self._log.info("✓  %s  (%.2f s)", self.phase, elapsed)
        return False   # never suppress exceptions


# ═════════════════════════════════════════════════════════════════════════════
# TIMING DECORATOR
# ═════════════════════════════════════════════════════════════════════════════

def log_timing(logger: Optional[logging.Logger] = None):
    """
    Decorator that logs a function's execution time at INFO level.

    Example
    -------
        @log_timing(get_logger(__name__))
        def train_model(…): …
    """
    import functools

    def decorator(fn):
        _log = logger or get_logger(fn.__module__)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0     = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            _log.debug("⏱  %s  →  %.3f s", fn.__qualname__, elapsed)
            return result

        return wrapper
    return decorator
