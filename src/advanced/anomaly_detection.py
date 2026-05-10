"""
anomaly_detection.py — Anomaly detection for CASA deposit time series.

Implements three complementary detectors:
  1. IQR Fence         — classical box-plot outlier detection
  2. Z-Score           — standard deviation gating
  3. Rolling-Sigma     — adaptive rolling-window z-score

Each detector returns a labelled DataFrame so results can be overlaid
on charts or fed into alert pipelines.

Usage
-----
    from src.advanced.anomaly_detection import detect_anomalies, AnomalyReport

    report = detect_anomalies(series, methods=["iqr", "zscore", "rolling"])
    print(report.summary())
    flagged = report.flagged_periods()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from src.logger import get_logger

_log = get_logger(__name__)

Method = Literal["iqr", "zscore", "rolling"]


# ═════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL DETECTORS
# ═════════════════════════════════════════════════════════════════════════════

def _iqr_detector(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    IQR fence detector.

    Flags values outside [Q1 - k·IQR, Q3 + k·IQR].

    Returns a boolean Series (True = anomaly).
    """
    q1  = series.quantile(0.25)
    q3  = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (series < lower) | (series > upper)


def _zscore_detector(series: pd.Series, threshold: float = 2.5) -> pd.Series:
    """
    Global Z-score detector.

    Flags |z| > threshold (default 2.5σ).

    Returns a boolean Series (True = anomaly).
    """
    mu  = series.mean()
    std = series.std()
    z   = (series - mu) / (std + 1e-9)
    return z.abs() > threshold


def _rolling_detector(
    series: pd.Series,
    window: int = 6,
    threshold: float = 2.0,
) -> pd.Series:
    """
    Rolling-window z-score detector.

    Compares each point against its local rolling mean and std.
    Good for series with a structural upward trend.

    Returns a boolean Series (True = anomaly).
    """
    roll_mean = series.rolling(window, min_periods=3).mean()
    roll_std  = series.rolling(window, min_periods=3).std().replace(0, 1e-9)
    z_roll    = (series - roll_mean) / roll_std
    return z_roll.abs() > threshold


# ═════════════════════════════════════════════════════════════════════════════
# ANOMALY REPORT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class AnomalyReport:
    """Structured container for multi-method anomaly detection results."""

    series:   pd.Series
    results:  Dict[str, pd.Series] = field(default_factory=dict)   # method → bool series
    scores:   pd.DataFrame         = field(default_factory=pd.DataFrame)

    # ── Convenience accessors ─────────────────────────────────────────────

    def flagged_periods(self, min_methods: int = 1) -> pd.DataFrame:
        """
        Return periods flagged by at least *min_methods* detectors.

        Parameters
        ----------
        min_methods : minimum number of methods that must agree (default 1)
        """
        if not self.results:
            return pd.DataFrame()
        vote = pd.DataFrame(self.results).astype(int).sum(axis=1)
        mask = vote >= min_methods
        flagged = self.series[mask].to_frame("Deposit_Amount")
        flagged["votes"]  = vote[mask]
        flagged["methods"] = [
            ", ".join(m for m, s in self.results.items() if s[idx])
            for idx in flagged.index
        ]
        return flagged

    def anomaly_rate(self) -> Dict[str, float]:
        """Return detection rate (%) per method."""
        n = len(self.series)
        return {
            method: round(flags.sum() / n * 100, 2)
            for method, flags in self.results.items()
        }

    def consensus_flags(self) -> pd.Series:
        """Return boolean Series — True where ALL methods agree it's an anomaly."""
        if not self.results:
            return pd.Series(False, index=self.series.index)
        df = pd.DataFrame(self.results)
        return df.all(axis=1)

    def summary(self) -> str:
        n       = len(self.series)
        rates   = self.anomaly_rate()
        flagged = self.flagged_periods(min_methods=1)
        consenus = self.consensus_flags().sum()
        lines = [
            "╔══════════════════════════════════════════╗",
            "║   ANOMALY DETECTION REPORT               ║",
            "╚══════════════════════════════════════════╝",
            f"  Series length   : {n} periods",
            f"  Consensus flags : {consenus} periods (all methods agree)",
            "",
            "  Method rates:",
        ]
        for method, rate in rates.items():
            lines.append(f"    {method:<14}  {rate:.1f}%")
        if not flagged.empty:
            lines += ["", "  Flagged periods:"]
            for idx, row in flagged.iterrows():
                lines.append(
                    f"    {str(idx)[:10]}  ₹{row['Deposit_Amount']/1e7:.2f}Cr"
                    f"  [{row['methods']}]"
                )
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def detect_anomalies(
    series:      pd.Series,
    methods:     List[Method] = ("iqr", "zscore", "rolling"),
    iqr_k:       float = 1.5,
    zscore_thr:  float = 2.5,
    rolling_win: int   = 6,
    rolling_thr: float = 2.0,
) -> AnomalyReport:
    """
    Run one or more anomaly detectors on a CASA deposit time series.

    Parameters
    ----------
    series      : pd.Series of deposit amounts (numeric, sorted by date)
    methods     : list of detectors to run
    iqr_k       : IQR multiplier (default 1.5)
    zscore_thr  : global Z-score threshold (default 2.5)
    rolling_win : rolling window size in periods (default 6)
    rolling_thr : rolling Z-score threshold (default 2.0)

    Returns
    -------
    AnomalyReport
    """
    s = series.dropna()
    results: Dict[str, pd.Series] = {}

    for method in methods:
        if method == "iqr":
            results["iqr"]     = _iqr_detector(s, multiplier=iqr_k)
        elif method == "zscore":
            results["zscore"]  = _zscore_detector(s, threshold=zscore_thr)
        elif method == "rolling":
            results["rolling"] = _rolling_detector(s, window=rolling_win,
                                                    threshold=rolling_thr)
        else:
            _log.warning("Unknown anomaly method '%s' — skipped.", method)

    n_flagged = sum(
        results[m].sum() for m in results
    ) // max(len(results), 1)
    _log.info(
        "Anomaly detection complete. Methods: %s  |  ~%d periods flagged",
        list(results.keys()), n_flagged,
    )

    return AnomalyReport(series=s, results=results)
