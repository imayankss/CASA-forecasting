"""
drift_detection.py — Forecast drift & data-drift detection for CASA deposits.

Implements two complementary detectors:
  1. KS-Drift       — Kolmogorov-Smirnov test comparing train vs live distributions
  2. CUSUM           — Cumulative sum control chart for mean-shift detection
  3. Residual-Drift  — detects when live residuals exceed historical bounds

Usage
-----
    from src.advanced.drift_detection import DriftDetector

    detector = DriftDetector(train_series, residuals_history)
    report   = detector.check(live_series, live_residuals)
    print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.logger import get_logger

_log = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# DRIFT REPORT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DriftReport:
    """Container for all drift test results."""
    ks_statistic:     float
    ks_p_value:       float
    ks_drift:         bool
    cusum_signal:     bool
    cusum_max:        float
    residual_drift:   bool
    residual_z:       float
    overall_drift:    bool
    severity:         str    # "None" | "Mild" | "Moderate" | "Severe"

    def summary(self) -> str:
        sev_emoji = {"None": "✅", "Mild": "⚠️", "Moderate": "🔶", "Severe": "🚨"}
        lines = [
            "══════════════════════════════════════════",
            "  DRIFT DETECTION REPORT",
            "══════════════════════════════════════════",
            f"  Overall Drift   : {'YES' if self.overall_drift else 'NO'}  "
            f"{sev_emoji.get(self.severity, '')} {self.severity}",
            "",
            "  KS Distribution Test",
            f"    Statistic  : {self.ks_statistic:.4f}",
            f"    p-value    : {self.ks_p_value:.4f}",
            f"    Drift      : {'YES ⚠️' if self.ks_drift else 'NO ✅'}",
            "",
            "  CUSUM Control Chart",
            f"    Max Signal  : {self.cusum_max:.4f}",
            f"    Alert       : {'YES ⚠️' if self.cusum_signal else 'NO ✅'}",
            "",
            "  Residual Drift",
            f"    Z-Score    : {self.residual_z:.4f}",
            f"    Drift      : {'YES ⚠️' if self.residual_drift else 'NO ✅'}",
            "",
            "  Recommendation:",
        ]
        if self.severity == "None":
            lines.append("    Model is performing within expected bounds. No action needed.")
        elif self.severity == "Mild":
            lines.append("    Monitor closely. Consider re-training at next quarterly update.")
        elif self.severity == "Moderate":
            lines.append("    Drift detected. Schedule re-training before next forecast cycle.")
        else:
            lines.append("    SEVERE drift. Immediate re-training recommended.")
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# DRIFT DETECTOR
# ═════════════════════════════════════════════════════════════════════════════

class DriftDetector:
    """
    Statistical drift detector for CASA deposit forecasting.

    Parameters
    ----------
    train_series       : historical training series used to fit models
    train_residuals    : in-sample residuals from the best model
    ks_alpha           : KS test significance level (default 0.05)
    cusum_threshold    : CUSUM alert threshold in standard deviations (default 4)
    residual_z_thresh  : residual z-score alert threshold (default 3.0)
    """

    def __init__(
        self,
        train_series:    pd.Series,
        train_residuals: np.ndarray,
        ks_alpha:        float = 0.05,
        cusum_threshold: float = 4.0,
        residual_z_thresh: float = 3.0,
    ) -> None:
        self.train_series    = train_series
        self.train_residuals = np.array(train_residuals, dtype=float)
        self.ks_alpha        = ks_alpha
        self.cusum_threshold = cusum_threshold
        self.residual_z_thresh = residual_z_thresh

        # Pre-compute reference statistics
        self._ref_mean = float(np.mean(self.train_residuals))
        self._ref_std  = float(np.std(self.train_residuals)) or 1.0

    # ── Individual tests ──────────────────────────────────────────────────

    def _ks_test(self, live_series: pd.Series) -> Tuple[float, float, bool]:
        """KS test: compare live deposit distribution vs training distribution."""
        ks_stat, ks_p = stats.ks_2samp(
            self.train_series.dropna().values,
            live_series.dropna().values,
        )
        return float(ks_stat), float(ks_p), bool(ks_p < self.ks_alpha)

    def _cusum_test(self, live_residuals: np.ndarray) -> Tuple[bool, float]:
        """CUSUM control chart on live residuals."""
        r      = np.array(live_residuals, dtype=float)
        mu     = self._ref_mean
        sigma  = self._ref_std
        k      = 0.5   # allowance
        cusum_pos = np.zeros(len(r))
        cusum_neg = np.zeros(len(r))
        for i in range(1, len(r)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + (r[i] - mu) / sigma - k)
            cusum_neg[i] = max(0, cusum_neg[i-1] - (r[i] - mu) / sigma - k)
        max_cusum = float(max(cusum_pos.max(), cusum_neg.max()))
        return max_cusum > self.cusum_threshold, max_cusum

    def _residual_drift_test(self, live_residuals: np.ndarray) -> Tuple[bool, float]:
        """Check if mean of live residuals deviates significantly from training."""
        r = np.array(live_residuals, dtype=float)
        if len(r) == 0:
            return False, 0.0
        live_mean = float(np.mean(r))
        z = abs(live_mean - self._ref_mean) / self._ref_std
        return z > self.residual_z_thresh, round(z, 4)

    # ── Master check ──────────────────────────────────────────────────────

    def check(
        self,
        live_series:    pd.Series,
        live_residuals: Optional[np.ndarray] = None,
    ) -> DriftReport:
        """
        Run all drift tests against live data.

        Parameters
        ----------
        live_series    : recent deposit observations (e.g. last quarter)
        live_residuals : residuals from live forecast (optional)

        Returns
        -------
        DriftReport with per-test results and overall severity.
        """
        live_res = live_residuals if live_residuals is not None else np.array([0.0])

        ks_stat, ks_p, ks_drift   = self._ks_test(live_series)
        cusum_signal, cusum_max    = self._cusum_test(live_res)
        res_drift, res_z           = self._residual_drift_test(live_res)

        # Aggregate severity
        n_alerts = sum([ks_drift, cusum_signal, res_drift])
        if n_alerts == 0:
            severity = "None"
        elif n_alerts == 1:
            severity = "Mild"
        elif n_alerts == 2:
            severity = "Moderate"
        else:
            severity = "Severe"

        _log.info(
            "Drift check → KS=%s | CUSUM=%s | ResidualZ=%s | Severity=%s",
            "DRIFT" if ks_drift else "OK",
            "ALERT" if cusum_signal else "OK",
            "DRIFT" if res_drift else "OK",
            severity,
        )

        return DriftReport(
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
            ks_drift=ks_drift,
            cusum_signal=cusum_signal,
            cusum_max=cusum_max,
            residual_drift=res_drift,
            residual_z=res_z,
            overall_drift=n_alerts > 0,
            severity=severity,
        )
