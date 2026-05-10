"""
confidence_scoring.py — Forecast confidence scorer for CASA deposit predictions.

Produces a 0-100 confidence score per forecast period based on:
  - Model MAPE (lower error → higher confidence)
  - Residual stability (lower variance → higher confidence)
  - Forecast horizon (further ahead → lower confidence, decay function)
  - CI width relative to forecast value (narrower band → higher confidence)
  - Drift status (drift detected → confidence penalty)

Usage
-----
    from src.advanced.confidence_scoring import ForecastConfidenceScorer

    scorer = ForecastConfidenceScorer(metrics, residuals)
    df     = scorer.score(forecast_series, ci_df, horizon=4)
    print(df)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.logger import get_logger

_log = get_logger(__name__)


class ForecastConfidenceScorer:
    """
    Compute per-period confidence scores for a model's forecast.

    Parameters
    ----------
    metrics    : model evaluation metrics dict (from compute_full_metrics)
    residuals  : in-sample residuals array
    drift_penalty : confidence penalty (0-30) when drift is detected
    """

    def __init__(
        self,
        metrics:       Dict[str, float],
        residuals:     np.ndarray,
        drift_penalty: float = 15.0,
    ) -> None:
        self.metrics       = metrics
        self.residuals     = np.array(residuals, dtype=float)
        self.drift_penalty = drift_penalty

    # ── Component scores ─────────────────────────────────────────────────

    def _mape_score(self) -> float:
        """Convert MAPE to a 0-40 score (lower MAPE → higher score)."""
        mape = self.metrics.get("MAPE", 50.0)
        # Sigmoid-style mapping: MAPE=0→40, MAPE=5→30, MAPE=10→20, MAPE=20→10
        score = 40.0 * np.exp(-0.07 * max(mape, 0))
        return round(float(np.clip(score, 0, 40)), 2)

    def _stability_score(self) -> float:
        """Convert residual stability to a 0-30 score."""
        stab = self.metrics.get("Stability", 50.0)
        return round(float(stab * 0.30), 2)   # 100 stability → 30 pts

    def _horizon_decay(self, step: int, total_steps: int) -> float:
        """
        Horizon penalty — confidence decays with forecast distance.
        Returns a 0-20 score (step 1 → 20, last step → 5).
        """
        decay = 20.0 * np.exp(-0.18 * (step - 1))
        return round(float(np.clip(decay, 5, 20)), 2)

    def _ci_width_score(
        self,
        forecast_val: float,
        ci_lower: float,
        ci_upper: float,
    ) -> float:
        """
        Narrow CI relative to forecast → higher score (0-10).
        Width ratio = (upper - lower) / forecast_val
        """
        if forecast_val <= 0:
            return 5.0
        width_ratio = (ci_upper - ci_lower) / abs(forecast_val)
        score = 10.0 * np.exp(-2.5 * width_ratio)
        return round(float(np.clip(score, 0, 10)), 2)

    # ── Master scoring ────────────────────────────────────────────────────

    def score(
        self,
        forecast:      pd.Series,
        ci_df:         pd.DataFrame,
        drift_detected: bool = False,
    ) -> pd.DataFrame:
        """
        Compute confidence score for each forecast period.

        Parameters
        ----------
        forecast       : forecast Series (indexed by date)
        ci_df          : DataFrame with 'lower' and 'upper' columns
        drift_detected : whether drift was detected (applies penalty)

        Returns
        -------
        pd.DataFrame with columns:
            forecast, lower, upper, mape_score, stability_score,
            horizon_score, ci_score, drift_penalty, total_score, label
        """
        n      = len(forecast)
        rows   = []

        mape_sc = self._mape_score()
        stab_sc = self._stability_score()
        penalty = self.drift_penalty if drift_detected else 0.0

        for step, (idx, fc_val) in enumerate(forecast.items(), 1):
            lo  = float(ci_df.loc[idx, "lower"]) if idx in ci_df.index else fc_val * 0.9
            hi  = float(ci_df.loc[idx, "upper"]) if idx in ci_df.index else fc_val * 1.1

            hor_sc = self._horizon_decay(step, n)
            ci_sc  = self._ci_width_score(fc_val, lo, hi)
            total  = float(np.clip(mape_sc + stab_sc + hor_sc + ci_sc - penalty, 0, 100))

            label = (
                "Very High" if total >= 80 else
                "High"      if total >= 65 else
                "Medium"    if total >= 45 else
                "Low"       if total >= 25 else
                "Very Low"
            )

            rows.append({
                "period":           str(idx)[:10],
                "forecast":         round(float(fc_val), 0),
                "lower_95":         round(lo, 0),
                "upper_95":         round(hi, 0),
                "mape_score":       mape_sc,
                "stability_score":  stab_sc,
                "horizon_score":    hor_sc,
                "ci_score":         ci_sc,
                "drift_penalty":    penalty,
                "confidence_score": round(total, 1),
                "confidence_label": label,
            })

        df = pd.DataFrame(rows).set_index("period")
        _log.info(
            "Confidence scores computed: %d periods | Range: %.1f – %.1f",
            n,
            df["confidence_score"].min(),
            df["confidence_score"].max(),
        )
        return df

    def summary(self, df: pd.DataFrame) -> str:
        """Return a human-readable summary of confidence scores."""
        avg = df["confidence_score"].mean()
        lines = [
            "FORECAST CONFIDENCE SUMMARY",
            f"{'─'*40}",
            f"  Avg Score   : {avg:.1f}/100",
            f"  Min Score   : {df['confidence_score'].min():.1f}",
            f"  Max Score   : {df['confidence_score'].max():.1f}",
            f"  MAPE Score  : {df['mape_score'].iloc[0]:.1f}/40",
            f"  Stab Score  : {df['stability_score'].iloc[0]:.1f}/30",
            "",
            "  Per-period:",
        ]
        for period, row in df.iterrows():
            lines.append(
                f"    {period}  ₹{row['forecast']/1e7:.2f}Cr  "
                f"→ {row['confidence_score']:.0f}/100 ({row['confidence_label']})"
            )
        return "\n".join(lines)
