"""
ensemble.py — Weighted Ensemble Forecaster for BOI CASA Deposits.

Combines multiple model forecasts into a single robust prediction using
inverse-MAPE weighting (better models get higher weight automatically).

Usage
-----
    from src.advanced.ensemble import EnsembleForecaster

    ens = EnsembleForecaster(results, strategy="inverse_mape")
    fc, ci = ens.forecast()
    print(ens.weights)
    print(ens.metrics)
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from src.logger import get_logger
from src.evaluation.metrics import compute_full_metrics

_log = get_logger(__name__)

Strategy = Literal["inverse_mape", "inverse_rmse", "equal", "rank"]


class EnsembleForecaster:
    """
    Combine multiple model forecasts via weighted averaging.

    Parameters
    ----------
    results  : pipeline results dict  {model_name → {forecast, ci, metrics, …}}
    test     : actual test-period series (for evaluation)
    strategy : weighting strategy
        - ``inverse_mape`` — weight ∝ 1/MAPE (default, best models weighted most)
        - ``inverse_rmse`` — weight ∝ 1/RMSE_%
        - ``rank``         — weight ∝ 1/rank (best=1 gets highest weight)
        - ``equal``        — uniform weighting (simple average)
    top_k    : use only top-k models by MAPE (None = use all)
    """

    def __init__(
        self,
        results:  Dict,
        test:     pd.Series,
        strategy: Strategy = "inverse_mape",
        top_k:    Optional[int] = None,
    ) -> None:
        self.results  = results
        self.test     = test
        self.strategy = strategy
        self.top_k    = top_k

        self.weights: Dict[str, float] = {}
        self.ensemble_forecast: Optional[pd.Series] = None
        self.ensemble_ci: Optional[pd.DataFrame]    = None
        self.metrics: Dict[str, float] = {}

        self._compute_weights()

    # ── Weight computation ────────────────────────────────────────────────

    def _compute_weights(self) -> None:
        """Compute normalised weights for each model."""
        models = list(self.results.keys())

        # Sort by MAPE ascending
        sorted_models = sorted(
            models,
            key=lambda m: self.results[m]["metrics"].get("MAPE", 9999),
        )
        if self.top_k:
            sorted_models = sorted_models[: self.top_k]

        raw: Dict[str, float] = {}
        for rank, name in enumerate(sorted_models, 1):
            m = self.results[name]["metrics"]
            if self.strategy == "inverse_mape":
                mape = m.get("MAPE", 9999)
                raw[name] = 1.0 / max(mape, 0.01)
            elif self.strategy == "inverse_rmse":
                rmse = m.get("RMSE_%", 9999)
                raw[name] = 1.0 / max(rmse, 0.01)
            elif self.strategy == "rank":
                raw[name] = 1.0 / rank
            else:  # equal
                raw[name] = 1.0

        total = sum(raw.values()) or 1.0
        self.weights = {k: round(v / total, 4) for k, v in raw.items()}
        _log.info(
            "Ensemble weights [%s]: %s",
            self.strategy,
            {k: f"{v:.3f}" for k, v in self.weights.items()},
        )

    # ── Forecast ─────────────────────────────────────────────────────────

    def forecast(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Compute the weighted ensemble forecast and a simple CI.

        Returns
        -------
        (ensemble_series, ci_dataframe)
        """
        ref_idx = self.test.index

        # Weighted mean forecast
        fc_matrix = np.zeros((len(ref_idx), len(self.weights)))
        for j, (name, w) in enumerate(self.weights.items()):
            fc_vals = self.results[name]["forecast"].reindex(ref_idx).values
            fc_matrix[:, j] = fc_vals * w

        ens_vals = fc_matrix.sum(axis=1)

        # Std across model forecasts for uncertainty band
        raw_matrix = np.column_stack(
            [self.results[n]["forecast"].reindex(ref_idx).values
             for n in self.weights]
        )
        spread = raw_matrix.std(axis=1) * 1.96   # 95% band

        self.ensemble_forecast = pd.Series(ens_vals, index=ref_idx, name="Ensemble")
        self.ensemble_ci = pd.DataFrame(
            {"lower": ens_vals - spread, "upper": ens_vals + spread},
            index=ref_idx,
        )

        # Evaluate
        self.metrics = compute_full_metrics(
            self.test.values,
            self.ensemble_forecast.values,
        )
        _log.info(
            "Ensemble MAPE=%.2f%%  RMSE%%=%.2f%%  R²=%.4f",
            self.metrics["MAPE"],
            self.metrics["RMSE_%"],
            self.metrics["R2"],
        )
        return self.ensemble_forecast, self.ensemble_ci

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"Ensemble Strategy : {self.strategy}",
            f"Models used       : {len(self.weights)}",
            "Weights:",
        ]
        for name, w in sorted(self.weights.items(), key=lambda x: -x[1]):
            lines.append(f"  {name:<14} {w:.4f}  ({w*100:.1f}%)")
        if self.metrics:
            lines += [
                "Ensemble Metrics:",
                f"  MAPE     : {self.metrics['MAPE']:.3f}%",
                f"  RMSE %   : {self.metrics['RMSE_%']:.3f}%",
                f"  R²       : {self.metrics['R2']:.4f}",
                f"  Stability: {self.metrics['Stability']:.1f}/100",
            ]
        return "\n".join(lines)
