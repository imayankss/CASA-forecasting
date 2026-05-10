"""
comparison.py — Model comparison, ranking, and recommendation engine.
Produces leaderboards, composite scores, and business insights.
"""

import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_full_metrics, composite_score, SCORE_WEIGHTS

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# COMPARISON ENGINE
# ─────────────────────────────────────────────────────────────

class ModelComparison:
    """
    Aggregates results from multiple models and produces:
    - metric DataFrames
    - leaderboard with composite scores
    - best / most stable model identification
    - natural-language recommendations
    """

    def __init__(self) -> None:
        self._records: Dict[str, Dict] = {}   # model_name → full result dict

    # ── Registration ─────────────────────────────────────────────────────

    def add(
        self,
        model_name: str,
        actual:     np.ndarray,
        predicted:  np.ndarray,
        residuals:  np.ndarray,
        train_time: float = 0.0,
        extra:      Optional[Dict] = None,
    ) -> None:
        """Register a model result."""
        metrics = compute_full_metrics(actual, predicted, residuals)
        self._records[model_name] = {
            "metrics":    metrics,
            "actual":     np.array(actual,    dtype=float),
            "predicted":  np.array(predicted, dtype=float),
            "residuals":  np.array(residuals, dtype=float),
            "train_time": train_time,
            **(extra or {}),
        }

    # ── DataFrames ───────────────────────────────────────────────────────

    def metrics_df(self) -> pd.DataFrame:
        """Return a DataFrame of raw metrics for all models."""
        rows = {k: v["metrics"] for k, v in self._records.items()}
        return pd.DataFrame(rows).T.round(4)

    def leaderboard(self) -> pd.DataFrame:
        """
        Return a ranked DataFrame with composite scores.
        Columns: rank, all metrics, Composite_Score, Recommended.
        """
        mdf = self.metrics_df()
        if mdf.empty:
            return mdf

        scores = {
            name: composite_score(rec["metrics"], mdf)
            for name, rec in self._records.items()
        }
        mdf["Composite_Score"] = pd.Series(scores)
        mdf["Train_Time_s"]    = pd.Series(
            {k: round(v["train_time"], 2) for k, v in self._records.items()}
        )
        mdf.sort_values("Composite_Score", ascending=False, inplace=True)
        mdf.insert(0, "Rank", range(1, len(mdf) + 1))
        mdf["Recommended"] = mdf.index == mdf.index[0]
        return mdf

    # ── Key models ───────────────────────────────────────────────────────

    def best_model(self) -> str:
        """Name of the model with the highest composite score."""
        lb = self.leaderboard()
        return lb.index[0] if not lb.empty else ""

    def most_stable_model(self) -> str:
        """Name of the model with the highest residual stability score."""
        if not self._records:
            return ""
        return max(
            self._records,
            key=lambda k: self._records[k]["metrics"].get("Stability", 0),
        )

    def lowest_mape_model(self) -> str:
        """Name of the model with the lowest MAPE."""
        if not self._records:
            return ""
        return min(
            self._records,
            key=lambda k: self._records[k]["metrics"].get("MAPE", 9999),
        )

    # ── Insight generation ───────────────────────────────────────────────

    def generate_insights(self) -> Dict:
        """
        Produce a structured insights report with business-friendly language.
        """
        lb = self.leaderboard()
        if lb.empty:
            return {}

        best   = self.best_model()
        stable = self.most_stable_model()
        lowest = self.lowest_mape_model()
        bm     = self._records[best]["metrics"]

        # Trend analysis
        all_mapes  = {k: v["metrics"]["MAPE"] for k, v in self._records.items()}
        all_stab   = {k: v["metrics"]["Stability"] for k, v in self._records.items()}
        worst      = max(all_mapes, key=all_mapes.get)
        mape_range = max(all_mapes.values()) - min(all_mapes.values())

        insights = {
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
            "n_models":     len(self._records),
            "best_model":   best,
            "stable_model": stable,
            "lowest_mape":  lowest,
            "worst_model":  worst,
            "best_metrics": bm,
            "mape_range":   round(mape_range, 3),
            "avg_mape":     round(np.mean(list(all_mapes.values())), 3),
            "model_ranking": lb["Composite_Score"].round(2).to_dict(),

            "executive_summary": (
                f"Across {len(self._records)} evaluated models, **{best}** achieves the best "
                f"overall performance with a composite score of "
                f"{lb.loc[best, 'Composite_Score']:.1f}/100 and a MAPE of "
                f"{bm['MAPE']:.2f}%. "
                f"The forecast bias is {bm['Bias_%']:+.2f}%, indicating "
                f"{'slight over-forecasting' if bm['Bias_%'] > 0 else 'slight under-forecasting' if bm['Bias_%'] < 0 else 'no systematic bias'}. "
                f"Model stability score: {bm['Stability']:.1f}/100."
            ),

            "model_recommendations": [
                f"✅ Use **{best}** as the primary production model — highest composite score.",
                f"🔒 **{stable}** is the most stable model — ideal for risk-averse forecasting.",
                f"📉 **{worst}** shows the highest MAPE ({all_mapes[worst]:.2f}%) — use with caution.",
                f"📊 MAPE spread across models is {mape_range:.2f}% — "
                + ("models are closely matched." if mape_range < 2 else "significant variation exists across models."),
            ],

            "banking_insights": [
                "📈 CASA deposit growth follows quarterly seasonality with Q4 typically strongest.",
                f"🏆 Best forecasting accuracy: {bm['MAPE']:.2f}% MAPE ({best}) — "
                  "suitable for treasury planning.",
                "⚙️ Recommend running a re-training cycle quarterly as new deposit data arrives.",
                f"📐 R² Score of {bm['R2']:.4f} for {best} indicates "
                  + ("strong" if bm["R2"] > 0.9 else "moderate" if bm["R2"] > 0.7 else "weak")
                  + " explanatory power.",
            ],
        }
        return insights

    # ── Export helpers ────────────────────────────────────────────────────

    def to_csv(self, path: str) -> None:
        self.leaderboard().to_csv(path)

    def to_markdown(self) -> str:
        lb = self.leaderboard()
        cols = ["Rank", "MAE_%", "RMSE_%", "MAPE", "R2", "Stability", "Composite_Score"]
        display = [c for c in cols if c in lb.columns]
        return lb[display].round(3).to_markdown()
