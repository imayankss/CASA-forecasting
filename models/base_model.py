"""
base_model.py — Abstract base class that every forecasting model inherits from.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from src.utils import compute_metrics, get_logger

logger = get_logger(__name__)


class BaseForecastModel(ABC):
    """
    Abstract base for all BOI CASA forecasting models.

    Subclasses must implement: fit(), forecast(), and fitted_values().
    """

    name: str = "BaseModel"

    def __init__(self) -> None:
        self._fitted = False
        self.residuals_: Optional[np.ndarray] = None
        self.metrics_: Dict[str, float] = {}

    # ── Abstract Interface ────────────────────────────────────────────────

    @abstractmethod
    def fit(self, train: pd.Series) -> "BaseForecastModel":
        """Fit the model to training data."""
        ...

    @abstractmethod
    def forecast(self, steps: int) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """
        Generate an out-of-sample forecast.

        Returns
        -------
        (forecast_series, conf_int_df_or_None)
        """
        ...

    @abstractmethod
    def fitted_values(self) -> pd.Series:
        """Return in-sample fitted values (original scale)."""
        ...

    # ── Shared helpers ────────────────────────────────────────────────────

    def evaluate(self, actual: pd.Series) -> Dict[str, float]:
        """
        Compute evaluation metrics against provided actuals.

        Aligns fitted values to the actual index before scoring.
        """
        fitted = self.fitted_values()
        common  = actual.index.intersection(fitted.index)
        self.metrics_ = compute_metrics(actual.loc[common], fitted.loc[common])
        self.residuals_ = (actual.loc[common] - fitted.loc[common]).values
        return self.metrics_

    def summary(self) -> str:
        """Return a short text summary of evaluation metrics."""
        if not self.metrics_:
            return f"{self.name}: not yet evaluated."
        m = self.metrics_
        return (
            f"{'─'*40}\n"
            f"  Model   : {self.name}\n"
            f"  MAE     : {m['MAE_pct']:.2f}%\n"
            f"  RMSE    : {m['RMSE_pct']:.2f}%\n"
            f"  MAPE    : {m['MAPE']:.2f}%\n"
            f"{'─'*40}"
        )
