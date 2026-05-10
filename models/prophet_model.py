"""
prophet_model.py — Facebook Prophet forecasting model.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import warnings

from models.base_model import BaseForecastModel
from src.config import PROPHET_CONFIG, FORECAST_HORIZON
from src.utils import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


class ProphetModel(BaseForecastModel):
    """Facebook Prophet model for BOI CASA quarterly deposits."""

    name = "Prophet"

    def __init__(self, **prophet_kwargs) -> None:
        super().__init__()
        self._kwargs  = {**PROPHET_CONFIG, **prophet_kwargs}
        self._model   = None
        self._forecast_df: Optional[pd.DataFrame] = None
        self._train_df: Optional[pd.DataFrame] = None

    # ── Internal helper ──────────────────────────────────────────────────

    def _to_prophet_df(self, series: pd.Series) -> pd.DataFrame:
        return series.reset_index().rename(columns={"index": "ds", series.name or 0: "y"})[["ds", "y"]]

    # ── API ──────────────────────────────────────────────────────────────

    def fit(self, train: pd.Series) -> "ProphetModel":
        from prophet import Prophet
        self._train_df = self._to_prophet_df(train)
        logger.info("Fitting Prophet …")
        self._model = Prophet(**self._kwargs)
        self._model.fit(self._train_df)
        # Store in-sample forecast
        self._forecast_df = self._model.predict(self._train_df[["ds"]])
        logger.info("Prophet fitted.")
        return self

    def fitted_values(self) -> pd.Series:
        yhat = self._forecast_df.set_index("ds")["yhat"]
        yhat.index = pd.DatetimeIndex(yhat.index)
        return yhat.rename("Prophet_fitted")

    def forecast(self, steps: int) -> Tuple[pd.Series, pd.DataFrame]:
        last_ds  = self._train_df["ds"].iloc[-1]
        future_idx = pd.date_range(start=last_ds, periods=steps + 1, freq="QE")[1:]
        future_df  = pd.DataFrame({"ds": future_idx})

        fc_df = self._model.predict(future_df)
        fc    = fc_df.set_index("ds")["yhat"]
        lower = fc_df.set_index("ds")["yhat_lower"]
        upper = fc_df.set_index("ds")["yhat_upper"]

        fc.index = lower.index = upper.index = future_idx
        ci_df = pd.DataFrame({"lower": lower, "upper": upper})
        logger.info(f"Prophet forecast ({steps} steps) complete.")
        return fc.rename("Prophet"), ci_df
