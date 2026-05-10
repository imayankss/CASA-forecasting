"""
auto_arima_model.py — Automatic ARIMA model selection via pmdarima.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import warnings

from models.base_model import BaseForecastModel
from src.config import AUTO_ARIMA_CONFIG
from src.utils import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


class AutoARIMAModel(BaseForecastModel):
    """Auto-ARIMA with seasonal support (pmdarima)."""

    name = "AutoARIMA"

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._cfg   = {**AUTO_ARIMA_CONFIG, **kwargs}
        self._model = None
        self._train_log: Optional[pd.Series] = None

    def fit(self, train: pd.Series) -> "AutoARIMAModel":
        import pmdarima as pm
        train_log = np.log(train.replace(0, np.nan)).dropna()
        self._train_log = train_log
        logger.info("Running Auto-ARIMA grid search …")
        self._model = pm.auto_arima(train_log, **self._cfg)
        logger.info(f"Auto-ARIMA selected: {self._model.order} seasonal {self._model.seasonal_order}")
        return self

    def fitted_values(self) -> pd.Series:
        fitted_log = self._model.predict_in_sample()
        fitted     = np.exp(fitted_log)
        return pd.Series(fitted, index=self._train_log.index, name="AutoARIMA_fitted")

    def forecast(self, steps: int) -> Tuple[pd.Series, pd.DataFrame]:
        fc_log, ci_log = self._model.predict(n_periods=steps, return_conf_int=True)
        fc       = np.exp(fc_log)
        ci_lower = np.exp(ci_log[:, 0])
        ci_upper = np.exp(ci_log[:, 1])

        last_idx = self._train_log.index[-1]
        idx = pd.date_range(start=last_idx + pd.offsets.QuarterEnd(), periods=steps, freq="QE")
        fc_series = pd.Series(fc, index=idx, name="AutoARIMA")
        ci_df     = pd.DataFrame({"lower": ci_lower, "upper": ci_upper}, index=idx)
        logger.info(f"Auto-ARIMA forecast ({steps} steps) complete.")
        return fc_series, ci_df
