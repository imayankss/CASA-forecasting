"""
arima_model.py — ARIMA forecasting model with log-transform and back-transform.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA

from models.base_model import BaseForecastModel
from src.config import ARIMA_ORDER
from src.utils import get_logger, adf_test

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


class ARIMAModel(BaseForecastModel):
    """ARIMA model fitted on log-transformed CASA deposits."""

    name = "ARIMA"

    def __init__(self, order: tuple = ARIMA_ORDER) -> None:
        super().__init__()
        self.order = order
        self._results = None
        self._train_log: Optional[pd.Series] = None
        self._train_orig: Optional[pd.Series] = None

    def fit(self, train: pd.Series) -> "ARIMAModel":
        self._train_orig = train.copy()
        train_log = np.log(train.replace(0, np.nan)).dropna()
        self._train_log = train_log

        # Auto-detect differencing order
        d = 0
        series = train_log.copy()
        while not adf_test(series, f"ARIMA d={d}") and d < 2:
            d += 1
            series = series.diff().dropna()

        order = (self.order[0], d, self.order[2])
        logger.info(f"Fitting ARIMA{order} …")
        model = ARIMA(train_log, order=order)
        self._results = model.fit()
        self._fitted_flag = True
        logger.info(f"ARIMA fitted  AIC={self._results.aic:.2f}")
        return self

    def fitted_values(self) -> pd.Series:
        fitted_log = self._results.fittedvalues
        return np.exp(fitted_log).rename("ARIMA_fitted")

    def forecast(self, steps: int) -> Tuple[pd.Series, pd.DataFrame]:
        fc_obj   = self._results.get_forecast(steps=steps)
        fc_log   = fc_obj.predicted_mean
        ci_log   = fc_obj.conf_int()

        fc       = np.exp(fc_log)
        ci_lower = np.exp(ci_log.iloc[:, 0])
        ci_upper = np.exp(ci_log.iloc[:, 1])

        # Build a proper quarterly DatetimeIndex for the forecast
        last_idx = self._train_log.index[-1]
        idx = pd.date_range(start=last_idx + pd.offsets.QuarterEnd(), periods=steps, freq="QE")
        fc.index = ci_lower.index = ci_upper.index = idx

        ci_df = pd.DataFrame({"lower": ci_lower, "upper": ci_upper})
        logger.info(f"ARIMA forecast ({steps} steps) complete.")
        return fc.rename("ARIMA"), ci_df
