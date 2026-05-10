"""
sarima_model.py — SARIMA / SARIMAX forecasting models.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm

from models.base_model import BaseForecastModel
from src.config import SARIMA_ORDER, SARIMA_SEASONAL_ORDER, SARIMAX_ORDER, SARIMAX_SEASONAL_ORDER
from src.utils import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


class SARIMAModel(BaseForecastModel):
    """SARIMA model on log-transformed CASA deposits."""

    name = "SARIMA"

    def __init__(
        self,
        order: tuple = SARIMA_ORDER,
        seasonal_order: tuple = SARIMA_SEASONAL_ORDER,
    ) -> None:
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
        self._results = None
        self._train_log: Optional[pd.Series] = None

    def fit(self, train: pd.Series) -> "SARIMAModel":
        train_log = np.log(train.replace(0, np.nan)).dropna()
        self._train_log = train_log
        logger.info(f"Fitting SARIMA{self.order}x{self.seasonal_order} …")
        model = sm.tsa.SARIMAX(
            train_log,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._results = model.fit(disp=False)
        logger.info(f"SARIMA fitted  AIC={self._results.aic:.2f}")
        return self

    def fitted_values(self) -> pd.Series:
        return np.exp(self._results.fittedvalues).rename("SARIMA_fitted")

    def forecast(self, steps: int) -> Tuple[pd.Series, pd.DataFrame]:
        fc_obj   = self._results.get_forecast(steps=steps)
        fc_log   = fc_obj.predicted_mean
        ci_log   = fc_obj.conf_int()
        fc       = np.exp(fc_log)
        ci_lower = np.exp(ci_log.iloc[:, 0])
        ci_upper = np.exp(ci_log.iloc[:, 1])

        last_idx = self._train_log.index[-1]
        idx = pd.date_range(start=last_idx + pd.offsets.QuarterEnd(), periods=steps, freq="QE")
        fc.index = ci_lower.index = ci_upper.index = idx

        ci_df = pd.DataFrame({"lower": ci_lower, "upper": ci_upper})
        logger.info(f"SARIMA forecast ({steps} steps) complete.")
        return fc.rename("SARIMA"), ci_df


class SARIMAXModel(BaseForecastModel):
    """SARIMAX — SARIMA with exogenous regressors."""

    name = "SARIMAX"

    def __init__(
        self,
        order: tuple = SARIMAX_ORDER,
        seasonal_order: tuple = SARIMAX_SEASONAL_ORDER,
    ) -> None:
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
        self._results = None
        self._train_log: Optional[pd.Series] = None

    def fit(self, train: pd.Series, exog_train: Optional[pd.DataFrame] = None) -> "SARIMAXModel":
        train_log = np.log(train.replace(0, np.nan)).dropna()
        self._train_log = train_log
        logger.info(f"Fitting SARIMAX{self.order}x{self.seasonal_order} …")
        model = sm.tsa.SARIMAX(
            train_log,
            exog=exog_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._results = model.fit(disp=False)
        logger.info(f"SARIMAX fitted  AIC={self._results.aic:.2f}")
        return self

    def fitted_values(self) -> pd.Series:
        return np.exp(self._results.fittedvalues).rename("SARIMAX_fitted")

    def forecast(self, steps: int, exog_future: Optional[pd.DataFrame] = None) -> Tuple[pd.Series, pd.DataFrame]:
        fc_obj   = self._results.get_forecast(steps=steps, exog=exog_future)
        fc_log   = fc_obj.predicted_mean
        ci_log   = fc_obj.conf_int()
        fc       = np.exp(fc_log)
        ci_lower = np.exp(ci_log.iloc[:, 0])
        ci_upper = np.exp(ci_log.iloc[:, 1])

        last_idx = self._train_log.index[-1]
        idx = pd.date_range(start=last_idx + pd.offsets.QuarterEnd(), periods=steps, freq="QE")
        fc.index = ci_lower.index = ci_upper.index = idx

        ci_df = pd.DataFrame({"lower": ci_lower, "upper": ci_upper})
        logger.info(f"SARIMAX forecast ({steps} steps) complete.")
        return fc.rename("SARIMAX"), ci_df
