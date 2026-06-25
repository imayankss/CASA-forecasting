# 📊 BOI CASA Deposit Forecasting Report

> **Bank of India · Data Science Division**  
> Time-Series Forecasting — CASA Deposits (2015–2024)

![Python](https://img.shields.io/badge/Python-3.10+-blue)  ![Models](https://img.shields.io/badge/Models-6-green)  ![Best_Model](https://img.shields.io/badge/Best_Model-HoltWinters-brightgreen)  ![MAPE](https://img.shields.io/badge/MAPE-1.52pct-success)  ![License](https://img.shields.io/badge/License-MIT-lightgrey)


---

## 🎯 At a Glance

| Metric | Value |
|--------|-------|
| 🏆 Best Model | **HoltWinters** |
| 📉 Best MAPE | `1.52%` |
| 📐 Best R² | `0.8061` |
| 📊 Models Evaluated | `6` |
| 🗓️ Data Points | `40 quarters` |
| ⏱️ Report Generated | `2026-06-24 20:22` |



---


## 1. Executive Summary

This report presents the results of a comprehensive time-series forecasting study on Bank of India CASA (Current Account Savings Account) deposit data. The study evaluated 6 forecasting models, benchmarked using multiple statistical metrics and walk-forward cross-validation on quarterly deposit data spanning 2015–2024.

The HoltWinters model achieved the highest composite score (98.8/100) with a Mean Absolute Percentage Error (MAPE) of 1.52%, an RMSE of 1.80% relative to mean deposits, and an R² of 0.8061. Forecast bias stands at +0.84%, indicating a slight tendency toward over-forecasting. Residual stability score is 37.9/100.

Across all 6 models the average MAPE was 3.00%, with a model-to-model spread of 2.42 percentage points — indicating a closely-matched field. Based on the composite evaluation, HoltWinters is recommended as the primary production forecasting model for treasury planning cycles, with re-training advised at each quarterly data release.


---


## 2. Model Performance Leaderboard

|             |   Rank |   MAE_% |   RMSE_% |   MAPE |     R2 |   Stability |   Composite_Score |   Train_Time_s |
|:------------|-------:|--------:|---------:|-------:|-------:|------------:|------------------:|---------------:|
| HoltWinters |      1 |   1.496 |    1.797 |  1.518 |  0.806 |       37.92 |             98.82 |           0.03 |
| Prophet     |      2 |   2.021 |    2.43  |  2.033 |  0.646 |       40.29 |             82.1  |           0.46 |
| ARIMA       |      3 |   3.243 |    4.148 |  3.214 | -0.033 |        0    |             18.31 |           0.26 |
| SARIMAX     |      4 |   3.471 |    3.885 |  3.453 |  0.094 |        0    |             15.58 |           0.07 |
| SARIMA      |      5 |   3.846 |    4.214 |  3.831 | -0.066 |        0    |              3.85 |           0.23 |
| AutoARIMA   |      6 |   3.958 |    4.351 |  3.942 | -0.136 |        0    |              0    |           1.46 |

> **Composite Score** = weighted combination of MAPE (35%), RMSE % (25%), MAE % (20%), Stability (20%). Higher is better.


## 3. Forecast Interpretation

The HoltWinters model forecasts CASA deposits over the next 8 quarter(s). The first forecast period projects ₹0.06 Cr, which is 9.44% below the last observed deposit level of ₹0.07 Cr. Across the full forecast horizon, the model anticipates continued growth of 9.86%, reaching ₹0.07 Cr by the final forecast period. These projections carry a 95% confidence interval and should be reviewed against macroeconomic conditions before operational use.

### Performance Context

Model performance spans a MAPE range of 1.52%–3.94%. Excellent tier (MAPE < 3%): HoltWinters, Prophet. Good tier (3–6%): ARIMA, SARIMAX, SARIMA, AutoARIMA. The HoltWinters model leads the leaderboard, while AutoARIMA shows the highest forecast error. Composite scores — which weight MAPE, RMSE, MAE, and residual stability — confirm this ranking.


---


## 4. Model Explanations

### 4.1 ARIMA

![MAPE](https://img.shields.io/badge/MAPE-3.21%25-green)  ![RMSE%](https://img.shields.io/badge/RMSE%25-4.15%25-blue)  ![R2](https://img.shields.io/badge/R2--0.0326-blueviolet)  ![Stability](https://img.shields.io/badge/Stability-0/100-informational)

ARIMA (AutoRegressive Integrated Moving Average) is a classical univariate statistical model. It captures linear trends and short-term autocorrelation in the deposit series. The model operates on log-transformed values for variance stabilisation.

Performance (test set):
  MAPE        : 3.21%
  RMSE %      : 4.15%
  R²          : -0.0326
  Forecast Bias: -0.74%
  Stability   : 0.0/100
  Train time  : 0.26 s

Strengths:
    • Simple and interpretable
    • Low compute cost
    • Strong baseline for comparison

Limitations:
    • Does not model seasonality
    • Sensitive to order selection


### 4.2 SARIMA

![MAPE](https://img.shields.io/badge/MAPE-3.83%25-green)  ![RMSE%](https://img.shields.io/badge/RMSE%25-4.21%25-blue)  ![R2](https://img.shields.io/badge/R2--0.0658-blueviolet)  ![Stability](https://img.shields.io/badge/Stability-0/100-informational)

SARIMA (Seasonal ARIMA) extends ARIMA with seasonal components, making it well-suited for quarterly banking data that exhibits recurring patterns across fiscal years. Seasonal differencing removes periodic non-stationarity.

Performance (test set):
  MAPE        : 3.83%
  RMSE %      : 4.21%
  R²          : -0.0658
  Forecast Bias: +3.85%
  Stability   : 0.0/100
  Train time  : 0.23 s

Strengths:
    • Explicitly models quarterly seasonality
    • Well-understood statistical properties

Limitations:
    • Many parameters — risk of overfitting on short series


### 4.3 SARIMAX

![MAPE](https://img.shields.io/badge/MAPE-3.45%25-green)  ![RMSE%](https://img.shields.io/badge/RMSE%25-3.88%25-blue)  ![R2](https://img.shields.io/badge/R2-0.0940-blueviolet)  ![Stability](https://img.shields.io/badge/Stability-0/100-informational)

SARIMAX (Seasonal ARIMA with Exogenous Variables) enriches SARIMA with external regressors. A time-index regressor captures the long-run structural growth trend in deposits, resulting in the lowest error among classical models.

Performance (test set):
  MAPE        : 3.45%
  RMSE %      : 3.88%
  R²          : 0.0940
  Forecast Bias: +3.47%
  Stability   : 0.0/100
  Train time  : 0.07 s

Strengths:
    • Best classical accuracy in this study
    • Captures structural growth via exogenous regressor

Limitations:
    • Requires exogenous variable for future periods


### 4.4 AutoARIMA

![MAPE](https://img.shields.io/badge/MAPE-3.94%25-green)  ![RMSE%](https://img.shields.io/badge/RMSE%25-4.35%25-blue)  ![R2](https://img.shields.io/badge/R2--0.1363-blueviolet)  ![Stability](https://img.shields.io/badge/Stability-0/100-informational)

AutoARIMA uses a stepwise grid-search (via pmdarima) to automatically select the optimal ARIMA order (p, d, q) and seasonal order. This removes the manual tuning burden and often finds parsimonious models that generalise well.

Performance (test set):
  MAPE        : 3.94%
  RMSE %      : 4.35%
  R²          : -0.1363
  Forecast Bias: +3.96%
  Stability   : 0.0/100
  Train time  : 1.46 s

Strengths:
    • No manual order selection
    • Robust to model misspecification
    • Good CV performance

Limitations:
    • Slower training
    • Grid search may miss non-standard orders


### 4.5 HoltWinters

![MAPE](https://img.shields.io/badge/MAPE-1.52%25-brightgreen)  ![RMSE%](https://img.shields.io/badge/RMSE%25-1.80%25-blue)  ![R2](https://img.shields.io/badge/R2-0.8061-blueviolet)  ![Stability](https://img.shields.io/badge/Stability-38/100-informational)

HoltWinters is a time-series forecasting model.

Performance (test set):
  MAPE        : 1.52%
  RMSE %      : 1.80%
  R²          : 0.8061
  Forecast Bias: +0.84%
  Stability   : 37.9/100
  Train time  : 0.03 s

Strengths:
    • —

Limitations:
    • —


### 4.6 Prophet

![MAPE](https://img.shields.io/badge/MAPE-2.03%25-brightgreen)  ![RMSE%](https://img.shields.io/badge/RMSE%25-2.43%25-blue)  ![R2](https://img.shields.io/badge/R2-0.6457-blueviolet)  ![Stability](https://img.shields.io/badge/Stability-40/100-informational)

Prophet (Meta) is an additive decomposition model that separates deposits into trend, seasonality, and residual components. Its multiplicative seasonality mode accommodates the growing amplitude of quarterly swings seen in CASA deposit data.

Performance (test set):
  MAPE        : 2.03%
  RMSE %      : 2.43%
  R²          : 0.6457
  Forecast Bias: +1.88%
  Stability   : 40.3/100
  Train time  : 0.46 s

Strengths:
    • Highest composite score
    • Excellent residual diagnostics
    • Handles trend changepoints automatically

Limitations:
    • Slight over-forecasting bias (+1.9%)
    • Less interpretable internals



---


## 5. Residual Diagnostics

| Model | Shapiro-Wilk | Jarque-Bera | Ljung-Box | Health Score |
|---|---|---|---|---|
| **ARIMA** | ❌ Non-normal | ❌ Non-normal | ✅ No AutoCorr | 50/100 — Good |
| **SARIMA** | ❌ Non-normal | ❌ Non-normal | ✅ No AutoCorr | 50/100 — Good |
| **SARIMAX** | ❌ Non-normal | ❌ Non-normal | ✅ No AutoCorr | 50/100 — Good |
| **AutoARIMA** | ❌ Non-normal | ❌ Non-normal | ❌ Autocorr. | 25/100 — Fair |
| **HoltWinters** | ✅ Normal | ✅ Normal | ✅ No AutoCorr | 100/100 — Excellent |
| **Prophet** | ✅ Normal | ✅ Normal | ✅ No AutoCorr | 100/100 — Excellent |

> **Health Score** = % of diagnostic tests passed. 75–100 = Excellent · 50–74 = Good · 25–49 = Fair · 0–24 = Poor.


## 6. Walk-Forward Cross-Validation

| Model       |   CV_MAPE_Mean |   CV_MAPE_Std |   CV_RMSE_%_Mean |   CV_R2_Mean |   CV_Dir_Acc_Mean |   CV_Folds |
|:------------|---------------:|--------------:|-----------------:|-------------:|------------------:|-----------:|
| ARIMA       |          6.097 |         1.726 |            6.981 |       -6.73  |             58.33 |          3 |
| SARIMA      |          1.719 |         0.309 |            1.914 |        0.504 |            100    |          3 |
| SARIMAX     |          9.954 |         7.022 |           12.593 |      -27.39  |             83.33 |          3 |
| AutoARIMA   |          1.976 |         0.301 |            2.471 |        0.035 |            100    |          3 |
| HoltWinters |          1.945 |         1.001 |            2.408 |        0.46  |            100    |          3 |
| Prophet     |          2.433 |         1.276 |            2.982 |       -0.014 |             91.67 |          3 |

> **CV_MAPE_Mean** = average MAPE across all folds. **CV_MAPE_Std** = standard deviation across folds — lower indicates more stable generalisation.


---


## 7. Banking Domain Analysis

Bank of India CASA deposits grew 92.6% over the study period, representing a compound annual growth rate (CAGR) of 6.77%. Deposit volatility (coefficient of variation) is 19.14%, indicating moderate intra-series variability.

Seasonality analysis reveals that Q4 consistently records the highest deposit inflows, while Q2 records the lowest — a pattern consistent with end-of-fiscal-year banking behaviour in India. Average year-over-year growth stands at 6.77%.

These seasonal and growth patterns were successfully captured by the top-performing models. For operational treasury management, the quarterly seasonality should be factored into liquidity buffers and lending plans, particularly ahead of Q4 where deposit inflows are historically strongest.


## 8. Recommendations

1. Deploy HoltWinters as the primary CASA deposit forecasting model for quarterly treasury planning. Its composite score and MAPE lead all evaluated models.

2. Use Prophet as a secondary 'conservative' forecast when risk-averse estimates are required (e.g., regulatory reporting).

3. Re-train all models at the close of each quarter using the latest deposit data to maintain forecast accuracy as structural conditions evolve.

4. Retire AutoARIMA from the production model set or subject it to additional hyperparameter tuning before further consideration.

5. Investigate ensemble forecasting (weighted average of top-3 models) in the next project phase to potentially reduce MAPE further.

6. Implement automated drift detection (e.g., ADWIN) to trigger re-training when the live deposit stream deviates from model expectations.

7. Walk-forward cross-validation confirms SARIMA as the most robust model (CV MAPE = 1.72%), validating the test-set findings.



---


## Appendix — Full Metrics Table

|             |      MAE |    RMSE |   MAPE |   SMAPE |     R2 |   Bias_% |   Stability |   Theil_U |
|:------------|---------:|--------:|-------:|--------:|-------:|---------:|------------:|----------:|
| ARIMA       | 20951.8  | 26796.6 |  3.214 |   3.238 | -0.033 |   -0.742 |        0    |     0.904 |
| SARIMA      | 24849.8  | 27223.3 |  3.831 |   3.746 | -0.066 |    3.846 |        0    |     0.911 |
| SARIMAX     | 22422.5  | 25099.3 |  3.453 |   3.38  |  0.094 |    3.471 |        0    |     0.846 |
| AutoARIMA   | 25569.4  | 28109.8 |  3.942 |   3.851 | -0.136 |    3.958 |        0    |     0.943 |
| HoltWinters |  9665.04 | 11611.5 |  1.518 |   1.504 |  0.806 |    0.843 |       37.92 |     0.392 |
| Prophet     | 13054.9  | 15696.2 |  2.033 |   2.004 |  0.646 |    1.883 |       40.29 |     0.53  |


---


*Report generated by BOI CASA Forecasting Platform v1.0 · 2026-06-24 20:22*
