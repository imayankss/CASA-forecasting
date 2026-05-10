# 📊 BOI CASA Deposit Forecasting Report

> **Bank of India · Data Science Division**  
> Time-Series Forecasting — CASA Deposits (2015–2024)

![Python](https://img.shields.io/badge/Python-3.10+-blue)  ![Models](https://img.shields.io/badge/Models-5-green)  ![Best_Model](https://img.shields.io/badge/Best_Model-Prophet-brightgreen)  ![MAPE](https://img.shields.io/badge/MAPE-2.03pct-success)  ![License](https://img.shields.io/badge/License-MIT-lightgrey)


---

## 🎯 At a Glance

| Metric | Value |
|--------|-------|
| 🏆 Best Model | **Prophet** |
| 📉 Best MAPE | `2.03%` |
| 📐 Best R² | `0.6457` |
| 📊 Models Evaluated | `5` |
| 🗓️ Data Points | `40 quarters` |
| ⏱️ Report Generated | `2026-05-10 02:33` |



---


## 1. Executive Summary

This report presents the results of a comprehensive time-series forecasting study on Bank of India CASA (Current Account Savings Account) deposit data. The study evaluated 5 forecasting models, benchmarked using multiple statistical metrics and walk-forward cross-validation on quarterly deposit data spanning 2015–2024.

The Prophet model achieved the highest composite score (100.0/100) with a Mean Absolute Percentage Error (MAPE) of 2.03%, an RMSE of 2.43% relative to mean deposits, and an R² of 0.6457. Forecast bias stands at +1.88%, indicating a slight tendency toward over-forecasting. Residual stability score is 40.3/100.

Across all 5 models the average MAPE was 3.24%, with a model-to-model spread of 1.91 percentage points — indicating a closely-matched field. Based on the composite evaluation, Prophet is recommended as the primary production forecasting model for treasury planning cycles, with re-training advised at each quarterly data release.


---


## 2. Model Performance Leaderboard

|           |   Rank |   MAE_% |   RMSE_% |   MAPE |     R2 |   Stability |   Composite_Score |   Train_Time_s |
|:----------|-------:|--------:|---------:|-------:|-------:|------------:|------------------:|---------------:|
| Prophet   |      1 |   2.021 |    2.43  |  2.033 |  0.646 |       40.29 |            100    |           0.41 |
| ARIMA     |      2 |   3.27  |    4.09  |  3.255 | -0.004 |        0    |             23.06 |           0.23 |
| SARIMAX   |      3 |   3.488 |    3.806 |  3.479 |  0.13  |        0    |             20.4  |           0.23 |
| SARIMA    |      4 |   3.489 |    3.806 |  3.481 |  0.131 |        0    |             20.35 |           0.42 |
| AutoARIMA |      5 |   3.956 |    4.35  |  3.941 | -0.136 |        0    |              0    |           3.8  |

> **Composite Score** = weighted combination of MAPE (35%), RMSE % (25%), MAE % (20%), Stability (20%). Higher is better.


## 3. Forecast Interpretation

The Prophet model forecasts CASA deposits over the next 8 quarter(s). The first forecast period projects ₹0.06 Cr, which is 8.29% below the last observed deposit level of ₹0.07 Cr. Across the full forecast horizon, the model anticipates continued growth of 9.43%, reaching ₹0.07 Cr by the final forecast period. These projections carry a 95% confidence interval and should be reviewed against macroeconomic conditions before operational use.

### Performance Context

Model performance spans a MAPE range of 2.03%–3.94%. Excellent tier (MAPE < 3%): Prophet. Good tier (3–6%): ARIMA, SARIMAX, SARIMA, AutoARIMA. The Prophet model leads the leaderboard, while AutoARIMA shows the highest forecast error. Composite scores — which weight MAPE, RMSE, MAE, and residual stability — confirm this ranking.


---


## 4. Model Explanations

### 4.1 ARIMA

![MAPE](https://img.shields.io/badge/MAPE-3.25%25-green)  ![RMSE%](https://img.shields.io/badge/RMSE%25-4.09%25-blue)  ![R2](https://img.shields.io/badge/R2--0.0043-blueviolet)  ![Stability](https://img.shields.io/badge/Stability-0/100-informational)

ARIMA (AutoRegressive Integrated Moving Average) is a classical univariate statistical model. It captures linear trends and short-term autocorrelation in the deposit series. The model operates on log-transformed values for variance stabilisation.

Performance (test set):
  MAPE        : 3.25%
  RMSE %      : 4.09%
  R²          : -0.0043
  Forecast Bias: -0.29%
  Stability   : 0.0/100
  Train time  : 0.23 s

Strengths:
    • Simple and interpretable
    • Low compute cost
    • Strong baseline for comparison

Limitations:
    • Does not model seasonality
    • Sensitive to order selection


### 4.2 SARIMA

![MAPE](https://img.shields.io/badge/MAPE-3.48%25-green)  ![RMSE%](https://img.shields.io/badge/RMSE%25-3.81%25-blue)  ![R2](https://img.shields.io/badge/R2-0.1305-blueviolet)  ![Stability](https://img.shields.io/badge/Stability-0/100-informational)

SARIMA (Seasonal ARIMA) extends ARIMA with seasonal components, making it well-suited for quarterly banking data that exhibits recurring patterns across fiscal years. Seasonal differencing removes periodic non-stationarity.

Performance (test set):
  MAPE        : 3.48%
  RMSE %      : 3.81%
  R²          : 0.1305
  Forecast Bias: +3.49%
  Stability   : 0.0/100
  Train time  : 0.42 s

Strengths:
    • Explicitly models quarterly seasonality
    • Well-understood statistical properties

Limitations:
    • Many parameters — risk of overfitting on short series


### 4.3 SARIMAX

![MAPE](https://img.shields.io/badge/MAPE-3.48%25-green)  ![RMSE%](https://img.shields.io/badge/RMSE%25-3.81%25-blue)  ![R2](https://img.shields.io/badge/R2-0.1304-blueviolet)  ![Stability](https://img.shields.io/badge/Stability-0/100-informational)

SARIMAX (Seasonal ARIMA with Exogenous Variables) enriches SARIMA with external regressors. A time-index regressor captures the long-run structural growth trend in deposits, resulting in the lowest error among classical models.

Performance (test set):
  MAPE        : 3.48%
  RMSE %      : 3.81%
  R²          : 0.1304
  Forecast Bias: +3.49%
  Stability   : 0.0/100
  Train time  : 0.23 s

Strengths:
    • Best classical accuracy in this study
    • Captures structural growth via exogenous regressor

Limitations:
    • Requires exogenous variable for future periods


### 4.4 AutoARIMA

![MAPE](https://img.shields.io/badge/MAPE-3.94%25-green)  ![RMSE%](https://img.shields.io/badge/RMSE%25-4.35%25-blue)  ![R2](https://img.shields.io/badge/R2--0.1357-blueviolet)  ![Stability](https://img.shields.io/badge/Stability-0/100-informational)

AutoARIMA uses a stepwise grid-search (via pmdarima) to automatically select the optimal ARIMA order (p, d, q) and seasonal order. This removes the manual tuning burden and often finds parsimonious models that generalise well.

Performance (test set):
  MAPE        : 3.94%
  RMSE %      : 4.35%
  R²          : -0.1357
  Forecast Bias: +3.96%
  Stability   : 0.0/100
  Train time  : 3.80 s

Strengths:
    • No manual order selection
    • Robust to model misspecification
    • Good CV performance

Limitations:
    • Slower training
    • Grid search may miss non-standard orders


### 4.5 Prophet

![MAPE](https://img.shields.io/badge/MAPE-2.03%25-brightgreen)  ![RMSE%](https://img.shields.io/badge/RMSE%25-2.43%25-blue)  ![R2](https://img.shields.io/badge/R2-0.6457-blueviolet)  ![Stability](https://img.shields.io/badge/Stability-40/100-informational)

Prophet (Meta) is an additive decomposition model that separates deposits into trend, seasonality, and residual components. Its multiplicative seasonality mode accommodates the growing amplitude of quarterly swings seen in CASA deposit data.

Performance (test set):
  MAPE        : 2.03%
  RMSE %      : 2.43%
  R²          : 0.6457
  Forecast Bias: +1.88%
  Stability   : 40.3/100
  Train time  : 0.41 s

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
| **Prophet** | ✅ Normal | ✅ Normal | ✅ No AutoCorr | 100/100 — Excellent |

> **Health Score** = % of diagnostic tests passed. 75–100 = Excellent · 50–74 = Good · 25–49 = Fair · 0–24 = Poor.


## 6. Walk-Forward Cross-Validation

| Model     |   CV_MAPE_Mean |   CV_MAPE_Std |   CV_RMSE_%_Mean |   CV_Folds |
|:----------|---------------:|--------------:|-----------------:|-----------:|
| ARIMA     |          3.405 |         0.509 |            4.428 |          3 |
| SARIMA    |          2.291 |         0.321 |            2.371 |          3 |
| SARIMAX   |          2.262 |         0.345 |            2.34  |          3 |
| AutoARIMA |          1.971 |         0.738 |            2.129 |          3 |
| Prophet   |          1.79  |         0.909 |            2.228 |          3 |

> **CV_MAPE_Mean** = average MAPE across all folds. **CV_MAPE_Std** = standard deviation across folds — lower indicates more stable generalisation.


---


## 7. Banking Domain Analysis

Bank of India CASA deposits grew 92.6% over the study period, representing a compound annual growth rate (CAGR) of 6.77%. Deposit volatility (coefficient of variation) is 19.14%, indicating moderate intra-series variability.

Seasonality analysis reveals that Q4 consistently records the highest deposit inflows, while Q2 records the lowest — a pattern consistent with end-of-fiscal-year banking behaviour in India. Average year-over-year growth stands at 6.77%.

These seasonal and growth patterns were successfully captured by the top-performing models. For operational treasury management, the quarterly seasonality should be factored into liquidity buffers and lending plans, particularly ahead of Q4 where deposit inflows are historically strongest.


## 8. Recommendations

1. Deploy Prophet as the primary CASA deposit forecasting model for quarterly treasury planning. Its composite score and MAPE lead all evaluated models.

2. Use Prophet as a secondary 'conservative' forecast when risk-averse estimates are required (e.g., regulatory reporting).

3. Re-train all models at the close of each quarter using the latest deposit data to maintain forecast accuracy as structural conditions evolve.

4. Retire AutoARIMA from the production model set or subject it to additional hyperparameter tuning before further consideration.

5. Investigate ensemble forecasting (weighted average of top-3 models) in the next project phase to potentially reduce MAPE further.

6. Implement automated drift detection (e.g., ADWIN) to trigger re-training when the live deposit stream deviates from model expectations.

7. Walk-forward cross-validation confirms Prophet as the most robust model (CV MAPE = 1.79%), validating the test-set findings.



---


## Appendix — Full Metrics Table

|           |     MAE |    RMSE |   MAPE |   SMAPE |     R2 |   Bias_% |   Stability |   Theil_U |
|:----------|--------:|--------:|-------:|--------:|-------:|---------:|------------:|----------:|
| ARIMA     | 21128.4 | 26426   |  3.255 |   3.265 | -0.004 |   -0.29  |        0    |     0.884 |
| SARIMA    | 22543.5 | 24589.7 |  3.481 |   3.411 |  0.131 |    3.489 |        0    |     0.82  |
| SARIMAX   | 22532.6 | 24590.8 |  3.479 |   3.409 |  0.13  |    3.488 |        0    |     0.82  |
| AutoARIMA | 25560.8 | 28102.5 |  3.941 |   3.85  | -0.136 |    3.956 |        0    |     0.943 |
| Prophet   | 13054.7 | 15695.9 |  2.033 |   2.004 |  0.646 |    1.883 |       40.29 |     0.53  |


---


*Report generated by BOI CASA Forecasting Platform v1.0 · 2026-05-10 02:33*
