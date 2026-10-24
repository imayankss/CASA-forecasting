# What I fixed

All numbers below are on the **synthetic** 40-quarter series (see `data/raw/README.md`), on the same 8-quarter holdout (2023Q1–2024Q4). "Before" is commit `1eb3361`; "after" is the pipeline output in `data/processed/`.

## The bugs

| Problem | Why it mattered | Fix |
|---|---|---|
| **SARIMAX used future data.** It forecast with the *actual* repo rate, CPI and GDP of the test quarters. | Nobody knows next quarter's GDP at forecast time. Train-only scaling did not fix this. | Macro inputs are lagged by the full forecast horizon (8 quarters), so every value used is known at the forecast origin. `assert_features_known_at` enforces it. |
| **ARIMA forecast a flat line.** ARIMA(1,1,0) on log values had no drift term, so it predicted 641,272 for 7 of 8 quarters. | The model could not represent growth at all. | Added a drift term (`trend="t"`). |
| **The drift test could not fail.** KS compared the *levels* of a trending series (KS = 0.97, always "drift"). | A monitor that always alarms carries no information. | KS now compares quarter-on-quarter log growth (KS = 0.22, no drift). |
| **No baselines, no significance test.** | With 8 test points, "best model" claims were untested. | Added SeasonalNaive and Drift baselines, MASE, and a Diebold–Mariano test against SeasonalNaive. |

## Before vs after

| Model | MAPE before | MAPE after | MASE after | DM vs SeasonalNaive (p) |
|---|---:|---:|---:|---:|
| HoltWinters | 1.52% | 1.52% | 0.29 | 0.008 |
| Prophet | 2.03% | 2.03% | 0.39 | 0.009 |
| SARIMAX | 3.45% | 3.65% | 0.70 | 0.034 |
| SARIMA | 3.83% | 3.83% | 0.75 | 0.029 |
| AutoARIMA | 3.94% | 3.94% | 0.77 | 0.034 |
| ARIMA | 3.21% | 6.70% | 1.29 | 0.878 |
| Drift (baseline) | — | 5.30% | 1.02 | 0.311 |
| SeasonalNaive (baseline) | — | 7.03% | 1.38 | — |

## What the results say

- **Removing the look-ahead cost SARIMAX accuracy** (3.45% → 3.65% MAPE), and its cross-validated MAPE is still the worst of the fitted models (6.6% ± 7.1%). The macro inputs never added real predictive value.
- **ARIMA got worse once it could trend.** The flat line happened to sit near a period when the series stalled. A model that is right by accident is not a good model.
- **Holt-Winters and Prophet clearly beat the seasonal-naive baseline** (MASE 0.29 and 0.39, DM p < 0.01). Caveat: 8 test points from a single forecast origin is a weak basis. Treat the p-values as indicative, not proof.
- **Directional accuracy of 100% is not informative** with 7 quarter-on-quarter changes, so it is no longer used to rank models.

The same pipeline is re-run on Bank of India's real published CASA series in `data/real/` (see the README).
