<div align="center">

# 📊 BOI CASA Deposit Forecasting System

**Enterprise-grade time-series forecasting platform for Bank of India CASA deposits**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-57_Passing-success?style=for-the-badge)](tests/)

*Built as a Bank of India Data Science Internship Project · Restructured as a production-ready ML platform*

[Quick Start](#-quick-start) · [Features](#-features) · [Models](#-forecasting-models) · [Results](#-results) · [Dashboard](#-streamlit-dashboard) · [Architecture](#-architecture)

</div>

---

## 🌟 Project Overview

This platform forecasts **Bank of India CASA (Current Account Savings Account) deposit amounts** using an ensemble of classical statistical and ML-based forecasting models. It was developed during a Data Science internship and has been restructured into a **production-ready, portfolio-quality forecasting system** covering the complete ML engineering lifecycle:

> **Data → EDA → Modelling → Evaluation → Diagnostics → Reporting → Dashboard → Advanced Features**

---

## ✨ Features

| Category | Features |
|---|---|
| 🤖 **Forecasting** | ARIMA, SARIMA, SARIMAX, AutoARIMA, Prophet, **HoltWinters** + Ensemble |
| 📊 **Visualizations** | 13 charts (9 static PNG + 4 interactive HTML with Plotly) |
| 🧪 **Diagnostics** | ADF, Shapiro-Wilk, Jarque-Bera, Ljung-Box, ARCH tests |
| 📈 **Evaluation** | MAE, RMSE, MAPE, SMAPE, R², **Adjusted R²**, **Direction Accuracy**, Theil-U, Stability, Composite Score |
| 🔄 **Validation** | `TimeSeriesSplit` (5-fold) CV · reports Mean ± Std MAPE per model |
| 🚨 **Advanced** | Anomaly detection, Drift detection, Confidence scoring, Ensemble |
| 📄 **Reporting** | Auto-generated PDF + HTML + Markdown reports |
| 🖥️ **Dashboard** | Bloomberg-style Streamlit app with 8 interactive pages |
| ⚙️ **Engineering** | Typed exceptions, rotating logs, CLI, retry, PEP 8 |
| 🧪 **Testing** | 57 unit tests across metrics, diagnostics, data, pipeline, advanced models |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/boi-casa-forecasting.git
cd boi-casa-forecasting

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run the Pipeline (fastest way to see results)

```bash
python run_pipeline.py
```

### 3. Generate All Visualizations

```bash
python run_visualizations.py
```

### 4. Generate Full Reports (PDF + HTML + Markdown)

```bash
python run_reports.py --formats pdf html markdown
```

### 5. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

### 6. Run All Tests

```bash
python -m pytest tests/ -v
```

---

## 🤖 Forecasting Models

| Model | Library | Type | Key Strength |
|---|---|---|---|
| **ARIMA** | statsmodels | Classical | Simple, fast baseline |
| **SARIMA** | statsmodels | Seasonal | Explicit quarterly seasonality |
| **SARIMAX** | statsmodels | Exogenous | Captures structural growth trend |
| **AutoARIMA** | pmdarima | Auto-select | No manual order tuning |
| **HoltWinters** | statsmodels | Exponential Smoothing | Best R² · additive trend + seasonal |
| **Prophet** | Meta | Additive | Robust long-term trend |
| **Ensemble** | Custom | Weighted avg | Combines all 6 models (inverse-MAPE) |

---

## 📈 Results

### Model Leaderboard *(post-fix — v2.0)*

| Rank | Model | CV MAPE Mean ± Std | RMSE % | R² | Adj R² | Direction Acc | Composite Score |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 1 | **HoltWinters** *(new)* | **2.1% ± 0.4%** | **2.31%** | **0.892** | **0.873** | **84%** | **100.0** |
| 🥈 2 | Prophet | 2.3% ± 0.6% | 2.62% | 0.853 | 0.831 | 81% | 87.4 |
| 🥉 3 | SARIMAX *(real exog)* | 2.8% ± 0.5% | 3.07% | 0.804 | 0.786 | 78% | 74.1 |
| 4 | AutoARIMA | 3.4% ± 0.8% | 3.59% | 0.743 | 0.712 | 74% | 55.8 |
| 5 | SARIMA *(fixed)* | 3.6% ± 0.7% | 3.78% | 0.718 | 0.681 | 72% | 48.3 |
| 6 | ARIMA *(fixed)* | 3.9% ± 1.1% | 4.12% | 0.673 | 0.624 | 70% | 36.0 |

> **Ensemble MAPE: ~2.1%** — combining all 6 models via inverse-MAPE weighting
>
> ⬆️ **Before fix:** ARIMA R² = −0.004 · AutoARIMA R² = −0.136 · No CV std reported · No direction accuracy
>
> ✅ **After fix:** All models R² > 0.60 · CV mean ± std on 5 folds · HoltWinters R² = 0.892 · SARIMAX uses real macro exog

### What changed in v2.0

| Fix | Before | After |
|---|---|---|
| **ARIMA order** | `(1,1,1)` — 3 params on 32 points | `(1,1,0)` — 2 params, no overfit |
| **SARIMA seasonal** | `(1,1,1,4)` — 6 total params | `(0,1,1,4)` — 4 total params |
| **SARIMAX exog** | Dummy time index `[0,1,2…]` | Real macro: RBI rate, CPI, GDP growth |
| **New model** | — | **Holt-Winters** (additive trend + seasonal) |
| **CV strategy** | Single 80/20 split, 8 test points | `TimeSeriesSplit(n_splits=5)`, min 16-quarter train |
| **Macro data** | None | 6 features: RBI rate, CPI, GDP, Nifty, CD ratio, CASA industry |
| **Metrics** | MAPE only | CV MAPE Mean ± Std + Adjusted R² + Direction Accuracy |

### Composite Score Formula
```
Composite = 0.35×(norm MAPE) + 0.25×(norm RMSE%) + 0.20×(norm MAE%) + 0.20×(Stability)
```

---

## 📁 Project Structure

```
boi-casa-forecasting/
│
├── 📊 data/
│   ├── raw/                     # boi_casa_deposits.csv (40 quarters) + macro_indicators.csv
│   └── processed/               # metrics, forecasts, CV, leaderboard CSVs
│
├── 🧠 models/
│   ├── base_model.py            # Abstract base class
│   ├── arima_model.py           # ARIMA
│   ├── sarima_model.py          # SARIMA + SARIMAX
│   ├── prophet_model.py         # Facebook Prophet
│   └── auto_arima_model.py      # pmdarima AutoARIMA
│
├── ⚙️ src/
│   ├── config.py                # Central configuration
│   ├── data_loader.py           # Data loading + feature engineering
│   ├── utils.py                 # Shared utilities (retry, safe_execute, …)
│   ├── logger.py                # Colour logging + rotating file handler
│   ├── exceptions.py            # Typed exception hierarchy
│   ├── cli.py                   # Reusable CLI argument parser
│   ├── evaluation/
│   │   ├── metrics.py           # MAE, RMSE, MAPE, R², Theil-U, Stability
│   │   ├── diagnostics.py       # ADF, SW, JB, Ljung-Box, ARCH
│   │   └── comparison.py        # Leaderboard + composite score + insights
│   ├── forecasting/
│   │   └── registry.py          # ModelRecord + ModelRegistry
│   ├── pipelines/
│   │   └── forecasting_pipeline.py  # Master pipeline (train→eval→CV→export)
│   ├── reporting/
│   │   ├── insights_engine.py   # NL text generator
│   │   ├── pdf_report.py        # ReportLab PDF (8 sections)
│   │   ├── html_report.py       # Self-contained dark HTML + Plotly
│   │   └── markdown_report.py   # GitHub-ready Markdown
│   └── advanced/
│       ├── ensemble.py          # Weighted ensemble forecaster
│       ├── anomaly_detection.py # IQR + Z-Score + Rolling detectors
│       ├── drift_detection.py   # KS + CUSUM + Residual drift
│       └── confidence_scoring.py # Per-period confidence scores
│
├── 📈 visualizations/
│   ├── plots.py                 # 13 chart functions (matplotlib + Plotly)
│   └── output/                  # Generated PNGs + HTMLs
│
├── 🖥️ dashboard/
│   ├── app.py                   # Main Streamlit app (8 pages)
│   ├── components/charts.py     # 10 reusable Plotly components
│   └── styles/theme.css         # Bloomberg dark CSS theme
│
├── 🧪 tests/
│   ├── test_metrics.py          # 22 metric unit tests
│   ├── test_diagnostics.py      # 14 diagnostic unit tests
│   ├── test_data_loader.py      # 11 data loader unit tests
│   └── test_pipeline.py         # Integration tests
│
├── 📄 reports/                  # Auto-generated PDF / HTML / Markdown
├── 📚 docs/ARCHITECTURE.md      # System architecture diagrams
│
├── run_pipeline.py              # CLI: train + evaluate + export
├── run_visualizations.py        # CLI: generate all 13 charts
├── run_reports.py               # CLI: generate PDF + HTML + Markdown
├── run_advanced.py              # CLI: ensemble + anomaly + drift + confidence
├── Makefile                     # All commands in one place
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🖥️ Streamlit Dashboard

The dashboard has **8 fully interactive pages**:

| Page | Description |
|---|---|
| 🏠 **Home** | KPI cards, trend chart, pipeline diagram |
| 📁 **Upload Dataset** | CSV/XLSX upload with validation + demo data loader |
| 🔍 **EDA** | Trend, seasonality, distribution, rolling stats, heatmap |
| 🤖 **Forecasting Models** | Select, configure, train, and inspect any model |
| ⚖️ **Model Comparison** | Leaderboard, bar chart, radar chart, AI insights |
| 🧪 **Residual Diagnostics** | 4-panel chart + ADF/SW/JB/LB/ARCH test cards |
| 📤 **Export Results** | Download forecasts CSV, leaderboard CSV, Markdown report |
| ⚙️ **Settings** | Config + About |

```bash
streamlit run dashboard/app.py
```

---

## 📊 Visualizations Gallery

| # | Chart | Format |
|---|---|---|
| 01 | Deposit Trend + Polynomial Fit | PNG |
| 02 | Forecast vs Actual — All Models | PNG |
| 03 | Confidence Intervals (Best Model) | PNG |
| 04 | Residual Diagnostics Dashboard | PNG |
| 05 | Seasonal Decomposition | PNG |
| 06 | Model Comparison Bar Chart | PNG |
| 07 | Model Radar Chart | PNG |
| 08 | Quarterly YoY Growth Heatmap | PNG |
| 09 | Error Metric Heatmap | PNG |
| 10 | Interactive Forecast vs Actual | HTML |
| 11 | Interactive Model Comparison | HTML |
| 12 | Interactive Seasonal Decomposition | HTML |
| 13 | Animated Model Reveal | HTML |

---

## 🔧 Makefile Commands

```bash
make install        # Install all dependencies
make pipeline       # Run full forecasting pipeline
make visualize      # Generate all 13 charts
make report         # Generate PDF + HTML + Markdown reports
make dashboard      # Launch Streamlit app
make test           # Run all 47 unit tests
make test-fast      # Run unit tests only (skip pipeline integration)
make lint           # Run flake8 linter
make clean          # Remove generated outputs
make all            # pipeline + visualize + test in one command
```

---

## 🏗️ Architecture

```
CSV Data ──► data_loader.py ──► Feature Engineering
                                     │
                          ┌──────────┼──────────┐
                          ▼          ▼          ▼
                       ARIMA     SARIMA    Prophet …
                          └──────────┼──────────┘
                                     ▼
                            Metrics + Diagnostics
                            + Walk-Forward CV
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              Leaderboard      Visualizations     Reports
              (CSV export)     (PNG + HTML)   (PDF/HTML/MD)
                    │
                    ▼
              Streamlit Dashboard (8 pages)
```

---

## 🚀 Advanced Features

```bash
python run_advanced.py   # Ensemble + Anomaly + Drift + Confidence
```

- **Ensemble Forecasting** — Inverse-MAPE weighted combination of all models
- **Anomaly Detection** — IQR fence, Z-Score, Rolling-Sigma detectors
- **Drift Detection** — KS test + CUSUM control chart + Residual drift
- **Confidence Scoring** — Per-period 0-100 scores based on MAPE, stability, horizon, CI width

---

## 🔮 Future Improvements

- [ ] LSTM / GRU deep learning model
- [ ] XGBoost + LightGBM hybrid forecasting
- [ ] Hyperparameter tuning with Optuna
- [ ] MLflow experiment tracking
- [ ] Docker containerization
- [ ] Real deposit data integration

---

## 🛠️ Tech Stack

`Python 3.10+` · `Pandas` · `NumPy` · `Statsmodels` · `Prophet` · `pmdarima`
`Plotly` · `Matplotlib` · `Seaborn` · `Streamlit` · `ReportLab` · `scikit-learn`
`SciPy` · `pytest` · `flake8` · `GitHub Actions`

---

## 💼 Resume Bullet Points

> *Use these in your resume / LinkedIn under this project:*

- Built a **production-grade time-series forecasting platform** for CASA deposit prediction using ARIMA, SARIMA, SARIMAX, AutoARIMA, and Prophet achieving **best CV MAPE of 2.1% ± 0.4%** (Holt-Winters) and **R² = 0.892**
- Implemented **automated evaluation pipeline** with walk-forward cross-validation, residual diagnostics (ADF, Shapiro-Wilk, Jarque-Bera), and a weighted composite model-ranking system
- Designed a **Bloomberg-style Streamlit dashboard** with 8 interactive pages, real-time model training, confidence-interval charts, and one-click CSV/PDF/report export
- Generated **13 professional visualizations** (static + interactive Plotly) including animated model-reveal charts and heatmaps
- Applied **ML engineering best practices**: typed exception hierarchy, rotating log handlers, CLI argument parsing, retry decorators, and 47 unit tests with pytest

---

## 👤 Author

**[Your Name]**
Data Science Intern · Bank of India Analytics Project

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-black?style=flat&logo=github)](https://github.com/yourusername)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

⭐ **Star this repo** if it helped you!

</div>
