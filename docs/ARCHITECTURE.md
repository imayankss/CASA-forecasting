# System Architecture

## Overview

The BOI CASA Forecasting Platform is organized as a layered Python monorepo.

```
┌──────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│  dashboard/app.py  (Streamlit UI)                           │
│  dashboard/components/charts.py  (Plotly components)        │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                   APPLICATION LAYER                          │
│  run_pipeline.py       (CLI entry point — Point 8)          │
│  run_visualizations.py (Visualization runner — Point 6)     │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                   PIPELINE LAYER                             │
│  src/pipelines/forecasting_pipeline.py                      │
│    ├── load() → train/test split                            │
│    ├── _train_model() × N models                            │
│    ├── _run_diagnostics()                                   │
│    ├── _run_cv() → walk-forward validation                  │
│    └── save_results() → CSV + JSON                         │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                  EVALUATION LAYER                            │
│  src/evaluation/metrics.py       (MAE, RMSE, MAPE, R², …)  │
│  src/evaluation/diagnostics.py   (ADF, SW, JB, LB, ARCH)   │
│  src/evaluation/comparison.py    (leaderboard, insights)    │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                    MODEL LAYER                               │
│  models/base_model.py       Abstract base                   │
│  models/arima_model.py      ARIMA                           │
│  models/sarima_model.py     SARIMA / SARIMAX                │
│  models/prophet_model.py    Prophet                         │
│  models/auto_arima_model.py AutoARIMA                       │
│  src/forecasting/registry.py  ModelRecord + ModelRegistry   │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                     DATA LAYER                               │
│  src/data_loader.py   load_casa_data(), train/test split    │
│  src/utils.py         shared helpers, ADF, logging          │
│  src/config.py        all constants and environment vars    │
│  data/raw/            original CSV                          │
│  data/processed/      generated metrics, forecasts, CV CSVs │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow

```
CSV File
  │
  ▼
load_casa_data()
  │  parse Quarter_Year → DatetimeIndex
  │  compute Log, YoY_Growth, dummies, rolling avg
  ▼
train / test split (80/20 chronological)
  │
  ├──► ARIMA fitter      ──┐
  ├──► SARIMA fitter     ──┤
  ├──► SARIMAX fitter    ──┼──► forecast + CI + fitted + residuals
  ├──► AutoARIMA fitter  ──┤
  └──► Prophet fitter    ──┘
                            │
                            ▼
                    compute_full_metrics()
                    run_full_diagnostics()
                    walk_forward_cv()
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
           ModelComparison       ModelRegistry
           leaderboard()         save_index()
           generate_insights()
                  │
         ┌────────┴────────────┐
         ▼                     ▼
    CSV exports           Visualizations
    (processed/)          (visualizations/output/)
```

## Forecasting Pipeline Sequence

```
ForecastingPipeline.run()
  │
  ├─ load()
  │    └─ load_casa_data(data_path)
  │    └─ get_train_test(df, train_frac)
  │
  ├─ for name in model_names:
  │    └─ fitter(train, test) → {forecast, ci, fitted, residuals, metrics}
  │    └─ comparison.add(...)
  │    └─ registry.register(ModelRecord(...))
  │
  ├─ for name in results:
  │    └─ run_full_diagnostics(residuals, series) → health_score
  │
  ├─ for name in model_names:
  │    └─ walk_forward_cv(series, fitter, n_splits) → cv_df
  │
  └─ save_results()
       └─ leaderboard CSV
       └─ metrics CSV
       └─ forecasts CSV
       └─ cv_summary CSV
       └─ model_registry JSON
```

## Dashboard Architecture

```
dashboard/app.py
  │
  ├─ _init_state()         — Streamlit session state initialisation
  ├─ render_sidebar()      — Navigation + dataset status widget
  │
  ├─ page_home()           — KPI cards + trend chart
  ├─ page_upload()         — File upload + demo data loader
  ├─ page_eda()            — Trend, seasonality, distribution, heatmap
  ├─ page_forecasting()    — Model selector + trainer + forecast chart
  ├─ page_comparison()     — Leaderboard + bar + radar + insights
  ├─ page_diagnostics()    — 4-panel residuals + statistical tests
  ├─ page_export()         — CSV + Markdown report downloads
  └─ page_settings()       — Config + About
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Log-transform before ARIMA | Stabilises variance; back-transformed for display |
| Separate fitter functions (not class instances) | Easier to test, cache, and parallelize |
| ModelComparison collects raw arrays | Enables composite scoring relative to all models |
| Walk-forward CV with expanding window | Mimics real deployment scenario for time series |
| Composite score = weighted normalised average | Single rankable number that balances MAPE, RMSE, Stability |
| Streamlit session state | Avoids re-training on every user interaction |
