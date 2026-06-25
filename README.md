# BOI CASA Deposit Forecasting System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-Dashboard-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](web/)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-107_Passing-success?style=for-the-badge)](tests/)

CASA Intelligence Dashboard is a banking forecasting intelligence project for Bank of India CASA deposits. The Python pipeline trains and evaluates time-series models, exports verified artifacts, and feeds a modern static Next.js dashboard for model comparison, risk monitoring, confidence scoring, seasonality analytics, and scenario simulation.

The Streamlit dashboard remains available in `dashboard/` as a legacy/internal demo. The portfolio product layer now lives in `web/`.

## Verified Results

Last regenerated in this workspace on June 24, 2026.

| Rank | Model | MAPE | RMSE % | R2 | Adj R2 | Direction Acc | Stability | Composite |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | HoltWinters | 1.518% | 1.797% | 0.8061 | 0.7738 | 100.00% | 37.92 | 98.82 |
| 2 | Prophet | 2.033% | 2.430% | 0.6457 | 0.5866 | 100.00% | 40.29 | 82.10 |
| 3 | ARIMA | 3.214% | 4.148% | -0.0326 | -0.2047 | 71.43% | 0.00 | 18.31 |
| 4 | SARIMAX | 3.453% | 3.885% | 0.0940 | -0.0569 | 100.00% | 0.00 | 15.58 |
| 5 | SARIMA | 3.831% | 4.214% | -0.0658 | -0.2434 | 100.00% | 0.00 | 3.85 |
| 6 | AutoARIMA | 3.942% | 4.351% | -0.1363 | -0.3257 | 100.00% | 0.00 | 0.00 |

Additional verified artifacts:

- `data/processed/model_leaderboard.csv` contains all 6 models.
- `reports/model_registry.json` contains all 6 trained model records.
- `data/processed/ensemble_forecast.csv` uses all 6 models with inverse-MAPE weights.
- `web/public/data/manifest.json` reports `model_count: 6`, `best_model: HoltWinters`, and `best_mape: 1.518`.
- Full Python test suite: `107 passed, 1 warning`.

## Product Architecture

```text
Raw CASA + macro CSV
        |
        v
Python forecasting pipeline
        |
        v
Processed CSVs + model registry + advanced monitoring outputs
        |
        v
scripts/export_web_data.py
        |
        v
web/public/data/*.json
        |
        v
Next.js CASA Intelligence Dashboard
```

## Project Structure

```text
data/
  raw/
  processed/
src/
  data_loader.py
  pipelines/
  evaluation/
  advanced/
  reporting/
  export/web_exporter.py
scripts/
  export_web_data.py
dashboard/
  app.py                  # legacy Streamlit dashboard
web/
  app/                    # Next.js App Router pages
  components/             # cards, charts, layout, dashboard components
  lib/                    # data contract, types, formatters
  public/data/            # generated static JSON
reports/
visualizations/
tests/
```

## ML Pipeline

The pipeline evaluates:

- ARIMA
- SARIMA
- SARIMAX with real macro exogenous variables
- AutoARIMA
- HoltWinters
- Prophet
- Weighted ensemble forecasting through `run_advanced.py`

Quality safeguards include chronological train/test splitting, walk-forward validation, residual diagnostics, confidence scoring, anomaly detection, drift monitoring, model registry export, and tests for train-only exogenous scaling.

## Next.js Dashboard

The new dashboard in `web/` is static and Vercel-ready. It does not call Python and does not duplicate model training logic.

Pages:

- Overview
- Forecast
- Models
- Risk
- Diagnostics
- Seasonality
- Scenario
- Methodology

Dashboard preview placeholders:

- Overview: KPI cockpit, best model card, forecast preview, risk status.
- Forecast: actual vs forecast chart, model selector, confidence band, forecast table.
- Models: leaderboard, comparison charts, radar profile, recommendation.
- Risk: confidence timeline, anomaly status, drift status.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the ML pipeline:

```bash
python run_pipeline.py
python run_advanced.py
```

Generate reports:

```bash
python run_reports.py --formats html markdown --report-dir reports
```

Export static dashboard data:

```bash
python scripts/export_web_data.py
# or
make export-web
```

Run the Next.js dashboard:

```bash
cd web
npm install
npm run dev
```

Run the legacy Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

## Quality Checks

```bash
python -m pytest tests/
python scripts/export_web_data.py
cd web && npm run lint
cd web && npm run build
```

Verified locally:

```text
python -m pytest tests/        -> 107 passed, 1 warning
python scripts/export_web_data.py -> 10 JSON files generated
cd web && npm run lint         -> passed
cd web && npm run build        -> passed
```

## Deploy on Vercel

1. Push the repository to GitHub.
2. In Vercel, set the project root to `web/`.
3. Use:
   - Install command: `npm install`
   - Build command: `npm run build`
   - Output: Next.js default
4. Regenerate `web/public/data/*.json` before deployment whenever the Python pipeline changes.

## Resume Bullets

- Built an end-to-end CASA deposit forecasting system with ARIMA, SARIMA, SARIMAX, AutoARIMA, HoltWinters, Prophet, and ensemble forecasting.
- Added train-only exogenous scaling safeguards to reduce data leakage risk in SARIMAX forecasting.
- Created a static JSON export layer that decouples Python ML artifacts from a modern Next.js analytics dashboard.
- Designed a recruiter-ready banking intelligence dashboard with model leaderboard, forecast explorer, confidence timeline, drift status, and scenario simulator.
- Validated the project with 107 Python tests plus Next.js lint and production build checks.
