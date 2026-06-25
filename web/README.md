# CASA Intelligence Dashboard

Modern static dashboard for the BOI CASA Deposit Forecasting System.

## Run

```bash
cd ..
python scripts/export_web_data.py

cd web
npm install
npm run dev
```

## Build

```bash
cd web
npm run lint
npm run build
```

The app reads exported JSON from `web/public/data/` and does not call Python or a backend API.

