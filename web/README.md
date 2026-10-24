# Results dashboard

A single static page (Next.js) that shows the committed results of the three real-data tracks: Bank of India, the FDIC bank panel and Santander lead scoring. Live at https://casa-forecasting.vercel.app.

Every control switches between measured results read from `public/data/*.json`; nothing is recomputed in the browser.

## Run

```bash
python scripts/export_web_data.py   # from the repo root: rebuild the JSON from reports/ and data/
cd web
npm install
npm run dev
```

## Check

```bash
npm run lint
npm run build
```

## Deploy

Vercel project with root directory `web`; every push to `main` redeploys.
