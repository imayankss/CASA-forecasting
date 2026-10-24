# Results Dashboard: Strategy and Plan

**Goal:** Replace the 11-page synthetic-era dashboard with one results page in the style of the SecureSwipe methodology page (https://secure-swipe.vercel.app/secureswipe-methodology.html). It shows the measured results of the three real-data tracks, not the datasets.

## Strategy

**Audience.** A recruiter or interviewer with two minutes. They should leave knowing three numbers, why each is honest, and that they can check them.

**Message.** "Deposit forecasts are easy to make look good. This project measures them against baselines, on later periods only, at three scales, and reports when the simple baseline wins."

**What stays and what goes.**

| Keep (results) | Drop (dataset browsing) |
|---|---|
| One headline number per track, with its baseline | Raw series tables, row dumps |
| One interactive chart per track | Synthetic-data pages (forecast, risk, scenario, seasonality, diagnostics, models) |
| Short leaderboards (top models + baselines) | Radar charts, heatmaps, KPI cards on synthetic data |
| Evaluation protocol and "what I fixed" | Per-feature tables |

**Design: copy the reference page's system.**
- Single editorial column (40rem text, 60rem wide blocks), IBM Plex Serif for headings and body prose, Plex Sans for UI, Plex Mono for labels and numbers.
- The reference page's tokens: light `#fcfcfb` / dark `#1a1a19` surfaces, blue `--series-1` for the model, orange `--series-2` for the baseline, red for misses.
- Mono uppercase metadata grid under the title; numbered `01 …` section headings; bordered stat cells; light/dark toggle.

**Interaction (the page is dynamic, but numbers are never recomputed in the browser).** Every control selects among measured, committed results.
1. **BOI forecast explorer:** choose a model and see its holdout forecast against BOI's actual CASA, with its MAPE. Includes the FDIC-trained transfer model and Chronos-2.
2. **Bank-panel origins:** choose models and see MASE at each of the 9 forecast origins, with the 2023Q1 outflow quarter marked.
3. **Call-capacity frontier:** slide from the top 10% to 100% of customers ranked by the model and see the share of new CASA openers reached, the hit rate and the lift. Measured deciles only, no interpolation.

## Sections

0. Header: title, one-line claim, metadata grid (origin, data, evaluation, scale, commit, licence), theme toggle.
1. Lede: the trap (a straight line beats most models on BOI) and the three headline cells.
2. `01` Three scales: what each track asks.
3. `02` Evaluation protocol: later periods only, leakage guards that raise, baselines, uncertainty ranges.
4. `03` Bank of India: forecast explorer plus a short leaderboard.
5. `04` 6,814 US banks: origins chart, leaderboard, paired bootstrap result, sum-of-banks check.
6. `05` Lead scoring: capacity frontier plus the model comparison (lift with 95% CI).
7. `06` What I fixed: the four bugs, one line each.
8. `07` Reproduce: commands and data provenance.

## Tasks

1. **Data.** Extend `src/export/web_exporter.py`: `boi_real.json` gains `forecasts` (from `reports/panel/boi_transfer_forecasts.csv`); `panel.json` gains `mase_by_origin` (from `reports/panel/mase_by_origin.csv`). Re-export. Extend `tests/test_web_exporter.py` only if it covers these payloads.
2. **Theme.** Replace `web/app/globals.css` with the reference tokens and typography. Load IBM Plex through `next/font/google` in `web/app/layout.tsx`. Set the theme before paint (no flash).
3. **Page.** `web/app/page.tsx` (server) reads the three JSON files and renders the sections. Interactive parts are small client components in `web/components/results/`. Charts use the installed `recharts`, coloured by CSS tokens.
4. **Delete.** Remove the old pages and any component, lib file or type that nothing imports any more. Leave the Python exporter's synthetic JSON alone (its tests depend on it).
5. **Verify.** `npm run lint && npm run build`; check the page in both themes at desktop width and at 375px; `pytest tests/` stays green; `git status` shows no stray files.

## Out of scope
- New analysis or re-running models: the page shows committed results only.
- Deployment (Vercel or other): not requested.
