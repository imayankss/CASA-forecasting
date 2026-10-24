# CASA Scale-Up: 2-Day Plan

> **For the executing session:** run this with `superpowers:executing-plans`. Stop at each **Checkpoint** and report numbers before continuing. Phases marked ‖ run in parallel via `superpowers:dispatching-parallel-agents`. Never run two **[HEAVY]** jobs at once (8 GB RAM).

**Goal:** Turn a 40-row synthetic forecasting project into one built on real data at three scales: (0) Bank of India's real quarterly CASA series, (A) CASA-acquisition lead scoring on 956,645 customers, and (B) deposit forecasting across ~7,500 banks. Every headline number must be leakage-checked, compared against a baseline and reported with an uncertainty range.

**Architecture:** New code follows the repo's existing layout: modules under `src/`, entry points as `run_*.py`, and results registered in `reports/model_registry.json`. Raw downloads go to `data/external/` (gitignored). Public, citable data goes to `data/real/` (committed). The synthetic CSVs stay as fixed test inputs only.

**Tech stack:** Python 3.12 **arm64** venv, Polars + PyArrow, LightGBM, statsforecast 2.1.1, mlforecast 1.1.0, utilsforecast 0.2.17, chronos-forecasting 2.3.2 (Chronos-2, CPU), pdfplumber, kaggle CLI.

**Sources:** `docs/research/customer-level-datasets.md`, `docs/research/panel-deposit-data-and-tools.md`, `docs/research/boi-public-casa.md`. Check details there before guessing.

---

## Decisions (settled with the user, 2026-10-20)

| # | Decision |
|---|---|
| Scope | Track 0 (fixes + real BOI series), Track A (Santander), Track B (FDIC panel). Hierarchical reconciliation is a stretch goal only |
| Compute | This Mac (M2, 8 GB). Polars streaming, Parquet, compact dtypes, no full pandas loads. Kaggle Notebooks only as fallback |
| Branch | `scale-up` in `~/CASA-forecasting`. Commit at the end of each task. Never push without the user's OK |
| Time | 2 × 10 h, with the cut list below applied at each checkpoint |
| Tracking | Extend `reports/model_registry.json`: every run records config, data snapshot (row count + file hash) and git SHA. No MLflow |
| Tests | Test-first **only** for leakage guards, the CASA identity, and the lift-CI and MASE functions. Keep the existing 107 tests green |
| Honesty | README gets a "What I fixed" before/after section. Synthetic data stays labelled. Santander is "Santander's public 13.6M-row dataset", **never "real customers"** (its data page says it contains no real customers). FDIC is US data. The lead-scoring 3.7x lift belongs to the lead-scoring project, not here |
| Dashboard | One page per track via the existing JSON exporter, capped at 2 h |

## Hard constraints

- **The raw Santander data is never committed** (competition rules). Code may be published. Add `data/external/` to `.gitignore` in Task 0.2.
- **Every reported metric is measured on later time periods than the training data.** No random splits anywhere.
- **Kaggle submission is the user's action.** Prepare `submission.csv` and the command; the user runs or approves it.
- **[HEAVY] jobs, run one at a time:** Santander ingest/features, Santander LightGBM training, FDIC LightGBM, Chronos-2 inference.

---

## Cut list (apply in this order when behind at a checkpoint)

1. Kaggle late submission (Task A.7)
2. Dashboard pages (Task 5.2)
3. Chronos-2 on the full panel: drop to a 1,000-bank stratified sample (Task B.5)
4. BOI filings before 2020: ship 2020–2026 (~26 quarters), labelled as partial (Task 0.6)

**Never cut:** leakage tests, baselines, confidence intervals, the provenance columns.

---

## Schedule

| When | Main session | Parallel agent 1 | Parallel agent 2 | [HEAVY] slot |
|---|---|---|---|---|
| D1 0–1 h | Phase 0 setup | — | — | Santander download (I/O only) |
| D1 1–4 h | Track 0 fixes (0.3–0.5) | Real BOI series (0.6–0.7) | FDIC ingest + series build (B.1–B.3) | — |
| **Checkpoint 1** | fixes merged; FDIC Parquet built; BOI ≥ 2020–2026 | | | |
| D1 4–7 h | Track B baselines + global LGBM code (B.4) | Santander target + base-rate prototype (A.2) | — | Santander ingest (A.1), then FDIC CV runs |
| D1 7–10 h | Track A features (A.3) | Re-run pipeline on real BOI (0.7) | — | FDIC LGBM CV |
| **Checkpoint 2** | base rates known; Track B MASE table (baselines vs LGBM) | | | |
| D2 0–3 h | Track A model code + metrics (A.4–A.5) | Track B evaluation + stress window (B.6) | — | Chronos-2 (B.5), then Santander features |
| D2 3–5 h | Track A training + lift CI | Transfer to BOI (B.7) | — | Santander LGBM |
| **Checkpoint 3** | headline numbers for all three tracks | | | |
| D2 5–6 h | Kaggle submission prep (A.7) | — | — | — |
| D2 6–8 h | Registry + dashboard pages (5.1–5.2) | — | — | — |
| D2 8–10 h | README, resume bullets, verification, PR (5.3–5.5) | — | — | — |

---

## Phase 0: Setup (D1, 0–1 h)

### Task 0.1: User pre-flight (you, 10 min)
- [ ] Create a Kaggle API token (kaggle.com, Settings, API) and save it to `~/.kaggle/kaggle.json`, then run `chmod 600 ~/.kaggle/kaggle.json`
- [ ] Accept the rules at https://www.kaggle.com/competitions/santander-product-recommendation/rules

### Task 0.2: Environment
- [ ] `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv` (the Homebrew `uv` on this Mac is an x86_64 build, and torch has no Intel-mac wheels, so don't use it)
- [ ] Add `requirements-scale.txt`: polars, pyarrow, lightgbm, statsforecast==2.1.1, mlforecast==1.1.0, utilsforecast==0.2.17, chronos-forecasting==2.3.2, torch, kaggle, pdfplumber, requests; then `pip install -r requirements.txt -r requirements-scale.txt`
- [ ] Add `data/external/` to `.gitignore`
- [ ] Verify: `.venv/bin/python -c "import platform, polars, lightgbm, statsforecast, mlforecast, chronos; print(platform.machine())"` prints `arm64`
- [ ] Start in background: `kaggle competitions download -c santander-product-recommendation -p data/external/santander` (240 MB)
- [ ] Commit

## Phase 1, Track 0: Foundation fixes + real BOI series

### Task 0.3: Leakage guards (test-first)
- **Files:** create `src/evaluation/temporal.py` and `tests/test_temporal.py`
- [ ] Tests: `assert_temporal_split(train_idx, test_idx)` fails when train ≥ test; `assert_features_known_at(feature_df, origin)` fails if any feature timestamp > origin
- [ ] Implement, then run `pytest tests/test_temporal.py` and see it pass
- [ ] Commit

### Task 0.4: Fix the three pipeline bugs (`src/pipelines/forecasting_pipeline.py`)
- [ ] **SARIMAX look-ahead:** `_fit_sarimax` currently forecasts with the test quarters' actual repo rate, CPI and GDP. Use exog lagged by the forecast horizon (values known at the origin), guarded by `assert_features_known_at`. Keep the old variant only as an explicitly named "scenario (oracle exog)" run, if at all.
- [ ] **Flat ARIMA:** ARIMA(1,1,0) on log values has no drift, so it forecasts a constant. Add `trend="t"`.
- [ ] **Drift test that can't fail:** `src/advanced/drift_detection.py` runs KS on the *levels* of a trending series. Run it on growth rates or residuals instead.
- [ ] Run `pytest tests/`. All tests must pass (update only tests that encoded the bugs, and say which)
- [ ] Commit

### Task 0.5: Honest baselines and metrics
- [ ] Add SeasonalNaive and Drift baselines to the leaderboard
- [ ] Add MASE (m=4) to `src/evaluation/metrics.py`, test-first with a hand-computed case
- [ ] Add a Diebold–Mariano test (best model vs SeasonalNaive) to `src/evaluation/comparison.py`. With 8 test points, report it even if it isn't significant
- [ ] Re-run `run_pipeline.py` and `run_advanced.py`; save the before/after leaderboard to `docs/what-i-fixed.md`
- [ ] Commit

### Task 0.6: Real BOI CASA series ‖ (agent 1)
- **Source:** `docs/research/boi-public-casa.md`: NSE archive filings for BANKINDIA, press releases and investor decks. URLs can't be guessed and must be enumerated.
- **Files:** create `src/real_data/boi_filings.py`, `run_boi_real.py`, `data/real/boi_casa_quarterly.csv`, `tests/test_boi_real.py`
- [ ] Output columns: `quarter_end, fy_quarter, casa_cr, ca_cr, sa_cr, casa_ratio_reported, domestic_deposits_cr, global_deposits_cr, source_url, source_page`. Every row cites its source.
- [ ] Test-first: the five verified quarters in the research note (Mar-15 114,386; Mar-19 181,765; Dec-20 214,826; Mar-23 252,149; Dec-24 277,190) must match exactly
- [ ] Extract from press releases (text-based); fill messy 2016–2019 filings by hand from the PDFs; the Q3FY25 deck is image-only, so use its press release
- [ ] Gotchas: BOI's FY runs April–March (Q3FY25 = Dec-2024). The reported CASA ratio is not CASA ÷ headline domestic deposits, so store it as reported and never recompute it. The research note's "30-Jun-2025" row cites a Q1FY27 (Jun-2026) filing, so check that date
- [ ] Cut rule: if not done by the end of D1, ship 2020–2026 and label it partial
- [ ] Commit (public figures, cited, so it's safe to commit)

### Task 0.7: Re-run the forecasting pipeline on real BOI data ‖ (agent 1)
- [ ] `BOI_DATA_FILE=data/real/boi_casa_quarterly.csv python run_pipeline.py` (`src/config.py` already reads this variable). Adapt `load_casa_data` to accept quarter-end dates if needed, keeping the synthetic path working for tests
- [ ] Expect structural breaks (BOI shed bulk deposits in FY16) and record them
- [ ] Commit

**Checkpoint 1:** fixes merged with tests green; real BOI rows collected so far; FDIC Parquet built (B.1–B.3). Report: the before/after leaderboard and the BOI row count.

## Phase 2, Track B: FDIC panel forecasting ‖ (agent 2 for B.1–B.3)

### Task B.1: Ingest
- **Files:** create `src/panel/fdic.py` and `run_panel.py`
- [ ] Pull `https://api.fdic.gov/banks/financials` for the 102 quarter-ends 2001Q1–2026Q2 (no key; `limit` ≤ 10,000; paginate). Fields: `CERT, REPDTE, NAME, STALP, TRN, NTRSMMDA, NTRSOTH, DEPDOM, NTRTIME, DEP, ASSET`. Cache one Parquet file per quarter in `data/external/fdic/`
- [ ] Verify: about 710,691 rows and 11,243 banks (from the research note); values are in $ thousands
- [ ] Pull `/history` for merger and failure events

### Task B.2: CASA identity (test-first)
- [ ] Test: `TRN + NTRSMMDA + NTRSOTH == DEPDOM − NTRTIME` within rounding on a sample. The research note checked this exactly on JPMorgan 2019Q4
- [ ] Target `y = TRN + NTRSMMDA + NTRSOTH`. **Forecast the sum only, never the parts:** after Reg D (2020) banks reclassified savings as transaction accounts (Wells Fargo ~$730bn in 2020Q4)

### Task B.3: Panel build
- [ ] Keep banks with ≥ 40 quarters (~7,488). Cut series at mergers and large breaks flagged by `/history`. **Keep failed banks** (SVB and Signature end 2022Q4) so backtests have no survivorship bias
- [ ] Write `data/external/fdic/panel.parquet` in the `unique_id, ds, y` format
- [ ] Commit

### Task B.4: Baselines + global model
- [ ] Re-forecast from every quarter-end 2021Q4–2023Q4 (9 origins), h = 1–4, on log y
- [ ] statsforecast: SeasonalNaive, Naive, AutoETS, AutoARIMA, AutoTheta, per bank
- [ ] mlforecast: **one** LightGBM model across all banks: lags 1–8, rolling means, quarter-of-year, differences target transform. **[HEAVY]**
- [ ] Commit

### Task B.5: Chronos-2 zero-shot **[HEAVY]**
- [ ] Run it alone. Cut rule: if the full panel × 9 origins exceeds ~2 h on CPU, use a 1,000-bank sample stratified by asset size and say so

### Task B.6: Evaluation
- [ ] Use utilsforecast: MASE (m=4) per bank, origin and horizon; scaled CRPS for the quantile models
- [ ] Paired bootstrap across banks for global LGBM vs AutoETS (a 95% CI on the MASE difference)
- [ ] **Stress window:** 2023Q1 (industry CASA fell about $847bn). Compare performance on that quarter vs the rest
- [ ] Sanity check: sum of bank forecasts vs the industry total
- [ ] Commit `reports/panel/*.csv`

### Task B.7: Transfer to BOI
- [ ] Apply the FDIC-trained LGBM (`mlforecast predict(new_df=...)`) and Chronos-2 zero-shot to the real BOI series; compare with Holt-Winters and SeasonalNaive on the same holdout
- [ ] Whatever the result, report it. "The global model didn't beat Holt-Winters on BOI" is a valid finding
- [ ] Commit

**Checkpoint 2 (end of D1):** the Track B MASE table (baselines vs LGBM) and the Santander base rates from A.2.

## Phase 3, Track A: CASA-acquisition lead scoring (Santander)

### Task A.1: Ingest **[HEAVY]**
- **Files:** create `src/propensity/santander.py` and `run_propensity.py`
- [ ] Polars `scan_csv` with a schema (product flags `Int8`, strings `Categorical`) → `sink_parquet` to `data/external/santander/`, partitioned by `fecha_dato`
- [ ] Verify: 13,647,309 rows, 956,645 customers, 17 months (2015-01-28 to 2016-05-28)

### Task A.2: Target + base-rate prototype (`mattpocock-skills:prototype`, throwaway)
- **CASA set (assumption, written into the README):** `ind_cco` (current), `ind_ctop`/`ind_ctpp`/`ind_ctma` (particular accounts), `ind_ecue` (e-account), `ind_cno` (payroll), `ind_ahor` (savings). Exclude `ind_ctju` (junior). Kaggle doesn't classify the particular accounts, so this is our reading
- [ ] Target `casa_add(t+1) = 1` if the customer holds a CASA-set product at t+1 that they didn't hold at t. The population is customers present at both t and t+1. **Never use "holds"**, which inflates lift
- [ ] Prototype: print the monthly base rate and adds by product. Expect spikes in Jun and Dec 2015; savings alone is too rare to model. Then delete the prototype
- [ ] Test-first: target construction on a 3-customer, 3-month fixture

### Task A.3: Features (everything known at t)
- [ ] Demographics (age, `renta` + missing flag, `segmento`, `antiguedad`, `canal_entrada`, `ind_actividad_cliente`, `tiprel_1mes`), current ownership, ownership lags t-1 to t-5, product count, months since the last add or drop
- [ ] Guard with `assert_features_known_at`
- [ ] Commit

### Task A.4: Splits + model
- [ ] Train on target months 2015-07 to 2016-03; validate on 2016-04; **test on 2016-05** (features up to 2016-04)
- [ ] Baselines: base rate; a rule (active customers ranked by product count); logistic regression
- [ ] LightGBM binary with early stopping on validation. Downsample negatives if memory requires it, and reweight for calibration **[HEAVY]**
- [ ] Commit

### Task A.5: Metrics (test-first for the lift CI)
- [ ] `lift_at_k(y, score, k=0.1)` and a customer-level bootstrap 95% CI (1,000 resamples), tested on a hand-computed case
- [ ] Report on the 2016-05 test month: base rate, lift@10% with CI (vs the rule baseline), cumulative gains curve, PR-AUC, Brier score, and lift by `segmento`
- [ ] Save to `reports/propensity/*.csv` and register the run
- [ ] Commit

**Checkpoint 3:** the headline table for all three tracks. Apply the cut list here.

### Task A.6: Stretch, uplift by month
- [ ] Optional: does lift hold on 2015-12 (a seasonal spike)? Skip if behind

### Task A.7: Kaggle late submission (cut first)
- [ ] Train 24 per-product models (or one multiclass model) on "added products" and write `submission.csv` for 2016-06-28 (top 7 per customer)
- [ ] Hand the user the command: `kaggle competitions submit -c santander-product-recommendation -f submission.csv -m "CASA scale-up"`. **The user runs it.** Record the private MAP@7 separately from the CASA lift (it covers all 24 products). For reference, 1st place scored 0.03140

## Phase 4: Integration (D2, 6–10 h)

### Task 5.1: Registry
- [ ] Extend `src/forecasting/registry.py`: each record gets `config`, `data_snapshot` (rows + sha256 of the input Parquet or CSV) and `git_sha`. Test that a record round-trips
- [ ] Commit

### Task 5.2: Dashboard (cap 2 h, cut second)
- [ ] Extend `src/export/web_exporter.py` with `propensity.json`, `panel.json` and `boi_real.json`; add `web/app/propensity` and `web/app/panel` pages reusing the existing chart and table components
- [ ] Mark synthetic vs real data on every page
- [ ] `cd web && npm install && npm run lint && npm run build`
- [ ] Commit

### Task 5.3: README rewrite
- [ ] Order: real BOI results → Track B panel → Track A lead scoring → "What I fixed" → data provenance table (source, licence, real or synthetic) → how to reproduce. Remove the "Results on synthetic data" table from the top, or move it under the synthetic demo
- [ ] Resume bullets, filled only from the measured numbers:
  - "Rebuilt a Bank of India CASA forecasting pipeline on 46 quarters of BOI's published CASA figures; found and removed a look-ahead leak; [best model] MASE [x] vs seasonal-naive [y]."
  - "Scaled deposit forecasting to ~7,500 US banks (FDIC Call Reports, 2001–2026): one global LightGBM model cut MASE by [x]% (95% CI [a, b]) vs per-bank AutoETS, including during the 2023 deposit outflows."
  - "Built CASA-acquisition lead scoring on Santander's public 13.6M-row, 957k-customer dataset: top-decile lift [x] (95% CI [a, b]), measured on a later month than training [, Kaggle private MAP@7 [z]]."

### Task 5.4: Verification (`superpowers:verification-before-completion`)
- [ ] `pytest tests/` passes; `run_pipeline.py`, `run_boi_real.py`, `run_panel.py` and `run_propensity.py` run end to end
- [ ] Every number in the README matches a file in `reports/` or `data/processed/`. Check each one
- [ ] `git status` shows no files from `data/external/`

### Task 5.5: PR
- [ ] Ask the user before pushing. Then push `scale-up` and open a PR against `main`

---

## Risks

| Risk | Mitigation |
|---|---|
| NSE blocks scripted downloads | Fetch filings one by one with a browser user agent and backoff; fall back to the URLs in the research note plus manual entry; apply the cut rule |
| 8 GB memory pressure on Santander | Streaming Parquet, `Int8` flags, train on downsampled negatives; Kaggle Notebook as fallback |
| Chronos-2 is slow on CPU | Stratified 1,000-bank sample (cut rule 3) |
| Real BOI results look worse than the synthetic ones | Expected. Report them honestly; that's the point of the exercise |
