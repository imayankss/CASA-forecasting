# Panel deposit data and forecasting tools: scaling the BOI CASA forecaster

Research date: 2026-10-20. Every fact is followed by its source. "Live API query" means I ran that exact query (small, `limit<=10` or aggregate-only) on 2026-10-20. Anything not confirmed from a primary source is marked **unverified**.

---

## Summary and recommendation

**Dataset: use FDIC BankFind `/financials` (Call Report data) as the training panel.**

- It is free, needs no key, and returns JSON or CSV. It covers every FDIC-insured institution quarterly from 1984Q1 to 2026Q2. Since 2001Q1 that is **710,691 bank-quarter rows from 11,243 institutions**. Of those, **7,488 banks have at least 40 quarters**, the same length as the BOI series, and **3,886 have all 102 quarters**. (Source: live API queries, section 1.)
- The **US CASA analog is `TRN + NTRSMMDA + NTRSOTH`**: transaction accounts plus MMDA savings plus other savings. This equals `DEPDOM - NTRTIME` exactly, i.e. domestic deposits minus time deposits. (Source: FDIC dictionary and an identity check in section 1.)
- **Always model the sum, never its parts.** Banks have moved hundreds of billions of dollars between "other savings" and "transaction" after the 2020 Regulation D change: Wells Fargo in 2020Q4 and Bank of America in 2025Q4. The CASA sum is unaffected by these moves.
- The whole pull is about 102 API calls, one per quarter-end with `limit=10000`, and about 60 MB in memory. It fits easily on a 16 GB Mac.

**FRED H.8 (weekly, 1973 onward)** is a good macro covariate and sanity check. It gives total deposits, large time deposits and "other deposits", but no CASA split. (Section 5.)

**Indian data**: RBI's Basic Statistical Returns are the only Indian source with a current/savings/term split across many series:
- BSR-7 quarterly by state and district, with type-wise deposits from March 2012 to December 2022.
- Quarterly BSR-2 from March 2023: state, district, population group and bank group.

They are useful for validating the transfer and as a secondary panel, but they are hard to automate:
- Downloads are DBIE Excel files, and I found no documented API.
- Every RBI host returned 403, "request blocked" or a timeout from this environment, including in a real browser.
- There is a methodology break at 2023 (BSR-7 to BSR-2).

(Section 4.)

**Tool stack for a 2-day CPU sprint** (versions from PyPI, 2026-10-20):

| Role | Package (version) | Why |
|---|---|---|
| Local baselines on thousands of series | `statsforecast` 2.1.1 | AutoETS, AutoARIMA, AutoTheta and SeasonalNaive with built-in cross-validation, multi-core |
| Global ML model | `mlforecast` 1.1.0 + `lightgbm` | One model across all banks. Lag, rolling and seasonal-rolling features, target transforms (`Differences`, `LocalStandardScaler`, `LocalBoxCox`), conformal intervals, and `predict(new_df=...)` to transfer straight to the BOI series |
| Zero-shot foundation model | `chronos-forecasting` 2.3.2 with `amazon/chronos-2` | Apache-2.0, CPU inference supported, covariates, context up to 8192 |
| Evaluation | `utilsforecast` 0.2.17 | `mase`, `scaled_crps`, `quantile_loss`, `scaled_mqloss`, `coverage` |
| Optional | `hierarchicalforecast` 1.5.3 (MinTrace; bank to state to national), `neuralforecast` 3.3.0 (NHITS) | Only if time remains. Skip TFT on CPU in 2 days |
| Avoid for a bank-facing deliverable | TimesFM 3.0 weights (non-commercial license), Moirai 2.0 (CC-BY-NC-4.0) | Licensing. TimesFM 2.5 (Apache-2.0) is fine as a second foundation model |

**Most credible story:** ARIMA/ETS/Theta local baselines, then a LightGBM global model trained on about 7.5k US bank CASA series, then Chronos-2 zero-shot.
- Score all three with rolling-origin cross-validation using MASE and scaled CRPS.
- Then transfer to the BOI series: run the global model via `predict(new_df=BOI)` and Chronos-2 zero-shot.
- Compare against the original single-series ARIMA/SARIMA/Holt-Winters/Prophet results.

---

## 1. FDIC BankFind Suite: financial (Call Report) data

### API mechanics

| Fact | Source |
|---|---|
| The API lives under `https://api.fdic.gov/banks`. The OpenAPI spec declares `servers: url: '/banks'` and the docs are at `/banks/docs/`. | https://api.fdic.gov/banks/docs/swagger.yaml ; https://api.fdic.gov/banks/docs/ |
| Endpoints: `/institutions`, `/locations`, `/summary`, `/failures`, `/history`, `/financials`, `/demographics`, `/sod` | https://api.fdic.gov/banks/docs/swagger.yaml |
| `/financials` parameters: `filters`, `fields`, `sort_by` (default `REPDTE`), `sort_order` (ASC/DESC, default ASC), `limit`, `offset`, `agg_by`, `agg_term_fields`, `agg_sum_fields`, `agg_limit`, `format` (json/csv), `download`, `filename`, and an optional `api_key` | https://api.fdic.gov/banks/docs/swagger.yaml |
| Spec text for `limit`: "Default is 10 and maximum is 10,000. However, if the fields request is for more than 250 fields (variables), the maximum limit is 500." `agg_limit` maximum is 10000. | https://api.fdic.gov/banks/docs/swagger.yaml |
| Field names and filter values must be UPPERCASE | https://api.fdic.gov/banks/docs/swagger.yaml |
| Filters use Elasticsearch query-string syntax. Examples: `NAME:"First Bank" AND STALP:IA`, `!(BKCLASS:NM OR BKCLASS:N)`, ranges with `[]` (inclusive) or `{}` (exclusive) and `*`, e.g. `DEPDOM:[* TO 10000]` | https://api.fdic.gov/banks/docs/ |
| Dollar values are in **thousands** ("domestic deposits under 10,000 ($thousands)") | https://api.fdic.gov/banks/docs/ |
| API key: "Users are currently not required to register and obtain a unique 'key' to use with every query" | https://api.fdic.gov/banks/docs/ |
| `REPDTE` filter format is `yyyymmdd`, e.g. `REPDTE:20260630` (the docs' `yyyy-mm-dd` example applies to other date fields) | Live API query: `https://api.fdic.gov/banks/financials?filters=REPDTE:20260630&fields=CERT&limit=1` |
| Deep `offset` works: `offset=700000` returned rows | Live API query: `.../financials?filters=REPDTE:[20010331 TO 20261231]&fields=CERT,REPDTE&limit=5&offset=700000` |
| Rate limit: no limit is documented. Responses carry `x-ratelimit-limit: 20` (an api-umbrella gateway header), and `x-ratelimit-remaining` stayed at 19 across sequential calls. **The window is unverified**, so pace about 1 request per second. | Live response headers; docs page silent |
| Aggregation works server-side, e.g. sum by quarter: `agg_by=REPDTE&agg_sum_fields=TRN,NTRSMMDA,...&limit=0` | Live API query (section 1, aggregates) |
| Index snapshot used on 2026-10-20: `risview_20260819185831`. The latest `REPDTE` is 20260630. | Live API query `meta.index` |

Sample call that confirms field names (CERT 14, `limit=5`):
`https://api.fdic.gov/banks/financials?filters=CERT:14&fields=CERT,REPDTE,ASSET,DEP,DEPDOM,TRN,NTRSMMDA,NTRSOTH,NTRTIME,COREDEP,DEPUNINS&limit=5&sort_by=REPDTE&sort_order=DESC`
It returned all requested fields for 20260630 back to 20250630. (Live API query.)

### History range and row counts

| Fact | Value | Source |
|---|---|---|
| Earliest / latest report date | 19840331 / 20260630 | Live: `.../financials?fields=CERT,REPDTE&limit=2&sort_by=REPDTE&sort_order=ASC` (and DESC) |
| Total rows, all years | 1,682,616 | Live: `meta.total` of the query above |
| Rows since 1992Q1 | 1,140,484 | Live: `filters=REPDTE:[19920101 TO 20261231]` |
| **Rows since 2001Q1 (102 quarters)** | **710,691** | Live: `filters=REPDTE:[20010331 TO 20261231]` |
| Institutions reporting, 2001Q4 / 2010Q4 / 2019Q4 / 2020Q2 / 2023Q4 / 2025Q2 / 2026Q2 | 9,731 / 7,747 / 5,244 / 5,131 / 4,658 / 4,494 / 4,313 | Live: `filters=REPDTE:<date>&limit=1` (`meta.total`) |
| Distinct CERTs since 2001Q1 | 11,243 | Live: `agg_by=CERT`, split into CERT ranges `[0 TO 15000]` and `[15001 TO *]` to stay under `agg_limit` 10,000 |
| CERTs with at least 8 / 20 / **40** / 60 / 80 / 100 / all 102 quarters (2001Q1 to 2026Q2) | 10,500 / 9,380 / **7,488** / 6,035 / 4,833 / 3,993 / **3,886** | Same live aggregation |
| Practical pull | About 102 calls (one per `REPDTE`; the largest quarter is 2001Q1 with 9,940 rows, under the 10k limit). About 0.7M rows x 12 numeric columns is about 70 MB as float64. | Derived from the counts above; live `filters=REPDTE:20010331&limit=1` gives 9,940 |

### Field codes (FDIC financial data dictionary `risview_properties.yaml`)

| Concept | Code | Official title in dictionary | Source |
|---|---|---|---|
| Total assets | `ASSET` | "Total assets" | https://api.fdic.gov/banks/docs/risview_properties.yaml |
| Total deposits | `DEP` | "Total deposits", the sum of demand, money market, other savings, time deposits and deposits in foreign offices | same |
| Domestic deposits | `DEPDOM` | "Deposits held in domestic offices", the sum of domestic demand, money market, other savings and time deposits | same |
| Foreign-office deposits | `DEPFOR` | "TOTAL DEPOSITS-FOR" | same |
| **Transaction accounts** | `TRN` | "TRANSACTION-TOTAL" (also `TRNIPC`, `TRNMUNI`, `TRNUSGOV`) | same |
| Demand deposits (subset of TRN) | `DDT` | "DDA TRANS-TOTAL" | same |
| Nontransaction total | `NTR` | "NONTRANSACTION-TOTAL" | same |
| **Savings: MMDA** | `NTRSMMDA` | "SAVINGS DEP-MMDA" | same |
| **Savings: other** | `NTRSOTH` | "SAVINGS DEP-OTHER" | same |
| Time deposits | `NTRTIME` | "TIME DEPOSITS-TOTAL" (also `NTRTMLGJ` = time deposits over $250,000, `NTRTMMED` = time deposits of $250,000 or less) | same |
| Time + savings | `TS` | "TIME & SAVINGS DEPOSITS-TOTAL" | same |
| Core deposits | `COREDEP` | "CORE DEPOSITS" (no description in the dictionary) | same |
| **Estimated uninsured** | `DEPUNINS` | "ESTIMATED UNINSURED DEPOSITS IN DOMESTIC OFFICES AND IN INSURED BRANCHES IN US TERRITORIES AND POSSESSIONS" | same |
| Estimated insured | `DEPINS` | "ESTIMATED INSURED DEPOSITS" | same |
| Accounts over / at or under $250k (amounts) | `DEPLGAMT` / `DEPSMAMT` | "AMT DEP ACC GREATER THAN $250,000" / "AMT DEP ACC AT $250,000 OR LESS" | same |
| Noninterest / interest-bearing domestic | `DEPNIDOM` / `DEPIDOM` | | same |
| Brokered | `BRO` | "BROKERED DEP" | same |
| Report form | `CALLFORM` | "Call Form Number" | same |
| Merger flag on the financial row | `ACTEVT` | "Structure activity event code. Merger or closing codes only." | same |

**CASA analog, verified by identity.** For JPMorgan Chase (CERT 628) at 20191231:

`TRN` 352,272,000 + `NTRSMMDA` 546,395,000 + `NTRSOTH` 355,601,000 + `NTRTIME` 100,800,000 = **1,355,068,000 = `DEPDOM`** exactly ($ thousands).

So `CASA_US = TRN + NTRSMMDA + NTRSOTH = DEPDOM - NTRTIME`. (Live: `.../financials?filters=CERT:628 AND REPDTE:[20191231 TO 20210331]&fields=...`.)

The same identity holds for thrifts that filed the TFR: their 2011Q4 sums are TRN 66.4 + MMDA 315.0 + other savings 86.8 + time 193.7 = 661.9 vs DEPDOM 661.8 ($bn). (Live: `filters=REPDTE:20111231 AND CALLFORM:100&agg_by=REPDTE&agg_sum_fields=...`.)

**`COREDEP`, empirically.**
- It matches `DEPDOM - BRO - NTRTMLGJ` exactly for Citizens (CERT 57957, 2025Q4).
- It is close to that formula for JPM, BofA and State Street. This is consistent with the UBPR-style definition (domestic deposits minus time deposits over $250k minus insured brokered deposits).
- The FDIC Quarterly Banking Profile instead defines core deposits as "non-brokered deposits in domestic offices" for its community-bank screen.

(Sources: live query; https://www.fdic.gov/quarterly-banking-profile/qbp-notes-users-fourth-quarter-2025.pdf.) The exact `COREDEP` formula is **unverified**, so do not use it as the target.

**`DEPUNINS` provenance.** "Institutions with $1 billion or more in total assets report estimated uninsured deposits... For institutions that do not report estimated uninsured deposits, the FDIC calculates this amount" from accounts above the insurance limit. (Source: https://www.fdic.gov/quarterly-banking-profile/qbp-notes-users-fourth-quarter-2025.pdf.)

### Industry aggregates (all reporters, $bn; CASA = TRN + MMDA + other savings)

| REPDTE | n | TRN | MMDA | Other sav. | Time | DEPDOM | CASA | Uninsured |
|---|---|---|---|---|---|---|---|---|
| 20191231 | 5244 | 2339 | 5647 | 3205 | 2042 | 13301 | 11191 | 5874 |
| 20200930 | 5098 | 3596 | 6597 | 3855 | 1653 | 15765 | 14048 | 7282 |
| **20201231** | 5067 | **4759** | 6782 | **3257** | 1523 | 16387 | 14798 | 7683 |
| 20221231 | 4773 | 6381 | 6048 | 3621 | 1709 | 17830 | 16049 | 8169 |
| **20230331** | 4740 | 6296 | **5509** | 3396 | **2133** | **17406** | **15202** | **7526** |
| 20230630 | 4714 | 6187 | 5372 | 3233 | 2440 | 17305 | 14791 | 7330 |
| 20250930 | 4452 | 6923 | 5235 | 3114 | 2885 | 18246 | 15273 | 7925 |
| **20251231** | 4411 | **7989** | 5309 | **2319** | 2864 | 18573 | 15617 | 8138 |
| 20260630 | 4313 | 8395 | 5334 | 2365 | 2927 | 19119 | 16094 | 8693 |

Source: live API query `https://api.fdic.gov/banks/financials?filters=REPDTE:[20190331 TO 20261231]&agg_by=REPDTE&agg_sum_fields=TRN,NTRSMMDA,NTRSOTH,NTRTIME,DEPDOM,DEPUNINS&agg_limit=100&limit=0`. Bold cells mark the breaks discussed below.

### Merger and structure history

| Fact | Source |
|---|---|
| `/history` returns structure-change events with `CERT`, `CHANGECODE`, `CHANGECODE_DESC`, `EFFDATE`, `ACQ_CERT`, `OUT_CERT`, `SUR_CERT`, `FRM_CERT` (previous CERT), `TRANSNUM` | https://api.fdic.gov/banks/docs/history_properties.yaml |
| Merger and closure codes seen in the data:<br>211 "Failure - Whole Institution" (2,550 events)<br>213 "Merger - Assisted"<br>216 Bridge Bank<br>217 Passthrough receivership<br>221 "Absorbtion - Without Assistance" (1,026)<br>222 "Consolidated - Without Assistance"<br>223 "Merger -Without Assistance" (18,379)<br>224 affiliated merger<br>810/811 "Participated in ... Merger" (acquirer side)<br>150 "Cert Changed from Old Cert" | Live: `https://api.fdic.gov/banks/history?filters=CHANGECODE_DESC:*&agg_by=CHANGECODE&agg_term_fields=CHANGECODE_DESC&agg_limit=60&limit=0` |
| Example: `SUR_CERT:628` (JPM) shows First Republic (OUT_CERT 59017, code 211/811) on 2023-05-01, and Chase Bank USA (23702, code 223/810) on 2019-05-18. Each one steps JPM's series up. | Live: `https://api.fdic.gov/banks/history?filters=SUR_CERT:628&fields=CERT,INSTNAME,CHANGECODE,CHANGECODE_DESC,EFFDATE,ACQ_CERT,OUT_CERT,SUR_CERT,OUT_INSTNAME&limit=5&sort_by=EFFDATE&sort_order=DESC` |
| `/institutions` has `ACTIVE`, `INACTIVE`, `ENDEFYMD` (end date), `CHANGEC1`, `FED_RSSD` for linking to FFIEC/Fed data | https://api.fdic.gov/banks/docs/institution_properties.yaml |

### 2023 regional-bank outflows: visible

| Fact | Source |
|---|---|
| Industry 2022Q4 to 2023Q1: DEPDOM fell $424bn (17,830 to 17,406), CASA fell $847bn (16,049 to 15,202), uninsured fell $643bn, time deposits rose $424bn (CASA-to-term migration) | Live aggregate query above |
| Silicon Valley Bank (CERT 24735): last report 20221231 (DEPDOM $161.5bn, DEPUNINS $151.6bn). It is absent afterwards, so it is a survivorship case. | Live: `.../financials?filters=CERT:24735&...&sort_order=DESC&limit=3` |
| Signature Bank (CERT 57053): last report 20221231 | Live: `.../financials?filters=CERT:57053&...&limit=1` |
| First Republic (CERT 59017): DEPDOM 176.4 to 104.5 $bn and CASA 151.2 to 54.6 $bn (-64%) from 2022Q4 to 2023Q1, then absorbed by JPM on 2023-05-01 | Live: `.../financials?filters=CERT:59017 AND REPDTE:[20220630 TO 20231231]&...` ; history query above |

---

## 2. FFIEC CDR bulk download (alternative)

| Fact | Source |
|---|---|
| URL: `https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx`. Products: "Call Reports -- Single Period", "Call Reports -- Balance Sheet, Income Statement, Past Due -- Four Periods", UBPR Ratio / Rank / Stats | https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx |
| "All of the data items for a single quarter can be downloaded in Tab Delimited format or eXtensible Business Reporting Language (XBRL) format" | https://cdr.ffiec.gov/public/HelpFiles/WelcomeAdditionalInfo.htm |
| Files are available "45 calendar days after the report date" and "regenerated monthly to capture amendments", so there are no as-reported vintages | same |
| Earliest UBPR period is December 2002 | same |
| Covers Call Report filers only (commercial banks, plus savings institutions since 2012Q1). Raw MDRM codes (e.g. RCON2215 transaction accounts, RCON6810 MMDA, RCON0352 other savings) instead of FDIC's harmonized names. **MDRM codes are unverified here.** | Coverage: QBP notes (TFR filers to Call Report from 2012Q1), https://www.fdic.gov/quarterly-banking-profile/qbp-notes-users-fourth-quarter-2025.pdf |
| Verdict: use only if you need an item FDIC does not expose. FDIC already harmonizes TFR and Call data and is far easier to pull. | Judgment |
| Earliest Call Report bulk period (believed to be 2001) | **unverified** (the period dropdown is loaded dynamically and could not be read) |

Side option: FDIC **Summary of Deposits** (`/sod`). It gives branch-level total deposits (`DEPSUMBR`) annually (June 30) from 1994. There were 76,097 branches in 2025 and 2,898,836 rows in total. It has **no CASA split**. (Live: `https://api.fdic.gov/banks/sod?filters=YEAR:2025&fields=CERT,BRNUM,YEAR,DEPSUMBR,STALPBR&limit=2`.)

---

## 3. Reporting-form landscape (needed to understand breaks)

| Fact | Source |
|---|---|
| `CALLFORM` counts, from live `agg_by=CALLFORM`:<br>2010Q4: 41 = 6,886; 31 = 113; 100 = 731; 2 = 17<br>2012Q1: 41 = 7,279; 31 = 98; 2 = 17<br>2017Q1: **51 = 3,505**; 41 = 2,333; 31 = 70<br>2026Q2: 51 = 3,282; 41 = 934; 31 = 80; 2 = 17 | Live: `https://api.fdic.gov/banks/financials?filters=REPDTE:20170331&agg_by=CALLFORM&agg_limit=20&limit=0` |
| CALLFORM 100 is the OTS Thrift Financial Report. **This is inferred**: it disappears exactly at 2012Q1, and "TFR filers began filing Call Reports effective with the quarter ending March 31, 2012". FDIC also notes "certain adjustments are made to the OTS Thrift Financial Reports to provide closer conformance". | https://www.fdic.gov/quarterly-banking-profile/qbp-notes-users-fourth-quarter-2025.pdf |
| FFIEC 051 (streamlined form) took effect March 31, 2017 for institutions with domestic offices only and total assets under $1bn. Filing it was optional. | https://www.fdic.gov/news/inactive-financial-institution-letters/2017/fil17001.html |
| FFIEC 051 eligibility expanded to under $5bn (final rule effective July 22, 2019). Newly eligible banks could switch from 2019Q3 or 2019Q4. | https://www.fdic.gov/news/inactive-financial-institution-letters/2019/fil19035.pdf ; https://www.fdic.gov/news/inactive-financial-institution-letters/2019/fil19040.html |

---

## 4. Indian sources (RBI)

**Access warning.** From this environment, `rbi.org.in`, `website.rbi.org.in`, `rbidocs.rbi.org.in`, `dbie.rbi.org.in` and `data.rbi.org.in` all returned HTTP 403, "The request is blocked" or a timeout, both via fetch and in a real browser. The facts below come from search-engine extracts of RBI's own pages: the source text is RBI's, but I could not open the pages. Plan for manual Excel downloads from an Indian browser session.

| Source | Granularity / split | Frequency and history | Format | Source URL |
|---|---|---|---|---|
| **WSS "Scheduled Commercial Banks - Business in India"** | All-India aggregate deposits split into **demand** and **time** only | Fortnightly | PDF/HTML tables; series in DBIE | https://website.rbi.org.in/en/web/rbi/-/publications/weekly-statistical-supplement/scheduled-commercial-banks-business-in-india-17993 |
| Handbook of Statistics, "SCB - Select Aggregates" | Demand / time / aggregate deposits | Annual from 1950-51 (Table 47). A fortnightly table covers recent years only (Table 212). | PDF/Excel | https://website.rbi.org.in/web/rbi/-/publications/table-47-scheduled-commercial-banks-select-aggregates-15837 |
| **BSR-7 "Quarterly Statistics on Deposits and Credit of SCBs"** | State, district, centre, population group, bank group. **Type-wise (current / savings / term) state and district time series from the March 2012 round.** | Quarterly. Reference dates were 31 March and the last Friday of June, September and December. **Discontinued after December 2022.** | DBIE | Type-wise from 2012: RBI March 2016 release as reproduced at https://www.theasianbanker.com/press-releases/rbi-releases-quarterly-statistics-on-deposits-and-credit-of-scheduled-commercial-banks:-march-2016 (secondary copy; RBI original blocked). Discontinuation: https://website.rbi.org.in/web/rbi/-/press-releases/quarterly-bsr-1-outstanding-credit-of-scheduled-commercial-banks-june-2023-56294 |
| **Quarterly BSR-2 "Deposits with SCBs"** (replaces BSR-7 for deposits) | Branch-level returns released by **type of deposit (current, savings, term)**, population group, bank group, state, district, centre, interest-rate range, size and maturity. Excludes RRBs. | Quarterly from March 2023. The latest found is June 2026, released 2026-08-31. | DBIE (data.rbi.org.in, "Publications") | https://website.rbi.org.in/as/web/rbi/-/press-releases/quarterly-bsr-2-deposits-with-scheduled-commercial-banks-june-2023-56293 ; https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=63476 |
| **Annual BSR-2 "Composition and Ownership Pattern of Deposits"** | Type (current / savings / term) x institutional ownership x age. Disaggregated by population group, bank group, state, district, centre. Savings share was 29.1% in March 2025 vs 30.8% a year earlier. | Annual (March) | DBIE Excel tables | https://website.rbi.org.in/web/rbi/-/press-releases/deposits-with-scheduled-commercial-banks-march-2025-annual-bsr-2- |
| **Statistical Tables Relating to Banks in India** | "Entity-wise information on major items of liabilities and assets", i.e. **bank-wise**, SCBs excluding RRBs. 2024-25 edition released 2025-12-29. Electronic-only on DBIE since 2013-14. | Annual | DBIE | https://website.rbi.org.in/web/rbi/-/press-releases/rbi-releases-statistical-tables-relating-to-banks-in-india-2024-25- ; https://website.rbi.org.in/web/rbi/-/press-releases/rbi-releases-statistical-tables-relating-to-banks-in-india-2013-14-32875 |
| DBIE portal | Moved to `https://data.rbi.org.in` from 2024-06-21. Excel/CSV/PDF downloads. **No public API found.** | | | https://currentaffairs.adda247.com/reserve-bank-of-india-updates-url-for-database-on-indian-economy-portal/ (secondary report of the RBI notice) |

**Conclusion (India).**
- The largest Indian panel with a CASA split is **district x deposit type** (and state x bank group x type): BSR-7 quarterly from 2012Q1 to 2022Q4, spliced to quarterly BSR-2 from 2023Q1.
  - The number of districts, and whether type-wise district series are continuous across the 2023 switch, are **unverified**.
- Bank-wise CASA (STRBI) is annual only. That table-level split is **unverified**, though Indian bank balance sheets report demand / savings / term deposits.
- Effort: manual Excel downloads from DBIE, no API, and RBI hosts block scripted access from outside India. Use it as a validation and secondary panel, not the main training set, in a 2-day sprint.

---

## 5. FRED: H.8 weekly deposits

| Series ID | Title | Frequency / SA | First obs | Last obs (as of 2026-10-20) | Source |
|---|---|---|---|---|---|
| `DPSACBW027SBOG` | Deposits, All Commercial Banks | Weekly (Wed), SA, $bn | 1973-01-03 | 2026-10-07 (19,771.5) | https://fred.stlouisfed.org/series/DPSACBW027SBOG ; CSV `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DPSACBW027SBOG,LTDACBW027SBOG,ODSACBW027SBOG` |
| `DPSACBW027NBOG` | same, not seasonally adjusted | Weekly, NSA | 1973-01-03 | 2026-10-07 | same series page ("Other formats") and CSV |
| `DPSACBM027SBOG` / `DPSACBM027NBOG` | same, monthly | Monthly | not checked | | https://fred.stlouisfed.org/series/DPSACBW027SBOG |
| `LTDACBW027SBOG` | Large Time Deposits, All Commercial Banks | Weekly, SA | 1973-01-03 | 2026-10-07 (2,605.7) | https://fred.stlouisfed.org/series/LTDACBW027SBOG |
| `ODSACBW027SBOG` | Other Deposits, All Commercial Banks | Weekly, SA | **2009-07-01** (902 obs) | 2026-10-07 (17,165.8) | https://fred.stlouisfed.org/series/ODSACBW027SBOG ; CSV above |
| `DPSLCBW027SBOG` | Deposits, Large Domestically Chartered Commercial Banks | Weekly, SA | 1985-04-03 | 2026-10-07 | https://fred.stlouisfed.org/series/DPSSCBW027SBOG ; CSV `...fredgraph.csv?id=DPSLCBW027SBOG,DPSSCBW027SBOG,DPSFRIW027SBOG,DPSACBW027NBOG` |
| `DPSSCBW027SBOG` | Deposits, Small Domestically Chartered Commercial Banks | Weekly, SA | 1973-01-03 | 2026-10-07 | same |
| `DPSFRIW027SBOG` | Deposits, foreign-related institutions (title **unverified**; ID confirmed in CSV) | Weekly, SA | 1973-01-03 | 2026-10-07 | CSV above |

All are from the Board of Governors H.8 release, "Assets and Liabilities of Commercial Banks in the United States". (Source: https://fred.stlouisfed.org/series/DPSACBW027SBOG.)

There is no CASA split. Use these as exogenous or macro covariates (e.g. quarterly growth of large-bank vs small-bank deposits), or as an aggregate sanity check against the FDIC panel.

`fredgraph.csv` downloads need no key. The FRED API (`api.stlouisfed.org`) requires a free key; that is general knowledge, **not re-verified here**.

---

## 6. Forecasting tooling (late 2026)

### Nixtla ecosystem: versions from the PyPI JSON API on 2026-10-20

| Package | Version (upload date) | Python | License | Verified capabilities | Source |
|---|---|---|---|---|---|
| `statsforecast` | **2.1.1** (2026-07-16) | >=3.10 | Apache-2.0 | Model classes include `AutoARIMA`, `AutoETS`, `AutoCES`, `AutoTheta`, `AutoMFLES`, `AutoTBATS`, `HoltWinters`, `SeasonalNaive`, `MSTL`. Self-reported speed: "20x faster than pmdarima", "500x faster than Prophet", "Fit 10 benchmark models on 1,000,000 series in under 5 min". Spark/Dask/Ray compatible. | https://pypi.org/pypi/statsforecast/json ; https://raw.githubusercontent.com/Nixtla/statsforecast/main/python/statsforecast/models.py ; https://github.com/Nixtla/statsforecast |
| `mlforecast` | **1.1.0** (2026-07-10) | >=3.10 | Apache-2.0 | Target transforms: `Differences`, `AutoDifferences`, `AutoSeasonalDifferences`, `LocalStandardScaler`, `LocalMinMaxScaler`, `LocalRobustScaler`, `LocalBoxCox`, `GlobalSklearnTransformer`.<br>Lag transforms: `RollingMean/Std/Quantile`, `SeasonalRollingMean`, `ExpandingMean`, `ExponentiallyWeightedMean`, `Offset`, `Combine`.<br>`fit(..., prediction_intervals=PredictionIntervals(...))` gives conformal intervals. `cross_validation`, `update`.<br>**`predict(h, new_df=...)`** forecasts series not seen in training, which is the transfer path to BOI. | https://pypi.org/pypi/mlforecast/json ; https://raw.githubusercontent.com/Nixtla/mlforecast/main/mlforecast/target_transforms.py ; .../lag_transforms.py ; .../forecast.py |
| `neuralforecast` | **3.3.0** (2026-10-07) | >=3.10 | Apache-2.0 | Models include `NHITS`, `NBEATSx`, `TFT`, `TiDE`, `PatchTST`, `DeepAR`, `TSMixerx`, `TimeXer`, `xLSTM` and others | https://pypi.org/pypi/neuralforecast/json ; https://raw.githubusercontent.com/Nixtla/neuralforecast/main/neuralforecast/models/__init__.py |
| `hierarchicalforecast` | **1.5.3** (2026-09-24) | >=3.10,<4 | Apache-2.0 | `BottomUp`, `TopDown`, `MiddleOut`, `MinTrace` (+ `MinTraceSparse`), `OptimalCombination`, `ERM` | https://pypi.org/pypi/hierarchicalforecast/json ; https://raw.githubusercontent.com/Nixtla/hierarchicalforecast/main/hierarchicalforecast/methods.py |
| `utilsforecast` | **0.2.17** (2026-10-05) | >=3.10 | Apache-2.0 | `losses`: `mase`, `rmsse`, `msse`, `mae`, `smape`, `quantile_loss`, `scaled_quantile_loss`, `mqloss`, `scaled_mqloss`, **`scaled_crps`**, `coverage`, `winkler_score`, `calibration` | https://pypi.org/pypi/utilsforecast/json ; https://raw.githubusercontent.com/Nixtla/utilsforecast/main/utilsforecast/losses.py |

Speed on about 7.5k quarterly series of length 40 to 102 on an 8 to 10 core Mac: I estimate minutes for AutoETS/AutoARIMA (`n_jobs=-1`) and seconds for LightGBM. These are **unverified** estimates; Nixtla's published numbers use clusters.

### Time-series foundation models (CPU)

| Model | pip install | Size | Max context / horizon | Covariates | CPU | License | Source |
|---|---|---|---|---|---|---|---|
| **Chronos-2** (`amazon/chronos-2`) | `pip install "chronos-forecasting>=2.0"` (latest **2.3.2**, 2026-09-08) | 120M params (HF metadata 119.5M) | **8192 / 1024** | **Yes**: past-only and known-future (real and categorical), multivariate, cross-learning across items | "supporting both GPU and CPU inference". Over 300 series per second on one A10G; CPU throughput **unverified**. | Apache-2.0 | https://huggingface.co/amazon/chronos-2 ; https://pypi.org/pypi/chronos-forecasting/json |
| Chronos-Bolt (`amazon/chronos-bolt-{tiny,mini,small,base}`) | same package | small = 47.7M | 2048 / 64 | No native covariates (external regressor workaround) | "up to 250 times faster and 20 times more memory-efficient than the original Chronos". CPU-friendly. | Apache-2.0 | https://huggingface.co/amazon/chronos-bolt-small ; https://huggingface.co/amazon/chronos-2 (comparison table) |
| TimesFM 2.5 (`google/timesfm-2.5-200m-pytorch`) | `pip install timesfm[torch]` (package latest **3.0.2**, 2026-09-09) | 200M (+ optional 30M quantile head; HF metadata 231.3M) | up to 16k context; quantiles up to 1k horizon | XReg covariates ("added back" 2025-10-29) | Not explicitly documented (examples use CUDA) | **Apache-2.0** weights | https://raw.githubusercontent.com/google-research/timesfm/master/README.md ; https://huggingface.co/google/timesfm-2.5-200m-pytorch ; https://pypi.org/pypi/timesfm/json |
| TimesFM 3.0 (`google/timesfm-3.0-pytorch`, Aug 2026) | `pip install timesfm[torch]` or **`timesfm[mlx]` for Apple silicon** | about 330M per the MLX benchmark (**unverified**); 20 layers, d = 1280 | global context 15,360 | Native past-only and past-and-future covariates, multivariate | MLX backend on Mac | **Non-commercial license** ("timesfm-non-commercial-license-v1.0"). Commercial use only via Google Cloud/BigQuery ML. | https://huggingface.co/google/timesfm-3.0-pytorch ; GitHub README above |
| Moirai 2.0 (`Salesforce/moirai-2.0-R-small`) | `pip install uni2ts` (latest **2.0.0**, 2025-11-04); the model card says build from source | 11.4M | Not documented (example uses `context_length=1680`); **unverified** | Moirai 1.x supports `feat_dynamic_real` / `past_feat_dynamic_real`. The Moirai2 example sets both dims to 0, so 2.0 covariate support is **unverified**. | Tiny, so CPU is fine (inferred) | Weights **CC-BY-NC-4.0**, "for research purposes only"; uni2ts code Apache-2.0 | https://huggingface.co/Salesforce/moirai-2.0-R-small ; https://raw.githubusercontent.com/SalesforceAIResearch/uni2ts/main/README.md ; https://pypi.org/pypi/uni2ts/json |

### Recommended stack for the 2-day sprint

- **Day 1, data.**
  - Pull FDIC `/financials` for `REPDTE` 2001Q1 to 2026Q2. Fields: `CERT, REPDTE, CALLFORM, ASSET, DEP, DEPDOM, TRN, NTRSMMDA, NTRSOTH, NTRTIME, DEPUNINS, BRO, ACTEVT`.
  - Pull `/history` merger events.
  - Build `y = log(TRN + NTRSMMDA + NTRSOTH)`.
  - Drop or split series at merger, failure and form-change breakpoints.
  - Keep banks with at least 40 clean quarters.
- **Day 1, baselines.**
  - `statsforecast` with `SeasonalNaive(4)`, `AutoETS`, `AutoARIMA`, `AutoTheta`.
  - `cross_validation(h=4, n_windows=4..8, step_size=1)`, scored with `utilsforecast` `mase(seasonality=4)` and `scaled_crps`.
- **Day 2, global and foundation models.**
  - `mlforecast` + LightGBM: lags 1 to 8, `RollingMean(4)`, `SeasonalRollingMean`, quarter-of-year feature, `Differences([1])` or `LocalStandardScaler`, `PredictionIntervals`.
  - Then Chronos-2 zero-shot on the same windows. Chronos-Bolt-small is the fast fallback.
- **Day 2, transfer.**
  - Run `MLForecast.predict(h, new_df=boi_df)` and Chronos-2 on the 40-point BOI series.
  - Compare against the existing ARIMA/SARIMA/HW/Prophet on the same holdout.
  - Report MASE and CRPS with intervals.
  - Optional: `hierarchicalforecast` MinTrace on bank, state and national levels.

---

## Structural breaks and gotchas

### Regulation D: savings reclassified as transaction accounts (largest trap)

- The Fed deleted "the six-per-month limit on convenient transfers from the 'savings deposit' definition" (interim final rule announced 2020-04-24). Banks "may immediately suspend enforcement". (Source: https://www.federalreserve.gov/newsevents/pressreleases/bcreg20200424a.htm)
- The Call Report glossary says that where the limit is suspended, the account is reported "as a savings account or a transaction account based on an assessment of the characteristics of the account":
  - If the bank does not keep the 7-day notice right, the account is a demand deposit and transaction account.
  - If it keeps that right, the account is a NOW (transaction) account or a savings deposit.
  - (Source: FFIEC 031/041 instruction book update March 2021, Glossary "deposits", https://www.fdic.gov/resources/bankers/call-reports/crinst-031-041/2021/2021-03-inserts.pdf)
- Observed in the data:
  - **Wells Fargo (CERT 3511), 2020Q3 to 2020Q4**: `NTRSOTH` fell from 766.6 to 36.1 $bn while `TRN` rose from 125.7 to 954.0 $bn.
  - **Bank of America (CERT 3510), 2025Q3 to 2025Q4**: `NTRSOTH` fell from 986.8 to 166.8 $bn while `TRN` rose from 421.1 to 1,261.8 $bn.
  - Industry `TRN` jumped by $1.16tn in 2020Q4 and $1.07tn in 2025Q4, while CASA moved smoothly.
  - (Source: live queries `.../financials?filters=CERT:3511 AND REPDTE:[20200630 TO 20210630]&...` and `.../financials?filters=REPDTE:20251231&fields=...&sort_by=NTRSOTH&sort_order=DESC&limit=5`. The cause of the 2025 move is **unverified**.)
- **Fix:** forecast the CASA sum (`DEPDOM - NTRTIME`). Never forecast `TRN` or savings alone. Never use their ratio as a target or feature without break dummies.

### Mergers, failures and survivorship

- Acquirer series step up at each merger, e.g. JPM absorbing Chase Bank USA in 2019 and First Republic in 2023. Failed banks just stop: SVB and Signature last reported 2022Q4.
- Detect breaks with `/history` (codes 211, 213, 216, 217, 221, 222, 223, 224, 810, 811 and `EFFDATE`), the `ACTEVT` flag on financial rows, and a jump rule (e.g. |Δlog CASA| > 0.25 in one quarter).
- **Evaluate on all banks alive at the forecast origin, not only survivors**, otherwise backtests look too good. (Sources: live history and financial queries in section 1.)

### Form and coverage changes

- TFR thrifts moved to Call Reports in 2012Q1 (638 TFR filers at 2011Q4). (Source: QBP notes.)
- The FFIEC 051 small-bank form started 2017Q1 (3,505 filers) and was expanded to banks under $5bn in 2019. (Sources: FIL-1-2017, FIL-35-2019.)
- The Call Report does not require these forms to change the RC-E deposit items. Whether any CASA item is missing on the 051 is **unverified**, though the spot checks returned all deposit fields.

### Uninsured deposits are partly modelled

- Only banks with $1bn or more in assets report estimated uninsured deposits. The FDIC computes the figure for everyone else.
- Insurance-limit changes affect insured and uninsured series: $100k to $250k on 2009-09-30, and unlimited noninterest-bearing transaction coverage from 2010-12-31 to 2012-12-31.
- (Source: https://www.fdic.gov/quarterly-banking-profile/qbp-notes-users-fourth-quarter-2025.pdf)

### No as-reported vintages

- Data are revised: CDR is "regenerated monthly to capture amendments", and the FDIC index is a snapshot (e.g. `risview_20260819185831`). Backtests therefore use revised data, a mild look-ahead.
- (Source: https://cdr.ffiec.gov/public/HelpFiles/WelcomeAdditionalInfo.htm ; live `meta.index`)

### Units and keys

- FDIC values are in $ thousands. `REPDTE` is a `yyyymmdd` string. Field names in `fields`, `filters` and `sort_by` must be uppercase.
- Some fields have no description in the dictionary (e.g. `TRN`, `COREDEP`), so confirm definitions by identity checks as done above.

### Survey cadence

- Call Report data are quarter-end snapshots, so intra-quarter runs are smoothed. SVB failed on 2023-03-10 and therefore never shows a 2023Q1 row.

### India: demand/time is not current/savings/term

- RBI's "demand deposits" include current deposits plus the **demand-liability portion of savings deposits**. The time-liability portion of savings goes into "time deposits".
- The split is recomputed from minimum balances as of March 31 and September 30.
- So WSS demand/time cannot be mapped to CASA. Only BSR data carry current/savings/term. (Source: RBI CRR/SLR directions as extracted via search, https://website.rbi.org.in/documents/87730/39016390/88500.pdf. Page not directly reachable.)

### India: series breaks

- BSR-7 was discontinued after December 2022 and replaced by quarterly BSR-2 from March 2023.
- WSS and Bulletin tables note that data "include the impact of the merger of a non-bank with a bank" from 2023-07-01. The HDFC Ltd / HDFC Bank identity is my inference and **unverified**.
- The reporting fortnight was redefined to the 15th and last calendar day of each month from 2025-12-15 (Banking Laws (Amendment) Act, 2025).
- (Sources: https://website.rbi.org.in/web/rbi/-/press-releases/quarterly-bsr-1-outstanding-credit-of-scheduled-commercial-banks-june-2023-56294 ; https://www.rbi.org.in/Scripts/BS_ViewWssExtractdetails.aspx?id=63718, via search extract)

### Transfer-learning domain shift

- Indian nominal deposit growth and seasonality (fiscal year-end in March) differ from the US. Train the global model on scale-free targets (log levels with differencing, or quarter-on-quarter growth) and per-series normalization.
- Check whether BOI's "2015Q1" is a calendar or Indian-fiscal quarter before aligning with US calendar quarters. (Judgment.)

### Licensing

- TimesFM 3.0 weights are non-commercial and Moirai 2.0 is CC-BY-NC-4.0. For anything shown to a bank as a usable tool, stick to Chronos-2, Chronos-Bolt or TimesFM 2.5 (all Apache-2.0). (Sources: HF model cards above.)

---

## Unverified

1. **FDIC API rate-limit window.** The header says `x-ratelimit-limit: 20`, but the period (per second, minute or hour) and per-IP behaviour are undocumented.
2. **The exact formula behind FDIC `COREDEP`.** It matches `DEPDOM - BRO - NTRTMLGJ` exactly for one bank and only approximately for others.
3. **CALLFORM code 100 = TFR.** Inferred from the 2012Q1 disappearance plus the QBP note.
4. **Cause of Bank of America's 2025Q4 savings-to-transaction shift.** The shift itself is verified in the data.
5. **FFIEC CDR bulk history.** The earliest Call Report bulk period (believed 2001Q1) and the MDRM codes RCON2215 / RCON6810 / RCON0352 are unverified.
6. **RBI pages: content seen only as search-engine extracts.** All RBI hosts were blocked or timed out from here. Specifically unverified:
   - fortnightly WSS history length in DBIE;
   - number of districts and continuity of type-wise district series across the BSR-7 to BSR-2 switch;
   - whether STRBI bank-wise tables split deposits into demand / savings / term;
   - absence of any DBIE API (none found, but not confirmed by RBI);
   - HDFC as the "non-bank merged with a bank" in July 2023.
7. **Title of FRED `DPSFRIW027SBOG`** (foreign-related institutions). The ID and range are verified.
8. **TimesFM 3.0 parameter count** (about 330M, from an MLX benchmark mention) and TimesFM CPU performance.
9. **Moirai 2.0 maximum context and covariate support.**
10. **Chronos-2 CPU throughput, and statsforecast/LightGBM runtimes on a 16 GB Mac** (my estimates only).
11. **Exact `lightgbm` version.** Not checked; any recent 4.x should work with mlforecast 1.1.0 (assumption).
