# Public datasets for a customer-level CASA propensity model

Research date: 2026-10-20. Every fact below is followed by the URL it was checked against. Kaggle pages need JavaScript, so they were read in a real browser session (logged out); numbers that Kaggle only exposes after accepting the rules were taken from public notebooks by competition participants and are labelled as such. Anything not confirmed is listed under [Unverified](#unverified).

---

## Summary and recommendation

**Use Kaggle "Santander Product Recommendation" (2016).** It is the only large public dataset found that has month-by-month, customer-level ownership flags for real current/savings-type bank accounts.

| | Santander Product Recommendation |
|---|---|
| Customers | 956,645 unique `ncodpers` in train; 929,615 in test, all of whom also appear in train |
| Rows | 13,647,309 customer-month rows × 48 columns in `train_ver2.csv` |
| Time span | 17 monthly snapshots, 2015-01-28 to 2016-05-28; the test set is 2016-06-28 |
| CASA-like flags | `ind_cco_fin_ult1` (current), `ind_ahor_fin_ult1` (saving), `ind_cno_fin_ult1` (payroll account), `ind_ecue_fin_ult1` (e-account), `ind_ctop/ctpp/ctma_fin_ult1` (particular / particular Plus / más particular accounts), `ind_ctju_fin_ult1` (junior) |
| Download | 240.06 MB zipped (train zip 224.7 MB) |
| Benchmark | MAP@7 on newly added products. Private LB: 1st 0.03140; gold cutoff 0.03097; silver 0.03050; bronze 0.03027 |
| Late submissions | Open: the competition page shows a "Late Submission" button |

Sources for the table: data page https://www.kaggle.com/competitions/santander-product-recommendation/data ; leaderboard https://www.kaggle.com/competitions/santander-product-recommendation/leaderboard ; row and customer counts from participants' notebooks https://github.com/SudalaiRajkumar/Kaggle/blob/master/SantanderReco/santander_exploartion.ipynb and https://github.com/kevinmtian/kaggle_santander/blob/master/notebook/demo_part1_cleaning.ipynb

**Ranked recommendation**

| Rank | Dataset | (a) CASA open/hold propensity next month, lift@top-decile, 100k+ customers | (b) Kaggle-comparable benchmark |
|---|---|---|---|
| 1 | Santander Product Recommendation | **Yes.** Monthly CASA ownership for about 0.93–0.96M customers, so "adds a CASA-type account next month" can be built for any month from 2015-02 to 2016-05. | **Yes.** Late submission gives an official private-LB MAP@7 score. This covers all 24 products, not CASA alone. |
| 2 | Berka / PKDD'99 Czech bank | Partly. It has a running balance per transaction, which suits balance-growth targets, but only about 4,500 accounts. That is too small to answer the "500 customers" criticism. | No leaderboard |
| 3 | UCI Bank Marketing | No. The target is term-deposit subscription from phone campaigns, about 41–45k rows. It is not CASA. | No leaderboard |
| — | Santander Customer Transaction Prediction, Santander Customer Satisfaction, AmEx Default Prediction, BAF suite | Not relevant to CASA (see section 3) | — |

**How to report the result honestly:** quote two separate numbers.
1. The Kaggle private-LB MAP@7 from a late submission. This one is comparable with the leaderboard.
2. Your own lift@top-decile for the target "adds any CASA-type account at month t+1, among customers who did not hold it at t". Compute it on a held-out month that has labels, for example features up to 2016-04 and labels from 2016-05. Report it next to the base rate and next to a simple rule-based baseline. This number is not Kaggle-comparable, and the write-up should say so.

---

## 1. Santander Product Recommendation (Kaggle, 2016): slug `santander-product-recommendation`

### 1.1 Facts

| Fact | Value | Source |
|---|---|---|
| Host / type | Banco Santander, Featured Prediction Competition, $60,000 prizes | https://www.kaggle.com/competitions/santander-product-recommendation/overview |
| Dates | Start 10/26/2016 7:00 PM UTC; entry/merger deadline 12/14/2016; end 12/21/2016 11:59 PM UTC | https://www.kaggle.com/competitions/santander-product-recommendation/rules |
| Participation | 2,997 entrants, 2,051 participants, 1,779 teams, 28,733 submissions | https://www.kaggle.com/competitions/santander-product-recommendation/overview |
| Task, in Kaggle's words | "predict what **additional** products a customer will get in the last month, 2016-06-28, in addition to what they already have at 2016-05-28" | https://www.kaggle.com/competitions/santander-product-recommendation/data |
| Date range | "The data starts at 2015-01-28 and has monthly records"; described as "1.5 years". Train covers 17 month-ends, 2015-01-28 to 2016-05-28; test is 2016-06-28 | data page (above); 17 timestamps also stated at https://github.com/kweonwooj/kaggle_santander_product_recommendation |
| Split | "The test and train sets are split by time, and public and private leaderboard sets are split randomly." Private LB is about 70% of test, public about 30%. | data page; https://www.kaggle.com/competitions/santander-product-recommendation/leaderboard |
| Data provenance caveat | "This sample does not include any real Santander Spain customers, and thus it is not representative of Spain's customer base." | data page |
| Files (zipped, bytes) | `train_ver2.csv.zip` 224,672,878; `test_ver2.csv.zip` 12,995,524; `sample_submission.csv.zip` 2,393,683; total 240,062,085 ("240.06 MB, 3 files") | data page; byte sizes read from the data page's file metadata (Kaggle web API `GetDatabundleExternal`, competitionId 5558) |
| Uncompressed train size | about 2.3 GB (secondary source) | https://nycdatascience.com/blog/student-works/18037/ |
| Train rows | 13,647,309 rows, 48 columns | SRK notebook output: https://github.com/SudalaiRajkumar/Kaggle/blob/master/SantanderReco/santander_exploartion.ipynb ; `(13647309, 48)` in https://github.com/kevinmtian/kaggle_santander/blob/master/notebook/demo_part1_cleaning.ipynb |
| Train unique customers | 956,645 | SRK notebook (above); also stated in the kevinmtian notebook |
| Test rows / customers | 929,615 rows = 929,615 unique customers, all present in train ("Number of common customers: 929615") | SRK notebook; Kaggle competition metadata `totalSolutionRows: 929615` |
| `fecha_alta` (customer since) range | 1995-01 to 2016-05 | SRK notebook |
| Metric | MAP@7 (see 1.4) | https://www.kaggle.com/competitions/santander-product-recommendation/overview/evaluation |
| Data licence | "Subject to Competition Rules" | data page |

### 1.2 The 24 product columns (columns 25–48), with Kaggle's English labels

The English meanings are exactly as printed in the data-page table at https://www.kaggle.com/competitions/santander-product-recommendation/data. The CASA flag is this report's mapping, not Kaggle's.

| Column | Kaggle label | CASA-like? |
|---|---|---|
| `ind_ahor_fin_ult1` | Saving Account | **Yes, SA.** Additions are extremely rare (see Gotchas) |
| `ind_aval_fin_ult1` | Guarantees | No |
| `ind_cco_fin_ult1` | Current Accounts | **Yes, CA.** Strongly seasonal additions |
| `ind_cder_fin_ult1` | Derivada Account | No (derivatives) |
| `ind_cno_fin_ult1` | Payroll Account | **Yes.** A salary-credit current account |
| `ind_ctju_fin_ult1` | Junior Account | **Yes** (minors' account) |
| `ind_ctma_fin_ult1` | Más particular Account | **Yes** (current-account variant) |
| `ind_ctop_fin_ult1` | particular Account | **Yes** (current-account variant) |
| `ind_ctpp_fin_ult1` | particular Plus Account | **Yes** (current-account variant) |
| `ind_deco_fin_ult1` | Short-term deposits | No (term deposit) |
| `ind_deme_fin_ult1` | Medium-term deposits | No (term deposit) |
| `ind_dela_fin_ult1` | Long-term deposits | No (term deposit) |
| `ind_ecue_fin_ult1` | e-account | **Yes** (online account) |
| `ind_fond_fin_ult1` | Funds | No |
| `ind_hip_fin_ult1` | Mortgage | No |
| `ind_plan_fin_ult1` | Pensions | No (pension plan) |
| `ind_pres_fin_ult1` | Loans | No |
| `ind_reca_fin_ult1` | Taxes | No |
| `ind_tjcr_fin_ult1` | Credit Card | No |
| `ind_valo_fin_ult1` | Securities | No |
| `ind_viv_fin_ult1` | Home Account | No |
| `ind_nomina_ult1` | Payroll | No; it is a payroll *service* flag, but it moves together with `cno` |
| `ind_nom_pens_ult1` | Pensions | No (pension direct credit) |
| `ind_recibo_ult1` | Direct Debit | No (a CASA *engagement* signal, not an account) |

Two CASA target definitions are defensible:
- **Narrow:** `cco ∪ ahor`.
- **Broad:** `cco ∪ ahor ∪ cno ∪ ecue ∪ ctop ∪ ctpp ∪ ctma ∪ ctju`.

The ownership of `ctop/ctpp/ctma` by Spanish product name is from the data page. Treating them as current-account variants is this report's interpretation; Kaggle does not say so.

### 1.3 Customer attribute columns (columns 1–24)

Definitions are from the data page, https://www.kaggle.com/competitions/santander-product-recommendation/data

| Column | Meaning (Kaggle) |
|---|---|
| `fecha_dato` | Snapshot month ("The table is partitioned for this column") |
| `ncodpers` | Customer code |
| `ind_empleado` | Employee index: A active, B ex employed, F filial, N not employee, P pasive |
| `pais_residencia` | Country of residence |
| `sexo` | Sex |
| `age` | Age |
| `fecha_alta` | Date the customer first became holder of a contract with the bank |
| `ind_nuevo` | 1 if the customer registered in the last 6 months |
| `antiguedad` | Customer seniority (tenure) in months |
| `indrel` | 1 = primary; 99 = primary during the month but not at month end |
| `ult_fec_cli_1t` | Last date as primary customer |
| `indrel_1mes` | Customer type at start of month: 1 primary, 2 co-owner, P potential, 3 former primary, 4 former co-owner |
| `tiprel_1mes` | Relation type at start of month: A active, I inactive, P former, R potential |
| `indresi` | Residence country = bank country (S/N) |
| `indext` | Birth country ≠ bank country (S/N) |
| `conyuemp` | Spouse of an employee |
| `canal_entrada` | Channel used to join |
| `indfall` | Deceased (N/S) |
| `tipodom` | Address type |
| `cod_prov` / `nomprov` | Province code / name |
| `ind_actividad_cliente` | Activity index (1 active, 0 inactive) |
| `renta` | Gross household income |
| `segmento` | Segment: "01 - VIP, 02 - Individuals 03 - college graduated". The raw values are actually `01 - TOP`, `02 - PARTICULARES`, `03 - UNIVERSITARIO` (see https://www.kaggle.com/code/apryor6/detailed-cleaning-visualization-python) |

### 1.4 Evaluation metric, exactly as defined

From https://www.kaggle.com/competitions/santander-product-recommendation/overview/evaluation :

> MAP@7 = (1/|U|) · Σ_{u=1..|U|} [ 1/min(m,7) · Σ_{k=1..min(n,7)} P(k) ]
> "where |U| is the number of rows (users in two time points), P(k) is the precision at cutoff k, n is the number of predicted products, and m is the number of added products for the given user at that time point. If m = 0, the precision is defined to be 0."

The submission format is `ncodpers,added_products`, with up to 7 product names separated by spaces, one row per test customer (929,615 rows).

Practical consequences:
- Customers who add nothing (m = 0) score 0 whatever you predict. The leaderboard therefore only rewards ranking *which* products are added, given that something is added. It does not reward *who* adds. This is why the top scores are only about 0.031.
- A Kaggle MAP@7 score does **not** measure the "who will open a CASA" skill that a lift@decile metric measures. BreakfastPirate (5th place) says so directly: "We are not trying to determine who will add a new product, we are only trying to predict which products they would add if they did." (https://www.kaggle.com/competitions/santander-product-recommendation/discussion/25579)

### 1.5 Leaderboard benchmarks

Scores read from https://www.kaggle.com/competitions/santander-product-recommendation/leaderboard (Private tab; 1,780 rows; the medal flags come from the same page's data).

| Private rank | Team | Private MAP@7 | Public MAP@7 |
|---|---|---|---|
| 1 | idle_speculation | **0.03140** | 0.03100 |
| 2 | Tom Van de Wiele | 0.03131 | 0.03100 |
| 3 | Jack (Japan) | 0.03131 | 0.03099 |
| 4 | yoniko | 0.03130 | 0.03092 |
| 5 | Jared Turkewitz and BreakfastPirate | 0.03126 | 0.03087 |
| 13 (last gold) | — | 0.03097 | — |
| 88 (last silver) | — | 0.03050 | — |
| 177 (last bronze) | — | 0.03027 | — |
| 200 | — | 0.03024 | — |
| 300 | — | 0.02947 | — |
| 500 | — | 0.02927 | — |
| 890 (about the median) | — | 0.02723 | — |
| 1000 | — | 0.02501 | — |

Reference points for a 2-day sprint:
- The 3rd-place author's single-month xgboost (train on the 2016-05-28 snapshot, 142 features) scored **0.0312074 private / 0.0308403 public** on an 8 GB RAM laptop. Source: https://www.kaggle.com/competitions/santander-product-recommendation/writeups/jack-japan-3rd-place-solution-with-code
- raddar (9th place) wrote: "with correct feature space, 5-minute model is enough to reach 0.030" using lag-5 product features. Source: https://www.kaggle.com/competitions/santander-product-recommendation/discussion/25579

A realistic, defensible target is **about 0.0300–0.0305 private**, which falls between bronze (0.03027) and the top 200.

### 1.6 Late submissions

- The competition header shows a **"Late Submission"** button (observed 2026-10-20 on https://www.kaggle.com/competitions/santander-product-recommendation and its sub-pages), so new submissions are still accepted.
- Late submissions are scored but do not change the official leaderboard. Quote a late score as "late submission, private LB MAP@7 = X (would have ranked ~N)", never as a rank you achieved. See Unverified for Kaggle's general late-submission policy.

### 1.7 Rules on data use and publishing

All quotes are from https://www.kaggle.com/competitions/santander-product-recommendation/rules :

- **Raw data cannot be redistributed.** "Participants must use the Data solely for the purpose and duration of the Competition … agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Data to any party not participating in the Competition."
- **Code can be published.** "Participants are permitted to publicly share source or executable code developed in connection with or based upon the Data … By so sharing, the sharing Participant is thereby deemed to have licensed the shared code under the MIT License."
- **External data is not allowed** for competition submissions: "Participants must not use data other than the Data to develop and test their models and Submissions."
- **Data page licence:** "Subject to Competition Rules" (https://www.kaggle.com/competitions/santander-product-recommendation/data).
- **Practical reading:**
  - Publish code, aggregate metrics, plots and the README on GitHub, under MIT or compatible.
  - Do **not** commit any CSV, parquet or feature-matrix files, or row-level predictions keyed by `ncodpers`. Add `data/` to `.gitignore`.
  - Tell readers to download the data themselves after accepting the rules.
  - The "solely for the purpose and duration of the Competition" wording is a grey area for portfolio use after the competition. Framing the work as a late-submission entry is the cleanest position (see Unverified).

### 1.8 Download command

```bash
pip install kaggle
kaggle auth login                     # or: export KAGGLE_API_TOKEN=...  (token from https://www.kaggle.com/settings/api)
# FIRST accept the competition rules on the website while logged in (the data page says
# "To see this data you need to agree to the competition rules")
kaggle competitions download santander-product-recommendation -p data --unzip
#   -> data/train_ver2.csv.zip, data/test_ver2.csv.zip, data/sample_submission.csv.zip (inner zips; unzip again)
kaggle competitions submit santander-product-recommendation -f submission.csv -m "late sub: lgbm lag5" --wait 600
```

Sources:
- Command syntax (the competition name is positional; options `-p`, `--unzip`, `-f`, `-m`, `--wait`): https://github.com/Kaggle/kaggle-api/blob/main/docs/competitions.md
- Authentication options (`kaggle auth login`, `KAGGLE_API_TOKEN`, `~/.kaggle/access_token`, legacy `~/.kaggle/kaggle.json`): https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md
- Rules-acceptance requirement: data page.
- Older tutorials use `-c <slug>`. The current docs show the positional form.

### 1.9 What the top solutions say

| Insight | Who / source |
|---|---|
| The key features are lags of product ownership, time since a product was present or last purchased, and product averages. Also useful: lags and time-since-change of `segmento`, `ind_actividad_cliente`, `cod_prov`, `canal_entrada`, `indrel_1mes`, `tiprel_1mes`. | 1st place: https://www.kaggle.com/competitions/santander-product-recommendation/writeups/idle-speculation-1-solution |
| Seasonality: **`reca` (taxes) is scored "as Jun-15" and `cco` (current account) "as Dec-15"**. Scores are zeroed for products the customer held the previous month. | 1st place (above) |
| `cco` "showed particularly different behavior in June and December 15 … end of (tax) year specific patterns". Most model weight went to June 2015. | 2nd place: https://www.kaggle.com/competitions/santander-product-recommendation/writeups/tom-van-de-wiele-2nd-place-solution |
| **`renta` is non-unique for about 30% of users**, mostly in pairs or small groups, probably households. Household features added little. | 2nd place (above); top repeated `renta` values listed in https://www.kaggle.com/competitions/santander-product-recommendation/discussion/26786 |
| `cco` was predicted only from 2015-12-28 data and `reca` only from 2015-06-28 data. **`ahor` (savings), `aval`, `deco`, `deme` were fixed to probability 1e-10 because "these product are purchased very rarely".** | 3rd place: https://www.kaggle.com/competitions/santander-product-recommendation/writeups/jack-japan-3rd-place-solution-with-code |
| About half of the 5th-place ensemble was models trained on June 2015 only. 8 GB RAM was enough by training only on rows with a new product. | 5th place: https://www.kaggle.com/competitions/santander-product-recommendation/writeups/jared-turkewitz-and-breakfastpirate-5-solution |
| "When Less is More": train on only the **about 37,000 accounts that added a product in June 2015** (about 46,000 rows after one row per added product). This reached about 0.030. | https://www.kaggle.com/competitions/santander-product-recommendation/discussion/25579 |
| `nom_pens` is always added with `nomina` if not already held. Post-rule: rank `nom_pens` above `nomina`. | https://www.kaggle.com/competitions/santander-product-recommendation/discussion/26786 |
| Winners **probed the public leaderboard** per product to recalibrate probabilities ("baselined to level of public leaderboard … by submitting only that product"; "extrapolated public leaderboard count"). | 1st and 2nd place write-ups (above) |
| Hardware: 1st place used 64 GB RAM ("RAM was the biggest bottleneck"); 2nd used 48 GB and about 3 days of compute; 3rd and 5th used 8 GB by subsetting. | 1st/2nd/3rd/5th write-ups (above) |

---

## 2. Berka / PKDD'99 Czech financial dataset

| Fact | Value | Source |
|---|---|---|
| Official host today | CTU Prague Relational Dataset Repository, dataset "Financial" (alt. name "loan application"). Served as a public read-only MariaDB (guest account published on the page) with CSV/SQL export. The original PKDD'99 page host `lisp.vse.cz` no longer resolves (DNS failure on 2026-10-20). | https://relational.fel.cvut.cz/dataset/Financial |
| Size (CTU version) | 8 tables, 1,090,086 rows in total, 55 columns, 78.8 MB | https://relational.fel.cvut.cz/dataset/Financial |
| Standard task | Loan outcome: "606 successful and 76 unsuccessful loans"; target `loan.status` | https://relational.fel.cvut.cz/dataset/Financial |
| Table counts | account 4,500; client 5,369; disposition 5,369; transaction 1,056,320; loan 682; credit card 892. This course page restates the original challenge description. | https://webpages.charlotte.edu/mirsad/itcs6265/group1/domain.html (secondary) |
| Other tables | permanent order ≈6,471 rows; district 77 rows (after dropping one junk row) | https://data-dict.tidyverse.org/examples/loan-application.yaml |
| Date range | `trans.date` 1993-01-01 to 1998-12-31; account creation 1993-01-01 to 1997-12-29; `loan.date` 1993-07-05 to 1998-12-08 | https://data-dict.tidyverse.org/examples/loan-application.yaml |
| Running balance | **Yes.** `trans.balance` = "account balance after the transaction", range −41,126 to 209,637. Other `trans` fields: `type` (PRIJEM credit / VYDAJ debit / VYBER cash withdrawal), `operation`, `amount`, `k_symbol`, `bank`, `account` | https://data-dict.tidyverse.org/examples/loan-application.yaml |
| Licence | **Not stated** on the CTU page, and none found | https://relational.fel.cvut.cz/dataset/Financial ; https://github.com/hadley/loan-application (no licence file) |
| Data-quality warning | "The source database is publicly writable and has collected two junk rows" (a `district` row and an all-null `loan` row), which makes some id/date columns export as text | https://github.com/hadley/loan-application |
| Leakage warning (CTU) | If the temporal constraint is ignored, good and bad loans are "trivially separable" using the minimum transaction balance | https://relational.fel.cvut.cz/dataset/Financial |

**CASA suitability:**
- It *can* support "balance growth" and balance-forecast targets, because it has a per-transaction running balance (monthly end-of-month balance per account is easy to derive).
- It cannot support "open a CASA product" propensity: every account is already a current account, and the only product-opening events are loans (682) and cards (892).
- About 4,500 accounts repeats the small-sample criticism (about 9× the student's 500, not 1,000×).
- Use it, if at all, as a secondary "balance dynamics" notebook, not as the headline.

---

## 3. Other candidates (honest relevance check)

| Dataset | Verified facts | CASA relevance (one line) |
|---|---|---|
| UCI Bank Marketing | 45,211 rows (bank-full) / 41,188 (bank-additional-full), 16–20 inputs. Target `y` = "if the client will subscribe a term deposit". Portuguese bank phone campaigns, May 2008–Nov 2010. CC BY 4.0. Source: https://archive.ics.uci.edu/dataset/222/bank+marketing | Low. It is a term-deposit campaign response, not CASA, and at about 41–45k rows it is too small for the "big data" point. |
| Kaggle Santander Customer Transaction Prediction (2019) | "anonymized dataset containing numeric feature variables, the binary target column, and a string ID_code"; 3 files, 606.35 MB. Source: https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data | None. The features are anonymised and nothing maps to accounts or balances. |
| Kaggle Santander Customer Satisfaction (2016) | "anonymized dataset containing a large number of numeric variables"; TARGET = 1 for unsatisfied customers; 119.04 MB. Source: https://www.kaggle.com/competitions/santander-customer-satisfaction/data | None (anonymised satisfaction label). |
| Kaggle American Express Default Prediction (2022) | Predicts whether a customer fails to pay within 120 days of the latest statement (18-month window). Features are anonymised D/S/P/B/R; negatives subsampled at 5%; 4 CSV files, **50.31 GB**. Source: https://www.kaggle.com/competitions/amex-default-prediction/data | None. It is credit-card default, and 50 GB of CSV does not fit a 16 GB laptop sprint. |
| Bank Account Fraud (BAF) suite, NeurIPS 2022 | 6 synthetic datasets × 1M instances, 30 features plus `month` plus protected attributes. Generated with CTGAN plus DP noise from a real account-opening fraud dataset. Label `fraud_bool`. CC BY-NC-SA 4.0. Base.csv 213 MB. Sources: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 ; https://github.com/feedzai/bank-account-fraud | Very low. The event is an account-opening *application*, but the label is fraud, the data are synthetic, and there is no post-opening behaviour. |
| Others with ≥1M rows and real account/deposit ownership over time | — | **None found** besides Santander PR. Berka has 1.06M transactions but only 4.5k accounts. |

---

## Gotchas / leakage risks

1. **"Hold" vs "add" leakage (the biggest trap).**
   - Predicting whether a customer *holds* `cco` next month is almost free, because ownership persists month to month. A lift@decile on a "hold" target will look spectacular and will be meaningless.
   - Define the target as a **new addition**: `owned(t) = 0 and owned(t+1) = 1`. Restrict the scored population to customers who do not hold the product at t.
   - Kaggle defines "added" this way. Top solutions zero out products held last month (1st-place write-up).

2. **Time-based split only.**
   - Features must use months ≤ t; labels come from t+1.
   - Never shuffle customer-months randomly across time. The same `ncodpers` appears up to 17 times, so a random split leaks the customer's future into training.
   - The test set (2016-06) has no labels, so local validation must use a labelled month. The 3rd-place author validated on 2016-05-28.

3. **Seasonality.**
   - `cco` additions behave differently in June and December 2015 (year-end/tax), and `reca` spikes in June (1st, 2nd, 3rd and 5th-place write-ups).
   - A CASA model trained on one month and evaluated on another will mis-calibrate. Report the evaluation month, and compare against the same calendar month where possible.

4. **Savings (`ahor`) is nearly empty as a target.** The 3rd-place author fixed its probability to 1e-10 because additions are so rare. A "savings-account propensity" model on this data is not feasible. Use the broad CASA union, or `cco` alone.

5. **Most customers add nothing in a given month.**
   - Only about 37k accounts added any product in June 2015 (BreakfastPirate).
   - The CASA-add base rate will be well under 1–5% of the active base (exact rate unverified; compute it).
   - Lift@decile must be reported with the base rate and the absolute count of positives in the top decile.

6. **Missing data and broken history.**
   - `renta` is missing for 2,794,375 of 13,647,309 rows (about 20.5%) (kevinmtian 16th-place cleaning notebook).
   - New customers after late September 2015 have **no `renta` at all**, so missingness shifts over time (BreakfastPirate, discussion/26786).
   - Before July 2015, rows for customers with zero products seem to be missing. A customer can "reappear", creating spurious "adds".
   - Non-product attributes (age, `antiguedad`, …) are frozen before July 2015 (same post).
   - 27,734 rows lack almost every demographic field, and 16,063 rows have NaN in `ind_nomina_ult1` / `ind_nom_pens_ult1` (kevinmtian notebook).

7. **Messy types.**
   - `age`, `antiguedad` and `indrel_1mes` load as mixed types; pandas raises a DtypeWarning. `indrel_1mes` mixes `1`, `1.0`, `"1"`, `"P"`.
   - `segmento` raw values are `01 - TOP`, `02 - PARTICULARES`, `03 - UNIVERSITARIO`.
   - Source: https://www.kaggle.com/code/apryor6/detailed-cleaning-visualization-python
   - Pass explicit `dtype=`/`usecols=`, or use polars or DuckDB.

8. **Memory on a 16 GB Mac.**
   - Top-2 teams used 48–64 GB. The 3rd and 5th-place teams managed with 8 GB by subsetting rows and months (write-ups above).
   - Do not `pd.read_csv` the full 2.3 GB file with object dtypes.
   - Use `usecols` plus int8 product flags plus categorical strings, or DuckDB/polars with lazy scan. Build lag features with a self-join on (`ncodpers`, month).

9. **Leaderboard comparability.**
   - Top scores include public-LB probing and per-product recalibration (1st and 2nd-place write-ups).
   - MAP@7 covers all 24 products, and payroll (`nomina`/`nom_pens`), `recibo`, `reca`, `cco` and `tjcr` dominate it.
   - A CASA-only model cannot be submitted alone to get a fair MAP@7. Submit an all-product model for the Kaggle number and report CASA lift separately.

10. **Payroll coupling.**
    - `cno` (payroll account), `nomina` and `nom_pens` move together.
    - If the broad CASA target includes `cno`, much of the lift will come from detecting salary-switch events. Say so in the write-up, or report `cco`-only lift as well.

11. **Same-household duplicates.** Identical `renta` values (about 30% non-unique) and adjacent `ncodpers` suggest households (2nd place; discussion/26786). Use group-aware splitting if you split by customer.

12. **Licence.** Do not push data, features or `ncodpers`-level outputs to GitHub (rules, section 1.7).

13. **Berka, if used.**
    - Respect time: compute balance features only from transactions before the prediction date. Otherwise targets leak; the CTU page notes that loans become "trivially separable" by minimum balance when time is ignored.
    - Drop the junk rows the publicly writable source DB has accumulated (https://github.com/hadley/loan-application).
    - No licence is stated, so treat redistribution as not permitted.

---

## Unverified

Claims not confirmed against a primary source in this session:

- **Kaggle's general late-submission policy.** It is commonly said that late submissions get public and private scores but no rank or medal. Only the "Late Submission" button on the competition page was observed. https://www.kaggle.com/docs/competitions does not describe late submissions. Verify after submitting: the score should show on your Submissions tab.
- **Whether post-competition portfolio use is covered** by "solely for the purpose and duration of the Competition". This is a legal reading, not something Kaggle states. Kaggle still serves the data and accepts late submissions, but no explicit permission for post-competition use was found.
- **Uncompressed size of `train_ver2.csv` (about 2.3 GB).** From a secondary blog (nycdatascience.com), not Kaggle. Kaggle shows only the zip sizes.
- **Monthly base rate of CASA-type additions.** Not published anywhere found. The only anchor is about 37k accounts adding *any* product in June 2015 (BreakfastPirate). Compute it after download.
- **Theoretical maximum MAP@7** (share of test customers who add ≥1 product in 2016-06). Not found.
- **Berka per-table counts** (4,500 / 5,369 / 1,056,320 / 682 / 892). Taken from a university course page restating the original challenge guide; the original berka.htm host is down. The CTU repository gives only the total (1,090,086 rows), and the per-table counts above sum to about 1.08M, so the CTU copy may differ slightly.
- **Row counts for Santander CTP (often quoted as 200k/200k) and Santander Customer Satisfaction (about 76k train).** Kaggle shows only file sizes without login.
- **AmEx customer and row counts** (often quoted as about 459k customers / about 5.5M rows). Not shown without login; only the 50.31 GB total was verified.
- **The 2.3 GB figure and the claim that a full pandas load with object dtypes exceeds 16 GB.** This is an engineering expectation, not measured here. Measure with `usecols` and dtypes on day 1.
- **Interpreting `ctop` / `ctpp` / `ctma` as current-account variants.** Kaggle labels them only "particular / particular Plus / Más particular Account".
