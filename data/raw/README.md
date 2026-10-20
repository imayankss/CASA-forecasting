# Raw data provenance

Both files in this folder are **synthetic**. Bank of India's internal data was confidential, so neither file contains real Bank of India figures.

| File | Rows | How it was produced | Status |
|---|---:|---|---|
| `boi_casa_deposits.csv` | 40 quarters (2015 Q1 to 2024 Q4) | Generated with an LLM (ChatGPT) to resemble a bank's quarterly CASA deposit series | Synthetic. Not Bank of India's actual deposits |
| `macro_indicators.csv` | 40 quarters (2015 Q1 to 2024 Q4) | Produced with an LLM (ChatGPT); values approximate public Indian macro figures (repo rate, CPI, GDP growth, Nifty Bank, credit-deposit ratio, industry CASA ratio) | Not checked against official sources. Do not cite these values |

Do not quote any value from these files as a real Bank of India, RBI or MOSPI figure.

The pipeline, tests and dashboard use these files as fixed demo inputs. Results on real public data will be added separately and labelled with their sources.
