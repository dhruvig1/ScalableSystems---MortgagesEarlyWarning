# Results log (verified numbers, updated as runs finish)

## Data scale
- Freddie Mac: 49,483,298 loans; 2,921,816,568 loan-month rows; 110 quarterly sources (1999Q1 to 2026Q2)
- CFPB: 464,583 mortgage complaints, 147,249 with narratives; FRED: 103 series
- Layers on GCS: bronze 40.65 GiB raw, silver 25.44 GiB Parquet, gold panel 40.84 GiB

## Gold panel (Spark on Dataproc, 2 workers x e2-standard-4 = 8 worker vCPUs)
| Chunk | Sources | Input rows | Duration | Rows/min |
|---|---|---|---|---|
| c1 | 1999-2009 | 1.28 B | 1 h 58 min | 10.9 M |
| c2 | 2010-2014 | 655 M | 1 h 11 min | 9.3 M |
| c3 | 2015-2019 | 422 M | 55 min | 7.7 M |
| c4 | 2020-2026 | 563 M | 1 h 10 min | 8.0 M |
| Total | | 2.92 B | 5 h 13 min | about 9.3 M |

Final panel: 2,854,956,909 at-risk loan-months; 2,696,005,409 labeled (12-month look-ahead); positive rate 1.210%.
At-risk filter: UPB > 0, no zero-balance code, dq_months < 3.

## Ask it twice: DuckDB vs Spark panel (2008 and 2019, 128.6 M rows)
Row counts, labeled rows, positives and delinquency sums identical in all 8 quarters.
UPB sum differs by 0.36 on about 1e12 (floating-point summation order). Verdict: OK.
Single-machine DuckDB (4 vCPU) built 8 quarters (about 150 M rows) in about 3 minutes.

## Models (time split: train to 2021-06, test from 2022-07, 12-month embargo; 0.2% loan sample)
Train 3,720,354 rows (1.33% positive) | val 411,135 | test 948,255 (0.71% positive)

| Model | AUC | AP | Brier | ECE | Mean pred (actual 0.71%) |
|---|---|---|---|---|---|
| Logistic | 0.8663 | 0.2377 | 0.00612 | 0.00123 | 0.68% |
| LightGBM | 0.8925 | 0.2773 | 0.00585 | 0.00110 | 0.63% |
| LightGBM + isotonic | 0.8905 | 0.2636 | 0.00587 | 0.00149 | 0.57% |

Ablation, no macro features:

| Model | AUC | AP | Brier | ECE | Mean pred |
|---|---|---|---|---|---|
| Logistic | 0.8790 | 0.2404 | 0.00618 | 0.00672 | 1.39% |
| LightGBM | 0.8919 | 0.2635 | 0.00612 | 0.00631 | 1.35% |
| LightGBM + isotonic | 0.8915 | 0.2504 | 0.00611 | 0.00546 | 1.26% |

Finding: macro inputs barely change ranking (AUC +0.0006) but fix the level of predicted risk.
Without them the model predicts about 2x the actual rate (it learned the high-delinquency training years).
Caveats: one run, no confidence intervals, 0.2% sample, test window 2022-2025 is a calm macro period.

## Complaint themes (Claude Haiku 5.5, 3,000 complaints, 6 themes)
- Human reference: 70 complaints labeled by an independent rater
- Agreement (6 themes): 65.7%, Cohen's kappa 0.556; merged 4 groups: 70.0%
- Author's own first pass (200 rows): 41.5% (kappa 0.22); same labeler vs own second pass on 70 rows: 41.4%
  The first pass contained clear errors (for example "closing on a mortgage" labeled foreclosure).
- CFPB-issue mapping agreement (weak sanity check): 46%, kappa 0.31
- Themes overlap (escrow vs servicing error; hardship vs modification), so human agreement is also low.

## Atlas
- mew.complaints: 147,249 documents; Atlas Search index complaints_search (Active); 3,000 documents with theme.
