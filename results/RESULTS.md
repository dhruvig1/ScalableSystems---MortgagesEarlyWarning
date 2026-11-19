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

## Ask it twice: state-by-quarter 90+ delinquency rate, full history (silver, 2.9 B rows joined to loan state)
DuckDB (one machine, SQL over the raw Parquet) vs MongoDB Atlas (aggregation over the stored state-month rollup):
5,939 state-quarter rows on both sides, 0 missing keys, 0 count mismatches, max rate difference 0.0 -> AGREE.
Rollup in Atlas: 17,794 state-month documents (mew.state_month_rollup).
Third engine, Spark SQL (local mode on the VM, 8 GB driver) vs DuckDB on a slice (loans originated 2008 and 2019, all
reporting months through 2026): 3,996 state-quarter rows, 0 missing keys, 0 count mismatches, max rate difference 0.0 -> AGREE.
Spark SQL took about 2.5 minutes on the slice. (Cluster-scale Spark is validated by the panel build, which matched DuckDB on 128.6 M rows.)
Definition: rate = loans 90+ days late (dq_months >= 3) / active loans (UPB > 0, no zero-balance code) at quarter-end months.

## Query latency (VM e2-standard-4 in GCP us-central1; Atlas M0 free tier on AWS us-east-1; wall-clock ms, warm-up excluded)
| Engine | Query | Runs | p50 | p95 |
|---|---|---|---|---|
| DuckDB (Parquet on local disk) | aggregate by state, 2019 vintages (69.1 M panel rows) | 3 | 1050.8 | 1082.0 |
| DuckDB | one loan's history, partition pruned | 3 | 91.2 | 92.9 |
| MongoDB | find scored loan by _id | 30 | 29.3 | 29.6 |
| MongoDB | top 20 riskiest loans in a state | 30 | 29.7 | 30.2 |
| MongoDB | quarterly delinquency rates for a state (rollup aggregation) | 30 | 61.1 | 65.7 |
| Atlas Search | full-text over 147 K narratives, top 10 | 30 | 39.4 | 50.1 |
| Atlas Search | same, filtered to one state | 30 | 39.2 | 43.1 |
Caveats: Mongo timings include a cross-cloud network hop (about 29 ms floor); the DuckDB scan is warm-cache (files already in the OS cache);
M0 is a shared tier. Indicative numbers, not a vendor benchmark.

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

Ablation, no delinquency-history features (dq_months, dq_lag1, dq_lag3, dq_max_6m, dq_max_12m, n_late_12m), time split:

| Model | AUC | AP | Brier | ECE | Mean pred (actual 0.71%) |
|---|---|---|---|---|---|
| Logistic | 0.7835 | 0.0330 | 0.00702 | 0.00189 | 0.53% |
| LightGBM | 0.7964 | 0.0425 | 0.00698 | 0.00235 | 0.48% |
| LightGBM + isotonic | 0.7954 | 0.0390 | 0.00698 | 0.00231 | 0.48% |

Vintage split (train vintages <= 2017, test vintages >= 2019; loans never seen in training; base rate 0.76%):

| Model | AUC | AP | Brier | ECE | Mean pred |
|---|---|---|---|---|---|
| Logistic | 0.8403 | 0.2215 | 0.00663 | 0.00149 | 0.64% |
| LightGBM | 0.8689 | 0.2520 | 0.00640 | 0.00143 | 0.63% |
| LightGBM + isotonic | 0.8679 | 0.2385 | 0.00641 | 0.00162 | 0.60% |

Finding (history): payment history carries most of the signal. Without it AP falls from 0.277 to 0.043
(about 39x base rate down to about 6x), AUC from 0.893 to 0.796. Static loan features plus macro still rank risk.
Finding (vintage): on loans never seen in training, LightGBM keeps AUC 0.869 and ECE 0.0014; the time split
(same loans, later months) is the more optimistic number. The vintage split still shares calendar time.

Finding (macro): macro inputs barely change ranking (AUC +0.0006) but fix the level of predicted risk.
Without them the model predicts about 2x the actual rate (it learned the high-delinquency training years).
Caveats: one run, no confidence intervals, 0.2% sample, test window 2022-2025 is a calm macro period.

## Question 2: do complaint surges lead delinquency? (state-month panel, state and month fixed effects, clustered SE)
Outcome: change in a state's 90+ delinquency rate (pp) h months after a complaint surge (z-score vs the state's own trailing 12 months),
controlling for current rate and 3-month momentum. Placebo: surge series shuffled across states within month, 20 reps.

| Series | h (months) | Coef (pp per 1 SD) | p | Largest placebo | Beats placebo |
|---|---|---|---|---|---|
| all mortgage | 1 | -0.0044 | 0.010 | 0.0045 | no |
| all mortgage | 3 | -0.0093 | 0.127 | 0.0115 | no |
| all mortgage | 6 | -0.0077 | 0.122 | 0.0148 | no |
| all mortgage | 9 | -0.0061 | 0.283 | 0.0142 | no |
| all mortgage | 12 | -0.0055 | 0.372 | 0.0223 | no |
| hardship ("struggling to pay") | 1 | -0.0014 | 0.170 | 0.0050 | no |
| hardship | 3 | not estimated (SVD did not converge) | | | |
| hardship | 6 | -0.0078 | 0.038 | 0.0065 | yes |
| hardship | 9 | -0.0099 | 0.012 | 0.0079 | yes |
| hardship | 12 | +0.0013 | 0.864 | 0.0124 | no |

Reading: no evidence that complaint surges PRECEDE rises in delinquency. Every all-mortgage coefficient is small, negative and within placebo range.
The only results that beat the placebo (hardship, 6 and 9 months) are NEGATIVE: a surge is followed by a slightly lower rate (about 0.01 pp per SD),
the opposite of the early-warning hypothesis. 9 tests were run, so one or two nominal hits are not surprising; treat as weak.
Caveats: complaints cover all lenders and servicers, delinquency is Freddie Mac loans only; state-month level; association, not causation.

## Portfolio scoring and stress scenarios (LightGBM + isotonic, June 2026 snapshot)
Scored 500,000 current loans: mean 12-month PD 0.77%, UPB-weighted 0.81% (recent actual rate 0.71%).

Monte Carlo: 200 macro-shock draws per scenario on a 100,000-loan sample (about $24 B UPB); loss = PD x UPB x LGD, LGD = 0.35 (ASSUMED, not estimated).

| Scenario | Delinquency rate p05 / p50 / p95 | Change vs baseline (p50) | Expected loss p50 |
|---|---|---|---|
| baseline | 0.756% / 0.776% / 0.794% | 0 | $65.3 M |
| unemployment +2 pp | 0.811% / 0.831% / 0.843% | +0.055 pp (+7%) | $69.8 M |
| mortgage rates +200 bp | 0.757% / 0.773% / 0.790% | -0.003 pp (about 0) | $63.8 M |
| stagflation (+2 pp, +200 bp) | 0.806% / 0.831% / 0.840% | +0.055 pp (+7%) | $68.3 M |
| severe recession (+4 pp, -100 bp) | 0.881% / 0.949% / 1.005% | +0.173 pp (+22%) | $82.1 M |

Reading: the model responds to unemployment but the response is modest (+4 pp unemployment raises the
predicted rate by about 22%). Rates have almost no effect (most loans are fixed-rate). Only macro inputs
are shocked; payment-history features stay fixed, so second-round effects (more loans slipping late) are
not captured. Treat as a lower bound on macro sensitivity, not a forecast. Bands reflect shock draws only,
not model uncertainty.

## Complaint themes (Claude Haiku 5.5, 3,000 complaints, 6 themes)
- Human reference: 70 complaints labeled by an independent rater
- Agreement (6 themes): 65.7%, Cohen's kappa 0.556; merged 4 groups: 70.0%
- Author's own first pass (200 rows): 41.5% (kappa 0.22); same labeler vs own second pass on 70 rows: 41.4%
  The first pass contained clear errors (for example "closing on a mortgage" labeled foreclosure).
- CFPB-issue mapping agreement (weak sanity check): 46%, kappa 0.31
- Themes overlap (escrow vs servicing error; hardship vs modification), so human agreement is also low.

## Atlas
- mew.complaints: 147,249 documents; Atlas Search index complaints_search (Active); 3,000 documents with theme.
