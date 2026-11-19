# Key queries

Six non-trivial queries. Each has a file, what it does, and a result from the real data.

| # | Query | Where | Result |
|---|---|---|---|
| 1 | 12-month forward label with lags: for every loan-month, did a 90+ day delinquency or credit-event exit occur in the next 12 months (`max(event)` over `rangeBetween(1, 12)` per loan), plus `dq_lag1`, `dq_lag3`, rolling 6- and 12-month maxima and counts | `src/features/spark_panel.py`, `build_panel_df` (Spark windows partitioned by loan, ordered by month) | 2,854,956,909 at-risk loan-months, 2,696,005,409 labeled, 1.21% positive; 5 h 13 min on 8 vCPUs |
| 2 | Same panel in SQL on one machine, as an independent check | `src/features/build_panel.py` (DuckDB) compared by `src/queries/compare_panels.py` | Identical to Spark on 128.6 M rows (2008 and 2019 vintages) |
| 3 | State-by-quarter 90+ delinquency rate, three engines | `src/queries/state_quarter.py` (DuckDB, Spark SQL, MongoDB aggregation over `state_month_rollup`) | 5,939 rows DuckDB vs MongoDB, 3,996 rows DuckDB vs Spark SQL, 0 mismatches |
| 4 | Lead-lag: change in a state's delinquency rate h months after a complaint surge (z-score vs own trailing 12 months), state and month fixed effects, clustered errors, placebo by shuffling across states | `src/analysis/complaints_lead_lag.py` | No early-warning signal; see `results/RESULTS.md` |
| 5 | Portfolio scoring and Monte Carlo: score the latest loan snapshot, shock only macro inputs, loss = PD x UPB x 0.35 | `src/models/score.py`, `src/scenarios/` | Severe recession +22% delinquency, $65.3 M to $82.1 M loss |
| 6 | Atlas Search over narratives with a state filter and a facet count by issue | `src/serve/search_demo.py` | 39 ms p50 over 147 K narratives |

Other checks: join integrity and orphans (`src/queries/loan_counts.py`), per-source quality report
(`src/queries/silver_report.py`), latency benchmark (`src/bench/benchmark.py`).

## To show in the recording
- Query 1: open `src/features/spark_panel.py` at the window definitions (`fut = w.rangeBetween(1, 12)`),
  explain it, then show the chunk table in `results/RESULTS.md`.
- Query 3: run `python -m src.queries.state_quarter` and show the zero-mismatch output.
- Query 6: `python -m src.serve.search_demo --query "escrow payment increased" --state TX`.
- Queries 4 and 5: dashboard tabs (lead-lag, scenarios) or the CSVs in `results/`.

## Refreshing piece
`python -m src.ingest.cfpb_live --dry-run` pulls new mortgage complaints from CFPB's API (updated daily)
and prints counts by month; without `--dry-run` it upserts them into the Atlas collection `complaints_live`.
CFPB's API returns metadata only (date, state, issue, company), not narrative text, so narratives stay
archive-only and the live feed refreshes the complaint counts that Q2 uses. Daily Airflow DAG
`mew_cfpb_daily_live` runs this step (written, not executed in Airflow).
