# Data reconciliation: our counts vs what Freddie Mac publishes

Freddie Mac does not publish loan counts per origination quarter, so a quarter-by-quarter comparison
with published totals is not possible. What can be compared, and what we checked instead:

| Check | Published fact | Ours | Result |
|---|---|---|---|
| Coverage | Loans originated 1 Jan 1999 to 30 Jun 2026; monthly performance through 30 Jun 2026 | 110 quarterly sources, 1999Q1 to 2026Q2; last reporting month 2026-06 | Match |
| Total loans | "About 56 million" mortgages for the **Standard and Non-Standard** datasets combined | 49,483,298 loans from the Standard files we downloaded | Not like-for-like (Non-Standard files not used); no match or mismatch is claimed |

Published reference points (Freddie Mac, accessed Nov 2026):
- Dataset page: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
- User guide: https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf
- Release notes: https://www.freddiemac.com/fmac-resources/research/pdf/release_notes.pdf

## Internal consistency checks (no external numbers needed)

Run by `src/queries/silver_report.py` and `src/queries/loan_counts.py` on every source:
- Origination loans equal distinct performance loans per source (join integrity); 293 loans from the newest
  quarters have origination records but no performance rows yet.
- No performance loans without an origination record.
- Delinquency status within 0..99, 90+ share plausible in every source (peak 5.94% of rows in 2007Q4),
  null rates of credit score, LTV, DTI and state under thresholds.
- The same state-by-quarter rate reproduced exactly by DuckDB, Spark SQL and MongoDB (see `results/RESULTS.md`).
- The Spark panel reproduced exactly by DuckDB on 128.6 M rows.

`docs/freddie_published_counts.csv` is left empty on purpose; `loan_counts.py` and `silver_report.py` skip the
comparison when it has no values.
