"""Build the loan-month modeling panel with the 12-month serious-delinquency label.

Usage:
    python -m src.features.build_panel --parquet data/parquet --out data/panel \
        --threads 4 --memory-limit 8GB --tmp-dir data/tmp

One row per loan per month, restricted to loans AT RISK at that month:
  * still active (UPB > 0, no zero-balance code), and
  * currently < 90 days delinquent (dq_months < 3).

Label y_12m = 1 if in any of the NEXT 12 months (t+1 .. t+12) the loan is 90+ days
delinquent (dq_months >= 3, includes REO 'RA' = 99) or exits with a credit-event
zero-balance code. y_12m is NULL when the 12-month window is not fully observed
(t + 12 months > last month in the data); drop those rows for training.
A loan that prepays inside the window with no earlier 90+ status gets y_12m = 0.

No leakage: every feature uses only rows with month <= t. Only the label looks ahead.
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import duckdb

log = logging.getLogger("panel")

# Zero-balance codes treated as credit events: 02 third-party sale, 03 short sale,
# 09 REO disposition, 15 note sale. (01 prepay, 06 repurchase, 96 removal are not.)
CREDIT_EVENT_CODES = "'02','03','09','15'"

ORIG_FEATURES = [
    "credit_score", "first_time_homebuyer_flag", "mi_pct", "num_units", "occupancy_status",
    "cltv", "dti", "orig_upb", "ltv", "orig_interest_rate", "channel", "property_state",
    "property_type", "loan_purpose", "orig_loan_term", "num_borrowers", "first_payment_date",
]


def panel_sql(source: str, parquet: Path, global_max_idx: int) -> str:
    perf = f"read_parquet('{parquet}/performance/source={source}/*.parquet', union_by_name=true)"
    orig = f"read_parquet('{parquet}/origination/source={source}/*.parquet', union_by_name=true)"
    o_cols = ", ".join(f"o.{c}" for c in ORIG_FEATURES)
    return f"""
    WITH p AS (
      SELECT loan_sequence_number AS loan_id,
             monthly_reporting_period AS rep_month,
             year(monthly_reporting_period) * 12 + month(monthly_reporting_period) AS m_idx,
             current_actual_upb AS upb, dq_months, loan_age, zero_balance_code,
             modification_flag, estimated_ltv, current_interest_rate
      FROM {perf}
    ),
    ev AS (
      SELECT *, (COALESCE(dq_months, 0) >= 3
                 OR zero_balance_code IN ({CREDIT_EVENT_CODES}))::INT AS event
      FROM p
    ),
    w AS (
      SELECT *,
        MAX(event) OVER (PARTITION BY loan_id ORDER BY m_idx
                         RANGE BETWEEN 1 FOLLOWING AND 12 FOLLOWING)         AS fut_event,
        LAG(dq_months, 1) OVER (PARTITION BY loan_id ORDER BY m_idx)         AS dq_lag1,
        LAG(dq_months, 3) OVER (PARTITION BY loan_id ORDER BY m_idx)         AS dq_lag3,
        MAX(COALESCE(dq_months, 0)) OVER (PARTITION BY loan_id ORDER BY m_idx
                         RANGE BETWEEN 5 PRECEDING AND CURRENT ROW)          AS dq_max_6m,
        MAX(COALESCE(dq_months, 0)) OVER (PARTITION BY loan_id ORDER BY m_idx
                         RANGE BETWEEN 11 PRECEDING AND CURRENT ROW)         AS dq_max_12m,
        SUM((COALESCE(dq_months, 0) >= 1)::INT) OVER (PARTITION BY loan_id ORDER BY m_idx
                         RANGE BETWEEN 11 PRECEDING AND CURRENT ROW)         AS n_late_12m
      FROM ev
    )
    SELECT w.loan_id, w.rep_month, w.m_idx, '{source}' AS source,
           w.loan_age, w.dq_months, w.dq_lag1, w.dq_lag3, w.dq_max_6m, w.dq_max_12m,
           w.n_late_12m, w.upb, w.upb / NULLIF(o.orig_upb, 0) AS upb_ratio,
           w.current_interest_rate, w.current_interest_rate - o.orig_interest_rate AS rate_chg,
           w.estimated_ltv, (w.modification_flag = 'Y')::INT AS modified,
           {o_cols},
           CASE WHEN w.m_idx + 12 <= {global_max_idx}
                THEN COALESCE(w.fut_event, 0) END AS y_12m
    FROM w JOIN {orig} o ON o.loan_sequence_number = w.loan_id
    WHERE w.upb > 0 AND w.zero_balance_code IS NULL AND COALESCE(w.dq_months, 99) < 3
    """


def global_max_month_idx(con, parquet: Path) -> int:
    return con.execute(
        f"SELECT max(year(monthly_reporting_period) * 12 + month(monthly_reporting_period)) "
        f"FROM read_parquet('{parquet}/performance/*/*.parquet', union_by_name=true)"
    ).fetchone()[0]


def build(parquet: Path, out: Path, threads: int, memory_limit: str, tmp_dir: str | None,
          only: list[str] | None = None) -> None:
    con = duckdb.connect()
    con.execute(f"SET threads={threads}; SET memory_limit='{memory_limit}';")
    if tmp_dir:
        con.execute(f"SET temp_directory='{tmp_dir}';")
    g = global_max_month_idx(con, parquet)
    log.info("Last observed month index: %d (%d-%02d)", g, (g - 1) // 12, (g - 1) % 12 + 1)
    sources = sorted(p.name.split("=", 1)[1] for p in (parquet / "performance").glob("source=*"))
    if only:  # prefixes like "2019" or "2019Q2"; the last-month cutoff still uses ALL data
        sources = [x for x in sources if any(x.startswith(o) for o in only)]
    for s in sources:
        dest = out / f"source={s}"
        if dest.exists():
            shutil.rmtree(dest)  # rebuild cleanly; avoids duplicate files on re-run
        dest.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY ({panel_sql(s, parquet, g)}) TO '{dest}' "
                    f"(FORMAT PARQUET, COMPRESSION ZSTD, PER_THREAD_OUTPUT true, "
                    f"OVERWRITE_OR_IGNORE true)")
        n, lab, pos = con.execute(
            f"SELECT count(*), count(y_12m), sum(y_12m) FROM read_parquet('{dest}/*.parquet')"
        ).fetchone()
        log.info("%s: %d rows, %d labeled, positive rate %.3f%%", s, n, lab,
                 100 * (pos or 0) / max(lab, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory-limit", default="8GB")
    ap.add_argument("--tmp-dir", default=None)
    ap.add_argument("--sources", nargs="*", default=None, help="only these sources/prefixes, e.g. 2019")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build(a.parquet, a.out, a.threads, a.memory_limit, a.tmp_dir, a.sources)


if __name__ == "__main__":
    main()
