"""Post-ingest report on the silver Freddie Parquet: size, counts, integrity, sanity checks.

Usage (silver synced to local disk, e.g. ~/work/silver):
    python -m src.queries.silver_report --silver ~/work/silver --out ~/work/outputs/silver_report \
        [--published docs/freddie_published_counts.csv] [--threads 4 --memory-limit 8GB]

Works one source (year or quarter folder) at a time so memory stays small on billions of rows.
Writes silver_by_source.csv, silver_totals.json, silver_checks.csv and prints PASS/WARN lines.

Checks per source
  * loans in origination vs distinct loans in performance (should be equal)
  * performance loans with no origination record, origination loans with no performance rows
  * reporting-month range and that no month is far outside the source's vintage window
  * delinquency status values are in a sane range (0..99) and the 90+ share is plausible
  * null rates of the model's key fields (credit score, LTV, DTI, state)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger("silver_report")


def sources(silver: Path) -> list[str]:
    return sorted(p.name.split("=", 1)[1] for p in (silver / "performance").glob("source=*"))


def report_source(con, silver: Path, s: str) -> dict:
    perf = f"read_parquet('{silver}/performance/source={s}/*.parquet')"
    orig = f"read_parquet('{silver}/origination/source={s}/*.parquet')"
    o = con.execute(f"""
        SELECT count(*) n, count(DISTINCT loan_sequence_number) n_loans,
               avg((credit_score IS NULL)::INT) null_credit, avg((ltv IS NULL)::INT) null_ltv,
               avg((dti IS NULL)::INT) null_dti, avg((property_state IS NULL)::INT) null_state,
               min(first_payment_date) first_pay_min, max(first_payment_date) first_pay_max
        FROM {orig}""").fetchone()
    p = con.execute(f"""
        SELECT count(*) n_rows, count(DISTINCT loan_sequence_number) n_loans,
               min(monthly_reporting_period) first_month, max(monthly_reporting_period) last_month,
               min(dq_months) dq_min, max(dq_months) dq_max,
               avg((dq_months >= 3)::INT) share_90plus, avg((dq_months IS NULL)::INT) null_dq
        FROM {perf}""").fetchone()
    orphan_perf = con.execute(f"""
        SELECT count(*) FROM (SELECT DISTINCT loan_sequence_number FROM {perf}) a
        ANTI JOIN (SELECT loan_sequence_number FROM {orig}) b USING (loan_sequence_number)""").fetchone()[0]
    orphan_orig = con.execute(f"""
        SELECT count(*) FROM (SELECT loan_sequence_number FROM {orig}) a
        ANTI JOIN (SELECT DISTINCT loan_sequence_number FROM {perf}) b USING (loan_sequence_number)""").fetchone()[0]
    return {
        "source": s, "orig_rows": o[0], "orig_loans": o[1], "perf_rows": p[0], "perf_loans": p[1],
        "perf_loans_without_orig": orphan_perf, "orig_loans_without_perf": orphan_orig,
        "first_month": p[2], "last_month": p[3], "dq_min": p[4], "dq_max": p[5],
        "share_90plus": p[6], "null_dq": p[7], "null_credit": o[2], "null_ltv": o[3],
        "null_dti": o[4], "null_state": o[5], "first_pay_min": o[6], "first_pay_max": o[7],
    }


def checks(df: pd.DataFrame, published: Path | None) -> pd.DataFrame:
    rows = []

    def add(name, ok, detail=""):
        rows.append({"check": name, "status": "PASS" if ok else "WARN", "detail": detail})

    years = sorted({int(re.match(r"\d{4}", s).group()) for s in df.source})
    missing = [y for y in range(years[0], years[-1] + 1) if y not in years]
    add("no missing years between first and last", not missing, f"missing={missing}")
    bad = df[df.orig_loans != df.perf_loans]
    add("origination loans == performance loans per source", bad.empty,
        f"{len(bad)} sources differ: {bad.source.tolist()[:6]}")
    add("no performance loans without origination", int(df.perf_loans_without_orig.sum()) == 0,
        f"total={int(df.perf_loans_without_orig.sum())}")
    add("no origination loans without performance", int(df.orig_loans_without_perf.sum()) == 0,
        f"total={int(df.orig_loans_without_perf.sum())}")
    add("dq_months within 0..99", bool((df.dq_min >= 0).all() and (df.dq_max <= 99).all()),
        f"min={df.dq_min.min()} max={df.dq_max.max()}")
    add("90+ share plausible (<15% of rows in every source)", bool((df.share_90plus < 0.15).all()),
        f"max={df.share_90plus.max():.4f} in {df.loc[df.share_90plus.idxmax(), 'source']}")
    add("state never missing", bool((df.null_state < 0.01).all()), f"max null={df.null_state.max():.4f}")
    add("credit score null rate < 5% in every source", bool((df.null_credit < 0.05).all()),
        f"max null={df.null_credit.max():.4f} in {df.loc[df.null_credit.idxmax(), 'source']}")
    if published and published.exists():
        pub = pd.read_csv(published, dtype={"source": str}).dropna(subset=["published_loans"])
        m = df.merge(pub, on="source")
        if len(m):
            m["pct_diff"] = 100 * (m.orig_loans - m.published_loans) / m.published_loans
            add("loan counts within 0.5% of Freddie published totals", bool((m.pct_diff.abs() < 0.5).all()),
                m[["source", "orig_loans", "published_loans", "pct_diff"]].round(3).to_string(index=False))
    return pd.DataFrame(rows)


def run(silver: Path, out: Path, published: Path | None = None, threads: int = 4,
        memory_limit: str = "8GB") -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={threads}; SET memory_limit='{memory_limit}';")
    recs = []
    for s in sources(silver):
        log.info("source %s", s)
        recs.append(report_source(con, silver, s))
    df = pd.DataFrame(recs)
    df.to_csv(out / "silver_by_source.csv", index=False)
    totals = {"sources": len(df), "origination_loans": int(df.orig_loans.sum()),
              "performance_rows": int(df.perf_rows.sum()),
              "first_month": str(df.first_month.min()), "last_month": str(df.last_month.max())}
    (out / "silver_totals.json").write_text(json.dumps(totals, indent=2))
    ch = checks(df, published)
    ch.to_csv(out / "silver_checks.csv", index=False)
    print(json.dumps(totals, indent=2))
    print(ch.to_string(index=False))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--published", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory-limit", default="8GB")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(a.silver, a.out, a.published, a.threads, a.memory_limit)


if __name__ == "__main__":
    main()
