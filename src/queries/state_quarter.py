"""State-by-quarter serious-delinquency rate, computed three ways and compared.

Definition (identical in all three engines):
    For each property state and quarter-end month (Mar, Jun, Sep, Dec):
        n_active = loans with UPB > 0 and no zero-balance code in that month
        n_90     = those with dq_months >= 3 (90+ days late, includes REO = 99)
        rate     = n_90 / n_active

Engines
    1. DuckDB SQL straight over the raw Parquet (one machine)
    2. Spark SQL over the same Parquet (cluster), also writes a state-MONTH rollup
    3. MongoDB aggregation over that stored state-month rollup (re-derives the quarter-end rows)
They use different code paths (raw rows vs stored rollup), so agreement is a real check.

Usage:
    python -m src.queries.state_quarter --parquet data/parquet --out data/outputs \
        [--sources 2019] [--spark] [--mongo]
Exits non-zero if any engine disagrees (counts exact, rates within 1e-9).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger("state_quarter")

ACTIVE = "p.current_actual_upb > 0 AND p.zero_balance_code IS NULL"

# {perf}/{orig} are relations: read_parquet(...) in DuckDB, temp views in Spark.
QUARTER_SQL = """
SELECT o.property_state AS state,
       year(p.monthly_reporting_period)    AS yr,
       quarter(p.monthly_reporting_period) AS qtr,
       COUNT(*)                                         AS n_active,
       SUM(CASE WHEN p.dq_months >= 3 THEN 1 ELSE 0 END) AS n_90
FROM {perf} p JOIN {orig} o ON o.loan_sequence_number = p.loan_sequence_number
WHERE month(p.monthly_reporting_period) IN (3, 6, 9, 12) AND """ + ACTIVE + """
  AND o.property_state IS NOT NULL
GROUP BY 1, 2, 3
"""

MONTH_SQL = """
SELECT o.property_state AS state,
       CAST(p.monthly_reporting_period AS DATE) AS month,
       COUNT(*)                                         AS n_active,
       SUM(CASE WHEN p.dq_months >= 3 THEN 1 ELSE 0 END) AS n_90
FROM {perf} p JOIN {orig} o ON o.loan_sequence_number = p.loan_sequence_number
WHERE """ + ACTIVE + """ AND o.property_state IS NOT NULL
GROUP BY 1, 2
"""


def _rel(root: Path, kind: str, sources: list[str] | None) -> str:
    if sources:
        files = ", ".join(f"'{root}/{kind}/source={s}/*.parquet'" for s in sources)
        return f"read_parquet([{files}])"
    return f"read_parquet('{root}/{kind}/*/*.parquet')"


def duckdb_quarter(root: Path, sources=None, threads: int = 4) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute(f"SET threads={threads}; SET memory_limit='9GB'; "
                f"SET temp_directory='{os.path.expanduser('~/work/tmp')}'")
    sql = QUARTER_SQL.format(perf=_rel(root, "performance", sources),
                             orig=_rel(root, "origination", sources))
    return con.execute(sql).df()


def duckdb_month(root: Path, sources=None, threads: int = 4) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute(f"SET threads={threads}; SET memory_limit='9GB'; "
                f"SET temp_directory='{os.path.expanduser('~/work/tmp')}'")
    sql = MONTH_SQL.format(perf=_rel(root, "performance", sources),
                           orig=_rel(root, "origination", sources))
    return con.execute(sql).df()


def spark_run(root: Path, sources=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("mew-state-quarter").getOrCreate()
    for kind, view in (("performance", "perf_v"), ("origination", "orig_v")):
        df = spark.read.option("mergeSchema", "true").parquet(f"{root}/{kind}")
        if sources:
            df = df.filter(df.source.isin(sources))
        df.createOrReplaceTempView(view)
    q = spark.sql(QUARTER_SQL.format(perf="perf_v", orig="orig_v")).toPandas()
    m = spark.sql(MONTH_SQL.format(perf="perf_v", orig="orig_v")).toPandas()
    m["month"] = pd.to_datetime(m["month"])
    spark.stop()
    return q, m


def mongo_quarter(db) -> pd.DataFrame:
    """Aggregation over the stored state-month rollup: keep quarter-end months, sum by quarter."""
    pipeline = [
        {"$addFields": {"yr": {"$year": "$month"}, "mo": {"$month": "$month"}}},
        {"$match": {"mo": {"$in": [3, 6, 9, 12]}}},
        {"$group": {"_id": {"state": "$state", "yr": "$yr", "qtr": {"$ceil": {"$divide": ["$mo", 3]}}},
                    "n_active": {"$sum": "$n_active"}, "n_90": {"$sum": "$n_90"}}},
        {"$project": {"_id": 0, "state": "$_id.state", "yr": "$_id.yr", "qtr": "$_id.qtr",
                      "n_active": 1, "n_90": 1}},
    ]
    return pd.DataFrame(list(db["state_month_rollup"].aggregate(pipeline, allowDiskUse=True)))


def compare(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> dict:
    keys = ["state", "yr", "qtr"]
    x, y = a.copy(), b.copy()
    for d in (x, y):
        d[["yr", "qtr", "n_active", "n_90"]] = d[["yr", "qtr", "n_active", "n_90"]].astype("int64")
        d["rate"] = d.n_90 / d.n_active
    m = x.merge(y, on=keys, how="outer", suffixes=("_a", "_b"), indicator=True)
    missing = int((m["_merge"] != "both").sum())
    both = m[m["_merge"] == "both"]
    cnt_mismatch = int(((both.n_active_a != both.n_active_b) | (both.n_90_a != both.n_90_b)).sum())
    max_rate_diff = float((both.rate_a - both.rate_b).abs().max()) if len(both) else float("nan")
    ok = missing == 0 and cnt_mismatch == 0 and max_rate_diff < 1e-9
    log.info("%s vs %s: rows %d/%d, missing keys %d, count mismatches %d, max rate diff %.2e -> %s",
             name_a, name_b, len(x), len(y), missing, cnt_mismatch, max_rate_diff,
             "AGREE" if ok else "DISAGREE")
    return {"pair": f"{name_a} vs {name_b}", "rows_a": len(x), "rows_b": len(y),
            "missing_keys": missing, "count_mismatches": cnt_mismatch,
            "max_rate_diff": max_rate_diff, "agree": bool(ok)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--spark", action="store_true", help="also run Spark SQL (and write rollup)")
    ap.add_argument("--mongo", action="store_true",
                    help="load the Spark/DuckDB rollup into Atlas and aggregate there")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    a.out.mkdir(parents=True, exist_ok=True)

    d_q = duckdb_quarter(a.parquet, a.sources, a.threads)
    d_q.to_csv(a.out / "state_quarter_duckdb.csv", index=False)
    results = []
    rollup = None
    if a.spark:
        s_q, rollup = spark_run(a.parquet, a.sources)
        s_q.to_csv(a.out / "state_quarter_spark.csv", index=False)
        results.append(compare(d_q, s_q, "duckdb", "spark"))
    if rollup is None:
        rollup = duckdb_month(a.parquet, a.sources, a.threads)  # fallback if Spark not run
        rollup["month"] = pd.to_datetime(rollup["month"])
    rollup.to_parquet(a.out / "state_month_rollup.parquet", index=False)
    if a.mongo:
        from src.serve.mongo_client import get_db
        from src.serve.mongo_load import load_rollups
        db = get_db()
        load_rollups(db, rollup)
        m_q = mongo_quarter(db)
        m_q.to_csv(a.out / "state_quarter_mongo.csv", index=False)
        results.append(compare(d_q, m_q, "duckdb", "mongodb"))
    (a.out / "state_quarter_check.json").write_text(json.dumps(results, indent=2))
    if results and not all(r["agree"] for r in results):
        raise SystemExit("Engines disagree - see state_quarter_check.json")
    log.info("Done. %d state-quarter rows. Checks: %s", len(d_q),
             "all agree" if results else "none requested")


if __name__ == "__main__":
    main()
