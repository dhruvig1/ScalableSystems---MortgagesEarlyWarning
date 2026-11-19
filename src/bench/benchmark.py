"""Query latency benchmark: DuckDB over Parquet vs MongoDB Atlas (indexed lookups, rollups, Atlas Search).

Usage (on the VM; MONGODB_URI must be set for the Mongo part):
    python -m src.bench.benchmark --panel ~/work/gold/panel --out results/benchmark \
        [--runs 30] [--scan-runs 3] [--skip-duckdb] [--skip-mongo]

What is measured (wall-clock milliseconds per call, p50 / p95 over repeated runs):
  DuckDB, one machine, scanning Parquet on the VM disk
    d_scan_year      aggregate the labeled panel for one origination year (about 20M+ rows)
    d_loan_history   one loan's history, partition pruned by the quarter in the loan id
  MongoDB Atlas (M0 free tier, AWS us-east-1; includes network from this machine)
    m_loan_lookup    find one scored loan by _id (indexed)
    m_top_state      top 20 riskiest loans in a state (compound index state + pd_12m)
    m_state_rates    quarterly delinquency rates for one state from the state-month rollup
    m_search         Atlas Search full-text query over complaint narratives
    m_search_state   same, filtered to one state
The point is the contrast: Parquet scans scale with data size and are great for analytics,
Mongo/Atlas answers key lookups and text search in milliseconds. M0 is a shared free tier, so
absolute numbers are indicative, not a performance claim about Atlas itself.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def timeit(fn, runs: int) -> list[float]:
    fn()  # warm-up (connection, cache) is not counted
    out = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return out


def summarize(engine: str, name: str, desc: str, ms: list[float]) -> dict:
    a = np.array(ms)
    return {"engine": engine, "query": name, "description": desc, "runs": len(a),
            "p50_ms": round(float(np.percentile(a, 50)), 1),
            "p95_ms": round(float(np.percentile(a, 95)), 1),
            "mean_ms": round(float(a.mean()), 1), "min_ms": round(float(a.min()), 1)}


def bench_duckdb(panel: Path, runs: int) -> list[dict]:
    import duckdb
    con = duckdb.connect()
    con.execute("SET threads=4; SET memory_limit='6GB'")
    rows = []
    year_q = (f"SELECT property_state, count(*) n, avg(y_12m) pos_rate "
              f"FROM read_parquet('{panel}/source=2019Q*/*.parquet') "
              f"WHERE y_12m IS NOT NULL GROUP BY 1")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{panel}/source=2019Q*/*.parquet')").fetchone()[0]
    rows.append(summarize("duckdb", "d_scan_year", f"aggregate by state, 2019 vintages ({n/1e6:.0f} M rows)",
                          timeit(lambda: con.execute(year_q).fetchall(), runs)))
    lid = con.execute(f"SELECT loan_id FROM read_parquet('{panel}/source=2019Q1/*.parquet') LIMIT 1").fetchone()[0]
    quarter = "20" + lid[1:3] + lid[3:5]  # loan ids look like F19Q1...; the folder is source=2019Q1
    hist_q = f"SELECT * FROM read_parquet('{panel}/source={quarter}/*.parquet') WHERE loan_id = '{lid}'"
    rows.append(summarize("duckdb", "d_loan_history", "one loan's monthly history (partition pruned)",
                          timeit(lambda: con.execute(hist_q).fetchall(), runs)))
    return rows


def bench_mongo(runs: int) -> list[dict]:
    from src.serve.mongo_client import get_db
    db = get_db()
    sc, rl, cp = db["scored_loans"], db["state_month_rollup"], db["complaints"]
    lid = sc.find_one({}, {"_id": 1})["_id"]
    state = "TX"
    rows = []
    rows.append(summarize("mongodb", "m_loan_lookup", "find one scored loan by _id",
                          timeit(lambda: sc.find_one({"_id": lid}), runs)))
    rows.append(summarize("mongodb", "m_top_state", f"top 20 riskiest loans in {state}",
                          timeit(lambda: list(sc.find({"property_state": state}).sort("pd_12m", -1).limit(20)), runs)))
    pipe = [{"$match": {"state": state}},
            {"$addFields": {"yr": {"$year": "$month"}, "mo": {"$month": "$month"}}},
            {"$match": {"mo": {"$in": [3, 6, 9, 12]}}},
            {"$group": {"_id": {"yr": "$yr", "mo": "$mo"}, "n_active": {"$sum": "$n_active"},
                        "n_90": {"$sum": "$n_90"}}}]
    rows.append(summarize("mongodb", "m_state_rates", f"quarterly delinquency rates for {state}",
                          timeit(lambda: list(rl.aggregate(pipe)), runs)))
    text = {"text": {"query": "escrow payment increased", "path": "narrative"}}
    s1 = [{"$search": {"index": "complaints_search", **text}}, {"$limit": 10},
          {"$project": {"issue": 1, "state": 1, "score": {"$meta": "searchScore"}}}]
    s2 = [{"$search": {"index": "complaints_search", "compound": {
              "must": [text], "filter": [{"equals": {"path": "state", "value": state}}]}}},
          {"$limit": 10}, {"$project": {"issue": 1, "state": 1}}]
    rows.append(summarize("atlas_search", "m_search", "full-text query, top 10 of 147 K narratives",
                          timeit(lambda: list(cp.aggregate(s1)), runs)))
    rows.append(summarize("atlas_search", "m_search_state", f"same query filtered to {state}",
                          timeit(lambda: list(cp.aggregate(s2)), runs)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--scan-runs", type=int, default=3)
    ap.add_argument("--skip-duckdb", action="store_true")
    ap.add_argument("--skip-mongo", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    if not a.skip_duckdb:
        rows += bench_duckdb(a.panel.expanduser(), a.scan_runs)
    if not a.skip_mongo:
        rows += bench_mongo(a.runs)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "latency_benchmark.csv", index=False)
    meta = {"host": os.uname().nodename, "date": time.strftime("%Y-%m-%d %H:%M"),
            "note": "Atlas M0 shared tier; Mongo timings include network from this host"}
    (a.out / "benchmark_meta.json").write_text(json.dumps(meta, indent=2))
    print(df.drop(columns=["description"]).to_string(index=False))


if __name__ == "__main__":
    main()
