"""Compare the DuckDB panel with the Spark panel on the same sources.

Usage:
    python -m src.queries.compare_panels --duck data/panel_duck --spark data/panel_spark_test

Both folders hold source=<tag>/*.parquet. Compares, per source: rows, labeled rows,
positives, and the sum of a few features. Exits non-zero on any difference.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def per_source(root: Path):
    return duckdb.connect().execute(f"""
        SELECT regexp_extract(filename, 'source=([^/]+)', 1) AS source,
               count(*) AS n_rows, count(y_12m) AS n_labeled, sum(y_12m) AS n_pos,
               round(sum(coalesce(dq_max_12m, 0)), 3) AS s_dq12, round(sum(coalesce(upb, 0)), 2) AS s_upb
        FROM read_parquet('{root}/*/*.parquet', filename=true)
        GROUP BY 1 ORDER BY 1""").df().set_index("source")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duck", required=True, type=Path)
    ap.add_argument("--spark", required=True, type=Path)
    a = ap.parse_args()
    d, s = per_source(a.duck), per_source(a.spark)
    j = d.join(s, how="outer", lsuffix="_duck", rsuffix="_spark")
    print(j.to_string())
    bad = False
    for c in d.columns:
        diff = (j[f"{c}_duck"].fillna(-1) - j[f"{c}_spark"].fillna(-1)).abs().max()
        ok = diff < (0.5 if c.startswith("n_") else 1e-3 * max(1.0, float(j[f"{c}_duck"].abs().max())))
        print(f"{c}: max diff {diff:g} -> {'OK' if ok else 'MISMATCH'}")
        bad |= not ok
    pos = j.n_pos_spark.sum() / max(j.n_labeled_spark.sum(), 1)
    print(f"Spark positive rate on labeled rows: {100 * pos:.2f}%")
    if bad:
        raise SystemExit("Panels differ")


if __name__ == "__main__":
    main()
