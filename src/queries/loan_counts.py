"""Check our loan counts against totals Freddie Mac publishes.

Usage:
    python -m src.queries.loan_counts --parquet data/parquet --published docs/freddie_published_counts.csv

`published` is a CSV you fill in from the Freddie Mac dataset summary / user guide:
    source,published_loans        e.g.  2019,1780000
`source` matches the tag used by the loader (2019, 2019Q1, ...). Prints our distinct-loan count
per source, the published count, and the % difference. Also reports loans with no performance
rows and performance rows with no origination record (join integrity).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger("loan_counts")


def counts(parquet: Path) -> pd.DataFrame:
    con = duckdb.connect()
    return con.execute(f"""
      WITH o AS (SELECT source, count(DISTINCT loan_sequence_number) AS n_orig
                 FROM read_parquet('{parquet}/origination/*/*.parquet', hive_partitioning=true)
                 GROUP BY 1),
           p AS (SELECT source, count(DISTINCT loan_sequence_number) AS n_perf,
                        count(*) AS perf_rows
                 FROM read_parquet('{parquet}/performance/*/*.parquet', hive_partitioning=true)
                 GROUP BY 1)
      SELECT o.source, o.n_orig, p.n_perf, p.perf_rows
      FROM o LEFT JOIN p USING (source) ORDER BY 1""").df()


def orphans(parquet: Path) -> dict:
    con = duckdb.connect()
    r = con.execute(f"""
      SELECT
        (SELECT count(DISTINCT loan_sequence_number)
           FROM read_parquet('{parquet}/performance/*/*.parquet')
          WHERE loan_sequence_number NOT IN
            (SELECT loan_sequence_number FROM read_parquet('{parquet}/origination/*/*.parquet')))
          AS perf_without_orig,
        (SELECT count(*) FROM read_parquet('{parquet}/origination/*/*.parquet')
          WHERE loan_sequence_number NOT IN
            (SELECT loan_sequence_number FROM read_parquet('{parquet}/performance/*/*.parquet')))
          AS orig_without_perf""").fetchone()
    return {"perf_without_orig": r[0], "orig_without_perf": r[1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--published", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = counts(a.parquet)
    if a.published and a.published.exists():
        pub = pd.read_csv(a.published, dtype={"source": str})
        df = df.merge(pub, on="source", how="left")
        df["pct_diff"] = 100 * (df.n_orig - df.published_loans) / df.published_loans
    print(df.to_string(index=False))
    print("Join integrity:", orphans(a.parquet))
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
