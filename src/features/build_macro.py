"""Build a state-by-month macro table from the FRED download, using only data known at month t.

Usage:
    python -m src.features.build_macro --fred data/parquet/fred/fred_series.parquet \
        --out data/parquet/macro/state_month.parquet

Columns (one row per state x month, first-of-month dates):
    unemp_rate      state unemployment rate for month t-1 (released ~1 month late, so t-1 is
                    what a lender knew at t)
    unemp_chg_12m   unemp_rate(t) - unemp_rate(t-12), same lag
    hpi_yoy         year-over-year change of the state house price index. The quarterly value
                    is treated as known 5 months after the quarter starts (quarter end + ~2
                    months release lag) and forward-filled.
    mortgage_rate   monthly average of the weekly 30-year fixed rate (MORTGAGE30US)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb

log = logging.getLogger("macro")

MACRO_SQL = """
WITH s AS (SELECT series_id, obs_date, value FROM read_parquet('{fred}')),
months AS (
  SELECT CAST(unnest(generate_series(DATE '1999-01-01', DATE '2026-12-01', INTERVAL 1 MONTH))
              AS DATE) AS month
),
states AS (
  SELECT DISTINCT substr(series_id, 1, 2) AS state FROM s
  WHERE series_id LIKE '__UR' AND series_id <> 'MORTGAGE30US'
),
grid AS (SELECT state, month FROM states CROSS JOIN months),
ur AS (  -- shift one month: UR for month t-1 becomes known at t
  SELECT substr(series_id, 1, 2) AS state,
         CAST(date_trunc('month', obs_date) + INTERVAL 1 MONTH AS DATE) AS month,
         value AS unemp_rate
  FROM s WHERE series_id LIKE '__UR'
),
hpi AS (  -- quarterly value available 5 months after the quarter start
  SELECT substr(series_id, 1, 2) AS state,
         CAST(date_trunc('month', obs_date) + INTERVAL 5 MONTH AS DATE) AS month,
         value AS hpi_raw
  FROM s WHERE series_id LIKE '__STHPI'
),
mr AS (
  SELECT CAST(date_trunc('month', obs_date) AS DATE) AS month, avg(value) AS mortgage_rate
  FROM s WHERE series_id = 'MORTGAGE30US' GROUP BY 1
),
j AS (
  SELECT g.state, g.month, ur.unemp_rate, hpi.hpi_raw,
         last_value(hpi.hpi_raw IGNORE NULLS) OVER (
            PARTITION BY g.state ORDER BY g.month) AS hpi_ff
  FROM grid g
  LEFT JOIN ur  ON ur.state = g.state AND ur.month = g.month
  LEFT JOIN hpi ON hpi.state = g.state AND hpi.month = g.month
),
f AS (
  SELECT state, month, unemp_rate,
         unemp_rate - lag(unemp_rate, 12) OVER (PARTITION BY state ORDER BY month) AS unemp_chg_12m,
         hpi_ff / NULLIF(lag(hpi_ff, 12) OVER (PARTITION BY state ORDER BY month), 0) - 1 AS hpi_yoy
  FROM j
)
SELECT f.state, f.month, f.unemp_rate, f.unemp_chg_12m, f.hpi_yoy, mr.mortgage_rate
FROM f LEFT JOIN mr ON mr.month = f.month
WHERE f.unemp_rate IS NOT NULL OR mr.mortgage_rate IS NOT NULL
ORDER BY f.state, f.month
"""


def build(fred: Path, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"COPY ({MACRO_SQL.format(fred=fred)}) TO '{out}' (FORMAT PARQUET)")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    log.info("Wrote %d state-month macro rows to %s", n, out)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fred", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build(a.fred, a.out)


if __name__ == "__main__":
    main()
