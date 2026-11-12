"""Download FRED macro series to Parquet-ready CSV/Parquet.

Usage:
    export FRED_API_KEY=...
    python -m src.ingest.fred_download --out data/parquet/fred

Series:
  MORTGAGE30US        30-yr fixed mortgage rate (weekly, Freddie PMMS)
  <ST>UR              state unemployment rate (monthly), e.g. CAUR
  <ST>STHPI           state all-transactions house price index (quarterly), e.g. CASTHPI
Series IDs follow FRED's naming pattern; the script reports any that return an error so you can fix them.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import duckdb
import requests

log = logging.getLogger("fred")
URL = "https://api.stlouisfed.org/fred/series/observations"
STATES = ("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE "
          "NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY").split()


def fetch(series_id: str, key: str) -> list[tuple[str, str, float | None]]:
    r = requests.get(URL, params={"series_id": series_id, "api_key": key, "file_type": "json"},
                     timeout=30)
    r.raise_for_status()
    rows = []
    for o in r.json()["observations"]:
        v = None if o["value"] == "." else float(o["value"])
        rows.append((series_id, o["date"], v))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise SystemExit("Set FRED_API_KEY")
    ids = ["MORTGAGE30US"] + [f"{s}UR" for s in STATES] + [f"{s}STHPI" for s in STATES]
    rows, failed = [], []
    for sid in ids:
        try:
            rows += fetch(sid, key)
        except Exception as e:  # keep going, report at end
            failed.append((sid, str(e)))
        time.sleep(0.2)  # stay well under rate limit
    args.out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("CREATE TABLE s(series_id VARCHAR, obs_date DATE, value DOUBLE)")
    con.executemany("INSERT INTO s VALUES (?, ?, ?)", rows)
    con.execute(f"COPY s TO '{args.out / 'fred_series.parquet'}' (FORMAT PARQUET)")
    log.info("Wrote %d observations for %d series", len(rows), len(ids) - len(failed))
    for sid, err in failed:
        log.warning("FAILED %s: %s", sid, err)


if __name__ == "__main__":
    main()
