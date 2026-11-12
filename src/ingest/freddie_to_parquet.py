"""Convert Freddie Mac loan-level zips to partitioned Parquet with DuckDB.

Usage:
    python -m src.ingest.freddie_to_parquet --zips data/raw/freddie/*.zip --out data/parquet

Output:
    <out>/origination/source=<tag>/*.parquet
    <out>/performance/source=<tag>/*.parquet
<tag> comes from the zip name (e.g. historical_data_2019.zip -> 2019, ..._2019Q1.zip -> 2019Q1).

Design choices:
  * Everything is read as VARCHAR first, then cast explicitly (no silent type guessing).
  * "Not available" codes (9999 credit score, 999 DTI, ...) become NULL. Nothing is imputed.
  * Column layout is detected per file; unknown widths are logged, extras kept as extra_<n>.
  * dq_months: 0..N months delinquent; 'RA' (REO acquisition) -> 99; unparseable -> NULL.
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import duckdb

from src.common.schemas import (
    ORIG_NULL_CODES, ORIG_NUMERIC, ORIGINATION_COLS, PERF_NULL_CODES,
    PERF_NUMERIC, PERFORMANCE_COLS,
)

log = logging.getLogger("freddie")


def detect_ncols(path: Path) -> int:
    with open(path, "r", encoding="latin-1") as f:
        return f.readline().rstrip("\n").count("|") + 1


ORIG_TRUSTED = 24  # 2019 data checked: columns 1-24 (through seller_name) are correct at 31 cols


def layout(known: list[str], n: int, kind: str = "performance") -> list[str]:
    if n == len(known):
        return list(known)
    log.warning("%s layout has %d columns, schema knows %d. Columns past the verified prefix "
                "are named unverified_<n>; check the Freddie user guide before using them.",
                kind, n, len(known))
    trusted = ORIG_TRUSTED if kind == "origination" else len(known)
    return known[:min(trusted, n)] + [f"unverified_{i}" for i in range(trusted + 1, n + 1)]


def _select_expr(names, numeric, null_codes, date_cols, month_cols=()):
    out = []
    for c in names:
        base = f"NULLIF(TRIM({c}), '')"
        if c in null_codes:
            base = f"NULLIF({base}, '{null_codes[c]}')"
        if c in numeric:
            out.append(f"TRY_CAST({base} AS {numeric[c]}) AS {c}")
        elif c in month_cols:  # YYYYMM -> first of month
            out.append(f"TRY_STRPTIME({base} || '01', '%Y%m%d')::DATE AS {c}")
        elif c in date_cols:
            out.append(f"TRY_STRPTIME({base} || '01', '%Y%m%d')::DATE AS {c}")
        else:
            out.append(f"{base} AS {c}")
    return out


def convert_file(con, src: Path, kind: str, tag: str, out_root: Path) -> int:
    known = ORIGINATION_COLS if kind == "origination" else PERFORMANCE_COLS
    names = layout(known, detect_ncols(src), kind)
    cols = ", ".join(f"'{c}': 'VARCHAR'" for c in names)
    raw = (f"read_csv('{src}', delim='|', header=false, quote='', escape='', "
           f"columns={{{cols}}}, auto_detect=false)")
    if kind == "origination":
        exprs = _select_expr(names, ORIG_NUMERIC, ORIG_NULL_CODES, set(),
                             {"first_payment_date", "maturity_date"})
    else:
        exprs = _select_expr(names, PERF_NUMERIC, PERF_NULL_CODES, set(),
                             {"monthly_reporting_period", "zero_balance_effective_date"})
        exprs.append(
            "CASE WHEN TRIM(current_loan_delinquency_status) = 'RA' THEN 99 "
            "ELSE TRY_CAST(NULLIF(TRIM(current_loan_delinquency_status), '') AS INTEGER) "
            "END AS dq_months")
    dest = out_root / kind / f"source={tag}"
    if dest.exists():
        shutil.rmtree(dest)  # clear any partial output from an interrupted run
    dest.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT {', '.join(exprs)} FROM {raw}) TO '{dest}' "
                f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000, "
                f"PER_THREAD_OUTPUT true, OVERWRITE_OR_IGNORE true)")
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{dest}/*.parquet')").fetchone()[0]
    log.info("%s %s: %d rows, %d cols", kind, tag, n, len(names))
    return n


def process_zip(zip_path: Path, out_root: Path, threads: int, memory_limit: str,
                tmp_dir: str | None = None, force: bool = False) -> None:
    """Handles both a zip holding the txt files and a yearly zip holding quarterly zips.

    Nested zips are extracted and converted one at a time to keep temp disk use low.
    A quarter that finished is marked in <out>/_done/<tag> and skipped on re-runs
    unless force=True, so an interrupted multi-year load can simply be restarted.
    """
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        with zipfile.ZipFile(zip_path) as z:
            inner = sorted(n for n in z.namelist() if n.lower().endswith(".zip"))
            if inner:
                log.info("%s contains %d nested zips", zip_path.name, len(inner))
                for n in inner:
                    mm = re.search(r"(\d{4}Q[1-4])", n)
                    if mm and not force and (out_root / "_done" / mm.group(1)).exists():
                        log.info("skip %s (already done)", mm.group(1))
                        continue
                    z.extract(n, tmp)
                    nested = Path(tmp) / n
                    process_zip(nested, out_root, threads, memory_limit, tmp_dir, force)
                    nested.unlink()
                return
            m = re.search(r"(\d{4}(?:Q[1-4])?)", zip_path.stem)
            if not m:
                raise ValueError(f"Cannot get year/quarter tag from {zip_path.name}")
            tag = m.group(1)
            marker = out_root / "_done" / tag
            if marker.exists() and not force:
                log.info("skip %s (already done)", tag)
                return
            z.extractall(tmp)
        con = duckdb.connect()
        con.execute(f"SET threads={threads}; SET memory_limit='{memory_limit}';")
        files = [p for p in Path(tmp).rglob("*.txt")]
        # Older naming: historical_data_time_YYYYQn.txt; newer: perf_YYYYQn.txt / orig_YYYYQn.txt
        perf = [p for p in files if "_time_" in p.name or p.name.lower().startswith("perf")]
        orig = [p for p in files if p not in perf]
        if len(perf) != 1 or len(orig) != 1:
            raise RuntimeError(f"{zip_path.name}: expected 1 orig + 1 perf file, "
                               f"got {[p.name for p in files]}")
        convert_file(con, orig[0], "origination", tag, out_root)
        convert_file(con, perf[0], "performance", tag, out_root)  # large; streams to disk
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory-limit", default="8GB")
    ap.add_argument("--tmp-dir", default=None, help="where to unzip (needs several GB free)")
    ap.add_argument("--force", action="store_true", help="redo quarters already marked done")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for z in sorted(args.zips):
        process_zip(z, args.out, args.threads, args.memory_limit, args.tmp_dir, args.force)


if __name__ == "__main__":
    main()
