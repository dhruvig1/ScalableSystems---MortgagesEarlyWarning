"""Filter CFPB complaint archive files to mortgage complaints and write Parquet.

Usage:
    python -m src.ingest.cfpb_to_parquet --inputs data/raw/cfpb/*.zip --out data/parquet/cfpb

Inputs are the archive zips/CSVs you downloaded from the CFPB archive page (each zip holds one
CSV). Overlapping archives are de-duplicated on Complaint ID, keeping the latest file.
Prints total mortgage complaints and how many have a narrative, to compare with the
459,836 / 147,249 counts in the proposal.
"""
from __future__ import annotations

import argparse
import logging
import tempfile
import zipfile
from pathlib import Path

import duckdb

log = logging.getLogger("cfpb")

COLS = {
    "Date received": "date_received", "Product": "product", "Sub-product": "sub_product",
    "Issue": "issue", "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "narrative", "Company": "company", "State": "state",
    "ZIP code": "zip", "Submitted via": "submitted_via", "Complaint ID": "complaint_id",
}


def collect_csvs(inputs: list[Path], tmp: str) -> list[Path]:
    csvs = []
    for p in sorted(inputs):
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                for name in z.namelist():
                    if name.lower().endswith(".csv"):
                        dest = Path(tmp) / f"{p.stem}__{Path(name).name}"
                        dest.write_bytes(z.read(name))
                        csvs.append(dest)
        elif p.suffix == ".csv":
            csvs.append(p)
    return csvs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    sel = ", ".join(f'"{k}" AS {v}' for k, v in COLS.items())
    with tempfile.TemporaryDirectory() as tmp:
        csvs = collect_csvs(args.inputs, tmp)
        if not csvs:
            raise SystemExit("No CSV/zip inputs found")
        parts = []
        for i, c in enumerate(csvs):  # order = recency rank (later file wins)
            parts.append(
                f"SELECT {sel}, {i} AS file_rank FROM read_csv('{c}', header=true, "
                f"all_varchar=true, sample_size=-1, ignore_errors=true)")
        union = " UNION ALL ".join(parts)
        con.execute(f"""
            COPY (
              SELECT * EXCLUDE (file_rank, rn) FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY complaint_id
                                              ORDER BY file_rank DESC) AS rn
                FROM ({union})
                WHERE product ILIKE '%mortgage%'
              ) WHERE rn = 1
            ) TO '{args.out / "mortgage_complaints.parquet"}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    total, with_narr = con.execute(
        f"SELECT COUNT(*), COUNT(*) FILTER (WHERE narrative IS NOT NULL AND TRIM(narrative) <> '') "
        f"FROM read_parquet('{args.out / 'mortgage_complaints.parquet'}')").fetchone()
    log.info("Mortgage complaints: %d (with narrative: %d)", total, with_narr)


if __name__ == "__main__":
    main()
