# Mortgage Credit Early-Warning

Predict and stress-test 12-month serious delinquency (90+ days) from Freddie Mac loan-level data,
FRED macro series, and CFPB mortgage complaints.

## Layout

```
src/common/schemas.py          Freddie column names, null codes, types
src/ingest/freddie_to_parquet.py   zip -> partitioned Parquet (DuckDB)
src/ingest/cfpb_to_parquet.py      CFPB archive -> mortgage complaints Parquet
src/ingest/fred_download.py        FRED API -> Parquet
tests/                         synthetic-data tests
data/                          NOT in the repo (gitignored)
```

## Get the data (raw data is not included)

1. **Freddie Mac**: register (free) at the Freddie Mac Clarity portal, download the Single-Family
   Loan-Level Dataset yearly (or quarterly) zips into `data/raw/freddie/`. Manual one-time step.
2. **CFPB**: download the archive zip files from the CFPB complaint archive page into `data/raw/cfpb/`.
3. **FRED**: get a free API key, `export FRED_API_KEY=...`.

## Run

```bash
pip install -r requirements.txt
python -m src.ingest.freddie_to_parquet --zips data/raw/freddie/*.zip --out data/parquet
python -m src.ingest.cfpb_to_parquet --inputs data/raw/cfpb/*.zip --out data/parquet/cfpb
python -m src.ingest.fred_download --out data/parquet/fred
pytest -q
```

## Notes

- Column layouts differ by year. The loader logs a warning whenever a file's width differs from
  the schema in `schemas.py`; check it against the Freddie user guide for that vintage.
- Codes like 9999 (credit score) and 999 (DTI/LTV) become NULL, never imputed.
- `dq_months`: months delinquent; `RA` (REO acquisition) = 99.
- For GCS: mount with gcsfuse or `gsutil cp` zips to the VM disk and write Parquet locally, then
  `gsutil rsync` to your bucket.
