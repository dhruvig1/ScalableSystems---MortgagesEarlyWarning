#!/usr/bin/env bash
# Convert every Freddie zip in the bucket to Parquet, one year at a time, on the VM.
#
# For each gs://$BUCKET/raw/freddie/historical_data_YYYY.zip not yet marked done:
#   download -> freddie_to_parquet (DuckDB) -> upload Parquet to gs://$BUCKET/silver/ ->
#   mark done in the bucket -> delete local zip and Parquet (keeps disk use to ~1 year).
# Safe to stop and re-run: finished years are skipped (markers live in the bucket).
#
# Run inside tmux so a dropped SSH session does not kill it:
#     tmux new -s ingest
#     BUCKET=dhruvi-mortgage-project-1 bash scripts/vm_ingest.sh 2>&1 | tee -a ~/ingest.log
# Optional: pass years to do only those, e.g.  bash scripts/vm_ingest.sh 2007 2019
set -uo pipefail

: "${BUCKET:?set BUCKET=your-bucket-name (no gs://)}"
REPO="${REPO:-$HOME/mew}"
WORK="${WORK:-$HOME/work}"
THREADS="${THREADS:-4}"
MEM="${MEM:-10GB}"
RAW="gs://$BUCKET/raw/freddie"
OUT="gs://$BUCKET/silver"

source "$HOME/venv/bin/activate"
cd "$REPO"
mkdir -p "$WORK/zips" "$WORK/parquet" "$WORK/tmp"

if [ $# -gt 0 ]; then
  YEARS=("$@")
else
  mapfile -t YEARS < <(gsutil ls "$RAW/" | sed -n 's#.*historical_data_\([0-9]\{4\}\)\.zip#\1#p' | sort)
fi
echo "$(date) years to consider: ${YEARS[*]}"

for Y in "${YEARS[@]}"; do
  if gsutil -q stat "$OUT/_year_done/$Y"; then
    echo "$(date) skip $Y (done)"; continue
  fi
  echo "$(date) ===== $Y: download ====="
  rm -rf "$WORK/zips/"* "$WORK/parquet/"* "$WORK/tmp/"*
  if ! gsutil -q cp "$RAW/historical_data_$Y.zip" "$WORK/zips/"; then
    echo "$(date) FAILED download $Y"; continue
  fi
  echo "$(date) $Y: convert"
  if ! python -m src.ingest.freddie_to_parquet --zips "$WORK/zips/historical_data_$Y.zip" \
        --out "$WORK/parquet" --tmp-dir "$WORK/tmp" --threads "$THREADS" --memory-limit "$MEM"; then
    echo "$(date) FAILED convert $Y (see log above); continuing with next year"; continue
  fi
  echo "$(date) $Y: upload"
  if gsutil -m -q rsync -r "$WORK/parquet" "$OUT"; then
    echo "$Y $(date -u +%FT%TZ)" | gsutil -q cp - "$OUT/_year_done/$Y"
    echo "$(date) $Y done"
  else
    echo "$(date) FAILED upload $Y"; continue
  fi
  df -h / | tail -1
done

echo "$(date) ===== finished. Layout warnings and failures: ====="
grep -E "layout has|FAILED|Traceback" "$HOME/ingest.log" | sort | uniq -c | head -50 || true
