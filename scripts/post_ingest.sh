#!/usr/bin/env bash
# After the Freddie ingest finishes: confirm all years are done, sync silver to the VM disk,
# run the integrity report, and (optionally) the EDA. Run in tmux on the VM:
#     BUCKET=dhruvi-mortgage-project-1 bash scripts/post_ingest.sh 2>&1 | tee -a ~/post_ingest.log
# Set EDA=1 to also run src.eda.profile at the end.
set -uo pipefail
: "${BUCKET:?set BUCKET=your-bucket-name (no gs://)}"
REPO="${REPO:-$HOME/mew}"
WORK="${WORK:-$HOME/work}"
FIRST="${FIRST:-1999}"; LAST="${LAST:-2026}"
source "$HOME/venv/bin/activate"
cd "$REPO"

echo "== years marked done =="
DONE=$(gsutil ls "gs://$BUCKET/silver/_year_done/" 2>/dev/null | sed 's#.*/##' | sort)
echo "$DONE" | tr '\n' ' '; echo
N=$(echo "$DONE" | grep -c .)
WANT=$((LAST - FIRST + 1))
if [ "$N" -ne "$WANT" ]; then echo "WARNING: $N of $WANT years done. Not all years are in yet."; fi

echo "== sizes =="
echo -n "bronze (raw Freddie zips): "; gsutil du -sh "gs://$BUCKET/raw/freddie" | awk '{print $1,$2}'
echo -n "silver (all):              "; gsutil du -sh "gs://$BUCKET/silver" | awk '{print $1,$2}'

echo "== free disk =="
df -h / | tail -1
echo "Silver is synced to $WORK/silver. Stop here (Ctrl-C) if free space is too low."
mkdir -p "$WORK/silver"
gsutil -m rsync -r "gs://$BUCKET/silver" "$WORK/silver"

echo "== integrity report =="
python -m src.queries.silver_report --silver "$WORK/silver" --out "$WORK/outputs/silver_report" \
    --published docs/freddie_published_counts.csv --threads "${THREADS:-4}" --memory-limit "${MEM:-8GB}"

if [ "${EDA:-0}" = "1" ]; then
  echo "== EDA =="
  python -m src.eda.profile --parquet "$WORK/silver" --cfpb "$WORK/silver/cfpb/mortgage_complaints.parquet" \
      --out "$WORK/outputs/eda" --threads "${THREADS:-4}"
fi
echo "Done. Outputs in $WORK/outputs. Copy to bucket: gsutil -m rsync -r $WORK/outputs gs://$BUCKET/gold/outputs"
