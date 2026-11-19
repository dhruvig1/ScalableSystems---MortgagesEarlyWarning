#!/usr/bin/env bash
# Build the gold loan-month panel for ALL years on a Dataproc cluster, in chunks.
# Each chunk is a separate Spark job. A finished chunk is marked in the bucket and skipped on re-run,
# so if one fails you only repeat that chunk. Output: gs://<bucket>/gold/panel/source=<quarter>/
#
#   export BUCKET=gs://dhruvi-mortgage-project-1 REGION=us-central1 CLUSTER=mew-full
#   bash scripts/run_full_panel.sh 2>&1 | tee -a ~/panel_run.log
# Optional: pass chunk names to run only those, e.g.  bash scripts/run_full_panel.sh c3 c4
set -uo pipefail
: "${BUCKET:?set BUCKET=gs://your-bucket}" "${REGION:?set REGION}" "${CLUSTER:?set CLUSTER}"
cd "$(dirname "$0")/.."

declare -A CHUNKS=(
  [c1]="1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009"
  [c2]="2010 2011 2012 2013 2014"
  [c3]="2015 2016 2017 2018 2019"
  [c4]="2020 2021 2022 2023 2024 2025 2026"
)
WANT=("${@:-c1 c2 c3 c4}")
# shellcheck disable=SC2206
WANT=(${WANT[@]})

for name in "${WANT[@]}"; do
  marker="$BUCKET/gold/_chunk_done/$name"
  if gcloud storage ls "$marker" >/dev/null 2>&1; then echo "$(date -u +%T) skip $name (done)"; continue; fi
  echo "$(date -u +%T) ===== $name: ${CHUNKS[$name]} ====="
  # shellcheck disable=SC2086
  if gcloud dataproc jobs submit pyspark src/features/spark_panel.py \
        --cluster "$CLUSTER" --region "$REGION" --properties spark.sql.adaptive.enabled=true -- \
        --parquet "$BUCKET/silver" --macro "$BUCKET/silver/macro/state_month.parquet" \
        --out "$BUCKET/gold/panel" --shuffle-partitions 1000 --sources ${CHUNKS[$name]}; then
    echo "$name $(date -u +%FT%TZ)" | gcloud storage cp - "$marker" >/dev/null
    echo "$(date -u +%T) $name done"
  else
    echo "$(date -u +%T) FAILED $name - re-run this script to retry it"; exit 1
  fi
done
echo "$(date -u +%T) all requested chunks finished"
