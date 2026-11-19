#!/usr/bin/env bash
# Run the same Spark jobs on Dataproc (managed Spark, quickest if you do not have GKE image set up).
# Required env: BUCKET (gs://...), REGION (e.g. us-central1), CLUSTER (Dataproc cluster name)
# Usage: bash infra/submit_dataproc.sh panel [extra args]
set -euo pipefail
: "${BUCKET:?set BUCKET}" "${REGION:?set REGION}" "${CLUSTER:?set CLUSTER}"
JOB="${1:-panel}"; shift || true
cd "$(dirname "$0")/.."
zip -qr /tmp/mew_src.zip src

case "$JOB" in
  panel)
    gcloud dataproc jobs submit pyspark src/features/spark_panel.py \
      --cluster "$CLUSTER" --region "$REGION" --py-files /tmp/mew_src.zip \
      --properties spark.sql.adaptive.enabled=true -- \
      --parquet "$BUCKET/silver" --macro "$BUCKET/silver/macro/state_month.parquet" \
      --out "$BUCKET/gold/panel" --shuffle-partitions 4000 "$@" ;;
  *) echo "unknown job $JOB"; exit 1 ;;
esac
