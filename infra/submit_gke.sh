#!/usr/bin/env bash
# Run the Spark panel build on a GKE cluster (Spark on Kubernetes, cluster mode).
#
# Prereqs (once): a GKE cluster, a service account with GCS access bound to a K8s service
# account named "spark", and a Spark image that has the GCS connector + this repo's src/:
#     docker build -t $IMAGE -f infra/Dockerfile.spark . && docker push $IMAGE
# Required env (set in your shell, not in this file):
#     BUCKET      gs://your-bucket        PROJECT   your-gcp-project
#     IMAGE       us-docker.pkg.dev/$PROJECT/mew/spark-mew:latest
#     K8S_MASTER  from: kubectl cluster-info  (https://<endpoint>)
# Usage: bash infra/submit_gke.sh panel [extra args]
# (The DuckDB-vs-Spark-vs-Mongo check runs on a VM where both engines read the same local
#  Parquet; see README step 6.)
set -euo pipefail
: "${BUCKET:?set BUCKET}" "${IMAGE:?set IMAGE}" "${K8S_MASTER:?set K8S_MASTER}"
JOB="${1:-panel}"; shift || true
EXECUTORS="${EXECUTORS:-20}"

case "$JOB" in
  panel)
    APP="local:///opt/mew/src/features/spark_panel.py"
    ARGS=(--parquet "$BUCKET/silver" --macro "$BUCKET/silver/macro/state_month.parquet"
          --out "$BUCKET/gold/panel" --shuffle-partitions 4000 "$@") ;;
  *) echo "unknown job $JOB"; exit 1 ;;
esac

spark-submit \
  --master "k8s://$K8S_MASTER" --deploy-mode cluster \
  --name "mew-$JOB" \
  --conf spark.executor.instances="$EXECUTORS" \
  --conf spark.executor.memory=12g --conf spark.executor.cores=3 \
  --conf spark.kubernetes.container.image="$IMAGE" \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
  --conf spark.hadoop.fs.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem \
  --conf spark.sql.adaptive.enabled=true \
  "$APP" "${ARGS[@]}"
