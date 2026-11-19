"""Airflow DAG: monthly refresh of the early-warning pipeline.

Each task is a thin shell step so the same commands work by hand (see README). Set the
variables below in Airflow (Admin -> Variables or env); no secrets live in this file.
    MEW_HOME   repo checkout on the Airflow worker
    MEW_BUCKET gs://<bucket> with raw/ (bronze), silver/, gold/
Secrets (FRED_API_KEY, MONGODB_URI, ANTHROPIC_API_KEY) come from the worker environment
or an Airflow connection / secret backend.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

HOME = "{{ var.value.get('MEW_HOME', '/opt/mew') }}"
BUCKET = "{{ var.value.get('MEW_BUCKET', 'gs://CHANGE_ME') }}"

default_args = {"owner": "dhruvi", "retries": 1, "retry_delay": timedelta(minutes=10)}

with DAG(
    dag_id="mew_monthly_refresh",
    start_date=datetime(2026, 12, 1),
    schedule="0 3 5 * *",          # 5th of each month, after FRED/CFPB publish
    catchup=False,
    default_args=default_args,
    tags=["mortgage", "early-warning"],
) as dag:
    fred = BashOperator(
        task_id="download_fred",
        bash_command=f"cd {HOME} && python -m src.ingest.fred_download --out data/parquet/fred")
    macro = BashOperator(
        task_id="build_macro",
        bash_command=f"cd {HOME} && python -m src.features.build_macro "
                     "--fred data/parquet/fred/fred_series.parquet --out data/parquet/macro/state_month.parquet "
                     f"&& gsutil -m cp data/parquet/macro/state_month.parquet {BUCKET}/silver/macro/")
    cfpb = BashOperator(
        task_id="ingest_cfpb",
        bash_command=f"cd {HOME} && python -m src.ingest.cfpb_to_parquet "
                     "--inputs data/raw/cfpb/*.zip --out data/parquet/cfpb")
    spark_panel = BashOperator(
        task_id="spark_panel_on_gke",
        bash_command=f"cd {HOME} && BUCKET={BUCKET} bash infra/submit_dataproc.sh panel")
    train = BashOperator(
        task_id="train_and_score",
        bash_command=f"cd {HOME} && python -m src.models.train --panel data/panel_spark --out data/models "
                     "--split time --sample-frac 0.1 "
                     "&& python -m src.models.score --panel data/panel_spark --models data/models "
                     "--out data/outputs/portfolio_snapshot.parquet")
    checks = BashOperator(
        task_id="state_quarter_three_way_check",
        bash_command=f"cd {HOME} && python -m src.queries.state_quarter --parquet data/parquet "
                     "--out data/outputs --spark --mongo")
    scenarios = BashOperator(
        task_id="monte_carlo",
        bash_command=f"cd {HOME} && python -m src.scenarios.monte_carlo --snapshot "
                     "data/outputs/portfolio_snapshot.parquet --models data/models --out data/outputs")
    mongo = BashOperator(
        task_id="load_atlas",
        bash_command=f"cd {HOME} && python -m src.serve.mongo_load complaints --parquet "
                     "data/parquet/cfpb/mortgage_complaints.parquet "
                     "&& python -m src.serve.mongo_load scores --parquet data/outputs/portfolio_snapshot.parquet "
                     "&& python -m src.serve.mongo_load scenarios --csv data/outputs/scenario_summary.csv "
                     "&& python -m src.serve.mongo_load indexes")
    label = BashOperator(
        task_id="claude_label_capped_sample",
        bash_command=f"cd {HOME} && python -m src.ai.label_complaints --parquet "
                     "data/parquet/cfpb/mortgage_complaints.parquet --out data/labels --n 3000 --write-mongo")
    leadlag = BashOperator(
        task_id="complaints_lead_lag",
        bash_command=f"cd {HOME} && python -m src.analysis.complaints_lead_lag --rollup "
                     "data/outputs/state_month_rollup.parquet --complaints "
                     "data/parquet/cfpb/mortgage_complaints.parquet --out data/outputs")

    [fred, cfpb] >> macro >> spark_panel >> train >> [checks, scenarios]
    cfpb >> mongo >> label
    [checks, scenarios] >> leadlag


# Daily refresh: CFPB's database updates daily, so pull only new complaints and label them.
# Written, not executed by Airflow; the same two commands were run by hand.
with DAG(
    dag_id="mew_cfpb_daily_live",
    start_date=datetime(2026, 12, 1),
    schedule="0 7 * * *",
    catchup=False,
    default_args=default_args,
    tags=["mortgage", "early-warning", "live"],
) as live_dag:
    pull = BashOperator(
        task_id="pull_new_complaints",
        bash_command=f"cd {HOME} && python -m src.ingest.cfpb_live  ")
