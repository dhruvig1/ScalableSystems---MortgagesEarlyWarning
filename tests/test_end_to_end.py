"""Whole chain on synthetic data: FRED -> macro -> Spark panel -> train -> score -> scenarios."""
from argparse import Namespace

import duckdb
import pandas as pd
import pytest

pytest.importorskip("pyspark")

STATES = ["IL", "CA", "TX", "FL"]


def _fred(path):
    rows = []
    for s in STATES:
        for i, d in enumerate(pd.date_range("2018-01-01", "2022-12-01", freq="MS")):
            rows.append((f"{s}UR", d.date(), 4.0 + 0.05 * i))
        for i, d in enumerate(pd.date_range("2018-01-01", "2022-10-01", freq="QS")):
            rows.append((f"{s}STHPI", d.date(), 100.0 + 2 * i))
    for d in pd.date_range("2018-01-01", "2022-12-01", freq="MS"):
        rows.append(("MORTGAGE30US", d.date(), 4.0))
    df = pd.DataFrame(rows, columns=["series_id", "obs_date", "value"])
    duckdb.connect().execute(f"COPY (SELECT * FROM df) TO '{path}' (FORMAT PARQUET)")


def test_pipeline(freddie, tmp_path):
    from pyspark.sql import SparkSession
    from src.features.build_macro import build as build_macro
    from src.features.spark_panel import build_panel_df
    from src.models.score import latest_snapshot, load_model, predict
    from src.models.train import run as train
    from src.scenarios.monte_carlo import simulate, summarize

    _fred(tmp_path / "fred.parquet")
    build_macro(tmp_path / "fred.parquet", tmp_path / "macro.parquet")

    spark = (SparkSession.builder.master("local[2]").config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.driver.host", "127.0.0.1").config("spark.driver.bindAddress", "127.0.0.1")
             .getOrCreate())
    try:
        perf = spark.read.parquet(f"{freddie}/performance")
        orig = spark.read.parquet(f"{freddie}/origination")
        macro = spark.read.parquet(str(tmp_path / "macro.parquet"))
        gmax = perf.selectExpr("max(year(monthly_reporting_period)*12+month(monthly_reporting_period))").first()[0]
        build_panel_df(perf, orig, macro, gmax).write.partitionBy("source").parquet(str(tmp_path / "panel"))
    finally:
        spark.stop()

    cols = {r[0] for r in duckdb.connect().execute(
        f"DESCRIBE SELECT * FROM read_parquet('{tmp_path}/panel/*/*.parquet')").fetchall()}
    assert {"unemp_rate", "unemp_chg_12m", "hpi_yoy", "mortgage_rate", "rate_gap"} <= cols

    res = train(Namespace(panel=str(tmp_path / "panel"), out=str(tmp_path / "models"), split="time",
                          train_end="2019-12", test_start="2021-01", train_vintage_max=2017,
                          test_vintage_min=2019, sample_frac=1.0, max_logit_rows=100000, threads=1))
    assert 0.0 < res["lgbm"]["auc"] < 1.0  # plumbing check; model quality is covered in test_train
    booster, iso, feats = load_model(tmp_path / "models")
    assert "unemp_rate" in feats["num"]

    snap = latest_snapshot(tmp_path / "panel", feats, 500)
    assert len(snap) > 50 and snap.rep_month.nunique() == 1
    snap["pd_12m"] = predict(booster, iso, snap, feats)
    assert snap.pd_12m.between(0, 1).all()

    sims = pd.concat([simulate(snap, booster, iso, feats, sc, 3, 0.35)
                      for sc in ("baseline", "stagflation")], ignore_index=True)
    assert len(summarize(sims)) == 2
