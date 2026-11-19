"""Spark panel must equal the DuckDB panel (same rows, same labels, same features)."""
import duckdb
import pytest

pytest.importorskip("pyspark")

from src.features.build_panel import build  # noqa: E402


def test_spark_matches_duckdb(freddie, tmp_path):
    from pyspark.sql import SparkSession
    from src.features.spark_panel import build_panel_df

    out_duck = tmp_path / "duck"
    build(freddie, out_duck, threads=1, memory_limit="1GB", tmp_dir=None)

    spark = (SparkSession.builder.master("local[2]").appName("t")
             .config("spark.sql.shuffle.partitions", "4").config("spark.ui.enabled", "false")
             .config("spark.driver.host", "127.0.0.1").config("spark.driver.bindAddress", "127.0.0.1")
             .getOrCreate())
    try:
        perf = spark.read.parquet(f"{freddie}/performance")
        orig = spark.read.parquet(f"{freddie}/origination")
        gmax = perf.selectExpr("max(year(monthly_reporting_period)*12+month(monthly_reporting_period))").first()[0]
        panel = build_panel_df(perf, orig, None, gmax)
        out_sp = tmp_path / "spark"
        panel.write.partitionBy("source").parquet(str(out_sp))
    finally:
        spark.stop()

    con = duckdb.connect()
    cols = "loan_id, m_idx, y_12m, dq_lag1, dq_lag3, dq_max_6m, dq_max_12m, n_late_12m, upb_ratio, rate_chg"
    a = con.execute(f"SELECT {cols} FROM read_parquet('{out_duck}/*/*.parquet') ORDER BY loan_id, m_idx").df()
    b = con.execute(f"SELECT {cols} FROM read_parquet('{out_sp}/*/*.parquet') ORDER BY loan_id, m_idx").df()
    assert len(a) == len(b) > 1000
    assert a.loan_id.tolist() == b.loan_id.tolist()
    assert a.m_idx.tolist() == b.m_idx.tolist()
    for c in ["y_12m", "dq_lag1", "dq_lag3", "dq_max_6m", "dq_max_12m", "n_late_12m", "upb_ratio", "rate_chg"]:
        assert a[c].fillna(-9).round(9).tolist() == b[c].fillna(-9).round(9).tolist(), c
