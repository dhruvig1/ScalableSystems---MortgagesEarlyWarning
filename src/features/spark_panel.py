"""Spark version of the loan-month panel build, plus the macro join. Runs on GKE / Dataproc.

Why Spark: the Freddie history is billions of loan-month rows. The per-loan window functions
(12-month look-ahead label, lags, rolling maxima) need every row of a loan on one executor,
so the job shuffles by loan_id and scales out by adding executors. DuckDB (build_panel.py)
produces the same panel on one machine and is used here as the cross-check on small slices.

Usage (local test on a slice):
    spark-submit src/features/spark_panel.py --parquet data/parquet \
        --macro data/parquet/macro/state_month.parquet --out data/panel_spark --sources 2019

Usage (cluster): see infra/submit_gke.sh or infra/submit_dataproc.sh.

Label and filters match build_panel.py exactly:
  * at-risk rows only: UPB > 0, no zero-balance code, dq_months < 3
  * y_12m = 1 if any of months t+1..t+12 is 90+ days late or a credit-event exit
  * y_12m is NULL when t+12 is past the last observed month
Macro features (unemp_rate, unemp_chg_12m, hpi_yoy, mortgage_rate, rate_gap) come from
build_macro.py and only use information known at month t.
"""
from __future__ import annotations

import argparse
import logging

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

log = logging.getLogger("spark_panel")

CREDIT_EVENT_CODES = ["02", "03", "09", "15"]
ORIG_FEATURES = [
    "credit_score", "first_time_homebuyer_flag", "mi_pct", "num_units", "occupancy_status",
    "cltv", "dti", "orig_upb", "ltv", "orig_interest_rate", "channel", "property_state",
    "property_type", "loan_purpose", "orig_loan_term", "num_borrowers", "first_payment_date",
]


def _read(spark, root: str, kind: str, sources: list[str] | None):
    df = spark.read.option("mergeSchema", "true").parquet(f"{root}/{kind}")
    if sources:  # "2019" matches 2019Q1..2019Q4; "2019Q2" matches only that quarter
        cond = None
        for s_ in sources:
            c_ = F.col("source").startswith(s_)
            cond = c_ if cond is None else (cond | c_)
        df = df.filter(cond)
    return df


def build_panel_df(perf, orig, macro=None, global_max_idx: int | None = None):
    """Pure DataFrame transform (unit-testable on a local SparkSession)."""
    p = perf.select(
        F.col("loan_sequence_number").alias("loan_id"),
        F.col("monthly_reporting_period").alias("rep_month"),
        (F.year("monthly_reporting_period") * 12 + F.month("monthly_reporting_period")).alias("m_idx"),
        F.col("current_actual_upb").alias("upb"),
        "dq_months", "loan_age", "zero_balance_code", "modification_flag", "estimated_ltv",
        "current_interest_rate", "source",
    )
    dq0 = F.coalesce(F.col("dq_months"), F.lit(0))
    ev = p.withColumn(
        "event", ((dq0 >= 3) | F.col("zero_balance_code").isin(CREDIT_EVENT_CODES)).cast("int"))
    w = Window.partitionBy("loan_id").orderBy("m_idx")
    fut = w.rangeBetween(1, 12)
    p6 = w.rangeBetween(-5, 0)
    p12 = w.rangeBetween(-11, 0)
    ev = (ev.withColumn("fut_event", F.max("event").over(fut))
            .withColumn("dq_lag1", F.lag("dq_months", 1).over(w))
            .withColumn("dq_lag3", F.lag("dq_months", 3).over(w))
            .withColumn("dq_max_6m", F.max(dq0).over(p6))
            .withColumn("dq_max_12m", F.max(dq0).over(p12))
            .withColumn("n_late_12m", F.sum((dq0 >= 1).cast("int")).over(p12)))
    if global_max_idx is None:
        global_max_idx = ev.agg(F.max("m_idx")).first()[0]

    o = orig.select(F.col("loan_sequence_number").alias("loan_id"), *ORIG_FEATURES)
    j = ev.join(o, "loan_id")
    j = (j.withColumn("upb_ratio", F.col("upb") / F.when(F.col("orig_upb") != 0, F.col("orig_upb")))
          .withColumn("rate_chg", F.col("current_interest_rate") - F.col("orig_interest_rate"))
          .withColumn("modified", (F.col("modification_flag") == "Y").cast("int"))
          .withColumn("y_12m", F.when(F.col("m_idx") + 12 <= global_max_idx,
                                      F.coalesce(F.col("fut_event"), F.lit(0)))))
    j = j.filter((F.col("upb") > 0) & F.col("zero_balance_code").isNull()
                 & (F.coalesce(F.col("dq_months"), F.lit(99)) < 3))
    if macro is not None:
        m = macro.select(F.col("state").alias("property_state"), F.col("month").alias("rep_month"),
                         "unemp_rate", "unemp_chg_12m", "hpi_yoy", "mortgage_rate")
        j = j.join(F.broadcast(m), ["property_state", "rep_month"], "left")
        j = j.withColumn("rate_gap", F.col("current_interest_rate") - F.col("mortgage_rate"))
    drop = ["fut_event", "event", "zero_balance_code", "modification_flag"]
    return j.drop(*drop)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="root with origination/ and performance/")
    ap.add_argument("--macro", default=None, help="state_month.parquet from build_macro.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="*", default=None, help="e.g. 2019 2020Q1 (default: all)")
    ap.add_argument("--shuffle-partitions", type=int, default=2000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    spark = (SparkSession.builder.appName("mew-panel")
             .config("spark.sql.shuffle.partitions", a.shuffle_partitions)
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.sql.adaptive.skewJoin.enabled", "true")
             .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
             .getOrCreate())
    perf_all = _read(spark, a.parquet, "performance", None)
    g = perf_all.agg(F.max(F.year("monthly_reporting_period") * 12
                           + F.month("monthly_reporting_period"))).first()[0]
    log.info("Last observed month index: %d", g)
    perf = _read(spark, a.parquet, "performance", a.sources)
    orig = _read(spark, a.parquet, "origination", a.sources)
    macro = spark.read.parquet(a.macro) if a.macro else None

    panel = build_panel_df(perf, orig, macro, g)
    (panel.write.mode("overwrite").option("compression", "zstd")
          .partitionBy("source").parquet(a.out))
    chk = (spark.read.parquet(a.out)
           .agg(F.count("*").alias("rows"), F.count("y_12m").alias("labeled"),
                F.sum("y_12m").alias("pos")).first())
    log.info("rows=%d labeled=%d positive_rate=%.3f%%", chk.rows, chk.labeled,
             100 * (chk.pos or 0) / max(chk.labeled, 1))
    spark.stop()


if __name__ == "__main__":
    main()
