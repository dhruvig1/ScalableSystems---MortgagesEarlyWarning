"""Exploratory data analysis: shape, missing values, distributions, outliers, label behavior.

Usage:
    python -m src.eda.profile --parquet data/parquet --cfpb data/parquet/cfpb/mortgage_complaints.parquet \
        --panel data/panel_spark --out data/outputs/eda [--sources 2019]

Writes CSV tables and PNG charts to --out. Each table answers a design question:
  shape.csv              size of each source -> justifies Spark / Parquet
  orig_missing.csv       share of loans with an unknown (coded -> NULL) value per field
  orig_percentiles.csv   p1/p50/p99 and IQR-rule outlier counts for numeric fields
  dq_status.csv          how delinquency status is distributed (rare-event check)
  label_by_year.csv      12-month default rate by reporting year (2008/2020 regime shifts)
  vintage_curves.csv     cumulative share of loans ever 90+ days late by loan age and vintage
  complaints_monthly.csv volume and narrative coverage by month
  complaints_issues.csv  top CFPB issue categories
Charts are saved next to the tables.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("eda")

ORIG_FIELDS = ["credit_score", "dti", "ltv", "cltv", "mi_pct", "num_units", "num_borrowers",
               "occupancy_status", "channel", "property_type", "loan_purpose",
               "first_time_homebuyer_flag", "property_state", "postal_code", "msa"]
NUMERIC = ["credit_score", "dti", "ltv", "cltv", "orig_upb", "orig_interest_rate", "orig_loan_term"]


def rel(root: Path, kind: str, sources) -> str:
    if sources:
        return "read_parquet([" + ", ".join(
            f"'{root}/{kind}/source={s}/*.parquet'" for s in sources) + "])"
    return f"read_parquet('{root}/{kind}/*/*.parquet')"


def bar(df, x, y, path, title, rot=0):
    ax = df.plot.bar(x=x, y=y, legend=False, figsize=(8, 4), rot=rot)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def run(parquet: Path, out: Path, cfpb: Path | None = None, panel: Path | None = None,
        sources=None, threads: int = 4) -> None:
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    orig, perf = rel(parquet, "origination", sources), rel(parquet, "performance", sources)

    # ---- shape ----
    shape = con.execute(f"""
      SELECT 'origination' AS table_name, count(*) AS rows, count(DISTINCT loan_sequence_number) AS loans FROM {orig}
      UNION ALL
      SELECT 'performance', count(*), count(DISTINCT loan_sequence_number) FROM {perf}""").df()
    shape.to_csv(out / "shape.csv", index=False)
    log.info("\n%s", shape)

    # ---- missing (coded unknowns became NULL in the loader) ----
    sel = ", ".join(f"avg(({c} IS NULL)::INT) AS {c}" for c in ORIG_FIELDS)
    miss = con.execute(f"SELECT {sel} FROM {orig}").df().T.reset_index()
    miss.columns = ["field", "share_missing"]
    miss.sort_values("share_missing", ascending=False).to_csv(out / "orig_missing.csv", index=False)
    top = miss.sort_values("share_missing", ascending=False).head(10)
    bar(top, "field", "share_missing", out / "orig_missing.png",
        "Share of loans with unknown value (coded -> NULL)", rot=45)

    # ---- percentiles + IQR outliers ----
    rows = []
    for c in NUMERIC:
        r = con.execute(f"""
          WITH q AS (SELECT quantile_cont({c}, 0.25) q1, quantile_cont({c}, 0.75) q3,
                            quantile_cont({c}, 0.01) p1, quantile_cont({c}, 0.5) p50,
                            quantile_cont({c}, 0.99) p99, min({c}) mn, max({c}) mx
                     FROM {orig})
          SELECT '{c}', p1, p50, p99, mn, mx, q1, q3,
                 (SELECT count(*) FROM {orig}, q WHERE {c} < q1 - 1.5 * (q3 - q1)
                                                  OR {c} > q3 + 1.5 * (q3 - q1)) AS n_outliers_iqr
          FROM q""").fetchone()
        rows.append(r)
    import pandas as pd
    pct = pd.DataFrame(rows, columns=["field", "p1", "p50", "p99", "min", "max", "q1", "q3",
                                      "n_outliers_iqr"])
    pct.to_csv(out / "orig_percentiles.csv", index=False)

    # ---- distributions (histograms) ----
    for c, lo, hi, step in [("credit_score", 300, 850, 10), ("dti", 0, 70, 2), ("ltv", 0, 110, 3)]:
        h = con.execute(f"""
          SELECT floor({c}/{step})*{step} AS b, count(*) AS n FROM {orig}
          WHERE {c} BETWEEN {lo} AND {hi} GROUP BY 1 ORDER BY 1""").df()
        h.to_csv(out / f"hist_{c}.csv", index=False)
        bar(h, "b", "n", out / f"hist_{c}.png", f"Distribution of {c}", rot=90)

    # ---- delinquency status (rare-event check) ----
    dq = con.execute(f"""
      SELECT CASE WHEN dq_months IS NULL THEN 'unknown' WHEN dq_months >= 6 THEN '6+'
                  ELSE CAST(dq_months AS VARCHAR) END AS dq_status, count(*) AS rows
      FROM {perf} GROUP BY 1 ORDER BY 1""").df()
    dq["share"] = dq.rows / dq.rows.sum()
    dq.to_csv(out / "dq_status.csv", index=False)

    # ---- vintage curves: cumulative share ever 90+ by loan age ----
    vc = con.execute(f"""
      WITH first90 AS (
        SELECT loan_sequence_number AS loan_id, min(loan_age) AS age90
        FROM {perf} WHERE dq_months >= 3 GROUP BY 1),
      loans AS (SELECT loan_sequence_number AS loan_id,
                       year(first_payment_date) AS vintage FROM {orig}),
      ages AS (SELECT unnest(range(12, 121, 12)) AS age)
      SELECT l.vintage, a.age,
             count(*) FILTER (WHERE f.age90 <= a.age) * 1.0 / count(*) AS cum_share_90
      FROM loans l CROSS JOIN ages a LEFT JOIN first90 f USING (loan_id)
      GROUP BY 1, 2 ORDER BY 1, 2""").df()
    vc.to_csv(out / "vintage_curves.csv", index=False)
    if len(vc):
        piv = vc.pivot(index="age", columns="vintage", values="cum_share_90")
        ax = piv.plot(figsize=(8, 4.5), marker="o")
        ax.set_title("Cumulative share of loans ever 90+ days late, by vintage")
        ax.set_xlabel("loan age (months)")
        plt.tight_layout()
        plt.savefig(out / "vintage_curves.png", dpi=140)
        plt.close()

    # ---- label behavior by reporting year (needs the panel) ----
    if panel and Path(panel).exists():
        ly = con.execute(f"""
          SELECT year(rep_month) AS yr, count(y_12m) AS labeled, avg(y_12m) AS default_rate
          FROM read_parquet('{panel}/*/*.parquet') GROUP BY 1 ORDER BY 1""").df()
        ly.to_csv(out / "label_by_year.csv", index=False)
        bar(ly.dropna(), "yr", "default_rate", out / "label_by_year.png",
            "12-month serious-delinquency rate by reporting year", rot=90)

    # ---- complaints ----
    if cfpb and Path(cfpb).exists():
        cm = con.execute(f"""
          SELECT date_trunc('month', CAST(date_received AS DATE)) AS month, count(*) AS complaints,
                 count(*) FILTER (WHERE narrative IS NOT NULL AND length(trim(narrative)) > 0) AS with_narrative
          FROM read_parquet('{cfpb}') GROUP BY 1 ORDER BY 1""").df()
        cm.to_csv(out / "complaints_monthly.csv", index=False)
        ax = cm.plot(x="month", y=["complaints", "with_narrative"], figsize=(9, 4))
        ax.set_title("CFPB mortgage complaints per month (and with narrative)")
        plt.tight_layout()
        plt.savefig(out / "complaints_monthly.png", dpi=140)
        plt.close()
        con.execute(f"""
          COPY (SELECT issue, count(*) AS n FROM read_parquet('{cfpb}')
                GROUP BY 1 ORDER BY 2 DESC LIMIT 15) TO '{out / "complaints_issues.csv"}' (HEADER)""")
    log.info("EDA written to %s", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cfpb", type=Path, default=None)
    ap.add_argument("--panel", type=Path, default=None)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(a.parquet, a.out, a.cfpb, a.panel, a.sources, a.threads)


if __name__ == "__main__":
    main()
