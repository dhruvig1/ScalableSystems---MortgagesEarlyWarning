"""Query cross-checks and EDA on synthetic data."""
import pandas as pd

from src.eda.profile import run as run_eda
from src.queries.loan_counts import counts, orphans
from src.queries.state_quarter import compare, duckdb_month, duckdb_quarter


def test_quarter_matches_rollup_derivation(freddie):
    """Quarter-end rows derived from the state-month rollup (the Mongo path) equal the direct query."""
    q = duckdb_quarter(freddie, ["2019"], threads=1)
    m = duckdb_month(freddie, ["2019"], threads=1)
    m["month"] = pd.to_datetime(m["month"])
    m = m[m.month.dt.month.isin([3, 6, 9, 12])]
    m["yr"], m["qtr"] = m.month.dt.year, (m.month.dt.month - 1) // 3 + 1
    derived = m.groupby(["state", "yr", "qtr"], as_index=False)[["n_active", "n_90"]].sum()
    res = compare(q, derived, "direct", "rollup")
    assert res["agree"] and res["rows_a"] == res["rows_b"] > 0


def test_compare_detects_difference(freddie):
    q = duckdb_quarter(freddie, ["2019"], threads=1)
    bad = q.copy()
    bad.loc[0, "n_90"] += 1
    assert not compare(q, bad, "a", "b")["agree"]


def test_rates_are_valid(freddie):
    q = duckdb_quarter(freddie, ["2019"], threads=1)
    assert (q.n_90 <= q.n_active).all() and (q.n_active > 0).all()


def test_loan_counts(freddie):
    c = counts(freddie)
    assert int(c.n_orig.iloc[0]) == 400
    o = orphans(freddie)
    assert o == {"perf_without_orig": 0, "orig_without_perf": 0}


def test_eda_outputs(freddie, tmp_path):
    run_eda(freddie, tmp_path, sources=["2019"], threads=1)
    for f in ["shape.csv", "orig_missing.csv", "orig_percentiles.csv", "dq_status.csv",
              "vintage_curves.csv", "hist_credit_score.png", "vintage_curves.png"]:
        assert (tmp_path / f).exists(), f
    shape = pd.read_csv(tmp_path / "shape.csv")
    assert shape.loc[shape.table_name == "origination", "loans"].iloc[0] == 400
    vc = pd.read_csv(tmp_path / "vintage_curves.csv")
    assert (vc.groupby("vintage").cum_share_90.diff().dropna() >= -1e-12).all()  # curves rise
