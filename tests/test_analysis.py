"""Lead-lag test recovers a planted effect; scenario engine behaves; macro table has no look-ahead."""
import duckdb
import numpy as np
import pandas as pd

from src.analysis.complaints_lead_lag import build_panel, complaint_surge, fit_horizon, run


def _synthetic(lead=6, seed=0, planted=True):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2012-01-01", "2020-12-01", freq="MS")
    states = [f"S{i:02d}" for i in range(30)]
    roll, comp = [], []
    for st in states:
        shock = np.zeros(len(months))
        for _ in range(4):
            shock[rng.integers(20, len(months) - 12)] = rng.uniform(2, 4)  # complaint surge events
        n_c = np.maximum(0, (30 + 8 * shock + rng.normal(0, 2, len(months)))).round().astype(int)
        base = 2.0 + np.cumsum(rng.normal(0, 0.03, len(months)))
        if planted:
            base = base + np.concatenate([np.zeros(lead), 0.35 * np.cumsum(shock)[:-lead]])
        n_active = 5000
        for m, b, c in zip(months, base, n_c):
            roll.append({"state": st, "month": m, "n_active": n_active,
                         "n_90": int(n_active * max(b, 0) / 100)})
            comp += [{"state": st, "date_received": m, "issue": "Struggling to pay mortgage"}] * int(c)
    return pd.DataFrame(roll), pd.DataFrame(comp)


def test_surge_uses_only_past():
    counts = pd.DataFrame({"state": "A", "month": pd.date_range("2020-01-01", periods=30, freq="MS"),
                           "n": [10] * 29 + [500]})
    s = complaint_surge(counts)
    assert s.surge.iloc[:12].isna().all()
    assert s.surge.iloc[-1] > 3  # last month spikes vs its own history


def test_leadlag_detects_planted_effect():
    roll, comp = _synthetic(planted=True)
    res = run(roll, comp, horizons=(6,), placebo_reps=5)
    r = res[(res.series == "hardship")].iloc[0]
    assert r.coef > 0 and r.p < 0.05 and r.beats_placebo


def test_leadlag_null_when_no_effect():
    roll, comp = _synthetic(planted=False, seed=3)
    surge = complaint_surge(comp.groupby(["state", pd.to_datetime(comp.date_received)])
                            .size().rename("n").reset_index().rename(columns={"date_received": "month"}))
    r = fit_horizon(build_panel(roll, surge, [6]), 6)
    assert abs(r["coef"]) < 0.05


def test_macro_no_lookahead(tmp_path):
    from src.features.build_macro import build
    rows = []
    for st in ("IL", "CA"):
        for i, d in enumerate(pd.date_range("2018-01-01", "2020-12-01", freq="MS")):
            rows.append((f"{st}UR", d, float(i)))             # UR(month i) = i
        for d, v in zip(pd.date_range("2018-01-01", "2020-10-01", freq="QS"), range(12)):
            rows.append((f"{st}STHPI", d, 100.0 + v))
    for d in pd.date_range("2018-01-01", "2020-12-01", freq="MS"):
        rows.append(("MORTGAGE30US", d, 4.0))
    df = pd.DataFrame(rows, columns=["series_id", "obs_date", "value"])
    p = tmp_path / "fred.parquet"
    duckdb.connect().execute(f"COPY (SELECT * FROM df) TO '{p}' (FORMAT PARQUET)")
    out = tmp_path / "macro.parquet"
    build(p, out)
    m = duckdb.connect().execute(f"SELECT * FROM read_parquet('{out}') WHERE state='IL'").df()
    m["month"] = pd.to_datetime(m["month"])
    t = m[m.month == "2019-06-01"].iloc[0]
    # UR for May 2019 is index 16; June 2019's row must carry May's value
    assert t.unemp_rate == 16.0
    assert t.unemp_chg_12m == 12.0
    # HPI for quarter starting 2019-04 (v=5) is known from 2019-09, not before
    sep = m[m.month == "2019-09-01"].iloc[0]
    aug = m[m.month == "2019-08-01"].iloc[0]
    assert sep.hpi_yoy is not None and not np.isnan(sep.hpi_yoy)
    assert aug.hpi_yoy == m[m.month == "2019-07-01"].iloc[0].hpi_yoy  # unchanged until the next quarter arrives
