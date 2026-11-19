"""Streamlit dashboard over the pipeline outputs.

Run:  streamlit run app/dashboard.py -- --outputs data/outputs --models data/models
Reads only local files written by the pipeline (no keys needed), so it works offline for the demo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import streamlit as st


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="data/outputs")
    ap.add_argument("--models", default="data/models")
    return ap.parse_known_args()[0]


A = args()
OUT, MOD = Path(A.outputs), Path(A.models)


def load_csv(p: Path):
    return pd.read_csv(p) if p.exists() else None


st.set_page_config(page_title="Mortgage Early-Warning", layout="wide")
st.title("Mortgage Credit Early-Warning")
st.caption("Freddie Mac loan-level history + FRED macro + CFPB complaints. "
           "Q1 risk ranking, Q2 complaint lead-lag, Q3 scenarios.")

t1, t2, t3, t4, t5 = st.tabs(["Risk model", "State delinquency", "Complaint lead-lag",
                              "Scenarios", "Complaint themes"])

with t1:
    mp = MOD / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        rows = [{"model": k, **{x: v[x] for x in ("auc", "avg_precision", "brier", "ece",
                                                  "base_rate", "mean_pred")}}
                for k, v in m.items() if isinstance(v, dict) and "auc" in v]
        st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True)
        c1, c2 = st.columns(2)
        cal = load_csv(MOD / "calibration_lgbm_isotonic.csv")
        if cal is not None:
            c1.subheader("Calibration (test period)")
            c1.line_chart(cal.rename(columns={"p": "predicted", "y": "observed"})
                          .set_index("predicted")[["observed"]])
            c1.caption("A well calibrated model follows observed = predicted.")
        fi = load_csv(MOD / "feature_importance_lgbm.csv")
        if fi is not None:
            c2.subheader("What drives risk")
            c2.bar_chart(fi.head(15).set_index("feature")["gain"])
        g = load_csv(MOD / "groups_lgbm_isotonic.csv")
        if g is not None:
            st.subheader("Does it work for every borrower group?")
            grp = st.selectbox("Group", sorted(g.group.unique()))
            st.dataframe(g[g.group == grp].round(4), use_container_width=True)
    else:
        st.info("Run src.models.train to create metrics.json")

with t2:
    r = OUT / "state_month_rollup.parquet"
    if r.exists():
        d = pd.read_parquet(r)
        d["month"] = pd.to_datetime(d["month"])
        d["rate_pct"] = 100 * d.n_90 / d.n_active
        states = st.multiselect("States", sorted(d.state.unique()), default=["IL", "CA", "TX", "FL"])
        p = d[d.state.isin(states)].pivot(index="month", columns="state", values="rate_pct")
        st.line_chart(p)
        chk = OUT / "state_quarter_check.json"
        if chk.exists():
            st.subheader("Same number computed more than one way")
            st.dataframe(pd.DataFrame(json.loads(chk.read_text())), use_container_width=True)
    else:
        st.info("Run src.queries.state_quarter")

with t3:
    ll = load_csv(OUT / "complaints_lead_lag.csv")
    if ll is not None:
        st.write("Effect of a one-standard-deviation complaint surge in month t on the change in "
                 "the state's 90+ day delinquency rate (percentage points) from t to t+h.")
        for s, g in ll.groupby("series"):
            st.subheader(s)
            st.dataframe(g.round(4), use_container_width=True)
            st.bar_chart(g.set_index("h")["coef"])
        st.caption("Lead-lag, not causation. Placebo columns show what chance alone produces.")
    else:
        st.info("Run src.analysis.complaints_lead_lag")

with t4:
    sc = load_csv(OUT / "scenario_summary.csv")
    if sc is not None:
        st.dataframe(sc.round(4), use_container_width=True)
        st.bar_chart(sc.set_index("scenario")[["dq_rate_p05", "dq_rate_p50", "dq_rate_p95"]])
        st.caption("Range = 5th to 95th percentile across Monte Carlo draws. Loss assumes the "
                   "LGD set when the engine was run.")
    else:
        st.info("Run src.scenarios.monte_carlo")

with t5:
    lab = OUT.parent / "labels" / "claude_labels.csv"
    if lab.exists():
        l = pd.read_csv(lab)
        st.bar_chart(l.theme.value_counts())
        for f, title in (("label_check_cfpb.json", "vs CFPB issue mapping"),
                         ("label_check_hand.json", "vs hand labels")):
            p = OUT / f
            if p.exists():
                j = json.loads(p.read_text())
                st.metric(f"Claude accuracy {title}", f"{j['accuracy']:.1%}", f"kappa {j['kappa']:.2f}")
    else:
        st.info("Run src.ai.label_complaints")
    ei = OUT / "eda" / "complaints_monthly.png"
    if ei.exists():
        st.image(str(ei))
