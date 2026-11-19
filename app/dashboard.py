"""Streamlit dashboard over the pipeline outputs.

Run:  streamlit run app/dashboard.py            (uses data/ if present, else the small snapshot in app/demo_data/)
      streamlit run app/dashboard.py -- --outputs data/outputs --models data/models
The first five tabs read local files only. The "Live Atlas" tab needs MONGODB_URI (a read-only user).
app/demo_data/ holds a copy of the small result files so the hosted version works without the full data.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st


def args():
    ap = argparse.ArgumentParser()
    here = Path(__file__).parent / "demo_data"
    have = Path("data/outputs").exists()
    ap.add_argument("--outputs", default="data/outputs" if have else str(here / "outputs"))
    ap.add_argument("--models", default="data/models" if have else str(here / "models"))
    return ap.parse_known_args()[0]


A = args()
OUT, MOD = Path(A.outputs), Path(A.models)


def load_csv(p: Path):
    return pd.read_csv(p) if p.exists() else None


st.set_page_config(page_title="Mortgage Early-Warning", layout="wide")
st.title("Mortgage Credit Early-Warning")
st.caption("Freddie Mac loan-level history + FRED macro + CFPB complaints. "
           "Q1 risk ranking, Q2 complaint lead-lag, Q3 scenarios.")

t1, t2, t3, t4, t5, t6 = st.tabs(["Risk model", "State delinquency", "Complaint lead-lag",
                                  "Scenarios", "Complaint themes", "Live Atlas"])

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

with t6:
    st.subheader("Live queries against MongoDB Atlas")
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        st.info("Set MONGODB_URI in the shell that starts Streamlit (a read-only user is enough) "
                "to enable live Atlas Search and the scored-loan lookup.")
    else:
        from pymongo import MongoClient
        db = MongoClient(uri, serverSelectionTimeoutMS=8000)[os.environ.get("MONGODB_DB", "mew")]
        st.markdown("**Full-text search over CFPB complaint narratives (Atlas Search)**")
        c1, c2, c3 = st.columns([3, 1, 2])
        query = c1.text_input("Search text", "escrow payment increased")
        state = c2.text_input("State (optional, e.g. TX)", "").strip().upper()
        theme = c3.selectbox("Claude theme (optional)", [""] + [
            "payment_hardship", "servicing_error", "escrow", "modification_denial",
            "foreclosure", "origination_other"])
        if query:
            text = {"text": {"query": query, "path": "narrative"}}
            filt = []
            if state:
                filt.append({"equals": {"path": "state", "value": state}})
            if theme:
                filt.append({"equals": {"path": "theme", "value": theme}})
            op = {"compound": {"must": [text], "filter": filt}} if filt else text
            t0 = time.perf_counter()
            hits = list(db["complaints"].aggregate([
                {"$search": {"index": "complaints_search", **op}}, {"$limit": 10},
                {"$project": {"issue": 1, "state": 1, "theme": 1, "narrative": 1,
                              "score": {"$meta": "searchScore"}}}]))
            ms = (time.perf_counter() - t0) * 1000
            st.caption(f"{len(hits)} results in {ms:.0f} ms (includes network to Atlas)")
            if hits:
                df = pd.DataFrame(hits).drop(columns=["_id"])
                df["narrative"] = df["narrative"].fillna("").str.slice(0, 300)
                st.dataframe(df, use_container_width=True)
        st.markdown("**Riskiest scored loans (June 2026 snapshot)**")
        s = st.text_input("Property state", "IL").strip().upper()
        rows = list(db["scored_loans"].find({"property_state": s}, {"_id": 1, "pd_12m": 1, "upb": 1,
                    "credit_score": 1, "ltv": 1, "dti": 1, "dq_months": 1, "loan_age": 1})
                    .sort("pd_12m", -1).limit(15))
        if rows:
            st.dataframe(pd.DataFrame(rows).rename(columns={"_id": "loan_id"}).round(4),
                         use_container_width=True)
        st.caption("Scores come from the calibrated LightGBM model (probability of 90+ days late within 12 months).")
