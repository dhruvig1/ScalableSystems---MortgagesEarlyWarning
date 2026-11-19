"""Question 3: how much does portfolio delinquency change if unemployment or rates rise?

Usage:
    python -m src.scenarios.monte_carlo --snapshot data/outputs/portfolio_snapshot.parquet \
        --models data/models --out data/outputs --n-sims 200 --lgd 0.35

For each scenario we draw n_sims macro shocks, re-score every loan in the current portfolio
with the trained model, and report the distribution of expected 12-month serious-delinquency
rate and expected loss.
  * national unemployment shock du ~ N(du_mean, du_sd) percentage points
  * state shock = du + N(0, state_sd): states move together but not identically
  * mortgage-rate shock dr ~ N(dr_mean, dr_sd) percentage points
Shocks change: unemp_rate, unemp_chg_12m (by du), mortgage_rate (by dr), rate_gap (follows).
Expected loss = sum(PD * UPB * LGD). LGD is an ASSUMPTION (flag --lgd), not estimated here.

Caveat to state in the talk: the model learned macro sensitivity from history. The scenario
is "what the model predicts if inputs look like this", not a causal estimate. It is only as good
as the macro features' predictive signal (see the no-macro ablation in train.py).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.score import load_model, predict

log = logging.getLogger("scenarios")

SCENARIOS = {  # du_mean, du_sd, dr_mean, dr_sd  (percentage points)
    "baseline": (0.0, 0.3, 0.0, 0.15),
    "unemployment_+2pp": (2.0, 0.5, 0.0, 0.15),
    "rates_+200bp": (0.0, 0.3, 2.0, 0.40),
    "stagflation": (2.0, 0.5, 2.0, 0.40),
    "severe_recession": (4.0, 1.0, -1.0, 0.40),
}


def apply_shock(df: pd.DataFrame, du_by_state: dict, dr: float) -> pd.DataFrame:
    x = df.copy()
    du = x["property_state"].astype(str).map(du_by_state).fillna(0.0).to_numpy()
    for c in ("unemp_rate", "unemp_chg_12m"):
        if c in x:
            x[c] = x[c] + du
    if "mortgage_rate" in x:
        x["mortgage_rate"] = x["mortgage_rate"] + dr
        if "rate_gap" in x:
            x["rate_gap"] = x["current_interest_rate"] - x["mortgage_rate"]
    return x


def simulate(df, booster, iso, feats, scenario, n_sims, lgd, state_sd=0.4, seed=0) -> pd.DataFrame:
    du_m, du_sd, dr_m, dr_sd = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    states = sorted(df["property_state"].astype(str).unique())
    upb = df["upb"].fillna(0).to_numpy()
    rows = []
    for s in range(n_sims):
        du = rng.normal(du_m, du_sd)
        dr = rng.normal(dr_m, dr_sd)
        du_state = {st: du + rng.normal(0, state_sd) for st in states}
        pd_ = predict(booster, iso, apply_shock(df, du_state, dr), feats)
        rows.append({"scenario": scenario, "sim": s, "du": du, "dr": dr,
                     "exp_dq_rate": float(pd_.mean()),
                     "exp_dq_rate_upb_w": float(np.average(pd_, weights=upb + 1e-9)),
                     "exp_loss": float((pd_ * upb * lgd).sum()),
                     "upb_total": float(upb.sum())})
    return pd.DataFrame(rows)


def summarize(sims: pd.DataFrame) -> pd.DataFrame:
    base = sims[sims.scenario == "baseline"]["exp_dq_rate"].median() if (
        sims.scenario == "baseline").any() else np.nan
    out = []
    for sc, g in sims.groupby("scenario", sort=False):
        q = g.exp_dq_rate.quantile([0.05, 0.5, 0.95])
        out.append({"scenario": sc, "n_sims": len(g),
                    "dq_rate_p05": q[0.05], "dq_rate_p50": q[0.5], "dq_rate_p95": q[0.95],
                    "dq_rate_change_vs_baseline": g.exp_dq_rate.median() - base,
                    "loss_p05": g.exp_loss.quantile(0.05), "loss_p50": g.exp_loss.median(),
                    "loss_p95": g.exp_loss.quantile(0.95)})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--models", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-sims", type=int, default=200)
    ap.add_argument("--max-loans", type=int, default=100_000)
    ap.add_argument("--lgd", type=float, default=0.35)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    booster, iso, feats = load_model(a.models)
    if "unemp_rate" not in feats["num"]:
        raise SystemExit("Model has no macro features; rebuild the panel with spark_panel.py "
                         "--macro ... and retrain.")
    df = pd.read_parquet(a.snapshot).head(a.max_loans)
    for c in feats["cat"]:
        df[c] = df[c].astype("category")
    sims = pd.concat([simulate(df, booster, iso, feats, sc, a.n_sims, a.lgd)
                      for sc in SCENARIOS], ignore_index=True)
    a.out.mkdir(parents=True, exist_ok=True)
    sims.to_parquet(a.out / "scenario_sims.parquet", index=False)
    summ = summarize(sims)
    summ.to_csv(a.out / "scenario_summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
