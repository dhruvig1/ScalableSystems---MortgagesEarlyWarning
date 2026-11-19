"""Question 2: do complaint surges in a state show up BEFORE its delinquency rate rises?

Usage:
    python -m src.analysis.complaints_lead_lag \
        --rollup data/outputs/state_month_rollup.parquet \
        --complaints data/parquet/cfpb/mortgage_complaints.parquet --out data/outputs

Design
  * complaint surge_t = z-score of log(1 + monthly complaints) against that state's own trailing
    12 months (months t-12..t-1), so it only uses information known at t and removes state size.
  * outcome: change in the state's 90+ day delinquency rate from t to t+h, h in 0..12 months
    (percentage points).
  * model, per horizon:  d_rate(t->t+h) = b * surge_t + c * rate_t + e * past_change_t
                                           + state fixed effects + month fixed effects
    with standard errors clustered by state. `past_change` (rate_t - rate_{t-3}) controls for
    momentum, so b measures what complaints add beyond delinquency already visible.
  * placebo: shuffle the surge series across states within each month and re-fit. A real
    effect should be much larger than the placebo coefficients.
  * two complaint series: all mortgage complaints, and "hardship" complaints (CFPB Issue
    mentioning struggling to pay), which needs no AI labeling.
Caveats: complaints are for all lenders, delinquency is Freddie Mac loans only; the join is at
state-month level. A significant coefficient is evidence of lead-lag, not of causation.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

log = logging.getLogger("leadlag")


def complaint_surge(counts: pd.DataFrame, trailing: int = 12) -> pd.DataFrame:
    """counts: state, month, n. Returns state, month, surge (z of log1p vs own trailing window)."""
    out = []
    for st, g in counts.groupby("state"):
        g = g.set_index("month").sort_index()
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="MS"), fill_value=0)
        x = np.log1p(g["n"])
        mu = x.shift(1).rolling(trailing, min_periods=trailing).mean()
        sd = x.shift(1).rolling(trailing, min_periods=trailing).std().clip(lower=0.25)
        out.append(pd.DataFrame({"state": st, "month": g.index, "surge": (x - mu) / sd}))
    return pd.concat(out, ignore_index=True)


def build_panel(rollup: pd.DataFrame, surge: pd.DataFrame, horizons=range(0, 13)) -> pd.DataFrame:
    r = rollup.copy()
    r["month"] = pd.to_datetime(r["month"])
    r = r[r.n_active >= 200].sort_values(["state", "month"])  # ignore tiny state-months
    r["rate"] = 100 * r.n_90 / r.n_active
    g = r.groupby("state")["rate"]
    r["past_change"] = r["rate"] - g.shift(3)
    for h in horizons:
        r[f"d{h}"] = g.shift(-h) - r["rate"]
    return r.merge(surge, on=["state", "month"], how="inner").dropna(subset=["surge", "past_change"])


def fit_horizon(df: pd.DataFrame, h: int) -> dict:
    d = df.dropna(subset=[f"d{h}"]).copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[f"d{h}", "surge", "rate", "past_change"])
    d["mth"] = d["month"].dt.strftime("%Y-%m")
    m = smf.ols(f"d{h} ~ surge + rate + past_change + C(state) + C(mth)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["state"])[0]})
    return {"h": h, "coef": float(m.params["surge"]), "se": float(m.bse["surge"]),
            "p": float(m.pvalues["surge"]), "n": int(m.nobs)}


def placebo(df: pd.DataFrame, h: int, reps: int = 20, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    coefs = []
    for _ in range(reps):
        d = df.copy()
        d["surge"] = d.groupby("month")["surge"].transform(lambda s: rng.permutation(s.values))
        coefs.append(fit_horizon(d, h)["coef"])
    return coefs


def run(rollup: pd.DataFrame, complaints: pd.DataFrame, horizons=(1, 3, 6, 9, 12),
        placebo_reps: int = 20) -> pd.DataFrame:
    c = complaints.copy()
    c["date_received"] = pd.to_datetime(c["date_received"], errors="coerce")
    c = c.dropna(subset=["date_received", "state"])
    c["month"] = c["date_received"].dt.to_period("M").dt.to_timestamp()
    series = {
        "all_mortgage": c,
        "hardship": c[c["issue"].fillna("").str.lower().str.contains("struggling to pay")],
    }
    rows = []
    for name, sub in series.items():
        counts = sub.groupby(["state", "month"]).size().rename("n").reset_index()
        panel = build_panel(rollup, complaint_surge(counts), horizons)
        for h in horizons:
            try:
                res = fit_horizon(panel, h)
                pl = placebo(panel, h, placebo_reps)
            except np.linalg.LinAlgError as e:
                log.warning("skip %s h=%d: %s", name, h, e)
                continue
            res.update({"series": name, "placebo_abs_mean": float(np.mean(np.abs(pl))),
                        "placebo_abs_max": float(np.max(np.abs(pl))),
                        "beats_placebo": bool(abs(res["coef"]) > np.max(np.abs(pl)))})
            rows.append(res)
            log.info("%s h=%2d coef=%+.4f p=%.3f placebo_max=%.4f", name, h, res["coef"],
                     res["p"], res["placebo_abs_max"])
    return pd.DataFrame(rows)[["series", "h", "coef", "se", "p", "n", "placebo_abs_mean",
                               "placebo_abs_max", "beats_placebo"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollup", required=True, type=Path)
    ap.add_argument("--complaints", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--placebo-reps", type=int, default=20)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    res = run(pd.read_parquet(a.rollup),
              pd.read_parquet(a.complaints, columns=["date_received", "state", "issue"]),
              placebo_reps=a.placebo_reps)
    a.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out / "complaints_lead_lag.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
