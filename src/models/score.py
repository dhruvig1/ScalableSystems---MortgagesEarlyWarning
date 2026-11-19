"""Score the current portfolio: every at-risk loan in the latest reporting month.

Usage:
    python -m src.models.score --panel data/panel_spark --models data/models \
        --out data/outputs/portfolio_snapshot.parquet --max-loans 500000

Output columns: loan_id, rep_month, property_state, upb, pd_12m (calibrated probability of
90+ days late within 12 months) plus every model feature, so the scenario engine can re-score
the same loans under shocked macro inputs without re-reading the panel.
Rows with a known label (y_12m not null) are NOT scored here; the latest month has none.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger("score")


def load_model(models: Path):
    booster = lgb.Booster(model_file=str(models / "lgbm.txt"))
    iso = joblib.load(models / "isotonic.joblib")
    feats = json.loads((models / "features.json").read_text())
    return booster, iso, feats


def predict(booster, iso, df: pd.DataFrame, feats: dict) -> np.ndarray:
    X = df[feats["num"] + feats["cat"]]
    return np.clip(iso.predict(booster.predict(X)), 1e-6, 1 - 1e-6)


def latest_snapshot(panel: Path, feats: dict, max_loans: int) -> pd.DataFrame:
    cols = ", ".join(sorted(set(feats["num"] + feats["cat"] + ["property_state", "orig_upb"])))
    q = f"""
      WITH last AS (SELECT max(rep_month) AS m FROM read_parquet('{panel}/*/*.parquet'))
      SELECT loan_id, rep_month, upb, {cols}
      FROM read_parquet('{panel}/*/*.parquet'), last
      WHERE rep_month = last.m
      ORDER BY hash(loan_id)
      LIMIT {int(max_loans)}"""  # deterministic pseudo-random subset of the current book
    df = duckdb.connect().execute(q).df()
    for c in feats["num"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in feats["cat"]:
        df[c] = df[c].astype("category")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, type=Path)
    ap.add_argument("--models", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-loans", type=int, default=500_000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    booster, iso, feats = load_model(a.models)
    df = latest_snapshot(a.panel, feats, a.max_loans)
    df["pd_12m"] = predict(booster, iso, df, feats)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    log.info("Scored %d loans (month %s). Mean PD %.4f, UPB-weighted PD %.4f",
             len(df), df.rep_month.max(), df.pd_12m.mean(),
             float(np.average(df.pd_12m, weights=df.upb.fillna(0) + 1e-9)))


if __name__ == "__main__":
    main()
