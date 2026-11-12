"""Synthetic panel with a known signal: both models must beat chance; embargo enforced."""
from argparse import Namespace

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.models.train import CAT, NUM, run


def _panel(tmp_path):
    rng = np.random.default_rng(0)
    n_loans, per = 4000, 12
    rows = []
    months = pd.date_range("2019-01-01", "2025-03-01", freq="MS")
    for i in range(n_loans):
        score = rng.normal(720, 50)
        ltv = rng.uniform(40, 97)
        dqmax = rng.choice([0, 0, 0, 1, 2], p=[.5, .2, .1, .1, .1])
        for m in rng.choice(months, per, replace=False):
            z = -4 + 0.9 * dqmax - 0.012 * (score - 720) + 0.02 * (ltv - 75)
            y = int(rng.random() < 1 / (1 + np.exp(-z)))
            rows.append({"loan_id": f"F19Q1{i:07d}", "rep_month": m, "y_12m": y,
                         "loan_age": rng.integers(1, 80), "dq_months": 0, "dq_lag1": 0,
                         "dq_lag3": 0, "dq_max_6m": dqmax, "dq_max_12m": dqmax,
                         "n_late_12m": dqmax, "upb_ratio": 0.9, "current_interest_rate": 4.5,
                         "rate_chg": 0.0, "estimated_ltv": ltv, "modified": 0,
                         "credit_score": score, "mi_pct": 0, "num_units": 1, "cltv": ltv,
                         "dti": rng.uniform(10, 50), "orig_upb": 200000, "ltv": ltv,
                         "orig_interest_rate": 4.5, "orig_loan_term": 360, "num_borrowers": 2,
                         "first_time_homebuyer_flag": "N", "occupancy_status": "P",
                         "channel": "R", "property_state": rng.choice(["IL", "CA", "TX"]),
                         "property_type": "SF", "loan_purpose": "P"})
    df = pd.DataFrame(rows)
    assert set(NUM + CAT) <= set(df.columns)
    d = tmp_path / "panel" / "source=T"
    d.mkdir(parents=True)
    duckdb.connect().execute(f"COPY (SELECT * FROM df) TO '{d}/p.parquet' (FORMAT PARQUET)")
    return tmp_path / "panel"


def _args(panel, out, **kw):
    base = dict(panel=str(panel), out=str(out), split="time", train_end="2021-12",
                test_start="2023-02", train_vintage_max=2017, test_vintage_min=2019,
                sample_frac=1.0, max_logit_rows=200000, threads=1)
    base.update(kw)
    return Namespace(**base)


def test_models_beat_chance(tmp_path):
    panel = _panel(tmp_path)
    res = run(_args(panel, tmp_path / "out"))
    assert res["logistic"]["auc"] > 0.62
    assert res["lgbm"]["auc"] > 0.62
    assert (tmp_path / "out" / "metrics.json").exists()
    assert (tmp_path / "out" / "calibration_lgbm.csv").exists()


def test_embargo_enforced(tmp_path):
    panel = _panel(tmp_path)
    with pytest.raises(ValueError):
        run(_args(panel, tmp_path / "out2", test_start="2022-06"))  # < train_end + 12 months


def test_ablation_flags(tmp_path):
    panel = _panel(tmp_path)
    res = run(_args(panel, tmp_path / "o3", current_only=True, no_history=True, disjoint=True))
    assert res["lgbm"]["auc"] > 0.55  # credit score / LTV still carry signal
