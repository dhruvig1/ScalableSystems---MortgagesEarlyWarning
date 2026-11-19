"""Shared synthetic Freddie-style Parquet so every stage can be tested without real data."""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_freddie(root: Path, n_loans: int = 400, months: int = 40, seed: int = 1) -> Path:
    rng = np.random.default_rng(seed)
    states = np.array(["IL", "CA", "TX", "FL"])
    orig_rows, perf_rows = [], []
    for i in range(n_loans):
        lid = f"F19Q1{i:07d}"
        st = states[i % 4]
        score = int(rng.normal(720, 50))
        ltv = int(rng.uniform(40, 97))
        risky = rng.random() < 0.25
        onset = int(rng.integers(5, 36))  # staggered so positives exist in every period
        orig_rows.append({
            "loan_sequence_number": lid, "property_state": st, "credit_score": score,
            "first_payment_date": pd.Timestamp("2019-01-01"), "dti": int(rng.uniform(10, 50)),
            "ltv": ltv, "cltv": ltv, "orig_upb": 200000.0, "orig_interest_rate": 4.5,
            "orig_loan_term": 360, "mi_pct": 0, "num_units": 1, "num_borrowers": 2,
            "occupancy_status": "P", "channel": "R", "property_type": "SF", "loan_purpose": "P",
            "first_time_homebuyer_flag": "N", "postal_code": "60600", "msa": "16980"})
        dq, dead = 0, False
        for t in range(months):
            if dead:
                break
            if risky and t > onset:
                dq = min(dq + 1, 4) if rng.random() < 0.5 else dq
            zb = None
            upb = 200000.0 - 100 * t
            if rng.random() < 0.01:
                zb, upb, dead = "01", 0.0, True
            perf_rows.append({
                "loan_sequence_number": lid,
                "monthly_reporting_period": pd.Timestamp("2019-01-01") + pd.DateOffset(months=t),
                "current_actual_upb": upb, "dq_months": dq, "loan_age": t + 1,
                "zero_balance_code": zb, "modification_flag": "N", "estimated_ltv": ltv,
                "current_interest_rate": 4.5})
    for kind, rows in (("origination", orig_rows), ("performance", perf_rows)):
        d = root / kind / "source=2019"
        d.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        for c in df.columns:  # real loader writes DATE, not nanosecond timestamps
            if str(df[c].dtype).startswith("datetime64"):
                df[c] = df[c].dt.date
        duckdb.connect().execute(f"COPY (SELECT * FROM df) TO '{d}/p.parquet' (FORMAT PARQUET)")
    return root


@pytest.fixture(scope="session")
def freddie(tmp_path_factory):
    return make_freddie(tmp_path_factory.mktemp("freddie"))
