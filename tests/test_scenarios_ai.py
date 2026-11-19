"""Scenario engine and Claude-labeling plumbing (no network, no real model calls)."""
import json
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from src.ai.label_complaints import classify, parse_reply, run as label_run
from src.ai.themes import cfpb_theme
from src.ai.validate_labels import vs_cfpb, vs_hand
from src.scenarios.monte_carlo import SCENARIOS, apply_shock, simulate, summarize


def _toy():
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({
        "property_state": rng.choice(["IL", "CA"], n),
        "unemp_rate": rng.uniform(3, 8, n), "unemp_chg_12m": rng.normal(0, 1, n),
        "mortgage_rate": rng.uniform(3, 7, n), "current_interest_rate": rng.uniform(3, 7, n),
        "credit_score": rng.normal(700, 50, n), "upb": rng.uniform(1e5, 3e5, n)})
    df["rate_gap"] = df.current_interest_rate - df.mortgage_rate
    df["property_state"] = df["property_state"].astype("category")
    p = 1 / (1 + np.exp(-(-4 + 0.5 * (df.unemp_rate - 5) - 0.01 * (df.credit_score - 700))))
    y = (rng.random(n) < p).astype(int)
    feats = {"num": ["unemp_rate", "unemp_chg_12m", "mortgage_rate", "rate_gap", "credit_score"],
             "cat": ["property_state"], "macro": ["unemp_rate"]}
    m = lgb.LGBMClassifier(n_estimators=60, verbose=-1).fit(df[feats["num"] + feats["cat"]], y)
    raw = m.predict_proba(df[feats["num"] + feats["cat"]])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
    return df, m.booster_, iso, feats


def test_apply_shock_changes_only_macro():
    df, *_ = _toy()
    x = apply_shock(df, {"IL": 2.0, "CA": 1.0}, 1.5)
    il = df.property_state == "IL"
    assert np.allclose((x.unemp_rate - df.unemp_rate)[il], 2.0)
    assert np.allclose(x.mortgage_rate - df.mortgage_rate, 1.5)
    assert np.allclose(x.rate_gap, x.current_interest_rate - x.mortgage_rate)
    assert (x.credit_score == df.credit_score).all()


def test_higher_unemployment_raises_expected_delinquency():
    df, booster, iso, feats = _toy()
    sims = pd.concat([simulate(df, booster, iso, feats, sc, 10, 0.35)
                      for sc in ("baseline", "unemployment_+2pp")], ignore_index=True)
    s = summarize(sims).set_index("scenario")
    assert s.loc["unemployment_+2pp", "dq_rate_p50"] > s.loc["baseline", "dq_rate_p50"]
    assert (s.dq_rate_p05 <= s.dq_rate_p50).all() and (s.dq_rate_p50 <= s.dq_rate_p95).all()
    assert set(SCENARIOS) >= {"baseline", "stagflation"}


def test_parse_reply():
    assert parse_reply('{"theme": "escrow", "confidence": 0.9}') == ("escrow", 0.9)
    assert parse_reply('Sure: {"theme":"payment_hardship","confidence":0.7} done')[0] == "payment_hardship"
    with pytest.raises(ValueError):
        parse_reply('{"theme": "banana", "confidence": 1}')


class FakeClient:
    """Mimics anthropic.Anthropic().messages.create."""
    def __init__(self):
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls += 1
        text = kw["messages"][0]["content"].lower()
        theme = "escrow" if "escrow" in text else "payment_hardship"
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps({"theme": theme, "confidence": 0.8}))])


def test_labeling_is_resumable_and_capped(tmp_path):
    rows = [{"complaint_id": str(i), "issue": "Struggling to pay mortgage" if i % 2 else "Trouble during payment process",
             "sub_issue": None, "state": "IL", "date_received": "2020-05-01",
             "narrative": ("My escrow payment doubled. " if i % 2 == 0 else "I lost my job and cannot pay. ") * 3}
            for i in range(30)]
    p = tmp_path / "c.parquet"
    pd.DataFrame(rows).to_parquet(p)
    fc = FakeClient()
    assert label_run(p, tmp_path / "lab", 10, "m", False, client=fc) == 10
    assert fc.calls == 10                                   # cap respected
    assert label_run(p, tmp_path / "lab", 10, "m", False, client=fc) == 0
    assert fc.calls == 10                                   # rerun pays nothing
    assert label_run(p, tmp_path / "lab", 20, "m", False, client=fc) == 10  # extends by the difference
    rep, conf = vs_cfpb(p, tmp_path / "lab" / "claude_labels.csv")
    assert rep["n"] == 20 and 0 <= rep["accuracy"] <= 1


def test_hand_validation(tmp_path):
    lab = pd.DataFrame({"complaint_id": ["1", "2", "3"], "theme": ["escrow", "escrow", "foreclosure"],
                        "confidence": 0.9, "model": "m"})
    hand = pd.DataFrame({"complaint_id": ["1", "2", "3"],
                         "human_theme": ["escrow", "payment_hardship", "foreclosure"]})
    lab.to_csv(tmp_path / "l.csv", index=False)
    hand.to_csv(tmp_path / "h.csv", index=False)
    r = vs_hand(tmp_path / "l.csv", tmp_path / "h.csv")
    assert abs(r["accuracy"] - 2 / 3) < 1e-9


def test_cfpb_mapping():
    assert cfpb_theme("Struggling to pay your mortgage") == "payment_hardship"
    assert cfpb_theme("Trouble during payment process") == "servicing_error"
    assert cfpb_theme("Applying for a mortgage or refinancing an existing mortgage") == "origination_other"
    assert cfpb_theme("Dealing with my lender or servicer", "Problem with escrow") == "escrow"
