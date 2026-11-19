"""Train and evaluate 12-month serious-delinquency models on the loan-month panel.

Usage (time split, recommended):
    python -m src.models.train --panel data/panel --out data/models \
        --split time --train-end 2021-06 --test-start 2022-07 --sample-frac 0.1

Usage (vintage split, as in the proposal):
    python -m src.models.train --panel data/panel --out data/models \
        --split vintage --train-vintage-max 2017 --test-vintage-min 2019

Leakage controls
  * Time split enforces a >12-month embargo: the label for a train row at month t looks
    to t+12, so test rows must start after train_end + 12 months.
  * The train/validation split (early stopping, calibration fit) is by LOAN, so the same
    loan never sits on both sides.
  * Features never look ahead (see build_panel.py). Sampling is by loan, not by row.
  * Caveat: in the time split, the same loans can appear in train (early months) and test
    (later months). That matches deployment (re-scoring existing loans) but is more
    optimistic than the vintage split, whose loans are disjoint. Report both.
  * Caveat: the vintage split still shares calendar time (a 2020 shock hits every vintage),
    so it does not remove macro-regime effects. Report results by period.

Outputs in --out: metrics.json, calibration_<model>.csv, groups_<model>.csv,
feature_importance_lgbm.csv, lgbm.txt
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = logging.getLogger("train")

NUM = [
    "loan_age", "dq_months", "dq_lag1", "dq_lag3", "dq_max_6m", "dq_max_12m", "n_late_12m",
    "upb_ratio", "current_interest_rate", "rate_chg", "estimated_ltv", "modified",
    "credit_score", "mi_pct", "num_units", "cltv", "dti", "orig_upb", "ltv",
    "orig_interest_rate", "orig_loan_term", "num_borrowers",
]
HISTORY = ["dq_months", "dq_lag1", "dq_lag3", "dq_max_6m", "dq_max_12m", "n_late_12m"]
CAT = ["first_time_homebuyer_flag", "occupancy_status", "channel", "property_state",
       "property_type", "loan_purpose"]


# Macro columns exist only in panels built by spark_panel.py (state/month FRED join).
MACRO = ["unemp_rate", "unemp_chg_12m", "hpi_yoy", "mortgage_rate", "rate_gap"]


def panel_macro_cols(panel: Path) -> list[str]:
    have = {r[0] for r in duckdb.connect().execute(
        f"DESCRIBE SELECT * FROM read_parquet('{panel}/*/*.parquet')").fetchall()}
    return [c for c in MACRO if c in have]


def load_panel(panel: Path, sample_frac: float) -> pd.DataFrame:
    macro = panel_macro_cols(panel)
    cols = ", ".join(NUM + macro + CAT)
    keep = int(round(sample_frac * 1000))
    q = f"""
      SELECT loan_id, rep_month,
             CASE WHEN CAST(substr(loan_id, 2, 2) AS INTEGER) >= 90 THEN 1900 ELSE 2000 END
               + CAST(substr(loan_id, 2, 2) AS INTEGER) AS vintage,
             y_12m::INTEGER AS y, (hash(loan_id) // 1000) % 100 AS hbucket, {cols}
      FROM read_parquet('{panel}/*/*.parquet')
      WHERE y_12m IS NOT NULL AND (hash(loan_id) % 1000) < {keep}"""
    con = duckdb.connect(); con.execute("SET memory_limit='5GB'"); df = con.execute(q).df()
    for c in NUM + macro:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in CAT:
        df[c] = df[c].astype("category")  # categories fixed on full frame -> same codes everywhere
    df["rep_month"] = pd.to_datetime(df["rep_month"])
    return df


def make_split(df: pd.DataFrame, a) -> tuple[pd.DataFrame, pd.DataFrame]:
    if a.split == "time":
        train_end = pd.Timestamp(a.train_end + "-01")
        test_start = pd.Timestamp(a.test_start + "-01")
        if test_start <= train_end + pd.DateOffset(months=12):
            raise ValueError("test-start must be more than 12 months after train-end "
                             "(label look-ahead embargo)")
        return df[df.rep_month <= train_end], df[df.rep_month >= test_start]
    train = df[df.vintage <= a.train_vintage_max]
    test = df[df.vintage >= a.test_vintage_min]
    return train, test


def ece(y, p, bins=10):
    d = pd.DataFrame({"y": y, "p": p})
    d["b"] = pd.qcut(d.p, bins, duplicates="drop")
    g = d.groupby("b", observed=True).agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean"))
    return float((g.n * (g.y - g.p).abs()).sum() / g.n.sum()), g.reset_index(drop=True)


def metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    e, _ = ece(y, p)
    return {"auc": float(roc_auc_score(y, p)), "avg_precision": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p)),
            "ece": e, "base_rate": float(np.mean(y)), "mean_pred": float(np.mean(p)),
            "n": int(len(y))}


def group_table(test: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    t = test.assign(p=p)
    bands = {
        "credit_score": pd.cut(t.credit_score, [0, 620, 680, 740, 900], right=False),
        "ltv": pd.cut(t.ltv, [0, 60, 80, 95, 1000], right=True),
        "dti": pd.cut(t.dti, [0, 30, 40, 50, 100], right=True),
        "occupancy_status": t.occupancy_status.astype(str),
        "channel": t.channel.astype(str),
        "vintage": t.vintage.astype(str),
        "report_year": t.rep_month.dt.year.astype(str),
    }
    rows = []
    for name, band in bands.items():
        for lvl, g in t.groupby(band, observed=True):
            auc = roc_auc_score(g.y, g.p) if g.y.nunique() == 2 else np.nan
            rows.append({"group": name, "level": str(lvl), "n": len(g),
                         "observed": g.y.mean(), "predicted": g.p.mean(), "auc": auc})
    return pd.DataFrame(rows)


def run(a) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_panel(Path(a.panel), a.sample_frac)
    if getattr(a, "current_only", False):  # score only loans that are fully current today
        df = df[df.dq_months == 0]
    log.info("Loaded %d labeled rows, %d loans", len(df), df.loan_id.nunique())
    train_all, test = make_split(df, a)
    if getattr(a, "disjoint", False):  # no loan appears in both train/val and test
        test = test[test.hbucket < 10]
        train_all = train_all[train_all.hbucket >= 10]
        val = train_all[train_all.hbucket < 20]
        train = train_all[train_all.hbucket >= 20]
    else:
        val = train_all[train_all.hbucket < 10]
        train = train_all[train_all.hbucket >= 10]
    if len(train) == 0 or len(test) == 0:
        raise SystemExit(f"Empty split: train={len(train)}, test={len(test)}")
    log.info("train %d (pos %.2f%%) | val %d | test %d (pos %.2f%%)", len(train),
             100 * train.y.mean(), len(val), len(test), 100 * test.y.mean())
    macro = [c for c in MACRO if c in df.columns]
    if getattr(a, "no_macro", False):
        macro = []
    num = [c for c in NUM if not (getattr(a, "no_history", False) and c in HISTORY)] + macro
    feats = num + CAT
    res: dict = {"split": a.split, "n_train": len(train), "n_val": len(val), "n_test": len(test)}

    # ---- logistic baseline ----
    lr_train = train.sample(min(len(train), a.max_logit_rows), random_state=0)
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=100), CAT)])
    logit = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=300, C=1.0))])
    logit.fit(lr_train[feats], lr_train.y)
    p_lr = logit.predict_proba(test[feats])[:, 1]
    res["logistic"] = metrics(test.y.values, p_lr)

    # ---- LightGBM ----
    gbm = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63,
                             min_child_samples=200, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, n_jobs=a.threads, verbose=-1)
    gbm.fit(train[feats], train.y, eval_set=[(val[feats], val.y)], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)])
    p_gb = gbm.predict_proba(test[feats])[:, 1]
    res["lgbm"] = metrics(test.y.values, p_gb)
    res["lgbm"]["best_iteration"] = int(gbm.best_iteration_ or gbm.n_estimators)

    # ---- isotonic recalibration fit on the validation (train-period) loans ----
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    iso.fit(gbm.predict_proba(val[feats])[:, 1], val.y)
    p_iso = iso.predict(p_gb)
    res["lgbm_isotonic"] = metrics(test.y.values, p_iso)

    for name, p in [("logistic", p_lr), ("lgbm", p_gb), ("lgbm_isotonic", p_iso)]:
        ece(test.y.values, np.clip(p, 1e-6, 1 - 1e-6))[1].to_csv(out / f"calibration_{name}.csv", index=False)
        group_table(test, p).to_csv(out / f"groups_{name}.csv", index=False)
    pd.DataFrame({"feature": feats,
                  "gain": gbm.booster_.feature_importance("gain")}).sort_values(
        "gain", ascending=False).to_csv(out / "feature_importance_lgbm.csv", index=False)
    gbm.booster_.save_model(str(out / "lgbm.txt"))
    joblib.dump(iso, out / "isotonic.joblib")  # used by score.py and the scenario engine
    (out / "features.json").write_text(json.dumps({"num": num, "cat": CAT, "macro": macro}))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    for k in ("logistic", "lgbm", "lgbm_isotonic"):
        m = res[k]
        log.info("%-14s AUC %.4f  AP %.4f  Brier %.5f  ECE %.5f  base %.4f  mean_pred %.4f",
                 k, m["auc"], m["avg_precision"], m["brier"], m["ece"], m["base_rate"], m["mean_pred"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["time", "vintage"], default="time")
    ap.add_argument("--train-end", default="2021-06")
    ap.add_argument("--test-start", default="2022-07")
    ap.add_argument("--train-vintage-max", type=int, default=2017)
    ap.add_argument("--test-vintage-min", type=int, default=2019)
    ap.add_argument("--sample-frac", type=float, default=0.1, help="fraction of LOANS kept")
    ap.add_argument("--max-logit-rows", type=int, default=1_000_000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--current-only", action="store_true",
                    help="keep only loan-months with dq_months == 0 (early-warning on current loans)")
    ap.add_argument("--no-history", action="store_true",
                    help="drop payment-history features (ablation)")
    ap.add_argument("--no-macro", action="store_true",
                    help="drop FRED macro features (ablation: do they add signal?)")
    ap.add_argument("--disjoint", action="store_true",
                    help="no loan appears in both train/val and test")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(a)


if __name__ == "__main__":
    main()
