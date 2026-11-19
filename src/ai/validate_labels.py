"""Check Claude's theme labels two ways: against CFPB's own issue categories and a hand sample.

Usage:
    python -m src.ai.validate_labels cfpb --parquet data/parquet/cfpb/mortgage_complaints.parquet \
        --labels data/labels/claude_labels.csv --out data/outputs
    python -m src.ai.validate_labels export-hand --parquet ... --labels ... --out data/labels --n 200
        (fill the empty `human_theme` column in data/labels/hand_label_template.csv, save as
         data/labels/hand_labels.csv)
    python -m src.ai.validate_labels hand --labels data/labels/claude_labels.csv \
        --hand data/labels/hand_labels.csv --out data/outputs

`cfpb` reports agreement and Cohen's kappa against a rule-based mapping of CFPB Issue/Sub-issue.
That mapping is imperfect, so it is a sanity check. `hand` is the accuracy number to quote.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, classification_report

from src.ai.themes import THEME_LIST, cfpb_theme


def _report(y_true, y_pred, name: str) -> dict:
    return {
        "name": name, "n": int(len(y_true)),
        "accuracy": float((pd.Series(list(y_true)) == pd.Series(list(y_pred))).mean()),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=THEME_LIST)),
        "per_class": classification_report(y_true, y_pred, labels=THEME_LIST,
                                          output_dict=True, zero_division=0),
    }


def vs_cfpb(parquet: Path, labels: Path) -> tuple[dict, pd.DataFrame]:
    c = pd.read_parquet(parquet, columns=["complaint_id", "issue", "sub_issue"])
    l = pd.read_csv(labels, dtype={"complaint_id": str})
    c["complaint_id"] = c["complaint_id"].astype(str)
    m = l.merge(c, on="complaint_id")
    m["cfpb_theme"] = [cfpb_theme(i, s) for i, s in zip(m.issue, m.sub_issue)]
    rep = _report(m.cfpb_theme, m.theme, "claude_vs_cfpb_issue_mapping")
    conf = pd.crosstab(m.cfpb_theme, m.theme)
    return rep, conf


def vs_hand(labels: Path, hand: Path) -> dict:
    l = pd.read_csv(labels, dtype={"complaint_id": str})
    h = pd.read_csv(hand, dtype={"complaint_id": str}).dropna(subset=["human_theme"])
    m = l.merge(h[["complaint_id", "human_theme"]], on="complaint_id")
    bad = set(m.human_theme) - set(THEME_LIST)
    if bad:
        raise SystemExit(f"Unknown human themes: {bad}")
    return _report(m.human_theme, m.theme, "claude_vs_hand_labels")


def export_hand(parquet: Path, labels: Path, out: Path, n: int) -> Path:
    c = pd.read_parquet(parquet, columns=["complaint_id", "narrative", "issue"])
    c["complaint_id"] = c["complaint_id"].astype(str)
    l = pd.read_csv(labels, dtype={"complaint_id": str})
    m = l.merge(c, on="complaint_id").sample(min(n, len(l)), random_state=7)
    # Blind: do NOT show Claude's theme to the human labeler.
    t = m[["complaint_id", "issue", "narrative"]].assign(human_theme="")
    out.mkdir(parents=True, exist_ok=True)
    p = out / "hand_label_template.csv"
    t.to_csv(p, index=False)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("cfpb")
    a1.add_argument("--parquet", required=True, type=Path)
    a1.add_argument("--labels", required=True, type=Path)
    a1.add_argument("--out", required=True, type=Path)
    a2 = sub.add_parser("export-hand")
    a2.add_argument("--parquet", required=True, type=Path)
    a2.add_argument("--labels", required=True, type=Path)
    a2.add_argument("--out", required=True, type=Path)
    a2.add_argument("--n", type=int, default=200)
    a3 = sub.add_parser("hand")
    a3.add_argument("--labels", required=True, type=Path)
    a3.add_argument("--hand", required=True, type=Path)
    a3.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if a.cmd == "cfpb":
        rep, conf = vs_cfpb(a.parquet, a.labels)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "label_check_cfpb.json").write_text(json.dumps(rep, indent=2))
        conf.to_csv(a.out / "label_confusion_cfpb.csv")
        print(f"n={rep['n']} accuracy={rep['accuracy']:.3f} kappa={rep['kappa']:.3f}")
        print(conf)
    elif a.cmd == "export-hand":
        print("wrote", export_hand(a.parquet, a.labels, a.out, a.n))
    else:
        rep = vs_hand(a.labels, a.hand)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "label_check_hand.json").write_text(json.dumps(rep, indent=2))
        print(f"n={rep['n']} accuracy={rep['accuracy']:.3f} kappa={rep['kappa']:.3f}")


if __name__ == "__main__":
    main()
