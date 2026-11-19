"""Fast terminal tool for hand-labeling complaint themes (one keystroke per row).

Usage:
    python -m src.ai.hand_label_cli --template data/labels/hand_label_template.csv \
        --out data/labels/hand_labels.csv [--max-chars 900]

Shows the narrative only (not Claude's theme, not the CFPB issue) so the label is blind.
Keys: 1-6 = theme, s = skip, u = undo last, q = quit. Progress saves after every label,
so you can stop and resume any time; `validate_labels hand` works on a partial file.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd

from src.ai.themes import THEME_LIST, THEMES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-chars", type=int, default=900)
    a = ap.parse_args()

    t = pd.read_csv(a.template, dtype={"complaint_id": str})
    done = {}
    if a.out.exists():
        old = pd.read_csv(a.out, dtype={"complaint_id": str}).dropna(subset=["human_theme"])
        done = dict(zip(old.complaint_id, old.human_theme))
    order = []  # labeling order, for undo

    def save() -> None:
        out = t[["complaint_id", "issue", "narrative"]].copy()
        out["human_theme"] = out.complaint_id.map(done)
        out.to_csv(a.out, index=False)

    print("\nThemes:")
    for i, k in enumerate(THEME_LIST, 1):
        print(f"  {i}. {k}: {THEMES[k]}")
    print("\nKeys: 1-6 theme | s skip | u undo | q quit\n")

    todo = [r for r in t.itertuples(index=False) if r.complaint_id not in done]
    print(f"{len(done)} already labeled, {len(todo)} to go.\n")
    skipped = set()
    for r in todo:
        if r.complaint_id in skipped:
            continue
        while True:
            print("-" * 78)
            print(f"[{len(done) + 1}]  " + "  ".join(f"{i}={k}" for i, k in enumerate(THEME_LIST, 1)))
            print(textwrap.fill(str(r.narrative)[:a.max_chars], 78))
            key = input("\nTheme (1-6, s, u, q): ").strip().lower()
            if key == "q":
                save()
                print(f"Saved {len(done)} labels to {a.out}")
                return
            if key == "u" and order:
                done.pop(order.pop(), None)
                save()
                print("(undid last label)")
                continue
            if key == "s":
                skipped.add(r.complaint_id)
                break
            if key.isdigit() and 1 <= int(key) <= len(THEME_LIST):
                done[r.complaint_id] = THEME_LIST[int(key) - 1]
                order.append(r.complaint_id)
                save()
                break
            print("Press 1-6, s, u or q.")
    save()
    print(f"Done. {len(done)} labels saved to {a.out}")


if __name__ == "__main__":
    main()
