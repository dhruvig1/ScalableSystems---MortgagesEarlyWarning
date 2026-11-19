"""Claude labels a capped, deterministic sample of complaint narratives into themes.

Usage:
    export ANTHROPIC_API_KEY=...            # environment only, never in a file
    python -m src.ai.label_complaints --parquet data/parquet/cfpb/mortgage_complaints.parquet \
        --out data/labels --n 3000 [--write-mongo]

Steps
  1. Pick the sample: complaints with a narrative, ordered by hash(complaint_id), first N.
     Same N always gives the same sample, so reruns resume instead of paying twice.
  2. Ask Claude for {"theme", "confidence"} per narrative (one call each, JSON only).
  3. Append to <out>/claude_labels.csv (skips ids already labeled). Optionally write the
     theme back onto the Atlas documents so Atlas Search can filter and facet on it.

Interactive, ad hoc analysis over the same data runs through the MongoDB MCP server
(see docs/demo_script.md). This script is the batch labeling step with a cost cap.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

from src.ai.themes import THEME_LIST, THEMES

log = logging.getLogger("label")
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5")
MAX_CHARS = 1500

SYSTEM = (
    "You classify US mortgage complaints filed with the CFPB into exactly one theme. "
    "Text may contain XXXX redactions. Themes:\n"
    + "\n".join(f"- {k}: {v}" for k, v in THEMES.items())
    + '\nReply with ONLY a JSON object: {"theme": "<one theme>", "confidence": <0..1>}.'
)


def parse_reply(text: str) -> tuple[str, float]:
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {text[:80]!r}")
    obj = json.loads(m.group(0))
    theme = str(obj.get("theme", "")).strip()
    if theme not in THEME_LIST:
        raise ValueError(f"unknown theme {theme!r}")
    return theme, float(obj.get("confidence", 0.0))


def classify(client, narrative: str, model: str = DEFAULT_MODEL, retries: int = 2) -> tuple[str, float]:
    last = None
    for i in range(retries):
        try:
            msg = client.messages.create(
                model=model, max_tokens=800, system=SYSTEM,
                messages=[{"role": "user", "content": narrative[:MAX_CHARS]}])
            # the reply may start with a thinking block; keep only the text blocks
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            return parse_reply(text)
        except Exception as e:  # rate limit, bad JSON, etc.
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"classification failed after {retries} tries: {last}")


def pick_sample(parquet: Path, n: int):
    return duckdb.connect().execute(f"""
        SELECT complaint_id, narrative, issue, sub_issue, state, date_received
        FROM read_parquet('{parquet}')
        WHERE narrative IS NOT NULL AND length(trim(narrative)) > 40
        ORDER BY hash(complaint_id) LIMIT {int(n)}""").df()


def run(parquet: Path, out: Path, n: int, model: str, write_mongo: bool, client=None,
        workers: int = 4) -> int:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "claude_labels.csv"
    done = set()
    if path.exists():
        with open(path, newline="") as f:
            done = {r["complaint_id"] for r in csv.DictReader(f)}
    sample = pick_sample(parquet, n)
    todo = sample[~sample.complaint_id.isin(done)]
    log.info("sample %d, already labeled %d, to label %d", len(sample), len(done), len(todo))
    if client is None:
        import anthropic
        # reads ANTHROPIC_API_KEY; extra retries ride out flaky home-network connection drops
        # gzip only: some installs have a zstd decoder version mismatch that breaks every reply
        client = anthropic.Anthropic(max_retries=5, timeout=45,
                                     default_headers={"Accept-Encoding": "gzip, deflate"})
    new = not path.exists()
    labeled = []

    def one(r):
        try:
            return r.complaint_id, classify(client, r.narrative, model), None
        except RuntimeError as e:
            return r.complaint_id, None, e

    # a few calls in flight at once: ~3 s per call serially is hours for thousands of rows.
    # Keep workers low so a new account's requests-per-minute limit is not exceeded.
    with open(path, "a", newline="") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        w = csv.writer(f)
        if new:
            w.writerow(["complaint_id", "theme", "confidence", "model"])
        rows = list(todo.itertuples(index=False))
        for i, (cid, res, err) in enumerate(pool.map(one, rows), 1):
            if err is not None:
                log.warning("skip %s: %s", cid, err)
                continue
            theme, conf = res
            w.writerow([cid, theme, f"{conf:.2f}", model])
            f.flush()
            labeled.append((cid, theme, conf))
            if i % 100 == 0:
                log.info("labeled %d / %d", i, len(todo))
    if write_mongo:  # push every label in the CSV, including ones from earlier runs
        from src.serve.mongo_client import get_db
        from pymongo import UpdateOne
        with open(path, newline="") as f:
            allrows = list(csv.DictReader(f))
        db = get_db()
        ops = [UpdateOne({"_id": str(r["complaint_id"])},
                         {"$set": {"theme": r["theme"], "theme_confidence": float(r["confidence"]),
                                   "theme_model": r["model"]}}) for r in allrows]
        for j in range(0, len(ops), 1000):
            db.complaints.bulk_write(ops[j:j + 1000], ordered=False)
        log.info("wrote %d themes to Atlas", len(ops))
    return len(labeled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=3000, help="cap on narratives labeled (cost control)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--write-mongo", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="parallel API calls")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(a.parquet, a.out, a.n, a.model, a.write_mongo, workers=a.workers)


if __name__ == "__main__":
    main()
