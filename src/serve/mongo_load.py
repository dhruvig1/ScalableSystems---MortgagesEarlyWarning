"""Load pipeline outputs into MongoDB Atlas (idempotent: re-running upserts by _id).

Collections
    complaints          one document per CFPB mortgage complaint (text + metadata, later `theme`)
    state_month_rollup  state x month loan counts (n_active, n_90) from Spark
    scored_loans        top-risk loans from the latest scoring run
    scenario_results    Monte Carlo summaries per scenario

Why Mongo here: complaints are variable-shape documents with free text that need full-text
search (Atlas Search) and get new fields as Claude labels them. Rollups and scenario summaries
are small documents that the dashboard and the MCP server read by key.

Usage:
    python -m src.serve.mongo_load complaints --parquet data/parquet/cfpb/mortgage_complaints.parquet
    python -m src.serve.mongo_load rollups    --parquet data/outputs/state_month_rollup.parquet
    python -m src.serve.mongo_load scores     --parquet data/outputs/portfolio_snapshot.parquet --top 50000
    python -m src.serve.mongo_load scenarios  --csv data/outputs/scenario_summary.csv
    python -m src.serve.mongo_load indexes
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from pymongo import ReplaceOne
from pymongo.operations import SearchIndexModel

log = logging.getLogger("mongo_load")

SEARCH_INDEX = "complaints_search"
SEARCH_DEFINITION = {
    "mappings": {
        "dynamic": False,
        "fields": {
            "narrative": {"type": "string", "analyzer": "lucene.english"},
            "issue": {"type": "token"},
            "sub_issue": {"type": "token"},
            "state": {"type": "token"},
            "company": {"type": "token"},
            "theme": {"type": "token"},
            "date_received": {"type": "date"},
        },
    }
}


def _bulk(coll, docs: list[dict], batch: int = 5000) -> int:
    n = 0
    for i in range(0, len(docs), batch):
        ops = [ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in docs[i:i + batch]]
        if ops:
            coll.bulk_write(ops, ordered=False)
            n += len(ops)
    return n


def _clean(df: pd.DataFrame) -> list[dict]:
    df = df.astype(object).where(pd.notna(df), None)
    return df.to_dict("records")


def load_complaints(db, parquet: Path, keep_labels: bool = True,
                    only_narratives: bool = False, max_docs: int | None = None) -> int:
    """only_narratives / max_docs keep the load inside the 512 MB Atlas free tier."""
    df = pd.read_parquet(parquet)
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
    df["has_narrative"] = df["narrative"].fillna("").str.strip().ne("")
    df["narrative"] = df["narrative"].where(df["has_narrative"], None)
    if only_narratives:
        df = df[df["has_narrative"]]
    if max_docs and len(df) > max_docs:  # newest first, so recent months survive the cap
        df = df.sort_values("date_received", ascending=False).head(max_docs)
    df["_id"] = df["complaint_id"].astype(str)
    docs = _clean(df.drop(columns=["complaint_id"]))
    for d in docs:  # pandas NaT -> None, python datetimes for BSON dates
        ts = d.get("date_received")
        d["date_received"] = None if ts is None or pd.isna(ts) else pd.Timestamp(ts).to_pydatetime()
    if keep_labels:  # do not wipe labels Claude already wrote
        done = {x["_id"]: x for x in db.complaints.find(
            {"theme": {"$exists": True}}, {"theme": 1, "theme_confidence": 1, "theme_model": 1})}
        for d in docs:
            if d["_id"] in done:
                d.update({k: v for k, v in done[d["_id"]].items() if k != "_id"})
    n = _bulk(db.complaints, docs)
    db.complaints.create_index([("state", 1), ("date_received", 1)])
    db.complaints.create_index([("has_narrative", 1)])
    log.info("complaints upserted: %d", n)
    return n


def load_rollups(db, df: pd.DataFrame) -> int:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    docs = []
    for r in df.to_dict("records"):
        docs.append({"_id": f"{r['state']}|{r['month']:%Y-%m}", "state": r["state"],
                     "month": r["month"].to_pydatetime(), "n_active": int(r["n_active"]),
                     "n_90": int(r["n_90"])})
    n = _bulk(db.state_month_rollup, docs)
    db.state_month_rollup.create_index([("state", 1), ("month", 1)])
    log.info("state_month_rollup upserted: %d", n)
    return n


def load_scores(db, parquet: Path, top: int) -> int:
    df = pd.read_parquet(parquet).sort_values("pd_12m", ascending=False).head(top)
    keep = ["loan_id", "rep_month", "property_state", "upb", "pd_12m", "credit_score", "ltv",
            "dti", "loan_age", "dq_months", "occupancy_status"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["rep_month"] = pd.to_datetime(df["rep_month"])
    docs = _clean(df)
    for d in docs:
        d["_id"] = d["loan_id"]
        d["rep_month"] = pd.Timestamp(d["rep_month"]).to_pydatetime()
    n = _bulk(db.scored_loans, docs)
    db.scored_loans.create_index([("pd_12m", -1)])
    db.scored_loans.create_index([("property_state", 1), ("pd_12m", -1)])
    log.info("scored_loans upserted: %d", n)
    return n


def load_scenarios(db, csv: Path) -> int:
    df = pd.read_csv(csv)
    docs = [{"_id": r["scenario"], **r} for r in _clean(df)]
    n = _bulk(db.scenario_results, docs)
    log.info("scenario_results upserted: %d", n)
    return n


def ensure_search_index(db) -> None:
    existing = {i["name"] for i in db.complaints.list_search_indexes()}
    if SEARCH_INDEX in existing:
        log.info("search index %s already exists", SEARCH_INDEX)
        return
    db.complaints.create_search_index(
        SearchIndexModel(definition=SEARCH_DEFINITION, name=SEARCH_INDEX, type="search"))
    log.info("created search index %s (builds in the background; check Atlas UI)", SEARCH_INDEX)


def main() -> None:
    from src.serve.mongo_client import get_db
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("complaints"); c.add_argument("--parquet", required=True, type=Path)
    c.add_argument("--only-narratives", action="store_true")
    c.add_argument("--max-docs", type=int, default=None)
    r = sub.add_parser("rollups"); r.add_argument("--parquet", required=True, type=Path)
    s = sub.add_parser("scores"); s.add_argument("--parquet", required=True, type=Path)
    s.add_argument("--top", type=int, default=50_000)
    sc = sub.add_parser("scenarios"); sc.add_argument("--csv", required=True, type=Path)
    sub.add_parser("indexes")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    db = get_db()
    if a.cmd == "complaints":
        load_complaints(db, a.parquet, only_narratives=a.only_narratives, max_docs=a.max_docs)
    elif a.cmd == "rollups":
        load_rollups(db, pd.read_parquet(a.parquet))
    elif a.cmd == "scores":
        load_scores(db, a.parquet, a.top)
    elif a.cmd == "scenarios":
        load_scenarios(db, a.csv)
    elif a.cmd == "indexes":
        ensure_search_index(db)


if __name__ == "__main__":
    main()
