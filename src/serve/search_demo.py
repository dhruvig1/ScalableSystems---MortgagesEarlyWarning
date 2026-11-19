"""Atlas Search demo queries over complaint narratives (run after `mongo_load indexes`).

Usage:
    python -m src.serve.search_demo --query "lost job forbearance ended" --state TX
Prints the top hits, then facet counts of issue types among all matches.
"""
from __future__ import annotations

import argparse

from src.serve.mongo_client import get_db
from src.serve.mongo_load import SEARCH_INDEX


def search(db, query: str, state: str | None = None, limit: int = 5):
    must = [{"text": {"query": query, "path": "narrative"}}]
    flt = [{"equals": {"path": "state", "value": state}}] if state else []
    pipeline = [
        {"$search": {"index": SEARCH_INDEX, "compound": {"must": must, "filter": flt}}},
        {"$limit": limit},
        {"$project": {"_id": 1, "state": 1, "issue": 1, "theme": 1, "date_received": 1,
                      "score": {"$meta": "searchScore"},
                      "snippet": {"$substrCP": ["$narrative", 0, 240]}}},
    ]
    return list(db.complaints.aggregate(pipeline))


def facets(db, query: str, state: str | None = None):
    flt = [{"equals": {"path": "state", "value": state}}] if state else []
    pipeline = [{"$searchMeta": {
        "index": SEARCH_INDEX,
        "facet": {"operator": {"compound": {"must": [{"text": {"query": query, "path": "narrative"}}],
                                            "filter": flt}},
                  "facets": {"issues": {"type": "string", "path": "issue", "numBuckets": 8}}}}}]
    return list(db.complaints.aggregate(pipeline))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--state", default=None)
    a = ap.parse_args()
    db = get_db()
    for h in search(db, a.query, a.state):
        print(f"{h['score']:.2f} {h.get('state')} {h.get('issue')} | {h['snippet']}")
    print(facets(db, a.query, a.state))


if __name__ == "__main__":
    main()
