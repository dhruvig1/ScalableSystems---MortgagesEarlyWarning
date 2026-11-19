"""Refresh complaint counts from CFPB's public API (the database updates daily).

This is the live piece of the pipeline. The API returns complaint metadata only (date, state,
issue, company, product); it does NOT return narrative text, so narratives come from the archive
files and the search collection stays as it is. The metadata is enough for what Q2 uses: complaint
counts by state and month.

    export MONGODB_URI='mongodb+srv://...'          # admin user (needs write access)
    python -m src.ingest.cfpb_live --dry-run        # fetch and print counts, write nothing
    python -m src.ingest.cfpb_live                  # upsert into mew.complaints_live
    python -m src.ingest.cfpb_live --since 2026-07-01

Default start date = newest stored live complaint minus 7 days (or 2026-07-01 the first time).
Re-running is safe: documents are keyed by complaint id.
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
from collections import Counter
from datetime import datetime, timedelta

import requests
from pymongo import UpdateOne

from src.serve.mongo_client import get_db

log = logging.getLogger("cfpb_live")
API = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
# CFPB's server rejects the default python-requests user agent, so identify the script plainly.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; mortgage-early-warning class project)",
           "Accept": "text/csv"}
COLL = "complaints_live"


def list_new(since: str) -> list[dict]:
    """All Mortgage complaints received on/after `since` (CSV export, metadata only)."""
    r = requests.get(API, timeout=180, headers=HEADERS, params={
        "product": "Mortgage", "date_received_min": since,
        "sort": "created_date_desc", "format": "csv", "no_aggs": "true"})
    if not r.ok:
        raise SystemExit(f"CFPB API returned {r.status_code}: {r.text[:300]!r}")
    return list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))


def to_doc(row: dict) -> dict:
    day = (row.get("Date received") or "")[:10]
    return {
        "_id": str(row["Complaint ID"]),
        "date_received": datetime.strptime(day, "%Y-%m-%d") if "-" in day else datetime.strptime(day, "%m/%d/%y"),
        "product": row.get("Product"), "sub_product": row.get("Sub-product"),
        "issue": row.get("Issue"), "sub_issue": row.get("Sub-issue"),
        "company": row.get("Company"), "state": row.get("State"),
        "submitted_via": row.get("Submitted via"), "source": "cfpb_api_live",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="YYYY-MM-DD")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    db = get_db()
    since = a.since
    if not since:
        newest = db[COLL].find_one(sort=[("date_received", -1)], projection={"date_received": 1})
        since = ((newest["date_received"] - timedelta(days=7)).strftime("%Y-%m-%d")
                 if newest else "2026-07-01")
    log.info("fetching Mortgage complaints received since %s", since)
    docs = [to_doc(r) for r in list_new(since)]
    by_month = Counter(d["date_received"].strftime("%Y-%m") for d in docs)
    log.info("API returned %d complaints; by month: %s", len(docs), dict(sorted(by_month.items())))
    if a.dry_run or not docs:
        return
    # $setOnInsert keeps existing documents unchanged, so re-runs only add new complaints
    res = db[COLL].bulk_write(
        [UpdateOne({"_id": d["_id"]}, {"$setOnInsert": d}, upsert=True) for d in docs], ordered=False)
    db[COLL].create_index([("state", 1), ("date_received", 1)])
    log.info("new complaints inserted: %d (already present: %d)", res.upserted_count,
             len(docs) - res.upserted_count)
    newest = db[COLL].find_one(sort=[("date_received", -1)], projection={"date_received": 1})
    log.info("newest complaint now stored: %s | total in %s: %d", newest["date_received"].date(),
             COLL, db[COLL].count_documents({}))


if __name__ == "__main__":
    main()
