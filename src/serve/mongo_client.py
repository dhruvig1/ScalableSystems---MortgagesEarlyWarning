"""MongoDB Atlas connection. The URI comes from the environment only, never from a file.

    export MONGODB_URI='mongodb+srv://<user>:<password>@<cluster>/...'   # set in your shell
    export MONGODB_DB=mew                                                # optional
"""
from __future__ import annotations

import os

from pymongo import MongoClient


def get_db(uri: str | None = None, name: str | None = None):
    uri = uri or os.environ.get("MONGODB_URI")
    if not uri:
        raise SystemExit("Set MONGODB_URI in your environment (do not commit it).")
    client = MongoClient(uri, appname="mortgage-early-warning", serverSelectionTimeoutMS=15000)
    return client[name or os.environ.get("MONGODB_DB", "mew")]
