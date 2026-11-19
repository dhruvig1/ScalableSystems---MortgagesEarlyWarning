"""CFPB ingest merges an archive with narratives and a newer file that has none."""
import subprocess
import sys

import duckdb


def test_narrative_kept_when_newer_file_has_none(tmp_path):
    old = tmp_path / "a_archive.csv"   # sorts first = older
    new = tmp_path / "z_current.csv"   # sorts last = newest, no narrative column
    hdr_old = ("Date received,Product,Sub-product,Issue,Sub-issue,Consumer complaint narrative,"
               "Company,State,ZIP code,Submitted via,Complaint ID")
    old.write_text(hdr_old + "\n"
                   "2020-05-01,Mortgage,FHA,Struggling to pay mortgage,,I lost my job,ACME,IL,606XX,Web,1\n"
                   "2020-06-01,Mortgage,FHA,Trouble during payment process,,,ACME,TX,750XX,Web,2\n"
                   "2020-06-02,Credit card,,Fees,,hello,BANK,CA,900XX,Web,3\n")
    new.write_text("Date received,Product,Sub-product,Issue,Sub-issue,Company,State,ZIP code,"
                   "Submitted via,Complaint ID\n"
                   "2020-05-01,Mortgage,FHA,Struggling to pay mortgage,,ACME,IL,606XX,Web,1\n"
                   "2026-09-01,Mortgage,Conventional,Escrow,,ACME,FL,331XX,Web,4\n")
    out = tmp_path / "out"
    subprocess.run([sys.executable, "-m", "src.ingest.cfpb_to_parquet", "--inputs", str(old), str(new),
                    "--out", str(out)], check=True)
    rows = duckdb.connect().execute(
        f"SELECT complaint_id, narrative, issue FROM read_parquet('{out}/mortgage_complaints.parquet') "
        f"ORDER BY complaint_id").fetchall()
    assert [r[0] for r in rows] == ["1", "2", "4"]            # non-mortgage dropped, no duplicates
    assert rows[0][1] == "I lost my job"                       # narrative survived the newer file
    assert rows[1][1] is None and rows[2][1] is None           # genuinely no narrative
