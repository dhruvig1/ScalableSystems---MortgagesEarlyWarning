"""Synthetic end-to-end check of the Freddie loader (no real data needed)."""
import zipfile
from pathlib import Path

import duckdb

from src.ingest.freddie_to_parquet import process_zip


def _make_zip(tmp: Path, o_name="historical_data", p_name="historical_data_time") -> Path:
    orig = ["|".join(["9999", "201903", "N", "204902", "16974", "000", "1", "P", "80", "999",
                      "200000", "80", "4.5", "R", "N", "FRM", "IL", "SF", "606", "F19Q1000001",
                      "P", "360", "2", "SELLER", "SERVICER", "", "", "", "", "9", "N", "N"]),
            "|".join(["720", "201903", "9", "204902", "16974", "000", "1", "P", "70", "35",
                      "150000", "70", "4.25", "R", "N", "FRM", "CA", "SF", "900", "F19Q1000002",
                      "P", "360", "1", "SELLER", "SERVICER", "", "", "", "", "9", "N", "N"])]
    perf = []
    for loan in ("F19Q1000001", "F19Q1000002"):
        for i, (ym, st) in enumerate([("201904", "0"), ("201905", "1"), ("201906", "RA")]):
            row = [loan, ym, "199000.0", st, str(i + 1), "357"] + [""] * 20 + ["999"] + [""] * 5
            perf.append("|".join(row))
    z = tmp / "historical_data_2019Q1.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(f"{o_name}_2019Q1.txt", "\n".join(orig) + "\n")
        zf.writestr(f"{p_name}_2019Q1.txt", "\n".join(perf) + "\n")
    return z


def test_loader(tmp_path):
    z = _make_zip(tmp_path)
    out = tmp_path / "pq"
    process_zip(z, out, threads=1, memory_limit="1GB")
    con = duckdb.connect()
    o = con.execute(f"SELECT credit_score, dti, first_payment_date FROM "
                    f"read_parquet('{out}/origination/*/*.parquet') ORDER BY loan_sequence_number"
                    ).fetchall()
    assert o[0][0] is None and o[0][1] is None          # 9999 / 999 -> NULL
    assert o[1][0] == 720 and o[1][1] == 35
    assert str(o[0][2]) == "2019-03-01"
    p = con.execute(f"SELECT dq_months, estimated_ltv FROM read_parquet("
                    f"'{out}/performance/*/*.parquet') ORDER BY monthly_reporting_period"
                    ).fetchall()
    assert len(p) == 6
    assert sorted({r[0] for r in p}) == [0, 1, 99]       # RA -> 99
    assert all(r[1] is None for r in p)                  # 999 -> NULL


def test_nested_yearly_zip(tmp_path):
    inner = _make_zip(tmp_path)  # historical_data_2019Q1.zip
    outer = tmp_path / "historical_data_2019.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.write(inner, inner.name)
    out = tmp_path / "pq2"
    process_zip(outer, out, threads=1, memory_limit="1GB")
    n = duckdb.connect().execute(
        f"SELECT COUNT(*) FROM read_parquet('{out}/performance/source=2019Q1/*.parquet')"
    ).fetchone()[0]
    assert n == 6


def test_new_naming(tmp_path):
    z = _make_zip(tmp_path, "orig", "perf")
    out = tmp_path / "pq3"
    process_zip(z, out, threads=1, memory_limit="1GB")
    n = duckdb.connect().execute(
        f"SELECT COUNT(*) FROM read_parquet('{out}/performance/*/*.parquet')").fetchone()[0]
    assert n == 6


def test_rerun_is_idempotent(tmp_path):
    z = _make_zip(tmp_path)
    out = tmp_path / "pq4"
    q = f"SELECT COUNT(*) FROM read_parquet('{out}/performance/*/*.parquet')"
    process_zip(z, out, threads=1, memory_limit="1GB")
    n1 = duckdb.connect().execute(q).fetchone()[0]
    assert (out / "_done" / "2019Q1").exists()
    process_zip(z, out, threads=1, memory_limit="1GB")            # skipped
    process_zip(z, out, threads=1, memory_limit="1GB", force=True)  # redone, no duplicates
    n2 = duckdb.connect().execute(q).fetchone()[0]
    assert n1 == n2 == 6


def test_rerun_after_partial_output(tmp_path):
    z = _make_zip(tmp_path)
    out = tmp_path / "pq5"
    stale = out / "performance" / "source=2019Q1"
    stale.mkdir(parents=True)
    (stale / "stale.parquet").write_bytes(b"junk")  # simulates interrupted run
    process_zip(z, out, threads=1, memory_limit="1GB")
    n = duckdb.connect().execute(
        f"SELECT COUNT(*) FROM read_parquet('{out}/performance/*/*.parquet')").fetchone()[0]
    assert n == 6
