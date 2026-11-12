"""Synthetic check of label logic: look-ahead window, censoring, at-risk filter."""
from pathlib import Path

import duckdb

from src.features.build_panel import ORIG_FEATURES, build


def _idx_to_date(i):  # month 1 = 2019-01
    y, m = divmod(i - 1, 12)
    return f"{2019 + y}-{m + 1:02d}-01"


def _write(tmp: Path):
    con = duckdb.connect()
    # A: never late, active 1..24. B: becomes 3 months late at month 8, stays. C: prepays at month 5.
    perf = []
    for t in range(1, 25):
        perf.append(("A", t, 100000.0, 0, None))
    for t in range(1, 15):
        perf.append(("B", t, 100000.0, 3 if t >= 8 else 0, None))
    for t in range(1, 6):
        perf.append(("C", t, 100000.0 if t < 5 else 0.0, 0, "01" if t == 5 else None))
    con.execute("CREATE TABLE perf(loan_sequence_number VARCHAR, t INT, current_actual_upb DOUBLE,"
                " dq_months INT, zero_balance_code VARCHAR)")
    con.executemany("INSERT INTO perf VALUES (?,?,?,?,?)", perf)
    con.execute("""CREATE VIEW perf_v AS SELECT loan_sequence_number,
        CAST(('2019-01-01'::DATE + INTERVAL (t-1) MONTH) AS DATE) AS monthly_reporting_period,
        current_actual_upb, dq_months, zero_balance_code, t AS loan_age,
        CAST(NULL AS VARCHAR) AS modification_flag, CAST(NULL AS INT) AS estimated_ltv,
        4.5 AS current_interest_rate FROM perf""")
    (tmp / "performance/source=T").mkdir(parents=True)
    (tmp / "origination/source=T").mkdir(parents=True)
    con.execute(f"COPY perf_v TO '{tmp}/performance/source=T/p.parquet' (FORMAT PARQUET)")
    cols = []
    for c in ORIG_FEATURES:
        if c in ("orig_upb",):
            cols.append("100000.0 AS orig_upb")
        elif c == "orig_interest_rate":
            cols.append("4.5 AS orig_interest_rate")
        elif c == "first_payment_date":
            cols.append("DATE '2019-01-01' AS first_payment_date")
        else:
            cols.append(f"CAST(NULL AS VARCHAR) AS {c}")
    con.execute(f"COPY (SELECT x AS loan_sequence_number, {', '.join(cols)} "
                f"FROM (VALUES ('A'),('B'),('C')) v(x)) "
                f"TO '{tmp}/origination/source=T/o.parquet' (FORMAT PARQUET)")


def test_labels(tmp_path):
    _write(tmp_path)
    out = tmp_path / "panel"
    build(tmp_path, out, threads=1, memory_limit="1GB", tmp_dir=None)
    rows = duckdb.connect().execute(
        f"SELECT loan_id, m_idx - (SELECT min(m_idx) FROM read_parquet('{out}/*/*.parquet')) + 1 AS t, "
        f"y_12m, dq_max_6m FROM read_parquet('{out}/*/*.parquet')").fetchall()
    lab = {(r[0], r[1]): r[2] for r in rows}
    # global last month = 24 (loan A). Window complete only if t + 12 <= 24 -> t <= 12.
    assert lab[("A", 1)] == 0 and lab[("A", 12)] == 0
    assert lab[("A", 13)] is None and lab[("A", 24)] is None
    # B turns 90+ at t=8: rows t=1..7 are positives; t>=8 excluded (already delinquent)
    assert all(lab[("B", t)] == 1 for t in range(1, 8))
    assert ("B", 8) not in lab
    # C prepays at 5: row 5 excluded (zero balance); t=1..4 labeled 0 (never late)
    assert all(lab[("C", t)] == 0 for t in range(1, 5)) and ("C", 5) not in lab
    # no leakage: history features at t=1 can't see later delinquency
    assert {r[3] for r in rows if r[0] == "B" and r[1] == 1} == {0}
