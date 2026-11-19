from src.queries.silver_report import run


def test_report_on_synthetic(freddie, tmp_path):
    df = run(freddie, tmp_path / "rep", threads=1, memory_limit="1GB")
    assert df.loc[0, "orig_loans"] == 400
    assert df.loc[0, "orig_loans"] == df.loc[0, "perf_loans"]
    assert df.loc[0, "perf_loans_without_orig"] == 0
    for f in ("silver_by_source.csv", "silver_totals.json", "silver_checks.csv"):
        assert (tmp_path / "rep" / f).exists()
