"""Source-shape and dated exposure safeguards for AEWR inputs."""

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import us

from h2a.aewr import (
    build_aewr_artifacts,
    interval_rate,
    parse_fls_table,
    validate_schedule,
)
from h2a.paths import CODE, RAW


def expect_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected malformed source to fail")


def test_parser_rejects_application_means_and_incomplete_schedules():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "fixture.xml"
        rows = [f"<ROW><ENT>{s.name}</ENT><ENT>10.50</ENT></ROW>"
                for s in us.STATES if s.abbr != "AK"]
        prefix = "<ROOT><GPOTABLE><TTITLE>State Adverse Effect Wage Rates</TTITLE>"
        path.write_text(prefix + "".join(rows) + "</GPOTABLE></ROOT>")
        assert len(parse_fls_table(path)) == 49
        path.write_text(prefix + "".join(rows[:-1]) + "</GPOTABLE></ROOT>")
        expect_value_error(lambda: parse_fls_table(path))
        path.write_text(prefix + "".join(rows + rows[:1]) + "</GPOTABLE></ROOT>")
        expect_value_error(lambda: parse_fls_table(path))
        path.write_text(prefix + "<ROW><ENT>AL</ENT><ENT>8</ENT><ENT>9</ENT>"
                        "<ENT>10</ENT><ENT>11</ENT><ENT>12</ENT></ROW></GPOTABLE></ROOT>")
        expect_value_error(lambda: parse_fls_table(path))


def test_effective_date_intersections_and_gap_are_exact():
    frame = pl.DataFrame({
        "state_fips": ["01", "01"], "occupation_scope": ["reference"] * 2,
        "effective_start": [date(2020, 2, 1), date(2020, 3, 1)],
        "effective_end_exclusive": [date(2020, 3, 1), date(2020, 4, 1)],
        "aewr": [10.0, 12.0]})
    validate_schedule(frame)
    result = interval_rate(frame, "01", date(2020, 2, 28), date(2020, 3, 2))
    assert result["reference_covered_days"] == 3  # Includes February 29.
    assert result["reference_rate_day_sum"] == 32
    assert interval_rate(frame, "01", date(2020, 1, 31), date(2020, 2, 2))["reference_uncovered_days"] == 1
    expect_value_error(lambda: validate_schedule(pl.concat([frame, frame.head(1)])))


def test_cached_sources_preserve_history_and_quarantine_2009():
    artifacts = build_aewr_artifacts(RAW / "aewr", CODE / "a01_sources/aewr_rate_schedule_manifest.csv")
    annual = artifacts["aewr.parquet"]
    assert annual.height == 49 * 36
    assert annual.filter(pl.col("year") == 2009)["aewr"].null_count() == 49
    assert annual.filter((pl.col("year") == 2008) & (pl.col("state_fips") == "06"))["aewr"].item() == 9.72
    assert annual.filter((pl.col("year") == 2003) & (pl.col("state_fips") == "15"))["aewr"].item() == 9.42
    assert artifacts["aewr_reference_rate_source_audit.parquet"].height == annual.height
    gap = interval_rate(artifacts["aewr_rate_schedule.parquet"], "01", date(2009, 1, 17), date(2010, 3, 15))
    assert gap["reference_covered_days"] == 0
    assert artifacts["aewr_2009_nonstatutory_wages.parquet"]["is_statutory_aewr"].not_().all()
    skill = artifacts["aewr_occupation_skill_rates.parquet"]
    assert set(skill["worker_category"]) == {"US", "H2A"}
    assert set(skill["skill_level"]) == {1, 2}
    assert skill.filter(pl.col("hourly_aewr") <= 0).is_empty()
