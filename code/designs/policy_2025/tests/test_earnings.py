"""Meaningful extraction edge cases; uses the exported notebook in test mode."""

import os
import runpy
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import polars as pl


def load_functions():
    os.environ["CD_EARNINGS_MODE"] = "test"
    return runpy.run_path(os.environ["CD_EXPORTED_EARNINGS"])


def expect_error(exception, text, function):
    try:
        function()
    except exception as error:
        assert text in str(error)
    else:
        raise AssertionError(f"Expected {exception.__name__}: {text}")


def row(area="01001", wage="100", employment="10", disclosure=""):
    return {"area_fips": area, "year": "2024", "qtr": "1", "industry_code": "111",
            "own_code": "5", "agglvl_code": "75", "month1_emplvl": employment,
            "month2_emplvl": employment, "month3_emplvl": employment,
            "avg_wkly_wage": wage, "total_qtrly_wages": "13000", "disclosure_code": disclosure}


def test_suppressed_wages_are_unavailable(functions, tmp_path):
    result = functions["aggregate_earnings"](pl.DataFrame([row(disclosure="N")]))
    assert result["disclosed"].to_list() == [False]
    assert result["avg_wkly_wage"].null_count() == 1
    assert result["weekly_wage_from_totals"].null_count() == 1


def test_nonpositive_denominator_and_geography(functions, tmp_path):
    result = functions["aggregate_earnings"](pl.DataFrame([row(employment="0"), row("01000"), row("01999")]))
    assert result["county_fips"].to_list() == ["01001"]
    assert result["county_fips"].dtype == pl.String
    assert result["avg_wkly_wage"].null_count() == 1


def test_duplicate_county_quarter_rejected(functions, tmp_path):
    expect_error(ValueError, "Duplicate", lambda: functions["aggregate_earnings"](pl.DataFrame([row(), row()])))


def test_canonical_county_priority(functions, tmp_path):
    result = functions["aggregate_earnings"](pl.DataFrame([row("46102", "100"), row("46113", "200")]))
    assert result["county_fips"].to_list() == ["46113"]
    assert result["avg_wkly_wage"].to_list() == [200.0]


def test_hash_mismatch_rejected(functions, tmp_path):
    manifest = tmp_path / "hashes"
    manifest.write_text("0" * 64 + "  code/designs/policy_2025/design.json\n")
    expect_error(ValueError, "Hash mismatch", lambda: functions["verify_hashes"](manifest))


def test_pre_policy_gate_precedes_metadata_and_acquisition(functions, tmp_path):
    if datetime.now(UTC) >= datetime(2026, 12, 2, 15, tzinfo=UTC):
        print("Calendar gate has opened; publication metadata is then required")
        return
    environment = dict(os.environ, CD_EARNINGS_MODE="post", POLICY2025_WORK=str(tmp_path))
    environment.pop("CD_RELEASE_METADATA", None)
    result = subprocess.run([sys.executable, os.environ["CD_EXPORTED_EARNINGS"]],
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "Awaiting 2026 Q2 publication" in result.stderr
    assert not list(tmp_path.iterdir())


def test_lead_rejects_unexpected_quarter_without_reading_history(functions, tmp_path):
    bad = row()
    bad.update(year="2025", qtr="4")
    pl.DataFrame([bad]).write_csv(tmp_path / "qcew_2025Q1_111.csv")
    os.environ["POLICY2025_WORK"] = str(tmp_path)
    os.environ["CD_EARNINGS_MODE"] = "lead"
    expect_error(ValueError, "unexpected quarters", lambda: runpy.run_path(os.environ["CD_EXPORTED_EARNINGS"]))
    assert not (tmp_path / "earnings_lead.parquet").exists()


if __name__ == "__main__":
    definitions = load_functions()
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        with TemporaryDirectory(prefix="chapter-earnings-test-") as folder, patch.dict(os.environ):
            test(definitions, Path(folder))
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} earnings edge checks passed")
