"""Regression checks for disclosure-era semantics and whole-pair fallbacks."""

import runpy

import polars as pl

from h2a.paths import CODE

notebook = runpy.run_path(str(CODE / "a01_sources/02_01_h2a_match_locations.py"))
_, date_helpers = notebook["disclosure_date_helpers"].run(pl=pl)
harmonize_disclosure_dates = date_helpers["harmonize_disclosure_dates"]


def source_row(year, **overrides):
    row = {"fiscal_year": year}
    for prefix in ("certification", "requested", "employment", "job_order"):
        row[f"{prefix}_begin_date"] = f"{year}-02-01"
        row[f"{prefix}_end_date"] = f"{year}-09-30"
    row.update(overrides)
    return row


def test_each_era_selects_its_employment_period_and_never_job_order_dates():
    rows = [
        source_row(year, job_order_begin_date=f"{year - 1}-12-01",
                   job_order_end_date=f"{year}-04-01",
                   requested_begin_date=f"{year}-03-01",
                   employment_begin_date=f"{year}-04-01")
        for year in (2008, 2009, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2025)
    ]
    result = harmonize_disclosure_dates(pl.DataFrame(rows))
    assert result["period_date_basis"].to_list() == [
        "certification_dates", "certification_dates", "certification_dates",
        "certification_dates", "legacy_need_dates", "legacy_need_dates",
        "granted_need_dates", "granted_need_dates", "employment_dates", "employment_dates",
    ]
    assert result["period_begin_date"].to_list() == [
        "2008-02-01", "2009-02-01", "2014-02-01", "2015-02-01",
        "2016-03-01", "2017-03-01", "2018-03-01", "2019-03-01",
        "2020-04-01", "2025-04-01",
    ]


def test_requested_fallback_uses_a_complete_pair_and_a_common_application_cohort():
    rows = [
        source_row(2020, employment_begin_date="2020-03-01", employment_end_date=""),
        source_row(2010, certification_begin_date="2010-10-01",
                   certification_end_date="2010-02-01"),
        source_row(2020, employment_begin_date="3020-01-01"),
        source_row(2020, employment_begin_date="2020-04-01",
                   requested_begin_date="2019-12-01"),
    ]
    result = harmonize_disclosure_dates(pl.DataFrame(rows))
    assert result["period_begin_date"].to_list() == [
        "2020-02-01", "2010-02-01", "2020-02-01", "2020-04-01",
    ]
    assert result["period_date_basis"].to_list() == [
        "requested_fallback", "requested_fallback", "requested_fallback", "employment_dates",
    ]
    assert result["application_begin_date"][-1] == "2019-12-01"


def test_missing_end_preserves_only_start_and_invalid_dates_stay_missing():
    rows = [
        source_row(2008, certification_end_date="", requested_begin_date="",
                   requested_end_date=""),
        source_row(2020, employment_begin_date="", employment_end_date="",
                   requested_begin_date="not-a-date", requested_end_date=""),
        source_row(2020, employment_begin_date="", requested_end_date=""),
    ]
    result = harmonize_disclosure_dates(pl.DataFrame(rows))
    assert result["period_begin_date"].to_list() == ["2008-02-01", None, "2020-02-01"]
    assert result["period_end_date"].to_list() == [None, None, None]
    assert result["period_date_basis"].to_list() == [
        "certification_dates_start_only", "missing", "requested_fallback_start_only",
    ]


def test_fy2026_disclosures_retain_calendar_2025_starts():
    result = harmonize_disclosure_dates(pl.DataFrame([
        source_row(2026, employment_begin_date="2025-11-15",
                   employment_end_date="2026-06-30",
                   requested_begin_date="2025-11-01"),
    ]))
    assert result["period_begin_date"].to_list() == ["2025-11-15"]
    assert result["period_date_basis"].to_list() == ["employment_dates"]
    assert result["application_begin_date"].to_list() == ["2025-11-01"]
