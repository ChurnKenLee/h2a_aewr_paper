"""Conservation and missingness checks for allocated certification capacity."""

from datetime import date

import polars as pl

from h2a.worksite_exposure import (
    aggregate_exposure,
    attach_fls_reference,
    expand_months,
    prepare_worksites,
)


def fixture():
    rows = []
    for i, workers, begin, end, county in (
        (1, 60.0, "2020-01-31", "2020-02-01", "01001,01003"),
        (2, 40.0, "2020-02-01", "2020-02-29", "00000"),
    ):
        rows.append({"allocation_id": str(i), "case_number": "case", "fiscal_year": 2020,
            "worksite_requested_begin_date": begin, "worksite_requested_end_date": end,
            "certification_begin_date": None, "certification_end_date": None,
            "job_begin_date": "2020-01-01", "job_end_date": "2020-03-01",
            "requested_begin_date": None, "requested_end_date": None,
            "number_of_hours": "40", "case_nbr_workers_certified": 100.0,
            "master_n_workers_certified": None, "adjusted_nbr_workers_certified": workers,
            "county_fips_list": county, "allocation_method": "case_certifications_requested_staffing_observed"})
    return pl.DataFrame(rows)


def test_inclusive_site_days_county_shares_and_unmatched_conserve_mass():
    work, cases = prepare_worksites(fixture())
    assert cases["case_allocation_reconciled"].all()
    month = expand_months(work)
    assert month["allocated_certification_position_days"].sum() == 60 * 2 + 40 * 29
    assert month.filter(pl.col("county_fips") == "01001")["allocated_certification_position_days"].sum() == 60
    assert month.filter(pl.col("county_fips").is_null())["allocated_certification_position_days"].sum() == 1160
    assert month["date_basis"].unique().to_list() == ["observed_worksite_requested_dates"]


def test_unreconciled_case_is_audited_and_not_reallocated():
    bad = fixture().with_columns(pl.lit(99.0).alias("case_nbr_workers_certified"))
    work, cases = prepare_worksites(bad)
    assert not cases["case_allocation_reconciled"].any()
    assert cases["case_total_difference"].item() == 1
    assert expand_months(work).is_empty()


def test_legacy_equal_subcase_allocation_is_labeled_imputed_staffing():
    work, _ = prepare_worksites(fixture().with_columns(pl.lit(
        "distribute_master_workers_across_sub_entries_drop_master").alias("allocation_method")))
    month = expand_months(work)
    schedule = pl.DataFrame({"state_fips": ["01"], "effective_start": [date(2020, 1, 1)],
        "effective_end_exclusive": [date(2021, 1, 1)], "aewr": [10.0]})
    county = aggregate_exposure(attach_fls_reference(month, schedule))
    assert county["position_days_imputed_staffing"].sum() == 120
    assert county["allocated_certification_position_days"].sum() == 120


def test_date_fallback_and_missing_hours_remain_visible():
    source = fixture().with_columns(pl.lit(None).alias("worksite_requested_begin_date"),
                                    pl.lit(None).alias("number_of_hours"))
    work, _ = prepare_worksites(source)
    month = expand_months(work)
    assert month["date_basis"].unique().to_list() == ["case_date_fallback"]
    assert month["allocated_certification_hours_with_reported_case_schedule"].null_count() == month.height
    schedule = pl.DataFrame({"state_fips": ["01"], "effective_start": [date(2020, 2, 1)],
        "effective_end_exclusive": [date(2020, 3, 1)], "aewr": [10.0]})
    matched = attach_fls_reference(month, schedule)
    assert matched.filter(pl.col("month") == date(2020, 1, 1))["fls_reference_covered_days"].sum() == 0
    assert matched.filter((pl.col("county_fips") == "01001")
                          & (pl.col("month") == date(2020, 2, 1)))["fls_reference_rate_day_sum"].item() == 290
