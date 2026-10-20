"""Calendar allocation of certified positions, retaining source uncertainty.

Position-days describe allocated certification capacity, not actual employment.
The allocation itself has exactly one owner: B01's case/worksite producer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from h2a.aewr import assert_unique
from h2a.geography import assert_geo_columns, harmonize_county_fips_2010


def parsed_date(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.String).str.to_date("%Y-%m-%d", strict=False)


def valid_interval(start: str, end: str) -> pl.Expr:
    return (
        pl.col(start).is_not_null() & pl.col(end).is_not_null()
        & (pl.col(end) >= pl.col(start))
        & ((pl.col(start).dt.year() - pl.col("fiscal_year")).abs() <= 1)
        & ((pl.col(end).dt.year() - pl.col("fiscal_year")).abs() <= 1)
    ).fill_null(False)


def prepare_worksites(source: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    assert_unique(source, ["allocation_id"])
    work = source.with_columns(
        parsed_date("worksite_requested_begin_date").alias("site_begin"),
        parsed_date("worksite_requested_end_date").alias("site_end"),
        pl.coalesce(parsed_date("certification_begin_date"), parsed_date("job_begin_date"),
                    parsed_date("requested_begin_date")).alias("case_begin"),
        pl.coalesce(parsed_date("certification_end_date"), parsed_date("job_end_date"),
                    parsed_date("requested_end_date")).alias("case_end"),
        pl.col("number_of_hours").cast(pl.Float64, strict=False).alias("weekly_hours"),
        pl.coalesce("case_nbr_workers_certified", "master_n_workers_certified",
                    "adjusted_nbr_workers_certified").alias("case_target_candidate"),
    ).with_columns(
        valid_interval("site_begin", "site_end").alias("valid_site_interval"),
        valid_interval("case_begin", "case_end").alias("valid_case_interval"),
    ).with_columns(
        pl.when(pl.col("valid_site_interval")).then(pl.col("site_begin"))
        .when(pl.col("valid_case_interval")).then(pl.col("case_begin"))
        .otherwise(None).alias("exposure_begin"),
        pl.when(pl.col("valid_site_interval")).then(pl.col("site_end"))
        .when(pl.col("valid_case_interval")).then(pl.col("case_end"))
        .otherwise(None).alias("exposure_end"),
        pl.when(pl.col("valid_site_interval")).then(pl.lit("observed_worksite_requested_dates"))
        .when(pl.col("valid_case_interval")).then(pl.lit("case_date_fallback"))
        .otherwise(pl.lit("no_valid_interval")).alias("date_basis"),
        (pl.col("valid_site_interval") & pl.col("valid_case_interval")
         & ((pl.col("site_begin") != pl.col("case_begin"))
            | (pl.col("site_end") != pl.col("case_end")))).fill_null(False).alias("site_case_dates_differ"),
    )
    cases = work.group_by("case_number", "fiscal_year").agg(
        pl.len().alias("allocated_rows"),
        pl.col("case_target_candidate").drop_nulls().n_unique().alias("n_case_targets"),
        pl.col("case_target_candidate").first().alias("case_certified_target"),
        pl.col("adjusted_nbr_workers_certified").sum().alias("allocated_certified_positions"),
        (~pl.col("adjusted_nbr_workers_certified").is_finite()
         | (pl.col("adjusted_nbr_workers_certified") < 0)).fill_null(True).sum().alias("invalid_allocation_rows"),
        (~pl.col("valid_site_interval")).sum().alias("rows_without_valid_site_dates"),
        pl.col("site_case_dates_differ").sum().alias("rows_with_distinct_site_dates"),
    ).with_columns(
        (pl.col("allocated_certified_positions") - pl.col("case_certified_target")).alias("case_total_difference")
    ).with_columns(
        ((pl.col("n_case_targets") == 1) & (pl.col("invalid_allocation_rows") == 0)
         & (pl.col("case_total_difference").abs()
            <= 1e-8 * pl.max_horizontal(1.0, pl.col("case_certified_target").abs())))
        .fill_null(False).alias("case_allocation_reconciled"))
    work = work.join(cases.select("case_number", "fiscal_year", "case_allocation_reconciled"),
                     on=["case_number", "fiscal_year"], validate="m:1")
    return work, cases


def candidate_counties(value: str | None) -> list[str | None]:
    counties = sorted({harmonize_county_fips_2010(v.strip())
                       for v in (value or "").split(",") if v.strip() and v.strip() != "00000"})
    return counties or [None]


def expand_months(work: pl.DataFrame) -> pl.DataFrame:
    keep = ["allocation_id", "case_number", "fiscal_year", "county_fips_list",
            "adjusted_nbr_workers_certified", "weekly_hours", "date_basis",
            "valid_site_interval", "site_case_dates_differ", "allocation_method",
            "exposure_begin", "exposure_end"]
    month = work.filter(pl.col("exposure_begin").is_not_null()
                        & pl.col("case_allocation_reconciled")).select(keep).with_columns(
        pl.col("county_fips_list").map_elements(candidate_counties, return_dtype=pl.List(pl.String),
                                                 skip_nulls=False).alias("counties"),
        pl.date_ranges(pl.col("exposure_begin").dt.month_start(),
                       pl.col("exposure_end").dt.month_start(), interval="1mo").alias("month"),
    ).with_columns(pl.col("counties").list.len().alias("n_county_candidates")).explode("counties").rename(
        {"counties": "county_fips"}).drop("county_fips_list").explode("month").with_columns(
        pl.col("county_fips").str.slice(0, 2).alias("state_fips"),
        pl.max_horizontal("exposure_begin", "month").alias("segment_begin"),
        pl.min_horizontal("exposure_end", pl.col("month").dt.month_end()).alias("segment_end"),
        (pl.col("adjusted_nbr_workers_certified") / pl.col("n_county_candidates")).alias("allocated_positions"),
    ).with_columns(
        ((pl.col("segment_end") - pl.col("segment_begin")).dt.total_days() + 1).alias("segment_days"),
    ).with_columns(
        (pl.col("allocated_positions") * pl.col("segment_days")).alias("allocated_certification_position_days"),
    ).with_columns(
        pl.when(pl.col("weekly_hours").is_finite() & (pl.col("weekly_hours") > 0)
                & (pl.col("weekly_hours") <= 168))
        .then(pl.col("allocated_certification_position_days") * pl.col("weekly_hours") / 7)
        .otherwise(None).alias("allocated_certification_hours_with_reported_case_schedule"),
    )
    # Unknown county is retained as an explicit unmatched record, never a centroid.
    assert_geo_columns(month, ["county_fips", "state_fips"], allow_null=["county_fips", "state_fips"])
    if month.select("allocation_id", "county_fips", "month").is_duplicated().any():
        raise ValueError("Duplicated worksite/county/month exposure")
    expected = work.filter(pl.col("exposure_begin").is_not_null() & pl.col("case_allocation_reconciled")).select(
        (pl.col("adjusted_nbr_workers_certified")
         * ((pl.col("exposure_end") - pl.col("exposure_begin")).dt.total_days() + 1)).sum()).item()
    actual = month["allocated_certification_position_days"].sum()
    if abs(actual - expected) > 1e-8 * max(1, abs(expected)):
        raise ValueError("Calendar/county expansion does not conserve allocated position-days")
    return month


def aggregate_exposure(month: pl.DataFrame) -> pl.DataFrame:
    return month.filter(pl.col("county_fips").is_not_null()).with_columns(
        pl.col("month").dt.year().alias("year")).group_by("county_fips", "year").agg(
        pl.col("allocated_certification_position_days").sum(),
        pl.when(pl.col("valid_site_interval").any()).then(
            pl.col("allocated_certification_position_days").filter(pl.col("valid_site_interval")).sum()
        ).otherwise(None).alias("position_days_with_observed_worksite_dates"),
        pl.col("allocated_certification_position_days").filter(
            pl.col("date_basis") == "case_date_fallback").sum().alias("position_days_case_date_fallback"),
        pl.col("allocated_certification_position_days").filter(
            pl.col("n_county_candidates") > 1).sum().alias("position_days_ambiguous_county_split"),
        pl.col("allocated_certification_position_days").filter(
            pl.col("allocation_method").str.contains(
                "imputed|evenly|distribute_master_workers_across_sub_entries_drop_master"
            )).sum().alias("position_days_imputed_staffing"),
        pl.when(pl.col("allocated_certification_hours_with_reported_case_schedule").is_not_null().any())
        .then(pl.col("allocated_certification_hours_with_reported_case_schedule").sum()).otherwise(None)
        .alias("allocated_certification_hours_with_reported_case_schedule"),
        pl.col("allocated_certification_position_days").filter(
            pl.col("allocated_certification_hours_with_reported_case_schedule").is_not_null())
        .sum().alias("position_days_with_reported_weekly_hours"),
        (pl.col("allocated_positions") * pl.col("fls_reference_rate_day_sum")).sum()
        .alias("fls_reference_dollar_position_days"),
        (pl.col("allocated_positions") * pl.col("fls_reference_covered_days")).sum()
        .alias("position_days_with_fls_reference"),
    )


def attach_fls_reference(month: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """Match reference effective dates without asserting occupational eligibility."""
    indexed = month.with_row_index("exposure_row")
    if indexed.is_empty():
        return indexed.with_columns(pl.lit(0.0).alias("fls_reference_rate_day_sum"),
                                    pl.lit(0).alias("fls_reference_covered_days"))
    start, end = indexed["segment_begin"].min(), indexed["segment_end"].max()
    rates = schedule.filter((pl.col("effective_start") <= end)
                            & (pl.col("effective_end_exclusive") > start)).select(
        "state_fips", "effective_start", "effective_end_exclusive", "aewr")
    covered = indexed.select("exposure_row", "state_fips", "segment_begin", "segment_end").join(
        rates, on="state_fips", how="inner").with_columns(
        (pl.min_horizontal(pl.col("segment_end") + pl.duration(days=1), "effective_end_exclusive")
         - pl.max_horizontal("segment_begin", "effective_start")).dt.total_days().alias("overlap_days")
    ).filter(pl.col("overlap_days") > 0).group_by("exposure_row").agg(
        pl.col("overlap_days").sum().alias("fls_reference_covered_days"),
        (pl.col("overlap_days") * pl.col("aewr")).sum().alias("fls_reference_rate_day_sum"))
    result = indexed.join(covered, on="exposure_row", how="left", validate="1:1").with_columns(
        pl.col("fls_reference_covered_days").fill_null(0),
        pl.col("fls_reference_rate_day_sum").fill_null(0.0),
        pl.lit(True).alias("fls_reference_requires_job_cohort_and_competing_floor_review"))
    if result.filter(pl.col("fls_reference_covered_days") > pl.col("segment_days")).height:
        raise ValueError("Overlapping FLS references during monthly matching")
    return result.drop("exposure_row")


def build_worksite_exposure(intermediate: Path) -> dict:
    source_path = intermediate / "h2a_allocated_worksites.parquet"
    schedule_path = intermediate / "aewr_rate_schedule.parquet"
    schedule = pl.read_parquet(schedule_path)
    work, cases = prepare_worksites(pl.read_parquet(source_path))
    work.write_parquet(intermediate / "h2a_worksite_exposure_audit.parquet")
    cases.write_parquet(intermediate / "h2a_case_allocation_audit.parquet")
    destination = intermediate / "h2a_worksite_months"
    destination.mkdir(parents=True, exist_ok=True)
    summaries = []
    counts = []
    for year in sorted(work["fiscal_year"].unique().to_list()):
        month = expand_months(work.filter(pl.col("fiscal_year") == year))
        month = attach_fls_reference(month, schedule)
        month.write_parquet(destination / f"fy_{year}.parquet")
        summaries.append(aggregate_exposure(month))
        counts.append({"fiscal_year": year, "worksite_county_month_rows": month.height,
                       "position_days": month["allocated_certification_position_days"].sum(),
                       "unmatched_county_position_days": month.filter(pl.col("county_fips").is_null())[
                           "allocated_certification_position_days"].sum()})
    annual = pl.concat(summaries).group_by("county_fips", "year").agg(
        pl.exclude("county_fips", "year", "position_days_with_observed_worksite_dates",
                   "allocated_certification_hours_with_reported_case_schedule").sum(),
        pl.when(pl.col("position_days_with_observed_worksite_dates").is_not_null().any())
        .then(pl.col("position_days_with_observed_worksite_dates").sum()).otherwise(None)
        .alias("position_days_with_observed_worksite_dates"),
        pl.when(pl.col("allocated_certification_hours_with_reported_case_schedule").is_not_null().any())
        .then(pl.col("allocated_certification_hours_with_reported_case_schedule").sum()).otherwise(None)
        .alias("allocated_certification_hours_with_reported_case_schedule"),
    ).sort(["county_fips", "year"])
    assert_unique(annual, ["county_fips", "year"])
    annual.write_parquet(intermediate / "h2a_worksite_county_year_exposure.parquet")
    pl.DataFrame(counts).write_parquet(intermediate / "h2a_worksite_exposure_coverage.parquet")
    manifest = {
        "schema_version": "allocated_certification_calendar_v1",
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "fls_reference_schedule_sha256": hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
        "wage_reference": "day-weighted FLS reference only; never asserted to be the binding job wage",
        "measure": "allocated certification position-days, not actual workers, entries or employment",
        "date_convention": "inclusive requested site dates; explicit valid-case-date fallback",
        "date_guard": "source begin/end years within one year of fiscal year; no modal or typo correction",
        "county_allocation": "equal shares across distinct source county candidates; unmatched retained",
        "case_allocation": "existing B01 allocation; unreconciled/invalid cases excluded and audited",
        "sequential_worksites": "a crew may rotate; proportional certification allocation is not observed site staffing",
        "weekly_hours": "reported or case-inherited only; no mean imputation; bounds 0 < hours <= 168",
        "monthly_rows": sum(x["worksite_county_month_rows"] for x in counts),
        "unreconciled_cases": cases.filter(~pl.col("case_allocation_reconciled")).height,
        "annual_county_rows": annual.height,
    }
    (intermediate / "h2a_worksite_exposure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
