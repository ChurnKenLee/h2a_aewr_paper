from __future__ import annotations

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    """Recover FLS geographic weights from public composition and wage moments.

    This single script owns the one-off feature construction, entropy recovery, and
    artifact writes for the panel-IV design.
    """
    return


@app.cell
def _():
    import argparse
    import math
    import os
    import zlib
    from collections import defaultdict
    from collections.abc import Iterable, Sequence
    from pathlib import Path
    from typing import Any

    import numpy as np
    import polars as pl

    from h2a.paths import INTERMEDIATE

    return (
        Any,
        INTERMEDIATE,
        Iterable,
        Path,
        Sequence,
        argparse,
        defaultdict,
        math,
        np,
        os,
        pl,
        zlib,
    )


@app.cell
def _():
    SUPPORTED_YEARS = tuple(range(2011, 2022))
    BASELINE_WEIGHT_SPEC = "census_hired_workers_qcew_updated"
    # The supported FLS series starts in 2011.  Keep the 2022 terminal Census
    # benchmark available for temporal feature borrowing; earlier OEWS crosswalk
    # vintages are not needed and are incomplete for a small number of counties.
    FRAME_FEATURE_YEARS = tuple(range(2011, 2023))
    REFERENCE_QUARTERS = ("january", "april", "july", "october")
    QUARTER_NUMBER = {
        quarter: number for number, quarter in enumerate(REFERENCE_QUARTERS, start=1)
    }
    DURATION_CELLS = ("long", "short")
    MOMENT_SPEC = "fls_joint_quarter_duration_plus_field_livestock_wage"
    GEOGRAPHIC_ALLOCATION_SPEC = "oews_township_share_within_county"
    MINIMUM_FRAME_COVERAGE = 0.90
    MINIMUM_CONTRAST_SCALE = 1e-12
    BOUNDARY_EPSILON = 1e-9
    return (
        BASELINE_WEIGHT_SPEC,
        BOUNDARY_EPSILON,
        FRAME_FEATURE_YEARS,
        GEOGRAPHIC_ALLOCATION_SPEC,
        MINIMUM_CONTRAST_SCALE,
        MINIMUM_FRAME_COVERAGE,
        MOMENT_SPEC,
        QUARTER_NUMBER,
        REFERENCE_QUARTERS,
        SUPPORTED_YEARS,
    )


@app.cell
def _(
    Any,
    BASELINE_WEIGHT_SPEC,
    BOUNDARY_EPSILON,
    FRAME_FEATURE_YEARS,
    GEOGRAPHIC_ALLOCATION_SPEC,
    INTERMEDIATE,
    Iterable,
    MINIMUM_CONTRAST_SCALE,
    MINIMUM_FRAME_COVERAGE,
    MOMENT_SPEC,
    Path,
    QUARTER_NUMBER,
    REFERENCE_QUARTERS,
    SUPPORTED_YEARS,
    Sequence,
    argparse,
    defaultdict,
    math,
    np,
    os,
    pl,
    zlib,
):
    _AREA_KEYS = ["aewr_region_id", "source_year", "oews_area_code"]
    _AREA_QUARTER_KEYS = [*_AREA_KEYS, "qtr"]
    _COUNTY_YEAR_KEYS = ["county_fips", "source_year"]
    _COUNTY_QUARTER_KEYS = [*_COUNTY_YEAR_KEYS, "qtr"]
    _EXPECTED_INDUSTRIES = {"111", "112"}

    def _require_columns(
        frame: pl.DataFrame,
        columns: Iterable[str],
        label: str,
    ) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{label} is missing columns: {', '.join(missing)}")

    def _require_unique(
        frame: pl.DataFrame,
        keys: Sequence[str],
        label: str,
    ) -> None:
        duplicate_count = (
            frame.group_by(list(keys)).len().filter(pl.col("len") > 1).height
        )
        if duplicate_count:
            raise ValueError(
                f"{label} has {duplicate_count} duplicate cells on {', '.join(keys)}"
            )

    def _finite_nonnegative(value: Any) -> bool:
        return (
            value is not None
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) >= 0
        )

    def _finite_positive(value: Any) -> bool:
        return _finite_nonnegative(value) and float(value) > 0

    def _logit(value: float, epsilon: float = BOUNDARY_EPSILON) -> float:
        clipped = min(max(float(value), epsilon), 1 - epsilon)
        return math.log(clipped / (1 - clipped))

    def _expit(value: float) -> float:
        if value >= 0:
            inverse = math.exp(-value)
            return 1 / (1 + inverse)
        exponential = math.exp(value)
        return exponential / (1 + exponential)

    def _first_nonempty(values: Sequence[Any]) -> str | None:
        for value in values:
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    def build_fls_joint_targets(
        quarterly_workers: pl.DataFrame,
        *,
        years: Sequence[int] = SUPPORTED_YEARS,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Create normalized observed quarter-by-duration FLS target cells.

        A quarter is usable only when both duration counts are observed and
        nonnegative.  The separately reported all-worker count is retained only as
        a source/rounding diagnostic.
        """

        _require_columns(
            quarterly_workers,
            [
                "year",
                "quarter",
                "aewr_region_id",
                "region_name",
                "fls_hired_workers",
                "fls_hired_workers_150_days_or_more",
                "fls_hired_workers_149_days_or_less",
            ],
            "FLS quarterly workers",
        )
        years = tuple(int(year) for year in years)
        source = (
            quarterly_workers.filter(pl.col("year").cast(pl.Int32).is_in(years))
            .with_columns(
                pl.col("year").cast(pl.Int32).alias("source_year"),
                pl.col("aewr_region_id").cast(pl.String),
                pl.col("quarter").cast(pl.String).str.to_lowercase(),
            )
            .sort("aewr_region_id", "source_year", "quarter")
        )
        _require_unique(
            source,
            ["aewr_region_id", "source_year", "quarter"],
            "FLS quarterly workers",
        )

        target_rows: list[dict[str, Any]] = []
        diagnostic_rows: list[dict[str, Any]] = []
        for group in source.partition_by(
            ["aewr_region_id", "source_year"],
            maintain_order=True,
        ):
            first = group.row(0, named=True)
            region = first["aewr_region_id"]
            year = int(first["source_year"])
            by_quarter = {row["quarter"]: row for row in group.iter_rows(named=True)}
            usable: list[dict[str, Any]] = []
            for quarter in REFERENCE_QUARTERS:
                row = by_quarter.get(quarter)
                if row is None:
                    continue
                long_count = row["fls_hired_workers_150_days_or_more"]
                short_count = row["fls_hired_workers_149_days_or_less"]
                if not (
                    _finite_nonnegative(long_count) and _finite_nonnegative(short_count)
                ):
                    continue
                duration_total = float(long_count) + float(short_count)
                if duration_total <= 0:
                    continue
                usable.append(row)

            target_total = sum(
                float(row["fls_hired_workers_150_days_or_more"])
                + float(row["fls_hired_workers_149_days_or_less"])
                for row in usable
            )
            available = len(usable) >= 3 and target_total > 0
            status = (
                "available"
                if available
                else (
                    "fewer_than_three_usable_quarters"
                    if len(usable) < 3
                    else "nonpositive_target_total"
                )
            )
            all_worker_total = sum(
                float(row["fls_hired_workers"])
                for row in usable
                if _finite_nonnegative(row["fls_hired_workers"])
            )
            duration_all_discrepancy = (
                target_total - all_worker_total
                if all(_finite_nonnegative(row["fls_hired_workers"]) for row in usable)
                else math.nan
            )
            diagnostic_rows.append(
                {
                    "aewr_region_id": region,
                    "source_year": year,
                    "region_name": first["region_name"],
                    "usable_quarter_count": len(usable),
                    "joint_cell_count": 2 * len(usable),
                    "independent_contrast_count": max(2 * len(usable) - 1, 0),
                    "target_duration_count_total": target_total,
                    "reported_all_worker_count_total": (
                        all_worker_total
                        if all(
                            _finite_nonnegative(row["fls_hired_workers"])
                            for row in usable
                        )
                        else None
                    ),
                    "duration_minus_all_worker_count": (
                        duration_all_discrepancy
                        if math.isfinite(duration_all_discrepancy)
                        else None
                    ),
                    "duration_minus_all_worker_relative": (
                        duration_all_discrepancy / all_worker_total
                        if math.isfinite(duration_all_discrepancy)
                        and all_worker_total > 0
                        else None
                    ),
                    "target_status": status,
                    "moment_spec": MOMENT_SPEC,
                }
            )
            if not available:
                continue

            cell_index = 0
            for quarter in REFERENCE_QUARTERS:
                row = next(
                    (
                        candidate
                        for candidate in usable
                        if candidate["quarter"] == quarter
                    ),
                    None,
                )
                if row is None:
                    continue
                long_count = float(row["fls_hired_workers_150_days_or_more"])
                short_count = float(row["fls_hired_workers_149_days_or_less"])
                duration_total = long_count + short_count
                all_workers = row["fls_hired_workers"]
                for duration, count in (
                    ("long", long_count),
                    ("short", short_count),
                ):
                    target_rows.append(
                        {
                            "aewr_region_id": region,
                            "source_year": year,
                            "region_name": first["region_name"],
                            "quarter": quarter,
                            "qtr": QUARTER_NUMBER[quarter],
                            "duration": duration,
                            "cell_index": cell_index,
                            "fls_worker_count": count,
                            "fls_target_share": count / target_total,
                            "fls_duration_quarter_total": duration_total,
                            "fls_reported_all_worker_count": (
                                float(all_workers)
                                if _finite_nonnegative(all_workers)
                                else None
                            ),
                            "fls_duration_minus_all_worker_count": (
                                duration_total - float(all_workers)
                                if _finite_nonnegative(all_workers)
                                else None
                            ),
                            "usable_quarter_count": len(usable),
                            "joint_cell_count": 2 * len(usable),
                            "independent_contrast_count": 2 * len(usable) - 1,
                            "moment_spec": MOMENT_SPEC,
                        }
                    )
                    cell_index += 1

        if not diagnostic_rows:
            raise ValueError("No FLS region-year target rows were found")
        targets = pl.DataFrame(target_rows, infer_schema_length=None).sort(
            "aewr_region_id", "source_year", "cell_index"
        )
        diagnostics = pl.DataFrame(diagnostic_rows, infer_schema_length=None).sort(
            "aewr_region_id", "source_year"
        )
        return targets, diagnostics

    def build_county_area_prior(
        frame_employment: pl.DataFrame,
        area_definitions: pl.DataFrame,
        *,
        years: Sequence[int] = FRAME_FEATURE_YEARS,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Allocate county frame mass using within-county township shares."""

        _require_columns(
            frame_employment,
            [
                "county_fips",
                "aewr_region_id",
                "source_year",
                "weight_spec",
                "weight_draw_id",
                "frame_employment_mass",
            ],
            "FLS frame employment analog",
        )
        _require_columns(
            area_definitions,
            [
                "county_fips",
                "year",
                "oews_township_code",
                "oews_area_code",
                "oews_area_name",
            ],
            "OEWS area definitions",
        )
        years = tuple(int(year) for year in years)
        frame = (
            frame_employment.filter(
                pl.col("source_year").cast(pl.Int32).is_in(years),
                pl.col("weight_spec") == BASELINE_WEIGHT_SPEC,
                pl.col("weight_draw_id").is_null(),
            )
            .with_columns(
                pl.col("source_year").cast(pl.Int32),
                pl.col("county_fips").cast(pl.String),
                pl.col("aewr_region_id").cast(pl.String),
                pl.col("frame_employment_mass").cast(pl.Float64),
            )
            .select(
                "county_fips",
                "aewr_region_id",
                "source_year",
                "frame_employment_mass",
            )
            .sort("county_fips", "source_year")
        )
        _require_unique(frame, _COUNTY_YEAR_KEYS, "FLS frame employment analog")
        if (
            frame.filter(
                ~pl.col("frame_employment_mass").is_finite()
                | (pl.col("frame_employment_mass") < 0)
            ).height
            > 0
        ):
            raise ValueError("Frame employment mass must be finite and nonnegative")

        mapping = (
            area_definitions.with_columns(
                pl.col("year").cast(pl.Int32).alias("source_year"),
                pl.col("county_fips").cast(pl.String),
                pl.col("oews_township_code").cast(pl.String).str.strip_chars(),
                pl.col("oews_area_code").cast(pl.String).str.strip_chars(),
            )
            .filter(
                pl.col("source_year").is_in(years),
                pl.col("oews_township_code").is_not_null(),
                pl.col("oews_township_code") != "",
                pl.col("oews_area_code").is_not_null(),
                pl.col("oews_area_code") != "",
            )
            .select(
                "county_fips",
                "source_year",
                "oews_township_code",
                "oews_area_code",
                "oews_area_name",
            )
            .unique(
                subset=[
                    "county_fips",
                    "source_year",
                    "oews_township_code",
                    "oews_area_code",
                ],
                maintain_order=True,
            )
        )
        township_contract = (
            mapping.group_by("county_fips", "source_year", "oews_township_code")
            .agg(pl.col("oews_area_code").n_unique().alias("mapped_area_count"))
            .filter(pl.col("mapped_area_count") != 1)
        )
        if township_contract.height:
            raise ValueError("An OEWS township maps to multiple areas")

        mapping = (
            mapping.group_by("county_fips", "source_year", "oews_area_code")
            .agg(
                pl.col("oews_area_name").drop_nulls().first(),
                pl.col("oews_township_code")
                .n_unique()
                .alias("oews_area_mapped_townships"),
            )
            .with_columns(
                pl.col("oews_area_mapped_townships")
                .sum()
                .over("county_fips", "source_year")
                .alias("county_mapped_townships")
            )
            .with_columns(
                (
                    pl.col("oews_area_mapped_townships")
                    / pl.col("county_mapped_townships")
                ).alias("county_oews_area_share")
            )
        )
        share_contract = (
            mapping.group_by(_COUNTY_YEAR_KEYS)
            .agg(pl.col("county_oews_area_share").sum().alias("share_sum"))
            .filter((pl.col("share_sum") - 1).abs() > 1e-12)
        )
        if share_contract.height:
            raise ValueError("County-to-area township shares do not sum to one")

        missing_mapping = frame.filter(pl.col("frame_employment_mass") > 0).join(
            mapping.select(_COUNTY_YEAR_KEYS).unique(),
            on=_COUNTY_YEAR_KEYS,
            how="anti",
        )
        if missing_mapping.height:
            examples = ", ".join(
                f"{row['county_fips']}-{row['source_year']}"
                for row in missing_mapping.head(10).iter_rows(named=True)
            )
            raise ValueError(
                f"Positive frame counties lack an OEWS township mapping: {examples}"
            )

        allocation = (
            frame.join(
                mapping,
                on=_COUNTY_YEAR_KEYS,
                how="inner",
                validate="1:m",
            )
            .with_columns(
                (
                    pl.col("frame_employment_mass") * pl.col("county_oews_area_share")
                ).alias("baseline_county_area_mass")
            )
            .sort(
                "aewr_region_id",
                "source_year",
                "oews_area_code",
                "county_fips",
            )
        )
        allocation_contract = (
            allocation.group_by(_COUNTY_YEAR_KEYS)
            .agg(
                pl.col("frame_employment_mass").first().alias("county_mass"),
                pl.col("baseline_county_area_mass").sum().alias("allocated_mass"),
            )
            .with_columns(
                (pl.col("county_mass") - pl.col("allocated_mass")).abs().alias("gap")
            )
        )
        if allocation_contract.get_column("gap").max() > 1e-8:
            raise ValueError("County-to-area allocation does not conserve frame mass")

        area_prior = (
            allocation.group_by(_AREA_KEYS)
            .agg(
                pl.col("oews_area_name").drop_nulls().first(),
                pl.col("baseline_county_area_mass").sum().alias("area_frame_mass"),
                pl.col("county_fips").n_unique().alias("mapped_county_count"),
                pl.col("oews_area_mapped_townships")
                .sum()
                .alias("mapped_township_count"),
            )
            .with_columns(
                pl.col("area_frame_mass")
                .sum()
                .over("aewr_region_id", "source_year")
                .alias("region_frame_mass")
            )
            .with_columns(
                pl.when(
                    (pl.col("area_frame_mass") > 0) & (pl.col("region_frame_mass") > 0)
                )
                .then(pl.col("area_frame_mass") / pl.col("region_frame_mass"))
                .otherwise(0.0)
                .alias("frame_prior_weight"),
                (pl.col("area_frame_mass") > 0).alias("supported_frame"),
                pl.lit(BASELINE_WEIGHT_SPEC).alias("baseline_weight_spec"),
                pl.lit(GEOGRAPHIC_ALLOCATION_SPEC).alias("geographic_allocation_spec"),
            )
            .sort(_AREA_KEYS)
        )
        prior_contract = (
            area_prior.group_by("aewr_region_id", "source_year")
            .agg(
                pl.col("frame_prior_weight").sum().alias("prior_sum"),
                pl.col("region_frame_mass").first(),
            )
            .filter(
                (pl.col("region_frame_mass") > 0)
                & ((pl.col("prior_sum") - 1).abs() > 1e-12)
            )
        )
        if prior_contract.height:
            raise ValueError("Positive area priors do not sum to one")

        allocation = allocation.join(
            area_prior.select(
                *_AREA_KEYS,
                "area_frame_mass",
                "region_frame_mass",
                "frame_prior_weight",
                "supported_frame",
            ),
            on=_AREA_KEYS,
            how="left",
            validate="m:1",
        ).with_columns(
            pl.when(pl.col("area_frame_mass") > 0)
            .then(pl.col("baseline_county_area_mass") / pl.col("area_frame_mass"))
            .otherwise(None)
            .alias("baseline_county_conditional_within_area"),
            pl.lit(BASELINE_WEIGHT_SPEC).alias("baseline_weight_spec"),
            pl.lit(GEOGRAPHIC_ALLOCATION_SPEC).alias("geographic_allocation_spec"),
        )
        return allocation, area_prior

    def _aggregate_qcew_county_quarter(qcew: pl.DataFrame) -> pl.DataFrame:
        _require_columns(
            qcew,
            [
                "county_fips",
                "year",
                "qtr",
                "industry_code",
                "qcew_employment_disclosed",
                "qcew_reference_month_emplvl",
            ],
            "QCEW quarterly employment",
        )
        valid = (
            pl.col("qcew_employment_disclosed").fill_null(False)
            & pl.col("qcew_reference_month_emplvl").is_not_null()
            & pl.col("qcew_reference_month_emplvl").is_finite()
            & (pl.col("qcew_reference_month_emplvl") >= 0)
        )
        return (
            qcew.filter(
                pl.col("industry_code").cast(pl.String).is_in(_EXPECTED_INDUSTRIES),
                pl.col("qtr").cast(pl.Int32).is_between(1, 4),
            )
            .with_columns(
                pl.col("year").cast(pl.Int32).alias("source_year"),
                pl.col("qtr").cast(pl.Int32),
                pl.col("industry_code").cast(pl.String),
                pl.col("qcew_reference_month_emplvl").cast(pl.Float64),
            )
            .group_by(_COUNTY_QUARTER_KEYS)
            .agg(
                pl.len().alias("qcew_source_rows"),
                pl.col("industry_code").n_unique().alias("qcew_industry_count"),
                valid.sum().alias("qcew_valid_industry_count"),
                pl.when(valid)
                .then(pl.col("qcew_reference_month_emplvl"))
                .otherwise(None)
                .sum()
                .alias("_qcew_employment_sum"),
            )
            .with_columns(
                (
                    (pl.col("qcew_source_rows") == 2)
                    & (pl.col("qcew_industry_count") == 2)
                    & (pl.col("qcew_valid_industry_count") == 2)
                ).alias("qcew_complete")
            )
            .with_columns(
                pl.when(pl.col("qcew_complete"))
                .then(pl.col("_qcew_employment_sum"))
                .otherwise(None)
                .alias("qcew_ag_employment")
            )
            .drop("_qcew_employment_sum")
        )

    def _aggregate_qwi_county_quarter(qwi: pl.DataFrame) -> pl.DataFrame:
        _require_columns(
            qwi,
            [
                "county_fips",
                "year",
                "qtr",
                "industry_code",
                "qwi_beginning_quarter_employment",
                "qwi_stable_employment",
            ],
            "QWI quarterly employment",
        )
        valid_beginning = (
            pl.col("qwi_beginning_quarter_employment").is_not_null()
            & pl.col("qwi_beginning_quarter_employment").is_finite()
            & (pl.col("qwi_beginning_quarter_employment") >= 0)
        )
        valid_stable = (
            pl.col("qwi_stable_employment").is_not_null()
            & pl.col("qwi_stable_employment").is_finite()
            & (pl.col("qwi_stable_employment") >= 0)
        )
        return (
            qwi.filter(
                pl.col("industry_code").cast(pl.String).is_in(_EXPECTED_INDUSTRIES),
                pl.col("qtr").cast(pl.Int32).is_between(1, 4),
            )
            .with_columns(
                pl.col("year").cast(pl.Int32).alias("source_year"),
                pl.col("qtr").cast(pl.Int32),
                pl.col("industry_code").cast(pl.String),
                pl.col("qwi_beginning_quarter_employment").cast(pl.Float64),
                pl.col("qwi_stable_employment").cast(pl.Float64),
            )
            .group_by(_COUNTY_QUARTER_KEYS)
            .agg(
                pl.len().alias("qwi_source_rows"),
                pl.col("industry_code").n_unique().alias("qwi_industry_count"),
                valid_beginning.sum().alias("qwi_beginning_valid_count"),
                valid_stable.sum().alias("qwi_stable_valid_count"),
                pl.when(valid_beginning)
                .then(pl.col("qwi_beginning_quarter_employment"))
                .otherwise(None)
                .sum()
                .alias("_qwi_beginning_sum"),
                pl.when(valid_stable)
                .then(pl.col("qwi_stable_employment"))
                .otherwise(None)
                .sum()
                .alias("_qwi_stable_sum"),
            )
            .with_columns(
                (
                    (pl.col("qwi_source_rows") == 2)
                    & (pl.col("qwi_industry_count") == 2)
                    & (pl.col("qwi_beginning_valid_count") == 2)
                ).alias("qwi_beginning_complete"),
                (
                    (pl.col("qwi_source_rows") == 2)
                    & (pl.col("qwi_industry_count") == 2)
                    & (pl.col("qwi_beginning_valid_count") == 2)
                    & (pl.col("qwi_stable_valid_count") == 2)
                    & (pl.col("_qwi_beginning_sum") > 0)
                    & (pl.col("_qwi_stable_sum") <= pl.col("_qwi_beginning_sum"))
                ).alias("qwi_duration_complete"),
            )
            .with_columns(
                pl.when(pl.col("qwi_beginning_complete"))
                .then(pl.col("_qwi_beginning_sum"))
                .otherwise(None)
                .alias("qwi_beginning_employment"),
                pl.when(pl.col("qwi_duration_complete"))
                .then(pl.col("_qwi_stable_sum"))
                .otherwise(None)
                .alias("qwi_stable_employment"),
            )
            .drop("_qwi_beginning_sum", "_qwi_stable_sum")
        )

    def build_area_quarter_public_features(
        county_area_prior: pl.DataFrame,
        area_prior: pl.DataFrame,
        qcew: pl.DataFrame,
        qwi: pl.DataFrame,
        *,
        minimum_coverage: float = MINIMUM_FRAME_COVERAGE,
    ) -> pl.DataFrame:
        """Aggregate strict QCEW with QWI fills to region–OEWS areas."""

        qcew_county = _aggregate_qcew_county_quarter(qcew)
        qwi_county = _aggregate_qwi_county_quarter(qwi)
        skeleton = county_area_prior.join(
            pl.DataFrame(
                {
                    "qtr": list(range(1, 5)),
                    "quarter": list(REFERENCE_QUARTERS),
                }
            ),
            how="cross",
        )
        county_quarter = (
            skeleton.join(
                qcew_county,
                on=_COUNTY_QUARTER_KEYS,
                how="left",
                validate="m:1",
            )
            .join(
                qwi_county,
                on=_COUNTY_QUARTER_KEYS,
                how="left",
                validate="m:1",
            )
            .with_columns(
                pl.col("qcew_complete").fill_null(False),
                pl.col("qwi_beginning_complete").fill_null(False),
                pl.col("qwi_duration_complete").fill_null(False),
            )
            .with_columns(
                (
                    pl.col("qcew_complete")
                    | (~pl.col("qcew_complete") & pl.col("qwi_beginning_complete"))
                ).alias("public_employment_complete"),
                pl.coalesce("qcew_ag_employment", "qwi_beginning_employment").alias(
                    "public_ag_employment"
                ),
                (~pl.col("qcew_complete") & pl.col("qwi_beginning_complete")).alias(
                    "qwi_employment_fill"
                ),
            )
            .with_columns(
                (
                    pl.col("public_ag_employment") * pl.col("county_oews_area_share")
                ).alias("_allocated_public_employment"),
                (
                    pl.col("qwi_beginning_employment")
                    * pl.col("county_oews_area_share")
                ).alias("_allocated_qwi_beginning"),
                (
                    pl.col("qwi_stable_employment") * pl.col("county_oews_area_share")
                ).alias("_allocated_qwi_stable"),
            )
        )
        area_quarter = (
            county_quarter.group_by(_AREA_QUARTER_KEYS)
            .agg(
                pl.col("quarter").first(),
                pl.col("oews_area_name").drop_nulls().first(),
                pl.col("area_frame_mass").first(),
                pl.col("region_frame_mass").first(),
                pl.col("frame_prior_weight").first(),
                pl.col("supported_frame").first(),
                pl.when(pl.col("public_employment_complete"))
                .then(pl.col("_allocated_public_employment"))
                .otherwise(None)
                .sum()
                .alias("public_area_ag_employment_partial"),
                pl.when(pl.col("public_employment_complete"))
                .then(pl.col("baseline_county_area_mass"))
                .otherwise(0.0)
                .sum()
                .alias("employment_observed_frame_mass"),
                pl.when(pl.col("qcew_complete"))
                .then(pl.col("baseline_county_area_mass"))
                .otherwise(0.0)
                .sum()
                .alias("qcew_observed_frame_mass"),
                pl.when(pl.col("qwi_employment_fill"))
                .then(pl.col("baseline_county_area_mass"))
                .otherwise(0.0)
                .sum()
                .alias("qwi_fill_frame_mass"),
                pl.when(pl.col("qwi_duration_complete"))
                .then(pl.col("_allocated_qwi_beginning"))
                .otherwise(None)
                .sum()
                .alias("qwi_area_beginning_employment_partial"),
                pl.when(pl.col("qwi_duration_complete"))
                .then(pl.col("_allocated_qwi_stable"))
                .otherwise(None)
                .sum()
                .alias("qwi_area_stable_employment_partial"),
                pl.when(pl.col("qwi_duration_complete"))
                .then(pl.col("baseline_county_area_mass"))
                .otherwise(0.0)
                .sum()
                .alias("qwi_duration_observed_frame_mass"),
                pl.when(pl.col("public_employment_complete"))
                .then(pl.col("county_fips"))
                .otherwise(None)
                .n_unique()
                .alias("employment_observed_counties"),
                pl.col("county_fips").n_unique().alias("mapped_counties"),
            )
            .with_columns(
                pl.when(pl.col("area_frame_mass") > 0)
                .then(
                    pl.col("employment_observed_frame_mass") / pl.col("area_frame_mass")
                )
                .otherwise(None)
                .alias("employment_observed_frame_share"),
                pl.when(pl.col("area_frame_mass") > 0)
                .then(pl.col("qcew_observed_frame_mass") / pl.col("area_frame_mass"))
                .otherwise(None)
                .alias("qcew_observed_frame_share"),
                pl.when(pl.col("area_frame_mass") > 0)
                .then(pl.col("qwi_fill_frame_mass") / pl.col("area_frame_mass"))
                .otherwise(None)
                .alias("qwi_fill_frame_share"),
                pl.when(pl.col("area_frame_mass") > 0)
                .then(
                    pl.col("qwi_duration_observed_frame_mass")
                    / pl.col("area_frame_mass")
                )
                .otherwise(None)
                .alias("qwi_duration_observed_frame_share"),
            )
            .with_columns(
                pl.when(
                    (pl.col("area_frame_mass") > 0)
                    & (pl.col("employment_observed_frame_share") >= minimum_coverage)
                    & (pl.col("public_area_ag_employment_partial") > 0)
                )
                .then(
                    pl.col("public_area_ag_employment_partial")
                    / pl.col("area_frame_mass")
                )
                .otherwise(None)
                .alias("employment_intensity_raw"),
                pl.when(
                    (pl.col("qwi_duration_observed_frame_share") >= minimum_coverage)
                    & (pl.col("qwi_area_beginning_employment_partial") > 0)
                    & (pl.col("qwi_area_stable_employment_partial") >= 0)
                    & (
                        pl.col("qwi_area_stable_employment_partial")
                        <= pl.col("qwi_area_beginning_employment_partial")
                    )
                )
                .then(
                    pl.col("qwi_area_stable_employment_partial")
                    / pl.col("qwi_area_beginning_employment_partial")
                )
                .otherwise(None)
                .alias("qwi_persistence_raw"),
            )
            .with_columns(
                pl.when(pl.col("employment_intensity_raw").is_null())
                .then(None)
                .when(pl.col("qwi_fill_frame_mass") <= 1e-15)
                .then(pl.lit("qcew"))
                .when(pl.col("qcew_observed_frame_mass") <= 1e-15)
                .then(pl.lit("qwi_fill"))
                .otherwise(pl.lit("qcew_with_qwi_fill"))
                .alias("employment_raw_source")
            )
            .sort(_AREA_QUARTER_KEYS)
        )
        _require_unique(area_quarter, _AREA_QUARTER_KEYS, "area-quarter features")
        expected = area_prior.height * 4
        if area_quarter.height != expected:
            raise ValueError(
                "Area-quarter skeleton is incomplete: "
                f"expected {expected}, found {area_quarter.height}"
            )
        return area_quarter

    def build_census_area_duration(
        county_area_prior: pl.DataFrame,
        census_duration: pl.DataFrame,
        *,
        minimum_coverage: float = MINIMUM_FRAME_COVERAGE,
    ) -> pl.DataFrame:
        """Allocate published Census duration counts to OEWS areas."""

        _require_columns(
            census_duration,
            [
                "county_fips",
                "year",
                "census_hired_workers_150_days_or_more",
                "census_hired_workers_less_than_150_days",
                "census_hired_worker_duration_complete",
            ],
            "Census hired-worker duration",
        )
        census = census_duration.with_columns(
            pl.col("year").cast(pl.Int32).alias("source_year"),
            pl.col("county_fips").cast(pl.String),
            pl.col("census_hired_workers_150_days_or_more").cast(pl.Float64),
            pl.col("census_hired_workers_less_than_150_days").cast(pl.Float64),
        ).select(
            "county_fips",
            "source_year",
            "census_hired_workers_150_days_or_more",
            "census_hired_workers_less_than_150_days",
            "census_hired_worker_duration_complete",
        )
        _require_unique(census, _COUNTY_YEAR_KEYS, "Census hired-worker duration")
        joined = county_area_prior.join(
            census,
            on=_COUNTY_YEAR_KEYS,
            how="left",
            validate="m:1",
        )
        valid = (
            pl.col("census_hired_worker_duration_complete").fill_null(False)
            & pl.col("census_hired_workers_150_days_or_more").is_not_null()
            & pl.col("census_hired_workers_150_days_or_more").is_finite()
            & (pl.col("census_hired_workers_150_days_or_more") >= 0)
            & pl.col("census_hired_workers_less_than_150_days").is_not_null()
            & pl.col("census_hired_workers_less_than_150_days").is_finite()
            & (pl.col("census_hired_workers_less_than_150_days") >= 0)
        )
        return (
            joined.group_by(_AREA_KEYS)
            .agg(
                pl.col("area_frame_mass").first(),
                pl.when(valid)
                .then(
                    pl.col("census_hired_workers_150_days_or_more")
                    * pl.col("county_oews_area_share")
                )
                .otherwise(None)
                .sum()
                .alias("census_area_workers_long_partial"),
                pl.when(valid)
                .then(
                    pl.col("census_hired_workers_less_than_150_days")
                    * pl.col("county_oews_area_share")
                )
                .otherwise(None)
                .sum()
                .alias("census_area_workers_short_partial"),
                pl.when(valid)
                .then(pl.col("baseline_county_area_mass"))
                .otherwise(0.0)
                .sum()
                .alias("census_duration_observed_frame_mass"),
            )
            .with_columns(
                pl.when(pl.col("area_frame_mass") > 0)
                .then(
                    pl.col("census_duration_observed_frame_mass")
                    / pl.col("area_frame_mass")
                )
                .otherwise(None)
                .alias("census_duration_observed_frame_share"),
                (
                    pl.col("census_area_workers_long_partial")
                    + pl.col("census_area_workers_short_partial")
                ).alias("census_area_workers_duration_total_partial"),
            )
            .with_columns(
                pl.when(
                    (pl.col("census_duration_observed_frame_share") >= minimum_coverage)
                    & (pl.col("census_area_workers_duration_total_partial") > 0)
                )
                .then(
                    pl.col("census_area_workers_long_partial")
                    / pl.col("census_area_workers_duration_total_partial")
                )
                .otherwise(None)
                .alias("census_long_share_benchmark")
            )
            .sort(_AREA_KEYS)
        )

    def interpolate_census_duration(
        area_prior: pl.DataFrame,
        census_area_duration: pl.DataFrame,
    ) -> pl.DataFrame:
        """Interpolate Census long-duration shares within areas on the logit scale."""

        annual = area_prior.select(
            *_AREA_KEYS,
            "area_frame_mass",
            "frame_prior_weight",
            "supported_frame",
        ).join(
            census_area_duration.select(
                *_AREA_KEYS,
                "census_long_share_benchmark",
                "census_duration_observed_frame_share",
            ),
            on=_AREA_KEYS,
            how="left",
            validate="1:1",
        )
        output: list[dict[str, Any]] = []
        for group in annual.partition_by(
            ["aewr_region_id", "oews_area_code"],
            maintain_order=True,
        ):
            rows = sorted(
                group.iter_rows(named=True),
                key=lambda row: int(row["source_year"]),
            )
            observed = [
                row
                for row in rows
                if row["census_long_share_benchmark"] is not None
                and math.isfinite(float(row["census_long_share_benchmark"]))
                and 0 <= float(row["census_long_share_benchmark"]) <= 1
            ]
            for row in rows:
                value = row["census_long_share_benchmark"]
                source: str | None = None
                interpolated: float | None = None
                if value is not None and math.isfinite(float(value)):
                    interpolated = float(value)
                    source = "census_benchmark"
                else:
                    year = int(row["source_year"])
                    lower = [
                        candidate
                        for candidate in observed
                        if int(candidate["source_year"]) < year
                    ]
                    upper = [
                        candidate
                        for candidate in observed
                        if int(candidate["source_year"]) > year
                    ]
                    if lower and upper:
                        before = max(lower, key=lambda item: int(item["source_year"]))
                        after = min(upper, key=lambda item: int(item["source_year"]))
                        before_year = int(before["source_year"])
                        after_year = int(after["source_year"])
                        fraction = (year - before_year) / (after_year - before_year)
                        interpolated = _expit(
                            _logit(float(before["census_long_share_benchmark"]))
                            + fraction
                            * (
                                _logit(float(after["census_long_share_benchmark"]))
                                - _logit(float(before["census_long_share_benchmark"]))
                            )
                        )
                        source = "census_logit_interpolation"
                output.append(
                    {
                        **row,
                        "census_long_share_interpolated": interpolated,
                        "census_duration_source": source,
                    }
                )
        return pl.DataFrame(output, infer_schema_length=None).sort(_AREA_KEYS)

    def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        """Return the first value whose normalized cumulative weight reaches 0.5."""

        keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not np.any(keep):
            return math.nan
        values = values[keep]
        weights = weights[keep]
        order = np.argsort(values, kind="stable")
        values = values[order]
        weights = weights[order] / weights[order].sum()
        return float(values[np.flatnonzero(np.cumsum(weights) >= 0.5)[0]])

    def estimate_duration_odds_bridge(
        area_quarter_public: pl.DataFrame,
        census_area_duration: pl.DataFrame,
        *,
        boundary_epsilon: float = 1e-6,
    ) -> tuple[float, pl.DataFrame]:
        """Estimate odds(QWI persistence) / odds(Census long share)."""

        matched = area_quarter_public.join(
            census_area_duration.select(
                *_AREA_KEYS,
                "census_long_share_benchmark",
                "census_duration_observed_frame_share",
            ),
            on=_AREA_KEYS,
            how="inner",
            validate="m:1",
        ).filter(
            pl.col("qwi_persistence_raw").is_not_null(),
            pl.col("census_long_share_benchmark").is_not_null(),
            pl.col("qwi_persistence_raw").is_between(
                boundary_epsilon, 1 - boundary_epsilon
            ),
            pl.col("census_long_share_benchmark").is_between(
                boundary_epsilon, 1 - boundary_epsilon
            ),
            pl.col("area_frame_mass") > 0,
        )
        if matched.is_empty():
            ratio = math.nan
        else:
            qwi_share = matched.get_column("qwi_persistence_raw").to_numpy()
            census_share = matched.get_column("census_long_share_benchmark").to_numpy()
            odds_ratio = (
                qwi_share / (1 - qwi_share) / (census_share / (1 - census_share))
            )
            ratio = weighted_median(
                odds_ratio,
                matched.get_column("area_frame_mass").to_numpy(),
            )
        diagnostics = pl.DataFrame(
            {
                "bridge_method": ["frame_weighted_median_qwi_to_census_duration_odds"],
                "duration_odds_bridge_ratio": [ratio if math.isfinite(ratio) else None],
                "matched_area_quarter_cells": [matched.height],
                "matched_area_year_cells": [matched.select(_AREA_KEYS).unique().height],
                "minimum_frame_coverage": [MINIMUM_FRAME_COVERAGE],
                "uses_fls_data": [False],
            }
        )
        return ratio, diagnostics

    def apply_inverse_duration_odds_bridge(
        qwi_persistence: float,
        odds_bridge_ratio: float,
    ) -> float:
        """Translate QWI persistence to a 150-plus-day share analog."""

        if not (
            math.isfinite(qwi_persistence)
            and 0 <= qwi_persistence <= 1
            and math.isfinite(odds_bridge_ratio)
            and odds_bridge_ratio > 0
        ):
            return math.nan
        return _expit(_logit(qwi_persistence) - math.log(odds_bridge_ratio))

    def _linear_interpolation(
        year: int,
        before_year: int,
        before_value: float,
        after_year: int,
        after_value: float,
        *,
        scale: str,
    ) -> float | None:
        if before_year >= year or after_year <= year or before_year >= after_year:
            return None
        fraction = (year - before_year) / (after_year - before_year)
        if scale == "log":
            if before_value <= 0 or after_value <= 0:
                return None
            return math.exp(
                math.log(before_value)
                + fraction * (math.log(after_value) - math.log(before_value))
            )
        if scale == "logit":
            return _expit(
                _logit(before_value)
                + fraction * (_logit(after_value) - _logit(before_value))
            )
        raise ValueError(f"Unsupported interpolation scale: {scale}")

    def _hierarchical_impute(
        rows: list[dict[str, Any]],
        *,
        value_name: str,
        source_name: str,
        imputed_name: str,
        scale: str,
    ) -> None:
        """Apply the declared temporal/area hierarchy in place."""

        indices_by_area_quarter: defaultdict[tuple[str, str, int], list[int]] = (
            defaultdict(list)
        )
        for index, row in enumerate(rows):
            indices_by_area_quarter[
                (
                    row["aewr_region_id"],
                    row["oews_area_code"],
                    int(row["qtr"]),
                )
            ].append(index)

        # Stage 1: interior same-area interpolation from original public values.
        for indices in indices_by_area_quarter.values():
            indices.sort(key=lambda index: int(rows[index]["source_year"]))
            original = [
                index
                for index in indices
                if rows[index][value_name] is not None
                and math.isfinite(float(rows[index][value_name]))
            ]
            for index in indices:
                row = rows[index]
                if not row["supported_frame"] or (
                    row[value_name] is not None
                    and math.isfinite(float(row[value_name]))
                ):
                    continue
                year = int(row["source_year"])
                lower = [
                    candidate
                    for candidate in original
                    if int(rows[candidate]["source_year"]) < year
                ]
                upper = [
                    candidate
                    for candidate in original
                    if int(rows[candidate]["source_year"]) > year
                ]
                if not lower or not upper:
                    continue
                before = max(
                    lower, key=lambda candidate: int(rows[candidate]["source_year"])
                )
                after = min(
                    upper, key=lambda candidate: int(rows[candidate]["source_year"])
                )
                value = _linear_interpolation(
                    year,
                    int(rows[before]["source_year"]),
                    float(rows[before][value_name]),
                    int(rows[after]["source_year"]),
                    float(rows[after][value_name]),
                    scale=scale,
                )
                if value is not None and math.isfinite(value):
                    row[value_name] = value
                    row[source_name] = f"same_area_{scale}_linear_interpolation"
                    row[imputed_name] = True

        # Stage 2: nearest value from the same area and quarter within two years.
        for indices in indices_by_area_quarter.values():
            available = [
                index
                for index in indices
                if rows[index][value_name] is not None
                and math.isfinite(float(rows[index][value_name]))
            ]
            for index in indices:
                row = rows[index]
                if not row["supported_frame"] or (
                    row[value_name] is not None
                    and math.isfinite(float(row[value_name]))
                ):
                    continue
                year = int(row["source_year"])
                candidates = [
                    candidate
                    for candidate in available
                    if abs(int(rows[candidate]["source_year"]) - year) <= 2
                ]
                if not candidates:
                    continue
                nearest = min(
                    candidates,
                    key=lambda candidate: (
                        abs(int(rows[candidate]["source_year"]) - year),
                        int(rows[candidate]["source_year"]),
                    ),
                )
                row[value_name] = float(rows[nearest][value_name])
                row[source_name] = "same_area_nearest_within_2_years"
                row[imputed_name] = True

        # Stage 3: frame-prior-weighted region-year mean for the same quarter.
        indices_by_region_year_quarter: defaultdict[tuple[str, int, int], list[int]] = (
            defaultdict(list)
        )
        for index, row in enumerate(rows):
            indices_by_region_year_quarter[
                (
                    row["aewr_region_id"],
                    int(row["source_year"]),
                    int(row["qtr"]),
                )
            ].append(index)
        for indices in indices_by_region_year_quarter.values():
            observed = [
                index
                for index in indices
                if rows[index]["supported_frame"]
                and rows[index][value_name] is not None
                and math.isfinite(float(rows[index][value_name]))
                and float(rows[index]["frame_prior_weight"]) > 0
            ]
            weight_total = sum(
                float(rows[index]["frame_prior_weight"]) for index in observed
            )
            regional_mean = (
                sum(
                    float(rows[index]["frame_prior_weight"])
                    * float(rows[index][value_name])
                    for index in observed
                )
                / weight_total
                if weight_total > 0
                else None
            )
            for index in indices:
                row = rows[index]
                if not row["supported_frame"] or (
                    row[value_name] is not None
                    and math.isfinite(float(row[value_name]))
                ):
                    continue
                if regional_mean is not None and math.isfinite(regional_mean):
                    row[value_name] = regional_mean
                    row[source_name] = "frame_weighted_region_year_mean"
                    row[imputed_name] = True

    def build_imputed_area_quarter_features(
        area_quarter_public: pl.DataFrame,
        census_duration_interpolated: pl.DataFrame,
        odds_bridge_ratio: float,
    ) -> pl.DataFrame:
        """Build duration analogs and fill remaining public-feature gaps."""

        joined = area_quarter_public.join(
            census_duration_interpolated.select(
                *_AREA_KEYS,
                "census_long_share_interpolated",
                "census_duration_source",
                "census_duration_observed_frame_share",
            ),
            on=_AREA_KEYS,
            how="left",
            validate="m:1",
        )
        rows = joined.to_dicts()
        for row in rows:
            row["employment_intensity"] = row["employment_intensity_raw"]
            row["employment_feature_source"] = row["employment_raw_source"]
            row["employment_feature_imputed"] = False
            qwi_persistence = row["qwi_persistence_raw"]
            if (
                qwi_persistence is not None
                and math.isfinite(float(qwi_persistence))
                and math.isfinite(odds_bridge_ratio)
                and odds_bridge_ratio > 0
            ):
                row["duration_150_plus_analog"] = apply_inverse_duration_odds_bridge(
                    float(qwi_persistence), odds_bridge_ratio
                )
                row["duration_feature_source"] = "qwi_inverse_odds_bridge"
                row["duration_feature_imputed"] = False
            elif row["census_long_share_interpolated"] is not None and math.isfinite(
                float(row["census_long_share_interpolated"])
            ):
                row["duration_150_plus_analog"] = float(
                    row["census_long_share_interpolated"]
                )
                row["duration_feature_source"] = row["census_duration_source"]
                row["duration_feature_imputed"] = True
            else:
                row["duration_150_plus_analog"] = None
                row["duration_feature_source"] = None
                row["duration_feature_imputed"] = True

            if not row["supported_frame"]:
                row["employment_intensity"] = None
                row["employment_feature_source"] = "outside_supported_frame"
                row["employment_feature_imputed"] = False
                row["duration_150_plus_analog"] = None
                row["duration_feature_source"] = "outside_supported_frame"
                row["duration_feature_imputed"] = False

        _hierarchical_impute(
            rows,
            value_name="employment_intensity",
            source_name="employment_feature_source",
            imputed_name="employment_feature_imputed",
            scale="log",
        )
        _hierarchical_impute(
            rows,
            value_name="duration_150_plus_analog",
            source_name="duration_feature_source",
            imputed_name="duration_feature_imputed",
            scale="logit",
        )
        output = pl.DataFrame(rows, infer_schema_length=None).sort(_AREA_QUARTER_KEYS)
        return output.with_columns(
            pl.lit(odds_bridge_ratio).alias("duration_odds_bridge_ratio")
        )

    def helmert_basis(cell_count: int) -> np.ndarray:
        """Return a K by K-1 orthonormal Helmert contrast basis."""

        if cell_count < 1:
            raise ValueError("cell_count must be positive")
        basis = np.zeros((cell_count, max(cell_count - 1, 0)), dtype=float)
        for contrast in range(cell_count - 1):
            denominator = math.sqrt((contrast + 1) * (contrast + 2))
            basis[: contrast + 1, contrast] = 1 / denominator
            basis[contrast + 1, contrast] = -(contrast + 1) / denominator
        return basis

    def _area_quarter_lookup(
        area_quarter: pl.DataFrame,
    ) -> dict[tuple[str, int, str, int], dict[str, Any]]:
        return {
            (
                row["aewr_region_id"],
                int(row["source_year"]),
                row["oews_area_code"],
                int(row["qtr"]),
            ): row
            for row in area_quarter.iter_rows(named=True)
        }

    def assemble_joint_feature_artifact(
        targets: pl.DataFrame,
        target_diagnostics: pl.DataFrame,
        area_prior: pl.DataFrame,
        area_quarter: pl.DataFrame,
        *,
        years: Sequence[int] = SUPPORTED_YEARS,
        regions: Sequence[str] | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Assemble joint cells and prior-standardized Helmert contrasts."""

        years_set = {int(year) for year in years}
        regions_set = {str(region) for region in regions} if regions else None
        selected_targets = targets.filter(pl.col("source_year").is_in(years_set))
        selected_prior = area_prior.filter(pl.col("source_year").is_in(years_set))
        if regions_set is not None:
            selected_targets = selected_targets.filter(
                pl.col("aewr_region_id").is_in(regions_set)
            )
            selected_prior = selected_prior.filter(
                pl.col("aewr_region_id").is_in(regions_set)
            )
        public_lookup = _area_quarter_lookup(area_quarter)
        output_rows: list[dict[str, Any]] = []
        diagnostic_rows: list[dict[str, Any]] = []

        target_groups = {
            (group["aewr_region_id"][0], int(group["source_year"][0])): group
            for group in selected_targets.partition_by(
                ["aewr_region_id", "source_year"],
                maintain_order=True,
            )
        }
        prior_groups = {
            (group["aewr_region_id"][0], int(group["source_year"][0])): group
            for group in selected_prior.partition_by(
                ["aewr_region_id", "source_year"],
                maintain_order=True,
            )
        }
        selected_target_diagnostics = target_diagnostics.filter(
            pl.col("source_year").is_in(years_set)
        )
        if regions_set is not None:
            selected_target_diagnostics = selected_target_diagnostics.filter(
                pl.col("aewr_region_id").is_in(regions_set)
            )
        target_diagnostic_lookup = {
            (row["aewr_region_id"], int(row["source_year"])): row
            for row in selected_target_diagnostics.iter_rows(named=True)
        }

        all_keys = sorted(
            set(target_groups) | set(prior_groups),
            key=lambda key: (int(key[0]), key[1]),
        )
        for region, year in all_keys:
            target_group = target_groups.get((region, year))
            prior_group = prior_groups.get((region, year))
            target_diag = target_diagnostic_lookup.get((region, year), {})
            if target_group is None or prior_group is None:
                diagnostic_rows.append(
                    {
                        "aewr_region_id": region,
                        "source_year": year,
                        "feature_status": (
                            "missing_fls_target"
                            if target_group is None
                            else "missing_frame_prior"
                        ),
                        "joint_cell_count": (
                            target_group.height if target_group is not None else 0
                        ),
                        "active_contrast_count": 0,
                        "positive_prior_area_count": (
                            prior_group.filter(pl.col("frame_prior_weight") > 0).height
                            if prior_group is not None
                            else 0
                        ),
                        "imputed_prior_mass": None,
                        "moment_spec": MOMENT_SPEC,
                    }
                )
                continue

            target_rows = sorted(
                target_group.iter_rows(named=True),
                key=lambda row: int(row["cell_index"]),
            )
            area_rows = sorted(
                prior_group.iter_rows(named=True),
                key=lambda row: row["oews_area_code"],
            )
            supported_indices = [
                index
                for index, row in enumerate(area_rows)
                if float(row["frame_prior_weight"]) > 0
            ]
            if not supported_indices:
                diagnostic_rows.append(
                    {
                        "aewr_region_id": region,
                        "source_year": year,
                        "feature_status": "nonpositive_region_frame_mass",
                        "joint_cell_count": len(target_rows),
                        "active_contrast_count": 0,
                        "positive_prior_area_count": 0,
                        "imputed_prior_mass": None,
                        "moment_spec": MOMENT_SPEC,
                    }
                )
                continue

            cell_count = len(target_rows)
            tau = np.asarray(
                [float(row["fls_target_share"]) for row in target_rows],
                dtype=float,
            )
            area_compositions = np.full(
                (len(area_rows), cell_count), np.nan, dtype=float
            )
            area_quarter_values: dict[tuple[int, int], dict[str, Any]] = {}
            cell_failure = False
            for area_index, area in enumerate(area_rows):
                if area_index not in supported_indices:
                    continue
                quarters = sorted({int(row["qtr"]) for row in target_rows})
                public_rows: dict[int, dict[str, Any]] = {}
                for qtr in quarters:
                    public = public_lookup.get(
                        (region, year, area["oews_area_code"], qtr)
                    )
                    if public is None:
                        cell_failure = True
                        continue
                    intensity = public["employment_intensity"]
                    duration = public["duration_150_plus_analog"]
                    if not (
                        _finite_positive(intensity)
                        and duration is not None
                        and math.isfinite(float(duration))
                        and 0 <= float(duration) <= 1
                    ):
                        cell_failure = True
                        continue
                    public_rows[qtr] = public
                    area_quarter_values[(area_index, qtr)] = public
                if len(public_rows) != len(quarters):
                    continue
                intensity_total = sum(
                    float(public_rows[qtr]["employment_intensity"]) for qtr in quarters
                )
                if intensity_total <= 0:
                    cell_failure = True
                    continue
                seasonal_share = {
                    qtr: float(public_rows[qtr]["employment_intensity"])
                    / intensity_total
                    for qtr in quarters
                }
                for target in target_rows:
                    cell_index = int(target["cell_index"])
                    qtr = int(target["qtr"])
                    duration_share = float(public_rows[qtr]["duration_150_plus_analog"])
                    area_compositions[area_index, cell_index] = seasonal_share[qtr] * (
                        duration_share
                        if target["duration"] == "long"
                        else 1 - duration_share
                    )

            supported_compositions = area_compositions[supported_indices]
            if (
                cell_failure
                or not np.all(np.isfinite(supported_compositions))
                or not np.allclose(
                    supported_compositions.sum(axis=1),
                    1,
                    atol=1e-12,
                    rtol=0,
                )
            ):
                diagnostic_rows.append(
                    {
                        "aewr_region_id": region,
                        "source_year": year,
                        "feature_status": "public_feature_imputation_failed",
                        "joint_cell_count": cell_count,
                        "active_contrast_count": 0,
                        "positive_prior_area_count": len(supported_indices),
                        "imputed_prior_mass": None,
                        "moment_spec": MOMENT_SPEC,
                    }
                )
                continue

            prior_weight = np.asarray(
                [
                    float(area_rows[index]["frame_prior_weight"])
                    for index in supported_indices
                ],
                dtype=float,
            )
            prior_weight /= prior_weight.sum()
            basis = helmert_basis(cell_count)
            area_contrasts = area_compositions @ basis
            target_contrasts = tau @ basis
            supported_contrasts = area_contrasts[supported_indices]
            contrast_centers = prior_weight @ supported_contrasts
            contrast_scales = np.sqrt(
                prior_weight @ (supported_contrasts - contrast_centers) ** 2
            )
            active = np.isfinite(contrast_scales) & (
                contrast_scales > MINIMUM_CONTRAST_SCALE
            )

            employment_imputed_by_area = np.zeros(len(area_rows), dtype=bool)
            duration_imputed_by_area = np.zeros(len(area_rows), dtype=bool)
            minimum_employment_coverage = math.inf
            minimum_duration_coverage = math.inf
            quarter_employment_imputed_mass: list[float] = []
            quarter_duration_imputed_mass: list[float] = []
            for qtr in sorted({int(row["qtr"]) for row in target_rows}):
                employment_flags = []
                duration_flags = []
                for area_index in supported_indices:
                    public = area_quarter_values[(area_index, qtr)]
                    employment_flag = bool(public["employment_feature_imputed"])
                    duration_flag = bool(public["duration_feature_imputed"])
                    employment_flags.append(employment_flag)
                    duration_flags.append(duration_flag)
                    employment_imputed_by_area[area_index] |= employment_flag
                    duration_imputed_by_area[area_index] |= duration_flag
                    coverage = public["employment_observed_frame_share"]
                    if coverage is not None and math.isfinite(float(coverage)):
                        minimum_employment_coverage = min(
                            minimum_employment_coverage, float(coverage)
                        )
                    duration_coverage = public["qwi_duration_observed_frame_share"]
                    if duration_coverage is not None and math.isfinite(
                        float(duration_coverage)
                    ):
                        minimum_duration_coverage = min(
                            minimum_duration_coverage,
                            float(duration_coverage),
                        )
                quarter_employment_imputed_mass.append(
                    float(prior_weight @ np.asarray(employment_flags, dtype=float))
                )
                quarter_duration_imputed_mass.append(
                    float(prior_weight @ np.asarray(duration_flags, dtype=float))
                )
            supported_employment_imputed = employment_imputed_by_area[supported_indices]
            supported_duration_imputed = duration_imputed_by_area[supported_indices]
            supported_any_imputed = (
                supported_employment_imputed | supported_duration_imputed
            )
            employment_imputed_mass = float(
                prior_weight @ supported_employment_imputed.astype(float)
            )
            duration_imputed_mass = float(
                prior_weight @ supported_duration_imputed.astype(float)
            )
            any_imputed_mass = float(prior_weight @ supported_any_imputed.astype(float))

            common = {
                "weight_spec": "fls_realized_geography_dirichlet_entropy",
                "baseline_weight_spec": BASELINE_WEIGHT_SPEC,
                "moment_spec": MOMENT_SPEC,
                "wage_target_used": True,
                "geographic_allocation_spec": GEOGRAPHIC_ALLOCATION_SPEC,
                "feature_status": "available",
            }
            for area_index, area in enumerate(area_rows):
                supported = area_index in supported_indices
                target_quarters = sorted({int(row["qtr"]) for row in target_rows})
                area_intensity_total = (
                    sum(
                        float(
                            area_quarter_values[(area_index, target_qtr)][
                                "employment_intensity"
                            ]
                        )
                        for target_qtr in target_quarters
                    )
                    if supported
                    else None
                )
                for target in target_rows:
                    cell_index = int(target["cell_index"])
                    qtr = int(target["qtr"])
                    public = (
                        area_quarter_values.get((area_index, qtr))
                        if supported
                        else None
                    )
                    output_rows.append(
                        {
                            "aewr_region_id": region,
                            "source_year": year,
                            "oews_area_code": area["oews_area_code"],
                            "oews_area_name": area["oews_area_name"],
                            "feature_row_type": "joint_cell",
                            "frame_prior_weight": float(area["frame_prior_weight"]),
                            "area_frame_mass": float(area["area_frame_mass"]),
                            "region_frame_mass": float(area["region_frame_mass"]),
                            "supported_frame": supported,
                            "quarter": target["quarter"],
                            "qtr": qtr,
                            "duration": target["duration"],
                            "cell_index": cell_index,
                            "fls_worker_count": float(target["fls_worker_count"]),
                            "fls_target_share": float(target["fls_target_share"]),
                            "fls_reported_all_worker_count": target[
                                "fls_reported_all_worker_count"
                            ],
                            "fls_duration_minus_all_worker_count": target[
                                "fls_duration_minus_all_worker_count"
                            ],
                            "seasonal_employment_share": (
                                float(
                                    public["employment_intensity"]
                                    / area_intensity_total
                                )
                                if supported and public is not None
                                else None
                            ),
                            "duration_150_plus_analog": (
                                float(public["duration_150_plus_analog"])
                                if public is not None
                                else None
                            ),
                            "area_joint_share": (
                                float(area_compositions[area_index, cell_index])
                                if supported
                                else None
                            ),
                            "employment_intensity": (
                                float(public["employment_intensity"])
                                if public is not None
                                else None
                            ),
                            "employment_feature_source": (
                                public["employment_feature_source"]
                                if public is not None
                                else "outside_supported_frame"
                            ),
                            "duration_feature_source": (
                                public["duration_feature_source"]
                                if public is not None
                                else "outside_supported_frame"
                            ),
                            "employment_feature_imputed": (
                                bool(public["employment_feature_imputed"])
                                if public is not None
                                else False
                            ),
                            "duration_feature_imputed": (
                                bool(public["duration_feature_imputed"])
                                if public is not None
                                else False
                            ),
                            "employment_observed_frame_share": (
                                public["employment_observed_frame_share"]
                                if public is not None
                                else None
                            ),
                            "qcew_observed_frame_share": (
                                public["qcew_observed_frame_share"]
                                if public is not None
                                else None
                            ),
                            "qwi_fill_frame_share": (
                                public["qwi_fill_frame_share"]
                                if public is not None
                                else None
                            ),
                            "qwi_duration_observed_frame_share": (
                                public["qwi_duration_observed_frame_share"]
                                if public is not None
                                else None
                            ),
                            "census_duration_observed_frame_share": (
                                public["census_duration_observed_frame_share"]
                                if public is not None
                                else None
                            ),
                            "duration_odds_bridge_ratio": (
                                public["duration_odds_bridge_ratio"]
                                if public is not None
                                else None
                            ),
                            "contrast_id": None,
                            "area_helmert_contrast": None,
                            "target_helmert_contrast": None,
                            "contrast_prior_center": None,
                            "contrast_prior_scale": None,
                            "area_standardized_contrast": None,
                            "target_standardized_contrast": None,
                            "contrast_active": None,
                            "contrast_drop_reason": None,
                            **common,
                        }
                    )
                for contrast_index in range(cell_count - 1):
                    scale = float(contrast_scales[contrast_index])
                    is_active = bool(active[contrast_index])
                    output_rows.append(
                        {
                            "aewr_region_id": region,
                            "source_year": year,
                            "oews_area_code": area["oews_area_code"],
                            "oews_area_name": area["oews_area_name"],
                            "feature_row_type": "helmert_contrast",
                            "frame_prior_weight": float(area["frame_prior_weight"]),
                            "area_frame_mass": float(area["area_frame_mass"]),
                            "region_frame_mass": float(area["region_frame_mass"]),
                            "supported_frame": supported,
                            "quarter": None,
                            "qtr": None,
                            "duration": None,
                            "cell_index": None,
                            "fls_worker_count": None,
                            "fls_target_share": None,
                            "fls_reported_all_worker_count": None,
                            "fls_duration_minus_all_worker_count": None,
                            "seasonal_employment_share": None,
                            "duration_150_plus_analog": None,
                            "area_joint_share": None,
                            "employment_intensity": None,
                            "employment_feature_source": None,
                            "duration_feature_source": None,
                            "employment_feature_imputed": None,
                            "duration_feature_imputed": None,
                            "employment_observed_frame_share": None,
                            "qcew_observed_frame_share": None,
                            "qwi_fill_frame_share": None,
                            "qwi_duration_observed_frame_share": None,
                            "census_duration_observed_frame_share": None,
                            "duration_odds_bridge_ratio": None,
                            "contrast_id": contrast_index + 1,
                            "area_helmert_contrast": (
                                float(area_contrasts[area_index, contrast_index])
                                if supported
                                else None
                            ),
                            "target_helmert_contrast": float(
                                target_contrasts[contrast_index]
                            ),
                            "contrast_prior_center": float(
                                contrast_centers[contrast_index]
                            ),
                            "contrast_prior_scale": scale,
                            "area_standardized_contrast": (
                                float(
                                    (
                                        area_contrasts[area_index, contrast_index]
                                        - contrast_centers[contrast_index]
                                    )
                                    / scale
                                )
                                if supported and is_active
                                else None
                            ),
                            "target_standardized_contrast": (
                                float(
                                    (
                                        target_contrasts[contrast_index]
                                        - contrast_centers[contrast_index]
                                    )
                                    / scale
                                )
                                if is_active
                                else None
                            ),
                            "contrast_active": is_active,
                            "contrast_drop_reason": (
                                None
                                if is_active
                                else "effectively_zero_cross_area_variation"
                            ),
                            **common,
                        }
                    )

            diagnostic_rows.append(
                {
                    "aewr_region_id": region,
                    "source_year": year,
                    "feature_status": "available",
                    "usable_quarter_count": int(
                        target_diag.get("usable_quarter_count", cell_count // 2)
                    ),
                    "joint_cell_count": cell_count,
                    "independent_contrast_count": cell_count - 1,
                    "active_contrast_count": int(active.sum()),
                    "dropped_contrast_count": int((~active).sum()),
                    "positive_prior_area_count": len(supported_indices),
                    "outside_frame_area_count": (
                        len(area_rows) - len(supported_indices)
                    ),
                    "frame_prior_sum": float(prior_weight.sum()),
                    "fls_target_sum": float(tau.sum()),
                    "maximum_area_joint_sum_gap": float(
                        np.max(np.abs(supported_compositions.sum(axis=1) - 1))
                    ),
                    "employment_imputed_prior_mass": employment_imputed_mass,
                    "duration_imputed_prior_mass": duration_imputed_mass,
                    "imputed_prior_mass": any_imputed_mass,
                    "maximum_quarter_employment_imputed_prior_mass": max(
                        quarter_employment_imputed_mass
                    ),
                    "maximum_quarter_duration_imputed_prior_mass": max(
                        quarter_duration_imputed_mass
                    ),
                    "minimum_employment_observed_frame_share": (
                        minimum_employment_coverage
                        if math.isfinite(minimum_employment_coverage)
                        else None
                    ),
                    "minimum_qwi_duration_observed_frame_share": (
                        minimum_duration_coverage
                        if math.isfinite(minimum_duration_coverage)
                        else None
                    ),
                    "target_duration_count_total": target_diag.get(
                        "target_duration_count_total"
                    ),
                    "reported_all_worker_count_total": target_diag.get(
                        "reported_all_worker_count_total"
                    ),
                    "duration_minus_all_worker_count": target_diag.get(
                        "duration_minus_all_worker_count"
                    ),
                    "duration_minus_all_worker_relative": target_diag.get(
                        "duration_minus_all_worker_relative"
                    ),
                    "weight_spec": "fls_realized_geography_dirichlet_entropy",
                    "baseline_weight_spec": BASELINE_WEIGHT_SPEC,
                    "moment_spec": MOMENT_SPEC,
                    "wage_target_used": True,
                }
            )

        features = (
            pl.DataFrame(output_rows, infer_schema_length=None).sort(
                "aewr_region_id",
                "source_year",
                "oews_area_code",
                "feature_row_type",
                "cell_index",
                "contrast_id",
            )
            if output_rows
            else pl.DataFrame()
        )
        diagnostics = (
            pl.DataFrame(diagnostic_rows, infer_schema_length=None).sort(
                "aewr_region_id", "source_year"
            )
            if diagnostic_rows
            else pl.DataFrame()
        )
        return features, diagnostics

    def build_feature_artifacts(
        *,
        frame_employment: pl.DataFrame,
        area_definitions: pl.DataFrame,
        quarterly_workers: pl.DataFrame,
        qcew: pl.DataFrame,
        qwi: pl.DataFrame,
        census_duration: pl.DataFrame,
        years: Sequence[int] = SUPPORTED_YEARS,
        regions: Sequence[str] | None = None,
    ) -> dict[str, pl.DataFrame]:
        """Build the composition features and frame priors used below."""

        targets, target_diagnostics = build_fls_joint_targets(
            quarterly_workers, years=years
        )
        county_area_prior, area_prior = build_county_area_prior(
            frame_employment,
            area_definitions,
            years=FRAME_FEATURE_YEARS,
        )
        area_quarter_public = build_area_quarter_public_features(
            county_area_prior,
            area_prior,
            qcew,
            qwi,
        )
        census_area = build_census_area_duration(
            county_area_prior,
            census_duration,
        )
        census_interpolated = interpolate_census_duration(
            area_prior,
            census_area,
        )
        odds_bridge_ratio, bridge_diagnostics = estimate_duration_odds_bridge(
            area_quarter_public,
            census_area,
        )
        if not math.isfinite(odds_bridge_ratio) or odds_bridge_ratio <= 0:
            raise ValueError(
                "No positive public QWI-to-Census duration odds bridge could be estimated"
            )
        area_quarter_imputed = build_imputed_area_quarter_features(
            area_quarter_public,
            census_interpolated,
            odds_bridge_ratio,
        )
        features, feature_diagnostics = assemble_joint_feature_artifact(
            targets,
            target_diagnostics,
            area_prior,
            area_quarter_imputed,
            years=years,
            regions=regions,
        )

        years_set = {int(year) for year in years}
        selected_area_prior = area_prior.filter(pl.col("source_year").is_in(years_set))
        selected_county_area = county_area_prior.filter(
            pl.col("source_year").is_in(years_set)
        )
        selected_target_diagnostics = target_diagnostics.filter(
            pl.col("source_year").is_in(years_set)
        )
        if regions is not None:
            regions_set = {str(region) for region in regions}
            selected_area_prior = selected_area_prior.filter(
                pl.col("aewr_region_id").is_in(regions_set)
            )
            selected_county_area = selected_county_area.filter(
                pl.col("aewr_region_id").is_in(regions_set)
            )
            selected_target_diagnostics = selected_target_diagnostics.filter(
                pl.col("aewr_region_id").is_in(regions_set)
            )

        return {
            "features": features,
            "county_area_prior": selected_county_area,
            "area_prior": selected_area_prior,
            "target_diagnostics": selected_target_diagnostics,
            "feature_diagnostics": feature_diagnostics,
            "bridge_diagnostics": bridge_diagnostics,
        }

    def expand_area_weights_to_counties(
        area_weights: pl.DataFrame,
        county_area_prior: pl.DataFrame,
        *,
        weight_column: str = "oews_area_weight",
    ) -> pl.DataFrame:
        """Expand area weights with the fixed baseline conditional county mass."""

        keys = ["aewr_region_id", "source_year", "oews_area_code"]
        _require_columns(area_weights, [*keys, weight_column], "area weights")
        _require_columns(
            county_area_prior,
            [*keys, "county_fips", "baseline_county_conditional_within_area"],
            "county-area prior",
        )
        draw_keys = [
            column
            for column in ("specification", "weight_draw_id")
            if column in area_weights.columns
        ]
        expanded = area_weights.join(
            county_area_prior.select(
                *keys,
                "county_fips",
                "baseline_county_conditional_within_area",
            ),
            on=keys,
            how="inner",
            validate="m:m" if draw_keys else "1:m",
        ).with_columns(
            (
                pl.col(weight_column)
                * pl.col("baseline_county_conditional_within_area")
            ).alias("county_area_weight")
        )
        check_keys = [*keys, *draw_keys]
        contract = (
            expanded.group_by(check_keys)
            .agg(
                pl.col(weight_column).first().alias("area_weight"),
                pl.col("county_area_weight").sum().alias("expanded_weight"),
            )
            .filter((pl.col("area_weight") - pl.col("expanded_weight")).abs() > 1e-12)
        )
        if contract.height:
            raise ValueError("County expansion does not reproduce area weights")
        return expanded

    WEIGHT_SPEC = "fls_realized_geography_dirichlet_entropy"
    RHO_VALUES = (0.01, 0.03, 0.10, 0.30, 1.00)
    KAPPA_MULTIPLIERS = (2.0, 5.0, 10.0, 20.0)
    PRIMARY_RHO = 0.10
    PRIMARY_KAPPA_MULTIPLIER = 10.0
    PRIMARY_SPECIFICATION = "fls_geo_field_livestock_dirichlet_m10_rho010"
    PRIMARY_DRAW_COUNT = 999
    SENSITIVITY_DRAW_COUNT = 199
    SIMULATION_SEED = 20260726
    WEIGHT_SUM_TOLERANCE = 1e-10
    OPTIMIZER_GRADIENT_TOLERANCE = 1e-10
    OPTIMIZER_MAX_ITERATIONS = 80
    BIG_SIX_OCC_CODES = (
        "45-2041",
        "45-2091",
        "45-2092",
        "45-2093",
        "53-7064",
        "45-2099",
        "79011",
        "79021",
        "79856",
        "79858",
        "98902",
    )

    def rho_code(rho: float) -> str:
        return f"{int(round(100 * rho)):03d}"

    def multiplier_code(multiplier: float) -> str:
        if float(multiplier).is_integer():
            return f"{int(multiplier):02d}"
        return str(multiplier).replace(".", "p")

    def specification_label(multiplier: float, rho: float) -> str:
        return (
            "fls_geo_field_livestock_dirichlet_"
            f"m{multiplier_code(multiplier)}_rho{rho_code(rho)}"
        )

    def specification_grid() -> tuple[dict[str, Any], ...]:
        """Return the fixed primary and one-dimensional sensitivity grid."""

        specifications: list[dict[str, Any]] = [
            {
                "specification": PRIMARY_SPECIFICATION,
                "rho": PRIMARY_RHO,
                "kappa_multiplier": PRIMARY_KAPPA_MULTIPLIER,
                "draw_count": PRIMARY_DRAW_COUNT,
                "is_primary": True,
            }
        ]
        for rho in RHO_VALUES:
            if math.isclose(rho, PRIMARY_RHO):
                continue
            specifications.append(
                {
                    "specification": specification_label(PRIMARY_KAPPA_MULTIPLIER, rho),
                    "rho": rho,
                    "kappa_multiplier": PRIMARY_KAPPA_MULTIPLIER,
                    "draw_count": SENSITIVITY_DRAW_COUNT,
                    "is_primary": False,
                }
            )
        for multiplier in KAPPA_MULTIPLIERS:
            if math.isclose(multiplier, PRIMARY_KAPPA_MULTIPLIER):
                continue
            specifications.append(
                {
                    "specification": specification_label(multiplier, PRIMARY_RHO),
                    "rho": PRIMARY_RHO,
                    "kappa_multiplier": multiplier,
                    "draw_count": SENSITIVITY_DRAW_COUNT,
                    "is_primary": False,
                }
            )
        labels = [specification["specification"] for specification in specifications]
        if len(labels) != len(set(labels)):
            raise AssertionError("Duplicate FLS realized-geography specifications")
        return tuple(specifications)

    def deterministic_seed(
        aewr_region_id: str,
        source_year: int,
        kappa_multiplier: float,
        *,
        base_seed: int = SIMULATION_SEED,
    ) -> int:
        """Return the stable seed for a region-year Dirichlet prior path.

        Rho is deliberately absent so all m=10 rho sensitivities reuse the first
        199 draws of the primary prior path.
        """

        try:
            region_component = int(aewr_region_id)
        except ValueError:
            region_component = zlib.crc32(aewr_region_id.encode("utf-8")) % 1000
        multiplier_component = int(round(kappa_multiplier * 10))
        seed = (
            int(base_seed)
            + region_component * 1_000_000
            + int(source_year) * 100
            + multiplier_component
        )
        return seed % (2**32 - 1)

    def dirichlet_prior_draws(
        prior_weight: np.ndarray,
        *,
        kappa_multiplier: float,
        draw_count: int,
        seed: int,
    ) -> tuple[np.ndarray, float, float]:
        """Draw reproducible priors with concentration m times effective areas."""

        prior = np.asarray(prior_weight, dtype=float)
        if prior.ndim != 1 or not np.all(np.isfinite(prior)) or np.any(prior <= 0):
            raise ValueError("Dirichlet prior weights must be finite and positive")
        prior = prior / prior.sum()
        effective_area_count = float(1 / np.sum(prior**2))
        kappa = float(kappa_multiplier * effective_area_count)
        alpha = kappa * prior
        rng = np.random.default_rng(seed)
        draws = rng.dirichlet(alpha, size=int(draw_count))
        # Very small alpha values may underflow to exact zero in finite precision.
        # Restore the mathematical positive support at the smallest representable
        # positive float and renormalize.  This affects no reported decimal place.
        draws = np.maximum(draws, np.nextafter(0.0, 1.0))
        draws /= draws.sum(axis=1, keepdims=True)
        return draws, kappa, effective_area_count

    def _dual_state(
        log_prior: np.ndarray,
        design: np.ndarray,
        target: np.ndarray,
        rho: float,
        multipliers: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        linear_predictor = multipliers @ design.T
        log_unnormalized = log_prior + linear_predictor
        maximum = np.max(log_unnormalized, axis=1, keepdims=True)
        exponential = np.exp(log_unnormalized - maximum)
        normalizer = exponential.sum(axis=1, keepdims=True)
        weights = exponential / normalizer
        moments = weights @ design
        objective = (
            maximum[:, 0]
            + np.log(normalizer[:, 0])
            - np.sum(multipliers * target[None, :], axis=1)
            + np.sum(multipliers**2, axis=1) / (2 * rho)
        )
        gradient = moments - target[None, :] + multipliers / rho
        return weights, moments, objective, gradient

    def solve_soft_entropy_batch(
        prior_weights: np.ndarray,
        design: np.ndarray,
        target: np.ndarray,
        *,
        rho: float,
        gradient_tolerance: float = OPTIMIZER_GRADIENT_TOLERANCE,
        maximum_iterations: int = OPTIMIZER_MAX_ITERATIONS,
    ) -> dict[str, np.ndarray]:
        """Solve KL(w||q) + rho/2 times squared standardized imbalance.

        This is the no-exact-constraint specialization of the project's existing
        entropy dual.  A damped Newton method is used here so hundreds of common
        Dirichlet draws can be solved together while preserving the same convex
        objective.
        """

        priors = np.asarray(prior_weights, dtype=float)
        if priors.ndim == 1:
            priors = priors[None, :]
        design = np.asarray(design, dtype=float)
        target = np.asarray(target, dtype=float)
        if priors.ndim != 2:
            raise ValueError("prior_weights must be a vector or matrix")
        if design.ndim != 2 or design.shape[0] != priors.shape[1]:
            raise ValueError("design rows must equal the number of areas")
        if target.shape != (design.shape[1],):
            raise ValueError("target length must equal the design column count")
        if rho <= 0 or not math.isfinite(rho):
            raise ValueError("rho must be finite and positive")
        if (
            not np.all(np.isfinite(priors))
            or np.any(priors < 0)
            or np.any(priors.sum(axis=1) <= 0)
        ):
            raise ValueError("Every prior must be finite, nonnegative, and positive")
        priors = priors / priors.sum(axis=1, keepdims=True)
        batch_size, area_count = priors.shape
        moment_count = design.shape[1]

        if moment_count == 0:
            zeros = np.zeros(batch_size, dtype=float)
            return {
                "weights": priors,
                "success": np.ones(batch_size, dtype=bool),
                "status": np.full(batch_size, "calibrated_prior", dtype=object),
                "iterations": np.zeros(batch_size, dtype=np.int32),
                "input_imbalance_norm": zeros,
                "calibrated_imbalance_norm": zeros,
                "maximum_absolute_imbalance": zeros,
                "kl_divergence": zeros,
                "effective_area_count": 1 / np.sum(priors**2, axis=1),
                "maximum_area_share": np.max(priors, axis=1),
            }

        if not np.all(np.isfinite(design)) or not np.all(np.isfinite(target)):
            raise ValueError("Active standardized moments must be finite")
        log_prior = np.log(np.maximum(priors, np.nextafter(0.0, 1.0)))
        multipliers = np.zeros((batch_size, moment_count), dtype=float)
        converged = np.zeros(batch_size, dtype=bool)
        failed_line_search = np.zeros(batch_size, dtype=bool)
        iterations = np.zeros(batch_size, dtype=np.int32)
        identity = np.eye(moment_count)

        for iteration in range(1, maximum_iterations + 1):
            weights, moments, objective, gradient = _dual_state(
                log_prior, design, target, rho, multipliers
            )
            gradient_norm = np.max(np.abs(gradient), axis=1)
            newly_converged = gradient_norm <= gradient_tolerance
            converged |= newly_converged
            active = ~(converged | failed_line_search)
            if not np.any(active):
                break

            active_indices = np.flatnonzero(active)
            active_weights = weights[active]
            active_moments = moments[active]
            second_moment = np.einsum(
                "bn,np,nq->bpq",
                active_weights,
                design,
                design,
                optimize=True,
            )
            hessian = (
                second_moment
                - np.einsum(
                    "bp,bq->bpq",
                    active_moments,
                    active_moments,
                    optimize=True,
                )
                + identity[None, :, :] / rho
            )
            try:
                step_direction = np.linalg.solve(hessian, gradient[active, :, None])[
                    :, :, 0
                ]
            except np.linalg.LinAlgError:
                # A per-row fallback identifies only the singular cells as failed.
                step_direction = np.full((len(active_indices), moment_count), np.nan)
                for local_index, batch_index in enumerate(active_indices):
                    try:
                        step_direction[local_index] = np.linalg.solve(
                            hessian[local_index], gradient[batch_index]
                        )
                    except np.linalg.LinAlgError:
                        failed_line_search[batch_index] = True

            finite_direction = np.all(np.isfinite(step_direction), axis=1)
            for local_index, batch_index in enumerate(active_indices):
                if not finite_direction[local_index]:
                    failed_line_search[batch_index] = True
                    continue
                direction = step_direction[local_index]
                directional_derivative = float(gradient[batch_index] @ direction)
                step_size = 1.0
                accepted = False
                while step_size >= 2**-20:
                    proposal = (multipliers[batch_index] - step_size * direction)[
                        None, :
                    ]
                    _, _, proposal_objective, _ = _dual_state(
                        log_prior[batch_index : batch_index + 1],
                        design,
                        target,
                        rho,
                        proposal,
                    )
                    numerical_objective_tolerance = 1e-12 * (
                        1 + abs(float(objective[batch_index]))
                    )
                    if math.isfinite(float(proposal_objective[0])) and (
                        proposal_objective[0]
                        <= objective[batch_index]
                        - 1e-4 * step_size * directional_derivative
                        or proposal_objective[0]
                        <= objective[batch_index] + numerical_objective_tolerance
                    ):
                        multipliers[batch_index] = proposal[0]
                        iterations[batch_index] = iteration
                        accepted = True
                        break
                    step_size *= 0.5
                if not accepted:
                    failed_line_search[batch_index] = True

        final_weights, final_moments, _, final_gradient = _dual_state(
            log_prior, design, target, rho, multipliers
        )
        final_weights /= final_weights.sum(axis=1, keepdims=True)
        gradient_norm = np.max(np.abs(final_gradient), axis=1)
        converged |= gradient_norm <= gradient_tolerance

        input_imbalance = priors @ design - target[None, :]
        calibrated_imbalance = final_moments - target[None, :]
        input_norm = np.linalg.norm(input_imbalance, axis=1)
        calibrated_norm = np.linalg.norm(calibrated_imbalance, axis=1)
        maximum_absolute = np.max(np.abs(calibrated_imbalance), axis=1)
        weight_sum_valid = np.abs(final_weights.sum(axis=1) - 1) <= WEIGHT_SUM_TOLERANCE
        nonnegative = np.all(final_weights >= 0, axis=1)
        finite = np.all(np.isfinite(final_weights), axis=1)
        balance_not_worse = calibrated_norm <= input_norm + 1e-9
        success = (
            converged
            & ~failed_line_search
            & weight_sum_valid
            & nonnegative
            & finite
            & balance_not_worse
        )
        status = np.full(batch_size, "optimizer_failed", dtype=object)
        status[success] = "calibrated_soft"
        status[failed_line_search] = "line_search_failed"
        status[~failed_line_search & ~converged] = "maximum_iterations_reached"
        status[converged & ~(weight_sum_valid & nonnegative & finite)] = (
            "invalid_weight_solution"
        )
        status[
            converged & weight_sum_valid & nonnegative & finite & ~balance_not_worse
        ] = "imbalance_increased"

        kl_divergence = np.sum(
            np.where(
                final_weights > 0,
                final_weights
                * (
                    np.log(np.maximum(final_weights, np.nextafter(0.0, 1.0)))
                    - log_prior
                ),
                0.0,
            ),
            axis=1,
        )
        return {
            "weights": final_weights,
            "success": success,
            "status": status,
            "iterations": iterations,
            "input_imbalance_norm": input_norm,
            "calibrated_imbalance_norm": calibrated_norm,
            "maximum_absolute_imbalance": maximum_absolute,
            "kl_divergence": kl_divergence,
            "effective_area_count": 1 / np.sum(final_weights**2, axis=1),
            "maximum_area_share": np.max(final_weights, axis=1),
        }

    def build_wage_features(
        area_prior: pl.DataFrame,
        oews: pl.DataFrame,
        fls_region: pl.DataFrame,
        *,
        years: Sequence[int],
    ) -> pl.DataFrame:
        """Put the OEWS area wage and FLS regional wage on a common scale."""

        _require_columns(
            oews,
            ["area", "area_name", "occ_code", "tot_emp", "h_mean", "year"],
            "OEWS",
        )
        _require_columns(
            fls_region,
            [
                "estimate_year",
                "aewr_region_id",
                "revised_year",
                "preliminary_year",
                "field_livestock_revised",
                "field_livestock_preliminary",
            ],
            "FLS regional wages",
        )
        years = tuple(int(year) for year in years)

        area_wages = (
            oews.filter(
                pl.col("year").cast(pl.Int32).is_in(years),
                pl.col("occ_code").cast(pl.String).is_in(BIG_SIX_OCC_CODES),
            )
            .with_columns(
                pl.col("area")
                .cast(pl.String)
                .str.strip_chars()
                .str.replace(r"\.0+$", "")
                .alias("oews_area_code"),
                pl.col("year").cast(pl.Int32).alias("source_year"),
                pl.col("tot_emp")
                .cast(pl.Float64, strict=False)
                .alias("oews_occupation_employment"),
                pl.col("h_mean")
                .cast(pl.Float64, strict=False)
                .alias("oews_occupation_mean_hourly_wage"),
            )
            .with_columns(
                (
                    pl.col("oews_occupation_employment").is_finite()
                    & (pl.col("oews_occupation_employment") > 0)
                    & pl.col("oews_occupation_mean_hourly_wage").is_finite()
                    & (pl.col("oews_occupation_mean_hourly_wage") > 0)
                ).alias("usable_wage")
            )
            .with_columns(
                pl.when("usable_wage")
                .then("oews_occupation_employment")
                .otherwise(0.0)
                .alias("oews_area_wage_covered_employment"),
                pl.when("usable_wage")
                .then(
                    pl.col("oews_occupation_employment")
                    * pl.col("oews_occupation_mean_hourly_wage")
                )
                .otherwise(0.0)
                .alias("oews_area_hourly_wage_bill"),
            )
            .group_by("oews_area_code", "source_year")
            .agg(
                pl.col("area_name").drop_nulls().first().alias("oews_area_name_data"),
                pl.col("oews_area_wage_covered_employment").sum(),
                pl.col("oews_area_hourly_wage_bill").sum(),
                pl.col("usable_wage").sum().alias("oews_occupation_count"),
            )
            .with_columns(
                pl.when(pl.col("oews_area_wage_covered_employment") > 0)
                .then(
                    pl.col("oews_area_hourly_wage_bill")
                    / pl.col("oews_area_wage_covered_employment")
                )
                .alias("oews_area_mean_hourly_wage")
            )
        )

        revised_targets = fls_region.select(
            pl.col("aewr_region_id").cast(pl.String),
            pl.col("revised_year").cast(pl.Int32).alias("source_year"),
            pl.col("field_livestock_revised")
            .cast(pl.Float64, strict=False)
            .alias("fls_field_livestock_mean_hourly_wage"),
            pl.col("estimate_year").cast(pl.Int32).alias("fls_release_year"),
            pl.lit("revised").alias("fls_wage_vintage"),
            pl.lit(1).alias("vintage_priority"),
        )
        preliminary_targets = fls_region.select(
            pl.col("aewr_region_id").cast(pl.String),
            pl.col("preliminary_year").cast(pl.Int32).alias("source_year"),
            pl.col("field_livestock_preliminary")
            .cast(pl.Float64, strict=False)
            .alias("fls_field_livestock_mean_hourly_wage"),
            pl.col("estimate_year").cast(pl.Int32).alias("fls_release_year"),
            pl.lit("preliminary").alias("fls_wage_vintage"),
            pl.lit(0).alias("vintage_priority"),
        )
        wage_targets = (
            pl.concat([revised_targets, preliminary_targets], how="vertical")
            .filter(
                pl.col("source_year").is_in(years),
                pl.col("fls_field_livestock_mean_hourly_wage").is_finite(),
                pl.col("fls_field_livestock_mean_hourly_wage") > 0,
            )
            .sort(
                "aewr_region_id",
                "source_year",
                "vintage_priority",
                "fls_release_year",
            )
            .unique(
                subset=["aewr_region_id", "source_year"],
                keep="first",
                maintain_order=True,
            )
            .drop("vintage_priority")
        )

        joined = area_prior.join(
            area_wages,
            on=["oews_area_code", "source_year"],
            how="left",
            validate="m:1",
        ).join(
            wage_targets,
            on=["aewr_region_id", "source_year"],
            how="left",
            validate="m:1",
        )
        output_rows: list[dict[str, Any]] = []
        for cell in joined.partition_by(
            ["aewr_region_id", "source_year"],
            maintain_order=True,
        ):
            first = cell.row(0, named=True)
            region = str(first["aewr_region_id"])
            year = int(first["source_year"])
            target_wage = first["fls_field_livestock_mean_hourly_wage"]
            if not _finite_positive(target_wage):
                raise ValueError(
                    "No FLS field-and-livestock wage target for "
                    f"region {region}, {year}"
                )

            prior = cell.get_column("frame_prior_weight").to_numpy()
            supported = np.isfinite(prior) & (prior > 0)
            prior = np.where(supported, prior, 0.0)
            prior /= prior.sum()
            raw_wages = cell.get_column("oews_area_mean_hourly_wage").to_numpy()
            observed = np.isfinite(raw_wages) & (raw_wages > 0)
            observed_mass = float(prior @ observed.astype(float))
            if observed_mass <= 0:
                raise ValueError(f"No OEWS area wages for region {region}, {year}")

            observed_mean = float(
                (prior[observed] @ raw_wages[observed]) / prior[observed].sum()
            )
            filled_wages = np.where(observed, raw_wages, observed_mean)
            center = float(prior @ filled_wages)
            scale = float(np.sqrt(prior @ (filled_wages - center) ** 2))
            if not math.isfinite(scale) or scale <= MINIMUM_CONTRAST_SCALE:
                raise ValueError(
                    f"No usable OEWS wage variation for region {region}, {year}"
                )
            standardized_target = (float(target_wage) - center) / scale

            for row_index, row in enumerate(cell.iter_rows(named=True)):
                output_rows.append(
                    {
                        "aewr_region_id": region,
                        "source_year": year,
                        "oews_area_code": row["oews_area_code"],
                        "oews_area_name": row["oews_area_name"],
                        "frame_prior_weight": float(prior[row_index]),
                        "oews_area_mean_hourly_wage": float(filled_wages[row_index]),
                        "oews_area_wage_observed": bool(observed[row_index]),
                        "oews_area_wage_imputed": bool(
                            supported[row_index] and not observed[row_index]
                        ),
                        "oews_wage_observed_frame_share": observed_mass,
                        "oews_wage_prior_center": center,
                        "oews_wage_prior_scale": scale,
                        "oews_area_standardized_wage": float(
                            (filled_wages[row_index] - center) / scale
                        ),
                        "fls_field_livestock_mean_hourly_wage": float(target_wage),
                        "fls_standardized_wage_target": standardized_target,
                        "fls_release_year": int(first["fls_release_year"]),
                        "fls_wage_vintage": first["fls_wage_vintage"],
                        "moment_spec": MOMENT_SPEC,
                        "wage_target_used": True,
                    }
                )

        wage_features = pl.DataFrame(output_rows, infer_schema_length=None).sort(
            "aewr_region_id", "source_year", "oews_area_code"
        )
        _require_unique(
            wage_features,
            ["aewr_region_id", "source_year", "oews_area_code"],
            "wage features",
        )
        return wage_features

    def _extract_recovery_cell(
        features: pl.DataFrame,
        area_prior: pl.DataFrame,
        wage_features: pl.DataFrame,
        *,
        aewr_region_id: str,
        source_year: int,
    ) -> dict[str, Any]:
        area = area_prior.filter(
            pl.col("aewr_region_id") == str(aewr_region_id),
            pl.col("source_year") == int(source_year),
        ).sort("oews_area_code")
        if area.is_empty():
            raise ValueError(
                f"No frame prior for region {aewr_region_id}, year {source_year}"
            )
        supported = area.filter(pl.col("frame_prior_weight") > 0)
        if supported.is_empty():
            raise ValueError(
                f"No positive frame prior for region {aewr_region_id}, year {source_year}"
            )
        feature_cell = features.filter(
            pl.col("aewr_region_id") == str(aewr_region_id),
            pl.col("source_year") == int(source_year),
        )
        contrast = feature_cell.filter(
            pl.col("feature_row_type") == "helmert_contrast",
            pl.col("supported_frame"),
            pl.col("contrast_active"),
        )
        active_ids = sorted(
            int(value)
            for value in contrast.get_column("contrast_id").drop_nulls().unique()
        )
        supported_codes = supported.get_column("oews_area_code").to_list()
        if active_ids:
            design_lookup = {
                (row["oews_area_code"], int(row["contrast_id"])): float(
                    row["area_standardized_contrast"]
                )
                for row in contrast.iter_rows(named=True)
            }
            target_lookup = {
                int(row["contrast_id"]): float(row["target_standardized_contrast"])
                for row in contrast.unique(
                    subset=["contrast_id"], maintain_order=True
                ).iter_rows(named=True)
            }
            design = np.asarray(
                [
                    [
                        design_lookup[(area_code, contrast_id)]
                        for contrast_id in active_ids
                    ]
                    for area_code in supported_codes
                ],
                dtype=float,
            )
            target = np.asarray(
                [target_lookup[contrast_id] for contrast_id in active_ids],
                dtype=float,
            )
        else:
            design = np.empty((supported.height, 0), dtype=float)
            target = np.empty(0, dtype=float)

        wage_cell = wage_features.filter(
            pl.col("aewr_region_id") == str(aewr_region_id),
            pl.col("source_year") == int(source_year),
            pl.col("frame_prior_weight") > 0,
        )
        wage_lookup = {
            row["oews_area_code"]: row for row in wage_cell.iter_rows(named=True)
        }
        missing_wage_features = [
            area_code for area_code in supported_codes if area_code not in wage_lookup
        ]
        if missing_wage_features:
            raise ValueError(
                f"Missing wage features for region {aewr_region_id}, {source_year}: "
                + ", ".join(missing_wage_features[:5])
            )
        wage_design = np.asarray(
            [
                wage_lookup[area_code]["oews_area_standardized_wage"]
                for area_code in supported_codes
            ],
            dtype=float,
        )
        wage_targets = {
            float(row["fls_standardized_wage_target"]) for row in wage_lookup.values()
        }
        if len(wage_targets) != 1:
            raise ValueError(
                f"FLS wage target is not unique for region {aewr_region_id}, {source_year}"
            )
        design = np.column_stack([design, wage_design])
        target = np.append(target, wage_targets.pop())

        joint = feature_cell.filter(
            pl.col("feature_row_type") == "joint_cell",
            pl.col("supported_frame"),
        )
        imputed_lookup: dict[str, bool] = defaultdict(bool)
        for row in joint.iter_rows(named=True):
            imputed_lookup[row["oews_area_code"]] |= bool(
                row["employment_feature_imputed"] or row["duration_feature_imputed"]
            )
        for area_code in supported_codes:
            imputed_lookup[area_code] |= bool(
                wage_lookup[area_code]["oews_area_wage_imputed"]
            )
        first_wage = wage_lookup[supported_codes[0]]
        return {
            "area": area,
            "supported": supported,
            "supported_codes": supported_codes,
            "prior": supported.get_column("frame_prior_weight").to_numpy(),
            "design": design,
            "target": target,
            "active_contrast_ids": active_ids,
            "fls_wage_target": float(
                first_wage["fls_field_livestock_mean_hourly_wage"]
            ),
            "oews_wage_observed_frame_share": float(
                first_wage["oews_wage_observed_frame_share"]
            ),
            "raw_area_wages": np.asarray(
                [
                    wage_lookup[area_code]["oews_area_mean_hourly_wage"]
                    for area_code in supported_codes
                ],
                dtype=float,
            ),
            "imputed": np.asarray(
                [imputed_lookup[code] for code in supported_codes],
                dtype=bool,
            ),
        }

    def _metadata(
        specification: dict[str, Any],
        *,
        simulation_seed: int,
    ) -> dict[str, Any]:
        return {
            "specification": specification["specification"],
            "weight_spec": WEIGHT_SPEC,
            "baseline_weight_spec": BASELINE_WEIGHT_SPEC,
            "moment_spec": MOMENT_SPEC,
            "wage_target_used": True,
            "rho": specification["rho"],
            "kappa_multiplier": specification["kappa_multiplier"],
            "is_primary": specification["is_primary"],
            "simulation_seed": simulation_seed,
        }

    def _diagnostic_rows(
        *,
        aewr_region_id: str,
        source_year: int,
        specification: dict[str, Any],
        simulation_seed: int,
        kappa: float,
        frame_effective_area_count: float,
        solution: dict[str, np.ndarray],
        weight_draw_ids: np.ndarray,
        weight_kind: str,
        prior_imputed_mass: np.ndarray,
        calibrated_imputed_mass: np.ndarray,
        composition_contrast_count: int,
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "aewr_region_id": [str(aewr_region_id)] * len(weight_draw_ids),
                "source_year": [int(source_year)] * len(weight_draw_ids),
                "weight_draw_id": weight_draw_ids.astype(np.int64),
                "weight_kind": [weight_kind] * len(weight_draw_ids),
                "solver_status": solution["status"].tolist(),
                "optimizer_convergence": solution["success"].tolist(),
                "optimizer_iterations": solution["iterations"].tolist(),
                "targeted_composition_contrast_count": [composition_contrast_count]
                * len(weight_draw_ids),
                "targeted_wage_moment_count": [1] * len(weight_draw_ids),
                "targeted_moment_count": [composition_contrast_count + 1]
                * len(weight_draw_ids),
                "input_prior_standardized_imbalance_norm": (
                    solution["input_imbalance_norm"].tolist()
                ),
                "calibrated_standardized_imbalance_norm": (
                    solution["calibrated_imbalance_norm"].tolist()
                ),
                "maximum_absolute_standardized_imbalance": (
                    solution["maximum_absolute_imbalance"].tolist()
                ),
                "kl_divergence": solution["kl_divergence"].tolist(),
                "effective_area_count": solution["effective_area_count"].tolist(),
                "maximum_area_share": solution["maximum_area_share"].tolist(),
                "imputed_prior_mass": prior_imputed_mass.tolist(),
                "calibrated_imputed_weight_share": (calibrated_imputed_mass.tolist()),
                "kappa": [kappa] * len(weight_draw_ids),
                "frame_effective_area_count": [frame_effective_area_count]
                * len(weight_draw_ids),
                **{
                    key: [value] * len(weight_draw_ids)
                    for key, value in _metadata(
                        specification, simulation_seed=simulation_seed
                    ).items()
                },
            }
        )

    def recover_region_year_specification(
        *,
        features: pl.DataFrame,
        area_prior: pl.DataFrame,
        wage_features: pl.DataFrame,
        aewr_region_id: str,
        source_year: int,
        specification: dict[str, Any],
        prior_draws: np.ndarray | None = None,
        kappa: float | None = None,
        frame_effective_area_count: float | None = None,
        simulation_seed: int | None = None,
    ) -> dict[str, pl.DataFrame]:
        """Recover a deterministic center and one predeclared draw ensemble."""

        cell = _extract_recovery_cell(
            features,
            area_prior,
            wage_features,
            aewr_region_id=aewr_region_id,
            source_year=source_year,
        )
        area = cell["area"]
        prior = np.array(cell["prior"], dtype=float, copy=True)
        prior /= prior.sum()
        design = cell["design"]
        target = cell["target"]
        imputed = cell["imputed"]
        raw_area_wages = cell["raw_area_wages"]
        if simulation_seed is None:
            simulation_seed = deterministic_seed(
                aewr_region_id,
                source_year,
                specification["kappa_multiplier"],
            )
        if prior_draws is None:
            prior_draws, generated_kappa, generated_effective = dirichlet_prior_draws(
                prior,
                kappa_multiplier=specification["kappa_multiplier"],
                draw_count=specification["draw_count"],
                seed=simulation_seed,
            )
            kappa = generated_kappa
            frame_effective_area_count = generated_effective
        else:
            prior_draws = np.asarray(prior_draws, dtype=float)
            if prior_draws.shape != (specification["draw_count"], len(prior)):
                raise ValueError("Provided prior draws have an unexpected shape")
            if frame_effective_area_count is None:
                frame_effective_area_count = float(1 / np.sum(prior**2))
            if kappa is None:
                kappa = float(
                    specification["kappa_multiplier"] * frame_effective_area_count
                )
        assert kappa is not None
        assert frame_effective_area_count is not None

        center_solution = solve_soft_entropy_batch(
            prior,
            design,
            target,
            rho=specification["rho"],
        )
        draw_solution = solve_soft_entropy_batch(
            prior_draws,
            design,
            target,
            rho=specification["rho"],
        )
        successful_weights = draw_solution["weights"][draw_solution["success"]]

        area_codes = area.get_column("oews_area_code").to_list()
        area_names = area.get_column("oews_area_name").to_list()
        all_prior = area.get_column("frame_prior_weight").to_numpy()
        supported_codes = cell["supported_codes"]
        supported_index = {code: index for index, code in enumerate(supported_codes)}
        center_all = np.zeros(area.height, dtype=float)
        if center_solution["success"][0]:
            for index, code in enumerate(area_codes):
                if code in supported_index:
                    center_all[index] = center_solution["weights"][
                        0, supported_index[code]
                    ]
        else:
            center_all[:] = np.nan

        draw_all = np.zeros((specification["draw_count"], area.height), dtype=float)
        prior_draw_all = np.zeros_like(draw_all)
        for area_index, code in enumerate(area_codes):
            if code not in supported_index:
                continue
            supported_area_index = supported_index[code]
            draw_all[:, area_index] = draw_solution["weights"][:, supported_area_index]
            prior_draw_all[:, area_index] = prior_draws[:, supported_area_index]
        draw_all[~draw_solution["success"], :] = np.nan

        draw_ids = np.arange(1, specification["draw_count"] + 1, dtype=np.int64)
        repeated_success = np.repeat(draw_solution["success"], area.height)
        flattened_draw = draw_all.reshape(-1)
        flattened_draw[~repeated_success] = np.nan
        draw_frame = pl.DataFrame(
            {
                "aewr_region_id": np.repeat(
                    str(aewr_region_id),
                    specification["draw_count"] * area.height,
                ),
                "source_year": np.repeat(
                    int(source_year),
                    specification["draw_count"] * area.height,
                ),
                "weight_draw_id": np.repeat(draw_ids, area.height),
                "oews_area_code": np.tile(area_codes, specification["draw_count"]),
                "oews_area_name": np.tile(area_names, specification["draw_count"]),
                "frame_prior_weight": np.tile(all_prior, specification["draw_count"]),
                "prior_draw_weight": prior_draw_all.reshape(-1),
                "oews_area_weight": flattened_draw,
                "solver_status": np.repeat(draw_solution["status"], area.height),
                "kappa": np.repeat(kappa, specification["draw_count"] * area.height),
                "frame_effective_area_count": np.repeat(
                    frame_effective_area_count,
                    specification["draw_count"] * area.height,
                ),
                **{
                    key: np.repeat(value, specification["draw_count"] * area.height)
                    for key, value in _metadata(
                        specification, simulation_seed=simulation_seed
                    ).items()
                },
            }
        ).with_columns(pl.col("oews_area_weight").fill_nan(None))

        summary_rows: list[dict[str, Any]] = []
        for area_index, (area_code, area_name) in enumerate(
            zip(area_codes, area_names, strict=True)
        ):
            values = (
                successful_weights[:, supported_index[area_code]]
                if area_code in supported_index and successful_weights.size
                else (
                    np.zeros(draw_solution["success"].sum())
                    if successful_weights.size
                    else np.empty(0)
                )
            )
            summary_rows.append(
                {
                    "aewr_region_id": str(aewr_region_id),
                    "source_year": int(source_year),
                    "weight_draw_id": None,
                    "oews_area_code": area_code,
                    "oews_area_name": area_name,
                    "frame_prior_weight": float(all_prior[area_index]),
                    "calibrated_center_weight": (
                        float(center_all[area_index])
                        if center_solution["success"][0]
                        else None
                    ),
                    "draw_mean_weight": (
                        float(np.mean(values)) if values.size else None
                    ),
                    "draw_standard_deviation_weight": (
                        float(np.std(values, ddof=1))
                        if values.size > 1
                        else (0.0 if values.size == 1 else None)
                    ),
                    "simulation_envelope_p025_weight": (
                        float(np.quantile(values, 0.025)) if values.size else None
                    ),
                    "simulation_envelope_p50_weight": (
                        float(np.quantile(values, 0.50)) if values.size else None
                    ),
                    "simulation_envelope_p975_weight": (
                        float(np.quantile(values, 0.975)) if values.size else None
                    ),
                    "center_solver_status": center_solution["status"][0],
                    "draws_requested": specification["draw_count"],
                    "draws_succeeded": int(draw_solution["success"].sum()),
                    "draw_success_rate": float(draw_solution["success"].mean()),
                    "kappa": kappa,
                    "frame_effective_area_count": frame_effective_area_count,
                    "fls_field_livestock_mean_hourly_wage": cell[
                        "fls_wage_target"
                    ],
                    "oews_wage_observed_frame_share": cell[
                        "oews_wage_observed_frame_share"
                    ],
                    "frame_prior_oews_mean_hourly_wage": float(prior @ raw_area_wages),
                    "calibrated_center_oews_mean_hourly_wage": (
                        float(center_solution["weights"][0] @ raw_area_wages)
                        if center_solution["success"][0]
                        else None
                    ),
                    **_metadata(specification, simulation_seed=simulation_seed),
                }
            )
        summary = pl.DataFrame(summary_rows, infer_schema_length=None).with_columns(
            pl.col("weight_draw_id").cast(pl.Int64)
        )

        center_prior_imputed_mass = np.asarray([float(prior @ imputed.astype(float))])
        center_calibrated_imputed_mass = np.asarray(
            [float(center_solution["weights"][0] @ imputed.astype(float))]
        )
        center_diagnostic = _diagnostic_rows(
            aewr_region_id=aewr_region_id,
            source_year=source_year,
            specification=specification,
            simulation_seed=simulation_seed,
            kappa=kappa,
            frame_effective_area_count=frame_effective_area_count,
            solution=center_solution,
            weight_draw_ids=np.asarray([0], dtype=np.int64),
            weight_kind="deterministic_center",
            prior_imputed_mass=center_prior_imputed_mass,
            calibrated_imputed_mass=center_calibrated_imputed_mass,
            composition_contrast_count=len(cell["active_contrast_ids"]),
        )
        draw_diagnostic = _diagnostic_rows(
            aewr_region_id=aewr_region_id,
            source_year=source_year,
            specification=specification,
            simulation_seed=simulation_seed,
            kappa=kappa,
            frame_effective_area_count=frame_effective_area_count,
            solution=draw_solution,
            weight_draw_ids=draw_ids,
            weight_kind="dirichlet_draw",
            prior_imputed_mass=prior_draws @ imputed.astype(float),
            calibrated_imputed_mass=(draw_solution["weights"] @ imputed.astype(float)),
            composition_contrast_count=len(cell["active_contrast_ids"]),
        )
        diagnostics = pl.concat(
            [center_diagnostic, draw_diagnostic], how="vertical_relaxed"
        )
        return {
            "draws": draw_frame,
            "summary": summary,
            "diagnostics": diagnostics,
        }

    def common_prior_draws_for_region_year(
        *,
        features: pl.DataFrame,
        area_prior: pl.DataFrame,
        wage_features: pl.DataFrame,
        aewr_region_id: str,
        source_year: int,
        specifications: Sequence[dict[str, Any]] | None = None,
    ) -> dict[float, tuple[np.ndarray, float, float, int]]:
        """Generate each multiplier's path once for reuse over rho."""

        if specifications is None:
            specifications = specification_grid()
        cell = _extract_recovery_cell(
            features,
            area_prior,
            wage_features,
            aewr_region_id=aewr_region_id,
            source_year=source_year,
        )
        prior = np.array(cell["prior"], dtype=float, copy=True)
        prior /= prior.sum()
        paths: dict[float, tuple[np.ndarray, float, float, int]] = {}
        for multiplier in sorted(
            {specification["kappa_multiplier"] for specification in specifications}
        ):
            draw_count = max(
                specification["draw_count"]
                for specification in specifications
                if math.isclose(specification["kappa_multiplier"], multiplier)
            )
            seed = deterministic_seed(aewr_region_id, source_year, multiplier)
            draws, kappa, effective = dirichlet_prior_draws(
                prior,
                kappa_multiplier=multiplier,
                draw_count=draw_count,
                seed=seed,
            )
            paths[multiplier] = (draws, kappa, effective, seed)
        return paths

    def validate_primary_acceptance(
        feature_diagnostics: pl.DataFrame,
        recovery_diagnostics: pl.DataFrame,
    ) -> None:
        """Fail a build that misses a predeclared primary acceptance threshold."""

        unavailable = feature_diagnostics.filter(
            pl.col("feature_status") != "available"
        )
        if unavailable.height:
            cells = ", ".join(
                f"{row['aewr_region_id']}-{row['source_year']}:{row['feature_status']}"
                for row in unavailable.iter_rows(named=True)
            )
            raise RuntimeError(f"Unavailable public feature cells: {cells}")

        primary = recovery_diagnostics.filter(pl.col("is_primary"))
        centers = primary.filter(pl.col("weight_kind") == "deterministic_center")
        failed_centers = centers.filter(~pl.col("optimizer_convergence"))
        if failed_centers.height:
            cells = ", ".join(
                f"{row['aewr_region_id']}-{row['source_year']}"
                for row in failed_centers.iter_rows(named=True)
            )
            raise RuntimeError(f"Primary deterministic centers failed: {cells}")

        draw_rates = (
            primary.filter(pl.col("weight_kind") == "dirichlet_draw")
            .group_by("aewr_region_id", "source_year")
            .agg(pl.col("optimizer_convergence").mean().alias("draw_success_rate"))
        )
        failed_rates = draw_rates.filter(pl.col("draw_success_rate") < 0.99)
        if failed_rates.height:
            cells = ", ".join(
                f"{row['aewr_region_id']}-{row['source_year']}:"
                f"{row['draw_success_rate']:.3f}"
                for row in failed_rates.iter_rows(named=True)
            )
            raise RuntimeError(
                "Primary draw success rate is below 99 percent: " + cells
            )

    FEATURE_PATH = INTERMEDIATE / "panel_iv_fls_geography_features.parquet"
    COUNTY_AREA_PRIOR_PATH = (
        INTERMEDIATE / "panel_iv_fls_geography_county_area_prior.parquet"
    )
    AREA_PRIOR_PATH = INTERMEDIATE / "panel_iv_fls_geography_area_prior.parquet"
    WAGE_FEATURE_PATH = INTERMEDIATE / "panel_iv_fls_geography_wage_features.parquet"
    TARGET_DIAGNOSTIC_PATH = (
        INTERMEDIATE / "panel_iv_fls_geography_target_diagnostics.parquet"
    )
    FEATURE_DIAGNOSTIC_PATH = (
        INTERMEDIATE / "panel_iv_fls_geography_feature_diagnostics.parquet"
    )
    BRIDGE_DIAGNOSTIC_PATH = (
        INTERMEDIATE / "panel_iv_fls_geography_duration_bridge.parquet"
    )
    DRAW_ROOT = INTERMEDIATE / "panel_iv_fls_geography_draws"
    SUMMARY_PATH = INTERMEDIATE / "panel_iv_fls_geography_weight_summary.parquet"
    RECOVERY_DIAGNOSTIC_PATH = (
        INTERMEDIATE / "panel_iv_fls_geography_diagnostics.parquet"
    )

    RECOVERY_INPUT_FILENAMES = (
        "panel_iv_fls_frame.parquet",
        "oews_area_definitions.parquet",
        "oews.parquet",
        "fls_region.parquet",
        "fls_region_quarterly_workers.parquet",
        "qcew_county_ag_quarterly_employment.parquet",
        "qwi_county_ag_quarterly_employment.parquet",
        "census_ag_hired_worker_duration_county.parquet",
    )

    def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            frame.write_parquet(temporary, compression="zstd")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _selected_expression(
        *,
        years: list[int],
        regions: list[str] | None,
    ) -> pl.Expr:
        expression = pl.col("source_year").is_in(years)
        if regions is not None:
            expression = expression & pl.col("aewr_region_id").is_in(regions)
        return expression

    def _replace_selected_cells(
        path: Path,
        replacement: pl.DataFrame,
        *,
        years: list[int],
        regions: list[str] | None,
        sort_columns: list[str],
    ) -> None:
        retained: pl.DataFrame | None = None
        if path.exists():
            existing = pl.read_parquet(path)
            retained = existing.filter(
                ~_selected_expression(years=years, regions=regions)
            )
        pieces = [
            frame
            for frame in (retained, replacement)
            if frame is not None and not frame.is_empty()
        ]
        if not pieces:
            return
        combined = (
            pl.concat(pieces, how="diagonal_relaxed") if len(pieces) > 1 else pieces[0]
        )
        _atomic_write_parquet(combined.sort(sort_columns), path)

    def _write_feature_artifacts(
        artifacts: dict[str, pl.DataFrame],
        *,
        years: list[int],
        regions: list[str] | None,
    ) -> None:
        tables = (
            (
                FEATURE_PATH,
                artifacts["features"],
                [
                    "aewr_region_id",
                    "source_year",
                    "oews_area_code",
                    "feature_row_type",
                    "cell_index",
                    "contrast_id",
                ],
            ),
            (
                COUNTY_AREA_PRIOR_PATH,
                artifacts["county_area_prior"],
                [
                    "aewr_region_id",
                    "source_year",
                    "oews_area_code",
                    "county_fips",
                ],
            ),
            (
                AREA_PRIOR_PATH,
                artifacts["area_prior"],
                ["aewr_region_id", "source_year", "oews_area_code"],
            ),
            (
                WAGE_FEATURE_PATH,
                artifacts["wage_features"],
                ["aewr_region_id", "source_year", "oews_area_code"],
            ),
            (
                TARGET_DIAGNOSTIC_PATH,
                artifacts["target_diagnostics"],
                ["aewr_region_id", "source_year"],
            ),
            (
                FEATURE_DIAGNOSTIC_PATH,
                artifacts["feature_diagnostics"],
                ["aewr_region_id", "source_year"],
            ),
        )
        for path, table, sort_columns in tables:
            _replace_selected_cells(
                path,
                table,
                years=years,
                regions=regions,
                sort_columns=sort_columns,
            )
        _atomic_write_parquet(artifacts["bridge_diagnostics"], BRIDGE_DIAGNOSTIC_PATH)

    def _draw_partition_path(
        aewr_region_id: str,
        source_year: int,
        specification: str,
    ) -> Path:
        return (
            DRAW_ROOT
            / f"aewr_region_id={aewr_region_id}"
            / f"source_year={source_year}"
            / f"specification={specification}"
            / "part-00000.parquet"
        )

    def _write_recovery_partition(
        partition: dict[str, pl.DataFrame],
        *,
        aewr_region_id: str,
        source_year: int,
        specification: str,
    ) -> None:
        _atomic_write_parquet(
            partition["draws"],
            _draw_partition_path(aewr_region_id, source_year, specification),
        )

    def _read_inputs() -> dict[str, pl.DataFrame]:
        missing = [
            name
            for name in RECOVERY_INPUT_FILENAMES
            if not (INTERMEDIATE / name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing realized-geography public inputs: " + ", ".join(missing)
            )
        return {
            name.removesuffix(".parquet"): pl.read_parquet(INTERMEDIATE / name)
            for name in RECOVERY_INPUT_FILENAMES
        }

    def run_recovery(
        *,
        years: list[int],
        regions: list[str] | None,
    ) -> None:
        inputs = _read_inputs()
        print("Building public FLS quarter-duration features", flush=True)
        artifacts = build_feature_artifacts(
            frame_employment=inputs["panel_iv_fls_frame"],
            area_definitions=inputs["oews_area_definitions"],
            quarterly_workers=inputs["fls_region_quarterly_workers"],
            qcew=inputs["qcew_county_ag_quarterly_employment"],
            qwi=inputs["qwi_county_ag_quarterly_employment"],
            census_duration=inputs["census_ag_hired_worker_duration_county"],
            years=years,
            regions=regions,
        )
        artifacts["wage_features"] = build_wage_features(
            artifacts["area_prior"],
            inputs["oews"],
            inputs["fls_region"],
            years=years,
        )
        _write_feature_artifacts(artifacts, years=years, regions=regions)

        available_cells = (
            artifacts["feature_diagnostics"]
            .filter(pl.col("feature_status") == "available")
            .select("aewr_region_id", "source_year")
        )
        specifications = specification_grid()
        summary_frames: list[pl.DataFrame] = []
        diagnostic_frames: list[pl.DataFrame] = []
        for cell_number, cell in enumerate(
            available_cells.iter_rows(named=True), start=1
        ):
            region = cell["aewr_region_id"]
            year = int(cell["source_year"])
            print(
                f"Recovering region {region}, {year} "
                f"({cell_number}/{available_cells.height})",
                flush=True,
            )
            common_paths = common_prior_draws_for_region_year(
                features=artifacts["features"],
                area_prior=artifacts["area_prior"],
                wage_features=artifacts["wage_features"],
                aewr_region_id=region,
                source_year=year,
                specifications=specifications,
            )
            for specification in specifications:
                draws, kappa, effective_area_count, seed = common_paths[
                    specification["kappa_multiplier"]
                ]
                partition = recover_region_year_specification(
                    features=artifacts["features"],
                    area_prior=artifacts["area_prior"],
                    wage_features=artifacts["wage_features"],
                    aewr_region_id=region,
                    source_year=year,
                    specification=specification,
                    prior_draws=draws[: specification["draw_count"]],
                    kappa=kappa,
                    frame_effective_area_count=effective_area_count,
                    simulation_seed=seed,
                )
                _write_recovery_partition(
                    partition,
                    aewr_region_id=region,
                    source_year=year,
                    specification=specification["specification"],
                )
                summary_frames.append(partition["summary"])
                diagnostic_frames.append(partition["diagnostics"])

        if not summary_frames or not diagnostic_frames:
            raise RuntimeError("No FLS realized-geography cells were recovered")
        summaries = pl.concat(summary_frames, how="vertical_relaxed")
        diagnostics = pl.concat(diagnostic_frames, how="vertical_relaxed")
        _replace_selected_cells(
            SUMMARY_PATH,
            summaries,
            years=years,
            regions=regions,
            sort_columns=[
                "aewr_region_id",
                "source_year",
                "specification",
                "oews_area_code",
            ],
        )
        _replace_selected_cells(
            RECOVERY_DIAGNOSTIC_PATH,
            diagnostics,
            years=years,
            regions=regions,
            sort_columns=[
                "aewr_region_id",
                "source_year",
                "specification",
                "weight_draw_id",
            ],
        )
        validate_primary_acceptance(artifacts["feature_diagnostics"], diagnostics)
        primary = diagnostics.filter(
            pl.col("specification") == PRIMARY_SPECIFICATION,
            pl.col("weight_kind") == "dirichlet_draw",
        )
        print(
            f"Wrote {summaries.height:,} area summaries and "
            f"{diagnostics.height:,} diagnostics; primary draw success "
            f"{primary.get_column('optimizer_convergence').mean():.3%}",
            flush=True,
        )

    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--years",
            nargs="+",
            type=int,
            default=list(SUPPORTED_YEARS),
            help="source years to build (default: 2011 through 2021)",
        )
        parser.add_argument(
            "--regions",
            nargs="+",
            default=None,
            help="optional AEWR/FLS region identifiers for a partial rebuild",
        )
        return parser.parse_args()

    def main() -> None:
        args = parse_args()
        unsupported = sorted(set(args.years).difference(SUPPORTED_YEARS))
        if unsupported:
            raise ValueError(
                "Unsupported FLS realized-geography years: "
                + ", ".join(str(year) for year in unsupported)
            )
        regions = (
            sorted({str(int(region)) for region in args.regions}, key=int)
            if args.regions
            else None
        )
        if regions and any(not 1 <= int(region) <= 17 for region in regions):
            raise ValueError("FLS region identifiers must be between 1 and 17")
        run_recovery(
            years=sorted(set(args.years)),
            regions=regions,
        )

    if __name__ == "__main__":
        main()
    return


if __name__ == "__main__":
    app.run()
