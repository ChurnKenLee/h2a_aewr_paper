from __future__ import annotations

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    # Build the rolling Census–QCEW/QWI/BEA FLS-frame employment analog.
    # Inputs are existing local source/clean artifacts only. No first-stage,outcome, OEWS employment, wage-calibration, or realized-weight data enter.
    return


@app.cell
def _():
    import polars as pl
    from h2a.geography import assert_geo_columns
    from h2a.paths import INTERMEDIATE, PROCESSED



    import math
    from collections import defaultdict
    from collections.abc import Iterable
    from itertools import pairwise
    from typing import Any


    BENCHMARK_YEARS = (2007, 2012, 2017, 2022)
    FRAME_YEARS = tuple(range(BENCHMARK_YEARS[0], BENCHMARK_YEARS[-1] + 1))
    WEIGHT_SPEC = "census_hired_workers_qcew_updated"
    ANNUAL_UPDATE_SPEC = "qcew_qwi_bea_two_sided_state_raked"
    EXTREME_LOG1P_GROWTH_THRESHOLD = math.log(2)

    _COUNTY_KEYS = ["county_fips"]
    _ANNUAL_KEYS = ["county_fips", "source_year"]
    _EXPECTED_INDUSTRIES = {"111", "112"}
    _EXPECTED_QUARTERS = {1, 2, 3, 4}


    def _finite_nonnegative(value: Any) -> bool:
        return (
            value is not None
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) >= 0
        )


    def _finite_positive(value: Any) -> bool:
        return _finite_nonnegative(value) and float(value) > 0


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
        keys: list[str],
        label: str,
    ) -> None:
        _require_columns(frame, keys, label)
        duplicate_count = frame.group_by(keys).len().filter(pl.col("len") > 1).height
        if duplicate_count:
            raise ValueError(
                f"{label} contains {duplicate_count} duplicate key cells on "
                f"{', '.join(keys)}"
            )


    def _strict_quarterly_annual_measure(
        frame: pl.DataFrame,
        *,
        value_column: str,
        prefix: str,
        disclosed_column: str | None = None,
    ) -> pl.DataFrame:
        """Average quarterly NAICS 111+112 employment after an eight-cell check."""
        required = [
            "county_fips",
            "year",
            "qtr",
            "industry_code",
            value_column,
        ]
        if disclosed_column is not None:
            required.append(disclosed_column)
        _require_columns(frame, required, prefix.upper())

        relevant = (
            frame.filter(
                pl.col("industry_code").cast(pl.String).is_in(_EXPECTED_INDUSTRIES),
                pl.col("qtr").cast(pl.Int32).is_in(_EXPECTED_QUARTERS),
            )
            .with_columns(
                pl.col("year").cast(pl.Int32).alias("source_year"),
                pl.col("qtr").cast(pl.Int32),
                pl.col("industry_code").cast(pl.String),
                pl.col(value_column).cast(pl.Float64, strict=False),
            )
            .select(
                "county_fips",
                "source_year",
                "qtr",
                "industry_code",
                value_column,
                *([disclosed_column] if disclosed_column is not None else []),
            )
        )

        valid_value = (
            pl.col(value_column).is_not_null()
            & pl.col(value_column).is_finite()
            & (pl.col(value_column) >= 0)
        )
        valid_row = valid_value
        if disclosed_column is not None:
            valid_row = valid_row & pl.col(disclosed_column).fill_null(False)

        cells = (
            relevant.group_by(
                "county_fips",
                "source_year",
                "qtr",
                "industry_code",
            )
            .agg(
                pl.len().alias("source_rows"),
                valid_row.sum().alias("valid_source_rows"),
                pl.when(valid_row)
                .then(pl.col(value_column))
                .otherwise(None)
                .first()
                .alias("cell_employment"),
            )
            .with_columns(
                (
                    (pl.col("source_rows") == 1)
                    & (pl.col("valid_source_rows") == 1)
                ).alias("valid_cell")
            )
        )

        annual = (
            cells.group_by(*_ANNUAL_KEYS)
            .agg(
                pl.len().alias(f"{prefix}_observed_cells"),
                pl.col("valid_cell").sum().alias(f"{prefix}_valid_cells"),
                pl.col("qtr").n_unique().alias(f"{prefix}_observed_quarters"),
                pl.col("industry_code")
                .n_unique()
                .alias(f"{prefix}_observed_industries"),
                pl.when(pl.col("valid_cell"))
                .then(pl.col("cell_employment"))
                .otherwise(None)
                .sum()
                .alias("_valid_employment_sum"),
            )
            .with_columns(
                (
                    (pl.col(f"{prefix}_observed_cells") == 8)
                    & (pl.col(f"{prefix}_valid_cells") == 8)
                    & (pl.col(f"{prefix}_observed_quarters") == 4)
                    & (pl.col(f"{prefix}_observed_industries") == 2)
                ).alias(f"{prefix}_strict_complete")
            )
            .with_columns(
                pl.when(pl.col(f"{prefix}_strict_complete"))
                .then(pl.col("_valid_employment_sum") / 4)
                .otherwise(None)
                .alias(f"{prefix}_ag_employment")
            )
            .drop("_valid_employment_sum")
            .sort(*_ANNUAL_KEYS)
        )
        _require_unique(annual, _ANNUAL_KEYS, f"{prefix} annual employment")
        return annual


    def build_strict_qcew_annual(qcew: pl.DataFrame) -> pl.DataFrame:
        """Build strict private NAICS 111+112 QCEW annual employment."""
        return _strict_quarterly_annual_measure(
            qcew,
            value_column="qcew_reference_month_emplvl",
            prefix="qcew",
            disclosed_column="qcew_employment_disclosed",
        )


    def build_strict_qwi_annual(qwi: pl.DataFrame) -> pl.DataFrame:
        """Build strict QWI beginning-quarter NAICS 111+112 employment."""
        return _strict_quarterly_annual_measure(
            qwi,
            value_column="qwi_beginning_quarter_employment",
            prefix="qwi",
        )


    def build_bea_hired_farm_jobs(bea: pl.DataFrame) -> pl.DataFrame:
        """Return max(BEA farm employment - farm proprietors, 0)."""
        _require_columns(
            bea,
            ["county_fips", "year", "emp_farm", "emp_farm_propr"],
            "BEA county employment",
        )
        result = (
            bea.with_columns(
                pl.col("year").cast(pl.Int32).alias("source_year"),
                pl.col("emp_farm").cast(pl.Float64, strict=False),
                pl.col("emp_farm_propr").cast(pl.Float64, strict=False),
            )
            .with_columns(
                pl.when(
                    pl.col("emp_farm").is_not_null()
                    & pl.col("emp_farm").is_finite()
                    & (pl.col("emp_farm") >= 0)
                    & pl.col("emp_farm_propr").is_not_null()
                    & pl.col("emp_farm_propr").is_finite()
                    & (pl.col("emp_farm_propr") >= 0)
                )
                .then(
                    (pl.col("emp_farm") - pl.col("emp_farm_propr")).clip(
                        lower_bound=0
                    )
                )
                .otherwise(None)
                .alias("bea_hired_farm_jobs")
            )
            .select(*_ANNUAL_KEYS, "bea_hired_farm_jobs")
            .filter(pl.col("source_year").is_in(FRAME_YEARS))
            .sort(*_ANNUAL_KEYS)
        )
        _require_unique(result, _ANNUAL_KEYS, "BEA hired-farm employment")
        return result


    def _positive_signal_by_county(
        census_county: pl.DataFrame,
        qcew: pl.DataFrame,
        qwi: pl.DataFrame,
        bea_annual: pl.DataFrame,
    ) -> dict[str, bool]:
        signal: defaultdict[str, bool] = defaultdict(bool)
        for row in census_county.select(
            "county_fips", "census_hired_workers_total"
        ).iter_rows(named=True):
            if _finite_positive(row["census_hired_workers_total"]):
                signal[row["county_fips"]] = True
        for row in qcew.select(
            "county_fips",
            "qcew_employment_disclosed",
            "qcew_reference_month_emplvl",
        ).iter_rows(named=True):
            if row["qcew_employment_disclosed"] and _finite_positive(
                row["qcew_reference_month_emplvl"]
            ):
                signal[row["county_fips"]] = True
        for row in qwi.select(
            "county_fips", "qwi_beginning_quarter_employment"
        ).iter_rows(named=True):
            if _finite_positive(row["qwi_beginning_quarter_employment"]):
                signal[row["county_fips"]] = True
        for row in bea_annual.iter_rows(named=True):
            if _finite_positive(row["bea_hired_farm_jobs"]):
                signal[row["county_fips"]] = True
        return dict(signal)


    def build_annual_employment_updates(
        counties: pl.DataFrame,
        qcew: pl.DataFrame,
        qwi: pl.DataFrame,
        bea: pl.DataFrame,
        *,
        frame_years: Iterable[int] = FRAME_YEARS,
    ) -> pl.DataFrame:
        """Apply annual-level and same-source growth fallback hierarchies."""
        _require_unique(counties, _COUNTY_KEYS, "county universe")
        years = tuple(int(year) for year in frame_years)
        qcew_annual = build_strict_qcew_annual(qcew)
        qwi_annual = build_strict_qwi_annual(qwi)
        bea_annual = build_bea_hired_farm_jobs(bea)

        grid = (
            counties.select("county_fips")
            .join(pl.DataFrame({"source_year": years}), how="cross")
            .join(qcew_annual, on=_ANNUAL_KEYS, how="left", validate="1:1")
            .join(qwi_annual, on=_ANNUAL_KEYS, how="left", validate="1:1")
            .join(bea_annual, on=_ANNUAL_KEYS, how="left", validate="1:1")
            .with_columns(
                pl.col("qcew_strict_complete").fill_null(False),
                pl.col("qwi_strict_complete").fill_null(False),
            )
            .sort(*_ANNUAL_KEYS)
        )

        rows = grid.to_dicts()
        previous_by_county: dict[str, dict[str, Any]] = {}
        for row in rows:
            qcew_valid = bool(row["qcew_strict_complete"]) and _finite_nonnegative(
                row["qcew_ag_employment"]
            )
            qwi_valid = bool(row["qwi_strict_complete"]) and _finite_nonnegative(
                row["qwi_ag_employment"]
            )
            bea_valid = _finite_nonnegative(row["bea_hired_farm_jobs"])
            if qcew_valid:
                update_source = "qcew"
                update_employment = float(row["qcew_ag_employment"])
            elif qwi_valid:
                update_source = "qwi"
                update_employment = float(row["qwi_ag_employment"])
            elif bea_valid:
                update_source = "bea"
                update_employment = float(row["bea_hired_farm_jobs"])
            else:
                update_source = "unavailable"
                update_employment = None

            previous = previous_by_county.get(row["county_fips"])
            growth_source = "not_applicable"
            log_growth = 0.0
            if previous is not None:
                consecutive = previous["source_year"] == row["source_year"] - 1
                source_pairs = (
                    (
                        "qcew",
                        qcew_valid,
                        bool(previous["qcew_strict_complete"]),
                        row["qcew_ag_employment"],
                        previous["qcew_ag_employment"],
                    ),
                    (
                        "qwi",
                        qwi_valid,
                        bool(previous["qwi_strict_complete"]),
                        row["qwi_ag_employment"],
                        previous["qwi_ag_employment"],
                    ),
                    (
                        "bea",
                        bea_valid,
                        _finite_nonnegative(previous["bea_hired_farm_jobs"]),
                        row["bea_hired_farm_jobs"],
                        previous["bea_hired_farm_jobs"],
                    ),
                )
                for (
                    candidate_source,
                    current_valid,
                    previous_valid,
                    current_value,
                    previous_value,
                ) in source_pairs:
                    if consecutive and current_valid and previous_valid:
                        growth_source = candidate_source
                        log_growth = math.log1p(float(current_value)) - math.log1p(
                            float(previous_value)
                        )
                        break
                else:
                    growth_source = "unit_growth"

            row["annual_update_source"] = update_source
            row["annual_update_employment"] = update_employment
            row["annual_growth_source"] = growth_source
            row["annual_log1p_growth"] = log_growth
            row["annual_growth_factor_log1p"] = math.exp(log_growth)
            row["extreme_annual_change"] = (
                growth_source not in {"not_applicable", "unit_growth"}
                and abs(log_growth) > EXTREME_LOG1P_GROWTH_THRESHOLD
            )
            row["qwi_annual_fallback_used"] = update_source == "qwi"
            row["bea_annual_fallback_used"] = update_source == "bea"
            row["unit_growth_fallback_used"] = growth_source == "unit_growth"
            previous_by_county[row["county_fips"]] = row

        result = pl.DataFrame(rows)
        _require_unique(result, _ANNUAL_KEYS, "annual employment update grid")
        return result.sort(*_ANNUAL_KEYS)


    def _cumulative_growth(
        growth: dict[tuple[str, int], float],
        county_fips: str,
        start_year: int,
        end_year: int,
    ) -> float:
        if start_year == end_year:
            return 0.0
        if start_year < end_year:
            return sum(
                growth[(county_fips, year)]
                for year in range(start_year + 1, end_year + 1)
            )
        return -sum(
            growth[(county_fips, year)]
            for year in range(end_year + 1, start_year + 1)
        )


    def _interval_for_year(
        year: int,
        benchmark_years: tuple[int, ...],
    ) -> tuple[int, int]:
        for lower, upper in pairwise(benchmark_years):
            if lower <= year <= upper:
                return lower, upper
        raise ValueError(f"Year {year} is outside the benchmark interval")


    def _quality_flags(row: dict[str, Any]) -> str:
        flags: list[str] = []
        if row["census_benchmark_reported"]:
            flags.append("census_reported")
        if row["census_published_zero"]:
            flags.append("census_published_zero")
        if (
            row["census_benchmark_fill_method"] is not None
            and not row["census_benchmark_reported"]
        ):
            flags.append("census_benchmark_imputed")
        if row["structural_zero"]:
            flags.append("structural_zero")
        if row["qwi_annual_fallback_used"]:
            flags.append("qwi_annual_fallback")
        if row["bea_annual_fallback_used"]:
            flags.append("bea_annual_fallback")
        if row["unit_growth_fallback_used"]:
            flags.append("unit_growth_fallback")
        if row["annual_update_source"] == "unavailable":
            flags.append("annual_update_unavailable")
        if row["extreme_annual_change"]:
            flags.append("extreme_annual_change")
        if row["nonnegative_floor_applied"]:
            flags.append("nonnegative_floor")
        if abs(row["state_rake_factor"] - 1) > 1e-10:
            flags.append("state_raked")
        return "|".join(flags) if flags else "none"


    def build_frame_employment_analog(
        counties: pl.DataFrame,
        census_county: pl.DataFrame,
        census_state: pl.DataFrame,
        census_farms: pl.DataFrame,
        qcew: pl.DataFrame,
        qwi: pl.DataFrame,
        bea: pl.DataFrame,
        *,
        benchmark_years: Iterable[int] = BENCHMARK_YEARS,
        frame_years: Iterable[int] = FRAME_YEARS,
    ) -> pl.DataFrame:
        """Construct the county-year Census frame analog.

        Reported county benchmarks are fixed. Missing county benchmarks are first
        filled by county interpolation/projection or employment ratios and are
        then scaled only within the unreported state residual. Annual paths use
        same-source QCEW/QWI/BEA log-growth and a two-sided endpoint correction,
        followed by state-year raking to interpolated published state totals.
        """
        benchmark_years = tuple(int(year) for year in benchmark_years)
        frame_years = tuple(int(year) for year in frame_years)
        if benchmark_years != tuple(sorted(benchmark_years)):
            raise ValueError("Benchmark years must be sorted")
        if frame_years != tuple(
            range(benchmark_years[0], benchmark_years[-1] + 1)
        ):
            raise ValueError("Frame years must cover every benchmark interval year")

        required_county_columns = [
            "county_fips",
            "state_fips",
            "state_abbrev",
            "aewr_region_id",
        ]
        _require_columns(counties, required_county_columns, "county universe")
        counties = counties.select(required_county_columns).unique()
        _require_unique(counties, _COUNTY_KEYS, "county universe")
        county_rows = counties.sort("county_fips").to_dicts()
        county_ids = {row["county_fips"] for row in county_rows}

        _require_columns(
            census_county,
            ["county_fips", "year", "census_hired_workers_total"],
            "county Census benchmarks",
        )
        census_county = census_county.filter(
            pl.col("county_fips").is_in(county_ids),
            pl.col("year").cast(pl.Int32).is_in(benchmark_years),
        ).with_columns(pl.col("year").cast(pl.Int32))
        _require_unique(
            census_county,
            ["county_fips", "year"],
            "county Census benchmarks",
        )

        _require_columns(
            census_state,
            ["state_fips", "year", "state_census_hired_workers_reported"],
            "state Census benchmarks",
        )
        census_state = census_state.filter(
            pl.col("state_fips").is_in(
                counties.get_column("state_fips").unique()
            ),
            pl.col("year").cast(pl.Int32).is_in(benchmark_years),
        ).with_columns(pl.col("year").cast(pl.Int32))
        _require_unique(
            census_state,
            ["state_fips", "year"],
            "state Census benchmarks",
        )

        expected_state_cells = (
            counties.get_column("state_fips").n_unique() * len(benchmark_years)
        )
        if census_state.height != expected_state_cells:
            raise ValueError(
                "State Census benchmarks do not cover every state-vintage cell"
            )
        if not all(
            _finite_nonnegative(value)
            for value in census_state.get_column(
                "state_census_hired_workers_reported"
            ).to_list()
        ):
            raise ValueError("State Census hired-worker totals must be reported")

        updates = build_annual_employment_updates(
            counties,
            qcew,
            qwi,
            bea,
            frame_years=frame_years,
        )
        bea_annual = build_bea_hired_farm_jobs(bea)
        positive_signal = _positive_signal_by_county(
            census_county,
            qcew,
            qwi,
            bea_annual,
        )
        update_map = {
            (row["county_fips"], row["source_year"]): row
            for row in updates.iter_rows(named=True)
        }
        growth = {
            key: float(row["annual_log1p_growth"])
            for key, row in update_map.items()
        }

        reported_map: dict[tuple[str, int], float] = {}
        observed_years: defaultdict[str, list[int]] = defaultdict(list)
        for row in census_county.iter_rows(named=True):
            value = row["census_hired_workers_total"]
            if _finite_nonnegative(value):
                key = (row["county_fips"], int(row["year"]))
                reported_map[key] = float(value)
                observed_years[row["county_fips"]].append(int(row["year"]))
        for county_fips in observed_years:
            observed_years[county_fips].sort()

        state_total = {
            (row["state_fips"], int(row["year"])): float(
                row["state_census_hired_workers_reported"]
            )
            for row in census_state.iter_rows(named=True)
        }

        state_denominator: defaultdict[tuple[str, int], float] = defaultdict(float)
        region_denominator: defaultdict[tuple[str, int], float] = defaultdict(float)
        national_denominator: defaultdict[int, float] = defaultdict(float)
        state_to_region = {
            row["state_fips"]: row["aewr_region_id"] for row in county_rows
        }
        for county in county_rows:
            for year in benchmark_years:
                employment = update_map[
                    (county["county_fips"], year)
                ]["annual_update_employment"]
                if _finite_nonnegative(employment):
                    value = float(employment)
                    state_denominator[(county["state_fips"], year)] += value
                    region_denominator[(county["aewr_region_id"], year)] += value
                    national_denominator[year] += value

        region_numerator: defaultdict[tuple[str, int], float] = defaultdict(float)
        national_numerator: defaultdict[int, float] = defaultdict(float)
        for (state_fips, year), value in state_total.items():
            region = state_to_region[state_fips]
            region_numerator[(region, year)] += value
            national_numerator[year] += value

        state_ratio = {
            key: state_total[key] / denominator
            for key, denominator in state_denominator.items()
            if denominator > 0
        }
        region_ratio = {
            key: region_numerator[key] / denominator
            for key, denominator in region_denominator.items()
            if denominator > 0
        }
        national_ratio = {
            year: national_numerator[year] / denominator
            for year, denominator in national_denominator.items()
            if denominator > 0
        }

        prefill: dict[tuple[str, int], dict[str, Any]] = {}
        for county in county_rows:
            county_fips = county["county_fips"]
            reported_years = observed_years[county_fips]
            for year in benchmark_years:
                key = (county_fips, year)
                if key in reported_map:
                    value = reported_map[key]
                    prefill[key] = {
                        "value": value,
                        "method": "reported",
                        "reported": True,
                        "published_zero": value == 0,
                        "structural_zero": False,
                    }
                    continue

                has_any_signal = positive_signal.get(county_fips, False)
                if not has_any_signal:
                    prefill[key] = {
                        "value": 0.0,
                        "method": "structural_zero",
                        "reported": False,
                        "published_zero": False,
                        "structural_zero": True,
                    }
                    continue

                lower_years = [
                    observed_year
                    for observed_year in reported_years
                    if observed_year < year
                ]
                upper_years = [
                    observed_year
                    for observed_year in reported_years
                    if observed_year > year
                ]
                if lower_years and upper_years:
                    lower = max(lower_years)
                    upper = min(upper_years)
                    share = (year - lower) / (upper - lower)
                    value = math.expm1(
                        (1 - share) * math.log1p(reported_map[(county_fips, lower)])
                        + share * math.log1p(reported_map[(county_fips, upper)])
                    )
                    method = "county_log1p_interpolation"
                elif reported_years:
                    neighbor = min(
                        reported_years,
                        key=lambda observed_year: (abs(observed_year - year), observed_year),
                    )
                    projected_log = math.log1p(
                        reported_map[(county_fips, neighbor)]
                    ) + _cumulative_growth(growth, county_fips, neighbor, year)
                    value = math.expm1(projected_log)
                    method = f"county_growth_projection_from_{neighbor}"
                    if value < 0:
                        value = 0.0
                        method += "_nonnegative_floor"
                else:
                    update_values = [
                        (
                            abs(candidate_year - year),
                            candidate_year,
                            update_map[
                                (county_fips, candidate_year)
                            ]["annual_update_employment"],
                        )
                        for candidate_year in frame_years
                        if _finite_nonnegative(
                            update_map[
                                (county_fips, candidate_year)
                            ]["annual_update_employment"]
                        )
                    ]
                    if update_values:
                        _, proxy_year, proxy = min(update_values)
                        proxy = float(proxy)
                    else:
                        proxy_year = year
                        proxy = 1.0

                    state_key = (county["state_fips"], year)
                    region_key = (county["aewr_region_id"], year)
                    if state_key in state_ratio:
                        ratio = state_ratio[state_key]
                        method = "state_census_to_employment_ratio"
                    elif region_key in region_ratio:
                        ratio = region_ratio[region_key]
                        method = "aewr_region_census_to_employment_ratio"
                    elif year in national_ratio:
                        ratio = national_ratio[year]
                        method = "national_census_to_employment_ratio"
                    else:
                        raise ValueError(
                            f"No Census-to-employment ratio for {county_fips}, {year}"
                        )
                    value = ratio * proxy
                    if proxy_year != year:
                        method += f"_nearest_employment_{proxy_year}"

                prefill[key] = {
                    "value": max(float(value), 0.0),
                    "method": method,
                    "reported": False,
                    "published_zero": False,
                    "structural_zero": False,
                }

        benchmark: dict[tuple[str, int], dict[str, Any]] = {}
        counties_by_state: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for county in county_rows:
            counties_by_state[county["state_fips"]].append(county)

        for state_fips, state_counties in counties_by_state.items():
            for year in benchmark_years:
                cells = [
                    prefill[(county["county_fips"], year)]
                    for county in state_counties
                ]
                reported_sum = sum(
                    cell["value"] for cell in cells if cell["reported"]
                )
                target = state_total[(state_fips, year)]
                residual = target - reported_sum
                tolerance = 1e-8 * max(1.0, target)
                if residual < -tolerance:
                    raise ValueError(
                        f"Reported county Census workers exceed the state total "
                        f"for {state_fips}, {year}"
                    )
                residual = max(residual, 0.0)
                missing_indexes = [
                    index for index, cell in enumerate(cells) if not cell["reported"]
                ]
                eligible_indexes = [
                    index
                    for index in missing_indexes
                    if not cells[index]["structural_zero"]
                ]
                initial_total = sum(cells[index]["value"] for index in missing_indexes)

                if not missing_indexes:
                    if residual > tolerance:
                        raise ValueError(
                            f"State residual has no unreported counties for "
                            f"{state_fips}, {year}"
                        )
                elif initial_total > 0:
                    factor = residual / initial_total
                    for index in missing_indexes:
                        cells[index]["value"] *= factor
                        cells[index]["state_residual_factor"] = factor
                        cells[index]["method"] += "_state_residual_scaled"
                elif residual > tolerance:
                    if not eligible_indexes:
                        raise ValueError(
                            f"A positive state residual would violate structural "
                            f"zeros for {state_fips}, {year}"
                        )
                    employment_weights = [
                        update_map[
                            (state_counties[index]["county_fips"], year)
                        ]["annual_update_employment"]
                        for index in eligible_indexes
                    ]
                    employment_total = sum(
                        float(value)
                        for value in employment_weights
                        if _finite_positive(value)
                    )
                    for index, employment in zip(
                        eligible_indexes, employment_weights
                    ):
                        if employment_total > 0 and _finite_positive(employment):
                            share = float(employment) / employment_total
                            method_suffix = "_state_residual_employment_allocated"
                        else:
                            share = 1 / len(eligible_indexes)
                            method_suffix = "_state_residual_equal_allocated"
                        cells[index]["value"] = residual * share
                        cells[index]["state_residual_factor"] = None
                        cells[index]["method"] += method_suffix
                else:
                    for index in missing_indexes:
                        cells[index]["value"] = 0.0
                        cells[index]["state_residual_factor"] = 0.0
                        cells[index]["method"] += "_zero_state_residual"

                for county, cell in zip(state_counties, cells):
                    if cell["reported"]:
                        cell["state_residual_factor"] = 1.0
                    benchmark[(county["county_fips"], year)] = dict(cell)

                filled_sum = sum(cell["value"] for cell in cells)
                if not math.isclose(
                    filled_sum,
                    target,
                    rel_tol=1e-10,
                    abs_tol=1e-7,
                ):
                    raise AssertionError(
                        f"Filled Census benchmarks miss state total for "
                        f"{state_fips}, {year}: {filled_sum} != {target}"
                    )

        farm_map: dict[tuple[str, int], float] = {}
        for row in census_farms.iter_rows(named=True):
            farm_map[(row["county_fips"], int(row["year"]))] = float(row["census_eligible_farms"])

        path_rows: list[dict[str, Any]] = []
        for county in county_rows:
            county_fips = county["county_fips"]
            for year in frame_years:
                lower_candidates = [b for b in benchmark_years if b <= year]
                if not lower_candidates:
                    raise ValueError(f"No benchmark year <= {year}")
                lower = max(lower_candidates)
            
                lower_value = benchmark[(county_fips, lower)]["value"]
                cumulative = _cumulative_growth(growth, county_fips, lower, year)
                forward_log = math.log1p(lower_value) + cumulative
                projected_value = math.expm1(forward_log)
                nonnegative_floor = projected_value < 0
                if nonnegative_floor:
                    projected_value = 0.0

                base_farms = farm_map.get((county_fips, lower), 0.0)
                frame_mass = projected_value + 1.0 * base_farms

                update = update_map[(county_fips, year)]
                is_benchmark = year in benchmark_years
                benchmark_cell = benchmark[(county_fips, year)] if is_benchmark else None
                reported_value = reported_map.get((county_fips, year))
                if not is_benchmark:
                    benchmark_status = "non_benchmark_year"
                elif reported_value is None:
                    benchmark_status = "suppressed_or_absent"
                elif reported_value == 0:
                    benchmark_status = "reported_zero"
                else:
                    benchmark_status = "reported_positive"

                path_rows.append(
                    {
                        **county,
                        "source_year": year,
                        "weight_spec": WEIGHT_SPEC,
                        "annual_update_spec": ANNUAL_UPDATE_SPEC,
                        "weight_draw_id": None,
                        "census_benchmark_status": benchmark_status,
                        "census_benchmark_reported": bool(
                            benchmark_cell and benchmark_cell["reported"]
                        ),
                        "census_published_zero": bool(
                            benchmark_cell and benchmark_cell["published_zero"]
                        ),
                        "census_hired_workers_reported": reported_value,
                        "census_hired_workers_benchmark_filled": (
                            benchmark_cell["value"] if benchmark_cell else None
                        ),
                        "census_benchmark_fill_method": (
                            benchmark_cell["method"] if benchmark_cell else None
                        ),
                        "census_benchmark_prefill": (
                            prefill[(county_fips, year)]["value"]
                            if is_benchmark
                            else None
                        ),
                        "census_benchmark_state_residual_factor": (
                            benchmark_cell["state_residual_factor"]
                            if benchmark_cell
                            else None
                        ),
                        "structural_zero": bool(
                            benchmark_cell and benchmark_cell["structural_zero"]
                        ),
                        "lower_census_benchmark_year": lower,
                        "upper_census_benchmark_year": lower,
                        "lower_census_hired_workers_filled": lower_value,
                        "upper_census_hired_workers_filled": lower_value,
                        "interval_share": 0.0,
                        "qcew_ag_employment": update["qcew_ag_employment"],
                        "qcew_strict_complete": update["qcew_strict_complete"],
                        "qcew_observed_cells": update["qcew_observed_cells"],
                        "qcew_valid_cells": update["qcew_valid_cells"],
                        "qwi_ag_employment": update["qwi_ag_employment"],
                        "qwi_strict_complete": update["qwi_strict_complete"],
                        "qwi_observed_cells": update["qwi_observed_cells"],
                        "qwi_valid_cells": update["qwi_valid_cells"],
                        "bea_hired_farm_jobs": update["bea_hired_farm_jobs"],
                        "annual_update_employment": update["annual_update_employment"],
                        "annual_update_source": update["annual_update_source"],
                        "annual_growth_source": update["annual_growth_source"],
                        "annual_log1p_growth": update["annual_log1p_growth"],
                        "annual_growth_factor_log1p": update["annual_growth_factor_log1p"],
                        "two_sided_log1p_drift": 0.0,
                        "frame_employment_mass_prerake": projected_value,
                        "state_census_hired_workers_target": 0.0,
                        "state_frame_employment_prerake": 0.0,
                        "state_rake_factor": 1.0,
                        "frame_employment_mass": frame_mass,
                        "qwi_annual_fallback_used": update["qwi_annual_fallback_used"],
                        "bea_annual_fallback_used": update["bea_annual_fallback_used"],
                        "unit_growth_fallback_used": update["unit_growth_fallback_used"],
                        "extreme_annual_change": update["extreme_annual_change"],
                        "nonnegative_floor_applied": nonnegative_floor,
                    }
                )

        for row in path_rows:
            row["update_imputation_source"] = (
                row["census_benchmark_fill_method"]
                if row["census_benchmark_fill_method"] is not None
                else row["annual_growth_source"]
            )
            row["quality_flags"] = _quality_flags(row)

        result = pl.DataFrame(path_rows).with_columns(
            pl.col("source_year").cast(pl.Int32),
            pl.col("weight_draw_id").cast(pl.Int64),
            pl.col("qcew_observed_cells").fill_null(0).cast(pl.Int32),
            pl.col("qcew_valid_cells").fill_null(0).cast(pl.Int32),
            pl.col("qwi_observed_cells").fill_null(0).cast(pl.Int32),
            pl.col("qwi_valid_cells").fill_null(0).cast(pl.Int32),
        )
        _require_unique(result, _ANNUAL_KEYS, "frame employment analog")
        masses = result.get_column("frame_employment_mass")
        if masses.null_count() or not masses.is_finite().all() or (masses < 0).any():
            raise AssertionError("Frame masses must be finite and nonnegative")
        return result.sort(*_ANNUAL_KEYS)

    return (
        ANNUAL_UPDATE_SPEC,
        BENCHMARK_YEARS,
        INTERMEDIATE,
        PROCESSED,
        WEIGHT_SPEC,
        assert_geo_columns,
        build_frame_employment_analog,
        pl,
    )


@app.cell
def _(INTERMEDIATE):
    OUTPUT_PATH = INTERMEDIATE / "panel_iv_fls_frame.parquet"
    DIAGNOSTIC_PATH = INTERMEDIATE / "panel_iv_fls_frame_diagnostics.parquet"
    return DIAGNOSTIC_PATH, OUTPUT_PATH


@app.cell
def _(BENCHMARK_YEARS, INTERMEDIATE, pl):
    def _parse_quickstats_value() -> pl.Expr:
        return (
            pl.when(pl.col("value") == "(Z)")
            .then(pl.lit(0.0))
            .otherwise(
                pl.col("value").str.replace_all(",", "").cast(pl.Float64, strict=False)
            )
        )

    def read_state_census_benchmarks() -> pl.DataFrame:
        """
        Read published state hired-worker totals from the local Quick Stats file.
        """
        return (
            pl.scan_parquet(INTERMEDIATE / "qs_census_economics.parquet")
            .filter(
                pl.col("year").is_in(BENCHMARK_YEARS),
                pl.col("agg_level_desc") == "STATE",
                pl.col("freq_desc") == "ANNUAL",
                pl.col("reference_period_desc") == "YEAR",
                pl.col("commodity_desc") == "LABOR",
                pl.col("domain_desc") == "TOTAL",
                pl.col("prodn_practice_desc") == "ALL PRODUCTION PRACTICES",
                pl.col("short_desc") == "LABOR, HIRED - NUMBER OF WORKERS",
            )
            .select(
                pl.col("state_fips").cast(pl.String).str.pad_start(2, "0"),
                pl.col("year").cast(pl.Int32),
                _parse_quickstats_value().alias("state_census_hired_workers_reported"),
            )
            .collect()
            .sort("state_fips", "year")
        )

    def read_county_census_farms() -> pl.DataFrame:
        """
        Read published county farm counts from the local Quick Stats file for the prior floor.
        """
        return (
            pl.scan_parquet(INTERMEDIATE / "qs_census_economics.parquet")
            .filter(
                pl.col("year").is_in(BENCHMARK_YEARS),
                pl.col("agg_level_desc") == "COUNTY",
                pl.col("short_desc").cast(pl.String) == "FARM OPERATIONS - NUMBER OF OPERATIONS",
            )
            .select(
                (pl.col("state_fips").cast(pl.String).str.pad_start(2, "0") + pl.col("county_code").cast(pl.String).str.pad_start(3, "0")).alias("county_fips"),
                pl.col("year").cast(pl.Int32),
                _parse_quickstats_value().alias("census_eligible_farms"),
            )
            .collect()
            .sort("county_fips", "year")
        )

    return (read_state_census_benchmarks, read_county_census_farms)


@app.cell
def _(
    ANNUAL_UPDATE_SPEC,
    DIAGNOSTIC_PATH,
    INTERMEDIATE,
    OUTPUT_PATH,
    PROCESSED,
    WEIGHT_SPEC,
    assert_geo_columns,
    build_frame_employment_analog,
    pl,
    read_state_census_benchmarks,
    read_county_census_farms,
):
    county_panel = pl.read_parquet(PROCESSED / "county_year_panel.parquet")
    counties = county_panel.select(
        "county_fips",
        "state_fips",
        "state_abbrev",
        "aewr_region_id",
    ).unique()
    census_county = pl.read_parquet(
        INTERMEDIATE / "census_ag_hired_worker_duration_county.parquet"
    )
    census_state = read_state_census_benchmarks()
    census_farms = read_county_census_farms()
    qcew = pl.read_parquet(INTERMEDIATE / "qcew_county_ag_quarterly_employment.parquet")
    qwi = pl.read_parquet(INTERMEDIATE / "qwi_county_ag_quarterly_employment.parquet")
    bea = pl.read_parquet(INTERMEDIATE / "bea_caemp25n_data_year.parquet")

    frame = build_frame_employment_analog(
        counties,
        census_county,
        census_state,
        census_farms,
        qcew,
        qwi,
        bea,
    )
    assert_geo_columns(
        frame,
        ["county_fips", "state_fips", "aewr_region_id"],
    )
    if frame.get_column("weight_spec").unique().to_list() != [WEIGHT_SPEC]:
        raise AssertionError("Unexpected frame weight specification")
    if frame.get_column("annual_update_spec").unique().to_list() != [
        ANNUAL_UPDATE_SPEC
    ]:
        raise AssertionError("Unexpected annual update specification")
    frame.write_parquet(OUTPUT_PATH)

    diagnostics = frame.filter(
        (pl.col("census_benchmark_reported").not_())
        & pl.col("census_hired_workers_benchmark_filled").is_not_null()
        | pl.col("qwi_annual_fallback_used")
        | pl.col("bea_annual_fallback_used")
        | pl.col("unit_growth_fallback_used")
        | pl.col("extreme_annual_change")
        | pl.col("nonnegative_floor_applied")
    )
    diagnostics.write_parquet(DIAGNOSTIC_PATH)

    summary = (
        frame.group_by("source_year")
        .agg(
            pl.len().alias("county_rows"),
            (pl.col("annual_update_source") == "qcew").sum().alias("qcew_updates"),
            (pl.col("annual_update_source") == "qwi").sum().alias("qwi_fallbacks"),
            (pl.col("annual_update_source") == "bea").sum().alias("bea_fallbacks"),
            pl.col("unit_growth_fallback_used").sum().alias("unit_growth_fallbacks"),
            pl.col("extreme_annual_change").sum().alias("extreme_changes"),
            pl.col("state_rake_factor").min().alias("minimum_rake_factor"),
            pl.col("state_rake_factor").max().alias("maximum_rake_factor"),
        )
        .sort("source_year")
    )
    print(summary)
    print(f"Wrote {frame.height:,} rows to {OUTPUT_PATH}")
    print(f"Wrote {diagnostics.height:,} flagged rows to {DIAGNOSTIC_PATH}")
    return


if __name__ == "__main__":
    app.run()
