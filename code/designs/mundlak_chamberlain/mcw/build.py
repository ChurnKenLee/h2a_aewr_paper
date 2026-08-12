"""Build the v4 Mundlak--Chamberlain--Wooldridge county-year panel.

The source Parquet is scanned by DuckDB and every returned object is a Polars
``DataFrame``.  This module deliberately contains no pandas or pickle path.

Stable treatment coordinates
----------------------------
For calendar coordinate ``h`` in 2011--2022, the public mapping
``TREATMENT_PATH_COLUMNS[family][h]`` names the raw column.  In particular,

``mc_aewr_log_level_h``
    ``100 * (log(real AEWR[r,h]) - log(real AEWR[r,2010]))``;
``mc_aewr_dollar_level_h``
    ``real AEWR[r,h] - real AEWR[r,2010]``;
``mc_aewr_log_change_h``
    ``100 * (log(real AEWR[r,h]) - log(real AEWR[r,h-1]))``;
``mc_bite_f0809_h`` and ``mc_bite_f0810_h``
    frozen-distribution dollar shortfalls at the nominal AEWR and the
    contemporaneous non-AEWR agricultural wage floor;
``mc_exposure_log_f0809_h`` and ``mc_exposure_log_f0810_h``
    the corresponding pre-period mean fraction affected times
    ``mc_aewr_log_level_h``.

The v4 treatment history is 2011--2022 and the outcome calendar is 2012--2022,
as explicitly approved after rejecting 2011 as an outcome-side anchor.  The
2011 treatment coordinate remains available to every subsequent outcome year.

Use :func:`full_history_columns` to select exactly the admissible 2011--t
coordinates.  Use :func:`one_lag_columns` for the declared current-plus-lag-1
benchmark.  The row-relative primary benchmark is also materialized as
``mc_aewr_log_level_current`` and ``mc_aewr_log_level_lag1``.

The bite is explicitly an approximation: each of p10/p25/p50/p75/p90 is a
point mass whose probability is the nearest-quantile cell width.  Its weights
are 0.175, 0.200, 0.250, 0.200, and 0.175.  Frozen quantiles are county means
over 2008--09 or 2008--10.  Support-count columns expose incomplete windows
rather than silently pretending that every county has the same baseline.

``mc_baseline_farm_employment`` is the finite-value 2008--10 mean of
``emp_farm``.  It is the declared downstream eligibility variable and
constructed-outcome denominator; this builder does not substitute the
single-year ``emp_farm_2011`` field.  Baseline moderator z scores use moments
from counties with positive ``mc_baseline_farm_employment``.  Missing moderator
means are retained raw, marked by ``*_missing``, completed in separate
``*_imputed`` columns by eligible-region median and then eligible-national
median, and labeled by ``mc_baseline_imputation_method``.
"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import duckdb
import polars as pl

AEWR_BASE_YEAR = 2010
FULL_HISTORY_YEARS = tuple(range(2011, 2023))
ANALYSIS_YEARS = tuple(range(2012, 2023))

BASELINE_WINDOWS: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "f0809": (2008, 2009),
        "f0810": (2008, 2009, 2010),
    }
)

QUANTILE_COLUMNS = (
    "wage_p10",
    "wage_p25",
    "wage_p50",
    "wage_p75",
    "wage_p90",
)
QUANTILE_POINT_MASS_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "wage_p10": 0.175,
        "wage_p25": 0.200,
        "wage_p50": 0.250,
        "wage_p75": 0.200,
        "wage_p90": 0.175,
    }
)
BITE_APPROXIMATION_LABEL = "five_quantile_nearest_cell_point_mass"

# Mirrors BASELINE_VARIABLES in the executable v4 Python design contract.  The
# construction layer exposes raw 2008--10 means; downstream design code decides
# how to project or interact them.  Keeping this mapping public avoids a second
# hard-coded source list.
BASELINE_VARIABLE_SOURCES: Mapping[str, str] = MappingProxyType(
    {
        "h2a_applications": "nbr_applications_start_year",
        "h2a_certified_positions": "nbr_workers_certified_start_year",
        "log_population": "ln_pop_census",
        "farm_employment_share": "farm_emp_share",
        "employment_population_ratio": "emp_pop_ratio",
        "crop_income_share": "share_farm_crop_cashandinc",
        "hired_labor_cost_share": "share_farm_laborexp_prodexp",
        "low_wage": "wage_p25",
        "animal_income_share": "share_farm_animal_cashandinc",
        "production_expense_share": "share_farm_prodexp_cashandinc",
        "median_wage": "wage_p50",
        "cropland": "census_cropland_2007",
    }
)
BASELINE_MEAN_COLUMNS: Mapping[str, str] = MappingProxyType(
    {name: f"mc_baseline_{name}" for name in BASELINE_VARIABLE_SOURCES}
)
BASELINE_IMPUTED_COLUMNS: Mapping[str, str] = MappingProxyType(
    {
        name: f"mc_baseline_{name}_imputed"
        for name in ("bite", *BASELINE_VARIABLE_SOURCES)
    }
)
BASELINE_MISSING_INDICATOR_COLUMNS: Mapping[str, str] = MappingProxyType(
    {
        name: f"mc_baseline_{name}_missing"
        for name in ("bite", *BASELINE_VARIABLE_SOURCES)
    }
)
BASELINE_IMPUTATION_LABEL = "eligible_aewr_region_median_then_eligible_national_median"
BASELINE_FARM_EMPLOYMENT_COLUMN = "mc_baseline_farm_employment"
BASELINE_BITE_COLUMN = "mc_baseline_bite"
STANDARDIZED_BASELINE_COLUMNS: Mapping[str, str] = MappingProxyType(
    {
        "bite": "mc_baseline_bite_z",
        **{name: f"mc_baseline_{name}_z" for name in BASELINE_VARIABLE_SOURCES},
    }
)

SOURCE_OUTCOME_COLUMNS: Mapping[str, str] = MappingProxyType(
    {
        "applications": "nbr_applications_start_year",
        "employers": "nbr_employers_balanced_start_year",
        "requested_positions": "nbr_workers_requested_start_year",
        "certified_positions": "nbr_workers_certified_start_year",
        "certified_hours": "man_hours_certified_start_year",
    }
)
OUTCOME_COLUMNS: Mapping[str, str] = MappingProxyType(
    {
        "applications": "mc_y_applications",
        "employers": "mc_y_employers_balanced",
        "requested_positions": "mc_y_requested_positions",
        "certified_positions": "mc_y_certified_positions",
        "certified_hours": "mc_y_certified_hours",
        "any_application": "mc_y_any_application",
    }
)

CLUSTER_COLUMNS: Mapping[str, str] = MappingProxyType(
    {
        "region": "mc_cluster_region",
        "county": "mc_cluster_county",
        "state": "mc_cluster_state",
        "cz_region": "mc_cluster_cz_region",
    }
)

_PATH_COLUMN_TEMPLATES: Mapping[str, str] = MappingProxyType(
    {
        "aewr_log_level": "mc_aewr_log_level_{year}",
        "aewr_dollar_level": "mc_aewr_dollar_level_{year}",
        "aewr_log_change": "mc_aewr_log_change_{year}",
        "bite_f0809": "mc_bite_f0809_{year}",
        "bite_f0810": "mc_bite_f0810_{year}",
        "exposure_log_f0809": "mc_exposure_log_f0809_{year}",
        "exposure_log_f0810": "mc_exposure_log_f0810_{year}",
    }
)
TREATMENT_PATH_COLUMNS: Mapping[str, Mapping[int, str]] = MappingProxyType(
    {
        family: MappingProxyType(
            {year: template.format(year=year) for year in FULL_HISTORY_YEARS}
        )
        for family, template in _PATH_COLUMN_TEMPLATES.items()
    }
)

FRACTION_AFFECTED_COLUMNS: Mapping[str, str] = MappingProxyType(
    {tag: f"mc_fraction_affected_{tag}" for tag in BASELINE_WINDOWS}
)
ONE_LAG_BENCHMARK_COLUMNS = (
    "mc_aewr_log_level_current",
    "mc_aewr_log_level_lag1",
)

GEOGRAPHY_COLUMNS = (
    "county_fips",
    "state_fips",
    "cz_id",
    "aewr_region_id",
)
REQUIRED_SOURCE_COLUMNS = (
    *GEOGRAPHY_COLUMNS,
    "year",
    "aewr",
    "aewr_ppi",
    "prevailing_ag_min_wage",
    *QUANTILE_COLUMNS,
    "emp_farm",
    *BASELINE_VARIABLE_SOURCES.values(),
    *SOURCE_OUTCOME_COLUMNS.values(),
)


class PanelBuildError(ValueError):
    """Raised when an input or constructed panel violates the build contract."""


def treatment_coordinate_column(family: str, year: int) -> str:
    """Return the stable raw-column name for one treatment coordinate."""

    if family not in TREATMENT_PATH_COLUMNS:
        choices = ", ".join(TREATMENT_PATH_COLUMNS)
        raise PanelBuildError(
            f"unknown treatment family {family!r}; choose from {choices}"
        )
    if year not in TREATMENT_PATH_COLUMNS[family]:
        raise PanelBuildError(
            f"treatment coordinate year {year} is outside "
            f"{FULL_HISTORY_YEARS[0]}--{FULL_HISTORY_YEARS[-1]}"
        )
    return TREATMENT_PATH_COLUMNS[family][year]


def full_history_columns(
    outcome_year: int,
    family: str = "aewr_log_level",
) -> tuple[str, ...]:
    """Return chronological 2011--``outcome_year`` treatment coordinates."""

    if outcome_year not in FULL_HISTORY_YEARS:
        raise PanelBuildError(f"invalid full-history outcome year {outcome_year}")
    return tuple(
        treatment_coordinate_column(family, year)
        for year in FULL_HISTORY_YEARS
        if year <= outcome_year
    )


def one_lag_columns(
    outcome_year: int,
    family: str = "aewr_log_level",
) -> tuple[str, ...]:
    """Return chronological ``max(2011, t-1)..t`` benchmark coordinates."""

    if outcome_year not in FULL_HISTORY_YEARS:
        raise PanelBuildError(f"invalid one-lag outcome year {outcome_year}")
    return tuple(
        treatment_coordinate_column(family, year)
        for year in range(
            max(FULL_HISTORY_YEARS[0], outcome_year - 1), outcome_year + 1
        )
    )


def _validate_year_arguments(
    history_years: Sequence[int],
    analysis_years: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    history = tuple(history_years)
    analysis = tuple(analysis_years)
    if not history or history != tuple(range(history[0], history[-1] + 1)):
        raise PanelBuildError("history_years must be a nonempty consecutive sequence")
    if history[0] != FULL_HISTORY_YEARS[0]:
        raise PanelBuildError("history_years must start in 2011")
    if any(year not in FULL_HISTORY_YEARS for year in history):
        raise PanelBuildError("history_years must lie within 2011--2022")
    if not analysis or tuple(sorted(set(analysis))) != analysis:
        raise PanelBuildError("analysis_years must be nonempty, sorted, and unique")
    if any(year not in history for year in analysis):
        raise PanelBuildError("analysis_years must be contained in history_years")
    return history, analysis


def _scan_parquet(source_path: str | Path) -> pl.DataFrame:
    path = Path(source_path)
    if not path.is_file():
        raise PanelBuildError(f"source Parquet does not exist: {path}")

    con = duckdb.connect(database=":memory:")
    try:
        description = con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
        ).fetchall()
        source_columns = tuple(row[0] for row in description)
        missing = sorted(set(REQUIRED_SOURCE_COLUMNS) - set(source_columns))
        if missing:
            raise PanelBuildError(f"source Parquet is missing columns: {missing}")
        arrow = con.execute(
            "SELECT * FROM read_parquet(?)", [str(path)]
        ).to_arrow_table()
    finally:
        con.close()
    frame = pl.from_arrow(arrow)
    if not isinstance(frame, pl.DataFrame):
        frame = frame.collect()
    if frame.is_empty():
        raise PanelBuildError("source Parquet contains no rows")
    return frame


def _normalize_and_validate_keys(frame: pl.DataFrame) -> pl.DataFrame:
    for column in GEOGRAPHY_COLUMNS:
        if frame.schema[column] != pl.String:
            raise PanelBuildError(
                f"{column} must be stored as a string; refusing a cast that could lose zeros"
            )
        if frame[column].null_count() or (frame[column].str.len_chars() == 0).any():
            raise PanelBuildError(f"{column} must be nonempty and nonmissing")

    year = frame["year"].cast(pl.Float64, strict=True)
    invalid_year = (
        year.is_null() | year.is_nan() | year.is_infinite() | (year != year.floor())
    )
    if invalid_year.any():
        raise PanelBuildError("year must contain finite integer-valued observations")
    frame = frame.with_columns(pl.col("year").cast(pl.Int32, strict=True))

    duplicates = (
        frame.group_by(["county_fips", "year"]).len().filter(pl.col("len") != 1)
    )
    if not duplicates.is_empty():
        example = duplicates.head(3).to_dicts()
        raise PanelBuildError(f"county-year keys are not unique; examples: {example}")

    mapping = frame.group_by("county_fips").agg(
        *(pl.col(column).n_unique().alias(column) for column in GEOGRAPHY_COLUMNS[1:])
    )
    unstable = mapping.filter(
        pl.any_horizontal(*(pl.col(column) != 1 for column in GEOGRAPHY_COLUMNS[1:]))
    )
    if not unstable.is_empty():
        raise PanelBuildError(
            "county geography changes over time; examples: "
            f"{unstable.head(3).to_dicts()}"
        )
    return frame


def _assert_finite_nonnegative_outcomes(frame: pl.DataFrame) -> None:
    failures: list[str] = []
    for column in SOURCE_OUTCOME_COLUMNS.values():
        values = frame[column].cast(pl.Float64, strict=True)
        invalid = values.is_not_null() & (
            values.is_nan() | values.is_infinite() | (values < 0)
        )
        if invalid.any():
            failures.append(column)
    if failures:
        raise PanelBuildError(
            f"primitive outcome sources must be finite and nonnegative: {failures}"
        )
    farm_employment = frame["emp_farm"].cast(pl.Float64, strict=True)
    if (farm_employment.is_finite() & (farm_employment < 0)).any():
        raise PanelBuildError("emp_farm must be nonnegative when observed")


def _region_aewr_paths(
    frame: pl.DataFrame,
    history_years: tuple[int, ...],
) -> pl.DataFrame:
    required_years = (AEWR_BASE_YEAR, *history_years)
    region_rows = frame.filter(pl.col("year").is_in(required_years)).select(
        "aewr_region_id", "year", "aewr", "aewr_ppi"
    )
    inconsistent = (
        region_rows.group_by(["aewr_region_id", "year"])
        .agg(
            pl.col("aewr").n_unique().alias("nominal_values"),
            pl.col("aewr_ppi").n_unique().alias("real_values"),
        )
        .filter((pl.col("nominal_values") != 1) | (pl.col("real_values") != 1))
    )
    if not inconsistent.is_empty():
        raise PanelBuildError(
            "AEWR must be unique within region-year from 2010 onward; examples: "
            f"{inconsistent.head(3).to_dicts()}"
        )

    region = region_rows.unique().sort(["aewr_region_id", "year"])
    expected = len(required_years)
    incomplete = (
        region.group_by("aewr_region_id").len().filter(pl.col("len") != expected)
    )
    if not incomplete.is_empty():
        raise PanelBuildError(
            "each analysis region needs complete 2010--history AEWR support; examples: "
            f"{incomplete.head(3).to_dicts()}"
        )

    for column in ("aewr", "aewr_ppi"):
        values = region[column].cast(pl.Float64, strict=True)
        invalid = (
            values.is_null() | values.is_nan() | values.is_infinite() | (values <= 0)
        )
        if invalid.any():
            raise PanelBuildError(
                f"{column} must be finite and positive for AEWR paths"
            )
    return region


def _validate_baseline_quantiles(frame: pl.DataFrame) -> None:
    baseline_years = sorted(
        {year for years in BASELINE_WINDOWS.values() for year in years}
    )
    baseline = frame.filter(pl.col("year").is_in(baseline_years))
    numeric = [
        pl.col(column).cast(pl.Float64, strict=True) for column in QUANTILE_COLUMNS
    ]
    complete = pl.all_horizontal(
        *(value.is_not_null() & value.is_finite() & (value > 0) for value in numeric)
    )
    bad_order = complete & ~pl.all_horizontal(
        *(numeric[index] <= numeric[index + 1] for index in range(len(numeric) - 1))
    )
    if baseline.select(bad_order.any()).item():
        raise PanelBuildError(
            "baseline wage quantiles must be positive and nondecreasing"
        )


def _frozen_quantiles(frame: pl.DataFrame, tag: str) -> pl.DataFrame:
    years = BASELINE_WINDOWS[tag]
    numeric = [
        pl.col(column).cast(pl.Float64, strict=True) for column in QUANTILE_COLUMNS
    ]
    complete = pl.all_horizontal(
        *(value.is_not_null() & value.is_finite() & (value > 0) for value in numeric)
    )
    aliases = {column: f"mc_frozen_{column}_{tag}" for column in QUANTILE_COLUMNS}
    return (
        frame.filter(pl.col("year").is_in(years) & complete)
        .group_by("county_fips")
        .agg(
            pl.len().cast(pl.Int8).alias(f"mc_baseline_year_count_{tag}"),
            *(
                pl.col(column).mean().alias(aliases[column])
                for column in QUANTILE_COLUMNS
            ),
        )
    )


def _finite_baseline_mean(column: str, alias: str) -> pl.Expr:
    value = pl.col(column).cast(pl.Float64, strict=True)
    return pl.when(value.is_finite()).then(value).otherwise(None).mean().alias(alias)


def _baseline_means(frame: pl.DataFrame) -> pl.DataFrame:
    baseline = frame.filter(pl.col("year").is_in(BASELINE_WINDOWS["f0810"]))
    farm_employment = pl.col("emp_farm").cast(pl.Float64, strict=True)
    real_aewr = pl.col("aewr_ppi").cast(pl.Float64, strict=True)
    low_wage = pl.col("wage_p25").cast(pl.Float64, strict=True)
    baseline_bite = real_aewr - low_wage
    return baseline.group_by("county_fips").agg(
        _finite_baseline_mean("emp_farm", BASELINE_FARM_EMPLOYMENT_COLUMN),
        pl.when(real_aewr.is_finite() & low_wage.is_finite())
        .then(baseline_bite)
        .otherwise(None)
        .mean()
        .alias(BASELINE_BITE_COLUMN),
        pl.when(farm_employment.is_finite())
        .then(1)
        .otherwise(None)
        .count()
        .cast(pl.Int8)
        .alias("mc_baseline_farm_employment_year_count"),
        *(
            _finite_baseline_mean(source, BASELINE_MEAN_COLUMNS[name])
            for name, source in BASELINE_VARIABLE_SOURCES.items()
        ),
    )


def _add_standardized_baselines(county_paths: pl.DataFrame) -> pl.DataFrame:
    eligible = pl.col(BASELINE_FARM_EMPLOYMENT_COLUMN).is_finite() & (
        pl.col(BASELINE_FARM_EMPLOYMENT_COLUMN) > 0
    )
    eligible_frame = county_paths.filter(eligible)
    if eligible_frame.is_empty():
        raise PanelBuildError("no county has positive 2008--10 mean farm employment")

    # Raw means remain untouched.  Every imputation gets its own value column
    # and missing indicator.  Donor medians use only the declared eligible
    # population, first within AEWR region and then nationally as a fallback.
    imputation_sources = {
        "bite": BASELINE_BITE_COLUMN,
        **BASELINE_MEAN_COLUMNS,
    }
    for name, source in imputation_sources.items():
        imputed = BASELINE_IMPUTED_COLUMNS[name]
        missing = BASELINE_MISSING_INDICATOR_COLUMNS[name]
        source_is_finite = pl.col(source).is_finite().fill_null(False)
        national_values = eligible_frame.filter(pl.col(source).is_finite())[source]
        national_median = national_values.median()
        if national_median is None or not math.isfinite(national_median):
            raise PanelBuildError(
                f"no finite eligible national median for baseline variable {source}"
            )
        region_median = eligible_frame.group_by("__baseline_region").agg(
            pl.col(source)
            .filter(pl.col(source).is_finite())
            .median()
            .alias("__region_median")
        )
        county_paths = (
            county_paths.join(
                region_median,
                on="__baseline_region",
                how="left",
                validate="m:1",
            )
            .with_columns(
                (~source_is_finite).cast(pl.Int8).alias(missing),
                pl.when(source_is_finite)
                .then(pl.col(source))
                .otherwise(pl.coalesce("__region_median", pl.lit(national_median)))
                .alias(imputed),
            )
            .drop("__region_median")
        )
        eligible_frame = county_paths.filter(eligible)

    z_sources = {name: BASELINE_IMPUTED_COLUMNS[name] for name in imputation_sources}
    expressions: list[pl.Expr] = []
    for name, source in z_sources.items():
        finite = eligible_frame.filter(pl.col(source).is_finite())
        if finite.is_empty():
            raise PanelBuildError(
                f"no finite eligible values for baseline moderator {source}"
            )
        mean = finite[source].mean()
        standard_deviation = finite[source].std(ddof=0)
        if standard_deviation is None or not math.isfinite(standard_deviation):
            raise PanelBuildError(
                f"baseline moderator {source} has no finite dispersion"
            )
        if standard_deviation <= 0:
            raise PanelBuildError(f"baseline moderator {source} has zero dispersion")
        expressions.append(
            ((pl.col(source) - mean) / standard_deviation).alias(
                STANDARDIZED_BASELINE_COLUMNS[name]
            )
        )
    return county_paths.with_columns(
        *expressions,
        pl.lit(BASELINE_IMPUTATION_LABEL).alias("mc_baseline_imputation_method"),
    ).drop("__baseline_region")


def _bite_expression(tag: str, year: int) -> pl.Expr:
    aewr = pl.col(f"__nominal_aewr_{year}")
    floor = pl.col(f"__ag_floor_{year}")
    quantiles = [pl.col(f"mc_frozen_{column}_{tag}") for column in QUANTILE_COLUMNS]
    complete = (
        aewr.is_not_null()
        & aewr.is_finite()
        & floor.is_not_null()
        & floor.is_finite()
        & pl.all_horizontal(*(quantile.is_not_null() for quantile in quantiles))
    )
    shortfalls = [
        (aewr - pl.max_horizontal(quantile, floor)).clip(lower_bound=0.0)
        * QUANTILE_POINT_MASS_WEIGHTS[column]
        for column, quantile in zip(QUANTILE_COLUMNS, quantiles, strict=True)
    ]
    return (
        pl.when(complete)
        .then(pl.sum_horizontal(*shortfalls))
        .otherwise(None)
        .alias(treatment_coordinate_column(f"bite_{tag}", year))
    )


def _baseline_fraction_affected(frame: pl.DataFrame, tag: str) -> pl.DataFrame:
    """Return a pre-period mean binding share with no special post-2010 year."""

    aewr = pl.col("aewr").cast(pl.Float64, strict=True)
    floor = pl.col("prevailing_ag_min_wage").cast(pl.Float64, strict=True)
    quantiles = [
        pl.col(column).cast(pl.Float64, strict=True) for column in QUANTILE_COLUMNS
    ]
    complete = (
        aewr.is_finite()
        & floor.is_finite()
        & pl.all_horizontal(*(quantile.is_finite() for quantile in quantiles))
    )
    affected = [
        (pl.max_horizontal(quantile, floor) < aewr).cast(pl.Float64)
        * QUANTILE_POINT_MASS_WEIGHTS[column]
        for column, quantile in zip(QUANTILE_COLUMNS, quantiles, strict=True)
    ]
    fraction = (
        pl.when(complete)
        .then(pl.sum_horizontal(*affected))
        .otherwise(None)
        .alias("__baseline_fraction_affected")
    )
    return (
        frame.filter(pl.col("year").is_in(BASELINE_WINDOWS[tag]))
        .with_columns(fraction)
        .group_by("county_fips")
        .agg(
            pl.col("__baseline_fraction_affected")
            .mean()
            .alias(FRACTION_AFFECTED_COLUMNS[tag])
        )
    )


def _wide_county_paths(
    frame: pl.DataFrame,
    history_years: tuple[int, ...],
) -> pl.DataFrame:
    county_paths = (
        frame.select("county_fips", pl.col("aewr_region_id").alias("__baseline_region"))
        .unique()
        .join(_baseline_means(frame), on="county_fips", how="left", validate="1:1")
    )
    for tag in BASELINE_WINDOWS:
        county_paths = county_paths.join(
            _frozen_quantiles(frame, tag), on="county_fips", how="left"
        ).join(
            _baseline_fraction_affected(frame, tag),
            on="county_fips",
            how="left",
            validate="1:1",
        )

    for year in history_years:
        year_floor = frame.filter(pl.col("year") == year).select(
            "county_fips",
            pl.col("aewr").cast(pl.Float64).alias(f"__nominal_aewr_{year}"),
            pl.col("prevailing_ag_min_wage")
            .cast(pl.Float64)
            .alias(f"__ag_floor_{year}"),
        )
        county_paths = county_paths.join(year_floor, on="county_fips", how="left")
        expressions: list[pl.Expr] = []
        for tag in BASELINE_WINDOWS:
            expressions.append(_bite_expression(tag, year))
        county_paths = county_paths.with_columns(expressions)

    internal = [
        column
        for column in county_paths.columns
        if column.startswith("__") and column != "__baseline_region"
    ]
    county_paths = county_paths.drop(internal)
    return _add_standardized_baselines(county_paths)


def _wide_region_paths(
    region: pl.DataFrame,
    history_years: tuple[int, ...],
) -> pl.DataFrame:
    base = region.filter(pl.col("year") == AEWR_BASE_YEAR).select(
        "aewr_region_id",
        pl.col("aewr_ppi").cast(pl.Float64).alias("__real_aewr_base"),
    )
    wide = (
        region.select("aewr_region_id")
        .unique()
        .join(base, on="aewr_region_id", how="left")
    )
    for year in history_years:
        current = region.filter(pl.col("year") == year).select(
            "aewr_region_id",
            pl.col("aewr_ppi").cast(pl.Float64).alias("__real_aewr_current"),
        )
        prior = region.filter(pl.col("year") == year - 1).select(
            "aewr_region_id",
            pl.col("aewr_ppi").cast(pl.Float64).alias("__real_aewr_prior"),
        )
        wide = wide.join(current, on="aewr_region_id", how="left").join(
            prior, on="aewr_region_id", how="left"
        )
        wide = wide.with_columns(
            (
                100.0
                * (
                    pl.col("__real_aewr_current").log()
                    - pl.col("__real_aewr_base").log()
                )
            ).alias(treatment_coordinate_column("aewr_log_level", year)),
            (pl.col("__real_aewr_current") - pl.col("__real_aewr_base")).alias(
                treatment_coordinate_column("aewr_dollar_level", year)
            ),
            (
                100.0
                * (
                    pl.col("__real_aewr_current").log()
                    - pl.col("__real_aewr_prior").log()
                )
            ).alias(treatment_coordinate_column("aewr_log_change", year)),
        ).drop("__real_aewr_current", "__real_aewr_prior")
    return wide.drop("__real_aewr_base")


def _add_outcomes_and_clusters(frame: pl.DataFrame) -> pl.DataFrame:
    outcome_expressions = [
        pl.col(source).cast(pl.Float64).alias(OUTCOME_COLUMNS[name])
        for name, source in SOURCE_OUTCOME_COLUMNS.items()
    ]
    applications = pl.col(SOURCE_OUTCOME_COLUMNS["applications"]).cast(pl.Float64)
    outcome_expressions.append(
        pl.when(applications.is_null())
        .then(None)
        .otherwise((applications > 0).cast(pl.Int8))
        .alias(OUTCOME_COLUMNS["any_application"])
    )
    return frame.with_columns(
        *outcome_expressions,
        pl.col("aewr_region_id").alias(CLUSTER_COLUMNS["region"]),
        pl.col("county_fips").alias(CLUSTER_COLUMNS["county"]),
        pl.col("state_fips").alias(CLUSTER_COLUMNS["state"]),
        (pl.col("cz_id") + pl.lit("::") + pl.col("aewr_region_id")).alias(
            CLUSTER_COLUMNS["cz_region"]
        ),
        pl.lit(BITE_APPROXIMATION_LABEL).alias("mc_bite_approximation_method"),
    )


def _add_exposures_and_benchmark(
    frame: pl.DataFrame,
    history_years: tuple[int, ...],
    analysis_years: tuple[int, ...],
) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for tag in BASELINE_WINDOWS:
        fraction = pl.col(FRACTION_AFFECTED_COLUMNS[tag])
        for year in history_years:
            expressions.append(
                (
                    fraction
                    * pl.col(treatment_coordinate_column("aewr_log_level", year))
                ).alias(treatment_coordinate_column(f"exposure_log_{tag}", year))
            )

    current = pl.when(pl.lit(False)).then(None)
    lag1 = pl.when(pl.lit(False)).then(None)
    for year in analysis_years:
        current = current.when(pl.col("year") == year).then(
            pl.col(treatment_coordinate_column("aewr_log_level", year))
        )
        lag1_column = (
            pl.lit(0.0)
            if year == FULL_HISTORY_YEARS[0]
            else pl.col(treatment_coordinate_column("aewr_log_level", year - 1))
        )
        lag1 = lag1.when(pl.col("year") == year).then(lag1_column)
    expressions.extend(
        (
            current.otherwise(None).alias(ONE_LAG_BENCHMARK_COLUMNS[0]),
            lag1.otherwise(None).alias(ONE_LAG_BENCHMARK_COLUMNS[1]),
        )
    )
    return frame.with_columns(expressions)


def _assert_output_contract(frame: pl.DataFrame) -> None:
    duplicates = (
        frame.group_by(["county_fips", "year"]).len().filter(pl.col("len") != 1)
    )
    if not duplicates.is_empty():
        raise PanelBuildError("constructed panel is not unique by county-year")
    for column in (*GEOGRAPHY_COLUMNS, *CLUSTER_COLUMNS.values()):
        if frame.schema[column] != pl.String:
            raise PanelBuildError(f"constructed geography {column} is not a string")


def build_mcw_panel(
    source_path: str | Path,
    *,
    history_years: Sequence[int] = FULL_HISTORY_YEARS,
    analysis_years: Sequence[int] = ANALYSIS_YEARS,
) -> pl.DataFrame:
    """Scan a source Parquet and return the v4 analysis panel as Polars.

    Baseline and treatment history rows are used for construction; the returned
    rows are restricted only to ``analysis_years``.  No eligibility or
    outcome-based sample restriction is imposed here.
    """

    history, analysis = _validate_year_arguments(history_years, analysis_years)
    frame = _normalize_and_validate_keys(_scan_parquet(source_path))
    _assert_finite_nonnegative_outcomes(frame)
    _validate_baseline_quantiles(frame)
    region = _region_aewr_paths(frame, history)
    region_paths = _wide_region_paths(region, history)
    county_paths = _wide_county_paths(frame, history)

    derived_names = {
        *OUTCOME_COLUMNS.values(),
        *CLUSTER_COLUMNS.values(),
        *FRACTION_AFFECTED_COLUMNS.values(),
        *ONE_LAG_BENCHMARK_COLUMNS,
        BASELINE_FARM_EMPLOYMENT_COLUMN,
        BASELINE_BITE_COLUMN,
        "mc_baseline_farm_employment_year_count",
        *BASELINE_MEAN_COLUMNS.values(),
        *BASELINE_IMPUTED_COLUMNS.values(),
        *BASELINE_MISSING_INDICATOR_COLUMNS.values(),
        *STANDARDIZED_BASELINE_COLUMNS.values(),
        "mc_baseline_imputation_method",
        *(f"mc_baseline_year_count_{tag}" for tag in BASELINE_WINDOWS),
        *(
            f"mc_frozen_{column}_{tag}"
            for tag in BASELINE_WINDOWS
            for column in QUANTILE_COLUMNS
        ),
        "mc_bite_approximation_method",
        *(
            column
            for family in TREATMENT_PATH_COLUMNS.values()
            for column in family.values()
        ),
    }
    collisions = sorted(derived_names.intersection(frame.columns))
    if collisions:
        raise PanelBuildError(
            f"source already contains reserved output columns: {collisions}"
        )

    output = (
        frame.filter(pl.col("year").is_in(analysis))
        .join(region_paths, on="aewr_region_id", how="left", validate="m:1")
        .join(county_paths, on="county_fips", how="left", validate="m:1")
    )
    output = _add_outcomes_and_clusters(output)
    output = _add_exposures_and_benchmark(output, history, analysis)
    output = output.sort(["county_fips", "year"])
    _assert_output_contract(output)
    return output


def write_mcw_panel(
    source_path: str | Path,
    output_path: str | Path,
    *,
    history_years: Sequence[int] = FULL_HISTORY_YEARS,
    analysis_years: Sequence[int] = ANALYSIS_YEARS,
) -> pl.DataFrame:
    """Build and write a Parquet panel, returning the exact Polars frame written."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = build_mcw_panel(
        source_path,
        history_years=history_years,
        analysis_years=analysis_years,
    )
    frame.write_parquet(output)
    return frame
