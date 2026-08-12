"""Construct version-4 causal and correlated-effects design matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.linalg
import scipy.stats

from .design import (
    ANALYSIS_YEARS,
    BASELINE_VARIABLES,
    FULL_HISTORY_REFERENCE_OUTCOME_YEAR,
    FULL_HISTORY_REFERENCE_PATH_YEARS,
    MODERATOR_SETS,
    PRIMITIVE_OUTCOMES,
    TREATMENT_DEFINITIONS,
    TREATMENT_HISTORY_YEARS,
    Specification,
)
from .fwl import NestedFixedEffectProjector, NoFixedEffectProjector, OLSProjector
from .resources import guard_dense_allocation

BASELINE_COLUMN_OVERRIDES = {
    "bite": "mc_baseline_bite_z",
    "h2a_applications": "mc_baseline_h2a_applications_z",
    "h2a_certified_positions": "mc_baseline_h2a_certified_positions_z",
    "log_population": "mc_baseline_log_population_z",
    "farm_employment_share": "mc_baseline_farm_employment_share_z",
    "employment_population_ratio": "mc_baseline_employment_population_ratio_z",
    "crop_income_share": "mc_baseline_crop_income_share_z",
    "hired_labor_cost_share": "mc_baseline_hired_labor_cost_share_z",
    "low_wage": "mc_baseline_low_wage_z",
    "animal_income_share": "mc_baseline_animal_income_share_z",
    "production_expense_share": "mc_baseline_production_expense_share_z",
    "median_wage": "mc_baseline_median_wage_z",
    "cropland": "mc_baseline_cropland_z",
}


@dataclass(frozen=True, slots=True)
class ModelMatrices:
    causal: np.ndarray
    nuisance: np.ndarray
    outcomes: np.ndarray
    causal_names: tuple[str, ...]
    nuisance_names: tuple[str, ...]
    outcome_names: tuple[str, ...]
    projector: OLSProjector
    cluster: np.ndarray
    row_count: int
    frame: pl.DataFrame
    causal_metadata: tuple[dict[str, object], ...]


def _required_columns(specification: Specification) -> set[str]:
    definition = TREATMENT_DEFINITIONS[specification.treatment]
    prefix = str(definition["column_prefix"])
    columns = {
        "county_fips",
        "year",
        "state_fips",
        "aewr_region_id",
        "cz_id",
        "mc_market_id",
        "mc_baseline_farm_employment",
        *PRIMITIVE_OUTCOMES.values(),
        *MODERATOR_SETS[specification.moderator_set],
        *(f"{prefix}{year}" for year in TREATMENT_HISTORY_YEARS),
    }
    columns.update(BASELINE_COLUMN_OVERRIDES.values())
    columns.update(
        f"{column.removesuffix('_z')}_missing"
        for column in BASELINE_COLUMN_OVERRIDES.values()
    )
    return columns


def _assert_frame_contract(frame: pl.DataFrame, specification: Specification) -> None:
    missing = sorted(_required_columns(specification).difference(frame.columns))
    if missing:
        raise ValueError(f"Analysis panel lacks required columns: {missing}")
    if frame.height == 0:
        raise ValueError("Analysis panel is empty.")
    duplicate_count = (
        frame.group_by(["county_fips", "year"]).len().filter(pl.col("len") != 1).height
    )
    if duplicate_count:
        raise ValueError("Analysis panel is not unique by county_fips and year.")
    for column in ("county_fips", "state_fips", "aewr_region_id", "cz_id"):
        if frame.schema[column] != pl.String:
            raise TypeError(f"Geographic identifier {column} must be a string.")


def _codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    levels, codes = np.unique(values.astype(str), return_inverse=True)
    return codes.astype(np.int64, copy=False), levels


def _unique_county_values(
    frame: pl.DataFrame, column: str, unit_codes: np.ndarray, n_units: int
) -> np.ndarray:
    values = frame[column].cast(pl.Float64).to_numpy()
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Baseline column {column} contains non-finite values.")
    minimum = np.full(n_units, np.inf)
    maximum = np.full(n_units, -np.inf)
    np.minimum.at(minimum, unit_codes, values)
    np.maximum.at(maximum, unit_codes, values)
    if np.any(np.abs(maximum - minimum) > 1e-10):
        raise ValueError(f"Baseline column {column} varies within county.")
    return minimum


def _unit_geography(
    frame: pl.DataFrame, column: str, unit_codes: np.ndarray, n_units: int
) -> np.ndarray:
    values = frame[column].cast(pl.String).to_numpy()
    first = np.empty(n_units, dtype=object)
    first[:] = None
    for row, unit in enumerate(unit_codes):
        value = str(values[row])
        if first[unit] is None:
            first[unit] = value
        elif first[unit] != value:
            raise ValueError(f"{column} varies within county.")
    return first.astype(str)


def _group_mean(values: np.ndarray, codes: np.ndarray, count: int) -> np.ndarray:
    sums = np.bincount(codes, weights=values, minlength=count)
    sizes = np.bincount(codes, minlength=count)
    if np.any(sizes == 0):
        raise ValueError("Empty group in hierarchical baseline construction.")
    return sums / sizes


def _standardize(values: np.ndarray, label: str) -> np.ndarray | None:
    centered = values - np.mean(values)
    scale = np.sqrt(np.mean(np.square(centered)))
    if not np.isfinite(scale) or scale <= 1e-12:
        return None
    standardized = centered / scale
    if not np.all(np.isfinite(standardized)):
        raise ValueError(f"Non-finite standardized component: {label}")
    return standardized


def hierarchical_baseline_components(
    frame: pl.DataFrame,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Construct standardized telescoping components from county baselines."""

    unit_codes, unit_levels = _codes(frame["county_fips"].to_numpy())
    n_units = unit_levels.size
    market = _unit_geography(frame, "mc_market_id", unit_codes, n_units)
    state = _unit_geography(frame, "state_fips", unit_codes, n_units)
    region = _unit_geography(frame, "aewr_region_id", unit_codes, n_units)
    market_codes, market_levels = _codes(market)
    state_codes, state_levels = _codes(state)
    region_codes, region_levels = _codes(region)

    components: list[np.ndarray] = []
    names: list[str] = []
    for key in BASELINE_VARIABLES:
        column = BASELINE_COLUMN_OVERRIDES[key]
        county_value = _unique_county_values(frame, column, unit_codes, n_units)
        market_mean = _group_mean(county_value, market_codes, market_levels.size)[
            market_codes
        ]
        state_mean = _group_mean(county_value, state_codes, state_levels.size)[
            state_codes
        ]
        region_mean = _group_mean(county_value, region_codes, region_levels.size)[
            region_codes
        ]
        raw_components = {
            "county": county_value - market_mean,
            "market": market_mean - state_mean,
            "state": state_mean - region_mean,
            "region": region_mean - np.mean(county_value),
        }
        reconstruction = sum(raw_components.values())
        if not np.allclose(
            reconstruction,
            county_value - np.mean(county_value),
            rtol=1e-10,
            atol=1e-10,
        ):
            raise AssertionError(f"Hierarchical decomposition failed for {key}.")
        for level, values in raw_components.items():
            name = f"cre_{key}_{level}"
            standardized = _standardize(values, name)
            if standardized is None:
                continue
            components.append(standardized[unit_codes])
            names.append(name)
    return np.column_stack(components), tuple(names)


def categorical_year_contrasts(year: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return orthonormal categorical contrasts, never a polynomial trend."""

    year = np.asarray(year).astype(int)
    levels = np.array(sorted(np.unique(year)), dtype=int)
    expected = np.array(ANALYSIS_YEARS, dtype=int)
    if not np.array_equal(levels, expected):
        raise ValueError(
            f"Analysis years must be exactly {tuple(expected)}; got {tuple(levels)}."
        )
    codes = np.searchsorted(levels, year)
    # scipy's Helmert matrix spans the categorical zero-sum space.  It is only
    # a fixed-effect contrast and does not smooth or compress treatment lags.
    contrast_by_level = scipy.linalg.helmert(levels.size, full=False).T
    matrix = contrast_by_level[codes]
    names = tuple(
        f"calendar_helmert_{index + 1:02d}" for index in range(matrix.shape[1])
    )
    return matrix, names


def categorical_region_contrasts(
    region: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return a full-rank explicit AEWR-region contrast block."""

    region = np.asarray(region).astype(str)
    levels = np.array(sorted(np.unique(region)), dtype=str)
    if levels.size < 2:
        raise ValueError("Pooled WMC projection requires at least two AEWR regions.")
    codes = np.searchsorted(levels, region)
    matrix = scipy.linalg.helmert(levels.size, full=False).T[codes]
    names = tuple(f"region_helmert_{index + 1:02d}" for index in range(matrix.shape[1]))
    return matrix, names


def build_nuisance_matrix(
    frame: pl.DataFrame,
    specification: Specification,
) -> tuple[np.ndarray, tuple[str, ...]]:
    component_upper_bound = len(BASELINE_VARIABLES) * 4
    calendar_width = len(ANALYSIS_YEARS) - 1
    missing_upper_bound = len(BASELINE_VARIABLES)
    if specification.fixed_effects == "pooled_wmc":
        region_width = frame["aewr_region_id"].n_unique() - 1
        nuisance_upper_bound = (
            1
            + calendar_width
            + region_width
            + component_upper_bound
            + component_upper_bound * calendar_width
            + 3 * len(BASELINE_VARIABLES) * region_width
            + missing_upper_bound * (1 + calendar_width + region_width)
        )
    else:
        nuisance_upper_bound = (
            component_upper_bound
            + component_upper_bound * calendar_width
            + missing_upper_bound * (1 + calendar_width)
        )
    guard_dense_allocation(
        frame.height, nuisance_upper_bound, label="WMC nuisance dictionary upper bound"
    )
    components, component_names = hierarchical_baseline_components(frame)
    contrasts, contrast_names = categorical_year_contrasts(frame["year"].to_numpy())
    pooled = specification.fixed_effects == "pooled_wmc"
    if pooled:
        region_contrasts, region_contrast_names = categorical_region_contrasts(
            frame["aewr_region_id"].to_numpy()
        )
        blocks: list[np.ndarray] = [
            np.ones((frame.height, 1)),
            contrasts,
            region_contrasts,
            components,
        ]
        names: list[str] = [
            "pooled_intercept",
            *contrast_names,
            *region_contrast_names,
            *component_names,
        ]
    else:
        region_contrasts = np.empty((frame.height, 0))
        region_contrast_names = ()
        blocks = [components]
        names = list(component_names)
    for component_index, component_name in enumerate(component_names):
        block = components[:, component_index, None] * contrasts
        blocks.append(block)
        names.extend(
            f"{component_name}__x__{contrast_name}" for contrast_name in contrast_names
        )
        if pooled and not component_name.endswith("_region"):
            region_block = components[:, component_index, None] * region_contrasts
            blocks.append(region_block)
            names.extend(
                f"{component_name}__x__{contrast_name}"
                for contrast_name in region_contrast_names
            )
    for column in BASELINE_COLUMN_OVERRIDES.values():
        missing_column = f"{column.removesuffix('_z')}_missing"
        values = frame[missing_column].cast(pl.Float64).to_numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Missingness indicator {missing_column} is non-finite.")
        if np.std(values, ddof=0) > 0:
            blocks.append(values[:, None])
            names.append(missing_column)
            blocks.append(values[:, None] * contrasts)
            names.extend(
                f"{missing_column}__x__{contrast_name}"
                for contrast_name in contrast_names
            )
            if pooled:
                blocks.append(values[:, None] * region_contrasts)
                names.extend(
                    f"{missing_column}__x__{contrast_name}"
                    for contrast_name in region_contrast_names
                )
    return np.column_stack(blocks), tuple(names)


def _within_region_transform(
    values: np.ndarray,
    region_codes: np.ndarray,
    transform: str,
) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    for region in np.unique(region_codes):
        index = np.flatnonzero(region_codes == region)
        group = values[index]
        if transform == "continuous_within_region_z":
            scale = np.std(group, ddof=0)
            if scale <= 1e-12:
                raise ValueError("Treatment has no within-region variation.")
            result[index] = (group - np.mean(group)) / scale
        elif transform == "binary_median":
            result[index] = (group > np.median(group)).astype(float)
        elif transform == "binary_upper_quartile":
            result[index] = (group > np.quantile(group, 0.75)).astype(float)
        elif transform == "within_region_rank":
            result[index] = (scipy.stats.rankdata(group, method="average") - 0.5) / len(
                group
            )
        else:
            raise ValueError(f"Unknown lower-geography transform: {transform}")
    return result


def _treatment_paths(
    frame: pl.DataFrame, specification: Specification
) -> dict[int, np.ndarray]:
    prefix = str(TREATMENT_DEFINITIONS[specification.treatment]["column_prefix"])
    transform = specification.treatment_transform
    region_codes, _ = _codes(frame["aewr_region_id"].to_numpy())
    paths = {}
    for history_year in TREATMENT_HISTORY_YEARS:
        values = frame[f"{prefix}{history_year}"].cast(pl.Float64).to_numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Treatment coordinate {prefix}{history_year} is non-finite."
            )
        if transform != "continuous_raw":
            values = _within_region_transform(values, region_codes, transform)
        paths[history_year] = values
    return paths


def causal_moderator_values(frame: pl.DataFrame, name: str) -> np.ndarray:
    """Return the declared moderator as a within-AEWR-region deviation.

    The late WMC specification centers causal moderators within the treatment
    region.  This transformation is deliberately separate from the nuisance
    hierarchy: region and calendar interactions continue to use the explicit
    component dictionary constructed above.
    """

    if name not in frame.columns:
        raise ValueError(f"Moderator column is missing: {name}")
    values = frame[name].cast(pl.Float64).to_numpy()
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Moderator {name} is non-finite.")
    region_codes, region_levels = _codes(frame["aewr_region_id"].to_numpy())
    means = _group_mean(values, region_codes, region_levels.size)
    centered = values - means[region_codes]
    if not np.all(np.isfinite(centered)):
        raise ValueError(f"Centered moderator {name} is non-finite.")
    return centered


def build_causal_matrix(
    frame: pl.DataFrame, specification: Specification
) -> tuple[np.ndarray, tuple[str, ...], tuple[dict[str, object], ...]]:
    """Build separate linear history coordinates and allowed interactions."""

    if specification.history == "full":
        cells = sum(
            outcome_year - TREATMENT_HISTORY_YEARS[0] + 1
            for outcome_year in ANALYSIS_YEARS
        )
        if specification.fixed_effects != "pooled_wmc":
            cells -= len(FULL_HISTORY_REFERENCE_PATH_YEARS)
    else:
        cells = sum(
            min(2, outcome_year - TREATMENT_HISTORY_YEARS[0] + 1)
            for outcome_year in ANALYSIS_YEARS
        )
    guard_dense_allocation(
        frame.height,
        cells * (1 + len(MODERATOR_SETS[specification.moderator_set])),
        label="WMC causal dictionary",
    )
    year = frame["year"].cast(pl.Int32).to_numpy()
    paths = _treatment_paths(frame, specification)
    moderator_names = MODERATOR_SETS[specification.moderator_set]
    moderators = {
        name: causal_moderator_values(frame, name) for name in moderator_names
    }

    columns: list[np.ndarray] = []
    names: list[str] = []
    metadata: list[dict[str, object]] = []
    for outcome_year in ANALYSIS_YEARS:
        if specification.history == "full":
            history_years = range(TREATMENT_HISTORY_YEARS[0], outcome_year + 1)
        else:
            history_years = range(
                max(TREATMENT_HISTORY_YEARS[0], outcome_year - 1), outcome_year + 1
            )
        outcome_cell = (year == outcome_year).astype(np.float64)
        for history_year in history_years:
            # In the full model, every history path available at the first
            # outcome year is present in every outcome-year cell. Each such
            # block sums to a county-invariant path absorbed by county FEs.
            # Omit the first outcome cell and name the remaining coefficients
            # as differences from that reference. This retains the complete
            # identified history space without fitting projection noise.
            if (
                specification.fixed_effects != "pooled_wmc"
                and specification.history == "full"
                and history_year in FULL_HISTORY_REFERENCE_PATH_YEARS
                and outcome_year == FULL_HISTORY_REFERENCE_OUTCOME_YEAR
            ):
                continue
            base = outcome_cell * paths[history_year]
            if (
                specification.fixed_effects != "pooled_wmc"
                and specification.history == "full"
                and history_year in FULL_HISTORY_REFERENCE_PATH_YEARS
            ):
                main_name = (
                    f"effect_difference_y{outcome_year}"
                    f"_vs_y{FULL_HISTORY_REFERENCE_OUTCOME_YEAR}_h{history_year}"
                )
                identification = "difference_from_first_outcome_year"
                reference_year: int | None = FULL_HISTORY_REFERENCE_OUTCOME_YEAR
            else:
                main_name = f"effect_y{outcome_year}_h{history_year}"
                identification = (
                    "level_pooled_wmc"
                    if specification.fixed_effects == "pooled_wmc"
                    else "level_after_county_fe"
                )
                reference_year = None
            columns.append(base)
            names.append(main_name)
            metadata.append(
                {
                    "name": main_name,
                    "outcome_year": outcome_year,
                    "history_year": history_year,
                    "lag": outcome_year - history_year,
                    "moderator": None,
                    "moderator_transform": None,
                    "identification": identification,
                    "reference_year": reference_year,
                }
            )
            for moderator_name, moderator in moderators.items():
                interaction_name = f"{main_name}__x__{moderator_name}"
                columns.append(base * moderator)
                names.append(interaction_name)
                metadata.append(
                    {
                        "name": interaction_name,
                        "outcome_year": outcome_year,
                        "history_year": history_year,
                        "lag": outcome_year - history_year,
                        "moderator": moderator_name,
                        "moderator_transform": "within_aewr_region_deviation",
                        "identification": identification,
                        "reference_year": reference_year,
                    }
                )
    return np.column_stack(columns), tuple(names), tuple(metadata)


def _fixed_effect_parent(
    frame: pl.DataFrame, specification: Specification
) -> np.ndarray:
    if specification.fixed_effects == "county_year":
        return np.repeat("national", frame.height)
    if specification.fixed_effects == "county_state_year":
        return frame["state_fips"].to_numpy()
    if specification.fixed_effects == "county_region_year":
        return frame["aewr_region_id"].to_numpy()
    raise ValueError(f"Unknown fixed effects: {specification.fixed_effects}")


def build_model_matrices(
    frame: pl.DataFrame,
    specification: Specification,
    cluster_column: str,
) -> ModelMatrices:
    """Compile one specification after enforcing the common-sample contract."""

    specification.validate()
    _assert_frame_contract(frame, specification)
    frame = frame.sort(["county_fips", "year"])
    causal, causal_names, causal_metadata = build_causal_matrix(frame, specification)
    nuisance, nuisance_names = build_nuisance_matrix(frame, specification)
    outcome_names = tuple(PRIMITIVE_OUTCOMES)
    outcomes = frame.select(PRIMITIVE_OUTCOMES.values()).to_numpy().astype(np.float64)
    if specification.fixed_effects == "pooled_wmc":
        projector: OLSProjector = NoFixedEffectProjector.from_row_count(frame.height)
    else:
        projector = NestedFixedEffectProjector.from_arrays(
            frame["county_fips"].to_numpy(),
            frame["year"].to_numpy(),
            _fixed_effect_parent(frame, specification),
        )
    if cluster_column not in frame.columns:
        raise ValueError(f"Cluster column is missing: {cluster_column}")
    cluster = frame[cluster_column].cast(pl.String).to_numpy()
    if np.any(pl.Series(cluster).is_null().to_numpy()):
        raise ValueError(f"Cluster column {cluster_column} contains nulls.")
    return ModelMatrices(
        causal=causal,
        nuisance=nuisance,
        outcomes=outcomes,
        causal_names=causal_names,
        nuisance_names=nuisance_names,
        outcome_names=outcome_names,
        projector=projector,
        cluster=cluster,
        row_count=frame.height,
        frame=frame,
        causal_metadata=causal_metadata,
    )
