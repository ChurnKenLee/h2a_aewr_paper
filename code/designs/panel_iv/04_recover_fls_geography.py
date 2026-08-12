"""Recover one pseudo-FLS county-weight distribution per region and year.

The Census/QCEW/QWI/BEA frame supplies the prior.  The secondary publication
specification softly targets the annual FLS combined wage with county-mapped
OEWS hourly wages.  The preferred specification adds three independent QCEW
seasonal contrasts and four undivided FLS field/livestock composition
residuals.  Worker levels and duration measures are retained as diagnostics;
they never constrain regional employment levels.
"""

from __future__ import annotations

import argparse
import math
import os
import zlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from h2a.geography import assert_geo_columns
from h2a.paths import INTERMEDIATE, PROCESSED

SUPPORTED_YEARS = tuple(range(2010, 2022))
REFERENCE_QUARTERS = ("january", "april", "july", "october")
QUARTER_NUMBER = {quarter: index for index, quarter in enumerate(REFERENCE_QUARTERS, 1)}
BASELINE_WEIGHT_SPEC = "census_hired_workers_qcew_annual_updated_v2"
ANNUAL_UPDATE_SPEC = "qcew_annual_qwi_bea_two_sided_state_raked_v2"
WEIGHT_SPEC = "fls_pseudo_county_entropy_v2"
WAGE_ONLY_SPECIFICATION = "fls_county_wage_only_soft_rho010_v2"
PRIMARY_SPECIFICATION = "fls_county_wage_seasonal_composition_soft_rho010_v2"
WAGE_ONLY_MOMENT_SPEC = "fls_field_livestock_wage_only"
PRIMARY_MOMENT_SPEC = "fls_oews_wage_plus_qcew_seasonal_and_composition"
PRIMARY_RHO = 0.10
KAPPA_MULTIPLIER = 10.0
DIAGNOSTIC_DRAW_COUNT = 8
SIMULATION_SEED = 20260812
MINIMUM_SCALE_RELATIVE = 1e-12
WEIGHT_SUM_TOLERANCE = 1e-10
OPTIMIZER_GRADIENT_TOLERANCE = 1e-10
OPTIMIZER_MAX_ITERATIONS = 100

FEATURE_PATH = INTERMEDIATE / "panel_iv_fls_county_features.parquet"
WEIGHT_SUMMARY_PATH = INTERMEDIATE / "panel_iv_fls_county_weight_summary.parquet"
CALIBRATION_DIAGNOSTIC_PATH = (
    INTERMEDIATE / "panel_iv_fls_county_calibration_diagnostics.parquet"
)
MOMENT_DIAGNOSTIC_PATH = (
    INTERMEDIATE / "panel_iv_fls_county_moment_diagnostics.parquet"
)
DRAW_ROOT = INTERMEDIATE / "panel_iv_fls_county_draws"

COUNTY_KEYS = ["aewr_region_id", "source_year", "county_fips"]
CELL_KEYS = ["aewr_region_id", "source_year"]


def require_columns(frame: pl.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def require_unique(frame: pl.DataFrame, keys: list[str], label: str) -> None:
    require_columns(frame, keys, label)
    duplicates = frame.group_by(keys).len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise ValueError(
            f"{label} has {duplicates.height} duplicate cells on {', '.join(keys)}"
        )


def finite_positive(value: Any) -> bool:
    return (
        value is not None
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def helmert_basis(cell_count: int) -> np.ndarray:
    """Return an orthonormal basis for contrasts among ``cell_count`` cells."""
    if cell_count < 2:
        raise ValueError("At least two cells are required for a contrast basis")
    basis = np.zeros((cell_count, cell_count - 1), dtype=float)
    for column in range(cell_count - 1):
        basis[: column + 1, column] = 1 / math.sqrt(
            (column + 1) * (column + 2)
        )
        basis[column + 1, column] = -(column + 1) / math.sqrt(
            (column + 1) * (column + 2)
        )
    return basis


def composition_residual(
    field_wage: float,
    livestock_wage: float,
    combined_wage: float,
    crop_employment: np.ndarray | Sequence[float],
    animal_employment: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return the undivided field/livestock composition residual.

    No implied employment share is formed, so equal or rounded FLS wage rates
    are valid inputs and never require division or clipping.
    """
    crop = np.asarray(crop_employment, dtype=float)
    animal = np.asarray(animal_employment, dtype=float)
    if crop.shape != animal.shape:
        raise ValueError("Crop and animal employment arrays must have equal shape")
    return (field_wage - combined_wage) * crop + (
        livestock_wage - combined_wage
    ) * animal


def standardize_moment(
    values: np.ndarray | Sequence[float],
    target: float,
    prior: np.ndarray | Sequence[float],
) -> dict[str, Any]:
    """Standardize one moment under the county prior and flag zero variation."""
    raw = np.asarray(values, dtype=float)
    weights = np.asarray(prior, dtype=float)
    if raw.ndim != 1 or weights.shape != raw.shape:
        raise ValueError("Moment values and prior must be equal-length vectors")
    if (
        not np.all(np.isfinite(raw))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or weights.sum() <= 0
        or not math.isfinite(float(target))
    ):
        raise ValueError("Moment values, target, and prior must be finite")
    weights = weights / weights.sum()
    center = float(weights @ raw)
    scale = float(np.sqrt(weights @ (raw - center) ** 2))
    reference = max(1.0, float(np.max(np.abs(raw))), abs(center), abs(float(target)))
    active = scale > MINIMUM_SCALE_RELATIVE * reference
    return {
        "prior_center": center,
        "prior_scale": scale,
        "active": active,
        "status": "active" if active else "inactive_zero_prior_variation",
        "standardized_values": (raw - center) / scale if active else None,
        "standardized_target": (float(target) - center) / scale if active else None,
    }


def specification_grid() -> tuple[dict[str, Any], ...]:
    return (
        {
            "specification": WAGE_ONLY_SPECIFICATION,
            "moment_spec": WAGE_ONLY_MOMENT_SPEC,
            "included_families": {"annual_wage"},
            "is_primary": False,
        },
        {
            "specification": PRIMARY_SPECIFICATION,
            "moment_spec": PRIMARY_MOMENT_SPEC,
            "included_families": {"annual_wage", "seasonal", "composition"},
            "is_primary": True,
        },
    )


def deterministic_seed(aewr_region_id: str, source_year: int) -> int:
    try:
        region_component = int(aewr_region_id)
    except ValueError:
        region_component = zlib.crc32(aewr_region_id.encode()) % 1000
    return (
        SIMULATION_SEED + region_component * 1_000_000 + int(source_year) * 100
    ) % (2**32 - 1)


def dirichlet_prior_draws(
    prior_weight: np.ndarray,
    *,
    draw_count: int,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    prior = np.asarray(prior_weight, dtype=float)
    if prior.ndim != 1 or not np.all(np.isfinite(prior)) or np.any(prior <= 0):
        raise ValueError("Dirichlet prior weights must be finite and positive")
    prior /= prior.sum()
    effective_county_count = float(1 / np.sum(prior**2))
    kappa = float(KAPPA_MULTIPLIER * effective_county_count)
    rng = np.random.default_rng(seed)
    draws = rng.dirichlet(kappa * prior, size=draw_count)
    draws = np.maximum(draws, np.nextafter(0.0, 1.0))
    draws /= draws.sum(axis=1, keepdims=True)
    return draws, kappa, effective_county_count


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
    rho: float = PRIMARY_RHO,
) -> dict[str, np.ndarray]:
    """Solve soft entropy calibration for one or more county-prior vectors."""
    priors = np.asarray(prior_weights, dtype=float)
    if priors.ndim == 1:
        priors = priors[None, :]
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    if priors.ndim != 2 or design.ndim != 2 or design.shape[0] != priors.shape[1]:
        raise ValueError("Calibration design and prior dimensions disagree")
    if target.shape != (design.shape[1],):
        raise ValueError("Target length does not match the calibration design")
    if (
        not np.all(np.isfinite(priors))
        or np.any(priors < 0)
        or np.any(priors.sum(axis=1) <= 0)
    ):
        raise ValueError("Every calibration prior must be finite and nonnegative")
    priors /= priors.sum(axis=1, keepdims=True)
    batch_size = priors.shape[0]
    moment_count = design.shape[1]
    if moment_count == 0:
        zeros = np.zeros(batch_size)
        return {
            "weights": priors,
            "success": np.ones(batch_size, dtype=bool),
            "status": np.full(batch_size, "calibrated_prior_no_active_moments", object),
            "iterations": np.zeros(batch_size, dtype=np.int32),
            "input_imbalance_norm": zeros,
            "calibrated_imbalance_norm": zeros,
            "maximum_absolute_imbalance": zeros,
            "kl_divergence": zeros,
            "effective_county_count": 1 / np.sum(priors**2, axis=1),
            "maximum_county_weight": np.max(priors, axis=1),
        }
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(target)):
        raise ValueError("Active standardized moments must be finite")

    log_prior = np.log(np.maximum(priors, np.nextafter(0.0, 1.0)))
    multipliers = np.zeros((batch_size, moment_count))
    converged = np.zeros(batch_size, dtype=bool)
    failed = np.zeros(batch_size, dtype=bool)
    iterations = np.zeros(batch_size, dtype=np.int32)
    identity = np.eye(moment_count)

    for iteration in range(1, OPTIMIZER_MAX_ITERATIONS + 1):
        weights, moments, objective, gradient = _dual_state(
            log_prior, design, target, rho, multipliers
        )
        converged |= np.max(np.abs(gradient), axis=1) <= OPTIMIZER_GRADIENT_TOLERANCE
        active = ~(converged | failed)
        if not np.any(active):
            break
        for batch_index in np.flatnonzero(active):
            second = np.einsum(
                "n,np,nq->pq",
                weights[batch_index],
                design,
                design,
                optimize=True,
            )
            hessian = (
                second
                - np.outer(moments[batch_index], moments[batch_index])
                + identity / rho
            )
            try:
                direction = np.linalg.solve(hessian, gradient[batch_index])
            except np.linalg.LinAlgError:
                failed[batch_index] = True
                continue
            directional_derivative = float(gradient[batch_index] @ direction)
            step_size = 1.0
            accepted = False
            while step_size >= 2**-20:
                proposal = (multipliers[batch_index] - step_size * direction)[None, :]
                _, _, proposal_objective, _ = _dual_state(
                    log_prior[batch_index : batch_index + 1],
                    design,
                    target,
                    rho,
                    proposal,
                )
                tolerance = 1e-12 * (1 + abs(float(objective[batch_index])))
                if math.isfinite(float(proposal_objective[0])) and (
                    proposal_objective[0]
                    <= objective[batch_index]
                    - 1e-4 * step_size * directional_derivative
                    or proposal_objective[0] <= objective[batch_index] + tolerance
                ):
                    multipliers[batch_index] = proposal[0]
                    iterations[batch_index] = iteration
                    accepted = True
                    break
                step_size *= 0.5
            if not accepted:
                failed[batch_index] = True

    final_weights, final_moments, _, final_gradient = _dual_state(
        log_prior, design, target, rho, multipliers
    )
    final_weights /= final_weights.sum(axis=1, keepdims=True)
    converged |= (
        np.max(np.abs(final_gradient), axis=1) <= OPTIMIZER_GRADIENT_TOLERANCE
    )
    input_imbalance = priors @ design - target[None, :]
    calibrated_imbalance = final_moments - target[None, :]
    input_norm = np.linalg.norm(input_imbalance, axis=1)
    calibrated_norm = np.linalg.norm(calibrated_imbalance, axis=1)
    weight_valid = (
        np.abs(final_weights.sum(axis=1) - 1) <= WEIGHT_SUM_TOLERANCE
    ) & np.all(np.isfinite(final_weights) & (final_weights >= 0), axis=1)
    success = converged & ~failed & weight_valid & (
        calibrated_norm <= input_norm + 1e-9
    )
    status = np.full(batch_size, "optimizer_failed", object)
    status[success] = "calibrated_soft"
    status[failed] = "line_search_or_hessian_failed"
    status[~failed & ~converged] = "maximum_iterations_reached"
    status[converged & ~weight_valid] = "invalid_weight_solution"
    kl = np.sum(
        final_weights
        * (
            np.log(np.maximum(final_weights, np.nextafter(0.0, 1.0)))
            - log_prior
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
        "maximum_absolute_imbalance": np.max(
            np.abs(calibrated_imbalance), axis=1
        ),
        "kl_divergence": kl,
        "effective_county_count": 1 / np.sum(final_weights**2, axis=1),
        "maximum_county_weight": np.max(final_weights, axis=1),
    }


def _annual_fls_targets(fls_region: pl.DataFrame, years: Sequence[int]) -> pl.DataFrame:
    require_columns(
        fls_region,
        [
            "estimate_year",
            "aewr_region_id",
            "revised_year",
            "preliminary_year",
            "field_livestock_revised",
            "field_livestock_preliminary",
            "source_zip",
            "source_csv",
        ],
        "FLS annual regional wages",
    )
    revised = fls_region.select(
        pl.col("aewr_region_id").cast(pl.String),
        pl.col("revised_year").cast(pl.Int32).alias("source_year"),
        pl.col("field_livestock_revised")
        .cast(pl.Float64, strict=False)
        .alias("fls_annual_field_livestock_hourly_wage"),
        pl.col("estimate_year").cast(pl.Int32).alias("fls_annual_release_year"),
        pl.lit("revised").alias("fls_annual_wage_vintage"),
        pl.col("source_zip").alias("fls_annual_source_zip"),
        pl.col("source_csv").alias("fls_annual_source_csv"),
        pl.lit(1).alias("_vintage_priority"),
    )
    preliminary = fls_region.select(
        pl.col("aewr_region_id").cast(pl.String),
        pl.col("preliminary_year").cast(pl.Int32).alias("source_year"),
        pl.col("field_livestock_preliminary")
        .cast(pl.Float64, strict=False)
        .alias("fls_annual_field_livestock_hourly_wage"),
        pl.col("estimate_year").cast(pl.Int32).alias("fls_annual_release_year"),
        pl.lit("preliminary").alias("fls_annual_wage_vintage"),
        pl.col("source_zip").alias("fls_annual_source_zip"),
        pl.col("source_csv").alias("fls_annual_source_csv"),
        pl.lit(0).alias("_vintage_priority"),
    )
    targets = (
        pl.concat([preliminary, revised], how="vertical")
        .filter(
            pl.col("source_year").is_in(years),
            pl.col("fls_annual_field_livestock_hourly_wage").is_finite(),
            pl.col("fls_annual_field_livestock_hourly_wage") > 0,
        )
        .sort(
            "aewr_region_id",
            "source_year",
            "_vintage_priority",
            "fls_annual_release_year",
        )
        .unique(subset=CELL_KEYS, keep="last", maintain_order=True)
        .drop("_vintage_priority")
    )
    require_unique(targets, CELL_KEYS, "annual FLS wage targets")
    return targets


def _paired_quarterly_targets(
    workers: pl.DataFrame,
    wages: pl.DataFrame,
    years: Sequence[int],
) -> pl.DataFrame:
    keys = ["aewr_region_id", "year", "quarter"]
    require_unique(workers, keys, "FLS quarterly workers")
    require_unique(wages, keys, "FLS quarterly wages")
    worker_columns = [
        *keys,
        "fls_hired_workers",
        "fls_hired_workers_150_days_or_more",
        "fls_hired_workers_149_days_or_less",
        "fls_gross_hours_worked",
        "source_zip",
        "release_year",
        "release_month",
        "release_day",
        "fls_pair_values_available",
        "fls_pair_value_status",
    ]
    wage_columns = [
        *keys,
        "fls_field_hourly_wage",
        "fls_livestock_hourly_wage",
        "fls_field_livestock_hourly_wage",
        "fls_all_hired_hourly_wage",
        "source_zip",
        "release_year",
        "release_month",
        "release_day",
        "wage_source_csv",
        "wage_table_title",
    ]
    require_columns(workers, worker_columns, "FLS quarterly workers")
    require_columns(wages, wage_columns, "FLS quarterly wages")
    paired = workers.select(worker_columns).rename(
        {
            "year": "source_year",
            "source_zip": "worker_source_zip",
            "release_year": "worker_release_year",
            "release_month": "worker_release_month",
            "release_day": "worker_release_day",
        }
    ).join(
        wages.select(wage_columns).rename(
            {
                "year": "source_year",
                "source_zip": "wage_source_zip",
                "release_year": "wage_release_year",
                "release_month": "wage_release_month",
                "release_day": "wage_release_day",
            }
        ),
        on=["aewr_region_id", "source_year", "quarter"],
        how="inner",
        validate="1:1",
    ).filter(pl.col("source_year").is_in(years))
    mismatched = paired.filter(
        (pl.col("worker_source_zip") != pl.col("wage_source_zip"))
        | (pl.col("worker_release_year") != pl.col("wage_release_year"))
        | (pl.col("worker_release_month") != pl.col("wage_release_month"))
        | (pl.col("worker_release_day") != pl.col("wage_release_day"))
    )
    if mismatched.height:
        raise ValueError("FLS quarterly worker and wage targets use different releases")
    expected = len(years) * 17 * len(REFERENCE_QUARTERS)
    if paired.height != expected:
        raise ValueError(
            f"Expected {expected} paired FLS region-week targets, found {paired.height}"
        )
    invalid_available = paired.filter(
        pl.col("fls_pair_values_available")
        & (
            ~pl.col("fls_hired_workers").is_finite()
            | (pl.col("fls_hired_workers") < 0)
            | ~pl.col("fls_field_hourly_wage").is_finite()
            | ~pl.col("fls_livestock_hourly_wage").is_finite()
            | ~pl.col("fls_field_livestock_hourly_wage").is_finite()
            | ~pl.col("fls_all_hired_hourly_wage").is_finite()
        )
    )
    if invalid_available.height:
        raise ValueError("Available paired FLS quarterly targets contain invalid values")
    unavailable = paired.filter(~pl.col("fls_pair_values_available"))
    documented_gap = unavailable.filter(
        (pl.col("source_year") == 2011)
        & (pl.col("quarter") == "april")
        & (pl.col("fls_pair_value_status") == "survey_not_conducted")
    )
    if unavailable.height != 17 or documented_gap.height != 17:
        raise ValueError("Unexpected unavailable paired FLS quarterly targets")
    if paired.filter(~pl.col("quarter").is_in(REFERENCE_QUARTERS)).height:
        raise ValueError("Paired FLS targets contain an invalid survey quarter")
    return paired.sort("aewr_region_id", "source_year", "quarter")


def _qcew_lookup(qcew: pl.DataFrame, years: Sequence[int]) -> dict[tuple[str, int, int, str], float]:
    keys = ["county_fips", "year", "qtr", "industry_code"]
    require_columns(
        qcew,
        [*keys, "qcew_employment_disclosed", "qcew_reference_month_emplvl"],
        "quarterly QCEW",
    )
    require_unique(qcew, keys, "quarterly QCEW")
    selected = qcew.filter(
        pl.col("year").is_in(years),
        pl.col("industry_code").is_in(["111", "112"]),
        pl.col("qcew_employment_disclosed").fill_null(False),
        pl.col("qcew_reference_month_emplvl").is_finite(),
        pl.col("qcew_reference_month_emplvl") >= 0,
    )
    return {
        (
            row["county_fips"],
            int(row["year"]),
            int(row["qtr"]),
            row["industry_code"],
        ): float(row["qcew_reference_month_emplvl"])
        for row in selected.iter_rows(named=True)
    }


def _impute_under_prior(
    values: np.ndarray,
    observed: np.ndarray,
    prior: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    if values.ndim == 1:
        values_2d = values[:, None]
    else:
        values_2d = values
    observed = np.asarray(observed, dtype=bool)
    observed_mass = float(prior[observed].sum())
    if observed_mass <= 0:
        raise ValueError("A county moment has no observed positive prior mass")
    mean = (prior[observed, None] * values_2d[observed]).sum(axis=0) / observed_mass
    filled = np.array(values_2d, copy=True)
    filled[~observed] = mean
    if values.ndim == 1:
        return filled[:, 0], observed_mass, mean
    return filled, observed_mass, mean


def build_county_features(
    frame: pl.DataFrame,
    county_panel: pl.DataFrame,
    qcew: pl.DataFrame,
    annual_targets: pl.DataFrame,
    quarterly_targets: pl.DataFrame,
    *,
    years: Sequence[int],
    regions: Sequence[str] | None,
) -> tuple[pl.DataFrame, dict[tuple[str, int], dict[str, Any]]]:
    require_columns(
        frame,
        [
            "county_fips",
            "aewr_region_id",
            "source_year",
            "weight_spec",
            "annual_update_spec",
            "weight_draw_id",
            "frame_employment_mass",
            "annual_update_source",
        ],
        "FLS county frame",
    )
    prior = frame.filter(
        pl.col("source_year").is_in(years),
        pl.col("weight_spec") == BASELINE_WEIGHT_SPEC,
        pl.col("annual_update_spec") == ANNUAL_UPDATE_SPEC,
        pl.col("weight_draw_id").is_null(),
    )
    if regions is not None:
        prior = prior.filter(pl.col("aewr_region_id").is_in(regions))
    prior = prior.select(
        "county_fips",
        "state_fips",
        "aewr_region_id",
        "source_year",
        "frame_employment_mass",
        "annual_update_source",
        "qcew_strict_complete",
        "qwi_annual_fallback_used",
        "bea_annual_fallback_used",
        "quality_flags",
    ).sort(*COUNTY_KEYS)
    require_unique(prior, COUNTY_KEYS, "FLS county prior")
    assert_geo_columns(prior, ["county_fips", "state_fips", "aewr_region_id"])

    panel_wages = county_panel.select(
        pl.col("county_fips").cast(pl.String),
        pl.col("year").cast(pl.Int32).alias("source_year"),
        pl.col("oews_big_six_mean_hourly_wage").cast(pl.Float64, strict=False),
        pl.col("oews_wage_observed").cast(pl.Boolean),
    )
    require_unique(panel_wages, ["county_fips", "source_year"], "county OEWS wages")
    prior = prior.join(
        panel_wages,
        on=["county_fips", "source_year"],
        how="left",
        validate="1:1",
    )
    qcew_values = _qcew_lookup(qcew, years)
    annual_lookup = {
        (row["aewr_region_id"], int(row["source_year"])): row
        for row in annual_targets.iter_rows(named=True)
    }
    quarterly_lookup = {
        (row["aewr_region_id"], int(row["source_year"]), row["quarter"]): row
        for row in quarterly_targets.iter_rows(named=True)
    }

    feature_rows: list[dict[str, Any]] = []
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    for cell in prior.partition_by(CELL_KEYS, maintain_order=True):
        first = cell.row(0, named=True)
        region = first["aewr_region_id"]
        year = int(first["source_year"])
        county_codes = cell.get_column("county_fips").to_list()
        mass = cell.get_column("frame_employment_mass").cast(pl.Float64).to_numpy()
        if not np.all(np.isfinite(mass)) or np.any(mass < 0) or mass.sum() <= 0:
            raise ValueError(f"Invalid county prior mass for region {region}, {year}")
        county_prior = mass / mass.sum()

        quarterly = np.full((cell.height, 4, 2), np.nan)
        for county_index, county in enumerate(county_codes):
            for quarter_index, quarter in enumerate(REFERENCE_QUARTERS):
                qtr = QUARTER_NUMBER[quarter]
                for industry_index, industry in enumerate(("111", "112")):
                    value = qcew_values.get((county, year, qtr, industry))
                    if value is not None:
                        quarterly[county_index, quarter_index, industry_index] = value

        target_rows = [
            quarterly_lookup[(region, year, quarter)]
            for quarter in REFERENCE_QUARTERS
        ]
        worker_counts = np.asarray(
            [
                float(row["fls_hired_workers"])
                if row["fls_hired_workers"] is not None
                else np.nan
                for row in target_rows
            ]
        )
        available_quarters = np.flatnonzero(
            np.isfinite(worker_counts) & (worker_counts >= 0)
        )
        if (
            available_quarters.size < 2
            or worker_counts[available_quarters].sum() <= 0
        ):
            raise ValueError(
                f"Insufficient FLS worker targets for region {region}, {year}"
            )

        seasonal_quarterly = quarterly[:, available_quarters, :]
        seasonal_observed = np.all(
            np.isfinite(seasonal_quarterly), axis=(1, 2)
        )
        annual_quarter_total = np.sum(seasonal_quarterly, axis=(1, 2))
        seasonal_observed &= annual_quarter_total > 0
        seasonal_raw = np.full((cell.height, available_quarters.size), np.nan)
        seasonal_raw[seasonal_observed] = (
            seasonal_quarterly[seasonal_observed].sum(axis=2)
            / annual_quarter_total[seasonal_observed, None]
        )
        seasonal_filled, seasonal_coverage, _ = _impute_under_prior(
            seasonal_raw, seasonal_observed, county_prior
        )
        seasonal_target = (
            worker_counts[available_quarters]
            / worker_counts[available_quarters].sum()
        )
        basis = helmert_basis(available_quarters.size)

        raw_moments: list[dict[str, Any]] = []
        for contrast in range(available_quarters.size - 1):
            raw_moments.append(
                {
                    "moment_id": f"seasonal_helmert_{contrast + 1}",
                    "moment_family": "seasonal",
                    "raw_values": seasonal_filled @ basis[:, contrast],
                    "raw_target": float(seasonal_target @ basis[:, contrast]),
                    "observed": seasonal_observed,
                    "observed_prior_mass": seasonal_coverage,
                    "feature_source": "qcew_111_112_quarterly_share",
                    "target_source": "fls_quarterly_worker_share",
                    "quarter": None,
                }
            )
        for contrast in range(available_quarters.size - 1, 3):
            raw_moments.append(
                {
                    "moment_id": f"seasonal_helmert_{contrast + 1}",
                    "moment_family": "seasonal",
                    "raw_values": np.zeros(cell.height),
                    "raw_target": 0.0,
                    "observed": np.zeros(cell.height, dtype=bool),
                    "observed_prior_mass": 0.0,
                    "feature_source": "qcew_111_112_quarterly_share",
                    "target_source": "fls_survey_not_conducted",
                    "quarter": None,
                    "forced_status": "inactive_fls_survey_not_conducted",
                }
            )

        for quarter_index, (quarter, target_row) in enumerate(
            zip(REFERENCE_QUARTERS, target_rows, strict=True)
        ):
            composition_observed = np.all(
                np.isfinite(quarterly[:, quarter_index, :]), axis=1
            )
            wage_values = np.asarray(
                [
                    target_row["fls_field_hourly_wage"],
                    target_row["fls_livestock_hourly_wage"],
                    target_row["fls_field_livestock_hourly_wage"],
                ],
                dtype=float,
            )
            wage_target_available = bool(np.all(np.isfinite(wage_values)))
            raw_composition = np.full(cell.height, np.nan)
            if wage_target_available and np.any(composition_observed):
                raw_composition[composition_observed] = composition_residual(
                    wage_values[0],
                    wage_values[1],
                    wage_values[2],
                    quarterly[composition_observed, quarter_index, 0],
                    quarterly[composition_observed, quarter_index, 1],
                )
            if wage_target_available:
                composition_filled, composition_coverage, _ = _impute_under_prior(
                    raw_composition, composition_observed, county_prior
                )
            else:
                composition_observed = np.zeros(cell.height, dtype=bool)
                composition_filled = np.zeros(cell.height)
                composition_coverage = 0.0
            raw_moments.append(
                {
                    "moment_id": f"composition_{quarter}",
                    "moment_family": "composition",
                    "raw_values": composition_filled,
                    "raw_target": 0.0,
                    "observed": composition_observed,
                    "observed_prior_mass": composition_coverage,
                    "feature_source": "qcew_111_112_undivided_residual",
                    "target_source": "fls_quarterly_field_livestock_wages",
                    "quarter": quarter,
                    **(
                        {}
                        if wage_target_available
                        else {
                            "forced_status": "inactive_fls_survey_not_conducted"
                        }
                    ),
                }
            )

        annual_target = annual_lookup.get((region, year))
        if annual_target is None:
            raise ValueError(f"Missing annual FLS wage target for {region}, {year}")
        oews = cell.get_column("oews_big_six_mean_hourly_wage").to_numpy()
        oews_observed = np.isfinite(oews) & (oews > 0)
        oews_filled, oews_coverage, _ = _impute_under_prior(
            oews, oews_observed, county_prior
        )
        raw_moments.append(
            {
                "moment_id": "annual_fls_oews_hourly_wage",
                "moment_family": "annual_wage",
                "raw_values": oews_filled,
                "raw_target": float(
                    annual_target["fls_annual_field_livestock_hourly_wage"]
                ),
                "observed": oews_observed,
                "observed_prior_mass": oews_coverage,
                "feature_source": "county_mapped_oews_big_six_hourly_wage",
                "target_source": "fls_annual_field_livestock_hourly_wage",
                "quarter": None,
            }
        )

        moment_records: list[dict[str, Any]] = []
        for moment in raw_moments:
            if "forced_status" in moment:
                standardized = {
                    "prior_center": float(county_prior @ moment["raw_values"]),
                    "prior_scale": 0.0,
                    "active": False,
                    "status": moment["forced_status"],
                    "standardized_values": None,
                    "standardized_target": None,
                }
            else:
                standardized = standardize_moment(
                    moment["raw_values"], moment["raw_target"], county_prior
                )
            moment.update(standardized)
            moment_records.append(moment)
            for index, county in enumerate(county_codes):
                feature_rows.append(
                    {
                        "aewr_region_id": region,
                        "source_year": year,
                        "county_fips": county,
                        "moment_id": moment["moment_id"],
                        "moment_family": moment["moment_family"],
                        "quarter": moment["quarter"],
                        "frame_prior_weight": float(county_prior[index]),
                        "raw_feature_value": float(moment["raw_values"][index]),
                        "feature_observed": bool(moment["observed"][index]),
                        "feature_imputed": not bool(moment["observed"][index]),
                        "observed_prior_mass": moment["observed_prior_mass"],
                        "raw_target": moment["raw_target"],
                        "prior_center": moment["prior_center"],
                        "prior_scale": moment["prior_scale"],
                        "standardized_feature_value": (
                            float(moment["standardized_values"][index])
                            if moment["active"]
                            else None
                        ),
                        "standardized_target": (
                            float(moment["standardized_target"])
                            if moment["active"]
                            else None
                        ),
                        "moment_active": moment["active"],
                        "moment_status": moment["status"],
                        "feature_source": moment["feature_source"],
                        "target_source": moment["target_source"],
                        "baseline_weight_spec": BASELINE_WEIGHT_SPEC,
                        "annual_update_spec": ANNUAL_UPDATE_SPEC,
                        "weight_spec": WEIGHT_SPEC,
                        "fls_worker_reference_week_total_diagnostic": float(
                            np.nansum(worker_counts)
                        ),
                        "fls_annual_wage_vintage": annual_target[
                            "fls_annual_wage_vintage"
                        ],
                        "fls_annual_release_year": annual_target[
                            "fls_annual_release_year"
                        ],
                    }
                )
        cells[(region, year)] = {
            "county_codes": county_codes,
            "prior": county_prior,
            "moments": moment_records,
            "annual_update_sources": cell.get_column(
                "annual_update_source"
            ).to_list(),
        }

    features = pl.DataFrame(feature_rows, infer_schema_length=None).sort(
        *COUNTY_KEYS, "moment_id"
    )
    require_unique(features, [*COUNTY_KEYS, "moment_id"], "county moment features")
    return features, cells


def _spec_design(
    cell: dict[str, Any], specification: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    selected = [
        moment
        for moment in cell["moments"]
        if moment["moment_family"] in specification["included_families"]
    ]
    active = [moment for moment in selected if moment["active"]]
    if active:
        design = np.column_stack(
            [moment["standardized_values"] for moment in active]
        )
        target = np.asarray(
            [moment["standardized_target"] for moment in active], dtype=float
        )
    else:
        design = np.empty((len(cell["county_codes"]), 0))
        target = np.empty(0)
    return design, target, selected


def _solver_diagnostic_rows(
    *,
    region: str,
    year: int,
    specification: dict[str, Any],
    solution: dict[str, np.ndarray],
    weight_kind: str,
    draw_ids: Sequence[int | None],
    kappa: float,
    frame_effective_county_count: float,
    active_moment_count: int,
    inactive_moment_count: int,
    simulation_seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for index, draw_id in enumerate(draw_ids):
        rows.append(
            {
                "aewr_region_id": region,
                "source_year": year,
                "specification": specification["specification"],
                "moment_spec": specification["moment_spec"],
                "is_primary": specification["is_primary"],
                "weight_spec": WEIGHT_SPEC,
                "baseline_weight_spec": BASELINE_WEIGHT_SPEC,
                "rho": PRIMARY_RHO,
                "kappa_multiplier": KAPPA_MULTIPLIER,
                "kappa": kappa,
                "frame_effective_county_count": frame_effective_county_count,
                "weight_kind": weight_kind,
                "weight_draw_id": draw_id,
                "simulation_seed": simulation_seed,
                "active_moment_count": active_moment_count,
                "inactive_moment_count": inactive_moment_count,
                "optimizer_success": bool(solution["success"][index]),
                "optimizer_status": str(solution["status"][index]),
                "optimizer_iterations": int(solution["iterations"][index]),
                "input_standardized_residual_norm": float(
                    solution["input_imbalance_norm"][index]
                ),
                "calibrated_standardized_residual_norm": float(
                    solution["calibrated_imbalance_norm"][index]
                ),
                "maximum_absolute_standardized_residual": float(
                    solution["maximum_absolute_imbalance"][index]
                ),
                "kl_divergence": float(solution["kl_divergence"][index]),
                "calibrated_effective_county_count": float(
                    solution["effective_county_count"][index]
                ),
                "maximum_calibrated_county_weight": float(
                    solution["maximum_county_weight"][index]
                ),
            }
        )
    return rows


def recover_cells(
    cells: dict[tuple[str, int], dict[str, Any]],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[tuple[str, int, str], pl.DataFrame]]:
    summary_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    moment_rows: list[dict[str, Any]] = []
    draw_partitions: dict[tuple[str, int, str], pl.DataFrame] = {}
    specifications = specification_grid()

    for cell_number, ((region, year), cell) in enumerate(cells.items(), start=1):
        print(
            f"Recovering county weights for AEWR region {region}, {year} "
            f"({cell_number}/{len(cells)})",
            flush=True,
        )
        prior = np.asarray(cell["prior"], dtype=float)
        supported = prior > 0
        supported_prior = prior[supported]
        supported_prior /= supported_prior.sum()
        seed = deterministic_seed(region, year)
        prior_draws, kappa, frame_effective = dirichlet_prior_draws(
            supported_prior,
            draw_count=DIAGNOSTIC_DRAW_COUNT,
            seed=seed,
        )

        for specification in specifications:
            design_all, target, selected_moments = _spec_design(cell, specification)
            design = design_all[supported]
            center_solution = solve_soft_entropy_batch(
                supported_prior, design, target, rho=PRIMARY_RHO
            )
            draw_solution = solve_soft_entropy_batch(
                prior_draws, design, target, rho=PRIMARY_RHO
            )
            active_count = sum(moment["active"] for moment in selected_moments)
            inactive_count = len(selected_moments) - active_count
            diagnostic_rows.extend(
                _solver_diagnostic_rows(
                    region=region,
                    year=year,
                    specification=specification,
                    solution=center_solution,
                    weight_kind="deterministic_center",
                    draw_ids=[None],
                    kappa=kappa,
                    frame_effective_county_count=frame_effective,
                    active_moment_count=active_count,
                    inactive_moment_count=inactive_count,
                    simulation_seed=seed,
                )
            )
            diagnostic_rows.extend(
                _solver_diagnostic_rows(
                    region=region,
                    year=year,
                    specification=specification,
                    solution=draw_solution,
                    weight_kind="dirichlet_draw",
                    draw_ids=list(range(1, DIAGNOSTIC_DRAW_COUNT + 1)),
                    kappa=kappa,
                    frame_effective_county_count=frame_effective,
                    active_moment_count=active_count,
                    inactive_moment_count=inactive_count,
                    simulation_seed=seed,
                )
            )
            if not bool(center_solution["success"][0]):
                raise RuntimeError(
                    f"County calibration failed for {region}, {year}, "
                    f"{specification['specification']}: {center_solution['status'][0]}"
                )

            center_all = np.zeros_like(prior)
            center_all[supported] = center_solution["weights"][0]
            draws_all = np.zeros((DIAGNOSTIC_DRAW_COUNT, len(prior)))
            draws_all[:, supported] = draw_solution["weights"]
            prior_draws_all = np.zeros_like(draws_all)
            prior_draws_all[:, supported] = prior_draws
            county_codes = cell["county_codes"]

            for county_index, county in enumerate(county_codes):
                values = draws_all[draw_solution["success"], county_index]
                summary_rows.append(
                    {
                        "aewr_region_id": region,
                        "source_year": year,
                        "county_fips": county,
                        "weight_draw_id": None,
                        "frame_prior_weight": float(prior[county_index]),
                        "calibrated_center_weight": float(center_all[county_index]),
                        "draw_mean_weight": float(values.mean()) if values.size else None,
                        "draw_standard_deviation_weight": (
                            float(values.std(ddof=1)) if values.size > 1 else 0.0
                        ),
                        "simulation_envelope_p025_weight": (
                            float(np.quantile(values, 0.025)) if values.size else None
                        ),
                        "simulation_envelope_p50_weight": (
                            float(np.quantile(values, 0.5)) if values.size else None
                        ),
                        "simulation_envelope_p975_weight": (
                            float(np.quantile(values, 0.975)) if values.size else None
                        ),
                        "center_solver_status": str(center_solution["status"][0]),
                        "draws_requested": DIAGNOSTIC_DRAW_COUNT,
                        "draws_succeeded": int(draw_solution["success"].sum()),
                        "draw_success_rate": float(draw_solution["success"].mean()),
                        "active_moment_count": active_count,
                        "inactive_moment_count": inactive_count,
                        "kappa": kappa,
                        "frame_effective_county_count": frame_effective,
                        "calibrated_effective_county_count": float(
                            center_solution["effective_county_count"][0]
                        ),
                        "maximum_calibrated_county_weight": float(
                            center_solution["maximum_county_weight"][0]
                        ),
                        "specification": specification["specification"],
                        "moment_spec": specification["moment_spec"],
                        "weight_spec": WEIGHT_SPEC,
                        "baseline_weight_spec": BASELINE_WEIGHT_SPEC,
                        "rho": PRIMARY_RHO,
                        "kappa_multiplier": KAPPA_MULTIPLIER,
                        "is_primary": specification["is_primary"],
                        "simulation_seed": seed,
                    }
                )

            draw_rows = []
            for draw_index in range(DIAGNOSTIC_DRAW_COUNT):
                for county_index, county in enumerate(county_codes):
                    draw_rows.append(
                        {
                            "aewr_region_id": region,
                            "source_year": year,
                            "county_fips": county,
                            "specification": specification["specification"],
                            "moment_spec": specification["moment_spec"],
                            "weight_draw_id": draw_index + 1,
                            "prior_draw_weight": float(
                                prior_draws_all[draw_index, county_index]
                            ),
                            "calibrated_draw_weight": (
                                float(draws_all[draw_index, county_index])
                                if draw_solution["success"][draw_index]
                                else None
                            ),
                            "optimizer_success": bool(
                                draw_solution["success"][draw_index]
                            ),
                            "optimizer_status": str(
                                draw_solution["status"][draw_index]
                            ),
                            "weight_spec": WEIGHT_SPEC,
                            "simulation_seed": seed,
                        }
                    )
            draw_partitions[(region, year, specification["specification"])] = (
                pl.DataFrame(draw_rows, infer_schema_length=None).sort(
                    "weight_draw_id", "county_fips"
                )
            )

            for moment in selected_moments:
                prior_raw = float(prior @ moment["raw_values"])
                calibrated_raw = float(center_all @ moment["raw_values"])
                prior_standardized = (
                    float(
                        supported_prior
                        @ moment["standardized_values"][supported]
                    )
                    if moment["active"]
                    else None
                )
                calibrated_standardized = (
                    float(
                        center_all[supported]
                        @ moment["standardized_values"][supported]
                    )
                    if moment["active"]
                    else None
                )
                moment_rows.append(
                    {
                        "aewr_region_id": region,
                        "source_year": year,
                        "specification": specification["specification"],
                        "moment_spec": specification["moment_spec"],
                        "is_primary": specification["is_primary"],
                        "moment_id": moment["moment_id"],
                        "moment_family": moment["moment_family"],
                        "quarter": moment["quarter"],
                        "moment_active": moment["active"],
                        "moment_status": moment["status"],
                        "observed_prior_mass": moment["observed_prior_mass"],
                        "raw_target": moment["raw_target"],
                        "prior_raw_moment": prior_raw,
                        "calibrated_raw_moment": calibrated_raw,
                        "prior_raw_residual": prior_raw - moment["raw_target"],
                        "calibrated_raw_residual": (
                            calibrated_raw - moment["raw_target"]
                        ),
                        "standardized_target": moment["standardized_target"],
                        "prior_standardized_moment": prior_standardized,
                        "calibrated_standardized_moment": calibrated_standardized,
                        "calibrated_standardized_residual": (
                            calibrated_standardized
                            - float(moment["standardized_target"])
                            if moment["active"]
                            else None
                        ),
                        "prior_center": moment["prior_center"],
                        "prior_scale": moment["prior_scale"],
                        "weight_spec": WEIGHT_SPEC,
                        "rho": PRIMARY_RHO,
                    }
                )

    summary = pl.DataFrame(summary_rows, infer_schema_length=None).with_columns(
        pl.col("weight_draw_id").cast(pl.Int64)
    )
    diagnostics = pl.DataFrame(diagnostic_rows, infer_schema_length=None).with_columns(
        pl.col("weight_draw_id").cast(pl.Int64)
    )
    moments = pl.DataFrame(moment_rows, infer_schema_length=None)
    require_unique(
        summary,
        [*COUNTY_KEYS, "specification"],
        "county calibration weight summary",
    )
    sums = summary.group_by(*CELL_KEYS, "specification").agg(
        pl.col("calibrated_center_weight").sum().alias("weight_sum")
    )
    if sums.filter((pl.col("weight_sum") - 1).abs() > WEIGHT_SUM_TOLERANCE).height:
        raise ValueError("Recovered county weights do not sum to one")
    return summary, diagnostics, moments, draw_partitions


def atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.write_parquet(temporary, compression="zstd")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def selected_expression(years: Sequence[int], regions: Sequence[str] | None) -> pl.Expr:
    expression = pl.col("source_year").is_in(years)
    if regions is not None:
        expression &= pl.col("aewr_region_id").is_in(regions)
    return expression


def replace_selected(
    path: Path,
    replacement: pl.DataFrame,
    *,
    years: Sequence[int],
    regions: Sequence[str] | None,
    sort_columns: list[str],
) -> None:
    pieces = []
    if path.exists():
        retained = pl.read_parquet(path).filter(
            ~selected_expression(years=years, regions=regions)
        )
        if not retained.is_empty():
            pieces.append(retained)
    if not replacement.is_empty():
        pieces.append(replacement)
    if not pieces:
        return
    combined = (
        pieces[0]
        if len(pieces) == 1
        else pl.concat(pieces, how="diagonal_relaxed")
    )
    atomic_write_parquet(combined.sort(sort_columns), path)


def draw_partition_path(region: str, year: int, specification: str) -> Path:
    return (
        DRAW_ROOT
        / f"aewr_region_id={region}"
        / f"source_year={year}"
        / f"specification={specification}"
        / "part-00000.parquet"
    )


def read_inputs() -> dict[str, pl.DataFrame]:
    paths = {
        "frame": INTERMEDIATE / "panel_iv_fls_frame.parquet",
        "county_panel": PROCESSED / "county_year_panel.parquet",
        "qcew": INTERMEDIATE / "qcew_county_ag_quarterly_employment.parquet",
        "fls_region": INTERMEDIATE / "fls_region.parquet",
        "fls_workers": INTERMEDIATE / "fls_region_quarterly_workers.parquet",
        "fls_wages": INTERMEDIATE / "fls_region_quarterly_wages.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing county-recovery inputs: " + ", ".join(missing))
    return {name: pl.read_parquet(path) for name, path in paths.items()}


def run_recovery(*, years: list[int], regions: list[str] | None) -> None:
    inputs = read_inputs()
    annual_targets = _annual_fls_targets(inputs["fls_region"], years)
    quarterly_targets = _paired_quarterly_targets(
        inputs["fls_workers"], inputs["fls_wages"], years
    )
    if regions is not None:
        annual_targets = annual_targets.filter(pl.col("aewr_region_id").is_in(regions))
        quarterly_targets = quarterly_targets.filter(
            pl.col("aewr_region_id").is_in(regions)
        )
    features, cells = build_county_features(
        inputs["frame"],
        inputs["county_panel"],
        inputs["qcew"],
        annual_targets,
        quarterly_targets,
        years=years,
        regions=regions,
    )
    expected_cells = len(years) * (len(regions) if regions is not None else 17)
    if len(cells) != expected_cells:
        raise ValueError(f"Expected {expected_cells} region-year cells, found {len(cells)}")
    summary, diagnostics, moments, draw_partitions = recover_cells(cells)

    replace_selected(
        FEATURE_PATH,
        features,
        years=years,
        regions=regions,
        sort_columns=[*COUNTY_KEYS, "moment_id"],
    )
    replace_selected(
        WEIGHT_SUMMARY_PATH,
        summary,
        years=years,
        regions=regions,
        sort_columns=[*CELL_KEYS, "specification", "county_fips"],
    )
    replace_selected(
        CALIBRATION_DIAGNOSTIC_PATH,
        diagnostics,
        years=years,
        regions=regions,
        sort_columns=[*CELL_KEYS, "specification", "weight_kind", "weight_draw_id"],
    )
    replace_selected(
        MOMENT_DIAGNOSTIC_PATH,
        moments,
        years=years,
        regions=regions,
        sort_columns=[*CELL_KEYS, "specification", "moment_id"],
    )
    for (region, year, specification), partition in draw_partitions.items():
        atomic_write_parquet(
            partition, draw_partition_path(region, year, specification)
        )

    primary_centers = diagnostics.filter(
        pl.col("specification") == PRIMARY_SPECIFICATION,
        pl.col("weight_kind") == "deterministic_center",
    )
    primary_draws = diagnostics.filter(
        pl.col("specification") == PRIMARY_SPECIFICATION,
        pl.col("weight_kind") == "dirichlet_draw",
    )
    if primary_centers.filter(~pl.col("optimizer_success")).height:
        raise RuntimeError("At least one preferred county calibration center failed")
    if primary_draws.group_by(*CELL_KEYS).agg(
        pl.col("optimizer_success").mean().alias("success_rate")
    ).filter(pl.col("success_rate") < 0.875).height:
        raise RuntimeError("Preferred diagnostic-draw success is below 87.5 percent")
    print(
        f"Wrote {features.height:,} county-moment features, "
        f"{summary.height:,} county weight summaries, and "
        f"{diagnostics.height:,} calibration diagnostics",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(SUPPORTED_YEARS))
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="optional AEWR region identifiers for a restartable partial rebuild",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = sorted(set(args.years))
    unsupported = sorted(set(years).difference(SUPPORTED_YEARS))
    if unsupported:
        raise ValueError(
            "Unsupported FLS county-recovery years: "
            + ", ".join(str(year) for year in unsupported)
        )
    regions = (
        sorted({str(int(region)) for region in args.regions}, key=int)
        if args.regions
        else None
    )
    if regions and any(not 1 <= int(region) <= 17 for region in regions):
        raise ValueError("AEWR region identifiers must be between 1 and 17")
    run_recovery(years=years, regions=regions)


if __name__ == "__main__":
    main()
