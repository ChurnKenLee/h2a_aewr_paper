"""Compare the Rust PPML solver with the saved JAX PPML fit.

The default comparison sends the complete JAX all-level design to Rust,
including coefficients that JAX estimated as zero. Higher-order L1 heredity
multipliers are frozen at the values implied by the saved JAX solution and
Rust fits the resulting ordinary convex elastic-net problem. This supports a
full inactive-variable KKT check while keeping the encoding, transforms,
exposures, targets, clipping, and penalties matched. The next year is retained
as a predeclared holdout and is never used by the Rust refit.

An explicitly requested ``--scope active`` run preserves the earlier,
restricted active-set diagnostic, but it is not the default comparison.

The wheel is extracted to a temporary directory at runtime, so it does not
need to be installed in the project environment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

PROJECT = Path(__file__).resolve().parents[2]
INTERMEDIATE = PROJECT / "data" / "intermediate"
WHEEL = (
    PROJECT
    / "code"
    / "a01_sources"
    / ("ppml_estimator-0.1.0-cp310-abi3-linux_x86_64.whl")
)

CAT_COLS = ("taxgrtgroup", "drainagecl", "nirrcapcl")
PATCH_CONT_COLS = (
    "slope_r",
    "slopegradwta",
    "resdept_r",
    "aws025wta",
    "aws050wta",
    "aws0100wta",
    "aws0150wta",
    "wtdepannmin",
    "wtdepaprjunmin",
    "brockdepmin",
    "cropprodindex",
    "slope_r_obs_share",
    "slopegradwta_obs_share",
    "resdept_r_obs_share",
    "aws025wta_obs_share",
    "aws050wta_obs_share",
    "aws0100wta_obs_share",
    "aws0150wta_obs_share",
    "wtdepannmin_obs_share",
    "wtdepaprjunmin_obs_share",
    "brockdepmin_obs_share",
    "cropprodindex_obs_share",
)


@dataclass(frozen=True)
class ActiveFeature:
    rust_index: int
    order: int
    feature_id: int
    coefficient_index: int
    feature_name: str
    categorical_columns: tuple[str, ...]
    categorical_values: tuple[str, ...]
    covariate_scope: str | None
    covariate: str | None
    jax_coefficient: float
    heredity_multiplier: float


def _softplus(value: float) -> float:
    return math.log1p(math.exp(-abs(value))) + max(value, 0.0)


def _load_penalties(cutoff_year: int) -> tuple[dict[int, float], dict[int, float]]:
    checkpoint_path = (
        PROJECT
        / "code"
        / "json"
        / f"meta_ppml_opt_checkpoint_cutoff_{cutoff_year}.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text())
    leaves = checkpoint["best_hparams_leaves"]
    if len(leaves) != 6:
        raise ValueError(
            f"Expected six best-hyperparameter leaves in {checkpoint_path}; "
            f"found {len(leaves)}."
        )
    l1 = {order: _softplus(float(leaves[order - 1])) for order in range(1, 4)}
    l2 = {order: _softplus(float(leaves[order + 2])) + 1e-4 for order in range(1, 4)}
    return l1, l2


def _load_saved_model(
    cutoff_year: int,
) -> tuple[
    pl.DataFrame,
    float,
    list[str],
    list[str],
    dict[int, np.ndarray],
    dict[int, list[dict[str, Any]]],
    dict[int, list[tuple[str, ...]]],
]:
    path = (
        INTERMEDIATE / f"h2a_prediction_elastic_net_model_cutoff_{cutoff_year}.parquet"
    )
    model = pl.read_parquet(path)
    bias_rows = model.filter(pl.col("record_type") == "global_bias")
    if bias_rows.height != 1:
        raise ValueError(f"{path} must contain exactly one global bias row.")
    bias = float(np.float32(bias_rows["coefficient"][0]))

    transforms = model.filter(pl.col("record_type") == "continuous_transform")
    county_cont_cols = transforms.filter(pl.col("covariate_scope") == "county")[
        "covariate"
    ].to_list()
    patch_cont_cols = transforms.filter(pl.col("covariate_scope") == "patch")[
        "covariate"
    ].to_list()
    if tuple(patch_cont_cols) != PATCH_CONT_COLS:
        raise ValueError("Saved patch covariates do not match the JAX specification.")
    covariates = county_cont_cols + patch_cont_cols
    width = 1 + len(covariates)

    weights = model.filter(pl.col("record_type") == "weight")
    matrices: dict[int, np.ndarray] = {}
    metadata: dict[int, list[dict[str, Any]]] = {}
    combos: dict[int, list[tuple[str, ...]]] = {}
    for order in range(1, 4):
        order_rows = weights.filter(pl.col("interaction_order") == order)
        feature_count = int(order_rows["feature_id"].max()) + 1
        if order_rows.height != feature_count * width:
            raise ValueError(f"Order-{order} saved coefficient table is incomplete.")
        feature_sequence = (
            order_rows["feature_id"].to_numpy().reshape(feature_count, width)
        )
        expected_sequence = np.arange(feature_count, dtype=feature_sequence.dtype)
        if not np.array_equal(feature_sequence[:, 0], expected_sequence) or not np.all(
            feature_sequence == feature_sequence[:, :1]
        ):
            raise ValueError(
                f"Order-{order} coefficients have unexpected row ordering."
            )
        matrices[order] = (
            order_rows["coefficient"]
            .to_numpy()
            .astype(np.float32, copy=False)
            .reshape(feature_count, width)
        )

        feature_rows = (
            order_rows.select(
                "feature_id",
                "feature_name",
                "categorical_columns",
                "categorical_values",
            )
            .unique(subset=["feature_id"], maintain_order=True)
            .sort("feature_id")
        )
        order_metadata: list[dict[str, Any]] = []
        combo_first_id: dict[tuple[str, ...], int] = {}
        for row in feature_rows.iter_rows(named=True):
            columns = tuple(row["categorical_columns"])
            values = tuple(row["categorical_values"])
            feature_id = int(row["feature_id"])
            order_metadata.append(
                {
                    "feature_id": feature_id,
                    "feature_name": str(row["feature_name"]),
                    "columns": columns,
                    "values": values,
                }
            )
            combo_first_id.setdefault(columns, feature_id)
        if [row["feature_id"] for row in order_metadata] != list(range(feature_count)):
            raise ValueError(f"Order-{order} categorical metadata is incomplete.")
        metadata[order] = order_metadata
        combos[order] = sorted(combo_first_id, key=combo_first_id.__getitem__)

    return (
        model,
        bias,
        county_cont_cols,
        patch_cont_cols,
        matrices,
        metadata,
        combos,
    )


def _prepare_training_frame(
    cutoff_year: int,
    model: pl.DataFrame,
    county_cont_cols: list[str],
    patch_cont_cols: list[str],
    start_year: int = 2008,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    bea = (
        pl.read_parquet(INTERMEDIATE / "bea_farm_nonfarm_emp.parquet")
        .filter(pl.col("year") == 2011)
        .rename({"bea_farm_emp": "bea_farm_emp_2011"})
        .drop("bea_nonfarm_emp", "year")
    )
    h2a = (
        pl.read_parquet(INTERMEDIATE / "h2a_aggregated.parquet")
        .with_columns(pl.col("year").cast(pl.Int32))
        .rename({"nbr_workers_certified_start_year": "h2a_certified"})
        .select("year", "county_fips", "h2a_certified")
        .filter(pl.col("year").is_between(start_year, cutoff_year))
    )
    climate = (
        pl.read_parquet(
            INTERMEDIATE / "county_h2a_prediction_climate_basis_annual.parquet"
        )
        .select("year", "county_fips", pl.col("^normal_cb_.*$"))
        .filter(pl.col("year").is_between(start_year, cutoff_year))
    )
    soil = pl.read_parquet(
        INTERMEDIATE / "county_h2a_prediction_gnatsgo_soil_cells.parquet"
    ).with_row_index("soil_component_key")

    merged = (
        soil.join(climate, on="county_fips", how="inner")
        .join(h2a, on=["county_fips", "year"], how="left")
        .with_columns(pl.col("h2a_certified").fill_null(0))
        .join(bea, on="county_fips", how="inner")
        .filter(
            pl.col("bea_farm_emp_2011").is_not_null()
            & (pl.col("bea_farm_emp_2011") > 0)
        )
        .with_columns(
            (pl.col("h2a_certified") / pl.col("bea_farm_emp_2011"))
            .clip(0.0, 2.0)
            .alias("h2a_rate")
        )
        .with_columns(
            (pl.col("h2a_rate") * pl.col("bea_farm_emp_2011")).alias(
                "h2a_target_count"
            ),
            pl.struct(["county_fips", "year"]).rank("dense").alias("group_id") - 1,
        )
    )

    transform_rows = model.filter(pl.col("record_type") == "continuous_transform")
    transform_map = {
        (row["covariate_scope"], row["covariate"]): row
        for row in transform_rows.iter_rows(named=True)
    }
    continuous = county_cont_cols + patch_cont_cols
    merged = merged.with_columns(
        [
            pl.when(pl.col(column).is_nan())
            .then(None)
            .otherwise(pl.col(column))
            .alias(column)
            for column in continuous
        ]
    ).with_columns(
        [
            pl.col(column).fill_null(
                pl.lit(transform_map[("county", column)]["imputation_value"])
            )
            for column in county_cont_cols
        ]
        + [
            pl.col(column).fill_null(
                pl.lit(transform_map[("patch", column)]["imputation_value"])
            )
            for column in patch_cont_cols
        ]
        + [pl.col(column).fill_null("MISSING") for column in CAT_COLS]
    )
    merged = (
        merged.with_columns(
            (
                pl.col("total_acres") / pl.col("total_acres").sum().over("group_id")
            ).alias("acreage_frac")
        )
        .with_columns(
            (pl.col("acreage_frac") * pl.col("bea_farm_emp_2011")).alias(
                "patch_exposure"
            )
        )
        .sort("group_id")
    )

    group_frame = (
        merged.select(
            "group_id",
            "county_fips",
            "year",
            "h2a_target_count",
            *county_cont_cols,
        )
        .unique(subset=["group_id"], maintain_order=True)
        .sort("group_id")
    )
    num_groups = int(merged["group_id"].max()) + 1
    if group_frame.height != num_groups:
        raise ValueError("Training frame does not contain one row per dense group ID.")

    def transform_matrix(raw: np.ndarray, scope: str, columns: list[str]) -> np.ndarray:
        output = np.empty_like(raw, dtype=np.float32)
        for column_index, column in enumerate(columns):
            transform = transform_map[(scope, column)]
            output[:, column_index] = (
                raw[:, column_index] - np.float32(transform["center"])
            ) / np.float32(transform["scale"])
        return output

    county_raw = group_frame.select(county_cont_cols).to_numpy().astype(np.float32)
    patch_raw = merged.select(patch_cont_cols).to_numpy().astype(np.float32)
    x_county = transform_matrix(county_raw, "county", county_cont_cols)
    x_patch = transform_matrix(patch_raw, "patch", patch_cont_cols)
    group_ids = merged["group_id"].to_numpy().astype(np.uint64, copy=False)
    exposure = merged["patch_exposure"].to_numpy().astype(np.float32).astype(np.float64)
    outcomes = (
        group_frame["h2a_target_count"].to_numpy().astype(np.float32).astype(np.float64)
    )
    component_tokens = merged["soil_component_key"].to_numpy()
    _, component_indices = np.unique(component_tokens, return_inverse=True)
    return (
        merged,
        group_frame,
        x_county,
        x_patch,
        group_ids,
        exposure,
        outcomes,
        component_indices.astype(np.uint64, copy=False),
    )


def _categorical_ids(
    merged: pl.DataFrame,
    metadata: dict[int, list[dict[str, Any]]],
    combos: dict[int, list[tuple[str, ...]]],
) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for order in range(1, 4):
        columns_for_order: list[np.ndarray] = []
        for combo in combos[order]:
            lookup = {
                row["values"]: row["feature_id"]
                for row in metadata[order]
                if row["columns"] == combo
            }
            column_values = [
                merged[column].cast(pl.String).to_list() for column in combo
            ]
            row_values = zip(*column_values, strict=True)
            try:
                encoded = np.fromiter(
                    (lookup[values] for values in row_values),
                    dtype=np.int32,
                    count=merged.height,
                )
            except KeyError as error:
                raise ValueError(
                    f"Saved order-{order} categorical mapping is missing {error.args[0]!r}."
                ) from error
            columns_for_order.append(encoded)
        output[order] = np.column_stack(columns_for_order)
    return output


def _heredity_multipliers(
    matrices: dict[int, np.ndarray],
    metadata: dict[int, list[dict[str, Any]]],
) -> dict[int, np.ndarray]:
    multipliers = {1: np.ones(matrices[1].shape[0], dtype=np.float64)}
    lookup = {
        (order, row["columns"], row["values"]): row["feature_id"]
        for order in range(1, 4)
        for row in metadata[order]
    }
    for order in range(2, 4):
        values = np.empty(matrices[order].shape[0], dtype=np.float64)
        for row in metadata[order]:
            parent_strengths = []
            columns = row["columns"]
            categorical_values = row["values"]
            for positions in itertools.combinations(range(order), order - 1):
                parent_columns = tuple(columns[index] for index in positions)
                parent_values = tuple(categorical_values[index] for index in positions)
                parent_id = lookup[(order - 1, parent_columns, parent_values)]
                parent_weights = matrices[order - 1][parent_id].astype(
                    np.float64, copy=False
                )
                parent_strengths.append(float(np.mean(parent_weights**2)))
            parent_strength = min(parent_strengths)
            values[row["feature_id"]] = max(1.0, 1e-3 / (parent_strength + 1e-8))
        multipliers[order] = values
    return multipliers


def _select_active_features(
    bias: float,
    county_cont_cols: list[str],
    patch_cont_cols: list[str],
    matrices: dict[int, np.ndarray],
    metadata: dict[int, list[dict[str, Any]]],
    multipliers: dict[int, np.ndarray],
    threshold: float,
) -> tuple[list[ActiveFeature], dict[int, list[list[ActiveFeature]]], np.ndarray]:
    covariates = [("county", name) for name in county_cont_cols] + [
        ("patch", name) for name in patch_cont_cols
    ]
    active: list[ActiveFeature] = []
    by_order: dict[int, list[list[ActiveFeature]]] = {}
    initial = [bias]
    rust_index = 1
    for order in range(1, 4):
        by_feature = [[] for _ in range(matrices[order].shape[0])]
        feature_ids, coefficient_ids = np.nonzero(np.abs(matrices[order]) > threshold)
        for feature_id, coefficient_index in zip(
            feature_ids.tolist(), coefficient_ids.tolist(), strict=True
        ):
            row = metadata[order][feature_id]
            if coefficient_index == 0:
                scope = None
                covariate = None
            else:
                scope, covariate = covariates[coefficient_index - 1]
            feature = ActiveFeature(
                rust_index=rust_index,
                order=order,
                feature_id=feature_id,
                coefficient_index=coefficient_index,
                feature_name=row["feature_name"],
                categorical_columns=row["columns"],
                categorical_values=row["values"],
                covariate_scope=scope,
                covariate=covariate,
                jax_coefficient=float(matrices[order][feature_id, coefficient_index]),
                heredity_multiplier=float(multipliers[order][feature_id]),
            )
            active.append(feature)
            by_feature[feature_id].append(feature)
            initial.append(feature.jax_coefficient)
            rust_index += 1
        by_order[order] = by_feature
    return active, by_order, np.asarray(initial, dtype=np.float64)


def _jax_predictions(
    bias: float,
    matrices: dict[int, np.ndarray],
    categorical_ids: dict[int, np.ndarray],
    x_county: np.ndarray,
    x_patch: np.ndarray,
    group_ids: np.ndarray,
    exposure: np.ndarray,
    num_groups: int,
) -> np.ndarray:
    x_county_by_membership = x_county[group_ids]
    county_width = x_county.shape[1]
    linear_predictor = np.full(exposure.size, np.float32(bias), dtype=np.float32)
    for order in range(1, 4):
        for combo_index in range(categorical_ids[order].shape[1]):
            selected = matrices[order][categorical_ids[order][:, combo_index]]
            linear_predictor += selected[:, 0]
            linear_predictor += np.sum(
                selected[:, 1 : 1 + county_width] * x_county_by_membership,
                axis=1,
                dtype=np.float32,
            )
            linear_predictor += np.sum(
                selected[:, 1 + county_width :] * x_patch,
                axis=1,
                dtype=np.float32,
            )
            del selected
    np.minimum(linear_predictor, np.float32(15.0), out=linear_predictor)
    membership_predictions = exposure * np.exp(linear_predictor.astype(np.float64))
    return np.bincount(
        group_ids.astype(np.int64, copy=False),
        weights=membership_predictions,
        minlength=num_groups,
    )


def _build_native_data(
    core: Any,
    active_by_order: dict[int, list[list[ActiveFeature]]],
    categorical_ids: dict[int, np.ndarray],
    combos: dict[int, list[tuple[str, ...]]],
    x_county: np.ndarray,
    x_patch: np.ndarray,
    group_ids: np.ndarray,
    component_indices: np.ndarray,
    exposure: np.ndarray,
    outcomes: np.ndarray,
) -> Any:
    num_memberships = exposure.size
    num_groups = outcomes.size
    num_components = int(component_indices.max()) + 1

    block_counts: list[np.ndarray] = []
    block_specs: list[tuple[int, int, np.ndarray]] = []
    row_counts = np.ones(num_memberships, dtype=np.uint64)
    for order in range(1, 4):
        counts_lookup = np.asarray(
            [len(features) for features in active_by_order[order]], dtype=np.uint16
        )
        for combo_index in range(len(combos[order])):
            ids = categorical_ids[order][:, combo_index]
            counts = counts_lookup[ids]
            block_counts.append(counts)
            block_specs.append((order, combo_index, ids))
            row_counts += counts

    row_offsets = np.empty(num_memberships + 1, dtype=np.uint64)
    row_offsets[0] = 0
    np.cumsum(row_counts, dtype=np.uint64, out=row_offsets[1:])
    total_entries = int(row_offsets[-1])
    print(
        f"Active-set Rust design: {total_entries:,} nonzeros "
        f"({total_entries / num_memberships:.1f} per membership).",
        flush=True,
    )
    feature_indices = np.empty(total_entries, dtype=np.uint64)
    feature_values = np.empty(total_entries, dtype=np.float64)
    row_starts = row_offsets[:-1]
    feature_indices[row_starts] = 0
    feature_values[row_starts] = 1.0
    cursor = row_starts.copy() + 1

    county_width = x_county.shape[1]
    for block_number, ((order, combo_index, ids), counts) in enumerate(
        zip(block_specs, block_counts, strict=True), start=1
    ):
        permutation = np.argsort(ids, kind="stable")
        sorted_ids = ids[permutation]
        boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [num_memberships]))
        for start, stop in zip(starts, stops, strict=True):
            feature_id = int(sorted_ids[start])
            features = active_by_order[order][feature_id]
            if not features:
                continue
            rows = permutation[start:stop]
            positions = cursor[rows]
            for within_feature, feature in enumerate(features):
                write_positions = positions + within_feature
                feature_indices[write_positions] = feature.rust_index
                coefficient_index = feature.coefficient_index
                if coefficient_index == 0:
                    feature_values[write_positions] = 1.0
                elif coefficient_index <= county_width:
                    feature_values[write_positions] = x_county[
                        group_ids[rows], coefficient_index - 1
                    ]
                else:
                    feature_values[write_positions] = x_patch[
                        rows, coefficient_index - 1 - county_width
                    ]
        cursor += counts
        print(
            f"  populated categorical block {block_number}/{len(block_specs)}",
            flush=True,
        )
    if not np.array_equal(cursor, row_offsets[1:]):
        raise AssertionError("Active-set CSR row offsets were not filled exactly.")

    group_counts = np.bincount(
        group_ids.astype(np.int64, copy=False), minlength=num_groups
    )
    group_offsets = np.empty(num_groups + 1, dtype=np.uint64)
    group_offsets[0] = 0
    np.cumsum(group_counts, dtype=np.uint64, out=group_offsets[1:])

    signature_columns = np.column_stack(
        [
            categorical_ids[order][:, combo_index]
            for order in range(1, 4)
            for combo_index in range(len(combos[order]))
        ]
    )
    _, signature_ids = np.unique(signature_columns, axis=0, return_inverse=True)
    num_signatures = int(signature_ids.max()) + 1
    num_group_signature_pairs = np.unique(
        np.column_stack((group_ids, signature_ids)), axis=0
    ).shape[0]
    num_component_signature_pairs = np.unique(
        np.column_stack((component_indices, signature_ids)), axis=0
    ).shape[0]
    del signature_columns, signature_ids

    num_features = (
        max(
            feature.rust_index
            for order_features in active_by_order.values()
            for features in order_features
            for feature in features
        )
        + 1
    )
    feature_orders = np.empty(num_features, dtype=np.int8)
    feature_orders[0] = -1
    for order, order_features in active_by_order.items():
        for features in order_features:
            for feature in features:
                feature_orders[feature.rust_index] = order
    parent_offsets = np.zeros(num_features + 1, dtype=np.uint64)
    parents = np.empty(0, dtype=np.uint64)

    print("Transferring the CSR design to the Rust extension...", flush=True)
    buffers = {
        "group_offsets": group_offsets.tobytes(),
        "component_indices": component_indices.astype("<u8", copy=False).tobytes(),
        "exposure": exposure.astype("<f8", copy=False).tobytes(),
        "outcomes": outcomes.astype("<f8", copy=False).tobytes(),
        "weights": np.ones(num_groups, dtype="<f8").tobytes(),
        "offsets": np.zeros(num_groups, dtype="<f8").tobytes(),
        "row_offsets": row_offsets.tobytes(),
        "feature_indices": feature_indices.tobytes(),
        "feature_values": feature_values.tobytes(),
        "feature_orders": feature_orders.tobytes(),
        "parent_offsets": parent_offsets.tobytes(),
        "parents": parents.tobytes(),
    }
    native = core.PreparedData(
        num_groups,
        num_components,
        num_signatures,
        num_group_signature_pairs,
        num_component_signature_pairs,
        buffers["group_offsets"],
        buffers["component_indices"],
        buffers["exposure"],
        buffers["outcomes"],
        buffers["weights"],
        buffers["offsets"],
        buffers["row_offsets"],
        buffers["feature_indices"],
        buffers["feature_values"],
        buffers["feature_orders"],
        buffers["parent_offsets"],
        buffers["parents"],
        15.0,
    )
    del buffers, feature_indices, feature_values
    gc.collect()
    print(f"Rust data dimensions: {native.dimensions_json}", flush=True)
    return native


def _summary_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = candidate - reference
    if reference.size > 1 and np.std(reference) > 0 and np.std(candidate) > 0:
        correlation = float(np.corrcoef(reference, candidate)[0, 1])
    else:
        correlation = float("nan")
    return {
        "n": int(reference.size),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "mean_abs_difference": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "correlation": correlation,
        "max_relative_difference_floor_1": float(
            np.max(np.abs(difference) / np.maximum(1.0, np.abs(reference)))
        ),
    }


def _fit_metrics(outcomes: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    residual = predictions - outcomes
    if np.std(outcomes) > 0 and np.std(predictions) > 0:
        correlation = float(np.corrcoef(outcomes, predictions)[0, 1])
    else:
        correlation = float("nan")
    return {
        "total_predicted": float(np.sum(predictions)),
        "mean_predicted": float(np.mean(predictions)),
        "mean_absolute_error": float(np.mean(np.abs(residual))),
        "root_mean_squared_error": float(np.sqrt(np.mean(residual**2))),
        "poisson_nll": float(
            np.mean(predictions - outcomes * np.log(predictions + 1e-7))
        ),
        "outcome_prediction_correlation": correlation,
    }


def _full_parameter_vectors(
    bias: float,
    matrices: dict[int, np.ndarray],
    multipliers: dict[int, np.ndarray],
    l1: dict[int, float],
    l2: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    initial_parts = [np.asarray([bias], dtype=np.float64)]
    l1_parts = [np.zeros(1, dtype=np.float64)]
    l2_parts = [np.zeros(1, dtype=np.float64)]
    order_parts = [np.asarray([-1], dtype=np.int8)]
    for order in range(1, 4):
        width = matrices[order].shape[1]
        initial_parts.append(matrices[order].astype(np.float64, copy=False).ravel())
        l1_parts.append(
            l1[order] * np.repeat(multipliers[order].astype(np.float64), width)
        )
        l2_parts.append(np.full(matrices[order].size, l2[order], dtype=np.float64))
        order_parts.append(np.full(matrices[order].size, order, dtype=np.int8))
    return (
        np.concatenate(initial_parts),
        np.concatenate(l1_parts),
        np.concatenate(l2_parts),
        np.concatenate(order_parts),
    )


def _build_native_data_full(
    core: Any,
    matrices: dict[int, np.ndarray],
    categorical_ids: dict[int, np.ndarray],
    combos: dict[int, list[tuple[str, ...]]],
    x_county: np.ndarray,
    x_patch: np.ndarray,
    group_ids: np.ndarray,
    component_indices: np.ndarray,
    exposure: np.ndarray,
    outcomes: np.ndarray,
    *,
    max_design_bytes: int,
) -> Any:
    num_memberships = exposure.size
    num_groups = outcomes.size
    num_components = int(component_indices.max()) + 1
    width = 1 + x_county.shape[1] + x_patch.shape[1]
    block_count = sum(len(combos[order]) for order in range(1, 4))
    row_width = 1 + block_count * width
    total_entries = num_memberships * row_width
    design_bytes = total_entries * 16
    if design_bytes > max_design_bytes:
        raise MemoryError(
            f"Full sparse design requires {design_bytes:,} bytes, exceeding the "
            f"explicit --max-design-bytes limit of {max_design_bytes:,}."
        )
    print(
        f"Full Rust design: {total_entries:,} nonzeros, {row_width} per membership, "
        f"{design_bytes / (1024**3):.2f} GiB forward storage.",
        flush=True,
    )

    feature_indices = np.empty((num_memberships, row_width), dtype=np.uint64)
    feature_values = np.empty((num_memberships, row_width), dtype=np.float64)
    feature_indices[:, 0] = 0
    feature_values[:, 0] = 1.0
    basis = np.empty((num_memberships, width), dtype=np.float64)
    basis[:, 0] = 1.0
    county_width = x_county.shape[1]
    basis[:, 1 : 1 + county_width] = x_county[group_ids]
    basis[:, 1 + county_width :] = x_patch

    column_start = 1
    feature_start = 1
    completed_block = 0
    coefficient_offsets = np.arange(width, dtype=np.uint64)
    for order in range(1, 4):
        for combo_index in range(len(combos[order])):
            ids = categorical_ids[order][:, combo_index].astype(np.uint64, copy=False)
            row_feature_start = feature_start + ids * width
            block_slice = slice(column_start, column_start + width)
            feature_indices[:, block_slice] = (
                row_feature_start[:, None] + coefficient_offsets
            )
            feature_values[:, block_slice] = basis
            column_start += width
            completed_block += 1
            print(
                f"  populated full categorical block {completed_block}/{block_count}",
                flush=True,
            )
        feature_start += matrices[order].size
    if column_start != row_width:
        raise AssertionError("Full CSR design width was not filled exactly.")
    del basis
    gc.collect()

    group_counts = np.bincount(
        group_ids.astype(np.int64, copy=False), minlength=num_groups
    )
    group_offsets = np.empty(num_groups + 1, dtype=np.uint64)
    group_offsets[0] = 0
    np.cumsum(group_counts, dtype=np.uint64, out=group_offsets[1:])
    row_offsets = np.arange(
        0,
        (num_memberships + 1) * row_width,
        row_width,
        dtype=np.uint64,
    )

    signature_columns = np.column_stack(
        [
            categorical_ids[order][:, combo_index]
            for order in range(1, 4)
            for combo_index in range(len(combos[order]))
        ]
    )
    _, signature_ids = np.unique(signature_columns, axis=0, return_inverse=True)
    num_signatures = int(signature_ids.max()) + 1
    num_group_signature_pairs = np.unique(
        np.column_stack((group_ids, signature_ids)), axis=0
    ).shape[0]
    num_component_signature_pairs = np.unique(
        np.column_stack((component_indices, signature_ids)), axis=0
    ).shape[0]
    del signature_columns, signature_ids

    feature_orders = np.concatenate(
        [np.asarray([-1], dtype=np.int8)]
        + [np.full(matrices[order].size, order, dtype=np.int8) for order in range(1, 4)]
    )
    parent_offsets = np.zeros(feature_orders.size + 1, dtype=np.uint64)
    parents = np.empty(0, dtype=np.uint64)

    print("Serializing the complete CSR design for the Rust extension...", flush=True)
    feature_indices_bytes = feature_indices.ravel().tobytes()
    del feature_indices
    gc.collect()
    feature_values_bytes = feature_values.ravel().tobytes()
    del feature_values
    gc.collect()
    buffers = {
        "group_offsets": group_offsets.tobytes(),
        "component_indices": component_indices.astype("<u8", copy=False).tobytes(),
        "exposure": exposure.astype("<f8", copy=False).tobytes(),
        "outcomes": outcomes.astype("<f8", copy=False).tobytes(),
        "weights": np.ones(num_groups, dtype="<f8").tobytes(),
        "offsets": np.zeros(num_groups, dtype="<f8").tobytes(),
        "row_offsets": row_offsets.tobytes(),
        "feature_indices": feature_indices_bytes,
        "feature_values": feature_values_bytes,
        "feature_orders": feature_orders.tobytes(),
        "parent_offsets": parent_offsets.tobytes(),
        "parents": parents.tobytes(),
    }
    native = core.PreparedData(
        num_groups,
        num_components,
        num_signatures,
        num_group_signature_pairs,
        num_component_signature_pairs,
        buffers["group_offsets"],
        buffers["component_indices"],
        buffers["exposure"],
        buffers["outcomes"],
        buffers["weights"],
        buffers["offsets"],
        buffers["row_offsets"],
        buffers["feature_indices"],
        buffers["feature_values"],
        buffers["feature_orders"],
        buffers["parent_offsets"],
        buffers["parents"],
        15.0,
    )
    del buffers, feature_indices_bytes, feature_values_bytes
    gc.collect()
    print(f"Rust full-data dimensions: {native.dimensions_json}", flush=True)
    return native


def _rust_diagnostics(
    core: Any,
    native_data: Any,
    coefficients: np.ndarray,
    l1_features: np.ndarray,
    l2_features: np.ndarray,
) -> tuple[float, np.ndarray]:
    native = core.diagnostics(
        native_data,
        coefficients.astype("<f8", copy=False).tobytes(),
        l1_features.astype("<f8", copy=False).tobytes(),
        l2_features.astype("<f8", copy=False).tobytes(),
        np.zeros(coefficients.size, dtype="<f8").tobytes(),
        "f64",
        "fisher",
        None,
    )
    return (
        float(native.objective),
        np.frombuffer(native.gradient, dtype="<f8").copy(),
    )


def _kkt_metrics(
    coefficients: np.ndarray,
    smooth_gradient: np.ndarray,
    l1_features: np.ndarray,
    feature_orders: np.ndarray,
    active_threshold: float,
    kkt_tolerance: float,
) -> tuple[dict[str, float | int], np.ndarray]:
    active = np.abs(coefficients) > active_threshold
    violation = np.where(
        active,
        np.abs(smooth_gradient + l1_features * np.sign(coefficients)),
        np.maximum(np.abs(smooth_gradient) - l1_features, 0.0),
    )
    penalized = feature_orders >= 0
    inactive = penalized & ~active
    return (
        {
            "active_parameters_including_bias": int(np.count_nonzero(active)),
            "inactive_penalized_parameters": int(np.count_nonzero(inactive)),
            "max_violation": float(np.max(violation, initial=0.0)),
            "max_inactive_violation": float(np.max(violation[inactive], initial=0.0)),
            "inactive_violations_above_kkt_tolerance": int(
                np.count_nonzero(violation[inactive] > kkt_tolerance)
            ),
        },
        violation,
    )


def _unflatten_coefficients(
    coefficients: np.ndarray,
    matrices: dict[int, np.ndarray],
) -> tuple[float, dict[int, np.ndarray]]:
    output: dict[int, np.ndarray] = {}
    cursor = 1
    for order in range(1, 4):
        stop = cursor + matrices[order].size
        output[order] = coefficients[cursor:stop].reshape(matrices[order].shape)
        cursor = stop
    if cursor != coefficients.size:
        raise ValueError("Rust coefficient vector has an unexpected length.")
    return float(coefficients[0]), output


def _predictions_from_coefficients(
    bias: float,
    matrices: dict[int, np.ndarray],
    categorical_ids: dict[int, np.ndarray],
    x_county: np.ndarray,
    x_patch: np.ndarray,
    group_ids: np.ndarray,
    exposure: np.ndarray,
    num_groups: int,
    *,
    dtype: np.dtype[Any],
) -> np.ndarray:
    x_county_by_membership = x_county[group_ids].astype(dtype, copy=False)
    x_patch_typed = x_patch.astype(dtype, copy=False)
    county_width = x_county.shape[1]
    linear_predictor = np.full(exposure.size, dtype.type(bias), dtype=dtype)
    for order in range(1, 4):
        matrix = matrices[order].astype(dtype, copy=False)
        for combo_index in range(categorical_ids[order].shape[1]):
            selected = matrix[categorical_ids[order][:, combo_index]]
            linear_predictor += selected[:, 0]
            linear_predictor += np.sum(
                selected[:, 1 : 1 + county_width] * x_county_by_membership,
                axis=1,
                dtype=dtype,
            )
            linear_predictor += np.sum(
                selected[:, 1 + county_width :] * x_patch_typed,
                axis=1,
                dtype=dtype,
            )
            del selected
    np.minimum(linear_predictor, dtype.type(15.0), out=linear_predictor)
    membership_predictions = exposure * np.exp(
        linear_predictor.astype(np.float64, copy=False)
    )
    return np.bincount(
        group_ids.astype(np.int64, copy=False),
        weights=membership_predictions,
        minlength=num_groups,
    )


def _run_active(args: argparse.Namespace, core: Any, package_version: str) -> None:
    started = time.perf_counter()
    (
        model,
        bias,
        county_cont_cols,
        patch_cont_cols,
        matrices,
        metadata,
        combos,
    ) = _load_saved_model(args.cutoff_year)
    l1, l2 = _load_penalties(args.cutoff_year)
    print(f"JAX L1 rates: {l1}")
    print(f"JAX L2 rates: {l2}")

    (
        merged,
        group_frame,
        x_county,
        x_patch,
        group_ids,
        exposure,
        outcomes,
        component_indices,
    ) = _prepare_training_frame(
        args.cutoff_year, model, county_cont_cols, patch_cont_cols
    )
    categorical_ids = _categorical_ids(merged, metadata, combos)
    multipliers = _heredity_multipliers(matrices, metadata)
    active, active_by_order, initial = _select_active_features(
        bias,
        county_cont_cols,
        patch_cont_cols,
        matrices,
        metadata,
        multipliers,
        args.active_threshold,
    )
    print(
        f"JAX active set at |coefficient| > {args.active_threshold:g}: "
        f"{len(active):,} penalized coefficients plus the bias."
    )
    jax_predictions = _jax_predictions(
        bias,
        matrices,
        categorical_ids,
        x_county,
        x_patch,
        group_ids,
        exposure,
        outcomes.size,
    )

    l1_features = np.zeros(initial.size, dtype=np.float64)
    l2_features = np.zeros(initial.size, dtype=np.float64)
    for feature in active:
        l1_features[feature.rust_index] = (
            l1[feature.order] * feature.heredity_multiplier
        )
        l2_features[feature.rust_index] = l2[feature.order]

    initial_nll = np.mean(jax_predictions - outcomes * np.log(jax_predictions + 1e-7))
    initial_objective = float(
        initial_nll
        + np.sum(l1_features * np.abs(initial))
        + np.sum(l2_features * initial**2)
    )

    native_data = _build_native_data(
        core,
        active_by_order,
        categorical_ids,
        combos,
        x_county,
        x_patch,
        group_ids,
        component_indices,
        exposure,
        outcomes,
    )
    config = json.dumps(
        {
            "max_iterations": args.max_iterations,
            "tolerance": args.tolerance,
            "initial_lipschitz": 1.0,
            "backtracking_growth": 1.5,
            "max_backtracking_steps": 40,
            "heredity": "none",
            "precision": "f64",
            "gpu_device": None,
        },
        separators=(",", ":"),
    )
    fit_started = time.perf_counter()
    native_fit = core.fit(
        native_data,
        l1_features.astype("<f8", copy=False).tobytes(),
        l2_features.astype("<f8", copy=False).tobytes(),
        config,
        initial.astype("<f8", copy=False).tobytes(),
    )
    fit_seconds = time.perf_counter() - fit_started
    rust_coefficients = np.frombuffer(native_fit.coefficients, dtype="<f8").copy()
    rust_predictions = np.frombuffer(native_fit.predictions, dtype="<f8").copy()
    objective_history = np.frombuffer(native_fit.objective_history, dtype="<f8").copy()
    convergence = json.loads(native_fit.report_json)

    coefficient_metrics = _summary_metrics(initial, rust_coefficients)
    prediction_metrics = _summary_metrics(jax_predictions, rust_predictions)
    summary = {
        "cutoff_year": args.cutoff_year,
        "comparison_scope": "jax_selected_active_set",
        "active_threshold": args.active_threshold,
        "rust_package_version": package_version,
        "rust_wheel_sha256": hashlib.sha256(WHEEL.read_bytes()).hexdigest(),
        "rust_backend": "cpu",
        "rust_precision": "f64",
        "jax_precision": "f32",
        "groups": int(outcomes.size),
        "memberships": int(exposure.size),
        "jax_total_penalized_coefficients": int(
            sum(matrix.size for matrix in matrices.values())
        ),
        "active_penalized_coefficients": len(active),
        "initial_jax_objective_on_frozen_active_set": initial_objective,
        "rust_final_objective": float(convergence["objective"]),
        "rust_iterations": int(convergence["iterations"]),
        "rust_converged": bool(convergence["converged"]),
        "rust_requested_max_iterations": args.max_iterations,
        "rust_requested_tolerance": args.tolerance,
        "rust_fit_seconds": fit_seconds,
        "total_seconds": time.perf_counter() - started,
        "l1": l1,
        "l2": l2,
        "coefficient_metrics": coefficient_metrics,
        "prediction_metrics": prediction_metrics,
        "observed_total": float(np.sum(outcomes)),
        "observed_mean": float(np.mean(outcomes)),
        "jax_fit_metrics": _fit_metrics(outcomes, jax_predictions),
        "rust_fit_metrics": _fit_metrics(outcomes, rust_predictions),
        "rust_active_penalized_coefficients": int(
            np.sum(np.abs(rust_coefficients[1:]) > args.active_threshold)
        ),
        "rust_objective_history_first": (
            None if objective_history.size == 0 else float(objective_history[0])
        ),
        "rust_objective_history_last": (
            None if objective_history.size == 0 else float(objective_history[-1])
        ),
        "rust_convergence_report": convergence,
        "interpretation": (
            "Rust f64 convex refinement of the exact JAX all-level design, restricted "
            "to JAX coefficients above the active threshold. Higher-order L1 heredity "
            "multipliers are frozen at their saved-JAX values."
        ),
    }

    output_stem = f"h2a_prediction_ppml_rust_comparison_cutoff_{args.cutoff_year}"
    summary_path = INTERMEDIATE / f"{output_stem}.json"
    coefficient_path = INTERMEDIATE / f"{output_stem}_coefficients.parquet"
    prediction_path = INTERMEDIATE / f"{output_stem}_predictions.parquet"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    coefficient_rows = [
        {
            "cutoff_year": args.cutoff_year,
            "rust_index": 0,
            "record_type": "global_bias",
            "interaction_order": None,
            "feature_id": None,
            "feature_name": None,
            "categorical_columns": None,
            "categorical_values": None,
            "covariate_scope": None,
            "covariate": None,
            "heredity_multiplier": 1.0,
            "l1_penalty": 0.0,
            "l2_penalty": 0.0,
            "jax_coefficient": float(initial[0]),
            "rust_coefficient": float(rust_coefficients[0]),
        }
    ]
    for feature in active:
        coefficient_rows.append(
            {
                "cutoff_year": args.cutoff_year,
                "rust_index": feature.rust_index,
                "record_type": "weight",
                "interaction_order": feature.order,
                "feature_id": feature.feature_id,
                "feature_name": feature.feature_name,
                "categorical_columns": list(feature.categorical_columns),
                "categorical_values": list(feature.categorical_values),
                "covariate_scope": feature.covariate_scope,
                "covariate": feature.covariate,
                "heredity_multiplier": feature.heredity_multiplier,
                "l1_penalty": float(l1_features[feature.rust_index]),
                "l2_penalty": float(l2_features[feature.rust_index]),
                "jax_coefficient": feature.jax_coefficient,
                "rust_coefficient": float(rust_coefficients[feature.rust_index]),
            }
        )
    coefficient_df = pl.DataFrame(coefficient_rows).with_columns(
        (pl.col("rust_coefficient") - pl.col("jax_coefficient")).alias(
            "coefficient_difference"
        ),
        (pl.col("rust_coefficient") - pl.col("jax_coefficient"))
        .abs()
        .alias("absolute_coefficient_difference"),
    )
    coefficient_df.write_parquet(coefficient_path)

    prediction_df = (
        group_frame.select(
            "county_fips",
            "year",
            pl.col("h2a_target_count").alias("observed_h2a_count"),
        )
        .with_columns(
            pl.Series("jax_predicted_h2a_count", jax_predictions),
            pl.Series("rust_predicted_h2a_count", rust_predictions),
        )
        .with_columns(
            (
                pl.col("rust_predicted_h2a_count") - pl.col("jax_predicted_h2a_count")
            ).alias("prediction_difference"),
            (pl.col("rust_predicted_h2a_count") - pl.col("jax_predicted_h2a_count"))
            .abs()
            .alias("absolute_prediction_difference"),
        )
    )
    prediction_df.write_parquet(prediction_path)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {summary_path}")
    print(f"Wrote {coefficient_path}")
    print(f"Wrote {prediction_path}")


def _run_full(args: argparse.Namespace, core: Any, package_version: str) -> None:
    started = time.perf_counter()
    (
        model,
        bias,
        county_cont_cols,
        patch_cont_cols,
        matrices,
        metadata,
        combos,
    ) = _load_saved_model(args.cutoff_year)
    l1, l2 = _load_penalties(args.cutoff_year)
    multipliers = _heredity_multipliers(matrices, metadata)
    initial, l1_features, l2_features, feature_orders = _full_parameter_vectors(
        bias, matrices, multipliers, l1, l2
    )
    print(f"JAX L1 rates: {l1}")
    print(f"JAX L2 rates: {l2}")
    print(
        f"Complete matched parameter vector: {initial.size:,} coefficients "
        f"({np.count_nonzero(np.abs(initial[1:]) > args.active_threshold):,} "
        "JAX-nonzero penalized coefficients).",
        flush=True,
    )

    (
        train_merged,
        train_groups,
        train_x_county,
        train_x_patch,
        train_group_ids,
        train_exposure,
        train_outcomes,
        train_component_indices,
    ) = _prepare_training_frame(
        args.cutoff_year, model, county_cont_cols, patch_cont_cols
    )
    train_categorical_ids = _categorical_ids(train_merged, metadata, combos)
    jax_train_predictions = _predictions_from_coefficients(
        bias,
        matrices,
        train_categorical_ids,
        train_x_county,
        train_x_patch,
        train_group_ids,
        train_exposure,
        train_outcomes.size,
        dtype=np.dtype(np.float32),
    )

    holdout_year = args.cutoff_year + 1
    (
        holdout_merged,
        holdout_groups,
        holdout_x_county,
        holdout_x_patch,
        holdout_group_ids,
        holdout_exposure,
        holdout_outcomes,
        _,
    ) = _prepare_training_frame(
        holdout_year,
        model,
        county_cont_cols,
        patch_cont_cols,
        start_year=holdout_year,
    )
    holdout_categorical_ids = _categorical_ids(holdout_merged, metadata, combos)
    jax_holdout_predictions = _predictions_from_coefficients(
        bias,
        matrices,
        holdout_categorical_ids,
        holdout_x_county,
        holdout_x_patch,
        holdout_group_ids,
        holdout_exposure,
        holdout_outcomes.size,
        dtype=np.dtype(np.float32),
    )

    native_data = _build_native_data_full(
        core,
        matrices,
        train_categorical_ids,
        combos,
        train_x_county,
        train_x_patch,
        train_group_ids,
        train_component_indices,
        train_exposure,
        train_outcomes,
        max_design_bytes=args.max_design_bytes,
    )
    initial_objective, initial_gradient = _rust_diagnostics(
        core, native_data, initial, l1_features, l2_features
    )
    initial_kkt, initial_violation = _kkt_metrics(
        initial,
        initial_gradient,
        l1_features,
        feature_orders,
        args.active_threshold,
        args.kkt_tolerance,
    )
    print(
        f"JAX start on matched Rust objective: {initial_objective:.9f}; "
        f"max KKT violation: {initial_kkt['max_violation']:.6g}; "
        f"max inactive violation: {initial_kkt['max_inactive_violation']:.6g}.",
        flush=True,
    )

    config = json.dumps(
        {
            "max_iterations": args.max_iterations,
            "tolerance": args.tolerance,
            "stopping_rule": args.stopping_rule,
            "objective_tolerance": args.objective_tolerance,
            "kkt_tolerance": args.kkt_tolerance,
            "initial_lipschitz": 1.0,
            "backtracking_growth": 1.5,
            "max_backtracking_steps": 40,
            "heredity": "none",
            "precision": "f64",
            "gpu_device": None,
        },
        separators=(",", ":"),
    )
    checkpoint_path = (
        INTERMEDIATE
        / f"h2a_prediction_ppml_rust_full_cutoff_{args.cutoff_year}.checkpoint"
    )
    fit_started = time.perf_counter()
    native_fit = core.fit(
        native_data,
        l1_features.astype("<f8", copy=False).tobytes(),
        l2_features.astype("<f8", copy=False).tobytes(),
        config,
        None if args.resume else initial.astype("<f8", copy=False).tobytes(),
        checkpoint_path,
        args.checkpoint_every,
        args.resume,
    )
    fit_seconds = time.perf_counter() - fit_started
    rust_coefficients = np.frombuffer(native_fit.coefficients, dtype="<f8").copy()
    rust_train_predictions = np.frombuffer(native_fit.predictions, dtype="<f8").copy()
    objective_history = np.frombuffer(native_fit.objective_history, dtype="<f8").copy()
    convergence = json.loads(native_fit.report_json)

    rust_objective, rust_gradient = _rust_diagnostics(
        core, native_data, rust_coefficients, l1_features, l2_features
    )
    rust_kkt, rust_violation = _kkt_metrics(
        rust_coefficients,
        rust_gradient,
        l1_features,
        feature_orders,
        args.active_threshold,
        args.kkt_tolerance,
    )
    rust_bias, rust_matrices = _unflatten_coefficients(rust_coefficients, matrices)
    rust_holdout_predictions = _predictions_from_coefficients(
        rust_bias,
        rust_matrices,
        holdout_categorical_ids,
        holdout_x_county,
        holdout_x_patch,
        holdout_group_ids,
        holdout_exposure,
        holdout_outcomes.size,
        dtype=np.dtype(np.float64),
    )

    jax_active = np.abs(initial) > args.active_threshold
    coefficient_metrics_all = _summary_metrics(initial, rust_coefficients)
    coefficient_metrics_jax_active = _summary_metrics(
        initial[jax_active], rust_coefficients[jax_active]
    )
    train_prediction_metrics = _summary_metrics(
        jax_train_predictions, rust_train_predictions
    )
    holdout_prediction_metrics = _summary_metrics(
        jax_holdout_predictions, rust_holdout_predictions
    )
    summary = {
        "cutoff_year": args.cutoff_year,
        "holdout_year": holdout_year,
        "comparison_scope": "complete_jax_design_frozen_heredity",
        "active_threshold": args.active_threshold,
        "rust_package_version": package_version,
        "rust_wheel_sha256": hashlib.sha256(WHEEL.read_bytes()).hexdigest(),
        "rust_backend": "cpu",
        "rust_precision": "f64",
        "jax_precision": "f32",
        "training_groups": int(train_outcomes.size),
        "training_memberships": int(train_exposure.size),
        "holdout_groups": int(holdout_outcomes.size),
        "holdout_memberships": int(holdout_exposure.size),
        "total_coefficients": int(initial.size),
        "jax_nonzero_penalized_coefficients": int(
            np.count_nonzero(np.abs(initial[1:]) > args.active_threshold)
        ),
        "rust_nonzero_penalized_coefficients": int(
            np.count_nonzero(np.abs(rust_coefficients[1:]) > args.active_threshold)
        ),
        "l1": l1,
        "l2": l2,
        "initial_jax_objective": initial_objective,
        "rust_final_objective_from_diagnostics": rust_objective,
        "rust_final_objective_from_fit": float(convergence["objective"]),
        "jax_initial_kkt": initial_kkt,
        "rust_final_kkt": rust_kkt,
        "coefficient_metrics_all": coefficient_metrics_all,
        "coefficient_metrics_jax_active": coefficient_metrics_jax_active,
        "training_prediction_metrics": train_prediction_metrics,
        "holdout_prediction_metrics": holdout_prediction_metrics,
        "training_observed_total": float(np.sum(train_outcomes)),
        "holdout_observed_total": float(np.sum(holdout_outcomes)),
        "jax_training_fit_metrics": _fit_metrics(train_outcomes, jax_train_predictions),
        "rust_training_fit_metrics": _fit_metrics(
            train_outcomes, rust_train_predictions
        ),
        "jax_holdout_fit_metrics": _fit_metrics(
            holdout_outcomes, jax_holdout_predictions
        ),
        "rust_holdout_fit_metrics": _fit_metrics(
            holdout_outcomes, rust_holdout_predictions
        ),
        "rust_convergence_report": convergence,
        "rust_converged": bool(convergence["converged"]),
        "rust_iterations": int(convergence["iterations"]),
        "rust_requested_max_iterations": args.max_iterations,
        "rust_requested_parameter_tolerance": args.tolerance,
        "rust_requested_objective_tolerance": args.objective_tolerance,
        "rust_requested_kkt_tolerance": args.kkt_tolerance,
        "rust_stopping_rule": args.stopping_rule,
        "rust_fit_seconds": fit_seconds,
        "total_seconds": time.perf_counter() - started,
        "rust_objective_history_first": (
            None if objective_history.size == 0 else float(objective_history[0])
        ),
        "rust_objective_history_last": (
            None if objective_history.size == 0 else float(objective_history[-1])
        ),
        "interpretation": (
            "Complete JAX all-level coefficient space, including JAX-zero terms; "
            "JAX final adaptive heredity multipliers frozen as feature-specific L1 "
            "weights; Rust heredity disabled; cutoff+1 held out from fitting."
        ),
    }

    output_stem = f"h2a_prediction_ppml_rust_full_comparison_cutoff_{args.cutoff_year}"
    summary_path = INTERMEDIATE / f"{output_stem}.json"
    coefficient_path = INTERMEDIATE / f"{output_stem}_coefficients.parquet"
    prediction_path = INTERMEDIATE / f"{output_stem}_predictions.parquet"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    saved_coefficients = model.filter(
        pl.col("record_type").is_in(["global_bias", "weight"])
    )
    if saved_coefficients.height != initial.size:
        raise ValueError("Saved JAX coefficient rows do not match the full vector.")
    coefficient_df = (
        saved_coefficients.with_columns(
            pl.Series("rust_index", np.arange(initial.size, dtype=np.int32)),
            pl.Series("heredity_adjusted_l1_penalty", l1_features),
            pl.Series("l2_penalty", l2_features),
            pl.Series("jax_kkt_violation", initial_violation),
            pl.Series("rust_coefficient", rust_coefficients),
            pl.Series("rust_kkt_violation", rust_violation),
        )
        .rename({"coefficient": "jax_coefficient"})
        .with_columns(
            (pl.col("rust_coefficient") - pl.col("jax_coefficient")).alias(
                "coefficient_difference"
            ),
            (pl.col("rust_coefficient") - pl.col("jax_coefficient"))
            .abs()
            .alias("absolute_coefficient_difference"),
        )
    )
    coefficient_df.write_parquet(coefficient_path)

    def prediction_frame(
        sample: str,
        groups: pl.DataFrame,
        outcomes: np.ndarray,
        jax_predictions: np.ndarray,
        rust_predictions: np.ndarray,
    ) -> pl.DataFrame:
        return (
            groups.select("county_fips", "year")
            .with_columns(
                pl.lit(sample).alias("sample"),
                pl.Series("observed_h2a_count", outcomes),
                pl.Series("jax_predicted_h2a_count", jax_predictions),
                pl.Series("rust_predicted_h2a_count", rust_predictions),
            )
            .with_columns(
                (
                    pl.col("rust_predicted_h2a_count")
                    - pl.col("jax_predicted_h2a_count")
                ).alias("prediction_difference"),
                (pl.col("rust_predicted_h2a_count") - pl.col("jax_predicted_h2a_count"))
                .abs()
                .alias("absolute_prediction_difference"),
            )
        )

    prediction_df = pl.concat(
        [
            prediction_frame(
                "training",
                train_groups,
                train_outcomes,
                jax_train_predictions,
                rust_train_predictions,
            ),
            prediction_frame(
                "holdout",
                holdout_groups,
                holdout_outcomes,
                jax_holdout_predictions,
                rust_holdout_predictions,
            ),
        ]
    )
    prediction_df.write_parquet(prediction_path)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {summary_path}")
    print(f"Wrote {coefficient_path}")
    print(f"Wrote {prediction_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff-year", type=int, default=2008)
    parser.add_argument("--scope", choices=("full", "active"), default="full")
    parser.add_argument("--active-threshold", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--stopping-rule",
        choices=("all", "kkt", "objective", "parameter", "parameter_objective"),
        default="kkt",
    )
    parser.add_argument("--objective-tolerance", type=float, default=1e-8)
    parser.add_argument("--kkt-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-design-bytes", type=int, default=5_000_000_000)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 2008 <= args.cutoff_year <= 2025:
        parser.error("--cutoff-year must be between 2008 and 2025")
    if args.active_threshold < 0 or not math.isfinite(args.active_threshold):
        parser.error("--active-threshold must be finite and nonnegative")
    if args.max_design_bytes <= 0:
        parser.error("--max-design-bytes must be positive")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be nonnegative")
    if not WHEEL.exists():
        parser.error(f"Rust estimator wheel not found: {WHEEL}")

    with tempfile.TemporaryDirectory(prefix="h2a-ppml-estimator-") as temp_dir:
        with zipfile.ZipFile(WHEEL) as archive:
            archive.extractall(temp_dir)
        sys.path.insert(0, temp_dir)
        import ppml_estimator  # type: ignore[import-not-found]
        from ppml_estimator import _core  # type: ignore[import-not-found]

        if args.scope == "full":
            _run_full(args, _core, ppml_estimator.__version__)
        else:
            _run_active(args, _core, ppml_estimator.__version__)


if __name__ == "__main__":
    main()
