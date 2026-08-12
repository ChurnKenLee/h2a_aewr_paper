"""Fit, diagnose, and report the version-4 MCW registry."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from .design import (
    CCV_STATUS,
    CLUSTER_DEFINITIONS,
    CURRENT_EFFECT_REPORTING_YEARS,
    DESIGN_VERSION,
    DIRECT_LOG_AEWR_TREATMENTS,
    PRIMITIVE_OUTCOMES,
    Specification,
)
from .estimands import (
    CoefficientLayout,
    CommonCoefficientMatrix,
    NamedGradient,
    RowGradient,
    RowVector,
    TargetPopulation,
    fixed_observed_mean_elasticity,
    fixed_observed_mean_percent_per_treatment_unit,
    hours_per_position_derivative,
    per_baseline_worker_effect,
    positions_per_application_derivative,
)
from .fwl import CommonOLSFit, fit_common_ols
from .inference import (
    batch_linear_gradient_cross_outcome_inference,
    experimental_scalar_ccv_hc3,
    residualized_contrast_direction,
)
from .io import (
    ANALYSIS_PANEL,
    COEFFICIENTS_PATH,
    DIAGNOSTICS_PATH,
    MANIFEST_PATH,
    REGISTRY_PATH,
    RESULTS_PATH,
    SOURCE_PANEL,
    atomic_write_frame,
    atomic_write_json,
    code_hash,
    environment_record,
    panel_key_hash,
    sha256_file,
    sha256_json,
)
from .model import ModelMatrices, build_model_matrices, causal_moderator_values

FIT_CACHE = ANALYSIS_PANEL.parent.parent / "intermediate" / "mcw_v4_fits"
RESIDUAL_VARIANCE_SHARE_MINIMUM = 1e-10
CURRENT_EFFECT_START_YEAR = min(CURRENT_EFFECT_REPORTING_YEARS)
CURRENT_EFFECT_END_YEAR = max(CURRENT_EFFECT_REPORTING_YEARS)
RETAINED_ARTIFACT_PATHS = {
    "coefficients_sha256": COEFFICIENTS_PATH,
    "diagnostics_sha256": DIAGNOSTICS_PATH,
    "results_sha256": RESULTS_PATH,
}


def _load_registry() -> pl.DataFrame:
    if not REGISTRY_PATH.is_file():
        raise FileNotFoundError(f"Run 02_build_registry.py first: {REGISTRY_PATH}")
    registry = pl.read_csv(REGISTRY_PATH)
    if (
        registry.is_empty()
        or registry["specification_id"].n_unique() != registry.height
    ):
        raise ValueError("Specification registry is empty or has duplicate IDs.")
    return registry


def _specification(row: dict[str, object]) -> Specification:
    fields = {
        key: row[key]
        for key in (
            "specification_id",
            "stage",
            "treatment",
            "history",
            "fixed_effects",
            "moderator_set",
            "cluster",
            "treatment_transform",
            "interpretation_status",
        )
    }
    spec = Specification(**fields)  # type: ignore[arg-type]
    spec.validate()
    return spec


def _fit_path(specification_id: str) -> Path:
    return FIT_CACHE / f"{specification_id}.npz"


def _manifest_path(specification_id: str) -> Path:
    return FIT_CACHE / f"{specification_id}.json"


def _cache_hash(
    spec: Specification,
    panel_hash: str,
    sample_hash: str,
    selected_names: tuple[str, ...],
) -> str:
    return sha256_json(
        {
            "specification": asdict(spec),
            "panel_sha256": panel_hash,
            "sample_hash": sample_hash,
            "code_hash": code_hash(),
            "selected_names": selected_names,
        }
    )


def _report_manifest(sample: pl.DataFrame, registry: pl.DataFrame) -> dict[str, object]:
    """Load an estimation manifest only when every report input is current."""

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Run 03_estimate.py first: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text())
    expected = {
        "design_version": DESIGN_VERSION,
        "registry_sha256": sha256_file(REGISTRY_PATH),
        "panel_sha256": sha256_file(ANALYSIS_PANEL),
        "source_panel_sha256": sha256_file(SOURCE_PANEL),
        "sample_hash": panel_key_hash(sample),
        "code_hash": code_hash(),
        "specification_count": registry.height,
        "row_count": sample.height,
        "outcomes": list(PRIMITIVE_OUTCOMES),
        "ccv_status": CCV_STATUS,
    }
    mismatches = sorted(
        key for key, value in expected.items() if manifest.get(key) != value
    )
    if mismatches:
        raise ValueError(
            "Estimation manifest is stale for report inputs: "
            f"{', '.join(mismatches)}. Re-run 03_estimate.py."
        )
    return manifest


def _sample_frame(panel: pl.DataFrame) -> pl.DataFrame:
    eligible = pl.col("mc_baseline_farm_employment").is_finite() & (
        pl.col("mc_baseline_farm_employment") > 0
    )
    sample = panel.filter(eligible)
    if sample.is_empty():
        raise ValueError("No positive-preperiod-farm-employment counties.")
    counts = sample.group_by("county_fips").agg(
        pl.col("year").n_unique().alias("years")
    )
    maximum = counts["years"].max()
    if maximum is None or counts.filter(pl.col("years") != maximum).height:
        raise ValueError(
            "The eligible v4 analysis panel must be balanced by county-year."
        )
    return sample.sort("county_fips", "year")


def _coefficient_rows(
    spec: Specification,
    fit: CommonOLSFit,
    causal_metadata: tuple[dict[str, object], ...],
) -> list[dict[str, object]]:
    rows = []
    for outcome_index, outcome in enumerate(fit.outcome_names):
        for coefficient_index, term in enumerate(fit.design_names):
            metadata = (
                causal_metadata[coefficient_index]
                if coefficient_index < fit.causal_count
                else {}
            )
            rows.append(
                {
                    "specification_id": spec.specification_id,
                    "outcome": outcome,
                    "term": term,
                    "estimate": fit.coefficient[coefficient_index, outcome_index],
                    "causal_term": coefficient_index < fit.causal_count,
                    "causal_outcome_year": metadata.get("outcome_year"),
                    "causal_history_year": metadata.get("history_year"),
                    "causal_lag": metadata.get("lag"),
                    "causal_moderator": metadata.get("moderator"),
                    "causal_moderator_transform": metadata.get("moderator_transform"),
                    "causal_identification": metadata.get("identification"),
                    "causal_reference_year": metadata.get("reference_year"),
                }
            )
    return rows


def _diagnostic_rows(
    spec: Specification,
    matrices: ModelMatrices,
    fit: CommonOLSFit,
    panel_hash: str,
    sample_hash: str,
) -> list[dict[str, object]]:
    raw_causal = matrices.causal
    within_causal = matrices.projector.within(raw_causal)
    raw_centered = raw_causal - np.mean(raw_causal, axis=0)
    denominator = float(np.sum(np.square(raw_centered)))
    residualized_share = float(np.sum(np.square(within_causal)) / denominator)
    cluster_sizes = (
        pl.DataFrame({"cluster": matrices.cluster}).group_by("cluster").len()
    )
    any_application_index = fit.outcome_names.index("any_application")
    lpm_fitted = fit.fitted[:, any_application_index]
    below_zero_share = float(np.mean(lpm_fitted < 0.0))
    above_one_share = float(np.mean(lpm_fitted > 1.0))
    common = {
        "specification_id": spec.specification_id,
        "design_version": DESIGN_VERSION,
        "panel_sha256": panel_hash,
        "sample_hash": sample_hash,
    }
    rows = [
        {
            **common,
            "diagnostic": "model_rank",
            "value": float(fit.model_rank),
            "status": "pass",
            "detail": "full rank including fixed effects",
        },
        {
            **common,
            "diagnostic": "condition_number",
            "value": fit.condition_number,
            "status": "warning" if fit.condition_number > 1e10 else "pass",
            "detail": "scaled within-design Gram",
        },
        {
            **common,
            "diagnostic": "solve_relative_residual",
            "value": fit.solve_relative_residual,
            "status": "warning" if fit.solve_relative_residual > 1e-9 else "pass",
            "detail": "norm(X' residual) / norm(X' outcome)",
        },
        {
            **common,
            "diagnostic": "residualized_treatment_variance_share",
            "value": residualized_share,
            "status": (
                "warning"
                if residualized_share < RESIDUAL_VARIANCE_SHARE_MINIMUM
                else "pass"
            ),
            "detail": "norm(M_FE D)^2 / norm(D - mean(D))^2",
        },
        {
            **common,
            "diagnostic": "maximum_full_model_leverage",
            "value": float(np.max(fit.leverage)),
            "status": "warning" if np.max(fit.leverage) > 0.5 else "pass",
            "detail": "includes absorbed fixed effects and selected design",
        },
        {
            **common,
            "diagnostic": "cluster_count",
            "value": float(cluster_sizes.height),
            "status": "pass",
            "detail": f"minimum cluster rows={cluster_sizes['len'].min()}",
        },
        {
            **common,
            "diagnostic": "dropped_nuisance_columns",
            "value": float(len(fit.dropped_nuisance_names)),
            "status": "warning" if fit.dropped_nuisance_names else "pass",
            "detail": "|".join(fit.dropped_nuisance_names),
        },
        {
            **common,
            "diagnostic": "lpm_fitted_below_zero_share",
            "value": below_zero_share,
            "status": "warning" if below_zero_share > 0 else "pass",
            "detail": f"minimum fitted value={np.min(lpm_fitted):.8g}",
        },
        {
            **common,
            "diagnostic": "lpm_fitted_above_one_share",
            "value": above_one_share,
            "status": "warning" if above_one_share > 0 else "pass",
            "detail": f"maximum fitted value={np.max(lpm_fitted):.8g}",
        },
    ]
    return rows


def fit_registry() -> None:
    """Fit every selected registry row and retain provenance-checked arrays."""

    panel = pl.read_parquet(ANALYSIS_PANEL)
    sample = _sample_frame(panel)
    registry = _load_registry()
    panel_hash = sha256_file(ANALYSIS_PANEL)
    sample_hash = panel_key_hash(sample)
    force = os.getenv("MC_SPEC_FORCE", "0") == "1"
    FIT_CACHE.mkdir(parents=True, exist_ok=True)
    coefficient_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    for row in registry.iter_rows(named=True):
        spec = _specification(row)
        matrices = build_model_matrices(
            sample,
            spec,
            CLUSTER_DEFINITIONS[spec.cluster],
        )
        fit = fit_common_ols(
            matrices.projector,
            matrices.causal,
            matrices.nuisance,
            matrices.outcomes,
            matrices.causal_names,
            matrices.nuisance_names,
            matrices.outcome_names,
        )
        digest = _cache_hash(spec, panel_hash, sample_hash, fit.design_names)
        fit_path = _fit_path(spec.specification_id)
        manifest_path = _manifest_path(spec.specification_id)
        if not force and fit_path.is_file() and manifest_path.is_file():
            previous = json.loads(manifest_path.read_text())
            if previous.get("cache_hash") == digest:
                print(
                    f"Compatible cache exists; recomputing summary: {spec.specification_id}"
                )
        np.savez_compressed(
            fit_path,
            coefficient=fit.coefficient,
            bread=fit.bread,
            residual=fit.residual,
            leverage=fit.leverage,
            within_design=fit.within_design,
            cluster=np.asarray(matrices.cluster, dtype=str),
            design_names=np.asarray(fit.design_names, dtype=str),
            outcome_names=np.asarray(fit.outcome_names, dtype=str),
            causal_count=np.asarray([fit.causal_count]),
            model_rank=np.asarray([fit.model_rank]),
            causal_metadata_json=np.asarray(
                [json.dumps(matrices.causal_metadata, sort_keys=True)], dtype=str
            ),
        )
        atomic_write_json(
            {
                "cache_hash": digest,
                "specification": asdict(spec),
                "panel_sha256": panel_hash,
                "source_panel_sha256": sha256_file(SOURCE_PANEL),
                "sample_hash": sample_hash,
                "code_hash": code_hash(),
                "environment": environment_record(),
                "row_count": matrices.row_count,
                "model_rank": fit.model_rank,
                "residual_df": fit.residual_df,
                "condition_number": fit.condition_number,
                "solve_relative_residual": fit.solve_relative_residual,
                "design_names": fit.design_names,
                "causal_metadata": matrices.causal_metadata,
                "dropped_nuisance_names": fit.dropped_nuisance_names,
            },
            manifest_path,
        )
        coefficient_rows.extend(_coefficient_rows(spec, fit, matrices.causal_metadata))
        diagnostic_rows.extend(
            _diagnostic_rows(spec, matrices, fit, panel_hash, sample_hash)
        )
        print(
            f"Fit {spec.specification_id}: N={matrices.row_count:,}, "
            f"K={fit.model_rank:,}, max(h)={fit.leverage.max():.4f}."
        )

    atomic_write_frame(
        pl.DataFrame(coefficient_rows, infer_schema_length=None), COEFFICIENTS_PATH
    )
    atomic_write_frame(
        pl.DataFrame(diagnostic_rows, infer_schema_length=None), DIAGNOSTICS_PATH
    )
    atomic_write_json(
        {
            "design_version": DESIGN_VERSION,
            "registry_sha256": sha256_file(REGISTRY_PATH),
            "panel_sha256": panel_hash,
            "source_panel_sha256": sha256_file(SOURCE_PANEL),
            "sample_hash": sample_hash,
            "code_hash": code_hash(),
            "environment": environment_record(),
            "specification_count": registry.height,
            "row_count": sample.height,
            "outcomes": list(PRIMITIVE_OUTCOMES),
            "ccv_status": CCV_STATUS,
        },
        MANIFEST_PATH,
    )


def _average_current_contrast(
    frame: pl.DataFrame,
    names: tuple[str, ...],
    causal_count: int,
) -> np.ndarray:
    gradient = _current_row_gradient(frame, names, causal_count).values
    target = np.isin(
        frame["year"].cast(pl.Int32).to_numpy(), CURRENT_EFFECT_REPORTING_YEARS
    )
    if not np.any(target):
        raise ValueError("Current-effect reporting target is empty.")
    return np.mean(gradient[target], axis=0)


def _block_view(
    covariance: np.ndarray, coefficient_count: int, outcomes: int
) -> np.ndarray:
    expected = coefficient_count * outcomes
    if covariance.shape != (expected, expected):
        raise ValueError("Stacked covariance has the wrong coefficient dimensions.")
    # Outcome-major stacking lets this be a non-copying view. Materializing
    # three (J,J,K,K) copies is needlessly expensive in the rich pooled model.
    return covariance.reshape(
        outcomes, coefficient_count, outcomes, coefficient_count
    ).transpose(0, 2, 1, 3)


def _row_ids(frame: pl.DataFrame) -> tuple[str, ...]:
    return tuple(
        frame.select(
            pl.concat_str(
                [pl.col("county_fips"), pl.col("year").cast(pl.Int32).cast(pl.String)],
                separator="::",
            ).alias("row_id")
        )["row_id"].to_list()
    )


def _current_row_gradient(
    frame: pl.DataFrame,
    design_names: tuple[str, ...],
    causal_count: int,
) -> RowGradient:
    years = frame["year"].cast(pl.Int32).to_numpy()
    gradient = np.zeros((frame.height, len(design_names)))
    for index, name in enumerate(design_names[:causal_count]):
        text = str(name)
        if not text.startswith("effect_y"):
            continue
        base_name, *interaction = text.split("__x__", maxsplit=1)
        outcome_year, history_year = base_name.removeprefix("effect_y").split("_h")
        if (
            outcome_year == history_year
            and int(outcome_year) in CURRENT_EFFECT_REPORTING_YEARS
        ):
            loading = np.ones(frame.height)
            if interaction:
                moderator = interaction[0]
                if moderator not in frame.columns:
                    raise ValueError(
                        f"Current-effect moderator column is missing: {moderator}"
                    )
                loading = causal_moderator_values(frame, moderator)
            gradient[:, index] = (years == int(outcome_year)) * loading
    if not np.any(gradient):
        raise ValueError("No current-coordinate row gradients were constructed.")
    return RowGradient(
        name="current_coordinate",
        row_ids=_row_ids(frame),
        values=gradient,
        coefficient_names=design_names,
    )


def _proportional_outcome_gradient(
    gradient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Factor a ``K x J`` gradient into one direction and outcome scales."""

    derivative = np.asarray(gradient, dtype=np.float64)
    if derivative.ndim != 2:
        raise ValueError("Outcome gradient must be a K x J matrix.")
    active = np.flatnonzero(np.linalg.norm(derivative, axis=0) > np.finfo(float).tiny)
    if active.size == 0:
        return None
    direction = derivative[:, active[0]]
    denominator = float(direction @ direction)
    if denominator <= 0:
        return None
    scales = np.array(
        [float(direction @ derivative[:, index]) / denominator for index in active]
    )
    reconstructed = direction[:, None] * scales[None, :]
    if not np.allclose(
        reconstructed,
        derivative[:, active],
        rtol=1e-10,
        atol=1e-12,
    ):
        return None
    return direction, active, scales


def _fixed_observed_mean_metric(
    spec: Specification,
    *,
    outcome: str,
    row_gradient: RowGradient,
    observed_outcome: RowVector,
    target: TargetPopulation,
    layout: CoefficientLayout,
) -> NamedGradient:
    """Name and scale a mean-normalized derivative from its treatment unit."""

    if (
        spec.treatment in DIRECT_LOG_AEWR_TREATMENTS
        and spec.treatment_transform == "continuous_raw"
    ):
        return fixed_observed_mean_elasticity(
            name=f"{outcome}_elasticity_at_observed_mean",
            outcome=outcome,
            row_gradient=row_gradient,
            observed_outcome=observed_outcome,
            target=target,
            layout=layout,
        )
    return fixed_observed_mean_percent_per_treatment_unit(
        name=f"{outcome}_percent_of_observed_mean_per_treatment_unit",
        outcome=outcome,
        row_gradient=row_gradient,
        observed_outcome=observed_outcome,
        target=target,
        layout=layout,
    )


def _postfit_rows(
    spec: Specification,
    coefficient: np.ndarray,
    names: tuple[str, ...],
    outcomes: tuple[str, ...],
    causal_count: int,
    frame: pl.DataFrame,
    *,
    design: np.ndarray,
    residual: np.ndarray,
    bread: np.ndarray,
    leverage: np.ndarray,
    cluster: np.ndarray,
    model_rank: int,
) -> list[dict[str, object]]:
    layout = CoefficientLayout(names, outcomes)
    coefficients = CommonCoefficientMatrix(coefficient, layout)
    row_ids = _row_ids(frame)
    analysis_year = frame["year"].cast(pl.Int32).to_numpy()
    target = TargetPopulation(
        name=(
            f"eligible_county_years_{CURRENT_EFFECT_START_YEAR}_"
            f"{CURRENT_EFFECT_END_YEAR}_"
            "with_identified_current_effects"
        ),
        row_ids=row_ids,
        include=np.isin(analysis_year, CURRENT_EFFECT_REPORTING_YEARS),
        weights=np.ones(frame.height),
    )
    current = _current_row_gradient(frame, names, causal_count)
    observed = {
        outcome: RowVector(
            name=f"observed_{outcome}",
            row_ids=row_ids,
            values=frame[PRIMITIVE_OUTCOMES[outcome]].cast(pl.Float64).to_numpy(),
        )
        for outcome in outcomes
    }
    baseline_workers = RowVector(
        name="baseline_farm_employment",
        row_ids=row_ids,
        values=frame["mc_baseline_farm_employment"].cast(pl.Float64).to_numpy(),
    )
    gradients = []
    for outcome in ("applications", "requested_positions", "certified_positions"):
        gradients.append(
            per_baseline_worker_effect(
                name=f"{outcome}_per_1000_baseline_farm_workers",
                outcome=outcome,
                row_gradient=current,
                baseline_workers=baseline_workers,
                target=target,
                layout=layout,
                scale=1000.0,
            )
        )
    gradients.append(
        per_baseline_worker_effect(
            name="certified_hours_per_baseline_farm_worker",
            outcome="certified_hours",
            row_gradient=current,
            baseline_workers=baseline_workers,
            target=target,
            layout=layout,
        )
    )
    for outcome in outcomes:
        gradients.append(
            _fixed_observed_mean_metric(
                spec,
                outcome=outcome,
                row_gradient=current,
                observed_outcome=observed[outcome],
                target=target,
                layout=layout,
            )
        )
    gradients.extend(
        (
            positions_per_application_derivative(
                positions_row_gradient=current,
                applications_row_gradient=current,
                observed_positions=observed["certified_positions"],
                observed_applications=observed["applications"],
                target=target,
                layout=layout,
            ),
            hours_per_position_derivative(
                hours_row_gradient=current,
                positions_row_gradient=current,
                observed_hours=observed["certified_hours"],
                observed_positions=observed["certified_positions"],
                target=target,
                layout=layout,
            ),
        )
    )
    sandwiches = batch_linear_gradient_cross_outcome_inference(
        design,
        residual,
        bread,
        leverage,
        cluster,
        np.stack([gradient.values for gradient in gradients]),
        n_parameters=model_rank,
    )
    rows = []
    for gradient, sandwich in zip(gradients, sandwiches, strict=True):
        estimate = gradient.evaluate(coefficients)
        common = {
            "specification_id": spec.specification_id,
            "outcome": "constructed",
            "estimand": gradient.name,
            "estimate": estimate,
            "cluster": spec.cluster,
            "target_population": gradient.target_name,
            "target_observations": target.observations,
            "target_weight_sum": target.weight_sum,
            "target_weighting": "equal_county_year",
        }
        rows.extend(
            (
                {
                    **common,
                    "standard_error": sandwich.hc3_standard_error,
                    "inference_method": "hc3_full_model_leverage_joint_delta",
                    "cluster_count": None,
                    "experimental_ccv": False,
                },
                {
                    **common,
                    "standard_error": sandwich.cr0_standard_error,
                    "inference_method": "cr0_cluster_sandwich_joint_delta",
                    "cluster_count": np.unique(cluster).size,
                    "experimental_ccv": False,
                },
                {
                    **common,
                    "standard_error": sandwich.cr1_standard_error,
                    "inference_method": "cr1_cluster_sandwich_joint_delta",
                    "cluster_count": np.unique(cluster).size,
                    "experimental_ccv": False,
                },
            )
        )
        if (
            sandwich.common_contrast_direction is not None
            and sandwich.outcome_loadings is not None
        ):
            scalar = experimental_scalar_ccv_hc3(
                sandwich.common_contrast_direction,
                residual @ sandwich.outcome_loadings[:, None],
                leverage,
                cluster,
                n_parameters=model_rank,
            )
            rows.extend(
                (
                    {
                        **common,
                        "standard_error": float(
                            np.sqrt(max(scalar.ccv_hc3[0, 0], 0.0))
                        ),
                        "inference_method": (
                            "ccv_hc3_scalar_mixture_experimental_delta"
                        ),
                        "cluster_count": scalar.n_clusters,
                        "experimental_ccv": True,
                    },
                    {
                        **common,
                        "standard_error": float(
                            np.sqrt(max(scalar.ccv_hc3_cr1[0, 0], 0.0))
                        ),
                        "inference_method": (
                            "ccv_hc3_cr1_scalar_mixture_experimental_delta"
                        ),
                        "cluster_count": scalar.n_clusters,
                        "experimental_ccv": True,
                    },
                )
            )
    return rows


def _history_disagreement_diagnostics(
    results: pl.DataFrame,
    registry: pl.DataFrame,
    *,
    panel_hash: str,
    sample_hash: str,
) -> pl.DataFrame:
    """Compare like-for-like full and one-lag current-effect summaries."""

    comparison_columns = (
        "treatment",
        "fixed_effects",
        "moderator_set",
        "cluster",
        "treatment_transform",
    )
    primitive = (
        results.filter(
            (pl.col("outcome") != "constructed")
            & (pl.col("inference_method") == "hc3_full_model_leverage")
        )
        .join(
            registry.select("specification_id", "history", *comparison_columns),
            on="specification_id",
            how="left",
            validate="m:1",
        )
        .select(
            "specification_id",
            "history",
            "outcome",
            "estimate",
            *comparison_columns,
        )
    )
    join_columns = [*comparison_columns, "outcome"]
    full = primitive.filter(pl.col("history") == "full").select(
        *join_columns,
        pl.col("specification_id").alias("full_id"),
        pl.col("estimate").alias("full_estimate"),
    )
    one_lag = primitive.filter(pl.col("history") == "one_lag").select(
        *join_columns,
        pl.col("specification_id").alias("one_lag_id"),
        pl.col("estimate").alias("one_lag_estimate"),
    )
    pairs = full.join(one_lag, on=join_columns, how="inner", validate="1:1")
    rows: list[dict[str, object]] = []
    for pair in pairs.iter_rows(named=True):
        difference = abs(float(pair["full_estimate"]) - float(pair["one_lag_estimate"]))
        outcome = str(pair["outcome"])
        for specification_id, comparator in (
            (str(pair["full_id"]), str(pair["one_lag_id"])),
            (str(pair["one_lag_id"]), str(pair["full_id"])),
        ):
            rows.append(
                {
                    "specification_id": specification_id,
                    "design_version": DESIGN_VERSION,
                    "panel_sha256": panel_hash,
                    "sample_hash": sample_hash,
                    "diagnostic": f"full_one_lag_absolute_difference::{outcome}",
                    "value": difference,
                    "status": "warning",
                    "detail": (
                        f"descriptive comparison with {comparator}; not a validity gate"
                    ),
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={
                "specification_id": pl.String,
                "design_version": pl.String,
                "panel_sha256": pl.String,
                "sample_hash": pl.String,
                "diagnostic": pl.String,
                "value": pl.Float64,
                "status": pl.String,
                "detail": pl.String,
            }
        )
    return pl.DataFrame(rows)


def report_registry() -> None:
    """Report one transparent common estimand under every inference comparator."""

    registry = _load_registry()
    sample = _sample_frame(pl.read_parquet(ANALYSIS_PANEL))
    manifest = _report_manifest(sample, registry)
    diagnostic_provenance = {
        "design_version": DESIGN_VERSION,
        "panel_sha256": str(manifest["panel_sha256"]),
        "sample_hash": str(manifest["sample_hash"]),
    }
    target_rows = np.isin(
        sample["year"].cast(pl.Int32).to_numpy(), CURRENT_EFFECT_REPORTING_YEARS
    )
    target_observations = int(np.count_nonzero(target_rows))
    target_name = (
        f"eligible_county_years_{CURRENT_EFFECT_START_YEAR}_"
        f"{CURRENT_EFFECT_END_YEAR}_with_identified_current_effects"
    )
    rows: list[dict[str, object]] = []
    inference_diagnostics: list[dict[str, object]] = []
    for row in registry.iter_rows(named=True):
        spec = _specification(row)
        fit_path = _fit_path(spec.specification_id)
        fit_manifest_path = _manifest_path(spec.specification_id)
        if not fit_path.is_file() or not fit_manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing fitted arrays for {spec.specification_id}; "
                "re-run 03_estimate.py."
            )
        arrays = np.load(fit_path, allow_pickle=False)
        coefficient = arrays["coefficient"]
        bread = arrays["bread"]
        residual = arrays["residual"]
        leverage = arrays["leverage"]
        design = arrays["within_design"]
        cluster = arrays["cluster"]
        names = tuple(str(value) for value in arrays["design_names"])
        outcomes = tuple(str(value) for value in arrays["outcome_names"])
        fit_manifest = json.loads(fit_manifest_path.read_text())
        expected_cache_hash = _cache_hash(
            spec,
            str(manifest["panel_sha256"]),
            str(manifest["sample_hash"]),
            names,
        )
        if fit_manifest.get("cache_hash") != expected_cache_hash:
            raise ValueError(
                f"Fitted arrays are stale for {spec.specification_id}; "
                "re-run 03_estimate.py."
            )
        causal_count = int(arrays["causal_count"][0])
        model_rank = int(arrays["model_rank"][0])
        contrast = _average_current_contrast(sample, names, causal_count)
        estimate = contrast @ coefficient

        direction = residualized_contrast_direction(design, bread, contrast)
        experimental = experimental_scalar_ccv_hc3(
            direction,
            residual,
            leverage,
            cluster,
            n_parameters=model_rank,
        )
        for outcome_index, outcome in enumerate(outcomes):
            variances = {
                "hc3_full_model_leverage": float(
                    experimental.hc3[outcome_index, outcome_index]
                ),
                "cr0_cluster_sandwich": float(
                    experimental.cr0[outcome_index, outcome_index]
                ),
                "cr1_cluster_sandwich": float(
                    experimental.cr1[outcome_index, outcome_index]
                ),
                "ccv_hc3_scalar_mixture_experimental": float(
                    experimental.ccv_hc3[outcome_index, outcome_index]
                ),
                "ccv_hc3_cr1_scalar_mixture_experimental": float(
                    experimental.ccv_hc3_cr1[outcome_index, outcome_index]
                ),
            }
            for method, variance in variances.items():
                rows.append(
                    {
                        "specification_id": spec.specification_id,
                        "outcome": str(outcome),
                        "estimand": "average_current_coordinate_effect_"
                        f"{CURRENT_EFFECT_START_YEAR}_{CURRENT_EFFECT_END_YEAR}",
                        "estimate": float(estimate[outcome_index]),
                        "standard_error": float(np.sqrt(max(variance, 0.0))),
                        "inference_method": method,
                        "cluster": spec.cluster,
                        "cluster_count": (
                            None
                            if method == "hc3_full_model_leverage"
                            else experimental.n_clusters
                        ),
                        "experimental_ccv": "experimental" in method,
                        "target_population": target_name,
                        "target_observations": target_observations,
                        "target_weight_sum": float(target_observations),
                        "target_weighting": "equal_county_year",
                    }
                )
        rows.extend(
            _postfit_rows(
                spec,
                coefficient,
                names,
                outcomes,
                causal_count,
                sample,
                design=design,
                residual=residual,
                bread=bread,
                leverage=leverage,
                cluster=cluster,
                model_rank=model_rank,
            )
        )
        inference_diagnostics.extend(
            (
                {
                    "specification_id": spec.specification_id,
                    **diagnostic_provenance,
                    "diagnostic": "experimental_ccv_lambda",
                    "value": experimental.lambda_weight,
                    "status": "warning",
                    "detail": CCV_STATUS,
                },
                {
                    "specification_id": spec.specification_id,
                    **diagnostic_provenance,
                    "diagnostic": "experimental_ccv_omega_cv",
                    "value": experimental.omega_cv,
                    "status": "warning",
                    "detail": "descriptive; not a validity gate",
                },
                {
                    "specification_id": spec.specification_id,
                    **diagnostic_provenance,
                    "diagnostic": "experimental_ccv_kappa_cv",
                    "value": experimental.kappa_cv,
                    "status": "warning",
                    "detail": "descriptive heuristic; not a Lean theorem",
                },
            )
        )
    result_frame = pl.DataFrame(rows, infer_schema_length=None)
    atomic_write_frame(result_frame, RESULTS_PATH)
    existing = pl.read_csv(DIAGNOSTICS_PATH)
    disagreement = _history_disagreement_diagnostics(
        result_frame,
        registry,
        panel_hash=diagnostic_provenance["panel_sha256"],
        sample_hash=diagnostic_provenance["sample_hash"],
    )
    atomic_write_frame(
        pl.concat(
            (existing, pl.DataFrame(inference_diagnostics), disagreement),
            how="diagonal_relaxed",
        ),
        DIAGNOSTICS_PATH,
    )
    manifest.update(
        {field: sha256_file(path) for field, path in RETAINED_ARTIFACT_PATHS.items()}
    )
    atomic_write_json(manifest, MANIFEST_PATH)
    print(f"Wrote {len(rows):,} constructed-estimand inference rows.")
