"""Validate version-4 artifacts and the explicit rejected-method contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

BRANCH_DIR = Path(__file__).resolve().parent
if str(BRANCH_DIR) not in sys.path:
    sys.path.insert(0, str(BRANCH_DIR))

from mcw.design import (
    ALLOW_BOOTSTRAP,
    ALLOW_DIMENSION_REDUCING_LAG_PROFILES,
    ALLOW_NONLINEAR_MODELS,
    ALLOW_POLYNOMIAL_TREATMENT_TERMS,
    ALLOW_RANDOMIZATION_INFERENCE,
    ANALYSIS_YEARS,
    CCV_STATUS,
    DESIGN_VERSION,
    DIRECT_LOG_AEWR_TREATMENTS,
    FULL_HISTORY_REFERENCE_OUTCOME_YEAR,
    INFERENCE_METHODS,
    PRIMITIVE_OUTCOMES,
    TREATMENT_HISTORY_YEARS,
)
from mcw.io import (
    ANALYSIS_PANEL,
    COEFFICIENTS_PATH,
    DIAGNOSTICS_PATH,
    MANIFEST_PATH,
    REGISTRY_PATH,
    RESULTS_PATH,
    SOURCE_PANEL,
    code_hash,
    panel_key_hash,
    sha256_file,
)

VALIDATED_ANALYSIS_YEARS = tuple(range(2012, 2023))
COEFFICIENT_KEY = ("specification_id", "outcome", "term")
DIAGNOSTIC_KEY = ("specification_id", "diagnostic")
RESULT_KEY = (
    "specification_id",
    "outcome",
    "estimand",
    "inference_method",
    "cluster",
)
CAUSAL_METADATA_COLUMNS = (
    "causal_outcome_year",
    "causal_history_year",
    "causal_lag",
    "causal_moderator",
    "causal_moderator_transform",
    "causal_identification",
    "causal_reference_year",
)
RETAINED_ARTIFACT_PATHS = {
    "coefficients_sha256": COEFFICIENTS_PATH,
    "diagnostics_sha256": DIAGNOSTICS_PATH,
    "results_sha256": RESULTS_PATH,
}


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Required version-4 artifact is missing or empty: {path}")


def _require_columns(frame: pl.DataFrame, label: str, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} lack required columns: {missing}")


def _require_unique_key(
    frame: pl.DataFrame, label: str, columns: tuple[str, ...]
) -> None:
    _require_columns(frame, label, columns)
    duplicates = frame.group_by(list(columns)).len().filter(pl.col("len") != 1)
    if duplicates.height:
        raise ValueError(f"{label} are not unique by {columns}.")


def _specification_ids(frame: pl.DataFrame, label: str) -> set[str]:
    _require_columns(frame, label, ("specification_id",))
    if frame.is_empty():
        raise ValueError(f"{label} are empty.")
    if frame["specification_id"].null_count():
        raise ValueError(f"{label} contain null specification IDs.")
    return set(frame["specification_id"].cast(pl.String).to_list())


def _validate_exact_coverage(
    registry: pl.DataFrame,
    results: pl.DataFrame,
    diagnostics: pl.DataFrame,
    coefficients: pl.DataFrame,
    manifest: dict[str, object],
) -> None:
    """Require exact registry membership and unique persistent artifact keys."""

    expected = _specification_ids(registry, "Specification registry")
    if len(expected) != registry.height:
        raise ValueError("Specification registry IDs are not unique.")
    if manifest.get("specification_count") != registry.height:
        raise ValueError("Manifest specification count does not match the registry.")
    for label, frame in (
        ("Results", results),
        ("Diagnostics", diagnostics),
        ("Coefficients", coefficients),
    ):
        observed = _specification_ids(frame, label)
        if observed != expected:
            missing = sorted(expected.difference(observed))
            extra = sorted(observed.difference(expected))
            raise ValueError(
                f"{label} specification coverage is not exact; "
                f"missing={missing}, extra={extra}."
            )
    _require_unique_key(results, "Results", RESULT_KEY)
    _require_unique_key(diagnostics, "Diagnostics", DIAGNOSTIC_KEY)
    _require_unique_key(coefficients, "Coefficients", COEFFICIENT_KEY)


def _validate_retained_artifact_hashes(
    manifest: dict[str, object],
    artifact_paths: dict[str, Path] = RETAINED_ARTIFACT_PATHS,
) -> None:
    """Tie every retained tabular result to the completed report manifest."""

    missing = sorted(set(artifact_paths).difference(manifest))
    if missing:
        raise ValueError(f"Manifest lacks retained-artifact hashes: {missing}")
    mismatches = [
        field
        for field, path in artifact_paths.items()
        if manifest[field] != sha256_file(path)
    ]
    if mismatches:
        raise ValueError(
            "Retained artifacts changed after reporting: "
            f"{', '.join(sorted(mismatches))}."
        )


def _validate_balanced_panel_keys(
    panel: pl.DataFrame, manifest: dict[str, object]
) -> None:
    """Validate full and eligible county panels on the exact 2012--2022 grid."""

    if tuple(ANALYSIS_YEARS) != VALIDATED_ANALYSIS_YEARS:
        raise ValueError(
            "Executable analysis years drifted from the validated 2012--2022 window."
        )
    required = ("county_fips", "year", "mc_baseline_farm_employment")
    _require_columns(panel, "Analysis panel", required)
    if panel.is_empty():
        raise ValueError("Analysis panel is empty.")
    if panel.schema["county_fips"] != pl.String:
        raise TypeError("Analysis-panel county_fips must be stored as a string.")
    if panel.select(
        pl.any_horizontal(
            pl.col(column).is_null() for column in ("county_fips", "year")
        ).any()
    ).item():
        raise ValueError("Analysis panel contains null county-year keys.")
    observed_baseline = panel.filter(
        pl.col("mc_baseline_farm_employment").is_not_null()
    )
    if (
        observed_baseline.height
        and not observed_baseline["mc_baseline_farm_employment"].is_finite().all()
    ):
        raise ValueError("Observed baseline farm employment must be finite.")

    observed_years = tuple(panel["year"].cast(pl.Int64).unique().sort().to_list())
    if observed_years != VALIDATED_ANALYSIS_YEARS:
        raise ValueError(
            "Analysis panel must contain exactly the 2012--2022 outcome years; "
            f"got {observed_years}."
        )
    _require_unique_key(panel, "Analysis panel", ("county_fips", "year"))

    expected_year_count = len(VALIDATED_ANALYSIS_YEARS)
    county_counts = panel.group_by("county_fips").agg(
        pl.len().alias("rows"),
        pl.col("year").n_unique().alias("years"),
    )
    if county_counts.filter(
        (pl.col("rows") != expected_year_count)
        | (pl.col("years") != expected_year_count)
    ).height:
        raise ValueError("Analysis panel is not a complete balanced county-year grid.")

    baseline_by_county = panel.group_by("county_fips").agg(
        pl.col("mc_baseline_farm_employment").null_count().alias("null_rows"),
        pl.col("mc_baseline_farm_employment")
        .drop_nulls()
        .n_unique()
        .alias("observed_values"),
    )
    if baseline_by_county.filter(
        (pl.col("observed_values") > 1)
        | ~pl.col("null_rows").is_in((0, expected_year_count))
    ).height:
        raise ValueError(
            "Baseline farm employment must be a county-invariant repeated value "
            "or uniformly null for an ineligible county."
        )

    eligible = panel.filter(
        pl.col("mc_baseline_farm_employment").is_finite()
        & (pl.col("mc_baseline_farm_employment") > 0)
    ).sort("county_fips", "year")
    if eligible.is_empty():
        raise ValueError("Eligible analysis sample is empty.")
    eligible_counts = eligible.group_by("county_fips").agg(
        pl.len().alias("rows"),
        pl.col("year").n_unique().alias("years"),
    )
    if eligible_counts.filter(
        (pl.col("rows") != expected_year_count)
        | (pl.col("years") != expected_year_count)
    ).height:
        raise ValueError(
            "Positive-baseline-farm-employment sample is not balanced over 2012--2022."
        )
    if manifest.get("row_count") != eligible.height:
        raise ValueError("Manifest row count does not match the eligible panel keys.")
    if manifest.get("sample_hash") != panel_key_hash(eligible):
        raise ValueError("Manifest sample hash does not match the eligible panel keys.")


def _validate_finite_column(
    frame: pl.DataFrame,
    label: str,
    column: str,
    *,
    nonnegative: bool = False,
) -> None:
    _require_columns(frame, label, (column,))
    try:
        values = frame[column].cast(pl.Float64)
    except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
        raise TypeError(f"{label}.{column} must be numeric.") from error
    if values.null_count() or not values.is_finite().all():
        raise ValueError(f"{label}.{column} contains a non-finite value.")
    if nonnegative and (values < 0).any():
        raise ValueError(f"{label}.{column} contains a negative value.")


def _validate_finite_outputs(
    results: pl.DataFrame,
    diagnostics: pl.DataFrame,
    coefficients: pl.DataFrame,
) -> None:
    _validate_finite_column(coefficients, "Coefficients", "estimate")
    _validate_finite_column(results, "Results", "estimate")
    _validate_finite_column(results, "Results", "standard_error", nonnegative=True)
    _validate_finite_column(diagnostics, "Diagnostics", "value")


def _validate_cluster_counts(results: pl.DataFrame) -> None:
    """Require cluster counts only for inference methods that use clusters."""

    _require_columns(
        results,
        "Results",
        ("inference_method", "experimental_ccv", "cluster_count"),
    )
    try:
        cluster_count = pl.col("cluster_count").cast(pl.Float64)
        observed = results.filter(pl.col("cluster_count").is_not_null()).select(
            cluster_count.alias("cluster_count")
        )
    except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
        raise TypeError(
            "Results.cluster_count must be numeric when observed."
        ) from error
    if observed.height and (
        not observed["cluster_count"].is_finite().all()
        or (observed["cluster_count"] < 1).any()
    ):
        raise ValueError("Observed result cluster counts must be finite and positive.")

    requires_clusters = pl.col("inference_method").str.contains("cluster") | pl.col(
        "experimental_ccv"
    )
    if results.filter(requires_clusters & pl.col("cluster_count").is_null()).height:
        raise ValueError(
            "Clustered or experimental inference rows lack a positive cluster count."
        )
    nonclustered_hc3 = pl.col("inference_method").str.starts_with(
        "hc3_full_model_leverage"
    )
    if results.filter(nonclustered_hc3 & pl.col("cluster_count").is_not_null()).height:
        raise ValueError("Nonclustered HC3 result rows must have null cluster counts.")


def _validate_diagnostic_provenance(
    diagnostics: pl.DataFrame, manifest: dict[str, object]
) -> None:
    """Require every fit- and report-stage diagnostic to identify its inputs."""

    expected = {
        "design_version": manifest.get("design_version"),
        "panel_sha256": manifest.get("panel_sha256"),
        "sample_hash": manifest.get("sample_hash"),
    }
    _require_columns(diagnostics, "Diagnostics", tuple(expected))
    for column, value in expected.items():
        if value is None:
            raise ValueError(f"Manifest lacks diagnostic provenance field {column}.")
        observed = diagnostics[column].cast(pl.String)
        if observed.null_count() or not (observed == str(value)).all():
            raise ValueError(
                f"Diagnostics.{column} does not uniformly match the report manifest."
            )


def _validate_observed_mean_metric_labels(
    registry: pl.DataFrame, results: pl.DataFrame
) -> None:
    """Reserve elasticity labels for raw coordinates in log-AEWR units."""

    _require_columns(
        registry,
        "Specification registry",
        ("specification_id", "treatment", "treatment_transform"),
    )
    _require_columns(results, "Results", ("specification_id", "estimand"))
    enriched = results.join(
        registry.select("specification_id", "treatment", "treatment_transform"),
        on="specification_id",
        how="left",
        validate="m:1",
    )
    direct_log_coordinate = pl.col("treatment").is_in(DIRECT_LOG_AEWR_TREATMENTS) & (
        pl.col("treatment_transform") == "continuous_raw"
    )
    elasticity = pl.col("estimand").str.ends_with("_elasticity_at_observed_mean")
    percent_per_unit = pl.col("estimand").str.ends_with(
        "_percent_of_observed_mean_per_treatment_unit"
    )
    normalized = enriched.filter(elasticity | percent_per_unit)
    if normalized.is_empty():
        raise ValueError("Results contain no observed-mean normalized effects.")
    if normalized.filter(elasticity & ~direct_log_coordinate).height:
        raise ValueError(
            "Elasticity labels appear on a transformed or non-log-AEWR treatment."
        )
    if normalized.filter(percent_per_unit & direct_log_coordinate).height:
        raise ValueError(
            "Raw log-AEWR treatments use a per-treatment-unit label instead of "
            "the identified elasticity label."
        )
    expected_ids = set(registry["specification_id"].cast(pl.String).to_list())
    observed_ids = set(
        normalized["specification_id"].cast(pl.String).unique().to_list()
    )
    if observed_ids != expected_ids:
        raise ValueError(
            "Observed-mean normalized effects do not cover every specification."
        )


def _validate_result_targets(results: pl.DataFrame) -> None:
    """Require every reported scalar to persist its population and weights."""

    columns = (
        "target_population",
        "target_observations",
        "target_weight_sum",
        "target_weighting",
    )
    _require_columns(results, "Results", columns)
    if results.select(
        pl.any_horizontal(pl.col(column).is_null() for column in columns).any()
    ).item():
        raise ValueError("Result rows have incomplete target-population metadata.")
    observations = results["target_observations"].cast(pl.Int64, strict=True)
    weights = results["target_weight_sum"].cast(pl.Float64, strict=True)
    if (
        (observations < 1).any()
        or not weights.is_finite().all()
        or (weights <= 0).any()
    ):
        raise ValueError("Result target counts and weight sums must be positive.")
    if results["target_population"].n_unique() != 1:
        raise ValueError("Compact results silently mix target populations.")


def _validate_causal_metadata(
    registry: pl.DataFrame, coefficients: pl.DataFrame
) -> None:
    """Check causal coordinate semantics and common-outcome coefficient keys."""

    coefficient_columns = (
        "specification_id",
        "outcome",
        "term",
        "causal_term",
        *CAUSAL_METADATA_COLUMNS,
    )
    _require_columns(coefficients, "Coefficients", coefficient_columns)
    _require_columns(
        registry,
        "Specification registry",
        ("specification_id", "history", "fixed_effects"),
    )
    if coefficients["causal_term"].null_count():
        raise ValueError("Coefficient causal-term flags contain nulls.")

    expected_outcomes = set(PRIMITIVE_OUTCOMES)
    for specification_id in registry["specification_id"].to_list():
        observed = set(
            coefficients.filter(pl.col("specification_id") == specification_id)[
                "outcome"
            ].to_list()
        )
        if observed != expected_outcomes:
            raise ValueError(
                f"Coefficient outcomes for {specification_id} are not the exact "
                f"common six-outcome set: {sorted(observed)}."
            )
    term_coverage = coefficients.group_by("specification_id", "term").agg(
        pl.len().alias("rows"),
        pl.col("outcome").n_unique().alias("outcomes"),
    )
    outcome_count = len(expected_outcomes)
    if term_coverage.filter(
        (pl.col("rows") != outcome_count) | (pl.col("outcomes") != outcome_count)
    ).height:
        raise ValueError(
            "Every selected coefficient term must occur once for each primitive outcome."
        )

    metadata_consistency = coefficients.group_by("specification_id", "term").agg(
        *(
            pl.col(column).n_unique().alias(column)
            for column in ("causal_term", *CAUSAL_METADATA_COLUMNS)
        )
    )
    if (
        metadata_consistency.select(
            pl.any_horizontal(
                pl.col(column) != 1 for column in metadata_consistency.columns[2:]
            )
        )
        .to_series()
        .any()
    ):
        raise ValueError("Causal metadata vary across outcomes for a common term.")

    causal = coefficients.filter(pl.col("causal_term"))
    nuisance = coefficients.filter(~pl.col("causal_term"))
    if causal.is_empty():
        raise ValueError("Coefficient artifact contains no named causal terms.")
    required_causal = (
        "causal_outcome_year",
        "causal_history_year",
        "causal_lag",
        "causal_identification",
    )
    if causal.select(
        pl.any_horizontal(pl.col(column).is_null() for column in required_causal).any()
    ).item():
        raise ValueError("Named causal terms have incomplete coordinate metadata.")
    if nuisance.select(
        pl.any_horizontal(
            pl.col(column).is_not_null() for column in CAUSAL_METADATA_COLUMNS
        ).any()
    ).item():
        raise ValueError("Nuisance terms carry causal-coordinate metadata.")

    outcome_year = causal["causal_outcome_year"].cast(pl.Int64)
    history_year = causal["causal_history_year"].cast(pl.Int64)
    lag = causal["causal_lag"].cast(pl.Int64)
    if not outcome_year.is_in(ANALYSIS_YEARS).all():
        raise ValueError("Causal outcome-year metadata fall outside 2012--2022.")
    if not history_year.is_in(TREATMENT_HISTORY_YEARS).all():
        raise ValueError("Causal history-year metadata fall outside 2011--2022.")
    if not (lag == outcome_year - history_year).all() or (lag < 0).any():
        raise ValueError(
            "Causal lag metadata do not equal outcome year minus history year."
        )

    enriched = causal.join(
        registry.select("specification_id", "history", "fixed_effects"),
        on="specification_id",
        how="left",
    )
    if enriched.filter(
        (pl.col("history") == "one_lag")
        & ~pl.col("causal_lag").cast(pl.Int64).is_in((0, 1))
    ).height:
        raise ValueError("One-lag specifications contain a longer causal history.")

    pooled = enriched.filter(pl.col("fixed_effects") == "pooled_wmc")
    if pooled.filter(pl.col("causal_identification") != "level_pooled_wmc").height:
        raise ValueError(
            "Pooled WMC causal terms lack pooled-level identification labels."
        )
    if pooled["causal_reference_year"].null_count() != pooled.height:
        raise ValueError(
            "Pooled WMC causal terms must not declare an FE reference year."
        )

    reference = enriched.filter(
        pl.col("causal_identification") == "difference_from_first_outcome_year"
    )
    if reference.height:
        reference_year = reference["causal_reference_year"].cast(pl.Int64, strict=False)
        if (
            reference_year.null_count()
            or not (reference_year == FULL_HISTORY_REFERENCE_OUTCOME_YEAR).all()
        ):
            raise ValueError(
                "County-FE difference coordinates have the wrong reference year."
            )
    levels = enriched.filter(
        pl.col("causal_identification") != "difference_from_first_outcome_year"
    )
    if levels["causal_reference_year"].null_count() != levels.height:
        raise ValueError("Level causal coordinates must not declare a reference year.")

    moderated = causal.filter(pl.col("causal_moderator").is_not_null())
    unmoderated = causal.filter(pl.col("causal_moderator").is_null())
    if moderated.filter(
        pl.col("causal_moderator_transform").is_null()
        | (pl.col("causal_moderator_transform") != "within_aewr_region_deviation")
    ).height:
        raise ValueError("Moderated causal terms lack their centering transformation.")
    if unmoderated["causal_moderator_transform"].null_count() != unmoderated.height:
        raise ValueError("Unmoderated causal terms declare a moderator transformation.")


def main() -> None:
    for path in (
        SOURCE_PANEL,
        ANALYSIS_PANEL,
        REGISTRY_PATH,
        COEFFICIENTS_PATH,
        DIAGNOSTICS_PATH,
        RESULTS_PATH,
        MANIFEST_PATH,
    ):
        _require_file(path)
    manifest = json.loads(MANIFEST_PATH.read_text())
    required_manifest_fields = {
        "ccv_status",
        "code_hash",
        "coefficients_sha256",
        "design_version",
        "diagnostics_sha256",
        "outcomes",
        "panel_sha256",
        "registry_sha256",
        "results_sha256",
        "row_count",
        "sample_hash",
        "source_panel_sha256",
        "specification_count",
    }
    missing_manifest = sorted(required_manifest_fields.difference(manifest))
    if missing_manifest:
        raise ValueError(f"Manifest lacks required fields: {missing_manifest}")
    if manifest["design_version"] != DESIGN_VERSION:
        raise ValueError("Manifest design version does not match executable contract.")
    if manifest["ccv_status"] != CCV_STATUS:
        raise ValueError("Manifest CCV status does not match executable contract.")
    if manifest["panel_sha256"] != sha256_file(ANALYSIS_PANEL):
        raise ValueError("Analysis panel changed after estimation.")
    if manifest["source_panel_sha256"] != sha256_file(SOURCE_PANEL):
        raise ValueError("Shared source panel changed after estimation.")
    if manifest["registry_sha256"] != sha256_file(REGISTRY_PATH):
        raise ValueError("Specification registry changed after estimation.")
    if manifest["code_hash"] != code_hash():
        raise ValueError("MCW implementation changed after estimation.")
    if manifest["outcomes"] != list(PRIMITIVE_OUTCOMES):
        raise ValueError("Manifest outcomes do not match the executable registry.")
    _validate_retained_artifact_hashes(manifest)

    panel = pl.read_parquet(ANALYSIS_PANEL)
    registry = pl.read_csv(REGISTRY_PATH)
    results = pl.read_csv(RESULTS_PATH)
    diagnostics = pl.read_csv(DIAGNOSTICS_PATH)
    coefficients = pl.read_csv(COEFFICIENTS_PATH)
    _validate_exact_coverage(registry, results, diagnostics, coefficients, manifest)
    _validate_balanced_panel_keys(panel, manifest)
    _validate_finite_outputs(results, diagnostics, coefficients)
    _validate_causal_metadata(registry, coefficients)
    _validate_result_targets(results)
    _validate_diagnostic_provenance(diagnostics, manifest)
    _validate_observed_mean_metric_labels(registry, results)

    _require_columns(
        results,
        "Results",
        ("experimental_ccv", "inference_method", "cluster_count"),
    )
    if results["experimental_ccv"].null_count():
        raise ValueError("Results contain null experimental-CCV labels.")
    experimental_name = pl.col("inference_method").str.contains("experimental")
    if results.filter(pl.col("experimental_ccv") != experimental_name).height:
        raise ValueError(
            "Experimental-CCV flags do not match the named inference methods."
        )
    _validate_cluster_counts(results)
    experimental = results.filter(pl.col("experimental_ccv"))
    if experimental.is_empty() or CCV_STATUS != "experimental_not_lean_validated":
        raise ValueError(
            "Experimental continuous CCV labeling is missing or inaccurate."
        )
    observed_methods = set(results["inference_method"].to_list())
    declared_methods = set(INFERENCE_METHODS)
    if observed_methods != declared_methods:
        raise ValueError(
            "Reported inference methods do not match the executable contract; "
            f"missing={sorted(declared_methods - observed_methods)}, "
            f"extra={sorted(observed_methods - declared_methods)}."
        )
    if any(
        (
            ALLOW_RANDOMIZATION_INFERENCE,
            ALLOW_BOOTSTRAP,
            ALLOW_NONLINEAR_MODELS,
            ALLOW_POLYNOMIAL_TREATMENT_TERMS,
            ALLOW_DIMENSION_REDUCING_LAG_PROFILES,
        )
    ):
        raise ValueError("A rejected estimator or functional-form flag is enabled.")
    warnings = diagnostics.filter(pl.col("status") == "warning").height
    print(
        f"Validated {registry.height:,} MCW v4 specifications and "
        f"{results.height:,} estimand rows; retained {warnings:,} diagnostic warnings."
    )


if __name__ == "__main__":
    main()
