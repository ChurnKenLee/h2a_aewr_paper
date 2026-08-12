"""Focused artifact-contract tests for the version-4 validator."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

BRANCH_ROOT = Path(__file__).resolve().parents[1]
if str(BRANCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BRANCH_ROOT))

SPEC = importlib.util.spec_from_file_location(
    "mcw_v4_validate", BRANCH_ROOT / "05_validate.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load the version-4 validator.")
VALIDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATE)

from mcw.design import PRIMITIVE_OUTCOMES
from mcw.io import panel_key_hash


def _panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "county_fips": [
                county
                for county in ("01001", "01003")
                for _ in VALIDATE.VALIDATED_ANALYSIS_YEARS
            ],
            "year": list(VALIDATE.VALIDATED_ANALYSIS_YEARS) * 2,
            "mc_baseline_farm_employment": [100.0] * 11 + [None] * 11,
        },
        schema={
            "county_fips": pl.String,
            "year": pl.Int32,
            "mc_baseline_farm_employment": pl.Float64,
        },
    )


def _registry() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "specification_id": ["pooled", "county"],
            "history": ["full", "one_lag"],
            "fixed_effects": ["pooled_wmc", "county_year"],
            "treatment": ["aewr_log_level", "aewr_dollar_level"],
            "treatment_transform": ["continuous_raw", "continuous_raw"],
        }
    )


def _coefficients() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    definitions = {
        "pooled": (
            "effect_y2012_h2011__x__mc_baseline_bite_z",
            2012,
            2011,
            1,
            "mc_baseline_bite_z",
            "within_aewr_region_deviation",
            "level_pooled_wmc",
        ),
        "county": (
            "effect_y2012_h2012",
            2012,
            2012,
            0,
            None,
            None,
            "level_after_county_fe",
        ),
    }
    for specification_id, definition in definitions.items():
        term, outcome_year, history_year, lag, moderator, transform, identity = (
            definition
        )
        for outcome in PRIMITIVE_OUTCOMES:
            rows.append(
                {
                    "specification_id": specification_id,
                    "outcome": outcome,
                    "term": term,
                    "estimate": 0.25,
                    "causal_term": True,
                    "causal_outcome_year": outcome_year,
                    "causal_history_year": history_year,
                    "causal_lag": lag,
                    "causal_moderator": moderator,
                    "causal_moderator_transform": transform,
                    "causal_identification": identity,
                    "causal_reference_year": None,
                }
            )
            rows.append(
                {
                    "specification_id": specification_id,
                    "outcome": outcome,
                    "term": "explicit_nuisance",
                    "estimate": -0.1,
                    "causal_term": False,
                    "causal_outcome_year": None,
                    "causal_history_year": None,
                    "causal_lag": None,
                    "causal_moderator": None,
                    "causal_moderator_transform": None,
                    "causal_identification": None,
                    "causal_reference_year": None,
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


def _artifacts() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    results = pl.DataFrame(
        {
            "specification_id": ["pooled", "county"],
            "outcome": ["applications", "applications"],
            "estimand": ["current", "current"],
            "estimate": [0.1, -0.2],
            "standard_error": [0.03, 0.04],
            "inference_method": ["hc3", "hc3"],
            "cluster": ["aewr_region", "aewr_region"],
            "target_population": ["eligible_2013_2022"] * 2,
            "target_observations": [20, 20],
            "target_weight_sum": [20.0, 20.0],
            "target_weighting": ["equal_county_year"] * 2,
        }
    )
    diagnostics = pl.DataFrame(
        {
            "specification_id": ["pooled", "county"],
            "design_version": [VALIDATE.DESIGN_VERSION] * 2,
            "panel_sha256": ["panel-hash"] * 2,
            "sample_hash": ["sample-hash"] * 2,
            "diagnostic": ["rank", "rank"],
            "value": [10.0, 12.0],
        }
    )
    return results, diagnostics, _coefficients()


def _cluster_results() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "inference_method": [
                "hc3_full_model_leverage_joint_delta",
                "cr1_cluster_sandwich",
                "ccv_hc3_scalar_mixture_experimental",
            ],
            "experimental_ccv": [False, False, True],
            "cluster_count": [None, 17, 17],
        }
    )


def _manifest(panel: pl.DataFrame) -> dict[str, object]:
    eligible = panel.filter(pl.col("mc_baseline_farm_employment") > 0)
    return {
        "design_version": VALIDATE.DESIGN_VERSION,
        "panel_sha256": "panel-hash",
        "specification_count": 2,
        "row_count": eligible.height,
        "sample_hash": panel_key_hash(eligible),
    }


class ArtifactValidationTest(unittest.TestCase):
    def test_valid_synthetic_contract_passes(self) -> None:
        panel = _panel()
        registry = _registry()
        results, diagnostics, coefficients = _artifacts()
        manifest = _manifest(panel)

        VALIDATE._validate_exact_coverage(
            registry, results, diagnostics, coefficients, manifest
        )
        VALIDATE._validate_balanced_panel_keys(panel, manifest)
        VALIDATE._validate_finite_outputs(results, diagnostics, coefficients)
        VALIDATE._validate_causal_metadata(registry, coefficients)
        VALIDATE._validate_cluster_counts(_cluster_results())
        VALIDATE._validate_result_targets(results)
        diagnostics = diagnostics.with_columns(
            pl.lit(manifest["sample_hash"]).alias("sample_hash")
        )
        VALIDATE._validate_diagnostic_provenance(diagnostics, manifest)

    def test_registry_coverage_must_be_exact(self) -> None:
        registry = _registry()
        results, diagnostics, coefficients = _artifacts()
        with self.assertRaisesRegex(ValueError, "coverage is not exact"):
            VALIDATE._validate_exact_coverage(
                registry,
                results.filter(pl.col("specification_id") != "county"),
                diagnostics,
                coefficients,
                {"specification_count": 2},
            )

    def test_retained_artifacts_must_match_manifest_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "results.csv"
            artifact.write_text("estimate\n1.0\n")
            paths = {"results_sha256": artifact}
            manifest = {"results_sha256": VALIDATE.sha256_file(artifact)}
            VALIDATE._validate_retained_artifact_hashes(manifest, paths)

            artifact.write_text("estimate\n2.0\n")
            with self.assertRaisesRegex(ValueError, "changed after reporting"):
                VALIDATE._validate_retained_artifact_hashes(manifest, paths)

    def test_balanced_panel_requires_every_2012_2022_key(self) -> None:
        panel = _panel()
        incomplete = panel.filter(
            ~((pl.col("county_fips") == "01003") & (pl.col("year") == 2017))
        )
        with self.assertRaisesRegex(ValueError, "balanced county-year grid"):
            VALIDATE._validate_balanced_panel_keys(incomplete, _manifest(panel))

    def test_nonfinite_estimate_and_negative_standard_error_fail(self) -> None:
        results, diagnostics, coefficients = _artifacts()
        with self.subTest("infinite estimate"):
            bad_coefficients = coefficients.with_columns(
                pl.when(pl.int_range(pl.len()) == 0)
                .then(float("inf"))
                .otherwise(pl.col("estimate"))
                .alias("estimate")
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                VALIDATE._validate_finite_outputs(
                    results, diagnostics, bad_coefficients
                )
        with self.subTest("negative standard error"):
            bad_results = results.with_columns(
                pl.when(pl.int_range(pl.len()) == 0)
                .then(-0.01)
                .otherwise(pl.col("standard_error"))
                .alias("standard_error")
            )
            with self.assertRaisesRegex(ValueError, "negative"):
                VALIDATE._validate_finite_outputs(
                    bad_results, diagnostics, coefficients
                )

    def test_cluster_count_required_only_for_clustered_inference(self) -> None:
        results = _cluster_results()
        VALIDATE._validate_cluster_counts(results)

        with self.subTest("clustered row missing count"):
            missing = results.with_columns(
                pl.when(pl.col("inference_method") == "cr1_cluster_sandwich")
                .then(pl.lit(None, dtype=pl.Int64))
                .otherwise(pl.col("cluster_count"))
                .alias("cluster_count")
            )
            with self.assertRaisesRegex(ValueError, "lack a positive cluster count"):
                VALIDATE._validate_cluster_counts(missing)

        with self.subTest("nonclustered HC3 count must be null"):
            observed_hc3 = results.with_columns(
                pl.when(
                    pl.col("inference_method") == "hc3_full_model_leverage_joint_delta"
                )
                .then(17)
                .otherwise(pl.col("cluster_count"))
                .alias("cluster_count")
            )
            with self.assertRaisesRegex(ValueError, "must have null cluster counts"):
                VALIDATE._validate_cluster_counts(observed_hc3)

    def test_diagnostic_provenance_must_match_manifest(self) -> None:
        panel = _panel()
        _, diagnostics, _ = _artifacts()
        manifest = _manifest(panel)
        diagnostics = diagnostics.with_columns(
            pl.lit(manifest["sample_hash"]).alias("sample_hash")
        )
        VALIDATE._validate_diagnostic_provenance(diagnostics, manifest)
        with self.assertRaisesRegex(ValueError, "does not uniformly match"):
            VALIDATE._validate_diagnostic_provenance(
                diagnostics.with_columns(
                    pl.when(pl.int_range(pl.len()) == 0)
                    .then(pl.lit(None, dtype=pl.String))
                    .otherwise(pl.col("panel_sha256"))
                    .alias("panel_sha256")
                ),
                manifest,
            )

    def test_observed_mean_metric_labels_follow_treatment_units(self) -> None:
        registry = pl.DataFrame(
            {
                "specification_id": ["log", "transformed", "dollar"],
                "treatment": [
                    "aewr_log_level",
                    "aewr_log_level",
                    "aewr_dollar_level",
                ],
                "treatment_transform": [
                    "continuous_raw",
                    "binary_median",
                    "continuous_raw",
                ],
            }
        )
        results = pl.DataFrame(
            {
                "specification_id": ["log", "transformed", "dollar"],
                "estimand": [
                    "applications_elasticity_at_observed_mean",
                    "applications_percent_of_observed_mean_per_treatment_unit",
                    "applications_percent_of_observed_mean_per_treatment_unit",
                ],
            }
        )
        VALIDATE._validate_observed_mean_metric_labels(registry, results)
        mislabeled = results.with_columns(
            pl.when(pl.col("specification_id") == "dollar")
            .then(pl.lit("applications_elasticity_at_observed_mean"))
            .otherwise(pl.col("estimand"))
            .alias("estimand")
        )
        with self.assertRaisesRegex(ValueError, "transformed or non-log-AEWR"):
            VALIDATE._validate_observed_mean_metric_labels(registry, mislabeled)

    def test_result_target_metadata_is_required(self) -> None:
        results, _, _ = _artifacts()
        VALIDATE._validate_result_targets(results)
        with self.assertRaisesRegex(ValueError, "incomplete target-population"):
            VALIDATE._validate_result_targets(
                results.with_columns(
                    pl.when(pl.int_range(pl.len()) == 0)
                    .then(pl.lit(None, dtype=pl.String))
                    .otherwise(pl.col("target_population"))
                    .alias("target_population")
                )
            )

    def test_pooled_causal_metadata_requires_label_and_transform(self) -> None:
        registry = _registry()
        coefficients = _coefficients()
        with self.subTest("wrong pooled identification"):
            wrong_label = coefficients.with_columns(
                pl.when(
                    (pl.col("specification_id") == "pooled") & pl.col("causal_term")
                )
                .then(pl.lit("level_after_county_fe"))
                .otherwise(pl.col("causal_identification"))
                .alias("causal_identification")
            )
            with self.assertRaisesRegex(ValueError, "pooled-level"):
                VALIDATE._validate_causal_metadata(registry, wrong_label)
        with self.subTest("missing moderator transform"):
            missing_transform = coefficients.with_columns(
                pl.when(pl.col("specification_id") == "pooled")
                .then(pl.lit(None, dtype=pl.String))
                .otherwise(pl.col("causal_moderator_transform"))
                .alias("causal_moderator_transform")
            )
            with self.assertRaisesRegex(ValueError, "centering transformation"):
                VALIDATE._validate_causal_metadata(registry, missing_transform)


if __name__ == "__main__":
    unittest.main()
