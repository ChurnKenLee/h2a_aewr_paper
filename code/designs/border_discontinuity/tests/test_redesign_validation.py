"""Mutation checks for independently reconstructed redesign artifact invariants."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


def validator_functions():
    """Load pure validation functions without running the Marimo application."""
    path = Path(__file__).parents[1] / "06_validate_artifacts.py"
    names = {
        "expected_model_ids", "validate_redesign_geographies",
        "validate_region_disjoint_matching",
        "validate_redesign_panel",
        "validate_redesign_inference",
        "validate_power_counts",
        "validate_power_curve_registry",
        "validate_assignment_audit",
        "validate_redesign_decisions",
    }
    tree = ast.parse(path.read_text())
    definitions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in definitions} == names
    namespace = {"np": np, "pd": pd}
    # Only trusted, named local function definitions are loaded; no app cells run.
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace


def geography_fixture():
    county_ids = ["01001", "01003", "01005"]
    specification = {
        "spec_id": "screened",
        "role": "redesign_candidate",
        "membership_mode": "disjoint",
        "ring_bandwidth": 1,
        "all_member_agricultural_screen": True,
        "strict_retained_locality": True,
    }
    config = {
        "caliper_quantile": .9,
        "geography_specs": [specification],
        "agricultural_county_pair_filter": {
            "farm_employment_start_year": 2008,
            "farm_employment_end_year": 2010,
            "minimum_county_cropland_share": .075,
            "minimum_county_farm_employment_per_100_square_miles": 30,
            "minimum_county_measurement_coverage": .9,
        },
    }
    memberships = pd.DataFrame({
        "spec_id": ["screened"] * 3,
        "county_fips": county_ids,
        "aewr_region_id": ["1", "2", "1"],
        "stratum_id": ["screened__01_02__c01"] * 3,
        "membership_id": county_ids,
        "region_pair_id": ["01_02"] * 3,
        "analysis_weight": [1.] * 3,
        "role": ["redesign_candidate"] * 3,
        "membership_mode": ["disjoint"] * 3,
        "ring_bandwidth": [1] * 3,
        "caliper_threshold": [1.] * 3,
        "border_depth": [0, 0, 1],
    })
    features = pd.DataFrame({
        "county_fips": county_ids,
        "cropland_acres_2007": [1000.] * 3,
        "land_area_acres_2010": [10000.] * 3,
        "mean_farm_employment_2008_2010": [100.] * 3,
        "farm_employment_years_observed": [3] * 3,
        "scaled__feature": [0., 1., .2],
    })
    edges = {("01001", "01003")}
    neighbors = {
        "01001": {"01003", "01005"},
        "01003": {"01001"},
        "01005": {"01001"},
    }
    regions = {"01001": "1", "01003": "2", "01005": "1"}
    return config, memberships, features, edges, neighbors, regions


def assert_rejected(arguments, message):
    try:
        validator_functions()["validate_redesign_geographies"](*arguments)
    except ValueError as error:
        assert message in str(error)
    else:
        raise AssertionError(f"Expected rejection containing {message!r}")


def test_legacy_model_registry_retains_original_audit_without_primary_role():
    config = {
        "benchmark_geography_ids": ["old_h0", "old_h1"],
        "diagnostic_audit": {"primary_spec_id": "old_h1"},
        "geography_specs": [
            {"spec_id": "old_h0", "role": "benchmark"},
            {"spec_id": "old_h1", "role": "legacy_primary_benchmark"},
            {"spec_id": "new_h1", "role": "redesign_candidate"},
        ],
    }
    assert validator_functions()["expected_model_ids"](config) == {
        "old_h0__base", "old_h1__base", "old_h1__equal_stratum_region",
        "old_h1__baseline_trends", "old_h1__preperiod_placebo",
    }


def test_supported_redesign_geography_passes():
    validator_functions()["validate_redesign_geographies"](*geography_fixture())


def test_added_county_must_pass_same_agricultural_screen():
    arguments = geography_fixture()
    arguments[2].loc[2, "cropland_acres_2007"] = 500
    assert_rejected(arguments, "all-member agricultural screen")


def test_added_county_requires_full_preperiod_coverage():
    arguments = geography_fixture()
    arguments[2].loc[2, "farm_employment_years_observed"] = 2
    assert_rejected(arguments, "all-member agricultural screen")


def test_connected_component_does_not_replace_own_region_path():
    arguments = geography_fixture()
    neighbors = arguments[4]
    neighbors["01001"].remove("01005")
    neighbors["01003"].add("01005")
    neighbors["01005"] = {"01003"}
    assert_rejected(arguments, "own-region locality")


def test_caliper_must_use_surviving_opposite_core():
    arguments = geography_fixture()
    arguments[2].loc[2, "scaled__feature"] = 3
    assert_rejected(arguments, "surviving-opposite-core feature caliper")


def test_numeric_weight_sum_alone_is_insufficient():
    arguments = geography_fixture()
    arguments[1].loc[2, "analysis_weight"] = -1
    assert_rejected(arguments, "positive inverse membership counts")


def test_stored_caliper_cannot_be_widened_to_pass_locality():
    arguments = geography_fixture()
    arguments[1]["caliper_threshold"] = 100
    assert_rejected(arguments, "frozen eligible-edge feature distances")


def test_stored_border_depth_must_reproduce_actual_locality():
    arguments = geography_fixture()
    arguments[1].loc[2, "border_depth"] = 0
    assert_rejected(arguments, "stored border depth")


def test_membership_region_must_match_source_county():
    arguments = geography_fixture()
    arguments[1].loc[2, "aewr_region_id"] = "2"
    assert_rejected(arguments, "AEWR regions differ")


def test_membership_role_must_match_registry():
    arguments = geography_fixture()
    arguments[1].loc[2, "role"] = "primary"
    assert_rejected(arguments, "membership role differs")


def matching_fixture(selected_pairs):
    pairs = {
        "01_02": ("1", "2"), "01_03": ("1", "3"),
        "02_04": ("2", "4"), "03_04": ("3", "4"),
    }
    rows, edges = [], set()
    for index, (pair_id, endpoints) in enumerate(pairs.items()):
        counties = (f"01{index * 2 + 1:03d}", f"01{index * 2 + 2:03d}")
        edges.add(counties)
        for county, region in zip(counties, endpoints, strict=True):
            rows.append({
                "spec_id": "source", "county_fips": county,
                "region_pair_id": pair_id, "aewr_region_id": region,
            })
    source = pd.DataFrame(rows)
    selected = source.loc[source["region_pair_id"].isin(selected_pairs)].copy()
    selected["spec_id"] = "matched"
    config = {
        "geography_specs": [
            {"spec_id": "source"},
            {"spec_id": "matched", "region_disjoint_matching": True},
        ],
        "redesign": {"candidate_spec_id": "source"},
    }
    return config, pd.concat([source, selected], ignore_index=True), edges


def test_matching_verifies_declared_lexical_optimum():
    validator_functions()["validate_region_disjoint_matching"](
        *matching_fixture(["01_02", "03_04"])
    )


def test_matching_rejects_fewer_independent_pairs():
    try:
        validator_functions()["validate_region_disjoint_matching"](
            *matching_fixture(["01_02"])
        )
    except ValueError as error:
        assert "maximum cardinality" in str(error)
    else:
        raise AssertionError("A nonmaximum region matching must fail")


def test_matching_rejects_reused_regions():
    try:
        validator_functions()["validate_region_disjoint_matching"](
            *matching_fixture(["01_02", "01_03"])
        )
    except ValueError as error:
        assert "reuse" in str(error)
    else:
        raise AssertionError("Pairs sharing region 1 must fail")


def test_matching_rejects_wrong_tie_break():
    try:
        validator_functions()["validate_region_disjoint_matching"](
            *matching_fixture(["01_03", "02_04"])
        )
    except ValueError as error:
        assert "lexical tie rule" in str(error)
    else:
        raise AssertionError("A different matching tie-break must fail")


def inference_fixture():
    row = {
        "model_id": "candidate", "dyadic_raw_variance": 1.,
        "dyadic_standard_error": 1., "dyadic_p_value": .5,
        "dyadic_confidence_low": -2., "dyadic_confidence_high": 2.,
        "dyadic_status": "small_node_sensitivity", "crv3_status": "ok",
        "wcb_status": "ok", "wcb_p_value": .5,
        "wcb_confidence_low": -2., "wcb_confidence_high": 2.,
        "wcb_repetitions": 9999, "wcb_seed": 201103, "wcb_method": "WCR11-FNW",
    }
    for prefix in ("crv1", "two_way", "crv3_jackknife"):
        row.update({f"{prefix}_standard_error": 1., f"{prefix}_p_value": .5,
            f"{prefix}_confidence_low": -2., f"{prefix}_confidence_high": 2.})
    bootstrap = {"repetitions": 9999, "seed": 201103, "method": "WCR11-FNW"}
    return pd.DataFrame([row]), bootstrap


def test_explicit_unbounded_wcb_remains_valid_evidence():
    frame, bootstrap = inference_fixture()
    frame["wcb_status"] = "unbounded"
    frame["wcb_confidence_low"] = -np.inf
    frame["wcb_confidence_high"] = np.inf
    validator_functions()["validate_redesign_inference"](frame, bootstrap)


def test_nonpositive_dyadic_variance_is_retained_as_unavailable():
    frame, bootstrap = inference_fixture()
    frame["dyadic_status"] = "nonpositive_variance_unavailable"
    frame["dyadic_raw_variance"] = -.2
    frame[["dyadic_standard_error", "dyadic_p_value", "dyadic_confidence_low", "dyadic_confidence_high"]] = np.nan
    validator_functions()["validate_redesign_inference"](frame, bootstrap)


def test_nonpositive_dyadic_variance_cannot_receive_fabricated_interval():
    frame, bootstrap = inference_fixture()
    frame["dyadic_status"] = "nonpositive_variance_unavailable"
    frame["dyadic_raw_variance"] = -.2
    try:
        validator_functions()["validate_redesign_inference"](frame, bootstrap)
    except ValueError as error:
        assert "manufacture an interval" in str(error)
    else:
        raise AssertionError("A negative variance cannot be repaired silently")


def test_bootstrap_failure_can_remain_explicit():
    frame, bootstrap = inference_fixture()
    frame["wcb_status"] = "failed: synthetic singularity"
    frame[["wcb_p_value", "wcb_confidence_low", "wcb_confidence_high"]] = np.nan
    validator_functions()["validate_redesign_inference"](frame, bootstrap)


def test_invalid_bootstrap_p_value_is_rejected():
    frame, bootstrap = inference_fixture()
    frame["wcb_p_value"] = 1.1
    try:
        validator_functions()["validate_redesign_inference"](frame, bootstrap)
    except ValueError as error:
        assert "between zero and one" in str(error)
    else:
        raise AssertionError("An invalid bootstrap p-value must fail")


def panel_fixture():
    config = {
        "outcome": "outcome", "treatment": "wage", "baseline_control_columns": [],
        "feature_start_year": 2008, "feature_end_year": 2010,
        "policy_start_year": 2011, "policy_end_year": 2022,
        "geography_specs": [{"spec_id": "screened", "membership_mode": "disjoint"}],
    }
    panel = pd.DataFrame({
        "county_fips": ["01001"] * 4 + ["01003"] * 4,
        "aewr_region_id": ["1"] * 4 + ["2"] * 4,
        "year": [2008, 2009, 2010, 2011] * 2,
        "outcome": [1., 2., 3., 10., 2., 4., 6., 20.],
        "wage": [10., 11., 12., 20., 12., 13., 14., 30.],
        "border_baseline_farm_employment": [100.] * 4 + [200.] * 4,
        "stratum__screened": ["same_stratum"] * 8,
    })
    panel["year_centered"] = panel["year"] - 2011
    baseline = panel.loc[panel["year"].le(2010)].groupby("county_fips")[["outcome", "wage"]].mean()
    for column in ("outcome", "wage"):
        z = (baseline[column] - baseline[column].mean()) / baseline[column].std()
        panel[f"baseline_{column}_2008_2010_z"] = panel["county_fips"].map(z)
    features = pd.DataFrame({"county_fips": ["01001", "01003"],
        "mean_farm_employment_2008_2010": [100., 200.]})
    memberships = pd.DataFrame({"county_fips": ["01001", "01003"],
        "spec_id": ["screened"] * 2, "stratum_id": ["same_stratum"] * 2})
    return config, panel, features, memberships


def test_policy_outcomes_cannot_change_baseline_history_validation():
    arguments = panel_fixture()
    arguments[1].loc[arguments[1]["year"].eq(2011), "outcome"] = 1000.
    validator_functions()["validate_redesign_panel"](*arguments)


def test_changed_preperiod_outcomes_invalidate_stale_history():
    arguments = panel_fixture()
    arguments[1].loc[0, "outcome"] = 1000.
    try:
        validator_functions()["validate_redesign_panel"](*arguments)
    except ValueError as error:
        assert "predetermined history" in str(error)
    else:
        raise AssertionError("Stale baseline history must fail")


def test_stale_preemployment_weight_is_rejected():
    arguments = panel_fixture()
    arguments[1].loc[0, "border_baseline_farm_employment"] = 200.
    try:
        validator_functions()["validate_redesign_panel"](*arguments)
    except ValueError as error:
        assert "baseline producer" in str(error)
    else:
        raise AssertionError("A changed baseline weight must fail")


def power_count_fixture(invalid=0):
    n, rejected = 999, 50
    rate = rejected / n
    z = 1.959963984540054
    center = (rate + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    frame = pd.DataFrame([{
        "model_id": "candidate", "dgp_id": "shared_region", "scale_information": "preperiod_only",
        "inference_method": "pair_CRV1", "effect": 0., "repetitions": n,
        "valid_repetitions": n - invalid, "invalid_repetitions": invalid,
        "rejections": rejected, "rejection_rate": rate,
        "rejection_rate_valid_only": rejected / (n - invalid),
        "rejection_rate_mc_se": np.sqrt(rate * (1 - rate) / n),
        "mc_confidence_low": center - half, "mc_confidence_high": center + half,
        "size_gate_pass": invalid == 0,
    }])
    return frame, {"repetitions": n}, {"size_maximum": .075}


def test_invalid_variance_draws_remain_in_power_denominator():
    validator_functions()["validate_power_counts"](*power_count_fixture(invalid=1))


def test_power_gate_cannot_ignore_invalid_draws():
    arguments = power_count_fixture(invalid=1)
    arguments[0]["size_gate_pass"] = True
    try:
        validator_functions()["validate_power_counts"](*arguments)
    except ValueError as error:
        assert "size gate" in str(error)
    else:
        raise AssertionError("Invalid simulated variances must fail the size gate")


def test_power_cannot_report_valid_only_rate_as_unconditional():
    arguments = power_count_fixture(invalid=100)
    arguments[0]["rejection_rate"] = arguments[0]["rejection_rate_valid_only"]
    try:
        validator_functions()["validate_power_counts"](*arguments)
    except ValueError as error:
        assert "rejection counts" in str(error)
    else:
        raise AssertionError("Dropping invalid draws cannot increase reported power")


def test_assignment_audit_cannot_relabel_2009_as_statutory():
    config, panel, _, _ = panel_fixture()
    panel.loc[panel["year"].eq(2009), "wage"] = np.nan
    assignment = panel.groupby(["aewr_region_id", "year"])["wage"].agg(
        minimum="min", maximum="max", n_values="nunique"
    ).reset_index()
    assignment["treatment_range"] = assignment["maximum"] - assignment["minimum"]
    assignment["measurement_status"] = np.where(
        assignment["year"].eq(2009),
        "statutory_AEWR_unavailable_nonstatutory_mean_quarantined", "recorded_statutory_AEWR",
    )
    validator_functions()["validate_assignment_audit"](config, assignment, panel)
    assignment["measurement_status"] = "recorded_statutory_AEWR"
    try:
        validator_functions()["validate_assignment_audit"](config, assignment, panel)
    except ValueError as error:
        assert "2009 nonstatutory wage warning" in str(error)
    else:
        raise AssertionError("Known 2009 wage contamination must remain visible")


def power_registry_fixture():
    key = ("candidate", "shared_region", "preperiod_only", "pair_CV3J")
    columns = ["model_id", "dgp_id", "scale_information", "inference_method", "effect"]
    return pd.DataFrame([(*key, effect) for effect in (-.01, 0., .01)], columns=columns), {key}, {-.01, 0., .01}


def test_roundoff_cannot_create_duplicate_power_alternatives():
    arguments = list(power_registry_fixture())
    extra = arguments[0].iloc[[-1]].copy()
    extra["effect"] = np.nextafter(.01, 1.)
    arguments[0] = pd.concat([arguments[0], extra], ignore_index=True)
    try:
        validator_functions()["validate_power_curve_registry"](*arguments)
    except ValueError as error:
        assert "repeat" in str(error)
    else:
        raise AssertionError("Roundoff variants cannot silently duplicate an intended effect")


def test_power_registry_retains_both_signed_alternatives():
    arguments = list(power_registry_fixture())
    validator_functions()["validate_power_curve_registry"](*arguments)
    arguments[0] = arguments[0].iloc[1:]
    try:
        validator_functions()["validate_power_curve_registry"](*arguments)
    except ValueError as error:
        assert "signed alternatives" in str(error)
    else:
        raise AssertionError("An omitted negative alternative must fail")


def decision_fixture():
    config = {"redesign": {
        "models": [{"model_id": "candidate"}],
        "precision_assumptions": {"primary_threshold_beta": .01, "target_power": .8},
    }}
    dgps = ("shared_region_serial", "shared_region_and_pair_serial", "spatial_region_serial")
    methods = ("pair_CV3J", "shared_endpoint_dyadic")
    power, mde, decisions = [], [], []
    for method in methods:
        for dgp in dgps:
            for sign in (-1, 1):
                common = {"model_id": "candidate", "inference_method": method, "dgp_id": dgp, "scale_information": "preperiod_only"}
                power.append({**common, "effect": sign * .01, "rejection_rate": .9, "size_gate_pass": True})
                mde.append({**common, "effect_sign": sign, "credible_grid_mde": .009})
        decisions.append({
            "model_id": "candidate", "inference_method": method, "central_effect": .01,
            "minimum_signed_central_power": .9, "all_shared_dgp_size_gates": True,
            "maximum_size_valid_mde": .009, "confidence_halfwidth": .008,
            "interval_precision_pass": True, "conditional_precision_pass": True,
            "identification_gate_pass": False, "overall_requirement_met": False,
        })
    estimates = pd.DataFrame([{
        "model_id": "candidate", "crv3_jackknife_confidence_low": -.008,
        "crv3_jackknife_confidence_high": .008, "dyadic_confidence_low": -.008,
        "dyadic_confidence_high": .008,
    }])
    tables = {"estimates": estimates, "power": pd.DataFrame(power), "mde": pd.DataFrame(mde), "decision": pd.DataFrame(decisions)}
    return config, tables, {"identification_status": "not_established_with_current_local_data"}


def test_conditional_precision_can_pass_without_identification():
    validator_functions()["validate_redesign_decisions"](*decision_fixture())


def test_overall_gate_cannot_bypass_identification_assessment():
    arguments = decision_fixture()
    arguments[1]["decision"]["overall_requirement_met"] = True
    try:
        validator_functions()["validate_redesign_decisions"](*arguments)
    except ValueError as error:
        assert "separate identification assessment" in str(error)
    else:
        raise AssertionError("Precise conditional estimates do not establish identification")


def test_decision_cannot_contradict_recorded_identification_status():
    arguments = decision_fixture()
    arguments[1]["decision"]["identification_gate_pass"] = True
    arguments[1]["decision"]["overall_requirement_met"] = True
    try:
        validator_functions()["validate_redesign_decisions"](*arguments)
    except ValueError as error:
        assert "unresolved assessment" in str(error)
    else:
        raise AssertionError("The decision cannot relabel an unresolved identifying assumption")


def test_precision_decision_cannot_select_favorable_sign():
    arguments = decision_fixture()
    arguments[1]["power"].loc[0, "rejection_rate"] = .1
    arguments[1]["decision"].loc[0, "minimum_signed_central_power"] = .1
    try:
        validator_functions()["validate_redesign_decisions"](*arguments)
    except ValueError as error:
        assert "declared conjunction" in str(error)
    else:
        raise AssertionError("Low power under one signed alternative must fail precision")


def test_decision_cannot_skip_an_unavailable_size_valid_mde():
    arguments = decision_fixture()
    arguments[1]["mde"].loc[0, "credible_grid_mde"] = np.nan
    try:
        validator_functions()["validate_redesign_decisions"](*arguments)
    except ValueError as error:
        assert "size-valid MDE" in str(error)
    else:
        raise AssertionError("An unavailable signed MDE cannot be dropped from the gate")


def test_decision_cannot_omit_shared_dependence_scenario():
    arguments = decision_fixture()
    arguments[1]["power"] = arguments[1]["power"].iloc[1:]
    try:
        validator_functions()["validate_redesign_decisions"](*arguments)
    except ValueError as error:
        assert "every shared-dependence DGP" in str(error)
    else:
        raise AssertionError("Central precision must retain all declared shared-dependence scenarios")
