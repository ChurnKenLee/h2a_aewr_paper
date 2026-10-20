import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    # ruff: noqa: N999
    """Validate AEWR-border ring geography, estimates, and robustness audit."""

    import hashlib
    import json
    import os
    import re
    import sys
    from runpy import run_path

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from h2a.geography import harmonize_county_fips_2010
    from h2a.paths import FIGURES, INTERMEDIATE, PROCESSED, ROOT, TABLES

    DESIGN_DIR = ROOT / "code" / "designs" / "border_discontinuity"
    geography = run_path(DESIGN_DIR / "geography.py")
    build_graph = geography["build_graph"]
    connected_components = geography["connected_components"]
    pair_identifier = geography["pair_identifier"]

    def is_arrow_string(data_type: pa.DataType) -> bool:
        return pa.types.is_string(data_type) or pa.types.is_large_string(data_type)


    def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
        missing = sorted(columns.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


    def require_finite(frame: pd.DataFrame, columns: list[str], name: str) -> None:
        values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains nonfinite values in {', '.join(columns)}")


    def expected_model_ids(config: dict[str, object]) -> set[str]:
        benchmark_ids = set(config["benchmark_geography_ids"])
        specifications = [
            value for value in config["geography_specs"]
            if value["spec_id"] in benchmark_ids
        ]
        if {value["spec_id"] for value in specifications} != benchmark_ids:
            raise ValueError("benchmark geography registry is incomplete")
        base = {f"{value['spec_id']}__base" for value in specifications}
        primary = [
            value for value in specifications
            if value["spec_id"] == config["diagnostic_audit"]["primary_spec_id"]
        ]
        if len(primary) != 1:
            raise ValueError("legacy diagnostic geography must be a unique benchmark")
        return base | {
            f"{value['spec_id']}__{suffix}"
            for value in primary
            for suffix in (
                "equal_stratum_region",
                "baseline_trends",
                "preperiod_placebo",
            )
        }

    def validate_redesign_geographies(
        config, memberships, features, eligible_edges, neighbors, regions
    ) -> None:
        """Reconstruct redesigned locality from surviving edges and source features."""
        specifications = config["geography_specs"]
        registry = {value["spec_id"]: value for value in specifications}
        if len(registry) != len(specifications):
            raise ValueError("geography specification IDs must be unique")
        for column in ("county_fips", "aewr_region_id", "stratum_id", "membership_id"):
            if memberships[column].isna().any() or memberships[column].astype(str).eq("").any():
                raise ValueError(f"membership {column} must be nonempty")
        expected_regions = memberships["county_fips"].map(regions)
        if not np.array_equal(expected_regions.to_numpy(), memberships["aewr_region_id"].to_numpy()):
            raise ValueError("membership AEWR regions differ from the county panel")
        weights = memberships["analysis_weight"].to_numpy(dtype=float)
        counts = memberships.groupby(["spec_id", "county_fips"])["county_fips"].transform("size")
        if not np.isfinite(weights).all() or (weights <= 0).any() or not np.allclose(weights, 1 / counts, atol=1e-12, rtol=0):
            raise ValueError("membership weights must be positive inverse membership counts")
        support = features.set_index("county_fips")
        scaled_columns = sorted(column for column in features if column.startswith("scaled__"))
        if not scaled_columns:
            raise ValueError("locality validation requires scaled baseline features")
        edge_distances = [
            np.linalg.norm(
                support.loc[left, scaled_columns].to_numpy(dtype=float)
                - support.loc[right, scaled_columns].to_numpy(dtype=float)
            )
            for left, right in eligible_edges
        ]
        expected_caliper = np.quantile(edge_distances, config["caliper_quantile"])
        if not np.isfinite(expected_caliper) or not np.allclose(
            memberships["caliper_threshold"], expected_caliper, atol=1e-12, rtol=1e-12
        ):
            raise ValueError("geography caliper differs from frozen eligible-edge feature distances")
        agricultural_filter = config["agricultural_county_pair_filter"]
        expected_years = (
            agricultural_filter["farm_employment_end_year"]
            - agricultural_filter["farm_employment_start_year"] + 1
        )
        for spec_id, group in memberships.groupby("spec_id", sort=True):
            specification = registry[spec_id]
            for column in ("role", "membership_mode", "ring_bandwidth"):
                if set(group[column]) != {specification[column]}:
                    raise ValueError(f"{spec_id} membership {column} differs from registry")
            if not specification.get("all_member_agricultural_screen", False):
                continue
            source = support.loc[group["county_fips"]]
            crop = source["cropland_acres_2007"].to_numpy(dtype=float)
            area = source["land_area_acres_2010"].to_numpy(dtype=float)
            employment = source["mean_farm_employment_2008_2010"].to_numpy(dtype=float)
            coverage = source["farm_employment_years_observed"].to_numpy(dtype=float) / expected_years
            if (
                not np.isfinite(np.column_stack((crop, area, employment, coverage))).all()
                or (area <= 0).any()
                or (crop / area < agricultural_filter["minimum_county_cropland_share"]).any()
                or (employment * 64000 / area < agricultural_filter["minimum_county_farm_employment_per_100_square_miles"]).any()
                or (coverage < agricultural_filter["minimum_county_measurement_coverage"]).any()
            ):
                raise ValueError(f"{spec_id} contains a county failing the all-member agricultural screen")
            if not specification.get("strict_retained_locality", False):
                continue
            for stratum_id, component in group.groupby("stratum_id", sort=True):
                counties = set(component["county_fips"])
                surviving_edges = [
                    edge for edge in eligible_edges
                    if edge[0] in counties and edge[1] in counties
                ]
                if not surviving_edges:
                    raise ValueError(f"{stratum_id} has no surviving qualifying edge")
                core = {county for edge in surviving_edges for county in edge}
                for region, side in component.groupby("aewr_region_id", sort=True):
                    side_counties = set(side["county_fips"])
                    distances = {county: 0 for county in core & side_counties}
                    frontier = set(distances)
                    while frontier:
                        next_frontier = {
                            neighbor for county in frontier
                            for neighbor in neighbors[county] & side_counties
                            if neighbor not in distances
                        }
                        next_depth = 1 + max(distances[county] for county in frontier)
                        distances.update({county: next_depth for county in next_frontier})
                        frontier = next_frontier
                    if set(distances) != side_counties or max(distances.values()) > specification["ring_bandwidth"]:
                        raise ValueError(f"{stratum_id} violates surviving-edge own-region locality")
                    if not np.array_equal(
                        side["county_fips"].map(distances).to_numpy(),
                        side["border_depth"].to_numpy(),
                    ):
                        raise ValueError(f"{stratum_id} stored border depth differs from surviving local support")
                    opposite_core = sorted(county for county in core if regions[county] != region)
                    if not opposite_core:
                        raise ValueError(f"{stratum_id} has no surviving opposite core")
                    opposite_matrix = support.loc[opposite_core, scaled_columns].to_numpy(dtype=float)
                    for row in side.itertuples(index=False):
                        if row.county_fips in core:
                            continue
                        county_vector = support.loc[row.county_fips, scaled_columns].to_numpy(dtype=float)
                        distance = np.linalg.norm(opposite_matrix - county_vector, axis=1).min()
                        if distance > row.caliper_threshold + 1e-12:
                            raise ValueError(f"{stratum_id} violates the surviving-opposite-core feature caliper")

    def validate_region_disjoint_matching(config, memberships, eligible_edges) -> None:
        """Check the declared maximum matching with independent subset recursion."""
        matched_specs = [
            value for value in config["geography_specs"]
            if value.get("region_disjoint_matching", False)
        ]
        if not matched_specs:
            return
        source = memberships.loc[
            memberships["spec_id"].eq(config["redesign"]["candidate_spec_id"])
        ]
        pair_endpoints = {}
        pair_scores = {}
        for pair_id, group in source.groupby("region_pair_id", sort=True):
            endpoints = tuple(sorted(set(group["aewr_region_id"]), key=int))
            if len(endpoints) != 2:
                raise ValueError("matching source pair must have two endpoint regions")
            pair_endpoints[pair_id] = endpoints
            counties = set(group["county_fips"])
            pair_scores[pair_id] = sum(
                left in counties and right in counties
                for left, right in eligible_edges
            )
            if pair_scores[pair_id] < 1:
                raise ValueError("matching source pair has no qualifying core edge")
        if not pair_endpoints:
            raise ValueError("region matching requires nonempty strict source geography")
        all_regions = tuple(sorted({region for pair in pair_endpoints.values() for region in pair}, key=int))
        memo = {}

        def optimum(remaining):
            if not remaining:
                return ()
            if remaining in memo:
                return memo[remaining]
            first = remaining[0]
            available = set(remaining)
            choices = [optimum(tuple(region for region in remaining if region != first))]
            for pair_id, endpoints in pair_endpoints.items():
                if first in endpoints and set(endpoints) <= available:
                    tail = optimum(tuple(region for region in remaining if region not in endpoints))
                    choices.append(tuple(sorted((*tail, pair_id))))
            answer = min(choices, key=lambda pairs: (
                -len(pairs), -sum(pair_scores[pair] for pair in pairs), pairs
            ))
            memo[remaining] = answer
            return answer

        expected_pairs = optimum(all_regions)
        for specification in matched_specs:
            matched = memberships.loc[memberships["spec_id"].eq(specification["spec_id"])]
            actual_pairs = tuple(sorted(matched["region_pair_id"].unique()))
            endpoints = [
                region for pair_id in actual_pairs
                for region in pair_endpoints.get(pair_id, ())
            ]
            if len(endpoints) != 2 * len(actual_pairs) or len(set(endpoints)) != len(endpoints):
                raise ValueError("matched pairs reuse or lack an endpoint region")
            if actual_pairs != expected_pairs:
                raise ValueError("region matching violates maximum cardinality, core-edge score or lexical tie rule")
            expected_members = source.loc[source["region_pair_id"].isin(actual_pairs)]
            key_columns = ["region_pair_id", "county_fips", "aewr_region_id"]
            expected_keys = set(expected_members[key_columns].itertuples(index=False, name=None))
            actual_keys = set(matched[key_columns].itertuples(index=False, name=None))
            if actual_keys != expected_keys:
                raise ValueError("matched geography must preserve every member of selected strict-source pairs")

    def validate_redesign_panel(config, panel, features, memberships) -> None:
        """Check predetermined weights/history and stored geographic assignments."""
        expected_employment = panel["county_fips"].map(
            features.set_index("county_fips")["mean_farm_employment_2008_2010"]
        )
        if "border_baseline_farm_employment" not in panel or not np.allclose(
            panel["border_baseline_farm_employment"], expected_employment,
            atol=1e-12, rtol=1e-12, equal_nan=True,
        ):
            raise ValueError("redesign employment weights differ from their baseline producer")
        baseline_columns = [*config["baseline_control_columns"], config["outcome"], config["treatment"]]
        baseline = panel.loc[
            panel["year"].between(config["feature_start_year"], config["feature_end_year"])
        ].groupby("county_fips")[baseline_columns].mean()
        for column in baseline_columns:
            scale = baseline[column].std(ddof=1)
            expected = panel["county_fips"].map((baseline[column] - baseline[column].mean()) / scale)
            stored_column = f"baseline_{column}_2008_2010_z"
            if not np.isfinite(scale) or scale <= 0 or stored_column not in panel or not np.allclose(
                panel[stored_column], expected, atol=1e-12, rtol=1e-12, equal_nan=True
            ):
                raise ValueError(f"predetermined history {stored_column} differs from preperiod data")
        if not np.array_equal(panel["year_centered"], panel["year"] - config["policy_start_year"]):
            raise ValueError("redesign trend timing differs from policy start year")
        for specification in config["geography_specs"]:
            if specification["membership_mode"] != "disjoint":
                continue
            selected = memberships.loc[memberships["spec_id"].eq(specification["spec_id"])]
            expected = panel["county_fips"].map(selected.set_index("county_fips")["stratum_id"])
            stored_column = f"stratum__{specification['spec_id']}"
            if stored_column not in panel or not np.array_equal(
                panel[stored_column].fillna("").to_numpy(), expected.fillna("").to_numpy()
            ):
                raise ValueError(f"panel assignment {stored_column} differs from memberships")
        policy = panel.loc[panel["year"].between(config["policy_start_year"], config["policy_end_year"])]
        if (policy.groupby(["aewr_region_id", "year"])[config["treatment"]].nunique() > 1).any():
            raise ValueError("policy-period direct AEWR treatment is not constant within region-year")

    def validate_redesign_inference(frame, bootstrap) -> None:
        """Accept explicitly unavailable inference while rejecting invented precision."""
        for row in frame.itertuples(index=False):
            for prefix in ("crv1", "two_way"):
                values = [getattr(row, f"{prefix}_{suffix}") for suffix in (
                    "standard_error", "p_value", "confidence_low", "confidence_high"
                )]
                if not np.isfinite(values).all() or values[0] <= 0 or not 0 <= values[1] <= 1 or values[2] > values[3]:
                    raise ValueError(f"{row.model_id} has invalid {prefix} inference")
            dyadic = np.asarray([row.dyadic_standard_error, row.dyadic_p_value,
                row.dyadic_confidence_low, row.dyadic_confidence_high], dtype=float)
            if not np.isfinite(row.dyadic_raw_variance):
                raise ValueError("dyadic raw variance must remain observable")
            if row.dyadic_status == "small_node_sensitivity":
                if row.dyadic_raw_variance <= 0 or not np.isfinite(dyadic).all() or dyadic[0] <= 0 or not 0 <= dyadic[1] <= 1 or dyadic[2] > dyadic[3]:
                    raise ValueError("positive dyadic inference does not match its status")
            elif row.dyadic_status == "nonpositive_variance_unavailable":
                if row.dyadic_raw_variance > 0 or not np.isnan(dyadic).all():
                    raise ValueError("nonpositive dyadic variance must not manufacture an interval")
            else:
                raise ValueError("unknown dyadic inference status")
            jackknife = np.asarray([row.crv3_jackknife_standard_error,
                row.crv3_jackknife_p_value, row.crv3_jackknife_confidence_low,
                row.crv3_jackknife_confidence_high], dtype=float)
            if row.crv3_status == "ok":
                if not np.isfinite(jackknife).all() or jackknife[0] <= 0 or not 0 <= jackknife[1] <= 1 or jackknife[2] > jackknife[3]:
                    raise ValueError("CV3J inference does not match its successful status")
            elif row.crv3_status == "unidentified_delete_pair":
                if not np.isnan(jackknife).all():
                    raise ValueError("unidentified CV3J must retain missing inference")
            else:
                raise ValueError("unknown CV3J inference status")
            endpoints = np.asarray([row.wcb_confidence_low, row.wcb_confidence_high], dtype=float)
            if row.wcb_status in ("ok", "reproduced_v5"):
                if not np.isfinite(endpoints).all() or endpoints[0] > endpoints[1]:
                    raise ValueError("successful WCB interval must have finite ordered endpoints")
            elif row.wcb_status == "unbounded":
                if np.isnan(endpoints).any() or not np.isinf(endpoints).any() or endpoints[0] > endpoints[1]:
                    raise ValueError("unbounded WCB status must preserve ordered infinite endpoints")
            elif row.wcb_status == "inversion_failed":
                if not np.isnan(endpoints).any():
                    raise ValueError("failed WCB inversion requires missing endpoints")
            elif str(row.wcb_status).startswith("failed: "):
                if not np.isnan(endpoints).all() or not np.isnan(row.wcb_p_value):
                    raise ValueError("failed WCB must not contain fabricated inference")
            else:
                raise ValueError("unknown WCB inference status")
            if not str(row.wcb_status).startswith("failed: ") and (
                not np.isfinite(row.wcb_p_value) or not 0 <= row.wcb_p_value <= 1
            ):
                raise ValueError("computed WCB p-value must lie between zero and one")
            if (row.wcb_repetitions != bootstrap["repetitions"] or row.wcb_seed != bootstrap["seed"] or row.wcb_method != bootstrap["method"]):
                raise ValueError("redesign WCB settings differ from the declared seed/draw/method contract")

    def validate_redesign_results(config, tables, legacy, memberships) -> None:
        """Cross-check new tables and statuses without relying on their summaries."""
        estimates = tables["estimates"]
        declared = {value["model_id"]: value for value in config["redesign"]["models"]}
        declared.update({f"{spec_id}__base": {
            "model_id": f"{spec_id}__base", "spec_id": spec_id,
            "weight_spec": "equal_county", "control_spec": "none", "role": "v5_benchmark",
        } for spec_id in config["benchmark_geography_ids"]})
        ids = set(declared)
        if set(estimates["model_id"]) != ids or estimates["model_id"].duplicated().any():
            raise ValueError("redesign estimate registry is incomplete or duplicated")
        for column in ("design_id", "outcome", "treatment"):
            if set(estimates[column]) != {config[column]}:
                raise ValueError(f"redesign {column} metadata differs from registry")
        for row in estimates.itertuples(index=False):
            if any(getattr(row, column) != declared[row.model_id][column] for column in (
                "spec_id", "weight_spec", "control_spec", "role"
            )):
                raise ValueError("redesign model semantics differ from frozen candidate registry")
        numerical = ["estimate", "fwl_coefficient_error", "fwl_score_sum",
            "residual_outcome_ss", "treatment_partial_ss", "treatment_raw_sd",
            "treatment_within_sd", "effective_pair_leverage_count", "maximum_pair_leverage",
            "region_graph_incidence_rank", "n_observations", "n_counties", "n_strata", "n_pairs", "n_regions"]
        require_finite(estimates, numerical, "redesign estimates")
        if (estimates["fwl_coefficient_error"].abs() > 1e-8).any() or (estimates["fwl_score_sum"].abs() > 1e-8).any():
            raise ValueError("redesign weighted FWL identity fails")
        if (estimates["treatment_partial_ss"] <= 0).any() or (estimates["n_pairs"] < 2).any() or (estimates["n_regions"] < 2).any():
            raise ValueError("redesign treatment or policy-unit support is insufficient for a fitted result")
        validate_redesign_inference(estimates, config["bootstrap"])
        legacy_by_id = legacy.set_index("model_id")
        for spec_id in config["benchmark_geography_ids"]:
            model_id = f"{spec_id}__base"
            new = estimates.set_index("model_id").loc[model_id]
            old = legacy_by_id.loc[model_id]
            for column in ("estimate", "crv1_standard_error", "wcb_p_value", "wcb_confidence_low", "wcb_confidence_high", "n_observations"):
                if not np.isclose(new[column], old[column], atol=1e-8, rtol=1e-8):
                    raise ValueError(f"redesign benchmark {model_id}/{column} differs from v5")
        audit = tables["sample_audit"]
        expected_audit = {(model_id, period) for model_id in ids for period in ("policy", "placebo", "calibration_preperiod")}
        if set(audit[["model_id", "period"]].itertuples(index=False, name=None)) != expected_audit or audit.duplicated(["model_id", "period"]).any():
            raise ValueError("redesign sample audit does not cover both periods for all models")
        if not np.array_equal(audit["joined_rows"] - audit["complete_case_rows"], audit["dropped_missing_rows"]) or not np.array_equal(audit["complete_case_rows"] - audit["estimation_rows"], audit["dropped_singleton_rows"]) or (audit[["dropped_missing_rows", "dropped_singleton_rows"]] < 0).any().any():
            raise ValueError("redesign sample-loss counts do not reconcile")
        placebo = tables["placebo"]
        if set(placebo["model_id"]) != ids or placebo["model_id"].duplicated().any() or set(placebo["period"]) != {"2008_and_2010_already_exposed_two_year_diagnostic"} or not placebo["zero_effect_not_guaranteed"].eq(True).all() or set(placebo["excluded_wage_year"]) != {2009}:
            raise ValueError("earlier-period diagnostic must preserve its already-exposed interpretation")
        require_finite(placebo, ["estimate", "crv1_standard_error", "crv1_p_value", "approximate_pair_mde"], "earlier-period diagnostic")
        for period, fitted in (("policy", estimates), ("placebo", placebo)):
            a = audit.loc[audit["period"].eq(period)].set_index("model_id")
            f = fitted.set_index("model_id").loc[a.index]
            for audit_column, fit_column in (("estimation_rows", "n_observations"), ("n_counties", "n_counties"), ("n_strata", "n_strata"), ("n_pairs", "n_pairs"), ("n_regions", "n_regions")):
                if not np.array_equal(a[audit_column], f[fit_column]):
                    raise ValueError("redesign sample counts disagree with fitted results")
        support = tables["support"]
        if set(support["model_id"]) != ids or support.duplicated(["model_id", "region_pair_id"]).any():
            raise ValueError("redesign pair support has an incomplete key")
        require_finite(support, ["partial_treatment_leverage", "residual_outcome_ss", "coefficient_contribution", "pair_score"], "redesign pair support")
        if (support[["partial_treatment_leverage", "residual_outcome_ss"]] < 0).any().any():
            raise ValueError("redesign leverage/noise components cannot be negative")
        result_by_id = estimates.set_index("model_id")
        for model_id, group in support.groupby("model_id", sort=True):
            row = result_by_id.loc[model_id]
            if len(group) != row["n_pairs"] or not np.isclose(group["partial_treatment_leverage"].sum(), 1) or not np.isclose(group["coefficient_contribution"].sum(), row["estimate"], atol=1e-8) or not np.isclose(group["pair_score"].sum(), 0, atol=1e-8) or not np.isclose(group["residual_outcome_ss"].sum(), row["residual_outcome_ss"], atol=1e-8):
                raise ValueError("redesign pair decomposition does not reconstruct the fitted model")
            if not np.isclose(1 / np.square(group["partial_treatment_leverage"]).sum(), row["effective_pair_leverage_count"]) or not np.isclose(group["partial_treatment_leverage"].max(), row["maximum_pair_leverage"]):
                raise ValueError("redesign effective pair count does not match concentration")
            endpoints = sorted({endpoint for pair_id in group["region_pair_id"] for endpoint in pair_id.split("_")})
            incidence = np.asarray([[float(endpoint in pair_id.split("_")) for pair_id in group["region_pair_id"]] for endpoint in endpoints])
            scores = group["pair_score"].to_numpy()
            raw = np.square(incidence @ scores).sum() - np.square(scores).sum()
            signed = incidence.copy()
            for index, pair_id in enumerate(group["region_pair_id"]):
                signed[endpoints.index(pair_id.split("_")[1]), index] = -1
            if not np.isclose(raw, row["dyadic_raw_variance"], atol=1e-12, rtol=1e-8) or len(endpoints) != row["n_regions"] or np.linalg.matrix_rank(signed) != row["region_graph_incidence_rank"]:
                raise ValueError("shared-endpoint covariance or policy graph rank does not reconstruct")
        fwl = tables["fwl"]
        if set(fwl["model_id"]) != ids or fwl["model_id"].duplicated().any():
            raise ValueError("redesign FWL/gap audit is incomplete")
        for row in fwl.itertuples(index=False):
            if abs(row.fwl_coefficient_error) > 1e-8 or abs(row.fwl_score_sum) > 1e-8:
                raise ValueError("redesign FWL audit failed")
            if row.gap_status == "exact_balanced_side_gap":
                if not np.isfinite([row.gap_coefficient, row.gap_error]).all() or abs(row.gap_error) > 1e-8 or not np.isclose(row.gap_coefficient, result_by_id.loc[row.model_id, "estimate"], atol=1e-8):
                    raise ValueError("equivalent side-gap regression does not reproduce county regression")
            elif row.gap_status in ("controls_not_collapsed", "unbalanced_not_naively_collapsed"):
                if not pd.isna(row.gap_error) or not pd.isna(row.gap_coefficient):
                    raise ValueError("uncollapsed gap model must not invent an equality result")
            else:
                raise ValueError("unknown side-gap comparison status")
        balance = tables["balance"]
        variables = {f"baseline_{column}_2008_2010_z" for column in (config["outcome"], *config["baseline_control_columns"])}
        if set(balance[["model_id", "variable"]].itertuples(index=False, name=None)) != {(model_id, variable) for model_id in ids for variable in variables} or balance.duplicated(["model_id", "variable"]).any():
            raise ValueError("redesign baseline balance registry is incomplete")
        require_finite(balance, ["preperiod_wage_association", "pair_standard_error", "pair_p_value", "weighted_mean", "weighted_sd", "n_counties"], "redesign balance")
        if set(balance["wage_baseline_years"]) != {"2008;2010"} or set(balance["excluded_wage_year"]) != {2009}:
            raise ValueError("new wage balance must exclude the 2009 nonstatutory measure")
        deletion = tables["region_deletion"]
        if set(deletion["model_id"]) != ids or deletion.duplicated(["model_id", "excluded_region_id"]).any():
            raise ValueError("redesign region-deletion registry is incomplete")
        for model_id, group in deletion.groupby("model_id", sort=True):
            row = result_by_id.loc[model_id]
            specification = memberships.loc[memberships["spec_id"].eq(row["spec_id"])]
            region_pairs = {region: set(part["region_pair_id"]) for region, part in specification.groupby("aewr_region_id")}
            if set(group["excluded_region_id"]) != set(region_pairs):
                raise ValueError("region deletion does not cover every policy endpoint")
            for deleted in group.itertuples(index=False):
                expected = region_pairs[deleted.excluded_region_id]
                if set(deleted.excluded_pairs.split(";")) != expected or deleted.n_excluded_pairs != len(expected) or deleted.n_retained_pairs != row["n_pairs"] - len(expected) or not np.isclose(deleted.full_estimate, row["estimate"]):
                    raise ValueError("region deletion does not remove every incident border pair")
                if deleted.status == "ok":
                    if not np.isfinite(deleted.estimate):
                        raise ValueError("identified region deletion requires finite estimate")
                elif str(deleted.status).startswith("unidentified: "):
                    if not pd.isna(deleted.estimate):
                        raise ValueError("unidentified region deletion must retain a missing estimate")
                else:
                    raise ValueError("unknown region-deletion status")
        common_ids = {"screened_h1__preemployment_weight", "screened_h1__baseline_history", "screened_h1__region_disjoint"}
        for model_id in common_ids:
            row = result_by_id.loc[model_id]
            values = row[["base_on_comparable_county_years", "comparable_county_years", "change_from_comparable_base"]].to_numpy(dtype=float)
            if not np.isfinite(values).all() or values[1] <= 0 or values[1] > row["n_observations"] or not np.isclose(row["estimate"] - values[0], values[2]):
                raise ValueError("redesign common-sample comparison does not reconcile")
        comparables = tables["comparable_samples"]
        comparison_pairs = [
            ("ring_h1_disjoint__base", "screened_h1__base"),
            ("ring_h0_disjoint__base", "screened_h1__base"),
            ("screened_h1__base", "screened_h1__preemployment_weight"),
            ("screened_h1__base", "screened_h1__baseline_history"),
            ("screened_h1__base", "screened_h1__region_disjoint"),
        ]
        expected_comparisons = {("__versus__".join(pair), model_id) for pair in comparison_pairs for model_id in pair}
        if set(comparables[["comparison_id", "model_id"]].itertuples(index=False, name=None)) != expected_comparisons or comparables.duplicated(["comparison_id", "model_id"]).any():
            raise ValueError("common-sample precision comparisons are incomplete")
        require_finite(comparables, ["estimate", "crv1_standard_error", "two_way_standard_error", "residual_outcome_ss", "treatment_partial_ss", "n_observations", "n_counties", "fwl_coefficient_error"], "common-sample precision comparisons")
        for _, group in comparables.groupby("comparison_id"):
            if group["n_observations"].nunique() != 1 or group["n_counties"].nunique() != 1 or (group["fwl_coefficient_error"].abs() > 1e-8).any():
                raise ValueError("precision comparison does not use equivalent observed samples")
        noise = tables["noise_diagnosis"]
        if noise["county_fips"].duplicated().any() or not noise["county_fips"].str.fullmatch(r"\d{5}").all():
            raise ValueError("noise diagnosis must retain unique string county identifiers")
        require_finite(noise, ["fit_rss", "fit_treatment_ss", "fit_coefficient", "rss_share", "treatment_ss_share"], "noise diagnosis")
        old = result_by_id.loc[f"{config['diagnostic_audit']['primary_spec_id']}__base"]
        if not np.isclose(noise["fit_rss"].sum(), old["residual_outcome_ss"]) or not np.isclose(noise["fit_coefficient"].sum(), old["estimate"]) or not np.allclose(noise["rss_share"], noise["fit_rss"] / noise["fit_rss"].sum()) or not np.allclose(noise["treatment_ss_share"], noise["fit_treatment_ss"] / noise["fit_treatment_ss"].sum()):
            raise ValueError("baseline noise diagnosis does not reconstruct its model")

    def validate_power_counts(power, settings, assumptions) -> None:
        """Recompute rejection rates, Monte Carlo intervals and null-size gates."""
        count_columns = ["repetitions", "valid_repetitions", "invalid_repetitions", "rejections"]
        counts = power[count_columns].to_numpy(dtype=float)
        if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
            raise ValueError("power counts must be finite nonnegative integers")
        n = power["repetitions"].to_numpy(dtype=float)
        valid = power["valid_repetitions"].to_numpy(dtype=float)
        rejected = power["rejections"].to_numpy(dtype=float)
        if not np.equal(n, settings["repetitions"]).all() or not np.array_equal(valid + power["invalid_repetitions"], n) or (rejected > valid).any():
            raise ValueError("power repetition accounting does not reconcile")
        rate = rejected / n
        expected_valid_only = np.divide(rejected, valid, out=np.full_like(rejected, np.nan), where=valid > 0)
        if not np.allclose(power["rejection_rate"], rate) or not np.allclose(power["rejection_rate_valid_only"], expected_valid_only, equal_nan=True) or not np.allclose(power["rejection_rate_mc_se"], np.sqrt(rate * (1 - rate) / n)):
            raise ValueError("power rates or Monte Carlo errors disagree with rejection counts")
        z = 1.959963984540054
        denominator = 1 + z * z / n
        center = (rate + z * z / (2 * n)) / denominator
        half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
        low = np.minimum(rate, np.maximum(0, center - half))
        high = np.maximum(rate, np.minimum(1, center + half))
        if not np.allclose(power["mc_confidence_low"], low, atol=1e-12, rtol=1e-10) or not np.allclose(power["mc_confidence_high"], high, atol=1e-12, rtol=1e-10):
            raise ValueError("power Wilson intervals disagree with rejection counts")
        key = ["model_id", "dgp_id", "scale_information", "inference_method"]
        for _, group in power.groupby(key, sort=False):
            null = group.loc[group["effect"].eq(0)]
            if len(null) != 1:
                raise ValueError("every power curve must include one null-size experiment")
            expected_gate = bool(null["invalid_repetitions"].iloc[0] == 0 and null["mc_confidence_high"].iloc[0] <= assumptions["size_maximum"])
            if not group["size_gate_pass"].eq(expected_gate).all():
                raise ValueError("power size gate ignores invalid draws or null Monte Carlo uncertainty")

    def validate_power_curve_registry(power, expected_keys, effects) -> None:
        key = ["model_id", "dgp_id", "scale_information", "inference_method"]
        if set(power[key].itertuples(index=False, name=None)) != expected_keys:
            raise ValueError("power curves omit declared models, dependence scenarios or inference methods")
        canonical = power[key].copy()
        canonical["effect"] = power["effect"].round(12)
        if canonical.duplicated([*key, "effect"]).any():
            raise ValueError("power curves repeat a declared model/DGP/method/effect key")
        for _, group in canonical.groupby(key, sort=False):
            if set(group["effect"]) != effects:
                raise ValueError("power effect grid differs from declared signed alternatives and MDE grid")

    def validate_redesign_power(config, tables, estimates) -> None:
        settings = config["redesign"]["power"]
        assumptions = config["redesign"]["precision_assumptions"]
        power, calibration, mde = tables["power"], tables["power_calibration"], tables["mde"]
        ids = set(estimates["model_id"])
        methods = {"pair_CRV1", "pair_CV3J", "pair_region_CRV1", "shared_endpoint_dyadic"}
        scales = {"preperiod_only", "policy_residual_sensitivity"}
        dgps = set(settings["dgp_ids"])
        grid = settings.get("mde_grid")
        if grid is None:
            count = round(settings["mde_grid_max"] / settings["mde_grid_step"])
            grid = [index * settings["mde_grid_step"] for index in range(1, count + 1)]
        grid = sorted(float(value) for value in grid)
        effects = {round(float(value), 12) for value in (*settings["effect_grid"], *grid, *[-value for value in grid])}
        key = ["model_id", "dgp_id", "scale_information", "inference_method"]
        expected_keys = {(model_id, dgp, scale, method) for model_id in ids for dgp in dgps for scale in scales for method in methods}
        validate_power_curve_registry(power, expected_keys, effects)
        if set(power["seed"]) != {settings["seed"]} or set(power["alpha"]) != {assumptions["alpha"]}:
            raise ValueError("power seed or test size differs from registry")
        validate_power_counts(power, settings, assumptions)
        calibration_key = key[:3]
        expected_calibration = {(model_id, dgp, scale) for model_id in ids for dgp in dgps for scale in scales}
        if set(calibration[calibration_key].itertuples(index=False, name=None)) != expected_calibration or calibration.duplicated(calibration_key).any():
            raise ValueError("power calibration registry is incomplete")
        require_finite(calibration, ["sigma", "observed_residual_ss", "expected_unit_residual_ss", "expected_unit_residual_ss_mc_se"], "power calibration")
        if (calibration[["sigma", "observed_residual_ss", "expected_unit_residual_ss"]] <= 0).any().any() or not np.allclose(np.square(calibration["sigma"]), calibration["observed_residual_ss"] / calibration["expected_unit_residual_ss"]):
            raise ValueError("power scale is not calibrated to the declared residual variation")
        if set(calibration["calibration_repetitions"]) != {settings["calibration_repetitions"]} or set(calibration["seed"]) != {settings["seed"]} or set(calibration["ar1"]) != {settings["ar1"]} or set(calibration["n_preperiod_years"]) != {3} or set(calibration["n_policy_years"]) != {12}:
            raise ValueError("power calibration history or resource guards differ from registry")
        for row in calibration.itertuples(index=False):
            expected_projection = "unit_and_stratum_year_FE_only_no_wage_regressor" if row.scale_information == "preperiod_only" else "unit_and_stratum_year_FE_plus_full_policy_regressors"
            if row.calibration_projection != expected_projection:
                raise ValueError("power preperiod calibration must not use contaminated wage regressors")
            fractions = settings["variance_fractions"][row.dgp_id]
            if any(not np.isclose(getattr(row, f"{name}_variance_fraction"), fractions[name]) for name in ("region", "pair", "county")) or not np.isclose(row.spatial_graph_coefficient, settings["spatial_graph_coefficient"] if row.dgp_id == "spatial_region_serial" else 0):
                raise ValueError("simulated dependence fractions differ from declared DGP")
        expected_mde = {(*values, sign) for values in expected_keys for sign in (-1, 1)}
        if set(mde[[*key, "effect_sign"]].itertuples(index=False, name=None)) != expected_mde or mde.duplicated([*key, "effect_sign"]).any():
            raise ValueError("MDE registry must retain both signs for every scenario")
        groups = {values: group for values, group in power.groupby(key, sort=False)}
        for row in mde.itertuples(index=False):
            curve = groups[tuple(getattr(row, column) for column in key)]
            null = curve.loc[curve["effect"].eq(0)].iloc[0]
            directional = curve.copy()
            directional["magnitude"] = (directional["effect"] * row.effect_sign).round(12)
            directional = directional.set_index("magnitude").loc[[round(value, 12) for value in grid]]
            def crossing(column):
                indices = np.flatnonzero(directional[column].to_numpy() >= assumptions["target_power"])
                return grid[indices[0]] if len(indices) else np.nan
            expected = crossing("rejection_rate")
            expected_credible = expected if null["size_gate_pass"] else np.nan
            if not np.allclose([row.grid_mde, row.credible_grid_mde, row.mc_mde_lower, row.mc_mde_upper], [expected, expected_credible, crossing("mc_confidence_high"), crossing("mc_confidence_low")], equal_nan=True) or row.mde_above_grid != pd.isna(expected) or row.size_gate_pass != null["size_gate_pass"]:
                raise ValueError("MDE crosses the wrong power curve or bypasses the size gate")
            if not np.isclose(row.null_rejection_rate, null["rejection_rate"]) or not np.isclose(row.null_mc_confidence_high, null["mc_confidence_high"]) or row.invalid_repetitions != null["invalid_repetitions"] or row.repetitions != settings["repetitions"] or row.seed != settings["seed"] or row.target_power != assumptions["target_power"]:
                raise ValueError("MDE null-size evidence does not match its power simulation")

    def validate_assignment_audit(config, assignment, panel) -> None:
        key = ["aewr_region_id", "year"]
        if assignment.duplicated(key).any():
            raise ValueError("assignment audit must be unique by region-year")
        expected = panel.groupby(key)[config["treatment"]].agg(
            minimum="min", maximum="max", n_values="nunique"
        )
        recorded = assignment.set_index(key)
        if set(recorded.index) != set(expected.index):
            raise ValueError("assignment audit must cover the full panel region-year support")
        recorded = recorded.loc[expected.index]
        if not np.allclose(recorded[["minimum", "maximum", "n_values"]], expected, atol=1e-12, rtol=1e-12, equal_nan=True) or not np.allclose(recorded["treatment_range"], expected["maximum"] - expected["minimum"], equal_nan=True):
            raise ValueError("assignment audit does not reconstruct observed wage support")
        years = assignment["year"]
        contaminated = years.eq(2009)
        expected_status = np.where(contaminated, "statutory_AEWR_unavailable_nonstatutory_mean_quarantined", "recorded_statutory_AEWR")
        if not np.array_equal(assignment["measurement_status"], expected_status):
            raise ValueError("assignment audit must retain the 2009 nonstatutory wage warning")
        policy = years.between(config["policy_start_year"], config["policy_end_year"])
        if (assignment.loc[policy, "treatment_range"] > 1e-10).any():
            raise ValueError("policy assignment contains within-region wage variation")

    def validate_bootstrap_power(config, tables, estimates) -> None:
        settings, bootstrap = config["redesign"]["power"], config["bootstrap"]
        assumptions = config["redesign"]["precision_assumptions"]
        frame = tables["bootstrap_power"]
        key = ["model_id", "dgp_id", "scale_information", "effect"]
        expected = {(model_id, dgp, scale, float(effect))
            for model_id in settings["bootstrap_model_ids"] for dgp in settings["dgp_ids"]
            for scale in ("preperiod_only", "policy_residual_sensitivity")
            for effect in settings["bootstrap_effect_grid"]}
        if set(frame[key].itertuples(index=False, name=None)) != expected or frame.duplicated(key).any():
            raise ValueError("bootstrap power must cover its entire bounded model/DGP/effect registry")
        validate_power_counts(frame, {"repetitions": settings["bootstrap_simulation_repetitions"]}, assumptions)
        metadata = {
            "inference_method": "pair_WCR11_FNW", "bootstrap_weights": "webb",
            "bootstrap_type": "fnw11", "bootstrap_package_version": "0.14.3",
            "bootstrap_pvalue_convention": "strict_absolute_exceedance_no_plus_one",
            "bootstrap_repetitions": bootstrap["repetitions"], "bootstrap_seed": bootstrap["seed"],
            "simulation_seed": settings["seed"], "n_regressors": 1,
            "simulation_draw_selection": "first_declared_repetitions_of_same_raw_power_draws",
            "fixed_effects_nested_in_pair": True,
        }
        if any(not frame[column].eq(value).all() for column, value in metadata.items()):
            raise ValueError("bootstrap simulation changed its validated scalar law or resource guards")
        calibration = tables["power_calibration"].set_index(["model_id", "dgp_id", "scale_information"])
        fitted = estimates.set_index("model_id")
        for row in frame.itertuples(index=False):
            if not np.isclose(row.sigma, calibration.loc[(row.model_id, row.dgp_id, row.scale_information), "sigma"]) or row.n_obs != fitted.loc[row.model_id, "n_observations"] or row.n_pairs != fitted.loc[row.model_id, "n_pairs"] or row.n_regions != fitted.loc[row.model_id, "n_regions"]:
                raise ValueError("bootstrap simulation does not use the declared fitted sample/calibration")
            if row.invalid_bootstrap_draws < 0 or row.infinite_bootstrap_draws < 0 or row.repetitions_with_invalid_bootstrap_draws > row.invalid_repetitions or row.repetitions_with_invalid_bootstrap_draws > row.invalid_bootstrap_draws:
                raise ValueError("bootstrap invalid-draw accounting is inconsistent")
        diagnostics = tables["simulation_diagnostics"]
        expected_diagnostics = {(model_id, dgp) for model_id in set(estimates["model_id"]) for dgp in settings["dgp_ids"]}
        if set(diagnostics[["model_id", "dgp_id"]].itertuples(index=False, name=None)) != expected_diagnostics or diagnostics.duplicated(["model_id", "dgp_id"]).any():
            raise ValueError("simulation diagnostics must retain every model/DGP")
        require_finite(diagnostics, ["negative_two_way_variances", "negative_dyadic_variances", "simulated_coefficient_error_mean", "simulated_coefficient_error_sd", "repetitions"], "simulation diagnostics")
        if set(diagnostics["repetitions"]) != {settings["repetitions"]} or (diagnostics[["negative_two_way_variances", "negative_dyadic_variances"]] < 0).any().any() or (diagnostics[["negative_two_way_variances", "negative_dyadic_variances"]] > settings["repetitions"]).any().any():
            raise ValueError("simulation covariance diagnostics have invalid draw counts")

    def validate_redesign_decisions(config, tables, provenance) -> None:
        """Reconstruct precision gates; check consistency of the recorded identification assessment."""
        assumptions = config["redesign"]["precision_assumptions"]
        central = assumptions["primary_threshold_beta"]
        shared_dgps = {"shared_region_serial", "shared_region_and_pair_serial", "spatial_region_serial"}
        methods = {"pair_CV3J", "shared_endpoint_dyadic"}
        model_ids = {model["model_id"] for model in config["redesign"]["models"]}
        decisions = tables["decision"]
        key = ["model_id", "inference_method"]
        expected_keys = {(model_id, method) for model_id in model_ids for method in methods}
        if set(decisions[key].itertuples(index=False, name=None)) != expected_keys or decisions.duplicated(key).any():
            raise ValueError("decision table must retain every declared candidate and inference comparison")
        gate_columns = ["all_shared_dgp_size_gates", "interval_precision_pass", "conditional_precision_pass", "identification_gate_pass", "overall_requirement_met"]
        if any(not pd.api.types.is_bool_dtype(decisions[column]) or decisions[column].isna().any() for column in gate_columns):
            raise ValueError("decision gates must be explicit nonmissing booleans")
        identification_status = provenance.get("identification_status")
        if not isinstance(identification_status, str) or not identification_status:
            raise ValueError("decision provenance must record a separate identification assessment")
        if identification_status == "not_established_with_current_local_data" and decisions["identification_gate_pass"].any():
            raise ValueError("decision identification gate contradicts the recorded unresolved assessment")
        estimates = tables["estimates"].set_index("model_id")
        for row in decisions.itertuples(index=False):
            curve = tables["power"]
            curve = curve.loc[curve["model_id"].eq(row.model_id) & curve["inference_method"].eq(row.inference_method) & curve["scale_information"].eq("preperiod_only") & curve["dgp_id"].isin(shared_dgps) & np.isclose(curve["effect"].abs(), central, atol=1e-12, rtol=0)]
            expected_support = {(dgp, sign) for dgp in shared_dgps for sign in (-1, 1)}
            signed_support = set(zip(curve["dgp_id"], np.sign(curve["effect"]), strict=True))
            if len(curve) != 6 or signed_support != expected_support:
                raise ValueError("decision central power must retain both effect signs and every shared-dependence DGP")
            mde = tables["mde"]
            mde = mde.loc[mde["model_id"].eq(row.model_id) & mde["inference_method"].eq(row.inference_method) & mde["scale_information"].eq("preperiod_only") & mde["dgp_id"].isin(shared_dgps)]
            if len(mde) != 6 or set(mde[["dgp_id", "effect_sign"]].itertuples(index=False, name=None)) != expected_support:
                raise ValueError("decision MDE must retain both effect signs and every shared-dependence DGP")
            minimum_power = curve["rejection_rate"].min()
            size_ok = bool(curve["size_gate_pass"].all())
            maximum_mde = mde["credible_grid_mde"].max() if np.isfinite(mde["credible_grid_mde"]).all() else np.nan
            estimate = estimates.loc[row.model_id]
            prefix = "crv3_jackknife" if row.inference_method == "pair_CV3J" else "dyadic"
            halfwidth = (estimate[f"{prefix}_confidence_high"] - estimate[f"{prefix}_confidence_low"]) / 2
            interval_ok = bool(np.isfinite(halfwidth) and 0 <= halfwidth <= central)
            precision_ok = bool(size_ok and interval_ok and np.isfinite(minimum_power) and minimum_power >= assumptions["target_power"] and np.isfinite(maximum_mde) and maximum_mde <= central)
            if not np.allclose([row.central_effect, row.minimum_signed_central_power, row.maximum_size_valid_mde, row.confidence_halfwidth], [central, minimum_power, maximum_mde, halfwidth], equal_nan=True):
                raise ValueError("decision evidence differs from signed power, size-valid MDE or observed intervals")
            if row.all_shared_dgp_size_gates != size_ok or row.interval_precision_pass != interval_ok or row.conditional_precision_pass != precision_ok:
                raise ValueError("decision precision gates do not follow the declared conjunction")
            if row.overall_requirement_met != (precision_ok and row.identification_gate_pass):
                raise ValueError("overall decision must require both conditional precision and the separate identification assessment")

    def main(*, inputs_only: bool = False) -> None:
        config = json.loads((DESIGN_DIR / "design.json").read_text())
        features_path = INTERMEDIATE / "border_discontinuity_county_features.parquet"
        memberships_path = (
            INTERMEDIATE / "border_discontinuity_geography_memberships.parquet"
        )
        diagnostics_path = (
            INTERMEDIATE / "border_discontinuity_geography_diagnostics.parquet"
        )
        exclusions_path = (
            INTERMEDIATE / "border_discontinuity_assignment_exclusions.parquet"
        )
        agricultural_audit_path = (
            INTERMEDIATE
            / "border_discontinuity_agricultural_border_audit.parquet"
        )
        panel_path = PROCESSED / "border_discontinuity_panel.parquet"
        for path in (
            features_path,
            memberships_path,
            diagnostics_path,
            exclusions_path,
            agricultural_audit_path,
            panel_path,
        ):
            if not path.exists() or path.stat().st_size == 0:
                raise ValueError(f"missing nonempty artifact: {path}")

        feature_schema = pq.read_schema(features_path)
        if not is_arrow_string(feature_schema.field("county_fips").type):
            raise TypeError("feature county_fips must be stored as a string")
        features = pq.read_table(features_path).to_pandas()
        scaled_columns = sorted(c for c in features if c.startswith("scaled__"))
        raw_columns = sorted(c for c in features if c.startswith("raw__"))
        if not scaled_columns or len(scaled_columns) != len(raw_columns):
            raise ValueError("raw and scaled feature inventories must be nonempty and equal")
        if features.empty or features["county_fips"].duplicated().any():
            raise ValueError("feature artifact must be nonempty and county-unique")
        if set(features["feature_start_year"]) != {config["feature_start_year"]} or set(
            features["feature_end_year"]
        ) != {config["feature_end_year"]}:
            raise ValueError("feature artifact does not use the frozen pre-period window")
        require_finite(features, scaled_columns, "scaled feature artifact")
        support_columns = {
            "cropland_acres_2007",
            "mean_farm_employment_2008_2010",
            "farm_employment_years_observed",
            "land_area_acres_2010",
        }
        require_columns(features, support_columns, "agricultural support")
        require_finite(
            features,
            ["farm_employment_years_observed", "land_area_acres_2010"],
            "agricultural support",
        )
        observed_support = features[
            ["cropland_acres_2007", "mean_farm_employment_2008_2010"]
        ]
        filter_config = config["agricultural_county_pair_filter"]
        expected_filter_config = {
            "filter_unit": "cross_border_county_pair",
            "cropland_year": 2007,
            "farm_employment_start_year": 2008,
            "farm_employment_end_year": 2010,
            "minimum_county_cropland_share": 0.075,
            "minimum_county_farm_employment_per_100_square_miles": 30.0,
            "minimum_county_measurement_coverage": 0.9,
        }
        if filter_config != expected_filter_config:
            raise ValueError("agricultural county-pair filter is not frozen")
        expected_farm_years = (
            int(filter_config["farm_employment_end_year"])
            - int(filter_config["farm_employment_start_year"])
            + 1
        )
        if (
            (features["land_area_acres_2010"] <= 0).any()
            or (observed_support < 0).any().any()
            or not features["farm_employment_years_observed"]
            .between(0, expected_farm_years)
            .all()
            or not np.equal(
                features["farm_employment_years_observed"],
                np.floor(features["farm_employment_years_observed"]),
            ).all()
        ):
            raise ValueError("agricultural support contains invalid values")

        membership_schema = pq.read_schema(memberships_path)
        for column in (
            "county_fips",
            "aewr_region_id",
            "region_pair_id",
            "spec_id",
            "stratum_id",
        ):
            if not is_arrow_string(membership_schema.field(column).type):
                raise TypeError(f"membership {column} must be stored as a string")
        memberships = pq.read_table(memberships_path).to_pandas()
        require_columns(
            memberships,
            {
                "spec_id",
                "county_fips",
                "aewr_region_id",
                "region_pair_id",
                "stratum_id",
                "membership_mode",
                "role",
                "analysis_weight",
                "border_depth",
            },
            "geography memberships",
        )
        if config["bootstrap"]["cluster"] != "region_pair_id":
            raise ValueError("border-design inference must cluster by region_pair_id")
        two_way_config = config["two_way_cluster_sensitivity"]
        if (
            two_way_config["method"] != "CRV1"
            or two_way_config["clusters"]
            != ["region_pair_id", "aewr_region_id"]
            or two_way_config["confidence_level"] != 0.95
        ):
            raise ValueError(
                "two-way sensitivity must use CRV1 by border pair and AEWR region"
            )
        if (
            not memberships["region_pair_id"].str.fullmatch(r"\d{2}_\d{2}").all()
            or (
                memberships.groupby(["spec_id", "stratum_id"])["region_pair_id"]
                .nunique()
                != 1
            ).any()
            or (
                memberships.groupby(["spec_id", "region_pair_id"])[
                    "aewr_region_id"
                ].nunique()
                != 2
            ).any()
        ):
            raise ValueError("border-pair identifiers violate their declared structure")
        declared_specs = {value["spec_id"] for value in config["geography_specs"]}
        if set(memberships["spec_id"]) != declared_specs:
            raise ValueError("membership specifications differ from design.json")
        if set(memberships["geography_family"]) != {"ring"}:
            raise ValueError("border memberships must contain only ring geographies")
        if memberships.duplicated(["spec_id", "stratum_id", "county_fips"]).any():
            raise ValueError("membership artifact violates its declared key")
        disjoint = memberships.loc[memberships["membership_mode"].eq("disjoint")]
        if disjoint.duplicated(["spec_id", "county_fips"]).any():
            raise ValueError("a disjoint specification assigns a county more than once")
        weight_sums = memberships.groupby(["spec_id", "county_fips"])[
            "analysis_weight"
        ].sum()
        if not np.allclose(weight_sums, 1.0, atol=1e-12, rtol=0):
            raise ValueError("membership weights do not sum to one by specification-county")
        require_finite(
            memberships, ["analysis_weight", "border_depth"], "geography memberships"
        )

        panel = pq.read_table(
            panel_path, columns=["county_fips", "year", "aewr_region_id"]
        ).to_pandas()
        panel["county_fips"] = panel["county_fips"].map(harmonize_county_fips_2010)
        regions_frame = panel[["county_fips", "aewr_region_id"]].drop_duplicates()
        if regions_frame["county_fips"].duplicated().any():
            raise ValueError("panel AEWR region must be time invariant")
        regions = dict(
            zip(
                regions_frame["county_fips"],
                regions_frame["aewr_region_id"].astype(str),
                strict=True,
            )
        )
        adjacency = pq.read_table(
            INTERMEDIATE / "county_adjacency2010.parquet",
            columns=["county_fips", "neighbor_county_fips"],
        ).to_pandas()
        adjacency["county_fips"] = adjacency["county_fips"].map(
            harmonize_county_fips_2010
        )
        adjacency["neighbor_county_fips"] = adjacency["neighbor_county_fips"].map(
            harmonize_county_fips_2010
        )
        neighbors, edges = build_graph(
            regions,
            zip(
                adjacency["county_fips"],
                adjacency["neighbor_county_fips"],
                strict=True,
            ),
        )
        agricultural_audit_schema = pq.read_schema(agricultural_audit_path)
        audit_string_columns = (
            "region_pair_id",
            "county_pair_id",
            "left_county_fips",
            "right_county_fips",
            "left_aewr_region_id",
            "right_aewr_region_id",
            "screen_stage",
        )
        for column in audit_string_columns:
            if not is_arrow_string(agricultural_audit_schema.field(column).type):
                raise TypeError(
                    f"agricultural county-pair audit {column} must be a string"
                )
        agricultural_audit = pq.read_table(agricultural_audit_path).to_pandas()
        required_audit_columns = {
            "cropland_year",
            "farm_employment_start_year",
            "farm_employment_end_year",
            "minimum_cropland_share",
            "minimum_farm_employment_per_100_square_miles",
            "minimum_measurement_coverage",
            "left_cropland_acres_2007",
            "left_land_area_acres_2010",
            "left_cropland_share_2007",
            "left_mean_annual_farm_employment_2008_2010",
            "left_farm_employment_per_100_square_miles",
            "left_cropland_coverage",
            "left_farm_employment_coverage",
            "left_cropland_coverage_pass",
            "left_farm_employment_coverage_pass",
            "left_cropland_share_pass",
            "left_farm_employment_density_pass",
            "left_county_eligible",
            "right_cropland_acres_2007",
            "right_land_area_acres_2010",
            "right_cropland_share_2007",
            "right_mean_annual_farm_employment_2008_2010",
            "right_farm_employment_per_100_square_miles",
            "right_cropland_coverage",
            "right_farm_employment_coverage",
            "right_cropland_coverage_pass",
            "right_farm_employment_coverage_pass",
            "right_cropland_share_pass",
            "right_farm_employment_density_pass",
            "right_county_eligible",
            "county_pair_eligible",
        }
        require_columns(
            agricultural_audit,
            required_audit_columns,
            "agricultural county-pair audit",
        )
        if (
            agricultural_audit.empty
            or agricultural_audit["county_pair_id"].duplicated().any()
            or not (
                agricultural_audit["left_county_fips"]
                < agricultural_audit["right_county_fips"]
            ).all()
        ):
            raise ValueError("agricultural audit county-pair keys are invalid")

        raw_cross_edges = {
            edge for edge in edges if regions[edge[0]] != regions[edge[1]]
        }
        audited_edges = set(
            zip(
                agricultural_audit["left_county_fips"],
                agricultural_audit["right_county_fips"],
                strict=True,
            )
        )
        if audited_edges != raw_cross_edges:
            raise ValueError("agricultural audit does not cover every border edge")
        if not (
            agricultural_audit["county_pair_id"]
            == agricultural_audit["left_county_fips"]
            + "__"
            + agricultural_audit["right_county_fips"]
        ).all():
            raise ValueError("agricultural audit county-pair labels are inconsistent")

        feature_support = features.set_index("county_fips")
        for label in ("left", "right"):
            counties = agricultural_audit[f"{label}_county_fips"]
            expected_regions = counties.map(regions)
            expected_pairs = [
                pair_identifier(left, right)
                for left, right in zip(
                    agricultural_audit["left_aewr_region_id"],
                    agricultural_audit["right_aewr_region_id"],
                    strict=True,
                )
            ]
            if (
                not expected_regions.equals(
                    agricultural_audit[f"{label}_aewr_region_id"]
                )
                or expected_pairs != agricultural_audit["region_pair_id"].tolist()
            ):
                raise ValueError("agricultural audit region labels are inconsistent")

            cropland = counties.map(feature_support["cropland_acres_2007"])
            farm_employment = counties.map(
                feature_support["mean_farm_employment_2008_2010"]
            )
            farm_years = counties.map(
                feature_support["farm_employment_years_observed"]
            )
            land_area = counties.map(feature_support["land_area_acres_2010"])
            expected_values = {
                f"{label}_cropland_acres_2007": cropland,
                f"{label}_land_area_acres_2010": land_area,
                f"{label}_cropland_share_2007": cropland / land_area,
                f"{label}_mean_annual_farm_employment_2008_2010": farm_employment,
                f"{label}_farm_employment_per_100_square_miles": (
                    farm_employment * 64_000 / land_area
                ),
                f"{label}_cropland_coverage": cropland.notna().astype(float),
                f"{label}_farm_employment_coverage": (
                    farm_years / expected_farm_years
                ),
            }
            for column, expected in expected_values.items():
                if not np.allclose(
                    agricultural_audit[column],
                    expected,
                    atol=1e-12,
                    rtol=1e-12,
                    equal_nan=True,
                ):
                    raise ValueError(f"agricultural audit does not reproduce {column}")
            expected_passes = {
                f"{label}_cropland_coverage_pass": (
                    cropland.notna().astype(float)
                    >= filter_config["minimum_county_measurement_coverage"]
                ),
                f"{label}_farm_employment_coverage_pass": (
                    farm_years / expected_farm_years
                    >= filter_config["minimum_county_measurement_coverage"]
                ),
                f"{label}_cropland_share_pass": (
                    cropland / land_area
                    >= filter_config["minimum_county_cropland_share"]
                ),
                f"{label}_farm_employment_density_pass": (
                    farm_employment * 64_000 / land_area
                    >= filter_config[
                        "minimum_county_farm_employment_per_100_square_miles"
                    ]
                ),
            }
            for column, expected in expected_passes.items():
                if not np.array_equal(
                    agricultural_audit[column].to_numpy(),
                    expected.fillna(False).to_numpy(),
                ):
                    raise ValueError(f"agricultural audit does not reproduce {column}")
            expected_county_eligible = np.logical_and.reduce(
                [value.fillna(False).to_numpy() for value in expected_passes.values()]
            )
            if not np.array_equal(
                agricultural_audit[f"{label}_county_eligible"].to_numpy(),
                expected_county_eligible,
            ):
                raise ValueError(
                    f"agricultural audit does not reproduce {label} eligibility"
                )

        expected_pair_eligible = (
            agricultural_audit["left_county_eligible"]
            & agricultural_audit["right_county_eligible"]
        )
        if not np.array_equal(
            agricultural_audit["county_pair_eligible"].to_numpy(),
            expected_pair_eligible.to_numpy(),
        ):
            raise ValueError("agricultural county-pair eligibility is inconsistent")
        metadata_checks = {
            "cropland_year": filter_config["cropland_year"],
            "farm_employment_start_year": filter_config[
                "farm_employment_start_year"
            ],
            "farm_employment_end_year": filter_config["farm_employment_end_year"],
            "minimum_cropland_share": filter_config[
                "minimum_county_cropland_share"
            ],
            "minimum_farm_employment_per_100_square_miles": filter_config[
                "minimum_county_farm_employment_per_100_square_miles"
            ],
            "minimum_measurement_coverage": filter_config[
                "minimum_county_measurement_coverage"
            ],
            "screen_stage": (
                "cross_border_edges_before_ring_expansion_and_disjoint_assignment"
            ),
        }
        if any(
            set(agricultural_audit[column]) != {value}
            for column, value in metadata_checks.items()
        ):
            raise ValueError("agricultural audit metadata differs from design.json")

        eligible_edges = set(
            zip(
                agricultural_audit.loc[
                    agricultural_audit["county_pair_eligible"],
                    "left_county_fips",
                ],
                agricultural_audit.loc[
                    agricultural_audit["county_pair_eligible"],
                    "right_county_fips",
                ],
                strict=True,
            )
        )
        eligible_region_pairs = set(
            agricultural_audit.loc[
                agricultural_audit["county_pair_eligible"], "region_pair_id"
            ]
        )
        if not eligible_edges:
            raise ValueError("agricultural county-pair screen retained no edges")
        if not set(memberships["region_pair_id"]).issubset(eligible_region_pairs):
            raise ValueError("memberships contain an ineligible AEWR border pair")
        stacked_pairs = set(
            memberships.loc[
                memberships["spec_id"].eq("ring_h1_stacked"), "region_pair_id"
            ]
        )
        if stacked_pairs != eligible_region_pairs:
            raise ValueError("stacked one-ring sample omits an eligible border pair")
        for (spec_id, stratum_id), group in memberships.groupby(
            ["spec_id", "stratum_id"], sort=True
        ):
            members = set(group["county_fips"])
            if len(connected_components(neighbors, members)) != 1:
                raise ValueError(f"{spec_id}/{stratum_id} is disconnected")
            qualifying_cross_edge = any(
                left in members
                and right in members
                for left, right in eligible_edges
            )
            if (
                len({regions[county] for county in members}) < 2
                or not qualifying_cross_edge
            ):
                raise ValueError(
                    f"{spec_id}/{stratum_id} lacks an eligible agricultural edge"
                )

        validate_redesign_geographies(
            config, memberships, features, eligible_edges, neighbors, regions
        )
        validate_region_disjoint_matching(config, memberships, eligible_edges)

        diagnostics = pq.read_table(diagnostics_path).to_pandas()
        exclusions = pq.read_table(exclusions_path).to_pandas()
        if diagnostics.empty or exclusions.empty:
            raise ValueError("geography diagnostics and exclusions must be nonempty")
        if set(diagnostics["spec_id"]) != declared_specs:
            raise ValueError("diagnostic specifications differ from design.json")
        if set(diagnostics["geography_family"]) != {"ring"}:
            raise ValueError("border diagnostics must contain only ring geographies")
        if not diagnostics["retained"].all():
            raise ValueError("ring diagnostics contain a nonretained stratum")
        require_columns(
            diagnostics,
            {"region_pair_id", "n_qualifying_cross_region_edges"},
            "geography diagnostics",
        )
        if (diagnostics["n_qualifying_cross_region_edges"] < 1).any():
            raise ValueError("geography diagnostics contain unsupported strata")
        diagnostic_keys = set(zip(diagnostics["spec_id"], diagnostics["stratum_id"]))
        membership_keys = set(zip(memberships["spec_id"], memberships["stratum_id"]))
        if diagnostic_keys != membership_keys:
            raise ValueError("ring diagnostics do not cover every retained stratum")

        panel_schema = pq.read_schema(panel_path)
        for column in ("county_fips", "aewr_region_id"):
            if not is_arrow_string(panel_schema.field(column).type):
                raise TypeError(f"panel {column} must be stored as a string")
        full_panel = pq.read_table(panel_path).to_pandas()
        if full_panel.empty or full_panel.duplicated(["county_fips", "year"]).any():
            raise ValueError("border panel must have unique, nonempty county-year keys")
        require_columns(
            full_panel,
            set(config["diagnostic_margin_columns"].values()),
            "border panel diagnostic margins",
        )
        linkage_columns = {
            "employer_linkage_unit",
            "employer_linkage_version",
            "employer_linkage_input_max_year",
            "employer_linkage_review_status",
            "employer_linkage_review_report_hash",
        }
        require_columns(full_panel, linkage_columns, "border panel linkage")
        linkage = full_panel[sorted(linkage_columns)].drop_duplicates()
        expected_linkage = config["employer_linkage"]
        if (
            len(linkage) != 1
            or linkage.iloc[0]["employer_linkage_unit"]
            != expected_linkage["unit"]
            or linkage.iloc[0]["employer_linkage_version"]
            != expected_linkage["version"]
            or int(linkage.iloc[0]["employer_linkage_input_max_year"])
            != int(expected_linkage["input_max_year"])
            or linkage.iloc[0]["employer_linkage_review_status"]
            != expected_linkage["review_status"]
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(linkage.iloc[0]["employer_linkage_review_report_hash"]),
            )
            is None
        ):
            raise ValueError("border panel employer linkage differs from design.json")
        if set(full_panel["border_design_id"]) != {config["design_id"]}:
            raise ValueError("panel design metadata differs from design.json")
        if full_panel["year"].min() != config["placebo_start_year"] or full_panel[
            "year"
        ].max() != config["policy_end_year"]:
            raise ValueError("border panel year support differs from the design contract")
        for value in config["geography_specs"]:
            if value["membership_mode"] == "disjoint":
                column = f"stratum__{value['spec_id']}"
                if column not in full_panel:
                    raise ValueError(f"panel is missing disjoint assignment {column}")

        validate_redesign_panel(config, full_panel, features, memberships)

        if inputs_only:
            print(
                "Validated AEWR-border ring pre-estimation artifacts: "
                f"{len(memberships):,} memberships."
            )
            return

        estimate_path = TABLES / "border_discontinuity_estimates.csv"
        audit_path = TABLES / "border_discontinuity_sample_audit.csv"
        balance_path = TABLES / "border_discontinuity_balance.csv"
        placebo_path = TABLES / "border_discontinuity_placebo.csv"
        comparison_path = (
            TABLES / "border_discontinuity_ring_inference_comparison.csv"
        )
        annual_path = TABLES / "border_discontinuity_ring_annual_coefficients.csv"
        leave_one_out_path = TABLES / (
            "border_discontinuity_ring_leave_one_border_pair_out.csv"
        )
        jackknife_path = TABLES / (
            "border_discontinuity_ring_border_pair_jackknife.csv"
        )
        influence_path = TABLES / (
            "border_discontinuity_ring_border_pair_influence.csv"
        )
        support_path = TABLES / "border_discontinuity_ring_treatment_support.csv"
        county_year_decomposition_path = (
            TABLES / "border_discontinuity_ring_county_year_decomposition.csv"
        )
        county_decomposition_path = (
            TABLES / "border_discontinuity_ring_county_decomposition.csv"
        )
        stratum_region_year_decomposition_path = TABLES / (
            "border_discontinuity_ring_stratum_region_year_decomposition.csv"
        )
        region_decomposition_path = (
            TABLES / "border_discontinuity_ring_region_decomposition.csv"
        )
        border_pair_decomposition_path = TABLES / (
            "border_discontinuity_ring_border_pair_decomposition.csv"
        )
        retained_tables = (
            estimate_path,
            audit_path,
            balance_path,
            placebo_path,
            comparison_path,
            annual_path,
            leave_one_out_path,
            jackknife_path,
            influence_path,
            support_path,
            county_year_decomposition_path,
            county_decomposition_path,
            stratum_region_year_decomposition_path,
            region_decomposition_path,
            border_pair_decomposition_path,
            TABLES / "border_discontinuity_geography_diagnostics.csv",
            TABLES / "border_discontinuity_exclusion_summary.csv",
            TABLES / "border_discontinuity_feature_diagnostics.csv",
            TABLES / "border_discontinuity_agricultural_county_pair_audit.csv",
        )
        retained_figures = (
            FIGURES / "border_discontinuity_estimates.png",
            FIGURES / "border_discontinuity_ring_windows.png",
            FIGURES / "border_discontinuity_ring_annual_coefficients.png",
            FIGURES / "border_discontinuity_ring_leave_one_border_pair_out.png",
            FIGURES / "border_discontinuity_ring_primary_inference.png",
            FIGURES / "border_discontinuity_ring_border_pair_contributions.png",
            FIGURES / "border_discontinuity_ring_county_contributions.png",
            FIGURES / "border_discontinuity_ring_region_contributions.png",
            FIGURES / "border_discontinuity_ring_stratum_year_contributions.png",
        )
        for path in (*retained_tables, *retained_figures):
            if not path.exists() or path.stat().st_size == 0:
                raise ValueError(f"missing retained output: {path}")

        estimates = pd.read_csv(estimate_path)
        expected_ids = expected_model_ids(config)
        if set(estimates["model_id"]) != expected_ids or estimates["model_id"].duplicated().any():
            raise ValueError("estimate registry does not match design.json")
        require_finite(
            estimates,
            [
                "estimate",
                "crv1_standard_error",
                "crv1_p_value",
                "crv1_confidence_low",
                "crv1_confidence_high",
                "two_way_crv1_standard_error",
                "two_way_crv1_p_value",
                "two_way_crv1_confidence_low",
                "two_way_crv1_confidence_high",
                "two_way_crv1_raw_minimum_eigenvalue",
                "two_way_crv1_maximum_psd_adjustment",
                "two_way_confidence_level",
                "n_two_way_pair_clusters",
                "n_two_way_region_clusters",
                "n_two_way_intersection_clusters",
                "crv3_jackknife_standard_error",
                "crv3_jackknife_p_value",
                "crv3_jackknife_confidence_low",
                "crv3_jackknife_confidence_high",
                "crv3_jackknife_confidence_level",
                "crv3_jackknife_clusters",
                "wcb_p_value",
                "wcb_confidence_low",
                "wcb_confidence_high",
                "n_observations",
                "n_counties",
                "n_strata",
                "n_inference_clusters",
            ],
            "estimate output",
        )
        bootstrap = config["bootstrap"]
        audit_config = config["diagnostic_audit"]
        if (
            set(estimates["wcb_method"]) != {bootstrap["method"]}
            or set(estimates["wcb_bootstrap_type"])
            != {bootstrap["bootstrap_type"]}
            or set(estimates["wcb_weights"]) != {bootstrap["weights"]}
            or set(estimates["wcb_repetitions"]) != {bootstrap["repetitions"]}
            or set(estimates["wcb_seed"]) != {bootstrap["seed"]}
            or set(estimates["inference_cluster"]) != {bootstrap["cluster"]}
            or set(estimates["confidence_level"]) != {bootstrap["confidence_level"]}
            or set(estimates["wcb_fixed_effect_handling"])
            != {"weighted_fwl_absorption"}
            or estimates["wcb_method"].str.contains("AAIW|TSCB", case=False).any()
        ):
            raise ValueError("estimate inference metadata violates the frozen contract")
        expected_two_way_dimensions = "+".join(two_way_config["clusters"])
        if (
            set(estimates["two_way_crv1_method"])
            != {two_way_config["method"]}
            or set(estimates["two_way_cluster_dimensions"])
            != {expected_two_way_dimensions}
            or set(estimates["two_way_confidence_level"])
            != {two_way_config["confidence_level"]}
            or not np.array_equal(
                estimates["n_two_way_pair_clusters"].to_numpy(),
                estimates["n_inference_clusters"].to_numpy(),
            )
            or (estimates["n_two_way_region_clusters"] < 2).any()
            or (
                estimates["n_two_way_intersection_clusters"]
                != 2 * estimates["n_two_way_pair_clusters"]
            ).any()
            or (estimates["two_way_crv1_maximum_psd_adjustment"] < 0).any()
            or not np.array_equal(
                estimates["two_way_crv1_psd_adjusted"].to_numpy(),
                (
                    estimates["two_way_crv1_maximum_psd_adjustment"] > 1e-12
                ).to_numpy(),
            )
        ):
            raise ValueError("two-way CRV1 metadata violates the sensitivity contract")
        if (
            (estimates["two_way_crv1_standard_error"] <= 0).any()
            or not estimates["two_way_crv1_p_value"].between(0, 1).all()
            or (
                estimates["two_way_crv1_confidence_low"] > estimates["estimate"]
            ).any()
            or (
                estimates["two_way_crv1_confidence_high"] < estimates["estimate"]
            ).any()
        ):
            raise ValueError("two-way CRV1 estimates contain invalid inference values")
        if (
            set(estimates["crv3_jackknife_method"])
            != {audit_config["cluster_jackknife_method"]}
            or set(estimates["crv3_jackknife_cluster"])
            != {bootstrap["cluster"]}
            or set(estimates["crv3_jackknife_confidence_level"])
            != {audit_config["cluster_jackknife_confidence_level"]}
            or not np.array_equal(
                estimates["crv3_jackknife_clusters"].to_numpy(),
                estimates["n_inference_clusters"].to_numpy(),
            )
        ):
            raise ValueError("estimate CV3J metadata violates the frozen contract")
        if (estimates["n_inference_clusters"] < 2).any():
            raise ValueError("estimates contain fewer than two inference clusters")
        pair_counts = memberships.groupby("spec_id")["region_pair_id"].nunique()
        expected_pair_counts = estimates["spec_id"].map(pair_counts)
        if not np.array_equal(
            estimates["n_inference_clusters"].to_numpy(),
            expected_pair_counts.to_numpy(),
        ):
            raise ValueError("estimate cluster counts do not cover every border pair")

        audit = pd.read_csv(audit_path)
        if set(audit["model_id"]) != expected_ids or audit["model_id"].duplicated().any():
            raise ValueError("sample audit does not match the estimate registry")
        require_finite(
            audit,
            ["n_inference_clusters"],
            "sample audit inference clusters",
        )
        estimate_region_counts = estimates.set_index("model_id")[
            "n_two_way_region_clusters"
        ]
        audit_region_counts = audit.set_index("model_id")["n_regions"]
        if not estimate_region_counts.equals(audit_region_counts):
            raise ValueError("two-way AEWR-region counts differ from the sample audit")
        balance = pd.read_csv(balance_path)
        if len(balance) != 1 + len(config["baseline_control_columns"]):
            raise ValueError("primary-ring balance registry has the wrong size")
        if set(balance["spec_id"]) != {config["diagnostic_audit"]["primary_spec_id"]}:
            raise ValueError("balance output is not restricted to the primary ring")
        require_finite(
            balance,
            ["estimate", "crv1_standard_error", "crv1_p_value", "n_counties"],
            "balance output",
        )

        placebo = pd.read_csv(placebo_path)
        placebo_ids = set(estimates.loc[estimates["role"].eq("placebo"), "model_id"])
        if (
            set(placebo["model_id"]) != placebo_ids
            or placebo["model_id"].duplicated().any()
        ):
            raise ValueError("placebo output does not match the placebo registry")

        comparison = pd.read_csv(comparison_path)
        if (
            set(comparison["model_id"]) != expected_ids
            or comparison["model_id"].duplicated().any()
        ):
            raise ValueError("inference comparison does not match the model registry")
        require_finite(
            comparison,
            [
                "estimate",
                "crv1_p_value",
                "crv3_jackknife_p_value",
                "wcb_p_value",
                "two_way_crv1_p_value",
                "p_value_difference_crv3_minus_crv1",
                "p_value_difference_wcb_minus_crv1",
                "p_value_difference_two_way_minus_pair_crv1",
                "n_inference_clusters",
            ],
            "inference comparison",
        )

        primary_spec_id = audit_config["primary_spec_id"]
        primary_rows = estimates.loc[
            estimates["model_id"].eq(f"{primary_spec_id}__base")
        ]
        if len(primary_rows) != 1:
            raise ValueError("primary base estimate is missing or duplicated")
        primary_row = primary_rows.iloc[0]
        primary_estimate = float(primary_row["estimate"])

        annual = pd.read_csv(annual_path)
        expected_years = set(
            range(
                int(audit_config["annual_start_year"]),
                int(audit_config["annual_end_year"]) + 1,
            )
        )
        if (
            set(annual["year"]) != expected_years
            or annual["year"].duplicated().any()
            or set(annual["spec_id"]) != {primary_spec_id}
            or set(annual["n_inference_clusters"])
            != {int(pair_counts.loc[primary_spec_id])}
        ):
            raise ValueError("annual coefficient output violates its declared registry")
        require_finite(
            annual,
            [
                "estimate",
                "crv1_standard_error",
                "crv1_p_value",
                "crv1_confidence_low",
                "crv1_confidence_high",
                "crv3_jackknife_standard_error",
                "crv3_jackknife_p_value",
                "crv3_jackknife_confidence_low",
                "crv3_jackknife_confidence_high",
                "crv3_jackknife_confidence_level",
                "crv3_jackknife_clusters",
                "wcb_p_value",
                "wcb_confidence_low",
                "wcb_confidence_high",
                "n_inference_clusters",
            ],
            "annual coefficient output",
        )
        if (
            set(annual["wcb_method"]) != {bootstrap["method"]}
            or set(annual["wcb_bootstrap_type"]) != {bootstrap["bootstrap_type"]}
            or set(annual["wcb_weights"]) != {bootstrap["weights"]}
            or set(annual["wcb_repetitions"]) != {bootstrap["repetitions"]}
            or set(annual["wcb_seed"]) != {bootstrap["seed"]}
            or set(annual["inference_cluster"]) != {bootstrap["cluster"]}
        ):
            raise ValueError("annual bootstrap metadata violates the frozen contract")
        if (
            set(annual["crv3_jackknife_method"])
            != {audit_config["cluster_jackknife_method"]}
            or set(annual["crv3_jackknife_cluster"])
            != {bootstrap["cluster"]}
            or set(annual["crv3_jackknife_confidence_level"])
            != {audit_config["cluster_jackknife_confidence_level"]}
            or not np.array_equal(
                annual["crv3_jackknife_clusters"].to_numpy(),
                annual["n_inference_clusters"].to_numpy(),
            )
        ):
            raise ValueError("annual CV3J metadata violates the frozen contract")

        leave_one_out = pd.read_csv(
            leave_one_out_path, dtype={"excluded_border_pair_id": "string"}
        )
        primary_pairs = set(
            memberships.loc[
                memberships["spec_id"].eq(primary_spec_id), "region_pair_id"
            ].astype(str)
        )
        primary_strata = set(
            memberships.loc[
                memberships["spec_id"].eq(primary_spec_id), "stratum_id"
            ].astype(str)
        )
        primary_regions = set(
            memberships.loc[
                memberships["spec_id"].eq(primary_spec_id), "aewr_region_id"
            ].astype(str)
        )
        if (
            set(leave_one_out["excluded_border_pair_id"].astype(str))
            != primary_pairs
            or leave_one_out["excluded_border_pair_id"].duplicated().any()
        ):
            raise ValueError(
                "leave-one-out output does not cover every primary border pair"
            )
        require_finite(
            leave_one_out,
            [
                "estimate",
                "difference_from_full_estimate",
                "crv1_standard_error",
                "crv1_p_value",
                "crv1_confidence_low",
                "crv1_confidence_high",
                "n_inference_clusters",
            ],
            "leave-one-out output",
        )
        if not (
            leave_one_out["n_inference_clusters"] == len(primary_pairs) - 1
        ).all():
            raise ValueError("leave-one-out fits do not drop exactly one border pair")

        influence = pd.read_csv(
            influence_path, dtype={"region_pair_id": "string"}
        )
        if (
            set(influence["region_pair_id"].astype(str)) != primary_pairs
            or influence["region_pair_id"].duplicated().any()
        ):
            raise ValueError(
                "cluster influence output has the wrong border-pair registry"
            )
        require_finite(
            influence,
            [
                "cluster_score",
                "treatment_sum_squares",
                "treatment_leverage",
                "absolute_score_share",
                "one_step_coefficient_contribution",
                "effective_inference_cluster_count",
                "leave_one_out_estimate",
                "difference_from_full_estimate",
            ],
            "cluster influence output",
        )
        if not np.isclose(influence["treatment_leverage"].sum(), 1.0, atol=1e-10):
            raise ValueError("cluster treatment leverage does not sum to one")
        if not np.isclose(influence["absolute_score_share"].sum(), 1.0, atol=1e-10):
            raise ValueError("absolute cluster score shares do not sum to one")
        effective_clusters = influence["effective_inference_cluster_count"]
        if (
            effective_clusters.nunique() != 1
            or effective_clusters.iloc[0] <= 1
            or effective_clusters.iloc[0] > len(primary_pairs)
        ):
            raise ValueError("effective inference-cluster count is invalid")

        jackknife = pd.read_csv(jackknife_path)
        if len(jackknife) != 1 or set(jackknife["spec_id"]) != {primary_spec_id}:
            raise ValueError("cluster jackknife must contain one primary-ring result")
        require_finite(
            jackknife,
            [
                "estimate",
                "jackknife_mean_leave_one_out_estimate",
                "cluster_jackknife_standard_error",
                "cluster_jackknife_p_value",
                "cluster_jackknife_confidence_low",
                "cluster_jackknife_confidence_high",
                "n_inference_clusters",
            ],
            "cluster jackknife output",
        )
        if not np.isclose(jackknife["estimate"].iloc[0], primary_estimate):
            raise ValueError("cluster jackknife does not report the primary estimate")
        if int(jackknife["n_inference_clusters"].iloc[0]) != len(primary_pairs):
            raise ValueError("cluster jackknife has the wrong border-pair count")
        expected_jackknife_se = np.sqrt(
            (len(leave_one_out) - 1)
            / len(leave_one_out)
            * np.square(
                leave_one_out["estimate"] - leave_one_out["estimate"].mean()
            ).sum()
        )
        if (
            set(jackknife["crv3_jackknife_method"])
            != {audit_config["cluster_jackknife_method"]}
            or set(jackknife["crv3_jackknife_cluster"])
            != {bootstrap["cluster"]}
            or set(jackknife["confidence_level"])
            != {audit_config["cluster_jackknife_confidence_level"]}
            or not np.isclose(
                jackknife["crv3_jackknife_standard_error"].iloc[0],
                expected_jackknife_se,
                atol=1e-12,
                rtol=1e-10,
            )
            or not np.isclose(
                primary_row["crv3_jackknife_standard_error"],
                expected_jackknife_se,
                atol=1e-12,
                rtol=1e-10,
            )
        ):
            raise ValueError("primary CV3J output does not reproduce the cluster jackknife")

        support = pd.read_csv(support_path, dtype={"region_pair_id": "string"})
        if (
            set(support["year"]) != expected_years
            or set(support["spec_id"]) != {primary_spec_id}
            or set(support["region_pair_id"].astype(str)) != primary_pairs
            or set(support["stratum_id"].astype(str)) != primary_strata
            or support.duplicated(["stratum_id", "year"]).any()
            or len(support) != len(primary_strata) * len(expected_years)
            or (
                support.groupby("stratum_id")["region_pair_id"].nunique() != 1
            ).any()
            or not support["informative_treatment_contrast"].any()
            or (support["n_regions"] < 2).any()
        ):
            raise ValueError("primary-ring treatment-support contract failed")
        require_finite(
            support,
            [
                "n_counties",
                "n_regions",
                "treatment_minimum",
                "treatment_maximum",
                "treatment_range",
            ],
            "treatment support output",
        )

        decomposition_dtypes = {
            "county_fips": "string",
            "aewr_region_id": "string",
            "region_pair_id": "string",
            "stratum_id": "string",
        }
        county_year_decomposition = pd.read_csv(
            county_year_decomposition_path,
            dtype=decomposition_dtypes,
        )
        require_columns(
            county_year_decomposition,
            {
                "model_id",
                "spec_id",
                "county_fips",
                "aewr_region_id",
                "region_pair_id",
                "stratum_id",
                "year",
                "observed_outcome",
                "observed_treatment",
                "requested_positions",
                "certified_positions",
                "applications",
                "balanced_linked_worksite_businesses",
                "baseline_farm_employment",
                "coefficient_numerator",
                "coefficient_contribution",
                "model_score",
                "treatment_sum_squares",
                "treatment_leverage",
                "absolute_observation_score_share",
            },
            "county-year decomposition",
        )
        if (
            county_year_decomposition.empty
            or county_year_decomposition.duplicated(["county_fips", "year"]).any()
            or len(county_year_decomposition) != int(primary_row["n_observations"])
            or county_year_decomposition["county_fips"].nunique()
            != int(primary_row["n_counties"])
            or set(county_year_decomposition["year"])
            != set(range(config["policy_start_year"], config["policy_end_year"] + 1))
            or set(county_year_decomposition["model_id"])
            != {f"{primary_spec_id}__base"}
            or set(county_year_decomposition["spec_id"]) != {primary_spec_id}
            or set(county_year_decomposition["region_pair_id"].astype(str))
            != primary_pairs
            or not county_year_decomposition["county_fips"]
            .str.fullmatch(r"\d{5}")
            .all()
        ):
            raise ValueError("county-year decomposition violates its primary-sample key")
        require_finite(
            county_year_decomposition,
            [
                "observed_outcome",
                "observed_treatment",
                "estimation_weight",
                "within_outcome",
                "within_treatment",
                "model_residual",
                "requested_positions",
                "certified_positions",
                "applications",
                "balanced_linked_worksite_businesses",
                "baseline_farm_employment",
                "coefficient_numerator",
                "coefficient_contribution",
                "model_score",
                "treatment_sum_squares",
                "treatment_leverage",
                "absolute_observation_score_share",
            ],
            "county-year decomposition",
        )
        reconstructed_outcome = (
            county_year_decomposition["certified_positions"]
            / county_year_decomposition["baseline_farm_employment"]
        )
        if (
            not np.allclose(
                reconstructed_outcome,
                county_year_decomposition["observed_outcome"],
                atol=1e-12,
                rtol=1e-10,
            )
            or not np.isclose(
                county_year_decomposition["coefficient_contribution"].sum(),
                primary_estimate,
                atol=1e-10,
                rtol=1e-9,
            )
            or not np.isclose(
                county_year_decomposition["model_score"].sum(),
                0,
                atol=1e-8,
                rtol=0,
            )
            or not np.isclose(
                county_year_decomposition["treatment_leverage"].sum(),
                1,
                atol=1e-10,
                rtol=0,
            )
            or not np.isclose(
                county_year_decomposition["absolute_observation_score_share"].sum(),
                1,
                atol=1e-10,
                rtol=0,
            )
        ):
            raise ValueError("county-year decomposition does not reproduce the model")
        change_columns = [
            "observed_treatment_annual_change",
            "requested_positions_annual_change",
            "certified_positions_annual_change",
            "applications_annual_change",
            "balanced_linked_worksite_businesses_annual_change",
        ]
        require_columns(
            county_year_decomposition,
            set(change_columns),
            "county-year annual-change decomposition",
        )
        first_year = county_year_decomposition["year"].eq(config["policy_start_year"])
        if (
            county_year_decomposition.loc[first_year, change_columns].notna().any().any()
            or not np.isfinite(
                county_year_decomposition.loc[~first_year, change_columns]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy()
            ).all()
        ):
            raise ValueError("county-year annual changes do not follow the panel timing")

        county_decomposition = pd.read_csv(
            county_decomposition_path,
            dtype=decomposition_dtypes,
        )
        if (
            county_decomposition.empty
            or county_decomposition["county_fips"].duplicated().any()
            or len(county_decomposition) != int(primary_row["n_counties"])
            or set(county_decomposition["county_fips"])
            != set(county_year_decomposition["county_fips"])
        ):
            raise ValueError("county decomposition does not cover the primary sample")
        require_finite(
            county_decomposition,
            [
                "n_years",
                "coefficient_contribution",
                "model_score",
                "treatment_sum_squares",
                "treatment_leverage",
                "absolute_observation_score_share",
                "absolute_aggregated_score_share",
            ],
            "county decomposition",
        )
        if (
            not (
                county_decomposition["n_years"]
                == config["policy_end_year"] - config["policy_start_year"] + 1
            ).all()
            or not np.isclose(
                county_decomposition["coefficient_contribution"].sum(),
                primary_estimate,
                atol=1e-10,
                rtol=1e-9,
            )
            or not np.isclose(
                county_decomposition["treatment_leverage"].sum(),
                1,
                atol=1e-10,
                rtol=0,
            )
            or not np.isclose(
                county_decomposition["absolute_aggregated_score_share"].sum(),
                1,
                atol=1e-10,
                rtol=0,
            )
        ):
            raise ValueError("county decomposition does not aggregate exactly")

        stratum_region_year = pd.read_csv(
            stratum_region_year_decomposition_path,
            dtype=decomposition_dtypes,
        )
        if (
            stratum_region_year.empty
            or stratum_region_year.duplicated(
                ["region_pair_id", "stratum_id", "aewr_region_id", "year"]
            ).any()
            or set(stratum_region_year["year"])
            != set(range(config["policy_start_year"], config["policy_end_year"] + 1))
            or not (
                stratum_region_year.groupby(["stratum_id", "year"])[
                    "aewr_region_id"
                ].nunique()
                == 2
            ).all()
            or (
                stratum_region_year.groupby("stratum_id")["region_pair_id"]
                .nunique()
                != 1
            ).any()
        ):
            raise ValueError("stratum-region-year decomposition has an invalid registry")
        require_finite(
            stratum_region_year,
            [
                "n_counties",
                "observed_treatment",
                "mean_outcome",
                "total_requested_positions",
                "total_certified_positions",
                "total_applications",
                "summed_balanced_linked_worksite_businesses",
                "total_baseline_farm_employment",
                "coefficient_contribution",
                "model_score",
                "treatment_sum_squares",
                "treatment_leverage",
                "absolute_aggregated_score_share",
            ],
            "stratum-region-year decomposition",
        )
        if (
            not np.isclose(
                stratum_region_year["coefficient_contribution"].sum(),
                primary_estimate,
                atol=1e-10,
                rtol=1e-9,
            )
            or not np.isclose(
                stratum_region_year["treatment_leverage"].sum(),
                1,
                atol=1e-10,
                rtol=0,
            )
            or not np.isclose(
                stratum_region_year["absolute_aggregated_score_share"].sum(),
                1,
                atol=1e-10,
                rtol=0,
            )
        ):
            raise ValueError("stratum-region-year decomposition does not aggregate exactly")

        region_decomposition = pd.read_csv(
            region_decomposition_path,
            dtype={"aewr_region_id": "string"},
        )
        if (
            region_decomposition.empty
            or region_decomposition["aewr_region_id"].duplicated().any()
            or set(region_decomposition["aewr_region_id"].astype(str))
            != primary_regions
        ):
            raise ValueError("region decomposition has the wrong AEWR registry")
        require_finite(
            region_decomposition,
            [
                "n_counties",
                "n_observations",
                "total_requested_positions",
                "total_certified_positions",
                "total_applications",
                "mean_balanced_linked_worksite_businesses",
                "active_county_year_share",
                "mean_outcome",
                "residual_standard_deviation",
                "coefficient_contribution",
                "model_score",
                "treatment_sum_squares",
                "treatment_leverage",
                "certified_position_share",
                "absolute_aggregated_score_share",
            ],
            "region decomposition",
        )
        if (
            not np.isclose(
                region_decomposition["coefficient_contribution"].sum(),
                primary_estimate,
                atol=1e-10,
                rtol=1e-9,
            )
            or not np.isclose(
                region_decomposition["certified_position_share"].sum(),
                1,
                atol=1e-10,
                rtol=0,
            )
            or not np.isclose(
                region_decomposition["absolute_aggregated_score_share"].sum(),
                1,
                atol=1e-10,
                rtol=0,
            )
        ):
            raise ValueError("region decomposition does not aggregate exactly")

        border_pair_decomposition = pd.read_csv(
            border_pair_decomposition_path,
            dtype={"region_pair_id": "string"},
        )
        if (
            border_pair_decomposition.empty
            or border_pair_decomposition["region_pair_id"].duplicated().any()
            or set(border_pair_decomposition["region_pair_id"].astype(str))
            != primary_pairs
        ):
            raise ValueError("border-pair decomposition has the wrong registry")
        require_finite(
            border_pair_decomposition,
            [
                "n_counties",
                "n_observations",
                "total_requested_positions",
                "total_certified_positions",
                "total_applications",
                "mean_balanced_linked_worksite_businesses",
                "active_county_year_share",
                "mean_outcome",
                "residual_standard_deviation",
                "coefficient_contribution",
                "model_score",
                "treatment_sum_squares",
                "treatment_leverage",
                "certified_position_share",
                "absolute_aggregated_score_share",
            ],
            "border-pair decomposition",
        )
        influence_by_pair = influence.set_index("region_pair_id")
        decomposition_by_pair = border_pair_decomposition.set_index("region_pair_id")
        decomposition_by_pair = decomposition_by_pair.loc[influence_by_pair.index]
        if (
            not np.allclose(
                decomposition_by_pair["model_score"],
                influence_by_pair["cluster_score"],
                atol=1e-10,
                rtol=1e-9,
            )
            or not np.allclose(
                decomposition_by_pair["treatment_leverage"],
                influence_by_pair["treatment_leverage"],
                atol=1e-10,
                rtol=1e-9,
            )
            or not np.isclose(
                border_pair_decomposition["coefficient_contribution"].sum(),
                primary_estimate,
                atol=1e-10,
                rtol=1e-9,
            )
            or not np.isclose(
                border_pair_decomposition["certified_position_share"].sum(),
                1,
                atol=1e-10,
                rtol=0,
            )
            or not np.isclose(
                border_pair_decomposition["absolute_aggregated_score_share"].sum(),
                1,
                atol=1e-10,
                rtol=0,
            )
        ):
            raise ValueError(
                "border-pair decomposition does not reproduce influence outputs"
            )

        obsolete_outputs = (
            FIGURES / "border_discontinuity_national_clusters.png",
            FIGURES / "border_discontinuity_ring_leave_one_aewr_out.png",
            TABLES / "border_discontinuity_ring_leave_one_aewr_out.csv",
            TABLES / "border_discontinuity_ring_cluster_jackknife.csv",
            TABLES / "border_discontinuity_ring_cluster_influence.csv",
        )
        obsolete = [path for path in obsolete_outputs if path.exists()]
        if obsolete:
            raise ValueError(
                "obsolete region-cluster output is still retained: "
                + ", ".join(str(path) for path in obsolete)
            )
        redesign_tables = {}
        for name in (
            "estimates", "sample_audit", "support", "region_deletion", "balance",
            "placebo", "fwl", "noise_diagnosis", "assignment_audit", "comparable_samples", "power",
            "power_calibration", "mde", "simulation_diagnostics", "bootstrap_power", "decision",
        ):
            path = TABLES / f"border_discontinuity_redesign_{name}.csv"
            if not path.exists() or path.stat().st_size == 0:
                raise ValueError(f"missing redesigned output: {path}")
            redesign_tables[name] = pd.read_csv(path, dtype={
                "county_fips": "string", "aewr_region_id": "string",
                "region_pair_id": "string", "excluded_region_id": "string",
            })
        validate_redesign_results(config, redesign_tables, estimates, memberships)
        validate_assignment_audit(config, redesign_tables["assignment_audit"], full_panel)
        validate_redesign_power(config, redesign_tables, redesign_tables["estimates"])
        validate_bootstrap_power(config, redesign_tables, redesign_tables["estimates"])
        for path in (
            INTERMEDIATE / "border_discontinuity_redesign_simulation_inputs.rds",
            ROOT / "outputs" / "reports" / "border_redesign.md",
            FIGURES / "border_discontinuity_redesign_precision.png",
        ):
            if not path.exists() or path.stat().st_size == 0:
                raise ValueError(f"missing retained redesign product: {path}")
        provenance_path = ROOT / "outputs" / "reports" / "border_discontinuity_redesign_provenance.json"
        provenance = json.loads(provenance_path.read_text())
        validate_redesign_decisions(config, redesign_tables, provenance)
        if provenance["design_id"] != config["design_id"] or not provenance["files"]:
            raise ValueError("redesign provenance does not identify the current registry and files")
        for relative, expected_hash in provenance["files"].items():
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
                raise ValueError("redesign provenance contains an invalid repository file")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise ValueError(f"redesign provenance is stale for {relative}")
        print(
            "Validated AEWR-border ring artifacts: "
            f"{len(memberships):,} memberships, {len(estimates)} models, "
            f"{len(primary_pairs)} legacy leave-one-border-pair-out fits, "
            f"{len(redesign_tables['estimates'])} redesigned comparisons and power artifacts. "
            "Artifact validity does not establish causal identification or informative precision."
        )

    return main, os, sys


@app.cell
def _(main, os, sys):
    inputs_only = os.environ.get("H2A_BORDER_INPUTS_ONLY", "0") == "1"
    try:
        main(inputs_only=inputs_only)
    except Exception as error:
        print(f"Border-discontinuity validation failed: {error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    app.run()
