import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    # ruff: noqa: N999
    """Construct the declared AEWR-border ring geographies."""

    import json
    import os
    import tempfile
    from collections import defaultdict
    from pathlib import Path
    from runpy import run_path

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scipy.spatial.distance import cdist

    from h2a.geography import harmonize_county_fips_2010, normalize_geo_code
    from h2a.paths import INTERMEDIATE, PROCESSED, ROOT

    DESIGN_DIR = ROOT / "code" / "designs" / "border_discontinuity"
    geography = run_path(DESIGN_DIR / "geography.py")
    add_graph_metrics = geography["add_graph_metrics"]
    build_graph = geography["build_graph"]
    connected_components = geography["connected_components"]
    pair_identifier = geography["pair_identifier"]
    same_region_ring_distances = geography["same_region_ring_distances"]
    screen_agricultural_county_pairs = geography[
        "screen_agricultural_county_pairs"
    ]
    construct_ring_memberships = geography["construct_ring_memberships"]
    agricultural_county_eligibility = geography["agricultural_county_eligibility"]
    FEATURE_PATH = INTERMEDIATE / "border_discontinuity_county_features.parquet"
    MEMBERSHIP_PATH = (
        INTERMEDIATE / "border_discontinuity_geography_memberships.parquet"
    )
    DIAGNOSTIC_PATH = (
        INTERMEDIATE / "border_discontinuity_geography_diagnostics.parquet"
    )
    EXCLUSION_PATH = (
        INTERMEDIATE / "border_discontinuity_assignment_exclusions.parquet"
    )
    AGRICULTURAL_AUDIT_PATH = (
        INTERMEDIATE / "border_discontinuity_agricultural_border_audit.parquet"
    )

    def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


    def normalize_county(values: pd.Series) -> pd.Series:
        normalized = values.map(harmonize_county_fips_2010)
        if normalized.isna().any():
            raise ValueError("county adjacency contains malformed county identifiers")
        return normalized.astype("string")


    def load_regions() -> dict[str, str]:
        panel = pq.read_table(
            PROCESSED / "county_year_panel.parquet",
            columns=["county_fips", "aewr_region_id"],
        ).to_pandas()
        panel["county_fips"] = normalize_county(panel["county_fips"])
        panel["aewr_region_id"] = panel["aewr_region_id"].map(
            lambda value: normalize_geo_code(value, "aewr_region_id")
        )
        if panel["aewr_region_id"].isna().any():
            raise ValueError("shared panel contains missing AEWR-region identifiers")
        unique = panel.drop_duplicates()
        if unique["county_fips"].duplicated().any():
            raise ValueError("AEWR-region membership must be time invariant by county")
        return dict(zip(unique["county_fips"], unique["aewr_region_id"], strict=True))


    def load_graph(
        counties: list[str],
    ) -> tuple[dict[str, set[str]], tuple[tuple[str, str], ...]]:
        adjacency = pq.read_table(
            INTERMEDIATE / "county_adjacency2010.parquet",
            columns=["county_fips", "neighbor_county_fips"],
        ).to_pandas()
        adjacency["county_fips"] = normalize_county(adjacency["county_fips"])
        adjacency["neighbor_county_fips"] = normalize_county(
            adjacency["neighbor_county_fips"]
        )
        neighbors, edges = build_graph(
            counties,
            zip(
                adjacency["county_fips"],
                adjacency["neighbor_county_fips"],
                strict=True,
            ),
        )
        if len(connected_components(neighbors, counties)) != 1:
            raise ValueError("analysis county adjacency graph must be connected")
        return neighbors, edges


    def cross_border_edges(
        edges: tuple[tuple[str, str], ...], regions: dict[str, str]
    ) -> dict[str, list[tuple[str, str]]]:
        result: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for left, right in edges:
            if regions[left] != regions[right]:
                result[pair_identifier(regions[left], regions[right])].append((left, right))
        if not result:
            raise ValueError("county graph has no cross-AEWR adjacency edges")
        return dict(sorted(result.items()))


    def build_ring_candidates(
        neighbors: dict[str, set[str]],
        regions: dict[str, str],
        pair_edges: dict[str, list[tuple[str, str]]],
        features: pd.DataFrame,
        maximum_ring: int,
        caliper_quantile: float,
    ) -> tuple[pd.DataFrame, float]:
        position = {county: index for index, county in enumerate(features.index)}
        matrix = features.to_numpy()
        adjacent_distances = np.asarray(
            [
                np.linalg.norm(matrix[position[left]] - matrix[position[right]])
                for edges in pair_edges.values()
                for left, right in edges
            ]
        )
        caliper = float(np.quantile(adjacent_distances, caliper_quantile))
        rows: list[dict[str, object]] = []
        for pair_id, edges in pair_edges.items():
            pair_regions = sorted(
                {regions[county] for edge in edges for county in edge}, key=int
            )
            if len(pair_regions) != 2:
                raise ValueError(f"{pair_id} does not identify exactly two regions")
            cores = {region: set() for region in pair_regions}
            for left, right in edges:
                cores[regions[left]].add(left)
                cores[regions[right]].add(right)
            for region in pair_regions:
                opposite = next(value for value in pair_regions if value != region)
                distances = same_region_ring_distances(
                    neighbors, regions, region, cores[region], maximum=maximum_ring
                )
                candidates = sorted(distances)
                opposite_positions = [position[county] for county in sorted(cores[opposite])]
                feature_distances = cdist(
                    matrix[[position[county] for county in candidates]],
                    matrix[opposite_positions],
                ).min(axis=1)
                for county, feature_distance in zip(
                    candidates, feature_distances, strict=True
                ):
                    ring_distance = distances[county]
                    rows.append(
                        {
                            "county_fips": county,
                            "aewr_region_id": region,
                            "region_pair_id": pair_id,
                            "ring_distance": ring_distance,
                            "feature_distance": float(feature_distance),
                            "caliper_threshold": caliper,
                            "caliper_eligible": bool(
                                ring_distance == 0 or feature_distance <= caliper
                            ),
                        }
                    )
        return pd.DataFrame(rows), caliper


    def main() -> None:
        config = json.loads((DESIGN_DIR / "design.json").read_text())
        regions = load_regions()
        counties = sorted(regions)
        neighbors, edges = load_graph(counties)
        pair_edges = cross_border_edges(edges, regions)
        unfiltered_pair_count = len(pair_edges)

        feature_artifact = pq.read_table(FEATURE_PATH).to_pandas()
        feature_artifact["county_fips"] = normalize_county(
            feature_artifact["county_fips"]
        )
        scaled_columns = sorted(
            column for column in feature_artifact if column.startswith("scaled__")
        )
        if not scaled_columns:
            raise ValueError("feature artifact contains no scaled features")
        features = feature_artifact.set_index("county_fips")[scaled_columns].reindex(
            counties
        )
        if features.isna().any().any() or not np.isfinite(features.to_numpy()).all():
            raise ValueError("scaled features must be finite for every panel county")

        support_columns = [
            "cropland_acres_2007",
            "mean_farm_employment_2008_2010",
            "farm_employment_years_observed",
            "land_area_acres_2010",
        ]
        missing_support = sorted(
            set(support_columns).difference(feature_artifact.columns)
        )
        if missing_support:
            raise ValueError(
                "feature artifact lacks agricultural support: "
                + ", ".join(missing_support)
            )
        agricultural_filter = config["agricultural_county_pair_filter"]
        if agricultural_filter["filter_unit"] != "cross_border_county_pair":
            raise ValueError("agricultural support must screen county pairs")
        expected_farm_years = (
            int(agricultural_filter["farm_employment_end_year"])
            - int(agricultural_filter["farm_employment_start_year"])
            + 1
        )
        agricultural_support = feature_artifact.set_index("county_fips")[
            support_columns
        ]
        pair_edges, agricultural_audit = screen_agricultural_county_pairs(
            pair_edges,
            regions,
            agricultural_support,
            minimum_cropland_share=float(
                agricultural_filter["minimum_county_cropland_share"]
            ),
            minimum_farm_employment_per_100_square_miles=float(
                agricultural_filter[
                    "minimum_county_farm_employment_per_100_square_miles"
                ]
            ),
            minimum_measurement_coverage=float(
                agricultural_filter["minimum_county_measurement_coverage"]
            ),
            expected_farm_employment_years=expected_farm_years,
        )
        agricultural_audit["cropland_year"] = int(
            agricultural_filter["cropland_year"]
        )
        agricultural_audit["farm_employment_start_year"] = int(
            agricultural_filter["farm_employment_start_year"]
        )
        agricultural_audit["farm_employment_end_year"] = int(
            agricultural_filter["farm_employment_end_year"]
        )
        agricultural_audit["screen_stage"] = (
            "cross_border_edges_before_ring_expansion_and_disjoint_assignment"
        )

        ring_specs = config["geography_specs"]
        if any(value["family"] != "ring" for value in ring_specs):
            raise ValueError("border design may declare only ring geographies")
        maximum_ring = max(int(value) for value in config["ring_bandwidths"])
        candidates, caliper = build_ring_candidates(
            neighbors,
            regions,
            pair_edges,
            features,
            maximum_ring,
            float(config["caliper_quantile"]),
        )
        county_eligible = agricultural_county_eligibility(
            agricultural_support,
            minimum_cropland_share=float(
                agricultural_filter["minimum_county_cropland_share"]
            ),
            minimum_farm_employment_per_100_square_miles=float(
                agricultural_filter[
                    "minimum_county_farm_employment_per_100_square_miles"
                ]
            ),
            minimum_measurement_coverage=float(
                agricultural_filter["minimum_county_measurement_coverage"]
            ),
            expected_farm_employment_years=expected_farm_years,
        )
        ring, ring_exclusions = construct_ring_memberships(
            candidates,
            neighbors,
            edges,
            regions,
            pair_edges,
            ring_specs,
            agricultural_eligible=county_eligible,
            features=features,
            matching_source_spec_id=config.get("redesign", {}).get("candidate_spec_id"),
        )
        ring, ring_diagnostics = add_graph_metrics(
            ring, neighbors, edges, regions, pair_edges
        )
        memberships = ring
        memberships["aewr_region_id"] = memberships["aewr_region_id"].astype("string")
        memberships["county_fips"] = memberships["county_fips"].astype("string")
        memberships = memberships.sort_values(
            ["spec_id", "stratum_id", "county_fips"], kind="stable"
        ).reset_index(drop=True)
        if memberships.duplicated(["spec_id", "stratum_id", "county_fips"]).any():
            raise ValueError("geography memberships violate their declared key")
        disjoint = memberships.loc[memberships["membership_mode"].eq("disjoint")]
        if disjoint.duplicated(["spec_id", "county_fips"]).any():
            raise ValueError("disjoint geography assigns a county more than once")

        diagnostics = pd.DataFrame(ring_diagnostics)
        diagnostics["caliper_quantile"] = float(config["caliper_quantile"])
        diagnostics["caliper_threshold"] = caliper
        diagnostics = diagnostics.sort_values(
            ["spec_id", "retained", "stratum_id"], kind="stable"
        ).reset_index(drop=True)

        exclusions = pd.DataFrame(ring_exclusions)
        exclusions["county_fips"] = exclusions["county_fips"].astype("string")
        exclusions = exclusions.sort_values(
            ["spec_id", "reason", "county_fips", "region_pair_id"],
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)

        write_parquet_atomic(memberships, MEMBERSHIP_PATH)
        write_parquet_atomic(diagnostics, DIAGNOSTIC_PATH)
        write_parquet_atomic(exclusions, EXCLUSION_PATH)
        write_parquet_atomic(agricultural_audit, AGRICULTURAL_AUDIT_PATH)
        print(
            f"Wrote {len(memberships):,} memberships across "
            f"{memberships['spec_id'].nunique()} specifications; "
            f"agricultural screen retained {len(pair_edges):,} of "
            f"{unfiltered_pair_count:,} border pairs; caliper={caliper:.4f}."
        )

    return (main,)


@app.cell
def _(main):
    main()


if __name__ == "__main__":
    app.run()
