import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    # ruff: noqa: N999
    """Construct state-border county pairs for the statewide-AEWR design."""

    import json
    import os
    import tempfile
    from collections import defaultdict
    from pathlib import Path
    from runpy import run_path

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from h2a.geography import (
        harmonize_county_fips_2010,
        normalize_geo_code,
        state_from_county_fips,
    )
    from h2a.paths import INTERMEDIATE, PROCESSED, ROOT

    DESIGN_DIR = ROOT / "code" / "designs" / "border_discontinuity"
    geography = run_path(DESIGN_DIR / "geography.py")
    build_graph = geography["build_graph"]
    pair_identifier = geography["pair_identifier"]
    agricultural_county_eligibility = geography["agricultural_county_eligibility"]
    REGISTRY = json.loads((DESIGN_DIR / "statewide_design.json").read_text())
    FEATURE_PATH = INTERMEDIATE / "border_discontinuity_county_features.parquet"
    MEMBERSHIP_PATH = INTERMEDIATE / "border_statewide_memberships.parquet"
    PAIR_AUDIT_PATH = INTERMEDIATE / "border_statewide_pair_audit.parquet"

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

    return (
        FEATURE_PATH,
        INTERMEDIATE,
        MEMBERSHIP_PATH,
        PAIR_AUDIT_PATH,
        PROCESSED,
        REGISTRY,
        agricultural_county_eligibility,
        build_graph,
        defaultdict,
        harmonize_county_fips_2010,
        normalize_geo_code,
        pair_identifier,
        pd,
        pq,
        state_from_county_fips,
        write_parquet_atomic,
    )


@app.cell
def _(
    FEATURE_PATH,
    INTERMEDIATE,
    PROCESSED,
    REGISTRY,
    agricultural_county_eligibility,
    build_graph,
    harmonize_county_fips_2010,
    normalize_geo_code,
    pd,
    pq,
    state_from_county_fips,
):
    def normalize_county(values: pd.Series) -> pd.Series:
        normalized = values.map(harmonize_county_fips_2010)
        if normalized.isna().any():
            raise ValueError("county identifiers must be valid 2010 FIPS strings")
        return normalized.astype("string")

    panel = pq.read_table(
        PROCESSED / "county_year_panel.parquet",
        columns=["county_fips", "aewr_region_id"],
    ).to_pandas()
    panel["county_fips"] = normalize_county(panel["county_fips"])
    panel["aewr_region_id"] = panel["aewr_region_id"].map(
        lambda value: normalize_geo_code(value, "aewr_region_id")
    )
    counties = panel.drop_duplicates()
    if counties["county_fips"].duplicated().any() or counties["aewr_region_id"].isna().any():
        raise ValueError("former AEWR-region membership must be unique and complete")
    counties["state_fips"] = counties["county_fips"].map(state_from_county_fips).astype("string")
    excluded = set(REGISTRY["geography"]["excluded_states"])
    counties = counties[~counties["state_fips"].isin(excluded)].set_index("county_fips")

    screen = REGISTRY["geography"]["agricultural_screen"]
    features = pq.read_table(FEATURE_PATH).to_pandas()
    features["county_fips"] = normalize_county(features["county_fips"])
    features = features.set_index("county_fips").reindex(counties.index)
    eligible = agricultural_county_eligibility(
        features[[
            "cropland_acres_2007",
            "mean_farm_employment_2008_2010",
            "farm_employment_years_observed",
            "land_area_acres_2010",
        ]],
        minimum_cropland_share=screen["minimum_county_cropland_share"],
        minimum_farm_employment_per_100_square_miles=screen[
            "minimum_county_farm_employment_per_100_square_miles"
        ],
        minimum_measurement_coverage=screen["minimum_county_measurement_coverage"],
        expected_farm_employment_years=screen["farm_employment_years"],
    )
    counties["agricultural_eligible"] = eligible.fillna(False).astype(bool)

    adjacency = pq.read_table(
        INTERMEDIATE / "county_adjacency2010.parquet",
        columns=["county_fips", "neighbor_county_fips"],
    ).to_pandas()
    adjacency["county_fips"] = normalize_county(adjacency["county_fips"])
    adjacency["neighbor_county_fips"] = normalize_county(adjacency["neighbor_county_fips"])
    _neighbors, edges = build_graph(
        counties.index.tolist(),
        zip(adjacency["county_fips"], adjacency["neighbor_county_fips"], strict=True),
    )
    return counties, edges


@app.cell
def _(REGISTRY, counties, defaultdict, edges, pair_identifier, pd):
    def assign_pairs(spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        state = counties["state_fips"].to_dict()
        region = counties["aewr_region_id"].to_dict()
        eligible = counties["agricultural_eligible"].to_dict()
        qualifying: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for left, right in edges:
            if state[left] == state[right]:
                continue
            if spec["within_former_aewr_region_only"] and region[left] != region[right]:
                continue
            if spec["screened"] and not (eligible[left] and eligible[right]):
                continue
            pair = pair_identifier(state[left], state[right])
            qualifying[left][pair].add(right)
            qualifying[right][pair].add(left)
        active = {pair for options in qualifying.values() for pair in options}
        while True:
            assignment = {}
            for county, options in qualifying.items():
                candidates = [(len(nbrs), pair) for pair, nbrs in options.items() if pair in active]
                if candidates:
                    best = max(count for count, _ in candidates)
                    assignment[county] = min(pair for count, pair in candidates if count == best)
            sides: dict[str, set[str]] = defaultdict(set)
            for county, pair in assignment.items():
                sides[pair].add(state[county])
            supported = {pair for pair in active if len(sides[pair]) == 2}
            if supported == active:
                break
            active = supported
        members = pd.DataFrame(
            [
                {
                    "spec_id": spec["spec_id"],
                    "role": spec["role"],
                    "county_fips": county,
                    "state_fips": state[county],
                    "former_aewr_region_id": region[county],
                    "region_pair_id": pair,
                    "stratum_id": pair,
                    "qualifying_cross_border_neighbors": len(qualifying[county][pair]),
                    "agricultural_eligible": bool(eligible[county]),
                }
                for county, pair in sorted(assignment.items())
            ]
        )
        if members.empty:
            raise ValueError(f"{spec['spec_id']} has no supported state-border pairs")
        pair_audit = (
            members.groupby("region_pair_id")
            .agg(
                counties=("county_fips", "size"),
                states=("state_fips", "nunique"),
                former_regions=("former_aewr_region_id", "nunique"),
            )
            .reset_index()
            .assign(spec_id=spec["spec_id"])
        )
        pair_audit["same_former_aewr_region"] = pair_audit["former_regions"] == 1
        if (pair_audit["states"] != 2).any():
            raise ValueError("every retained pair must have counties in both states")
        if spec["within_former_aewr_region_only"] and not pair_audit["same_former_aewr_region"].all():
            raise ValueError("within-region spec retained a cross-region pair")
        return members, pair_audit

    results = [assign_pairs(spec) for spec in REGISTRY["geography"]["specs"]]
    memberships = pd.concat([members for members, _ in results], ignore_index=True)
    pair_audit = pd.concat([audit for _, audit in results], ignore_index=True)
    if memberships.duplicated(["spec_id", "county_fips"]).any():
        raise ValueError("state-border assignment must be disjoint within spec")
    for column in ["county_fips", "state_fips", "former_aewr_region_id", "region_pair_id", "stratum_id"]:
        memberships[column] = memberships[column].astype("string")
    return memberships, pair_audit


@app.cell
def _(MEMBERSHIP_PATH, PAIR_AUDIT_PATH, memberships, pair_audit, write_parquet_atomic):
    write_parquet_atomic(memberships, MEMBERSHIP_PATH)
    write_parquet_atomic(pair_audit, PAIR_AUDIT_PATH)
    summary = memberships.groupby("spec_id").agg(
        counties=("county_fips", "size"),
        pairs=("region_pair_id", "nunique"),
        states=("state_fips", "nunique"),
    )
    print(summary)


if __name__ == "__main__":
    app.run()
