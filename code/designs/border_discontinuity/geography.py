"""Geographic construction helpers for the AEWR-border ring experiment."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from functools import cache

import numpy as np
import pandas as pd


def build_graph(
    counties: Iterable[str], edges: Iterable[tuple[str, str]]
) -> tuple[dict[str, set[str]], tuple[tuple[str, str], ...]]:
    """Return neighbor sets and canonical edges for an undirected graph."""
    county_set = set(counties)
    neighbors: dict[str, set[str]] = {county: set() for county in county_set}
    clean_edges: set[tuple[str, str]] = set()
    for left, right in edges:
        if left not in county_set or right not in county_set or left == right:
            continue
        edge = tuple(sorted((left, right)))
        clean_edges.add(edge)
        neighbors[edge[0]].add(edge[1])
        neighbors[edge[1]].add(edge[0])
    return neighbors, tuple(sorted(clean_edges))


def connected_components(
    neighbors: dict[str, set[str]], members: Iterable[str]
) -> list[set[str]]:
    member_set = set(members)
    unseen = set(member_set)
    components: list[set[str]] = []
    while unseen:
        start = min(unseen)
        component = {start}
        unseen.remove(start)
        queue = deque([start])
        while queue:
            county = queue.popleft()
            for neighbor in neighbors[county].intersection(unseen):
                unseen.remove(neighbor)
                component.add(neighbor)
                queue.append(neighbor)
        components.append(component)
    return sorted(components, key=lambda values: min(values))


def graph_distances(
    neighbors: dict[str, set[str]],
    members: set[str],
    seeds: Iterable[str],
    maximum: int | None = None,
) -> dict[str, int]:
    distances = {seed: 0 for seed in seeds if seed in members}
    queue = deque(sorted(distances))
    while queue:
        county = queue.popleft()
        if maximum is not None and distances[county] >= maximum:
            continue
        for neighbor in sorted(neighbors[county].intersection(members)):
            if neighbor not in distances:
                distances[neighbor] = distances[county] + 1
                queue.append(neighbor)
    return distances


def graph_diameter(neighbors: dict[str, set[str]], members: set[str]) -> int:
    if not members:
        return 0
    diameter = 0
    for county in members:
        distances = graph_distances(neighbors, members, [county])
        if len(distances) != len(members):
            raise ValueError("graph diameter requested for a disconnected set")
        diameter = max(diameter, max(distances.values()))
    return diameter


def pair_identifier(left_region: str, right_region: str) -> str:
    left, right = sorted((int(left_region), int(right_region)))
    return f"{left:02d}_{right:02d}"


def screen_agricultural_county_pairs(
    pair_edges: dict[str, list[tuple[str, str]]],
    regions: dict[str, str],
    support: pd.DataFrame,
    *,
    minimum_cropland_share: float,
    minimum_farm_employment_per_100_square_miles: float,
    minimum_measurement_coverage: float,
    expected_farm_employment_years: int,
) -> tuple[dict[str, list[tuple[str, str]]], pd.DataFrame]:
    """Retain cross-border county pairs with agricultural support on both ends."""
    required = {
        "cropland_acres_2007",
        "mean_farm_employment_2008_2010",
        "farm_employment_years_observed",
        "land_area_acres_2010",
    }
    missing = sorted(required.difference(support.columns))
    if missing:
        raise ValueError(f"agricultural support is missing: {', '.join(missing)}")
    if support.index.has_duplicates:
        raise ValueError("agricultural support must be county-unique")
    if expected_farm_employment_years <= 0:
        raise ValueError("expected farm-employment years must be positive")

    retained_edges: dict[str, list[tuple[str, str]]] = {}
    audit_rows: list[dict[str, object]] = []
    for pair_id, edges in sorted(pair_edges.items()):
        eligible_edges: list[tuple[str, str]] = []
        for left, right in sorted(edges):
            if left not in support.index or right not in support.index:
                raise ValueError(f"{left}--{right} lacks agricultural support")
            if pair_identifier(regions[left], regions[right]) != pair_id:
                raise ValueError(f"{left}--{right} has an inconsistent region pair")

            edge_values: dict[str, object] = {
                "region_pair_id": pair_id,
                "county_pair_id": f"{left}__{right}",
                "left_county_fips": left,
                "right_county_fips": right,
                "left_aewr_region_id": regions[left],
                "right_aewr_region_id": regions[right],
            }
            endpoint_eligibility: list[bool] = []
            for label, county in (("left", left), ("right", right)):
                row = support.loc[county]
                land_area = float(row["land_area_acres_2010"])
                observed_farm_years = float(row["farm_employment_years_observed"])
                cropland = pd.to_numeric(
                    pd.Series([row["cropland_acres_2007"]]), errors="coerce"
                ).iloc[0]
                farm_employment = pd.to_numeric(
                    pd.Series([row["mean_farm_employment_2008_2010"]]),
                    errors="coerce",
                ).iloc[0]
                if not np.isfinite(land_area) or land_area <= 0:
                    raise ValueError(f"{county} has invalid county land area")
                if (
                    not np.isfinite(observed_farm_years)
                    or observed_farm_years < 0
                    or observed_farm_years > expected_farm_employment_years
                ):
                    raise ValueError(f"{county} has invalid farm-employment coverage")
                if (pd.notna(cropland) and cropland < 0) or (
                    pd.notna(farm_employment) and farm_employment < 0
                ):
                    raise ValueError(f"{county} has negative support values")

                cropland_coverage = float(pd.notna(cropland))
                farm_employment_coverage = (
                    observed_farm_years / expected_farm_employment_years
                )
                cropland_share = (
                    float(cropland) / land_area if pd.notna(cropland) else np.nan
                )
                farm_employment_density = (
                    float(farm_employment) * 64_000 / land_area
                    if pd.notna(farm_employment)
                    else np.nan
                )
                cropland_coverage_pass = (
                    cropland_coverage >= minimum_measurement_coverage
                )
                farm_employment_coverage_pass = (
                    farm_employment_coverage >= minimum_measurement_coverage
                )
                cropland_share_pass = bool(
                    pd.notna(cropland_share)
                    and cropland_share >= minimum_cropland_share
                )
                farm_employment_density_pass = bool(
                    pd.notna(farm_employment_density)
                    and farm_employment_density
                    >= minimum_farm_employment_per_100_square_miles
                )
                endpoint_eligible = all(
                    (
                        cropland_coverage_pass,
                        farm_employment_coverage_pass,
                        cropland_share_pass,
                        farm_employment_density_pass,
                    )
                )
                endpoint_eligibility.append(endpoint_eligible)
                edge_values.update(
                    {
                        f"{label}_cropland_acres_2007": cropland,
                        f"{label}_land_area_acres_2010": land_area,
                        f"{label}_cropland_share_2007": cropland_share,
                        f"{label}_mean_annual_farm_employment_2008_2010": (
                            farm_employment
                        ),
                        f"{label}_farm_employment_per_100_square_miles": (
                            farm_employment_density
                        ),
                        f"{label}_cropland_coverage": cropland_coverage,
                        f"{label}_farm_employment_coverage": (
                            farm_employment_coverage
                        ),
                        f"{label}_cropland_coverage_pass": (
                            cropland_coverage_pass
                        ),
                        f"{label}_farm_employment_coverage_pass": (
                            farm_employment_coverage_pass
                        ),
                        f"{label}_cropland_share_pass": cropland_share_pass,
                        f"{label}_farm_employment_density_pass": (
                            farm_employment_density_pass
                        ),
                        f"{label}_county_eligible": endpoint_eligible,
                    }
                )

            county_pair_eligible = all(endpoint_eligibility)
            edge_values.update(
                {
                    "minimum_cropland_share": minimum_cropland_share,
                    "minimum_farm_employment_per_100_square_miles": (
                        minimum_farm_employment_per_100_square_miles
                    ),
                    "minimum_measurement_coverage": minimum_measurement_coverage,
                    "county_pair_eligible": county_pair_eligible,
                }
            )
            audit_rows.append(edge_values)
            if county_pair_eligible:
                eligible_edges.append((left, right))
        if eligible_edges:
            retained_edges[pair_id] = eligible_edges

    if not retained_edges:
        raise ValueError("agricultural screen removed every cross-border county pair")
    audit = pd.DataFrame(audit_rows).sort_values(
        ["region_pair_id", "county_pair_id"], kind="stable"
    )
    return dict(sorted(retained_edges.items())), audit.reset_index(drop=True)


def same_region_ring_distances(
    neighbors: dict[str, set[str]],
    regions: dict[str, str],
    region: str,
    seeds: set[str],
    maximum: int,
) -> dict[str, int]:
    eligible = {county for county, value in regions.items() if value == region}
    return graph_distances(neighbors, eligible, seeds, maximum=maximum)


def agricultural_county_eligibility(
    support: pd.DataFrame,
    *,
    minimum_cropland_share: float,
    minimum_farm_employment_per_100_square_miles: float,
    minimum_measurement_coverage: float,
    expected_farm_employment_years: int,
) -> pd.Series:
    """Apply the frozen endpoint rules to every potential ring member."""
    required = [
        "cropland_acres_2007",
        "mean_farm_employment_2008_2010",
        "farm_employment_years_observed",
        "land_area_acres_2010",
    ]
    if support.index.has_duplicates or not set(required).issubset(support):
        raise ValueError("county support must be unique and contain all screen fields")
    if expected_farm_employment_years <= 0:
        raise ValueError("expected farm-employment years must be positive")
    numeric = support[required].apply(pd.to_numeric, errors="coerce")
    land = numeric["land_area_acres_2010"]
    years = numeric["farm_employment_years_observed"]
    crop = numeric["cropland_acres_2007"]
    farm = numeric["mean_farm_employment_2008_2010"]
    if (
        not np.isfinite(land).all()
        or land.le(0).any()
        or not np.isfinite(years).all()
        or years.lt(0).any()
        or years.gt(expected_farm_employment_years).any()
        or crop.lt(0).any()
        or farm.lt(0).any()
    ):
        raise ValueError("county support contains invalid area, coverage, or values")
    return (
        crop.notna().astype(float).ge(minimum_measurement_coverage)
        & (years / expected_farm_employment_years).ge(minimum_measurement_coverage)
        & (crop / land).ge(minimum_cropland_share)
        & (farm * 64_000 / land).ge(minimum_farm_employment_per_100_square_miles)
    ).rename("all_member_agricultural_eligible")


def exclusion_records(
    frame: pd.DataFrame, spec_id: str, reason: str
) -> list[dict[str, object]]:
    """Keep the existing exclusion schema for each geography decision."""
    return [
        {
            "spec_id": spec_id,
            "county_fips": row.county_fips,
            "region_pair_id": row.region_pair_id,
            "reason": reason,
            "ring_distance": row.ring_distance,
            "feature_distance": row.feature_distance,
            "caliper_threshold": row.caliper_threshold,
        }
        for row in frame.itertuples(index=False)
    ]


def prune_retained_locality(
    frame: pd.DataFrame,
    neighbors: dict[str, set[str]],
    regions: dict[str, str],
    features: pd.DataFrame,
    qualifying_pair_edges: dict[str, list[tuple[str, str]]],
    *,
    maximum_ring: int,
    spec_id: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Prune until every member is local to an edge surviving in its pair.

    Paths stay within retained counties of the member's own region. Noncore
    members must match a surviving opposite core; eligible core endpoints keep
    the original caliper exemption. Recompute after removals destroy core edges.
    """
    current = frame.copy()
    exclusions: list[dict[str, object]] = []
    while not current.empty:
        kept: list[pd.DataFrame] = []
        for pair_id, group in current.groupby("region_pair_id", sort=True):
            members = set(group["county_fips"])
            surviving_edges = [
                (left, right)
                for left, right in qualifying_pair_edges.get(str(pair_id), [])
                if left in members and right in members
            ]
            if not surviving_edges:
                exclusions.extend(
                    exclusion_records(group, spec_id, "no_surviving_qualifying_edge")
                )
                continue
            seeds = {county for edge in surviving_edges for county in edge}
            distances: dict[str, int] = {}
            feature_distances: dict[str, float] = {}
            for region in sorted({regions[county] for county in members}):
                own_members = {c for c in members if regions[c] == region}
                own_seeds = own_members.intersection(seeds)
                opposite = sorted(c for c in seeds if regions[c] != region)
                distances.update(
                    graph_distances(
                        neighbors, own_members, own_seeds, maximum=maximum_ring
                    )
                )
                opposite_matrix = features.loc[opposite].to_numpy(dtype=float)
                for county in sorted(own_members):
                    delta = opposite_matrix - features.loc[county].to_numpy(dtype=float)
                    feature_distances[county] = float(
                        np.sqrt(np.square(delta).sum(axis=1)).min()
                    )
            updated = group.copy()
            new_ring = updated["county_fips"].map(distances)
            new_distance = updated["county_fips"].map(feature_distances)
            local = new_ring.notna() & new_ring.le(maximum_ring)
            caliper = updated["county_fips"].isin(seeds) | new_distance.le(
                updated["caliper_threshold"]
            )
            exclusions.extend(
                exclusion_records(updated.loc[~local], spec_id, "retained_core_locality")
            )
            exclusions.extend(
                exclusion_records(
                    updated.loc[local & ~caliper], spec_id, "retained_core_caliper"
                )
            )
            keep = local & caliper
            updated = updated.loc[keep].copy()
            updated["ring_distance"] = new_ring.loc[keep].astype("int16")
            updated["feature_distance"] = new_distance.loc[keep]
            updated["caliper_eligible"] = True
            kept.append(updated)
        result = pd.concat(kept, ignore_index=True) if kept else current.iloc[:0].copy()
        if len(result) == len(current):
            return result, exclusions
        current = result
    return current, exclusions


def maximum_region_disjoint_matching(
    frame: pd.DataFrame,
    regions: dict[str, str],
    qualifying_pair_edges: dict[str, list[tuple[str, str]]],
) -> tuple[str, ...]:
    """Maximize pair count, then retained core-edge count, then lexical IDs."""
    endpoints: dict[str, tuple[str, str]] = {}
    edge_counts: dict[str, int] = {}
    for pair_id, group in frame.groupby("region_pair_id", sort=True):
        members = set(group["county_fips"])
        pair_regions = tuple(sorted({regions[county] for county in members}))
        if len(pair_regions) != 2:
            raise ValueError("a matching candidate must span exactly two regions")
        count = sum(
            left in members and right in members
            for left, right in qualifying_pair_edges.get(str(pair_id), [])
        )
        if count == 0:
            raise ValueError("a matching candidate must retain a qualifying core edge")
        endpoints[str(pair_id)] = pair_regions
        edge_counts[str(pair_id)] = count
    nodes = sorted({region for pair in endpoints.values() for region in pair})
    positions = {region: index for index, region in enumerate(nodes)}
    incident: dict[int, list[tuple[int, str]]] = {i: [] for i in range(len(nodes))}
    for pair_id, (left, right) in endpoints.items():
        i, j = positions[left], positions[right]
        incident[i].append((j, pair_id))
        incident[j].append((i, pair_id))

    def rank(pairs: tuple[str, ...]) -> tuple[int, int, tuple[str, ...]]:
        return -len(pairs), -sum(edge_counts[pair] for pair in pairs), pairs

    @cache
    def solve(available: int) -> tuple[str, ...]:
        if available == 0:
            return ()
        first_bit = available & -available
        first = first_bit.bit_length() - 1
        rest = available ^ first_bit
        best = solve(rest)
        for other, pair_id in sorted(incident[first]):
            other_bit = 1 << other
            if not rest & other_bit:
                continue
            choice = tuple(sorted((pair_id, *solve(rest ^ other_bit))))
            if rank(choice) < rank(best):
                best = choice
        return best

    return solve((1 << len(nodes)) - 1)


def construct_ring_memberships(
    candidates: pd.DataFrame,
    neighbors: dict[str, set[str]],
    edges: tuple[tuple[str, str], ...],
    regions: dict[str, str],
    qualifying_pair_edges: dict[str, list[tuple[str, str]]],
    specifications: list[dict[str, object]],
    *,
    agricultural_eligible: pd.Series | None = None,
    features: pd.DataFrame | None = None,
    matching_source_spec_id: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Construct benchmark rings unchanged and separately declared redesigns."""
    outputs: list[pd.DataFrame] = []
    completed: dict[str, pd.DataFrame] = {}
    exclusions: list[dict[str, object]] = []
    for specification in specifications:
        spec_id = str(specification["spec_id"])
        bandwidth = int(specification["ring_bandwidth"])
        if specification.get("region_disjoint_matching", False):
            if matching_source_spec_id not in completed:
                raise ValueError("region matching requires its completed screened source")
            source = completed[matching_source_spec_id]
            selected = maximum_region_disjoint_matching(
                source, regions, qualifying_pair_edges
            )
            keep = source["region_pair_id"].isin(selected)
            exclusions.extend(
                exclusion_records(source.loc[~keep], spec_id, "region_matching_not_selected")
            )
            eligible = source.loc[keep].copy()
        else:
            considered = candidates.loc[candidates["ring_distance"].le(bandwidth)].copy()
            exclusions.extend(
                exclusion_records(
                    considered.loc[~considered["caliper_eligible"]],
                    spec_id,
                    "caliper_rejected",
                )
            )
            eligible = considered.loc[considered["caliper_eligible"]].copy()
            if specification.get("all_member_agricultural_screen", False):
                if agricultural_eligible is None:
                    raise ValueError("all-member screening requires frozen county support")
                county_eligible = eligible["county_fips"].map(agricultural_eligible)
                if county_eligible.isna().any():
                    raise ValueError("all-member screen lacks a candidate county")
                exclusions.extend(
                    exclusion_records(
                        eligible.loc[~county_eligible], spec_id, "all_member_agricultural_screen"
                    )
                )
                eligible = eligible.loc[county_eligible].copy()
            if specification["membership_mode"] == "disjoint":
                eligible = eligible.sort_values(
                    ["county_fips", "ring_distance", "feature_distance", "region_pair_id"],
                    kind="stable",
                )
                chosen_index = eligible.groupby("county_fips", sort=False).head(1).index
                exclusions.extend(
                    exclusion_records(
                        eligible.loc[~eligible.index.isin(chosen_index)],
                        spec_id,
                        "overlap_not_selected",
                    )
                )
                eligible = eligible.loc[chosen_index]
            if specification.get("strict_retained_locality", False):
                if features is None:
                    raise ValueError("strict locality requires the frozen features")
                eligible, local_exclusions = prune_retained_locality(
                    eligible,
                    neighbors,
                    regions,
                    features,
                    qualifying_pair_edges,
                    maximum_ring=bandwidth,
                    spec_id=spec_id,
                )
                exclusions.extend(local_exclusions)
        supported, component_exclusions = supported_components(
            eligible, neighbors, edges, regions, spec_id, qualifying_pair_edges
        )
        exclusions.extend(component_exclusions)
        supported["spec_id"] = spec_id
        supported["geography_family"] = "ring"
        supported["membership_mode"] = specification["membership_mode"]
        supported["role"] = specification["role"]
        supported["ring_bandwidth"] = bandwidth
        outputs.append(supported)
        completed[spec_id] = supported
    memberships = pd.concat(outputs, ignore_index=True)
    counts = memberships.groupby(["spec_id", "county_fips"])["stratum_id"].transform(
        "count"
    )
    memberships["analysis_weight"] = 1.0 / counts
    memberships["membership_id"] = (
        memberships["spec_id"]
        + "__"
        + memberships["stratum_id"]
        + "__"
        + memberships["county_fips"]
    )
    return memberships, exclusions


def supported_components(
    frame: pd.DataFrame,
    neighbors: dict[str, set[str]],
    edges: tuple[tuple[str, str], ...],
    regions: dict[str, str],
    spec_id: str,
    required_cross_edges: dict[str, list[tuple[str, str]]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    retained: list[pd.DataFrame] = []
    excluded: list[dict[str, object]] = []
    for pair_id, pair_frame in frame.groupby("region_pair_id", sort=True):
        members = set(pair_frame["county_fips"])
        supported: list[set[str]] = []
        unsupported: list[set[str]] = []
        for component in connected_components(neighbors, members):
            component_regions = {regions[county] for county in component}
            geographic_cross_edges = [
                edge
                for edge in edges
                if edge[0] in component
                and edge[1] in component
                and regions[edge[0]] != regions[edge[1]]
            ]
            pair_required_edges = (
                required_cross_edges.get(str(pair_id), [])
                if required_cross_edges is not None
                else geographic_cross_edges
            )
            qualifying_cross_edges = [
                edge
                for edge in pair_required_edges
                if edge[0] in component and edge[1] in component
            ]
            if len(component_regions) == 2 and qualifying_cross_edges:
                supported.append(component)
            else:
                unsupported.append(component)
        for component in unsupported:
            for county in component:
                row = pair_frame.loc[pair_frame["county_fips"].eq(county)].iloc[0]
                excluded.append(
                    {
                        "spec_id": spec_id,
                        "county_fips": county,
                        "region_pair_id": pair_id,
                        "reason": "unsupported_component",
                        "ring_distance": row["ring_distance"],
                        "feature_distance": row["feature_distance"],
                        "caliper_threshold": row["caliper_threshold"],
                    }
                )
        for component_index, component in enumerate(
            sorted(supported, key=min), start=1
        ):
            component_frame = pair_frame.loc[
                pair_frame["county_fips"].isin(component)
            ].copy()
            component_frame["stratum_id"] = (
                f"{spec_id}__{pair_id}__c{component_index:02d}"
            )
            retained.append(component_frame)
    if not retained:
        raise ValueError(f"{spec_id} has no supported cross-border components")
    return pd.concat(retained, ignore_index=True), excluded


def add_graph_metrics(
    frame: pd.DataFrame,
    neighbors: dict[str, set[str]],
    edges: tuple[tuple[str, str], ...],
    regions: dict[str, str],
    qualifying_pair_edges: dict[str, list[tuple[str, str]]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    memberships: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for (spec_id, stratum_id), group in frame.groupby(
        ["spec_id", "stratum_id"], sort=True
    ):
        members = set(group["county_fips"])
        internal_edges = [
            edge
            for edge in edges
            if edge[0] in members and edge[1] in members
        ]
        geographic_cross_edges = [
            edge for edge in internal_edges if regions[edge[0]] != regions[edge[1]]
        ]
        pair_ids = group["region_pair_id"].astype(str).unique()
        if len(pair_ids) != 1:
            raise ValueError(f"{spec_id}/{stratum_id} spans multiple border pairs")
        pair_id = pair_ids[0]
        required_edges = (
            qualifying_pair_edges.get(pair_id, [])
            if qualifying_pair_edges is not None
            else geographic_cross_edges
        )
        qualifying_cross_edges = [
            edge
            for edge in required_edges
            if edge[0] in members and edge[1] in members
        ]
        if (
            len(connected_components(neighbors, members)) != 1
            or not qualifying_cross_edges
        ):
            raise ValueError(f"{spec_id}/{stratum_id} violates the graph contract")
        border_seeds = {
            county for edge in qualifying_cross_edges for county in edge
        }
        depths = graph_distances(neighbors, members, border_seeds)
        updated = group.copy()
        updated["border_depth"] = updated["county_fips"].map(depths).astype("int16")
        memberships.append(updated)
        depth_values = np.asarray(list(depths.values()), dtype=float)
        diagnostics.append(
            {
                "spec_id": spec_id,
                "stratum_id": stratum_id,
                "region_pair_id": pair_id,
                "geography_family": group["geography_family"].iloc[0],
                "membership_mode": group["membership_mode"].iloc[0],
                "role": group["role"].iloc[0],
                "retained": True,
                "n_counties": len(members),
                "n_regions": len({regions[county] for county in members}),
                "n_internal_edges": len(internal_edges),
                "n_cross_region_edges": len(geographic_cross_edges),
                "n_qualifying_cross_region_edges": len(qualifying_cross_edges),
                "graph_diameter": graph_diameter(neighbors, members),
                "minimum_border_depth": float(np.min(depth_values)),
                "median_border_depth": float(np.median(depth_values)),
                "maximum_border_depth": float(np.max(depth_values)),
            }
        )
    return pd.concat(memberships, ignore_index=True), diagnostics
