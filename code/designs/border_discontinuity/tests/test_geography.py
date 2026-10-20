"""Synthetic graph checks for the AEWR-border ring construction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).parents[1] / "geography.py"
SPEC = importlib.util.spec_from_file_location("border_geography", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
GEOGRAPHY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GEOGRAPHY
SPEC.loader.exec_module(GEOGRAPHY)


def line_graph():
    return GEOGRAPHY.build_graph(
        ["00001", "00002", "00003", "00004", "00005"],
        [
            ("00001", "00002"),
            ("00002", "00003"),
            ("00003", "00004"),
            ("00004", "00005"),
        ],
    )


def test_components_and_distances_follow_adjacency() -> None:
    neighbors, _ = line_graph()
    components = GEOGRAPHY.connected_components(
        neighbors, {"00001", "00002", "00004", "00005"}
    )
    assert components == [{"00001", "00002"}, {"00004", "00005"}]
    distances = GEOGRAPHY.graph_distances(
        neighbors,
        {"00001", "00002", "00003", "00004"},
        ["00001"],
        maximum=2,
    )
    assert distances == {"00001": 0, "00002": 1, "00003": 2}
    assert (
        GEOGRAPHY.graph_diameter(
            neighbors, {"00001", "00002", "00003", "00004"}
        )
        == 3
    )


def test_supported_components_require_an_internal_border_edge() -> None:
    neighbors, edges = line_graph()
    regions = {
        "00001": "1",
        "00002": "1",
        "00003": "2",
        "00004": "2",
        "00005": "2",
    }
    frame = pd.DataFrame(
        {
            "county_fips": ["00001", "00002", "00003", "00005"],
            "region_pair_id": ["01_02"] * 4,
            "ring_distance": [1, 0, 0, 1],
            "feature_distance": [0.2, 0.1, 0.1, 0.3],
            "caliper_threshold": [1.0] * 4,
        }
    )
    retained, excluded = GEOGRAPHY.supported_components(
        frame, neighbors, edges, regions, "test"
    )
    assert set(retained["county_fips"]) == {"00001", "00002", "00003"}
    assert {row["county_fips"] for row in excluded} == {"00005"}


def test_agricultural_screen_filters_county_pairs_not_whole_borders() -> None:
    regions = {"00001": "1", "00002": "2", "00003": "1", "00004": "2"}
    pair_edges = {"01_02": [("00001", "00002"), ("00003", "00004")]}
    support = pd.DataFrame(
        {
            "cropland_acres_2007": [750.0, 1000.0, 5000.0, 6000.0],
            "mean_farm_employment_2008_2010": [50.0, 60.0, 500.0, 600.0],
            "farm_employment_years_observed": [3, 3, 3, 3],
            "land_area_acres_2010": [10_000.0, 10_000.0, 1_000_000.0, 1_000_000.0],
        },
        index=["00001", "00002", "00003", "00004"],
    )
    retained, audit = GEOGRAPHY.screen_agricultural_county_pairs(
        pair_edges,
        regions,
        support,
        minimum_cropland_share=0.075,
        minimum_farm_employment_per_100_square_miles=30.0,
        minimum_measurement_coverage=0.9,
        expected_farm_employment_years=3,
    )
    assert retained == {"01_02": [("00001", "00002")]}
    assert audit.set_index("county_pair_id")["county_pair_eligible"].to_dict() == {
        "00001__00002": True,
        "00003__00004": False,
    }
    assert not audit.loc[
        audit["county_pair_id"].eq("00003__00004"),
        "left_cropland_share_pass",
    ].item()


def test_agricultural_screen_requires_measurement_coverage() -> None:
    regions = {"00001": "1", "00002": "1", "00003": "2", "00004": "2"}
    support = pd.DataFrame(
        {
            "cropland_acres_2007": [1000.0, 1000.0, 1000.0, 1000.0],
            "mean_farm_employment_2008_2010": [100.0] * 4,
            "farm_employment_years_observed": [3, 3, 2, 3],
            "land_area_acres_2010": [5000.0] * 4,
        },
        index=["00001", "00002", "00003", "00004"],
    )
    try:
        GEOGRAPHY.screen_agricultural_county_pairs(
            {"01_02": [("00002", "00003")]},
            regions,
            support,
            minimum_cropland_share=0.075,
            minimum_farm_employment_per_100_square_miles=30.0,
            minimum_measurement_coverage=0.9,
            expected_farm_employment_years=3,
        )
    except ValueError as error:
        assert "removed every cross-border county pair" in str(error)
    else:
        raise AssertionError("incomplete farm-employment coverage must be rejected")


def test_all_member_screen_agrees_with_endpoint_rules() -> None:
    counties = [f"{i:05d}" for i in range(1, 7)]
    regions = {county: "1" if i % 2 == 0 else "2" for i, county in enumerate(counties)}
    support = pd.DataFrame(
        {
            "cropland_acres_2007": [750, 1000, 749, 1000, 1000, 1000],
            "mean_farm_employment_2008_2010": [50, 60, 100, 100, 4, 100],
            "farm_employment_years_observed": [3, 3, 3, 2, 3, 3],
            "land_area_acres_2010": [10_000] * 6,
        },
        index=counties,
    )
    rules = {
        "minimum_cropland_share": 0.075,
        "minimum_farm_employment_per_100_square_miles": 30.0,
        "minimum_measurement_coverage": 0.9,
        "expected_farm_employment_years": 3,
    }
    eligible = GEOGRAPHY.agricultural_county_eligibility(support, **rules)
    _, audit = GEOGRAPHY.screen_agricultural_county_pairs(
        {"01_02": list(zip(counties[::2], counties[1::2], strict=True))},
        regions,
        support,
        **rules,
    )
    for row in audit.itertuples(index=False):
        assert eligible[row.left_county_fips] == row.left_county_eligible
        assert eligible[row.right_county_fips] == row.right_county_eligible
    assert eligible.tolist() == [True, True, False, False, False, True]


def candidate_frame(counties, regions, distances=None, pair="01_02"):
    return pd.DataFrame(
        {
            "county_fips": counties,
            "aewr_region_id": [regions[county] for county in counties],
            "region_pair_id": [pair] * len(counties),
            "ring_distance": distances if distances is not None else [0] * len(counties),
            "feature_distance": [0.1] * len(counties),
            "caliper_threshold": [1.0] * len(counties),
            "caliper_eligible": [True] * len(counties),
        }
    )


def test_strict_locality_cannot_cross_opposite_region_or_reuse_lost_core() -> None:
    regions = {"00001": "1", "00002": "2", "00003": "1", "00004": "1", "00005": "1"}
    neighbors, _ = GEOGRAPHY.build_graph(
        regions,
        [
            ("00001", "00002"),
            ("00002", "00003"),
            ("00001", "00004"),
            ("00004", "00005"),
        ],
    )
    frame = candidate_frame(list(regions), regions, [0, 0, 1, 1, 0])
    features = pd.DataFrame({"x": [0.0] * 5}, index=list(regions))
    result, excluded = GEOGRAPHY.prune_retained_locality(
        frame,
        neighbors,
        regions,
        features,
        {"01_02": [("00001", "00002"), ("00005", "00006")]},
        maximum_ring=1,
        spec_id="strict",
    )
    assert set(result["county_fips"]) == {"00001", "00002", "00004"}
    assert {row["county_fips"] for row in excluded} == {"00003", "00005"}
    assert {row["reason"] for row in excluded} == {"retained_core_locality"}
    assert result.set_index("county_fips")["ring_distance"].to_dict() == {
        "00001": 0,
        "00002": 0,
        "00004": 1,
    }


def test_strict_caliper_uses_surviving_opposite_core_with_core_exemption() -> None:
    regions = {"00001": "1", "00002": "2", "00003": "1"}
    neighbors, _ = GEOGRAPHY.build_graph(
        regions, [("00001", "00002"), ("00001", "00003")]
    )
    frame = candidate_frame(list(regions), regions, [0, 0, 1])
    features = pd.DataFrame({"x": [0.0, 5.0, 0.0]}, index=list(regions))
    result, excluded = GEOGRAPHY.prune_retained_locality(
        frame,
        neighbors,
        regions,
        features,
        {"01_02": [("00001", "00002")]},
        maximum_ring=1,
        spec_id="strict",
    )
    assert set(result["county_fips"]) == {"00001", "00002"}
    assert result["feature_distance"].eq(5.0).all()
    assert excluded[0]["county_fips"] == "00003"
    assert excluded[0]["reason"] == "retained_core_caliper"
    second, second_exclusions = GEOGRAPHY.prune_retained_locality(
        result,
        neighbors,
        regions,
        features,
        {"01_02": [("00001", "00002")]},
        maximum_ring=1,
        spec_id="strict",
    )
    pd.testing.assert_frame_equal(result, second)
    assert second_exclusions == []


def matching_example(pair_counts):
    rows, regions, edges = [], {}, {}
    index = 1
    for pair, count in pair_counts.items():
        left_region, right_region = (str(int(value)) for value in pair.split("_"))
        edges[pair] = []
        for _ in range(count):
            left, right = f"{index:05d}", f"{index + 1:05d}"
            index += 2
            regions[left], regions[right] = left_region, right_region
            edges[pair].append((left, right))
            rows.extend(
                {"county_fips": county, "region_pair_id": pair}
                for county in (left, right)
            )
    return pd.DataFrame(rows), regions, edges


def test_matching_optimizes_cardinality_then_support_then_lexical() -> None:
    examples = [
        ({"01_02": 10, "01_03": 1, "02_04": 1}, ("01_03", "02_04")),
        (
            {"01_02": 1, "03_04": 1, "01_03": 2, "02_04": 1},
            ("01_03", "02_04"),
        ),
        (
            {"01_02": 1, "03_04": 1, "01_03": 1, "02_04": 1},
            ("01_02", "03_04"),
        ),
    ]
    for counts, expected in examples:
        frame, regions, edges = matching_example(counts)
        result = GEOGRAPHY.maximum_region_disjoint_matching(frame, regions, edges)
        assert result == expected
        assert GEOGRAPHY.maximum_region_disjoint_matching(
            frame.sample(frac=1, random_state=12),
            regions,
            dict(reversed(list(edges.items()))),
        ) == expected
        assert len({node for pair in result for node in pair.split("_")}) == 2 * len(result)


def test_new_screen_preserves_legacy_memberships_and_matching_is_subset() -> None:
    regions = {"00001": "1", "00002": "2", "00003": "1", "00004": "2"}
    neighbors, edges = GEOGRAPHY.build_graph(
        regions,
        [("00001", "00002"), ("00001", "00003"), ("00002", "00004")],
    )
    candidates = candidate_frame(list(regions), regions, [0, 0, 1, 1])
    required = {"01_02": [("00001", "00002")]}
    legacy = {
        "spec_id": "legacy",
        "membership_mode": "disjoint",
        "role": "legacy_primary_benchmark",
        "ring_bandwidth": 1,
    }
    screened = {
        **legacy,
        "spec_id": "screened",
        "all_member_agricultural_screen": True,
        "strict_retained_locality": True,
    }
    matched = {**screened, "spec_id": "matched", "region_disjoint_matching": True}
    old, old_excluded = GEOGRAPHY.construct_ring_memberships(
        candidates, neighbors, edges, regions, required, [legacy]
    )
    revised, excluded = GEOGRAPHY.construct_ring_memberships(
        candidates,
        neighbors,
        edges,
        regions,
        required,
        [legacy, screened, matched],
        agricultural_eligible=pd.Series([True, True, False, True], index=list(regions)),
        features=pd.DataFrame({"x": [0.0] * 4}, index=list(regions)),
        matching_source_spec_id="screened",
    )
    pd.testing.assert_frame_equal(
        old, revised.loc[revised["spec_id"].eq("legacy")].reset_index(drop=True)
    )
    assert old_excluded == []
    assert set(old["county_fips"]) == set(regions)
    for spec_id in ("screened", "matched"):
        selected = revised.loc[revised["spec_id"].eq(spec_id)]
        assert set(selected["county_fips"]) == {"00001", "00002", "00004"}
        assert selected["analysis_weight"].eq(1).all()
    assert any(
        row["county_fips"] == "00003" and row["reason"] == "all_member_agricultural_screen"
        for row in excluded
    )
