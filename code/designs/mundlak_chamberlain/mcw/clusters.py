"""Deterministic sensitivity partitions for the version-4 cluster registry."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import polars as pl
import scipy.stats
from sklearn.cluster import AgglomerativeClustering

AGRO_FEATURES = (
    "mc_baseline_farm_employment_share_z",
    "mc_baseline_crop_income_share_z",
    "mc_baseline_hired_labor_cost_share_z",
    "mc_baseline_cropland_z",
)
MIN_REGION_MASS_SHARE = 0.05


def _exposure_column(frame: pl.DataFrame) -> str:
    candidates = (
        "mc_fraction_affected_f0809",
        "mc_fraction_affected_quantile_approx_f0809",
        "mc_frozen_fraction_affected_f0809",
    )
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(
        "No frozen 2008-09 fraction-affected column is available for exposure "
        "partitions."
    )


def _county_frame(frame: pl.DataFrame, columns: Iterable[str]) -> pl.DataFrame:
    selected = ["county_fips", *columns]
    missing = sorted(set(selected).difference(frame.columns))
    if missing:
        raise ValueError(f"Cluster construction lacks columns: {missing}")
    variation = (
        frame.group_by("county_fips")
        .agg(pl.col(list(columns)).n_unique())
        .select(pl.any_horizontal(pl.exclude("county_fips") > 1).any())
        .item()
    )
    if variation:
        raise ValueError("A predetermined cluster input varies within county.")
    return frame.select(selected).unique(subset="county_fips", keep="first")


def _exposure_deciles(county: pl.DataFrame, exposure: str) -> pl.DataFrame:
    values = county[exposure].cast(pl.Float64).to_numpy()
    if not np.all(np.isfinite(values)):
        raise ValueError("Frozen exposure is non-finite.")
    ranks = scipy.stats.rankdata(values, method="average")
    decile = np.minimum(10, np.floor(10 * (ranks - 1) / len(values)).astype(int) + 1)
    region = county["aewr_region_id"].cast(pl.String).to_numpy()
    return county.select("county_fips").with_columns(
        pl.Series("mc_cluster_exposure_decile", [f"d{d:02d}" for d in decile]),
        pl.Series(
            "mc_cluster_exposure_decile_region",
            [f"{r}__d{d:02d}" for r, d in zip(region, decile, strict=True)],
        ),
    )


def _merge_low_mass_clusters(
    labels: np.ndarray,
    features: np.ndarray,
    mass: np.ndarray,
) -> np.ndarray:
    labels = labels.copy()
    total_mass = float(np.sum(mass))
    if total_mass <= 0:
        mass = np.ones_like(mass)
        total_mass = float(mass.size)
    while np.unique(labels).size > 1:
        groups = np.unique(labels)
        group_mass = {group: float(np.sum(mass[labels == group])) for group in groups}
        low = [
            group
            for group in groups
            if group_mass[group] < MIN_REGION_MASS_SHARE * total_mass
        ]
        if not low:
            break
        source = min(low, key=lambda group: (group_mass[group], int(group)))
        source_rows = labels == source
        source_centroid = np.average(
            features[source_rows], axis=0, weights=mass[source_rows]
        )
        distances = []
        for target in groups:
            if target == source:
                continue
            target_rows = labels == target
            target_centroid = np.average(
                features[target_rows], axis=0, weights=mass[target_rows]
            )
            distances.append(
                (float(np.linalg.norm(source_centroid - target_centroid)), target)
            )
        _, target = min(distances, key=lambda item: (item[0], int(item[1])))
        labels[source_rows] = target
    return labels


def _agro_partition(county: pl.DataFrame, requested_k: int) -> pl.DataFrame:
    county_rows = county.to_dicts()
    by_region: dict[str, list[dict[str, object]]] = {}
    for row in county_rows:
        by_region.setdefault(str(row["aewr_region_id"]), []).append(row)

    county_to_cluster: dict[str, str] = {}
    for region in sorted(by_region):
        rows = by_region[region]
        by_unit: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            unit = f"{region}__{row['cz_id']}"
            by_unit.setdefault(unit, []).append(row)
        unit_ids = sorted(by_unit)
        features = []
        masses = []
        for unit_id in unit_ids:
            unit_rows = by_unit[unit_id]
            county_mass = np.array(
                [
                    max(float(value), 0.0)
                    if (value := row["mc_baseline_farm_employment"]) is not None
                    and np.isfinite(float(value))
                    else 0.0
                    for row in unit_rows
                ]
            )
            if np.sum(county_mass) <= 0:
                county_mass = np.ones(len(unit_rows))
            feature_matrix = np.array(
                [[float(row[column]) for column in AGRO_FEATURES] for row in unit_rows]
            )
            features.append(np.average(feature_matrix, axis=0, weights=county_mass))
            masses.append(float(np.sum(county_mass)))
        feature_array = np.asarray(features, dtype=np.float64)
        mass_array = np.asarray(masses, dtype=np.float64)
        centered = feature_array - np.mean(feature_array, axis=0)
        scale = np.std(feature_array, axis=0, ddof=0)
        scale[scale <= 1e-12] = 1.0
        standardized = centered / scale
        n_clusters = min(requested_k, len(unit_ids))
        if n_clusters <= 1:
            labels = np.zeros(len(unit_ids), dtype=int)
        else:
            labels = AgglomerativeClustering(
                n_clusters=n_clusters, linkage="ward"
            ).fit_predict(standardized)
        labels = _merge_low_mass_clusters(labels, standardized, mass_array)
        ordered_labels = sorted(
            np.unique(labels),
            key=lambda label: min(
                unit_ids[index] for index in np.flatnonzero(labels == label)
            ),
        )
        relabel = {old: new + 1 for new, old in enumerate(ordered_labels)}
        for unit_index, unit_id in enumerate(unit_ids):
            cluster_id = (
                f"{region}__agro{requested_k}__{relabel[labels[unit_index]]:02d}"
            )
            for row in by_unit[unit_id]:
                county_to_cluster[str(row["county_fips"])] = cluster_id
    column = f"mc_cluster_agro{requested_k}"
    return pl.DataFrame(
        {
            "county_fips": sorted(county_to_cluster),
            column: [county_to_cluster[key] for key in sorted(county_to_cluster)],
        }
    )


def add_cluster_partitions(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the declared deterministic sensitivity-cluster columns."""

    required = {
        "county_fips",
        "year",
        "state_fips",
        "aewr_region_id",
        "cz_id",
        "mc_baseline_farm_employment",
        *AGRO_FEATURES,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot construct cluster partitions: {missing}")
    exposure = _exposure_column(frame)
    market = pl.concat_str(["state_fips", "cz_id", "aewr_region_id"], separator="__")
    result = frame.with_columns(
        market.alias("mc_market_id"),
        pl.concat_str(["cz_id", "aewr_region_id"], separator="__").alias(
            "mc_cluster_cz_region"
        ),
        pl.concat_str(
            ["aewr_region_id", pl.col("year").cast(pl.String)], separator="__"
        ).alias("mc_cluster_region_year"),
        pl.concat_str(
            ["state_fips", pl.col("year").cast(pl.String)], separator="__"
        ).alias("mc_cluster_state_year"),
        pl.col("year").cast(pl.Int32).cast(pl.String).alias("mc_cluster_year"),
    )
    county_columns = [
        "state_fips",
        "aewr_region_id",
        "cz_id",
        exposure,
        "mc_baseline_farm_employment",
        *AGRO_FEATURES,
    ]
    county = _county_frame(result, county_columns)
    mappings = [_exposure_deciles(county, exposure)]
    mappings.extend(_agro_partition(county, k) for k in (2, 3, 5))
    for mapping in mappings:
        result = result.join(
            mapping, on="county_fips", how="left", maintain_order="left"
        )
    cluster_columns = [
        column for column in result.columns if column.startswith("mc_cluster_")
    ]
    if result.select(pl.any_horizontal(pl.col(cluster_columns).is_null()).any()).item():
        raise ValueError("A declared cluster partition contains null identifiers.")
    return result
