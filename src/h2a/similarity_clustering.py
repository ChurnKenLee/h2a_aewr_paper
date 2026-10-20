"""Reusable predetermined matching-feature helpers."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from h2a.geography import harmonize_county_fips_2010

SOIL_CONTINUOUS = (
    "slope_r",
    "slopegradwta",
    "resdept_r",
    "aws025wta",
    "aws050wta",
    "aws0100wta",
    "aws0150wta",
    "wtdepannmin",
    "wtdepaprjunmin",
    "brockdepmin",
    "cropprodindex",
)
SOIL_CATEGORICAL = ("taxorder", "drainagecl", "hydgrp", "nirrcapcl")


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def normalize_counties(values: pd.Series) -> pd.Series:
    normalized = values.map(harmonize_county_fips_2010)
    if normalized.isna().any():
        raise ValueError("county_fips contains missing or malformed values")
    return normalized.astype("string")


def clean_feature_name(value: object) -> str:
    text = str(value).strip().lower().replace("&", " and ")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text)).strip("_")


def crop_features(
    crop: pd.DataFrame, counties: pd.Index, start_year: int, end_year: int
) -> pd.DataFrame:
    require_columns(
        crop, {"county_fips", "year", "crop_type_label", "acres"}, "crop acreage"
    )
    crop = crop.copy()
    crop["county_fips"] = normalize_counties(crop["county_fips"])
    crop = crop.loc[
        crop["year"].between(start_year, end_year)
        & crop["crop_type_label"].notna()
        & crop["crop_type_label"].ne("non-crop")
    ].copy()
    crop["feature"] = "share_cdl_" + crop["crop_type_label"].map(
        clean_feature_name
    )
    annual = (
        crop.groupby(["county_fips", "year", "feature"], observed=True)["acres"]
        .sum()
        .unstack("feature", fill_value=0.0)
    )
    totals = annual.sum(axis=1)
    annual = annual.loc[totals.gt(0)].div(totals.loc[totals.gt(0)], axis=0)
    result = annual.groupby(level="county_fips").mean().reindex(counties)
    return result.sort_index(axis=1)


def climate_features(
    climate: pd.DataFrame, counties: pd.Index, start_year: int, end_year: int
) -> pd.DataFrame:
    require_columns(climate, {"county_fips", "year"}, "climate basis")
    climate = climate.copy()
    climate["county_fips"] = normalize_counties(climate["county_fips"])
    columns = sorted(c for c in climate.columns if c.startswith("normal_cb_"))
    if not columns:
        raise ValueError("climate basis has no normal_cb_ columns")
    return (
        climate.loc[climate["year"].between(start_year, end_year)]
        .groupby("county_fips", observed=True)[columns]
        .mean()
        .reindex(counties)
    )


def soil_features(soil: pd.DataFrame, counties: pd.Index) -> pd.DataFrame:
    require_columns(
        soil,
        {"county_fips", "total_acres", *SOIL_CONTINUOUS, *SOIL_CATEGORICAL},
        "soil cells",
    )
    soil = soil.copy()
    soil["county_fips"] = normalize_counties(soil["county_fips"])
    weights = pd.to_numeric(soil["total_acres"], errors="coerce")

    continuous: dict[str, pd.Series] = {}
    for column in SOIL_CONTINUOUS:
        values = pd.to_numeric(soil[column], errors="coerce")
        valid = values.notna() & weights.notna() & weights.gt(0)
        numerator = (values.loc[valid] * weights.loc[valid]).groupby(
            soil.loc[valid, "county_fips"], observed=True
        ).sum()
        denominator = weights.loc[valid].groupby(
            soil.loc[valid, "county_fips"], observed=True
        ).sum()
        continuous[column] = numerator.div(denominator)
    continuous_frame = pd.DataFrame(continuous).reindex(counties)

    categorical_frames: list[pd.DataFrame] = []
    valid_weight = weights.notna() & weights.gt(0)
    for column in SOIL_CATEGORICAL:
        category = soil[column].fillna("missing").map(clean_feature_name)
        feature = "share_soil_" + column + "_" + category
        grouped = (
            pd.DataFrame(
                {
                    "county_fips": soil.loc[valid_weight, "county_fips"],
                    "feature": feature.loc[valid_weight],
                    "weight": weights.loc[valid_weight],
                }
            )
            .groupby(["county_fips", "feature"], observed=True)["weight"]
            .sum()
            .unstack("feature", fill_value=0.0)
        )
        grouped = grouped.div(grouped.sum(axis=1), axis=0).reindex(counties)
        categorical_frames.append(grouped)
    return pd.concat([continuous_frame, *categorical_frames], axis=1)


def scaled_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    share_columns = [
        column
        for column in raw.columns
        if column.startswith(("share_cdl_", "share_soil_"))
    ]
    transformed = raw.astype(float).copy()
    transformed[share_columns] = np.sqrt(
        np.maximum(transformed[share_columns], 0.0)
    )
    missing_count = transformed.isna().sum(axis=1).astype("int32")

    for column in transformed.columns:
        values = transformed[column].replace([np.inf, -np.inf], np.nan)
        median = values.median()
        values = values.fillna(0.0 if pd.isna(median) else median)
        standard_deviation = values.std(ddof=1)
        transformed[column] = (
            (values - values.mean()) / standard_deviation
            if np.isfinite(standard_deviation) and standard_deviation > 0
            else 0.0
        )

    blocks = (
        [column for column in raw.columns if column.startswith("share_cdl_")],
        [column for column in raw.columns if column.startswith("normal_cb_")],
        [column for column in SOIL_CONTINUOUS if column in raw.columns],
        [column for column in raw.columns if column.startswith("share_soil_")],
    )
    for columns in blocks:
        if columns:
            transformed[columns] = transformed[columns] / np.sqrt(len(columns))
    return transformed, missing_count


def build_county_features(
    crop: pd.DataFrame,
    climate: pd.DataFrame,
    soil: pd.DataFrame,
    counties: pd.Index,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build raw and standardized matching features for a sorted county index."""
    raw = pd.concat(
        [
            crop_features(crop, counties, start_year, end_year),
            climate_features(climate, counties, start_year, end_year),
            soil_features(soil, counties),
        ],
        axis=1,
    )
    if raw.columns.duplicated().any():
        raise ValueError("feature construction produced duplicate columns")
    if raw.empty or not raw.index.is_unique:
        raise ValueError("county feature artifact must be nonempty and county-unique")
    scaled, missing_count = scaled_features(raw)
    return raw, scaled, missing_count
