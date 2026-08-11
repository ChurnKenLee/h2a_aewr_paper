"""Canonical geographic identifiers for Python-owned pipeline artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import polars as pl

# docs-ground:start geographic-code-contract-python
_FIXED_WIDTH = {
    "state_fips": 2,
    "county_code": 3,
    "county_fips": 5,
    "neighbor_county_fips": 5,
}
_UNPADDED = {"cz_id", "aewr_region_id"}
_VARIABLE_WIDTH = {"oews_area_code"}


def clean_geo_code(value: Any) -> str | None:
    """Clean one source value without changing its substantive digits."""
    if value is None:
        return None
    text = str(value).strip().replace('"', "")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text or None


def normalize_geo_code(value: Any, name: str) -> str | None:
    """Return one canonical identifier or raise for a malformed value."""
    text = clean_geo_code(value)
    if text is None:
        return None
    if not text.isdigit():
        raise ValueError(f"{name} must contain digits only: {value!r}")

    if name in _FIXED_WIDTH:
        width = _FIXED_WIDTH[name]
        if len(text) > width:
            raise ValueError(f"{name} must contain at most {width} digits: {value!r}")
        text = text.zfill(width)
    elif name in _UNPADDED:
        text = text.lstrip("0") or "0"
    elif name in _VARIABLE_WIDTH:
        pass
    else:
        raise ValueError(f"Unknown geographic identifier: {name}")

    if name == "aewr_region_id" and text not in {str(i) for i in range(1, 18)}:
        raise ValueError(f"aewr_region_id must be between 1 and 17: {value!r}")
    return text


def harmonize_county_fips_2010(value: Any) -> str | None:
    """Normalize a county identifier to the project's 2010 county vintage."""
    county = normalize_geo_code(value, "county_fips")
    return "46113" if county == "46102" else county


def state_from_county_fips(value: Any) -> str | None:
    """Return the canonical state component of a county FIPS."""
    county = harmonize_county_fips_2010(value)
    return None if county is None else county[:2]


def county_code_from_county_fips(value: Any) -> str | None:
    """Return the canonical three-digit component of a county FIPS."""
    county = harmonize_county_fips_2010(value)
    return None if county is None else county[2:]


def normalize_county_fips_list(value: Any) -> str:
    """Normalize a comma-delimited set of candidate counties."""
    text = clean_geo_code(value)
    if text is None:
        return ""
    counties = [
        harmonize_county_fips_2010(part)
        for part in text.split(",")
        if part.strip()
    ]
    return ",".join(county for county in counties if county is not None)


def geo_expr(source: str, name: str, *, vintage_2010: bool = False) -> pl.Expr:
    """Build a Polars expression that normalizes a geographic column."""
    normalizer = (
        harmonize_county_fips_2010
        if vintage_2010
        else lambda value: normalize_geo_code(value, name)
    )
    return (
        pl.col(source)
        .map_elements(normalizer, return_dtype=pl.String)
        .alias(name)
    )


def assert_geo_columns(
    frame: pl.DataFrame,
    required: Iterable[str],
    *,
    allow_null: Iterable[str] = (),
) -> None:
    """Fail when an artifact does not satisfy the canonical geo contract."""
    required = tuple(required)
    allow_null = set(allow_null)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required geographic columns: {', '.join(missing)}")

    for name in required:
        if frame.schema[name] != pl.String:
            raise TypeError(f"{name} must use Polars String, got {frame.schema[name]}")
        values = frame.get_column(name)
        if name not in allow_null and values.null_count() > 0:
            raise ValueError(f"{name} contains missing values")
        for value in values.drop_nulls().unique().to_list():
            normalized = (
                harmonize_county_fips_2010(value)
                if name in {"county_fips", "neighbor_county_fips"}
                else normalize_geo_code(value, name)
            )
            if normalized != value:
                raise ValueError(f"{name} contains noncanonical value {value!r}")
# docs-ground:end geographic-code-contract-python
