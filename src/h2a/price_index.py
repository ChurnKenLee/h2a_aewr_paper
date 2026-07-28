"""Helpers for constructing the county crop price-index inputs."""

from __future__ import annotations

import polars as pl

_STATE_KEYS = ["year", "state_fips", "cdl_code"]
_NATIONAL_KEYS = ["year", "cdl_code"]


def _require_unique(frame: pl.DataFrame, keys: list[str], label: str) -> None:
    """Fail with a useful message when a price lookup is not many-to-one."""
    missing = [key for key in keys if key not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing key columns: {', '.join(missing)}")
    if frame.select(keys).is_duplicated().any():
        raise ValueError(f"{label} contains duplicate keys: {', '.join(keys)}")


def attach_synthetic_price_yield(
    cdl_acres: pl.DataFrame,
    state_synthetic_cdl: pl.DataFrame,
    national_synthetic_cdl: pl.DataFrame,
) -> pl.DataFrame:
    """Attach state values with an independent national fallback per field.

    The state and national tables must each be unique on their lookup keys.
    Source columns and source labels are retained for diagnostics; callers may
    select only ``cdl_syn_price`` and ``cdl_syn_yield`` for downstream use.
    """
    _require_unique(state_synthetic_cdl, _STATE_KEYS, "state synthetic CDL")
    _require_unique(
        national_synthetic_cdl,
        _NATIONAL_KEYS,
        "national synthetic CDL",
    )

    state_lookup = state_synthetic_cdl.select(
        *_STATE_KEYS,
        "p_syn_state",
        "y_syn_state",
    )
    national_lookup = national_synthetic_cdl.select(
        *_NATIONAL_KEYS,
        "p_syn_nat",
        "y_syn_nat",
    )

    result = (
        cdl_acres.join(
            state_lookup,
            on=_STATE_KEYS,
            how="left",
            validate="m:1",
        )
        .join(
            national_lookup,
            on=_NATIONAL_KEYS,
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.coalesce("p_syn_state", "p_syn_nat").alias("cdl_syn_price"),
            pl.coalesce("y_syn_state", "y_syn_nat").alias("cdl_syn_yield"),
            pl.when(pl.col("p_syn_state").is_not_null())
            .then(pl.lit("state"))
            .when(pl.col("p_syn_nat").is_not_null())
            .then(pl.lit("national"))
            .otherwise(pl.lit("missing"))
            .alias("price_source"),
            pl.when(pl.col("y_syn_state").is_not_null())
            .then(pl.lit("state"))
            .when(pl.col("y_syn_nat").is_not_null())
            .then(pl.lit("national"))
            .otherwise(pl.lit("missing"))
            .alias("yield_source"),
        )
    )

    unresolved_fallback = result.filter(
        (
            pl.col("p_syn_state").is_null()
            & pl.col("p_syn_nat").is_not_null()
            & pl.col("cdl_syn_price").is_null()
        )
        | (
            pl.col("y_syn_state").is_null()
            & pl.col("y_syn_nat").is_not_null()
            & pl.col("cdl_syn_yield").is_null()
        )
    )
    if unresolved_fallback.height:
        raise AssertionError(
            "Usable national synthetic values were not applied as fallback"
        )
    if result.height != cdl_acres.height:
        raise AssertionError("Synthetic price joins changed the acreage row count")

    return result
