# Purpose: Construct county-year chained Fisher crop price and quantity indexes.
# Inputs: county CDL acres and state/national synthetic price-yield tables.
# Outputs: data/intermediate/price_index_fisher_county_year.parquet

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from h2a.paths import RAW, INTERMEDIATE, CODE, CACHE
    import dotenv, os
    import polars as pl
    import pdfplumber
    import dspy
    from pydantic import BaseModel, Field
    from typing import List, Literal
    import json
    import tqdm
    from itertools import islice
    import time
    import copy
    import math
    from h2a.price_index import attach_synthetic_price_yield

    return CACHE, INTERMEDIATE, RAW, attach_synthetic_price_yield, math, mo, pl


@app.cell
def _(CACHE, INTERMEDIATE, RAW):
    binary_path = INTERMEDIATE
    json_path = CACHE
    cdl_path = RAW / "croplandcros_cdl"
    return (binary_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Combine county CDL acreage with previously calculated state+national synthetic CDL price and yield
    """)
    return


@app.cell
def _(binary_path, pl):
    from h2a.geography import (
        assert_geo_columns,
        state_from_county_fips,
    )

    # CroplandCROS CDL acreage aggregated to the county-year-crop level
    cdl_acres = pl.read_parquet(binary_path / "croplandcros_county_crop_acres.parquet")
    assert_geo_columns(cdl_acres, ["county_fips"])
    cdl_acres = cdl_acres.with_columns(
        pl.col("county_fips")
        .map_elements(state_from_county_fips, return_dtype=pl.String)
        .alias("state_fips")
    )
    # CDL codes between 80 and 200 are non-ag codes
    cdl_acres = (
        cdl_acres.with_columns(
            pl.col("crop_code").cast(dtype=pl.Int64).alias("cdl_code"),
            pl.col("crop_name").alias("cdl_name"),
        )
        .filter((pl.col("cdl_code") < 80) | (pl.col("cdl_code") > 200))
        .drop("crop_code")
    )
    return assert_geo_columns, cdl_acres


@app.cell
def _(binary_path, pl):
    # State and national synthetic CDL price and yield
    state_synthetic_cdl = pl.read_parquet(
        binary_path / "cdl_price_yield_synthetic_state.parquet"
    )
    national_synthetic_cdl = pl.read_parquet(
        binary_path / "cdl_price_yield_synthetic_national.parquet"
    )
    return national_synthetic_cdl, state_synthetic_cdl


@app.cell
def _(
    attach_synthetic_price_yield,
    cdl_acres,
    national_synthetic_cdl,
    pl,
    state_synthetic_cdl,
):
    # State values take precedence; missing fields fall back independently to
    # the national synthetic value for the same year and crop.
    county_cdl_values = attach_synthetic_price_yield(
        cdl_acres,
        state_synthetic_cdl,
        national_synthetic_cdl,
    )
    print(
        county_cdl_values.group_by("price_source", "yield_source")
        .len()
        .sort("price_source", "yield_source")
    )
    county_cdl_panel = (
        county_cdl_values
        .with_columns(
            (pl.col("acres") * pl.col("cdl_syn_yield")).alias("q_lbs"),
            pl.col("cdl_syn_price").alias("p_usd_lb"),
        )
        .select(
            "county_fips",
            "state_fips",
            "year",
            "cdl_code",
            "cdl_name",
            "p_usd_lb",
            "q_lbs",
            "acres",
        )
    )
    return (county_cdl_panel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chained price index
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We include only crops that are present in both t-1 and t
    """)
    return


@app.cell
def _(county_cdl_panel, pl):
    def compute_bilateral_links(df: pl.DataFrame):
        # Create a shifted version of the panel to align T with T-1
        df_prev = df.select(
            [
                "county_fips",
                "cdl_code",
                (pl.col("year") + 1).alias("year"),
                pl.col("p_usd_lb").alias("p_prev"),
                pl.col("q_lbs").alias("q_prev"),
            ]
        )

        # Inner Join ensures we only compare crops present in both years (Matched Set)
        links = (
            df.join(df_prev, on=["county_fips", "year", "cdl_code"], how="inner")
            .with_columns(
                [
                    (pl.col("p_usd_lb") * pl.col("q_prev")).alias("p1_q0"),
                    (pl.col("p_prev") * pl.col("q_prev")).alias("p0_q0"),
                    (pl.col("p_usd_lb") * pl.col("q_lbs")).alias("p1_q1"),
                    (pl.col("p_prev") * pl.col("q_lbs")).alias("p0_q1"),
                ]
            )
            .group_by(["county_fips", "year"])
            .agg(
                [
                    pl.sum("p1_q0").alias("sum_p1_q0"),
                    pl.sum("p0_q0").alias("sum_p0_q0"),
                    pl.sum("p1_q1").alias("sum_p1_q1"),
                    pl.sum("p0_q1").alias("sum_p0_q1"),
                ]
            )
            .with_columns(
                [
                    # Price indexes hold quantities fixed.
                    (pl.col("sum_p1_q0") / pl.col("sum_p0_q0")).alias("laspeyres"),
                    (pl.col("sum_p1_q1") / pl.col("sum_p0_q1")).alias("paasche"),
                    # Quantity indexes hold prices fixed.
                    (pl.col("sum_p0_q1") / pl.col("sum_p0_q0")).alias(
                        "quantity_laspeyres"
                    ),
                    (pl.col("sum_p1_q1") / pl.col("sum_p1_q0")).alias(
                        "quantity_paasche"
                    ),
                ]
            )
            .with_columns(
                [
                    # Fisher links are the geometric means of their
                    # Laspeyres and Paasche counterparts.
                    (pl.col("laspeyres") * pl.col("paasche")).sqrt().alias("fisher"),
                    (pl.col("quantity_laspeyres") * pl.col("quantity_paasche"))
                    .sqrt()
                    .alias("quantity_fisher"),
                ]
            )
            # Convert to log-space for additive chaining
            .with_columns(
                pl.col("fisher").log().alias("log_fisher"),
                pl.col("laspeyres").log().alias("log_laspeyres"),
                pl.col("paasche").log().alias("log_paasche"),
                pl.col("quantity_fisher").log().alias("log_quantity_fisher"),
            )
            .select(
                [
                    "county_fips",
                    "year",
                    "fisher",
                    "laspeyres",
                    "paasche",
                    "log_fisher",
                    "log_laspeyres",
                    "log_paasche",
                    "quantity_fisher",
                    "quantity_laspeyres",
                    "quantity_paasche",
                    "log_quantity_fisher",
                ]
            )
        )
        return links

    bilateral_links = compute_bilateral_links(county_cdl_panel)
    bilateral_links
    return (bilateral_links,)


@app.cell
def _(bilateral_links, county_cdl_panel, math, pl):
    base_year = 2011
    log100 = math.log(100.0)

    def chain_index(log_link_column: str, index_column: str) -> pl.DataFrame:
        # Forward chain (2012 -> 2024)
        forward_chain = (
            bilateral_links.filter(pl.col("year") > base_year)
            .sort(["county_fips", "year"])
            .with_columns(
                (pl.col(log_link_column).cum_sum().over("county_fips") + log100).alias(
                    "log_index"
                ),
                pl.col("year").cast(pl.Int32),
            )
        )

        # Backward chaining subtracts each link from the base-year anchor.
        backward_chain = (
            bilateral_links.filter(pl.col("year") <= base_year)
            .sort(["county_fips", "year"], descending=[False, True])
            .with_columns(
                (log100 - pl.col(log_link_column).cum_sum().over("county_fips")).alias(
                    "log_index"
                ),
                (pl.col("year") - 1).alias("target_year"),
            )
            .select(["county_fips", pl.col("target_year").alias("year"), "log_index"])
            .with_columns(pl.col("year").cast(pl.Int32))
        )

        base_anchor = (
            county_cdl_panel.select("county_fips")
            .unique()
            .with_columns(
                [
                    pl.lit(base_year).alias("year").cast(pl.Int32),
                    pl.lit(log100).alias("log_index"),
                ]
            )
        )

        return (
            pl.concat(
                [
                    forward_chain.select(["county_fips", "year", "log_index"]),
                    backward_chain,
                    base_anchor,
                ]
            )
            .with_columns(pl.col("log_index").exp().alias(index_column))
            .select("county_fips", "year", index_column)
            .sort(["county_fips", "year"])
        )

    chained_fisher = chain_index("log_fisher", "fisher_index").join(
        chain_index("log_quantity_fisher", "fisher_quantity_index"),
        on=["county_fips", "year"],
        how="inner",
        validate="1:1",
    )

    index_columns = ["fisher_index", "fisher_quantity_index"]
    invalid_index_rows = chained_fisher.filter(
        pl.any_horizontal(
            [
                ~pl.col(column).is_finite() | (pl.col(column) <= 0)
                for column in index_columns
            ]
        )
    ).height
    if invalid_index_rows:
        print(
            "Setting nonfinite or nonpositive Fisher indexes to null:",
            invalid_index_rows,
        )
        chained_fisher = chained_fisher.with_columns(
            [
                pl.when(pl.col(column).is_finite() & (pl.col(column) > 0))
                .then(pl.col(column))
                .otherwise(None)
                .alias(column)
                for column in index_columns
            ]
        )

    if chained_fisher.select("county_fips", "year").is_duplicated().any():
        raise AssertionError("Fisher index output contains duplicate county-years")
    invalid_anchor = chained_fisher.filter(
        (pl.col("year") == base_year)
        & (
            ((pl.col("fisher_index") - 100).abs() > 1e-10)
            | ((pl.col("fisher_quantity_index") - 100).abs() > 1e-10)
        )
    )
    if invalid_anchor.height:
        raise AssertionError("Fisher indexes must equal 100 in the 2011 base year")

    gap_count = (
        chained_fisher.sort("county_fips", "year")
        .with_columns(pl.col("year").diff().over("county_fips").alias("year_gap"))
        .filter(pl.col("year_gap") > 1)
        .select("county_fips")
        .n_unique()
    )
    print("Counties with unfilled year gaps in the chained index:", gap_count)
    return (chained_fisher,)


@app.cell
def _(assert_geo_columns, binary_path, chained_fisher):
    assert_geo_columns(chained_fisher, ["county_fips"])
    chained_fisher.write_parquet(binary_path / "price_index_fisher_county_year.parquet")
    return


@app.cell
def _(chained_fisher):
    chained_fisher
    return


if __name__ == "__main__":
    app.run()
