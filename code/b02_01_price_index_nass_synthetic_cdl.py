import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import pyprojroot
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

    return math, mo, pl, pyprojroot


@app.cell
def _(pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    binary_path = root_path / 'binaries'
    json_path = root_path / 'code' / 'json'
    cdl_path = root_path / 'data' / 'croplandcros_cdl'
    return (binary_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Combine county CDL acreage with previously calculated state+national synthetic CDL price and yield
    """)
    return


@app.cell
def _(binary_path, pl):
    # CroplandCROS CDL acreage aggregated to the county-year-crop level
    cdl_acres = pl.read_parquet(binary_path / 'croplandcros_county_crop_acres.parquet')
    cdl_acres = cdl_acres.with_columns(
        pl.col("fips").str.slice(0, 2).alias("state_ansi")
    )
    # CDL codes between 80 and 200 are non-ag codes
    cdl_acres = cdl_acres.with_columns(
        pl.col('crop_code').cast(dtype=pl.Int64).alias('cdl_code'),
        pl.col('crop_name').alias('cdl_name')
    ).filter(
        (pl.col('cdl_code') < 80) | 
        (pl.col('cdl_code') > 200)
    ).drop(
        'crop_code'
    )
    return (cdl_acres,)


@app.cell
def _(binary_path, pl):
    # State and national synthetic CDL price and yield
    state_synthetic_cdl = pl.read_parquet(binary_path / 'cdl_price_yield_synthetic_state.parquet')
    national_synthetic_cdl = pl.read_parquet(binary_path / 'cdl_price_yield_synthetic_national.parquet')
    return national_synthetic_cdl, state_synthetic_cdl


@app.cell
def _(national_synthetic_cdl, pl, state_synthetic_cdl):
    # We want to use national synthetic CDL price and yield as fallback
    synthetic_cdl = (state_synthetic_cdl
        .join(national_synthetic_cdl, on=["year", "cdl_code"], how="full")
        .with_columns([
            pl.coalesce(["p_syn_state", "p_syn_nat"]).alias("cdl_syn_price"),
            pl.coalesce(["y_syn_state", "y_syn_nat"]).alias("cdl_syn_yield")
        ])
    )
    return (synthetic_cdl,)


@app.cell
def _(cdl_acres, pl, synthetic_cdl):
    county_cdl_panel = (cdl_acres
        .join(
            synthetic_cdl,
            on=["year", "state_ansi", "cdl_code"],
            how="left")
    ).with_columns(
        (pl.col("acres") * pl.col("cdl_syn_yield")).alias("q_lbs"),
        pl.col("cdl_syn_price").alias("p_usd_lb")
    ).select(
        "fips", "state_ansi", "year", "cdl_code", "cdl_name",
        "p_usd_lb", "q_lbs",
        'acres'
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
        df_prev = (df
            .select([
                "fips", 
                "cdl_code", 
                (pl.col("year") + 1).alias("year"), 
                pl.col("p_usd_lb").alias("p_prev"), 
                pl.col("q_lbs").alias("q_prev")
            ])
        )

        # Inner Join ensures we only compare crops present in both years (Matched Set)
        links = (df
            .join(
                df_prev,
                on=["fips", "year", "cdl_code"],
                how="inner")
            .with_columns([
                (pl.col("p_usd_lb") * pl.col("q_prev")).alias("p1_q0"),
                (pl.col("p_prev") * pl.col("q_prev")).alias("p0_q0"),
                (pl.col("p_usd_lb") * pl.col("q_lbs")).alias("p1_q1"),
                (pl.col("p_prev") * pl.col("q_lbs")).alias("p0_q1"),
            ])
            .group_by(["fips", "year"])
            .agg([
                pl.sum("p1_q0").alias("sum_p1_q0"),
                pl.sum("p0_q0").alias("sum_p0_q0"),
                pl.sum("p1_q1").alias("sum_p1_q1"),
                pl.sum("p0_q1").alias("sum_p0_q1"),
            ])
            .with_columns([
                # Laspeyres: (P1*Q0 / P0*Q0)
                (pl.col("sum_p1_q0") / pl.col("sum_p0_q0")).alias("laspeyres"),
                # Paasche: (P1*Q1 / P0*Q1)
                (pl.col("sum_p1_q1") / pl.col("sum_p0_q1")).alias("paasche"),
            ])
            .with_columns(
                # Fisher Link: Sqrt(L * P)
                (pl.col("laspeyres") * pl.col("paasche")).sqrt().alias("fisher")
            )
            # Convert to log-space for additive chaining
            .with_columns(
                pl.col("fisher").log().alias("log_fisher"),
                pl.col("laspeyres").log().alias("log_laspeyres"),
                pl.col("paasche").log().alias("log_paasche")
            )
            .select([
                "fips", "year",
                "fisher", "laspeyres", "paasche",
                "log_fisher", "log_laspeyres", "log_paasche"
            ])
        )
        return links

    bilateral_links = compute_bilateral_links(county_cdl_panel)
    bilateral_links
    return (bilateral_links,)


@app.cell
def _(bilateral_links, county_cdl_panel, math, pl):
    base_year = 2011
    log100 = math.log(100.0)

    # Forward chain (2012 -> 2024)
    forward_chain = (bilateral_links
        .filter(pl.col("year") > base_year)
        .sort(["fips", "year"])
        .with_columns(
            (pl.col("log_fisher").cum_sum().over("fips") + log100).alias("log_index")
        )
    )

    # Backward chain (2010 -> 2008)
    # Chaining backward means subtracting the log-link from the base
    backward_chain = (bilateral_links
        .filter(pl.col("year") <= base_year)
        .sort(["fips", "year"], descending=[False, True]) # Sort years descending within FIPS
        .with_columns(
            (log100 - pl.col("log_fisher").cum_sum().over("fips")).alias("log_index"),
            (pl.col("year") - 1).alias("target_year") # The link at 2011 defines the step from 2010 to 2011
        )
        .select([
            "fips", 
            pl.col("target_year").alias("year"), 
            "log_index"
        ])
    )

    # Base year anchor
    base_anchor = (county_cdl_panel
        .select("fips").unique()
        .with_columns([
            pl.lit(base_year).alias("year").cast(pl.Int64),
            pl.lit(log100).alias("log_index")
        ])
    )
    return backward_chain, base_anchor, forward_chain


@app.cell
def _(backward_chain, base_anchor, binary_path, forward_chain, pl):
    # Combine and exponentiate
    chained_fisher = (
        pl.concat([
            forward_chain.select(["fips", "year", "log_index"]), 
            backward_chain, 
            base_anchor
        ])
        .with_columns(
            pl.col("log_index").exp().alias("fisher_index")
        )
        .sort(["fips", "year"])
    )
    chained_fisher.write_parquet(binary_path / 'price_index_fisher_county_year_nass_price_yield_cdl_acres.parquet')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
