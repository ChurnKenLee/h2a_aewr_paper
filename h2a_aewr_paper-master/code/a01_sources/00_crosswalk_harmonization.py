# Purpose: Normalize the Census county adjacency file to the project county key.
# Inputs: data/raw/geographic_crosswalks/census/county_adjacency2010.txt
# Outputs: data/intermediate/county_adjacency2010.parquet

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from h2a.paths import CODE, RAW, INTERMEDIATE, CACHE
    import dotenv, os
    import polars as pl

    return INTERMEDIATE, RAW, pl


@app.cell
def _(INTERMEDIATE, RAW, pl):
    from h2a.geography import (
        assert_geo_columns,
        harmonize_county_fips_2010,
    )

    county_adjacency = pl.read_csv(
        RAW / "geographic_crosswalks" / "census" / "county_adjacency2010.txt",
        separator="\t",
        new_columns=[
            "countyname",
            "county_fips",
            "neighborname",
            "neighbor_county_fips",
        ],
        infer_schema=False,
        has_header=False,
        encoding="cp1252",
    )
    county_adjacency = (
        county_adjacency.fill_null(strategy="forward")
        .with_columns(
            pl.col("county_fips").map_elements(
                harmonize_county_fips_2010,
                return_dtype=pl.String,
            ),
            pl.col("neighbor_county_fips").map_elements(
                harmonize_county_fips_2010,
                return_dtype=pl.String,
            ),
        )
        .sort(by=pl.all())
    )
    assert_geo_columns(
        county_adjacency,
        ["county_fips", "neighbor_county_fips"],
    )
    county_adjacency.write_parquet(INTERMEDIATE / "county_adjacency2010.parquet")
    return


if __name__ == "__main__":
    app.run()
