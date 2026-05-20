import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from h2a.paths import CODE, RAW, INTERMEDIATE
    import dotenv, os
    import polars as pl
    import numpy as np
    import json
    import requests
    import urllib
    import time
    import addfips
    import us

    DC_STATEHOOD = 1  # Enables DC to be included in the state list
    return CODE, INTERMEDIATE, dotenv, mo, os, pl, requests, us


@app.cell
def _(CODE, INTERMEDIATE, dotenv, os):
    binary_path = INTERMEDIATE
    code_path = CODE
    dotenv.load_dotenv()
    fred_api_key = os.getenv("FRED_API_KEY")  # FRED API key from my account
    return (fred_api_key,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use FRED API to get state-level non-farm minimum wage
    """)
    return


@app.cell
def _(fred_api_key, requests):
    # Get release ID of state minimum wages
    # Grab release ID of all releases
    fred_release_api_url = "https://api.stlouisfed.org/fred/releases"
    release_params = {"api_key": fred_api_key, "file_type": "json"}

    fred_release = requests.get(fred_release_api_url, params=release_params)
    release_list = fred_release.json()["releases"]

    # Take the release id of the state minimum wage series
    state_minimum_wage_release = list(
        filter(
            lambda release_dict: release_dict["name"] == "Minimum Wage Rate by State",
            release_list,
        )
    )
    state_minimum_wage_release_id = state_minimum_wage_release[0]["id"]
    state_minimum_wage_release_id
    return


@app.cell
def _(fred_api_key):
    # Base URL to call API
    fred_observations_api_url = "https://api.stlouisfed.org/fred/v2/release/observations"

    # Bearer token has to go in request header
    fred_observations_headers = {"Authorization": f"Bearer {fred_api_key}"}

    fred_observations_params = {"release_id": "387", "format": "json"}
    return (
        fred_observations_api_url,
        fred_observations_headers,
        fred_observations_params,
    )


@app.cell
def _(
    fred_observations_api_url,
    fred_observations_headers,
    fred_observations_params,
    requests,
):
    # Get all state-level observations
    response = requests.get(
        fred_observations_api_url,
        headers=fred_observations_headers,
        params=fred_observations_params,
    )
    list_obs = response.json()["series"]
    return (list_obs,)


@app.cell
def _(list_obs, pl):
    df = pl.json_normalize(list_obs)

    # Each observation is a series for a state, so has many years
    df = df.explode(pl.col("observations"))
    # min wage data is in JSON values within column
    min_wage_values = pl.json_normalize(df.drop_in_place("observations"))
    df = pl.concat([df, min_wage_values], how="horizontal")

    # Use title to obtain state name, series_id to obtain state abbreviation
    df = df.with_columns(
        pl.col("title")
        .str.split(by="for ")
        .list.to_struct(fields=["junk_prefix", "state_name"])
    ).unnest("title")
    df = df.with_columns(
        pl.col("date").str.slice(length=4, offset=0).alias("year"),
        pl.col("series_id").str.slice(length=2, offset=-2).alias("state_abbreviation"),
    )
    df = df.with_columns(
        pl.when(pl.col("series_id") == "STTMINWGFG")
        .then(pl.lit("USA"))
        .otherwise(pl.col("state_name"))
        .alias("state_name"),
        pl.when(pl.col("series_id") == "STTMINWGFG")
        .then(pl.lit("US"))
        .otherwise(pl.col("state_abbreviation"))
        .alias("state_abbreviation"),
    )
    # Georgia has month obs we can drop
    df = df.filter(pl.col("frequency") == "Annual")
    df = df.select(
        [
            "state_name",
            "state_abbreviation",
            "year",
            "value",
            "units",
            "seasonal_adjustment",
        ]
    )
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Add agricultural worker exemption data
    """)
    return


@app.cell
def _(INTERMEDIATE, df, pl, us):
    ag_exemption_list = [
        "AK",
        "DE",
        "GA",
        "IN",
        "KS",
        "KY",
        "ME",
        "MA",
        "NE",
        "NH",
        "NJ",
        "OK",
        "RI",
        "VT",
        "VA",
        "WA",
        "WV",
        "WY",
    ]

    # Add FIPS code and export
    min_wage_df = df.with_columns(
        pl.col("state_abbreviation").is_in(ag_exemption_list).alias("agriculture_exemption")
    )

    state_name_to_fips_dict = us.states.mapping("name", "fips")
    min_wage_df = min_wage_df.with_columns(
        pl.col("state_name").replace(state_name_to_fips_dict).alias("state_fips_code")
    )
    min_wage_df.write_parquet(INTERMEDIATE / "state_year_min_wage.parquet")
    return


if __name__ == "__main__":
    app.run()
