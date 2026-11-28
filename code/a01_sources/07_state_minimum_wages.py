# Purpose: Assemble the unified state-year minimum-wage panel from FRED release 387.
# Inputs: FRED API release 387 (FRED_API_KEY in .env) and fips_codes.csv.
# Outputs: data/intermediate/state_year_min_wage.parquet.

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import os

    import marimo as mo
    import polars as pl
    import requests
    from dotenv import load_dotenv

    from h2a.geography import assert_geo_columns, geo_expr
    from h2a.paths import INTERMEDIATE, RAW, ROOT

    return (
        INTERMEDIATE,
        RAW,
        ROOT,
        assert_geo_columns,
        geo_expr,
        load_dotenv,
        mo,
        os,
        pl,
        requests,
    )


@app.cell
def _(ROOT, load_dotenv, os):
    load_dotenv(ROOT / ".env")
    fred_api_key = os.getenv("FRED_API_KEY")
    if not fred_api_key or not fred_api_key.strip():
        raise RuntimeError("FRED_API_KEY must be set in .env or the environment.")
    return (fred_api_key,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fetch FRED release 387 ("Minimum Wage Rate by State"), keeping the annual
    federal series (`STTMINWGFG`) and all annual state series on a complete
    51-state/DC × federal-year grid. Agricultural coverage is applied downstream
    by `code/c01_clean/04_state_minimum_wages.R` from
    `code/a01_sources/state_ag_minwage_coverage.csv`.
    """)


@app.cell
def _(fred_api_key, requests):
    fred_observations_api_url = (
        "https://api.stlouisfed.org/fred/v2/release/observations"
    )
    fred_observations_headers = {"Authorization": f"Bearer {fred_api_key}"}

    list_obs = []
    next_cursor = None
    while True:
        params = {"release_id": "387", "format": "json"}
        if next_cursor:
            params["next_cursor"] = next_cursor
        response = requests.get(
            fred_observations_api_url,
            headers=fred_observations_headers,
            params=params,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        list_obs.extend(payload.get("series", []))
        if not payload.get("has_more"):
            break
        next_cursor = payload.get("next_cursor")
        if not next_cursor:
            break

    if not list_obs:
        raise RuntimeError("FRED release 387 returned no series observations.")
    return (list_obs,)


@app.cell
def _(INTERMEDIATE, RAW, assert_geo_columns, geo_expr, list_obs, pl):
    raw_rows = []
    for series in list_obs:
        if series.get("frequency") != "Annual":
            continue
        series_id = str(series["series_id"]).strip()
        units = series.get("units")
        seasonal_adjustment = series.get("seasonal_adjustment")
        for obs in series.get("observations", []):
            raw_rows.append(
                {
                    "series_id": series_id,
                    "date": str(obs["date"]),
                    "value": obs.get("value"),
                    "units": units,
                    "seasonal_adjustment": seasonal_adjustment,
                }
            )

    if not raw_rows:
        raise RuntimeError("FRED release 387 returned no Annual observations.")

    annual_df = (
        pl.DataFrame(raw_rows)
        .with_columns(
            pl.col("date").str.slice(0, 4).cast(pl.Int32).alias("year"),
            pl.col("value").cast(pl.Float64, strict=False).alias("wage_value"),
            pl.when(pl.col("series_id") == "STTMINWGFG")
            .then(pl.lit("US"))
            .otherwise(pl.col("series_id").str.slice(-2, 2))
            .alias("state_abbreviation"),
        )
    )

    federal_df = (
        annual_df.filter(pl.col("series_id") == "STTMINWGFG")
        .select(
            "year",
            pl.col("wage_value").alias("federal_min_wage"),
            "units",
            "seasonal_adjustment",
        )
        .filter(
            pl.col("federal_min_wage").is_not_null()
            & pl.col("federal_min_wage").is_finite()
            & (pl.col("federal_min_wage") > 0)
        )
        .sort("year")
    )
    if (
        federal_df.is_empty()
        or federal_df["year"].n_unique() != federal_df.height
    ):
        raise ValueError(
            "Federal series STTMINWGFG must be nonempty and unique by year."
        )

    all_fips_lookup = (
        pl.read_csv(RAW / "geographic_crosswalks" / "phil" / "fips_codes.csv")
        .select(
            pl.col("state").cast(pl.String).alias("state_name"),
            pl.col("state_abbrev").cast(pl.String).alias("state_abbreviation"),
            pl.col("fips").cast(pl.Int32).alias("fips_int"),
            geo_expr("fips", "state_fips"),
        )
        .unique()
    )

    observed_abbreviations = (
        annual_df.filter(pl.col("series_id") != "STTMINWGFG")
        .select("state_abbreviation")
        .unique()
    )
    unexpected_abbreviations = (
        observed_abbreviations.join(
            all_fips_lookup.select("state_abbreviation"),
            on="state_abbreviation",
            how="anti",
        )
        .get_column("state_abbreviation")
        .to_list()
    )
    if unexpected_abbreviations:
        raise ValueError(
            "Unexpected minimum-wage state abbreviations: "
            f"{sorted(unexpected_abbreviations)}"
        )

    states_51 = (
        all_fips_lookup.filter(pl.col("fips_int") <= 56)
        .select("state_fips", "state_abbreviation", "state_name")
        .sort("state_fips")
    )
    if (
        states_51.height != 51
        or states_51["state_fips"].n_unique() != 51
    ):
        raise ValueError(
            f"Expected 51 canonical states/DC in fips_codes.csv, got {states_51.height}."
        )

    state_obs = (
        annual_df.filter(pl.col("series_id") != "STTMINWGFG")
        .join(
            states_51.select("state_fips", "state_abbreviation"),
            on="state_abbreviation",
            how="inner",
            validate="m:1",
        )
        .select(
            "state_fips",
            "year",
            pl.col("wage_value").alias("state_min_wage"),
        )
    )
    state_dups = (
        state_obs.group_by(["state_fips", "year"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if not state_dups.is_empty():
        raise ValueError(
            "Annual FRED state minimum-wage observations are not unique by state_fips and year."
        )

    out_of_range_years = (
        state_obs.select("year")
        .unique()
        .join(federal_df.select("year"), on="year", how="anti")
        .get_column("year")
        .to_list()
    )
    if out_of_range_years:
        raise ValueError(
            "State minimum-wage observations fall outside STTMINWGFG annual years: "
            f"{sorted(out_of_range_years)}"
        )

    min_wage_df = (
        states_51.join(federal_df, how="cross")
        .join(state_obs, on=["state_fips", "year"], how="left", validate="1:1")
        .with_columns(
            pl.max_horizontal("state_min_wage", "federal_min_wage").alias(
                "prevailing_min_wage"
            )
        )
        .select(
            "state_fips",
            "state_abbreviation",
            "state_name",
            "year",
            "state_min_wage",
            "federal_min_wage",
            "prevailing_min_wage",
            "units",
            "seasonal_adjustment",
        )
        .sort(["state_fips", "year"])
    )

    assert_geo_columns(min_wage_df, ["state_fips"])
    expected_rows = 51 * federal_df.height
    if (
        min_wage_df.height != expected_rows
        or min_wage_df.schema["year"] != pl.Int32
        or min_wage_df["federal_min_wage"].null_count() > 0
        or min_wage_df["prevailing_min_wage"].null_count() > 0
        or not min_wage_df.select(
            (pl.col("prevailing_min_wage") >= pl.col("federal_min_wage")).all()
        ).item()
    ):
        raise ValueError("Constructed state_year_min_wage panel failed contract checks.")

    min_wage_df.write_parquet(INTERMEDIATE / "state_year_min_wage.parquet")
    print(
        f"Wrote state_year_min_wage.parquet: {min_wage_df.height:,} rows "
        f"(51 states/DC x {federal_df['year'].min()}-{federal_df['year'].max()})"
    )


if __name__ == "__main__":
    app.run()
