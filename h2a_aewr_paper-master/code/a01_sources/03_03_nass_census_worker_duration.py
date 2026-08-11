import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    # Purpose: Derive county hired-worker duration and payroll from the Census of Agriculture.
    # Inputs: data/intermediate/qs_census_economics.parquet
    # Outputs: data/intermediate/census_ag_hired_worker_duration_county.parquet

    from h2a.paths import INTERMEDIATE
    import polars as pl

    input_path = INTERMEDIATE / "qs_census_economics.parquet"
    output_path = INTERMEDIATE / "census_ag_hired_worker_duration_county.parquet"
    return input_path, output_path, pl


@app.cell
def _(input_path, pl):
    from h2a.geography import (
        county_code_from_county_fips,
        harmonize_county_fips_2010,
    )

    labor_items = {
        "LABOR, HIRED - NUMBER OF WORKERS": "census_hired_workers_total",
        "LABOR, HIRED - EXPENSE, MEASURED IN $": (
            "census_hired_labor_expense"
        ),
        "LABOR, HIRED, GE 150 DAYS - NUMBER OF WORKERS": (
            "census_hired_workers_150_days_or_more"
        ),
        "LABOR, HIRED, LT 150 DAYS - NUMBER OF WORKERS": (
            "census_hired_workers_less_than_150_days"
        ),
    }

    census_labor = (
        pl.scan_parquet(input_path)
        .filter(
            (pl.col("year") >= 2007)
            & (pl.col("agg_level_desc") == "COUNTY")
            & (pl.col("freq_desc") == "ANNUAL")
            & (pl.col("reference_period_desc") == "YEAR")
            & (pl.col("commodity_desc") == "LABOR")
            & (pl.col("domain_desc") == "TOTAL")
            & (pl.col("prodn_practice_desc") == "ALL PRODUCTION PRACTICES")
            & pl.col("short_desc").is_in(list(labor_items))
        )
        .with_columns(
            pl.when(pl.col("value") == "(Z)")
            .then(pl.lit(0.0))
            .otherwise(
                pl.col("value").str.replace_all(",", "").cast(pl.Float64, strict=False)
            )
            .alias("numeric_value"),
            pl.col("state_fips").cast(pl.String).str.pad_start(2, "0"),
            pl.col("county_code").cast(pl.String).str.pad_start(3, "0"),
            pl.col("state_alpha").cast(pl.String),
            pl.col("state_name").cast(pl.String),
            pl.col("county_name").cast(pl.String),
            pl.col("short_desc").cast(pl.String),
        )
        .select(
            "year",
            "state_fips",
            "state_alpha",
            "state_name",
            "county_code",
            "county_name",
            "short_desc",
            "numeric_value",
        )
        .collect()
    )

    census_labor = census_labor.pivot(
        on="short_desc",
        index=[
            "year",
            "state_fips",
            "state_alpha",
            "state_name",
            "county_code",
            "county_name",
        ],
        values="numeric_value",
    )

    census_labor = census_labor.rename(labor_items)

    census_labor = (
        census_labor.with_columns(
            pl.concat_str("state_fips", "county_code")
            .map_elements(
                harmonize_county_fips_2010,
                return_dtype=pl.String,
            )
            .alias("county_fips"),
            (
                pl.col("census_hired_workers_150_days_or_more")
                + pl.col("census_hired_workers_less_than_150_days")
            ).alias("census_hired_workers_duration_total"),
        )
        .with_columns(
            pl.col("county_fips")
            .map_elements(
                county_code_from_county_fips,
                return_dtype=pl.String,
            )
            .alias("county_code"),
            (
                pl.col("census_hired_workers_total").is_not_null()
                & pl.col("census_hired_workers_150_days_or_more").is_not_null()
                & pl.col("census_hired_workers_less_than_150_days").is_not_null()
            ).alias("census_hired_worker_duration_complete"),
            pl.when(pl.col("census_hired_workers_duration_total") > 0)
            .then(
                pl.col("census_hired_workers_150_days_or_more")
                / pl.col("census_hired_workers_duration_total")
            )
            .otherwise(None)
            .alias("census_hired_worker_150_plus_share"),
            pl.when(pl.col("census_hired_workers_duration_total") > 0)
            .then(
                pl.col("census_hired_workers_less_than_150_days")
                / pl.col("census_hired_workers_duration_total")
            )
            .otherwise(None)
            .alias("census_hired_worker_less_than_150_share"),
        )
        .select(
            "county_fips",
            "year",
            "state_fips",
            "state_alpha",
            "state_name",
            "county_code",
            "county_name",
            "census_hired_workers_total",
            "census_hired_labor_expense",
            "census_hired_workers_150_days_or_more",
            "census_hired_workers_less_than_150_days",
            "census_hired_workers_duration_total",
            "census_hired_worker_duration_complete",
            "census_hired_worker_150_plus_share",
            "census_hired_worker_less_than_150_share",
        )
        .sort("county_fips", "year")
    )
    return (census_labor,)


@app.cell
def _(census_labor, output_path):
    from h2a.geography import assert_geo_columns

    assert_geo_columns(
        census_labor,
        ["state_fips", "county_code", "county_fips"],
    )
    census_labor.write_parquet(output_path)

    print(f"Wrote {census_labor.height:,} county-census rows to {output_path}")
    return


if __name__ == "__main__":
    app.run()
