import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    # Purpose: Derive county hired-worker duration shares from the Census of Agriculture.
    # Inputs: data/intermediate/qs_census_economics.parquet
    # Outputs: data/intermediate/census_ag_hired_worker_duration_county.parquet

    from h2a.paths import INTERMEDIATE
    import polars as pl

    input_path = INTERMEDIATE / "qs_census_economics.parquet"
    output_path = INTERMEDIATE / "census_ag_hired_worker_duration_county.parquet"
    return input_path, output_path, pl


@app.cell
def _(input_path, pl):
    worker_items = {
        "LABOR, HIRED - NUMBER OF WORKERS": "census_hired_workers_total",
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
            & (pl.col("statisticcat_desc") == "WORKERS")
            & (pl.col("unit_desc") == "NUMBER")
            & (pl.col("domain_desc") == "TOTAL")
            & (pl.col("prodn_practice_desc") == "ALL PRODUCTION PRACTICES")
            & pl.col("short_desc").is_in(list(worker_items))
        )
        .with_columns(
            pl.when(pl.col("value") == "(Z)")
            .then(pl.lit(0.0))
            .otherwise(
                pl.col("value").str.replace_all(",", "").cast(pl.Float64, strict=False)
            )
            .alias("numeric_value"),
            pl.col("state_fips_code").cast(pl.String).str.pad_start(2, "0"),
            pl.col("county_code").cast(pl.String).str.pad_start(3, "0"),
            pl.col("state_alpha").cast(pl.String),
            pl.col("state_name").cast(pl.String),
            pl.col("county_name").cast(pl.String),
            pl.col("short_desc").cast(pl.String),
        )
        # Match the 2010 county definition used elsewhere in the project.
        .with_columns(
            pl.when(
                (pl.col("state_fips_code") == "46") & (pl.col("county_code") == "102")
            )
            .then(pl.lit("113"))
            .otherwise(pl.col("county_code"))
            .alias("county_code")
        )
        .select(
            "year",
            "state_fips_code",
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
            "state_fips_code",
            "state_alpha",
            "state_name",
            "county_code",
            "county_name",
        ],
        values="numeric_value",
    )

    census_labor = census_labor.rename(worker_items)

    census_labor = (
        census_labor.with_columns(
            pl.concat_str("state_fips_code", "county_code").alias("countyfips"),
            (
                pl.col("census_hired_workers_150_days_or_more")
                + pl.col("census_hired_workers_less_than_150_days")
            ).alias("census_hired_workers_duration_total"),
        )
        .with_columns(
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
            "countyfips",
            "year",
            "state_fips_code",
            "state_alpha",
            "state_name",
            "county_code",
            "county_name",
            "census_hired_workers_total",
            "census_hired_workers_150_days_or_more",
            "census_hired_workers_less_than_150_days",
            "census_hired_workers_duration_total",
            "census_hired_worker_duration_complete",
            "census_hired_worker_150_plus_share",
            "census_hired_worker_less_than_150_share",
        )
        .sort("countyfips", "year")
    )
    return (census_labor,)


@app.cell
def _(census_labor, output_path):
    census_labor.write_parquet(output_path)

    print(f"Wrote {census_labor.height:,} county-census rows to {output_path}")
    return


if __name__ == "__main__":
    app.run()
