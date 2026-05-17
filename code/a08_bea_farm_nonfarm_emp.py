import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from h2a.paths import RAW, INTERMEDIATE
    import polars as pl

    return INTERMEDIATE, RAW, pl


@app.cell
def _(RAW, pl):
    bea = pl.read_csv(
        RAW / "bea" / "CAEMP25N__ALL_AREAS_2001_2022.csv",
        infer_schema_length=0,
        schema_overrides={
            "GeoFIPS": pl.String,
            "LineCode": pl.String,
        },
        encoding="cp1252",
    )
    year_cols = [c for c in bea.columns if c.startswith("20")]
    return bea, year_cols


@app.cell
def _(bea, pl, year_cols):
    def employment_long(line_code: str, value_name: str) -> pl.DataFrame:
        return (
            bea.filter(pl.col("LineCode") == line_code)
            .unpivot(
                index="GeoFIPS",
                on=year_cols,
                variable_name="year",
                value_name=value_name,
            )
            .with_columns(
                pl.col("GeoFIPS").str.strip_chars().str.strip_chars('"').str.zfill(5),
                pl.col("year").cast(pl.Int32),
                pl.col(value_name).str.replace_all(",", "").cast(pl.Float64, strict=False),
            )
            .select("GeoFIPS", "year", value_name)
        )

    return (employment_long,)


@app.cell
def _(INTERMEDIATE, employment_long):
    bea_farm = employment_long("70", "bea_farm_emp")
    bea_nonfarm = employment_long("80", "bea_nonfarm_emp")

    bea_farm_nonfarm = bea_farm.join(
        bea_nonfarm, on=["GeoFIPS", "year"], how="left"
    ).rename({"GeoFIPS": "county_fips"})

    bea_farm_nonfarm.write_parquet(INTERMEDIATE / "bea_farm_nonfarm_emp.parquet")
    return


if __name__ == "__main__":
    app.run()
