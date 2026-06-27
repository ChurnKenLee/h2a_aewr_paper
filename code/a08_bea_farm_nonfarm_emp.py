import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from h2a.paths import RAW, INTERMEDIATE
    import polars as pl
    import zipfile

    return INTERMEDIATE, RAW, pl, zipfile


@app.cell
def _(RAW):
    caemp_zip = RAW / "bea" / "CAEMP25N.zip"
    cainc_zip = RAW / "bea" / "CAINC45.zip"
    return caemp_zip, cainc_zip


@app.cell
def _(caemp_zip, cainc_zip, pl, zipfile):
    # We have to extract csv as raw data before passing to polars
    # Tis because polars does not apply encoding when streamed data
    with zipfile.ZipFile(caemp_zip) as _z:
        _data = _z.read("CAEMP25N__ALL_AREAS_2001_2022.csv")
        caemp_raw = pl.read_csv(
            _data,
            infer_schema=False,
            schema_overrides={
                "GeoFIPS": pl.String,
                "LineCode": pl.String,
            },
            encoding="cp1252",
        )
    with zipfile.ZipFile(cainc_zip) as _z:
        _data = _z.read("CAINC45__ALL_AREAS_1969_2022.csv")
        cainc_raw = pl.read_csv(
            _data,
            infer_schema=False,
            schema_overrides={
                "GeoFIPS": pl.String,
                "LineCode": pl.String,
            },
            encoding="cp1252",
        )
    return caemp_raw, cainc_raw


@app.cell
def _(INTERMEDIATE, caemp_raw, cainc_raw, pl):
    # Remove footers and prepend "y" to year columns
    # This is needed for Phil's code, as he previously cleaned this by hand
    caemp_year_cols = [_c for _c in caemp_raw.columns if _c.startswith("1")] + [
        _c for _c in caemp_raw.columns if _c.startswith("2")
    ]
    caemp = caemp_raw.filter(
        # pl.col("GeoName") != "United States",
        pl.col("GeoName").is_not_null()
    )
    caemp_trim = caemp.with_columns(pl.col(caemp_year_cols).name.prefix("y")).drop(
        pl.col(caemp_year_cols)
    )
    caemp_trim.write_parquet(INTERMEDIATE / "bea_CAEMP25N_trim.parquet")

    cainc_year_cols = [_c for _c in cainc_raw.columns if _c.startswith("1")] + [
        _c for _c in cainc_raw.columns if _c.startswith("2")
    ]
    cainc = cainc_raw.filter(
        # pl.col("GeoName") != "United States",
        pl.col("GeoName").is_not_null()
    )
    cainc_trim = cainc.with_columns(pl.col(cainc_year_cols).name.prefix("y")).drop(
        pl.col(cainc_year_cols)
    )
    cainc_trim.write_parquet(INTERMEDIATE / "bea_CAINC45_trim.parquet")
    return caemp, caemp_year_cols


@app.cell
def _(pl):
    def employment_long(df, year_cols, line_code: str, value_name: str) -> pl.DataFrame:
        return (
            df.filter(pl.col("LineCode") == line_code)
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
def _(INTERMEDIATE, caemp, caemp_year_cols, employment_long):
    bea_farm = employment_long(caemp, caemp_year_cols, "70", "bea_farm_emp")
    bea_nonfarm = employment_long(caemp, caemp_year_cols, "80", "bea_nonfarm_emp")
    bea_farm_nonfarm = bea_farm.join(
        bea_nonfarm, on=["GeoFIPS", "year"], how="left"
    ).rename({"GeoFIPS": "county_fips"})

    bea_farm_nonfarm.write_parquet(INTERMEDIATE / "bea_farm_nonfarm_emp.parquet")
    return


if __name__ == "__main__":
    app.run()
