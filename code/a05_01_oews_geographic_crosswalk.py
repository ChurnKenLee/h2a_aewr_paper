import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from h2a.paths import CODE, RAW, INTERMEDIATE, CACHE
    import dotenv, os
    import polars as pl
    import polars.selectors as cs

    return INTERMEDIATE, RAW, mo, pl


@app.cell
def _(INTERMEDIATE, RAW):
    binary_path = INTERMEDIATE
    oews_path = RAW / "oews"
    return binary_path, oews_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load OEWS locality codes for years before 2019
    """)
    return


@app.cell
def _(oews_path, pl):
    oews_df = pl.read_excel(
        oews_path / "oews_area_definitions" / "area_definitions.xlsx"
    ).select(pl.all().cast(pl.String))
    oews_df = oews_df.rename(
        {
            "FIPS": "oews_state_fips",
            "County code": "oews_county_fips",
            "Township code": "oews_township_code",
            "County/town name": "oews_county_name",
            "state": "oews_state_name",
        }
    )

    # Keep only columns we need
    columns_to_keep = [
        "oews_state_fips",
        "oews_county_fips",
        "oews_township_code",
        "oews_state_name",
        "oews_county_name",
    ]
    area_cols = [c for c in oews_df.columns if c.startswith("Area code")]
    area_name_cols = [c for c in oews_df.columns if c.startswith("Area name")]
    oews_df = oews_df.select(columns_to_keep + area_cols + area_name_cols)

    # Fill in years that retain the same code/name as prior years
    oews_df = oews_df.with_columns(
        [
            pl.col("Area code 2003").alias("Area code 2004"),
            pl.col("Area code 2006").alias("Area code 2007"),
            pl.col("Area code 2008").alias("Area code 2009"),
            pl.col("Area code 2011").alias("Area code 2012"),
            pl.col("Area code 2011").alias("Area code 2013"),
            pl.col("Area code 2011").alias("Area code 2014"),
            *[
                pl.col("Area name pre_2005").alias(f"Area name {year}")
                for year in range(1996, 2005)
            ],
            pl.col("Area name 2006").alias("Area name 2007"),
            pl.col("Area name 2011").alias("Area name 2012"),
            pl.col("Area name 2011").alias("Area name 2013"),
            pl.col("Area name 2011").alias("Area name 2014"),
            pl.col("Area name 2015").alias("Area name 2016"),
            pl.when(
                pl.col("Area name 2018").is_null()
                | (pl.col("Area name 2018").str.strip_chars() == "")
            )
            .then(pl.col("Area name 2017"))
            .otherwise(pl.col("Area name 2018"))
            .alias("Area name 2018"),
        ]
    )

    # The source crosswalk leaves Clark County, NV blank before 2015, but the
    # OEWS wage files publish Las Vegas-Paradise under area code 29820.
    clark_county_nv = (
        (pl.col("oews_state_name") == "Nevada")
        & (pl.col("oews_county_name") == "Clark County")
    )
    oews_df = oews_df.with_columns(
        [
            pl.when(
                clark_county_nv
                & (
                    pl.col(f"Area code {year}").is_null()
                    | (pl.col(f"Area code {year}").str.strip_chars() == "")
                )
            )
            .then(pl.lit("29820"))
            .otherwise(pl.col(f"Area code {year}"))
            .alias(f"Area code {year}")
            for year in range(2005, 2015)
        ]
    )
    oews_df = oews_df.with_columns(
        [
            pl.when(
                clark_county_nv
                & (
                    pl.col(f"Area name {year}").is_null()
                    | (pl.col(f"Area name {year}").str.strip_chars() == "")
                )
            )
            .then(pl.lit("Las Vegas-Paradise, NV"))
            .otherwise(pl.col(f"Area name {year}"))
            .alias(f"Area name {year}")
            for year in range(2005, 2015)
        ]
    )

    # Drop Guam, Puerto Rico, and Virgin Islands
    oews_df = oews_df.filter(
        ~pl.col("oews_state_name").is_in(["Guam", "Puerto Rico", "Virgin Islands"])
    )
    return (oews_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Save a long version of OEWS area codes for merging with OEWS data to get county-level observation
    """)
    return


@app.cell
def _(oews_df, oews_path, pl):
    # Update area_cols to include the newly mapped years
    new_area_code_cols = [c for c in oews_df.columns if c.startswith("Area code")]
    new_area_name_cols = [
        c
        for c in oews_df.columns
        if c.startswith("Area name ") and c.removeprefix("Area name ").isdigit()
    ]
    id_cols = [
        "oews_state_fips",
        "oews_county_fips",
        "oews_township_code",
        "oews_state_name",
        "oews_county_name",
    ]

    # Reshape long
    oews_code_long_df = oews_df.unpivot(
        index=id_cols,
        on=new_area_code_cols,
        variable_name="year",
        value_name="oews_area_code",
    ).with_columns(pl.col("year").str.replace("Area code ", "").cast(pl.Int32))

    oews_name_long_df = oews_df.unpivot(
        index=id_cols,
        on=new_area_name_cols,
        variable_name="year",
        value_name="oews_area_name",
    ).with_columns(pl.col("year").str.replace("Area name ", "").cast(pl.Int32))

    oews_long_df = oews_code_long_df.join(
        oews_name_long_df,
        on=id_cols + ["year"],
        how="left",
    )

    oews_long_df.filter(
        pl.col("oews_area_code").is_null()
        | (pl.col("oews_area_code").str.strip_chars() == "")
    ).write_csv(oews_path / "missing_area_codes.csv")
    return (oews_long_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load OEWS area definitions for years after 2018
    """)
    return


@app.cell
def _(oews_path, pl):
    oews_df_list = []
    for y in range(2019, 2025):
        # Read excel and cast to string to maintain consistency
        oews_df_1 = pl.read_excel(
            oews_path / "oews_area_definitions" / f"area_definitions_m{y}.xlsx"
        ).select(pl.all().cast(pl.String))

        area_code_variable = next(
            c for c in oews_df_1.columns if c.startswith(f"May {y} MSA code")
        )
        area_name_variable = next(
            c for c in oews_df_1.columns if c.startswith("May ") and c.endswith("MSA name")
        )
        county_name_variable = next(
            c for c in oews_df_1.columns if c.startswith("County name")
        )

        rename_map = {
            "FIPS code": "oews_state_fips",
            "County code": "oews_county_fips",
            "Township code": "oews_township_code",
            county_name_variable: "oews_county_name",
            "State": "oews_state_name",
            area_code_variable: "oews_area_code",
            area_name_variable: "oews_area_name",
        }

        # Safe renaming in case certain variables are absent in a specific year
        rename_map = {k: v for k, v in rename_map.items() if k in oews_df_1.columns}

        oews_df_1 = oews_df_1.rename(rename_map).with_columns(
            pl.lit(y).cast(pl.Int32).alias("year")
        )
        oews_df_1 = oews_df_1.select(
            [
                c
                for c in [
                    "oews_state_fips",
                    "oews_county_fips",
                    "oews_township_code",
                    "oews_state_name",
                    "oews_county_name",
                    "oews_area_code",
                    "oews_area_name",
                    "year",
                ]
                if c in oews_df_1.columns
            ]
        )
        oews_df_list.append(oews_df_1)
    return (oews_df_list,)


@app.cell
def _(oews_df_list, pl):
    # Combine post 2019 periods
    oews_df_post_2019 = pl.concat(oews_df_list, how="diagonal")
    return (oews_df_post_2019,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ensure county definitions match Census 2010 definition as in Phil's workflow
    """)
    return


@app.cell
def _(INTERMEDIATE, pl):
    full_county_set = (
        pl.read_parquet(INTERMEDIATE / "county_adjacency2010.parquet")
        .select(["fipscounty", "countyname"])
        .unique()
        .cast(pl.String)
        .with_columns(pl.col("fipscounty").str.pad_start(5, "0").alias("fipscounty"))
        .sort("fipscounty")
    )
    return (full_county_set,)


@app.cell
def _(oews_df_post_2019, oews_long_df, pl):
    # Zero-pad FIPS codes to cleanly standardize the string definitions
    # Combine both periods
    oews_long_df_both_periods = pl.concat([oews_long_df, oews_df_post_2019], how="diagonal")

    oews_long_df_both_periods = oews_long_df_both_periods.with_columns(
        [pl.col("oews_state_fips").str.zfill(2), pl.col("oews_county_fips").str.zfill(3)]
    ).with_columns(
        (pl.col("oews_state_fips") + pl.col("oews_county_fips")).alias("fipscounty")
    )

    # Remove whitespace around area codes and FIPS fields. Some source cells
    # carry leading spaces; trimming both ends keeps joins stable after export.
    oews_long_df_both_periods = oews_long_df_both_periods.with_columns(
        pl.col("oews_area_code").str.strip_chars(),
        pl.col("oews_area_name").str.strip_chars(),
    )
    return (oews_long_df_both_periods,)


@app.cell
def _(full_county_set, oews_long_df_both_periods, pl):
    # Don't care about states outside CONUS
    wrong_oews = (
        oews_long_df_both_periods.join(full_county_set, on="fipscounty", how="anti")
        .unique("fipscounty")
        .sort("fipscounty")
        .filter(~pl.col("oews_state_name").is_in(["Alaska", "Puerto Rico"]))
    )
    wrong_oews
    # Seems like the only problem is Miami-Dade (remap), Yellowstone (merged with Gallatin and Park, can ignore), Oglala Lakota (remap), Oak Ridge Reservation (Manhattan Project, can ignore), Clifton Forge (merged with Alleghany, can ignore)
    return


@app.cell
def _(oews_long_df_both_periods, pl):
    dade_county_fl = (
        (pl.col("oews_state_name") == "Florida")
        & pl.col("oews_county_name").str.contains("old def", literal=True)
    )
    oglala_lakota_county_sd = (
        (pl.col("oews_state_name") == "South Dakota")
        & (pl.col("oews_county_name") == "Oglala Lakota County")
    )
    deprecated_counties = (
        (
            (pl.col("oews_state_name") == "Montana")
            & pl.col("oews_county_name").str.starts_with("Yellowstone National Park")
        )
        | (
            (pl.col("oews_state_name") == "Tennessee")
            & (pl.col("oews_county_name") == "Oak Ridge Reservation")
        )
        | (
            (pl.col("oews_state_name") == "Virginia")
            & (pl.col("oews_county_name") == "Clifton Forge city")
        )
    )

    oews_remapped = oews_long_df_both_periods.with_columns(
        # Dade to Miami-Dade, Oglala Lakota to Shannon
        pl.when(dade_county_fl)
        .then(pl.lit("086"))
        .when(oglala_lakota_county_sd)
        .then(pl.lit("113"))
        .otherwise(pl.col("oews_county_fips"))
        .alias("oews_county_fips")
    )
    # Drop deprecated counties
    oews_remapped = oews_remapped.filter(~deprecated_counties).drop("fipscounty")
    return (oews_remapped,)


@app.cell
def _(binary_path, oews_remapped):
    # Export as binary
    oews_remapped.write_parquet(
        binary_path / "oews_area_definitions.parquet",
    )
    return


if __name__ == "__main__":
    app.run()
