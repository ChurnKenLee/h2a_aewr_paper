# Purpose: Aggregate gNATSGO soil attributes to county prediction cells.
# Inputs: the gNATSGO archive and the 2010 county shapefile.
# Outputs: county_h2a_prediction_gnatsgo_soil_cells.parquet and extraction cache.

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from h2a.paths import CODE, RAW, INTERMEDIATE, CACHE
    import polars as pl
    import geopandas as gpd
    import numpy as np
    import py7zr
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.shutil import copy as rio_copy
    import exactextract
    import sqlite3

    return (
        CACHE,
        INTERMEDIATE,
        RAW,
        exactextract,
        gpd,
        mo,
        np,
        pl,
        py7zr,
        rasterio,
        sqlite3,
    )


@app.cell
def _(CACHE, INTERMEDIATE, RAW):
    binary_path = INTERMEDIATE

    gnatsgo_archive = RAW / "gnatsgo" / "gNATSGO_gpkg_02_13_2026.7z"
    gnatsgo_cache_path = CACHE / "gnatsgo"
    gnatsgo_cache_path.mkdir(parents=True, exist_ok=True)

    census_shp_path = (
        f"/vsizip/{RAW / 'county_shapefile' / 'tl_2010_us_county10.zip'}"
        "/tl_2010_us_county10.shp"
    )
    return binary_path, census_shp_path, gnatsgo_archive, gnatsgo_cache_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ExactExtract cannot handle uint32, we will have to convert raster to int32 first
    """)
    return


@app.cell
def _(gnatsgo_archive, gnatsgo_cache_path, np, py7zr, rasterio):
    raster_member = "gNATSGO_gpkg_01_30_2026/MURASTER_30m_CONUS_2026.tif"
    extracted_raster = gnatsgo_cache_path / raster_member
    converted_raster = gnatsgo_cache_path / "MURASTER_30m_CONUS_2026_int32_zstd.tif"

    if not converted_raster.exists():
        if not extracted_raster.exists():
            print("Extracting only MURASTER_30m_CONUS_2026.tif from gNATSGO archive...")
            with py7zr.SevenZipFile(gnatsgo_archive, mode="r") as _archive:
                _archive.extract(
                    path=gnatsgo_cache_path,
                    targets=[raster_member],
                )

        print("Converting gNATSGO raster from uint32 to compressed int32...")

        with rasterio.open(extracted_raster) as _src:
            _profile = _src.profile

            _nodata_val = int(_src.nodata) if _src.nodata is not None else None
            if _nodata_val is not None and _nodata_val > 2147483647:
                _nodata_val = -1

            _profile.update(
                dtype=rasterio.int32,
                nodata=_nodata_val,
                compress="ZSTD",
                zstd_level=9,
                tiled=True,
                blockxsize=512,
                blockysize=512,
                bigtiff="IF_SAFER",
                num_threads="ALL_CPUS",
            )

            tmp_raster = converted_raster.with_suffix(".tmp.tif")
            if tmp_raster.exists():
                tmp_raster.unlink()

            with rasterio.open(tmp_raster, "w", **_profile) as _dst:
                for _, _window in _src.block_windows(1):
                    _data = _src.read(1, window=_window)
                    _dst.write(_data.astype(np.int32), 1, window=_window)

            tmp_raster.replace(converted_raster)

        extracted_raster.unlink()
        print(f"Raster saved to {converted_raster}")
    else:
        print(f"Found existing converted raster: {converted_raster}")
    return (converted_raster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Table A: Acreage of map units within each county
    """)
    return


@app.cell
def _(census_shp_path, converted_raster, gpd, rasterio):
    # Raster
    src = rasterio.open(converted_raster)
    # County shapefile
    counties = gpd.read_file(census_shp_path)

    # Reproject county shapefiles to match raster CRS
    # This is sually EPSG:5070 for accurate area calculations
    if counties.crs != src.crs:
        counties = counties.to_crs(src.crs)

    # County identifier in shapefile
    county_id_col = "GEOID10"

    # Pixel acreage
    pixel_width, pixel_height = src.res  # this is in meters
    sq_meters_per_pixel = abs(pixel_width * pixel_height)
    sq_meters_per_acre = 4046.872
    acres_per_pixel = sq_meters_per_pixel / sq_meters_per_acre

    nodata_val = src.nodata
    return acres_per_pixel, counties, county_id_col, nodata_val, src


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If extract already performed, skip extraction (takes about 15 minutes)
    """)
    return


@app.cell
def _(
    acres_per_pixel,
    binary_path,
    counties,
    county_id_col,
    exactextract,
    nodata_val,
    pl,
    src,
):
    table_a_parquet = binary_path / "county_mapunit_pixel_count_exactextract.parquet"

    if not table_a_parquet.exists():
        print("Extracting raster pixels per county using exactextract...")
        # exact_extract evaluates operations requested in a list.
        # "values" = raw pixel array, "coverage" = exact intersection fraction array.
        extracted_features = exactextract.exact_extract(
            src, counties, ["values", "coverage"], include_cols=[county_id_col]
        )

        table_a_rows = []

        for feature in extracted_features:
            props = feature["properties"]
            geoid = props[county_id_col]

            # Grab the dynamic keys for values and coverage arrays
            val_key = next((k for k in props.keys() if "values" in k), "values")
            cov_key = next((k for k in props.keys() if "coverage" in k), "coverage")

            # Create a Polars DataFrame for this specific county
            df_county = pl.DataFrame({"mukey": props[val_key], "fraction": props[cov_key]})

            # Filter out NoData gaps
            if nodata_val is not None:
                df_county = df_county.filter(
                    pl.col("mukey").is_not_null() & (pl.col("mukey") != nodata_val)
                )

            # Group by Map Unit Key (mukey), sum exact fractions to get "pixel count"
            df_county_agg = (
                df_county.group_by("mukey")
                .agg(pl.col("fraction").sum().alias("pixel_count"))
                .with_columns(pl.lit(geoid).alias("county_id"))
            )

            table_a_rows.append(df_county_agg)

        # Combine into Table A (County | Map Unit | Acres)
        table_a = pl.concat(table_a_rows)

        # Convert pixel counts to acres and cast mukeys to String for relational join later
        table_a = table_a.with_columns(
            [
                (pl.col("pixel_count") * acres_per_pixel).alias("mapunit_acres"),
                pl.col("mukey").cast(pl.String),
            ]
        )

        table_a.write_parquet(table_a_parquet)
        print("Extraction and aggregation complete")
    else:
        print(f"{table_a_parquet} already exists")
        table_a = pl.read_parquet(table_a_parquet)
    return (table_a,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Table B: Components within each map unit
    """)
    return


@app.cell
def _(gnatsgo_archive, gnatsgo_cache_path, pl, py7zr, sqlite3):
    gpkg_member = "gNATSGO_gpkg_01_30_2026/gNATSGO_02_13_2026.gpkg"
    gpkg_path = gnatsgo_cache_path / gpkg_member

    if not gpkg_path.exists():
        print("Extracting only gNATSGO_02_13_2026.gpkg from gNATSGO archive...")
        with py7zr.SevenZipFile(gnatsgo_archive, mode="r") as _archive:
            _archive.extract(path=gnatsgo_cache_path, targets=[gpkg_member])

    conn = sqlite3.connect(f"file:{gpkg_path.resolve()}?mode=ro", uri=True)
    tables_query = (
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    tables = pl.read_database(tables_query, connection=conn)
    return (conn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Grab the tables we want then join
    """)
    return


@app.cell
def _(conn, pl):
    component_query = """
    SELECT
        mukey,
        cokey,
        compname,
        comppct_r,
        majcompflag,
        taxorder,
        taxsuborder,
        taxgrtgroup,
        taxsubgrp,
        taxpartsize,
        taxreaction,
        slope_r,
        drainagecl,
        hydgrp,
        hydricrating,
        nirrcapcl,
        irrcapcl,
        cropprodindex
    FROM component
    """

    mapunit_query = """
    SELECT
        mukey,
        muname
    FROM mapunit
    """

    restriction_query = """
    SELECT
        cokey,
        resdept_r
    FROM corestrictions
    """

    muaggatt_query = """
    SELECT
        mukey,
        slopegradwta,
        aws025wta,
        aws050wta,
        aws0100wta,
        aws0150wta,
        wtdepannmin,
        wtdepaprjunmin,
        brockdepmin,
        flodfreqmax,
        pondfreqprs
    FROM muaggatt
    """

    component = pl.read_database(
        query=component_query, infer_schema_length=None, connection=conn
    )
    mapunit = pl.read_database(
        query=mapunit_query, infer_schema_length=None, connection=conn
    )
    restriction = pl.read_database(
        query=restriction_query, infer_schema_length=None, connection=conn
    )
    muaggatt = pl.read_database(
        query=muaggatt_query, infer_schema_length=None, connection=conn
    )

    # Each cokey can have multiple resdept_r because restrictions have types
    # Need to aggregate to min depth before joining or we will create duplicate rows
    restriction = restriction.group_by("cokey").agg(
        pl.col("resdept_r").min().alias("resdept_r")
    )

    joined_table = (
        component.join(mapunit, on="mukey", how="inner")
        .join(restriction, on="cokey", how="left")
        .join(muaggatt, on="mukey", how="left")
    )
    return (joined_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # County soil cells
    """)
    return


@app.cell
def _():
    soil_cell_cols = [
        "taxorder",
        "taxsuborder",
        "taxgrtgroup",
        "drainagecl",
        "hydgrp",
        "nirrcapcl",
    ]

    optional_cat_cols = [
        "taxsubgrp",
        "taxpartsize",
        "taxreaction",
        "hydricrating",
        "irrcapcl",
        "flodfreqmax",
        "pondfreqprs",
    ]

    cont_cols = [
        "slope_r",
        "slopegradwta",
        "resdept_r",
        "aws025wta",
        "aws050wta",
        "aws0100wta",
        "aws0150wta",
        "wtdepannmin",
        "wtdepaprjunmin",
        "brockdepmin",
        "cropprodindex",
    ]
    return cont_cols, optional_cat_cols, soil_cell_cols


@app.cell
def _(cont_cols, joined_table, optional_cat_cols, pl, soil_cell_cols, table_a):
    component_panel = (
        table_a.with_columns(pl.col("mukey").cast(pl.Int64).alias("mukey"))
        .join(joined_table, on="mukey", how="inner")
        .with_columns(
            [pl.col(_col).fill_null("MISSING").cast(pl.String) for _col in soil_cell_cols]
            + [
                pl.col(_col).fill_null("MISSING").cast(pl.String)
                for _col in optional_cat_cols
            ]
            + [
                pl.col("mapunit_acres").cast(pl.Float32),
                pl.col("comppct_r").fill_null(0).clip(0, 100).cast(pl.Float32),
            ]
            + [pl.col(_col).cast(pl.Float32) for _col in cont_cols]
        )
        .with_columns(
            (pl.col("mapunit_acres") * pl.col("comppct_r") / 100.0).alias("component_acres")
        )
        .filter(pl.col("component_acres") > 0)
    )

    weighted_exprs = []
    for _col in cont_cols:
        _valid_acres = (
            pl.when(pl.col(_col).is_not_null())
            .then(pl.col("component_acres"))
            .otherwise(0.0)
        )
        _weighted_sum = (
            pl.when(pl.col(_col).is_not_null())
            .then(pl.col(_col) * pl.col("component_acres"))
            .otherwise(0.0)
        )
        weighted_exprs.extend(
            [
                (_weighted_sum.sum() / _valid_acres.sum()).cast(pl.Float32).alias(_col),
                (_valid_acres.sum() / pl.col("component_acres").sum())
                .cast(pl.Float32)
                .alias(f"{_col}_obs_share"),
            ]
        )

    optional_mode_exprs = []
    for _col in optional_cat_cols:
        optional_mode_exprs.append(
            pl.col(_col)
            .sort_by("component_acres", descending=True)
            .first()
            .alias(f"dominant_{_col}")
        )

    county_soil_cells = (
        component_panel.group_by(["county_id"] + soil_cell_cols)
        .agg(
            [
                pl.col("component_acres").sum().cast(pl.Float32).alias("total_acres"),
                pl.col("mukey").n_unique().alias("n_mapunits"),
                pl.col("cokey").n_unique().alias("n_components"),
            ]
            + weighted_exprs
            + optional_mode_exprs
        )
        .rename({"county_id": "county_ansi"})
        .with_columns(pl.concat_str(soil_cell_cols, separator="|").alias("soil_cell_id"))
        .with_columns(
            pl.col("total_acres").sum().over("county_ansi").alias("county_soil_acres")
        )
        .with_columns(
            (pl.col("total_acres") / pl.col("county_soil_acres"))
            .cast(pl.Float32)
            .alias("acreage_frac")
        )
        .select(
            [
                "county_ansi",
                "soil_cell_id",
                *soil_cell_cols,
                "total_acres",
                "county_soil_acres",
                "acreage_frac",
                "n_mapunits",
                "n_components",
                *cont_cols,
                *[f"{_col}_obs_share" for _col in cont_cols],
                *[f"dominant_{_col}" for _col in optional_cat_cols],
            ]
        )
        .sort(["county_ansi", "total_acres"], descending=[False, True])
    )
    return (county_soil_cells,)


@app.cell
def _(binary_path, county_soil_cells):
    output_file = binary_path / "county_h2a_prediction_gnatsgo_soil_cells.parquet"
    county_soil_cells.write_parquet(output_file)
    print(
        f"Saved {county_soil_cells.height} county-soil cells "
        f"for {county_soil_cells.select('county_ansi').n_unique()} counties to {output_file}"
    )
    return


if __name__ == "__main__":
    app.run()
