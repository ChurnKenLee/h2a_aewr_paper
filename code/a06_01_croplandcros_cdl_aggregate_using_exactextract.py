import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from h2a.paths import CODE, RAW, INTERMEDIATE, CACHE
    from zipfile import ZipFile
    import dotenv, os
    import polars as pl
    import polars.selectors as cs
    import pandas as pd
    import geopandas as gpd
    import rasterio as rio
    from rasterio.shutil import copy as rio_copy
    import exactextract
    from exactextract import exact_extract, operation
    import time

    return (
        CACHE,
        INTERMEDIATE,
        RAW,
        ZipFile,
        cs,
        exact_extract,
        gpd,
        pd,
        pl,
        rio,
        rio_copy,
    )


@app.cell
def _(CACHE, RAW):
    cdl_path = RAW / "croplandcros_cdl"
    cdl_cache_path = CACHE / "croplandcros_cdl_tif"

    # vsizip is a GDAL virtual path for reading files without extracting from archive
    county_shapefile_path = (
        f"/vsizip/{RAW / 'county_shapefile' / 'tl_2010_us_county10.zip'}"
        "/tl_2010_us_county10.shp"
    )
    return cdl_cache_path, cdl_path, county_shapefile_path


@app.cell
def _(ZipFile, cdl_cache_path, cdl_path):
    def materialize_cdl_metadata(year):
        zip_path = cdl_path / f"{year}_30m_cdls.zip"
        zip_htm_path = f"/vsizip/{zip_path}/Metadata_Cropland-Data-Layer.htm"
        out_htm_path = cdl_cache_path / f"{year}_metadata"

        if (
            out_htm_path.exists()
            and out_htm_path.stat().st_mtime >= zip_path.stat().st_mtime
        ):
            return out_htm_path

        cdl_cache_path.mkdir(parents=True, exist_ok=True)

        with ZipFile(zip_path, "r") as zip_ref:
            htm_file = [f for f in zip_ref.namelist() if f.endswith(".htm")][0]
            zip_ref.extract(htm_file, out_htm_path)

        return out_htm_path

    return (materialize_cdl_metadata,)


@app.cell
def _(cdl_cache_path, cdl_path, rio_copy):
    def materialize_cdl_tif(year):
        zip_path = cdl_path / f"{year}_30m_cdls.zip"
        zip_tif_path = f"/vsizip/{zip_path}/{year}_30m_cdls.tif"
        out_tif_path = cdl_cache_path / f"{year}_30m_cdls.tif"

        if (
            out_tif_path.exists()
            and out_tif_path.stat().st_mtime >= zip_path.stat().st_mtime
        ):
            return out_tif_path

        cdl_cache_path.mkdir(parents=True, exist_ok=True)
        tmp_tif_path = out_tif_path.with_suffix(".tmp.tif")
        if tmp_tif_path.exists():
            tmp_tif_path.unlink()

        rio_copy(
            zip_tif_path,
            tmp_tif_path,
            driver="GTiff",
            compress="ZSTD",
            zstd_level=9,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            bigtiff="IF_SAFER",
            num_threads="ALL_CPUS",
        )

        tmp_tif_path.replace(out_tif_path)
        return out_tif_path

    return (materialize_cdl_tif,)


@app.cell
def _(materialize_cdl_metadata, materialize_cdl_tif, rio):
    cdl_files = {}
    for _year in range(2008, 2025):
        _cropland_data_layer_path = materialize_cdl_tif(_year)
        _metadata_path = materialize_cdl_metadata(_year)

        cdl_files[_year] = _cropland_data_layer_path
        with rio.open(_cropland_data_layer_path) as _cropland_data_layer:
            _cdl_crs_epsg = _cropland_data_layer.crs.to_epsg()
            print(f"{_year} has EPSG {_cdl_crs_epsg}")
    return (cdl_files,)


@app.cell
def _(cdl_files, county_shapefile_path, gpd, rio):
    # Reproject county shapefile to match CDL CRS
    # Read county border vector file
    county_borders = gpd.read_file(county_shapefile_path)

    # 3. Open rasters and ensure CRS match
    # CDL is almost always EPSG:5070 (USA Contiguous Albers Equal Area)
    # TIGER files are usually EPSG:4269. We reproject the counties.
    with rio.open(list(cdl_files.values())[0]) as src:
        cdl_crs = src.crs
        county_borders = county_borders.to_crs(cdl_crs)

    # We only want CONUS
    statefp_drop_list = [
        "02",  # AK
        "11",  # DC
        "15",  # HI
        "60",  # AS
        "66",  # GU
        "69",  # MP
        "72",  # PR
        "78",  # VI
        "74",  # UM
    ]

    county_borders_conus = county_borders[
        ~county_borders["STATEFP10"].isin(statefp_drop_list)
    ]
    county_borders_conus = county_borders_conus[["GEOID10", "geometry"]].reset_index(
        drop=True
    )
    return (county_borders_conus,)


@app.cell
def _(INTERMEDIATE, cdl_files, county_borders_conus, exact_extract, pd, pl):
    # We open all years simultaneously and pass them as a list
    # exactextract will compute weights once and apply to all bands (years)
    print("Starting extraction (this may take a while for CONUS)...")
    raster_handles = [path for path in cdl_files.values()]

    # Calculate zonal stats if not already done
    temp_ee = INTERMEDIATE / "temp_exactextract.parquet"
    if temp_ee.is_file():
        print("Zonal stats already exist")
        results = pd.read_parquet(temp_ee)

    else:
        results = exact_extract(
            rast=raster_handles,
            vec=county_borders_conus,
            ops=["count", "unique", "frac"],
            include_cols=["GEOID10"],
            output="pandas",
        )
        results.to_parquet(temp_ee)
        print("Finished extracting")

    results = pl.from_pandas(results)
    return (results,)


@app.cell
def _(cs, pl):
    # Define a function that unpivots each column type separately
    def unpivot_to_long_metric(df, suffix, value_name):
        long_df = (
            df.select(["GEOID10", cs.contains(suffix)])
            .unpivot(index="GEOID10", variable_name="year", value_name=value_name)
            .with_columns(
                pl.col("year").str.extract(r"^(\d{4})", 1)  # Extract 2008, 2009, etc.
            )
        )

        return long_df

    return (unpivot_to_long_metric,)


@app.cell
def _(results, unpivot_to_long_metric):
    # Unpivot the three types separately
    # This avoids the type errors because each df has consistent types
    df_counts = unpivot_to_long_metric(results, "_count", "county_pixel_count")
    df_unique = unpivot_to_long_metric(results, "_unique", "crop_code")
    df_fracs = unpivot_to_long_metric(results, "_frac", "fraction")
    return df_counts, df_fracs, df_unique


@app.cell
def _(INTERMEDIATE, df_counts, df_fracs, df_unique, pl):
    # Join them back together on GEOID and Year
    # Using a join ensures that 2008 count matches 2008 unique
    long_df = df_counts.join(df_unique, on=["GEOID10", "year"]).join(
        df_fracs, on=["GEOID10", "year"]
    )

    # Explode the lists and calculate pixel counts
    # Note: Polars handles multiple list explosions in parallel if they have the same length
    long_df = long_df.explode(["crop_code", "fraction"]).with_columns(
        (pl.col("fraction") * pl.col("county_pixel_count")).alias("crop_pixel_count")
    )

    # Export as parquet
    binary_file_path = (
        INTERMEDIATE / "county_crop_pixel_count_2008_2024_exactextract.parquet"
    )
    long_df.write_parquet(binary_file_path)
    return


if __name__ == "__main__":
    app.run()
