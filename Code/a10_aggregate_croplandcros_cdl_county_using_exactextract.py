import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import numpy as np
    import geopandas as gpd
    import rasterio as rio
    import rasterio.mask as riomask
    import rasterio.features as riofeatures
    import rasterio.plot as rioplot
    import rioxarray as rxr
    import matplotlib.pyplot as plt
    import polars as pl
    import pandas as pd
    from exactextract import exact_extract
    import marimo as mo

    return Path, exact_extract, gpd, rio


@app.cell
def _(Path):
    cdl_path = Path(__file__).parent.parent / 'Data' / 'croplandcros_cdl'
    return (cdl_path,)


@app.cell
def _(cdl_path, rio):
    for _year in range(2008, 2024):
        _cropland_data_layer_path = cdl_path / f'{_year}_30m_cdls/{_year}_30m_cdls.tif'  # CDL path
        _cropland_data_layer = rio.open(_cropland_data_layer_path)
        _cdl_crs_epsg = _cropland_data_layer.crs.to_epsg()
        print(f'{_year} has EPSG {_cdl_crs_epsg}')  # Open connection to raster file
        _cropland_data_layer.close()  # CRS for crop data  # Close connection
    return


@app.cell
def _(Path, cdl_path, gpd, rio):
    # Reproject county shapefile to match CDL CRS
    raster_path = Path(__file__).parent.parent / 'Data' / 'rasters'
    raster_path.mkdir(parents=True, exist_ok=True)

    # Read county border vector file
    county_shapefile_path = Path(__file__).parent.parent / 'Data' / 'county_shapefile' / 'tl_2023_us_county' / 'tl_2023_us_county.shp'
    county_borders = gpd.read_file(county_shapefile_path)

    # Take CRS from CDL
    reference_cdl_path = cdl_path / '2023_30m_cdls/2023_30m_cdls.tif'
    reference_cdl = rio.open(reference_cdl_path)
    raster_crs = reference_cdl.crs

    # Reproject county shapefile to CDL CRS
    county_borders = county_borders.to_crs(raster_crs)

    # Remove Alaska, Hawaii, territories
    county_borders['statefp'] = county_borders['STATEFP'].astype(int)
    county_borders_continental = county_borders[(county_borders['statefp'] < 60) & (county_borders['statefp'] != 2) & (county_borders['statefp'] != 15)].copy()
    county_borders_continental = county_borders_continental.drop('statefp', axis='columns')
    county_borders_continental['FIPS'] = county_borders_continental['STATEFP'] + county_borders_continental['COUNTYFP']

    # Save reprojected county shapefile
    reprojected_county_shp_path = Path(__file__).parent.parent / 'Data' / 'county_shapefile' / 'reprojected_continental_us_county.shp'
    county_borders_continental.to_file(reprojected_county_shp_path)
    return (reprojected_county_shp_path,)


@app.cell
def _(Path, exact_extract, gpd, reprojected_county_shp_path, test_cdl_path):
    # Perform aggregation using exactextract
    county_shp = gpd.read_file(reprojected_county_shp_path)
    for year in range(2008, 2024):
        cdl_path = cdl_path / f'{year}_30m_cdls/{year}_30m_cdls.tif'
        results = exact_extract(
            rast=test_cdl_path,
            vec=county_shp,
            ops=['count', 'unique', 'frac'],
            include_cols=['GEOID'], # Include county FIPS code in output
            output='pandas'
        )

        # unique and frac are returned as list-like columns, so explode them into rows.
        df_long = results.explode(['unique', 'frac'])
    
        # frac is the percentage of county covered by that specific crop code
        # multiply by count to get crop pixel count
        df_long['pixel_count'] = df_long['frac'] * df_long['count']
    
        # Rename 'unique' to 'crop_code' and keep only relevant columns
        df_long = df_long.rename(columns={'unique': 'crop_code'})
        df_long = df_long[['GEOID', 'crop_code', 'pixel_count']]
    
        # Optional: Remove entries with zero pixels (e.g., crops not present in a county)
        df_long = df_long[df_long['pixel_count'] > 0].reset_index(drop=True)

        # Save binary
        binary_path = Path(__file__).parent.parent / 'binaries'
        df_long.to_parquet(binary_path / f'county_crop_pixel_count_{year}_exactextract.parquet')
    return (cdl_path,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
