import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
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

    return Path, pd


@app.cell
def _(Path):
    # Read binaries we want to compare
    year = 2010
    binary_path = Path(__file__).parent.parent / 'binaries'
    return binary_path, year


@app.cell
def _(binary_path, pd, year):
    exact_df = pd.read_parquet(binary_path / f'county_crop_pixel_count_{year}_exactextract.parquet')
    exact_df = exact_df.rename(
        columns={
            'GEOID':'fips',
            'pixel_count':'exact_pixel_count'
        }
    )
    return (exact_df,)


@app.cell
def _(binary_path, pd, year):
    original_df = pd.read_parquet(binary_path / f'county_crop_pixel_count_{year}.parquet')
    original_df['fips'] = original_df['fips'].astype('str').str.zfill(5)
    original_df = original_df.rename(columns = {'crop':'crop_code'})
    return (original_df,)


@app.cell
def _(exact_df, original_df):
    combined_df = original_df.merge(exact_df)
    combined_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
