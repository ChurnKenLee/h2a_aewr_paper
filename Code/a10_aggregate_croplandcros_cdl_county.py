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
    import matplotlib.pyplot as plt
    import polars as pl
    import marimo as mo

    return Path, gpd, mo, rio


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check equality of CRS across CDL years
    """)
    return


@app.cell
def _(Path):
    cdl_path = Path(__file__).parent.parent / 'Data' / 'croplandcros_cdl'
    return (cdl_path,)


@app.cell
def _(cdl_path, rio):
    for _year in range(2012, 2024):
        _cropland_data_layer_path = cdl_path / f'{_year}_30m_cdls/{_year}_30m_cdls.tif'  # CDL path
        _cropland_data_layer = rio.open(_cropland_data_layer_path)
        _cdl_crs_epsg = _cropland_data_layer.crs.to_epsg()
        print(f'{_year} has EPSG {_cdl_crs_epsg}')  # Open connection to raster file
        _cropland_data_layer.close()  # CRS for crop data  # Close connection
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Rasterize county borders shapefile
    """)
    return


@app.cell
def _(Path):
    raster_path = Path(__file__).parent.parent / 'Data' / 'rasters'
    raster_path.mkdir(parents=True, exist_ok=True)
    return


@app.cell
def _(Path, cdl_path, gpd, rio):
    # Read county border vector file
    county_shapefile_path = Path(__file__).parent.parent / 'Data' / 'county_shapefile' / 'tl_2023_us_county' / 'tl_2023_us_county.shp'
    county_borders = gpd.read_file(county_shapefile_path)

    # Reproject county shapefile to CDL CRS
    # Take CRS from CDL
    reference_cdl_path = cdl_path / '2023_30m_cdls/2023_30m_cdls.tif'
    reference_cdl = rio.open(reference_cdl_path)
    county_borders = county_borders.to_crs(reference_cdl.crs)
    county_borders['numeric_statefp'] = county_borders['STATEFP'].astype(int)
    county_borders_continental = county_borders[(county_borders['numeric_statefp'] < 60) & (county_borders['numeric_statefp'] != 2) & (county_borders['numeric_statefp'] != 15)].copy()

    # Remove Alaska, Hawaii, territories
    county_borders_continental['fips_code'] = county_borders_continental['STATEFP'] + county_borders_continental['COUNTYFP']
    county_borders_continental['fips_code_int'] = county_borders_continental['fips_code'].astype('int32')
    statefp_list = county_borders_continental['numeric_statefp'].unique()
    return


@app.cell
def _():
    # # FIPS code is the attribute we want to burn into our raster
    # # Use 2023 CDL as our template
    # # We want to rasterize individual states at a time due to memory limitations
    # for statefp in statefp_list:
    #     # Select only the counties in one state
    #     state_counties = county_borders_continental[county_borders_continental['numeric_statefp'] == statefp].copy()
    #     # Crop raster to only the state we want
    #     out_image, out_transform = riomask.mask(reference_cdl, state_counties.geometry, crop=True)
    #     out_meta = reference_cdl.meta
    #     # Use masked raster's dimensions for rasterizing
    #     out_meta.update({
    #         'height': out_image.shape[1],
    #         'width': out_image.shape[2],
    #         'transform': out_transform})
    #     # Create tuples of geometry, value pairs, where value is the attribute value you want to burn
    #     geom_value = ((geom,value) for geom, value in zip(state_counties.geometry, state_counties['fips_code_int']))
    #     # Rasterize vector using the shape and transform of the raster
    #     state_rasterized = riofeatures.rasterize(
    #         geom_value,
    #         out_shape = (out_image.shape[1], out_image.shape[2]),
    #         transform = out_transform,
    #         all_touched = False,
    #         fill = 0,   # background value
    #         dtype = 'uint16')
    #     # Save state raster
    #     with rio.open(
    #         raster_path / f'rasterized_state_{statefp}.tif',
    #         'w',
    #         driver = out_meta['driver'],
    #         crs = out_meta['crs'],
    #         dtype = 'uint16',
    #         count = 1,
    #         width = out_meta['width'],
    #         height = out_meta['height'],
    #         compression = 'lzw') as dst:
    #         dst.write(state_rasterized, indexes = 1
    #                  )
    # # Close rasterio connection
    # reference_cdl.close()
    return


@app.cell
def _():
    # # Iterate over years
    # for year in range(2008, 2009):
    #     us_crop_agg = pl.DataFrame()
    #     print(f'Year {year}')

    #     # Load CDL for the year
    #     year_cdl_path = cdl_path / f'{year}_30m_cdls/{year}_30m_cdls.tif'

    #     # Open connection to raster file
    #     year_cdl = rio.open(year_cdl_path)

    #     # CRS for crop data
    #     cdl_crs_epsg = year_cdl.crs.to_epsg()

    #     # Iterate over states
    #     for statefp in statefp_list:
    #         print(f'State {statefp}')

    #         # Open state raster
    #         state_raster_path = raster_path / f'rasterized_state_{statefp}.tif'
    #         state_raster = rio.open(state_raster_path)

    #         # Geometry of state from vector file
    #         state_geom = county_borders_continental[county_borders_continental['numeric_statefp'] == statefp].copy()

    #         # Crop cropland raster to only the state we want
    #         out_image, out_transform = riomask.mask(year_cdl, state_geom.geometry, crop=True)

    #         # Overlay state-county raster and crop raster
    #         county_ras = state_raster.read(1)
    #         crop_ras = out_image[0]

    #         # # Evaluate coordinate using rasterio's transform over grid of array coordinates
    #         # # Dimension of grid
    #         # height = county_ras.shape[0]
    #         # width = county_ras.shape[1]
    #         # cols, rows = np.meshgrid(np.arange(width), np.arange(height))

    #         # xs, ys = rio.transform.xy(state_raster.transform, rows, cols)

    #         # lons= np.array(xs)
    #         # lats = np.array(ys)

    #         # Aggregate county X crop
    #         fips_flat = county_ras.ravel()
    #         crop_flat = crop_ras.ravel()
        
    #         # Filter out zeros (non-county areas)
    #         mask = fips_flat > 0
    #         fips_flat = fips_flat[mask]
    #         crop_flat = crop_flat[mask]
        
    #         # Create a combined key of County X Crop
    #         # Assuming FIPS is up to 99999 and Crop is up to 999
    #         combined_key = fips_flat.astype(np.uint64) * 1000 + crop_flat.astype(np.uint64)
        
    #         # Use numpy to count unique County X Crop occurences
    #         unique_keys, counts = np.unique(combined_key, return_counts=True)
        
    #         # Turn back into FIPS/Crop and save
    #         county_crop_agg = pl.DataFrame({
    #             'fips': unique_keys // 1000,
    #             'crop': unique_keys % 1000,
    #             'pixel_count': counts
    #         })

    #         # Append to overall dataframe
    #         us_crop_agg = pl.concat([us_crop_agg, county_crop_agg], how='vertical')

    #         # Close rasterio connection to state raster
    #         state_raster.close()

    #     # Save US dataframe to binary
    #     binary_path = Path(__file__).parent.parent / 'binaries'
    #     us_crop_agg.write_parquet(binary_path / f'county_crop_pixel_count_{year}.parquet')

    #     # Close rasterio connection to CDL raster for the year
    #     year_cdl.close()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
