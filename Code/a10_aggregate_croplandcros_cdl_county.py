import marimo

__generated_with = "0.19.7"
app = marimo.App()


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
    import pandas as pd
    return gpd, rio


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check equality of CRS across CDL years
    """)
    return


@app.cell
def _(rio):
    for year in range(2012, 2024):
        _cropland_data_layer_path = f'../Data/croplandcros_cdl/{year}_30m_cdls/{year}_30m_cdls.tif'  # CDL path
        _cropland_data_layer = rio.open(_cropland_data_layer_path)
        cdl_crs_epsg = _cropland_data_layer.crs.to_epsg()
        print(f'{year} has EPSG {cdl_crs_epsg}')  # Open connection to raster file
        _cropland_data_layer.close()  # CRS for crop data  # Close connection
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Rasterize county borders shapefile
    """)
    return


@app.cell
def _(gpd, rio):
    # Read county border vector file
    county_shapefile_path = '../Data/county_shapefile/tl_2023_us_county/tl_2023_us_county.shp'
    county_borders = gpd.read_file(county_shapefile_path)
    _cropland_data_layer_path = '../Data/croplandcros_cdl/2023_30m_cdls/2023_30m_cdls.tif'
    # Reproject county shapefile to CDL CRS
    # Take CRS from CDL
    _cropland_data_layer = rio.open(_cropland_data_layer_path)
    county_borders = county_borders.to_crs(_cropland_data_layer.crs)
    _cropland_data_layer.close()
    county_borders['numeric_statefp'] = county_borders['STATEFP'].astype(int)
    county_borders_continental = county_borders[(county_borders['numeric_statefp'] < 60) & (county_borders['numeric_statefp'] != 2) & (county_borders['numeric_statefp'] != 15)].copy()
    # Remove Alaska, Hawaii, territories
    county_borders_continental['fips_code'] = county_borders_continental['STATEFP'] + county_borders_continental['COUNTYFP']
    county_borders_continental['fips_code_int'] = county_borders_continental['fips_code'].astype('int32')
    _cropland_data_layer = rio.open(_cropland_data_layer_path)
    statefp_list = county_borders_continental['numeric_statefp'].unique()
    # FIPS code is the attribute we want to burn into our raster
    # Use 2023 CDL as our template
    # We want to rasterize individual states at a time due to memory limitations
    # for statefp in statefp_list:
    #     # Select only the counties in one state
    #     state_counties = county_borders_continental[county_borders_continental['numeric_statefp'] == statefp].copy()
    #     # Crop raster to only the state we want
    #     out_image, out_transform = riomask.mask(cropland_data_layer, state_counties.geometry, crop=True)
    #     out_meta = cropland_data_layer.meta
    #     # Use masked raster's dimensions for rasterizing
    #     out_meta.update({
    #         "height": out_image.shape[1],
    #         "width": out_image.shape[2],
    #         "transform": out_transform})
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
    #         f"../rasters/rasterized_state_{statefp}.tif",
    #         "w",
    #         driver = out_meta['driver'],
    #         crs = out_meta['crs'],
    #         dtype = 'uint16',
    #         count = 1,
    #         width = out_meta['width'],
    #         height = out_meta['height'],
    #         compression = 'lzw') as dst:
    #         dst.write(state_rasterized, indexes = 1)
    # Close rasterio connection
    _cropland_data_layer.close()
    return


@app.cell
def _():
    # # Iterate over years
    # for year in range(2008, 2023):
    #     us_crop_agg = pd.DataFrame()
    #     print(f'Year {year}')

    #     # Load CDL for the year
    #     cropland_data_layer_path = f"../Data/croplandcros_cdl/{year}_30m_cdls/{year}_30m_cdls.tif"

    #     # Open connection to raster file
    #     cropland_data_layer = rio.open(cropland_data_layer_path)

    #     # CRS for crop data
    #     cdl_crs_epsg = cropland_data_layer.crs.to_epsg()

    #     # Iterate over states
    #     for statefp in statefp_list:
    #         print(f'State {statefp}')

    #         # Open state raster
    #         state_raster_path = f"../rasters/rasterized_state_{statefp}.tif"

    #         state_raster = rio.open(state_raster_path)

    #         # Geometry of state from vector file
    #         state_geom = county_borders_continental[county_borders_continental['numeric_statefp'] == statefp].copy()

    #         # Crop cropland raster to only the state we want
    #         out_image, out_transform = riomask.mask(cropland_data_layer, state_geom.geometry, crop=True)

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

    #         # Put rasters into dataframe
    #         df = pd.DataFrame()
    #         df['fips'] = county_ras.ravel()
    #         df['crop'] = crop_ras.ravel()
    #         # df['lon'] = lons.ravel()
    #         # df['lat'] = lats.ravel()

    #         # Remove pixels that are not in any county
    #         df = df[df['fips'] != 0].copy()

    #         # Aggregate by county by crop
    #         county_crop_agg = df.groupby(by = ['fips', 'crop']).size().reset_index(name='pixel_count')

    #         # Append to overall dataframe
    #         us_crop_agg = pd.concat([us_crop_agg, county_crop_agg])

    #         # Close rasterio connection to state raster
    #         state_raster.close()

    #     # Save US dataframe to binary
    #     us_crop_agg.to_parquet(f"../binaries/county_crop_pixel_count_{year}.parquet")

    #     # Close rasterio connection to CDL raster for the year
    #     cropland_data_layer.close()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
