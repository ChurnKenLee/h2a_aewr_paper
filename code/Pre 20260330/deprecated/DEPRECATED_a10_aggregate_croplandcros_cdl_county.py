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
    import rasterio.windows as riowindows
    import rasterio.mask as riomask
    import rasterio.features as riofeatures
    import rasterio.plot as rioplot
    import matplotlib.pyplot as plt
    import polars as pl
    import time
    from concurrent.futures import ProcessPoolExecutor

    return Path, gpd, mo, np, pl, rio, riomask, time


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
    cdl_files = {}
    for _year in range(2008, 2010):
        _cropland_data_layer_path = cdl_path / f'{_year}_30m_cdls/{_year}_30m_cdls.tif'  # CDL path
        cdl_files[_year] = _cropland_data_layer_path
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
    return (raster_path,)


@app.cell
def _(Path, cdl_path, gpd, np, raster_path, rio):
    reference_cdl_path = cdl_path / f'2023_30m_cdls/2023_30m_cdls.tif'
    with rio.open(reference_cdl_path) as src:
        meta = src.meta.copy()
        cdl_transform = src.transform
        cdl_crs = src.crs
        cdl_width = src.width
        cdl_height = src.height

    # Reproject county shapefile to match CDL CRS
    # Read county border vector file
    county_shapefile_path = Path(__file__).parent.parent / 'Data' / 'county_shapefile' / 'tl_2023_us_county' / 'tl_2023_us_county.shp'
    counties = gpd.read_file(county_shapefile_path)
    counties = counties.to_crs(cdl_crs)

    # Create a generator of (geometry, value) pairs
    # We convert FIPS string to int so it can be stored in a raster
    shapes = ((geom, int(value)) for geom, value in zip(counties.geometry, counties['GEOID']))

    # Initialize the output array (filled with zeros)
    # Use uint32 to accommodate FIPS codes (which go up to ~72000)
    out_arr = np.zeros((cdl_height, cdl_width), dtype=np.uint32)

    # Burn the geometries into the array
    rio.features.rasterize(
        shapes=shapes,
        out=out_arr,
        transform=cdl_transform,
        fill=0,
        all_touched=False # False ensures we only count pixels whose center is in the county
            )

    # Write to disk
    meta.update({
        'dtype': 'uint32',
        'count': 1,
        'nodata': 0,
        'compress': 'lzw' # LZW is great for categorized rasters
    })

    output_path = raster_path / 'county_reprojected_rasterized.tif'
    with rio.open(output_path, 'w', **meta) as dst:
        dst.write(out_arr, 1)
    
    print(f"Rasterized county file saved to {output_path}")
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
def _(np, pl, rio):
    def process_cdl_by_county(cdl_path, county_raster_path):
        results = []

        with rio.open(cdl_path) as src_cdl, rio.open(county_raster_path) as src_county:
            # Ensure both rasters have the same dimensions/transform
            # Iterate over the blocks (tiles) of the raster
            for stdout, window in src_cdl.block_windows():
                # Read only the specific chunk from both files
                cdl_window = src_cdl.read(1, window=window)
                county_window = src_county.read(1, window=window)

                # Flatten and mask (0 is usually 'No Data' in CDL and County rasters)
                # Masking here handles both non-county areas and non-crop areas
                mask = (county_window > 0) & (cdl_window > 0)
            
                if not np.any(mask):
                    continue

                fips_flat = county_window[mask].astype(np.uint32)
                crop_flat = cdl_window[mask].astype(np.uint16)

                # Use Polars for fast local aggregation of this chunk
                chunk_df = pl.DataFrame({
                    'fips': fips_flat,
                    'crop': crop_flat
                }).group_by(['fips', 'crop']).count()

                results.append(chunk_df)

        # Combine all chunks and aggregate one last time
        print("Combining chunk results...")
        final_agg = (
            pl.concat(results)
            .group_by(['fips', 'crop'])
            .sum()
            .rename({'count': 'pixel_count'})
        )
    
        return final_agg

    return


@app.cell
def _(
    Path,
    cdl_path,
    county_borders_continental,
    np,
    pl,
    raster_path,
    rio,
    riomask,
    statefp_list,
    time,
):
    # Iterate over years
    for year in range(2008, 2025):
        us_crop_agg = pl.DataFrame()
        print(f'Year {year}')
        start_time = time.perf_counter()

        # Load CDL for the year
        year_cdl_path = cdl_path / f'{year}_30m_cdls/{year}_30m_cdls.tif'

        # Open connection to raster file
        year_cdl = rio.open(year_cdl_path)

        # CRS for crop data
        cdl_crs_epsg = year_cdl.crs.to_epsg()

        print('Adding GEOID to CDL raster for year ', year)

        # Iterate over states
        for statefp in statefp_list:
            print(f'State {statefp}')

            # Open state raster
            state_raster_path = raster_path / f'rasterized_state_{statefp}.tif'
            state_raster = rio.open(state_raster_path)

            # Geometry of state from vector file
            state_geom = county_borders_continental[county_borders_continental['numeric_statefp'] == statefp].copy()

            # Crop cropland raster to only the state we want
            out_image, out_transform = riomask.mask(year_cdl, state_geom.geometry, crop=True)

            # Overlay state-county raster and crop raster
            county_ras = state_raster.read(1)
            crop_ras = out_image[0]

            # Aggregate county X crop
            fips_flat = county_ras.ravel()
            crop_flat = crop_ras.ravel()

            # Filter out zeros (non-county areas)
            mask = fips_flat > 0
            fips_flat = fips_flat[mask]
            crop_flat = crop_flat[mask]

            # Create a combined key of County X Crop
            # Assuming FIPS is up to 99999 and Crop is up to 999
            combined_key = fips_flat.astype(np.uint64) * 1000 + crop_flat.astype(np.uint64)

            # Use numpy to count unique County X Crop occurences
            unique_keys, counts = np.unique(combined_key, return_counts=True)

            # Turn back into FIPS/Crop and save
            state_crop_agg = pl.DataFrame({
                'fips': unique_keys // 1000,
                'crop': unique_keys % 1000,
                'pixel_count': counts
            })

            # Append to overall dataframe
            us_crop_agg = pl.concat([us_crop_agg, state_crop_agg], how='vertical')

            # Close rasterio connection to state raster
            state_raster.close()

        # Save US dataframe to binary
        binary_path = Path(__file__).parent.parent / 'binaries'
        us_crop_agg.write_parquet(binary_path / f'county_crop_pixel_count_{year}.parquet')

        # How long did this year take?
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f'Year {year} took {elapsed_time:.4f} seconds.')

        # Close rasterio connection to CDL raster for the year
        year_cdl.close()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
