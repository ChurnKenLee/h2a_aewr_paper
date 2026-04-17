import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import pyprojroot
    import dotenv, os
    import rasterio
    from rasterio.shutil import copy
    import numpy as np
    import subprocess

    return copy, os, pyprojroot, rasterio


@app.cell
def _(pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    cdl_path = root_path / 'data' / 'croplandcros_cdl'
    gnatsgo_path = root_path / 'data' / 'gnatsgo' / 'gNATSGO_gpkg_01_30_2026'
    return cdl_path, gnatsgo_path


@app.cell
def _(copy, os, rasterio):
    def compress_fast_cdl(file_path):
        temp_path = file_path.stem + ".tmp.tif"
        print(f"Starting multi-threaded C++ compression for: {file_path}...")
    
        try:
            with rasterio.open(file_path) as src:
                # rasterio.shutil.copy calls GDAL's C++ CreateCopy method directly.
                # It completely bypasses Python loops and uses all CPU cores.
                copy(
                    src, 
                    temp_path, 
                    # --- GDAL Creation Options ---
                    compress='deflate', 
                    predictor=2, 
                    tiled=True, 
                    blockxsize=512, 
                    blockysize=512, 
                    num_threads='all_cpus'  # This triggers the multi-threading!
                )
            
            # Safely overwrite the original file
            os.replace(temp_path, file_path)
            print(f"Success! {file_path} has been compressed and replaced.")
        
        except Exception as e:
            print(f"An error occurred: {e}")
            # Clean up the broken temp file if something crashes
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return (compress_fast_cdl,)


@app.cell
def _(cdl_path, compress_fast_cdl):
    # Compress CDLs
    for _year in range(2008, 2025):
        _cropland_data_layer_path = cdl_path / f'{_year}_30m_cdls/{_year}_30m_cdls.tif'  # CDL path
        compress_fast_cdl(_cropland_data_layer_path)
    return


@app.cell
def _(copy, os, rasterio):
    def compress_fast_gnatsgo(file_path):
        temp_path = file_path.stem + ".tmp.tif"
        print(f"Starting gNATSGO multi-threaded compression for: {file_path}...")
    
        try:
            with rasterio.open(file_path) as src:
                # Check the original data type just to be safe
                print(f"Original Data Type: {src.dtypes[0]}")
            
                # We copy the file natively (preserving the 32-bit MUKEYs),
                # but we apply aggressive tiling and integer-optimized compression.
                copy(
                    src, 
                    temp_path, 
                    compress='deflate', 
                    predictor=2,        # Still works beautifully for 32-bit MUKEYs!
                    tiled=True, 
                    blockxsize=512, 
                    blockysize=512, 
                    num_threads='all_cpus' 
                )
            
            # Safely overwrite the original file
            os.replace(temp_path, file_path)
            print(f"Success! {file_path} has been compressed, tiled, and replaced.")
        
        except Exception as e:
            print(f"An error occurred: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return (compress_fast_gnatsgo,)


@app.cell
def _(compress_fast_gnatsgo, gnatsgo_path):
    # Compress gNTATSGO
    gnatsgo_raster = gnatsgo_path / "MURASTER_30m_CONUS_2026.tif"
    compress_fast_gnatsgo(gnatsgo_raster)
    return


if __name__ == "__main__":
    app.run()
