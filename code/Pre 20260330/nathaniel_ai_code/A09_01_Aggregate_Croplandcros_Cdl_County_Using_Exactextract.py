---
title: A09 01 Aggregate Croplandcros Cdl County Using Exactextract
marimo-version: 0.20.2
width: full
---

```python {.marimo}
import marimo as mo
from pathlib import Path
import pyprojroot
import dotenv, os
import polars as pl
import polars.selectors as cs
import pandas as pd
import geopandas as gpd
import rasterio as rio
import exactextract
from exactextract import exact_extract, operation
import time
```

```python {.marimo}
root_path = pyprojroot.find_root(criterion='pyproject.toml')
cdl_path = root_path / 'data' / 'croplandcros_cdl'
```

```python {.marimo}
cdl_files = {}
for _year in range(2008, 2025):
    _cropland_data_layer_path = cdl_path / f'{_year}_30m_cdls/{_year}_30m_cdls.tif'  # CDL path
    cdl_files[_year] = _cropland_data_layer_path
    _cropland_data_layer = rio.open(_cropland_data_layer_path)
    _cdl_crs_epsg = _cropland_data_layer.crs.to_epsg()
    print(f'{_year} has EPSG {_cdl_crs_epsg}')  # Open connection to raster file
    _cropland_data_layer.close()  # CRS for crop data  # Close connection
```

```python {.marimo}
# Reproject county shapefile to match CDL CRS
# Read county border vector file
county_shapefile_path = root_path / 'data' / 'county_shapefile' / 'tl_2023_us_county' / 'tl_2023_us_county.shp'
county_borders = gpd.read_file(county_shapefile_path)

# 3. Open rasters and ensure CRS match
# CDL is almost always EPSG:5070 (USA Contiguous Albers Equal Area)
# TIGER files are usually EPSG:4269. We MUST reproject the counties.
with rio.open(list(cdl_files.values())[0]) as src:
    cdl_crs = src.crs
    print(f'Reprojecting county shapefile to {cdl_crs}...')
    county_borders = county_borders.to_crs(cdl_crs)

# We only want CONUS
statefp_drop_list = [
    '02', # AK
    '11', # DC
    '15', # HI
    '60', # AS
    '66', # GU
    '69', # MP
    '72', # PR
    '78', # VI
    '74' #UM
]

county_borders_conus = county_borders[~county_borders['STATEFP'].isin(statefp_drop_list)]
county_borders_conus = county_borders_conus[['GEOID', 'geometry']].reset_index(drop=True)
```

```python {.marimo}
# We open all years simultaneously and pass them as a list
# exactextract will compute weights once and apply to all bands (years)
print('Starting extraction (this may take a while for CONUS)...')
raster_handles = [path for path in cdl_files.values()]

# results = exact_extract(
#     rast=raster_handles, 
#     vec=county_borders_conus, 
#     ops=['count', 'unique', 'frac'], 
#     include_cols=['GEOID'],
#     output='pandas'
# )

# results.to_parquet(project_path / 'binaries' / 'temp_exactextract.parquet')
results = pd.read_parquet(root_path / 'binaries' / 'temp_exactextract.parquet')
results = pl.from_pandas(results)

print('Finished extracting')
```

```python {.marimo}
# Define a function that unpivots each column type separately
def unpivot_to_long_metric(df, suffix, value_name):
    long_df = df.select(
        ["GEOID", cs.contains(suffix)]
    ).unpivot(
        index="GEOID", variable_name="year", value_name=value_name
    ).with_columns(
        pl.col("year").str.extract(r"^(\d{4})", 1) # Extract 2008, 2009, etc.
    )

    return long_df
```

```python {.marimo}
# Unpivot the three types separately
# This avoids the type errors because each df has consistent types
df_counts = unpivot_to_long_metric(results, "_count", "county_pixel_count")
df_unique = unpivot_to_long_metric(results, "_unique", "crop_code")
df_fracs  = unpivot_to_long_metric(results, "_frac", "fraction")
```

```python {.marimo}
# Join them back together on GEOID and Year
# Using a join ensures that 2008 count matches 2008 unique
long_df = df_counts.join(
    df_unique, on=["GEOID", "year"]
).join(
    df_fracs, on=["GEOID", "year"]
)

# Explode the lists and calculate pixel counts
# Note: Polars handles multiple list explosions in parallel if they have the same length
long_df = long_df.explode(
    ["crop_code", "fraction"]
).with_columns(
        (pl.col("fraction") * pl.col("county_pixel_count")).alias("crop_pixel_count")
)

# Export as parquet
binary_file_path = root_path / 'binaries' / 'county_crop_pixel_count_2008_2024_exactextract.parquet'
long_df.write_parquet(binary_file_path)
```

```python {.marimo}

```