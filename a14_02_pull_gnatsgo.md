---
title: A14 02 Pull Gnatsgo
marimo-version: 0.22.3
width: full
---

```python {.marimo}
import marimo as mo
from pathlib import Path
import pyprojroot
import polars as pl
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import exactextract
import sqlite3
```

```python {.marimo}
root_path = pyprojroot.find_root(criterion='pyproject.toml')
binary_path = root_path / 'binaries'
gnatsgo_path = root_path / 'data' / 'gnatsgo' / 'gNATSGO_gpkg_01_30_2026'
census_shp_path = root_path / 'data' / 'county_shapefile' / 'tl_2010_us_county10' / 'tl_2010_us_county10.shp'
```

ExactExtract cannot handle uint32, we will have to convert raster to float64 first

```python {.marimo}
input_raster = gnatsgo_path / "MURASTER_30m_CONUS_2026.tif"
converted_raster = gnatsgo_path / "MURASTER_30m_CONUS_2026_int32.tif"

# We only want to run this conversion once. Check if the converted file exists.
if not converted_raster.exists():
    print(f"Converting {input_raster} from uint32 to int32. This may take a few minutes...")

    with rasterio.open(input_raster) as _src:
        _profile = _src.profile

        # Safely convert NoData value to integer if it exists
        _nodata_val = int(_src.nodata) if _src.nodata is not None else None

        # Update the raster _profile to int32
        _profile.update(
            dtype=rasterio.int32,
            nodata=_nodata_val
        )

        # Read and write block-by-block to avoid loading a ~24GB CONUS raster into RAM
        with rasterio.open(converted_raster, 'w', **_profile) as _dst:
            for _ji, _window in _src.block_windows(1):
                # Read chunk
                _data = _src.read(1, window=_window)
                # Cast to int32 and write chunk
                _dst.write(_data.astype(np.int32), 1, window=_window)

    print(f"Raster successfully converted and saved to {converted_raster}")
else:
    print(f"Found existing converted raster: {converted_raster}")
```

# Table A: Acreage of map units within each county

```python {.marimo}
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
pixel_width, pixel_height = src.res # this is in meters
sq_meters_per_pixel = abs(pixel_width * pixel_height)
sq_meters_per_acre = 4046.872
acres_per_pixel = sq_meters_per_pixel / sq_meters_per_acre

nodata_val = src.nodata
```

If extract already performed, skip extraction (takes about 15 minutes)

```python {.marimo}
table_a_parquet = binary_path / 'county_mapunit_pixel_count_exactextract.parquet'

if not table_a_parquet.exists():
    print("Extracting raster pixels per county using exactextract...")
    # exact_extract evaluates operations requested in a list. 
    # "values" = raw pixel array, "coverage" = exact intersection fraction array.
    extracted_features = exactextract.exact_extract(
        src,
        counties,
        ["values", "coverage"],
        include_cols=[county_id_col]
    )

    table_a_rows =[]

    for feature in extracted_features:
        props = feature["properties"]
        geoid = props[county_id_col]

        # Grab the dynamic keys for values and coverage arrays
        val_key = next((k for k in props.keys() if "values" in k), "values")
        cov_key = next((k for k in props.keys() if "coverage" in k), "coverage")

        # Create a Polars DataFrame for this specific county
        df_county = pl.DataFrame({
            "mukey": props[val_key],
            "fraction": props[cov_key]
        })

        # Filter out NoData gaps
        if nodata_val is not None:
            df_county = df_county.filter(
                pl.col("mukey").is_not_null() & (pl.col("mukey") != nodata_val)
            )

        # Group by Map Unit Key (mukey), sum exact fractions to get "pixel count"
        df_county_agg = df_county.group_by("mukey").agg(
            pl.col("fraction").sum().alias("pixel_count")
        ).with_columns(
            pl.lit(geoid).alias("county_id")
        )

        table_a_rows.append(df_county_agg)

    # Combine into Table A (County | Map Unit | Acres)
    table_a = pl.concat(table_a_rows)

    # Convert pixel counts to acres and cast mukeys to String for relational join later
    table_a = table_a.with_columns([
        (pl.col("pixel_count") * acres_per_pixel).alias("mapunit_acres"),
        pl.col("mukey").cast(pl.String)
    ])

    table_a.write_parquet(table_a_parquet)
    print("Extraction and aggregation complete")
else:
    print(f"{table_a_parquet} already exists")
    table_a = pl.read_parquet(table_a_parquet)
```

# Table B: Components within each map unit

```python {.marimo}
gpkg_path = gnatsgo_path / 'gNATSGO_02_13_2026.gpkg'
conn = sqlite3.connect(gpkg_path)
tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
tables = pl.read_database(tables_query, connection=conn)
```

Grab the tables we want then join

```python {.marimo}
joined_query = """
SELECT
    mapunit.mukey,
    mapunit.muname,
    component.compname,
    component.comppct_r,
    component.majcompflag,
    component.taxorder,
    component.taxsuborder,
    component.taxgrtgroup,
    component.slope_r,
    component.drainagecl,
    component.nirrcapcl,
    corestrictions.resdept_r
FROM
    mapunit
    INNER JOIN component ON mapunit.mukey = component.mukey
    INNER JOIN corestrictions ON component.cokey = corestrictions.cokey
"""
joined_table = pl.read_database(joined_query, infer_schema_length = None, connection=conn)
```

```python {.marimo}
# Bin continuous variables
table_b = joined_table.with_columns(
    # USDA uses 8% gradient as cutoff for machine contour farming
    pl.when(
        pl.col('slope_r') > 8
    ).then(
        pl.lit('steep')
    ).when(
        pl.col('slope_r') <= 8
    ).then(
        pl.lit('flat')
    ).otherwise(
        pl.lit('no_grade')
    ).alias('slope'),
    # USDA classification of root depths
    pl.when(
        pl.col('resdept_r') < 20
    ).then(
        pl.lit('too_shallow')
    ).when(
        (pl.col('resdept_r') >= 20) & (pl.col('resdept_r') < 50 )
    ).then(
        pl.lit('shallow')
    ).when(
        (pl.col('resdept_r') >= 50) & (pl.col('resdept_r') < 100 )
    ).then(
        pl.lit('moderate')
    ).when(
        (pl.col('resdept_r') >= 100) & (pl.col('resdept_r') < 150 )
    ).then(
        pl.lit('deep')
    ).when(
        pl.col('resdept_r') >= 150
    ).then(
        pl.lit('very_deep')
    ).otherwise(
        pl.lit('no_depth')
    ).alias('root_depth')
)

table_b_cat = table_b.select([
    'mukey', 'comppct_r',
    'taxorder', 'taxsuborder', 'taxgrtgroup',
    'slope', 'drainagecl', 'root_depth',
    'nirrcapcl'
]).group_by([
    'mukey',
    'taxorder', 'taxsuborder', 'taxgrtgroup',
    'slope', 'drainagecl', 'root_depth',
    'nirrcapcl'
]).agg(
    pl.col('comppct_r').sum()
)

table_b_cont = table_b.select([
    'mukey', 'comppct_r',
    'taxorder', 'taxsuborder', 'taxgrtgroup',
    'slope_r', 'drainagecl', 'resdept_r',
    'nirrcapcl'
]).group_by([
    'mukey',
    'taxorder', 'taxsuborder', 'taxgrtgroup',
    'slope_r', 'drainagecl', 'resdept_r',
    'nirrcapcl'
]).agg(
    pl.col('comppct_r').sum()
)
```

# Join tables, export

```python {.marimo}
# Inner join connects every component inside every map unit located in our counties
final_df = table_a.with_columns(
    pl.col('mukey').cast(pl.Int64).alias('mukey')
).join(
    table_b_cont, on="mukey", how="inner"
)

# Component acreage: (Map Unit Area) * (Component Percentage / 100)
final_df = final_df.with_columns([
    (pl.col("mapunit_acres") * (pl.col("comppct_r") / 100.0)).alias("component_acres")
])

# Summary characteristics
classification_cols_cat =[
    'taxorder',
    'slope', 'drainagecl', 'root_depth',
    'nirrcapcl'
]

classification_cols_cont =[
    'taxorder',
    'slope_r', 'drainagecl', 'resdept_r',
    'nirrcapcl'
]

# Final acreage grouping by county and soil characteristics
county_level_acreage = final_df.group_by(
    ["county_id"] + classification_cols_cont
).agg(
    pl.col("component_acres").sum().alias("total_acres")
).sort(
    ["county_id", "total_acres"], descending=[False, True]
).rename({
    'county_id':'county_ansi'
})
```

```python {.marimo}
county_level_acreage.write_parquet(binary_path / 'county_h2a_prediction_gnatsgo.parquet')
```

```python {.marimo}

```