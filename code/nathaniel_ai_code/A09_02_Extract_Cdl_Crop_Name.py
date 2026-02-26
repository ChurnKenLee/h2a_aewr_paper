---
title: A09 02 Extract Cdl Crop Name
marimo-version: 0.20.2
width: full
---

```python {.marimo}
import marimo as mo
from pathlib import Path
import polars as pl
import numpy as np
from bs4 import BeautifulSoup
import re
```

Get crop categorization codes from metadata file

```python {.marimo}
project_path = Path(__file__).parent.parent
cdl_path = project_path / 'Data' / 'croplandcros_cdl'
```

```python {.marimo}
# Iterate over years and place crop codes in a DataFrame
rows_list = []
for _year in range(2008, 2025):
    year_cdl_path = cdl_path / f'{_year}_30m_cdls'
    metadata_path = list(year_cdl_path.glob('*.htm'))

    with open(metadata_path[0]) as fp:
        soup = BeautifulSoup(fp, features='html.parser')  # Extract table
        # The categorization codes for the crop types are in tables that are in preformatted blocks of text
        # Get the last block, which should be the categorization table
        cat_table = soup.find_all('pre')[-1].string

    for line in cat_table.splitlines():
        # Regex match number enclosed in double quotes
        crop_code_present = re.search(r'"(\d+)".*', line) is not None

        crop_code_dict = {}
        if crop_code_present:
            crop_code_dict = {}
            # Number in quotes is crop code
            crop_code = re.search(r'"(\d+)"', line)[0].strip('"')
            # Alphabetical chars after is crop name
            crop_name = re.search(r'[A-Z].*', line)[0]
            crop_code_dict['year'] = _year  # Extract lines with crop and categorization code
            crop_code_dict['crop_code'] = crop_code
            crop_code_dict['crop_name'] = crop_name
            rows_list.append(crop_code_dict.copy())
        else:
            continue

    if _year < 2022:
        grass_dict = {}
        grass_dict['year'] = _year  
        # For years prior to 2022, code 176 is not defined for some reason
        grass_dict['crop_code'] = '176'
        grass_dict['crop_name'] = 'Grassland/Pasture'
        rows_list.append(grass_dict)
```

Add crop names to county crop aggregates, Calculate acreage, clean FIPS code, then export

```python {.marimo}
# Convert parsed metadata file rows into df
crop_code_keys = pl.from_dicts(rows_list)
```

```python {.marimo}
# Read CDL county aggregates
cdl_pixel_file = project_path / 'binaries' / 'county_crop_pixel_count_2008_2024_exactextract.parquet'
cdl_pixel = pl.read_parquet(cdl_pixel_file)
# Change type to match crop code data
cdl_pixel = cdl_pixel.with_columns(
    pl.col('crop_code').cast(pl.String),
    pl.col('year').cast(pl.Int64)
)
```

```python {.marimo}
# Merge
county_crop = cdl_pixel.join(
    other=crop_code_keys,
    on=['crop_code', 'year'],
    how='inner',
    validate='m:1'
)

# Convert pixels to acres
# Each pixel is 30mx30m = 900m^2
# 1 acre = 4046.86m^2
# 900m^2 = 900/4046.86 acres = 0.22239464671 acres
county_crop = county_crop.with_columns(
    (pl.col('crop_pixel_count')*900/4046.86).alias('acres')
)

# Keep relevant columns and export
county_crop = county_crop.rename(
    {'GEOID':'fips'}
).select(
    ['fips', 'year', 'crop_code', 'crop_name', 'acres']
)
```

```python {.marimo}
# Save binary
county_crop.write_parquet(project_path / 'binaries' / 'croplandcros_county_crop_acres.parquet')
county_crop.write_parquet(project_path / 'files_for_phil' / 'croplandcros_county_crop_acres.parquet')
```

```python {.marimo}

```