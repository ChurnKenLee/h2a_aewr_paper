---
title: A09 04 Evaluate Nass Cdl Match And Combine
marimo-version: 0.20.2
width: full
---

```python {.marimo}
import marimo as mo
from pathlib import Path
import pyprojroot
import dotenv, os
import ibis
import ibis.selectors as s
from ibis import _
import polars as pl
import pdfplumber
import dspy
from pydantic import BaseModel, Field
from typing import List, Literal
import json
import tqdm
from itertools import islice
import time
import tempfile
import duckdb
```

```python {.marimo}
root_path = pyprojroot.find_root(criterion='pyproject.toml')
binary_path = root_path / 'binaries'
json_path = root_path / 'code' / 'json'
cdl_path = root_path / 'data' / 'croplandcros_cdl'
# Configure Gemini
gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
gemini = dspy.LM('gemini/gemini-3-flash-preview', api_key=gemini_api_key)
dspy.configure(lm=gemini)
```

```python {.marimo}
# Load CroplandCROS CDL code table
cdl_names = pl.read_excel(Path(cdl_path / 'CDL_codes_names_colors.xlsx'), read_options={"header_row": 3})
cdl_names = (cdl_names
    .rename(str.lower)
    .select([
        pl.col('codes'),
        pl.col('class_names')
    ])
    .rename(mapping={
     'codes':'cdl_code','class_names':'cdl_name'
    })
)
cdl_code_lookup = dict(cdl_names.select("cdl_code", "cdl_name").iter_rows())

df_results = pl.read_ndjson(json_path / 'nass_to_cdl_mappings.jsonl')
df_results = df_results.with_columns(
    cdl_code = pl.col("mapping").struct.field('codes')
)
df_results = df_results.with_columns(
    cdl_name = pl.col("cdl_code").list.eval(
        pl.element().replace_strict(cdl_code_lookup)
    )
).explode([
    "cdl_code",
    'cdl_name'
])
```

```python {.marimo}
cdl_acres = pl.read_parquet(binary_path / 'county_crop_pixel_count_2008_2024_exactextract.parquet')
# 1 acre ~ 4047m2
cdl_acres = cdl_acres.with_columns(
    crop_acre = pl.col('crop_pixel_count')/4047
)
cdl_acres = cdl_acres.select([
    'GEOID',
    'year',
    'crop_code',
    'crop_acre'
]).rename({
    'GEOID':'fips',
    'crop_code':'cdl_code',
    'crop_acre':'cdl_acre'
})
cdl_acres
```

```python {.marimo}
# NASS crop definitions
with open(json_path / 'nass_quickstats_survey_obs_selection.json', 'r') as _f:
    nass_obs_selection_json = json.load(_f)
nass_obs_selection = pl.from_dicts(list(nass_obs_selection_json.values())) # Unpack JSON in pl df
nass_obs_selection = (nass_obs_selection
    .drop('reasoning') # shares same name with column inside definitions
    .unnest('output') # output only has 1 item inside: definitions of crops selected
    .explode('definitions') # definitions has multiple entries, each definition is a selected crop
    .unnest('definitions')
    .rename(mapping={
        'task_key':'nass_id',
        'group':'group_desc',
        'commodity':'commodity_desc',
        'reasoning':'nass_reasoning',
        'semantic_label':'nass_semantic_label'
    })) # each crop has multiple elements within each definition (class, prodn, util)

nass_obs_selection
```

```python {.marimo}
# Export cleaned NASS CDL crosswalk
nass_cdl_xwalk = df_results.select(['nass_id', 'cdl_code', 'cdl_name'])
nass_cdl_xwalk = nass_obs_selection.join(
    nass_cdl_xwalk,
    how='left',
    on=['nass_id']
).drop('nass_id')
nass_cdl_xwalk.write_parquet(binary_path/'nass_cdl_crosswalk.parquet')
```

```python {.marimo}

```