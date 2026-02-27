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
import polars as pl
import pdfplumber
import dspy
from pydantic import BaseModel, Field
from typing import List, Literal
import json
import tqdm
from itertools import islice
import time
import copy
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

# Load CDL code NASS crosswalk, CDL acreage, NASS survey prices
<!---->
Load CDL codes from USDA, then combine with matching responses from LLM stored in jsonl

```python {.marimo}
# Load CroplandCROS CDL code table
# Ignore dtype parsing errors, that is due to row padding by USDA in their codebook in Excel
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
```

```python {.marimo}
# NASS CDL crosswalk
nass_cdl_xwalk = df_results.select(['nass_id', 'cdl_code', 'cdl_name'])
nass_cdl_xwalk = nass_obs_selection.join(
    nass_cdl_xwalk,
    how='left',
    on=['nass_id']
)
nass_cdl_xwalk = nass_cdl_xwalk.select([
    'group_desc', 'commodity_desc', 'class_desc', 'prodn_practice_desc', 'util_practice_desc',
    'cdl_code', 'cdl_name'
])
nass_cdl_xwalk.write_parquet(binary_path / 'nass_cdl_crosswalk.parquet')
```

CDL pixel counts previously aggregated using exactextract

```python {.marimo}
# CroplandCROS CDL acreage aggregated to the county-year-crop level
cdl_acres = pl.read_parquet(binary_path / 'county_crop_pixel_count_2008_2024_exactextract.parquet')
# 1 acre ~ 4047m2
cdl_acres = cdl_acres.with_columns(
    crop_acre = pl.col('crop_pixel_count')/4047
)
cdl_acres = (cdl_acres
    .select([
        'GEOID',
        'year',
        'crop_code',
        'crop_acre'
    ])
    .rename({
        'GEOID':'fips',
        'crop_code':'cdl_code',
        'crop_acre':'cdl_acre'
    })
    .with_columns(
        pl.col("fips").str.slice(0, 2).alias("state_fips")
    ))
```

Subset of NASS survey data from filtering NASS QuickStats survey data with the NASS crop selection list

```python {.marimo}
qs_survey_crops = pl.read_parquet(binary_path / 'qs_survey_selected_obs.parquet')
# Drop stats we don't care about
qs_survey_crops = qs_survey_crops.filter(
    pl.col('observation_type') != '',
    ~pl.col('value').is_null()
)
```

# Prepare NASS data

```python {.marimo}
qs_survey_crops.select('observation_type', 'unit_desc').unique()
```

This code generates the JSON mapping for extracting the base units from unit_desc
```python
import pandas as pd
import json

df = pd.read_csv('input_file_0.csv')

mapping = {}

# Handle observation_type strings
for obs_type in df['observation_type'].unique():
    mapping[obs_type] = {
        "string_type": "observation_type",
        "dimension": obs_type
    }

# Handle unit_desc strings
for _, row in df.iterrows():
    obs_type = row['observation_type']
    unit_desc = row['unit_desc']

    if unit_desc in mapping:
        continue

    money_unit = None
    weight_unit = None
    area_unit = None
    basis = None

    parts = [p.strip() for p in unit_desc.split(',')]
    main_part = parts[0]
    if len(parts) > 1:
        basis = ', '.join(parts[1:])

    # Heuristics based on obs_type context (though we want unit_desc to be context-independent if possible)
    # Actually, let's just parse the unit_desc based on its content mostly, or use the first time we saw it.
    if obs_type == 'price':
        if '/' in main_part:
            num, den = [p.strip() for p in main_part.split('/')]
            money_unit = num
            weight_unit = den
        else:
            money_unit = main_part

    elif obs_type == 'yield':
        if '/' in main_part:
            num, den = [p.strip() for p in main_part.split('/')]
            weight_unit = num
            area_unit = den
        else:
            weight_unit = main_part

    elif obs_type == 'production':
        if '$' in main_part:
            money_unit = main_part
        else:
            weight_unit = main_part

    elif obs_type == 'area':
        area_unit = main_part

    mapping[unit_desc] = {
        "string_type": "unit_desc",
        "money_unit": money_unit,
        "weight_unit": weight_unit,
        "area_unit": area_unit,
        "basis": basis
    }

print(json.dumps(mapping, indent=2))
```

```python {.marimo}
# 1. Load the JSON mapping generated earlier
with open(root_path / 'code' / 'json' / "crop_unit_mapping.json", "r") as f:
    mapping = json.load(f)

# 2. Split the master mapping into individual column dictionaries
# This is the fastest and safest way to map multiple target columns in Polars
obs_type_map = {
    k: v["dimension"] 
    for k, v in mapping.items() 
    if v["string_type"] == "observation_type"
}

money_unit_map = {k: v["money_unit"] for k, v in mapping.items() if v["string_type"] == "unit_desc"}
weight_unit_map = {k: v["weight_unit"] for k, v in mapping.items() if v["string_type"] == "unit_desc"}
area_unit_map = {k: v["area_unit"] for k, v in mapping.items() if v["string_type"] == "unit_desc"}
basis_map = {k: v["basis"] for k, v in mapping.items() if v["string_type"] == "unit_desc"}

# 4. Apply the mapping to generate the new harmonized columns
qs_extracted_units = qs_survey_crops.with_columns(
    # Extract the true dimension from observation_type
    dimension = pl.col("observation_type").replace_strict(obs_type_map),

    # Extract base units and qualitative basis from unit_desc
    money_unit = pl.col("unit_desc").replace_strict(money_unit_map),
    weight_unit = pl.col("unit_desc").replace_strict(weight_unit_map),
    area_unit = pl.col("unit_desc").replace_strict(area_unit_map),
    basis = pl.col("unit_desc").replace_strict(basis_map)
)
qs_extracted_units
```

```python {.marimo}
# Want to harmonize differing spellings of common unit names for easier comparison
# These are dict mappings into common unit names
area_unit_common_name = {
  "ACRES WHERE TAPS SET": "Acre",
  "ACRE": "Acre",
  "ACRES": "Acre",
  "SQ FT": "Square Foot",
  "TAP": "Tap",
  "NET PLANTED ACRE": "Acre"
}

weight_unit_common_name = {
  "BUNCH": "Bunch",
  "FLAT": "Flat",
  "LB": "Pound",
  "480 LB BALES": "Bale",
  "CWT": "Hundredweight",
  "GALLON": "Gallon",
  "BARRELS": "Barrel",
  "TON": "Ton",
  "STEM": "Stem",
  "BOX": "Box",
  "BU": "Bushel",
  "BARREL": "Barrel",
  "TONS": "Ton",
  "POT": "Pot",
  "BLOOM": "Bloom",
  "GALLONS": "Gallon",
  "BASKET": "Basket",
  "BOXES": "Box",
  "SPIKE": "Spike"
}

money_unit_common_name = {
    "$": "Dollar",
    "CENTS": "Cent"
}
```

```python {.marimo}
qs_harmonized_unit_name = qs_extracted_units.with_columns(
    pl.col("area_unit").replace(area_unit_common_name).alias('area_unit'),
    pl.col("weight_unit").replace(weight_unit_common_name).alias('weight_unit'),
    pl.col("money_unit").replace(money_unit_common_name).alias('money_unit'),
)
qs_harmonized_unit_name
```

```python {.marimo}
# Identify unique Crop-Unit pairs from your harmonized dataframe
unique_pairs = (qs_harmonized_unit_name
    .select(['commodity_desc', "weight_unit"])
    .unique()
    .filter(pl.col("weight_unit").is_not_null())
)
```

```python {.marimo}
# Define pydantic output structure
class WeightConversionOutput(BaseModel):
    # This ensures a clean list of integers
    conversion_factor: float = Field(description="Multiplier to convert given unit into pounds.")

    # This captures the 'Why' for auditing
    reasoning: str = Field(description="Brief justification for the conversion multiplier chosen.")
```

```python {.marimo}
# Define DSpy signature
class AgUnitConversion(dspy.Signature):
    """
    Look up the standard USDA/NASS conversion factor to convert a specific agricultural commodity unit into Pounds (LBS).
    If USDA/NASS conversion factor is not available there may be state legislation that defines the conversion factor.
    Use standard test weights (e.g., Corn Bushel = 56 lbs, Wheat Bushel = 60 lbs).
    """
    crop_name = dspy.InputField(desc="The NASS commodity name (e.g., CORN, APPLES, PEACHES)")
    unit_name = dspy.InputField(desc="The harmonized unit name (e.g., Bushel, Basket, Barrel, Box)")

    output: WeightConversionOutput = dspy.OutputField()

analyzer = dspy.ChainOfThought(AgUnitConversion)
```

```python {.marimo}
# test_commodity='ORANGES'
# test_unit="Box"
# prediction = analyzer(crop_name=test_commodity, unit_name=test_unit)
# prediction
```

```python {.marimo}
# Unambiguous constants to skip LLM calls
static_weights = {
    "Pound": 1.0,
    "Ton": 2000.0,
    "Hundredweight": 100.0,
    "Bale": 480.0  # Per NASS standard for Cotton 480lb bales here
}

# Storage for results
conversion_results = []
checkpoint_path = Path(json_path / "ag_unit_conversions.jsonl")

# Load existing results if restarting an incomplete run
existing_keys = set()
if checkpoint_path.exists():
    with open(checkpoint_path, "r") as _f:
        for line in _f:
            data = json.loads(line)
            existing_keys.add((data['commodity_desc'], data['weight_unit']))
            conversion_results.append(data)

# Convert unique_pairs to list of dicts for iteration
pairs_to_process = unique_pairs.to_dicts()
```

```python {.marimo}
print(f"Processing {len(pairs_to_process)} unique crop-unit pairs...")

for pair in tqdm.tqdm(pairs_to_process):
    commodity = pair['commodity_desc']
    unit = pair['weight_unit']

    # 1. Skip if already processed in a previous run
    if (commodity, unit) in existing_keys:
        print(f"{commodity}-{unit} already processed")
        continue

    # 2. Handle static conversions, bypass LLM
    if unit in static_weights:
        print(f"{commodity}-{unit} is in standard weight unit")
        result = {
            "commodity_desc": commodity,
            "weight_unit": unit,
            "conversion_factor": static_weights[unit],
            "reasoning": "Static constant weight defined in pipeline logic.",
            "source": "static"
        }

    # 3. Handle LLM Conversions
    else:
        print(f"Processing {commodity}-{unit}")
        try:
            # Call your DSPy analyzer
            pred = analyzer(crop_name=commodity, unit_name=unit)

            # Extract values from the pydantic output
            result = {
                "commodity_desc": commodity,
                "weight_unit": unit,
                "conversion_factor": float(pred.output.conversion_factor),
                "reasoning": pred.output.reasoning,
                "source": "llm_dspy"
            }
        except Exception as e:
            print(f"Error processing {commodity}-{unit}: {e}")
            continue # Skip and move to next

    # 4. Store and Checkpoint
    conversion_results.append(result)
    with open(checkpoint_path, "a") as _f:
        _f.write(json.dumps(result) + "\n")
```

```python {.marimo}
# Convert final list to Polars DataFrame and join back to NASS
unit_conversion_lookup = pl.DataFrame(conversion_results)
qs_final_standardized = (qs_harmonized_unit_name
    .join(unit_conversion_lookup, on=["commodity_desc", "weight_unit"], how="left")
    .with_columns(
        pl.when( # Price (weight in denominator): Divide by factor
            (pl.col("observation_type") == "price") & 
            (~pl.col('conversion_factor').is_null())
        ).then(pl.col("value") / pl.col("conversion_factor"))
        .when( # Yield: multiply by factor
            (pl.col("observation_type") == "yield") & 
            (~pl.col('conversion_factor').is_null())
        ).then(pl.col("value") * pl.col("conversion_factor"))
        .when( # Non $ production : multiply by factor
            (pl.col("observation_type") == "production") & 
            (~pl.col('conversion_factor').is_null())
        ).then(pl.col("value") * pl.col("conversion_factor"))
        .when( # Area in sq ft, mostly horticulture, mushrooms
            (pl.col("observation_type")=='area') & 
            (pl.col('area_unit')=='Square Foot')
        ).then(pl.col('value') / 43560)
        .when( # Acreage in acres already
            (pl.col("observation_type") == 'area') & 
            (pl.col("area_unit") == 'Acre')
        ).then(pl.col('value')*1)
        .when( # Production in dollar terms already
            (pl.col("observation_type") == 'production') & 
            (pl.col("money_unit") == 'Dollar') &
            (pl.col('weight_unit').is_null())
        ).then(pl.col('value')*1)
        .otherwise(pl.lit(999999))
        .alias('standardized_value')
    ))
```

```python {.marimo}
# Every case is taken care of
qs_final_standardized.drop('source_desc','domain_desc','domaincat_desc','cv_%').filter(pl.col('standardized_value') == 999999)

# Update unit labels
qs_cleaned = (qs_final_standardized
    .filter(pl.col("standardized_value") != 999999)
    .drop_nulls(subset=["standardized_value"])
    # Standardize the unit labels for the final time
    .with_columns(
        pl.when(pl.col("weight_unit").is_not_null()).then(pl.lit("Pound"))
        .when(pl.col("area_unit").is_not_null()).then(pl.lit("Acre"))
        .when(pl.col("money_unit") == "Dollar").then(pl.lit("USD"))
        .otherwise(pl.lit("Unknown"))
        .alias("physical_dimension")
    )
)

# Add CDL code and crop
qs_cleaned = qs_cleaned.join(
    nass_cdl_xwalk,
    how='left',
    on=[
        "group_desc", "commodity_desc", 'class_desc', 'prodn_practice_desc', 'util_practice_desc', 
    ]
)
```

```python {.marimo}
# NASS crop identifier
# Every operation must respect these 8 columns to preserve dimensional and economic integrity.
nass_identity = [
    "group_desc",
    "commodity_desc", 
    "class_desc", 
    "prodn_practice_desc", 
    "util_practice_desc"
]

# The Geographic/Temporal identifier (note CDL crop as a category we are aggregating into)
geo_time_key = ["year", "state_ansi", "cdl_code"]

# The pivoting key
pivot_index = nass_identity + geo_time_key
```

```python {.marimo}
qs_state_only = qs_cleaned.filter(
    pl.col("agg_level_desc") == "STATE",
    ~pl.col('commodity_desc').str.contains('TOTALS')
)
# Calculate Revenue for every state-year
qs_state_revenue = (qs_state_only
    .with_columns(
        obs_key = pl.when((pl.col("observation_type") == "production") & (pl.col("physical_dimension") == "USD"))
                    .then(pl.lit("prod_v")) # prod value in $ already
                    .when(pl.col("observation_type") == "production")
                    .then(pl.lit("prod_q")) # prod in weights
                    .otherwise(pl.col("observation_type"))
    )
    .pivot(
        index=pivot_index,
        on="obs_key",
        values="standardized_value",
        aggregate_function="mean"
    )
    .with_columns(
        # Revenue is the anchor for the Portfolio Weight
        revenue = pl.coalesce([
            pl.col("prod_v"),
            pl.col("price") * pl.col("prod_q")
        ])
    )
    .drop_nulls(subset=["revenue"])
)
```

```python {.marimo}
# Calculate Portfolio Weights using the State-level Population revenue
state_rev_weights = (qs_state_revenue
    .with_columns(
        w = pl.col("revenue") / pl.col("revenue").sum().over(["year", "state_ansi", "cdl_code"])
    )
    .select(["year", "state_ansi", "cdl_code"] + nass_identity + ["w", "price", "revenue"])
)
```

```python {.marimo}
qs_national_only = qs_cleaned.filter(
    pl.col("agg_level_desc") == "NATIONAL",
    ~pl.col('commodity_desc').str.contains('TOTALS')
)
national_pivot_index = copy.deepcopy(pivot_index)
national_pivot_index.remove('state_ansi')
# Calculate Revenue for every state-year
qs_national_revenue = (qs_national_only
    .with_columns(
        obs_key = pl.when((pl.col("observation_type") == "production") & (pl.col("physical_dimension") == "USD"))
                    .then(pl.lit("prod_v")) # prod value in $ already
                    .when(pl.col("observation_type") == "production")
                    .then(pl.lit("prod_q")) # prod in weights
                    .otherwise(pl.col("observation_type"))
    )
    .pivot(
        index=national_pivot_index,
        on="obs_key",
        values="standardized_value",
        aggregate_function="mean"
    )
    .with_columns(
        # Revenue is the anchor for the Portfolio Weight
        revenue = pl.coalesce([
            pl.col("prod_v"),
            pl.col("price") * pl.col("prod_q")
        ])
    )
    .drop_nulls(subset=["revenue"])
)
qs_national_revenue
```

```python {.marimo}

```