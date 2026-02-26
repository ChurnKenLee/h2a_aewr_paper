import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
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

    return Path, dspy, json, mo, os, pl, pyprojroot


@app.cell
def _(dspy, os, pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    binary_path = root_path / 'binaries'
    json_path = root_path / 'code' / 'json'
    cdl_path = root_path / 'data' / 'croplandcros_cdl'
    # Configure Gemini
    gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    gemini = dspy.LM('gemini/gemini-3-flash-preview', api_key=gemini_api_key)
    dspy.configure(lm=gemini)
    return binary_path, cdl_path, json_path, root_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load CDL code NASS crosswalk, CDL acreage, NASS survey prices
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load CDL codes from USDA, then combine with matching responses from LLM stored in jsonl
    """)
    return


@app.cell
def _(Path, cdl_path, json_path, pl):
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
    return (df_results,)


@app.cell
def _(json, json_path, pl):
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
    return (nass_obs_selection,)


@app.cell
def _(binary_path, df_results, nass_obs_selection):
    # NASS CDL crosswalk
    nass_cdl_xwalk = df_results.select(['nass_id', 'cdl_code', 'cdl_name'])
    nass_cdl_xwalk = nass_obs_selection.join(
        nass_cdl_xwalk,
        how='left',
        on=['nass_id']
    )
    nass_cdl_xwalk.write_parquet(binary_path / 'nass_cdl_crosswalk.parquet')
    return (nass_cdl_xwalk,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    CDL pixel counts previously aggregated using exactextract
    """)
    return


@app.cell
def _(binary_path, pl):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Subset of NASS survey data from filtering NASS QuickStats survey data with the NASS crop selection list
    """)
    return


@app.cell
def _(binary_path, pl):
    qs_survey_crops = pl.read_parquet(binary_path / 'qs_survey_selected_obs.parquet')
    # Drop stats we don't care about
    qs_survey_crops = qs_survey_crops.filter(
        pl.col('observation_type') != ''
    )
    return (qs_survey_crops,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prepare NASS data
    """)
    return


@app.cell
def _(qs_survey_crops):
    qs_survey_crops.select('observation_type', 'unit_desc').unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    """)
    return


@app.cell
def _(json, pl, qs_survey_crops, root_path):
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
    df_extracted_units = qs_survey_crops.with_columns(
        # Extract the true dimension from observation_type
        dimension = pl.col("observation_type").replace_strict(obs_type_map),
    
        # Extract base units and qualitative basis from unit_desc
        money_unit = pl.col("unit_desc").replace_strict(money_unit_map),
        weight_unit = pl.col("unit_desc").replace_strict(weight_unit_map),
        area_unit = pl.col("unit_desc").replace_strict(area_unit_map),
        basis = pl.col("unit_desc").replace_strict(basis_map)
    )
    df_extracted_units
    return (df_extracted_units,)


@app.cell
def _():
    return


@app.cell
def _(df_extracted_units, nass_cdl_xwalk):
    # The Pivot Operation
    wide_panel = df_extracted_units.pivot(
        on="observation_type",
        index=["year",
               "state_ansi",
               "nass_id",
               'group_desc',
               'commodity_desc',
               'class_desc',
               'prodn_practice_desc',
               'util_practice_desc'
              ], # The unique row keys
        values=[
            "value",
            "money_unit", 
            "weight_unit", 
            "area_unit", 
            "basis",
            "unit_desc"
        ],
        aggregate_function="first" # Ensures we take the specific record's metadata
    )
    wide_panel = wide_panel.join(
        nass_cdl_xwalk,
        how='left',
        on=[
            "nass_id",
            'group_desc',
            'commodity_desc',
            'class_desc',
            'prodn_practice_desc',
            'util_practice_desc'
        ]
    )
    return (wide_panel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Harmonize unit names so that we can correctly identify when the units match
    """)
    return


@app.cell
def _(pl, wide_panel):
    unit_cols = wide_panel.select([
        "money_unit_yield",
        "money_unit_production",
        "money_unit_area",
        "money_unit_price",
        "weight_unit_yield",
        "weight_unit_production",
        "weight_unit_area",
        "weight_unit_price",
        "area_unit_yield",
        "area_unit_production",
        "area_unit_area",
        "area_unit_price"
    ])

    money_units = unit_cols.select([
        pl.col("^money_.*$")
    ]).unpivot()

    weight_units = unit_cols.select([
        pl.col("^weight_.*$")
    ]).unpivot()

    area_units = unit_cols.select([
        pl.col("^area_.*$")
    ]).unpivot()
    return area_units, money_units, weight_units


@app.cell
def _(money_units):
    money_units.select('value').unique().drop_nulls()
    return


@app.cell
def _(weight_units):
    weight_units.select('value').unique().drop_nulls()
    return


@app.cell
def _(area_units):
    area_units.select('value').unique().drop_nulls()
    return


@app.cell
def _():
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
    return (
        area_unit_common_name,
        money_unit_common_name,
        weight_unit_common_name,
    )


@app.cell
def _(
    area_unit_common_name,
    money_unit_common_name,
    pl,
    weight_unit_common_name,
    wide_panel,
):
    # Replace unit names with common ones
    wide_panel_common_unit_name = wide_panel.with_columns(
        pl.col("^weight_.*$").replace(weight_unit_common_name),
        pl.col("^area_.*$").replace(area_unit_common_name),
        pl.col("^money_.*$").replace(money_unit_common_name)
    )
    return (wide_panel_common_unit_name,)


@app.cell
def _(pl, root_path):
    usda_crop_weight = pl.read_json(root_path/'code'/'json'/'usda_crop_weight.json')
    nass_crop_weight_mapping = pl.read_json(root_path/'code'/'json'/'nass_crop_weight_mapping.json')
    nass_crop_weight = nass_crop_weight_mapping.join(
        usda_crop_weight,
        on=[
            'Category',
            'Crop'
        ],
        how='left'
    )
    return


@app.cell
def _():
    standard_units = [
        'Acre',
        'Square Foot',
        'Pound',
        'Bale',
        'Hundredweight',
        'Ton'
    ]

    def evaluate_unit_harmony(row):
        """
        Evaluates bi-directional unit consistency across the 4-way pivot.
        Returns flags indicating if conversion is needed for each linkage.
        """
    
        def check_link(unit_a, unit_b):
            # Case 0: Data is sparse
            if (unit_a is None) or (unit_b is None):
                return "no_unit_comparison"
        
            # Case 1: Direct unit match
            if unit_a == unit_b:
                return "ok"
        
            # Case 2: Weight-to-weight adjustment using standard units
            if unit_a in standard_units and unit_b in standard_units:
                return "weight_adjust"
            
            # Case 3: Colloquial mismatch (requires fallback to AvgWeightLbs)
            return "use_colloquial_conversion"

        # Case 1: Price ($/weight) vs Production (weight)
        price_prod_harmony = check_link(
            row["weight_unit_price"], row["weight_unit_production"]
        )

        if row["money_unit_production"] == 'Dollar':
            price_prod_harmony = "production_already_in_dollars"

        # Case 2: Area (acres) vs Yield (weight/area)
        area_yield_harmony = check_link(
            row["area_unit_area"], row["area_unit_yield"]
        )

        # Case 3: Yield (weight/area) vs Price ($/weight)
        yield_price_harmony = check_link(
            row["weight_unit_yield"], row["weight_unit_price"]
        )
    
        return {
            "price_prod_status": price_prod_harmony,
            "area_yield_status": area_yield_harmony,
            "yield_price_status": yield_price_harmony,
            # Flag if ANY link requires the USDA Colloquial Fallback
            "requires_colloquial_conversion": (
                price_prod_harmony == "use_colloquial_conversion" or 
                area_yield_harmony == "use_colloquial_conversion" or 
                yield_price_harmony == "use_colloquial_conversion"
            )
        }

    return (evaluate_unit_harmony,)


@app.cell
def _(evaluate_unit_harmony, wide_panel_common_unit_name):
    test_row = wide_panel_common_unit_name.row(index=10, named=True)
    evaluate_unit_harmony(test_row)
    return


@app.cell
def _(evaluate_unit_harmony, pl, wide_panel_common_unit_name):
    # Evaluate harmonization needs and execute
    wide_panel_harmony_report = wide_panel_common_unit_name.with_columns(
        pl.struct([
            "value_price", "value_production", "value_yield", "value_area",
            "weight_unit_price", "weight_unit_production", "weight_unit_yield",
            "area_unit_price", "area_unit_production", "area_unit_yield", "area_unit_area",
            "money_unit_production"
        ]).map_elements(evaluate_unit_harmony, return_dtype=pl.Struct([
            pl.Field("price_prod_status", pl.Utf8),
            pl.Field("area_yield_status", pl.Utf8),
            pl.Field("yield_price_status", pl.Utf8),
            pl.Field("requires_colloquial_conversion", pl.Boolean)
        ])).alias("harmony_report")
    ).unnest("harmony_report")
    return (wide_panel_harmony_report,)


@app.cell
def _(pl, wide_panel_harmony_report):
    # What are these crops that need a colloquial map?
    needs_colloquial_map = (wide_panel_harmony_report
        .filter(
            pl.col('requires_colloquial_conversion')
        ).select([
            "nass_id",
            'group_desc', 'commodity_desc', 'class_desc', 'prodn_practice_desc', 'util_practice_desc',
            "weight_unit_price", "weight_unit_production", "weight_unit_yield",
            'nass_semantic_label', 'cdl_name'
        ])
        .unique())
    needs_colloquial_map
    return


@app.cell
def _(pl, wide_panel_harmony_report):
    # Conversion factors to pound (weight) and acre (area)
    # Note: Bale is standardized to 480lbs (NASS standard in this dataset)
    unit_base_conversion_factor = {
        "Pound": 1.0,
        "Hundredweight": 100.0,
        "Ton": 2000.0,
        "Bale": 480.0,
        "Acre": 1.0,
        "Square Foot": 1.0 / 43560.0
    }

    # Colloquial conversion factors needed
    # Maple syrup is fine as is because it uses gallons
    # Sorghum: 1 bushel = 56 lbs
    # Apple: 1 bushel = 40 lbs
    # Cranberries: 1 barrel = 100 lbs
    # Peaches: 1 bushel = 50 lbs

    # Define a helper function to do both mappings
    def get_conversion_factor(unit_col_name: str):
        return (
            pl.when(pl.col("commodity_desc") == "SORGHUM", pl.col(unit_col_name) == "Bushel").then(56.0)
            .when(pl.col("commodity_desc") == "APPLES", pl.col(unit_col_name) == "Bushel").then(40.0)
            .when(pl.col("commodity_desc") == "PEACHES", pl.col(unit_col_name) == "Bushel").then(50.0)
            .when(pl.col("commodity_desc") == "CRANBERRIES", pl.col(unit_col_name) == "Barrel").then(100.0)
            .otherwise(
                pl.col(unit_col_name).replace_strict(unit_base_conversion_factor, default=1.0)
            )
        )

    wide_panel_harmonized_unit_values = wide_panel_harmony_report.with_columns([
        # Map factors for weights using the helper logic
        get_conversion_factor("weight_unit_price").alias("f_weight_price"),
        get_conversion_factor("weight_unit_production").alias("f_weight_prod"),
        get_conversion_factor("weight_unit_yield").alias("f_weight_yield"),
    
        # Map factors for area
        pl.col("area_unit_area").replace_strict(unit_base_conversion_factor, default=1.0).alias("f_area_area"),
        pl.col("area_unit_yield").replace_strict(unit_base_conversion_factor, default=1.0).alias("f_area_yield"),
    ])

    wide_panel_harmonized_unit_values
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
