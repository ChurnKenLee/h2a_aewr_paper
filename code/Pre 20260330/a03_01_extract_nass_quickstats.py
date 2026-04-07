# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.18.4",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import pyprojroot
    import numpy as np
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    return mo, pl, pyprojroot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NASS census and survey binaries
    """)
    return


@app.cell
def _(pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    binary_path = root_path/'binaries'
    return binary_path, root_path


@app.cell
def _(pl):
    quickstats_pl_type_dict = {
        'SOURCE_DESC': pl.Categorical,
        'SECTOR_DESC': pl.Categorical,
        'GROUP_DESC': pl.Categorical,
        'COMMODITY_DESC': pl.Categorical,
        'CLASS_DESC': pl.Categorical,
        'PRODN_PRACTICE_DESC': pl.Categorical,
        'UTIL_PRACTICE_DESC': pl.Categorical,
        'STATISTICCAT_DESC': pl.Categorical,
        'UNIT_DESC': pl.Categorical,
        'SHORT_DESC': pl.Categorical,
        'DOMAIN_DESC': pl.Categorical,
        'DOMAINCAT_DESC': pl.Categorical,
        'AGG_LEVEL_DESC': pl.Categorical,
        'STATE_ANSI': pl.Categorical,
        'STATE_FIPS_CODE': pl.Categorical,
        'STATE_ALPHA': pl.Categorical,
        'STATE_NAME': pl.Categorical,
        'ASD_CODE': pl.Categorical,
        'ASD_DESC': pl.Categorical,
        'COUNTY_ANSI': pl.Categorical,
        'COUNTY_CODE': pl.Categorical,
        'COUNTY_NAME': pl.Categorical,
        'REGION_DESC': pl.Categorical,
        'ZIP_5': pl.Categorical,
        'WATERSHED_CODE': pl.Categorical,
        'WATERSHED_DESC': pl.Categorical,
        'CONGR_DISTRICT_CODE': pl.Categorical,
        'COUNTRY_CODE': pl.Categorical,
        'COUNTRY_NAME': pl.Categorical,
        'LOCATION_DESC': pl.Categorical,
        'YEAR': pl.Int64,
        'FREQ_DESC': pl.Categorical,
        'BEGIN_CODE': pl.Categorical,
        'END_CODE': pl.Categorical,
        'REFERENCE_PERIOD_DESC': pl.Categorical,
        'WEEK_ENDING': pl.Categorical,
        'LOAD_TIME': pl.String,
        'VALUE': pl.String,
        'CV_%': pl.String,
    }
    return (quickstats_pl_type_dict,)


@app.cell
def _(binary_path, pl, quickstats_pl_type_dict, root_path):
    nass_qs_list = [
        'animals_products',
        'crops',
        'economics',
        'demographics'
        # 'environmental'
        ]
    # environmental csv is malformed or encoded incorrectly

    # Extract quickstats archive, read with polars, then export as binaries
    quickstats_path = root_path / 'Data' / 'quickstats'
    for qs_type in nass_qs_list:
        for file_path in quickstats_path.iterdir():
            file_name = file_path.name
            if qs_type in file_name:

                print(file_path)

                df = pl.read_csv(
                    file_path, 
                    separator='\t', 
                    has_header=True, 
                    schema=quickstats_pl_type_dict
                )
                df = df.select(pl.all().name.to_lowercase())

                census_df = df.filter(
                    pl.col('source_desc') == 'CENSUS'
                )
                census_df.write_parquet(
                    binary_path/f'qs_census_{qs_type}.parquet'
                )

                survey_df = df.filter(
                    pl.col('source_desc') == 'SURVEY'
                )
                survey_df.write_parquet(
                    binary_path/f'qs_survey_{qs_type}.parquet'
                )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
