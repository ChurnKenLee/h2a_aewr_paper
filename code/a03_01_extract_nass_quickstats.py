# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.18.4",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.19.11"
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
    # Census of Agriculture binaries
    """)
    return


@app.cell
def _(pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    return (root_path,)


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

    nass_qs_list = [
        'animals_products',
        'crops',
        'economics',
        'demographics'
        ]
    # don't include 'environmental' due to encoding error for now
    return nass_qs_list, quickstats_pl_type_dict


@app.cell
def _():
    quickstats_pd_type_dict = {
        'SOURCE_DESC': 'category',
        'SECTOR_DESC': 'category',
        'GROUP_DESC': 'category',
        'COMMODITY_DESC': 'category',
        'CLASS_DESC': 'category',
        'PRODN_PRACTICE_DESC': 'category',
        'UTIL_PRACTICE_DESC': 'category',
        'STATISTICCAT_DESC': 'category',
        'UNIT_DESC': 'category',
        'SHORT_DESC': 'category',
        'DOMAIN_DESC': 'category',
        'DOMAINCAT_DESC': 'category',
        'AGG_LEVEL_DESC': 'category',
        'STATE_ANSI': 'category',
        'STATE_FIPS_CODE': 'category',
        'STATE_ALPHA': 'category',
        'STATE_NAME': 'category',
        'ASD_CODE': 'category',
        'ASD_DESC': 'category',
        'COUNTY_ANSI': 'category',
        'COUNTY_CODE': 'category',
        'COUNTY_NAME': 'category',
        'REGION_DESC': 'category',
        'ZIP_5': 'category',
        'WATERSHED_CODE': 'category',
        'WATERSHED_DESC': 'category',
        'CONGR_DISTRICT_CODE': 'category',
        'COUNTRY_CODE': 'category',
        'COUNTRY_NAME': 'category',
        'LOCATION_DESC': 'category',
        'YEAR': 'int64',
        'FREQ_DESC': 'category',
        'BEGIN_CODE': 'category',
        'END_CODE': 'category',
        'REFERENCE_PERIOD_DESC': 'category',
        'WEEK_ENDING': 'category',
        'LOAD_TIME': 'str',
        'VALUE': 'str',
        'CV_%': 'str',
    }
    return


@app.cell
def _(pl, root_path):
    test = pl.read_parquet(root_path / 'binaries' / 'qs_census_crops.parquet')
    test['COMMODITY_DESC'].value_counts(sort=True)
    test.filter(pl.col('COMMODITY_DESC') == 'CHICKPEAS')
    return


@app.cell
def _(nass_qs_list, pl, quickstats_pl_type_dict, root_path):
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

                census_df = df.filter(
                    pl.col('SOURCE_DESC') == 'CENSUS'
                )
                census_df.write_parquet(
                    root_path / 'binaries' / f'qs_census_{qs_type}.parquet'
                )

                survey_df = df.filter(
                    pl.col('SOURCE_DESC') == 'SURVEY'
                )
                survey_df.write_parquet(
                    root_path / 'binaries' / f'qs_survey_{qs_type}.parquet'
                )
    return


if __name__ == "__main__":
    app.run()
