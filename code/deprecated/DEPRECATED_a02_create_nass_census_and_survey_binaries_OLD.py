# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.18.4",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import marimo as mo
    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Census of Agriculture binaries
    """)
    return


@app.cell
def _():
    census_2002_dtype_dict = {
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
        'STATE_ANSI': 'str',
        'STATE_FIPS_CODE': 'str',
        'STATE_ALPHA': 'str',
        'STATE_NAME': 'str',
        'ASD_CODE': 'str',
        'ASD_DESC': 'str',
        'COUNTY_ANSI': 'str',
        'COUNTY_CODE': 'str',
        'COUNTY_NAME': 'str',
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
    return (census_2002_dtype_dict,)


@app.cell
def _(pd):
    # Function that extracts Census of Agriculture compressed CSVs and exports them as parquet binaries
    def export_census(census_archive_path, dtype_dict, parquet_output_path):
        census_df = pd.read_csv(census_archive_path, compression = 'gzip', sep = '\t', skiprows=0, header=0, dtype=dtype_dict, parse_dates=['LOAD_TIME'])
        census_df.to_parquet(parquet_output_path)
    return (export_census,)


@app.cell
def _(census_2002_dtype_dict, export_census):
    for year in [2002, 2007, 2012, 2017, 2022]:
        census_archive_path = f'../Data/census_of_agriculture/qs.census{year}.txt.gz'
        parquet_output_path = f'../binaries/census_of_agriculture_{year}'
        print(f'Exporting year {year}')
        export_census(census_archive_path, census_2002_dtype_dict, parquet_output_path)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
