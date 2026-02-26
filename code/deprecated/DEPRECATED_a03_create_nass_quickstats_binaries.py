# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.17.0",
#     "nasspython>=1.0.0",
#     "numpy>=2.4.0",
#     "pandas>=2.3.3",
#     "pyzmq>=27.1.0",
#     "requests>=2.32.5",
# ]
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from nasspython import nass_api
    import requests
    from inspect import getmembers, isfunction
    from time import sleep
    return Path, pd


@app.cell
def _():
    # Import API key stored in text file
    with open("../tools/nass_quickstats_api_key.txt") as f:
        lines = f.readlines()

    api_key = lines[0]
    return (api_key,)


@app.cell
def _():
    # Pull records by state to keep number of records under 50,000, which is the API max number of records limit
    state_names = ["Alabama", "Arkansas", "Arizona", "California", 
    "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", 
    "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", 
    "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", 
    "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", 
    "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", 
    "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", 
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", 
    "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", 
    "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
    return


@app.cell
def _():
    records_url = 'https://quickstats.nass.usda.gov/api/api_GET/'
    count_url = 'https://quickstats.nass.usda.gov/api/get_counts/'
    possible_params_url = 'https://quickstats.nass.usda.gov/api/get_param_values/'
    return


@app.cell
def _(api_key):
    # Set request parameters to reduce number of records below 50000, which is the max number of records per request
    payload = {
        'key': api_key,
        'sector_desc': 'CROPS',
        'statisticcat_desc': 'PRICE RECEIVED',
        'year': '2023',
        'format': 'json'
    }
    return


@app.cell
def _():
    # r = requests.get(url = records_url, params = payload)
    return


@app.cell
def _():
    crops_dtype_dict = {
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
    return (crops_dtype_dict,)


@app.cell
def _(Path, crops_dtype_dict, pd):
    # Load CROPS data from NASS surveys, obtained from the NASS large datasets repository here: https://www.nass.usda.gov/datasets/
    gz_path = Path("../Data/nass_quickstats/qs.crops_20250815.txt.gz")
    crops_df = pd.read_csv(gz_path, compression='gzip', sep='\t', skiprows=0, header=0, dtype=crops_dtype_dict, parse_dates=['LOAD_TIME'])
    crops_df.to_parquet('../binaries/nass_quickstats_crops.parquet')
    return


if __name__ == "__main__":
    app.run()
