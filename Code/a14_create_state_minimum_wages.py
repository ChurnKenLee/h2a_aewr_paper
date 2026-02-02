import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import json
    from pandas.api.types import union_categoricals
    from itertools import islice
    import re
    import addfips
    import requests
    import urllib
    import time
    DC_STATEHOOD = 1 # Enables DC to be included in the state list
    import us
    import pickle
    import rapidfuzz
    return Path, pd, requests, us


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use FRED API to get state-level non-farm minimum wage
    """)
    return


@app.cell
def _():
    # FRED API key from my account
    # Import API key stored in text file
    with open("../tools/fred_api_key.txt") as f:
        lines = f.readlines()

    api_key = lines[0]
    return (api_key,)


@app.cell
def _(api_key, requests):
    # Get release ID of state minimum wages
    # Grab release ID of all releases
    fred_release_api_url = 'https://api.stlouisfed.org/fred/releases'
    release_params = {
        'api_key': api_key,
        'file_type': 'json'
        }

    fred_release = requests.get(fred_release_api_url, params = release_params)
    release_list = fred_release.json()['releases']

    # Take the release id of the state minimum wage series
    state_minimum_wage_release = list(filter(lambda release_dict: release_dict['name'] == 'Minimum Wage Rate by State', release_list))
    state_minimum_wage_release_id = state_minimum_wage_release[0]['id']
    state_minimum_wage_release_id
    return


@app.cell
def _(api_key):
    # Base URL to call API
    fred_observations_api_url = 'https://api.stlouisfed.org/fred/v2/release/observations'

    # Bearer token has to go in request header
    fred_observations_headers = {
        'Authorization': f'Bearer {api_key}'
    }

    fred_observations_params = {
        'release_id': '387',
        'format': 'json'
    }
    return (
        fred_observations_api_url,
        fred_observations_headers,
        fred_observations_params,
    )


@app.cell
def _(
    fred_observations_api_url,
    fred_observations_headers,
    fred_observations_params,
    requests,
):
    # Get all state-level observations
    response = requests.get(fred_observations_api_url, headers= fred_observations_headers, params = fred_observations_params)
    list_obs = response.json()['series']
    return (list_obs,)


@app.cell
def _(list_obs, pd):
    # Convert from JSON to dataframe
    df = pd.json_normalize(list_obs).reset_index(drop=True)
    df = df.explode('observations', ignore_index = True)
    return (df,)


@app.cell
def _(df, pd):
    # Observations column is still in JSON, unpack that as well as a separate dataframe, then merge back into original dataframe
    obs = pd.json_normalize(df.pop('observations')).reset_index(drop=True)
    df_1 = pd.concat([df, obs], axis=1)
    return (df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Preserve only information we want in dataframe
    """)
    return


@app.cell
def _(df_1):
    # Add state name, abbreviation, and year
    df_1['state_name'] = df_1['title'].str.split('for ', expand=True)[1]
    df_1['year'] = df_1['date'].str[0:4]
    df_1['state_abbreviation'] = df_1['series_id'].str[-2:]
    fed = df_1['series_id'] == 'STTMINWGFG'
    # Federal minimum wage
    df_1.loc[fed, 'state_name'] = 'USA'
    df_1.loc[fed, 'state_abbreviation'] = 'US'
    # Georgia has monthly observations that we can remove
    df_2 = df_1[df_1['frequency'] == 'Annual']
    return (df_2,)


@app.cell
def _(df_2):
    # min_wage_df = min_wage_df[min_wage_df['year'] > '1999']
    min_wage_df = df_2[['state_name', 'state_abbreviation', 'year', 'value', 'units', 'seasonal_adjustment']].copy()
    return (min_wage_df,)


@app.cell
def _(df_2):
    df_2.to_csv('min_wage.csv')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Add agricultural worker exemption data
    """)
    return


@app.cell
def _(min_wage_df):
    ag_exemption_list = ['AK', 'DE', 'GA', 'IN', 'KS', 'KY', 'ME', 'MA', 'NE', 'NH', 'NJ', 'OK', 'RI', 'VT', 'VA', 'WA', 'WV', 'WY']
    min_wage_df['agriculture_exemption'] = min_wage_df['state_abbreviation'].isin(ag_exemption_list)
    return


@app.cell
def _(min_wage_df, us):
    # Add state FIPS code and export
    min_wage_df['state_fips'] = min_wage_df['state_name'].map(us.states.mapping('name', 'fips'))
    min_wage_df_1 = min_wage_df.rename(columns={'state_fips': 'state_fips_code'})
    return (min_wage_df_1,)


@app.cell
def _(Path, min_wage_df_1):
    output_file = 'state_year_min_wage.parquet'
    output_dir = Path('../binaries')
    output_dir.mkdir(parents=True, exist_ok=True)
    min_wage_df_1.to_parquet(output_dir / output_file, index=False)  # can join path elements with / operator
    output_dir = Path('../files_for_phil')
    output_dir.mkdir(parents=True, exist_ok=True)
    min_wage_df_1.to_parquet(output_dir / output_file, index=False)
    return


@app.cell
def _(min_wage_df_1):
    min_wage_df_1
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
