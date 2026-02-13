import marimo

__generated_with = "0.19.7"
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
    return Path, pd, re, requests, time, us


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Obtain USDA MyMarketNews reports using their API
    """)
    return


@app.cell
def _():
    # USDA API key from my account
    # Import API key stored in text file
    with open("../tools/mymarketnews_api_key.txt") as f:
        lines = f.readlines()

    api_key = lines[0]
    return (api_key,)


@app.cell
def _(api_key):
    # Base URL to call API
    base_url = 'https://marsapi.ams.usda.gov/services/v1.2/reports/'

    # Use API key as basic authentication username, password not needed
    from requests.auth import HTTPBasicAuth
    api_auth = HTTPBasicAuth(api_key, '')
    return api_auth, base_url


@app.cell
def _():
    # # Make request
    # request_url = base_url
    # r = requests.get(url = request_url, auth = api_auth)

    # # Put response into spreadsheet for analysis
    # r_json = r.json()
    # data = [x for x in r_json]
    # all_reports = pd.DataFrame(data)
    # all_reports.to_excel("mymarketnews_reports.xlsx")
    return


@app.cell
def _(pd):
    all_reports = pd.read_excel("mymarketnews_reports.xlsx")
    # Get slug ID and name of all terminal (wholesale) market reports
    terminal_markets = all_reports[all_reports['market_types'] == '[\'Terminal\']']

    # Remove foreign terminal markets
    terminal_markets = terminal_markets[terminal_markets['offices'] != '[\'Washington, DC International - SC\']']

    # Get slug ID and name of all shipping point reports
    shipping_points = all_reports[all_reports['market_types'] == '[\'Shipping Point\']']
    return shipping_points, terminal_markets


@app.cell
def _():
    # report_header_df = pd.DataFrame()

    # for slug_id in terminal_markets['slug_id']:
    #     print(str(slug_id))

    #     time.sleep(1) # stagger each request by 1 sec

    #     # Pull report headers using Slug ID
    #     # Define request
    #     request_url = base_url + str(slug_id) # + '?q=' + 'report_begin_date=12/19/2024' + '&allSections=true'

    #     # Make request
    #     r = requests.get(url = request_url, auth = api_auth)

    #     # Put response into dataframe
    #     r_json = r.json()
    #     response_df = pd.DataFrame(r_json['results'])
    #     report_header_df = pd.concat([report_header_df, response_df])

    # # Put report headers into big parquet
    # report_header_df.to_parquet("../binaries/mymarketnews_terminal_report_headers.parquet", index=False)
    return


@app.cell
def _(Path):
    # Create directories to store CSVs and binaries
    try:
        Path.mkdir(f'../Data/mymarketnews_reports/')
    except FileExistsError:
        print('Data folder already exists.')
    for _year in range(2004, 2025):
        try:
            Path.mkdir(f'../Data/mymarketnews_reports/mymarketnews_reports_{_year}')
        except FileExistsError as err:
            print('Year ' + f'{_year}' + ' data folder already exists.')
    return


@app.cell
def _(Path, api_auth, base_url, pd, requests, terminal_markets, time):
    # Pull terminal market report details
    for _year in range(2004, 2025):
        terminal_report_details_df = pd.DataFrame()
        for _slug_id in terminal_markets['slug_id']:
            _slug_year_path = Path(f'../Data/mymarketnews_reports/mymarketnews_reports_{_year}/mymarketnews_terminal_market_reports_year_{_year}_slug_{_slug_id}.csv')
            if _slug_year_path.is_file():
                print('Slug ID', str(_slug_id), 'year', str(_year), 'already downloaded.')
                continue  # If slug and year is already downloaded, we can skip it
            print('Slug ID is', str(_slug_id), 'year is', str(_year))
            time.sleep(5)
            _request_url = base_url + str(_slug_id) + '?q=' + f'report_begin_date=01/01/{_year}:12/31/{_year}' + '&allSections=true'
            _r = requests.get(url=_request_url, auth=api_auth)
            try:
                _r_json = _r.json()
            except:
                _r = requests.get(url=_request_url, auth=api_auth)  # stagger each request by 5 sec
            _r_json = _r.json()
            _results = _r_json[1]['results']  # Pull report details using Slug ID
            _result_df = pd.DataFrame(_results)  # Define request
            print('Report has ' + str(len(_result_df)) + ' entries')
            _result_df.to_csv(f'../Data/mymarketnews_reports/mymarketnews_reports_{_year}/mymarketnews_terminal_market_reports_year_{_year}_slug_{_slug_id}.csv', index=False)  # Make request  # Put response into dataframe  # terminal_report_details_df = pd.concat([terminal_report_details_df, result_df])
    return


@app.cell
def _(Path, api_auth, base_url, pd, requests, shipping_points, time):
    # Pull shipping point report details
    for _year in range(2004, 2025):
        shipping_point_report_details_df = pd.DataFrame()
        print('Year is ' + str(_year))
        for _slug_id in shipping_points['slug_id']:
            _slug_year_path = Path(f'../Data/mymarketnews_reports/mymarketnews_reports_{_year}/mymarketnews_shipping_point_reports_year_{_year}_slug_{_slug_id}.csv')
            if _slug_year_path.is_file():
                print('Slug ID', str(_slug_id), 'year', str(_year), 'already downloaded.')  # If slug and year is already downloaded, we can skip it
                continue
            print('Slug ID is', str(_slug_id), 'year is', str(_year))
            time.sleep(5)
            _request_url = base_url + str(_slug_id) + '?q=' + f'report_begin_date=01/01/{_year}:12/31/{_year}' + '&allSections=true'
            _r = requests.get(url=_request_url, auth=api_auth)
            _r_json = _r.json()
            _results = _r_json[1]['results']
            _result_df = pd.DataFrame(_results)  # stagger each request by 5 sec
            print('Report has ' + str(len(_result_df)) + ' entries')
            _result_df.to_csv(f'../Data/mymarketnews_reports/mymarketnews_reports_{_year}/mymarketnews_shipping_point_reports_year_{_year}_slug_{_slug_id}.csv', index=False)  # Pull report details using Slug ID  # Define request  # Make request  # Put response into dataframe  # shipping_point_report_details_df = pd.concat([shipping_point_report_details_df, result_df])
    return


@app.cell
def _(Path, pd):
    for _year in range(2004, 2025):
        _year_df = pd.DataFrame()
        _binary_year_path = Path(f'../binaries/mymarketnews_shipping_point_reports_{_year}.parquet')
        if _binary_year_path.is_file():  # Skip if binary already created
            print(str(_year), 'already converted')
            continue
        else:
            print('Converting ', _year)
            for _csv_path in Path(f'../Data/mymarketnews_reports/mymarketnews_reports_{_year}').iterdir():
                if 'shipping_point' in _csv_path.stem:
                    try:
                        _report_df = pd.read_csv(_csv_path, dtype=str, engine='c')
                        _year_df = pd.concat([_year_df, _report_df], ignore_index=True)
                    except pd.errors.EmptyDataError:
                        print('Empty file:', _csv_path)
        _year_df.rename(columns={'district': 'origin'}, inplace=True)
        _year_df.to_parquet(f'../binaries/mymarketnews_shipping_point_reports_{_year}.parquet')  # Shipping Point reports use district instead of origin
    return


@app.cell
def _(Path, pd):
    for _year in range(2004, 2025):
        _year_df = pd.DataFrame()
        _binary_year_path = Path(f'../binaries/mymarketnews_terminal_market_reports_{_year}.parquet')
        if _binary_year_path.is_file():  # Skip if binary already created
            print(str(_year), 'already converted')
            continue
        else:
            print('Converting ', _year)
            for _csv_path in Path(f'../Data/mymarketnews_reports/mymarketnews_reports_{_year}').iterdir():
                if 'terminal_market' in _csv_path.stem:
                    try:
                        _report_df = pd.read_csv(_csv_path, dtype=str, engine='c')
                        _year_df = pd.concat([_year_df, _report_df], ignore_index=True)
                    except pd.errors.EmptyDataError:
                        print('Empty file:', _csv_path)
        _year_df.to_parquet(f'../binaries/mymarketnews_terminal_market_reports_{_year}.parquet')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Add origin state to our data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Define a function that reads origin string from market reports and determines origin state
    """)
    return


@app.cell
def _(re, us):
    def report_origin_state_search(origin_str):
        # Dict of states and abbreviations
        state_abbr_dict = {}
        for state in us.STATES:
            state_name = state.name.upper()
            state_abbr = state.abbr
            state_abbr_dict[state_name] = state_abbr

        # List of states and abbreviation
        state_abbr_list = list(state_abbr_dict.values())
        state_name_list = list(state_abbr_dict.keys())

        if origin_str is None:
            return []

        # Extracts a list of states from a single origin string
        origin_str_upper = origin_str.upper()
        found_states = set()

        # Strategy 1: Search for full state names (longest names first to avoid partial matches)
        # e.g., match "NORTH CAROLINA" before "CAROLINA"
        sorted_state_names = sorted(state_name_list, key=len, reverse=True)
        for state_name in sorted_state_names:
            # Use word boundaries to ensure whole word matching
            if re.search(r'\b' + re.escape(state_name) + r'\b', origin_str_upper):
                found_states.add(state_name)

        # Strategy 2: Search for state abbreviations (e.g., "CA", "AZ", "WV")
        # This is effective for patterns like "(MD, PA, VA, WV)" or "... CA, ... AZ"
        # We look for two consecutive uppercase letters that form a valid abbreviation
        potential_abbrs = re.findall(r'\b([A-Z]{2})\b', origin_str_upper)
        for abbr in potential_abbrs:
            if abbr in state_abbr_list:
                found_states.add(abbr)

        # We leave imports blank
        if re.search(r'\b' + re.escape('CROSSINGS') + r'\b', origin_str_upper):
            return([])

        if re.search(r'\b' + re.escape('IMPORTS') + r'\b', origin_str_upper):
            return([])

        # Return list of found states
        list_of_found_states = []
        for found_state in found_states:
            list_of_found_states.append(us.states.lookup(found_state).name)

        return(list_of_found_states)
    return (report_origin_state_search,)


@app.cell
def _(us):
    # Dict of state names with fips
    state_fips_dict = {}
    for state in us.STATES:
        state_name = state.name
        state_fips = state.fips
        state_fips_dict[state_name] = state_fips
    return


@app.cell
def _(pd, report_origin_state_search):
    # Add state of origin to Shipping Point reports
    for _year in range(2004, 2025):
        print('Doing', _year)
        _df = pd.read_parquet(f'../binaries/mymarketnews_shipping_point_reports_{_year}.parquet')
        _origins = _df['origin'].drop_duplicates(ignore_index=True)
        _origin_dict = {}
        for _origin in _origins:  # Create a dict mapping district names to states
            _origin_dict[_origin] = report_origin_state_search(_origin)
        _origin_dict['ARKANSAS VALLEY COLORADO'] = ['Colorado']
        _df['state_of_origin'] = _df['origin'].map(_origin_dict)
        _df.to_parquet(f'../binaries/mymarketnews_shipping_point_reports_{_year}.parquet')  # ARKANSAS VALLEY COLORADO refers only to Colorado as Arkansas Valley is a region in Colorado
    return


@app.cell
def _(pd, report_origin_state_search):
    for _year in range(2004, 2025):
        print('Doing', _year)
        _df = pd.read_parquet(f'../binaries/mymarketnews_terminal_market_reports_{_year}.parquet')
        _origins = _df['origin'].drop_duplicates(ignore_index=True)
        _origin_dict = {}
        for _origin in _origins:  # Create a dict mapping district names to states
            _origin_dict[_origin] = report_origin_state_search(_origin)
        _origin_dict['ARKANSAS VALLEY COLORADO'] = ['Colorado']
        _df['state_of_origin'] = _df['origin'].map(_origin_dict)
        _df.to_parquet(f'../binaries/mymarketnews_terminal_market_reports_{_year}.parquet')  # ARKANSAS VALLEY COLORADO refers only to Colorado as Arkansas Valley is a region in Colorado
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
