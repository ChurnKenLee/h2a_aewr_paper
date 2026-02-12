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
    from collections import defaultdict
    from bs4 import BeautifulSoup
    return addfips, json, pd, requests, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Scrape data from TRAC
    """)
    return


@app.cell
def _(requests):
    # Load table that has all state codes and put state codes into dict
    state_table = requests.get('https://tracreports.org/phptools/immigration/newdetain/table_json.php?stat=count&fy=All&dimension=LEA_state&sort=keyasc').json()
    state_keys = {}
    for _x in state_table['data']:
        state_keys[_x['label']] = _x['code']
    return


@app.cell
def _(requests):
    # Load table that has all the year codes and put year codes into dict
    year_table = requests.get('https://tracreports.org/phptools/immigration/newdetain/table_json.php?stat=count&LEA_state=All&dimension=fy&sort=keydesc').json()
    year_keys = {}
    for _x in year_table['data']:
        year_keys[_x['label']] = _x['code']
    return (year_keys,)


@app.cell
def _(requests, time, year_keys):
    fy_rows = []

    # Load county-facility table for each fiscal year, put response table into list
    for year, year_code in year_keys.items():
        county_fac_table = requests.get(f"https://tracreports.org/phptools/immigration/newdetain/table_json.php?stat=count&fy={year_code}&LEA_state=All&dimension=trac_fac_name_county&sort=keyasc").json()

        for entry in county_fac_table['data']:
            row = [year, entry['code'], entry['label'], entry['value']]
            fy_rows.append(row)

        time.sleep(1)
    return (fy_rows,)


@app.cell
def _(fy_rows, pd):
    # Put responses from list into dataframe
    detainer_counts = pd.DataFrame(fy_rows, columns = ['year', 'facility_code', 'facility_name', 'detainers_issued'])
    return (detainer_counts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Add FIPS code
    """)
    return


@app.cell
def _(detainer_counts):
    # Extract state and county name from facility name, to facilitate adding FIPS codes using the addFIPS package
    detainer_counts['county_name'] = detainer_counts['facility_name'].str.split(',').str[0]
    detainer_counts['county_name'] = detainer_counts['county_name'].astype(str)
    detainer_counts['state_abbrev'] = detainer_counts['facility_name'].str.extract(r'(,\s\D{2})')
    detainer_counts['state_abbrev'] = detainer_counts['state_abbrev'].astype(str) # Some entries don't have state names, i.e., aggregates
    detainer_counts['state_abbrev'] = detainer_counts['state_abbrev'].str.strip(', ')
    return


@app.cell
def _(addfips, detainer_counts):
    # Add FIPS code
    af = addfips.AddFIPS()
    detainer_counts['fips_from_addfips'] = detainer_counts.apply(lambda x: af.get_county_fips(_x['county_name'], state=_x['state_abbrev']), axis=1)
    return


@app.cell
def _(detainer_counts):
    # What are the unmatched locations?
    unmatched_df = detainer_counts[detainer_counts['fips_from_addfips'].isna()].copy()
    return (unmatched_df,)


@app.cell
def _(unmatched_df):
    unmatched_df.to_csv('test.csv')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use Google Find Places API to get county for the remaining facilities
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Start by finding the Place ID for each location using Find Place
    """)
    return


@app.cell
def _(unmatched_df):
    # For counties with 'None', these are for states that potentially don't have counties
    # Match using facility name instead
    unmatched_df['location_name'] = unmatched_df['county_name']
    unmatched_df.loc[unmatched_df['county_name']=='None', 'location_name'] = unmatched_df.loc[unmatched_df['county_name']=='None', 'facility_name']
    return


@app.cell
def _(unmatched_df):
    # Create ID for each row to link with API request responses
    unmatched_df['id'] = unmatched_df.reset_index().index.astype('str')
    return


@app.cell
def _(unmatched_df):
    # Split API calls into chunks of 100
    unmatched_df['chunk'] = unmatched_df['id'].astype(int)//100
    return


@app.cell
def _():
    # Google maps API key from my account
    # Import API key stored in text file
    with open("../tools/google_places_api_key.txt") as f:
        lines = f.readlines()

    api_key = lines[0]
    return


@app.cell
def _():
    # # Base url to call Find Place API
    # base_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json?"

    # for c in range(0, 8):
    #     unmatched_chunk = unmatched_df[unmatched_df['chunk'] == c]

    #     # Dict to store API responses
    #     api_placeid_dict = {}

    #     for ind in range(0, len(unmatched_chunk)):
    #         row = unmatched_chunk.iloc[ind]
    #         id = row['id']
    #         state_name = row['state_abbrev']
    #         place_name = row['location_name']
    #         name_to_search = place_name + ', ' + state_name

    #         print(id, name_to_search)

    #         # Create API request
    #         # URL'ed location name we want to search
    #         input = urllib.parse.quote(name_to_search) # Encode place name as URL string
    #         request_url = base_url + "input=" + input + "&inputtype=textquery" + "&fields=place_id" + "&key=" + api_key

    #         payload = {}
    #         headers = {}

    #         # Sleep one second between each API call
    #         time.sleep(1)

    #         # Make API call
    #         response = requests.request("GET", request_url, headers=headers, data=payload)
    #         response_json = response.json()
        
    #         # If API call is successful, then place response result into dict
    #         if response_json['status']=='OK':
    #             print('Successful')
    #             api_placeid_dict[id] = response_json
    #         else:
    #             # If API call is unsuccessful, then wait 5 seconds and retry
    #             print('NOT successful, retrying')
    #             time.sleep(5)
    #             response = requests.request("GET", request_url, headers=headers, data=payload)
    #             response_json = response.json()

    #             if response_json['status']=='OK':
    #                 print('Retry successful')
    #                 api_placeid_dict[id] = response_json
    #             else:
    #                 error_type = response_json['status']
    #                 print('Retry unsuccessful, error: ' + error_type)

    #     # Save API request results as JSON
    #     with open(f'json/trac_placeid_api_request_result_chunk_{c}.json', 'w') as f:
    #         json.dump(api_placeid_dict, f)
    return


@app.cell
def _(json):
    # Load JSON of API responses and put into DataFrame
    api_placeid_dict = {}
    for _c in range(0, 8):
        with open(f'json/trac_placeid_api_request_result_chunk_{_c}.json', 'r') as _infile:
            _api_dict = json.load(_infile)
        api_placeid_dict = api_placeid_dict | _api_dict
    return (api_placeid_dict,)


@app.cell
def _(api_placeid_dict, pd):
    # Put place IDs into DataFrame
    api_placeid_df = pd.DataFrame(columns=['id', 'placeid'])
    for id, _response in api_placeid_dict.items():
        number_of_candidates = len(_response['candidates'])
        for _response_ind in range(0, number_of_candidates):
            _placeid = _response['candidates'][_response_ind]['place_id']
            api_placeid_df.loc[len(api_placeid_df)] = [id, _placeid]
    return (api_placeid_df,)


@app.cell
def _(api_placeid_df):
    # Split API calls into chunks of 100
    api_placeid_df['chunk'] = api_placeid_df['id'].astype(int)//100
    return


@app.cell
def _():
    # # Use Place details API to get county names
    # base_url = 'https://maps.googleapis.com/maps/api/geocode/json?'

    # for c in range(0, 10):
    #     api_placeid_chunk = api_placeid_df[api_placeid_df['chunk'] == c]
    #     api_place_details_dict = {}

    #     # Iterate over each place ID
    #     for index, row in api_placeid_chunk.iterrows():
    #         print(row['id'], row['placeid'])

    #         # Create API request
    #         input = row['placeid']
    #         request_url = base_url + "place_id=" + input + "&key=" + api_key

    #         payload = {}
    #         headers = {}

    #         response = requests.request("GET", request_url, headers=headers, data=payload)
    #         response_json = response.json()

    #         # If API call is successful, then place response result into dict
    #         if response_json['status']=='OK':
    #             print('Successful')
    #             api_place_details_dict[input] = response_json
    #         else:
    #             # If API call is unsuccessful, then wait 5 seconds and retry
    #             print('NOT successful, retrying')
    #             time.sleep(5)
    #             response = requests.request("GET", request_url, headers=headers, data=payload)
    #             response_json = response.json()

    #             if response_json['status']=='OK':
    #                 print('Retry successful')
    #                 api_place_details_dict[input] = response_json
    #             else:
    #                 error_type = response_json['status']
    #                 print('Retry unsuccessful, error: ' + error_type)

    #     # Save API request results as JSON
    #     with open(f'json/trac_place_details_api_request_result_chunk_{c}.json', 'w') as f:
    #         json.dump(api_place_details_dict, f)
    return


@app.cell
def _(json):
    # Load JSON of API responses and put into DataFrame
    api_place_details_dict = {}
    for _c in range(0, 1):
        with open(f'json/trac_place_details_api_request_result_chunk_{_c}.json', 'r') as _infile:
            _api_dict = json.load(_infile)
        api_place_details_dict = api_place_details_dict | _api_dict
    return (api_place_details_dict,)


@app.cell
def _():
    # Store county name from place details into dictionary (store state names too as there may be incorrect states)
    county_name_dict = {}
    state_name_dict = {}
    return county_name_dict, state_name_dict


@app.cell
def _(api_place_details_dict, county_name_dict, state_name_dict):
    # Extract information we want from API response
    for _placeid, _response in api_place_details_dict.items():
        n_responses = len(_response['results'])
        for _response_ind in range(0, n_responses):
            individual_response = _response['results'][_response_ind]
            response_address_components_list = individual_response['address_components']
            n_components = len(response_address_components_list)
            for component_ind in range(0, n_components):
                component_dict = response_address_components_list[component_ind]
                component_type = component_dict['types'][0]
                if component_type == 'administrative_area_level_2':
                    county_name = component_dict['long_name']
                    county_name_dict[_placeid] = county_name
                if component_type == 'administrative_area_level_1':
                    state_name = component_dict['long_name']
                    state_name_dict[_placeid] = state_name
    return


@app.cell
def _(api_placeid_df, county_name_dict, state_name_dict):
    # Add county and state name columns to Place ID
    api_placeid_df['county_name_api'] = api_placeid_df['placeid'].map(county_name_dict)
    api_placeid_df['state_name_api'] = api_placeid_df['placeid'].map(state_name_dict)
    return


@app.cell
def _(api_placeid_df):
    # Some of these multiple responses per place name are in the same county, so we can collapse those
    api_placeid_df_1 = api_placeid_df.drop_duplicates(subset=['id', 'county_name_api', 'state_name_api'])
    return (api_placeid_df_1,)


@app.cell
def _(api_placeid_df_1, unmatched_df):
    # For the remainder, manually resolve
    api_placeid_df_2 = api_placeid_df_1.merge(unmatched_df[['state_abbrev', 'location_name', 'id']], how='left', on=['id'])
    return (api_placeid_df_2,)


@app.cell
def _(api_placeid_df_2):
    # These are unmatched independent cities
    api_placeid_df_2.loc[(api_placeid_df_2['location_name'] == 'Galax Police Department') & (api_placeid_df_2['state_abbrev'] == 'VA'), 'county_name_api'] = 'Galax'
    api_placeid_df_2.loc[(api_placeid_df_2['location_name'] == 'Lynchburg Pol Dept') & (api_placeid_df_2['state_abbrev'] == 'VA'), 'county_name_api'] = 'Lynchburg'
    api_placeid_df_2.loc[(api_placeid_df_2['location_name'] == 'City of Baltimore') & (api_placeid_df_2['state_abbrev'] == 'MD'), 'county_name_api'] = 'Baltimore'
    return


@app.cell
def _(api_placeid_df_2):
    # Recollapse after fixing
    api_placeid_df_3 = api_placeid_df_2.drop_duplicates(subset=['id', 'county_name_api', 'state_abbrev'])
    return (api_placeid_df_3,)


@app.cell
def _(addfips, api_placeid_df_3):
    af_1 = addfips.AddFIPS()
    api_placeid_df_4 = api_placeid_df_3[~api_placeid_df_3['county_name_api'].isna()].copy()
    api_placeid_df_4['fips_api'] = api_placeid_df_4.apply(lambda x: af_1.get_county_fips(_x['county_name_api'], state=_x['state_abbrev']), axis=1)
    return af_1, api_placeid_df_4


@app.cell
def _(api_placeid_df_4, unmatched_df):
    trac_api_df = unmatched_df.merge(api_placeid_df_4[['id', 'county_name_api', 'fips_api']], how='left', on=['id'])
    trac_api_df = trac_api_df[~trac_api_df['fips_api'].isna()].copy()
    trac_api_df = trac_api_df.drop_duplicates(subset=['state_abbrev', 'facility_code', 'fips_api'])
    trac_api_df = trac_api_df.groupby(['state_abbrev', 'facility_code']).agg({'fips_api': lambda x: ','.join(_x)}).reset_index()
    return (trac_api_df,)


@app.cell
def _(detainer_counts, trac_api_df):
    # Add FIPS from API back to original list of worksites
    county_df = detainer_counts.merge(trac_api_df, how = 'left', on = ['state_abbrev', 'facility_code'])
    return (county_df,)


@app.cell
def _(county_df):
    # Clean up
    county_df_1 = county_df.fillna(value='')
    county_df_1['fips'] = county_df_1['fips_from_addfips']
    county_df_1.loc[county_df_1['fips'] == '', 'fips'] = county_df_1.loc[county_df_1['fips'] == '', 'fips_api']
    return (county_df_1,)


@app.cell
def _(af_1, county_df_1):
    # There are a few more entries we can manually resolve
    # Brookhaven PD in Delaware County is in Pennsylvania, not Georgia
    county_df_1.loc[(county_df_1['facility_code'] == '350') & (county_df_1['state_abbrev'] == 'PA'), 'fips'] = af_1.get_county_fips('Delaware County', 'Pennsylvania')
    county_df_1.loc[(county_df_1['facility_code'] == '2984') & (county_df_1['state_abbrev'] == 'MS'), 'fips'] = af_1.get_county_fips('Kemper County', 'Mississippi')
    # Kemper-Neshoba Regional Correctional Facility, Kemper County is in Mississippi, not Michigan
    # Suffolk County - Ny State Police Farmingdale is in New York, not Arkansas
    # The rest are in U.S. Territories
    county_df_1.loc[(county_df_1['facility_code'] == '3260') & (county_df_1['state_abbrev'] == 'NY'), 'fips'] = af_1.get_county_fips('Suffolk County', 'New York')
    return


@app.cell
def _(county_df_1):
    # Export binary
    county_df_2 = county_df_1.drop(columns=['fips_from_addfips', 'fips_api'])
    county_df_2 = county_df_2.astype(str)
    county_df_2.to_parquet('../binaries/trac_detainer_counts.parquet', index=False)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
