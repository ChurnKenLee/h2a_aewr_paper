import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


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
    return Path, addfips, json, pd, pickle, rapidfuzz, us


@app.cell
def _(pd):
    # Create Census geographic codes file
    census_county = pd.read_csv("../Data/census_geography_codes/national_county2020.txt", sep='|', dtype = 'string', keep_default_na=False).apply(lambda x: x.str.upper())
    census_countysub = pd.read_csv("../Data/census_geography_codes/national_cousub2020.txt", sep='|', dtype = 'string', keep_default_na=False).apply(lambda x: x.str.upper())
    census_place = pd.read_csv("../Data/census_geography_codes/national_place2020.txt", sep='|', dtype = 'string', keep_default_na=False).apply(lambda x: x.str.upper())
    census_placebycounty = pd.read_csv("../Data/census_geography_codes/national_place_by_county2020.txt", sep='|', dtype = 'string', keep_default_na=False).apply(lambda x: x.str.upper())
    census_zip = pd.read_csv("../Data/census_geography_codes/tab20_zcta520_county20_natl.txt", sep='|', dtype = 'string', keep_default_na=False).apply(lambda x: x.str.upper())

    # Add FIPS column
    census_county['fips'] = census_county['STATEFP'] + census_county['COUNTYFP']
    census_placebycounty['fips'] = census_placebycounty['STATEFP'] + census_placebycounty['COUNTYFP']
    census_zip['fips'] = census_zip['GEOID_COUNTY_20']

    # There may be places and ZIP codes that map to multiple counties; collapse these into unique entries
    census_county = census_county[['STATE', 'COUNTYNAME', 'fips']]
    census_place_agg = census_placebycounty.groupby(['STATE', 'COUNTYNAME', 'PLACENAME']).agg({'fips':lambda x: ",".join(x)}).reset_index()
    census_zip_agg = census_zip.groupby(['GEOID_ZCTA5_20']).agg({'fips':lambda x: ",".join(x)}).reset_index()

    # Drop empty entries
    census_county = census_county[census_county['COUNTYNAME'] != '']
    census_place_agg = census_place_agg[census_place_agg['PLACENAME'] != '']
    census_zip_agg = census_zip_agg[census_zip_agg['GEOID_ZCTA5_20'] != '']
    census_zip_agg = census_zip_agg.rename(columns = {'GEOID_ZCTA5_20':'zip', 'fips':'fips_from_zip'})
    return census_placebycounty, census_zip_agg


@app.cell
def _():
    # Names of H-2A program Addendum B files from the DOL-OFLC
    h2a_file_name_dict = {
        '2020':'H-2A_FY2020_AddendumB_Employment.xlsx',
        '2021':'H-2A_Addendum_B_Employment_FY2021.xlsx',
        '2022':'H-2A_Addendum_B_Employment_Record_FY2022_Q4.xlsx'
    }
    return (h2a_file_name_dict,)


@app.cell
def _():
    # Define common set of variables we want from every fiscal year, and their types
    h2a_dtype_dict = {}

    h2a_dtype_dict['2020'] = {
        'CASE_NUMBER':'string',
        'NAME_OF_AGRICULTURAL_BUSINESS':'string',
        'PLACE_OF_EMPLOYMENT_CITY':'string',
        'PLACE_OF_EMPLOYMENT_STATE':'string',
        'PLACE_OF_EMPLOYMENT_POSTAL_CODE':'string',
        'TOTAL_WORKERS':'string'
    }

    h2a_dtype_dict['2021'] = h2a_dtype_dict['2020']
    h2a_dtype_dict['2022'] = h2a_dtype_dict['2020']
    return (h2a_dtype_dict,)


@app.cell
def _():
    # Define set of common names for concatenating
    h2a_rename_dict = {
        'CASE_NUMBER':'case_number',
        'NAME_OF_AGRICULTURAL_BUSINESS':'business_name',
        'PLACE_OF_EMPLOYMENT_CITY':'worksite_city',
        'PLACE_OF_EMPLOYMENT_STATE':'worksite_state',
        'PLACE_OF_EMPLOYMENT_POSTAL_CODE':'worksite_zip',
        'TOTAL_WORKERS':'total_h2a_workers_requested'
    }
    return (h2a_rename_dict,)


@app.cell
def _(Path, h2a_dtype_dict, h2a_file_name_dict, pd, pickle):
    h2a_df_dict = {}
    for year, file_name in h2a_file_name_dict.items():
        h2a_path = Path(f"../Data/h2a/{file_name}")
        print(h2a_path)

        dtype_dict = h2a_dtype_dict[year]
        col_list = list(dtype_dict.keys())
        h2a_df_dict[year] = pd.read_excel(h2a_path, usecols = col_list, dtype = dtype_dict, parse_dates=False)

    # Pickling
    with open("h2a_addendum_b_pickle", "wb") as fp:
        pickle.dump(h2a_df_dict, fp)
    return


@app.cell
def _(h2a_rename_dict, pd, pickle):
    # Unpickling
    with open("h2a_addendum_b_pickle", "rb") as fp:
        h2a_df_dict = pickle.load(fp)

    h2a_df = pd.DataFrame()
    for year, df in h2a_df_dict.items():
        df = df.rename(columns = h2a_rename_dict)
        df['fiscal_year'] = year
        h2a_df = pd.concat([h2a_df, df])
    return (h2a_df,)


@app.cell
def _(h2a_df):
    # Define consistent NAs, convert all entries to uppercase
    h2a_df_1 = h2a_df.fillna(value='').apply(lambda x: x.str.upper())
    h2a_df_1['worksite_zip'] = h2a_df_1['worksite_zip'].str.replace(pat=' ', repl='')
    return (h2a_df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Match FIPS codes for worksite locations
    """)
    return


@app.cell
def _(h2a_df_1):
    h2a_worksite_locations = h2a_df_1[['worksite_city', 'worksite_state', 'worksite_zip']]
    h2a_worksite_locations = h2a_worksite_locations.drop_duplicates()
    # Drop duplicated entries
    h2a_worksite_locations['city'] = h2a_worksite_locations['worksite_city']
    h2a_worksite_locations['zip'] = h2a_worksite_locations['worksite_zip']
    return (h2a_worksite_locations,)


@app.cell
def _(h2a_worksite_locations):
    # Clean up ZIP codes
    h2a_worksite_locations.loc[((h2a_worksite_locations['zip'].str.len() < 5) & (h2a_worksite_locations['zip'] != '')), 'zip'] = h2a_worksite_locations.loc[((h2a_worksite_locations['zip'].str.len() < 5) & (h2a_worksite_locations['zip'] != '')), 'zip'].str.pad(width = 5, side = 'left', fillchar = '0')
    h2a_worksite_locations.loc[(h2a_worksite_locations['zip'].str.len() > 5), 'zip'] = h2a_worksite_locations.loc[(h2a_worksite_locations['zip'].str.len() > 5), 'zip'].str.slice(start = 0, stop = 5)
    return


@app.cell
def _(census_zip_agg, h2a_worksite_locations, us):
    # Add state abbreviation where possible
    h2a_worksite_locations['state'] = h2a_worksite_locations['worksite_state'].apply(lambda x: us.states.lookup(x).abbr if (x != '') & (x != 'FEDERATED STATES OF MICRONESIA') else '')
    h2a_worksite_locations_1 = h2a_worksite_locations.merge(census_zip_agg, how='left', on=['zip'])
    # For entries without state but has ZIP code, add state that way
    h2a_worksite_locations_1['state_fip_from_zip'] = h2a_worksite_locations_1['fips_from_zip'].str.slice(start=0, stop=2)
    h2a_worksite_locations_1.loc[h2a_worksite_locations_1['state_fip_from_zip'].isna(), 'state_fip_from_zip'] = ''
    h2a_worksite_locations_1['state_from_zip'] = h2a_worksite_locations_1['state_fip_from_zip'].apply(lambda x: us.states.lookup(x).abbr if x != '' else '')
    h2a_worksite_locations_1.loc[h2a_worksite_locations_1['state'] == '', 'state'] = h2a_worksite_locations_1.loc[h2a_worksite_locations_1['state'] == '', 'state_from_zip']
    return (h2a_worksite_locations_1,)


@app.cell
def _(rapidfuzz):
    # Define function for fuzzy string matching
    def fuzz_search(census_df, census_col, state_to_search, name_to_match):

        def fuzz_match(x, y):
            return rapidfuzz.fuzz.WRatio(x, y)

        state_df = census_df[census_df['STATE'] == state_to_search].copy()
        state_df['score'] = state_df[census_col].apply(lambda x: fuzz_match(x, name_to_match))

        state_df = state_df.sort_values('score')

        max_score_row = state_df[state_df['score'] == state_df['score'].max()].reset_index()

        # Best match
        if len(max_score_row) >= 1:
            fips = str(max_score_row['fips'][0])
            score = str(max_score_row['score'][0])
            census_name = (max_score_row[census_col][0])
            return(fips, score, census_name)
        else:
            return('', '', '')
    return (fuzz_search,)


@app.cell
def _(census_placebycounty, fuzz_search, h2a_worksite_locations_1, pd):
    # Get matches and match score using city
    census_df = census_placebycounty
    census_col = 'PLACENAME'
    fuzzy_result_df = h2a_worksite_locations_1.apply(lambda x: fuzz_search(census_df, census_col, x.state, x.city), axis=1, result_type='expand')
    fuzzy_result_df = fuzzy_result_df.rename(columns={0: 'fips_from_city', 1: 'score_from_city', 2: 'census_name_city'})
    fuzzy_result_df['score_from_city'] = pd.to_numeric(fuzzy_result_df['score_from_city'], errors='coerce')
    return (fuzzy_result_df,)


@app.cell
def _(fuzzy_result_df):
    # It appears 85.5 is a good cutoff
    fuzzy_result_df.loc[fuzzy_result_df['score_from_city'] < 85.6, ['fips_from_city']] = ''
    return


@app.cell
def _(fuzzy_result_df, h2a_worksite_locations_1, pd):
    # Combine match list back in
    h2a_worksite_locations_2 = pd.concat([h2a_worksite_locations_1, fuzzy_result_df], axis=1)
    return (h2a_worksite_locations_2,)


@app.cell
def _(h2a_worksite_locations_2):
    # Create combined FIPS, and use Google Places API for the rest
    # Write function with logic for choosing FIPS
    def fips_choice(county_fips, zip_fips, city_fips):
        if zip_fips != '':
            return zip_fips
        if county_fips != '':
            return county_fips
        if city_fips != '':
            return city_fips
        else:
            return ''
    h2a_worksite_locations_3 = h2a_worksite_locations_2.fillna('')
    h2a_worksite_locations_3['fips_from_census'] = h2a_worksite_locations_3.apply(lambda x: fips_choice('', x.fips_from_zip, x.fips_from_city), axis=1)
    return (h2a_worksite_locations_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the remaining locations, find county using Google's Places API
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Start by finding the Place ID for each location using Find Place
    """)
    return


@app.cell
def _(h2a_worksite_locations_3):
    h2a_unmatched_census = h2a_worksite_locations_3[(h2a_worksite_locations_3['fips_from_census'] == '') & (h2a_worksite_locations_3['worksite_city'] != '') & (h2a_worksite_locations_3['worksite_state'] != '')][['worksite_city', 'worksite_state', 'worksite_zip', 'city', 'state', 'zip']]
    h2a_unmatched_census['state_name'] = h2a_unmatched_census['worksite_state']
    return (h2a_unmatched_census,)


@app.cell
def _(h2a_unmatched_census):
    # Create ID for each row to link with API request responses
    h2a_unmatched_census['id'] = h2a_unmatched_census.reset_index().index.astype('str')
    return


@app.cell
def _(h2a_unmatched_census):
    # Split API calls into chunks of 100
    h2a_unmatched_census['chunk'] = h2a_unmatched_census['id'].astype(int)//100
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
    #     h2a_chunk = h2a_unmatched_census[h2a_unmatched_census['chunk'] == c]

    #     # Dict to store API responses
    #     api_placeid_dict = {}

    #     for ind in range(0, len(h2a_chunk)):
    #         row = h2a_chunk.iloc[ind]
    #         id = row['id']
    #         state_name = row['state_name']
    #         place_name = row['city']
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
    #     with open(f'json/addendum_b_placeid_api_request_result_chunk_{c}.json', 'w') as f:
    #         json.dump(api_placeid_dict, f)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now use the Place ID to find the county name of each location
    """)
    return


@app.cell
def _(json):
    # Load JSON of API responses and put into DataFrame
    api_placeid_dict = {}
    for _c in range(0, 8):
        with open(f'json/addendum_b_placeid_api_request_result_chunk_{_c}.json', 'r') as _infile:
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
    #     with open(f'json/addendum_b_place_details_api_request_result_chunk_{c}.json', 'w') as f:
    #         json.dump(api_place_details_dict, f)
    return


@app.cell
def _(json):
    # Load JSON of API responses and put into DataFrame
    api_place_details_dict = {}
    for _c in range(0, 10):
        with open(f'json/addendum_b_place_details_api_request_result_chunk_{_c}.json', 'r') as _infile:
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
def _(api_placeid_df_1, h2a_unmatched_census):
    # For the remainder, manually resolve
    api_placeid_df_2 = api_placeid_df_1.merge(h2a_unmatched_census[['city', 'state_name', 'id']], how='left', on=['id'])
    multiple_response = api_placeid_df_2[api_placeid_df_2.duplicated(subset=['id'], keep=False)]
    multiple_response.to_csv('test.csv')
    return (api_placeid_df_2,)


@app.cell
def _(api_placeid_df_2):
    api_placeid_df_2.loc[(api_placeid_df_2['city'] == 'LA SELLE') & (api_placeid_df_2['state_name'] == 'ILLINOIS'), 'county_name_api'] = 'LaSalle County'
    api_placeid_df_2.loc[(api_placeid_df_2['city'] == 'HARRISON TWP') & (api_placeid_df_2['state_name'] == 'INDIANA'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['city'] == 'HARRISON TWP') & (api_placeid_df_2['state_name'] == 'INDIANA'), 'county_name_api'] = None  # Ambiguous
    api_placeid_df_2.loc[(api_placeid_df_2['city'] == 'TOWNSHIP') & (api_placeid_df_2['state_name'] == 'ILLINOIS'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['city'] == 'OWNSHIP') & (api_placeid_df_2['state_name'] == 'ILLINOIS'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['city'] == 'JORDAN/LIBERTY') & (api_placeid_df_2['state_name'] == 'INDIANA'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['city'] == 'PHOENIX AND SURROUNDING CITIES ') & (api_placeid_df_2['state_name'] == 'ARIZONA'), 'county_name_api'] = 'Maricopa County'
    api_placeid_df_2.loc[(api_placeid_df_2['city'] == 'BROOKFIELD TWPS') & (api_placeid_df_2['state_name'] == 'MICHIGAN'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['city'] == 'FAIRHAVEN TWPS') & (api_placeid_df_2['state_name'] == 'MICHIGAN'), 'county_name_api'] = 'Huron County'
    api_placeid_df_2.loc[(api_placeid_df_2['city'] == 'CYPRESS') & (api_placeid_df_2['state_name'] == 'SOUTH CAROLINA'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['city'] == 'CRYSTAL TWSP') & (api_placeid_df_2['state_name'] == 'MICHIGAN'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['city'] == 'LEROY TWSP') & (api_placeid_df_2['state_name'] == 'MICHIGAN'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['city'] == 'GREGREENFIELD') & (api_placeid_df_2['state_name'] == 'CALIFORNIA'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['city'] == 'NONE') & (api_placeid_df_2['state_name'] == 'SOUTH DAKOTA'), 'county_name_api'] = None
    return


@app.cell
def _(api_placeid_df_2):
    # Recollapse after fixing
    api_placeid_df_3 = api_placeid_df_2.drop_duplicates(subset=['id', 'county_name_api', 'state_name'])
    return (api_placeid_df_3,)


@app.cell
def _(addfips, api_placeid_df_3):
    # Get FIPS codes using addFIPS
    af = addfips.AddFIPS()
    api_placeid_df_4 = api_placeid_df_3[~api_placeid_df_3['county_name_api'].isna()].copy()
    api_placeid_df_4['fips_api'] = api_placeid_df_4.apply(lambda x: af.get_county_fips(x['county_name_api'], state=x['state_name']), axis=1)
    return (api_placeid_df_4,)


@app.cell
def _(api_placeid_df_4):
    # Drop API results that don't match states
    api_placeid_df_5 = api_placeid_df_4[api_placeid_df_4['state_name'] == api_placeid_df_4['state_name_api'].str.upper()]
    return (api_placeid_df_5,)


@app.cell
def _(api_placeid_df_5, h2a_unmatched_census):
    # Recollapse back into individual entries (some entries had multiple places per entry)
    h2a_api_df = h2a_unmatched_census.merge(api_placeid_df_5[['id', 'county_name_api', 'fips_api']], how='left', on=['id'])
    h2a_api_df = h2a_api_df[~h2a_api_df['fips_api'].isna()].copy()
    h2a_api_df = h2a_api_df.groupby(['worksite_city', 'worksite_state', 'worksite_zip']).agg({'fips_api': lambda x: ','.join(x)}).reset_index()
    return (h2a_api_df,)


@app.cell
def _(h2a_api_df, h2a_worksite_locations_3):
    # Add FIPS from API back to original list of worksites
    h2a_worksite_locations_4 = h2a_worksite_locations_3.merge(h2a_api_df, how='left', on=['worksite_city', 'worksite_state', 'worksite_zip'])
    return (h2a_worksite_locations_4,)


@app.cell
def _(h2a_worksite_locations_4):
    # Clean up
    h2a_worksite_locations_5 = h2a_worksite_locations_4.fillna(value='')
    h2a_worksite_locations_5['fips'] = h2a_worksite_locations_5['fips_from_census']
    h2a_worksite_locations_5.loc[h2a_worksite_locations_5['fips'] == '', 'fips'] = h2a_worksite_locations_5.loc[h2a_worksite_locations_5['fips'] == '', 'fips_api']
    return (h2a_worksite_locations_5,)


@app.cell
def _(h2a_df_1, h2a_worksite_locations_5):
    # Add FIPS to H-2A entries based on worksites
    h2a_df_final = h2a_df_1.merge(h2a_worksite_locations_5[['worksite_city', 'worksite_state', 'worksite_zip', 'fips']], how='left', on=['worksite_city', 'worksite_state', 'worksite_zip'])
    return (h2a_df_final,)


@app.cell
def _(h2a_df_final):
    # Export binary
    h2a_df_final.to_parquet("../binaries/h2a_addendum_b_with_fips.parquet", index=False)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
