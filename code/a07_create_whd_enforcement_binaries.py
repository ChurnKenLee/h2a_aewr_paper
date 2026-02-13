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
    import rapidfuzz
    from rapidfuzz import fuzz
    from collections import defaultdict
    return addfips, json, pd, rapidfuzz, us


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
    # Columns from DOL WHD enforcement data we want
    whd_dtype_dict = {
        'case_id': 'string',
        'trade_nm': 'string',
        'legal_name': 'string',
        'street_addr_1_txt': 'string',
        'cty_nm': 'string',
        'st_cd': 'string',
        'zip_cd': 'string',
        'naic_cd': 'string',
        'findings_start_date': 'string',
        'findings_end_date': 'string',
        'h2a_violtn_cnt': 'float',
        'h2a_bw_atp_amt': 'float',
        'h2a_ee_atp_cnt': 'float',
        'h2a_cmp_assd_amt': 'float',
        'mspa_violtn_cnt': 'float',
        'mspa_bw_atp_amt': 'float',
        'mspa_ee_atp_cnt': 'float',
        'mspa_cmp_assd_amt': 'float'
    }

    whd_cols_list = list(whd_dtype_dict.keys())
    return whd_cols_list, whd_dtype_dict


@app.cell
def _(pd, whd_cols_list, whd_dtype_dict):
    # DOL WHD enforcement data
    whd_df = pd.read_csv("../Data/whd_enforcement/whd_whisard.csv", usecols = whd_cols_list, dtype = whd_dtype_dict)
    return (whd_df,)


@app.cell
def _(whd_df):
    # We only care about H-2A violations
    whd_df_1 = whd_df[whd_df['h2a_violtn_cnt'] != 0]
    return (whd_df_1,)


@app.cell
def _(whd_df_1):
    # We want to add county FIPS codes to each entry
    # We have city names and ZIP codes
    # Get list of city names, states, and ZIP codes
    whd_df_1['city'] = whd_df_1['cty_nm'].str.upper()
    whd_df_1['zip'] = whd_df_1['zip_cd']
    whd_df_1['state'] = whd_df_1['st_cd']
    worksite_df = whd_df_1[['city', 'zip', 'state']]
    worksite_df = worksite_df.drop_duplicates()
    return (worksite_df,)


@app.cell
def _(worksite_df):
    # Clean up ZIP codes
    # There are some entries with ZIP codes shorter than 5; pad with zeros
    worksite_df.loc[((worksite_df['zip'].str.len() < 5) & (worksite_df['zip'] != '')), 'zip'] = worksite_df.loc[((worksite_df['zip'].str.len() < 5) & (worksite_df['zip'] != '')), 'zip'].str.pad(width = 5, side = 'left', fillchar = '0')
    return


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
def _(census_placebycounty, fuzz_search, pd, worksite_df):
    # Get matches and match score
    census_df = census_placebycounty
    census_col = 'PLACENAME'
    fuzzy_result_df = worksite_df.apply(lambda x: fuzz_search(census_df, census_col, x.state, x.city), axis=1, result_type='expand')
    fuzzy_result_df = fuzzy_result_df.rename(columns = {0:'fips_from_city', 1:'score_from_city', 2:'census_name_city'})
    fuzzy_result_df['score_from_city'] = pd.to_numeric(fuzzy_result_df['score_from_city'], errors='coerce')
    return (fuzzy_result_df,)


@app.cell
def _(fuzzy_result_df, worksite_df):
    # Check quality of matches, and define cutoff score
    match_quality = worksite_df.merge(fuzzy_result_df, left_index=True, right_index=True)
    match_quality.to_csv('test.csv')
    return


@app.cell
def _(fuzzy_result_df):
    # 86 appears to be a good cutoff score
    fuzzy_result_df.loc[fuzzy_result_df['score_from_city'] < 86, ['fips_from_city']] = ''
    return


@app.cell
def _(fuzzy_result_df, worksite_df):
    # Combine matched cities back in
    worksite_df_1 = worksite_df.merge(fuzzy_result_df, left_index=True, right_index=True)
    return (worksite_df_1,)


@app.cell
def _(census_zip_agg, worksite_df_1):
    # Add FIPS from matched ZIP codes as well
    worksite_df_2 = worksite_df_1.merge(census_zip_agg, how='left')
    return (worksite_df_2,)


@app.cell
def _(worksite_df_2):
    # Create combined FIPS, and use Google Places API for the rest
    # Write function with logic for choosing FIPS
    def fips_choice(county_fips, zip_fips, city_fips):
        if zip_fips != '':
            return zip_fips
        if city_fips != '':
            return city_fips
        if county_fips != '':
            return county_fips
        else:
            return ''
    worksite_df_3 = worksite_df_2.fillna('')
    worksite_df_3['fips_from_census'] = worksite_df_3.apply(lambda x: fips_choice('', x.fips_from_zip, x.fips_from_city), axis=1)
    return (worksite_df_3,)


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
def _(us, worksite_df_3):
    # These are the entries we need to use the API for
    whd_unmatched = worksite_df_3[(worksite_df_3['fips_from_census'] == '') & (worksite_df_3['city'] != '') & (worksite_df_3['state'] != '')][['city', 'state', 'zip']]
    whd_unmatched['state_name'] = whd_unmatched['state'].apply(lambda x: us.states.lookup(x).name)
    return (whd_unmatched,)


@app.cell
def _(whd_unmatched):
    # Create ID for each row to link with API request responses
    whd_unmatched['id'] = whd_unmatched.reset_index().index.astype('str')
    return


@app.cell
def _(whd_unmatched):
    # Split API calls into chunks of 100
    whd_unmatched['chunk'] = whd_unmatched['id'].astype(int)//100
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

    # for c in range(0, 1):
    #     whd_chunk = whd_unmatched[whd_unmatched['chunk'] == c]

    #     # Dict to store API responses
    #     api_placeid_dict = {}

    #     for ind in range(0, len(whd_chunk)):
    #         row = whd_chunk.iloc[ind]
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
    #     with open(f'json/whd_placeid_api_request_result_chunk_{c}.json', 'w') as f:
    #         json.dump(api_placeid_dict, f)
    return


@app.cell
def _(json):
    # Load JSON of API responses and put into DataFrame
    api_placeid_dict = {}
    for _c in range(0, 1):
        with open(f'json/whd_placeid_api_request_result_chunk_{_c}.json', 'r') as _infile:
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

    # for c in range(0, 1):
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
    #     with open(f'json/whd_place_details_api_request_result_chunk_{c}.json', 'w') as f:
    #         json.dump(api_place_details_dict, f)
    return


@app.cell
def _(json):
    # Load JSON of API responses and put into DataFrame
    api_place_details_dict = {}
    for _c in range(0, 1):
        with open(f'json/whd_place_details_api_request_result_chunk_{_c}.json', 'r') as _infile:
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
def _(api_placeid_df_1, whd_unmatched):
    # For the remainder, manually resolve
    api_placeid_df_2 = api_placeid_df_1.merge(whd_unmatched[['city', 'state_name', 'id']], how='left', on=['id'])
    multiple_response = api_placeid_df_2[api_placeid_df_2.duplicated(subset=['id'], keep=False)]
    multiple_response.to_csv('test.csv')
    return (api_placeid_df_2,)


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
    api_placeid_df_5 = api_placeid_df_4[api_placeid_df_4['state_name'] == api_placeid_df_4['state_name_api']]
    return (api_placeid_df_5,)


@app.cell
def _(api_placeid_df_5, whd_unmatched):
    # Recollapse back into individual entries (some entries had multiple places per entry)
    whd_api_df = whd_unmatched.merge(api_placeid_df_5[['id', 'county_name_api', 'fips_api']], how='left', on=['id'])
    whd_api_df = whd_api_df[~whd_api_df['fips_api'].isna()].copy()
    whd_api_df = whd_api_df.groupby(['city', 'state', 'zip']).agg({'fips_api': lambda x: ','.join(x)}).reset_index()
    return (whd_api_df,)


@app.cell
def _(whd_api_df, worksite_df_3):
    # Add FIPS from API back to original list of worksites
    worksite_df_4 = worksite_df_3.merge(whd_api_df, how='left', on=['city', 'state', 'zip'])
    return (worksite_df_4,)


@app.cell
def _(worksite_df_4):
    # Clean up
    worksite_df_5 = worksite_df_4.fillna(value='')
    worksite_df_5['fips'] = worksite_df_5['fips_from_census']
    worksite_df_5.loc[worksite_df_5['fips'] == '', 'fips'] = worksite_df_5.loc[worksite_df_5['fips'] == '', 'fips_api']
    return (worksite_df_5,)


@app.cell
def _(whd_df_1, worksite_df_5):
    # Add FIPS to H-2A entries based on worksites
    whd_final = whd_df_1.merge(worksite_df_5[['city', 'state', 'zip', 'fips']], how='left', on=['city', 'state', 'zip'])
    return (whd_final,)


@app.cell
def _(whd_final):
    # Export binary
    whd_final.to_parquet("../binaries/whd_with_fips.parquet", index=False)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
