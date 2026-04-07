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
    return addfips, json, np, pd


@app.cell
def _():
    # Variable list and types we want from H-2A application data
    h2a_dtype_dict = {
        'FiscalYear': 'string',
        'Matched': 'category',
        'WORKSITE_STATE': 'category',
        'WORKSITE_CITY': 'string',
        'WORKSITE_COUNTY': 'string',
        'ORIGINAL_WORKSITE_COUNTY': 'string',
        'WORKSITE_POSTAL_CODE': 'string',
        'EMPLOYER_NAME': 'string',
        'new_EMPLOYER_NAME': 'string',
        'CASE_NUMBER': 'string',
        'CASE_STATUS': 'category',
        'EMPLOYER_POSTAL_CODE': 'string',
        'JOB_TITLE': 'string',
        'NBR_WORKERS_REQUESTED': 'float32',
        'NBR_WORKERS_CERTIFIED': 'float32',
        'BASIC_NUMBER_OF_HOURS': 'float32',
        'BASIC_RATE_OF_PAY': 'float32',
        'BASIC_UNIT_OF_PAY': 'category',
        'CERTIFICATION_BEGIN_DATE': 'string',
        'CERTIFICATION_END_DATE': 'string',
        'DECISION_DATE': 'string',
        'CASE_RECEIVED_DATE': 'string',
        'ORGANIZATION_FLAG': 'category',
        'REQUESTED_START_DATE_OF_NEED': 'string',
        'REQUESTED_END_DATE_OF_NEED': 'string',
        'PRIMARY_CROP': 'string',
        'LAWFIRM_NAME': 'string',
        'SOC_CODE': 'category',
        'SOC_TITLE': 'category',
        'NAICS_CODE': 'category',
        'FULL_TIME_POSITION': 'category',
        'NATURE_OF_TEMPORARY_NEED': 'category',
        'OVERTIME_RATE_FROM': 'float32',
        'OVERTIME_RATE_TO': 'float32',
        'EDUCATION_LEVEL': 'category',
        'OTHER_EDU': 'category',
        'SWA_NAME': 'string',
        'JOB_IDNUMBER': 'string',
        'JOB_START_DATE': 'string',
        'JOB_END_DATE': 'string',
        'TRADE_NAME_DBA': 'string',
        'TYPE_OF_EMPLOYER_APPLICATION': 'category',
        'JOB_ORDER_NUMBER': 'string',
        'TOTAL_WORKERS_NEEDED': 'float32',
        'SPECIAL_REQUIREMENTS': 'string',
        'begin_year': 'string',
        'end_year': 'string',
        'case_year': 'string',
        'Abbreviation': 'category',
        'EMPLOYER_STATE': 'category',
        'EMPLOYER_CITY': 'string'
    }

    col_list = list(h2a_dtype_dict.keys())
    return


@app.cell
def _():
    small_rename_dict = {
        'EMPLOYMENT_BEGIN_DATE': 'JOB_START_DATE',
        'EMPLOYMENT_END_DATE': 'JOB_END_DATE',
        'REQUESTED_BEGIN_DATE': 'REQUESTED_START_DATE_OF_NEED',
        'REQUESTED_END_DATE': 'REQUESTED_END_DATE_OF_NEED',
        'TOTAL_WORKERS_H2A_REQUESTED': 'NBR_WORKERS_REQUESTED',
        'TOTAL_WORKERS_H2A_CERTIFIED': 'NBR_WORKERS_CERTIFIED'
    }

    small_h2a_dtype_dict = {
        'CASE_NUMBER': 'string',
        'CASE_STATUS': 'category',
        'TOTAL_WORKERS_NEEDED': 'float32',
        'TOTAL_WORKERS_H2A_REQUESTED': 'float32',
        'TOTAL_WORKERS_H2A_CERTIFIED': 'float32',
        'ANTICIPATED_NUMBER_OF_HOURS': 'float32',
        'REQUESTED_BEGIN_DATE': 'string',
        'REQUESTED_END_DATE': 'string',
        'EMPLOYMENT_BEGIN_DATE': 'string',
        'EMPLOYMENT_END_DATE': 'string',
        'WORKSITE_STATE': 'category',
        'WORKSITE_CITY': 'string',
        'WORKSITE_COUNTY': 'string',
        'WORKSITE_POSTAL_CODE': 'string'
    }

    small_col_list = list(small_h2a_dtype_dict.keys())
    return small_col_list, small_h2a_dtype_dict, small_rename_dict


@app.cell
def _():
    # # Convert from xlsx to parquet
    # df = pd.read_excel("../Data/h2a/H2A Final Worksite County Data.xlsx", usecols=col_list, dtype=h2a_dtype_dict, parse_dates=False)
    # df.to_parquet("../binaries/h2a_raw.parquet")
    return


@app.cell
def _(pd, small_col_list, small_h2a_dtype_dict):
    # Check 2021 and 2022 numbers
    df_2021 = pd.read_excel("../Data/h2a/H-2A_Disclosure_Data_FY2021.xlsx", usecols=small_col_list, dtype=small_h2a_dtype_dict, parse_dates=False)
    df_2021['FiscalYear'] = '2021'
    df_2022 = pd.read_excel("../Data/h2a/H-2A_Disclosure_Data_FY2022_Q4.xlsx", usecols=small_col_list, dtype=small_h2a_dtype_dict, parse_dates=False)
    df_2022['FiscalYear'] = '2022'
    return df_2021, df_2022


@app.cell
def _(df_2021, df_2022, pd, small_rename_dict):
    df_2021_2022 = pd.concat([df_2021, df_2022])
    df_2021_2022 = df_2021_2022.rename(columns = small_rename_dict)
    df_2021_2022['ken'] = True
    return (df_2021_2022,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    H-2A dataset without Places API imputed counties
    """)
    return


@app.cell
def _(df_2021_2022, pd):
    h2a_df = pd.read_parquet("../binaries/h2a_raw.parquet")

    # Years prior to 2008 do not have worksite location information
    h2a_df = h2a_df[h2a_df['FiscalYear'].astype(int) >= 2008]
    h2a_df['ken'] = False

    h2a_df = pd.concat([h2a_df, df_2021_2022])

    # Remove non-US locations
    h2a_df = h2a_df[h2a_df['WORKSITE_STATE'] != 'AB'].copy() # AB is Alberta, Canada
    h2a_df = h2a_df[h2a_df['WORKSITE_STATE'] != 'SK'].copy() # SK is Saskatchewan, Canada
    h2a_df = h2a_df[h2a_df['WORKSITE_STATE'] != 'MB'].copy() # MB is Manitoba, Canada
    return (h2a_df,)


@app.cell
def _(h2a_df):
    # Obtain list of states and counties in H-2A data
    state_county_df = h2a_df[['WORKSITE_STATE', 'WORKSITE_COUNTY']].drop_duplicates()
    return (state_county_df,)


@app.cell
def _(state_county_df):
    # Add column with fixed names for counties
    state_county_df['state'] = state_county_df['WORKSITE_STATE']
    state_county_df['county'] = state_county_df['WORKSITE_COUNTY']
    return


@app.cell
def _(state_county_df):
    # Fix county names with multiple counties, or with city names
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'HARRISON & BOURBON COUNTIES', 'county'] = 'HARRISON,BOURBON'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'STAMPING GROUND SCOTT COUNTY', 'county'] = 'SCOTT'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'HENRY & UNION COUNTY', 'county'] = 'HENRY,UNION'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'SILVER BOW AND MADISON', 'county'] = 'SILVER BOW,MADISON'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'CANDLER BULLOCH & EVANS COUNTIES', 'county'] = 'CANDLER,BULLOCH,EVANS'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'SAMPSON AND JOHNSTON COUNTIES', 'county'] = 'SAMPSOM,JOHNSTON'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'LINCOLN AND BAYFIELD COUNTIES', 'county'] = 'LINCOLN,BAYFIELD'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'ALLEN & MONROE COUNTIES', 'county'] = 'ALLEN,MONROE'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'OWEN & HENRY COUNTIES', 'county'] = 'OWEN,HENRY'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'UPTON/MAGNOLIA HART COUNTY', 'county'] = 'UPTON,MAGNOLIA,HART'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'WARREN & SIMPSON COUNTY', 'county'] = 'WARREN,SIMPSON'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'CASEY & BOYLE COUNTIES', 'county'] = 'CASEY,BOYLE'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'MONTGOMERY & HALIFAX COUNTIES', 'county'] = 'MONTOGOMERY,HALIFAX'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'COUNTIES: RISING SUN DILLISBORO DEARBORN', 'county'] = 'RISING SUN,DILLISBORO,DEARBORN'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'GEORGE AND RANKIN COUNTIES', 'county'] = 'GEORGE,RANKIN'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'MISSAUKEE WEXFORD OSCEOLA & ANTRIM COUNTIES', 'county'] = 'MISSAUKEE,WEXFORD,OSCEOLA,ANTRIM'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'WASHAKIE BIG HORN JOHNSON COUNTY', 'county'] = 'WASHAKIE,BIG HORN,JOHNSON'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'SOMERSET/BALD MOUNTAIN-UNORGANIZED TS', 'county'] = 'SOMERSET,BALD MOUNTAIN'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'MAHLEUR AND HARNEY COUNTIES', 'county'] = 'MAHLEUR,HARNEY'
    state_county_df.loc[state_county_df['WORKSITE_COUNTY'] == 'FRANKLIN & SOMERSET', 'county'] = 'FRANKLIN,SOMERSET'
    return


@app.cell
def _(pd):
    pd.set_option('display.max_rows', 20)
    return


@app.cell
def _(state_county_df):
    # Create duplicate entries for applications with multiple counties that are separated with a comma
    state_county_df['county_list'] = state_county_df['county'].str.split(',')
    state_county_df_1 = state_county_df.explode('county_list')
    return (state_county_df_1,)


@app.cell
def _(state_county_df_1):
    # Drop entries with no county or no state
    state_county_df_2 = state_county_df_1[state_county_df_1['county_list'].notna() & state_county_df_1['state'].notna()]
    return (state_county_df_2,)


@app.cell
def _(addfips, state_county_df_2):
    af = addfips.AddFIPS()
    state_county_df_2['fips_nonapi'] = state_county_df_2.apply(lambda x: af.get_county_fips(_x['county_list'], state=_x['state']), axis=1)
    return (af,)


@app.cell
def _(state_county_df_2):
    state_county_df_2['fips_nonapi'] = state_county_df_2['fips_nonapi'].astype('string').fillna('')
    state_county_df_2['fips_nonapi'] = state_county_df_2.groupby(['WORKSITE_STATE', 'WORKSITE_COUNTY'])['fips_nonapi'].transform(lambda x: ','.join(_x))
    state_county_df_2.drop_duplicates(subset=['WORKSITE_STATE', 'WORKSITE_COUNTY', 'fips_nonapi'], inplace=True)
    return


@app.cell
def _(h2a_df, state_county_df_2):
    # Add non-API county FIPS codes to entries
    h2a_nonapi_df = h2a_df.merge(state_county_df_2, left_on=['WORKSITE_STATE', 'WORKSITE_COUNTY'], right_on=['WORKSITE_STATE', 'WORKSITE_COUNTY'], how='left')
    h2a_nonapi_df = h2a_nonapi_df[h2a_nonapi_df['ken'] == False]
    return (h2a_nonapi_df,)


@app.cell
def _(df_2021_2022, state_county_df_2):
    h2a_2021_2022_nonapi_df = df_2021_2022.merge(state_county_df_2, left_on=['WORKSITE_STATE', 'WORKSITE_COUNTY'], right_on=['WORKSITE_STATE', 'WORKSITE_COUNTY'], how='left')
    # Save as binary to be processed in R
    h2a_2021_2022_nonapi_df.to_parquet('../binaries/h2a_2021_2022_with_fips.parquet', index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use Places API to impute counties for unmatched entries; put in separate parquet
    """)
    return


@app.cell
def _(state_county_df_2):
    # Export list of unmatched state-county pairs
    unmatched_counties_df = state_county_df_2[state_county_df_2['fips_nonapi'] == '']
    unmatched_counties_df = unmatched_counties_df.assign(place_name=unmatched_counties_df['county_list'] + ', ' + unmatched_counties_df['state'].astype(str))
    place_names_list = list(unmatched_counties_df['place_name'])
    return (unmatched_counties_df,)


@app.cell
def _():
    # Use Google Places API to fill in missing county names if possible
    # Google maps API key from my account
    # Import API key stored in text file
    with open("../tools/google_places_api_key.txt") as f:
        lines = f.readlines()

    api_key = lines[0]
    return


@app.cell
def _():
    # # Make API call for each place name to obtain Google Places placeID
    # Base url to call findplace API
    # base_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json?"
    # placeid_dict = {}
    # for place_ind in range(0, len(place_names_list)):
    
    #     place_name = place_names_list[place_ind]
    #     print('Currently looking up ' + place_name)

    #     # Create API request
    #     # URL'ed location name we want to search
    #     input = urllib.parse.quote(place_name) # Encode place name as URL string
    #     request_url = base_url + "input=" + input + "&inputtype=textquery" + "&key=" + api_key

    #     payload = {}
    #     headers = {}

    #     # Sleep one second between each API call
    #     time.sleep(1)

    #     # Make API call
    #     response = requests.request("GET", request_url, headers=headers, data=payload)
    #     response_json = response.json()
    
    #     # If API call is successful, then place response result into dict
    #     if response_json['status']=='OK':
    #         print('Successful')
    #         placeid_dict[place_name] = response_json
    #     else:
    #         # If API call is unsuccessful, then wait 5 seconds and retry
    #         print('NOT successful, retrying')
    #         time.sleep(5)
    #         response = requests.request("GET", request_url, headers=headers, data=payload)
    #         response_json = response.json()

    #         if response_json['status']=='OK':
    #             print('Retry successful')
    #             placeid_dict[place_name] = response_json
    #         else:
    #             error_type = response_json['status']
    #             print('Retry unsuccessful, error: ' + error_type)
    return


@app.cell
def _():
    # # Save API request results as JSON
    # with open('json/placeid_api_request_result.json', 'w') as f:
    #     json.dump(placeid_dict, f)
    return


@app.cell
def _(json):
    # Load JSON of API responses and put into DataFrame
    with open('json/placeid_api_request_result.json', 'r') as _infile:
        placeid_dict = json.load(_infile)
    return (placeid_dict,)


@app.cell
def _(pd, placeid_dict):
    # Put place IDs into DataFrame
    placeid_df = pd.DataFrame(columns=['name_to_search', 'placeid'])
    for _search_term, _response in placeid_dict.items():
        _number_of_candidates = len(_response['candidates'])
        for _response_ind in range(0, _number_of_candidates):
            _address = _response['candidates'][_response_ind]['place_id']
            placeid_df.loc[len(placeid_df)] = [_search_term, _address]
    return (placeid_df,)


@app.cell
def _():
    # # Use Place details API to get county names
    # # base_url = 'https://maps.googleapis.com/maps/api/place/details/json?'
    # place_details_dict = {}

    # # Iterate over each place ID
    # for index, row in placeid_df.iterrows():
    #     print(row['name_to_search'], row['placeid'])

    #     # Create API request
    #     input = row['placeid']
    #     place_name = row['name_to_search']
    #     request_url = base_url + "placeid=" + input + "&key=" + api_key

    #     payload = {}
    #     headers = {}

    #     response = requests.request("GET", request_url, headers=headers, data=payload)
    #     response_json = response.json()

    #     # If API call is successful, then place response result into dict
    #     if response_json['status']=='OK':
    #         print('Successful')
    #         place_details_dict[input] = response_json
    #     else:
    #         # If API call is unsuccessful, then wait 5 seconds and retry
    #         print('NOT successful, retrying')
    #         time.sleep(5)
    #         response = requests.request("GET", request_url, headers=headers, data=payload)
    #         response_json = response.json()

    #         if response_json['status']=='OK':
    #             print('Retry successful')
    #             place_details_dict[input] = response_json
    #         else:
    #             error_type = response_json['status']
    #             print('Retry unsuccessful, error: ' + error_type)
    return


@app.cell
def _():
    # # Save API request results as JSON
    # with open('json/place_details_api_request_result.json', 'w') as f:
    #     json.dump(place_details_dict, f)
    return


@app.cell
def _(json):
    # Load JSON of API responses and put into DataFrame
    with open('json/place_details_api_request_result.json', 'r') as _infile:
        place_details_dict = json.load(_infile)
    return (place_details_dict,)


@app.cell
def _(place_details_dict):
    # Store county name from place details into dictionary (store state names too as there may be incorrect states)
    county_name_dict = {}
    state_name_dict = {}
    for _placeid, _place_ind in place_details_dict.items():
        _address_components_dict_list = place_details_dict[_placeid]['result']['address_components']
        for _component_ind in range(len(_address_components_dict_list)):
            _component_admin_level = _address_components_dict_list[_component_ind]['types'][0]
            if _component_admin_level == 'administrative_area_level_2':  # Address components administrative level
                county_name_dict[_placeid] = _address_components_dict_list[_component_ind]['long_name']
            elif _component_admin_level == 'administrative_area_level_1':
                state_name_dict[_placeid] = _address_components_dict_list[_component_ind]['long_name']  # County name  # State name
    return county_name_dict, state_name_dict


@app.cell
def _(county_name_dict, pd, state_name_dict):
    # Convert state and county name dicts into DataFrames
    placeid_state_df = pd.DataFrame.from_dict(state_name_dict, orient='index', columns=['state'])
    placeid_state_df.reset_index(inplace=True)
    placeid_state_df.rename(columns={'index': 'placeid'}, inplace=True)

    placeid_county_df = pd.DataFrame.from_dict(county_name_dict, orient='index', columns=['county'])
    placeid_county_df.reset_index(inplace=True)
    placeid_county_df.rename(columns={'index':'placeid'}, inplace=True)
    return placeid_county_df, placeid_state_df


@app.cell
def _(placeid_county_df, placeid_df, placeid_state_df):
    # Add state and county names back into placeid_df
    placeid_df_1 = placeid_df.merge(placeid_state_df, left_on='placeid', right_on='placeid', how='left')
    placeid_df_1 = placeid_df_1.merge(placeid_county_df, left_on='placeid', right_on='placeid', how='left')
    return (placeid_df_1,)


@app.cell
def _(np, pd, placeid_df_1):
    # Fix a few incorrect entries
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'WASHINGOTN, ME', 'county'] = 'Washington County'
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'FREDRICK, VA', 'county'] = 'Frederick County'
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'UPTON, KY', 'county'] = 'Hardin County'  # Upton is mostly located in Hardin county
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'CA, CA', 'county'] = np.nan
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'MOFFAT, WY', 'county'] = 'Moffat County'
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'MOFFAT, WY', 'state'] = 'CO'  # Moffat County is on the WY/CO border in CO
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'MADERA, ID', 'county'] = np.nan  # Not sure what Madera, ID is
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'MADERA, ID', 'state'] = np.nan
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'GOODING, CO', 'county'] = 'Boulder County'  # Gooding is in Boulder County, CO
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'GOODING, CO', 'state'] = 'CO'
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'CUSTER, ND', 'county'] = np.nan  # Not sure what CUSTER, ND refers to
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'CUSTER, ND', 'state'] = np.nan
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'ORANGE, OK', 'county'] = np.nan  # Not sure what ORANGE, OK refers to
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'ORANGE, OK', 'state'] = np.nan
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'ALACHUA, GA', 'county'] = 'Alachua County'  # Alachua County is in Florida
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'ALACHUA, GA', 'state'] = 'Florida'
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'MOFFAT, FL', 'county'] = np.nan  # There is no place called Moffat in Florida
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'MOFFAT, FL', 'state'] = np.nan
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'JONES, SC', 'county'] = np.nan  # There is no Jones in SC
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'JONES, SC', 'state'] = np.nan
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'DUNN, SD', 'county'] = np.nan  # There is no Dunn in SD
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'DUNN, SD', 'state'] = np.nan
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'DICKEY, SD', 'county'] = np.nan  # There is no Dickey in SD
    placeid_df_1.loc[placeid_df_1['name_to_search'] == 'DICKEY, SD', 'state'] = np.nan
    connecticut_row = placeid_df_1[placeid_df_1['name_to_search'] == 'WESTERN CONNECTICUT, CT'].copy()
    # Western Connecticut consists of Lichtfield and Fairfield counties
    connecticut_row['county'] = 'Litchfield County'
    placeid_df_2 = pd.concat([placeid_df_1, connecticut_row], ignore_index=True)
    return (placeid_df_2,)


@app.cell
def _(placeid_df_2):
    # Drop locations that do not have a county name (including those incorrectly resolved that I removed)
    placeid_df_2.dropna(subset=['county'], inplace=True)
    return


@app.cell
def _(af, placeid_df_2):
    placeid_df_2['fips'] = placeid_df_2.apply(lambda x: af.get_county_fips(_x['county'], state=_x['state']), axis=1)
    return


@app.cell
def _(placeid_df_2):
    # LaSalle Parish in Louisiana is not in the addfips package(?), so add it manually
    placeid_df_2.loc[placeid_df_2['name_to_search'] == 'LASALLE, LA', 'fips'] = '22059'
    return


@app.cell
def _(placeid_df_2):
    placeid_df_3 = placeid_df_2[['name_to_search', 'fips']]
    placeid_df_3 = placeid_df_3.groupby(['name_to_search']).agg({'fips': lambda x: _x.tolist()}).reset_index()
    return (placeid_df_3,)


@app.cell
def _(placeid_df_3, unmatched_counties_df):
    # Prepare to append back into original DataFrame of states and counties
    placeid_df_3.rename(columns={'name_to_search': 'place_name'}, inplace=True)
    unmatched_counties_df.drop(columns=['fips_nonapi'], inplace=True)
    api_counties_df = unmatched_counties_df.merge(placeid_df_3, left_on='place_name', right_on='place_name', how='left')
    api_counties_df.drop(columns=['place_name'], inplace=True)
    return (api_counties_df,)


@app.cell
def _(api_counties_df, np):
    # Do some final cleaning
    api_counties_df.loc[api_counties_df['WORKSITE_COUNTY'] == 'KNOK', 'fips'] = '23027' # Knox town in Maine was misspelled as KNOK
    api_counties_df.loc[api_counties_df['WORKSITE_COUNTY'] == 'VARIOUS COUNTIES IN WESTERN', 'fips'] = np.nan # Various counties in NC not specified
    api_counties_df.loc[api_counties_df['WORKSITE_COUNTY'] == 'VARIOUS COUNTIES', 'fips'] = np.nan # Various counties in NC not specified
    return


@app.cell
def _(api_counties_df):
    api_counties_df_1 = api_counties_df[api_counties_df['fips'].notna()]
    api_counties_df_1['fips_string'] = api_counties_df_1['fips'].apply(lambda x: ','.join(map(str, _x)))
    return (api_counties_df_1,)


@app.cell
def _(api_counties_df_1):
    api_counties_df_1.loc[api_counties_df_1['WORKSITE_COUNTY'] == 'KNOK', 'fips_string'] = '23027'  # Somehow FIPS cannot be stored as a list, so join splits the string; this fixes it
    return


@app.cell
def _(api_counties_df_1):
    # We want to merge when all FIPS codes are stored as strings only
    api_counties_df_1.drop(columns=['fips'], inplace=True)
    api_counties_df_1['fips_string'] = api_counties_df_1['fips_string'].astype('string').fillna('')
    return


@app.cell
def _(api_counties_df_1, state_county_df_2):
    # Combine all FIPS codes for each WORKSITE_STATE and WORKSITE_COUNTY
    merged_df = state_county_df_2.merge(api_counties_df_1, left_on=['WORKSITE_STATE', 'WORKSITE_COUNTY', 'state', 'county', 'county_list'], right_on=['WORKSITE_STATE', 'WORKSITE_COUNTY', 'state', 'county', 'county_list'], how='left')
    return (merged_df,)


@app.cell
def _(merged_df):
    # Combine FIPS codes from both sources
    merged_df['fips_api'] = merged_df['fips_nonapi'].where(merged_df['fips_string'].isna(), merged_df['fips_string'])
    return


@app.cell
def _(merged_df):
    # Collapse back into unique WORKSITE_STATE and WORKSITE_COUNTY pairs
    merged_df_1 = merged_df[['WORKSITE_STATE', 'WORKSITE_COUNTY', 'fips_api']]
    merged_df_1.loc[:, 'fips_api'] = merged_df_1['fips_api'].astype(str)
    return (merged_df_1,)


@app.cell
def _(merged_df_1):
    merged_df_1['fips_api'] = merged_df_1.groupby(['WORKSITE_STATE', 'WORKSITE_COUNTY'])['fips_api'].transform(lambda x: ','.join(_x))
    merged_df_1.drop_duplicates(subset=['WORKSITE_STATE', 'WORKSITE_COUNTY', 'fips_api'], inplace=True)
    return


@app.cell
def _(h2a_nonapi_df, merged_df_1):
    # Merge back into H-2A data
    h2a_merged_df = h2a_nonapi_df.merge(merged_df_1, left_on=['WORKSITE_STATE', 'WORKSITE_COUNTY'], right_on=['WORKSITE_STATE', 'WORKSITE_COUNTY'], how='left')
    return (h2a_merged_df,)


@app.cell
def _(h2a_merged_df):
    # For the remaining entries, we will try to get counties from the WORKSITE_CITY names
    state_city_df = h2a_merged_df[h2a_merged_df['fips_api'].isna()][['WORKSITE_STATE', 'WORKSITE_CITY']].dropna().drop_duplicates().reset_index(drop=True)
    return (state_city_df,)


@app.cell
def _():
    # United States of America Python Dictionary to translate States,
    # Districts & Territories to Two-Letter codes and vice versa.
    #
    # Canonical URL: https://gist.github.com/rogerallen/1583593
    #
    # Dedicated to the public domain.  To the extent possible under law,
    # Roger Allen has waived all copyright and related or neighboring
    # rights to this code.  Data originally from Wikipedia at the url:
    # https://en.wikipedia.org/wiki/ISO_3166-2:US
    #
    # Automatically Generated 2021-09-11 18:04:36 via Jupyter Notebook from
    # https://gist.github.com/rogerallen/d75440e8e5ea4762374dfd5c1ddf84e0 

    us_state_to_abbrev = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
        "District of Columbia": "DC",
        "American Samoa": "AS",
        "Guam": "GU",
        "Northern Mariana Islands": "MP",
        "Puerto Rico": "PR",
        "United States Minor Outlying Islands": "UM",
        "U.S. Virgin Islands": "VI"
    }
    # invert the dictionary
    abbrev_to_us_state = dict(map(reversed, us_state_to_abbrev.items()))
    return (abbrev_to_us_state,)


@app.cell
def _(abbrev_to_us_state, state_city_df):
    # Use state name instead of abbreviation; suspect that Google Places API is more accurate with state names
    state_city_df['state_name'] = state_city_df['WORKSITE_STATE'].map(abbrev_to_us_state)
    state_city_df['city_state_name'] = state_city_df['WORKSITE_CITY'] + ', ' + state_city_df['state_name']
    return


@app.cell
def _():
    # # Split API search into chunks of 100
    # # Base url to call findplace API
    # base_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json?"

    # for x in range(1, 17):
    #     x1 = x*100
    #     x2 = (x+1)*100
    #     df = state_city_df.iloc[x1:x2]

    #     # Make API call for each place name to obtain Google Places placeID
    #     city_placeid_dict = {}

    #     for ind, row in df.iterrows():
    #         place_name = row['city_state_name']
    #         print('Currently looking up ' + place_name)

    #         # Create API request
    #         # URL'ed location name we want to search
    #         input = urllib.parse.quote(place_name) # Encode place name as URL string
    #         request_url = base_url + "input=" + input + "&inputtype=textquery" + "&key=" + api_key

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
    #             city_placeid_dict[place_name] = response_json
    #         else:
    #             # If API call is unsuccessful, then wait 5 seconds and retry
    #             print('NOT successful, retrying')
    #             time.sleep(5)
    #             response = requests.request("GET", request_url, headers=headers, data=payload)
    #             response_json = response.json()

    #             if response_json['status']=='OK':
    #                 print('Retry successful')
    #                 city_placeid_dict[place_name] = response_json
    #             else:
    #                 error_type = response_json['status']
    #                 print('Retry unsuccessful, error: ' + error_type)

    #     # Save API request results as JSON
    #     with open(f'json/city_placeid_api_request_result_{x}.json', 'w') as f:
    #         json.dump(city_placeid_dict, f)
    return


@app.cell
def _(json):
    # Load JSON of API responses containing place IDs and put into dictionary
    city_placeid_dict = {}
    for _x in range(0, 17):
        with open(f'json/city_placeid_api_request_result_{_x}.json', 'r') as _infile:
            city_placeid_dict[_x] = json.load(_infile)
    return (city_placeid_dict,)


@app.cell
def _(city_placeid_dict, pd):
    placeid_df_4 = pd.DataFrame(columns=['city_state_name', 'placeid'])
    for _x in range(0, 17):
        for _search_term, _response in city_placeid_dict[_x].items():
            _number_of_candidates = len(_response['candidates'])
            for _response_ind in range(0, _number_of_candidates):
                _address = _response['candidates'][_response_ind]['place_id']
                placeid_df_4.loc[len(placeid_df_4)] = [_search_term, _address]
    return (placeid_df_4,)


@app.cell
def _(placeid_df_4):
    # Keep only results with no uncertainty in location
    placeid_df_no_duplicates = placeid_df_4.drop_duplicates(subset=['city_state_name'], keep=False)
    return (placeid_df_no_duplicates,)


@app.cell
def _():
    # # Now split up API requests into chunks of 100
    # placeid_df_no_duplicates.reset_index(inplace=True, drop=True)
    # # Base url to call Place details API
    # base_url = 'https://maps.googleapis.com/maps/api/place/details/json?'

    # for x in range(1, 16):
    #     x1 = x*100
    #     x2 = (x+1)*100
    #     df = placeid_df_no_duplicates.iloc[x1:x2]

    #     # Use Place details API to get county names
    #     place_details_dict = {}

    #     # Iterate over each place ID
    #     for index, row in df.iterrows():
    #         print(row['city_state_name'], row['placeid'])

    #         # Create API request
    #         input = row['placeid']
    #         place_name = row['city_state_name']
    #         request_url = base_url + "placeid=" + input + "&key=" + api_key

    #         payload = {}
    #         headers = {}

    #         response = requests.request("GET", request_url, headers=headers, data=payload)
    #         response_json = response.json()

    #         # If API call is successful, then place response result into dict
    #         if response_json['status']=='OK':
    #             print('Successful')
    #             place_details_dict[input] = response_json
    #         else:
    #             # If API call is unsuccessful, then wait 5 seconds and retry
    #             print('NOT successful, retrying')
    #             time.sleep(5)
    #             response = requests.request("GET", request_url, headers=headers, data=payload)
    #             response_json = response.json()

    #             if response_json['status']=='OK':
    #                 print('Retry successful')
    #                 place_details_dict[input] = response_json
    #             else:
    #                 error_type = response_json['status']
    #                 print('Retry unsuccessful, error: ' + error_type)

    #     # Save API request results as JSON
    #     with open(f'json/city_place_details_api_request_result_{x}.json', 'w') as f:
    #         json.dump(place_details_dict, f)
    return


@app.cell
def _(json):
    county_name_dict_1 = {}
    state_name_dict_1 = {}
    for _x in range(0, 16):
        with open(f'json/city_place_details_api_request_result_{_x}.json', 'r') as _infile:
            place_details_dict_1 = json.load(_infile)
        for _placeid, _place_ind in place_details_dict_1.items():
            _address_components_dict_list = place_details_dict_1[_placeid]['result']['address_components']
            for _component_ind in range(len(_address_components_dict_list)):
                _component_admin_level = _address_components_dict_list[_component_ind]['types'][0]
                if _component_admin_level == 'administrative_area_level_2':
                    county_name_dict_1[_placeid] = _address_components_dict_list[_component_ind]['long_name']
                elif _component_admin_level == 'administrative_area_level_1':
                    state_name_dict_1[_placeid] = _address_components_dict_list[_component_ind]['long_name']
    return county_name_dict_1, state_name_dict_1


@app.cell
def _(county_name_dict_1, pd, state_name_dict_1):
    # Convert state and county name dicts into DataFrames
    placeid_state_df_1 = pd.DataFrame.from_dict(state_name_dict_1, orient='index', columns=['state'])
    placeid_state_df_1.reset_index(inplace=True)
    placeid_state_df_1.rename(columns={'index': 'placeid'}, inplace=True)
    placeid_county_df_1 = pd.DataFrame.from_dict(county_name_dict_1, orient='index', columns=['county'])
    placeid_county_df_1.reset_index(inplace=True)
    placeid_county_df_1.rename(columns={'index': 'placeid'}, inplace=True)
    return placeid_county_df_1, placeid_state_df_1


@app.cell
def _(placeid_county_df_1, placeid_df_no_duplicates, placeid_state_df_1):
    # Add state and county names back into placeid_df
    placeid_df_no_duplicates_1 = placeid_df_no_duplicates.merge(placeid_state_df_1, left_on='placeid', right_on='placeid', how='left')
    placeid_df_no_duplicates_1 = placeid_df_no_duplicates_1.merge(placeid_county_df_1, left_on='placeid', right_on='placeid', how='left')
    return (placeid_df_no_duplicates_1,)


@app.cell
def _(np, placeid_df_no_duplicates_1):
    # Fix a few incorrect entries
    placeid_df_no_duplicates_1.loc[placeid_df_no_duplicates_1['city_state_name'] == 'SEE BELOW, New York', 'county'] = np.nan
    # Drop locations that do not have a county name (including those incorrectly resolved that I removed)
    placeid_df_no_duplicates_1.dropna(subset=['state', 'county'], inplace=True)
    return


@app.cell
def _(af, placeid_df_no_duplicates_1):
    placeid_df_no_duplicates_1['fips_api_city'] = placeid_df_no_duplicates_1.apply(lambda x: af.get_county_fips(_x['county'], state=_x['state']), axis=1)
    return


@app.cell
def _(h2a_merged_df, placeid_df_no_duplicates_1, state_city_df):
    # Merge FIPS code back to original DataFrame
    placeid_df_no_duplicates_1.loc[:, 'fips_api_city'] = placeid_df_no_duplicates_1['fips_api_city'].astype('string')
    placeid_df_no_duplicates_2 = placeid_df_no_duplicates_1[['city_state_name', 'fips_api_city']]
    state_city_df_1 = state_city_df.merge(placeid_df_no_duplicates_2, left_on='city_state_name', right_on='city_state_name', how='left')
    state_city_df_1.drop(columns=['state_name', 'city_state_name'], inplace=True)
    h2a_merged_df_1 = h2a_merged_df.merge(state_city_df_1, left_on=['WORKSITE_STATE', 'WORKSITE_CITY'], right_on=['WORKSITE_STATE', 'WORKSITE_CITY'], how='left')
    return (h2a_merged_df_1,)


@app.cell
def _(h2a_merged_df_1):
    # Create finalized combined FIPS column
    h2a_merged_df_1['fips_api'] = h2a_merged_df_1['fips_api'].astype('string').fillna('')
    h2a_merged_df_1['fips_api_city'] = h2a_merged_df_1['fips_api_city'].astype('string').fillna('')
    h2a_merged_df_1.loc[h2a_merged_df_1['fips_api'] == '', 'fips_api'] = h2a_merged_df_1['fips_api_city']
    return


@app.cell
def _(h2a_merged_df_1):
    # Save as binary to be processed in R
    h2a_merged_df_1.to_parquet('../binaries/h2a_with_fips.parquet', index=False)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
