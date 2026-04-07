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
    import polars as pl
    return json, pd, pickle, rapidfuzz


@app.cell
def _(pd):
    # Create Census geographic codes file
    census_county = pd.read_csv('../Data/census_geography_codes/national_county2020.txt', sep='|', dtype='string', keep_default_na=False).apply(lambda _x: _x.str.upper())
    census_countysub = pd.read_csv('../Data/census_geography_codes/national_cousub2020.txt', sep='|', dtype='string', keep_default_na=False).apply(lambda _x: _x.str.upper())
    census_place = pd.read_csv('../Data/census_geography_codes/national_place2020.txt', sep='|', dtype='string', keep_default_na=False).apply(lambda _x: _x.str.upper())
    census_placebycounty = pd.read_csv('../Data/census_geography_codes/national_place_by_county2020.txt', sep='|', dtype='string', keep_default_na=False).apply(lambda _x: _x.str.upper())
    census_zip = pd.read_csv('../Data/census_geography_codes/tab20_zcta520_county20_natl.txt', sep='|', dtype='string', keep_default_na=False).apply(lambda _x: _x.str.upper())

    # Add FIPS column
    census_county['fips'] = census_county['STATEFP'] + census_county['COUNTYFP']
    census_placebycounty['fips'] = census_placebycounty['STATEFP'] + census_placebycounty['COUNTYFP']
    census_zip['fips'] = census_zip['GEOID_COUNTY_20']

    census_county = census_county[['STATE', 'COUNTYNAME', 'fips']]
    census_place_agg = census_placebycounty.groupby(['STATE', 'COUNTYNAME', 'PLACENAME']).agg({'fips': lambda _x: ','.join(_x)}).reset_index()
    census_zip_agg = census_zip.groupby(['GEOID_ZCTA5_20']).agg({'fips': lambda _x: ','.join(_x)}).reset_index()

    # Drop empty entries
    census_county = census_county[census_county['COUNTYNAME'] != '']
    census_place_agg = census_place_agg[census_place_agg['PLACENAME'] != '']
    census_zip_agg = census_zip_agg[census_zip_agg['GEOID_ZCTA5_20'] != '']

    # Clean names of census file columns
    census_county = census_county.rename(columns={'STATE': 'state', 'COUNTYNAME': 'county'})
    census_place_agg = census_place_agg.rename(columns={'STATE': 'state', 'COUNTYNAME': 'county', 'PLACENAME': 'place'})
    census_zip_agg = census_zip_agg.rename(columns={'GEOID_ZCTA5_20': 'zip'})
    return census_county, census_place_agg, census_zip_agg


@app.cell
def _():
    # Define common set of variables we want from every fiscal year, and their types
    # We want all as string type, but some dates cannot be read as string type due to storage as date type in Excel, so strip the trailing time later
    h2a_dtype_dict = {}

    h2a_dtype_dict['2008'] = {
        'CASE_NO':'string',
        'CASE_STATUS':'string',
        'EMPLOYER_NAME':'string',
        'EMPLOYER_CITY':'string',
        'EMPLOYER_STATE':'string',
        'EMPLOYER_POSTAL_CODE':'string',
        'NBR_WORKERS_CERTIFIED':'string',
        'CERTIFICATION_BEGIN_DATE':'string',
        'CERTIFICATION_END_DATE':'string',
        'BASIC_RATE_OF_PAY':'string',
        'BASIC_UNIT_OF_PAY':'string',
        'ALIEN_WORK_CITY':'string',
        'ALIEN_WORK_STATE':'string',
        'ORGANIZATION_FLAG':'string'
    }

    h2a_dtype_dict['2009'] = {
        'CASE_NO':'string',
        'CASE_STATUS':'string',
        'EMPLOYER_NAME':'string',
        'EMPLOYER_CITY':'string',
        'EMPLOYER_STATE':'string',
        'EMPLOYER_POSTAL_CODE':'string',
        'NBR_WORKERS_CERTIFIED':'string',
        'CERTIFICATION_BEGIN_DATE':'string',
        'CERTIFICATION_END_DATE':'string',
        'BASIC_RATE_OF_PAY':'string',
        'BASIC_UNIT_OF_PAY':'string',
        'ALIEN_WORK_CITY':'string',
        'ALIEN_WORK_STATE':'string',
        'ORGANIZATION_FLAG':'string'
    }

    h2a_dtype_dict['2010'] = h2a_dtype_dict['2009']

    h2a_dtype_dict['2011'] = {
        'CASE_NO':'string',
        'CASE_STATUS':'string',
        'EMPLOYER_NAME':'string',
        'EMPLOYER_CITY':'string',
        'EMPLOYER_STATE':'string',
        'EMPLOYER_POSTAL_CODE':'string',
        'NBR_WORKERS_REQUESTED':'string',
        'NBR_WORKERS_CERTIFIED':'string',
        'REQUESTED_START_DATE_OF_NEED':'object',
        'REQUESTED_END_DATE_OF_NEED':'object',
        'CERTIFICATION_BEGIN_DATE':'string',
        'CERTIFICATION_END_DATE':'string',
        'BASIC_NUMBER_OF_HOURS':'string',
        'BASIC_RATE_OF_PAY':'string',
        'BASIC_UNIT_OF_PAY':'string',
        'ALIEN_WORK_CITY':'string',
        'ALIEN_WORK_STATE':'string',
        'ORGANIZATION_FLAG':'string'
    }

    h2a_dtype_dict['2012'] = h2a_dtype_dict['2011']

    h2a_dtype_dict['2013'] = {
        'CASE_NO':'string',
        'CASE_STATUS':'string',
        'EMPLOYER_NAME':'string',
        'EMPLOYER_CITY':'string',
        'EMPLOYER_STATE':'string',
        'EMPLOYER_POSTAL_CODE':'string',
        'NBR_WORKERS_CERTIFIED':'string',
        'REQUESTED_START_DATE_OF_NEED':'object',
        'REQUESTED_END_DATE_OF_NEED':'object',
        'CERTIFICATION_BEGIN_DATE':'string',
        'CERTIFICATION_END_DATE':'string',
        'BASIC_NUMBER_OF_HOURS':'string',
        'BASIC_RATE_OF_PAY':'string',
        'BASIC_UNIT_OF_PAY':'string',
        'ALIEN_WORK_CITY':'string',
        'ALIEN_WORK_STATE':'string',
        'ORGANIZATION_FLAG':'string'
    }

    h2a_dtype_dict['2014'] = {
        'CASE_NO':'string',
        'CASE_STATUS':'string',
        'EMPLOYER_NAME':'string',
        'EMPLOYER_CITY':'string',
        'EMPLOYER_STATE':'string',
        'EMPLOYER_POSTAL_CODE':'string',
        'NBR_WORKERS_CERTIFIED':'string',
        'REQUESTED_START_DATE_OF_NEED':'object',
        'REQUESTED_END_DATE_OF_NEED':'object',
        'CERTIFICATION_BEGIN_DATE':'string',
        'CERTIFICATION_END_DATE':'string',
        'BASIC_NUMBER_OF_HOURS':'string',
        'BASIC_RATE_OF_PAY':'string',
        'BASIC_UNIT_OF_PAY':'string',
        'WORKSITE_LOCATION_CITY':'string',
        'WORKSITE_LOCATION_STATE':'string',
        'ORGANIZATION_FLAG':'string'
    }

    h2a_dtype_dict['2015'] = {
        'CASE_NUMBER':'string',
        'CASE_STATUS':'string',
        'EMPLOYER_NAME':'string',
        'EMPLOYER_CITY':'string',
        'EMPLOYER_STATE':'string',
        'EMPLOYER_POSTAL_CODE':'string',
        'NBR_WORKERS_REQUESTED':'string',
        'NBR_WORKERS_CERTIFIED':'string',
        'CERTIFICATION_BEGIN_DATE':'string',
        'CERTIFICATION_END_DATE':'string',
        'BASIC_NUMBER_OF_HOURS':'string',
        'BASIC_RATE_OF_PAY':'string',
        'BASIC_UNIT_OF_PAY':'string',
        'WORKSITE_CITY':'string',
        'WORKSITE_STATE':'string',
        'WORKSITE_POSTAL_CODE':'string',
        'ORGANIZATION_FLAG':'string'
    }

    h2a_dtype_dict['2016'] = {
        'CASE_NUMBER':'string',
        'CASE_STATUS':'string',
        'EMPLOYER_NAME':'string',
        'EMPLOYER_CITY':'string',
        'EMPLOYER_STATE':'string',
        'EMPLOYER_POSTAL_CODE':'string',
        'NBR_WORKERS_REQUESTED':'string',
        'NBR_WORKERS_CERTIFIED':'string',
        'REQUESTED_START_DATE_OF_NEED':'object',
        'REQUESTED_END_DATE_OF_NEED':'object',
        'JOB_START_DATE':'string',
        'JOB_END_DATE':'string',
        'BASIC_NUMBER_OF_HOURS':'string',
        'BASIC_RATE_OF_PAY':'string',
        'BASIC_UNIT_OF_PAY':'string',
        'WORKSITE_CITY':'string',
        'WORKSITE_STATE':'string',
        'WORKSITE_POSTAL_CODE':'string',
        'ORGANIZATION_FLAG':'string',
        'PRIMARY/SUB':'string',
    }

    h2a_dtype_dict['2017'] = {
        'CASE_NUMBER':'string',
        'CASE_STATUS':'string',
        'EMPLOYER_NAME':'string',
        'EMPLOYER_CITY':'string',
        'EMPLOYER_STATE':'string',
        'EMPLOYER_POSTAL_CODE':'string',
        'NBR_WORKERS_REQUESTED':'string',
        'NBR_WORKERS_CERTIFIED':'string',
        'REQUESTED_START_DATE_OF_NEED':'object',
        'REQUESTED_END_DATE_OF_NEED':'object',
        'JOB_START_DATE':'string', 
        'JOB_END_DATE':'string',
        'BASIC_NUMBER_OF_HOURS':'string',
        'BASIC_RATE_OF_PAY':'string',
        'BASIC_UNIT_OF_PAY':'string',
        'WORKSITE_CITY':'string',
        'WORKSITE_COUNTY':'string',
        'WORKSITE_STATE':'string',
        'WORKSITE_POSTAL_CODE':'string',
        'ORGANIZATION_FLAG':'string',
        'PRIMARY/SUB':'string'
    }

    h2a_dtype_dict['2018'] = {
        'CASE_NO':'string',
        'CASE_STATUS':'string',
        'EMPLOYER_NAME':'string',
        'EMPLOYER_CITY':'string',
        'EMPLOYER_STATE':'string',
        'EMPLOYER_POSTAL_CODE':'string',
        'NBR_WORKERS_REQUESTED':'string',
        'NBR_WORKERS_CERTIFIED':'string',
        'REQUESTED_START_DATE_OF_NEED':'object',
        'REQUESTED_END_DATE_OF_NEED':'object',
        'JOB_START_DATE':'string', 
        'JOB_END_DATE':'string',
        'BASIC_NUMBER_OF_HOURS':'string',
        'BASIC_RATE_OF_PAY':'string',
        'BASIC_UNIT_OF_PAY':'string',
        'WORKSITE_CITY':'string',
        'WORKSITE_COUNTY':'string',
        'WORKSITE_STATE':'string',
        'WORKSITE_POSTAL_CODE':'string',
        'ORGANIZATION_FLAG':'string',
        'PRIMARY_SUB':'string'
    }

    h2a_dtype_dict['2019'] = {
        'CASE_NUMBER':'string',
        'CASE_STATUS':'string',
        'EMPLOYER_NAME':'string',
        'EMPLOYER_CITY':'string',
        'EMPLOYER_STATE':'string',
        'EMPLOYER_POSTAL_CODE':'string',
        'NBR_WORKERS_REQUESTED':'string',
        'NBR_WORKERS_CERTIFIED':'string',
        'REQUESTED_START_DATE_OF_NEED':'object',
        'REQUESTED_END_DATE_OF_NEED':'object',
        'JOB_START_DATE':'string', 
        'JOB_END_DATE':'string',
        'BASIC_NUMBER_OF_HOURS':'string',
        'BASIC_RATE_OF_PAY':'string',
        'BASIC_UNIT_OF_PAY':'string',
        'WORKSITE_CITY':'string',
        'WORKSITE_COUNTY':'string',
        'WORKSITE_STATE':'string',
        'WORKSITE_POSTAL_CODE':'string',
        'ORGANIZATION_FLAG':'string',
        'PRMARY/SUB':'string'
    }

    h2a_dtype_dict['2020'] = {
        'CASE_NUMBER':'string',
        'CASE_STATUS':'string',
        'EMPLOYER_NAME':'string',
        'EMPLOYER_CITY':'string',
        'EMPLOYER_STATE':'string',
        'EMPLOYER_POSTAL_CODE':'string',
        'TOTAL_WORKERS_NEEDED':'string',
        'TOTAL_WORKERS_H2A_REQUESTED':'string',
        'TOTAL_WORKERS_H2A_CERTIFIED':'string',
        'REQUESTED_BEGIN_DATE':'string',
        'REQUESTED_END_DATE':'string',
        'EMPLOYMENT_BEGIN_DATE':'string',
        'EMPLOYMENT_END_DATE':'string',
        'ANTICIPATED_NUMBER_OF_HOURS':'string',
        'WAGE_OFFER':'string',
        'PER':'string',
        'WORKSITE_CITY':'string',
        'WORKSITE_COUNTY':'string',
        'WORKSITE_STATE':'string',
        'WORKSITE_POSTAL_CODE':'string',
        'TYPE_OF_EMPLOYER_APPLICATION':'string',
        'H2A_LABOR_CONTRACTOR':'string'
    }

    h2a_dtype_dict['2021'] = h2a_dtype_dict['2020']
    h2a_dtype_dict['2022'] = h2a_dtype_dict['2020']
    h2a_dtype_dict['2023'] = h2a_dtype_dict['2020']

    h2a_dtype_dict['2024'] = {
        'CASE_NUMBER':'string',
        'CASE_STATUS':'string',
        'EMPLOYER_NAME':'string',
        'EMPLOYER_CITY':'string',
        'EMPLOYER_STATE':'string',
        'EMPLOYER_POSTAL_CODE':'string',
        'TOTAL_WORKERS_NEEDED':'string',
        'TOTAL_WORKERS_H2A_REQUESTED':'string',
        'TOTAL_WORKERS_H2A_CERTIFIED':'string',
        'REQUESTED_BEGIN_DATE':'string',
        'REQUESTED_END_DATE':'string',
        'EMPLOYMENT_BEGIN_DATE':'string',
        'EMPLOYMENT_END_DATE':'string',
        'ANTICIPATED_NUMBER_OF_HOURS':'string',
        'WAGE_OFFER':'string',
        'PER':'string',
        'WORKSITE_CITY':'string',
        'WORKSITE_COUNTY':'string',
        'WORKSITE_STATE':'string',
        'WORKSITE_POSTAL_CODE':'string',
        'TYPE_OF_EMPLOYER_APPLICATION':'string',
        'AG_ASSN_OR_AGENCY_STATUS':'string',
        'H2A_LABOR_CONTRACTOR':'string'
    }
    return


@app.cell
def _():
    # Names of H-2A program disclosure files from the DOL-OFLC
    h2a_filenames_dict = {
        '2008':'H2A_FY2008.xlsx',
        '2009':'H2A_FY2009.xlsx',
        '2010':'H-2A_FY2010.xlsx',
        '2011':'H-2A_FY2011.xlsx',
        '2012':'H-2A_FY2012.xlsx',
        '2013':'H2A_FY2013.xls',
        '2014':'H-2A_FY14_Q4.xlsx',
        '2015':'H-2A_Disclosure_Data_FY15_Q4.xlsx',
        '2016':'H-2A_Disclosure_Data_FY16_updated.xlsx',
        '2017':'H-2A_Disclosure_Data_FY17.xlsx',
        '2018':'H-2A_Disclosure_Data_FY2018_EOY.xlsx',
        '2019':'H-2A_Disclosure_Data_FY2019.xlsx',
        '2020':'H-2A_Disclosure_Data_FY2020.xlsx',
        '2021':'H-2A_Disclosure_Data_FY2021.xlsx',
        '2022':'H-2A_Disclosure_Data_FY2022_Q4.xlsx',
        '2023':'H-2A_Disclosure_Data_FY2023_Q4.xlsx',
        '2024':'H-2A_Disclosure_Data_FY2024_Q4.xlsx'
    }
    return


@app.cell
def _(pd):
    # Function that reads lists of H-2A disclosure files and returns a dict of all dataframes
    def read_h2a_excel_files(year_filename_dict, year_list, year_dtype_dict):
        h2a_df_dict = {}
        for y in year_list:
            year = str(y)
            filename = year_filename_dict[year]
            print(filename)
            dtype_dict = year_dtype_dict[year]
            col_list = list(dtype_dict.keys())
            h2a_df_dict[year] = pd.read_excel('../Data/h2a/' + filename, dtype=dtype_dict, usecols=col_list, parse_dates=False, na_filter=False)

        return h2a_df_dict
    return


@app.cell
def _():
    # years_to_load = list(range(2008, 2025))
    # h2a_df_dict = read_h2a_excel_files(h2a_filenames_dict, years_to_load, h2a_dtype_dict)

    # # Pickle
    # with open("h2a_pickle", "wb") as _fp:
    #     pickle.dump(h2a_df_dict, _fp)
    return


@app.cell
def _(pickle):
    with open("h2a_pickle", "rb") as _fp:
        h2a_df_dict = pickle.load(_fp)
    return (h2a_df_dict,)


@app.cell
def _():
    # Define set of common names for concatenating
    h2a_rename_dict = {
        'CASE_NO':'case_number',
        'CASE_NUMBER':'case_number',
        'CASE_STATUS':'case_status',
        'EMPLOYER_NAME':'employer_name',
        'EMPLOYER_CITY':'employer_city',
        'EMPLOYER_STATE':'employer_state',
        'EMPLOYER_POSTAL_CODE':'employer_postal_code',
        'NBR_WORKERS_REQUESTED':'nbr_workers_requested',
        'NBR_WORKERS_CERTIFIED':'nbr_workers_certified',
        'TOTAL_WORKERS_NEEDED':'nbr_workers_needed',
        'TOTAL_WORKERS_H2A_REQUESTED':'nbr_workers_requested',
        'TOTAL_WORKERS_H2A_CERTIFIED':'nbr_workers_certified',
        'REQUESTED_START_DATE_OF_NEED':'requested_begin_date',
        'REQUESTED_END_DATE_OF_NEED':'requested_end_date',
        'CERTIFICATION_BEGIN_DATE':'certification_begin_date',
        'CERTIFICATION_END_DATE':'certification_end_date',
        'REQUESTED_BEGIN_DATE':'requested_begin_date',
        'REQUESTED_END_DATE':'requested_end_date',
        'EMPLOYMENT_BEGIN_DATE':'job_begin_date',
        'EMPLOYMENT_END_DATE':'job_end_date',
        'JOB_START_DATE':'job_begin_date',
        'JOB_END_DATE':'job_end_date',
        'BASIC_NUMBER_OF_HOURS':'number_of_hours',
        'ANTICIPATED_NUMBER_OF_HOURS':'number_of_hours',
        'BASIC_RATE_OF_PAY':'wage_rate',
        'WAGE_OFFER':'wage_rate',
        'BASIC_UNIT_OF_PAY':'wage_unit',
        'PER':'wage_unit',
        'ALIEN_WORK_CITY':'worksite_city',
        'ALIEN_WORK_STATE':'worksite_state',
        'WORKSITE_LOCATION_CITY':'worksite_city',
        'WORKSITE_LOCATION_STATE':'worksite_state',
        'WORKSITE_CITY':'worksite_city',
        'WORKSITE_COUNTY':'worksite_county',
        'WORKSITE_STATE':'worksite_state',
        'WORKSITE_POSTAL_CODE':'worksite_zip',
        'PRIMARY/SUB':'primary_sub',
        'PRIMARY_SUB':'primary_sub',
        'PRMARY/SUB':'primary_sub',
        'ORGANIZATION_FLAG':'organization_flag',
        'TYPE_OF_EMPLOYER_APPLICATION':'type_of_employer_application',
        'H2A_LABOR_CONTRACTOR':'h2a_labor_contractor',
        'AG_ASSN_OR_AGENCY_STATUS':'ag_association_or_agency',
    }
    return (h2a_rename_dict,)


@app.cell
def _(pd):
    # Define function that concatenates all years together
    def concatenate_h2a_years(h2a_df_dict, h2a_rename_dict):
        h2a_all_years_df = pd.DataFrame()
        for year, df in h2a_df_dict.items():
            df = df.rename(columns=h2a_rename_dict)
            df['fiscal_year'] = int(year)

            # Concatenate
            h2a_all_years_df = pd.concat([h2a_all_years_df, df], ignore_index=True, sort=False)

            h2a_all_years_df = h2a_all_years_df.fillna(value='')

        return h2a_all_years_df
    return (concatenate_h2a_years,)


@app.cell
def _(concatenate_h2a_years, h2a_df_dict, h2a_rename_dict):
    h2a_df = concatenate_h2a_years(h2a_df_dict, h2a_rename_dict)
    h2a_df = h2a_df.astype("string")
    h2a_df = h2a_df.replace(' 00:00:00', '')

    for columns in h2a_df.columns:
        h2a_df[columns] = h2a_df[columns].str.replace(' 00:00:00', '', regex=False)
        h2a_df[columns] = h2a_df[columns].str.upper()
    return (h2a_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Add FIPS codes using worksite location information
    """)
    return


@app.cell
def _(h2a_df):
    # Set of unique worksite locations we need to find counties for
    h2a_worksite_locations = h2a_df[['worksite_city', 'worksite_county', 'worksite_state', 'worksite_zip']]
    h2a_worksite_locations = h2a_worksite_locations.drop_duplicates()
    return (h2a_worksite_locations,)


@app.cell
def _(h2a_worksite_locations):
    # Locations are defined by worksite_city, worksite_county, worksite_state, worksite_zip
    # We will be operating on a new set of columns instead to preserve the original location identifiers
    h2a_worksite_locations['xcity'] = h2a_worksite_locations['worksite_city']
    h2a_worksite_locations['xcounty'] = h2a_worksite_locations['worksite_county']
    h2a_worksite_locations['xstate'] = h2a_worksite_locations['worksite_state']
    h2a_worksite_locations['xzip'] = h2a_worksite_locations['worksite_zip']
    return


@app.cell
def clean_county_zip_explode():
    # Write a function that fixes typos in county name and explodes county names
    def clean_county_explode(df, county_col, state_col):

        # Replace separators with commas
        for sep in [' AND ', ' & ', '/']:
            df[county_col] = df[county_col].str.replace(sep, ',')

        # Explode
        df['county_list'] = df[county_col].str.split(',')
        df_exploded = df.explode('county_list').reset_index(drop=True)

        # Remove whitespacess
        df_exploded['county_list'] = df_exploded['county_list'].str.strip()

        # Strip suffixes from names
        for suffix in [' COUNTY', ' COUNTIES', ' PARISH', ' PARRISH']:
            df_exploded['county_list'] = df_exploded['county_list'].str.replace(suffix, '')

        # Fix common typos for Lousiana counties
        la_typo_dict = {
            'ST\\.\\w':'ST. ',
            'NORTH\\. ':'NORTH ',
            'SOUTH\\. ':'SOUTH ',
            'EAST\\. ':'EAST ',
            'WEST\\. ':'WEST ',
            'BATON ROGUE':'BATON ROUGE',
            'JEFF DAVIS':'JEFFERSON DAVIS',
            'IBERIAL':'IBERIA'
        }

        for typo, fix in la_typo_dict.items():
            df_exploded.loc[df_exploded[state_col] == 'LA', 'county_list'] = df_exploded[df_exploded[state_col] == 'LA']['county_list'].str.replace(typo, fix, regex=True)

        return df_exploded

    def clean_zip(df, zip_col):

        # Remove period from ZIP codes
        df[zip_col] = df[zip_col].str.replace('.', '', regex=False)

        # Remove the 4 extra trailing digits after hyphen from ZIP codes
        df[zip_col] = df[zip_col].str.split('-').str[0]

        # Pad with 0s from the left
        # df['zip'] = df['zip'].str.pad(width=5, side='left', fillchar='0')

        return df
    return clean_county_explode, clean_zip


@app.cell
def _(clean_county_explode, clean_zip, h2a_worksite_locations):
    # Explode by counties, clean ZIP codes
    h2a_worksite_locations_exploded = clean_county_explode(h2a_worksite_locations, 'xcounty', 'xstate')
    h2a_worksite_locations_exploded = clean_zip(h2a_worksite_locations_exploded, 'xzip')
    return (h2a_worksite_locations_exploded,)


@app.function
# Write a function that add FIPS using addfips
def add_fips_using_addfips(df, county_col, state_col, name_of_new_fips_col):
    import addfips
    af = addfips.AddFIPS()
    df[name_of_new_fips_col] = df.apply(lambda x: af.get_county_fips(x[county_col], state=x[state_col]), axis=1)
    df = df.fillna(value='')
    return df


@app.function
# Write a function that add FIPS using ZIP and census ZIP to county crosswalk; also does a sanity check by matching states
def add_fips_using_census_zip(df, zip_col, state_col, name_of_new_fips_col, zip_fips_df, census_zip_col, census_fips_col, check_df, check_state_col, check_fips_col,):
    zip_to_fips_dict = dict(zip(zip_fips_df[census_zip_col], zip_fips_df[census_fips_col]))

    # Match ZIP codes to FIPS
    df[name_of_new_fips_col] = df[zip_col].map(zip_to_fips_dict)
    df = df.fillna(value='')

    # Sanity check by matching states
    fips_to_state_dict = dict(zip(check_df[check_fips_col], check_df[check_state_col]))
    df['state_check'] = df[name_of_new_fips_col].map(fips_to_state_dict)

    # Null out FIPS where state does not match
    df.loc[df['state_check'] != df[state_col], name_of_new_fips_col] = ''

    # Null out FIPS where ZIP codes match to multiple FIPS codes
    df.loc[df['state_check'].isna(), name_of_new_fips_col] = '' 
    df = df.fillna(value='')

    df = df.drop(columns=['state_check'])
    return df


@app.cell
def _(census_county, census_zip_agg, h2a_worksite_locations_exploded):
    # Add ZIP and examine unmatched entries
    h2a_exploded_with_addfips = add_fips_using_addfips(h2a_worksite_locations_exploded, 'county_list', 'xstate', 'fips_from_addfips')
    h2a_exploded_with_addfips_zip = add_fips_using_census_zip(h2a_exploded_with_addfips, 'xzip', 'xstate', 'fips_from_census_zip', census_zip_agg, 'zip', 'fips', census_county, 'state', 'fips')

    h2a_exploded_with_addfips_zip[(h2a_exploded_with_addfips_zip['fips_from_addfips'] != h2a_exploded_with_addfips_zip['fips_from_census_zip']) & 
                                  (h2a_exploded_with_addfips_zip['fips_from_addfips'] != '') & 
                                  (h2a_exploded_with_addfips_zip['fips_from_census_zip'] != '')]
    return (h2a_exploded_with_addfips_zip,)


@app.cell
def _(h2a_exploded_with_addfips_zip):
    h2a_exploded_addfips_zip_unmatched = h2a_exploded_with_addfips_zip[(h2a_exploded_with_addfips_zip['fips_from_addfips'] == '') & (h2a_exploded_with_addfips_zip['fips_from_census_zip'] == '')]
    h2a_exploded_addfips_zip_unmatched
    return (h2a_exploded_addfips_zip_unmatched,)


@app.cell
def _(rapidfuzz):
    # Define function for fuzzy string matching
    def fuzz_search(df, df_match_col, df_state_col, df_fips_col, state, name_to_match):

        # Search only within the same state
        state_df = df[df[df_state_col] == state]

        # Strip place suffixes from names
        place_suffixes = [' CITY', ' TOWN', ' VILLAGE', ' CDP', ' MUNICIPALITY', ' BOROUGH', ' TOWNSHIP', ' CENSUS AREA', ' CENSUS DESIGNATED PLACE', ' COUNTY', ' PARISH']
        for suffix in place_suffixes:
            state_df[df_match_col] = state_df[df_match_col].str.replace(suffix, '', regex=False)

        # Get match score
        state_df['score'] = state_df[df_match_col].apply(lambda _x: rapidfuzz.fuzz.partial_ratio_alignment(_x, name_to_match, processor = rapidfuzz.utils.default_process).score)
        state_df = state_df.sort_values('score')

        # Get highest scoring match
        state_df = state_df.sort_values('score')
        max_score_row = state_df[state_df['score'] == state_df['score'].max()].reset_index()

        # Check if there is a match
        if len(max_score_row) >= 1:
            fips = str(max_score_row[df_fips_col][0])
            score = str(max_score_row['score'][0])
            census_name = max_score_row[df_match_col][0]
            return (fips, score, census_name)  # Best match
        else:
            return ('', '', '')
    return (fuzz_search,)


@app.cell
def _(census_county, fuzz_search, h2a_exploded_addfips_zip_unmatched):
    # Do fuzzy match on county name then merge results back in
    county_name_fuzzy_match_df = h2a_exploded_addfips_zip_unmatched.apply(
        lambda _x: fuzz_search(
            census_county,
            'county',
            'state',
            'fips',
            _x['xstate'],
            _x['county_list']
        ),
        axis=1,
        result_type='expand'
    )

    county_name_fuzzy_match_df = county_name_fuzzy_match_df.rename(columns={0:'fips_from_fuzzy_county', 1:'fuzzy_county_score', 2:'fuzzy_county_name'})
    return (county_name_fuzzy_match_df,)


@app.cell
def _(county_name_fuzzy_match_df, h2a_exploded_addfips_zip_unmatched):
    h2a_exploded_fuzzy_county = h2a_exploded_addfips_zip_unmatched.merge(county_name_fuzzy_match_df, left_index=True, right_index=True, how='left')
    # Inspect score
    h2a_exploded_fuzzy_county.sort_values('fuzzy_county_score', ascending=False)[['county_list', 'fuzzy_county_name', 'fuzzy_county_score']]
    return (h2a_exploded_fuzzy_county,)


@app.cell
def _(census_place_agg, fuzz_search, h2a_exploded_addfips_zip_unmatched):
    # Do fuzzy match on city name then merge results back in
    city_name_fuzzy_match_df = h2a_exploded_addfips_zip_unmatched.apply(
        lambda _x: fuzz_search(
            census_place_agg,
            'place',
            'state',
            'fips',
            _x['xstate'],
            _x['xcity']
        ),
        axis=1,
        result_type='expand'
    )

    city_name_fuzzy_match_df = city_name_fuzzy_match_df.rename(columns={0:'fips_from_fuzzy_city', 1:'fuzzy_city_score', 2:'fuzzy_city_name'})
    return (city_name_fuzzy_match_df,)


@app.cell
def _(city_name_fuzzy_match_df, h2a_exploded_fuzzy_county):
    # Merge results back in
    h2a_exploded_fuzzy_county_city = h2a_exploded_fuzzy_county.merge(city_name_fuzzy_match_df, left_index=True, right_index=True, how='left')
    # Inspect score
    h2a_exploded_fuzzy_county_city.sort_values('fuzzy_city_score', ascending=False)[['xcity', 'fuzzy_city_name', 'fuzzy_city_score']]
    return (h2a_exploded_fuzzy_county_city,)


@app.cell
def _(census_place_agg, fuzz_search, h2a_exploded_fuzzy_county_city):
    # New England has to be matched again as they put place names in their county field
    ne_states = ['CT', 'ME', 'MA', 'NH', 'RI', 'VT']
    ne_fuzzy_match_df = h2a_exploded_fuzzy_county_city.apply(
        lambda _x: fuzz_search(
            census_place_agg,
            'place',
            'state',
            'fips',
            _x['state'],
            _x['county_list']
        ),
        axis=1,
        result_type='expand'
    )
    ne_fuzzy_match_df = ne_fuzzy_match_df.rename(columns={0:'fips_from_fuzzy_ne', 1:'fuzzy_ne_score', 2:'fuzzy_ne_name'})
    return (ne_fuzzy_match_df,)


@app.cell
def _(h2a_exploded_fuzzy_county_city, ne_fuzzy_match_df):
    # Merge results back in
    h2a_exploded_fuzzy_county_city_ne = h2a_exploded_fuzzy_county_city.merge(ne_fuzzy_match_df, left_index=True, right_index=True, how='left')
    # Inspect score
    h2a_exploded_fuzzy_county_city_ne.sort_values('fuzzy_ne_score', ascending=False)
    return (h2a_exploded_fuzzy_county_city_ne,)


@app.cell
def _(h2a_exploded_fuzzy_county_city_ne, pd):
    # Convert score columns to numeric
    score_columns = ['fuzzy_county_score', 'fuzzy_city_score', 'fuzzy_ne_score']
    for col in score_columns:
        h2a_exploded_fuzzy_county_city_ne[col] = pd.to_numeric(h2a_exploded_fuzzy_county_city_ne[col], errors='coerce')
    return


@app.function
# Define a function that validates and picks the best FIPS code for each row
def pick_best_fips(row, county_name_fips, zip_fips, fuzzy_county_name_fips, fuzzy_county_name_score, fuzzy_city_name_fips, fuzzy_city_name_score, fuzzy_ne_name_fips, fuzzy_ne_name_score, state_col, fuzzy_score):
    # Priority order:
    # Special case: New England states - if fuzzy NE score is high, use it
    if row[state_col] in ['CT', 'ME', 'MA', 'NH', 'RI', 'VT'] and row[fuzzy_ne_name_fips] != '' and (row[fuzzy_ne_name_score] > fuzzy_score + 10):
        return row[fuzzy_ne_name_fips]
    #1. Census ZIP to county crosswalk
    if row[zip_fips] != '':
        return row[zip_fips]
    #2. addfips county name match
    elif row[county_name_fips] != '':
        return row[county_name_fips]
    #3. Fuzzy county name and fuzzy city name both have high scores and agree
    elif (row[fuzzy_county_name_fips] != '') and (row[fuzzy_city_name_fips] != '') and (row[fuzzy_county_name_fips] == row[fuzzy_city_name_fips]) and (row[fuzzy_county_name_score] > fuzzy_score) and (row[fuzzy_city_name_score] > fuzzy_score):
        return row[fuzzy_county_name_fips]
    #4. Fuzzy county name and fuzzy city name both have high scores but disagree
    elif (row[fuzzy_county_name_fips] != '') and (row[fuzzy_city_name_fips] != '') and (row[fuzzy_county_name_fips] != row[fuzzy_city_name_fips]) and (row[fuzzy_county_name_score] > fuzzy_score) and (row[fuzzy_city_name_score] > fuzzy_score):
        # Pick the one with higher score with higher threshold
        if row[fuzzy_county_name_score] >= row[fuzzy_city_name_score] and (row[fuzzy_county_name_score] > (fuzzy_score + 5)):
            return row[fuzzy_county_name_fips]
        elif row[fuzzy_city_name_score] > (fuzzy_score + 5):
            return row[fuzzy_city_name_fips]
    #5. Fuzzy county name match only
    elif row[fuzzy_county_name_fips] != '' and (row[fuzzy_county_name_score] > fuzzy_score + 10):
        return row[fuzzy_county_name_fips]
    #6. Fuzzy city name match only
    elif row[fuzzy_city_name_fips] != '' and (row[fuzzy_city_name_score] > fuzzy_score + 10):
        return row[fuzzy_city_name_fips]
    else:
        return ''


@app.cell
def _(h2a_exploded_fuzzy_county_city_ne):
    # Apply FIPS selection and validation function to rows of matched FIPS
    h2a_exploded_fuzzy_county_city_ne['final_fips'] = h2a_exploded_fuzzy_county_city_ne.apply(
        lambda _x: pick_best_fips(
            _x,
            'fips_from_addfips',
            'fips_from_census_zip',
            'fips_from_fuzzy_county',
            'fuzzy_county_score',
            'fips_from_fuzzy_city',
            'fuzzy_city_score',
            'fips_from_fuzzy_ne',
            'fuzzy_ne_score',
            'state',
            80
        ),
        axis=1
    )
    return


@app.cell
def _(h2a_exploded_fuzzy_county_city_ne):
    h2a_exploded_fuzzy_county_city_ne[h2a_exploded_fuzzy_county_city_ne['final_fips'] == '']
    return


@app.cell
def _(mo):
    mo.md(r"""
    For the final set of locations that are still unmatched, we will use Google's Places API to match them
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Start by finding the Place ID for each location using Find Place
    """)
    return


@app.cell
def _(h2a_exploded_fuzzy_county_city_ne):
    h2a_remaining_unmatched = h2a_exploded_fuzzy_county_city_ne[h2a_exploded_fuzzy_county_city_ne['final_fips'] == '']
    h2a_remaining_unmatched
    return (h2a_remaining_unmatched,)


@app.cell
def _(h2a_remaining_unmatched):
    # For these remaining unmatched entries, we will use Gemini to clean the location names and then use Google Places API to get the FIPS codes.
    # Export these remaining unmatched locations to CSV for processing externally.
    h2a_remaining_unmatched_locations = h2a_remaining_unmatched[h2a_remaining_unmatched['zip'] != ''][['city', 'county_list', 'state', 'zip']].drop_duplicates()
    h2a_remaining_unmatched_locations.to_csv('h2a_remaining_unmatched_locations.csv', index=False)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
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

    # for c in range(0, 14):
    #     h2a_chunk = h2a_unmatched_census[h2a_unmatched_census['chunk'] == c]

    #     # Dict to store API responses
    #     api_placeid_dict = {}

    #     for ind in range(0, len(h2a_chunk)):
    #         row = h2a_chunk.iloc[ind]
    #         id = row['id']
    #         state_name = row['state_name']
    #         place_name = row['place_list']
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
    #     with open(f'json/placeid_api_request_result_chunk_{c}.json', 'w') as f:
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
    for _c in range(0, 14):
        with open(f'json/placeid_api_request_result_chunk_{_c}.json', 'r') as _infile:
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

    # for c in range(0, 16):
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
    #     with open(f'json/place_details_api_request_result_chunk_{c}.json', 'w') as f:
    #         json.dump(api_place_details_dict, f)
    return


@app.cell
def _(json):
    # Load JSON of API responses and put into DataFrame
    api_place_details_dict = {}
    for _c in range(0, 16):
        with open(f'json/place_details_api_request_result_chunk_{_c}.json', 'r') as _infile:
            _api_dict = json.load(_infile)
        api_place_details_dict = api_place_details_dict | _api_dict
    return (api_place_details_dict,)


@app.cell
def _():
    # Store county name from place details into dictionary (store state names too as there may be incorrect states)
    county_name_dict = {}
    state_name_dict_1 = {}
    return county_name_dict, state_name_dict_1


@app.cell
def _(api_place_details_dict, county_name_dict, state_name_dict_1):
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
                    state_name_dict_1[_placeid] = state_name
    return


@app.cell
def _(api_placeid_df, county_name_dict, state_name_dict_1):
    # Add county and state name columns to Place ID
    api_placeid_df['county_name_api'] = api_placeid_df['placeid'].map(county_name_dict)
    api_placeid_df['state_name_api'] = api_placeid_df['placeid'].map(state_name_dict_1)
    return


@app.cell
def _(api_placeid_df):
    # Some of these multiple responses per place name are in the same county, so we can collapse those
    api_placeid_df_1 = api_placeid_df.drop_duplicates(subset=['id', 'county_name_api', 'state_name_api'])
    return (api_placeid_df_1,)


@app.cell
def _(api_placeid_df_1, h2a_unmatched_census_2):
    # For the remainder, manually resolve
    api_placeid_df_2 = api_placeid_df_1.merge(h2a_unmatched_census_2[['place_list', 'state_name', 'id']], how='left', on=['id'])
    multiple_response = api_placeid_df_2[api_placeid_df_2.duplicated(subset=['id'], keep=False)]
    multiple_response.to_csv('test.csv')
    return (api_placeid_df_2,)


@app.cell
def _(api_placeid_df_2):
    # Caddo, Texas is in Stephens County
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == '33 CADDO') & (api_placeid_df_2['state_name'] == 'Texas'), 'county_name_api'] = 'Stephens County'
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'ALICE') & (api_placeid_df_2['state_name'] == 'Tennessee'), 'county_name_api'] = None
    # Alice, Tennessee is ambiguous
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'BOX 78') & (api_placeid_df_2['state_name'] == 'Kansas'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'BRITTON') & (api_placeid_df_2['state_name'] == 'North Dakota'), 'county_name_api'] = 'Marshall County'
    # Box 78, Kansas?
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'CASA GRANDE') & (api_placeid_df_2['state_name'] == 'Arkansas'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'CLARK') & (api_placeid_df_2['state_name'] == 'North Dakota'), 'county_name_api'] = None
    # Britton, South Dakota is in Marshall County
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'COTHERTOWN') & (api_placeid_df_2['state_name'] == 'Tennessee'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'CRAIG') & (api_placeid_df_2['state_name'] == 'Wyoming'), 'county_name_api'] = None
    # There is no Casa Grande in Arkansas
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == ' COUNTY') & (api_placeid_df_2['state_name'] == 'South Carolina'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'GLADSTONE') & (api_placeid_df_2['state_name'] == 'South Dakota'), 'county_name_api'] = None
    # There is no Clark in North Dakota
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'HOLLIS') & (api_placeid_df_2['state_name'] == 'Massachusetts'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'IDAHO COUNTY') & (api_placeid_df_2['state_name'] == 'Nevada'), 'county_name_api'] = None
    # There is no Cothertown in Tennessee
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'LONE TREE') & (api_placeid_df_2['state_name'] == 'Michigan'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'MACK (MAIL)') & (api_placeid_df_2['state_name'] == 'Colorado'), 'county_name_api'] = None
    # There is no Craig in Wyoming
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'MACK(MAIL)') & (api_placeid_df_2['state_name'] == 'Colorado'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'MAMOIA') & (api_placeid_df_2['state_name'] == 'Louisiana'), 'county_name_api'] = 'Evangeline Parish'
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'MANNNING') & (api_placeid_df_2['state_name'] == 'North Carolina'), 'county_name_api'] = 'Nash County'
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'MAURICECHURCH POINT') & (api_placeid_df_2['state_name'] == 'Louisiana'), 'county_name_api'] = 'Acadia Parish, Vermilion Parish'
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'NORTH SPRING VALLEY') & (api_placeid_df_2['state_name'] == 'Utah'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'PARKERS PRAIRIE') & (api_placeid_df_2['state_name'] == 'Missouri'), 'county_name_api'] = None
    # Ambiguous
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'PORTON') & (api_placeid_df_2['state_name'] == 'Arizona'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'ROYNE') & (api_placeid_df_2['state_name'] == 'Louisiana'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'SAVERY') & (api_placeid_df_2['state_name'] == 'Colorado'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'STRATFORD') & (api_placeid_df_2['state_name'] == 'North Dakota'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'THREE RIVERS') & (api_placeid_df_2['state_name'] == 'Florida'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'UNORGNAIZED TOWNSHIPS: AROUND') & (api_placeid_df_2['state_name'] == 'Maine'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'VASS') & (api_placeid_df_2['state_name'] == 'Tennessee'), 'county_name_api'] = None
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'WLIESBURG') & (api_placeid_df_2['state_name'] == 'Virginia'), 'county_name_api'] = 'Charlotte County'
    api_placeid_df_2.loc[(api_placeid_df_2['place_list'] == 'WOLCOTT') & (api_placeid_df_2['state_name'] == 'Idaho'), 'county_name_api'] = None
    return


@app.cell
def _(api_placeid_df_2):
    # Recollapse after fixing
    api_placeid_df_3 = api_placeid_df_2.drop_duplicates(subset=['id', 'county_name_api', 'state_name'])
    return (api_placeid_df_3,)


@app.cell
def _(api_placeid_df_3, pd):
    # MAURICECHURCH POINT, Louisiana is actually 2 cities
    extra_row = api_placeid_df_3[api_placeid_df_3['place_list'] == 'MAURICECHURCH POINT'].copy()
    api_placeid_df_3.loc[api_placeid_df_3['place_list'] == 'MAURICECHURCH POINT', 'county_name_api'] = 'Acadia Parish'
    extra_row['county_name_api'] = 'Vermilion Parish'
    api_placeid_df_4 = pd.concat([api_placeid_df_3, extra_row])
    return (api_placeid_df_4,)


@app.cell
def _(af, api_placeid_df_4):
    api_placeid_df_5 = api_placeid_df_4[~api_placeid_df_4['county_name_api'].isna()].copy()
    api_placeid_df_5['fips_api'] = api_placeid_df_5.apply(lambda x: af.get_county_fips(_x['county_name_api'], state=_x['state_name']), axis=1)
    return (api_placeid_df_5,)


@app.cell
def _(api_placeid_df_5):
    # Drop API results that don't match states
    api_placeid_df_6 = api_placeid_df_5[api_placeid_df_5['state_name'] == api_placeid_df_5['state_name_api']]
    return (api_placeid_df_6,)


@app.cell
def _(api_placeid_df_6, h2a_unmatched_census_2):
    h2a_api_df = h2a_unmatched_census_2.merge(api_placeid_df_6[['id', 'county_name_api', 'fips_api']], how='left', on=['id'])
    h2a_api_df = h2a_api_df[~h2a_api_df['fips_api'].isna()].copy()
    h2a_api_df = h2a_api_df.groupby(['worksite_city', 'worksite_county', 'worksite_state', 'worksite_zip']).agg({'fips_api': lambda x: ','.join(_x)}).reset_index()
    return (h2a_api_df,)


@app.cell
def _(h2a_api_df, h2a_df_with_fips):
    # Add FIPS from API to the H-2A entries based on worksites
    h2a_df_final = h2a_df_with_fips.merge(h2a_api_df, how='left', on=['worksite_city', 'worksite_county', 'worksite_state', 'worksite_zip'])
    h2a_df_final = h2a_df_final.fillna(value='')
    h2a_df_final.loc[h2a_df_final['fips'] == '', 'fips'] = h2a_df_final.loc[h2a_df_final['fips'] == '', 'fips_api']
    return (h2a_df_final,)


@app.cell
def _(h2a_df_final):
    # Export binary
    h2a_df_final_1 = h2a_df_final.drop(columns=['fips_api'])
    h2a_df_final_1.to_parquet('../binaries/h2a_with_fips.parquet', index=False)
    return (h2a_df_final_1,)


@app.cell
def _(h2a_df_final_1):
    h2a_df_final_1['wage_unit'].drop_duplicates()
    return


@app.cell
def _(h2a_df_final_1):
    h2a_df_final_1[h2a_df_final_1['fiscal_year'] == '2021']
    return


@app.cell
def _(h2a_df_final_1):
    h2a_df_final_1[h2a_df_final_1['fiscal_year'] == '2022']
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
