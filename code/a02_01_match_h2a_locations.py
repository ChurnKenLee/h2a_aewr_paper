import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import pyprojroot
    import polars as pl
    import numpy as np
    import json
    import re
    import addfips
    import requests
    import urllib
    import time
    DC_STATEHOOD = 1 # Enables DC to be included in the state list
    import us
    import pickle
    import rapidfuzz
    from functools import partial

    return Path, addfips, json, mo, pl, pyprojroot, rapidfuzz, re, us


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Extract all H-2A performance files and Census location crosswalks
    """)
    return


@app.cell
def _(pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    return (root_path,)


@app.cell
def _(pl):
    # Helper function to read and uppercase all string columns
    def read_census_file(path):
        df = (
            pl.read_csv(path, separator='|', infer_schema=False)
                .with_columns(
                    pl.col(pl.String).str.to_uppercase()
                )
        )
        return df

    return (read_census_file,)


@app.cell
def _(root_path):
    # Load and process Census files
    census_code_path = root_path / 'data' / 'census_geography_codes'
    return (census_code_path,)


@app.cell
def _(re):
    # Strip suffixes from locality names for fuzzy string matching later
    place_suffixes = [
        ' CITY',
        ' TOWN',
        ' VILLAGE',
        ' CDP',
        ' MUNICIPALITY',
        ' BOROUGH',
        ' TOWNSHIP',
        ' CENSUS AREA',
        ' CENSUS DESIGNATED PLACE',
        ' COUNTY',
        ' PARISH'
    ]

    # Construct the regex pattern: (?:suffix1|suffix2|...)$
    # re.escape is used to handle suffixes with special characters like '.'
    place_suffix_pattern = f'(?:{'|'.join(re.escape(s) for s in place_suffixes)})$'
    return (place_suffix_pattern,)


@app.cell
def _(census_code_path, pl, place_suffix_pattern, read_census_file):
    # Census county names
    census_county = read_census_file(census_code_path / 'national_county2020.txt')
    census_county = (
        census_county
            .with_columns(
                (pl.col('STATEFP') + pl.col('COUNTYFP')).alias('fips')
            )
            .select(
                [
                    pl.col('STATE').alias('state'),
                    pl.col('COUNTYNAME').alias('county'),
                    pl.col('fips')
                ]
            )
            .filter(
                pl.col('county') != ''
            ).with_columns(
                pl.col('county').str.replace_all(place_suffix_pattern, '')
            )
    )
    return (census_county,)


@app.cell
def _(census_code_path, pl, place_suffix_pattern, read_census_file):
    # Census counties aggregated to Census places
    census_placebycounty = read_census_file(census_code_path / 'national_place_by_county2020.txt')
    census_place_agg = (
        census_placebycounty
            .with_columns(
                (pl.col('STATEFP') + pl.col('COUNTYFP')).alias('fips')
            )
            .group_by(['STATE', 'COUNTYNAME', 'PLACENAME'])
            .agg(pl.col('fips').str.join(','))
            .filter(pl.col('PLACENAME') != '')
            .rename(
                {
                    'STATE': 'state',
                    'COUNTYNAME': 'county',
                    'PLACENAME': 'place'
                }
            ).with_columns(
                pl.col('place').str.replace_all(place_suffix_pattern, '')
            )
    )
    return (census_place_agg,)


@app.cell
def _(census_code_path, pl, read_census_file):
    # Census counties aggregated to ZIP
    census_zip = read_census_file(census_code_path / 'tab20_zcta520_county20_natl.txt')
    census_zip_agg = (
        census_zip
            .group_by('GEOID_ZCTA5_20')
            .agg(pl.col('GEOID_COUNTY_20').str.join(',')
                )
            .filter(pl.col('GEOID_ZCTA5_20') != '')
            .rename(
                {
                    'GEOID_ZCTA5_20': 'zip',
                    'GEOID_COUNTY_20': 'fips'
                }
            )
    )
    return (census_zip_agg,)


@app.cell
def _(pl):
    # Helper to create a state-based lookup with parallel lists
    def create_ref_lookup(df, name_col):
        lookup = {}
        # Group by state and get lists of names and fips
        grouped = df.group_by(
            'state'
        ).agg(
            [
                pl.col(name_col).alias('names'),
                pl.col('fips')
            ]
        ).to_dicts()
    
        for row in grouped:
            lookup[row['state']] = {
                'names': row['names'],
                'fips': row['fips']
            }
        return lookup

    return (create_ref_lookup,)


@app.cell
def _(census_county, census_place_agg, create_ref_lookup):
    # Create state-based maps for rapidfuzz to match
    census_county_ref_map = create_ref_lookup(census_county, "county")
    census_place_ref_map = create_ref_lookup(census_place_agg, "place")
    return census_county_ref_map, census_place_ref_map


@app.cell
def _(load_workbook):
    def get_excel_column_names(excel_file_path, sheet_name = None):
        wb = load_workbook(excel_file_path, read_only=True)

        # Select the active sheet if none is specified
        if sheet_name is None:
            sheet = wb.active
        else:
            sheet = wb[sheet_name]

        # Iterate through the first row only to get the headers
        for row in sheet.iter_rows(min_row=1, max_row=1, values_only=True):
            return list(row)

        return []

    return


@app.cell
def _(root_path):
    h2a_path = root_path / 'data' / 'h2a'
    return (h2a_path,)


@app.cell
def _():
    # Get column names of all the H-2A Excel files
    # Names of H-2A program disclosure files from the DOL-OFLC
    h2a_filenames_dict = {
        2008:'H2A_FY2008.xlsx',
        2009:'H2A_FY2009.xlsx',
        2010:'H-2A_FY2010.xlsx',
        2011:'H-2A_FY2011.xlsx',
        2012:'H-2A_FY2012.xlsx',
        2013:'H2A_FY2013.xls',
        2014:'H-2A_FY14_Q4.xlsx',
        2015:'H-2A_Disclosure_Data_FY15_Q4.xlsx',
        2016:'H-2A_Disclosure_Data_FY16_updated.xlsx',
        2017:'H-2A_Disclosure_Data_FY17.xlsx',
        2018:'H-2A_Disclosure_Data_FY2018_EOY.xlsx',
        2019:'H-2A_Disclosure_Data_FY2019.xlsx',
        2020:'H-2A_Disclosure_Data_FY2020.xlsx',
        2021:'H-2A_Disclosure_Data_FY2021.xlsx',
        2022:'H-2A_Disclosure_Data_FY2022_Q4.xlsx',
        2023:'H-2A_Disclosure_Data_FY2023_Q4.xlsx',
        2024:'H-2A_Disclosure_Data_FY2024_Q4.xlsx'
    }

    # Names of H-2A program Addendum B files from the DOL-OFLC
    add_b_filenames_dict = {
        2020:'H-2A_FY2020_AddendumB_Employment.xlsx',
        2021:'H-2A_Addendum_B_Employment_FY2021.xlsx',
        2022:'H-2A_Addendum_B_Employment_Record_FY2022_Q4.xlsx',
        2023:'H-2A_Addendum_B_Employment_Record_FY2023_Q4.xlsx',
        2024:'H-2A_Addendum_B_Employment_Record_FY2024_Q4.xlsx'
    }
    return add_b_filenames_dict, h2a_filenames_dict


@app.cell
def _(pl):
    # Define the set of columns we want to add and remove over the years
    def mutate_list(base, add, remove):
        final = base + add
        if remove:
            final = [s for s in final if s not in remove]
        return final

    def s_map(cols):
        s_dict = {}
        for c in cols:
            if 'DATE' in c:
                s_dict[c] = pl.Date
            else:
                s_dict[c] = pl.String
        return s_dict

    add_cols_dict = {}
    remove_cols_dict = {}

    add_cols_dict[2011] = [
        'BASIC_NUMBER_OF_HOURS',
        'NBR_WORKERS_REQUESTED',
        'REQUESTED_END_DATE_OF_NEED',
        'REQUESTED_START_DATE_OF_NEED'
    ]
    remove_cols_dict[2013] = [
        'NBR_WORKERS_REQUESTED'
    ]
    add_cols_dict[2014] = [
        'WORKSITE_LOCATION_CITY',
        'WORKSITE_LOCATION_STATE'
    ]
    remove_cols_dict[2014] =[
        'ALIEN_WORK_CITY',
        'ALIEN_WORK_STATE'
    ]
    add_cols_dict[2015] = [
        'CASE_NUMBER',
        'NBR_WORKERS_REQUESTED',
        'WORKSITE_CITY',
        'WORKSITE_POSTAL_CODE',
        'WORKSITE_STATE'
    ]
    remove_cols_dict[2015] = [
        'CASE_NO',
        'REQUESTED_END_DATE_OF_NEED',
        'REQUESTED_START_DATE_OF_NEED',
        'WORKSITE_LOCATION_CITY',
        'WORKSITE_LOCATION_STATE'
    ]
    add_cols_dict[2016] = [
        'JOB_END_DATE',
        'JOB_START_DATE',
        'PRIMARY/SUB',
        'REQUESTED_END_DATE_OF_NEED',
        'REQUESTED_START_DATE_OF_NEED'
    ]
    remove_cols_dict[2016] = [
        'CERTIFICATION_BEGIN_DATE',
        'CERTIFICATION_END_DATE'
    ]
    add_cols_dict[2017] = [
        'WORKSITE_COUNTY'
    ]
    add_cols_dict[2018] = [
        'CASE_NO',
        'PRIMARY_SUB'
    ]
    remove_cols_dict[2018] = [
        'CASE_NUMBER',
        'PRIMARY/SUB'
    ]
    add_cols_dict[2019] = [
        'CASE_NUMBER',
        'PRMARY/SUB'
    ]
    remove_cols_dict[2019] = [
        'CASE_NO',
        'PRIMARY_SUB'
    ]
    add_cols_dict[2020] = [
        'ANTICIPATED_NUMBER_OF_HOURS',
        'EMPLOYMENT_BEGIN_DATE',
        'EMPLOYMENT_END_DATE',
        'H2A_LABOR_CONTRACTOR',
        'PER',
        'REQUESTED_BEGIN_DATE',
        'REQUESTED_END_DATE',
        'TOTAL_WORKERS_H2A_CERTIFIED',
        'TOTAL_WORKERS_H2A_REQUESTED',
        'TOTAL_WORKERS_NEEDED',
        'TYPE_OF_EMPLOYER_APPLICATION',
        'WAGE_OFFER'
    ]
    remove_cols_dict[2020] = [
        'BASIC_NUMBER_OF_HOURS',
        'BASIC_RATE_OF_PAY',
        'BASIC_UNIT_OF_PAY',
        'JOB_END_DATE',
        'JOB_START_DATE',
        'NBR_WORKERS_CERTIFIED',
        'NBR_WORKERS_REQUESTED',
        'ORGANIZATION_FLAG',
        'PRMARY/SUB',
        'REQUESTED_END_DATE_OF_NEED',
        'REQUESTED_START_DATE_OF_NEED'
    ]
    add_cols_dict[2024] = [
        'AG_ASSN_OR_AGENCY_STATUS'
    ]
    return add_cols_dict, mutate_list, remove_cols_dict, s_map


@app.cell
def _(add_cols_dict, mutate_list, remove_cols_dict, s_map):
    # Generate set of schemas
    cols_dict = {}
    schema_dict = {}

    cols_dict[2008] = [
        'CASE_NO',
        'CASE_STATUS',
        'EMPLOYER_NAME',
        'EMPLOYER_CITY',
        'EMPLOYER_STATE',
        'EMPLOYER_POSTAL_CODE',
        'NBR_WORKERS_CERTIFIED',
        'CERTIFICATION_BEGIN_DATE',
        'CERTIFICATION_END_DATE',
        'BASIC_RATE_OF_PAY',
        'BASIC_UNIT_OF_PAY',
        'ALIEN_WORK_CITY',
        'ALIEN_WORK_STATE',
        'ORGANIZATION_FLAG'
    ]
    schema_dict[2008] = s_map(cols_dict[2008])

    for _y in range(2009, 2025):
        add_list = add_cols_dict.get(_y, [])
        remove_list = remove_cols_dict.get(_y, [])

        cols_dict[_y] = mutate_list(cols_dict[_y-1], add_list, remove_list)  
        schema_dict[_y] = s_map(cols_dict[_y])
    return cols_dict, schema_dict


@app.cell
def _(s_map):
    # Columns and schema from the Addendum B files
    add_b_cols_dict = {}
    add_b_schema_dict = {}

    add_b_cols_dict[2020] = [
        'CASE_NUMBER',
        'NAME_OF_AGRICULTURAL_BUSINESS',
        'PLACE_OF_EMPLOYMENT_CITY',
        'PLACE_OF_EMPLOYMENT_STATE',
        'PLACE_OF_EMPLOYMENT_POSTAL_CODE',
        'TOTAL_WORKERS'
    ]
    add_b_cols_dict[2021] = add_b_cols_dict[2020]
    add_b_cols_dict[2022] = add_b_cols_dict[2020]
    add_b_cols_dict[2023] = add_b_cols_dict[2020]
    add_b_cols_dict[2024] = add_b_cols_dict[2020]

    for _y, _cols in add_b_cols_dict.items():
        add_b_schema_dict[_y] = s_map(_cols)
    return add_b_cols_dict, add_b_schema_dict


@app.cell
def _():
    # Define common column names
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

    # Common names in Addendum B files
    add_b_rename_dict = {
        'CASE_NUMBER':'case_number',
        'NAME_OF_AGRICULTURAL_BUSINESS':'business_name',
        'PLACE_OF_EMPLOYMENT_CITY':'worksite_city',
        'PLACE_OF_EMPLOYMENT_STATE':'worksite_state',
        'PLACE_OF_EMPLOYMENT_POSTAL_CODE':'worksite_zip',
        'TOTAL_WORKERS':'total_h2a_workers_requested'
    }
    return add_b_rename_dict, h2a_rename_dict


@app.cell
def _(mo, pl):
    # Define a function that reads the Excel files using polars, with Marimo caching for speedy code iteration
    @mo.persistent_cache
    def read_h2a_excel(path, cols, schema, rename_map):
        df = pl.read_excel(
            source=path,
            columns=cols,
            schema_overrides=schema,
            engine='calamine'
        )

        df = df.with_columns(
            pl.col(pl.Date).dt.to_string(),
            pl.col(pl.String).fill_null(''),
        ).rename(
            mapping=rename_map, strict=False
        )

        return df

    return (read_h2a_excel,)


@app.cell
def _(
    add_b_cols_dict,
    add_b_filenames_dict,
    add_b_rename_dict,
    add_b_schema_dict,
    cols_dict,
    h2a_filenames_dict,
    h2a_path,
    h2a_rename_dict,
    pl,
    read_h2a_excel,
    schema_dict,
):
    # Read the H-2A performance data files
    h2a_df = pl.DataFrame()
    for _y in range(2008, 2025):
        _path = h2a_path / h2a_filenames_dict[_y]
        print(f'Reading H-2A for year {_y}')
        _df = read_h2a_excel(_path, cols_dict[_y], schema_dict[_y], h2a_rename_dict)

        _df = _df.with_columns(
            fiscal_year=_y
        )
        h2a_df = pl.concat(items=[h2a_df, _df], how='diagonal')

    # Same with Addendum B files
    # Read the H-2A performance data files
    add_b_df = pl.DataFrame()
    for _y in range(2020, 2025):
        _path = h2a_path / add_b_filenames_dict[_y]
        print(f'Reading Addendum B for year {_y}')
        _df = read_h2a_excel(_path, add_b_cols_dict[_y], add_b_schema_dict[_y], add_b_rename_dict)

        _df = _df.with_columns(
            fiscal_year=_y
        )

        add_b_df = pl.concat(items=[add_b_df, _df], how='diagonal')
    return add_b_df, h2a_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Add FIPS codes using worksite location information with local data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Given input location information (city, county, state, ZIP), we want to write functions that finds FIPS codes
    """)
    return


@app.cell
def _(add_b_df, h2a_df, pl):
    # Set of unique worksite locations
    h2a_worksite_locations = (
        h2a_df.select(
            ['worksite_city', 'worksite_county', 'worksite_state', 'worksite_zip']
        ).unique()
    )
    add_b_worksite_locations = (
        add_b_df.select(
            ['worksite_city', 'worksite_state', 'worksite_zip']
        )
        .unique()
    )
    # For Addendum B, we need to add the non-existent county column
    add_b_worksite_locations = add_b_worksite_locations.with_columns(pl.lit("").alias('worksite_county'))
    return add_b_worksite_locations, h2a_worksite_locations


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These are the sets of functions that are used to perform local matching of FIPS
    """)
    return


@app.cell
def _(pl):
    # Expand multi-county entries
    def clean_county_explode(df, county_col, state_col):
        # Replace separators and explode
        df = df.with_columns(
            pl.col(county_col)
            .str.replace_all(' AND ', ',', literal=True)
            .str.replace_all(' & ', ',', literal=True)
            .str.replace_all('/', ',', literal=True)
            .str.split(',')
            .alias('county_list')
        ).explode('county_list')

        # Clean whitespace and suffixes
        df = df.with_columns(
            pl.col('county_list').str.strip_chars()
        ).with_columns(
            pl.col('county_list')
            .str.replace_all(' COUNTY', '', literal=True)
            .str.replace_all(' COUNTIES', '', literal=True)
            .str.replace_all(' PARISH', '', literal=True)
            .str.replace_all(' PARRISH', '', literal=True)
        )

        # Fix Louisiana typos using a series of replaces (native Polars is faster than a loop)
        # Only apply where state is LA
        la_typos = {
            'ST\\.\\w': 'ST. ',
            'NORTH\\. ': 'NORTH ',
            'SOUTH\\. ': 'SOUTH ',
            'EAST\\. ': 'EAST ',
            'WEST\\. ': 'WEST ',
            'BATON ROGUE': 'BATON ROUGE',
            'JEFF DAVIS': 'JEFFERSON DAVIS',
            'IBERIAL': 'IBERIA'
        }
    
        df = df.with_columns(
            pl.when(pl.col(state_col) == 'LA')
            .then(pl.col('county_list').str.replace_many(la_typos))
            .otherwise(pl.col('county_list'))
            .alias('county_list')
        )

        # Fill nulls
        df = df.with_columns(
            pl.col(pl.String).fill_null('')
        )

        return df

    return (clean_county_explode,)


@app.cell
def _(pl):
    # Cleans ZIP codes, mostly by removing trailing 4 digits for ZIP+4 codes
    def clean_zip(df, zip_col):
        zip5 = df.with_columns(
            pl.col(zip_col)
            .str.replace_all('.', '', literal=True)
            .str.split('-').list.get(0)
            .alias(zip_col)
        )
        return zip5

    return (clean_zip,)


@app.cell
def _(addfips, pl):
    # Define a custom AddFIPS function that operates on a polars dataframe
    def add_fips_using_addfips(df, county_col, state_col, target_col):
        # Instantiate the library ONCE here
        af = addfips.AddFIPS()
    
        # The 'row' object passed from pl.struct is a dictionary
        return df.with_columns(
            pl.struct(
                [county_col, state_col]
            ).map_elements(
                lambda row: af.get_county_fips(county=row[county_col], state=row[state_col]) or '',
                return_dtype=pl.String
            ).alias(target_col)
        )

    return (add_fips_using_addfips,)


@app.cell
def _(rapidfuzz):
    # Uses rapidfuzz.process.extractOne and name maps to quickly find locality with best name match
    def fuzz_search(state, query, ref_map):
        # Sanity checks
        if not state or not query or state not in ref_map:
            return ("", 0.0, "")
    
        state_data = ref_map[state]
        choices = state_data["names"]
    
        # Rapidfuzz handles strings directly now
        # processor=utils.default_process handles lowercasing/stripping/whitespace
        match = rapidfuzz.process.extractOne(
            query, 
            choices, 
            processor=rapidfuzz.utils.default_process, 
            scorer=rapidfuzz.fuzz.partial_ratio
        )
    
        if match:
            # match is (found_string, score, index)
            _, score, index = match
            best_fips = state_data["fips"][index]
            best_name = state_data["names"][index]
            return (str(best_fips), float(score), str(best_name))
    
        else:
            return ("", 0.0, "")

    return (fuzz_search,)


@app.cell
def _(us):
    # Function cleans state names using the US package and returns 2 character abbreviations
    # Returns empty string if no match; DC is included
    # Needed for using fuzzy match function because Census sources use abbreviations
    def clean_state_to_abbr(input):
        if input == '':
            return ''

        # Include DC
        if str(input).upper() == 'DISTRICT OF COLUMBIA':
            return 'DC'
        if str(input).upper() == 'DC':
            return 'DC'

        # US package lookup can handle names, abbreviations, FIPs
        state_obj = us.states.lookup(str(input).strip())

        if state_obj:
            return state_obj.abbr
        else: 
            return ''

    return (clean_state_to_abbr,)


@app.cell
def _(
    add_fips_using_addfips,
    census_county_ref_map,
    census_place_ref_map,
    clean_county_explode,
    clean_state_to_abbr,
    clean_zip,
    fuzz_search,
    mo,
    pl,
):
    # Apply functions in sequence to an input dataframe and produce dataframe with matched FIPS
    @mo.persistent_cache
    def add_fips_with_addfips_and_fuzzy_matching(df, county_col, city_col, state_col, zip_col, census_zip_df, census_county_map, census_place_map):
        # Initialize helper columns
        df = df.with_columns([
            pl.col(city_col).alias('xcity'),
            pl.col(county_col).alias('xcounty'),
            pl.col(state_col).map_elements(clean_state_to_abbr, return_dtype=pl.String).alias('xstate'),
            pl.col(zip_col).alias('xzip')
        ])

        df = clean_county_explode(df, 'xcounty', 'xstate')
        df = clean_zip(df, 'xzip')

        # 1. Match via addfips
        df = add_fips_using_addfips(df, 'county_list', 'xstate', 'fips_from_addfips')

        # 2. Match via Census ZIP (Join instead of Map)
        df = df.join(
            census_zip_df.select(
                [
                    pl.col("zip").alias("xzip"),
                    pl.col("fips").alias("fips_from_census_zip")
                ]
            ),
            on="xzip",
            how="left"
        ).with_columns(
            pl.col("fips_from_census_zip").fill_null("")
        )

        # 3. Fuzzy Matching
        # Apply fuzzy logic
        df = df.with_columns([
            pl.struct(["county_list", "xstate"])
                .map_elements(
                    lambda x: fuzz_search(x['xstate'], x['county_list'], census_county_ref_map),
                    return_dtype=pl.Object
                ).alias("county_fuzzy_raw"),
            pl.struct(["xcity", "xstate"])
                .map_elements(
                    lambda x: fuzz_search(x['xstate'], x['xcity'], census_place_ref_map),
                    return_dtype=pl.Object
                ).alias("city_fuzzy_raw"),
            pl.struct(["county_list", "xstate"])
                .map_elements(
                    lambda x: fuzz_search(x['xstate'], x['county_list'], census_place_ref_map),
                    return_dtype=pl.Object
                ).alias("ne_fuzzy_raw")
        ])
        # Expand the fuzzy results into individual columns
        df = df.with_columns([
            pl.col("county_fuzzy_raw").map_elements(lambda x: x[0], return_dtype=pl.String).alias("fips_from_fuzzy_county"),
            pl.col("county_fuzzy_raw").map_elements(lambda x: x[1], return_dtype=pl.Float64).alias("fuzzy_county_score"),
            pl.col("city_fuzzy_raw").map_elements(lambda x: x[0], return_dtype=pl.String).alias("fips_from_fuzzy_city"),
            pl.col("city_fuzzy_raw").map_elements(lambda x: x[1], return_dtype=pl.Float64).alias("fuzzy_city_score"),
            pl.col("ne_fuzzy_raw").map_elements(lambda x: x[0], return_dtype=pl.String).alias("fips_from_fuzzy_ne"),
            pl.col("ne_fuzzy_raw").map_elements(lambda x: x[1], return_dtype=pl.Float64).alias("fuzzy_ne_score"),
        ]).drop(["county_fuzzy_raw", "city_fuzzy_raw", "ne_fuzzy_raw"])

        return df

    return (add_fips_with_addfips_and_fuzzy_matching,)


@app.cell
def _(pl):
    def pick_best_fips(df, fuzzy_threshold):
        # 4. Define FIPS selection logic
    
        df = df.with_columns(pl
            .when(pl.col("fips_from_census_zip") != "")
            .then(pl.col("fips_from_census_zip"))
            .when(pl.col("fips_from_addfips") != "")
            .then(pl.col("fips_from_addfips"))
            # New England puts county name in city column so has extra match check
            .when(
                (pl.col("xstate").is_in(['CT', 'ME', 'MA', 'NH', 'RI', 'VT'])) & 
                (pl.col("fips_from_fuzzy_ne") != "") & 
                (pl.col("fuzzy_ne_score") > (fuzzy_threshold + 10))
            )
            .then(pl.col("fips_from_fuzzy_ne"))
            # Use lower threshold fuzzy match result if matches agree
            .when(
                (pl.col("fips_from_fuzzy_county") != "") & 
                (pl.col("fips_from_fuzzy_city") != "") & 
                (pl.col("fips_from_fuzzy_county") == pl.col("fips_from_fuzzy_city")) & 
                (pl.col("fuzzy_county_score") > fuzzy_threshold) & 
                (pl.col("fuzzy_city_score") > fuzzy_threshold)
            )
            .then(pl.col("fips_from_fuzzy_county"))
            # Use higher threshold for fuzzy match result if matches disagree
            .when(
                (pl.col("fips_from_fuzzy_county") != "") & 
                (pl.col("fips_from_fuzzy_city") != "") & 
                (pl.col("fuzzy_county_score") > fuzzy_threshold + 5) & 
                (pl.col("fuzzy_city_score") > fuzzy_threshold) &
                (pl.col("fuzzy_county_score") > pl.col("fuzzy_city_score"))
            )
            .then(pl.col("fips_from_fuzzy_county"))
            .when(
                (pl.col("fips_from_fuzzy_county") != "") & 
                (pl.col("fips_from_fuzzy_city") != "") & 
                (pl.col("fuzzy_county_score") > fuzzy_threshold) & 
                (pl.col("fuzzy_city_score") > fuzzy_threshold + 5) &
                (pl.col("fuzzy_city_score") > pl.col("fuzzy_county_score"))
            )
            .then(pl.col("fips_from_fuzzy_city"))
            # Use highest threshold for singular fuzzy matches
            .when(
                (pl.col("fips_from_fuzzy_county") != "") &
                (pl.col("fuzzy_county_score") > fuzzy_threshold + 10)
            )
            .then(pl.col("fips_from_fuzzy_county"))
            .when(
                (pl.col("fips_from_fuzzy_city") != "") &
                (pl.col("fuzzy_city_score") > fuzzy_threshold + 10)
            )
            .then(pl.col("fips_from_fuzzy_city"))
            .otherwise(pl.lit(""))
            .alias("final_fips")
        )

        return df

    return (pick_best_fips,)


@app.cell
def _(
    add_b_worksite_locations,
    add_fips_with_addfips_and_fuzzy_matching,
    census_county_ref_map,
    census_place_ref_map,
    census_zip_agg,
    h2a_worksite_locations,
    pick_best_fips,
):
    # Match FIPS for H-2A worksites
    h2a_worksite_locations_added_fips = add_fips_with_addfips_and_fuzzy_matching(
        h2a_worksite_locations,
        'worksite_county', 'worksite_city', 'worksite_state', 'worksite_zip',
        census_zip_agg, census_county_ref_map, census_place_ref_map
    )
    h2a_worksite_locations_added_fips = pick_best_fips(h2a_worksite_locations_added_fips, 80)

    add_b_worksite_locations_added_fips = add_fips_with_addfips_and_fuzzy_matching(
        add_b_worksite_locations,
        'worksite_county', 'worksite_city', 'worksite_state', 'worksite_zip',
        census_zip_agg, census_county_ref_map, census_place_ref_map
    )
    add_b_worksite_locations_added_fips = pick_best_fips(add_b_worksite_locations_added_fips, 80)
    return (
        add_b_worksite_locations_added_fips,
        h2a_worksite_locations_added_fips,
    )


@app.cell
def _(h2a_worksite_locations_added_fips, pl):
    h2a_worksite_locations_added_fips.filter(pl.col('final_fips') == '')
    return


@app.cell
def _(add_b_worksite_locations_added_fips, pl):
    add_b_worksite_locations_added_fips.filter(pl.col('final_fips') == '')
    return


@app.cell
def _(
    Path,
    add_b_worksite_locations_added_fips,
    h2a_worksite_locations_added_fips,
    pl,
    root_path,
):
    # Export unmatched
    def export_unmatched_locations_pl(df, filename):
        (
            df
            .filter(pl.col('final_fips') == "")
            .select([
                pl.col('xcity').alias('city'),
                pl.col('xstate').alias('state'),
                pl.col('xzip').alias('zip'),
                pl.col('county_list').alias('county')
            ])
            .unique()
            .write_csv(filename)
        )
    json_path = root_path / 'code' / 'json'
    export_unmatched_locations_pl(h2a_worksite_locations_added_fips, Path(json_path / 'unmatched_h2a_locations.csv'))
    export_unmatched_locations_pl(add_b_worksite_locations_added_fips,  Path(json_path / 'unmatched_add_b_locations.csv'))
    return (json_path,)


@app.cell
def _(mo):
    mo.md(r"""
    # Use Gemini + Places API to get matches for the unmatched locations, then come back here and use them to fill in the missing FIPS
    """)
    return


@app.cell
def _(Path, json, json_path, pl, root_path):
    # Load mappings we obtained
    with open(Path(json_path / 'placeid_address_components_mapping_json.json')) as _fp:
        placeid_address_components_map = json.load(_fp)

    binary_path = root_path / 'binaries'
    h2a_placeid = pl.read_parquet(binary_path / 'h2a_location_placeids.parquet')
    add_b_placeid = pl.read_parquet(binary_path / 'add_b_location_placeids.parquet')
    return add_b_placeid, h2a_placeid, placeid_address_components_map


@app.cell
def _(json):
    # Function that takes a list of Place IDs and returns a list of counties
    def placeid_to_county(placeids, placeid_address_components_map):
        county_list = []
        for placeid in placeids:
            if placeid not in placeid_address_components_map:
                print(f'Place ID {placeid} not in mapping')
                continue

            responses = json.loads(placeid_address_components_map[placeid])['addressComponents']
            for response in responses:
                if not 'types' in response: # We are not interested in street address components
                    continue
                response_type = response['types']

                if 'locality' in response_type and 'political' in response_type:
                    locality = response['longText']
                elif 'administrative_area_level_2' in response_type and 'political' in response_type:
                    county = response['longText']
                    county_list.append(county)
                elif 'administrative_area_level_1' in response_type and 'political' in response_type:
                    state = response['longText']
                else:
                    continue

        return county_list

    return (placeid_to_county,)


@app.cell
def _(
    add_b_placeid,
    h2a_placeid,
    placeid_address_components_map,
    placeid_to_county,
):
    h2a_placeid['original_county'] = h2a_placeid.apply(lambda _x: placeid_to_county(_x['original_location_placeid'], placeid_address_components_map), axis=1)
    h2a_placeid['suggested_county'] = h2a_placeid.apply(lambda _x: placeid_to_county(_x['suggested_location_placeid'], placeid_address_components_map), axis=1)

    add_b_placeid['original_county'] = add_b_placeid.apply(lambda _x: placeid_to_county(_x['original_location_placeid'], placeid_address_components_map), axis=1)
    add_b_placeid['suggested_county'] = add_b_placeid.apply(lambda _x: placeid_to_county(_x['suggested_location_placeid'], placeid_address_components_map), axis=1)

    # Function produces common counties returned via Places API calls on both original and Gemini-suggested location names
    def common_counties(list1, list2):
        return list(set(list1).intersection(list2))

    def common_state(state1, state2):
        if state1 == state2:
            return state1
        else:
            return ''

    h2a_placeid['common_county'] = h2a_placeid.apply(lambda _x: common_counties(_x['original_county'], _x['suggested_county']), axis = 1)
    # h2a_placeid['common_state'] = h2a_placeid.apply(lambda _x: common_state(_x['state_name'], _x['state_name_suggested']), axis = 1)
    add_b_placeid['common_county'] = add_b_placeid.apply(lambda _x: common_counties(_x['original_county'], _x['suggested_county']), axis = 1)
    # add_b_placeid['common_state'] = add_b_placeid.apply(lambda _x: common_state(_x['state_name'], _x['state_name_suggested']), axis = 1)
    return


@app.cell
def _(add_b_placeid, h2a_placeid):
    # Function that add FIPS using addfips
    def add_fips_to_places_api_county(county_list, state):
        import addfips
        af = addfips.AddFIPS()
        string_of_new_fips = ''
        for county_name in county_list:
            county_fips = af.get_county_fips(county_name, state)
            if not county_fips:
                county_fips = ''
            string_of_new_fips = string_of_new_fips + county_fips + ','

        string_of_new_fips = string_of_new_fips.rstrip(',')
        return string_of_new_fips

    h2a_placeid['common_county_fips'] = h2a_placeid.apply(lambda _x: add_fips_to_places_api_county(_x['common_county'], _x['state_name']), axis=1)
    add_b_placeid['common_county_fips'] = add_b_placeid.apply(lambda _x: add_fips_to_places_api_county(_x['common_county'], _x['state_name']), axis=1)
    return


@app.cell
def _(add_b_placeid, h2a_placeid):
    # Combine FIPs from Places API back to original dataset
    # Change name back to column names from original dataset
    h2a_places_api_fips = h2a_placeid.rename(columns = {
        'city': 'xcity',
        'county': 'county_list',
        'state': 'xstate',
        'zip': 'xzip',
        'common_county_fips': 'places_api_fips'
    })
    h2a_places_api_fips = h2a_places_api_fips[['xcity', 'xstate', 'xzip', 'county_list', 'places_api_fips']]

    add_b_places_api_fips = add_b_placeid.rename(columns = {
        'city': 'xcity',
        'county': 'county_list',
        'state': 'xstate',
        'zip': 'xzip',
        'common_county_fips': 'places_api_fips'
    })
    add_b_places_api_fips = add_b_places_api_fips[['xcity', 'xstate', 'xzip', 'county_list', 'places_api_fips']]
    return add_b_places_api_fips, h2a_places_api_fips


@app.cell
def _(
    add_b_places_api_fips,
    add_b_worksite_locations_added_fips,
    h2a_places_api_fips,
    h2a_worksite_locations_added_fips,
    pd,
):
    # Define new finalized FIPs, keep only columns we need, then collapse back into form before I exploded xcounty into county_list
    # Location are defined by the worksite columns, the ones prefixed with x are just copies of worksite columns that I modify for explosion
    h2a_worksite_locations_finalized = pd.merge(h2a_worksite_locations_added_fips, h2a_places_api_fips, how='left' ,on = ['xcity', 'xstate', 'county_list', 'xzip']).fillna('')
    h2a_worksite_locations_finalized = h2a_worksite_locations_finalized[['worksite_city', 'worksite_county', 'worksite_state', 'worksite_zip', 'final_fips', 'places_api_fips']]

    add_b_worksite_locations_finalized = pd.merge(add_b_worksite_locations_added_fips, add_b_places_api_fips, how='left' ,on = ['xcity', 'xstate', 'county_list', 'xzip']).fillna('')
    add_b_worksite_locations_finalized = add_b_worksite_locations_finalized[['worksite_city', 'worksite_county', 'worksite_state', 'worksite_zip', 'final_fips', 'places_api_fips']]

    # Finalized FIPS
    def fips_selector(final_fips, places_api_fips):
        if final_fips != '':
            return final_fips
        elif places_api_fips:
            return places_api_fips
        else:
            return ''

    h2a_worksite_locations_finalized['fips'] = h2a_worksite_locations_finalized.apply(lambda _x: fips_selector(_x['final_fips'], _x['places_api_fips']), axis=1)
    add_b_worksite_locations_finalized['fips'] = add_b_worksite_locations_finalized.apply(lambda _x: fips_selector(_x['final_fips'], _x['places_api_fips']), axis=1)

    #h2a_worksite_locations_finalized = h2a_worksite_locations_finalized.drop(columns = ['final_fips', 'places_api_fips'])
    #add_b_worksite_locations_finalized = add_b_worksite_locations_finalized.drop(columns = ['final_fips', 'places_api_fips'])
    return add_b_worksite_locations_finalized, h2a_worksite_locations_finalized


@app.cell
def _(add_b_worksite_locations_finalized, h2a_worksite_locations_finalized):
    h2a_worksite_locations_collapsed = h2a_worksite_locations_finalized.groupby(['worksite_city', 'worksite_county', 'worksite_state', 'worksite_zip']).agg({'fips': ','.join})
    add_b_worksite_locations_collapsed = add_b_worksite_locations_finalized.groupby(['worksite_city', 'worksite_state', 'worksite_zip']).agg({'fips': ','.join})
    return add_b_worksite_locations_collapsed, h2a_worksite_locations_collapsed


@app.cell
def _(
    add_b_df,
    add_b_worksite_locations_collapsed,
    h2a_df,
    h2a_worksite_locations_collapsed,
    pd,
):
    h2a_with_fips_df = pd.merge(h2a_df, h2a_worksite_locations_collapsed, on=['worksite_city', 'worksite_county', 'worksite_state', 'worksite_zip'], how='left')
    add_b_with_fips_df = pd.merge(add_b_df, add_b_worksite_locations_collapsed, on=['worksite_city', 'worksite_state', 'worksite_zip'], how='left')
    return add_b_with_fips_df, h2a_with_fips_df


@app.cell
def _(add_b_with_fips_df, h2a_with_fips_df):
    # Export binary
    h2a_with_fips_df.to_parquet('../binaries/h2a_with_fips.parquet', index=False)
    add_b_with_fips_df.to_parquet('../binaries/h2a_addendum_b_with_fips.parquet', index=False)
    return


@app.cell
def _(h2a_with_fips_df):
    h2a_with_fips_df[h2a_with_fips_df['nbr_workers_needed'] != '']
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
