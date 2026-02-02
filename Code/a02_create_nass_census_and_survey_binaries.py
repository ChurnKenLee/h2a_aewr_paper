# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.18.4",
#     "pyzmq>=27.1.0",
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
    import polars as pl
    import marimo as mo
    import pyarrow as pa
    import pyarrow.parquet as pq
    return Path, mo, pd, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Census of Agriculture binaries
    """)
    return


@app.cell
def _(pl):
    quickstats_pl_type_dict = {
        'SOURCE_DESC': pl.Categorical,
        'SECTOR_DESC': pl.Categorical,
        'GROUP_DESC': pl.Categorical,
        'COMMODITY_DESC': pl.Categorical,
        'CLASS_DESC': pl.Categorical,
        'PRODN_PRACTICE_DESC': pl.Categorical,
        'UTIL_PRACTICE_DESC': pl.Categorical,
        'STATISTICCAT_DESC': pl.Categorical,
        'UNIT_DESC': pl.Categorical,
        'SHORT_DESC': pl.Categorical,
        'DOMAIN_DESC': pl.Categorical,
        'DOMAINCAT_DESC': pl.Categorical,
        'AGG_LEVEL_DESC': pl.Categorical,
        'STATE_ANSI': pl.Categorical,
        'STATE_FIPS_CODE': pl.Categorical,
        'STATE_ALPHA': pl.Categorical,
        'STATE_NAME': pl.Categorical,
        'ASD_CODE': pl.Categorical,
        'ASD_DESC': pl.Categorical,
        'COUNTY_ANSI': pl.Categorical,
        'COUNTY_CODE': pl.Categorical,
        'COUNTY_NAME': pl.Categorical,
        'REGION_DESC': pl.Categorical,
        'ZIP_5': pl.Categorical,
        'WATERSHED_CODE': pl.Categorical,
        'WATERSHED_DESC': pl.Categorical,
        'CONGR_DISTRICT_CODE': pl.Categorical,
        'COUNTRY_CODE': pl.Categorical,
        'COUNTRY_NAME': pl.Categorical,
        'LOCATION_DESC': pl.Categorical,
        'YEAR': pl.Int64,
        'FREQ_DESC': pl.Categorical,
        'BEGIN_CODE': pl.Categorical,
        'END_CODE': pl.Categorical,
        'REFERENCE_PERIOD_DESC': pl.Categorical,
        'WEEK_ENDING': pl.Categorical,
        'LOAD_TIME': pl.String,
        'VALUE': pl.String,
        'CV_%': pl.String,
    }

    nass_qs_list = [
        'animals_products',
        'crops',
        'economics',
        'demographics'
        ]
    # don't include 'environmental' due to encoding error for now
    return (nass_qs_list,)


@app.cell
def _():
    quickstats_pd_type_dict = {
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
        'STATE_ANSI': 'category',
        'STATE_FIPS_CODE': 'category',
        'STATE_ALPHA': 'category',
        'STATE_NAME': 'category',
        'ASD_CODE': 'category',
        'ASD_DESC': 'category',
        'COUNTY_ANSI': 'category',
        'COUNTY_CODE': 'category',
        'COUNTY_NAME': 'category',
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
    return (quickstats_pd_type_dict,)


@app.cell
def _():
    # year_list = [2002, 2007, 2012, 2017, 2022]
    # for year in year_list:
    #     census_archive_path = f'../Data/census_of_agriculture/qs.census{year}.txt.gz'
    #     parquet_output_path = f'../binaries/census_of_agriculture_{year}.parquet'
    #     print(f'Exporting year {year}')
    #     _df = pd.read_csv(census_archive_path, compression = 'gzip', sep = '\t', skiprows=0, header=0, dtype=quickstats_pd_type_dict, parse_dates=['LOAD_TIME'])
    #     _df.to_parquet(parquet_output_path)
    return


@app.cell
def _():
    # # Export as binaries
    # quickstats_folder = Path("../Data/quickstats")
    # for qs_type in nass_qs_list:
    #     for file_path in quickstats_folder.iterdir():
    #         file_name = file_path.name
    #         if qs_type in file_name:
    #             print(file_path)
    #             _df = pl.read_csv(file_path, separator='\t', has_header=True, schema=quickstats_pl_type_dict)
            
    #             _census_df = _df.filter(pl.col('SOURCE_DESC') == 'CENSUS')
    #             _census_df.write_parquet(f"../binaries/qs_census_{qs_type}.parquet")

    #             _survey_df = _df.filter(pl.col('SOURCE_DESC') == 'SURVEY')
    #             _survey_df.write_parquet(f"../binaries/qs_survey_{qs_type}.parquet")
    return


@app.cell
def _(Path, nass_qs_list, pd, quickstats_pd_type_dict):
    # Export as binaries
    quickstats_folder = Path("../Data/quickstats")
    for qs_type in nass_qs_list:
        for file_path in quickstats_folder.iterdir():
            file_name = file_path.name
            if qs_type in file_name:
                print(file_path)
                _df = pd.read_csv(file_path, compression = 'gzip', sep = '\t', skiprows=0, header=0, dtype=quickstats_pd_type_dict, parse_dates=['LOAD_TIME'])

                _census_df = _df[_df['SOURCE_DESC'] == 'CENSUS']
                _survey_df = _df[_df['SOURCE_DESC'] == 'SURVEY']
                # Export fixed parquet
                _census_df.to_parquet(f"../binaries/qs_census_{qs_type}.parquet")
                _survey_df.to_parquet(f"../binaries/qs_survey_{qs_type}.parquet")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
