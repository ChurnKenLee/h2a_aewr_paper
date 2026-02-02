import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    import pandas as pd
    import numpy as np
    from bs4 import BeautifulSoup
    import re
    return BeautifulSoup, Path, pd, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Get crop categorization codes from metadata file
    """)
    return


@app.cell
def _(BeautifulSoup, Path, pd, re):
    # Iterate over years and place crop codes in a DataFrame
    rows_list = []
    for _year in range(2008, 2024):
        metadata_htm_path = list(Path(f'../Data/croplandcros_cdl/{_year}_30m_cdls').glob('*.htm'))
        with open(metadata_htm_path[0]) as fp:
            soup = BeautifulSoup(fp)  # Extract table
        cat_table = soup.find_all('pre')[-1].string
        for line in cat_table.splitlines():
            crop_code_dict = {}
            result = re.search('\\"\\d*\\"', line)
            if result is None:
                continue  # The categorization codes for the crop types are in tables that are in preformatted blocks of text
            else:  # Get the last block, which should be the categorization table
                crop_code = re.search('\\"\\d*\\"', line)[0].strip('"')
                crop_name = re.search('[A-Z].*', line)[0]
                crop_code_dict['year'] = _year  # Extract lines with crop and categorization code
                crop_code_dict['crop_code'] = crop_code
                crop_code_dict['crop_name'] = crop_name
                rows_list.append(crop_code_dict)
        crop_code_dict = {}
        crop_code_dict['year'] = _year  # Regex match number enclosed in double quotes
        crop_code_dict['crop_code'] = '176'
        crop_code_dict['crop_name'] = 'Grassland/Pasture'
        rows_list.append(crop_code_dict)  # If line has categorization code, extract info from line
    crop_code_df = pd.DataFrame(rows_list)  # For years prior to 2022, code 176 is not defined for some reason
    return (crop_code_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Add crop names to county crop aggregates
    """)
    return


@app.cell
def _(pd):
    # Read CDL county aggregates
    cdl_df = pd.DataFrame()
    for _year in range(2008, 2024):
        df = pd.read_parquet(f'../binaries/county_crop_pixel_count_{_year}.parquet')
        df['year'] = _year
        cdl_df = pd.concat([cdl_df, df])
    return (cdl_df,)


@app.cell
def _(cdl_df, crop_code_df):
    # Coerce types to match
    crop_code_df['crop_code'] = crop_code_df['crop_code'].astype(int)
    cdl_df_1 = cdl_df.rename(columns={'crop': 'crop_code'})
    cdl_df_1['crop_code'] = cdl_df_1['crop_code'].astype(int)
    return (cdl_df_1,)


@app.cell
def _(cdl_df_1, crop_code_df):
    # Merge
    county_crop_df = cdl_df_1.merge(crop_code_df, how='left', on=['year', 'crop_code'])
    return (county_crop_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calculate acreage, clean FIPS code, then export
    """)
    return


@app.cell
def _(county_crop_df):
    # Convert pixels to acres
    # Each pixel is 30mx30m = 900m^2
    # 1 acre = 4046.86m^2
    # 900m^2 = 900/4046.86 acres = 0.22239464671 acres
    county_crop_df['acres'] = county_crop_df['pixel_count']*(900/4046.86)
    return


@app.cell
def _(county_crop_df):
    # Clean FIPS code
    county_crop_df['fips'] = county_crop_df['fips'].astype('string')
    county_crop_df['fips'] = county_crop_df['fips'].str.pad(width=5, side='left', fillchar='0')
    county_crop_df['state_fips_code'] = county_crop_df['fips'].str.slice(start=0, stop=2)
    county_crop_df['county_fips_code'] = county_crop_df['fips'].str.slice(start=2, stop=5)
    county_crop_df_1 = county_crop_df.drop(columns=['fips'])
    return (county_crop_df_1,)


@app.cell
def _(county_crop_df_1):
    # Save binary
    county_crop_df_1.to_parquet('../binaries/croplandcros_county_crop_acres.parquet')
    county_crop_df_1.to_parquet('../files_for_phil/croplandcros_county_crop_acres.parquet')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
