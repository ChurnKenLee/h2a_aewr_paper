import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import pandas as pd
    import marimo as mo
    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load OEWS locality codes for years before 2019
    """)
    return


@app.cell
def _(pd):
    oews_df = pd.read_excel("../Data/oews_area_definitions/area_definitions.xlsx", dtype='str')
    oews_df = oews_df.rename(columns={
        'FIPS':'oews_state_fips',
        'County code':'oews_county_fips',
        'Township code':'oews_township_code',
        'County/town name':'oews_county_name',
        'state':'oews_state_name'
    })


    # Remove leading blank spaces in front of area codes
    oews_df[oews_df.columns] = oews_df.apply(lambda x: x.str.lstrip(' '))

    # Keep only columns we need
    columns_to_keep = ['oews_state_fips', 'oews_county_fips', 'oews_township_code', 'oews_state_name', 'oews_county_name'] + [cols for cols in oews_df.columns if (cols[:9]=='Area code')]
    oews_df = oews_df[columns_to_keep].copy()

    # Fill in years that retain the same code as prior years
    oews_df['Area code 2004'] = oews_df['Area code 2003']
    oews_df['Area code 2007'] = oews_df['Area code 2006']
    oews_df['Area code 2009'] = oews_df['Area code 2008']
    oews_df['Area code 2012'] = oews_df['Area code 2011']
    oews_df['Area code 2013'] = oews_df['Area code 2011']
    oews_df['Area code 2014'] = oews_df['Area code 2011']

    # Drop Guam, Puerto Rico, and Virgin Islands
    oews_df = oews_df[~(oews_df['oews_state_name'].isin(['Guam', 'Puerto Rico', 'Virgin Islands']))]
    return (oews_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Save a long version of OEWS area codes for merging with OEWS data to get county-level observation
    """)
    return


@app.cell
def _(oews_df, pd):
    # Reshape long
    oews_long_df = pd.wide_to_long(oews_df, stubnames='Area code ', i=['oews_state_fips', 'oews_county_fips', 'oews_township_code', 'oews_state_name', 'oews_county_name'], j='year').reset_index()
    oews_long_df = oews_long_df.rename(columns={'Area code ':'oews_area_code'})
    return (oews_long_df,)


@app.cell
def _(oews_long_df):
    oews_long_df[oews_long_df['oews_area_code'].isna()].to_csv("../Data/oews_area_definitions/missing_area_codes.csv", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load OEWS area definitions for years after 2018
    """)
    return


@app.cell
def _(pd):
    oews_df_dict = {}
    for y in range(2019, 2023):
        oews_df_1 = pd.read_excel(f'../Data/oews_area_definitions/area_definitions_m{y}.xlsx', dtype='str')
        area_name_variable = f'May {y} MSA name'
        area_code_variable = f'May {y} MSA code '
        oews_df_1 = oews_df_1.rename(columns={'FIPS code': 'oews_state_fips', 'County code': 'oews_county_fips', 'Township code': 'oews_township_code', 'County name (or Township name for the New England states)': 'oews_county_name', 'State': 'oews_state_name', area_code_variable: 'oews_area_code', area_name_variable: 'oews_area_name'})
        oews_df_1['year'] = y
        oews_df_dict[y] = oews_df_1
    return (oews_df_dict,)


@app.cell
def _(oews_df_dict, oews_long_df, pd):
    # Combine both periods
    oews_df_post_2019 = pd.concat(oews_df_dict.values(), ignore_index=True).drop(columns = ['oews_area_name', 'State abbreviation'])
    oews_long_df_both_periods = pd.concat([oews_long_df, oews_df_post_2019], ignore_index=True)
    return (oews_long_df_both_periods,)


@app.cell
def _(oews_long_df_both_periods):
    # Export as binary
    oews_long_df_both_periods.to_parquet("../binaries/oews_area_definitions.parquet", index=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
