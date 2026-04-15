import marimo

__generated_with = "0.22.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import pyprojroot
    import dotenv, os
    import polars as pl
    import polars.selectors as cs
    import pandas as pd

    return mo, pl, pyprojroot


@app.cell
def _(pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    binary_path = root_path / 'binaries'
    oews_path = root_path / 'data' / 'oews_area_definitions'
    return binary_path, oews_path, root_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load OEWS locality codes for years before 2019
    """)
    return


@app.cell
def _(oews_path, pl):
    oews_df = pl.read_excel(oews_path / 'area_definitions.xlsx').select(pl.all().cast(pl.String))
    oews_df = oews_df.rename({
        'FIPS':'oews_state_fips',
        'County code':'oews_county_fips',
        'Township code':'oews_township_code',
        'County/town name':'oews_county_name',
        'state':'oews_state_name'
    })
    # Remove leading blank spaces in front of area codes
    oews_df = oews_df.with_columns(pl.all().str.strip_chars_start(' '))

    # Keep only columns we need
    columns_to_keep = ['oews_state_fips', 'oews_county_fips', 'oews_township_code', 'oews_state_name', 'oews_county_name']
    area_cols = [c for c in oews_df.columns if c.startswith('Area code')]
    oews_df = oews_df.select(columns_to_keep + area_cols)

    # Fill in years that retain the same code as prior years
    oews_df = oews_df.with_columns([
        pl.col('Area code 2003').alias('Area code 2004'),
        pl.col('Area code 2006').alias('Area code 2007'),
        pl.col('Area code 2008').alias('Area code 2009'),
        pl.col('Area code 2011').alias('Area code 2012'),
        pl.col('Area code 2011').alias('Area code 2013'),
        pl.col('Area code 2011').alias('Area code 2014'),
    ])

    # Drop Guam, Puerto Rico, and Virgin Islands
    oews_df = oews_df.filter(
        ~pl.col('oews_state_name').is_in(['Guam', 'Puerto Rico', 'Virgin Islands'])
    )
    oews_df
    return area_cols, oews_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Save a long version of OEWS area codes for merging with OEWS data to get county-level observation
    """)
    return


@app.cell
def _(area_cols, oews_df, oews_path, pl):
    # Update area_cols to include the newly mapped years
    new_area_cols = [c for c in oews_df.columns if c.startswith('Area code')]

    # Reshape long
    oews_long_df = oews_df.unpivot(
        index=['oews_state_fips', 'oews_county_fips', 'oews_township_code', 'oews_state_name', 'oews_county_name'],
        on=area_cols,
        variable_name='year',
        value_name='oews_area_code'
    ).with_columns(
        pl.col('year').str.replace('Area code ', '').cast(pl.Int32)
    )

    oews_long_df.filter(
        pl.col('oews_area_code').is_null() | (pl.col('oews_area_code').str.strip_chars() == "")
    ).write_csv(oews_path / 'missing_area_codes.csv')
    oews_long_df
    return (oews_long_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load OEWS area definitions for years after 2018
    """)
    return


@app.cell
def _(oews_path, pl):
    oews_df_list =[]
    for y in range(2019, 2023):
        # Read excel and cast to string to maintain consistency
        oews_df_1 = pl.read_excel(oews_path / f'area_definitions_m{y}.xlsx').select(pl.all().cast(pl.String))
    
        area_name_variable = f'May {y} MSA name'
        area_code_variable = f'May {y} MSA code '
    
        rename_map = {
            'FIPS code': 'oews_state_fips', 
            'County code': 'oews_county_fips', 
            'Township code': 'oews_township_code', 
            'County name (or Township name for the New England states)': 'oews_county_name', 
            'State': 'oews_state_name', 
            area_code_variable: 'oews_area_code', 
            area_name_variable: 'oews_area_name'
        }
    
        # Safe renaming in case certain variables are absent in a specific year
        rename_map = {k: v for k, v in rename_map.items() if k in oews_df_1.columns}
    
        oews_df_1 = oews_df_1.rename(rename_map).with_columns(pl.lit(y).cast(pl.Int32).alias('year'))
        oews_df_list.append(oews_df_1)
    return (oews_df_list,)


@app.cell
def _(oews_df_list, pl):
    # Combine post 2019 periods
    oews_df_post_2019 = pl.concat(oews_df_list, how="diagonal")
    oews_df_post_2019 = oews_df_post_2019.drop(['oews_area_name', 'State abbreviation'])
    return (oews_df_post_2019,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ensure county definitions match Census 2010 definition as in Phil's workflow
    """)
    return


@app.cell
def _(pl, root_path):
    full_county_set = pl.read_csv(
        root_path / "Data Int" / "county_adjacency2010.csv"
    ).select([
        'fipscounty', 'countyname'
    ]).unique().cast(
        pl.String
    ).with_columns(
        pl.col('fipscounty').str.pad_start(5, '0').alias('fipscounty')
    ).sort('fipscounty')
    full_county_set
    return (full_county_set,)


@app.cell
def _(oews_df_post_2019, oews_long_df, pl):
    # Zero-pad FIPS codes to cleanly standardize the string definitions
    # Combine both periods
    oews_long_df_both_periods = pl.concat([oews_long_df, oews_df_post_2019], how="diagonal")

    oews_long_df_both_periods = oews_long_df_both_periods.with_columns([
        pl.col('oews_state_fips').str.zfill(2),
        pl.col('oews_county_fips').str.zfill(3)
    ]).with_columns(
        (pl.col('oews_state_fips') + pl.col('oews_county_fips')).alias('fipscounty')
    )
    oews_long_df_both_periods
    return (oews_long_df_both_periods,)


@app.cell
def _(full_county_set, oews_long_df_both_periods, pl):
    # Don't care about states outside CONUS
    wrong_oews = oews_long_df_both_periods.join(
        full_county_set, 
        on='fipscounty', how='anti'
    ).unique(
        'fipscounty'
    ).sort(
        'fipscounty'
    ).filter(
        pl.col('oews_state_fips') != '02',
        pl.col('oews_state_fips') != '72'
    )
    wrong_oews
    # Seems like the only problem is Miami-Dade (remap), Yellowstone (merged with Gallatin and Park, can ignore), Oglala Lakota (remap), Oak Ridge Reservation (Manhattan Project, can ignore), Clifton Forge (merged with Alleghany, can ignore)
    return


@app.cell
def _(oews_long_df_both_periods, pl):
    oews_remapped = oews_long_df_both_periods.with_columns(
        # Dade to Miami-Dade, Oglala Lakota to Shannon
        pl.when(
            pl.col('fipscounty') =='12025'
        ).then(
            pl.lit('086')
        ).when(
            pl.col('fipscounty') =='46102'
        ).then(
            pl.lit('113')
        ).otherwise(
            pl.col('oews_county_fips')
        ).alias('oews_county_fips')
    )
    # Drop deprecated counties
    oews_remapped = oews_remapped.filter(
        ~pl.col('fipscounty').is_in(['30113', '47191', '51560'])
    ).drop('fipscounty')
    return (oews_remapped,)


@app.cell
def _(binary_path, oews_remapped):
    # Export as binary
    oews_remapped.write_parquet(binary_path / "oews_area_definitions.parquet",)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
