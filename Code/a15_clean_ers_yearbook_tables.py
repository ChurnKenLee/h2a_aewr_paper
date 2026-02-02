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
    import pickle
    import rapidfuzz
    return Path, pd


@app.cell
def _(Path, pd):
    # Load all the ERS yearbook tables
    path = r"../Data/ers_yearbook"
    files = Path(path).glob('Fruit*.csv')  # .rglob to get subdirectories

    dfs = list()

    # Fruit and tree nuts tables
    for f in files:
        data = pd.read_csv(f, dtype = 'string', keep_default_na=False)
        data['file'] = f.stem # add column denoting the filename
        dfs.append(data)

    fruit_df = pd.concat(dfs, ignore_index=True)

    # Vegetables, contains melons for earlier years as well
    vegetable_df = pd.read_csv("../Data/ers_yearbook/Vegetables_Pulses.csv", dtype = 'string', keep_default_na=False)
    return fruit_df, vegetable_df


@app.cell
def _(vegetable_df):
    # Harmonize vegetable table names for concat with fruit table
    vegetable_df_1 = vegetable_df.rename(columns={'Year': 'year_value', 'Item': 'variable', 'Commodity': 'commodity_element', 'EndUse': 'market_segment', 'Location': 'geographic_extent', 'PublishValue': 'value', 'Unit': 'unit'})
    return (vegetable_df_1,)


@app.cell
def _(fruit_df, pd, vegetable_df_1):
    # Concat and export as binary
    crop_df = pd.concat([fruit_df, vegetable_df_1])[['year_value', 'year_unit', 'variable', 'commodity_element', 'market_segment', 'geographic_extent', 'value', 'unit']]
    crop_df.to_parquet('../binaries/ers_yearbooks.parquet', index=False)
    return


if __name__ == "__main__":
    app.run()
