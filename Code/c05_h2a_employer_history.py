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
    from rapidfuzz import fuzz
    from collections import defaultdict
    return defaultdict, fuzz, pd, re


@app.cell
def _(pd):
    # Cleaned H-2A data; duplicate master entries, e.g., NCGA applications, are removed
    df = pd.read_parquet("../binaries/h2a_cleaned.parquet")
    return (df,)


@app.cell
def _(re):
    # Replace numbers
    def replace_numbers(name):
        # Define a dictionary of abbreviations and their replacements
        abbreviations = {
            r'\bONE\b': '1',
            r'\bTWO\b': '2',
            r'\bTHREE\b': '3',
            r'\bFOUR\b': '4',
            r'\bFIVE\b': '5',
            r'\bSIX\b': '6',
            r'\bSEVEN\b': '7',
            r'\bEIGHT\b': '8',
            r'\bNINE\b': '9',
            r'\bZERO\b': '0',
        }
    
        # Iterate over the abbreviations and replace them in the name
        for abbr, replacement in abbreviations.items():
            name = re.sub(abbr, replacement, name)
    
        return name
    return (replace_numbers,)


@app.cell
def _(re):
    # Harmonize common abbreviations
    def harmonize_abbreviations(name):
        # Define a dictionary of abbreviations and their replacements
        abbreviations = {
            r'\bINCORPORATED\b': 'INC',
            r'\bCORPORATION\b': 'CORP',
            r'\bLIMITED\b': 'LTD',
            r'\bDOING BUSINESS AS\b': 'DBA',
            r'\bCORPORATION\b': 'CORP',
            r'\bLIMITED LIABILITY COMPANY': 'LLC',
            r'\bLIMITED LIABILITY PARTNERSHIP': 'LLP'
        }
    
        # Iterate over the abbreviations and replace them in the name
        for abbr, replacement in abbreviations.items():
            name = re.sub(abbr, replacement, name)
    
        return name
    return (harmonize_abbreviations,)


@app.cell
def _(re):
    # Remove common terms that are not useful for matching
    def remove_common_terms(name):
        # Define a dictionary of abbreviations and their replacements
        abbreviations = {
            r'\bINC\b': '',
            r'\bCORP\b': '',
            r'\bLTD\b': '',
            r'\bDBA\b': '',
            r'\bCORP\b': '',
            r'\bPARTNERSHIP\b': '',
            r'\bFARM\b': '',
            r'\bFARMS\b': '',
            r'\bHARVESTING\b': '',
            r'\bLLC\b': '',
            r'\bLLP\b': '',
            r'\bCOMPANY\b': ''
        }

        # Iterate over the abbreviations and replace them in the name
        for abbr, replacement in abbreviations.items():
            name = re.sub(abbr, replacement, name)
    
        return name
    return (remove_common_terms,)


@app.cell
def _(df):
    # Clean employer names for string similarity clustering
    # Replace mising with empty string
    df['cleaned_employer_name'] = df['employer_name'].fillna("")
    df['cleaned_employer_name'] = df['cleaned_employer_name'].astype(str)
    return


@app.cell
def _(df, replace_numbers):
    # Replace numbers with letters
    df['cleaned_employer_name'] = df['cleaned_employer_name'].apply(lambda x: replace_numbers(x))
    return


@app.cell
def _(df):
    # Strip special characters, replace with whitespace
    df['cleaned_employer_name'] = df['cleaned_employer_name'].str.replace(pat = r'[^A-Z0-9\s]', repl = " ", regex = True)
    return


@app.cell
def _(df, harmonize_abbreviations):
    # Harmonize abbreviations
    df['cleaned_employer_name'] = df['cleaned_employer_name'].apply(lambda x: harmonize_abbreviations(x))
    return


@app.cell
def _(df, remove_common_terms):
    # Remove common terms that are not useful for matching
    df['cleaned_employer_name'] = df['cleaned_employer_name'].apply(lambda x: remove_common_terms(x))
    return


@app.cell
def _(df):
    # Harmonize whitespace
    df['cleaned_employer_name'] = df['cleaned_employer_name'].apply(lambda x: ' '.join(x.split()))
    return


@app.cell
def _(df):
    # Keep only columns we want, drop duplicates
    names_df = df[['cleaned_employer_name', 'employer_postal_code']]
    names_df = names_df.drop_duplicates()
    return (names_df,)


@app.cell
def _(defaultdict, fuzz):
    # Define a function to group employer names by string similarity
    def group_by_similarity(strings, threshold=90):
        groups = defaultdict(list)
    
        for string in strings:
            matched = False
        
            for key in groups.keys():
                if fuzz.WRatio(string, key) >= threshold:
                    groups[key].append(string)
                    matched = True
                    break
        
            if not matched:
                groups[string].append(string)
    
        return dict(groups)
    return (group_by_similarity,)


@app.function
# Define a function to assign the same harmonized name to each group
def assign_group_name(employer_name, grouped_names):

    for group_name, employer_name_list in grouped_names.items():

        if employer_name in employer_name_list:
            return group_name
        else:
            continue

    return("")


@app.cell
def _(group_by_similarity, names_df, pd):
    grouped_names_df = pd.DataFrame()
    # We restrict employer name matching to employers within the same ZIP code
    list_of_postal_codes = names_df['employer_postal_code'].unique()
    for postal_code in list_of_postal_codes:

        match_df = names_df[names_df['employer_postal_code'] == postal_code].copy()
        grouped_names = group_by_similarity(match_df['cleaned_employer_name'])
        match_df['group_name'] = match_df['cleaned_employer_name'].apply(lambda x: assign_group_name(x, grouped_names))

        grouped_names_df = pd.concat([grouped_names_df, match_df])
    return (grouped_names_df,)


@app.cell
def _(grouped_names_df, names_df):
    # Add group names back in
    names_df_1 = names_df.merge(grouped_names_df, how='left')
    return (names_df_1,)


@app.cell
def _(df, names_df_1):
    # Add group names to original data, add employer ID, and re-export
    df_1 = df.merge(names_df_1, how='left')
    df_1['employer_id'] = df_1.groupby(['group_name', 'employer_postal_code']).ngroup()
    return (df_1,)


@app.cell
def _(df_1):
    df_1.to_parquet('../files_for_phil/h2a_cleaned_with_grouped_employer_names.parquet')
    df_1.to_parquet('../binaries/h2a_cleaned_with_grouped_employer_names.parquet')
    return


if __name__ == "__main__":
    app.run()
