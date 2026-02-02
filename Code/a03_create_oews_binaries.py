import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import pandas as pd
    import json
    from pandas.api.types import union_categoricals
    from itertools import islice
    import re
    return Path, pd, re


@app.cell
def _(Path):
    oews_path = Path('../Data/oews')
    return (oews_path,)


@app.cell
def _(Path, oews_path, re):
    # Create dict with list of MSA files within each folder-year
    oews_path_dict = {}
    for folder in Path(oews_path).iterdir():
        folder_name = folder.name
        _year = re.findall('\\d+', folder_name)[0]
        if _year == '97' or _year == '98' or _year == '99':
            _year = '19' + _year
        else:
            _year = '20' + _year
        _file_list = []
        for file in Path(folder).iterdir():
            file_name = file.name
            file_prefix = file_name[0:3]
            if file_prefix == 'msa' or file_prefix == 'MSA' or file_prefix == 'oes' or (file_prefix == 'BOS'):
                _file_list.append(file)
        oews_path_dict[_year] = _file_list  # All MSA files is prefixed with 'msa' or 'MSA', except for 1997, which has 'oes' prefix  # Non-metro areas are in separate file with 'BOS' prefix beginning in 2006
    return (oews_path_dict,)


@app.cell
def _(Path):
    # Directory we are storing processed dataframes in binary format, for quick access later
    binary_path = Path('../binaries/')
    binary_path.mkdir(parents = True, exist_ok = True)

    # Dict to store paths to processed binaries
    oews_binary_path_dict = {}
    return (binary_path,)


@app.cell
def _():
    # These variables we want to store as strings
    oews_dtype_dict = {
        'area': 'string',
        'area_name': 'string',
        'occ_code': 'string',
        'occ_title': 'string'
    }
    return


@app.cell
def _(oews_path_dict, pd):
    oews_df = pd.DataFrame()
    for _year, _file_list in oews_path_dict.items():
        file_list_df_dict = {}  # Dict to store all relevant dfs in each year-folder to be concatenated into a single year df
        print(f'Loading files for year {_year}.')
        for file_path in _file_list:
            if int(_year) < 2001:
                df = pd.read_excel(file_path, dtype=str)
                header_row = df['Unnamed: 0'].isna().values.argmin()
                header = df.iloc[header_row]  # Prior to 2001, field descriptions were placed in the header rows
                new_df = df.iloc[header_row + 1:]
                new_df = new_df.rename(columns=header)
            else:
                new_df = pd.read_excel(file_path, header=0, dtype=str)
            if int(_year) == 2000:  # Find first row that is non-empty in first column, which should be the header row
                new_df.rename(columns={'occ_titl': 'occ_title'}, inplace=True)
            new_df.columns = new_df.columns.str.lower()
            if 1997 < int(_year) < 2003:  # Create new df without field description and header rows, and new headers
                new_df.rename(columns={'h_wpct10': 'h_pct10', 'h_wpct25': 'h_pct25', 'h_wpct75': 'h_pct75', 'h_wpct90': 'h_pct90', 'a_wpct10': 'a_pct10', 'a_wpct25': 'a_pct25', 'a_wpct75': 'a_pct75', 'a_wpct90': 'a_pct90'}, inplace=True)
            if int(_year) >= 2019:
                new_df.rename(columns={'area_title': 'area_name'}, inplace=True)
            if int(_year) != 1997:
                new_df = new_df[['area', 'area_name', 'occ_code', 'occ_title', 'tot_emp', 'h_mean', 'a_mean', 'h_pct10', 'h_pct25', 'h_median', 'h_pct75', 'h_pct90', 'a_pct10', 'a_pct25', 'a_median', 'a_pct75', 'a_pct90']]
            else:
                new_df = new_df[['area', 'area_name', 'occ_code', 'occ_title', 'tot_emp', 'h_mean', 'a_mean', 'h_median']]  # Keep only columns we want and store in df dict
            file_list_df_dict[file_path.name] = new_df  # For year 2000, occ_title has a typo
        combined_df = pd.concat([df for df in file_list_df_dict.values()])
        combined_df.reset_index(drop=True, inplace=True)
        combined_df['tot_emp'] = pd.to_numeric(combined_df['tot_emp'], errors='coerce')
        combined_df['a_mean'] = pd.to_numeric(combined_df['a_mean'], errors='coerce')  # Convert to lowercase column titles for all years
        combined_df['area'] = combined_df['area'].str.strip(' ')
        combined_df['year'] = int(_year)  # For years 1998-2002, wage percentile columns had 'h_wpct' and 'a_wpct' prefixes
        oews_df = pd.concat([oews_df, combined_df], ignore_index=True)  # Starting in 2019, area_name was changed to area_title  # Concat all dfs into one  # Convert total employment counts and mean wages into numeric  # Area code may have leading whitespace for 1999 and 2000  # Add year column  # Concat all years into one df
    return (oews_df,)


@app.cell
def _(Path, binary_path, oews_df):
    # Save combined df as parquet
    phil_path = Path("../files_for_phil")
    oews_df.to_parquet(binary_path.joinpath('oews.parquet'))
    # oews_df.to_parquet(phil_path.joinpath('oews.parquet'))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
