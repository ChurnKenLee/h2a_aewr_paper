import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import polars as pl
    from zipfile import ZipFile
    import pdfplumber
    import json
    return Path, ZipFile, pdfplumber, pl


@app.cell
def _(Path, pdfplumber):
    # Load column names from PDF file
    rma_sob_path = Path("../Data/risk_management_agency")
    for _file in rma_sob_path.iterdir():
        if _file.suffix == '.pdf':
            pdf_path = _file

    # Store column names in list
    col_names_dict = {}
    char_removal_tab = str.maketrans('', '', '()/') # remove these chars
    char_replacement_tab = str.maketrans(' \n', '__', '') # replace these chars with underscore

    with pdfplumber.open(_file) as pdf:
        for _i, _page in enumerate(pdf.pages):
            table = _page.extract_table()
            for _col_detail in table:
                col_number = _col_detail[0]
                col_name = _col_detail[1]
                # Clean column name
                col_name = col_name.lower().translate(char_removal_tab).translate(char_replacement_tab)
                if col_number != '':
                    col_names_dict[col_number] = col_name
                
    col_names_list = [_v for _v in col_names_dict.values()]
    col_names_list
    return col_names_list, rma_sob_path


@app.cell
def _(pl):
    # Some columns should remain strings
    _str_list = ['location_state_code', 'location_county_code', 'commodity_code', 'insurance_plan_code']
    overrides = {}
    for _col_name in _str_list:
        overrides[_col_name] = pl.String
    return (overrides,)


@app.cell
def _(ZipFile, col_names_list, overrides, pl, rma_sob_path):
    # Import all years of RMA Summary of Business data
    _df_list = []
    for _archive_path in rma_sob_path.iterdir():
        if _archive_path.suffix == '.zip':
            with ZipFile(_archive_path) as zf:
                zf_contents = [f for f in zf.namelist() if f.endswith('.txt')]
                for file_name in zf_contents:
                    _df_list.append(
                        pl.read_csv(
                            zf.read(file_name), 
                            has_header=False, 
                            separator='|', 
                            new_columns=col_names_list,
                            schema_overrides = overrides
                        ))

    sob_df = pl.concat(_df_list)
    sob_df
    return (sob_df,)


@app.cell
def _(sob_df):
    # Export to parquet for ingestion into R
    sob_df.write_parquet("../binaries/rma_sob.parquet")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
