import marimo

__generated_with = "0.19.7"
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
def _(Path):
    # Load column names from PDF file
    sob_path = Path(__file__).parent.parent / 'Data' / 'risk_management_agency' / 'summary_of_business'
    cov_pdf_path = sob_path / 'SOB_State_County_Crop_with_Coverage_Level_1989_Forward.pdf'
    tpu_pdf_path = sob_path / 'SOBTPU_External_All_Years.pdf'
    return cov_pdf_path, sob_path, tpu_pdf_path


@app.cell
def _(pdfplumber):
    # Define a function that extracts column names from Summary of Business file
    def extract_sob_col_name_format(pdf_path):
        # Store column names in list
        col_format_dict = {}
        char_removal_tab = str.maketrans('', '', '()/') # remove these chars
        char_replacement_tab = str.maketrans(' \n', '__', '') # replace these chars with underscore

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                table = page.extract_table()
                for col_detail in table:
                    col_number = col_detail[0]
                    col_name = col_detail[1]
                    col_type = col_detail[2]
                    # Clean column name
                    col_name = col_name.lower().translate(char_removal_tab).translate(char_replacement_tab)
                    if any(char.isdigit() for char in col_number): # Only include numbered rows, which excludes header
                        col_format_dict[col_name] = col_type

        return col_format_dict
    return (extract_sob_col_name_format,)


@app.cell
def _(cov_pdf_path, extract_sob_col_name_format, tpu_pdf_path):
    cov_col_format_dict = extract_sob_col_name_format(cov_pdf_path)
    tpu_col_format_dict = extract_sob_col_name_format(tpu_pdf_path)
    return cov_col_format_dict, tpu_col_format_dict


@app.cell
def _(pl):
    # Define type schema
    # By default, most columns should be integer
    # Define exceptions: string and float
    def define_sob_schema(format_dict):
        schema_dict = {}
        for col, format in format_dict.items():
            if 'V' in format: # V in format code indicates decimal
                schema_dict[col] = pl.Float64
            elif 'X' in format or 'code' in col: # X indicates alpha-numeric, and we want codes to be strings
                schema_dict[col] = pl.String
            else:
                schema_dict[col] = pl.Int64

        return schema_dict
    return (define_sob_schema,)


@app.cell
def _(cov_col_format_dict, define_sob_schema, pl, tpu_col_format_dict):
    cov_schema = define_sob_schema(cov_col_format_dict)
    tpu_schema = define_sob_schema(tpu_col_format_dict)

    # TPU schema has errors; just read all as strings for now and diagnose later
    tpu_str_schema = {}
    for _col in tpu_col_format_dict.keys():
        tpu_str_schema[_col] = pl.String
    return cov_schema, tpu_str_schema


@app.cell
def _(ZipFile, pl):
    # Define a function that reads Summary of Business files and converts to Polars DataFrame
    def read_sob_archive(archive_path, schema_overrides_dict, eol_char):
        col_names = [c for c in schema_overrides_dict.keys()]
        with ZipFile(archive_path) as zf:
            for file_name in zf.namelist():
                df = pl.read_csv(
                    zf.read(file_name), 
                    has_header=False, 
                    separator='|', 
                    new_columns=col_names,
                    schema_overrides=schema_overrides_dict,
                    eol_char=eol_char
                )
            return df
    return (read_sob_archive,)


@app.cell
def _(cov_schema, pl, read_sob_archive, sob_path, tpu_str_schema):
    # Get all COV and TPU files
    cov_df_list = []
    tpu_df_list = []

    for _file_path in sob_path.iterdir():
        if 'sobcov' in _file_path.name and _file_path.suffix == '.zip':
            cov_df_list.append(read_sob_archive(_file_path, cov_schema, '\n'))
        elif 'sobtpu' in _file_path.name and _file_path.suffix == '.zip':
            tpu_df_list.append(read_sob_archive(_file_path, tpu_str_schema, '\r'))

    cov_df = pl.concat(cov_df_list)
    tpu_df = pl.concat(tpu_df_list)
    return cov_df, tpu_df


@app.cell
def _(Path, cov_df, tpu_df):
    # Export to parquet for ingestion into R
    cov_df.write_parquet(Path(__file__).parent.parent / 'binaries' / 'rma_sob_cov.parquet')
    tpu_df.write_parquet(Path(__file__).parent.parent / 'binaries' / 'rma_sob_tpu.parquet')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
