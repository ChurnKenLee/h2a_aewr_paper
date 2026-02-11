import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import polars as pl
    import polars_io as pio
    from zipfile import ZipFile
    import io
    import pdfplumber
    import json
    import re

    return Path, ZipFile, io, mo, pio, pl, re


@app.cell
def _(Path):
    adm_path = Path(__file__).parent.parent / 'Data' / 'risk_management_agency' / 'actuarial_data_master'
    return (adm_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Actuarial Data Master files for 2000-2010
    """)
    return


@app.cell
def _(ZipFile, io, re):
    # Define function that extracts the relevant fwf file from the archive for 2000-2001
    def extract_00_10_adm_col_spec(meta_path, toc_file_name, col_spec_file_name, toc_encoding, col_spec_encoding, target_record_purpose):

        meta_zip = ZipFile(meta_path)

        # Data file number is contained in .TOC file, and column specification is contained in .TXT file
        toc_file = meta_zip.open(toc_file_name)

        # Wrap the binary stream in a text wrapper
        toc_content = io.TextIOWrapper(toc_file, encoding=toc_encoding).read()

        # Find TOC row that has RECORD PURPOSE we want to obtain record prefix
        for line in toc_content.splitlines():
            prefix_str = line[0:17].strip() # First 17 chars contains prefix
            remainder_str = line[17:] # There are >=5 blank spaces between record purpose and page number
            split = re.split(r'\s{4}', remainder_str, maxsplit=1)
            if len(split)>1:
                record_purpose_str = split[0].strip()
                page_str = split[1].strip()
                if record_purpose_str == target_record_purpose:
                    print('Found TOC row')
                    print(f'Prefix = {prefix_str}, record purpose = {record_purpose_str}, page = {page_str}')
                    break

        col_spec_file = meta_zip.open(col_spec_file_name)
        col_spec_content = io.TextIOWrapper(col_spec_file, encoding=col_spec_encoding).read()

        current_page = 0
        in_correct_page = False
        columns = []
        pages = []
        page_pattern = re.compile(r'\n(?=\d{1,2}/\d{1,2}/\d{4})')
        pages = page_pattern.split(col_spec_content)

        for page in pages:

            # Verify we are in the correct page
            correct_page = False
            for line in page.splitlines():
                if 'RECORD PURPOSE' in line:
                    split = re.split(r'\s{4}', line, maxsplit=1)
                    record_purpose_str = split[1].strip()
                    if record_purpose_str == target_record_purpose:
                        print('We found correct column spec page')
                        print('Col spec page record purpose:', line)
                        correct_page = True

            if correct_page:
                # Regex Breakdown for extracting column specifications:
                # 1. ^([A-Z0-9\-\* ]{1,24}) : Captures field name (start of line, up to 24 chars)
                # 2. \s+(\d+)               : Captures Start Position
                # 3. \s+(\d+)               : Captures Length
                # 4. \s+([X9V()0-9]+)       : Captures COBOL Picture/Format spec
                pattern = r"^([A-Z0-9\-\* ]{1,24})\s+(\d+)\s+(\d+)\s+([X9V()0-9]+)"

                col_specs = []
                for line in page.splitlines():
                    match = re.search(pattern, line)
                    if match:
                        col_specs.append({
                            "field_name": match.group(1).strip(),
                            "start": int(match.group(2)),
                            "length": int(match.group(3)),
                            "format": match.group(4).strip()
                        })

                return col_specs, prefix_str

    return (extract_00_10_adm_col_spec,)


@app.cell
def _(ZipFile, adm_path, extract_00_10_adm_col_spec, pio):
    # For 2000 and 2001, the relevant files are as follows:
    prefix_dict = {
        'FOUR': '4',
        'X':'X'
    }

    prices_00_10_dfs = {}
    cross_reference_00_10_dfs = {}

    for y in range(0, 11):
        year = str(y).zfill(2)

        meta_path = adm_path / f'{year}admrec.zip'
        data_path = adm_path / f'{year}admytd.zip'

        # 2000 and 2001 has .TOC and .TXT metadata files
        for file_name in ZipFile(meta_path).namelist():
            if y < 2 and file_name.endswith('.TOC'):
                toc_file_name = file_name
            elif y < 2 and file_name.endswith('.TXT'):
                col_spec_file_name = file_name

            # For 2002 to 2010, TOC file has TO suffix, column spec file has RE suffix
            elif y > 1 and file_name[6:8] == 'TO':
                toc_file_name = file_name
            elif y > 1 and file_name[6:8] == 'RE':
                col_spec_file_name = file_name

        target_record_purpose = 'PRICES'
        print(f'Getting column spec for year {year}')
        col_specs, prefix_str = extract_00_10_adm_col_spec(meta_path, toc_file_name, col_spec_file_name, 'cp1252', 'cp1252', target_record_purpose)

        # Define fwf dict from column specs, note adjustment for 0 indexing of character positions
        fwf_dict = {}
        for col_spec in col_specs:
            fwf_dict[col_spec['field_name']] = (col_spec['start']-1, col_spec['start']-1 + col_spec['length'])

        prefix = prefix_dict[prefix_str]
        # Use fwf specification to open file
        data_zip = ZipFile(data_path, 'r')
        data_file = data_zip.open(f'R{prefix}{year}YTD').read()
        prices_00_10_dfs[year] = pio.read_fwf(data_file, cols=fwf_dict, infer_schema=False, encoding='cp1252')

        target_record_purpose = 'CROSS REFERENCE DATA'
        print(f'Getting column spec for year {year}')
        col_specs, prefix_str = extract_00_10_adm_col_spec(meta_path, toc_file_name, col_spec_file_name, 'cp1252', 'cp1252', target_record_purpose)

        # Define fwf dict from column specs, note adjustment for 0 indexing of character positions
        fwf_dict = {}
        for col_spec in col_specs:
            fwf_dict[col_spec['field_name']] = (col_spec['start']-1, col_spec['start']-1 + col_spec['length'])

        prefix = prefix_dict[prefix_str]
        # Use fwf specification to open file
        data_zip = ZipFile(data_path, 'r')
        data_file = data_zip.open(f'R{prefix}{year}YTD').read()
        cross_reference_00_10_dfs[year] = pio.read_fwf(data_file, cols=fwf_dict, infer_schema=False, encoding='cp1252')
    return cross_reference_00_10_dfs, prices_00_10_dfs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Actuarial Data Master files for 2011-2023
    """)
    return


@app.cell
def _(adm_path):
    arc_path = adm_path / '2011_ADM_YTD.zip'
    return (arc_path,)


@app.cell
def _(ZipFile, arc_path):
    arc_zip = ZipFile(arc_path)
    for _file_name in arc_zip.namelist():
        if 'AdmLayout' in _file_name:
            layout_file_name = _file_name
        elif 'A00810' in _file_name:
            price_file_name = _file_name
    return arc_zip, layout_file_name, price_file_name


@app.cell
def _(arc_zip, layout_file_name, pl):
    layout_file = arc_zip.open(layout_file_name)
    layout_df = pl.read_csv(layout_file.read(), separator='|', infer_schema=False, encoding='1252')
    return (layout_df,)


@app.cell
def _(layout_df, pl):
    price_layout = layout_df.filter(
        pl.col('ADM Record Type Code') == 'A00810'
    )
    return


@app.cell
def _(arc_zip, pl, price_file_name):
    price_file = arc_zip.open(price_file_name)
    price_df = pl.read_csv(price_file.read(), separator='|', infer_schema=False, encoding='1252')
    return


@app.cell
def _(prices_00_10_dfs):
    prices_00_10_dfs['10']
    return


@app.cell
def _(cross_reference_00_10_dfs):
    cross_reference_00_10_dfs['10']
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
