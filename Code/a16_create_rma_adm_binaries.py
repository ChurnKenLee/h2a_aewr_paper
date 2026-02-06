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
    return Path, ZipFile


@app.cell
def _(Path):
    adm_path = Path(__file__).parent.parent / 'Data' / 'risk_management_agency' / 'actuarial_data_master'
    arc_path = adm_path / '00admytd.zip'
    rec_path = adm_path / '00admrec.zip'
    return (rec_path,)


@app.cell
def _(file):
    file.splitlines()[1]
    return


@app.cell
def _(ZipFile, rec_path):
    # Open record layout file
    zip = ZipFile(rec_path)
    file = zip.read('ADM2000V.TXT')
    for line in file.splitlines():
        if 'PRICES' in line:
            print(line)
    return (file,)


@app.cell
def _(file):
    file
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
