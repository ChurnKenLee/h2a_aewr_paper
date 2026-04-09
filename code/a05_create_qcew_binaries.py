import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import pyprojroot
    import dotenv, os
    import numpy as np
    import pandas as pd
    from zipfile import ZipFile

    return ZipFile, mo, pd, pyprojroot


@app.cell
def _(pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    binary_path = root_path / 'binaries'
    qcew_path = root_path / 'data' / 'qcew'
    return binary_path, qcew_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # QCEW binaries
    """)
    return


@app.cell
def _():
    qcew_dtype_dict = {
        'area_fips': 'string',
        'own_code': 'string',
        'industry_code': 'string',
        'agglvl_code': 'string',
        'size_code': 'string',
        'year': 'int16',
        'qtr': 'string',
        'disclosure_code': 'string',
        'annual_avg_estabs': 'float32',
        'annual_avg_emplvl': 'float32',
        'total_annual_wages': 'float32'
    }
    qcew_cols_list = list(qcew_dtype_dict.keys())
    return qcew_cols_list, qcew_dtype_dict


@app.cell
def _(ZipFile, binary_path, pd, qcew_cols_list, qcew_dtype_dict, qcew_path):
    for t in range(2005, 2018):
        print(t)
        zip_path = qcew_path / f"{t}_annual_singlefile.zip"
        zf = ZipFile(zip_path)
        qcew_df = pd.read_csv(zf.open(f'{t}.annual.singlefile.csv'), usecols = qcew_cols_list, dtype = qcew_dtype_dict)
        qcew_df.to_parquet(binary_path / f'qcew_{t}.parquet')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
