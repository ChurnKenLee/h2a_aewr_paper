import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from zipfile import ZipFile
    import marimo as mo
    return Path, ZipFile, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # QCEW binaries
    """)
    return


@app.cell
def _():
    qcew_dtype_dict = {
        'area_fips': 'category',
        'own_code': 'category',
        'industry_code': 'category',
        'agglvl_code': 'category',
        'size_code': 'category',
        'year': 'int16',
        'qtr': 'category',
        'disclosure_code': 'category',
        'annual_avg_estabs': 'float32',
        'annual_avg_emplvl': 'float32',
        'total_annual_wages': 'float32'
    }
    qcew_cols_list = list(qcew_dtype_dict.keys())
    return qcew_cols_list, qcew_dtype_dict


@app.cell
def _(Path, ZipFile, pd, qcew_cols_list, qcew_dtype_dict):
    for t in range(2005, 2018):
        print(t)
        zip_path = Path(f"../Data/qcew/{t}_annual_singlefile.zip")
        zf = ZipFile(zip_path)
        qcew_df = pd.read_csv(zf.open(f'{t}.annual.singlefile.csv'), usecols = qcew_cols_list, dtype = qcew_dtype_dict)
        qcew_df.to_parquet(f'../binaries/qcew_{t}.parquet')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
