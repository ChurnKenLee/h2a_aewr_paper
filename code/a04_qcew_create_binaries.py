import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from h2a.paths import CODE, RAW, INTERMEDIATE, CACHE
    import dotenv, os
    import polars as pl
    import zipfile

    return INTERMEDIATE, RAW, mo, pl, zipfile


@app.cell
def _(INTERMEDIATE, RAW):
    binary_path = INTERMEDIATE
    qcew_path = RAW / "qcew"
    return binary_path, qcew_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # QCEW binaries
    """)
    return


@app.cell
def _(pl):
    qcew_dtype_dict = {
        "area_fips": pl.String,
        "own_code": pl.String,
        "industry_code": pl.String,
        "agglvl_code": pl.String,
        "size_code": pl.String,
        "year": pl.Int16,
        "qtr": pl.String,
        "disclosure_code": pl.String,
        "annual_avg_estabs": pl.Float32,
        "annual_avg_emplvl": pl.Float32,
        "total_annual_wages": pl.Float32,
    }
    qcew_cols_list = list(qcew_dtype_dict.keys())
    return qcew_cols_list, qcew_dtype_dict


@app.cell
def _(binary_path, pl, qcew_cols_list, qcew_dtype_dict, qcew_path, zipfile):
    qcew_df = pl.DataFrame()
    for t in range(2000, 2025):
        print(t)
        zip_path = qcew_path / f"{t}_annual_singlefile.zip"
        target_csv = f"{t}.annual.singlefile.csv"

        with zipfile.ZipFile(zip_path, mode="r") as zf:
            with zf.open(target_csv) as extracted_file:
                _df = pl.read_csv(
                    extracted_file,
                    columns=qcew_cols_list,
                    schema_overrides=qcew_dtype_dict,
                )
        qcew_df = pl.concat([qcew_df, _df])

    qcew_df.write_parquet(binary_path / "qcew.parquet")
    return


if __name__ == "__main__":
    app.run()
