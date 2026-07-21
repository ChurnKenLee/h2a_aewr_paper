import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    # Extract QCEW reference-month employment for entropy-calibration moments.
    # Inputs: quarterly QCEW single-file ZIP archives in data/raw/qcew.
    # Outputs: data/intermediate/qcew_county_ag_quarterly_employment.parquet.
    import argparse
    import zipfile
    from pathlib import Path
    import polars as pl
    from h2a.paths import INTERMEDIATE, RAW

    return INTERMEDIATE, Path, RAW, pl, zipfile


@app.cell
def _(INTERMEDIATE):
    FIRST_YEAR = 2000
    LAST_YEAR = 2024
    OUTPUT_PATH = INTERMEDIATE / "qcew_county_ag_quarterly_employment.parquet"
    return FIRST_YEAR, LAST_YEAR, OUTPUT_PATH


@app.cell
def _(pl):
    REFERENCE_MONTH = {
        1: "january",
        2: "april",
        3: "july",
        4: "october",
    }
    QCEW_DTYPES = {
        "area_fips": pl.String,
        "own_code": pl.String,
        "industry_code": pl.String,
        "agglvl_code": pl.String,
        "year": pl.Int16,
        "qtr": pl.Int8,
        "disclosure_code": pl.String,
        "month1_emplvl": pl.Float32,
    }
    return QCEW_DTYPES, REFERENCE_MONTH


@app.cell
def _(Path, QCEW_DTYPES, REFERENCE_MONTH, pl, zipfile):
    def extract_year(zip_path: Path, year: int) -> pl.DataFrame:
        """Read and filter one quarterly single-file archive."""
        target_csv = f"{year}.q1-q4.singlefile.csv"
        with zipfile.ZipFile(zip_path, mode="r") as archive:
            with archive.open(target_csv) as extracted_file:
                year_df = pl.read_csv(
                    extracted_file,
                    columns=list(QCEW_DTYPES),
                    schema_overrides=QCEW_DTYPES,
                )

        return (
            year_df.filter(
                (pl.col("own_code") == "5")
                & (pl.col("agglvl_code") == "75")
                & pl.col("industry_code").is_in(["111", "112"])
            )
            .with_columns(
                pl.col("qtr")
                .replace_strict(REFERENCE_MONTH, return_dtype=pl.String)
                .alias("reference_month"),
                (pl.col("disclosure_code") == "").alias("qcew_employment_disclosed"),
                pl.when(pl.col("disclosure_code") == "")
                .then(pl.col("month1_emplvl"))
                .otherwise(None)
                .alias("qcew_reference_month_emplvl"),
            )
            .rename({"area_fips": "countyfips"})
            .select(
                "countyfips",
                "year",
                "qtr",
                "reference_month",
                "industry_code",
                "disclosure_code",
                "qcew_employment_disclosed",
                "qcew_reference_month_emplvl",
            )
        )

    return (extract_year,)


@app.cell
def _(FIRST_YEAR, LAST_YEAR, OUTPUT_PATH, Path, RAW, extract_year, pl):
    def extract_quarterly_employment(output_path: Path = OUTPUT_PATH) -> None:
        """Extract all years and write the compact calibration input."""
        qcew_path = RAW / "qcew"
        frames = []
        for year in range(FIRST_YEAR, LAST_YEAR + 1):
            print(f"Extracting quarterly QCEW {year}", flush=True)
            frames.append(
                extract_year(qcew_path / f"{year}_qtrly_singlefile.zip", year)
            )

        quarterly_employment = pl.concat(frames, how="vertical_relaxed").sort(
            "countyfips", "year", "qtr", "industry_code"
        )
        quarterly_employment.write_parquet(output_path)
        print(
            f"Wrote {quarterly_employment.height:,} county-industry-quarter rows "
            f"to {output_path}",
            flush=True,
        )

    return (extract_quarterly_employment,)


@app.cell
def _(OUTPUT_PATH, extract_quarterly_employment):
    if OUTPUT_PATH.exists():
        print(f"Output already exists at {OUTPUT_PATH}")
    else:
        extract_quarterly_employment()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
