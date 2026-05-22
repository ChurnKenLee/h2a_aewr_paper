import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from h2a.paths import CODE, RAW, INTERMEDIATE, CACHE
    import dotenv, os
    import polars as pl

    return INTERMEDIATE, RAW, pl


@app.cell
def _(INTERMEDIATE, RAW, pl):
    county_adjacency = pl.read_csv(
        RAW / "geographic_crosswalks" / "census" / "county_adjacency2010.txt",
        separator="\t",
        new_columns=["countyname", "fipscounty", "neighborname", "fipsneighbor"],
        infer_schema=False,
        has_header=False,
        encoding="cp1252",
    )
    county_adjacency = county_adjacency.fill_null(strategy="forward").sort(by=pl.all())
    county_adjacency.write_parquet(INTERMEDIATE / "county_adjacency2010.parquet")
    return


if __name__ == "__main__":
    app.run()
