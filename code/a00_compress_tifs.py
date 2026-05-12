import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Raw Data Compression

    This notebook keeps code-read rasters as GeoTIFFs with internal DEFLATE
    compression and repacks selected CSV/TXT/ZIP raw downloads into `.7z`
    archives. Downstream scripts still need archive-aware reads where they
    currently expect extracted files.
    """)
    return


@app.cell
def _():
    RUN_COMPRESSION = False
    REMOVE_ARCHIVED_SOURCES = True
    REPLACE_EXISTING_ARCHIVES = False
    return REMOVE_ARCHIVED_SOURCES, REPLACE_EXISTING_ARCHIVES, RUN_COMPRESSION


@app.cell
def _(
    RAW,
    REMOVE_ARCHIVED_SOURCES,
    REPLACE_EXISTING_ARCHIVES,
    RUN_COMPRESSION,
    print_results,
    run_compression_plan,
):
    results = run_compression_plan(
        RAW,
        dry_run=not RUN_COMPRESSION,
        remove_archived_sources=REMOVE_ARCHIVED_SOURCES,
        replace_existing_archives=REPLACE_EXISTING_ARCHIVES,
    )
    print_results(results)
    return


if __name__ == "__main__":
    app.run()
