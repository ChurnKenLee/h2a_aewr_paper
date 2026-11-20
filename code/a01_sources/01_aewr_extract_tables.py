# Purpose: Build audited annual AEWR references and dated FLS schedules.
# Inputs: cached data/raw/aewr, the cached 2023-rule OEWS workbooks under
#         data/raw/aewr_2023_oews_occupation/flag, and the reviewed source-date
#         registry beside this app.
# Outputs: aewr.parquet, aewr_rate_schedule.parquet, the occupation and skill
#          rate tables, source audit and manifest, and explicitly nonstatutory
#          2009 application wages.
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    from h2a.aewr import build_aewr_artifacts, write_artifacts
    from h2a.paths import CODE, INTERMEDIATE, RAW

    return CODE, INTERMEDIATE, RAW, build_aewr_artifacts, write_artifacts


@app.cell
def _(CODE, RAW, build_aewr_artifacts):
    # docs-ground:start aewr-source-contract
    # Missing cached sources fail before publishing; no network refresh occurs.
    # 2009 application means never enter the statutory AEWR column. The USDA
    # validation workbook does not determine the annual reference's support.
    artifacts = build_aewr_artifacts(
        RAW / "aewr", CODE / "a01_sources/aewr_rate_schedule_manifest.csv",
        RAW / "aewr_2023_oews_occupation" / "flag",
    )
    # docs-ground:end aewr-source-contract
    return (artifacts,)


@app.cell
def _(INTERMEDIATE, artifacts, write_artifacts):
    write_artifacts(artifacts, INTERMEDIATE)
    for _name, _artifact in artifacts.items():
        print(_name, getattr(_artifact, "shape", "source manifest"))
if __name__ == "__main__":
    app.run()
