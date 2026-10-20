# Purpose: Prepare and optionally geocode public worksite addresses, with cached provenance.
# Inputs: final Addendum B location file and cached FY2020--2024 main disclosure worksite streets.
# Outputs: h2a_worksite_geocode_{records,addresses,manifest}, requests/ and h2a_worksite_geocodes.parquet.
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import os

    from h2a.paths import INTERMEDIATE, RAW
    from h2a.worksite_geocoding import build_worksite_geocodes

    return INTERMEDIATE, RAW, build_worksite_geocodes, os


@app.cell
def _(INTERMEDIATE, RAW, build_worksite_geocodes, os):
    print(build_worksite_geocodes(RAW, INTERMEDIATE,
                                 acquire=os.getenv("H2A_RUN_PUBLIC_GEOCODER", "0") == "1"))


if __name__ == "__main__":
    app.run()
