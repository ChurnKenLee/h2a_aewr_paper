# Purpose: Derive audited worksite-month allocated certification capacity.
# Inputs: h2a_allocated_worksites.parquet, owned by B01/01.
# Outputs: h2a_worksite_months/, county-year exposure, case/row audits, coverage and manifest.
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    from h2a.paths import INTERMEDIATE
    from h2a.worksite_exposure import build_worksite_exposure

    return INTERMEDIATE, build_worksite_exposure


@app.cell
def _(INTERMEDIATE, build_worksite_exposure):
    print(build_worksite_exposure(INTERMEDIATE))


if __name__ == "__main__":
    app.run()
