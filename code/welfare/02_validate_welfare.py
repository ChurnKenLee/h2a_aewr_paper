"""Validate certification-space welfare bounds provenance and analytic identities."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import sys

    from h2a.paths import ROOT

    _welfare_dir = str(ROOT / "code" / "welfare")
    if _welfare_dir not in sys.path:
        sys.path.insert(0, _welfare_dir)

    from envelope import validate_welfare_outputs

    return (validate_welfare_outputs,)


@app.cell
def _(validate_welfare_outputs):
    validation_summary = validate_welfare_outputs()
    print(
        "Validated certification-space welfare bounds: "
        f"{validation_summary['checks_passed']}/{validation_summary['checks_passed']} "
        f"checks passed ({validation_summary['grid_rows']} grid rows, "
        f"{validation_summary['annual_rows']} annual rows)."
    )


if __name__ == "__main__":
    app.run()
