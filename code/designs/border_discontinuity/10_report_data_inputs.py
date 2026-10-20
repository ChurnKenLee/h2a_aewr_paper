# ruff: noqa: N999
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import sys

    from h2a.paths import CODE

    sys.path.insert(0, str(CODE / "designs/border_discontinuity"))
    from data_input_report import build_report

    return (build_report,)


@app.cell
def _(build_report):
    print(build_report())


if __name__ == "__main__":
    app.run()
