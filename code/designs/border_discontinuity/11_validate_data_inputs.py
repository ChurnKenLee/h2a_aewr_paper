# ruff: noqa: N999
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import sys

    from h2a.paths import CODE

    sys.path.insert(0, str(CODE / "designs/border_discontinuity"))
    from data_input_validation import validate_input_artifacts

    return (validate_input_artifacts,)


@app.cell
def _(validate_input_artifacts):
    print(validate_input_artifacts())


if __name__ == "__main__":
    app.run()
