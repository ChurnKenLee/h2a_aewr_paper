import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import sys

    from h2a.paths import CODE

    sys.path.insert(0, str(CODE / "designs" / "wage_reforms"))
    from report import validate

    return (validate,)


@app.cell
def _(validate):
    validate()


if __name__ == "__main__":
    app.run()
