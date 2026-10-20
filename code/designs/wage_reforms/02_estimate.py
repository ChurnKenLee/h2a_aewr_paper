import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import sys

    from h2a.paths import CODE

    sys.path.insert(0, str(CODE / "designs" / "wage_reforms"))
    from analysis import estimate

    return (estimate,)


@app.cell
def _(estimate):
    estimate()


if __name__ == "__main__":
    app.run()
