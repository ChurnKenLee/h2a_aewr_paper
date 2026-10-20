import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import sys

    from h2a.paths import CODE

    sys.path.insert(0, str(CODE / "designs" / "wage_reforms"))
    from analysis import build

    return (build,)


@app.cell
def _(build):
    build()


if __name__ == "__main__":
    app.run()
