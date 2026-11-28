# Purpose: Show how much the split of year-to-year change in certified
#   workers into continuing filers, entry and exit depends on the filer
#   identifier, and which filers carry the worker-weighted exits.
# Inputs: scratch filer_cases.parquet (01_filer_records.py) and
#   filer_ids.parquet (02_rule_harness.py).
# Outputs (scratch): usage_split.csv, usage_missed_return.csv,
#   usage_largest_exits.csv.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import polars as pl

    from h2a.paths import ROOT

    return Path, ROOT, mo, os, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Program usage and the filer identifier

    Total certified workers in a year do not depend on any linkage. The split
    of their change into continuing filers, entering filers and exiting
    filers does: a missed link moves a continuing filer's workers into both
    entry and exit.

    Shares below are of certified workers in the base year. The FEIN
    (FY2024 on) is the answer key.
    """)


@app.cell
def _(Path, ROOT, os, pl):
    scratch = Path(os.environ.get("EMPLOYER_PANEL_SCRATCH", "")).expanduser().resolve()
    if not os.environ.get("EMPLOYER_PANEL_SCRATCH") or scratch == ROOT or ROOT in scratch.parents:
        raise ValueError("Set EMPLOYER_PANEL_SCRATCH to a directory outside the repository")
    RULE_IDS = ["filer_R0", "filer_R5", "filer_R7", "filer_R8"]
    cases = (
        pl.read_parquet(scratch / "filer_cases.parquet")
        .join(
            pl.read_parquet(scratch / "filer_ids.parquet").select("fiscal_year", "case_number", *RULE_IDS),
            on=["fiscal_year", "case_number"],
            how="inner",
        )
        .with_columns(pl.col("certified_workers").fill_null(0.0))
    )
    if cases.is_empty() or cases.select("fiscal_year", "case_number").is_duplicated().any():
        raise ValueError("filer cases are empty or not unique by fiscal year and case")
    return RULE_IDS, cases, scratch


@app.cell
def _(RULE_IDS, cases, pl):
    def split_change(identifier, base_year):
        base = (
            cases.filter(pl.col("fiscal_year") == base_year)
            .group_by(identifier)
            .agg(pl.col("certified_workers").sum().alias("workers_base"))
        )
        later = (
            cases.filter(pl.col("fiscal_year") == base_year + 1)
            .group_by(identifier)
            .agg(pl.col("certified_workers").sum().alias("workers_later"))
        )
        both = base.join(later, on=identifier, how="full", coalesce=True)
        total_base = both["workers_base"].fill_null(0.0).sum()
        stay = both.filter(pl.col("workers_base").is_not_null() & pl.col("workers_later").is_not_null())
        leave = both.filter(pl.col("workers_later").is_null())
        enter = both.filter(pl.col("workers_base").is_null())
        return {
            "identifier": identifier,
            "base_year": base_year,
            "filers_base": base.height,
            "workers_base": round(total_base),
            "total_change": round((both["workers_later"].fill_null(0.0).sum() - total_base) / total_base, 4),
            "continuing_filers": round((stay["workers_later"].sum() - stay["workers_base"].sum()) / total_base, 4),
            "entry": round(enter["workers_later"].sum() / total_base, 4),
            "exit": round(-leave["workers_base"].sum() / total_base, 4),
            "share_of_filers_exiting": round(leave.height / base.height, 4),
        }

    last_year = int(cases["fiscal_year"].max())
    first_year = int(cases["fiscal_year"].min())
    fein_first = int(cases.filter(pl.col("fein").fill_null("") != "")["fiscal_year"].min())
    split_rows = [
        split_change(_identifier, _year) for _identifier in RULE_IDS for _year in range(first_year, last_year)
    ] + [split_change("fein", _year) for _year in range(fein_first, last_year)]
    usage_split = pl.DataFrame(split_rows).sort("base_year", "identifier")

    # Cases whose FEIN files again but whose rule identifier does not.
    missed_rows = []
    for _year in range(fein_first, last_year):
        _later = cases.filter(pl.col("fiscal_year") == _year + 1)
        _returning = cases.filter(
            (pl.col("fiscal_year") == _year) & pl.col("fein").is_in(_later["fein"].implode())
        )
        for _identifier in RULE_IDS:
            _missed = _returning.filter(~pl.col(_identifier).is_in(_later[_identifier].implode()))
            missed_rows.append({
                "identifier": _identifier,
                "base_year": _year,
                "returning_cases_missed": round(_missed.height / _returning.height, 4),
                "returning_workers_missed": round(
                    _missed["certified_workers"].sum() / _returning["certified_workers"].sum(), 4
                ),
            })
    usage_missed_return = pl.DataFrame(missed_rows)

    # The ten largest exiting filers under R7, by base-year workers.
    exit_frames = []
    for _year in range(first_year, last_year):
        _base = (
            cases.filter(pl.col("fiscal_year") == _year)
            .group_by("filer_R7")
            .agg(
                pl.col("certified_workers").sum().alias("workers"),
                pl.col("source_name_raw").first().alias("name"),
                pl.col("state").first(),
                pl.len().alias("cases"),
                pl.col("employer_record_role").first().alias("role"),
            )
        )
        _later_ids = cases.filter(pl.col("fiscal_year") == _year + 1).select("filer_R7").unique()
        _gone = _base.join(_later_ids, on="filer_R7", how="anti").sort("workers", descending=True)
        exit_frames.append(
            _gone.head(10).select(
                pl.lit(_year).alias("base_year"),
                "name",
                "state",
                "cases",
                "workers",
                "role",
                (pl.col("workers") / _base["workers"].sum()).round(4).alias("share_of_all_workers"),
                pl.lit(round(_gone["workers"].sum() / _base["workers"].sum(), 4)).alias("exit_share_of_workers"),
                pl.lit(round(_gone.head(10)["workers"].sum() / _gone["workers"].sum(), 4)).alias(
                    "top10_share_of_exit_workers"
                ),
            )
        )
    usage_largest_exits = pl.concat(exit_frames)
    return usage_largest_exits, usage_missed_return, usage_split


@app.cell
def _(mo, pl, scratch, usage_largest_exits, usage_missed_return, usage_split):
    usage_split.write_csv(scratch / "usage_split.csv")
    usage_missed_return.write_csv(scratch / "usage_missed_return.csv")
    usage_largest_exits.write_csv(scratch / "usage_largest_exits.csv")
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=240, fmt_str_lengths=60, tbl_hide_dataframe_shape=True):
        print(usage_split.filter(pl.col("base_year") >= 2024))
        print(usage_missed_return)
        print(usage_split.filter(pl.col("identifier") == "filer_R7"))
        print(
            usage_largest_exits.group_by("base_year", maintain_order=True).agg(
                pl.col("exit_share_of_workers").first(),
                pl.col("top10_share_of_exit_workers").first(),
                pl.col("name").head(3),
                pl.col("workers").head(3),
            )
        )
    mo.vstack([mo.ui.table(usage_split), mo.ui.table(usage_missed_return), mo.ui.table(usage_largest_exits)])
    return


if __name__ == "__main__":
    app.run()
