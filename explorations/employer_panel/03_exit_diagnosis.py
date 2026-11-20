# Purpose: Split the worksite businesses (v2 balanced identifier) that are
#   present in FY2024 and absent in FY2025 by what happened to their filer.
# Inputs: scratch filer_cases.parquet and business_rows.parquet from
#   01_filer_records.py; h2a_allocated_worksites.parquet.
# Outputs (scratch): exit_diagnosis.csv, exit_diagnosis_pairs.csv.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import polars as pl
    from rapidfuzz import fuzz, process

    from h2a.paths import INTERMEDIATE, ROOT

    return INTERMEDIATE, Path, ROOT, fuzz, mo, os, pl, process


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Where do apparent exits of worksite businesses come from?

    A worksite business that is in FY2024 and not in FY2025 falls into one of
    four groups, using the FEIN of the cases that named it in FY2024:

    1. its filer did not file in FY2025;
    2. its filer filed, and named a business with the same core name (a link
       the current rules missed);
    3. its filer filed, and named a business with a similar name;
    4. its filer filed and named nothing like it.
    """)


@app.cell
def _(INTERMEDIATE, Path, ROOT, os, pl):
    scratch = Path(os.environ.get("EMPLOYER_PANEL_SCRATCH", "")).expanduser().resolve()
    if not os.environ.get("EMPLOYER_PANEL_SCRATCH") or scratch == ROOT or ROOT in scratch.parents:
        raise ValueError("Set EMPLOYER_PANEL_SCRATCH to a directory outside the repository")
    BASE_YEAR = 2024
    filers = pl.read_parquet(scratch / "filer_cases.parquet").filter(
        pl.col("fiscal_year").is_in([BASE_YEAR, BASE_YEAR + 1])
    )
    businesses = (
        pl.read_parquet(scratch / "business_rows.parquet")
        .filter(pl.col("fiscal_year").is_in([BASE_YEAR, BASE_YEAR + 1]))
        .join(
            filers.select("fiscal_year", "case_number", "fein", "labor_contractor"),
            on=["fiscal_year", "case_number"],
            how="left",
        )
    )
    if businesses["fein"].null_count():
        raise ValueError("a worksite-business row has no filer case")
    workers = (
        pl.scan_parquet(INTERMEDIATE / "h2a_allocated_worksites.parquet")
        .filter((pl.col("fiscal_year") == BASE_YEAR) & pl.col("employer_is_worksite_business"))
        .group_by("employer_id_balanced")
        .agg(pl.col("adjusted_nbr_workers_certified").sum().alias("certified_workers"))
        .collect()
    )
    return BASE_YEAR, businesses, workers


@app.cell
def _(BASE_YEAR, businesses, fuzz, pl, process, workers):
    base = businesses.filter(pl.col("fiscal_year") == BASE_YEAR)
    later = businesses.filter(pl.col("fiscal_year") == BASE_YEAR + 1)
    later_ids = later.select("employer_id_balanced").unique()
    later_feins = later.select("fein").unique()

    gone_rows = base.join(later_ids, on="employer_id_balanced", how="anti")
    # Core names each filer names in the later year.
    later_names = later.group_by("fein").agg(pl.col("k_core").unique().alias("later_names"))
    gone_rows = gone_rows.join(later_feins.with_columns(pl.lit(True).alias("filer_returns")), on="fein", how="left").join(
        later_names, on="fein", how="left"
    )

    def best_match(row):
        names = row["later_names"]
        if not names or not row["k_core"]:
            return {"best_name": None, "best_ratio": 0.0}
        match = process.extractOne(row["k_core"], names, scorer=fuzz.ratio)
        return {"best_name": match[0], "best_ratio": float(match[1])}

    matched = gone_rows.with_columns(
        pl.struct("k_core", "later_names")
        .map_elements(best_match, return_dtype=pl.Struct({"best_name": pl.String, "best_ratio": pl.Float64}))
        .alias("best")
    ).unnest("best")
    row_groups = matched.with_columns(
        pl.when(pl.col("filer_returns").is_null())
        .then(pl.lit("1 filer did not file again"))
        .when(pl.col("best_ratio") >= 100)
        .then(pl.lit("2 filer filed; same core name named again"))
        .when(pl.col("best_ratio") >= 85)
        .then(pl.lit("3 filer filed; similar name named again"))
        .otherwise(pl.lit("4 filer filed; nothing like it named"))
        .alias("group")
    )
    # One group per business: the most favorable evidence across its rows.
    business_groups = (
        row_groups.sort("group", descending=False)
        .group_by("employer_id_balanced")
        .agg(
            pl.col("group").filter(pl.col("group").str.starts_with("2")).first().alias("g2"),
            pl.col("group").filter(pl.col("group").str.starts_with("3")).first().alias("g3"),
            pl.col("group").filter(pl.col("group").str.starts_with("4")).first().alias("g4"),
            pl.col("group").filter(pl.col("group").str.starts_with("1")).first().alias("g1"),
            pl.col("employer_record_role").first().alias("role"),
            pl.col("labor_contractor").any().alias("via_contractor"),
        )
        .with_columns(pl.coalesce("g2", "g3", "g4", "g1").alias("group"))
        .join(workers, on="employer_id_balanced", how="left")
        .with_columns(pl.col("certified_workers").fill_null(0.0))
    )
    base_ids = base.select("employer_id_balanced").unique().join(workers, on="employer_id_balanced", how="left")
    totals = {
        "businesses_base_year": base_ids.height,
        "workers_base_year": float(base_ids["certified_workers"].fill_null(0.0).sum()),
        "businesses_gone": business_groups.height,
    }
    diagnosis = (
        business_groups.group_by("group")
        .agg(
            pl.len().alias("businesses"),
            pl.col("certified_workers").sum().round(0).alias("certified_workers"),
            pl.col("via_contractor").mean().round(3).alias("share_named_by_a_contractor"),
            (pl.col("role") == "post2020_addendum_business").mean().round(3).alias("share_addendum_rows"),
        )
        .sort("group")
        .with_columns(
            (pl.col("businesses") / totals["businesses_gone"]).round(3).alias("share_of_gone"),
            (pl.col("businesses") / totals["businesses_base_year"]).round(3).alias("share_of_base_year_businesses"),
            (pl.col("certified_workers") / totals["workers_base_year"]).round(3).alias("share_of_base_year_workers"),
        )
    )
    pairs = (
        row_groups.filter(pl.col("group").str.starts_with("2") | pl.col("group").str.starts_with("3"))
        .select("group", "source_name_raw", "k_core", "best_name", "best_ratio", "city", "state", "zip", "fein")
        .unique(subset=["k_core", "best_name", "fein"])
        .sort("group", "best_ratio", "k_core")
    )
    return diagnosis, pairs, totals


@app.cell
def _(diagnosis, mo, pairs, pl, scratch, totals):
    diagnosis.write_csv(scratch / "exit_diagnosis.csv")
    pairs.write_csv(scratch / "exit_diagnosis_pairs.csv")
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=240, fmt_str_lengths=60, tbl_hide_dataframe_shape=True):
        print(totals)
        print(diagnosis)
        print(pairs.filter(pl.col("group").str.starts_with("2")).sample(12, seed=11).drop("group", "fein"))
        print(pairs.filter(pl.col("group").str.starts_with("3")).sample(12, seed=11).drop("group", "fein"))
    mo.vstack([mo.ui.table(diagnosis), mo.ui.table(pairs)])
    return


if __name__ == "__main__":
    app.run()
