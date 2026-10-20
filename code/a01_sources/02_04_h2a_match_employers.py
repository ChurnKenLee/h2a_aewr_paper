# Purpose: Link role-classified H-2A worksite businesses using FY2008--2024 evidence.
# Inputs: h2a_with_fips.parquet, h2a_addendum_b_with_fips.parquet, and manual labels.
# Outputs: canonical v2 crosswalk plus membership, edge, cluster, vintage, and review audits.

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import tempfile
    from pathlib import Path

    import marimo as mo
    import polars as pl

    from h2a.employer_linkage import (
        INPUT_MAX_YEAR,
        LINKAGE_UNIT,
        LINKAGE_VERSION,
        TIERS,
        build_linkage_artifacts,
        finalize_linkage_artifacts,
        read_review_labels,
    )
    from h2a.paths import INTERMEDIATE, RAW

    return (
        INPUT_MAX_YEAR,
        INTERMEDIATE,
        LINKAGE_UNIT,
        LINKAGE_VERSION,
        Path,
        RAW,
        TIERS,
        build_linkage_artifacts,
        finalize_linkage_artifacts,
        mo,
        os,
        pl,
        read_review_labels,
        tempfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # H-2A worksite-business linkage v2

    This producer uses FY2008--2024 identity evidence to create three nested,
    deterministic worksite-business partitions. Application outcomes, worker
    counts, wages, hours, and downstream design fields never enter matching.

    The application always writes the seeded 200-pair review sample first. It
    promotes the canonical crosswalk and audits only after every sampled pair
    has a valid manual `true`, `false`, or `uncertain` label.
    """)


@app.cell
def _(INTERMEDIATE, pl):
    h2a_with_fips = pl.read_parquet(INTERMEDIATE / "h2a_with_fips.parquet")
    h2a_addendum_b_with_fips = pl.read_parquet(
        INTERMEDIATE / "h2a_addendum_b_with_fips.parquet"
    )
    return h2a_addendum_b_with_fips, h2a_with_fips


@app.cell
def _(build_linkage_artifacts, h2a_addendum_b_with_fips, h2a_with_fips):
    candidate_artifacts = build_linkage_artifacts(
        h2a_with_fips,
        h2a_addendum_b_with_fips,
        verify_determinism=True,
    )
    return (candidate_artifacts,)


@app.cell
def _(candidate_artifacts):
    review_sample = candidate_artifacts["review_sample"].select(
        [
            "pair_id",
            "left_name",
            "left_trade_name",
            "left_address",
            "left_city",
            "left_state",
            "left_postal_code",
            "left_phone",
            "left_fein",
            "right_name",
            "right_trade_name",
            "right_address",
            "right_city",
            "right_state",
            "right_postal_code",
            "right_phone",
            "right_fein",
        ]
    )
    return (review_sample,)


@app.cell
def _(Path, os, pl, tempfile):
    def write_parquet_atomic(frame: pl.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            frame.write_parquet(temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def write_csv_atomic(frame: pl.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            frame.write_csv(temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    return write_csv_atomic, write_parquet_atomic


@app.cell
def _(INTERMEDIATE, review_sample, write_csv_atomic, write_parquet_atomic):
    review_sample_csv_path = INTERMEDIATE / "h2a_employer_linkage_review_sample.csv"
    review_sample_parquet_path = (
        INTERMEDIATE / "h2a_employer_linkage_review_sample.parquet"
    )
    write_csv_atomic(review_sample, review_sample_csv_path)
    write_parquet_atomic(review_sample, review_sample_parquet_path)
    return (review_sample_csv_path,)


@app.cell
def _(RAW, read_review_labels):
    review_labels_path = RAW / "h2a" / "employer_linkage_review_labels.csv"
    review_labels = read_review_labels(review_labels_path)
    return review_labels, review_labels_path


@app.cell
def _(candidate_artifacts, finalize_linkage_artifacts, review_labels):
    finalized_artifacts = finalize_linkage_artifacts(candidate_artifacts, review_labels)
    return (finalized_artifacts,)


@app.cell
def _(
    INTERMEDIATE,
    finalized_artifacts,
    write_csv_atomic,
    write_parquet_atomic,
):
    parquet_outputs = {
        "crosswalk": "h2a_employer_crosswalk.parquet",
        "membership": "h2a_employer_record_membership.parquet",
        "edge_audit": "h2a_employer_edge_audit.parquet",
        "cluster_diagnostics": "h2a_employer_cluster_diagnostics.parquet",
        "vintage_comparison": "h2a_employer_linkage_vintage_comparison.parquet",
        "review_sample": "h2a_employer_linkage_review_labeled.parquet",
        "review_report": "h2a_employer_linkage_review_report.parquet",
        "source_diagnostics": "h2a_employer_linkage_source_diagnostics.parquet",
    }
    output_paths = []
    for artifact_name, filename in parquet_outputs.items():
        output_path = INTERMEDIATE / filename
        write_parquet_atomic(finalized_artifacts[artifact_name], output_path)
        output_paths.append(output_path)
    review_report_csv = INTERMEDIATE / "h2a_employer_linkage_review_report.csv"
    write_csv_atomic(finalized_artifacts["review_report"], review_report_csv)
    output_paths.append(review_report_csv)
    return (output_paths,)


@app.cell
def _(
    INPUT_MAX_YEAR,
    LINKAGE_UNIT,
    LINKAGE_VERSION,
    TIERS,
    candidate_artifacts,
    finalized_artifacts,
    mo,
    output_paths,
    review_labels_path,
    review_sample_csv_path,
):
    cluster_summary = (
        finalized_artifacts["cluster_diagnostics"]
        .group_by("linkage_tier")
        .len(name="linked_worksite_businesses")
        .sort("linkage_tier")
    )
    edge_summary = (
        finalized_artifacts["edge_audit"]
        .group_by("linkage_tier", "decision", "rejection_reason")
        .len()
        .sort("linkage_tier", "decision", "rejection_reason")
    )
    mo.vstack(
        [
            mo.md(
                f"""
                - **Linkage unit:** `{LINKAGE_UNIT}`
                - **Version:** `{LINKAGE_VERSION}`
                - **Evidence vintage:** FY2008--{INPUT_MAX_YEAR}
                - **Tiers:** `{", ".join(TIERS)}`
                - **Manual labels:** `{review_labels_path}`
                - **Review sample:** `{review_sample_csv_path}`
                """
            ),
            mo.md("## Source-role coverage"),
            mo.ui.table(candidate_artifacts["source_diagnostics"]),
            mo.md("## Linked worksite businesses"),
            mo.ui.table(cluster_summary),
            mo.md("## Edge decisions"),
            mo.ui.table(edge_summary),
            mo.md("## Manual-review report"),
            mo.ui.table(finalized_artifacts["review_report"]),
            mo.md("## Published artifacts"),
            mo.md("\n".join(f"- `{path}`" for path in output_paths)),
        ]
    )


if __name__ == "__main__":
    app.run()
