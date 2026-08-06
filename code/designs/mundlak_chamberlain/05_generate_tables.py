"""Render Mundlak-Chamberlain tables with Posit's gt ecosystem.

The project already depends on ``great_tables``, the Python implementation of
the gt table grammar.  This script deliberately consumes the frozen CSV
results rather than recomputing estimates, so the values shown in HTML and
LaTeX are identical to the CCV/delta-method outputs produced by the R scripts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from great_tables import GT

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

CCV_NOTE = (
    "Standard errors use the continuous-treatment design-covariance CCV. "
    "The fitted residual vector is held fixed; the 17 observed AEWR policy "
    "paths are assigned cyclically across the 17 region labels; OLS is "
    "re-solved in every equally likely state; and the centered coefficient-"
    "error covariance uses probability weight 1/17. Intervals use t(16)."
)


def effect_cell(estimate: float, standard_error: float, digits: int = 2) -> str:
    """Format an estimate with its CCV standard error in parentheses."""

    return f"{estimate:,.{digits}f} ({standard_error:,.{digits}f})"


def write_gt_table(table: GT, stem: str) -> None:
    """Write both supported gt renderings from a single table object."""

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.write_raw_html(
        TABLE_DIR / f"{stem}.html",
        # Non-inlined CSS is Great Tables' dependency-free HTML path. The
        # optional css-inline extra is intentionally not required by this
        # project's locked environment.
        inline_css=False,
        make_page=True,
    )
    (TABLE_DIR / f"{stem}.tex").write_text(
        table.as_latex(tbl_pos="!htbp"),
        encoding="utf-8",
    )


def common_style(table: GT) -> GT:
    """Apply the restrained manuscript style shared by all retained tables."""

    return table.tab_options(
        table_font_size="10pt",
        heading_align="left",
        data_row_padding="4px",
        source_notes_font_size="8pt",
        table_width="100%",
    )


def dynamic_effects_table() -> GT:
    effects = pd.read_csv(TABLE_DIR / "mc_average_marginal_effects.csv")
    effects = effects.loc[
        (effects["scope"] == "overall")
        & (effects["scope_value"] == "all")
        & (effects["standardization"] == "county_year_equal")
    ].copy()
    outcome_order = effects["outcome_label"].drop_duplicates().tolist()
    effects["horizon"] = effects["horizon"].map(
        {
            "contemporaneous": "Current",
            "one_year": "One-year lag",
            "two_year": "Two-year lag",
        }
    )
    effects["effect"] = [
        effect_cell(estimate, standard_error)
        for estimate, standard_error in zip(
            effects["estimate"], effects["standard_error"], strict=True
        )
    ]
    display = (
        effects.pivot(
            index="outcome_label",
            columns="horizon",
            values="effect",
        )
        .reindex(
            index=outcome_order,
            columns=["Current", "One-year lag", "Two-year lag"],
        )
        .reset_index()
    )
    table = (
        GT(display, rowname_col="outcome_label")
        .tab_header(
            title="Dynamic Average Marginal Effects of AEWR Growth",
            subtitle="Estimate (design-covariance CCV standard error)",
        )
        .tab_stubhead(label="Outcome (reported unit)")
        .cols_align(align="center", columns=list(display.columns[1:]))
        .tab_source_note(
            "Effects are county-year-standardized responses to one log "
            "percentage point of AEWR growth; the any-application outcome is "
            "reported in percentage points. "
            + CCV_NOTE
        )
    )
    return common_style(table)


def heterogeneity_table() -> GT:
    effects = pd.read_csv(TABLE_DIR / "mc_average_marginal_effects.csv")
    effects = effects.loc[
        (effects["outcome_id"] == "certified_positions")
        & (effects["horizon"] == "contemporaneous")
        & effects["scope"].isin(
            ["mc_binding_quartile", "mc_baseline_h2a_quartile"]
        )
    ].copy()
    effects["dimension"] = effects["scope"].map(
        {
            "mc_binding_quartile": "Baseline AEWR bite",
            "mc_baseline_h2a_quartile": "Baseline H-2A intensity",
        }
    )
    effects["quartile"] = "Q" + effects["scope_value"].astype(int).astype(str)
    effects["effect"] = [
        effect_cell(estimate, standard_error)
        for estimate, standard_error in zip(
            effects["estimate"], effects["standard_error"], strict=True
        )
    ]
    display = (
        effects.pivot(index="dimension", columns="quartile", values="effect")
        .reindex(columns=["Q1", "Q2", "Q3", "Q4"])
        .reset_index()
    )
    table = (
        GT(display, rowname_col="dimension")
        .tab_header(
            title="Contemporaneous Certified-Position Effects by Baseline Quartile",
            subtitle="Average marginal effect (design-covariance CCV standard error)",
        )
        .tab_stubhead(label="Predetermined dimension")
        .cols_align(align="center", columns=["Q1", "Q2", "Q3", "Q4"])
        .tab_source_note(
            "Effects are certified positions per 1,000 baseline farm workers "
            "per one log percentage point of contemporaneous AEWR growth. "
            + CCV_NOTE
        )
    )
    return common_style(table)


def support_table() -> GT:
    display = pd.read_csv(TABLE_DIR / "mc_treatment_support.csv")[
        [
            "year",
            "assignment_cells",
            "mean",
            "standard_deviation",
            "minimum",
            "p10",
            "p90",
            "maximum",
        ]
    ]
    numeric_columns = [
        "mean",
        "standard_deviation",
        "minimum",
        "p10",
        "p90",
        "maximum",
    ]
    table = (
        GT(display)
        .tab_header(title="Support for Annual AEWR Growth")
        .cols_label(
            year="Year",
            assignment_cells="Cells",
            mean="Mean",
            standard_deviation="SD",
            minimum="Min.",
            p10="P10",
            p90="P90",
            maximum="Max.",
        )
        .fmt_integer(columns=["year", "assignment_cells"], use_seps=False)
        .fmt_number(columns=numeric_columns, decimals=2)
        .cols_align(align="center")
        .tab_source_note(
            "AEWR growth is 100 times the log change in the applicable "
            "regional AEWR. Each year contains 17 assignment cells; county "
            "replication does not create additional policy paths."
        )
    )
    return common_style(table)


def lead_placebo_table() -> GT:
    display = pd.read_csv(TABLE_DIR / "mc_lead_placebo_effects.csv")
    display = display.loc[display["estimand"] == "finite_dose_change"].copy()
    display["effect"] = [
        effect_cell(estimate, standard_error)
        for estimate, standard_error in zip(
            display["estimate"], display["standard_error"], strict=True
        )
    ]
    display = display[["outcome_label", "effect", "p_value"]]
    table = (
        GT(display, rowname_col="outcome_label")
        .tab_header(
            title="One-Year-Ahead AEWR-Growth Placebo Effects",
            subtitle="Five-log-point placebo with design-covariance CCV inference",
        )
        .tab_stubhead(label="Outcome")
        .cols_label(effect="Estimate (CCV SE)", p_value="p-value")
        .fmt_number(columns="p_value", decimals=3)
        .cols_align(align="center", columns=["effect", "p_value"])
        .tab_source_note(
            "A nonzero lead can indicate anticipation, feedback, omitted "
            "dynamics, or misspecification; a zero lead does not establish the "
            "identifying assumptions. "
            + CCV_NOTE
        )
    )
    return common_style(table)


def ccv_coefficient_table() -> GT:
    parameters = pd.read_csv(TABLE_DIR / "mc_parameter_estimates.csv")
    labels = (
        pd.read_csv(TABLE_DIR / "mc_model_diagnostics.csv")
        [["outcome_id", "outcome_label"]]
        .drop_duplicates()
    )
    display = parameters.loc[
        (parameters["model_id"] == "twfe_benchmark")
        & parameters["term"].isin(
            ["mc_dose_current", "mc_dose_lag1", "mc_dose_lag2"]
        )
    ].merge(labels, on="outcome_id", how="left")
    display["horizon"] = display["term"].map(
        {
            "mc_dose_current": "Current",
            "mc_dose_lag1": "Lag 1",
            "mc_dose_lag2": "Lag 2",
        }
    )
    display["ccv_to_cluster_ratio"] = (
        display["standard_error"]
        / display["conventional_cluster_standard_error"]
    )
    display = display[
        [
            "outcome_label",
            "horizon",
            "estimate",
            "standard_error",
            "conventional_cluster_standard_error",
            "ccv_to_cluster_ratio",
        ]
    ].rename(
        columns={
            "outcome_label": "Outcome",
            "horizon": "Horizon",
            "estimate": "Estimate",
            "standard_error": "CCV SE",
            "conventional_cluster_standard_error": "Region-clustered SE",
            "ccv_to_cluster_ratio": "CCV / clustered",
        }
    )
    table = (
        GT(display, rowname_col="Horizon", groupname_col="Outcome")
        .tab_header(
            title="TWFE AEWR-Growth Coefficients and CCV Standard Errors",
            subtitle="Conventional clustered standard errors shown only for comparison",
        )
        .tab_stubhead(label="Horizon")
        .fmt_number(
            columns=[
                "Estimate",
                "CCV SE",
                "Region-clustered SE",
                "CCV / clustered",
            ],
            decimals=3,
        )
        .cols_align(align="center")
        .tab_source_note(CCV_NOTE)
    )
    return common_style(table)


def main() -> None:
    tables = {
        "table_mc_dynamic_effects": dynamic_effects_table(),
        "table_mc_heterogeneity": heterogeneity_table(),
        "table_mc_support": support_table(),
        "table_mc_lead_placebos": lead_placebo_table(),
        "table_mc_ccv_coefficients": ccv_coefficient_table(),
    }
    for stem, table in tables.items():
        write_gt_table(table, stem)
    print(f"Rendered {len(tables)} Great Tables to HTML and LaTeX.")


if __name__ == "__main__":
    main()
