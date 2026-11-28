"""Evaluate certification-space producer incidence bounds under the envelope theorem."""

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

    from envelope import (
        ANNUAL_SERIES_PATH,
        BOUNDS_GRID_PATH,
        CONTRACT_PATH,
        DETAIL_PATH,
        REPORT_PATH,
        SKILL_RATES_PATH,
        SOURCE_PANEL,
        TABLE_PATH,
        add_coverage_adjustment,
        atomic_write_frame,
        atomic_write_text,
        build_county_year_detail,
        build_welfare_bounds_latex,
        build_welfare_bounds_report,
        evaluate_annual_series,
        evaluate_bounds_grid,
        load_oct2025_statutory_panel,
        load_shared_welfare_panel,
        load_welfare_contract,
        write_welfare_manifest,
    )

    ELASTICITY_DEFINITION = "dlog_certified_hours_dlog_real_aewr"
    APPLICATION_POPULATION = "all_shared_panel_county_years"
    COUNTERFACTUAL_DELTA = 1.0

    # docs-ground:start welfare-design-contract
    # The v2 calculation evaluates static partial-equilibrium producer incidence
    # bounds across a user-authorized certified-hours elasticity scenario grid
    # anchored to the complete FY2026 Q4 Track 1 statutory PPML hours jackknife
    # lower bound (never misrepresenting a bounds grid as a single point estimate).
    # It applies every scenario across all shared-panel counties in 2011--2025
    # for an anticipated +1 change in the AEWR (real PPI-2012 and 2024 nominal,
    # plus secondary +-10% and Oct. 2, 2025 statutory cut comparisons), decomposes
    # first-order envelope transfers from second-order behavioral cushions across
    # observed, coverage-adjusted, and F1 parametric exposures (x in {0.85, 0.90,
    # 0.94, 1.00}) and offered-wage pass-through sensitivity, and reports results
    # strictly in certification space. Paid-hours incidence equals rho times every
    # reported monetary result.
    # docs-ground:end welfare-design-contract

    return (
        ANNUAL_SERIES_PATH,
        APPLICATION_POPULATION,
        BOUNDS_GRID_PATH,
        CONTRACT_PATH,
        COUNTERFACTUAL_DELTA,
        DETAIL_PATH,
        ELASTICITY_DEFINITION,
        REPORT_PATH,
        SKILL_RATES_PATH,
        SOURCE_PANEL,
        TABLE_PATH,
        add_coverage_adjustment,
        atomic_write_frame,
        atomic_write_text,
        build_county_year_detail,
        build_welfare_bounds_latex,
        build_welfare_bounds_report,
        evaluate_annual_series,
        evaluate_bounds_grid,
        load_oct2025_statutory_panel,
        load_shared_welfare_panel,
        load_welfare_contract,
        write_welfare_manifest,
    )


@app.cell
def _(
    CONTRACT_PATH,
    SKILL_RATES_PATH,
    SOURCE_PANEL,
    add_coverage_adjustment,
    load_oct2025_statutory_panel,
    load_shared_welfare_panel,
    load_welfare_contract,
):
    contract = load_welfare_contract(CONTRACT_PATH)
    panel = add_coverage_adjustment(
        load_shared_welfare_panel(
            SOURCE_PANEL,
            start_year=contract["start_year"],
            end_year=contract["provisional_end_year"],
            real_aewr_column=contract["real_aewr_column"],
            nominal_aewr_column=contract["nominal_aewr_column"],
        )
    )
    oct2025_panel = load_oct2025_statutory_panel(
        panel,
        SKILL_RATES_PATH,
        baseline_year=contract["snapshot_year"],
        pre_cut_aewr_year=contract["provisional_end_year"],
    )
    return contract, oct2025_panel, panel


@app.cell
def _(
    build_county_year_detail,
    contract,
    evaluate_annual_series,
    evaluate_bounds_grid,
    oct2025_panel,
    panel,
):
    bounds_grid = evaluate_bounds_grid(panel, oct2025_panel, contract)
    annual_series = evaluate_annual_series(panel, contract)
    county_detail = build_county_year_detail(panel, contract)
    return annual_series, bounds_grid, county_detail


@app.cell
def _(
    ANNUAL_SERIES_PATH,
    BOUNDS_GRID_PATH,
    DETAIL_PATH,
    REPORT_PATH,
    TABLE_PATH,
    annual_series,
    atomic_write_frame,
    atomic_write_text,
    bounds_grid,
    build_welfare_bounds_latex,
    build_welfare_bounds_report,
    contract,
    county_detail,
    write_welfare_manifest,
):
    atomic_write_frame(county_detail, DETAIL_PATH)
    atomic_write_frame(bounds_grid, BOUNDS_GRID_PATH)
    atomic_write_frame(annual_series, ANNUAL_SERIES_PATH)
    atomic_write_text(build_welfare_bounds_latex(bounds_grid, contract), TABLE_PATH)
    atomic_write_text(
        build_welfare_bounds_report(bounds_grid, annual_series, contract),
        REPORT_PATH,
    )
    write_welfare_manifest(
        contract,
        [DETAIL_PATH, BOUNDS_GRID_PATH, ANNUAL_SERIES_PATH, TABLE_PATH, REPORT_PATH],
    )
    print(
        f"Wrote certification-space welfare bounds: {bounds_grid.height} grid rows, "
        f"{annual_series.height} annual rows, {county_detail.height} county-year rows."
    )


if __name__ == "__main__":
    app.run()
