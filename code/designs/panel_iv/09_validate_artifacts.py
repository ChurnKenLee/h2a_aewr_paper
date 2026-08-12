"""Validate QCEW county features and OEWS-hourly Panel-IV artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

ROOT = Path(__file__).parents[3]
INTERMEDIATE = ROOT / "data" / "intermediate"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

SUPPORTED_YEARS = tuple(range(2010, 2022))
QUARTERS = ("january", "april", "july", "october")
WAGE_ONLY_SPEC = "fls_county_wage_only_soft_rho010_v2"
PRIMARY_SPEC = "fls_county_wage_seasonal_composition_soft_rho010_v2"
PRIMARY_LABEL = "k5_d2_oews_hourly_wage_seasonal_composition_soft_rho010_center"
DONOR_WAGE_SPEC = "county_mapped_oews_area_big_six_hourly_v1"
DONOR_WAGE_SOURCE = "oews_area_big_six_hourly"


def require_columns(frame: pl.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def require_unique(frame: pl.DataFrame, keys: list[str], label: str) -> None:
    require_columns(frame, keys, label)
    if frame.group_by(keys).len().filter(pl.col("len") != 1).height:
        raise ValueError(f"{label} is not unique on {', '.join(keys)}")


def require_empty(frame: pl.DataFrame, message: str) -> None:
    if frame.height:
        raise ValueError(f"{message} ({frame.height:,} invalid rows)")


def read_required(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Panel-IV artifact is missing: {path}")
    if path.suffix == ".csv":
        return pl.read_csv(path)
    return pl.read_parquet(path)


def validate_qcew_and_shared_panel() -> None:
    quarterly = read_required(
        INTERMEDIATE / "qcew_county_ag_quarterly_employment.parquet"
    )
    require_unique(
        quarterly,
        ["county_fips", "year", "qtr", "industry_code"],
        "Quarterly QCEW",
    )
    require_empty(
        quarterly.filter(
            ~pl.col("county_fips").str.contains(r"^\d{5}$")
            | ~pl.col("industry_code").is_in(["111", "112"])
            | ~pl.col("qtr").is_between(1, 4)
        ),
        "Quarterly QCEW contains malformed keys",
    )
    require_empty(
        quarterly.filter(
            pl.col("qcew_employment_disclosed")
            != (
                pl.col("qcew_disclosed_ownership_components")
                == pl.col("qcew_reported_ownership_components")
            )
        ),
        "Quarterly QCEW disclosure aggregation is inconsistent",
    )
    require_empty(
        quarterly.filter(
            pl.col("qcew_employment_disclosed")
            & (
                ~pl.col("qcew_reference_month_emplvl").is_finite()
                | (pl.col("qcew_reference_month_emplvl") < 0)
            )
        ),
        "Disclosed quarterly QCEW values are invalid",
    )
    require_empty(
        quarterly.filter(
            ~pl.col("qcew_employment_disclosed")
            & pl.col("qcew_reference_month_emplvl").is_not_null()
        ),
        "Suppressed quarterly QCEW values must remain null",
    )

    for path in (
        INTERMEDIATE / "county_year_merged.parquet",
        PROCESSED / "county_year_panel.parquet",
    ):
        panel = read_required(path)
        require_unique(panel, ["county_fips", "year"], path.name)
        require_empty(
            panel.filter(~pl.col("county_fips").str.contains(r"^\d{5}$")),
            f"{path.name} contains malformed county identifiers",
        )
        for series in ("crop_sector", "animal_sector", "all_sectors"):
            disclosed = f"qcew_{series}_disclosed"
            employment = f"qcew_{series}_annual_avg_emplvl"
            wages = f"qcew_{series}_total_annual_wages"
            require_columns(panel, [disclosed, employment, wages], path.name)
            require_empty(
                panel.filter(
                    pl.col(disclosed).fill_null(False)
                    & (
                        ~pl.col(employment).is_finite()
                        | (pl.col(employment) < 0)
                        | ~pl.col(wages).is_finite()
                        | (pl.col(wages) < 0)
                    )
                ),
                f"{path.name} has invalid disclosed {series} QCEW values",
            )
            require_empty(
                panel.filter(
                    pl.col(disclosed).is_not_null()
                    & ~pl.col(disclosed)
                    & (pl.col(employment).is_not_null() | pl.col(wages).is_not_null())
                ),
                f"{path.name} exposes suppressed {series} QCEW values",
            )


def validate_fls_pairs() -> None:
    workers = read_required(INTERMEDIATE / "fls_region_quarterly_workers.parquet")
    wages = read_required(INTERMEDIATE / "fls_region_quarterly_wages.parquet")
    keys = ["aewr_region_id", "year", "quarter"]
    require_unique(workers, keys, "FLS quarterly workers")
    require_unique(wages, keys, "FLS quarterly wages")
    supported_workers = workers.filter(pl.col("year").is_in(SUPPORTED_YEARS))
    supported_wages = wages.filter(pl.col("year").is_in(SUPPORTED_YEARS))
    expected = 17 * len(SUPPORTED_YEARS) * len(QUARTERS)
    if supported_workers.height != expected or supported_wages.height != expected:
        raise ValueError("FLS worker/wage row coverage is incomplete for 2010-2021")
    paired = supported_workers.join(
        supported_wages,
        on=keys,
        how="inner",
        suffix="_wage",
        validate="1:1",
    )
    for column in ("source_zip", "release_year", "release_month", "release_day"):
        require_empty(
            paired.filter(pl.col(column) != pl.col(f"{column}_wage")),
            f"FLS worker and wage tables disagree on {column}",
        )
    require_empty(
        paired.filter(
            ~pl.col("quarter").is_in(QUARTERS)
            | ~pl.col("aewr_region_id").str.contains(r"^(?:[1-9]|1[0-7])$")
        ),
        "FLS paired keys are malformed",
    )
    published = paired.filter(pl.col("fls_pair_values_available"))
    if published.height != expected - 17:
        raise ValueError("Unexpected count of complete paired FLS survey weeks")
    value_columns = [
        "fls_hired_workers",
        "fls_field_hourly_wage",
        "fls_livestock_hourly_wage",
        "fls_field_livestock_hourly_wage",
        "fls_all_hired_hourly_wage",
    ]
    require_empty(
        published.filter(
            pl.any_horizontal(
                [~pl.col(column).is_finite() for column in value_columns]
            )
        ),
        "Available FLS pairs contain missing values",
    )
    gap = paired.filter(~pl.col("fls_pair_values_available"))
    require_empty(
        gap.filter(
            (pl.col("year") != 2011)
            | (pl.col("quarter") != "april")
            | (pl.col("fls_pair_value_status") != "survey_not_conducted")
        ),
        "Unavailable FLS pairs are not the documented April-2011 gap",
    )


def validate_calibration_and_instruments() -> None:
    weights = read_required(
        INTERMEDIATE / "panel_iv_fls_county_weight_summary.parquet"
    )
    require_unique(
        weights,
        ["aewr_region_id", "source_year", "county_fips", "specification"],
        "County calibration weights",
    )
    require_empty(
        weights.filter(
            ~pl.col("county_fips").str.contains(r"^\d{5}$")
            | ~pl.col("aewr_region_id").str.contains(r"^(?:[1-9]|1[0-7])$")
        ),
        "County calibration geographic keys are malformed",
    )
    sums = weights.group_by("aewr_region_id", "source_year", "specification").agg(
        pl.col("calibrated_center_weight").sum().alias("weight_sum")
    )
    if sums.height != 17 * len(SUPPORTED_YEARS) * 2:
        raise ValueError("County calibration cells are incomplete")
    require_empty(
        sums.filter((pl.col("weight_sum") - 1).abs() > 1e-10),
        "County calibration weights do not sum to one",
    )
    if set(weights["specification"].unique()) != {WAGE_ONLY_SPEC, PRIMARY_SPEC}:
        raise ValueError("Unexpected county calibration specification identifiers")

    diagnostics = read_required(
        INTERMEDIATE / "panel_iv_fls_county_calibration_diagnostics.parquet"
    )
    centers = diagnostics.filter(pl.col("weight_kind") == "deterministic_center")
    if centers.height != 17 * len(SUPPORTED_YEARS) * 2:
        raise ValueError("Deterministic calibration diagnostics are incomplete")
    require_empty(
        centers.filter(~pl.col("optimizer_success")),
        "At least one deterministic county calibration failed",
    )

    moments = read_required(
        INTERMEDIATE / "panel_iv_fls_county_moment_diagnostics.parquet"
    )
    gap_moments = moments.filter(
        (pl.col("source_year") == 2011)
        & (pl.col("moment_status") == "inactive_fls_survey_not_conducted")
    )
    if gap_moments.height != 34:
        raise ValueError("April-2011 inactive moment diagnostics are incomplete")

    donor_frame = read_required(INTERMEDIATE / "panel_iv_county_donor_frame.parquet")
    require_columns(
        donor_frame,
        [
            "donor_nominal_hourly_wage",
            "donor_real_hourly_wage",
            "donor_wage_available",
            "donor_wage_source",
            "donor_wage_spec",
            "oews_big_six_mean_hourly_wage",
            "oews_wage_observed",
            "qcew_strict_complete",
        ],
        "County donor frame",
    )
    require_empty(
        donor_frame.filter(pl.col("policy_year") != pl.col("source_year") + 1),
        "County donor frame violates t-1 source timing",
    )
    require_empty(
        donor_frame.filter(
            (pl.col("donor_wage_spec") != DONOR_WAGE_SPEC)
            | ~pl.col("donor_wage_source").is_in(
                [DONOR_WAGE_SOURCE, "unavailable"]
            )
        ),
        "County donor frame contains an undeclared wage proxy",
    )
    require_empty(
        donor_frame.filter(
            (pl.col("donor_wage_source") == DONOR_WAGE_SOURCE)
            & (
                ~pl.col("oews_wage_observed").fill_null(False)
                | ~pl.col("oews_big_six_mean_hourly_wage").is_finite()
                | (pl.col("oews_big_six_mean_hourly_wage") <= 0)
                | ~pl.col("donor_nominal_hourly_wage").is_finite()
                | ~pl.col("donor_real_hourly_wage").is_finite()
                | ~pl.col("donor_wage_available")
                | (
                    (
                        pl.col("donor_nominal_hourly_wage")
                        - pl.col("oews_big_six_mean_hourly_wage")
                    ).abs()
                    > 1e-12
                )
                | (
                    (
                        pl.col("donor_real_hourly_wage")
                        - pl.col("donor_nominal_hourly_wage")
                        / pl.col("source_year_ppi_2012")
                    ).abs()
                    > 1e-12
                )
            )
        ),
        "Selected OEWS hourly donor-wage proxies are invalid",
    )
    require_empty(
        donor_frame.filter(
            (pl.col("donor_wage_source") == "unavailable")
            & (
                pl.col("donor_nominal_hourly_wage").is_not_null()
                | pl.col("donor_real_hourly_wage").is_not_null()
                | pl.col("donor_wage_available")
            )
        ),
        "Unavailable donor counties expose an hourly wage",
    )
    retired_wage_fields = {
        "donor_nominal_annual_wage",
        "donor_real_annual_wage",
        "qcew_nominal_annual_wage",
        "oews_nominal_annual_wage",
        "bea_nominal_annual_wage",
    }.intersection(donor_frame.columns)
    if retired_wage_fields:
        raise ValueError(
            "Retired annual donor-wage fields remain: "
            + ", ".join(sorted(retired_wage_fields))
        )

    instruments = read_required(
        INTERMEDIATE / "panel_iv_instrument_cluster_year.parquet"
    )
    require_unique(
        instruments,
        ["aewr_region_id", "target_cluster", "source_year", "instrument_spec_label"],
        "Cluster-year instruments",
    )
    if instruments.height != 17 * 5 * len(SUPPORTED_YEARS) * 2:
        raise ValueError("Cluster-year instrument grid is incomplete")
    if PRIMARY_LABEL not in set(instruments["instrument_spec_label"]):
        raise ValueError("Preferred OEWS-hourly instrument label is missing")
    require_columns(
        instruments,
        [
            "donor_wage_spec",
            "qcew_employment_coverage_weight",
            "oews_wage_proxy_coverage_weight",
        ],
        "Cluster-year instruments",
    )
    require_empty(
        instruments.filter(pl.col("policy_year") != pl.col("source_year") + 1),
        "Cluster-year instrument grid violates t-1 timing",
    )
    require_empty(
        instruments.filter(
            (pl.col("donor_wage_spec") != DONOR_WAGE_SPEC)
            | ~pl.col("qcew_employment_coverage_weight").is_finite()
            | ~pl.col("qcew_employment_coverage_weight").is_between(0, 1)
            | ~pl.col("oews_wage_proxy_coverage_weight").is_finite()
            | ~pl.col("oews_wage_proxy_coverage_weight").is_between(0, 1)
        ),
        "Cluster-year wage-proxy or feature-coverage metadata are invalid",
    )
    retired_share_fields = {
        "qcew_wage_weight_share",
        "oews_fallback_weight_share",
        "bea_fallback_weight_share",
    }.intersection(instruments.columns)
    if retired_share_fields:
        raise ValueError(
            "Retired donor fallback fields remain: "
            + ", ".join(sorted(retired_share_fields))
        )


def validate_analysis_outputs() -> None:
    panel = read_required(PROCESSED / "panel_iv_county_year.parquet")
    require_unique(panel, ["county_fips", "year"], "Panel-IV county-year panel")
    require_columns(
        panel,
        [
            "z_wage_only_real",
            "z_wage_seasonal_composition_real",
            "aewr_iv_cluster_id",
        ],
        "Panel-IV county-year panel",
    )
    if "z_wage_seasonal_real" in panel.columns:
        raise ValueError("The retired preferred-instrument field remains in the panel")
    if panel["aewr_iv_cluster_id"].drop_nulls().n_unique() != 85:
        raise ValueError("Panel-IV panel must retain all 85 declared inference clusters")

    first_stage = read_required(TABLES / "iv_preferred_first_stage_estimates.csv")
    if first_stage["column"].to_list() != [1, 2, 3, 4]:
        raise ValueError("First-stage table does not retain the four-column order")
    if first_stage.row(3, named=True)["instrument"] != (
        "z_wage_seasonal_composition_real"
    ):
        raise ValueError("Column 4 is not the preferred seasonal/composition instrument")
    if first_stage["observations"].n_unique() != 1:
        raise ValueError("First-stage columns do not share a common sample")
    if first_stage["inference_clusters"].n_unique() != 1:
        raise ValueError("First-stage columns do not share an inference cluster count")
    require_empty(
        first_stage.filter(~pl.col("first_stage_f").is_finite()),
        "First-stage diagnostics are incomplete",
    )

    second_stage = read_required(TABLES / "iv_preferred_second_stage_estimates.csv")
    contract = second_stage.group_by("outcome").agg(
        pl.col("column").sort().alias("columns"),
        pl.col("observations").n_unique().alias("sample_counts"),
        pl.col("inference_clusters").n_unique().alias("cluster_count_values"),
    )
    require_empty(
        contract.filter(
            (pl.col("columns") != pl.lit([1, 2, 3, 4]))
            | (pl.col("sample_counts") != 1)
            | (pl.col("cluster_count_values") != 1)
        ),
        "Second-stage tables violate ordering, common-sample, or cluster contracts",
    )

    email_statistics = [
        "AEWR coefficient",
        "Clustered standard error",
        "p-value",
        "First-stage excluded-instrument F",
        "Observations",
        "Counties",
        "Inference clusters",
    ]
    email_csvs = [
        TABLES / "panel_iv_email_results_spec1_wage_only.csv",
        TABLES / "panel_iv_email_results_spec2_seasonal_composition.csv",
        TABLES / "panel_iv_email_results_spec3_wage_only_controls.csv",
        TABLES / "panel_iv_email_results_spec4_preferred.csv",
    ]
    for email_csv in email_csvs:
        table = read_required(email_csv)
        if (
            table.height != len(email_statistics)
            or table.width != second_stage["outcome"].n_unique() + 1
            or table.columns[0] != "statistic"
            or table["statistic"].to_list() != email_statistics
        ):
            raise ValueError(
                f"Email-ready IV table has an invalid orientation: {email_csv}"
            )
        require_empty(
            table.filter(
                pl.any_horizontal(
                    [
                        ~pl.col(column).is_finite()
                        for column in table.columns
                        if column != "statistic"
                    ]
                )
            ),
            f"Email-ready IV table contains incomplete estimates: {email_csv}",
        )
    email_pdf = TABLES / "panel_iv_email_results.pdf"
    if not email_pdf.exists() or email_pdf.stat().st_size == 0:
        raise ValueError("Email-ready Panel-IV PDF is missing or empty")

    retained_diagnostics = [
        ROOT / "outputs" / "figures" / "fig_iv_aewr_region_wage_calibration.png",
        ROOT / "outputs" / "figures" / "fig_iv_national_wage_calibration.png",
        ROOT / "outputs" / "figures" / "fig_iv_qcew_fls_moment_residuals.png",
        ROOT
        / "outputs"
        / "figures"
        / "fig_iv_county_entropy_weight_changes_pp.png",
        ROOT / "outputs" / "figures" / "fig_iv_oews_wage_proxy_coverage.png",
        ROOT / "outputs" / "figures" / "fig_iv_target_donor_support.png",
        TABLES / "iv_fls_county_wage_calibration.csv",
        TABLES / "iv_national_wage_calibration.csv",
        TABLES / "iv_qcew_fls_moment_residuals.csv",
        TABLES / "iv_county_entropy_weight_changes_pp.csv",
        TABLES / "iv_oews_wage_proxy_coverage.csv",
        TABLES / "iv_target_donor_support.csv",
    ]
    missing_diagnostics = [
        str(path) for path in retained_diagnostics if not path.exists() or path.stat().st_size == 0
    ]
    if missing_diagnostics:
        raise ValueError(
            "Retained Panel-IV diagnostics are missing: "
            + ", ".join(missing_diagnostics)
        )

    obsolete_diagnostics = [
        ROOT / "outputs" / "figures" / "fig_iv_fls_oews_cz_scatter.png",
        ROOT
        / "outputs"
        / "figures"
        / "fig_iv_cz_entropy_weight_changes_pp.png",
        TABLES / "iv_fls_oews_cz_scatter.csv",
        TABLES / "iv_cz_entropy_weight_changes_pp.csv",
        ROOT / "outputs" / "figures" / "fig_iv_donor_wage_source_shares.png",
        TABLES / "iv_donor_wage_source_shares.csv",
    ]
    stale_diagnostics = [str(path) for path in obsolete_diagnostics if path.exists()]
    if stale_diagnostics:
        raise ValueError(
            "Obsolete OEWS-area diagnostics remain: " + ", ".join(stale_diagnostics)
        )


def main() -> None:
    validate_qcew_and_shared_panel()
    validate_fls_pairs()
    validate_calibration_and_instruments()
    validate_analysis_outputs()
    print(
        "Validated QCEW/FLS county features, OEWS hourly donor-wage timing, "
        "and retained Panel-IV estimates."
    )


if __name__ == "__main__":
    main()
