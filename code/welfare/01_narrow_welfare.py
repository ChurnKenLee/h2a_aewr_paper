"""Calculate narrow certification-space producer incidence for a +$1 AEWR."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tomllib
from pathlib import Path
from statistics import NormalDist
from typing import Any

import polars as pl

from h2a.paths import INTERMEDIATE, PROCESSED, ROOT, TABLES

BRANCH_DIR = ROOT / "code" / "welfare"
CONTRACT_PATH = BRANCH_DIR / "elasticity_contract.toml"
SOURCE_PANEL = PROCESSED / "county_year_panel.parquet"
DETAIL_PATH = PROCESSED / "welfare_certification_space_county_year.parquet"
ANNUAL_PATH = TABLES / "welfare_certification_space_annual.csv"
SUMMARY_PATH = TABLES / "welfare_certification_space_summary.csv"
TABLE_PATH = TABLES / "table_welfare_certification_space.tex"
MANIFEST_PATH = INTERMEDIATE / "welfare_certification_space_manifest.json"

CONTRACT_VERSION = 1
ELASTICITY_DEFINITION = "dlog_certified_hours_dlog_real_aewr"
APPLICATION_POPULATION = "all_shared_panel_county_years"
COUNTERFACTUAL_DELTA = 1.0
EXPOSURE_TOLERANCE = 1e-8


# docs-ground:start welfare-design-contract
# The v1 calculation applies one certified-hours elasticity to every shared-panel
# county-year in the source estimand's declared target window. It evaluates an
# anticipated +1 change in the estimator-matched real AEWR, holds the binding
# share fixed along that local path, and reports certification-space producer
# incidence. Paid-hours incidence is rho times every reported monetary result.
# docs-ground:end welfare-design-contract


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_frame(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.suffix == ".parquet":
        frame.write_parquet(temporary, compression="zstd")
    elif path.suffix == ".csv":
        frame.write_csv(temporary)
    else:
        raise ValueError(f"Unsupported frame output: {path}")
    temporary.replace(path)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Frozen welfare contract requires a [{label}] section.")
    return value


def require_text(section: dict[str, Any], field: str, label: str) -> str:
    value = section.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{field} must be a nonempty string.")
    return value.strip()


def require_finite(section: dict[str, Any], field: str, label: str) -> float:
    value = section.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label}.{field} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}.{field} must be finite.")
    return result


def require_integer(section: dict[str, Any], field: str, label: str) -> int:
    value = require_finite(section, field, label)
    if value != math.floor(value):
        raise ValueError(f"{label}.{field} must be an integer.")
    return int(value)


def load_contract() -> dict[str, Any]:
    if not CONTRACT_PATH.is_file():
        raise ValueError(f"Missing welfare elasticity contract: {CONTRACT_PATH}")
    contract = tomllib.loads(CONTRACT_PATH.read_text())
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(
            f"Welfare contract_version must equal {CONTRACT_VERSION}."
        )
    if contract.get("status") != "frozen":
        raise ValueError(
            "Welfare elasticity contract is unfrozen. Select the final certified-"
            "hours elasticity, record its provenance, and set status = \"frozen\"."
        )

    elasticity = require_mapping(contract.get("elasticity"), "elasticity")
    source = require_mapping(contract.get("source"), "source")
    application = require_mapping(contract.get("application"), "application")

    definition = require_text(elasticity, "definition", "elasticity")
    if definition != ELASTICITY_DEFINITION:
        raise ValueError(
            "Welfare v1 requires elasticity.definition = "
            f"{ELASTICITY_DEFINITION!r}."
        )
    estimate = require_finite(elasticity, "estimate", "elasticity")
    standard_error = require_finite(
        elasticity, "standard_error", "elasticity"
    )
    if standard_error < 0:
        raise ValueError("elasticity.standard_error cannot be negative.")
    confidence_level = require_finite(
        elasticity, "confidence_level", "elasticity"
    )
    if not 0 < confidence_level < 1:
        raise ValueError("elasticity.confidence_level must lie in (0, 1).")
    inference_method = require_text(
        elasticity, "inference_method", "elasticity"
    )

    source_design = require_text(source, "design", "source")
    source_artifact_text = require_text(source, "artifact", "source")
    source_artifact = Path(source_artifact_text)
    if source_artifact.is_absolute():
        raise ValueError("source.artifact must be a repository-relative path.")
    resolved_source = (ROOT / source_artifact).resolve()
    try:
        resolved_source.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("source.artifact resolves outside the repository.") from error
    if not resolved_source.is_file():
        raise ValueError(f"Welfare source artifact does not exist: {resolved_source}")
    expected_source_hash = require_text(
        source, "artifact_sha256", "source"
    ).lower()
    if len(expected_source_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_source_hash
    ):
        raise ValueError("source.artifact_sha256 must be 64 lowercase hex digits.")
    observed_source_hash = sha256_file(resolved_source)
    if observed_source_hash != expected_source_hash:
        raise ValueError(
            "Welfare source artifact hash does not match the frozen contract."
        )
    row_selector = require_text(source, "row_selector", "source")
    source_target_population = require_text(
        source, "target_population", "source"
    )
    source_target_weighting = require_text(source, "target_weighting", "source")
    target_start_year = require_integer(source, "target_start_year", "source")
    target_end_year = require_integer(source, "target_end_year", "source")
    if target_start_year > target_end_year:
        raise ValueError("source target_start_year cannot exceed target_end_year.")

    population = require_text(application, "population", "application")
    if population != APPLICATION_POPULATION:
        raise ValueError(
            "Welfare v1 requires application.population = "
            f"{APPLICATION_POPULATION!r}."
        )
    real_aewr_column = require_text(
        application, "real_aewr_column", "application"
    )
    real_aewr_unit = require_text(application, "real_aewr_unit", "application")
    counterfactual_delta = require_finite(
        application, "counterfactual_delta", "application"
    )
    if counterfactual_delta != COUNTERFACTUAL_DELTA:
        raise ValueError(
            f"Welfare v1 requires a +{COUNTERFACTUAL_DELTA:g} AEWR counterfactual."
        )

    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "confidence_level": confidence_level,
        "inference_method": inference_method,
        "source_design": source_design,
        "source_artifact": source_artifact_text,
        "source_artifact_sha256": observed_source_hash,
        "row_selector": row_selector,
        "source_target_population": source_target_population,
        "source_target_weighting": source_target_weighting,
        "target_start_year": target_start_year,
        "target_end_year": target_end_year,
        "real_aewr_column": real_aewr_column,
        "real_aewr_unit": real_aewr_unit,
    }


def require_columns(frame: pl.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Shared panel lacks required welfare columns: {missing}")


def load_application_panel(contract: dict[str, Any]) -> pl.DataFrame:
    if not SOURCE_PANEL.is_file():
        raise ValueError(f"Missing shared county-year panel: {SOURCE_PANEL}")
    real_aewr_column = contract["real_aewr_column"]
    columns = [
        "county_fips",
        "year",
        "aewr_region_id",
        real_aewr_column,
        "man_hours_certified_start_year",
        "cert_hours_with_hourly_wage_and_aewr_start_year",
        "cert_hours_at_aewr_start_year",
    ]
    panel = pl.read_parquet(SOURCE_PANEL, columns=columns)
    require_columns(panel, columns)
    if panel.schema["county_fips"] != pl.String:
        raise TypeError("Welfare county_fips must be stored as a string.")
    if panel.schema["aewr_region_id"] != pl.String:
        raise TypeError("Welfare aewr_region_id must be stored as a string.")
    if panel.select(pl.col("year").is_null().any()).item():
        raise ValueError("Shared panel contains null welfare years.")
    year_as_float = pl.col("year").cast(pl.Float64)
    if panel.select((year_as_float != year_as_float.floor()).any()).item():
        raise ValueError("Shared-panel welfare years must be integer-valued.")
    panel = (
        panel.with_columns(pl.col("year").cast(pl.Int64))
        .filter(
            pl.col("year").is_between(
                contract["target_start_year"], contract["target_end_year"]
            )
        )
        .sort("county_fips", "year")
    )
    if panel.is_empty():
        raise ValueError("Welfare target window contains no shared-panel rows.")
    if panel.select(
        pl.any_horizontal(
            pl.col(column).is_null()
            for column in ("county_fips", "year", "aewr_region_id")
        ).any()
    ).item():
        raise ValueError("Welfare application keys or AEWR regions contain nulls.")
    duplicates = panel.group_by("county_fips", "year").len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise ValueError("Welfare application panel is not unique by county-year.")

    numeric_columns = [
        real_aewr_column,
        "man_hours_certified_start_year",
        "cert_hours_with_hourly_wage_and_aewr_start_year",
        "cert_hours_at_aewr_start_year",
    ]
    invalid_numeric = panel.select(
        pl.any_horizontal(
            pl.col(column).is_null() | ~pl.col(column).is_finite()
            for column in numeric_columns
        ).any()
    ).item()
    if invalid_numeric:
        raise ValueError("Welfare numeric inputs must be nonnull and finite.")
    if panel.select((pl.col(real_aewr_column) <= 0).any()).item():
        raise ValueError("Estimator-matched real AEWR must be positive.")
    if panel.select(
        pl.any_horizontal(pl.col(column) < 0 for column in numeric_columns[1:]).any()
    ).item():
        raise ValueError("Certified-hour welfare inputs cannot be negative.")

    total = pl.col("man_hours_certified_start_year")
    comparable = pl.col("cert_hours_with_hourly_wage_and_aewr_start_year")
    classified = pl.col("cert_hours_at_aewr_start_year")
    invalid_nesting = panel.select(
        (
            (comparable > total + EXPOSURE_TOLERANCE * pl.max_horizontal(1, total))
            | (
                classified
                > comparable
                + EXPOSURE_TOLERANCE * pl.max_horizontal(1, comparable)
            )
        ).any()
    ).item()
    if invalid_nesting:
        raise ValueError(
            "Welfare exposure inputs violate classified <= comparable <= total hours."
        )
    return panel


def add_coverage_adjustment(panel: pl.DataFrame) -> pl.DataFrame:
    comparable_name = "cert_hours_with_hourly_wage_and_aewr_start_year"
    classified_name = "cert_hours_at_aewr_start_year"
    region_support = panel.group_by("aewr_region_id", "year").agg(
        pl.col(classified_name).sum().alias("region_classified_hours"),
        pl.col(comparable_name).sum().alias("region_comparable_hours"),
    ).with_columns(
        pl.when(pl.col("region_comparable_hours") > 0)
        .then(pl.col("region_classified_hours") / pl.col("region_comparable_hours"))
        .otherwise(None)
        .alias("region_year_binding_share")
    )
    national_support = panel.group_by("year").agg(
        pl.col(classified_name).sum().alias("national_classified_hours"),
        pl.col(comparable_name).sum().alias("national_comparable_hours"),
    ).with_columns(
        pl.when(pl.col("national_comparable_hours") > 0)
        .then(
            pl.col("national_classified_hours")
            / pl.col("national_comparable_hours")
        )
        .otherwise(None)
        .alias("national_year_binding_share")
    )
    panel = panel.join(
        region_support.select(
            "aewr_region_id", "year", "region_year_binding_share"
        ),
        on=["aewr_region_id", "year"],
        how="left",
        validate="m:1",
    ).join(
        national_support.select("year", "national_year_binding_share"),
        on="year",
        how="left",
        validate="m:1",
    )

    total = pl.col("man_hours_certified_start_year")
    comparable = pl.col(comparable_name)
    classified = pl.col(classified_name)
    panel = panel.with_columns(
        pl.when(total == 0)
        .then(0.0)
        .when(comparable > 0)
        .then(classified / comparable)
        .when(pl.col("region_year_binding_share").is_not_null())
        .then(pl.col("region_year_binding_share"))
        .otherwise(pl.col("national_year_binding_share"))
        .alias("coverage_adjusted_binding_share"),
        pl.when(total == 0)
        .then(pl.lit("zero_certified_hours"))
        .when(comparable > 0)
        .then(pl.lit("county_year_observed"))
        .when(pl.col("region_year_binding_share").is_not_null())
        .then(pl.lit("aewr_region_year_fallback"))
        .otherwise(pl.lit("national_year_fallback"))
        .alias("coverage_adjustment_source"),
    )
    unsupported = panel.filter(
        (pl.col("man_hours_certified_start_year") > 0)
        & pl.col("coverage_adjusted_binding_share").is_null()
    )
    if unsupported.height:
        years = unsupported["year"].unique().sort().to_list()
        raise ValueError(
            "Coverage-adjusted welfare exposure lacks national-year support for "
            f"positive certified hours in years {years}."
        )
    invalid_share = panel.select(
        (
            (pl.col("coverage_adjusted_binding_share") < 0)
            | (pl.col("coverage_adjusted_binding_share") > 1)
            | ~pl.col("coverage_adjusted_binding_share").is_finite()
        ).any()
    ).item()
    if invalid_share:
        raise ValueError("Coverage-adjusted welfare binding shares must lie in [0, 1].")
    return panel


def integral_factor(real_aewr_column: str, elasticity: float) -> pl.Expr:
    baseline = pl.col(real_aewr_column)
    ratio = (baseline + COUNTERFACTUAL_DELTA) / baseline
    if abs(elasticity + 1) <= 1e-10:
        return baseline * ratio.log()
    return baseline * (ratio.pow(elasticity + 1) - 1) / (elasticity + 1)


def calculate_detail(
    panel: pl.DataFrame, contract: dict[str, Any]
) -> tuple[pl.DataFrame, float, float]:
    estimate = contract["estimate"]
    confidence_level = contract["confidence_level"]
    critical_value = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    elasticity_low = estimate - critical_value * contract["standard_error"]
    elasticity_high = estimate + critical_value * contract["standard_error"]
    real_aewr_column = contract["real_aewr_column"]
    ratio = (
        pl.col(real_aewr_column) + COUNTERFACTUAL_DELTA
    ) / pl.col(real_aewr_column)

    observed = panel.with_columns(
        pl.lit("observed_classified_at_aewr").alias("exposure_method"),
        pl.col("cert_hours_at_aewr_start_year").alias(
            "baseline_exposed_certified_hours"
        ),
        pl.when(pl.col("man_hours_certified_start_year") > 0)
        .then(
            pl.col("cert_hours_at_aewr_start_year")
            / pl.col("man_hours_certified_start_year")
        )
        .otherwise(0.0)
        .alias("applied_binding_share"),
        pl.lit("directly_classified_hours").alias("exposure_source"),
    )
    adjusted = panel.with_columns(
        pl.lit("coverage_adjusted").alias("exposure_method"),
        (
            pl.col("man_hours_certified_start_year")
            * pl.col("coverage_adjusted_binding_share")
        ).alias("baseline_exposed_certified_hours"),
        pl.col("coverage_adjusted_binding_share").alias("applied_binding_share"),
        pl.col("coverage_adjustment_source").alias("exposure_source"),
    )
    detail = pl.concat([observed, adjusted], how="vertical")
    detail = detail.with_columns(
        (pl.col(real_aewr_column) + COUNTERFACTUAL_DELTA).alias(
            "counterfactual_real_aewr"
        ),
        (
            pl.col("man_hours_certified_start_year") * ratio.pow(estimate)
        ).alias("counterfactual_certified_hours"),
        (
            pl.col("baseline_exposed_certified_hours") * ratio.pow(estimate)
        ).alias("counterfactual_exposed_certified_hours"),
        pl.lit(estimate).alias("elasticity"),
        pl.lit(contract["standard_error"]).alias("elasticity_standard_error"),
        pl.lit(elasticity_low).alias("elasticity_confidence_low"),
        pl.lit(elasticity_high).alias("elasticity_confidence_high"),
        pl.lit(contract["real_aewr_unit"]).alias("real_aewr_unit"),
        pl.lit(APPLICATION_POPULATION).alias("application_population"),
        pl.lit(True).alias("external_validity_extrapolation"),
    ).with_columns(
        (
            pl.col("counterfactual_certified_hours")
            - pl.col("man_hours_certified_start_year")
        ).alias("change_certified_hours"),
        (
            pl.col("counterfactual_exposed_certified_hours")
            - pl.col("baseline_exposed_certified_hours")
        ).alias("change_exposed_certified_hours"),
        (
            -pl.col("baseline_exposed_certified_hours")
            * integral_factor(real_aewr_column, estimate)
        ).alias("change_producer_surplus_certification_space"),
        (
            -pl.col("baseline_exposed_certified_hours")
            * integral_factor(real_aewr_column, elasticity_low)
        ).alias("change_producer_surplus_at_elasticity_low"),
        (
            -pl.col("baseline_exposed_certified_hours")
            * integral_factor(real_aewr_column, elasticity_high)
        ).alias("change_producer_surplus_at_elasticity_high"),
    ).with_columns(
        pl.min_horizontal(
            "change_producer_surplus_at_elasticity_low",
            "change_producer_surplus_at_elasticity_high",
        ).alias("change_producer_surplus_confidence_low"),
        pl.max_horizontal(
            "change_producer_surplus_at_elasticity_low",
            "change_producer_surplus_at_elasticity_high",
        ).alias("change_producer_surplus_confidence_high"),
    )
    result_columns = [
        "county_fips",
        "year",
        "aewr_region_id",
        "exposure_method",
        "exposure_source",
        real_aewr_column,
        "counterfactual_real_aewr",
        "real_aewr_unit",
        "man_hours_certified_start_year",
        "counterfactual_certified_hours",
        "change_certified_hours",
        "cert_hours_with_hourly_wage_and_aewr_start_year",
        "cert_hours_at_aewr_start_year",
        "applied_binding_share",
        "baseline_exposed_certified_hours",
        "counterfactual_exposed_certified_hours",
        "change_exposed_certified_hours",
        "elasticity",
        "elasticity_standard_error",
        "elasticity_confidence_low",
        "elasticity_confidence_high",
        "change_producer_surplus_certification_space",
        "change_producer_surplus_confidence_low",
        "change_producer_surplus_confidence_high",
        "application_population",
        "external_validity_extrapolation",
    ]
    detail = detail.select(result_columns).sort("exposure_method", "year", "county_fips")
    if detail.select(
        pl.any_horizontal(
            ~pl.col(column).is_finite()
            for column in (
                "counterfactual_certified_hours",
                "counterfactual_exposed_certified_hours",
                "change_producer_surplus_certification_space",
                "change_producer_surplus_confidence_low",
                "change_producer_surplus_confidence_high",
            )
        ).any()
    ).item():
        raise ValueError("Welfare calculation produced nonfinite results.")
    if detail.select(
        (
            pl.col("change_producer_surplus_confidence_low")
            > pl.col("change_producer_surplus_confidence_high")
        ).any()
    ).item():
        raise ValueError("Welfare confidence bounds are not ordered.")
    return detail, elasticity_low, elasticity_high


def aggregate_results(
    detail: pl.DataFrame, contract: dict[str, Any]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    annual = (
        detail.group_by("exposure_method", "year")
        .agg(
            pl.len().alias("county_count"),
            pl.col("man_hours_certified_start_year").sum().alias(
                "baseline_certified_hours"
            ),
            pl.col("cert_hours_with_hourly_wage_and_aewr_start_year")
            .sum()
            .alias("comparable_certified_hours"),
            pl.col("cert_hours_at_aewr_start_year")
            .sum()
            .alias("classified_at_aewr_certified_hours"),
            pl.col("baseline_exposed_certified_hours")
            .sum()
            .alias("baseline_exposed_certified_hours"),
            pl.col("change_certified_hours").sum().alias("change_certified_hours"),
            pl.col("change_exposed_certified_hours")
            .sum()
            .alias("change_exposed_certified_hours"),
            pl.col("change_producer_surplus_certification_space")
            .sum()
            .alias("change_producer_surplus_certification_space"),
            pl.col("change_producer_surplus_confidence_low")
            .sum()
            .alias("change_producer_surplus_confidence_low"),
            pl.col("change_producer_surplus_confidence_high")
            .sum()
            .alias("change_producer_surplus_confidence_high"),
            pl.col("real_aewr_unit").first().alias("real_aewr_unit"),
        )
        .with_columns(
            pl.when(pl.col("baseline_certified_hours") > 0)
            .then(
                pl.col("comparable_certified_hours")
                / pl.col("baseline_certified_hours")
            )
            .otherwise(None)
            .alias("comparable_hours_coverage"),
            pl.when(pl.col("comparable_certified_hours") > 0)
            .then(
                pl.col("classified_at_aewr_certified_hours")
                / pl.col("comparable_certified_hours")
            )
            .otherwise(None)
            .alias("binding_share_among_comparable_hours"),
            pl.lit(contract["estimate"]).alias("elasticity"),
            pl.lit(contract["standard_error"]).alias("elasticity_standard_error"),
            pl.lit(contract["inference_method"]).alias("inference_method"),
            pl.lit(contract["source_design"]).alias("elasticity_source_design"),
            pl.lit(contract["row_selector"]).alias("elasticity_source_row"),
            pl.lit("rho * change_producer_surplus_certification_space").alias(
                "paid_hours_scaling"
            ),
        )
        .sort("exposure_method", "year")
    )
    years = annual["year"].unique().sort().to_list()
    expected_years = list(
        range(contract["target_start_year"], contract["target_end_year"] + 1)
    )
    if years != expected_years:
        raise ValueError(
            f"Welfare annual output years {years} do not equal {expected_years}."
        )
    summary = (
        annual.group_by("exposure_method")
        .agg(
            pl.col("year").min().alias("target_start_year"),
            pl.col("year").max().alias("target_end_year"),
            pl.col("county_count").mean().alias("average_annual_county_count"),
            pl.col("baseline_certified_hours")
            .mean()
            .alias("average_annual_baseline_certified_hours"),
            pl.col("comparable_certified_hours")
            .mean()
            .alias("average_annual_comparable_certified_hours"),
            pl.col("classified_at_aewr_certified_hours")
            .mean()
            .alias("average_annual_classified_at_aewr_certified_hours"),
            pl.col("baseline_exposed_certified_hours")
            .mean()
            .alias("average_annual_baseline_exposed_certified_hours"),
            pl.col("change_certified_hours")
            .mean()
            .alias("average_annual_change_certified_hours"),
            pl.col("change_exposed_certified_hours")
            .mean()
            .alias("average_annual_change_exposed_certified_hours"),
            pl.col("change_producer_surplus_certification_space")
            .mean()
            .alias("average_annual_change_producer_surplus_certification_space"),
            pl.col("change_producer_surplus_confidence_low")
            .mean()
            .alias("average_annual_change_producer_surplus_confidence_low"),
            pl.col("change_producer_surplus_confidence_high")
            .mean()
            .alias("average_annual_change_producer_surplus_confidence_high"),
            pl.col("comparable_hours_coverage")
            .mean()
            .alias("average_comparable_hours_coverage"),
            pl.col("binding_share_among_comparable_hours")
            .mean()
            .alias("average_binding_share_among_comparable_hours"),
            pl.col("real_aewr_unit").first().alias("real_aewr_unit"),
            pl.col("elasticity").first().alias("elasticity"),
            pl.col("elasticity_standard_error")
            .first()
            .alias("elasticity_standard_error"),
            pl.col("inference_method").first().alias("inference_method"),
            pl.col("elasticity_source_design")
            .first()
            .alias("elasticity_source_design"),
            pl.col("elasticity_source_row").first().alias("elasticity_source_row"),
            pl.col("paid_hours_scaling").first().alias("paid_hours_scaling"),
        )
        .with_columns(
            pl.lit("unweighted_mean_of_annual_national_totals").alias(
                "headline_aggregation"
            ),
            pl.lit(APPLICATION_POPULATION).alias("application_population"),
            pl.lit(True).alias("external_validity_extrapolation"),
        )
        .sort("exposure_method")
    )
    return annual, summary


def format_number(value: float) -> str:
    return f"{value:,.0f}"


def build_latex_table(summary: pl.DataFrame, contract: dict[str, Any]) -> str:
    labels = {
        "observed_classified_at_aewr": "Observed classified exposure",
        "coverage_adjusted": "Coverage-adjusted exposure",
    }
    rows = []
    for row in summary.iter_rows(named=True):
        confidence = (
            f"[{format_number(row['average_annual_change_producer_surplus_confidence_low'])}, "
            f"{format_number(row['average_annual_change_producer_surplus_confidence_high'])}]"
        )
        rows.append(
            " & ".join(
                [
                    labels[row["exposure_method"]],
                    format_number(
                        row["average_annual_baseline_exposed_certified_hours"]
                    ),
                    format_number(row["average_annual_change_exposed_certified_hours"]),
                    format_number(
                        row[
                            "average_annual_change_producer_surplus_certification_space"
                        ]
                    ),
                    confidence,
                ]
            )
            + r" \\"
        )
    confidence_percent = 100 * contract["confidence_level"]
    return "\n".join(
        [
            r"\begin{table}[!htbp] \centering",
            r"\caption{Certification-space producer incidence of a real \$1 AEWR increase}",
            r"\label{tab:welfare_certification_space}",
            r"\begin{tabular}{lrrrr}",
            r"\hline\hline",
            (
                "Exposure measure & Baseline exposed hours & $\\Delta$ exposed hours "
                "& $\\Delta$ producer surplus & Interval \\\\"
            ),
            r"\hline",
            *rows,
            r"\hline\hline",
            r"\end{tabular}",
            r"\begin{minipage}{0.96\linewidth}\footnotesize",
            (
                f"Notes: The counterfactual is an anticipated +1 change in "
                f"{contract['real_aewr_unit']}. Values are unweighted means of annual "
                f"national totals over {contract['target_start_year']}--"
                f"{contract['target_end_year']}, applying the elasticity to all shared-"
                "panel counties. The observed result uses hours explicitly classified at "
                "the current or lagged AEWR. The sensitivity extrapolates binding exposure "
                "to uncovered certified hours with AEWR-region-year and then national-year "
                rf"fallbacks. The {confidence_percent:g}\% interval propagates only the "
                f"elasticity uncertainty from {contract['inference_method']}. Reported "
                "producer-surplus changes are in certification space; paid-hours changes "
                "equal $\\rho$ times the reported values. Output and domestic-labor prices "
                "are held fixed."
            ),
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )


def write_manifest(contract: dict[str, Any], outputs: list[Path]) -> None:
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_panel": str(SOURCE_PANEL.relative_to(ROOT)),
        "source_panel_sha256": sha256_file(SOURCE_PANEL),
        "elasticity_source_artifact": contract["source_artifact"],
        "elasticity_source_artifact_sha256": contract["source_artifact_sha256"],
        "elasticity_source_row": contract["row_selector"],
        "calculator_sha256": sha256_file(Path(__file__)),
        "outputs": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in outputs
        },
        "target_start_year": contract["target_start_year"],
        "target_end_year": contract["target_end_year"],
        "application_population": APPLICATION_POPULATION,
        "external_validity_extrapolation": True,
        "counterfactual_delta": COUNTERFACTUAL_DELTA,
        "real_aewr_column": contract["real_aewr_column"],
        "real_aewr_unit": contract["real_aewr_unit"],
        "realization_scaling": "paid-hours result = rho * certification-space result",
        "assumptions": [
            "constant certified-hours elasticity along the local +1 path",
            "fixed binding exposure along the local +1 path",
            "fixed output price and other domestic-labor wage",
            "no corresponding-employment hours",
            "elasticity applied outside its source population to all shared-panel counties",
        ],
    }
    atomic_write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", MANIFEST_PATH)


def main() -> None:
    contract = load_contract()
    panel = add_coverage_adjustment(load_application_panel(contract))
    detail, _, _ = calculate_detail(panel, contract)
    annual, summary = aggregate_results(detail, contract)
    atomic_write_frame(detail, DETAIL_PATH)
    atomic_write_frame(annual, ANNUAL_PATH)
    atomic_write_frame(summary, SUMMARY_PATH)
    atomic_write_text(build_latex_table(summary, contract), TABLE_PATH)
    write_manifest(
        contract,
        [DETAIL_PATH, ANNUAL_PATH, SUMMARY_PATH, TABLE_PATH],
    )
    print(
        "Wrote narrow certification-space welfare results for "
        f"{contract['target_start_year']}--{contract['target_end_year']}."
    )


if __name__ == "__main__":
    try:
        main()
    except (TypeError, ValueError) as error:
        raise SystemExit(f"Welfare calculation failed: {error}") from None
