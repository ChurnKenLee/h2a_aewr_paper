"""Pure imperative envelope-theorem welfare helpers for certification-space bounds."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tomllib
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from h2a.geography import assert_geo_columns
from h2a.paths import INTERMEDIATE, OUTPUTS, PROCESSED, ROOT, TABLES

BRANCH_DIR = ROOT / "code" / "welfare"
REPORTS = OUTPUTS / "reports"
CONTRACT_PATH = BRANCH_DIR / "elasticity_contract.toml"
SOURCE_PANEL = PROCESSED / "county_year_panel.parquet"
SKILL_RATES_PATH = INTERMEDIATE / "aewr_occupation_skill_rates.parquet"
CALCULATOR_PATH = BRANCH_DIR / "01_narrow_welfare.py"
HELPER_PATH = BRANCH_DIR / "envelope.py"

DETAIL_PATH = PROCESSED / "welfare_certification_space_county_year.parquet"
BOUNDS_GRID_PATH = TABLES / "welfare" / "welfare_bounds_grid.csv"
ANNUAL_SERIES_PATH = TABLES / "welfare" / "welfare_annual_series.csv"
TABLE_PATH = TABLES / "table_welfare_bounds.tex"
REPORT_PATH = REPORTS / "welfare_bounds.md"
MANIFEST_PATH = INTERMEDIATE / "welfare_certification_space_manifest.json"

CONTRACT_VERSION = 2
CONTRACT_STATUS = "scenario_bounds_grid"
ELASTICITY_DEFINITION = "dlog_certified_hours_dlog_real_aewr"
APPLICATION_POPULATION = "all_shared_panel_county_years"
COUNTERFACTUAL_DELTA = 1.0
EXPOSURE_TOLERANCE = 1e-8
LOG_LIMIT_TOLERANCE = 1e-10
BEA_THOUSANDS_TO_DOLLARS = 1000.0
PARAMETRIC_EXPOSURE_SHARES = (0.85, 0.90, 0.94, 1.00)
OCT2025_EFFECTIVE_DATE = date(2025, 10, 2)


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
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Welfare contract requires a [{label}] section.")
    return value


def _require_text(section: dict[str, Any], field: str, label: str) -> str:
    value = section.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{field} must be a nonempty string.")
    cleaned = value.strip()
    if "PENDING_PATH2" in cleaned:
        raise ValueError(
            f"{label}.{field} still contains unresolved placeholder: {cleaned!r}."
        )
    return cleaned


def _require_finite(section: dict[str, Any], field: str, label: str) -> float:
    value = section.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label}.{field} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}.{field} must be finite.")
    return result


def _require_integer(section: dict[str, Any], field: str, label: str) -> int:
    value = _require_finite(section, field, label)
    if value != math.floor(value):
        raise ValueError(f"{label}.{field} must be an integer.")
    return int(value)


def _require_float_list(
    section: dict[str, Any], field: str, label: str
) -> tuple[float, ...]:
    value = section.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label}.{field} must be a nonempty list of numbers.")
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise TypeError(f"{label}.{field} elements must be numeric.")
        num = float(item)
        if not math.isfinite(num):
            raise ValueError(f"{label}.{field} elements must be finite.")
        out.append(num)
    return tuple(out)


def _require_pinned_artifact(
    section: dict[str, Any],
    path_field: str,
    hash_field: str,
    label: str,
    root: Path,
) -> tuple[str, Path, str]:
    rel_text = _require_text(section, path_field, label)
    rel_path = Path(rel_text)
    if rel_path.is_absolute():
        raise ValueError(f"{label}.{path_field} must be repository-relative.")
    resolved = (root / rel_path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"{label}.{path_field} resolves outside repository."
        ) from error
    if not resolved.is_file():
        raise ValueError(f"{label}.{path_field} does not exist: {resolved}")
    expected = _require_text(section, hash_field, label).lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"{label}.{hash_field} must be 64 lowercase hex digits.")
    observed = sha256_file(resolved)
    if observed != expected:
        raise ValueError(
            f"{label}.{path_field} SHA-256 mismatch: expected {expected}, "
            f"observed {observed}."
        )
    return rel_text, resolved, observed


def load_welfare_contract(
    contract_path: Path = CONTRACT_PATH,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Load and strictly validate the v2 scenario-bounds welfare contract."""
    if not contract_path.is_file():
        raise ValueError(f"Missing welfare contract: {contract_path}")
    raw = tomllib.loads(contract_path.read_text(encoding="utf-8"))
    if raw.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"contract_version must equal {CONTRACT_VERSION}.")
    if raw.get("status") != CONTRACT_STATUS:
        raise ValueError(
            f"Welfare contract status must equal {CONTRACT_STATUS!r}; "
            f"got {raw.get('status')!r}."
        )

    auth = _require_mapping(raw.get("authorization"), "authorization")
    grid_sec = _require_mapping(raw.get("elasticity_grid"), "elasticity_grid")
    track1 = _require_mapping(raw.get("track1_source"), "track1_source")
    pt_sec = _require_mapping(raw.get("passthrough_source"), "passthrough_source")
    app = _require_mapping(raw.get("application"), "application")
    sec_cf = _require_mapping(
        raw.get("secondary_counterfactuals"), "secondary_counterfactuals"
    )

    authorized_on = _require_text(auth, "authorized_on", "authorization")
    authorized_note = _require_text(auth, "note", "authorization")

    definition = _require_text(grid_sec, "definition", "elasticity_grid")
    if definition != ELASTICITY_DEFINITION:
        raise ValueError(
            f"elasticity_grid.definition must equal {ELASTICITY_DEFINITION!r}."
        )
    canonical_etas = _require_float_list(grid_sec, "canonical_values", "elasticity_grid")
    if 0.0 not in canonical_etas or -1.0 not in canonical_etas:
        raise ValueError("elasticity_grid.canonical_values must include 0.0 and -1.0.")
    exposure_shares = _require_float_list(
        grid_sec, "parametric_exposure_shares", "elasticity_grid"
    )
    if any(not (0.0 < x <= 1.0) for x in exposure_shares):
        raise ValueError("parametric_exposure_shares must lie in (0, 1].")

    t1_design = _require_text(track1, "design", "track1_source")
    t1_art, t1_resolved, t1_hash = _require_pinned_artifact(
        track1, "artifact", "artifact_sha256", "track1_source", root
    )
    t1_row = _require_text(track1, "row_selector", "track1_source")
    t1_method = _require_text(track1, "inference_method", "track1_source")
    t1_point = _require_finite(track1, "point_estimate", "track1_source")
    t1_se = _require_finite(track1, "standard_error", "track1_source")
    t1_low = _require_finite(track1, "ci95_low", "track1_source")
    t1_high = _require_finite(track1, "ci95_high", "track1_source")
    t1_states = _require_integer(track1, "n_states", "track1_source")
    if t1_se < 0 or t1_low > t1_point or t1_point > t1_high or t1_states < 2:
        raise ValueError("track1_source bounds/SE/state count are inconsistent.")

    # Verify policy_2025 quantity_estimates.csv primary row matches declared Track 1 anchor.
    qty_df = pl.read_csv(t1_resolved, null_values=["NA", ""])
    qty_required = {
        "window",
        "contrast",
        "specification",
        "outcome",
        "estimator",
        "treatment",
        "elasticity",
        "elasticity_se_jk",
        "elasticity_low_jk",
        "elasticity_high_jk",
        "states",
    }
    if qty_required <= set(qty_df.columns):
        q_matched = qty_df.filter(
            (pl.col("window") == "full_fy")
            & (pl.col("contrast") == "fy2025_2026")
            & (pl.col("specification") == "B_primary")
            & (pl.col("outcome") == "hours")
            & (pl.col("estimator") == "PPML")
            & (pl.col("treatment") == "x_us_skill_1")
        )
        if q_matched.height != 1:
            raise ValueError(
                "Could not uniquely locate primary full_fy B_primary PPML hours "
                f"row in {t1_art}."
            )
        q_row = q_matched.row(0, named=True)
        for field_name, csv_col, declared in (
            ("point_estimate", "elasticity", t1_point),
            ("standard_error", "elasticity_se_jk", t1_se),
            ("ci95_low", "elasticity_low_jk", t1_low),
            ("ci95_high", "elasticity_high_jk", t1_high),
        ):
            observed_val = float(q_row[csv_col])
            if abs(observed_val - declared) > 1e-6:
                raise ValueError(
                    f"track1_source.{field_name} ({declared}) disagrees with "
                    f"{t1_art} {csv_col} ({observed_val})."
                )
        if int(q_row["states"]) != t1_states:
            raise ValueError(
                f"track1_source.n_states ({t1_states}) disagrees with "
                f"{t1_art} states ({q_row['states']})."
            )

    pt_design = _require_text(pt_sec, "design", "passthrough_source")
    pt_art, pt_resolved, pt_hash = _require_pinned_artifact(
        pt_sec, "artifact", "artifact_sha256", "passthrough_source", root
    )
    pt_row = _require_text(pt_sec, "row_selector", "passthrough_source")
    pt_est = _require_finite(pt_sec, "estimate", "passthrough_source")
    pt_se = _require_finite(pt_sec, "standard_error", "passthrough_source")
    if not (0.0 < pt_est <= 1.0) or pt_se < 0:
        raise ValueError("passthrough_source.estimate must lie in (0, 1].")

    # Verify policy_2025 wage_estimates.csv primary row matches declared pass-through.
    wage_df = pl.read_csv(pt_resolved, null_values=["NA", ""])
    if {"track", "specification", "tau", "tier", "weights", "term", "estimate"} <= set(
        wage_df.columns
    ):
        matched = wage_df.filter(
            (pl.col("track") == "wage")
            & (pl.col("specification") == "occupation_dw")
            & (pl.col("tau") == 2026)
            & (pl.col("tier") == "R7")
            & (pl.col("weights") == "h0")
            & (pl.col("term") == "dose")
        )
        if matched.height != 1:
            raise ValueError(
                "Could not uniquely locate primary R7 h0 occupation_dw tau=2026 row "
                f"in {pt_art}."
            )
        csv_est = float(matched["estimate"].item())
        if abs(csv_est - pt_est) > 1e-6:
            raise ValueError(
                f"passthrough_source.estimate ({pt_est}) disagrees with {pt_art} "
                f"({csv_est})."
            )

    population = _require_text(app, "population", "application")
    if population != APPLICATION_POPULATION:
        raise ValueError(
            f"application.population must equal {APPLICATION_POPULATION!r}."
        )
    panel_art, _, panel_hash = _require_pinned_artifact(
        app, "source_panel", "source_panel_sha256", "application", root
    )
    real_aewr_column = _require_text(app, "real_aewr_column", "application")
    real_aewr_unit = _require_text(app, "real_aewr_unit", "application")
    nominal_aewr_column = _require_text(app, "nominal_aewr_column", "application")
    nominal_aewr_unit = _require_text(app, "nominal_aewr_unit", "application")
    delta = _require_finite(app, "counterfactual_delta", "application")
    if delta != COUNTERFACTUAL_DELTA:
        raise ValueError(
            f"application.counterfactual_delta must equal {COUNTERFACTUAL_DELTA}."
        )
    start_year = _require_integer(app, "start_year", "application")
    complete_end_year = _require_integer(app, "complete_end_year", "application")
    provisional_end_year = _require_integer(app, "provisional_end_year", "application")
    snapshot_year = _require_integer(app, "snapshot_year", "application")
    if not (start_year <= snapshot_year <= complete_end_year <= provisional_end_year):
        raise ValueError("Invalid application year ordering in welfare contract.")

    pct_shifts = _require_float_list(
        sec_cf, "percent_shifts", "secondary_counterfactuals"
    )
    if any(p <= -1.0 or p == 0.0 for p in pct_shifts):
        raise ValueError("secondary_counterfactuals.percent_shifts must be >-1 and !=0.")
    skill_art, _, skill_hash = _require_pinned_artifact(
        sec_cf,
        "skill_rates_artifact",
        "skill_rates_sha256",
        "secondary_counterfactuals",
        root,
    )

    return {
        "contract_version": CONTRACT_VERSION,
        "status": CONTRACT_STATUS,
        "authorized_on": authorized_on,
        "authorized_note": authorized_note,
        "elasticity_definition": definition,
        "canonical_etas": canonical_etas,
        "parametric_exposure_shares": exposure_shares,
        "track1_design": t1_design,
        "track1_artifact": t1_art,
        "track1_artifact_sha256": t1_hash,
        "track1_row_selector": t1_row,
        "track1_inference_method": t1_method,
        "track1_point_estimate": t1_point,
        "track1_standard_error": t1_se,
        "track1_ci95_low": t1_low,
        "track1_ci95_high": t1_high,
        "track1_n_states": t1_states,
        "passthrough_design": pt_design,
        "passthrough_artifact": pt_art,
        "passthrough_artifact_sha256": pt_hash,
        "passthrough_row_selector": pt_row,
        "passthrough_estimate": pt_est,
        "passthrough_standard_error": pt_se,
        "population": population,
        "source_panel": panel_art,
        "source_panel_sha256": panel_hash,
        "real_aewr_column": real_aewr_column,
        "real_aewr_unit": real_aewr_unit,
        "nominal_aewr_column": nominal_aewr_column,
        "nominal_aewr_unit": nominal_aewr_unit,
        "counterfactual_delta": delta,
        "start_year": start_year,
        "complete_end_year": complete_end_year,
        "provisional_end_year": provisional_end_year,
        "snapshot_year": snapshot_year,
        "percent_shifts": pct_shifts,
        "skill_rates_artifact": skill_art,
        "skill_rates_sha256": skill_hash,
    }


def scalar_envelope_metrics(
    baseline_wage: float,
    baseline_exposed_hours: float,
    effective_delta: float,
    elasticity: float,
    functional_form: str = "constant_elasticity",
) -> dict[str, float]:
    """Exact scalar certification-space envelope metrics for one unit."""
    if baseline_wage <= 0.0 or not math.isfinite(baseline_wage):
        raise ValueError("baseline_wage must be finite and positive.")
    if baseline_exposed_hours < 0.0 or not math.isfinite(baseline_exposed_hours):
        raise ValueError("baseline_exposed_hours must be finite and nonnegative.")
    new_wage = baseline_wage + effective_delta
    if new_wage <= 0.0 or not math.isfinite(new_wage):
        raise ValueError("Counterfactual wage must remain positive.")

    ratio = new_wage / baseline_wage
    if functional_form == "constant_elasticity":
        hours_ratio = ratio**elasticity
        if abs(elasticity + 1.0) <= LOG_LIMIT_TOLERANCE:
            factor = baseline_wage * math.log(ratio)
        else:
            factor = (
                baseline_wage
                * (ratio ** (elasticity + 1.0) - 1.0)
                / (elasticity + 1.0)
            )
    elif functional_form == "constant_semi_elasticity":
        arg = elasticity * effective_delta / baseline_wage
        hours_ratio = math.exp(arg)
        if abs(elasticity) <= LOG_LIMIT_TOLERANCE:
            factor = effective_delta
        else:
            factor = baseline_wage * (math.exp(arg) - 1.0) / elasticity
    else:
        raise ValueError(f"Unsupported functional_form: {functional_form!r}")

    first_order = -baseline_exposed_hours * effective_delta
    delta_pi = -baseline_exposed_hours * factor
    second_order_cushion = delta_pi - first_order
    counterfactual_exposed_hours = baseline_exposed_hours * hours_ratio
    change_exposed_hours = counterfactual_exposed_hours - baseline_exposed_hours
    change_wage_bill = (
        new_wage * counterfactual_exposed_hours
        - baseline_wage * baseline_exposed_hours
    )
    pct_cushion = (
        100.0 * second_order_cushion / abs(first_order)
        if abs(first_order) > 0.0
        else 0.0
    )
    return {
        "counterfactual_wage": new_wage,
        "hours_ratio": hours_ratio,
        "counterfactual_exposed_hours": counterfactual_exposed_hours,
        "change_exposed_hours": change_exposed_hours,
        "first_order_surplus_change": first_order,
        "second_order_behavioral_cushion": second_order_cushion,
        "second_order_pct_of_abs_first_order": pct_cushion,
        "change_producer_surplus": delta_pi,
        "change_h2a_wage_bill": change_wage_bill,
    }


def _integral_factor_expr(
    wage_col: str,
    delta_expr: pl.Expr,
    elasticity: float,
    functional_form: str,
) -> pl.Expr:
    a0 = pl.col(wage_col)
    ratio = (a0 + delta_expr) / a0
    if functional_form == "constant_elasticity":
        if abs(elasticity + 1.0) <= LOG_LIMIT_TOLERANCE:
            return a0 * ratio.log()
        return a0 * (ratio.pow(elasticity + 1.0) - 1.0) / (elasticity + 1.0)
    if functional_form == "constant_semi_elasticity":
        if abs(elasticity) <= LOG_LIMIT_TOLERANCE:
            return delta_expr
        return a0 * ((elasticity * delta_expr / a0).exp() - 1.0) / elasticity
    raise ValueError(f"Unsupported functional_form: {functional_form!r}")


def _hours_ratio_expr(
    wage_col: str,
    delta_expr: pl.Expr,
    elasticity: float,
    functional_form: str,
) -> pl.Expr:
    a0 = pl.col(wage_col)
    if functional_form == "constant_elasticity":
        return ((a0 + delta_expr) / a0).pow(elasticity)
    if functional_form == "constant_semi_elasticity":
        return (elasticity * delta_expr / a0).exp()
    raise ValueError(f"Unsupported functional_form: {functional_form!r}")


def load_shared_welfare_panel(
    source_panel: Path,
    start_year: int,
    end_year: int,
    real_aewr_column: str,
    nominal_aewr_column: str,
) -> pl.DataFrame:
    """Load and validate all shared-panel county-years in [start_year, end_year]."""
    if not source_panel.is_file():
        raise ValueError(f"Missing shared county-year panel: {source_panel}")
    columns = [
        "county_fips",
        "state_fips",
        "aewr_region_id",
        "year",
        real_aewr_column,
        nominal_aewr_column,
        "man_hours_certified_start_year",
        "nbr_workers_certified_start_year",
        "cert_hours_with_hourly_wage_and_aewr_start_year",
        "cert_hours_at_aewr_start_year",
        "farm_laborexpense_ppi",
        "farm_cashcrops_ppi",
    ]
    panel = pl.read_parquet(source_panel, columns=columns)
    assert_geo_columns(panel, ["county_fips", "state_fips", "aewr_region_id"])
    if panel["year"].null_count() > 0:
        raise ValueError("Shared panel contains null years.")
    year_float = pl.col("year").cast(pl.Float64)
    if panel.select((year_float != year_float.floor()).any()).item():
        raise ValueError("Shared panel years must be integer-valued.")
    panel = (
        panel.with_columns(pl.col("year").cast(pl.Int64))
        .filter(pl.col("year").is_between(start_year, end_year))
        .sort(["year", "county_fips"])
    )
    if panel.is_empty():
        raise ValueError("Welfare panel slice is empty.")
    dupes = panel.group_by("county_fips", "year").len().filter(pl.col("len") > 1)
    if dupes.height:
        raise ValueError("Welfare panel is not unique by (county_fips, year).")

    expected_years = list(range(start_year, end_year + 1))
    observed_years = panel["year"].unique().sort().to_list()
    if observed_years != expected_years:
        raise ValueError(
            f"Welfare panel years {observed_years} do not equal {expected_years}."
        )
    counts_by_year = panel.group_by("year").len()["len"].unique().to_list()
    if len(counts_by_year) != 1:
        raise ValueError(f"Welfare panel is unbalanced across years: {counts_by_year}")

    strict_numeric = [
        real_aewr_column,
        nominal_aewr_column,
        "man_hours_certified_start_year",
        "nbr_workers_certified_start_year",
        "cert_hours_with_hourly_wage_and_aewr_start_year",
        "cert_hours_at_aewr_start_year",
    ]
    if panel.select(
        pl.any_horizontal(
            pl.col(c).is_null() | ~pl.col(c).is_finite() for c in strict_numeric
        ).any()
    ).item():
        raise ValueError("Welfare core numeric columns contain nulls or nonfinite values.")
    if panel.select(
        (pl.col(real_aewr_column) <= 0) | (pl.col(nominal_aewr_column) <= 0)
    ).to_series().any():
        raise ValueError("Baseline real and nominal AEWR must be strictly positive.")
    if panel.select(
        pl.any_horizontal(pl.col(c) < 0 for c in strict_numeric[2:]).any()
    ).item():
        raise ValueError("Certified H-2A hour/worker inputs cannot be negative.")

    total = pl.col("man_hours_certified_start_year")
    comp = pl.col("cert_hours_with_hourly_wage_and_aewr_start_year")
    at_aewr = pl.col("cert_hours_at_aewr_start_year")
    bad_nest = panel.select(
        (
            (comp > total + EXPOSURE_TOLERANCE * pl.max_horizontal(1.0, total))
            | (at_aewr > comp + EXPOSURE_TOLERANCE * pl.max_horizontal(1.0, comp))
        ).any()
    ).item()
    if bad_nest:
        raise ValueError("Hours violate classified <= comparable <= total.")
    return panel


def add_coverage_adjustment(panel: pl.DataFrame) -> pl.DataFrame:
    """Attach county -> region-year -> national-year coverage-adjusted binding shares."""
    comp_col = "cert_hours_with_hourly_wage_and_aewr_start_year"
    at_col = "cert_hours_at_aewr_start_year"
    region_support = (
        panel.group_by("aewr_region_id", "year")
        .agg(
            pl.col(at_col).sum().alias("_reg_at"),
            pl.col(comp_col).sum().alias("_reg_comp"),
        )
        .with_columns(
            pl.when(pl.col("_reg_comp") > 0)
            .then(pl.col("_reg_at") / pl.col("_reg_comp"))
            .otherwise(None)
            .alias("region_year_binding_share")
        )
    )
    national_support = (
        panel.group_by("year")
        .agg(
            pl.col(at_col).sum().alias("_nat_at"),
            pl.col(comp_col).sum().alias("_nat_comp"),
        )
        .with_columns(
            pl.when(pl.col("_nat_comp") > 0)
            .then(pl.col("_nat_at") / pl.col("_nat_comp"))
            .otherwise(None)
            .alias("national_year_binding_share")
        )
    )
    enriched = (
        panel.join(
            region_support.select("aewr_region_id", "year", "region_year_binding_share"),
            on=["aewr_region_id", "year"],
            how="left",
            validate="m:1",
        )
        .join(
            national_support.select("year", "national_year_binding_share"),
            on="year",
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.when(pl.col("man_hours_certified_start_year") == 0)
            .then(0.0)
            .when(pl.col(comp_col) > 0)
            .then(pl.col(at_col) / pl.col(comp_col))
            .when(pl.col("region_year_binding_share").is_not_null())
            .then(pl.col("region_year_binding_share"))
            .otherwise(pl.col("national_year_binding_share"))
            .alias("coverage_adjusted_binding_share"),
            pl.when(pl.col("man_hours_certified_start_year") == 0)
            .then(pl.lit("zero_certified_hours"))
            .when(pl.col(comp_col) > 0)
            .then(pl.lit("county_year_observed"))
            .when(pl.col("region_year_binding_share").is_not_null())
            .then(pl.lit("aewr_region_year_fallback"))
            .otherwise(pl.lit("national_year_fallback"))
            .alias("coverage_adjustment_source"),
        )
    )
    missing = enriched.filter(
        (pl.col("man_hours_certified_start_year") > 0)
        & pl.col("coverage_adjusted_binding_share").is_null()
    )
    if missing.height:
        raise ValueError("Missing coverage-adjusted binding share for positive hours.")
    if enriched.select(
        (
            (pl.col("coverage_adjusted_binding_share") < 0.0)
            | (pl.col("coverage_adjusted_binding_share") > 1.0 + EXPOSURE_TOLERANCE)
            | ~pl.col("coverage_adjusted_binding_share").is_finite()
        ).any()
    ).item():
        raise ValueError("Coverage-adjusted binding share outside [0, 1].")
    return enriched.with_columns(
        pl.col("coverage_adjusted_binding_share").clip(0.0, 1.0)
    )


def load_oct2025_statutory_panel(
    panel: pl.DataFrame,
    skill_rates_path: Path,
    baseline_year: int = 2024,
    pre_cut_aewr_year: int = 2025,
) -> pl.DataFrame:
    """Attach Oct 2, 2025 U.S. skill-I cut vs 2025 FLS AEWR onto 2024 baseline hours."""
    if not skill_rates_path.is_file():
        raise ValueError(f"Missing skill-rate table: {skill_rates_path}")
    skill = (
        pl.read_parquet(skill_rates_path)
        .filter(
            (pl.col("soc_code") == "FIELD_LIVESTOCK_COMBINED")
            & (pl.col("effective_start") == pl.lit(OCT2025_EFFECTIVE_DATE))
            & (pl.col("worker_category") == "US")
            & (pl.col("skill_level") == 1)
        )
        .select(
            "state_fips",
            pl.col("hourly_aewr").cast(pl.Float64).alias("oct2025_us_skill1_aewr"),
        )
        .unique()
    )
    assert_geo_columns(skill, ["state_fips"])
    if skill.group_by("state_fips").len().filter(pl.col("len") > 1).height:
        raise ValueError("Oct 2, 2025 skill-I rates are not unique by state_fips.")

    pre_fls = (
        panel.filter(pl.col("year") == pre_cut_aewr_year)
        .select("state_fips", pl.col("aewr").alias("fls_2025_nominal_aewr"))
        .unique()
    )
    if pre_fls.group_by("state_fips").len().filter(pl.col("len") > 1).height:
        raise ValueError("2025 FLS AEWR is not unique within state_fips.")

    base = (
        panel.filter(pl.col("year") == baseline_year)
        .join(pre_fls, on="state_fips", how="left", validate="m:1")
        .join(skill, on="state_fips", how="left", validate="m:1")
        .with_columns(
            (pl.col("oct2025_us_skill1_aewr") - pl.col("fls_2025_nominal_aewr")).alias(
                "oct2025_statutory_cut"
            )
        )
    )
    if base.select(
        pl.col("fls_2025_nominal_aewr").is_null().any()
        | pl.col("oct2025_us_skill1_aewr").is_null().any()
        | (pl.col("oct2025_us_skill1_aewr") <= 0).any()
    ).item():
        raise ValueError("Incomplete or nonpositive Oct 2, 2025 statutory cut panel.")
    return base


def _exposure_variants(
    parametric_shares: tuple[float, ...],
) -> list[tuple[str, pl.Expr]]:
    variants: list[tuple[str, pl.Expr]] = [
        ("observed_classified_at_aewr", pl.col("cert_hours_at_aewr_start_year")),
        (
            "coverage_adjusted",
            pl.col("man_hours_certified_start_year")
            * pl.col("coverage_adjusted_binding_share"),
        ),
    ]
    for share in parametric_shares:
        label = f"parametric_x_{round(share * 100):03d}"
        variants.append(
            (label, pl.col("man_hours_certified_start_year") * pl.lit(share))
        )
    return variants


def _elasticity_specs(contract: dict[str, Any]) -> list[tuple[str, float, str]]:
    specs: list[tuple[str, float, str]] = []
    for eta in contract["canonical_etas"]:
        role = (
            "first_order_inelastic_envelope"
            if eta == 0.0
            else "canonical_bounds_grid"
        )
        specs.append((f"canonical_eta_{eta:+.2f}".replace("+", ""), eta, role))
    specs.append(
        (
            "track1_q4_ci95_lower_bound",
            float(contract["track1_ci95_low"]),
            "empirical_lower_bound",
        )
    )
    specs.append(
        (
            "track1_q4_point_diagnostic",
            float(contract["track1_point_estimate"]),
            "diagnostic_point_not_precision_identified",
        )
    )
    return specs


def _summarize_scenario_frame(
    frame: pl.DataFrame,
    *,
    window_label: str,
    counterfactual_label: str,
    numeraire_label: str,
    wage_col: str,
    delta_expr: pl.Expr,
    exposure_method: str,
    exposed_expr: pl.Expr,
    passthrough: float,
    passthrough_label: str,
    elasticity_label: str,
    elasticity_value: float,
    elasticity_role: str,
    functional_form: str,
    includes_provisional_2025: bool,
) -> dict[str, Any]:
    eff_delta = delta_expr * pl.lit(passthrough)
    ratio_expr = _hours_ratio_expr(wage_col, eff_delta, elasticity_value, functional_form)
    factor_expr = _integral_factor_expr(
        wage_col, eff_delta, elasticity_value, functional_form
    )
    a0 = pl.col(wage_col)
    x0 = exposed_expr
    h0 = pl.col("man_hours_certified_start_year")
    n0 = pl.col("nbr_workers_certified_start_year")

    # Compute exact county-year quantities and aggregate to annual national totals first.
    calc = frame.select(
        "year",
        "county_fips",
        a0.alias("_a0"),
        eff_delta.alias("_eff_delta"),
        h0.alias("_h0"),
        n0.alias("_n0"),
        x0.alias("_x0"),
        (h0 * (ratio_expr - 1.0)).alias("_dh"),
        (x0 * (ratio_expr - 1.0)).alias("_dx"),
        (-x0 * eff_delta).alias("_fo"),
        (-x0 * factor_expr).alias("_dpi"),
        (x0 * ((a0 + eff_delta) * ratio_expr - a0)).alias("_deb"),
        pl.col("farm_laborexpense_ppi").alias("_bea_labor_k"),
        pl.col("farm_cashcrops_ppi").alias("_bea_crops_k"),
    )
    if calc.select(
        ((pl.col("_a0") + pl.col("_eff_delta")) <= 0.0).any()
        | ~pl.col("_dpi").is_finite().all()
    ).item():
        raise ValueError(
            "Nonpositive counterfactual wage or nonfinite surplus in "
            f"{window_label}/{counterfactual_label}/{elasticity_label}."
        )

    annual = calc.group_by("year").agg(
        pl.len().alias("county_count"),
        pl.col("_h0").sum().alias("baseline_certified_hours"),
        pl.col("_n0").sum().alias("baseline_certified_positions"),
        pl.col("_x0").sum().alias("baseline_exposed_hours"),
        pl.col("_dh").sum().alias("change_certified_hours"),
        pl.col("_dx").sum().alias("change_exposed_hours"),
        pl.col("_fo").sum().alias("first_order_surplus_change"),
        pl.col("_dpi").sum().alias("change_producer_surplus"),
        pl.col("_deb").sum().alias("change_h2a_wage_bill"),
        (pl.col("_a0") * pl.col("_x0")).sum().alias("_weighted_a0"),
        (pl.col("_eff_delta") * pl.col("_x0")).sum().alias("_weighted_delta"),
        pl.col("_bea_labor_k").sum().alias("_bea_labor_k_sum"),
        pl.col("_bea_crops_k").sum().alias("_bea_crops_k_sum"),
        pl.col("_bea_labor_k").is_not_null().sum().alias("_bea_nonnull"),
    )

    n_years = annual.height
    avg_h0 = float(annual["baseline_certified_hours"].mean())
    avg_n0 = float(annual["baseline_certified_positions"].mean())
    avg_x0 = float(annual["baseline_exposed_hours"].mean())
    avg_dh = float(annual["change_certified_hours"].mean())
    avg_dx = float(annual["change_exposed_hours"].mean())
    avg_fo = float(annual["first_order_surplus_change"].mean())
    avg_dpi = float(annual["change_producer_surplus"].mean())
    avg_deb = float(annual["change_h2a_wage_bill"].mean())
    second_order = avg_dpi - avg_fo
    second_order_pct = (
        100.0 * second_order / abs(avg_fo) if abs(avg_fo) > 0.0 else 0.0
    )
    total_x0 = float(annual["baseline_exposed_hours"].sum())
    mean_a0 = (
        float(annual["_weighted_a0"].sum()) / total_x0 if total_x0 > 0.0 else math.nan
    )
    mean_eff_delta = (
        float(annual["_weighted_delta"].sum()) / total_x0
        if total_x0 > 0.0
        else math.nan
    )

    # BEA scale ratios are computed strictly on complete BEA years (2011-2022) in real PPI dollars.
    bea_years = annual.filter(pl.col("_bea_nonnull") > 0)
    bea_labor_pct: float | None = None
    bea_crops_pct: float | None = None
    if numeraire_label == "real_ppi_2012" and bea_years.height > 0:
        bea_dpi = float(bea_years["change_producer_surplus"].mean())
        bea_labor_usd = float(bea_years["_bea_labor_k_sum"].mean()) * BEA_THOUSANDS_TO_DOLLARS
        bea_crops_usd = float(bea_years["_bea_crops_k_sum"].mean()) * BEA_THOUSANDS_TO_DOLLARS
        if bea_labor_usd > 0.0:
            bea_labor_pct = 100.0 * abs(bea_dpi) / bea_labor_usd
        if bea_crops_usd > 0.0:
            bea_crops_pct = 100.0 * abs(bea_dpi) / bea_crops_usd

    return {
        "window": window_label,
        "year_start": int(annual["year"].min()),
        "year_end": int(annual["year"].max()),
        "n_years": int(n_years),
        "includes_provisional_2025": bool(includes_provisional_2025),
        "county_count_per_year": round(float(annual["county_count"].mean())),
        "counterfactual": counterfactual_label,
        "numeraire": numeraire_label,
        "exposure_method": exposure_method,
        "passthrough_scenario": passthrough_label,
        "passthrough_rate": float(passthrough),
        "elasticity_label": elasticity_label,
        "elasticity": float(elasticity_value),
        "elasticity_role": elasticity_role,
        "functional_form": functional_form,
        "hours_weighted_baseline_aewr": mean_a0,
        "hours_weighted_effective_delta": mean_eff_delta,
        "avg_annual_baseline_certified_hours": avg_h0,
        "avg_annual_baseline_certified_positions": avg_n0,
        "avg_annual_baseline_exposed_hours": avg_x0,
        "implied_exposed_share": (avg_x0 / avg_h0) if avg_h0 > 0.0 else 0.0,
        "avg_annual_change_certified_hours": avg_dh,
        "avg_annual_change_exposed_hours": avg_dx,
        "avg_annual_first_order_surplus_change": avg_fo,
        "avg_annual_second_order_behavioral_cushion": second_order,
        "second_order_pct_of_abs_first_order": second_order_pct,
        "avg_annual_change_producer_surplus_cert_space": avg_dpi,
        "avg_annual_change_h2a_wage_bill_cert_space": avg_deb,
        "surplus_change_per_certified_position": (
            avg_dpi / avg_n0 if avg_n0 > 0.0 else math.nan
        ),
        "surplus_change_per_certified_hour": (
            avg_dpi / avg_h0 if avg_h0 > 0.0 else math.nan
        ),
        "abs_surplus_pct_bea_hired_labor_2011_2022": bea_labor_pct,
        "abs_surplus_pct_bea_crop_receipts_2011_2022": bea_crops_pct,
        "paid_hours_scaling": "rho * certification_space_value",
        "application_population": APPLICATION_POPULATION,
    }


def evaluate_bounds_grid(
    panel: pl.DataFrame,
    oct2025_panel: pl.DataFrame,
    contract: dict[str, Any],
) -> pl.DataFrame:
    """Evaluate the complete certification-space envelope bounds grid."""
    real_col = contract["real_aewr_column"]
    nom_col = contract["nominal_aewr_column"]
    delta_one = pl.lit(float(contract["counterfactual_delta"]))
    exposures = _exposure_variants(contract["parametric_exposure_shares"])
    etas = _elasticity_specs(contract)
    passthroughs = (
        ("statutory_full_pi_100", 1.0),
        ("policy_2025_offered_wage_pi", float(contract["passthrough_estimate"])),
    )

    p_2011_2024 = panel.filter(
        pl.col("year").is_between(contract["start_year"], contract["complete_end_year"])
    )
    p_2011_2025 = panel.filter(
        pl.col("year").is_between(
            contract["start_year"], contract["provisional_end_year"]
        )
    )
    p_2024 = panel.filter(pl.col("year") == contract["snapshot_year"])

    rows: list[dict[str, Any]] = []

    # 1. Primary +$1 counterfactual across core windows/numeraires, exposures,
    #    pass-throughs, elasticities, and functional forms.
    primary_specs = (
        ("2011_2024_complete", p_2011_2024, "real_ppi_2012", real_col, False),
        ("2011_2025_provisional", p_2011_2025, "real_ppi_2012", real_col, True),
        ("2024_snapshot_real", p_2024, "real_ppi_2012", real_col, False),
        ("2024_snapshot_nominal", p_2024, "nominal_2024_usd", nom_col, False),
    )
    for win_lbl, win_df, num_lbl, w_col, prov in primary_specs:
        for exp_lbl, exp_expr in exposures:
            for pt_lbl, pt_val in passthroughs:
                for eta_lbl, eta_val, eta_role in etas:
                    forms = (
                        ("constant_elasticity", "constant_semi_elasticity")
                        if pt_val == 1.0
                        and exp_lbl
                        in ("coverage_adjusted", "parametric_x_094", "parametric_x_100")
                        else ("constant_elasticity",)
                    )
                    for form in forms:
                        rows.append(
                            _summarize_scenario_frame(
                                win_df,
                                window_label=win_lbl,
                                counterfactual_label="plus_1_dollar",
                                numeraire_label=num_lbl,
                                wage_col=w_col,
                                delta_expr=delta_one,
                                exposure_method=exp_lbl,
                                exposed_expr=exp_expr,
                                passthrough=pt_val,
                                passthrough_label=pt_lbl,
                                elasticity_label=eta_lbl,
                                elasticity_value=eta_val,
                                elasticity_role=eta_role,
                                functional_form=form,
                                includes_provisional_2025=prov,
                            )
                        )

    # 2. Secondary uniform +-10% real AEWR curvature stress test (2011-2024 & 2024 real).
    key_exposures = [
        item
        for item in exposures
        if item[0] in ("coverage_adjusted", "parametric_x_090", "parametric_x_094", "parametric_x_100")
    ]
    for pct in contract["percent_shifts"]:
        cf_lbl = f"uniform_{round(pct * 100):+d}pct".replace("+", "plus_").replace(
            "-", "minus_"
        )
        for win_lbl, win_df in (
            ("2011_2024_complete", p_2011_2024),
            ("2024_snapshot_real", p_2024),
        ):
            for exp_lbl, exp_expr in key_exposures:
                for eta_lbl, eta_val, eta_role in etas:
                    for form in ("constant_elasticity", "constant_semi_elasticity"):
                        rows.append(
                            _summarize_scenario_frame(
                                win_df,
                                window_label=win_lbl,
                                counterfactual_label=cf_lbl,
                                numeraire_label="real_ppi_2012",
                                wage_col=real_col,
                                delta_expr=pl.col(real_col) * pl.lit(pct),
                                exposure_method=exp_lbl,
                                exposed_expr=exp_expr,
                                passthrough=1.0,
                                passthrough_label="statutory_full_pi_100",
                                elasticity_label=eta_lbl,
                                elasticity_value=eta_val,
                                elasticity_role=eta_role,
                                functional_form=form,
                                includes_provisional_2025=False,
                            )
                        )

    # 3. Secondary Oct 2, 2025 actual state-level statutory U.S. skill-I cut
    #    evaluated statically on 2024 certified hours (nominal 2025/2026 USD).
    for exp_lbl, exp_expr in exposures:
        for pt_lbl, pt_val in passthroughs:
            for eta_lbl, eta_val, eta_role in etas:
                rows.append(
                    _summarize_scenario_frame(
                        oct2025_panel,
                        window_label="2024_hours_oct2025_cut",
                        counterfactual_label="oct2025_statutory_skill1_cut",
                        numeraire_label="nominal_2025_2026_usd",
                        wage_col="fls_2025_nominal_aewr",
                        delta_expr=pl.col("oct2025_statutory_cut"),
                        exposure_method=exp_lbl,
                        exposed_expr=exp_expr,
                        passthrough=pt_val,
                        passthrough_label=pt_lbl,
                        elasticity_label=eta_lbl,
                        elasticity_value=eta_val,
                        elasticity_role=eta_role,
                        functional_form="constant_elasticity",
                        includes_provisional_2025=False,
                    )
                )

    return pl.DataFrame(rows)


def evaluate_annual_series(
    panel: pl.DataFrame,
    contract: dict[str, Any],
) -> pl.DataFrame:
    """Compute annual national totals for +$1 real AEWR across 2011-2025."""
    real_col = contract["real_aewr_column"]
    delta_one = pl.lit(float(contract["counterfactual_delta"]))
    exposures = [
        item
        for item in _exposure_variants(contract["parametric_exposure_shares"])
        if item[0]
        in (
            "observed_classified_at_aewr",
            "coverage_adjusted",
            "parametric_x_090",
            "parametric_x_094",
            "parametric_x_100",
        )
    ]
    key_etas = [
        ("canonical_eta_0.00", 0.0),
        ("canonical_eta_-0.30", -0.3),
        ("canonical_eta_-1.00", -1.0),
        ("track1_q4_ci95_lower_bound", float(contract["track1_ci95_low"])),
    ]
    frames: list[pl.DataFrame] = []
    for exp_lbl, exp_expr in exposures:
        for eta_lbl, eta_val in key_etas:
            ratio_expr = _hours_ratio_expr(
                real_col, delta_one, eta_val, "constant_elasticity"
            )
            factor_expr = _integral_factor_expr(
                real_col, delta_one, eta_val, "constant_elasticity"
            )
            a0 = pl.col(real_col)
            annual = (
                panel.group_by("year")
                .agg(
                    pl.len().alias("county_count"),
                    pl.col("man_hours_certified_start_year")
                    .sum()
                    .alias("baseline_certified_hours"),
                    pl.col("nbr_workers_certified_start_year")
                    .sum()
                    .alias("baseline_certified_positions"),
                    exp_expr.sum().alias("baseline_exposed_hours"),
                    (exp_expr * (ratio_expr - 1.0))
                    .sum()
                    .alias("change_exposed_hours"),
                    (-exp_expr * delta_one)
                    .sum()
                    .alias("first_order_surplus_change"),
                    (-exp_expr * factor_expr)
                    .sum()
                    .alias("change_producer_surplus_cert_space"),
                    (exp_expr * ((a0 + delta_one) * ratio_expr - a0))
                    .sum()
                    .alias("change_h2a_wage_bill_cert_space"),
                )
                .with_columns(
                    pl.lit(exp_lbl).alias("exposure_method"),
                    pl.lit(eta_lbl).alias("elasticity_label"),
                    pl.lit(float(eta_val)).alias("elasticity"),
                    (
                        pl.col("change_producer_surplus_cert_space")
                        - pl.col("first_order_surplus_change")
                    ).alias("second_order_behavioral_cushion"),
                    (pl.col("year") > contract["complete_end_year"]).alias(
                        "provisional_year"
                    ),
                )
                .with_columns(
                    pl.when(pl.col("first_order_surplus_change").abs() > 0)
                    .then(
                        100.0
                        * pl.col("second_order_behavioral_cushion")
                        / pl.col("first_order_surplus_change").abs()
                    )
                    .otherwise(0.0)
                    .alias("second_order_pct_of_abs_first_order")
                )
            )
            frames.append(annual)
    return pl.concat(frames, how="vertical").sort(
        ["exposure_method", "elasticity_label", "year"]
    )


def build_county_year_detail(
    panel: pl.DataFrame,
    contract: dict[str, Any],
) -> pl.DataFrame:
    """Write compact county-year certification-space bounds (2011-2025, +$1 real)."""
    real_col = contract["real_aewr_column"]
    delta_one = pl.lit(float(contract["counterfactual_delta"]))
    eta_low = float(contract["track1_ci95_low"])
    exp_cov = (
        pl.col("man_hours_certified_start_year")
        * pl.col("coverage_adjusted_binding_share")
    )
    f_m03 = _integral_factor_expr(real_col, delta_one, -0.3, "constant_elasticity")
    f_m10 = _integral_factor_expr(real_col, delta_one, -1.0, "constant_elasticity")
    f_t1 = _integral_factor_expr(real_col, delta_one, eta_low, "constant_elasticity")
    return (
        panel.select(
            "county_fips",
            "state_fips",
            "aewr_region_id",
            "year",
            real_col,
            contract["nominal_aewr_column"],
            "man_hours_certified_start_year",
            "nbr_workers_certified_start_year",
            "cert_hours_with_hourly_wage_and_aewr_start_year",
            "cert_hours_at_aewr_start_year",
            "coverage_adjusted_binding_share",
            "coverage_adjustment_source",
            exp_cov.alias("coverage_adjusted_exposed_hours"),
            (-exp_cov * delta_one).alias("delta_pi_cov_eta_000"),
            (-exp_cov * f_m03).alias("delta_pi_cov_eta_m030"),
            (-exp_cov * f_m10).alias("delta_pi_cov_eta_m100"),
            (-exp_cov * f_t1).alias("delta_pi_cov_eta_track1_low"),
            pl.lit(eta_low).alias("track1_q4_ci95_lower_bound"),
            (pl.col("year") > contract["complete_end_year"]).alias("provisional_year"),
            pl.lit(APPLICATION_POPULATION).alias("application_population"),
        )
        .sort(["year", "county_fips"])
    )


def _fmt_millions(value: float) -> str:
    return f"{value / 1e6:,.1f}"


def _fmt_pct(value: float) -> str:
    return f"{value:+.2f}\\%"


def build_welfare_bounds_latex(
    grid: pl.DataFrame,
    contract: dict[str, Any],
) -> str:
    """Build concise booktabs manuscript table for certification-space envelope bounds."""

    def _pick(
        window: str,
        cf: str,
        exp: str,
        pt: str,
        eta_lbl: str,
    ) -> dict[str, Any]:
        sub = grid.filter(
            (pl.col("window") == window)
            & (pl.col("counterfactual") == cf)
            & (pl.col("exposure_method") == exp)
            & (pl.col("passthrough_scenario") == pt)
            & (pl.col("elasticity_label") == eta_lbl)
            & (pl.col("functional_form") == "constant_elasticity")
        )
        if sub.height != 1:
            raise ValueError(
                f"Expected 1 row for {(window, cf, exp, pt, eta_lbl)}; got {sub.height}"
            )
        return sub.row(0, named=True)

    eta_order = [
        ("canonical_eta_0.00", r"$\eta = 0.00$ (First-order envelope)"),
        ("canonical_eta_-0.30", r"$\eta = -0.30$ (Moderate elasticity)"),
        ("canonical_eta_-0.75", r"$\eta = -0.75$ (High elasticity)"),
        ("canonical_eta_-1.00", r"$\eta = -1.00$ (Unit elastic benchmark)"),
        (
            "track1_q4_ci95_lower_bound",
            f"$\\eta = {contract['track1_ci95_low']:.2f}$ (Track 1 Q4 95\\% CI low)",
        ),
    ]

    panel_a_rows: list[str] = []
    for eta_lbl, row_title in eta_order:
        r_mean = _pick(
            "2011_2024_complete",
            "plus_1_dollar",
            "coverage_adjusted",
            "statutory_full_pi_100",
            eta_lbl,
        )
        r_2024r = _pick(
            "2024_snapshot_real",
            "plus_1_dollar",
            "coverage_adjusted",
            "statutory_full_pi_100",
            eta_lbl,
        )
        r_2024n = _pick(
            "2024_snapshot_nominal",
            "plus_1_dollar",
            "coverage_adjusted",
            "statutory_full_pi_100",
            eta_lbl,
        )
        panel_a_rows.append(
            " & ".join(
                [
                    row_title,
                    _fmt_millions(r_mean["avg_annual_change_producer_surplus_cert_space"]),
                    _fmt_pct(r_mean["second_order_pct_of_abs_first_order"]),
                    _fmt_millions(r_2024r["avg_annual_change_producer_surplus_cert_space"]),
                    _fmt_millions(r_2024n["avg_annual_change_producer_surplus_cert_space"]),
                    _fmt_millions(r_2024n["avg_annual_change_h2a_wage_bill_cert_space"]),
                ]
            )
            + r" \\"
        )

    panel_b_specs = [
        (
            "Coverage-adjusted exposure (baseline, $\\pi = 1.00$)",
            "2024_snapshot_nominal",
            "plus_1_dollar",
            "coverage_adjusted",
            "statutory_full_pi_100",
        ),
        (
            "Parametric exposure $x = 0.85$ (F1 lower polar, $\\pi = 1.00$)",
            "2024_snapshot_nominal",
            "plus_1_dollar",
            "parametric_x_085",
            "statutory_full_pi_100",
        ),
        (
            "Parametric exposure $x = 0.94$ (F1 upper central, $\\pi = 1.00$)",
            "2024_snapshot_nominal",
            "plus_1_dollar",
            "parametric_x_094",
            "statutory_full_pi_100",
        ),
        (
            "Parametric exposure $x = 1.00$ (full hours, $\\pi = 1.00$)",
            "2024_snapshot_nominal",
            "plus_1_dollar",
            "parametric_x_100",
            "statutory_full_pi_100",
        ),
        (
            f"Offered-wage pass-through sensitivity ($\\pi = {contract['passthrough_estimate']:.3f}$)",
            "2024_snapshot_nominal",
            "plus_1_dollar",
            "coverage_adjusted",
            "policy_2025_offered_wage_pi",
        ),
        (
            "Oct.\\ 2, 2025 statutory U.S.\\ skill-I cut (2024 hrs, $\\pi = 1.00$)",
            "2024_hours_oct2025_cut",
            "oct2025_statutory_skill1_cut",
            "coverage_adjusted",
            "statutory_full_pi_100",
        ),
    ]
    panel_b_rows: list[str] = []
    for title, win, cf, exp, pt in panel_b_specs:
        r0 = _pick(win, cf, exp, pt, "canonical_eta_0.00")
        r1 = _pick(win, cf, exp, pt, "canonical_eta_-1.00")
        rt = _pick(win, cf, exp, pt, "track1_q4_ci95_lower_bound")
        panel_b_rows.append(
            " & ".join(
                [
                    title,
                    _fmt_millions(r0["avg_annual_change_producer_surplus_cert_space"]),
                    _fmt_millions(r1["avg_annual_change_producer_surplus_cert_space"]),
                    _fmt_millions(rt["avg_annual_change_producer_surplus_cert_space"]),
                    _fmt_pct(rt["second_order_pct_of_abs_first_order"]),
                    _fmt_millions(r0["avg_annual_baseline_exposed_hours"]),
                ]
            )
            + r" \\"
        )

    return "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\caption{Certification-Space Producer Incidence Bounds Under the Envelope Theorem}",
            r"\label{tab:welfare_bounds}",
            r"\small",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"\multicolumn{6}{l}{\textit{Panel A: Anticipated $+\$1$/hr AEWR Increase (Coverage-Adjusted Exposure, $\pi = 1.00$, \$M)}} \\",
            r"Elasticity scenario & 2011--24 Avg & 2nd-Order & 2024 Real & 2024 Nom. & 2024 Nom. \\",
            r"(Bounds grid, not point est.) & $\Delta\Pi^{PE}$ & Cushion & $\Delta\Pi^{PE}$ & $\Delta\Pi^{PE}$ & $\Delta E^c_H$ \\",
            r"\midrule",
            *panel_a_rows,
            r"\midrule",
            r"\multicolumn{6}{l}{\textit{Panel B: Exposure, Pass-Through, and Oct.\ 2, 2025 Cut Bounds (2024 Nominal Baseline, \$M)}} \\",
            r"Exposure / counterfactual & $\eta=0.00$ & $\eta=-1.00$ & Track 1 Low & 2nd-Order & Exposed Hrs \\",
            r" & $\Delta\Pi^{PE}$ & $\Delta\Pi^{PE}$ & $\Delta\Pi^{PE}$ & Cushion & (Millions) \\",
            r"\midrule",
            *panel_b_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.96\linewidth}",
            r"\vspace{0.4em}\footnotesize",
            (
                r"\textit{Notes:} Static partial-equilibrium envelope bounds "
                r"$\Delta\Pi^{PE}=-\int X(s)\,ds$ applied to all 3,108 contiguous U.S.\ "
                r"shared-panel counties under a constant-elasticity certified-hours curve "
                r"$H^c(a)=H^c_0(a/a_0)^\eta$. The elasticity grid is anchored to the "
                r"complete FY2026 Q4 Track 1 statutory PPML hours 95\% jackknife lower "
                f"bound ($\\eta = {contract['track1_ci95_low']:.2f}$, $G={contract['track1_n_states']}$ states) "
                r"and is explicitly a bounds grid rather than a single point estimate. "
                r"Real values use 2012 farm-products-PPI-deflated AEWR (\texttt{aewr\_ppi}); "
                r"2024 nominal values use current-dollar \texttt{aewr}. 2nd-order cushion is "
                r"$100\times(\Delta\Pi^{PE}-\Delta\Pi^{PE}(0))/|\Delta\Pi^{PE}(0)| \ge 0$ "
                r"(loss mitigation for wage increases; gain amplification for cuts). "
                r"All monetary values are in certification space; paid-hours incidence "
                r"equals $\rho$ times every reported value. Output prices, domestic "
                r"wages, and corresponding-employment hours are held fixed."
            ),
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )


def build_welfare_bounds_report(
    grid: pl.DataFrame,
    annual: pl.DataFrame,
    contract: dict[str, Any],
) -> str:
    """Generate concise self-contained markdown report summarizing Path 1 bounds."""

    def _row(
        window: str,
        cf: str,
        exp: str,
        pt: str,
        eta_lbl: str,
        form: str = "constant_elasticity",
    ) -> dict[str, Any]:
        return (
            grid.filter(
                (pl.col("window") == window)
                & (pl.col("counterfactual") == cf)
                & (pl.col("exposure_method") == exp)
                & (pl.col("passthrough_scenario") == pt)
                & (pl.col("elasticity_label") == eta_lbl)
                & (pl.col("functional_form") == form)
            )
            .row(0, named=True)
        )

    r_mean_0 = _row("2011_2024_complete", "plus_1_dollar", "coverage_adjusted", "statutory_full_pi_100", "canonical_eta_0.00")
    r_mean_1 = _row("2011_2024_complete", "plus_1_dollar", "coverage_adjusted", "statutory_full_pi_100", "canonical_eta_-1.00")
    r_mean_t1 = _row("2011_2024_complete", "plus_1_dollar", "coverage_adjusted", "statutory_full_pi_100", "track1_q4_ci95_lower_bound")
    r_mean_semi = _row("2011_2024_complete", "plus_1_dollar", "coverage_adjusted", "statutory_full_pi_100", "canonical_eta_-1.00", "constant_semi_elasticity")
    r_24n_0 = _row("2024_snapshot_nominal", "plus_1_dollar", "coverage_adjusted", "statutory_full_pi_100", "canonical_eta_0.00")
    r_24n_1 = _row("2024_snapshot_nominal", "plus_1_dollar", "coverage_adjusted", "statutory_full_pi_100", "canonical_eta_-1.00")
    r_24n_t1 = _row("2024_snapshot_nominal", "plus_1_dollar", "coverage_adjusted", "statutory_full_pi_100", "track1_q4_ci95_lower_bound")
    r_24n_pi = _row("2024_snapshot_nominal", "plus_1_dollar", "coverage_adjusted", "policy_2025_offered_wage_pi", "canonical_eta_-1.00")
    r_cut_0 = _row("2024_hours_oct2025_cut", "oct2025_statutory_skill1_cut", "coverage_adjusted", "statutory_full_pi_100", "canonical_eta_0.00")
    r_cut_t1 = _row("2024_hours_oct2025_cut", "oct2025_statutory_skill1_cut", "coverage_adjusted", "statutory_full_pi_100", "track1_q4_ci95_lower_bound")
    r_p10_0 = _row("2024_snapshot_real", "uniform_plus_10pct", "coverage_adjusted", "statutory_full_pi_100", "canonical_eta_0.00")
    r_p10_1 = _row("2024_snapshot_real", "uniform_plus_10pct", "coverage_adjusted", "statutory_full_pi_100", "canonical_eta_-1.00")

    semi_diff_pct = (
        100.0
        * abs(
            r_mean_semi["avg_annual_change_producer_surplus_cert_space"]
            - r_mean_1["avg_annual_change_producer_surplus_cert_space"]
        )
        / abs(r_mean_1["avg_annual_change_producer_surplus_cert_space"])
    )

    lines = [
        "# Certification-Space Producer Incidence Bounds (Path 1 Envelope Grid)",
        "",
        f"- **Contract**: `code/welfare/elasticity_contract.toml` (`contract_version = {contract['contract_version']}`, `status = \"{contract['status']}\"`, authorized `{contract['authorized_on']}`).",
        f"- **Shared panel**: `{contract['source_panel']}` (SHA-256 `{contract['source_panel_sha256']}`).",
        f"- **Track 1 elasticity anchor**: `{contract['track1_artifact']}` (SHA-256 `{contract['track1_artifact_sha256']}`; `{contract['track1_row_selector']}`): point `{contract['track1_point_estimate']:.4f}` (diagnostic only), CV3J SE `{contract['track1_standard_error']:.4f}`, 95% CI `[{contract['track1_ci95_low']:.4f}, {contract['track1_ci95_high']:.4f}]` across `{contract['track1_n_states']}` states.",
        f"- **Offered-wage pass-through anchor**: `{contract['passthrough_artifact']}` (SHA-256 `{contract['passthrough_artifact_sha256']}`; `{contract['passthrough_row_selector']}`): `pi = {contract['passthrough_estimate']:.4f}` (SE `{contract['passthrough_standard_error']:.4f}`).",
        "- **Paid-hours scaling**: Every monetary figure is in **certification space**. If realized paid hours equal a constant fraction `rho` of certified hours, paid-hours producer incidence equals `rho` times the reported amount.",
        "",
        "## Headline Envelope Bounds",
        "",
        "1. **2011–2024 average annual `+$1` real (`aewr_ppi`) increase (coverage-adjusted, `pi = 1.00`)**:",
        f"   - First-order transfer (`eta = 0.00`): `${r_mean_0['avg_annual_change_producer_surplus_cert_space'] / 1e6:,.2f}M`/yr (`{r_mean_0['avg_annual_baseline_exposed_hours'] / 1e6:,.2f}M` exposed hrs).",
        f"   - Unit-elastic benchmark (`eta = -1.00`): `${r_mean_1['avg_annual_change_producer_surplus_cert_space'] / 1e6:,.2f}M`/yr (2nd-order behavioral cushion `{r_mean_1['second_order_pct_of_abs_first_order']:+.2f}%`; constant-semi-elasticity differs by only `{semi_diff_pct:.3f}%`).",
        f"   - Complete FY2026 Q4 Track 1 lower bound (`eta = {contract['track1_ci95_low']:.2f}`): `${r_mean_t1['avg_annual_change_producer_surplus_cert_space'] / 1e6:,.2f}M`/yr (2nd-order cushion `{r_mean_t1['second_order_pct_of_abs_first_order']:+.2f}%`).",
        f"   - BEA `CAINC45` scale (`2011–2022` real): `{r_mean_0['abs_surplus_pct_bea_hired_labor_2011_2022']:.2f}%` of U.S. farm hired-labor expense (`{r_mean_0['abs_surplus_pct_bea_crop_receipts_2011_2022']:.3f}%` of crop cash receipts) at `eta = 0`.",
        "2. **2024 latest complete year `+$1` nominal (`aewr`) increase (coverage-adjusted, `pi = 1.00`)**:",
        f"   - `eta = 0.00`: `${r_24n_0['avg_annual_change_producer_surplus_cert_space'] / 1e6:,.2f}M` (`{r_24n_0['avg_annual_baseline_exposed_hours'] / 1e6:,.2f}M` exposed hrs, hours-weighted baseline AEWR `${r_24n_0['hours_weighted_baseline_aewr']:.2f}`/hr).",
        f"   - `eta = -1.00`: `${r_24n_1['avg_annual_change_producer_surplus_cert_space'] / 1e6:,.2f}M` (cushion `{r_24n_1['second_order_pct_of_abs_first_order']:+.2f}%`, H-2A wage-bill change `${r_24n_1['avg_annual_change_h2a_wage_bill_cert_space'] / 1e6:,.2f}M`).",
        f"   - Track 1 Q4 lower bound (`eta = {contract['track1_ci95_low']:.2f}`): `${r_24n_t1['avg_annual_change_producer_surplus_cert_space'] / 1e6:,.2f}M` (cushion `{r_24n_t1['second_order_pct_of_abs_first_order']:+.2f}%`).",
        f"   - Offered-wage pass-through sensitivity (`pi = {contract['passthrough_estimate']:.3f}`, `eta = -1.00`): `${r_24n_pi['avg_annual_change_producer_surplus_cert_space'] / 1e6:,.2f}M`.",
        "3. **Secondary finite policy-scale comparisons**:",
        f"   - Uniform `+10%` real AEWR on 2024 hours: `${r_p10_0['avg_annual_change_producer_surplus_cert_space'] / 1e6:,.2f}M` (`eta = 0`) vs `${r_p10_1['avg_annual_change_producer_surplus_cert_space'] / 1e6:,.2f}M` (`eta = -1.00`, cushion `{r_p10_1['second_order_pct_of_abs_first_order']:+.2f}%`).",
        f"   - Actual Oct 2, 2025 state-level U.S. skill-I cut vs 2025 FLS AEWR (mean weighted cut `${r_cut_0['hours_weighted_effective_delta']:.2f}`/hr, static 2024 hours, nominal USD, `pi = 1.00`): `${r_cut_0['avg_annual_change_producer_surplus_cert_space'] / 1e6:+,.2f}M` (`eta = 0`) to `${r_cut_t1['avg_annual_change_producer_surplus_cert_space'] / 1e6:+,.2f}M` (Track 1 lower bound, cushion `{r_cut_t1['second_order_pct_of_abs_first_order']:+.2f}%`).",
        "",
        "## Annual National Totals (`2011–2025`, `+$1` Real AEWR, Coverage-Adjusted Exposure)",
        "",
        "| Year | Provisional | Baseline Hours (M) | Exposed Hours (M) | `eta=0.00` ΔΠ ($M) | `eta=-1.00` ΔΠ ($M) | Track 1 Low ΔΠ ($M) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    cov_annual = annual.filter(pl.col("exposure_method") == "coverage_adjusted")
    years = sorted(cov_annual["year"].unique().to_list())
    for yr in years:
        ydf = cov_annual.filter(pl.col("year") == yr)
        d0 = ydf.filter(pl.col("elasticity_label") == "canonical_eta_0.00").row(0, named=True)
        d1 = ydf.filter(pl.col("elasticity_label") == "canonical_eta_-1.00").row(0, named=True)
        dt = ydf.filter(pl.col("elasticity_label") == "track1_q4_ci95_lower_bound").row(0, named=True)
        prov_txt = "Yes" if d0["provisional_year"] else "No"
        lines.append(
            f"| {yr} | {prov_txt} | {d0['baseline_certified_hours'] / 1e6:,.2f} | "
            f"{d0['baseline_exposed_hours'] / 1e6:,.2f} | "
            f"{d0['change_producer_surplus_cert_space'] / 1e6:,.2f} | "
            f"{d1['change_producer_surplus_cert_space'] / 1e6:,.2f} | "
            f"{dt['change_producer_surplus_cert_space'] / 1e6:,.2f} |"
        )

    lines.extend(
        [
            "",
            "## Maintained Partial-Equilibrium Exclusions",
            "",
            "- Holds farm-gate/retail output prices and non-H-2A domestic wages fixed (`dΠ/da = -X(a)`).",
            "- Excludes corresponding-employment domestic hours, H-2A foreign worker outside-option surplus, administrative fee costs, and long-run crop/mechanization adjustments.",
            "- Treats the `WPU01` farm-products PPI (`aewr_ppi`) as an agricultural output-price numeraire rather than a general CPI/PCE household deflator, paired with nominal `2024` current-dollar snapshots.",
            "",
        ]
    )
    return "\n".join(lines)


def write_welfare_manifest(
    contract: dict[str, Any],
    outputs: list[Path],
    manifest_path: Path = MANIFEST_PATH,
    contract_path: Path = CONTRACT_PATH,
    calculator_path: Path = CALCULATOR_PATH,
    helper_path: Path = HELPER_PATH,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Write complete SHA-256 provenance manifest for welfare bounds artifacts."""
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "status": CONTRACT_STATUS,
        "contract_path": str(contract_path.relative_to(root)),
        "contract_sha256": sha256_file(contract_path),
        "calculator_path": str(calculator_path.relative_to(root)),
        "calculator_sha256": sha256_file(calculator_path),
        "helper_path": str(helper_path.relative_to(root)),
        "helper_sha256": sha256_file(helper_path),
        "source_panel": contract["source_panel"],
        "source_panel_sha256": contract["source_panel_sha256"],
        "track1_artifact": contract["track1_artifact"],
        "track1_artifact_sha256": contract["track1_artifact_sha256"],
        "track1_row_selector": contract["track1_row_selector"],
        "track1_ci95_low": contract["track1_ci95_low"],
        "track1_point_estimate": contract["track1_point_estimate"],
        "passthrough_artifact": contract["passthrough_artifact"],
        "passthrough_artifact_sha256": contract["passthrough_artifact_sha256"],
        "passthrough_estimate": contract["passthrough_estimate"],
        "skill_rates_artifact": contract["skill_rates_artifact"],
        "skill_rates_sha256": contract["skill_rates_sha256"],
        "application_population": APPLICATION_POPULATION,
        "counterfactual_delta": COUNTERFACTUAL_DELTA,
        "realization_scaling": "paid-hours result = rho * certification-space result",
        "outputs": {
            str(path.relative_to(root)): sha256_file(path) for path in outputs
        },
    }
    atomic_write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", manifest_path
    )
    return manifest


def validate_welfare_outputs(
    manifest_path: Path = MANIFEST_PATH,
    contract_path: Path = CONTRACT_PATH,
    root: Path = ROOT,
) -> dict[str, int]:
    """Independently verify SHA-256 parity, schema keys, and analytic envelope identities."""
    checks = 0
    contract = load_welfare_contract(contract_path, root)
    checks += 1
    if not manifest_path.is_file():
        raise ValueError(f"Missing welfare manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. Exact SHA-256 parity for inputs, code, contract, and outputs.
    hash_pairs = [
        (contract_path, manifest["contract_sha256"]),
        (root / manifest["calculator_path"], manifest["calculator_sha256"]),
        (root / manifest["helper_path"], manifest["helper_sha256"]),
        (root / manifest["source_panel"], manifest["source_panel_sha256"]),
        (root / manifest["track1_artifact"], manifest["track1_artifact_sha256"]),
        (root / manifest["passthrough_artifact"], manifest["passthrough_artifact_sha256"]),
        (root / manifest["skill_rates_artifact"], manifest["skill_rates_sha256"]),
    ]
    for path, expected in hash_pairs:
        if sha256_file(path) != expected:
            raise ValueError(f"Manifest input/code SHA-256 mismatch: {path}")
        checks += 1

    for rel_out, expected in manifest["outputs"].items():
        out_path = root / rel_out
        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise ValueError(f"Missing or empty welfare output: {out_path}")
        if sha256_file(out_path) != expected:
            raise ValueError(f"Manifest output SHA-256 mismatch: {out_path}")
        checks += 1

    # 2. Verify county-year detail Parquet schema, keys, and balance.
    detail = pl.read_parquet(DETAIL_PATH)
    assert_geo_columns(detail, ["county_fips", "state_fips", "aewr_region_id"])
    expected_years = list(
        range(contract["start_year"], contract["provisional_end_year"] + 1)
    )
    if detail["year"].unique().sort().to_list() != expected_years:
        raise ValueError("Detail Parquet years mismatch declared 2011-2025 range.")
    if detail.height != 3108 * len(expected_years):
        raise ValueError(
            f"Detail Parquet row count {detail.height} != {3108 * len(expected_years)}."
        )
    if detail.group_by("county_fips", "year").len().filter(pl.col("len") > 1).height:
        raise ValueError("Detail Parquet has duplicate (county_fips, year) keys.")
    checks += 4

    # 3. Verify analytic envelope identities directly on welfare_bounds_grid.csv.
    grid = pl.read_csv(BOUNDS_GRID_PATH)
    if grid.is_empty():
        raise ValueError("welfare_bounds_grid.csv is empty.")

    # (a) eta = 0 exact transfer identity
    eta0 = grid.filter(pl.col("elasticity") == 0.0)
    if eta0.is_empty():
        raise ValueError("Grid missing eta = 0.0 rows.")
    max_eta0_err = float(
        eta0.select(
            (
                pl.col("avg_annual_change_producer_surplus_cert_space")
                - pl.col("avg_annual_first_order_surplus_change")
            )
            .abs()
            .max()
        ).item()
    )
    if max_eta0_err > 1e-4:
        raise ValueError(f"eta=0 transfer identity violated: max_err={max_eta0_err}")
    checks += 1

    # (b) eta = -1 exact log-limit continuity & zero H-2A wage-bill change under CE
    m_lo = scalar_envelope_metrics(15.0, 1e6, 1.0, -1.0 - 1e-6, "constant_elasticity")
    m_at = scalar_envelope_metrics(15.0, 1e6, 1.0, -1.0, "constant_elasticity")
    m_hi = scalar_envelope_metrics(15.0, 1e6, 1.0, -1.0 + 1e-6, "constant_elasticity")
    if (
        abs(m_at["change_producer_surplus"] - m_lo["change_producer_surplus"]) > 1.0
        or abs(m_at["change_producer_surplus"] - m_hi["change_producer_surplus"]) > 1.0
    ):
        raise ValueError("eta = -1 log-limit continuity check failed.")
    eta_m1 = grid.filter(
        (pl.col("elasticity") == -1.0)
        & (pl.col("functional_form") == "constant_elasticity")
    )
    max_wb_err = float(
        eta_m1.select(pl.col("avg_annual_change_h2a_wage_bill_cert_space").abs().max()).item()
    )
    if max_wb_err > 1e-2:
        raise ValueError(f"Delta E_H(-1) == 0 identity violated: {max_wb_err}")
    checks += 2

    # (c) Nonnegative behavioral cushion for all eta <= 0 (convexity of profit)
    neg_eta = grid.filter(pl.col("elasticity") <= 0.0)
    min_cushion = float(
        neg_eta["avg_annual_second_order_behavioral_cushion"].min()
    )
    if min_cushion < -1e-4:
        raise ValueError(f"Negative behavioral cushion for eta <= 0: {min_cushion}")
    checks += 1

    # (d) Monotonicity in eta for primary +$1 rows (less negative surplus loss as eta drops)
    p1 = (
        grid.filter(
            (pl.col("window") == "2011_2024_complete")
            & (pl.col("counterfactual") == "plus_1_dollar")
            & (pl.col("exposure_method") == "coverage_adjusted")
            & (pl.col("passthrough_scenario") == "statutory_full_pi_100")
            & (pl.col("functional_form") == "constant_elasticity")
        )
        .sort("elasticity")
    )
    deltas = p1["avg_annual_change_producer_surplus_cert_space"].to_list()
    etas_sorted = p1["elasticity"].to_list()
    for i in range(len(deltas) - 1):
        if etas_sorted[i] < etas_sorted[i + 1] and deltas[i] < deltas[i + 1] - 1e-4:
            raise ValueError("Producer surplus failed monotonicity in elasticity.")
    checks += 1

    # (e) Constant-elasticity vs semi-elasticity closeness (< 0.5% for +$1, eta in [-1, 0])
    ce_row = p1.filter(pl.col("elasticity_label") == "canonical_eta_-1.00").row(
        0, named=True
    )
    se_row = (
        grid.filter(
            (pl.col("window") == "2011_2024_complete")
            & (pl.col("counterfactual") == "plus_1_dollar")
            & (pl.col("exposure_method") == "coverage_adjusted")
            & (pl.col("passthrough_scenario") == "statutory_full_pi_100")
            & (pl.col("elasticity_label") == "canonical_eta_-1.00")
            & (pl.col("functional_form") == "constant_semi_elasticity")
        )
        .row(0, named=True)
    )
    rel_diff = abs(
        ce_row["avg_annual_change_producer_surplus_cert_space"]
        - se_row["avg_annual_change_producer_surplus_cert_space"]
    ) / abs(ce_row["avg_annual_change_producer_surplus_cert_space"])
    if rel_diff >= 0.005:
        raise ValueError(f"CE vs semi-elasticity diverged unexpectedly: {rel_diff:.4%}")
    checks += 1

    # (f) Verify annual series, LaTeX table, and markdown report content.
    annual = pl.read_csv(ANNUAL_SERIES_PATH)
    if annual["year"].unique().sort().to_list() != expected_years:
        raise ValueError("welfare_annual_series.csv years incomplete.")
    tex = TABLE_PATH.read_text(encoding="utf-8")
    md = REPORT_PATH.read_text(encoding="utf-8")
    if r"\rho" not in tex or "bounds grid" not in tex.lower():
        raise ValueError("LaTeX welfare table missing rho or bounds-grid disclaimer.")
    if "rho" not in md or "certification space" not in md.lower():
        raise ValueError("Markdown welfare report missing rho or cert-space disclaimer.")
    checks += 3

    return {"checks_passed": checks, "grid_rows": grid.height, "annual_rows": annual.height}
