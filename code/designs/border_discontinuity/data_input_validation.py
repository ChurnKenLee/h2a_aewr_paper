"""Independent artifact accounting checks for the border input upgrade."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import polars as pl

from h2a.aewr import assert_unique, validate_schedule
from h2a.geography import assert_geo_columns
from h2a.paths import CODE, INTERMEDIATE, OUTPUTS, PROCESSED, RAW, ROOT, TABLES


def require_close(left, right, label, *, tolerance=1e-8):
    if not np.allclose(left, right, rtol=tolerance, atol=tolerance, equal_nan=True):
        raise ValueError(f"Accounting mismatch: {label}")


def validate_input_artifacts():
    annual = pl.read_parquet(INTERMEDIATE / "aewr.parquet")
    assert_unique(annual, ["state_fips", "year"])
    assert_geo_columns(annual, ["state_fips"])
    if annual.height != 49 * 36 or set(annual["year"]) != set(range(1990, 2026)):
        raise ValueError("Annual reference history truncated")
    if annual.filter(pl.col("year") == 2009)["aewr"].null_count() != 49:
        raise ValueError("Nonstatutory wage entered canonical 2009 AEWR")
    if annual.filter((pl.col("year") != 2009) & (~pl.col("aewr").is_finite())).height:
        raise ValueError("Missing published reference outside the declared OES gap")
    schedule = pl.read_parquet(INTERMEDIATE / "aewr_rate_schedule.parquet")
    validate_schedule(schedule)
    if schedule.filter((pl.col("effective_start") < pl.date(2010, 3, 15))
                       & (pl.col("effective_end_exclusive") > pl.date(2009, 1, 17))).height:
        raise ValueError("FLS schedule was filled across the OES-regime gap")
    audit = pl.read_parquet(INTERMEDIATE / "aewr_reference_rate_source_audit.parquet")
    if not audit.select("state_fips", "year").sort(["state_fips", "year"]).equals(
        annual.select("state_fips", "year").sort(["state_fips", "year"])):
        raise ValueError("Validation workbook altered annual support")
    source_manifest = json.loads((INTERMEDIATE / "aewr_source_manifest.json").read_text())
    for item in source_manifest["inputs"]:
        path = RAW / "aewr" / item["name"]
        if not path.exists():
            path = CODE / "a01_sources" / item["name"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Stale AEWR source provenance")
    skill = pl.read_parquet(INTERMEDIATE / "aewr_occupation_skill_rates.parquet")
    assert_geo_columns(skill, ["state_fips"])
    assert_unique(skill, ["state_fips", "soc_code", "skill_level", "worker_category", "effective_start"])
    if set(skill["skill_level"]) != {1, 2} or set(skill["worker_category"]) != {"US", "H2A"}:
        raise ValueError("Occupational wage dimensions lost")

    work = pl.scan_parquet(INTERMEDIATE / "h2a_worksite_exposure_audit.parquet")
    expected = work.filter(pl.col("case_allocation_reconciled") & pl.col("exposure_begin").is_not_null()).group_by(
        "fiscal_year").agg((pl.col("adjusted_nbr_workers_certified")
                            * ((pl.col("exposure_end") - pl.col("exposure_begin")).dt.total_days() + 1)).sum()
                            .alias("expected_position_days")).sort("fiscal_year").collect()
    files = sorted((INTERMEDIATE / "h2a_worksite_months").glob("fy_*.parquet"))
    if {int(p.stem.removeprefix("fy_")) for p in files} != set(expected["fiscal_year"]):
        raise ValueError("Missing or stale fiscal-year calendar partition")
    actual = []
    for path in files:
        month = pl.scan_parquet(path)
        if month.group_by("allocation_id", "county_fips", "month").len().filter(pl.col("len") != 1).limit(1).collect().height:
            raise ValueError("Duplicated worksite/county/month")
        if month.filter((pl.col("segment_days") <= 0)
                        | (pl.col("fls_reference_covered_days") > pl.col("segment_days"))
                        | (pl.col("allocated_certification_position_days") < 0)).limit(1).collect().height:
            raise ValueError("Invalid calendar accounting or overlapping reference dates")
        actual.append(month.group_by("fiscal_year").agg(
            pl.col("allocated_certification_position_days").sum().alias("actual_position_days"), pl.len().alias("rows")).collect())
    actual = pl.concat(actual).sort("fiscal_year")
    require_close(actual["actual_position_days"].to_numpy(), expected["expected_position_days"].to_numpy(), "fiscal position-days")
    month_county = pl.scan_parquet(files).filter(pl.col("county_fips").is_not_null()).with_columns(
        pl.col("month").dt.year().alias("year")).group_by("county_fips", "year").agg(
            pl.col("allocated_certification_position_days").sum()).sort(["county_fips", "year"]).collect()
    county = pl.read_parquet(INTERMEDIATE / "h2a_worksite_county_year_exposure.parquet").sort(["county_fips", "year"])
    assert_unique(county, ["county_fips", "year"])
    assert_geo_columns(county, ["county_fips"])
    if not county.select("county_fips", "year").equals(month_county.select("county_fips", "year")):
        raise ValueError("County-year calendar support mismatch")
    require_close(county["allocated_certification_position_days"].to_numpy(),
                  month_county["allocated_certification_position_days"].to_numpy(), "county-year position-days")

    spec = json.loads((CODE / "designs/border_discontinuity/data_input_design.json").read_text())
    design = json.loads((CODE / "designs/border_discontinuity/design.json").read_text())
    panel = pl.read_parquet(PROCESSED / "border_data_input_county_year.parquet")
    assert_unique(panel, ["county_fips", "year"])
    assert_geo_columns(panel, ["county_fips", "aewr_region_id"])
    for name in spec["denominators"]:
        require_close(panel[f"certifications_per_{name}"].to_numpy(),
                      (panel["nbr_workers_certified_start_year"] / panel[name]).to_numpy(), name)
    estimates = pl.read_csv(TABLES / "border_data_input_estimates.csv", null_values="NA")
    expected_ids = {f"{sample}__{denominator}" for sample in spec["sample_comparisons"] for denominator in spec["denominators"]}
    if set(estimates["model_id"]) != expected_ids or estimates.height != len(expected_ids):
        raise ValueError("Denominator comparison registry is incomplete")
    common = estimates.filter(pl.col("sample") == "common_all_denominators")
    if common["n_counties"].n_unique() != 1 or common["n_observations"].n_unique() != 1:
        raise ValueError("Denominator comparisons use different common samples")
    if estimates.filter(pl.col("fwl_coefficient_error").abs() > 1e-8).height:
        raise ValueError("Denominator fits fail FWL equivalence")
    power = pl.read_csv(TABLES / "border_data_input_power.csv", null_values="NA")
    settings = design["redesign"]["power"]
    keys = ["model_id", "dgp_id", "scale_information", "inference_method", "effect"]
    assert_unique(power, keys)
    if set(power["model_id"]) != set(common["model_id"]) or set(power["repetitions"]) != {settings["repetitions"]} or set(power["seed"]) != {settings["seed"]}:
        raise ValueError("Power registry or resource guard changed")
    require_close(power["rejection_rate"].to_numpy(), (power["rejections"] / power["repetitions"]).to_numpy(), "unconditional rejection rates")
    if power.filter(pl.col("valid_repetitions") + pl.col("invalid_repetitions") != pl.col("repetitions")).height:
        raise ValueError("Invalid simulation draws disappeared")
    nulls = power.filter(pl.col("effect") == 0)
    expected_gate = (pl.col("invalid_repetitions") == 0) & (pl.col("mc_confidence_high") <= design["redesign"]["precision_assumptions"]["size_maximum"])
    if nulls.filter(pl.col("size_gate_pass") != expected_gate).height:
        raise ValueError("Size gate bypassed")
    calibration = pl.read_csv(TABLES / "border_data_input_calibration.csv")
    if set(calibration["n_preperiod_years"]) != {3} or set(calibration["calibration_repetitions"]) != {settings["calibration_repetitions"]}:
        raise ValueError("Wage missingness truncated outcome noise history")

    geocodes = pl.read_parquet(INTERMEDIATE / "h2a_worksite_geocodes.parquet")
    assert_unique(geocodes, ["address_id"])
    if geocodes.filter((pl.col("geocode_status") != "Match")
                       & (pl.col("longitude").is_not_null() | pl.col("latitude").is_not_null())).height:
        raise ValueError("Unacquired/ambiguous geocode was replaced with a point")
    for filename in ("border_data_input_provenance.json", "border_data_input_report_manifest.json"):
        manifest = json.loads((OUTPUTS / "reports" / filename).read_text())
        for relative, digest in manifest["files"].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Stale data-input provenance: {relative}")
    result = {"status": "artifact_checks_passed", "annual_reference_rows": annual.height,
              "occupational_lookup_rows": skill.height, "worksite_county_month_rows": actual["rows"].sum(),
              "county_year_exposure_rows": county.height, "denominator_comparisons": estimates.height,
              "common_sample_counties": common["n_counties"][0],
              "county_level_inputs_validated": True,
              "nonpublic_nass_required": False,
              "external_data_complete": False,
              "remaining_external_requirements": ["2009 statutory OES determinations and uncached occupational archives",
                                                   "worksite geocoding and subsequent boundary/outcome-linkage audit"],
              "external_requirement_scope": "historical wage and finer spatial extensions; not required to run the current county comparisons",
              "excluded_data_route": "non-public NASS data/custom tabulations unavailable by user instruction"}
    (OUTPUTS / "reports/border_data_inputs_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
