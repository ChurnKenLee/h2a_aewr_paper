"""Imperative data and estimation helpers for the historical-reform investigation."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from h2a.geography import assert_geo_columns, geo_expr
from h2a.paths import CODE, INTERMEDIATE, OUTPUTS, PROCESSED, RAW, ROOT

BRANCH = CODE / "designs" / "wage_reforms"
TABLES = OUTPUTS / "tables" / "wage_reforms"
REPORTS = OUTPUTS / "reports"
CASE_KEY = ["fiscal_year", "case_number"]
MEMBERSHIP_KEYS = {
    "case_number": "case_number",
    "fiscal_year": "fiscal_year",
    "employer_name": "source_name_raw",
    "trade_name_dba": "source_trade_name_raw",
    "employer_address_1": "source_address_1_raw",
    "employer_address_2": "source_address_2_raw",
    "employer_city": "source_city_raw",
    "employer_state": "source_state_raw",
    "employer_postal_code": "source_postal_code_raw",
    "employer_phone": "source_phone_raw",
    "employer_fein": "source_fein_raw",
    "primary_sub": "source_primary_sub_raw",
    "organization_flag": "source_organization_flag_raw",
    "type_of_employer_application": "source_type_of_employer_application_raw",
    "ag_association_or_agency": "source_ag_association_or_agency_raw",
    "h2a_labor_contractor": "source_h2a_labor_contractor_raw",
}
LINKAGE_METADATA = [
    "employer_linkage_unit", "employer_linkage_version",
    "employer_linkage_input_max_year", "employer_linkage_review_status",
    "employer_linkage_review_report_hash",
]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def configuration() -> dict:
    return json.loads((BRANCH / "design.json").read_text())


def unique_key(frame: pl.DataFrame, key: list[str]) -> None:
    if not frame.height or any(frame[k].null_count() for k in key):
        raise ValueError(f"Empty artifact or missing key: {key}")
    if frame.select(key).is_duplicated().any():
        raise ValueError(f"Duplicate artifact key: {key}")
    for k in key:
        if frame.schema[k] == pl.String and (frame[k] == "").any():
            raise ValueError(f"Empty string key: {k}")


def prepare_rows(config: dict) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    source = pl.read_parquet(INTERMEDIATE / "h2a_with_fips.parquet").filter(
        pl.col("fiscal_year").is_between(*config["source_years"])
    ).unique(maintain_order=True)
    if source.height > config["resource_guard"]["maximum_source_rows"]:
        raise ValueError("Source-row resource guard exceeded")
    membership = pl.read_parquet(INTERMEDIATE / "h2a_employer_record_membership.parquet")
    metadata = membership.select(LINKAGE_METADATA).unique()
    if metadata.height != 1:
        raise ValueError("Multiple employer-linkage contracts")
    contract = metadata.row(0, named=True)
    for suffix in ["unit", "version", "input_max_year", "review_status"]:
        if contract[f"employer_linkage_{suffix}"] != config["linkage"][suffix]:
            raise ValueError(f"Unreviewed or incompatible linkage {suffix}")
    if len(contract["employer_linkage_review_report_hash"] or "") != 64:
        raise ValueError("Missing reviewed linkage provenance")
    member_columns = [
        "employer_id_balanced", "employer_record_role", "employer_is_worksite_business",
        "employer_id_conservative", "employer_id_high_recall",
    ]
    join_map = membership.filter(pl.col("source_dataset") == "h2a_with_fips").select(
        [pl.col(v).alias(k) for k, v in MEMBERSHIP_KEYS.items()] + member_columns
    )
    rows = source.with_columns(
        [pl.col(k).fill_null("") for k in MEMBERSHIP_KEYS if k != "fiscal_year"]
    ).join(join_map, on=list(MEMBERSHIP_KEYS), how="left", validate="m:1")
    if rows["employer_record_role"].null_count():
        raise ValueError("Employer membership does not cover source rows")
    if rows.filter(pl.col("employer_is_worksite_business"))["employer_id_balanced"].null_count():
        raise ValueError("Missing business identity")
    regions = pl.read_csv(RAW / "aewr" / "aewr_regions.csv", infer_schema=False).with_columns(
        geo_expr("aewr_region_num", "aewr_region_id")
    ).select(pl.col("state_abbrev").alias("worksite_state"), "aewr_region_id")
    unique_key(regions, ["worksite_state"])
    assert_geo_columns(regions, ["aewr_region_id"])
    valid_states = regions["worksite_state"].to_list()
    rows = rows.with_columns(
        certified=pl.col("nbr_workers_certified").cast(pl.Float64, strict=False),
        requested=pl.col("nbr_workers_requested").cast(pl.Float64, strict=False),
        offered_wage=pl.col("wage_rate").cast(pl.Float64, strict=False),
        hourly=pl.col("wage_unit").str.to_uppercase().is_in(["HOUR", "HR"]).fill_null(False),
        monthly=pl.col("wage_unit").str.to_uppercase().is_in(["MONTH", "MTH"]).fill_null(False),
        herder_title=pl.col("job_title").str.contains(config["herder_regex"]).fill_null(False),
        sheep_title=pl.col("job_title").str.contains(config["sheep_regex"]).fill_null(False),
        title_normalized=pl.col("job_title").fill_null("").str.to_uppercase()
        .str.replace_all(r"\s+", " ").str.strip_chars(),
        receipt_date=pl.col("case_received_date").str.to_date("%Y-%m-%d", strict=False),
        start_date=pl.coalesce([
            pl.col(c).str.to_date("%Y-%m-%d", strict=False)
            for c in ["certification_begin_date", "job_begin_date", "requested_begin_date"]
        ]),
        end_date=pl.coalesce([
            pl.col(c).str.to_date("%Y-%m-%d", strict=False)
            for c in ["certification_end_date", "job_end_date", "requested_end_date"]
        ]),
        valid_state=pl.col("worksite_state").is_in(valid_states).fill_null(False),
        is_master=pl.when(pl.col("fiscal_year") < 2016)
        .then(pl.col("organization_flag").fill_null("") != "")
        .otherwise(pl.col("primary_sub").fill_null("") == "PRI"),
    ).with_columns(
        strict_herder=pl.col("monthly") & pl.col("herder_title"),
        receipt_valid=pl.col("receipt_date").dt.year().is_between(
            pl.col("fiscal_year") - 2, pl.col("fiscal_year")
        ).fill_null(False),
        contract_valid=(pl.col("start_date").dt.year().is_between(
            pl.col("fiscal_year") - 1, pl.col("fiscal_year") + 1
        ) & pl.col("end_date").dt.year().is_between(
            pl.col("fiscal_year") - 1, pl.col("fiscal_year") + 1
        ) & (pl.col("end_date") >= pl.col("start_date"))).fill_null(False),
    ).join(regions, on="worksite_state", how="left", validate="m:1")
    positive = rows.filter(pl.col("certified") > 0).with_columns(
        positive_case_rows=pl.len().over(CASE_KEY)
    )
    cases = select_positive_cases(positive)
    case_states = positive.group_by(CASE_KEY).agg(
        pl.col("worksite_state").filter(pl.col("valid_state")).unique().sort()
        .alias("case_states"),
        (~pl.col("valid_state")).any().alias("case_state_missing"),
    )
    positive = positive.join(case_states, on=CASE_KEY, validate="m:1")
    cases = cases.join(case_states, on=CASE_KEY, validate="1:1")
    applications = rows.group_by(CASE_KEY).agg(
        pl.col("receipt_date").filter(pl.col("receipt_valid")).drop_nulls().unique().alias("receipt_candidates"),
        (pl.col("certified") > 0).any().alias("any_certified"),
        (~pl.col("receipt_valid")).any().alias("any_invalid_receipt_row"),
    ).with_columns(
        receipt_date=pl.when(pl.col("receipt_candidates").list.len() == 1)
        .then(pl.col("receipt_candidates").list.first()).otherwise(None),
    ).with_columns(receipt_valid=pl.col("receipt_date").is_not_null())
    unique_key(applications, CASE_KEY)
    return positive, cases, applications, {
        "linkage": contract,
        "unique_source_rows": source.height,
        "positive_source_rows": positive.height,
        "all_source_case_years": rows.select(CASE_KEY).unique().height,
        "positive_case_years": cases.height,
        "selected_positive_positions": float(cases["certified"].sum()),
        "nonpositive_case_years": rows.select(CASE_KEY).unique().height - cases.height,
        "application_case_years_without_unique_valid_receipt": applications.filter(~pl.col("receipt_valid")).height,
        "case_numbers_in_multiple_fiscal_vintages": cases.group_by("case_number").len().filter(pl.col("len") > 1).height,
    }


def select_positive_cases(positive: pl.DataFrame) -> pl.DataFrame:
    """Same singleton/master selection as the upstream national total audit."""
    cases = positive.filter((pl.col("positive_case_rows") == 1) | pl.col("is_master"))
    unique_key(cases, CASE_KEY)
    if cases.height != positive.select(CASE_KEY).unique().height:
        raise ValueError("Singleton/master selection leaves positive cases unresolved")
    return cases


def coverage_tables(positive: pl.DataFrame, cases: pl.DataFrame, config: dict) -> dict:
    coverage = cases.group_by("fiscal_year").agg(
        pl.len().alias("positive_cases"),
        pl.col("certified").sum().alias("all_certified_positions"),
        pl.col("certified").filter(pl.col("strict_herder")).sum().alias("herder_positions"),
        pl.col("strict_herder").sum().alias("herder_cases"),
        pl.col("certified").filter(pl.col("monthly")).sum().alias("all_monthly_positions"),
        pl.col("offered_wage").filter(pl.col("strict_herder")).median().alias("herder_median_monthly_wage"),
        pl.col("offered_wage").filter(pl.col("hourly")).median().alias("hourly_median_wage"),
        (~pl.col("receipt_valid")).sum().alias("invalid_receipt_dates"),
        (~pl.col("contract_valid")).sum().alias("invalid_contract_dates"),
        (pl.col("soc_code").fill_null("") != "").sum().alias("cases_with_soc"),
        pl.col("requested").is_not_null().sum().alias("cases_with_requested_positions"),
        pl.col("number_of_hours").cast(pl.Float64, strict=False).is_not_null().sum()
        .alias("cases_with_reported_weekly_hours"),
    )
    business_rows = positive.filter(pl.col("employer_is_worksite_business"))
    business_coverage = business_rows.group_by("fiscal_year").agg(
        pl.col("employer_id_balanced").n_unique().alias("all_businesses"),
        pl.col("employer_id_balanced").filter(pl.col("strict_herder")).n_unique()
        .alias("herder_businesses"),
    )
    coverage = coverage.join(business_coverage, on="fiscal_year", validate="1:1").with_columns(
        positions_share_percent=100 * pl.col("herder_positions") / pl.col("all_certified_positions"),
        businesses_share_percent=100 * pl.col("herder_businesses") / pl.col("all_businesses"),
    ).sort("fiscal_year")
    distribution = cases.filter(pl.col("strict_herder")).group_by("fiscal_year", "offered_wage").agg(
        pl.len().alias("cases"), pl.col("certified").sum().alias("certified_positions")
    ).sort("fiscal_year", "offered_wage")
    geography = business_rows.filter(pl.col("strict_herder")).group_by(
        "fiscal_year", "employer_id_balanced"
    ).agg(
        pl.col("worksite_state").filter(pl.col("valid_state")).unique().sort().alias("own_states"),
        pl.col("case_states").explode(empty_as_null=True).drop_nulls().unique().sort().alias("associated_case_states"),
        (~pl.col("valid_state")).any().alias("own_state_missing"),
        pl.col("case_state_missing").any().alias("case_state_missing"),
        pl.col("offered_wage").min().alias("minimum_reported_monthly_wage"),
        pl.col("sheep_title").all().alias("all_titles_explicit_sheep"),
    ).with_columns(
        own_state_count=pl.col("own_states").list.len(),
        associated_case_state_count=pl.col("associated_case_states").list.len(),
        any_protected_own=pl.col("own_states").list.set_intersection(config["candidate_protected_states"]).list.len() > 0,
        any_protected_case=pl.col("associated_case_states").list.set_intersection(config["candidate_protected_states"]).list.len() > 0,
    ).with_columns(
        geography_screen=pl.when(pl.col("own_state_missing") | pl.col("case_state_missing"))
        .then(pl.lit("missing_state_evidence"))
        .when(pl.col("associated_case_state_count") != 1).then(pl.lit("multiple_associated_case_states"))
        .when(pl.col("any_protected_case")).then(pl.lit("single_candidate_protected_state"))
        .otherwise(pl.lit("single_other_state"))
    )
    geo_counts = geography.group_by("fiscal_year", "geography_screen").agg(
        pl.len().alias("businesses"),
        pl.col("all_titles_explicit_sheep").sum().alias("explicit_sheep_businesses"),
        (pl.col("minimum_reported_monthly_wage") < 1206.31).sum().alias("reported_wage_below_initial_floor"),
    ).sort("fiscal_year", "geography_screen")
    title_support = cases.filter(pl.col("monthly")).group_by(
        "fiscal_year", "title_normalized", "strict_herder"
    ).agg(pl.len().alias("cases"), pl.col("certified").sum().alias("certified_positions"))
    wage_quality = cases.group_by("fiscal_year", "wage_unit").agg(
        pl.len().alias("cases"),
        pl.col("offered_wage").is_null().sum().alias("missing_numeric_wage"),
        (pl.col("offered_wage") <= 0).sum().alias("nonpositive_wage"),
        pl.col("offered_wage").min().alias("minimum"),
        pl.col("offered_wage").quantile(0.01).alias("p01"),
        pl.col("offered_wage").quantile(0.99).alias("p99"),
        pl.col("offered_wage").max().alias("maximum"),
    )
    return {"coverage": coverage, "herder_wage_distribution": distribution,
            "monthly_title_support": title_support.sort("fiscal_year", "title_normalized"),
            "wage_quality": wage_quality.sort("fiscal_year", "wage_unit"),
            "herder_geography_support": geo_counts,
            "herder_business_geography": geography.sort("fiscal_year", "employer_id_balanced")}


def classify_baseline(business_rows: pl.DataFrame, baseline_year: int) -> pl.DataFrame:
    """Classification is fixed before looking at subsequent outcomes."""
    baseline = business_rows.filter(pl.col("fiscal_year") == baseline_year).group_by(
        "employer_id_balanced"
    ).agg(
        pl.col("strict_herder").all().alias("all_herder"),
        (pl.col("hourly") & ~pl.col("herder_title")).all().alias("all_hourly"),
        pl.col("sheep_title").all().alias("baseline_explicit_sheep"),
        pl.col("worksite_state").filter(pl.col("valid_state")).unique().sort().alias("baseline_states"),
        pl.col("case_states").explode(empty_as_null=True).drop_nulls().unique().sort().alias("baseline_case_states"),
        (~pl.col("valid_state")).any().alias("baseline_own_state_missing"),
        pl.col("case_state_missing").any().alias("baseline_case_state_missing"),
        pl.col("aewr_region_id").drop_nulls().unique().sort().alias("baseline_regions"),
    ).with_columns(
        baseline_type=pl.when(pl.col("all_herder")).then(pl.lit("herder"))
        .when(pl.col("all_hourly")).then(pl.lit("hourly_nonherder"))
        .otherwise(pl.lit("mixed_or_unclassified")),
        baseline_state=pl.when(pl.col("baseline_states").list.len() == 1)
        .then(pl.col("baseline_states").list.first()).otherwise(None),
        baseline_region=pl.when(pl.col("baseline_regions").list.len() == 1)
        .then(pl.col("baseline_regions").list.first()).otherwise(None),
        geography_supported=(pl.col("baseline_states").list.len() == 1)
        & (pl.col("baseline_case_states").list.len() == 1)
        & ~pl.col("baseline_own_state_missing") & ~pl.col("baseline_case_state_missing"),
    )
    unique_key(baseline, ["employer_id_balanced"])
    return baseline


def cohort_panel(positive: pl.DataFrame, config: dict) -> pl.DataFrame:
    business_rows = positive.filter(pl.col("employer_is_worksite_business"))
    observed_any = business_rows.select("employer_id_balanced", "fiscal_year").unique().with_columns(
        certified_participation=pl.lit(1, dtype=pl.Int64)
    )
    panels = []
    for cohort in config["cohorts"]:
        baseline = classify_baseline(business_rows, cohort["baseline_year"])
        observations = business_rows.join(
            baseline.select("employer_id_balanced", "baseline_type"),
            on="employer_id_balanced", how="inner", validate="m:1",
        ).filter(
            ((pl.col("baseline_type") == "herder") & pl.col("strict_herder"))
            | ((pl.col("baseline_type") == "hourly_nonherder") & pl.col("hourly") & ~pl.col("herder_title"))
        )
        # Repeated worksite rows cannot give a case extra weight in its wage median.
        observations = observations.select(
            "employer_id_balanced", *CASE_KEY, "offered_wage"
        ).unique().group_by("employer_id_balanced", "fiscal_year").agg(
            pl.lit(1, dtype=pl.Int64).alias("same_type_participation"),
            pl.col("offered_wage").filter(
                (pl.col("offered_wage") > 0) & pl.col("offered_wage").is_finite()
            ).median().alias("median_wage"),
            pl.col("case_number").n_unique().alias("same_type_cases"),
        )
        years = pl.DataFrame({"fiscal_year": list(range(cohort["years"][0], cohort["years"][1] + 1))},
                             schema={"fiscal_year": pl.Int32})
        panel = baseline.join(years, how="cross").join(
            observed_any, on=["employer_id_balanced", "fiscal_year"], how="left", validate="1:1"
        ).join(observations, on=["employer_id_balanced", "fiscal_year"], how="left", validate="1:1").with_columns(
            pl.col("certified_participation", "same_type_participation", "same_type_cases").fill_null(0),
            log_wage=pl.col("median_wage").log(),
            cohort=pl.lit(cohort["name"]), baseline_year=pl.lit(cohort["baseline_year"]),
            reference_year=pl.lit(cohort["reference_year"]),
        )
        panels.append(panel)
    combined = pl.concat(panels).sort("cohort", "employer_id_balanced", "fiscal_year")
    unique_key(combined, ["cohort", "employer_id_balanced", "fiscal_year"])
    return combined


def candidate_ddd_support(panel: pl.DataFrame, config: dict) -> pl.DataFrame:
    baseline = panel.filter(
        (pl.col("cohort") == "herder") & (pl.col("fiscal_year") == pl.col("baseline_year"))
        & pl.col("geography_supported")
    ).with_columns(
        support_group=pl.when((pl.col("baseline_type") == "herder") & pl.col("baseline_explicit_sheep"))
        .then(pl.lit("explicit_monthly_sheep"))
        .when(pl.col("baseline_type") == "hourly_nonherder").then(pl.lit("hourly_nonherder"))
        .otherwise(pl.lit("other")),
        candidate_protected=pl.col("baseline_state").is_in(config["candidate_protected_states"]),
    )
    return baseline.group_by("baseline_state", "candidate_protected", "support_group").agg(
        pl.len().alias("baseline_businesses")
    ).sort("candidate_protected", "baseline_state", "support_group")


def linkage_retention_audit(positive: pl.DataFrame, panel: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Audit possible missed returns without changing a canonical business ID."""
    business = positive.filter(pl.col("employer_is_worksite_business"))
    exact_keys = ["employer_name", "employer_address_1", "employer_state", "employer_postal_code"]
    exact_observed = business.filter(pl.all_horizontal([
        pl.col(c).fill_null("").str.strip_chars() != "" for c in exact_keys
    ])).select(*exact_keys, "fiscal_year").unique()
    summaries = []
    for cohort in config["cohorts"]:
        base_rows = business.filter(pl.col("fiscal_year") == cohort["baseline_year"])
        data = panel.filter(pl.col("cohort") == cohort["name"])
        for tier in ["conservative", "high_recall"]:
            tier_key = f"employer_id_{tier}"
            bridge = base_rows.select("employer_id_balanced", tier_key).drop_nulls().unique()
            returns = business.select(tier_key, "fiscal_year").drop_nulls().unique()
            reached = bridge.join(returns, on=tier_key, how="inner", validate="m:m").select(
                "employer_id_balanced", "fiscal_year"
            ).unique().with_columns(pl.lit(1).alias(f"{tier}_return"))
            data = data.join(reached, on=["employer_id_balanced", "fiscal_year"], how="left", validate="1:1")
        bridge = base_rows.filter(pl.all_horizontal([
            pl.col(c).fill_null("").str.strip_chars() != "" for c in exact_keys
        ])).select("employer_id_balanced", *exact_keys).unique().with_columns(
            businesses_per_key=pl.col("employer_id_balanced").n_unique().over(exact_keys)
        ).filter(pl.col("businesses_per_key") == 1)
        reached = bridge.join(exact_observed, on=exact_keys, how="inner", validate="m:m").select(
            "employer_id_balanced", "fiscal_year"
        ).unique().with_columns(exact_identity_return=pl.lit(1))
        data = data.join(reached, on=["employer_id_balanced", "fiscal_year"], how="left", validate="1:1").with_columns(
            pl.col("conservative_return", "high_recall_return", "exact_identity_return").fill_null(0)
        )
        summaries.append(data.group_by("cohort", "baseline_type", "geography_supported", "fiscal_year").agg(
            pl.len().alias("baseline_businesses"),
            pl.col("certified_participation").sum().alias("balanced_returns"),
            pl.col("conservative_return").sum().alias("conservative_reachable_returns"),
            pl.col("high_recall_return").sum().alias("high_recall_reachable_returns"),
            ((pl.col("certified_participation") == 0) & (pl.col("exact_identity_return") == 1))
            .sum().alias("unlinked_exact_identity_return_candidates"),
        ))
    return pl.concat(summaries).sort("cohort", "baseline_type", "geography_supported", "fiscal_year")


def weekly_cutoff_counts(cases: pl.DataFrame, applications: pl.DataFrame, config: dict) -> pl.DataFrame:
    data = []
    for event in config["reversal_cutoffs"]:
        cutoff = date.fromisoformat(event)
        for year in [cutoff.year, 2011, 2012]:
            center = cutoff.replace(year=year)
            sample = cases.filter(pl.col("receipt_valid")).with_columns(
                days=(pl.col("receipt_date") - pl.lit(center)).dt.total_days()
            ).filter(pl.col("days").is_between(-90, 90)).with_columns(
                week=(pl.col("days") // 7), event=pl.lit(event), calendar_year=pl.lit(year),
            ).group_by("event", "calendar_year", "week").agg(
                pl.len().alias("all_positive_cases"),
                pl.col("hourly").sum().alias("hourly_cases"),
                pl.col("certified").sum().alias("certified_positions"),
                pl.col("offered_wage").filter(pl.col("hourly") & (pl.col("offered_wage") > 0))
                .median().alias("hourly_median_wage"),
                (pl.col("start_date") - pl.col("receipt_date")).dt.total_days()
                .filter(pl.col("contract_valid")).median().alias("median_receipt_to_start_days"),
            )
            grid = pl.DataFrame({"week": range(-13, 13)}).with_columns(
                event=pl.lit(event), calendar_year=pl.lit(year)
            )
            sample = grid.join(sample, on=["event", "calendar_year", "week"], how="left", validate="1:1").with_columns(
                pl.col("all_positive_cases", "hourly_cases", "certified_positions").fill_null(0)
            )
            all_counts = applications.filter(pl.col("receipt_valid")).with_columns(
                days=(pl.col("receipt_date") - pl.lit(center)).dt.total_days()
            ).filter(pl.col("days").is_between(-90, 90)).with_columns(week=pl.col("days") // 7).group_by("week").agg(
                pl.len().alias("all_application_case_years"),
                (~pl.col("any_certified")).sum().alias("nonpositive_application_case_years"),
            )
            sample = sample.join(all_counts, on="week", how="left", validate="1:1").with_columns(
                pl.col("all_application_case_years", "nonpositive_application_case_years").fill_null(0)
            )
            data.append(sample)
    return pl.concat(data).sort("event", "calendar_year", "week")


def build() -> dict:
    config = configuration()
    positive, cases, applications, audit = prepare_rows(config)
    tables = coverage_tables(positive, cases, config)
    panel = cohort_panel(positive, config)
    tables["cohort_trajectories"] = panel.group_by(
        "cohort", "baseline_type", "geography_supported", "fiscal_year"
    ).agg(
        pl.len().alias("cohort_businesses"),
        pl.col("certified_participation").sum().alias("participating_businesses"),
        pl.col("certified_participation").mean().alias("participation_rate"),
        pl.col("same_type_participation").mean().alias("same_type_participation_rate"),
        pl.col("median_wage").median().alias("median_offered_wage"),
        pl.col("log_wage").is_not_null().sum().alias("businesses_with_wage"),
    ).sort("cohort", "baseline_type", "geography_supported", "fiscal_year")
    tables["candidate_ddd_support"] = candidate_ddd_support(panel, config)
    tables["linkage_retention_audit"] = linkage_retention_audit(positive, panel, config)
    tables["weekly_cutoff_counts"] = weekly_cutoff_counts(cases, applications, config)
    TABLES.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, frame in tables.items():
        # Keep list-valued identity evidence as a JSON string, not Python repr.
        csv = frame.with_columns([
            pl.col(c).map_elements(lambda v: json.dumps(v.to_list()), return_dtype=pl.String)
            for c, dtype in frame.schema.items() if isinstance(dtype, pl.List)
        ])
        target = TABLES / f"{name}.csv"
        csv.write_csv(target)
        outputs.append(target)
    case_columns = [
        *CASE_KEY, "certified", "requested", "offered_wage", "wage_unit", "hourly", "monthly",
        "strict_herder", "sheep_title", "title_normalized", "soc_code", "worksite_state",
        "aewr_region_id", "receipt_date", "start_date", "end_date", "receipt_valid", "contract_valid",
        "positive_case_rows", "case_states", "case_state_missing",
    ]
    for name, frame in [("cases", cases.select(case_columns)), ("business_year", panel),
                        ("application_cases", applications)]:
        frame = frame.sort(["cohort", "employer_id_balanced", "fiscal_year"] if name == "business_year" else CASE_KEY)
        target = PROCESSED / f"wage_reforms_{name}.parquet"
        frame.write_parquet(target)
        outputs.append(target)
    inputs = [INTERMEDIATE / "h2a_with_fips.parquet",
              INTERMEDIATE / "h2a_employer_record_membership.parquet",
              RAW / "aewr" / "aewr_regions.csv", BRANCH / "design.json", BRANCH / "policy_sources.json",
              BRANCH / "analysis.py", BRANCH / "01_build.py"]
    manifest = {"version": config["version"], "audit": audit,
                "inputs": {relative(p): digest(p) for p in inputs},
                "outputs": {relative(p): digest(p) for p in outputs}}
    write_json(REPORTS / "wage_reforms_build_manifest.json", manifest)
    print(json.dumps({"positive_cases": cases.height, "business_year_rows": panel.height,
                      "build_outputs": len(outputs)}))
    return manifest


# docs-ground:start wage-reforms-estimation
def fixed_effect_rank(groups: list[np.ndarray]) -> int:
    """Exact intercept rank for one or two categorical fixed effects."""
    if len(groups) == 1:
        return int(groups[0].max()) + 1
    if len(groups) != 2:
        raise ValueError("Only one or two fixed effects are supported")
    first, second = groups
    count_first, count_second = int(first.max()) + 1, int(second.max()) + 1
    parents = list(range(count_first + count_second))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for a, b in zip(first, second, strict=True):
        a_root, b_root = root(int(a)), root(count_first + int(b))
        if a_root != b_root:
            parents[b_root] = a_root
    components = len({root(i) for i in range(len(parents))})
    return count_first + count_second - components


def absorb(values: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    residual = values.astype(float).copy()
    for _iteration in range(10000):
        largest_adjustment = 0.0
        for group in groups:
            counts = np.bincount(group)
            means = np.column_stack([
                np.bincount(group, weights=residual[:, j]) / counts
                for j in range(residual.shape[1])
            ])
            adjustment = means[group]
            largest_adjustment = max(largest_adjustment, float(np.max(np.abs(adjustment))))
            residual -= adjustment
        if largest_adjustment < 1e-10:
            return residual
    raise ValueError("Fixed-effect absorption did not converge")


def clustered_ols(y: np.ndarray, x: np.ndarray, effects: list[np.ndarray], clusters: np.ndarray,
                  guards: dict | None = None) -> dict:
    """OLS with absorbed FE; CRV1 counts the exact FE rank, including singletons."""
    if not len(y) or not np.isfinite(np.column_stack([y, x])).all():
        return {"status": "empty_or_nonfinite_sample", "n": len(y)}
    codes = [np.unique(g, return_inverse=True)[1] for g in effects]
    cluster_codes = np.unique(clusters, return_inverse=True)[1]
    count_clusters = int(cluster_codes.max()) + 1
    k_fe = fixed_effect_rank(codes)
    guards = guards or {"maximum_regression_rows": 100000, "maximum_design_columns": 1000}
    if len(y) > guards["maximum_regression_rows"] or k_fe + x.shape[1] > guards["maximum_design_columns"]:
        return {"status": "declared_resource_guard", "n": len(y), "clusters": count_clusters}
    within = absorb(np.column_stack([y, x]), codes)
    y_within, x_within = within[:, 0], within[:, 1:]
    k_x = np.linalg.matrix_rank(x_within, tol=1e-9)
    k = k_fe + k_x
    common = {"n": len(y), "clusters": count_clusters, "design_rank": k,
              "residual_treatment_ss": float(x_within[:, 0] @ x_within[:, 0])}
    if k_x != x.shape[1] or len(y) <= k or count_clusters < 2:
        return {"status": "rank_or_cluster_failure", **common}
    bread = np.linalg.inv(x_within.T @ x_within)
    beta = bread @ x_within.T @ y_within
    residual = y_within - x_within @ beta
    scores = np.zeros((count_clusters, x.shape[1]))
    np.add.at(scores, cluster_codes, x_within * residual[:, None])
    correction = count_clusters / (count_clusters - 1) * (len(y) - 1) / (len(y) - k)
    covariance = correction * bread @ (scores.T @ scores) @ bread
    variance = float(covariance[0, 0])
    if variance < 0 or not np.isfinite(variance):
        return {"status": "invalid_covariance", **common}
    se = float(np.sqrt(variance))
    critical = float(stats.t.ppf(0.975, count_clusters - 1))
    coefficient = float(beta[0])
    return {"status": "estimated_diagnostic", **common,
            "coefficient": coefficient, "std_error": se,
            "ci_low": coefficient - critical * se, "ci_high": coefficient + critical * se,
            "p_value": float(2 * stats.t.sf(abs(coefficient / se), count_clusters - 1)) if se > 0 else None,
            "conditional_mde80": (critical + float(stats.norm.ppf(0.8))) * se}


def local_wage_results(cases: pl.DataFrame, config: dict) -> pl.DataFrame:
    results = []
    for event in config["reversal_cutoffs"]:
        event_date = date.fromisoformat(event)
        for year in [event_date.year, 2011, 2012]:
            cutoff = event_date.replace(year=year)
            for window in config["local_windows_days"]:
                sample = cases.filter(
                    pl.col("receipt_valid") & pl.col("hourly") & (pl.col("offered_wage") > 0)
                    & pl.col("offered_wage").is_finite() & pl.col("aewr_region_id").is_not_null()
                    & (pl.col("title_normalized") != "")
                    & pl.col("receipt_date").is_between(cutoff - timedelta(days=window), cutoff + timedelta(days=window))
                )
                ambiguous = pl.lit(False)
                if event == "2010-03-15" and year == 2010:
                    ambiguous = (pl.col("receipt_date") < pl.lit(cutoff)) & (
                        ~pl.col("contract_valid") | (pl.col("start_date") >= pl.lit(date(2010, 6, 1)))
                    )
                ambiguous_n = sample.filter(ambiguous).height
                sample = sample.filter(~ambiguous)
                base = {"event": event, "calendar_year": year, "window_days": window,
                        "role": "actual_cutoff" if year == event_date.year else "calendar_placebo",
                        "ambiguous_transition_exclusions": ambiguous_n,
                        "pre_cases": sample.filter(pl.col("receipt_date") < pl.lit(cutoff)).height,
                        "post_cases": sample.filter(pl.col("receipt_date") >= pl.lit(cutoff)).height}
                if not sample.height:
                    results.append({**base, "status": "empty_sample", "n": 0})
                    continue
                days = (sample["receipt_date"] - cutoff).dt.total_days().to_numpy() / window
                post = (days >= 0).astype(float)
                fit = clustered_ols(np.log(sample["offered_wage"].to_numpy()),
                                    np.column_stack([post, days, post * days]),
                                    [sample["worksite_state"].to_numpy(), sample["title_normalized"].to_numpy()],
                                    sample["aewr_region_id"].to_numpy(), config["resource_guard"])
                results.append({**base, **fit})
    return pl.DataFrame(results, infer_schema_length=None).sort("event", "calendar_year", "window_days")


def cohort_response_results(panel: pl.DataFrame, config: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    results, deletions = [], []
    for cohort in config["cohorts"]:
        sample = panel.filter((pl.col("cohort") == cohort["name"]) & pl.col("geography_supported")
                              & pl.col("baseline_type").is_in(["herder", "hourly_nonherder"]))
        for outcome in ["certified_participation", "log_wage"]:
            reference = sample.filter(pl.col("fiscal_year") == cohort["reference_year"]).select(
                "employer_id_balanced", pl.col(outcome).alias("reference_outcome")
            )
            # Reference wage missingness stays missing; it never makes participation zero.
            for year in range(cohort["years"][0] + 1, cohort["years"][1] + 1):
                if year == cohort["reference_year"]:
                    continue
                data = sample.filter(pl.col("fiscal_year") == year).join(
                    reference, on="employer_id_balanced", validate="1:1"
                ).with_columns(delta=pl.col(outcome) - pl.col("reference_outcome")).filter(pl.col("delta").is_not_null())
                support = data.group_by("baseline_state").agg(pl.col("baseline_type").n_unique().alias("groups"))
                states = support.filter(pl.col("groups") == 2)["baseline_state"].to_list()
                data = data.filter(pl.col("baseline_state").is_in(states))
                base = {"cohort": cohort["name"], "year": year, "reference_year": cohort["reference_year"],
                        "outcome": outcome, "comparison": "baseline_herder_minus_hourly_within_state",
                        "herder_businesses": data.filter(pl.col("baseline_type") == "herder").height,
                        "hourly_businesses": data.filter(pl.col("baseline_type") == "hourly_nonherder").height}
                if len(states) < 2:
                    results.append({**base, "status": "insufficient_common_state_support", "n": data.height})
                    continue
                y = data["delta"].to_numpy().astype(float)
                x = (data["baseline_type"] == "herder").to_numpy().astype(float).reshape(-1, 1)
                state = data["baseline_state"].to_numpy()
                fit = clustered_ols(y, x, [state], state, config["resource_guard"])
                results.append({**base, **fit})
                for deleted in sorted(states):
                    keep = state != deleted
                    deletion = clustered_ols(y[keep], x[keep], [state[keep]], state[keep], config["resource_guard"])
                    deletions.append({**base, "deleted_state": deleted, **deletion})
    return (pl.DataFrame(results, infer_schema_length=None).sort("cohort", "outcome", "year"),
            pl.DataFrame(deletions, infer_schema_length=None).sort("cohort", "outcome", "year", "deleted_state"))
# docs-ground:end wage-reforms-estimation


def estimate() -> None:
    config = configuration()
    cases = pl.read_parquet(PROCESSED / "wage_reforms_cases.parquet")
    panel = pl.read_parquet(PROCESSED / "wage_reforms_business_year.parquet")
    local = local_wage_results(cases, config)
    cohort, deletion = cohort_response_results(panel, config)
    for name, frame in [("local_wage_responses", local), ("cohort_responses", cohort),
                        ("cohort_state_deletions", deletion)]:
        frame.write_csv(TABLES / f"{name}.csv")
    print(json.dumps({"local_fits": local.height, "cohort_fits": cohort.height,
                      "state_deletions": deletion.height,
                      "local_failures": local.filter(pl.col("status") != "estimated_diagnostic").height,
                      "cohort_failures": cohort.filter(pl.col("status") != "estimated_diagnostic").height}))
