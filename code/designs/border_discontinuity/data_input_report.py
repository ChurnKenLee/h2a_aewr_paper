"""Generate the border input audit and reviewable external-data requests."""

from __future__ import annotations

import hashlib
import json
import math

import polars as pl
import us

from h2a.geography import geo_expr, normalize_geo_code
from h2a.paths import CODE, INTERMEDIATE, OUTPUTS, PROCESSED, RAW, ROOT, TABLES


def markdown_table(frame: pl.DataFrame) -> list[str]:
    def value(item):
        if item is None or isinstance(item, float) and math.isnan(item):
            return "unavailable"
        if isinstance(item, float):
            return f"{item:.6g}"
        return str(item).replace("|", "\\|")
    return ["| " + " | ".join(frame.columns) + " |",
            "| " + " | ".join(["---"] * frame.width) + " |",
            *["| " + " | ".join(value(x) for x in row) + " |" for row in frame.iter_rows()]]


def source_coverage() -> pl.DataFrame:
    states = pl.DataFrame([{"state_abbrev": s.abbr, "state_fips": normalize_geo_code(s.fips, "state_fips")}
                           for s in [*us.STATES_AND_TERRITORIES, us.states.DC]])
    regions = pl.read_csv(RAW / "aewr/aewr_regions.csv", schema_overrides={"aewr_region_num": pl.String}).with_columns(
        geo_expr("aewr_region_num", "aewr_region_id")).join(states, on="state_abbrev", validate="1:1")
    records = []
    for name, filename, value, geography in [
        ("AEWR annual reference", "aewr.parquet", "aewr", "state_fips"),
        ("BEA total farm jobs", "bea_caemp25n_data_year.parquet", "emp_farm", "county_fips"),
        ("Census hired-worker reports", "census_ag_hired_worker_duration_county.parquet", "census_hired_workers_total", "county_fips"),
        ("QCEW 111/112 quarterly covered jobs", "qcew_county_ag_quarterly_employment.parquet", "qcew_reference_month_emplvl", "county_fips"),
        ("H2A allocated fiscal worksite/county records", "h2a_allocated_worksites.parquet", "adjusted_nbr_workers_certified", "county_fips"),
    ]:
        if filename == "h2a_allocated_worksites.parquet":
            frame = pl.read_parquet(INTERMEDIATE / filename).select(
                pl.col("county_fips_list").str.split(",").alias("county_fips"),
                pl.col("fiscal_year").alias("year"), value).explode("county_fips")
        else:
            frame = pl.read_parquet(INTERMEDIATE / filename).select(geography, "year", value)
        if geography == "county_fips":
            frame = frame.with_columns(pl.col(geography).str.slice(0, 2).alias("state_fips"))
        frame = frame.join(regions.select("state_fips", "aewr_region_id"), on="state_fips", how="left", validate="m:1")
        record = frame.group_by("year", "aewr_region_id").agg(
            pl.len().alias("source_cells"), pl.col(geography).n_unique().alias("geographies_reported"),
            pl.col(value).is_finite().fill_null(False).sum().alias("observed_numeric_cells"),
        ).with_columns(pl.lit(name).alias("source"), pl.col("year").cast(pl.Int32))
        records.append(record)
    result = pl.concat(records).sort(["source", "year", "aewr_region_id"])
    result.write_csv(TABLES / "border_data_input_source_coverage.csv")
    return result


def write_access_requests(report_dir):
    variables = [
        ("historical H2A case/job-order and worksite IDs", "2000--2024", "recover consistent pre-2008 outcomes and audit amendment duplicates"),
        ("certified/requested site counts and worksite intervals", "2000--2024", "audit allocation and requested timing without claiming realized employment"),
        ("OES wage determination and validity interval", "2009--2010", "recover area, SOC, skill, job-order cohort and actually applicable statutory wage"),
        ("job duties, occupation, skill and competing wage determinations", "2009--2010;2023--2024", "assign the legal wage floor rather than a state average"),
    ]
    frame = pl.DataFrame([{"variable_concept": a, "requested_years": b, "purpose": c,
                          "agency_variable_code": "confirm_with_provider", "access_status": "not_requested"}
                         for a, b, c in variables])
    frame.write_csv(TABLES / "border_data_access_variable_request.csv")
    text = [
        "# Border study: public-data plan and DOL archival request", "",
        "Non-public NASS data and custom tabulations are unavailable by user instruction. The former NASS access/tabulation proposal is withdrawn. No institution, eligible project lead or secure-access application is needed for the current county analysis.", "",
        "## Current county analysis", "",
        "Use the existing public BEA farm-job denominators and published 2007 Census hired-worker totals. These county totals cover agricultural activity beyond H-2A applicants; they do not require identifying which individual farms participate. The county comparison is already runnable from the retained public inputs, with its identification and precision limitations reported separately.", "",
        "Nonparticipant farm records would be needed for a farm-level probability of applying or adopting H-2A: the denominator would be all eligible farms, including those never observed in application files. That is a different outcome from certifications per county farm job. It is not a prerequisite for this project and is not pursued through restricted NASS access.", "",
        "## Public-data spatial extension, if pursued", "",
        "One possible different estimand is allocated certifications or position-days per predetermined agricultural acre. It would use actual worksite geocodes, a geographic frame defined independently of subsequent H-2A use, and public land-use data. Areas with no certifications would remain in the frame, while incomplete geocoding or source coverage must not be coded as zero. This measures geographic utilization intensity, not the share of farms adopting H-2A. No such finer estimator is implemented or selected by the current registry.", "",
        "## DOL/OFLC archival records request — draft, not sent", "",
        "For a longer historical extension, ask first whether downloadable electronic FY2000--2007 case/job-order records and dictionaries exist, including worksite records where retained. Relevant fields are case/job-order IDs, status/revision history, locations, requested/certified positions and date intervals. Request documentation of changes in reporting units and coverage and release/amendment IDs to avoid counting administrative versions twice.", "",
        "For 2009--early 2010, seek the OES wage determinations actually used: area definitions/crosswalks, SOC, skill, rate and validity dates, applicable rule/job-order cohort, and recorded prevailing/minimum/CBA overrides. Application-wage averages in the 2010 final rule cannot supply those rates. For 2023--2024, seek archived occupation lookups and job-order duties/cohort fields. Do not assert that the requested records were retained. These gaps do not prevent the current 2011--2022 county comparisons from running.", "",
        "The [OFLC performance page](https://www.dol.gov/agencies/eta/foreign-labor/performance) is the public release entry point. The [SeasonalJobs feeds](https://seasonaljobs.dol.gov/feeds) cover a recent update window and cannot reconstruct this historical panel. Report generation does not submit this draft.", "",
        "## Public worksite geocoding", "",
        "The generated batches contain actual worksite streets from public disclosures. `H2A_RUN_PUBLIC_GEOCODER=1` enables the Census batch requests; default execution prepares requests and reads caches. The source README documents the 10,000-address batches, 50-batch cap, cache and failure behavior. No restricted NASS permission is involved. The earlier execution environment failed to resolve the public Census/FLAG hostnames; this was a programmatic download failure, not an institutional data-access requirement. The geocoding manifest records the latest run and coverage.", "",
        "Address-range matches still need outcome-linkage, coordinate-quality and boundary-vintage review. See the [Census API documentation](https://geocoding.geo.census.gov/geocoder/Geocoding_Services_API.html). Employer mailing addresses or filing-agent locations cannot substitute for worksites.", "",
    ]
    (report_dir / "border_data_access_requests.md").write_text("\n".join(text))


def build_report():
    report_dir = OUTPUTS / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    coverage = source_coverage()
    history = coverage.group_by("source").agg(pl.col("year").min().alias("first_year"),
        pl.col("year").max().alias("last_year"), pl.col("source_cells").sum(),
        pl.col("observed_numeric_cells").sum()).sort("source")
    allocation = pl.read_parquet(INTERMEDIATE / "h2a_case_allocation_audit.parquet")
    exposure = pl.scan_parquet(INTERMEDIATE / "h2a_worksite_exposure_audit.parquet")
    dates = exposure.group_by("fiscal_year").agg(pl.len().alias("allocated_rows"),
        pl.col("valid_site_interval").sum().alias("rows_with_valid_site_dates"),
        pl.col("site_case_dates_differ").sum().alias("site_intervals_different_from_case"),
        pl.col("exposure_begin").is_null().sum().alias("rows_without_valid_interval"),
    ).sort("fiscal_year").collect()
    dates.write_csv(TABLES / "border_data_input_worksite_date_audit.csv")
    bad = allocation.filter(~pl.col("case_allocation_reconciled"))
    bad.write_csv(TABLES / "border_data_input_unreconciled_cases.csv")
    estimates = pl.read_csv(TABLES / "border_data_input_estimates.csv")
    results = estimates.select("sample", "denominator", "n_counties", "n_pairs",
        (pl.col("estimate") * 1000).alias("estimate_per_1000_denominator_units"),
        (pl.col("crv3_jackknife_standard_error") * 1000).alias("pair_CV3J_SE_per_1000"),
        (pl.col("dyadic_standard_error") * 1000).alias("dyadic_SE_per_1000"))
    power = pl.read_csv(TABLES / "border_data_input_power.csv")
    null = power.filter((pl.col("effect") == 0) & (pl.col("scale_information") == "preperiod_only")
                         & pl.col("inference_method").is_in(["pair_CV3J", "shared_endpoint_dyadic"]))
    size = null.group_by("model_id", "inference_method").agg(
        pl.col("rejection_rate").max().alias("maximum_null_rejection"),
        pl.col("mc_confidence_high").max().alias("maximum_null_Wilson_upper"),
        pl.col("size_gate_pass").all().alias("all_four_DGP_size_gates"))
    manifests = {key: json.loads((INTERMEDIATE / name).read_text()) for key, name in [
        ("wage", "aewr_source_manifest.json"), ("exposure", "h2a_worksite_exposure_manifest.json"),
        ("geocoding", "h2a_worksite_geocode_manifest.json")]}
    geo = manifests["geocoding"]
    acquisition_status = (
        f"The live request failed with `{geo['network_failure']}`."
        if geo["network_failure"] else
        "Acquisition was enabled; the manifest records completed and pending batches."
        if geo["acquisition_requested"] else
        "This run prepared requests and read caches without live acquisition."
    )
    annual = pl.read_parquet(INTERMEDIATE / "aewr.parquet")
    skill = pl.scan_parquet(INTERMEDIATE / "aewr_occupation_skill_rates.parquet").select(pl.len()).collect().item()
    events = pl.read_parquet(INTERMEDIATE / "aewr_policy_events.parquet")
    write_access_requests(report_dir)
    report = [
        "# Border study: data input improvements", "",
        "The cached-data corrections, worksite calendar, available historical extensions and denominator comparisons are implemented. The current county comparisons use public data and do not require nonparticipant farm microdata or point geocodes. Non-public NASS data/custom tabulations are unavailable by user instruction; the former access proposal is withdrawn. Exact 2009 statutory determinations and worksite coordinates remain gaps for historical and finer spatial extensions. No causal design is certified by this input work.", "",
        "## Wage sources and dated references", "",
        f"The source produces {annual.height:,} state-year references for 1990--2025. All 49 canonical 2009 rates are missing; the application means are separately named and retained. The old USDA validation join had dropped 1990--1999. The corrected source retains that history, independent of workbook validation coverage. The cached occupation/skill workbooks supply {skill:,} state/SOC/skill/worker-category lookup rows across the 2025--2027 releases.", "",
        "The annual reference is not every job's wage obligation. Dated FLS intervals and source hashes are retained separately. The 2009 OES regime needs area, occupation, skill and cohort determinations; 2023 onward requires occupational applicability; December 2024 has state/entity injunction timing; prevailing, collective-bargaining, federal/state minimum and promised wages can also bind. The three uncached historical OEWS lookup downloads failed under restricted networking. See the [AEWR contract](../../content/contracts/aewr-inputs.md) and [source audit](../tables/border_data_input_source_coverage.csv).", "",
        "The shared panel preserves 2009 outcomes when the statutory wage is missing. Its old certification and fixed-2011-denominator values are retained on common rows. Source normalization propagates the wage gap into its lags. Border stage04 explicitly reconstructs the historical nonstatutory comparator; stage07 uses canonical wages and does not require a wage observation for FE-only outcome calibration.", "",
        "The existing DiD complete-case rule now excludes 2010 because its lagged statutory wage is missing. The MC available-value baseline rule uses 2008 alone for the 2008--2009 fraction affected and 2008/2010 for the longer wage-based means. Separate wage-support counts distinguish these from complete quantile histories. These source corrections require downstream rebuilds; the design registries, prediction cutoff and policy-period treatment definitions are unchanged.", "",
        "## Worksite allocation and timing", "",
        f"The new producer writes {manifests['exposure']['monthly_rows']:,} worksite–county–month rows. It uses inclusive requested site intervals where valid, labels case-date fallback, retains missing weekly hours and equally splits only explicitly ambiguous county lists. It neither replaces site intervals with case-modal dates nor claims actual visa entries or employment. {bad.height} cases fail case-total reconciliation and are quarantined from the new calendar; the [case audit](../tables/border_data_input_unreconciled_cases.csv) retains them.", "",
        *markdown_table(dates.filter(pl.col("fiscal_year") >= 2020)), "",
        "The outcome is allocated certification position-days. Requested staffing weights are not observed certified staffing by site, and a crew can rotate among sequential worksites. Hours use reported or case-inherited weekly schedules without mean imputation. County-year artifacts distinguish observed site dates, fallback dates, ambiguous counties, imputed staffing and weekly-hours coverage. The historical case-date certification benchmark remains separate. FLS references are matched by interval overlap, with uncovered days retained. The capacity-weighted nominal reference is a timing diagnostic using outcome-dependent weights, not a predetermined treatment or instrument; none of the denominator regressions uses it. It does not establish the binding job wage.", "",
        "## Predetermined denominators and precision", "",
        "Four fixed denominators use the same screened geography, 2011--2022 annual real AEWR, equal county-year weights and county/component-year FE. BEA farm jobs include proprietors; hired farm jobs are the design-owned difference between farm and farm-proprietor jobs. The 2008--2010 means require all three positive observed years. The 2007 Census denominator counts hired-worker reports across farms, not a deduplicated population of people. Suppressed/missing values are not interpolated or repaired with later outcomes.", "",
        *markdown_table(results), "",
        "The common-sample rows hold counties and observations fixed across denominators. Changing the denominator changes units and can change the implicit weighting of heterogeneous effects; smaller numerical standard errors are not proof of more information. None of these ratios is a share of uniquely identified people.", "",
        "Common-sample power uses the existing four declared serial/shared-region/spatial DGPs, 999 simulation and calibration draws, seed 201103 and one thread. Reference effects and signed MDEs are expressed in each denominator's own units; they are not economically identical thresholds. The [power](../tables/border_data_input_power.csv), [calibration](../tables/border_data_input_calibration.csv), [MDE](../tables/border_data_input_mde.csv) and [invalid-variance diagnostics](../tables/border_data_input_simulation_diagnostics.csv) retain failures. Approximate CV3J MDEs in the estimates file are separate from simulation-based size-valid MDEs.", "",
        *markdown_table(size.sort(["model_id", "inference_method"])), "",
        "## Historical and policy coverage", "",
        *markdown_table(history), "",
        "The source/year/AEWR-region coverage CSV retains reporting cells and observed numeric cells. QCEW measures covered jobs and excludes proprietors and some agricultural workers; suppression is not zero. The wider BEA/Census histories do not extend the shared 2008--2022 estimation backbone or fabricate pre-2008 H-2A outcomes. Later disclosure years likewise do not automatically extend the annual wage design.", "",
        f"The sourced policy ledger contains {events.height} dated entries, including the enjoined 2009 suspension, 2010 return to FLS, 2021 timing, occupational changes and injunction exceptions. All carry an explicit absence of an exogeneity assumption. Routine FLS/OEWS wage updates reflect economic conditions; methodology changes also alter other program rules; occupation and injunction eligibility need job/entity information. The ledger does not identify a clean differential wage innovation by itself. Annual state minimum-wage references cannot resolve every agricultural exemption or local/contractual floor.", "",
        "## Spatial inputs and external dependencies", "",
        f"The geocoding producer extracted {geo['location_records']:,} public worksite records and {geo['unique_addresses']:,} distinct normalized addresses. {geo['requestable_addresses']:,} are requestable in {geo['expected_requests']} bounded Census batches. Actual matched coordinates: {geo['matched_addresses']:,}. {acquisition_status} Pending coordinates stay null. Main disclosure worksite streets are extracted separately from employer mailing addresses, with unresolved case/worksite linkage flagged. Addendum B record IDs align with the B01 allocation IDs when those rows are retained.", "",
        "Nonparticipant farm records would be needed to estimate a farm-level probability of applying or adopting H-2A: application files cannot enumerate farms that never apply. That is a different estimand from certifications per county farm job. The current county denominators already come from public BEA and Census totals, so this is not a missing input for the implemented analysis. Restricted NASS access and custom tabulations are not pursued.", "",
        "A possible public-data spatial extension could measure certifications or position-days per predetermined agricultural acre, using a complete geographic frame that includes zero-certification areas. It would measure utilization intensity rather than farm adoption, and would require actual worksite coordinates, boundary/coverage audits and an explicitly declared new outcome. Missing geocodes must not become zero activity. This extension is not implemented or promoted by the current registry.", "",
        "The [public-data plan and DOL archival draft](border_data_access_requests.md) replaces the former NASS request. The [variable request CSV](../tables/border_data_access_variable_request.csv) now lists only DOL historical concepts and purposes. No institutional-access details are requested for the county analysis. Public downloads failing in the execution environment are a separate technical limitation.", "",
        "## Reproduction", "",
        "Run `./scripts/run_border_data_inputs.sh` to reproduce cached source corrections, allocation/calendar and geocoding preparation, shared panel, descriptives, DiD, panel IV, bounded MC, ordinary border, and stages09--11. Inspect its order with `DRY_RUN=1` first. Public acquisition is a separate explicit environment flag. See the branch README for requirements. All generated artifacts have one producer; raw inputs, manifests and separate candidate estimates remain traceable. The input validator checks source keys, missingness, calendar conservation, common samples, simulation settings and provenance; external acquisition is a separate incomplete requirement.", "",
        "Public definitions: [QCEW coverage](https://www.bls.gov/cew/overview.htm), [Census geocoder](https://geocoding.geo.census.gov/geocoder/Geocoding_Services_API.html), [FLAG wage archives](https://flag.dol.gov/wage-data/adverse-effect-wage-rates).", "",
    ]
    (report_dir / "border_data_inputs.md").write_text("\n".join(report))
    paths = [CODE / "designs/border_discontinuity/data_input_report.py",
        CODE / "designs/border_discontinuity/data_input_design.json",
        PROCESSED / "border_data_input_county_year.parquet",
        INTERMEDIATE / "aewr.parquet", INTERMEDIATE / "aewr_source_manifest.json",
        INTERMEDIATE / "h2a_worksite_exposure_manifest.json", INTERMEDIATE / "h2a_worksite_geocode_manifest.json",
        TABLES / "border_data_input_estimates.csv", TABLES / "border_data_input_power.csv",
        TABLES / "border_data_input_source_coverage.csv"]
    manifest = {"design_id": "border_data_inputs_v1", "external_data_complete": False,
                "nonpublic_nass_required": False,
                "external_gap_scope": "historical wage and finer spatial extensions; current county comparisons use public inputs",
                "files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (report_dir / "border_data_input_report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"report": "outputs/reports/border_data_inputs.md", "external_data_complete": False}
