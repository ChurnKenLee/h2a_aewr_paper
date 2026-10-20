# Validate retained artifacts independently of the construction routines.
iv_check <- function(condition, message) {
  if (anyNA(condition) || !all(condition)) stop(message, call. = FALSE)
  invisible(TRUE)
}

iv_validate_linkage <- function(panel, filled) {
  fields <- c("employer_linkage_unit", "employer_linkage_version", "employer_linkage_input_max_year",
    "employer_linkage_review_status", "employer_linkage_review_report_hash")
  iv_require_columns(panel, fields, "Employer linkage")
  if (filled) iv_check(!anyNA(panel[fields]), "Unfilled employer-linkage metadata")
  contract <- unique(panel[!is.na(panel$employer_linkage_unit), fields])
  iv_check(nrow(contract) == 1L, "Ambiguous employer-linkage contract")
  iv_check(contract$employer_linkage_unit == "worksite_business" & contract$employer_linkage_version == "v2" &
    contract$employer_linkage_input_max_year == 2024L & contract$employer_linkage_review_status == "complete" &
    grepl("^[0-9a-f]{64}$", contract$employer_linkage_review_report_hash), "Invalid employer-linkage contract")
}

iv_validate_sources <- function() {
  quarterly <- iv_read(path_int("qcew_county_ag_quarterly_employment.parquet"))
  iv_require_unique(quarterly, c("county_fips", "year", "qtr", "industry_code"), "Quarterly QCEW")
  assert_geo_columns(quarterly, "county_fips")
  iv_check(quarterly$industry_code %in% c("111", "112") & quarterly$qtr %in% 1:4, "Malformed QCEW keys")
  iv_check(quarterly$qcew_employment_disclosed ==
    (quarterly$qcew_disclosed_ownership_components == quarterly$qcew_reported_ownership_components), "QCEW disclosure mismatch")
  observed <- quarterly$qcew_employment_disclosed
  iv_check(is.finite(quarterly$qcew_reference_month_emplvl[observed]) &
    quarterly$qcew_reference_month_emplvl[observed] >= 0, "Invalid disclosed QCEW employment")
  iv_check(is.na(quarterly$qcew_reference_month_emplvl[!observed]), "Suppressed QCEW is exposed")
  for (path in c(path_int("county_year_merged.parquet"), path_processed("county_year_panel.parquet"))) {
    panel <- iv_read(path)
    iv_require_unique(panel, c("county_fips", "year"), basename(path))
    assert_geo_columns(panel, "county_fips")
    iv_validate_linkage(panel, identical(path, path_processed("county_year_panel.parquet")))
    for (series in c("crop_sector", "animal_sector", "all_sectors")) {
      fields <- paste0("qcew_", series, c("_disclosed", "_annual_avg_emplvl", "_total_annual_wages"))
      iv_require_columns(panel, fields, "Shared QCEW")
      observed <- !is.na(panel[[fields[1]]]) & panel[[fields[1]]]
      suppressed <- !is.na(panel[[fields[1]]]) & !panel[[fields[1]]]
      for (field in fields[2:3]) {
        iv_check(is.finite(panel[[field]][observed]) & panel[[field]][observed] >= 0, "Invalid shared disclosed QCEW")
        iv_check(is.na(panel[[field]][suppressed]), "Shared QCEW exposes suppressed values")
      }
    }
  }
  workers <- iv_read(path_int("fls_region_quarterly_workers.parquet"))
  wages <- iv_read(path_int("fls_region_quarterly_wages.parquet"))
  keys <- c("aewr_region_id", "year", "quarter")
  for (data in list(workers, wages)) {
    iv_require_unique(data, keys, "FLS survey weeks")
    assert_geo_columns(data, "aewr_region_id")
    iv_check(sum(data$year %in% 2010:2021) == 816L, "Incomplete FLS survey grid")
  }
  paired <- dplyr::inner_join(workers[workers$year %in% 2010:2021, ], wages[wages$year %in% 2010:2021, ],
    by = keys, suffix = c("", "_wage"), relationship = "one-to-one")
  iv_check(nrow(paired) == 816L & paired$quarter %in% c("january", "april", "july", "october"), "Incomplete paired FLS keys")
  for (field in c("source_zip", "release_year", "release_month", "release_day"))
    iv_check(paired[[field]] == paired[[paste0(field, "_wage")]], "FLS release provenance differs")
  observed <- paired$fls_pair_values_available
  iv_check(sum(observed) == 799L, "Unexpected FLS availability")
  fields <- c("fls_hired_workers", "fls_field_hourly_wage", "fls_livestock_hourly_wage",
    "fls_field_livestock_hourly_wage", "fls_all_hired_hourly_wage")
  iv_check(is.finite(as.matrix(paired[observed, fields])), "Invalid published FLS values")
  iv_check(paired$year[!observed] == 2011L & paired$quarter[!observed] == "april" &
    paired$fls_pair_value_status[!observed] == "survey_not_conducted", "Unexpected FLS survey gap")
}

iv_validate_calibration <- function() {
  frame <- iv_read(path_int("panel_iv_fls_frame.parquet"))
  iv_require_unique(frame, c("county_fips", "source_year"), "County frame")
  assert_geo_columns(frame, c("county_fips", "state_fips", "aewr_region_id"))
  iv_check(is.finite(frame$frame_employment_mass) & frame$frame_employment_mass >= 0,
    "Invalid county frame mass")
  iv_check(!"weight_draw_id" %in% names(frame), "Retired frame draw field remains")
  iv_check(frame$state_rake_factor == 1 & frame$two_sided_log1p_drift == 0,
    "Original frame projection convention changed")
  weights <- iv_read(path_int("panel_iv_fls_county_weight_summary.parquet"))
  keys <- c("aewr_region_id", "source_year", "county_fips", "specification")
  iv_require_unique(weights, keys, "County weights")
  assert_geo_columns(weights, c("county_fips", "aewr_region_id"))
  iv_check(is.finite(weights$calibrated_center_weight) & weights$calibrated_center_weight >= 0, "Invalid county weights")
  expected_prior <- frame |>
    dplyr::filter(source_year %in% IV_SOURCE_YEARS) |>
    dplyr::group_by(aewr_region_id, source_year) |>
    dplyr::mutate(expected_prior = frame_employment_mass / sum(frame_employment_mass)) |>
    dplyr::ungroup() |>
    dplyr::select(county_fips, aewr_region_id, source_year, expected_prior)
  support <- dplyr::left_join(weights, expected_prior, by = c("county_fips", "aewr_region_id", "source_year"),
    relationship = "many-to-one")
  iv_check(abs(support$frame_prior_weight - support$expected_prior) <= 1e-12,
    "County prior does not match the frame")
  iv_check(support$calibrated_center_weight[support$expected_prior == 0] == 0,
    "Calibration created support outside the prior")
  totals <- weights |>
    dplyr::group_by(aewr_region_id, source_year, specification) |>
    dplyr::summarise(total = sum(calibrated_center_weight), .groups = "drop")
  iv_check(nrow(totals) == 408L & abs(totals$total - 1) <= 1e-10, "Incomplete or unnormalized weights")
  iv_check(setequal(weights$specification, c(IV_WAGE_SPEC, IV_PRIMARY_SPEC)), "Unexpected calibration specifications")
  diagnostics <- iv_read(path_int("panel_iv_fls_county_calibration_diagnostics.parquet"))
  iv_require_unique(diagnostics, c("aewr_region_id", "source_year", "specification"), "Deterministic diagnostics")
  iv_check(nrow(diagnostics) == 408L & diagnostics$optimizer_success, "Incomplete or failed calibration")
  iv_check(diagnostics$optimizer_iterations <= 100L &
    diagnostics$calibrated_standardized_residual_norm <= diagnostics$input_standardized_residual_norm + 1e-9, "Calibration guards failed")
  moments <- iv_read(path_int("panel_iv_fls_county_moment_diagnostics.parquet"))
  iv_require_unique(moments, c("aewr_region_id", "source_year", "specification", "moment_id"), "Moment diagnostics")
  iv_check(sum(moments$source_year == 2011 & moments$moment_status == "inactive_fls_survey_not_conducted") == 34L,
    "April 2011 inactive moments are incomplete")
  for (data in list(weights, diagnostics, moments, iv_read(path_int("panel_iv_fls_county_features.parquet")))) {
    iv_check("calibration_version" %in% names(data), "Missing deterministic calibration version")
    iv_check(data$calibration_version == IV_CALIBRATION_VERSION, "Stale calibration version")
    iv_check(!any(grepl("draw|simulation|kappa", names(data))), "Simulated calibration fields remain")
  }
  iv_check(!dir.exists(path_int("panel_iv_fls_county_draws")), "Retired calibration draw partitions remain")
  donor <- iv_read(path_int("panel_iv_county_donor_frame.parquet"))
  iv_check(donor$policy_year == donor$source_year + 1L, "Donor timing is not t-1")
  iv_check(donor$donor_wage_spec == "county_mapped_oews_area_big_six_hourly_v1" &
    donor$donor_wage_source %in% c("oews_area_big_six_hourly", "unavailable"), "Undeclared donor wage proxy")
  observed <- donor$donor_wage_source == "oews_area_big_six_hourly"
  selected <- donor[observed, ]
  iv_check(selected$oews_wage_observed & is.finite(selected$oews_big_six_mean_hourly_wage) &
    selected$oews_big_six_mean_hourly_wage > 0 & selected$donor_wage_available &
    is.finite(selected$donor_real_hourly_wage) &
    abs(selected$donor_nominal_hourly_wage - selected$oews_big_six_mean_hourly_wage) <= 1e-12 &
    abs(selected$donor_real_hourly_wage - selected$donor_nominal_hourly_wage / selected$source_year_ppi_2012) <= 1e-12,
    "Invalid selected OEWS hourly wage")
  iv_check(is.na(donor$donor_nominal_hourly_wage[!observed]) & is.na(donor$donor_real_hourly_wage[!observed]) &
    !donor$donor_wage_available[!observed], "Unavailable wage contributor exposes a value")
  iv_check(!any(c("donor_nominal_annual_wage", "donor_real_annual_wage", "qcew_nominal_annual_wage",
    "oews_nominal_annual_wage", "bea_nominal_annual_wage") %in% names(donor)), "Retired wage fallback fields remain")
  instruments <- iv_read(path_int("panel_iv_instrument_cluster_year.parquet"))
  iv_require_unique(instruments, c("aewr_region_id", "target_cluster", "source_year", "instrument_spec_label"), "Instruments")
  iv_check(nrow(instruments) == 2040L, "Incomplete instrument grid")
  iv_check(DISSIMILARITY_IV_PRIMARY_INSTRUMENT_LABEL %in% instruments$instrument_spec_label, "Missing preferred instrument")
  iv_check(instruments$policy_year == instruments$source_year + 1L &
    instruments$donor_wage_spec == "county_mapped_oews_area_big_six_hourly_v1", "Invalid instrument timing or wage proxy")
  supported <- is.finite(instruments$nonoverlap_calibrated_weight_mass) & instruments$nonoverlap_calibrated_weight_mass > 0
  for (field in c("qcew_employment_coverage_weight", "oews_wage_proxy_coverage_weight")) {
    iv_check(is.finite(instruments[[field]][supported]) & instruments[[field]][supported] >= 0 &
      instruments[[field]][supported] <= 1 + 1e-12, "Invalid instrument coverage")
    iv_check(is.na(instruments[[field]][!supported]), "Coverage is defined without nonoverlapping donor support")
  }
  iv_check(!any(c("qcew_wage_weight_share", "oews_fallback_weight_share", "bea_fallback_weight_share",
    "weight_draw_id", "kappa_multiplier") %in% names(instruments)), "Retired instrument fields remain")
}

iv_validate_estimates <- function() {
  panel <- iv_read(path_processed("panel_iv_county_year.parquet"))
  iv_require_unique(panel, c("county_fips", "year"), "Analysis panel")
  assert_geo_columns(panel, "county_fips")
  iv_validate_linkage(panel, TRUE)
  specification <- panel_iv_read_specification(path_code("designs", "panel_iv", "estimation_spec.json"))
  iv_require_columns(panel, c(unique(specification$models$instrument), "aewr_iv_cluster_id"), "Analysis panel")
  iv_check(!"z_wage_seasonal_real" %in% names(panel), "Retired instrument field remains")
  iv_check(length(unique(stats::na.omit(panel$aewr_iv_cluster_id))) == 85L, "Analysis panel must retain 85 clusters")
  first <- iv_read(path_tables("iv_preferred_first_stage_estimates.csv"))
  iv_check(identical(as.integer(first$column), 1:4) &
    first$instrument == specification$models$instrument, "First-stage ordering differs")
  iv_check(length(unique(first$observations)) == 1L & length(unique(first$inference_clusters)) == 1L &
    is.finite(first$first_stage_f), "First stages have inconsistent samples or missing diagnostics")
  second <- iv_read(path_tables("iv_preferred_second_stage_estimates.csv"))
  iv_check(setequal(second$outcome, specification$outcomes$outcome), "Unexpected outcome registry")
  for (outcome in specification$outcomes$outcome) {
    rows <- second[second$outcome == outcome, ]
    iv_check(identical(sort(as.integer(rows$column)), 1:4) & length(unique(rows$observations)) == 1L &
      length(unique(rows$inference_clusters)) == 1L, "Second-stage ordering or common samples differ")
  }
  statistics <- c("AEWR coefficient", "Clustered standard error", "p-value", "First-stage excluded-instrument F",
    "Observations", "Counties", "Inference clusters")
  for (suffix in c("spec1_wage_only", "spec2_seasonal_composition", "spec3_wage_only_controls", "spec4_preferred")) {
    table <- iv_read(path_tables(paste0("panel_iv_email_results_", suffix, ".csv")))
    iv_check(nrow(table) == 7L & ncol(table) == 13L & identical(names(table)[1], "statistic") &
      identical(table$statistic, statistics), "Invalid email table orientation")
    iv_check(is.finite(as.matrix(table[-1])), "Missing email table estimates")
  }
  required <- path_tables("panel_iv_email_results.pdf")
  for (suffix in c("aewr_region_wage_calibration", "national_wage_calibration", "qcew_fls_moment_residuals",
    "county_entropy_weight_changes_pp", "oews_wage_proxy_coverage", "target_donor_support"))
    required <- c(required, path_figures(paste0("fig_iv_", suffix, ".png")))
  for (suffix in c("fls_county_wage_calibration", "national_wage_calibration", "qcew_fls_moment_residuals",
    "county_entropy_weight_changes_pp", "oews_wage_proxy_coverage", "target_donor_support"))
    required <- c(required, path_tables(paste0("iv_", suffix, ".csv")))
  iv_check(file.exists(required) & file.info(required)$size > 0, "Missing retained diagnostic outputs")
  for (suffix in c("fls_oews_cz_scatter", "cz_entropy_weight_changes_pp", "donor_wage_source_shares"))
    iv_check(!file.exists(c(path_figures(paste0("fig_iv_", suffix, ".png")), path_tables(paste0("iv_", suffix, ".csv")))),
      "Obsolete OEWS-area diagnostics remain")
}
