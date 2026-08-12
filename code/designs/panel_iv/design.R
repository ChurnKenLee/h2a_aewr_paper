# Frozen constants for the panel-IV branch.
#
# The publication design compares two soft-entropy instruments constructed
# from the same Census hired-worker frame prior.  The first targets only the
# annual FLS combined field-and-livestock/OEWS hourly-wage moment. The
# preferred instrument adds QCEW seasonal and field/livestock-composition
# moments. Worker-duration moments and a separate Census-frame instrument are
# not publication specifications.

# docs-ground:start panel-iv-design-contract
DISSIMILARITY_IV_K_VALUES <- 5L
DISSIMILARITY_IV_PRIMARY_K <- 5L
DISSIMILARITY_IV_PRIMARY_DONOR_COUNT <- 2L

DISSIMILARITY_IV_FEATURE_START_YEAR <- 2008L
DISSIMILARITY_IV_FEATURE_END_YEAR <- 2011L
DISSIMILARITY_IV_POLICY_START_YEAR <- 2011L
DISSIMILARITY_IV_POLICY_END_YEAR <- 2022L

DISSIMILARITY_IV_INSTRUMENT_FAMILY <- "dissimilarity_cluster"
DISSIMILARITY_IV_AGGREGATION_SPEC <- "county"
DISSIMILARITY_IV_FRAME_WEIGHT_SPEC <-
  "census_hired_workers_qcew_annual_updated_v2"
DISSIMILARITY_IV_PRIMARY_WEIGHT_SPEC <-
  "fls_pseudo_county_entropy_v2"
DISSIMILARITY_IV_WAGE_ONLY_WEIGHT_SPECIFICATION <-
  "fls_county_wage_only_soft_rho010_v2"
DISSIMILARITY_IV_PRIMARY_WEIGHT_SPECIFICATION <-
  "fls_county_wage_seasonal_composition_soft_rho010_v2"
DISSIMILARITY_IV_PRIMARY_WEIGHT_COMPONENT <-
  "calibrated_center"
DISSIMILARITY_IV_WAGE_ONLY_MOMENT_SPEC <-
  "fls_field_livestock_wage_only"
DISSIMILARITY_IV_PRIMARY_MOMENT_SPEC <-
  "fls_oews_wage_plus_qcew_seasonal_and_composition"
DISSIMILARITY_IV_DONOR_WAGE_SPEC <-
  "county_mapped_oews_area_big_six_hourly_v1"
DISSIMILARITY_IV_DONOR_WAGE_SOURCE <-
  "oews_area_big_six_hourly"
DISSIMILARITY_IV_DONOR_WAGE_GEOGRAPHY <-
  "oews_reporting_area_mapped_to_county"
DISSIMILARITY_IV_PRIMARY_RHO <- 0.10
DISSIMILARITY_IV_PRIMARY_KAPPA_MULTIPLIER <- 10
DISSIMILARITY_IV_WAGE_ONLY_INSTRUMENT_LABEL <- paste0(
  "k",
  DISSIMILARITY_IV_PRIMARY_K,
  "_d",
  DISSIMILARITY_IV_PRIMARY_DONOR_COUNT,
  "_oews_hourly_wage_only_soft_rho010_center"
)
DISSIMILARITY_IV_PRIMARY_INSTRUMENT_LABEL <- paste0(
  "k",
  DISSIMILARITY_IV_PRIMARY_K,
  "_d",
  DISSIMILARITY_IV_PRIMARY_DONOR_COUNT,
  "_oews_hourly_wage_seasonal_composition_soft_rho010_center"
)
DISSIMILARITY_IV_BASELINE_FRAME_PROXY <-
  "census_ag_direct_hired_workers"
DISSIMILARITY_IV_ANNUAL_UPDATE_SPEC <-
  "qcew_annual_qwi_bea_two_sided_state_raked_v2"
DISSIMILARITY_IV_GEOGRAPHIC_ALLOCATION_SPEC <-
  "none_county_keyed"
DISSIMILARITY_IV_CLUSTER_SIZE_RULE <- "none"
DISSIMILARITY_IV_INFERENCE_CLUSTER <- "aewr_iv_cluster_id"

# All selected donor clusters must contribute a wage in a given year. Donor
# unit counts are diagnostics only; no minimum cluster-size rule is imposed.
DISSIMILARITY_IV_MIN_OBSERVED_DONOR_CLUSTERS <-
  DISSIMILARITY_IV_PRIMARY_DONOR_COUNT

DISSIMILARITY_IV_BIG_SIX_OCC_CODES <- c(
  "45-2041",
  "45-2091",
  "45-2092",
  "45-2093",
  "53-7064",
  "45-2099",
  "79011",
  "79021",
  "79856",
  "79858",
  "98902"
)

DISSIMILARITY_IV_BASELINE_CONTROL_COLUMNS <- c(
  "ln_pop_census_l1",
  "farm_emp_share_l1",
  "emp_pop_ratio_l1",
  "wage_p10_l1"
)

# The time-invariant PPML level is absorbed by county fixed effects. The
# controlled publication specifications instead allow counties with different
# baseline propensities to follow different linear trends.
DISSIMILARITY_IV_PROPENSITY_COLUMN <- "h2a_ppml_static_propensity_z"
DISSIMILARITY_IV_PROPENSITY_TREND_TERM <- paste0(
  DISSIMILARITY_IV_PROPENSITY_COLUMN,
  ":year_centered"
)

DISSIMILARITY_IV_CONTROL_COLUMNS <- c(
  DISSIMILARITY_IV_BASELINE_CONTROL_COLUMNS,
  DISSIMILARITY_IV_PROPENSITY_COLUMN,
  "year_centered"
)

DISSIMILARITY_IV_CONTROL_TERMS <- c(
  DISSIMILARITY_IV_BASELINE_CONTROL_COLUMNS,
  DISSIMILARITY_IV_PROPENSITY_TREND_TERM
)

make_dissimilarity_cluster_id <- function(aewr_region_id, target_cluster) {
  ifelse(
    is.na(aewr_region_id) | is.na(target_cluster),
    NA_character_,
    paste0(
      sprintf("%02d", as.integer(aewr_region_id)),
      "_",
      sprintf("%02d", as.integer(target_cluster))
    )
  )
}

make_panel_iv_target_unit_id <- function(cz_id, aewr_region_id) {
  ifelse(
    is.na(cz_id) | is.na(aewr_region_id),
    NA_character_,
    paste0(cz_id, "_", aewr_region_id)
  )
}
# docs-ground:end panel-iv-design-contract
