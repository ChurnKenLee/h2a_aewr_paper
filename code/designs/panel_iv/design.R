# Frozen constants for the panel-IV branch.
#
# The publication design compares two soft-entropy instruments constructed
# from the same Census hired-worker frame prior.  The first targets only the
# annual FLS combined field-and-livestock wage.  The preferred instrument adds
# published quarterly FLS worker-share targets.  Worker-duration moments and a
# separate Census-frame instrument are not publication specifications.

DISSIMILARITY_IV_K_VALUES <- 5L
DISSIMILARITY_IV_PRIMARY_K <- 5L
DISSIMILARITY_IV_PRIMARY_DONOR_COUNT <- 2L

DISSIMILARITY_IV_FEATURE_START_YEAR <- 2008L
DISSIMILARITY_IV_FEATURE_END_YEAR <- 2011L
DISSIMILARITY_IV_POLICY_START_YEAR <- 2011L
DISSIMILARITY_IV_POLICY_END_YEAR <- 2022L

DISSIMILARITY_IV_INSTRUMENT_FAMILY <- "dissimilarity_cluster"
DISSIMILARITY_IV_AGGREGATION_SPEC <- "unique_oews_area"
DISSIMILARITY_IV_FRAME_WEIGHT_SPEC <-
  "census_hired_workers_qcew_updated"
DISSIMILARITY_IV_PRIMARY_WEIGHT_SPEC <-
  "fls_realized_geography_dirichlet_entropy"
DISSIMILARITY_IV_WAGE_ONLY_WEIGHT_SPECIFICATION <-
  "fls_geo_wage_only_soft_rho010"
DISSIMILARITY_IV_PRIMARY_WEIGHT_SPECIFICATION <-
  "fls_geo_wage_seasonal_soft_rho010"
DISSIMILARITY_IV_PRIMARY_WEIGHT_COMPONENT <-
  "calibrated_center"
DISSIMILARITY_IV_WAGE_ONLY_MOMENT_SPEC <-
  "fls_field_livestock_wage_only"
DISSIMILARITY_IV_PRIMARY_MOMENT_SPEC <-
  "fls_field_livestock_wage_plus_quarterly_worker_shares"
DISSIMILARITY_IV_PRIMARY_RHO <- 0.10
DISSIMILARITY_IV_PRIMARY_KAPPA_MULTIPLIER <- 10
DISSIMILARITY_IV_WAGE_ONLY_INSTRUMENT_LABEL <- paste0(
  "k",
  DISSIMILARITY_IV_PRIMARY_K,
  "_d",
  DISSIMILARITY_IV_PRIMARY_DONOR_COUNT,
  "_wage_only_soft_rho010_center"
)
DISSIMILARITY_IV_PRIMARY_INSTRUMENT_LABEL <- paste0(
  "k",
  DISSIMILARITY_IV_PRIMARY_K,
  "_d",
  DISSIMILARITY_IV_PRIMARY_DONOR_COUNT,
  "_wage_seasonal_soft_rho010_center"
)
DISSIMILARITY_IV_BASELINE_FRAME_PROXY <-
  "census_ag_direct_hired_workers"
DISSIMILARITY_IV_ANNUAL_UPDATE_SPEC <-
  "qcew_qwi_bea_two_sided_state_raked"
DISSIMILARITY_IV_GEOGRAPHIC_ALLOCATION_SPEC <-
  "oews_township_share_within_county"
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

DISSIMILARITY_IV_CONTROL_COLUMNS <- c(
  "ln_pop_census_l1",
  "farm_emp_share_l1",
  "emp_pop_ratio_l1",
  "wage_p10_l1"
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
