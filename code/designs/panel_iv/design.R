# Frozen constants for the panel-IV branch.
#
# The primary design targets published FLS quarter-by-duration worker
# composition and the annual FLS combined field-and-livestock wage. Census of
# Agriculture directly hired workers, updated with QCEW, QWI, and BEA, remain
# the frame prior and an explicit benchmark. OEWS supplies the corresponding
# area wage.

DISSIMILARITY_IV_K_VALUES <- 5L
DISSIMILARITY_IV_PRIMARY_K <- 5L
DISSIMILARITY_IV_PRIMARY_DONOR_COUNT <- 2L

DISSIMILARITY_IV_FEATURE_START_YEAR <- 2008L
DISSIMILARITY_IV_FEATURE_END_YEAR <- 2011L
DISSIMILARITY_IV_POLICY_START_YEAR <- 2012L
DISSIMILARITY_IV_POLICY_END_YEAR <- 2022L

DISSIMILARITY_IV_INSTRUMENT_FAMILY <- "dissimilarity_cluster"
DISSIMILARITY_IV_AGGREGATION_SPEC <- "unique_oews_area"
DISSIMILARITY_IV_FRAME_WEIGHT_SPEC <-
  "census_hired_workers_qcew_updated"
DISSIMILARITY_IV_PRIMARY_WEIGHT_SPEC <-
  "fls_realized_geography_dirichlet_entropy"
DISSIMILARITY_IV_PRIMARY_WEIGHT_SPECIFICATION <-
  "fls_geo_field_livestock_dirichlet_m10_rho010"
DISSIMILARITY_IV_PRIMARY_WEIGHT_COMPONENT <-
  "calibrated_center"
DISSIMILARITY_IV_PRIMARY_MOMENT_SPEC <-
  "fls_joint_quarter_duration_plus_field_livestock_wage"
DISSIMILARITY_IV_PRIMARY_RHO <- 0.10
DISSIMILARITY_IV_PRIMARY_KAPPA_MULTIPLIER <- 10
DISSIMILARITY_IV_FRAME_INSTRUMENT_LABEL <- paste0(
  "k",
  DISSIMILARITY_IV_PRIMARY_K,
  "_d",
  DISSIMILARITY_IV_PRIMARY_DONOR_COUNT,
  "_census_frame"
)
DISSIMILARITY_IV_PRIMARY_INSTRUMENT_LABEL <- paste0(
  "k",
  DISSIMILARITY_IV_PRIMARY_K,
  "_d",
  DISSIMILARITY_IV_PRIMARY_DONOR_COUNT,
  "_fls_geo_field_livestock_dirichlet_m10_rho010_center"
)
DISSIMILARITY_IV_BASELINE_FRAME_PROXY <-
  "census_ag_direct_hired_workers"
DISSIMILARITY_IV_ANNUAL_UPDATE_SPEC <-
  "qcew_qwi_bea_two_sided_state_raked"
DISSIMILARITY_IV_GEOGRAPHIC_ALLOCATION_SPEC <-
  "oews_township_share_within_county"
DISSIMILARITY_IV_CLUSTER_SIZE_RULE <- "none"
DISSIMILARITY_IV_INFERENCE_CLUSTER <- "aewr_region_id"

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
  "wage_p25_l1"
)

DISSIMILARITY_IV_BOOTSTRAP_REPS <- 999L
DISSIMILARITY_IV_BOOTSTRAP_SEED <- 20260725L
DISSIMILARITY_IV_AR_GRID_POINTS <- 401L
DISSIMILARITY_IV_AR_LEVEL <- 0.95

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
