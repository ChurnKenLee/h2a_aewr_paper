# Shared constants for the publication IV design.

IV_PREFERRED_K <- 5L
IV_PREFERRED_DONOR_COUNT <- 2L
IV_PREFERRED_GAP_CLOSURE <- 1
IV_PREFERRED_WEIGHT_SPEC <- "wage_seasonal_exact"
IV_PREFERRED_CALIBRATION_MODE <- "exact"
IV_PREFERRED_PRIOR_SPEC <- "bea"
# Retained as an explicit missing value for older table code. The preferred
# design has exact constraints and therefore no rho.
IV_PREFERRED_SOFT_PENALTY <- NA_real_
IV_POLICY_START_YEAR <- 2011L

IV_WAGE_ONLY_INSTRUMENT <- paste0(
  "z_oews_entropy_agwage_l1_k",
  IV_PREFERRED_K,
  "_d",
  IV_PREFERRED_DONOR_COUNT,
  "_g100"
)

IV_AUXILIARY_INSTRUMENT <- paste0(
  IV_WAGE_ONLY_INSTRUMENT,
  "_",
  IV_PREFERRED_WEIGHT_SPEC
)

IV_CONTROL_COLUMNS <- c(
  "ln_pop_census_l1",
  "farm_emp_share_l1",
  "emp_pop_ratio_l1",
  "wage_p10_l1"
)

IV_CLUSTER_ASSIGNMENT_COLUMN <- paste0(
  "iv_cluster_k",
  IV_PREFERRED_K
)
IV_INFERENCE_CLUSTER_COLUMN <- "cz_id"

make_aewr_iv_cluster_id <- function(aewr_region_id, iv_cluster) {
  ifelse(
    is.na(aewr_region_id) | is.na(iv_cluster),
    NA_character_,
    paste0(
      sprintf("%02d", as.integer(aewr_region_id)),
      "_",
      as.integer(iv_cluster)
    )
  )
}
