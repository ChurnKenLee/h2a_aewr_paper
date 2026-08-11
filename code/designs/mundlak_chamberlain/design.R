# Frozen design contract for the multilevel Mundlak-Chamberlain-Wooldridge
# branch.
#
# The branch treats annual AEWR growth as a quantitative panel treatment.
# Treatment assignment may be correlated with persistent heterogeneity,
# observed histories, and heterogeneous treatment slopes.  Those correlations
# are parameterized directly through multilevel Mundlak summaries,
# Chamberlain pre-period histories, and their interactions with the treatment
# path.  Identification rests on the resulting conditional parallel-trends
# restriction, not on random assignment of AEWR changes.

MC_BASELINE_YEARS <- 2008:2010
MC_TREATMENT_HISTORY_YEARS <- 2011:2022
MC_ANALYSIS_YEARS <- 2013:2022
MC_REFERENCE_YEAR <- 2013L
MC_LEGACY_DESIGN_VERSION <- "2.3.0"
MC_DESIGN_VERSION <- "3.0.0"

# Version 3 is a specification program.  The constants above remain the
# calendar of the frozen version-2.3 compatibility record and the target
# calendar for the version-3 primary model.  All grid models read their
# calendar and causal dictionary from an explicit specification record.
MC_SPEC_PROGRAM_VERSION <- "3.0.0"
MC_SPEC_HISTORY_YEARS <- 2011:2022
MC_SPEC_ANALYSIS_END <- 2022L
MC_SPEC_PREPERIOD_LENGTHS <- 2:4
MC_SPEC_ANALYSIS_START_DELAYS <- 0:2
MC_SPEC_POLYNOMIAL_DEGREES <- 1:3
MC_SPEC_RICHNESS_TIERS <- 0:3
MC_SPEC_REGION_BUDGET <- 16L
# The per-year ledger is not the whole rank calculation once region and year
# effects are included jointly.  Six additional region-time coordinates are
# held out before the all-state basis audit; this reproduces the slack in the
# identified version-2.3 block without privileging a reference year.
MC_SPEC_GLOBAL_REGION_RESERVE <- 6L
MC_SPEC_FULL_SAMPLE_PARAMETER_ROW_MAX <- 0.25
MC_SPEC_RESTRICTED_SAMPLE_PARAMETER_ROW_MAX <- 0.15
MC_SPEC_DF_ADJUSTMENT <- "N_over_N_minus_K"
# Resource defaults are intentionally conservative.  One outcome process may
# still use several fixest threads, so two forked workers can otherwise occupy
# every logical CPU while duplicating multi-gigabyte design matrices.
MC_SPEC_DEFAULT_STAGE <- "compact"
MC_SPEC_DEFAULT_WORKERS <- 1L
MC_SPEC_DEFAULT_FIXEST_THREADS <- 4L
MC_SPEC_MAX_DENSE_MATRIX_GIB <- 1.25
MC_SPEC_MAX_ESTIMATED_PEAK_GIB <- 6
MC_SPEC_DENSE_PEAK_COPIES <- 4
MC_SPEC_GRAM_PEAK_COPIES <- 3
MC_SPEC_GRADIENT_TOLERANCE <- 1e-9

MC_SPEC_RICHNESS_LABELS <- c(
  "bite_only",
  "county_market_means",
  "county_market_trajectories",
  "bite_three_way_and_county_quadratic"
)

# The CCV reference design below has one state for each of the 17 AEWR-region
# treatment paths.  We use t_(17 - 1) critical values because the independent
# policy-path count, rather than the county-year row count, governs inference.
MC_CCV_REFERENCE_STATES <- 17L
MC_CCV_DF <- MC_CCV_REFERENCE_STATES - 1L
MC_CLUSTER_DF <- MC_CCV_DF
MC_CCV_METHOD <- "finite_design_covariance_ccv"
MC_CCV_REFERENCE_DESIGN <- "balanced_cyclic_aewr_path_assignment"
MC_PRIMARY_MODEL_ID <- "chamberlain_rich"

# A change is measured in log percentage points: 100 * Delta log(AEWR).
MC_COUNTERFACTUAL_DOSES <- c(1, 5, 10)
MC_LAG_ORDERS <- 0:2

# The master specification uses a quadratic dose response in every
# horizon-year cell.  With only 17 AEWR assignment regions, a cubic in every
# cell cannot coexist with the full leave-focal-out treatment-history vector:
# it would put more region-level columns than treatment cells into each year.
# Cubic terms remain in the panel for transparent sensitivity analysis.
MC_MASTER_POLYNOMIAL_DEGREES <- 1:2
MC_LITERAL_POLYNOMIAL_DEGREES <- 1:3

# Z_i is the standardized 2008-2010 mean local AEWR bite.  It is measured
# before the 2011-2022 treatment history and directly represents how binding
# the federal wage floor is for a county's local low-wage labor market.
MC_Z_VARIABLE <- "aewr_bite"
MC_Z_COLUMN <- "mc_z"
MC_Z_LABEL <- "Baseline AEWR bite (standard deviations)"
MC_DYNAMIC_HORIZONS <- c(
  contemporaneous = "mc_dose_current",
  one_year = "mc_dose_lag1",
  two_year = "mc_dose_lag2"
)

MC_TREATMENT_BASIS_TERMS <- c(
  unname(MC_DYNAMIC_HORIZONS),
  "mc_dose_current_sq",
  "mc_dose_current_cu",
  "mc_dose_lag1_sq",
  "mc_dose_lag1_cu",
  "mc_dose_lag2_sq",
  "mc_dose_lag2_cu",
  "mc_dose_current_x_lag1",
  "mc_dose_current_x_lag2",
  "mc_dose_lag1_x_lag2",
  "mc_dose_current_x_lag1_x_lag2"
)

# The state x CZ x AEWR-region cell creates a strict hierarchy:
# AEWR region > state > local market cell > county.
MC_HIERARCHY_LEVELS <- c("county", "market", "state", "region")

# Names are stable design names; values are columns in the shared panel.
MC_BASELINE_VARIABLES <- c(
  h2a_cert_intensity =
    "h2a_cert_share_farm_workers_2011_start_year",
  h2a_application_intensity =
    "h2a_applications_per_farm_worker_2011_start_year",
  aewr_bite = "aewr_cz_p25",
  log_population = "ln_pop_census",
  farm_employment_share = "farm_emp_share",
  employment_population_ratio = "emp_pop_ratio",
  crop_income_share = "share_farm_crop_cashandinc",
  animal_income_share = "share_farm_animal_cashandinc",
  hired_labor_cost_share = "share_farm_laborexp_prodexp",
  low_wage = "wage_p25",
  cropland = "census_cropland_2007",
  predicted_h2a_intensity = "h2a_predicted_share_2011"
)

# Chamberlain histories retain the separate 2008, 2009, and 2010 values for
# variables most directly related to selection into H-2A use, the local
# binding margin, farm structure, and untreated outcome paths.
MC_CHAMBERLAIN_VARIABLES <- c(
  "h2a_cert_intensity",
  "h2a_application_intensity",
  "aewr_bite",
  "log_population",
  "farm_employment_share",
  "employment_population_ratio",
  "crop_income_share",
  "hired_labor_cost_share",
  "low_wage",
  "predicted_h2a_intensity"
)

# These variables receive unrestricted calendar-year interactions in the
# untreated mean.  The list deliberately includes every principal moderator
# of the AEWR response.
MC_UNTREATED_TREND_VARIABLES <- c(
  "h2a_cert_intensity",
  "h2a_application_intensity",
  "aewr_bite",
  "log_population",
  "farm_employment_share",
  "crop_income_share",
  "hired_labor_cost_share",
  "low_wage",
  "predicted_h2a_intensity"
)

# Lagged treatment effects are allowed to vary with the most important
# predetermined selection and binding variables.  The contemporaneous slope
# receives the complete moderator set constructed by the build script.
MC_DYNAMIC_SLOPE_VARIABLES <- c(
  "h2a_cert_intensity",
  "aewr_bite",
  "farm_employment_share",
  "crop_income_share"
)

MC_OUTCOMES <- data.frame(
  outcome_id = c(
    "applications",
    "employers",
    "certified_positions",
    "certified_hours",
    "any_application",
    "positions_per_application",
    "hours_per_position"
  ),
  outcome_column = c(
    "mc_y_applications_per_1000",
    "mc_y_employers",
    "mc_y_certified_positions_per_1000",
    "mc_y_certified_hours_per_worker",
    "mc_y_any_application",
    "mc_y_positions_per_application",
    "mc_y_hours_per_position"
  ),
  outcome_label = c(
    "H-2A applications per 1,000 baseline farm workers",
    "H-2A employers (balanced linkage)",
    "H-2A certified positions per 1,000 baseline farm workers",
    "H-2A certified hours per baseline farm worker",
    "Any H-2A application",
    "Certified positions per application",
    "Certified hours per certified position"
  ),
  family = rep("gaussian", 7L),
  offset_column = rep(NA_character_, 7L),
  sample_rule = c(
    "all",
    "all",
    "all",
    "all",
    "all",
    "positive_applications",
    "positive_certified_positions"
  ),
  effect_unit = c(
    "applications_per_1000",
    "employers",
    "certified_positions_per_1000",
    "certified_hours_per_worker",
    "probability",
    "positions_per_application",
    "hours_per_position"
  ),
  primary_total = c(TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE),
  stringsAsFactors = FALSE
)

MC_VOLUME_OUTCOME_IDS <- MC_OUTCOMES$outcome_id[MC_OUTCOMES$primary_total]
MC_FARM_EMPLOYMENT_SCALED_OUTCOME_IDS <- setdiff(
  MC_VOLUME_OUTCOME_IDS,
  "employers"
)

MC_MODEL_IDS <- c(
  "twfe_benchmark",
  "mundlak_multilevel",
  "chamberlain_rich",
  "chamberlain_lead_test"
)

mc_make_market_id <- function(aewr_region_id, state_fips, cz_id) {
  ifelse(
    is.na(aewr_region_id) | is.na(state_fips) | is.na(cz_id),
    NA_character_,
    paste(
      sprintf("%02d", as.integer(aewr_region_id)),
      sprintf("%02d", as.integer(state_fips)),
      as.character(cz_id),
      sep = "_"
    )
  )
}
