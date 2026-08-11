# Purpose: Validate construction, estimability, postestimation algebra, and
# retained artifacts for the multilevel Mundlak-Chamberlain design.
# Inputs: all artifacts produced by scripts 01 through 05_generate_tables.
# Output: outputs/tables/mc_validation_summary.csv.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "mundlak_chamberlain", "design.R"))
source(path_code("designs", "mundlak_chamberlain", "helpers.R"))
library(arrow)
library(dplyr)
library(readr)
library(tibble)

required_artifacts <- c(
  path_processed("mundlak_chamberlain_county_year.parquet"),
  path_int("mundlak_chamberlain_metadata.rds"),
  path_int("mundlak_chamberlain_models.rds"),
  path_tables("mc_model_diagnostics.csv"),
  path_tables("mc_collinear_terms.csv"),
  path_tables("mc_model_warnings.csv"),
  path_tables("mc_parameter_estimates.csv"),
  path_tables("mc_ccv_diagnostics.csv"),
  path_tables("mc_finite_dose_effects.csv"),
  path_tables("mc_average_marginal_effects.csv"),
  path_tables("mc_year_effects.csv"),
  path_tables("mc_heterogeneity_effects.csv"),
  path_tables("mc_ame_grid.csv"),
  path_tables("mc_treatment_support.csv"),
  path_tables("mc_counterfactual_support.csv"),
  path_tables("mc_hierarchy_counts.csv"),
  path_tables("mc_identification_rank_audit.csv"),
  path_tables("mc_lead_placebo_effects.csv"),
  path_tables("table_mc_dynamic_effects.tex"),
  path_tables("table_mc_heterogeneity.tex"),
  path_tables("table_mc_support.tex"),
  path_tables("table_mc_lead_placebos.tex"),
  path_tables("table_mc_ccv_coefficients.tex"),
  path_tables("table_mc_dynamic_effects.html"),
  path_tables("table_mc_heterogeneity.html"),
  path_tables("table_mc_support.html"),
  path_tables("table_mc_lead_placebos.html"),
  path_tables("table_mc_ccv_coefficients.html"),
  path_figures("fig_mc_ccv_coefficients.png"),
  path_figures("fig_mc_dynamic_effects.png"),
  path_figures("fig_mc_year_effects.png"),
  path_figures("fig_mc_heterogeneity.png"),
  path_figures("fig_mc_treatment_support.png")
)
missing_artifacts <- required_artifacts[
  !file.exists(required_artifacts) |
    file.info(required_artifacts)$size <= 0
]
if (length(missing_artifacts) > 0L) {
  stop(
    "Missing or empty MC artifacts: ",
    paste(missing_artifacts, collapse = ", "),
    call. = FALSE
  )
}

panel <- read_parquet(
  path_processed("mundlak_chamberlain_county_year.parquet")
) %>%
  as.data.frame()
bundle <- readRDS(
  path_int("mundlak_chamberlain_models.rds")
)
metadata <- bundle$metadata
diagnostics <- read_csv(
  path_tables("mc_model_diagnostics.csv"),
  show_col_types = FALSE
)
ccv_diagnostics <- read_csv(
  path_tables("mc_ccv_diagnostics.csv"),
  show_col_types = FALSE
)
finite_effects <- read_csv(
  path_tables("mc_finite_dose_effects.csv"),
  show_col_types = FALSE
)
ame <- read_csv(
  path_tables("mc_average_marginal_effects.csv"),
  show_col_types = FALSE
)
ame_grid <- read_csv(
  path_tables("mc_ame_grid.csv"),
  show_col_types = FALSE
)
lead_placebos <- read_csv(
  path_tables("mc_lead_placebo_effects.csv"),
  show_col_types = FALSE
)

employer_specification <- metadata$outcomes %>%
  filter(outcome_id == "employers")
employer_finite_effects <- finite_effects %>%
  filter(outcome_id == "employers")
employer_ames <- ame %>%
  filter(outcome_id == "employers")
if (
  nrow(employer_specification) != 1L ||
    employer_specification$outcome_column[[1]] != "mc_y_employers" ||
    employer_specification$effect_unit[[1]] != "employers" ||
    !isTRUE(all(
      panel$mc_y_employers == panel$nbr_employers_balanced_start_year
    )) ||
    nrow(employer_finite_effects) == 0L ||
    nrow(employer_ames) == 0L ||
    any(employer_finite_effects$reported_unit != "employers") ||
    any(employer_ames$reported_unit != "employers")
) {
  stop("MC employer effects must remain on the raw count scale.", call. = FALSE)
}

if (
  nrow(panel) !=
    n_distinct(panel$county_fips) *
      length(metadata$analysis_years) ||
    anyDuplicated(panel[c("county_fips", "year")]) > 0L
) {
  stop("MC panel balance or uniqueness validation failed.", call. = FALSE)
}
if (
  n_distinct(panel$aewr_region_id) != 17L ||
    n_distinct(panel$state_fips) != 48L ||
    n_distinct(panel$mc_market_id) != 745L
) {
  stop("MC hierarchy counts changed unexpectedly.", call. = FALSE)
}
nesting_contract <- panel %>%
  distinct(
    county_fips,
    mc_market_id,
    state_fips,
    aewr_region_id
  )
if (
  any(
    nesting_contract %>%
      count(county_fips) %>%
      pull(n) != 1L
  ) ||
    any(
      nesting_contract %>%
        distinct(mc_market_id, state_fips) %>%
        count(mc_market_id) %>%
        pull(n) != 1L
    ) ||
    any(
      nesting_contract %>%
        distinct(state_fips, aewr_region_id) %>%
        count(state_fips) %>%
        pull(n) != 1L
    )
) {
  stop("MC geographic nesting validation failed.", call. = FALSE)
}

if (
  nrow(diagnostics) !=
    nrow(metadata$outcomes) * length(MC_MODEL_IDS) ||
    any(
      diagnostics$model_id == MC_PRIMARY_MODEL_ID &
        diagnostics$dropped_causal_terms != 0L
    )
) {
  stop("MC model-count or causal-rank validation failed.", call. = FALSE)
}
if (
  nrow(ccv_diagnostics) != nrow(diagnostics) ||
    any(ccv_diagnostics$method != MC_CCV_METHOD) ||
    any(ccv_diagnostics$reference_design != MC_CCV_REFERENCE_DESIGN) ||
    any(ccv_diagnostics$reference_states != MC_CCV_REFERENCE_STATES) ||
    any(ccv_diagnostics$design_df != MC_CCV_DF) ||
    any(ccv_diagnostics$covariance_rank > MC_CCV_DF) ||
    any(ccv_diagnostics$minimum_variance < -1e-10) ||
    any(abs(ccv_diagnostics$maximum_observed_state_error) > 1e-12) ||
    any(diagnostics$covariance_method != MC_CCV_METHOD)
) {
  stop("MC design-covariance CCV validation failed.", call. = FALSE)
}
if (
  any(!is.finite(finite_effects$estimate)) ||
    any(!is.finite(finite_effects$standard_error)) ||
    any(!is.finite(ame$estimate)) ||
    any(!is.finite(ame$standard_error)) ||
    any(!is.finite(ame_grid$estimate)) ||
    any(!is.finite(ame_grid$standard_error)) ||
    any(finite_effects$variance_method != MC_CCV_METHOD) ||
    any(ame$variance_method != MC_CCV_METHOD) ||
    any(ame_grid$variance_method != MC_CCV_METHOD)
) {
  stop("MC postestimation contains non-finite values.", call. = FALSE)
}

# Validate the exact delta-method gradient against the full formula matrix on
# a small sample that retains every year and AEWR-region factor level.
outcome_id <- "employers"
model <- bundle$models[[outcome_id]][[MC_PRIMARY_MODEL_ID]]
row_ids <-
  bundle$sample_row_ids[[outcome_id]][[MC_PRIMARY_MODEL_ID]]
estimation_data <- panel[
  match(row_ids, panel$mc_row_id),
  ,
  drop = FALSE
]
validation_data <- estimation_data %>%
  group_by(aewr_region_id, year) %>%
  slice_head(n = 1L) %>%
  ungroup() %>%
  as.data.frame()
analytic <- mc_master_sample_effect(
  model = model,
  data = validation_data,
  treatment_column = "mc_dose_current",
  dose_change = 5,
  normalize = TRUE,
  cluster_df = metadata$cluster_df
)

formula <- stats::as.formula(model$formula)
counterfactual_data <- mc_counterfactual_data(
  validation_data,
  "mc_dose_current",
  5
)
x0 <- stats::model.matrix(formula, data = validation_data)
x1 <- stats::model.matrix(formula, data = counterfactual_data)
coefficient_names <- names(stats::coef(model))
if (
  length(setdiff(coefficient_names, colnames(x0))) > 0L ||
    length(setdiff(coefficient_names, colnames(x1))) > 0L
) {
  stop("Validation model matrix lost estimated columns.", call. = FALSE)
}
direct_gradient <- colMeans(
  x1[, coefficient_names, drop = FALSE] -
    x0[, coefficient_names, drop = FALSE]
)
analytic_gradient <- attr(analytic, "gradient")
gradient_error <- max(
  abs(direct_gradient - analytic_gradient[coefficient_names])
)
direct_estimate <- sum(
  direct_gradient * stats::coef(model)[coefficient_names]
)
estimate_error <- abs(direct_estimate - analytic$estimate)
if (
  !is.finite(gradient_error) ||
    gradient_error > 1e-9 ||
    !is.finite(estimate_error) ||
    estimate_error > 1e-7
) {
  stop(
    "MC analytic delta gradient failed the formula-matrix check.",
    call. = FALSE
  )
}

finite_any <- finite_effects %>%
  filter(
    outcome_id == "any_application",
    dose_change == 5,
    standardization == "county_year_equal"
  )
lead_finite <- lead_placebos %>%
  filter(estimand == "finite_dose_change")

validation_summary <- tribble(
  ~check, ~status, ~value, ~detail,
  "Required artifacts",
  "pass",
  length(required_artifacts),
  "Every declared artifact exists and is nonempty.",
  "Balanced county-year panel",
  "pass",
  nrow(panel),
  paste(
    n_distinct(panel$county_fips),
    "counties observed in every analysis year."
  ),
  "Strict geographic nesting",
  "pass",
  n_distinct(panel$mc_market_id),
  "County < market < state < AEWR region is one-to-one upward.",
  "Independent treatment paths",
  "warning",
  n_distinct(panel$aewr_region_id),
  paste(
    "Only 17 AEWR paths; the finite-design CCV covariance rank is",
    "at most 16."
  ),
  "Design-covariance CCV",
  "pass",
  nrow(ccv_diagnostics),
  paste(
    "Every model uses the balanced 17-state reference law; covariance",
    "matrices are PSD cross-products and have rank at most 16."
  ),
  "Primary causal columns retained",
  "pass",
  sum(
    diagnostics$model_id == MC_PRIMARY_MODEL_ID &
      diagnostics$dropped_causal_terms == 0L
  ),
  "Every primary outcome retains every current/lag causal basis column.",
  "Raw employer-count scale",
  "pass",
  nrow(employer_finite_effects) + nrow(employer_ames),
  paste(
    "The modeled outcome equals the balanced-linkage count, and every",
    "employer effect is reported in employers."
  ),
  "Employer-outcome analytic delta gradient",
  "pass",
  gradient_error,
  paste(
    "Employer-effect named-parameter gradient equals the direct",
    "formula-matrix gradient."
  ),
  "Finite postestimation",
  "pass",
  nrow(finite_effects) + nrow(ame) + nrow(ame_grid),
  "All retained estimates and standard errors are finite.",
  "Linear-probability bounds",
  if (any(abs(finite_any$estimate) > 100)) "warning" else "pass",
  max(abs(finite_any$estimate)),
  paste(
    "A five-point probability effect outside [-100,100] signals",
    "functional-form extrapolation and weak support."
  ),
  "Future-dose placebos",
  if (any(lead_finite$p_value < 0.05)) "warning" else "pass",
  sum(lead_finite$p_value < 0.05),
  paste(
    "Number of eight outcomes rejecting a zero five-point lead",
    "effect at five percent."
  )
)
write_csv(
  validation_summary,
  path_tables("mc_validation_summary.csv")
)

message(
  "Validated MC design with ",
  sum(validation_summary$status == "pass"),
  " passing contracts and ",
  sum(validation_summary$status == "warning"),
  " substantive warnings."
)
