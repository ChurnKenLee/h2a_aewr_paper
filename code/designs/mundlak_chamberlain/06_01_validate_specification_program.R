# Purpose: Enforce the version-3 Mundlak-Chamberlain specification-program
# contract after estimation, promotion, diagnostics, and rendering.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "mundlak_chamberlain", "design.R"))
source(path_code("designs", "mundlak_chamberlain", "helpers.R"))
source(
  path_code(
    "designs",
    "mundlak_chamberlain",
    "specification_program.R"
  )
)
library(dplyr)
library(readr)
library(tidyr)

required_artifacts <- c(
  path_int("mundlak_chamberlain_specification_registry.rds"),
  path_int("mundlak_chamberlain_execution_registry.rds"),
  path_tables("mc_specification_registry.csv"),
  path_tables("mc_default_execution_registry.csv"),
  path_tables("mc_execution_registry.csv"),
  path_tables("mc_rank_budget_audit.csv"),
  path_tables("mc_specification_model_diagnostics.csv"),
  path_tables("mc_specification_effects.csv"),
  path_tables("mc_reference_state_influence.csv"),
  path_tables("mc_common_basis_audit.csv"),
  path_tables("mc_resolved_history_audit.csv"),
  path_tables("mc_primary_specification_selection.csv"),
  path_tables("mc_finite_dose_effects.csv"),
  path_tables("mc_average_marginal_effects.csv"),
  path_tables("mc_year_effects.csv"),
  path_tables("mc_heterogeneity_effects.csv"),
  path_tables("mc_delta_gradient_audit.csv"),
  path_tables("mc_specification_curve_data.csv"),
  path_tables("mc_moderator_alignment.csv"),
  path_tables("mc_treatment_support.csv"),
  path_tables("mc_counterfactual_support.csv"),
  path_tables("mc_hierarchy_counts.csv"),
  path_tables("mc_lead_placebo_diagnostics.csv"),
  path_tables("mc_lead_placebo_effects.csv"),
  path_tables("mc_parameter_estimates.csv"),
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
  path_figures("fig_mc_specification_curve.png"),
  path_figures("fig_mc_treatment_support.png")
)
missing_artifacts <- required_artifacts[
  !file.exists(required_artifacts) |
    file.info(required_artifacts)$size <= 0
]
if (length(missing_artifacts) > 0L) {
  stop(
    "Missing or empty specification-program artifacts: ",
    paste(missing_artifacts, collapse = ", "),
    call. = FALSE
  )
}

registry_bundle <- readRDS(
  path_int("mundlak_chamberlain_specification_registry.rds")
)
execution_bundle <- readRDS(
  path_int("mundlak_chamberlain_execution_registry.rds")
)
registry <- registry_bundle$registry
calendars <- registry_bundle$calendars
execution_registry <- execution_bundle$registry
budget <- read_csv(
  path_tables("mc_rank_budget_audit.csv"),
  show_col_types = FALSE
)
diagnostics <- read_csv(
  path_tables("mc_specification_model_diagnostics.csv"),
  show_col_types = FALSE
)
effects <- read_csv(
  path_tables("mc_specification_effects.csv"),
  show_col_types = FALSE
)
influence <- read_csv(
  path_tables("mc_reference_state_influence.csv"),
  show_col_types = FALSE
)
common_basis <- read_csv(
  path_tables("mc_common_basis_audit.csv"),
  show_col_types = FALSE
)
primary <- read_csv(
  path_tables("mc_primary_specification_selection.csv"),
  show_col_types = FALSE
)
finite_effects <- read_csv(
  path_tables("mc_finite_dose_effects.csv"),
  show_col_types = FALSE
)
ames <- read_csv(
  path_tables("mc_average_marginal_effects.csv"),
  show_col_types = FALSE
)
gradient_audit <- read_csv(
  path_tables("mc_delta_gradient_audit.csv"),
  show_col_types = FALSE
)
lead_diagnostics <- read_csv(
  path_tables("mc_lead_placebo_diagnostics.csv"),
  show_col_types = FALSE
)
lead_effects <- read_csv(
  path_tables("mc_lead_placebo_effects.csv"),
  show_col_types = FALSE
)

if (
  !identical(registry_bundle$design_version, MC_SPEC_PROGRAM_VERSION) ||
    nrow(calendars) != 54L ||
    nrow(registry) != 648L ||
    anyDuplicated(registry$spec_id) > 0L ||
    sum(registry$primary_target) != 1L ||
    !identical(
      unname(as.integer(table(calendars$horizon_count))),
      rep(18L, 3L)
    )
) {
  stop("Specification registry contract failed.", call. = FALSE)
}

if (
  any(budget$region_coordinates > MC_SPEC_REGION_BUDGET) ||
    any(budget$status != "budget_admissible") ||
    max(budget$region_coordinates) != MC_SPEC_REGION_BUDGET
) {
  stop("Region-level coordinate budget failed.", call. = FALSE)
}

expected_grid_rows <-
  nrow(execution_registry) * length(execution_bundle$outcomes)
if (
  nrow(diagnostics) != expected_grid_rows ||
    anyDuplicated(diagnostics[c("spec_id", "outcome_id")]) > 0L ||
    any(!diagnostics$spec_id %in% execution_registry$spec_id) ||
    any(!diagnostics$outcome_id %in% execution_bundle$outcomes) ||
    any(!diagnostics$status %in% c("estimated", "rejected")) ||
    any(
      diagnostics$status == "estimated" &
        diagnostics$dropped_causal_terms != 0L
    ) ||
    any(
      diagnostics$status == "estimated" &
        diagnostics$parameter_row_ratio >
          diagnostics$parameter_row_guard + 1e-12
    )
) {
  stop("Grid completion, rank, or row-guard contract failed.", call. = FALSE)
}
resource_columns <- c(
  "dense_matrix_gib",
  "estimated_peak_gib",
  "dense_matrix_guard_gib",
  "estimated_peak_guard_gib"
)
if (
  all(resource_columns %in% names(diagnostics)) &&
    any(
      diagnostics$status == "estimated" &
        (
          diagnostics$dense_matrix_gib >
            diagnostics$dense_matrix_guard_gib + 1e-12 |
            diagnostics$estimated_peak_gib >
              diagnostics$estimated_peak_guard_gib + 1e-12
        ),
      na.rm = TRUE
    )
) {
  stop("An estimated model exceeded its computational memory guard.", call. = FALSE)
}

estimated <- diagnostics |>
  filter(.data$status == "estimated")
if (
  nrow(common_basis) != nrow(estimated) * (MC_CCV_REFERENCE_STATES - 1L) ||
    any(!common_basis$common_basis) ||
    any(common_basis$missing_coefficient_count != 0L) ||
    any(estimated$ccv_covariance_rank > MC_CCV_DF) ||
    any(abs(estimated$maximum_observed_state_error) > 1e-12) ||
    any(abs(
      estimated$df_multiplier -
        estimated$observations /
          (estimated$observations - estimated$effective_parameters)
    ) > 1e-10)
) {
  stop("Common-basis or adjusted-dcCCV contract failed.", call. = FALSE)
}

if (
  nrow(primary) != nrow(MC_OUTCOMES) ||
    anyDuplicated(primary$outcome_id) > 0L ||
    any(!primary$outcome_id %in% MC_OUTCOMES$outcome_id)
) {
  stop("Primary selection must contain one record per outcome.", call. = FALSE)
}
for (outcome_id in MC_OUTCOMES$outcome_id) {
  selected <- primary[primary$outcome_id == outcome_id, , drop = FALSE]
  admissible <- diagnostics |>
    inner_join(
      registry |>
        filter(
          .data$horizon_count == 3L,
          .data$preperiod_start == min(MC_BASELINE_YEARS),
          .data$preperiod_end == max(MC_BASELINE_YEARS),
          .data$analysis_start == min(MC_ANALYSIS_YEARS),
          .data$analysis_end == max(MC_ANALYSIS_YEARS),
          .data$polynomial_degree == 2L
        ) |>
        select(.data$spec_id, .data$richness_tier),
      by = c("spec_id", "richness_tier")
    ) |>
    filter(
      .data$outcome_id == outcome_id,
      .data$status == "estimated"
    )
  if (
    nrow(admissible) == 0L ||
      selected$richness_tier[[1]] != max(admissible$richness_tier)
  ) {
    stop("Primary is not the richest admissible tier for ", outcome_id, ".")
  }
}

# Inspect the target panel dictionary once.  The bite channel must enter as a
# level rather than as a decomposition whose region component would silently
# consume a region-year coordinate.  R3 declares all 12 choose 2 pairs plus
# 12 squares, even when a sample-invariant component is later inactive.
target_calendar <- unique(primary$calendar_id)
if (length(target_calendar) != 1L) {
  stop("Primary calendars differ across outcomes.", call. = FALSE)
}
panel_bundle <- readRDS(file.path(
  path_int(),
  "mundlak_chamberlain_specification_panels",
  paste0(target_calendar, ".rds")
))
metadata <- panel_bundle$metadata
if (
  nrow(metadata$quadratic_lookup) != 78L ||
    anyDuplicated(metadata$quadratic_lookup$constructed_column) > 0L ||
    any(grepl("^mc_r_", c(
      metadata$mean_causal_columns,
      metadata$trajectory_causal_columns,
      metadata$bite_three_way_columns
    ))) ||
    metadata$bite_mean_column != "mc_z" ||
    !identical(
      panel_bundle$panel$mc_y_employers,
      panel_bundle$panel$nbr_employers_balanced_start_year
    )
) {
  stop("Moderator dictionary or raw employer outcome contract failed.", call. = FALSE)
}

finite_columns <- c(
  "estimate", "raw_standard_error", "standard_error",
  "conf_low", "conf_high", "df_multiplier"
)
if (
  any(!is.finite(as.matrix(effects[, finite_columns]))) ||
    any(effects$standard_error + 1e-12 < effects$raw_standard_error) ||
    any(abs(
      effects$standard_error /
        pmax(effects$raw_standard_error, .Machine$double.eps) -
        sqrt(effects$df_multiplier)
    ) > 1e-7 & effects$raw_standard_error > 1e-12) ||
    any(!influence$omitted_state %in% 0:(MC_CCV_REFERENCE_STATES - 1L)) ||
    any(abs(
      influence$critical_value -
        stats::qt(0.975, df = MC_CCV_DF - 1L)
    ) > 1e-12)
) {
  stop("Effect projection, df adjustment, or leave-one-out contract failed.", call. = FALSE)
}

if (
  nrow(gradient_audit) != nrow(MC_OUTCOMES) ||
    any(gradient_audit$unresolved_causal_terms != 0L) ||
    any(gradient_audit$unexpected_causal_terms != 0L) ||
    any(
      gradient_audit$maximum_gradient_error >
        MC_SPEC_GRADIENT_TOLERANCE
    ) ||
    any(gradient_audit$estimate_error > 1e-7)
) {
  stop("Named delta-method gradient contract failed.", call. = FALSE)
}

employer_finite <- finite_effects |>
  filter(.data$outcome_id == "employers")
employer_ames <- ames |>
  filter(.data$outcome_id == "employers")
employer_totals <- employer_finite |>
  filter(.data$standardization == "sample_period_total")
if (
  nrow(employer_finite) == 0L ||
    nrow(employer_ames) == 0L ||
    any(employer_finite$reported_unit != "employers") ||
    any(employer_ames$reported_unit != "employers") ||
    any(employer_finite$standardization == "farm_employment_weighted") ||
    any(employer_ames$standardization == "farm_employment_weighted") ||
    nrow(employer_totals) == 0L ||
    any(employer_totals$normalized) ||
    any(employer_totals$weight_sum != employer_totals$observations)
) {
  stop("Employer effects were scaled by farm employment.", call. = FALSE)
}

if (
  nrow(lead_diagnostics) != nrow(MC_OUTCOMES) ||
    any(lead_diagnostics$status != "estimated") ||
    any(lead_diagnostics$dropped_causal_terms != 0L) ||
    nrow(lead_effects) != nrow(MC_OUTCOMES) ||
    any(lead_effects$treatment_column != "mc_dose_lead1") ||
    any(lead_effects$dose_change != 5) ||
    any(!is.finite(lead_effects$p_value))
) {
  stop("Selected-primary lead-placebo contract failed.", call. = FALSE)
}

validation_summary <- data.frame(
  check = c(
    "Specification records",
    "Resource-bounded execution registry",
    "Region-coordinate budgets",
    "Completed outcome grid",
    "Common 17-state coefficient basis",
    "Richest admissible primary",
    "Named delta-method gradients",
    "Raw employer-count scale",
    "Adjusted dcCCV and leave-one-out",
    "Selected-primary lead placebos"
  ),
  status = c(
    rep("pass", 9L),
    if (any(lead_effects$p_value < 0.05)) "warning" else "pass"
  ),
  value = c(
    nrow(registry),
    nrow(execution_registry),
    max(budget$region_coordinates),
    nrow(diagnostics),
    nrow(common_basis),
    nrow(primary),
    max(gradient_audit$maximum_gradient_error),
    nrow(employer_finite) + nrow(employer_ames),
    max(estimated$df_multiplier),
    sum(lead_effects$p_value < 0.05)
  ),
  detail = c(
    "648 records over 54 calendars; exactly one predeclared target.",
    paste0(
      "Stage ", execution_bundle$execution_stage,
      " runs the declared subset with ", execution_bundle$workers,
      " worker(s) and ", execution_bundle$fixest_threads,
      " fixest thread(s)."
    ),
    "Every year-specific assignment block uses at most 16 coordinates.",
    "Every specification-outcome pair was estimated or guard-rejected.",
    "Every estimated model retains its basis in all 16 nonobserved states.",
    "Each outcome promotes the richest admissible H3-D2 target tier.",
    "Formula-matrix and analytical delta gradients agree to tolerance.",
    "Employer counts and totals use unit weights, never farm employment.",
    "Headline CCV uses N/(N-K); raw CCV and t(15) LOO are retained.",
    "Count of outcome-specific five-point lead effects significant at 5%."
  ),
  stringsAsFactors = FALSE
)
write_csv(
  validation_summary,
  path_tables("mc_validation_summary.csv")
)

message(
  "Validated the version-3 MC specification program: ",
  sum(validation_summary$status == "pass"),
  " passes and ",
  sum(validation_summary$status == "warning"),
  " substantive warnings."
)
