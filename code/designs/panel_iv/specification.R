# Plain helpers for the one canonical panel-IV outcome and model registry.

# docs-ground:start panel-iv-outcome-and-input-registry
panel_iv_read_specification <- function(path) {
  specification <- jsonlite::read_json(path, simplifyVector = TRUE)
  required <- c(
    "version", "policy_years", "endogenous", "instrument", "fixed_effects",
    "primary_inference_cluster", "controls", "models", "preferred_column",
    "primary_outcome", "derived_outcomes", "outcomes"
  )
  missing <- setdiff(required, names(specification))
  if (length(missing)) {
    stop("Panel-IV specification is missing: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  outcomes <- tibble::as_tibble(specification$outcomes)
  expected_columns <- c("outcome", "outcome_label", "table_stub", "adjustment_margin")
  if (!identical(names(outcomes), expected_columns) || nrow(outcomes) != 12L ||
      anyDuplicated(outcomes$outcome) || !specification$primary_outcome %in% outcomes$outcome) {
    stop("Panel-IV outcome registry must contain twelve unique declared outcomes.", call. = FALSE)
  }
  expected_years <- c(DISSIMILARITY_IV_POLICY_START_YEAR, DISSIMILARITY_IV_ESTIMATION_END_YEAR)
  if (!identical(as.integer(specification$policy_years), expected_years) ||
      !identical(as.character(specification$fixed_effects), c("county_fips", "year"))) {
    stop("Panel-IV timing or fixed effects changed without a design revision.", call. = FALSE)
  }
  if (!identical(specification$instrument, "z_wage_seasonal_composition_real") ||
      !identical(as.character(specification$controls), DISSIMILARITY_IV_CONTROL_TERMS)) {
    stop("Panel-IV instrument or controls changed without a design revision.", call. = FALSE)
  }
  models <- tibble::as_tibble(specification$models)
  if (!identical(names(models), c("column", "controls", "header", "file_stub")) ||
      !identical(as.integer(models$column), 1:2) ||
      !identical(as.logical(models$controls), c(FALSE, TRUE)) ||
      !identical(as.integer(specification$preferred_column), 2L)) {
    stop("Panel-IV model registry must contain the two ordered declared columns.", call. = FALSE)
  }
  specification$models <- models
  specification$outcomes <- outcomes
  specification
}

panel_iv_prepare_outcomes <- function(panel, specification) {
  definitions <- specification$derived_outcomes
  expected <- list(
    farm_cashandinc_ppi_per_farm_worker = list(
      numerator = "farm_cashandinc_ppi", denominator = "emp_farm", scale = 1000L,
      require_positive_denominator = TRUE
    ),
    h2a_linked_worksite_businesses_per_farm_worker_2011_start_year = list(
      numerator = "nbr_employers_balanced_start_year", denominator = "emp_farm_2011", scale = 1L,
      require_positive_denominator = TRUE
    )
  )
  if (!identical(definitions, expected)) {
    stop("Panel-IV derived-outcome definitions changed without executable support.", call. = FALSE)
  }
  dplyr::mutate(
    panel,
    year = as.integer(.data$year),
    farm_cashandinc_ppi_per_farm_worker = dplyr::if_else(
      is.finite(.data$farm_cashandinc_ppi) & is.finite(.data$emp_farm) & .data$emp_farm > 0,
      1000 * .data$farm_cashandinc_ppi / .data$emp_farm,
      NA_real_
    ),
    h2a_linked_worksite_businesses_per_farm_worker_2011_start_year = dplyr::if_else(
      is.finite(.data$nbr_employers_balanced_start_year) &
        is.finite(.data$emp_farm_2011) & .data$emp_farm_2011 > 0,
      .data$nbr_employers_balanced_start_year / .data$emp_farm_2011,
      NA_real_
    )
  )
}
# docs-ground:end panel-iv-outcome-and-input-registry
