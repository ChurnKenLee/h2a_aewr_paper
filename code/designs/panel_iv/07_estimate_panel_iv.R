# Purpose: Estimate the publication first stages, 2SLS outcomes, summary
# statistics, and email-ready transposed results on the county-year panel.
# Input: data/processed/panel_iv_county_year.parquet.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "panel_iv", "design.R"))
library(arrow)
library(dplyr)
library(fixest)
library(purrr)
library(readr)
library(stringr)
library(tibble)

dir.create(path_tables(), recursive = TRUE, showWarnings = FALSE)

obsolete_outputs <- c(
  path_tables("iv_dissimilarity_ar_intervals.csv"),
  path_tables("iv_dissimilarity_model_estimates.csv"),
  path_tables("table_iv_dissimilarity_panel.tex"),
  path_figures("fig_iv_dissimilarity_first_stage.png")
)
unlink(obsolete_outputs[file.exists(obsolete_outputs)])

# docs-ground:start panel-iv-outcome-and-input-registry
panel <- read_parquet(
  path_processed("panel_iv_county_year.parquet")
) %>%
  mutate(
    year = as.integer(year),
    farm_cashandinc_ppi_per_farm_worker = if_else(
      is.finite(farm_cashandinc_ppi) &
        is.finite(emp_farm) &
        emp_farm > 0,
      1000 * farm_cashandinc_ppi / emp_farm,
      NA_real_
    ),
    h2a_employers_per_farm_worker_2011_start_year = if_else(
      is.finite(nbr_employers_balanced_start_year) &
        is.finite(emp_farm_2011) &
        emp_farm_2011 > 0,
      nbr_employers_balanced_start_year / emp_farm_2011,
      NA_real_
    )
  )

endogenous <- "aewr_ppi"
fixed_effects <- "county_fips + year"
cluster_vcov <- ~aewr_iv_cluster_id
control_terms <- paste(
  DISSIMILARITY_IV_CONTROL_TERMS,
  collapse = " + "
)

outcomes <- tribble(
  ~outcome                                                             , ~outcome_label , ~table_stub ,
  "h2a_cert_share_farm_workers_2011_start_year"                        ,
  "H-2A certified workers / 2011 farm employment"                      ,
  "h2a_normalized"                                                     ,
  "h2a_cert_hours_per_farm_worker_2011_start_year"                     ,
  "H-2A certified hours / 2011 farm employment"                        ,
  "h2a_certified_hours"                                                ,
  "h2a_applications_per_farm_worker_2011_start_year"                   ,
  "H-2A applications / 2011 farm employment"                           ,
  "h2a_applications"                                                   ,
  "h2a_employers_per_farm_worker_2011_start_year"                      ,
  "H-2A employers / 2011 farm employment (balanced linkage)"          ,
  "h2a_employers"                                                      ,
  "h2a_cert_positions_per_application_start_year"                      ,
  "H-2A certified positions / application"                             ,
  "h2a_positions_per_application"                                      ,
  "h2a_cert_hours_per_position_start_year"                             ,
  "H-2A certified hours / certified position"                          ,
  "h2a_hours_per_position"                                             ,
  "fisher_index_ppi"                                                   ,
  "Real Fisher crop price index"                                       ,
  "prices"                                                             ,
  "emp_farm"                                                           ,
  "Farm employment"                                                    ,
  "farm_employment"                                                    ,
  "share_farm_prodexp_cashandinc"                                      ,
  "Farm production expenses / cash receipts and other income"          ,
  "production_expense_share"                                           ,
  "farm_cashandinc_ppi_per_farm_worker"                                ,
  "Farm cash receipts and other income per farm worker (2012 dollars)" ,
  "farm_income"                                                        ,
  "share_farm_laborexp_prodexp"                                        ,
  "Hired-labor share of farm production expenses"                      ,
  "farm_labor_share"                                                   ,
  "fisher_quantity_index"                                              ,
  "Fisher crop output-quantity index (2011 = 100)"                     ,
  "output_quantities"
)

h2a_adjustment_outcomes <- c(
  "h2a_cert_hours_per_farm_worker_2011_start_year",
  "h2a_applications_per_farm_worker_2011_start_year",
  "h2a_cert_positions_per_application_start_year",
  "h2a_cert_hours_per_position_start_year"
)

required_columns <- unique(c(
  "county_fips",
  "year",
  "aewr_region_id",
  "aewr_iv_cluster_id",
  "h2a_prediction_cutoff_year",
  "h2a_prediction_model_spec",
  "h2a_predicted_share_2011",
  "h2a_ppml_static_propensity_z",
  "year_centered",
  endogenous,
  "z_wage_only_real",
  "z_wage_seasonal_composition_real",
  DISSIMILARITY_IV_CONTROL_COLUMNS,
  outcomes$outcome
))
missing_columns <- setdiff(required_columns, names(panel))
if (length(missing_columns) > 0L) {
  stop(
    "County IV panel is missing: ",
    paste(missing_columns, collapse = ", "),
    call. = FALSE
  )
}
# docs-ground:end panel-iv-outcome-and-input-registry

prediction_cutoffs <- sort(unique(
  panel$h2a_prediction_cutoff_year[
    !is.na(panel$h2a_predicted_share_2011)
  ]
))
if (!identical(
  as.integer(prediction_cutoffs),
  H2A_PREDICTION_CUTOFF_YEAR
)) {
  stop(
    "Panel-IV controls must use only the canonical static H-2A prediction.",
    call. = FALSE
  )
}
prediction_model_specs <- unique(
  panel$h2a_prediction_model_spec[
    !is.na(panel$h2a_predicted_share_2011)
  ]
)
if (!identical(
  prediction_model_specs,
  H2A_PREDICTION_MODEL_SPEC
)) {
  stop("Panel-IV controls have an incompatible H-2A model spec.", call. = FALSE)
}

propensity_contract <- panel %>%
  filter(!is.na(h2a_ppml_static_propensity_z)) %>%
  distinct(
    county_fips,
    h2a_predicted_share_2011,
    h2a_ppml_static_propensity_z
  )
if (
  nrow(propensity_contract) == 0L ||
    anyDuplicated(propensity_contract$county_fips) > 0L ||
    any(!is.finite(propensity_contract$h2a_ppml_static_propensity_z)) ||
    abs(mean(propensity_contract$h2a_ppml_static_propensity_z)) > 1e-12 ||
    abs(sd(propensity_contract$h2a_ppml_static_propensity_z) - 1) > 1e-12 ||
    any(panel$year_centered != panel$year - 2011L)
) {
  stop(
    "Panel-IV static PPML propensities violate the equal-county trend contract.",
    call. = FALSE
  )
}

finite_complete <- function(data, numeric_columns, id_columns) {
  data %>%
    filter(
      year >= DISSIMILARITY_IV_POLICY_START_YEAR,
      year <= DISSIMILARITY_IV_POLICY_END_YEAR,
      if_all(
        all_of(numeric_columns),
        ~ !is.na(.x) & is.finite(.x)
      ),
      if_all(all_of(id_columns), ~ !is.na(.x))
    )
}

coefficient_row <- function(model, pattern) {
  table <- coeftable(model)
  selected <- grep(pattern, rownames(table), value = TRUE)
  if (length(selected) != 1L) {
    stop(
      "Expected one coefficient matching ",
      pattern,
      "; found: ",
      paste(selected, collapse = ", "),
      call. = FALSE
    )
  }
  row <- table[selected, , drop = FALSE]
  tibble(
    coefficient_name = selected,
    estimate = unname(row[1, "Estimate"]),
    standard_error = unname(row[1, "Std. Error"]),
    t_statistic = unname(row[1, "t value"]),
    p_value = unname(row[1, "Pr(>|t|)"])
  )
}

model_r2 <- function(model, type = "r2") {
  result <- tryCatch(
    as.numeric(r2(model, type)),
    error = function(error) NA_real_
  )
  if (length(result) == 0L) {
    return(NA_real_)
  }
  result[[1]]
}

format_number <- function(value, digits = 2L) {
  ifelse(
    is.na(value),
    "",
    formatC(
      value,
      format = "f",
      digits = digits,
      big.mark = ","
    )
  )
}

# Four-column first stage ----------------------------------------------------

first_stage_data <- finite_complete(
  panel,
  c(
    endogenous,
    "z_wage_only_real",
    "z_wage_seasonal_composition_real",
    DISSIMILARITY_IV_CONTROL_COLUMNS
  ),
  c("county_fips", "year", "aewr_iv_cluster_id")
)

first_stage_specs <- tribble(
  ~column                   , ~instrument , ~moment_targets , ~controls , ~header ,
  1L                        ,
  "z_wage_only_real"        ,
  "Wage only"               ,
  FALSE                     ,
  "Basic"                   ,
  2L                        ,
  "z_wage_seasonal_composition_real",
  "Wage + seasonal/composition",
  FALSE                     ,
  "Basic + seasonal/composition",
  3L                        ,
  "z_wage_only_real"        ,
  "Wage only"               ,
  TRUE                      ,
  "Controls"                ,
  4L                        ,
  "z_wage_seasonal_composition_real",
  "Wage + seasonal/composition",
  TRUE                      ,
  "Controls + seasonal/composition"
)

first_stage_models <- vector("list", nrow(first_stage_specs))
first_stage_rows <- vector("list", nrow(first_stage_specs))

for (index in seq_len(nrow(first_stage_specs))) {
  specification <- first_stage_specs[index, ]
  model_data <- first_stage_data %>%
    mutate(z = .data[[specification$instrument[[1]]]])
  rhs <- if (specification$controls[[1]]) {
    paste("z", control_terms, sep = " + ")
  } else {
    "z"
  }
  model <- feols(
    as.formula(paste(
      endogenous,
      "~",
      rhs,
      "|",
      fixed_effects
    )),
    data = model_data,
    vcov = cluster_vcov,
    warn = FALSE,
    notes = FALSE
  )
  estimation_data <- model_data[fixest::obs(model), , drop = FALSE]
  z_result <- coefficient_row(model, "^z$")

  first_stage_models[[index]] <- model
  first_stage_rows[[index]] <- bind_cols(
    specification,
    tibble(
      observations = nobs(model),
      counties = n_distinct(estimation_data$county_fips),
      aewr_regions = n_distinct(estimation_data$aewr_region_id),
      inference_clusters = n_distinct(
        estimation_data$aewr_iv_cluster_id
      ),
      minimum_year = min(estimation_data$year),
      maximum_year = max(estimation_data$year),
      r2 = model_r2(model),
      within_r2 = model_r2(model, "wr2")
    ),
    z_result,
    tibble(first_stage_f = z_result$t_statistic^2)
  )
}

first_stage_results <- bind_rows(first_stage_rows)
first_stage_observations <- map(first_stage_models, fixest::obs)
if (
  n_distinct(first_stage_results$observations) != 1L ||
    !all(
      map_lgl(
        first_stage_observations[-1],
        ~ identical(.x, first_stage_observations[[1]])
      )
    )
) {
  stop(
    "All four first-stage columns must use the same observations.",
    call. = FALSE
  )
}
if (
  first_stage_results$minimum_year[[1]] != DISSIMILARITY_IV_POLICY_START_YEAR ||
    first_stage_results$maximum_year[[1]] != DISSIMILARITY_IV_POLICY_END_YEAR
) {
  stop("The first-stage sample must span 2011 through 2022.", call. = FALSE)
}

write_csv(
  first_stage_results,
  path_tables("iv_preferred_first_stage_estimates.csv")
)

etable(
  first_stage_models,
  tex = TRUE,
  title = "First Stage: Real AEWR on OEWS-Area Donor-Wage Instruments",
  label = "tab:iv_preferred_first_stage",
  headers = list("Specification" = first_stage_specs$header),
  keep_raw = "^z$",
  dict = c(
    z = "Instrument $Z$",
    ln_pop_census_l1 = "Lagged log population",
    farm_emp_share_l1 = "Lagged farm-employment share",
    emp_pop_ratio_l1 = "Lagged employment/population",
    wage_p10_l1 = "Lagged real p10 wage",
    h2a_ppml_static_propensity_z = "Static PPML propensity (standardized)",
    year_centered = "Year minus 2011"
  ),
  fitstat = ~ n + r2,
  extralines = list(
    "_^Excluded-instrument F" = format_number(
      first_stage_results$first_stage_f
    ),
    "_Seasonal and composition moments" = c("No", "Yes", "No", "Yes"),
    "_Controls" = if_else(first_stage_specs$controls, "Yes", "No"),
    "_Static propensity differential trend" = if_else(
      first_stage_specs$controls,
      "Yes",
      "No"
    ),
    "_Prediction training cutoff" = rep(
      as.character(H2A_PREDICTION_CUTOFF_YEAR),
      4
    ),
    "_County fixed effects" = rep("Yes", 4),
    "_Year fixed effects" = rep("Yes", 4),
    "_AEWR-subregion clustered SEs" = rep("Yes", 4),
    "_Number of SE clusters" = as.character(
      first_stage_results$inference_clusters
    ),
    "_Common sample" = rep("Yes", 4)
  ),
  notes = c(
    paste0(
      "All columns use the same complete-case county-year sample from ",
      DISSIMILARITY_IV_POLICY_START_YEAR,
      " through ",
      DISSIMILARITY_IV_POLICY_END_YEAR,
      "."
    ),
    paste(
      "Column 4 is preferred. The excluded-instrument F is the squared",
      "AEWR-region-by-subregion clustered t statistic."
    ),
    paste(
      "Both instruments aggregate the county-mapped OEWS-area Big-Six",
      "hourly-wage proxy using the calibrated county weights; QCEW supplies",
      "county employment, seasonal, and field/livestock features."
    ),
    paste0(
      "Controlled columns include the standardized static PPML propensity ",
      "interacted with year minus 2011; ",
      "the selected PPML training cutoff is ",
      H2A_PREDICTION_CUTOFF_YEAR,
      "."
    )
  ),
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
  file = path_tables("table_iv_preferred_first_stage.tex"),
  replace = TRUE
)

# Four-column 2SLS outcomes -------------------------------------------------

second_stage_specs <- tribble(
  ~column                   , ~instrument , ~moment_targets , ~controls , ~header ,
  1L                        ,
  "z_wage_only_real"        ,
  "Wage only"               ,
  FALSE                     ,
  "Wage only"               ,
  2L                        ,
  "z_wage_seasonal_composition_real",
  "Wage + seasonal/composition",
  FALSE                     ,
  "Seasonal/composition"    ,
  3L                        ,
  "z_wage_only_real"        ,
  "Wage only"               ,
  TRUE                      ,
  "Wage only + controls"    ,
  4L                        ,
  "z_wage_seasonal_composition_real",
  "Wage + seasonal/composition",
  TRUE                      ,
  "Seasonal/composition + controls"
)

iv_result_rows <- list()
iv_sample_rows <- list()
iv_control_diagnostic_rows <- list()
h2a_adjustment_models <- list()

for (outcome_index in seq_len(nrow(outcomes))) {
  outcome_spec <- outcomes[outcome_index, ]
  outcome_name <- outcome_spec$outcome[[1]]
  outcome_data <- finite_complete(
    panel,
    c(
      outcome_name,
      endogenous,
      "z_wage_only_real",
      "z_wage_seasonal_composition_real",
      DISSIMILARITY_IV_CONTROL_COLUMNS
    ),
    c("county_fips", "year", "aewr_iv_cluster_id")
  )

  iv_models <- pmap(
    second_stage_specs,
    function(column, instrument, moment_targets, controls, header) {
      model_data <- outcome_data %>%
        mutate(z = .data[[instrument]])
      rhs <- if (controls) control_terms else "1"
      feols(
        as.formula(paste(
          outcome_name,
          "~",
          rhs,
          "|",
          fixed_effects,
          "|",
          endogenous,
          "~ z"
        )),
        data = model_data,
        vcov = cluster_vcov,
        warn = FALSE,
        notes = FALSE
      )
    }
  )
  model_observations <- map(iv_models, fixest::obs)
  if (
    !all(
      map_lgl(
        model_observations[-1],
        ~ identical(.x, model_observations[[1]])
      )
    )
  ) {
    stop(
      "The four IV columns for ",
      outcome_name,
      " do not use the same observations.",
      call. = FALSE
    )
  }
  estimation_data <- outcome_data[
    model_observations[[1]],
    ,
    drop = FALSE
  ]
  if (outcome_name %in% h2a_adjustment_outcomes) {
    h2a_adjustment_models[[outcome_name]] <- iv_models[[4]]
  }

  first_stage_models_outcome <- pmap(
    second_stage_specs,
    function(column, instrument, moment_targets, controls, header) {
      model_data <- estimation_data %>%
        mutate(z = .data[[instrument]])
      rhs <- if (controls) {
        paste("z", control_terms, sep = " + ")
      } else {
        "z"
      }
      feols(
        as.formula(paste(
          endogenous,
          "~",
          rhs,
          "|",
          fixed_effects
        )),
        data = model_data,
        vcov = cluster_vcov,
        warn = FALSE,
        notes = FALSE
      )
    }
  )
  first_stage_f <- map_dbl(
    first_stage_models_outcome,
    ~ coefficient_row(.x, "^z$")$t_statistic^2
  )
  inference_clusters <- n_distinct(
    estimation_data$aewr_iv_cluster_id
  )

  # Identical-sample diagnostic: compare the four committed controls with the
  # same controls plus the static PPML propensity differential trend.
  diagnostic_data <- estimation_data %>%
    mutate(z = z_wage_seasonal_composition_real)
  baseline_control_terms <- paste(
    DISSIMILARITY_IV_BASELINE_CONTROL_COLUMNS,
    collapse = " + "
  )
  four_control_iv <- feols(
    as.formula(paste(
      outcome_name,
      "~",
      baseline_control_terms,
      "|",
      fixed_effects,
      "|",
      endogenous,
      "~ z"
    )),
    data = diagnostic_data,
    vcov = cluster_vcov,
    warn = FALSE,
    notes = FALSE
  )
  four_control_first_stage <- feols(
    as.formula(paste(
      endogenous,
      "~ z +",
      baseline_control_terms,
      "|",
      fixed_effects
    )),
    data = diagnostic_data,
    vcov = cluster_vcov,
    warn = FALSE,
    notes = FALSE
  )
  trend_control_iv <- iv_models[[4]]
  trend_control_first_stage <- first_stage_models_outcome[[4]]
  observation_keys <- function(data, model) {
    selected <- data[
      fixest::obs(model),
      c("county_fips", "year"),
      drop = FALSE
    ]
    paste(selected$county_fips, selected$year, sep = "_")
  }
  diagnostic_observations <- list(
    four_control_iv = observation_keys(diagnostic_data, four_control_iv),
    trend_control_iv = observation_keys(outcome_data, trend_control_iv),
    four_control_first_stage = observation_keys(
      diagnostic_data,
      four_control_first_stage
    ),
    trend_control_first_stage = observation_keys(
      estimation_data,
      trend_control_first_stage
    )
  )
  if (
    !all(map_lgl(
      diagnostic_observations,
      ~ identical(.x, diagnostic_observations[[1]])
    )) ||
      length(diagnostic_observations[[1]]) != nrow(diagnostic_data)
  ) {
    stop(
      "Four-control and differential-trend diagnostics must use identical samples.",
      call. = FALSE
    )
  }
  four_iv_result <- coefficient_row(
    four_control_iv,
    "^(fit_)?aewr_ppi$"
  )
  trend_iv_result <- coefficient_row(
    trend_control_iv,
    "^(fit_)?aewr_ppi$"
  )
  four_first_stage_result <- coefficient_row(
    four_control_first_stage,
    "^z$"
  )
  trend_first_stage_result <- coefficient_row(
    trend_control_first_stage,
    "^z$"
  )
  iv_control_diagnostic_rows[[length(iv_control_diagnostic_rows) + 1L]] <-
    tibble(
      outcome = outcome_name,
      outcome_label = outcome_spec$outcome_label[[1]],
      observations = nobs(trend_control_iv),
      counties = n_distinct(diagnostic_data$county_fips),
      inference_clusters = inference_clusters,
      h2a_prediction_cutoff_year = H2A_PREDICTION_CUTOFF_YEAR,
      h2a_prediction_model_spec = H2A_PREDICTION_MODEL_SPEC,
      four_control_estimate = four_iv_result$estimate[[1]],
      trend_control_estimate = trend_iv_result$estimate[[1]],
      estimate_change = trend_control_estimate - four_control_estimate,
      four_control_clustered_se = four_iv_result$standard_error[[1]],
      trend_control_clustered_se = trend_iv_result$standard_error[[1]],
      clustered_se_change =
        trend_control_clustered_se - four_control_clustered_se,
      four_control_within_r2 = model_r2(four_control_iv, "wr2"),
      trend_control_within_r2 = model_r2(trend_control_iv, "wr2"),
      within_r2_change = trend_control_within_r2 - four_control_within_r2,
      four_control_first_stage_f =
        four_first_stage_result$t_statistic[[1]]^2,
      trend_control_first_stage_f =
        trend_first_stage_result$t_statistic[[1]]^2,
      first_stage_f_change =
        trend_control_first_stage_f - four_control_first_stage_f
    )

  for (model_index in seq_along(iv_models)) {
    result <- coefficient_row(
      iv_models[[model_index]],
      "^(fit_)?aewr_ppi$"
    )
    specification <- second_stage_specs[model_index, ]
    iv_result_rows[[length(iv_result_rows) + 1L]] <- bind_cols(
      outcome_spec,
      specification,
      tibble(
        observations = nobs(iv_models[[model_index]]),
        counties = n_distinct(estimation_data$county_fips),
        aewr_regions = n_distinct(estimation_data$aewr_region_id),
        inference_clusters = inference_clusters,
        minimum_year = min(estimation_data$year),
        maximum_year = max(estimation_data$year),
        r2 = model_r2(iv_models[[model_index]]),
        within_r2 = model_r2(iv_models[[model_index]], "wr2"),
        first_stage_f = first_stage_f[[model_index]]
      ),
      result
    )
  }

  iv_sample_rows[[length(iv_sample_rows) + 1L]] <- tibble(
    outcome = outcome_name,
    outcome_label = outcome_spec$outcome_label,
    complete_case_observations = nrow(outcome_data),
    singleton_observations_removed = nrow(outcome_data) -
      nrow(estimation_data),
    estimation_observations = nrow(estimation_data),
    counties = n_distinct(estimation_data$county_fips),
    aewr_regions = n_distinct(estimation_data$aewr_region_id),
    inference_clusters = inference_clusters,
    minimum_year = min(estimation_data$year),
    maximum_year = max(estimation_data$year)
  )

  if (n_distinct(map_int(iv_models, nobs)) != 1L) {
    stop(
      "The four IV models must have the same N for ",
      outcome_name,
      ".",
      call. = FALSE
    )
  }

  etable(
    iv_models,
    tex = TRUE,
    title = paste0(
      "IV Effect of the Real AEWR on ",
      outcome_spec$outcome_label[[1]]
    ),
    label = paste0("tab:iv_", outcome_spec$table_stub[[1]]),
    headers = list(
      "Specification" = second_stage_specs$header
    ),
    keep_raw = "^(fit_)?aewr_ppi$",
    dict = c(
      fit_aewr_ppi = "Real AEWR (2012 dollars)",
      aewr_ppi = "Real AEWR (2012 dollars)",
      ln_pop_census_l1 = "Lagged log population",
      farm_emp_share_l1 = "Lagged farm-employment share",
      emp_pop_ratio_l1 = "Lagged employment/population",
      wage_p10_l1 = "Lagged real p10 wage",
      h2a_ppml_static_propensity_z = "Static PPML propensity (standardized)",
      year_centered = "Year minus 2011"
    ),
    fitstat = ~ n + r2,
    extralines = list(
      "_^First-stage excluded-instrument F" = format_number(
        first_stage_f
      ),
      "_Seasonal and composition moments" = if_else(
        second_stage_specs$instrument == "z_wage_seasonal_composition_real",
        "Yes",
        "No"
      ),
      "_Controls" = if_else(
        second_stage_specs$controls,
        "Yes",
        "No"
      ),
      "_Static propensity differential trend" = if_else(
        second_stage_specs$controls,
        "Yes",
        "No"
      ),
      "_Prediction training cutoff" = rep(
        as.character(H2A_PREDICTION_CUTOFF_YEAR),
        4
      ),
      "_County fixed effects" = rep("Yes", 4),
      "_Year fixed effects" = rep("Yes", 4),
      "_AEWR-subregion clustered SEs" = rep("Yes", 4),
      "_Number of SE clusters" = rep(
        as.character(inference_clusters),
        4
      ),
      "_Common outcome sample" = rep("Yes", 4)
    ),
    notes = c(
      paste(
        "Columns 1 and 3 use the wage-only instrument; columns 2 and 4",
        "use the wage-plus-seasonal-and-composition instrument."
      ),
      paste(
        "Both instruments use the county-mapped OEWS-area Big-Six hourly",
        "wage proxy as the donor wage level; QCEW supplies county",
        "employment, seasonal, and field/livestock features."
      ),
      paste0(
        "All columns use the controlled specification's complete-case ",
        "county-year sample from ",
        DISSIMILARITY_IV_POLICY_START_YEAR,
        " through ",
        DISSIMILARITY_IV_POLICY_END_YEAR,
        ". Column 4 is preferred."
      ),
      paste0(
        "Controlled columns include the standardized static PPML propensity ",
        "interacted with year minus 2011; ",
        "the selected PPML training cutoff is ",
        H2A_PREDICTION_CUTOFF_YEAR,
        "."
      )
    ),
    signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
    file = path_tables(paste0(
      "table_iv_",
      outcome_spec$table_stub[[1]],
      ".tex"
    )),
    replace = TRUE
  )
}

iv_results <- bind_rows(iv_result_rows)
iv_samples <- bind_rows(iv_sample_rows)
iv_control_diagnostic <- bind_rows(iv_control_diagnostic_rows)

write_csv(
  iv_control_diagnostic,
  path_tables("iv_static_propensity_trend_diagnostic.csv")
)

diagnostic_table_rows <- vapply(
  seq_len(nrow(iv_control_diagnostic)),
  function(index) {
    row <- iv_control_diagnostic[index, ]
    sprintf(
      paste0(
        "%s & %.3f (%.3f) & %.3f (%.3f) & %.3f & %.3f & ",
        "%.3f $\\rightarrow$ %.3f & %.2f $\\rightarrow$ %.2f & %d & %d \\\\"
      ),
      row$outcome_label,
      row$four_control_estimate,
      row$four_control_clustered_se,
      row$trend_control_estimate,
      row$trend_control_clustered_se,
      row$estimate_change,
      row$clustered_se_change,
      row$four_control_within_r2,
      row$trend_control_within_r2,
      row$four_control_first_stage_f,
      row$trend_control_first_stage_f,
      row$observations,
      row$counties
    )
  },
  character(1)
)
diagnostic_table_tex <- c(
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{Identical-Sample Static PPML Differential-Trend Diagnostic}",
  "\\label{tab:iv_static_propensity_trend_diagnostic}",
  "\\resizebox{\\textwidth}{!}{%",
  "\\begin{tabular}{lrrrrrrrr}",
  "\\hline\\hline",
  paste0(
    "Outcome & Four controls & + Static trend & $\\Delta$ coefficient & ",
    "$\\Delta$ clustered SE & Within $R^2$ & Excluded-instrument F & ",
    "$N$ & Counties \\\\"
  ),
  "\\hline",
  diagnostic_table_rows,
  "\\hline\\hline",
  "\\end{tabular}%",
  "}",
  paste0(
    "\\begin{minipage}{0.98\\textwidth}\\footnotesize Notes: ",
    "Each pair uses the same outcome-specific county-year observations, ",
    "county and year fixed effects, the preferred wage-plus-seasonal-and-",
    "composition ",
    "instrument, and AEWR-subregion clustered standard errors. The static-trend ",
    "column adds standardized static PPML propensity interacted with year ",
    "minus 2011. The PPML model is trained through ",
    H2A_PREDICTION_CUTOFF_YEAR,
    ". Changes are static-trend minus ",
    "four-control; no precision improvement is imposed.\\end{minipage}"
  ),
  "\\end{table}"
)
writeLines(
  diagnostic_table_tex,
  path_tables("table_iv_static_propensity_trend_diagnostic.tex")
)

h2a_adjustment_specs <- tibble(
  outcome = h2a_adjustment_outcomes,
  column_header = c(
    "Certified hours",
    "Applications",
    "Positions/application",
    "Hours/position"
  ),
  conditioning = c(
    "None",
    "None",
    "Applications $>0$",
    "Certified positions $>0$"
  )
)

h2a_adjustment_results <- iv_results %>%
  filter(column == 4L) %>%
  inner_join(
    h2a_adjustment_specs,
    by = "outcome",
    relationship = "many-to-one"
  ) %>%
  arrange(match(outcome, h2a_adjustment_outcomes))

if (
  nrow(h2a_adjustment_results) != length(h2a_adjustment_outcomes) ||
    !identical(h2a_adjustment_results$outcome, h2a_adjustment_outcomes)
) {
  stop(
    "The preferred H-2A adjustment-margin results are incomplete.",
    call. = FALSE
  )
}

write_csv(
  h2a_adjustment_results,
  path_tables("iv_preferred_h2a_adjustment_margin_estimates.csv")
)

h2a_adjustment_models <- unname(
  h2a_adjustment_models[h2a_adjustment_outcomes]
)

etable(
  h2a_adjustment_models,
  tex = TRUE,
  title = "IV Effects of the Real AEWR on H-2A Adjustment Margins",
  label = "tab:iv_h2a_adjustment_margins",
  depvar = FALSE,
  headers = list(
    "Outcome" = h2a_adjustment_specs$column_header
  ),
  keep_raw = "^(fit_)?aewr_ppi$",
  dict = c(
    fit_aewr_ppi = "Real AEWR (2012 dollars)",
    aewr_ppi = "Real AEWR (2012 dollars)",
    ln_pop_census_l1 = "Lagged log population",
    farm_emp_share_l1 = "Lagged farm-employment share",
    emp_pop_ratio_l1 = "Lagged employment/population",
    wage_p10_l1 = "Lagged real p10 wage",
    h2a_ppml_static_propensity_z = "Static PPML propensity (standardized)",
    year_centered = "Year minus 2011"
  ),
  fitstat = ~ n + r2,
  extralines = list(
    "_^First-stage excluded-instrument F" = format_number(
      h2a_adjustment_results$first_stage_f
    ),
    "_Conditioning" = h2a_adjustment_specs$conditioning,
    "_Controls" = rep("Yes", length(h2a_adjustment_outcomes)),
    "_Static propensity differential trend" = rep(
      "Yes",
      length(h2a_adjustment_outcomes)
    ),
    "_Prediction training cutoff" = rep(
      as.character(H2A_PREDICTION_CUTOFF_YEAR),
      length(h2a_adjustment_outcomes)
    ),
    "_County fixed effects" = rep(
      "Yes",
      length(h2a_adjustment_outcomes)
    ),
    "_Year fixed effects" = rep(
      "Yes",
      length(h2a_adjustment_outcomes)
    ),
    "_AEWR-subregion clustered SEs" = rep(
      "Yes",
      length(h2a_adjustment_outcomes)
    ),
    "_Number of SE clusters" = as.character(
      h2a_adjustment_results$inference_clusters
    )
  ),
  notes = c(
    paste(
      "All columns use the preferred wage-plus-seasonal-and-composition",
      "instrument and",
      "the four lagged controls plus the static propensity differential trend."
    ),
    paste(
      "The donor wage level is the county-mapped OEWS-area Big-Six hourly",
      "wage proxy; QCEW supplies county employment and calibration features."
    ),
    paste(
      "Certified hours and applications are normalized by 2011 farm",
      "employment. Ratio outcomes retain only positive-denominator",
      "county-years."
    ),
    paste(
      "The identity H = A x (N/A) x (H/N) holds in the underlying data;",
      "the level-IV coefficients use different samples and are not",
      "algebraically additive."
    )
  ),
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
  file = path_tables("table_iv_h2a_adjustment_margins.tex"),
  replace = TRUE
)

sample_contract <- iv_results %>%
  count(outcome, observations) %>%
  count(outcome, name = "sample_counts")
if (any(sample_contract$sample_counts != 1L)) {
  stop("Every IV outcome table must use a common sample.", call. = FALSE)
}

write_csv(
  iv_results,
  path_tables("iv_preferred_second_stage_estimates.csv")
)
write_csv(
  iv_samples,
  path_tables("iv_preferred_second_stage_samples.csv")
)

# Email-ready transposed results -------------------------------------------

email_outcomes <- tribble(
  ~outcome, ~csv_label, ~pdf_label, ~outcome_group,
  "h2a_cert_share_farm_workers_2011_start_year",
  "Certified workers / 2011 farm employment",
  "Certified\nworkers /\n2011 farm emp.",
  "H-2A outcomes",
  "h2a_cert_hours_per_farm_worker_2011_start_year",
  "Certified hours / 2011 farm employment",
  "Certified\nhours /\n2011 farm emp.",
  "H-2A outcomes",
  "h2a_applications_per_farm_worker_2011_start_year",
  "Applications / 2011 farm employment",
  "Applications /\n2011 farm\nemployment",
  "H-2A outcomes",
  "h2a_employers_per_farm_worker_2011_start_year",
  "Employers / 2011 farm employment",
  "Employers /\n2011 farm\nemployment",
  "H-2A outcomes",
  "h2a_cert_positions_per_application_start_year",
  "Certified positions / application",
  "Certified\npositions /\napplication",
  "H-2A outcomes",
  "h2a_cert_hours_per_position_start_year",
  "Certified hours / position",
  "Certified\nhours /\nposition",
  "H-2A outcomes",
  "fisher_index_ppi",
  "Real Fisher crop price index",
  "Real crop\nprice\nindex",
  "Farm outcomes",
  "emp_farm",
  "Farm employment",
  "Farm\nemployment",
  "Farm outcomes",
  "share_farm_prodexp_cashandinc",
  "Production expenses / farm cash income",
  "Production\nexpense\nshare",
  "Farm outcomes",
  "farm_cashandinc_ppi_per_farm_worker",
  "Real farm cash income / farm worker",
  "Real farm\nincome /\nworker",
  "Farm outcomes",
  "share_farm_laborexp_prodexp",
  "Hired-labor share of production expenses",
  "Hired-labor\nexpense\nshare",
  "Farm outcomes",
  "fisher_quantity_index",
  "Fisher crop output-quantity index",
  "Crop output\nquantity\nindex",
  "Farm outcomes"
)

email_specs <- second_stage_specs %>%
  mutate(
    file_stub = c(
      "spec1_wage_only",
      "spec2_seasonal_composition",
      "spec3_wage_only_controls",
      "spec4_preferred"
    ),
    display_title = c(
      "Wage only",
      "Seasonal/composition",
      "Wage only + controls",
      "Seasonal/composition + controls (preferred)"
    )
  )

email_statistics <- c(
  "AEWR coefficient",
  "Clustered standard error",
  "p-value",
  "First-stage excluded-instrument F",
  "Observations",
  "Counties",
  "Inference clusters"
)

if (
  nrow(email_outcomes) != nrow(outcomes) ||
    !setequal(email_outcomes$outcome, outcomes$outcome) ||
    anyDuplicated(email_outcomes$csv_label) > 0L
) {
  stop("Email-ready IV outcomes do not match the estimation registry.",
    call. = FALSE)
}

make_email_csv <- function(specification_column) {
  selected <- iv_results %>%
    filter(column == specification_column) %>%
    right_join(email_outcomes, by = "outcome", relationship = "one-to-one") %>%
    arrange(match(outcome, email_outcomes$outcome))
  if (nrow(selected) != nrow(email_outcomes) || any(is.na(selected$estimate))) {
    stop("Email-ready IV table is missing an outcome estimate.", call. = FALSE)
  }
  result <- tibble(statistic = email_statistics)
  for (index in seq_len(nrow(selected))) {
    result[[selected$csv_label[[index]]]] <- c(
      selected$estimate[[index]],
      selected$standard_error[[index]],
      selected$p_value[[index]],
      selected$first_stage_f[[index]],
      selected$observations[[index]],
      selected$counties[[index]],
      selected$inference_clusters[[index]]
    )
  }
  result
}

format_email_effect <- function(value) {
  absolute <- abs(value)
  if (absolute >= 1000) {
    return(formatC(value, format = "f", digits = 0, big.mark = ","))
  }
  if (absolute >= 100) {
    return(formatC(value, format = "f", digits = 1, big.mark = ","))
  }
  if (absolute >= 10) {
    return(formatC(value, format = "f", digits = 2))
  }
  if (absolute >= 1) {
    return(formatC(value, format = "f", digits = 3))
  }
  if (absolute >= 0.001) {
    return(formatC(value, format = "f", digits = 4))
  }
  formatC(value, format = "e", digits = 2)
}

format_email_p <- function(value) {
  if (value < 0.001) {
    return("<0.001")
  }
  formatC(value, format = "f", digits = 3)
}

significance_stars <- function(value) {
  if (value < 0.01) {
    return("***")
  }
  if (value < 0.05) {
    return("**")
  }
  if (value < 0.10) {
    return("*")
  }
  ""
}

format_email_table <- function(data) {
  values <- as.matrix(data[-1])
  formatted <- matrix(
    "",
    nrow = nrow(values),
    ncol = ncol(values),
    dimnames = dimnames(values)
  )
  for (column_index in seq_len(ncol(values))) {
    formatted[1, column_index] <- paste0(
      format_email_effect(values[1, column_index]),
      significance_stars(values[3, column_index])
    )
    formatted[2, column_index] <- paste0(
      "(",
      format_email_effect(values[2, column_index]),
      ")"
    )
    formatted[3, column_index] <- format_email_p(values[3, column_index])
    formatted[4, column_index] <- formatC(
      values[4, column_index],
      format = "f",
      digits = 2
    )
    formatted[5, column_index] <- formatC(
      values[5, column_index],
      format = "f",
      digits = 0,
      big.mark = ","
    )
    formatted[6, column_index] <- formatC(
      values[6, column_index],
      format = "f",
      digits = 0,
      big.mark = ","
    )
    formatted[7, column_index] <- formatC(
      values[7, column_index],
      format = "f",
      digits = 0,
      big.mark = ","
    )
  }
  formatted
}

draw_email_table <- function(data, specification, page_number) {
  grid::grid.newpage()
  formatted <- format_email_table(data)
  page_left <- 0.025
  page_right <- 0.985
  statistic_width <- 0.135
  data_width <- (page_right - page_left - statistic_width) /
    nrow(email_outcomes)
  data_left <- page_left + statistic_width
  column_centers <- data_left +
    (seq_len(nrow(email_outcomes)) - 0.5) * data_width

  grid::grid.text(
    paste0(
      "Panel IV Results - Specification ",
      specification$column,
      ": ",
      specification$display_title
    ),
    x = page_left,
    y = 0.965,
    just = "left",
    gp = grid::gpar(fontsize = 17, fontface = "bold", col = "#17365D")
  )
  grid::grid.text(
    paste(
      "Treatment: real AEWR (2012 dollars). County and year fixed effects;",
      "standard errors clustered by AEWR-region x target subregion."
    ),
    x = page_left,
    y = 0.925,
    just = "left",
    gp = grid::gpar(fontsize = 9.5, col = "#333333")
  )
  grid::grid.text(
    paste(
      "Donor wage level: county-mapped OEWS-area Big-Six hourly-wage proxy;",
      "QCEW supplies county employment and calibration features; source year t-1."
    ),
    x = page_left,
    y = 0.898,
    just = "left",
    gp = grid::gpar(fontsize = 8.5, col = "#555555")
  )

  group_y <- 0.835
  group_height <- 0.043
  group_ranges <- email_outcomes %>%
    mutate(column_number = row_number()) %>%
    group_by(outcome_group) %>%
    summarise(
      first_column = min(column_number),
      last_column = max(column_number),
      .groups = "drop"
    )
  group_colors <- c(
    "H-2A outcomes" = "#D9EAF7",
    "Farm outcomes" = "#E4F1E8"
  )
  for (group_index in seq_len(nrow(group_ranges))) {
    group <- group_ranges[group_index, ]
    group_x <- data_left + (group$first_column - 1) * data_width
    group_width <- (group$last_column - group$first_column + 1) * data_width
    grid::grid.rect(
      x = group_x + group_width / 2,
      y = group_y,
      width = group_width,
      height = group_height,
      gp = grid::gpar(
        fill = group_colors[[group$outcome_group]],
        col = "white",
        lwd = 1
      )
    )
    grid::grid.text(
      group$outcome_group,
      x = group_x + group_width / 2,
      y = group_y,
      gp = grid::gpar(fontsize = 9, fontface = "bold", col = "#17365D")
    )
  }

  header_top <- group_y - group_height / 2
  header_bottom <- 0.675
  grid::grid.rect(
    x = page_left + statistic_width / 2,
    y = (header_top + header_bottom) / 2,
    width = statistic_width,
    height = header_top - header_bottom,
    gp = grid::gpar(fill = "#F2F2F2", col = "white")
  )
  grid::grid.text(
    "Statistic",
    x = page_left + 0.006,
    y = (header_top + header_bottom) / 2,
    just = "left",
    gp = grid::gpar(fontsize = 9, fontface = "bold")
  )
  for (column_index in seq_len(nrow(email_outcomes))) {
    fill_color <- if (
      email_outcomes$outcome_group[[column_index]] == "H-2A outcomes"
    ) {
      "#EEF6FB"
    } else {
      "#F1F8F3"
    }
    grid::grid.rect(
      x = column_centers[[column_index]],
      y = (header_top + header_bottom) / 2,
      width = data_width,
      height = header_top - header_bottom,
      gp = grid::gpar(fill = fill_color, col = "white")
    )
    grid::grid.text(
      email_outcomes$pdf_label[[column_index]],
      x = column_centers[[column_index]],
      y = (header_top + header_bottom) / 2,
      gp = grid::gpar(fontsize = 7.5, fontface = "bold", lineheight = 0.9)
    )
  }

  row_height <- 0.068
  row_top <- header_bottom
  for (row_index in seq_along(email_statistics)) {
    row_center <- row_top - (row_index - 0.5) * row_height
    row_fill <- if (row_index %% 2L == 0L) "#F8F8F8" else "white"
    grid::grid.rect(
      x = (page_left + page_right) / 2,
      y = row_center,
      width = page_right - page_left,
      height = row_height,
      gp = grid::gpar(fill = row_fill, col = "#E3E3E3", lwd = 0.6)
    )
    grid::grid.text(
      email_statistics[[row_index]],
      x = page_left + 0.006,
      y = row_center,
      just = "left",
      gp = grid::gpar(
        fontsize = 8.2,
        fontface = if (row_index == 1L) "bold" else "plain"
      )
    )
    for (column_index in seq_len(ncol(formatted))) {
      grid::grid.text(
        formatted[row_index, column_index],
        x = column_centers[[column_index]],
        y = row_center,
        gp = grid::gpar(
          fontsize = 8.1,
          fontface = if (row_index == 1L) "bold" else "plain"
        )
      )
    }
  }

  grid::grid.text(
    paste(
      "Notes: Coefficients report the effect of a one-dollar increase in real",
      "AEWR. Clustered standard errors are in parentheses; * p<0.10,",
      "** p<0.05, *** p<0.01. The first-stage F is the squared clustered",
      "t statistic for the excluded instrument."
    ),
    x = page_left,
    y = 0.135,
    just = "left",
    gp = grid::gpar(fontsize = 7.8, col = "#444444")
  )
  grid::grid.text(
    paste0("Page ", page_number, " of ", nrow(email_specs)),
    x = page_right,
    y = 0.055,
    just = "right",
    gp = grid::gpar(fontsize = 7.5, col = "#777777")
  )
}

email_tables <- vector("list", nrow(email_specs))
for (specification_index in seq_len(nrow(email_specs))) {
  specification <- email_specs[specification_index, ]
  email_tables[[specification_index]] <- make_email_csv(
    specification$column[[1]]
  )
  write_csv(
    email_tables[[specification_index]],
    path_tables(paste0(
      "panel_iv_email_results_",
      specification$file_stub[[1]],
      ".csv"
    ))
  )
}

grDevices::pdf(
  path_tables("panel_iv_email_results.pdf"),
  width = 17,
  height = 11,
  onefile = TRUE,
  paper = "special",
  family = "sans"
)
for (specification_index in seq_len(nrow(email_specs))) {
  draw_email_table(
    email_tables[[specification_index]],
    email_specs[specification_index, ],
    specification_index
  )
}
grDevices::dev.off()

# Summary statistics --------------------------------------------------------

summary_variable_labels <- c(
  aewr_ppi = "Real AEWR (2012 dollars)",
  z_wage_only_real = "Wage-only OEWS-area hourly-wage instrument",
  z_wage_seasonal_composition_real = paste(
    "Preferred OEWS-area hourly-wage plus seasonal/composition instrument"
  ),
  ln_pop_census_l1 = "Lagged log population",
  farm_emp_share_l1 = "Lagged farm-employment share",
  emp_pop_ratio_l1 = "Lagged employment/population",
  wage_p10_l1 = "Lagged real p10 wage",
  h2a_ppml_static_propensity_z = "Static PPML propensity (standardized)",
  year_centered = "Year minus 2011",
  setNames(outcomes$outcome_label, outcomes$outcome)
)

summary_data <- panel %>%
  filter(
    year >= DISSIMILARITY_IV_POLICY_START_YEAR,
    year <= DISSIMILARITY_IV_POLICY_END_YEAR
  )

summary_statistics <- imap_dfr(
  summary_variable_labels,
  function(label, variable) {
    values <- summary_data[[variable]]
    values <- values[is.finite(values)]
    tibble(
      variable,
      label,
      observations = length(values),
      mean = mean(values),
      standard_deviation = sd(values),
      minimum = min(values),
      maximum = max(values)
    )
  }
)

write_csv(
  summary_statistics,
  path_tables("iv_preferred_summary_statistics.csv")
)

summary_rows <- pmap_chr(
  summary_statistics,
  function(
    variable,
    label,
    observations,
    mean,
    standard_deviation,
    minimum,
    maximum
  ) {
    sprintf(
      "%s & %s & %.3f & %.3f & %.3f & %.3f \\\\",
      label,
      format(
        observations,
        big.mark = ",",
        scientific = FALSE
      ),
      mean,
      standard_deviation,
      minimum,
      maximum
    )
  }
)

summary_tex <- c(
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{Summary Statistics for Preferred IV Variables}",
  "\\label{tab:iv_preferred_sumstats}",
  "\\begin{tabular}{lrrrrr}",
  "\\hline\\hline",
  "Variable & N & Mean & SD & Min & Max \\\\",
  "\\hline",
  summary_rows,
  "\\hline\\hline",
  paste0(
    "\\multicolumn{6}{p{0.95\\linewidth}}{\\footnotesize Notes: ",
    "All finite county-year observations from ",
    DISSIMILARITY_IV_POLICY_START_YEAR,
    " through ",
    DISSIMILARITY_IV_POLICY_END_YEAR,
    ". Fixed-effect and cluster identifiers are excluded.} \\\\"
  ),
  "\\end{tabular}",
  "\\end{table}"
)

writeLines(
  summary_tex,
  path_tables("table_iv_preferred_summary_statistics.tex")
)

# fixest aligns TeX cells with trailing blanks. Normalize retained source text
# after every table has been written so diffs and downstream TeX checks remain
# deterministic without changing any table content.
panel_iv_tex_paths <- list.files(
  path_tables(),
  pattern = "^table_iv_.*\\.tex$",
  full.names = TRUE
)
for (tex_path in panel_iv_tex_paths) {
  tex_lines <- readLines(tex_path, warn = FALSE)
  writeLines(sub("[[:blank:]]+$", "", tex_lines), tex_path)
}
