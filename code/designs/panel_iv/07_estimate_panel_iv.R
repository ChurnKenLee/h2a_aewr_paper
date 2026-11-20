# Purpose: Estimate the publication first stages, 2SLS outcomes, summary
# statistics, and email-ready transposed results on the county-year panel.
# Input: data/processed/panel_iv_county_year.parquet.
# Every model uses the wage-plus-seasonal/composition instrument. Column 1 has
# no controls; column 2 adds the declared controls and is preferred.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "specification.R"))
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
  path_figures("fig_iv_dissimilarity_first_stage.png"),
  path_tables("table_iv_h2a_employers.tex")
)
unlink(obsolete_outputs[file.exists(obsolete_outputs)])

estimation_spec <- panel_iv_read_specification(
  path_code("designs", "panel_iv", "estimation_spec.json")
)
endogenous <- estimation_spec$endogenous
instrument <- estimation_spec$instrument
fixed_effects <- paste(estimation_spec$fixed_effects, collapse = " + ")
cluster_vcov <- stats::as.formula(paste("~", estimation_spec$primary_inference_cluster))
control_terms <- paste(estimation_spec$controls, collapse = " + ")
control_columns <- DISSIMILARITY_IV_CONTROL_COLUMNS
policy_end_year <- estimation_spec$policy_years[[2]]
preferred_column <- estimation_spec$preferred_column
model_specs <- estimation_spec$models %>%
  transmute(column, instrument = instrument, controls, header)
model_count <- nrow(model_specs)
outcomes <- estimation_spec$outcomes
h2a_adjustment_outcomes <- outcomes$outcome[outcomes$adjustment_margin]

panel <- panel_iv_prepare_outcomes(
  read_parquet(path_processed("panel_iv_county_year.parquet")),
  estimation_spec
)

required_columns <- unique(c(
  "county_fips",
  "year",
  "aewr_region_id",
  "aewr_iv_cluster_id",
  "h2a_prediction_cutoff_year",
  "h2a_prediction_model_spec",
  "h2a_predicted_share_2011",
  endogenous,
  instrument,
  control_columns,
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
# Every regression names the excluded instrument z.
panel <- panel %>%
  mutate(z = .data[[instrument]])

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

# Both columns share the controlled specification's complete cases.
complete_cases <- function(data, outcome = NULL) {
  data %>%
    filter(
      year >= DISSIMILARITY_IV_POLICY_START_YEAR,
      year <= policy_end_year,
      if_all(
        all_of(c(outcome, endogenous, "z", control_columns)),
        ~ !is.na(.x) & is.finite(.x)
      ),
      if_all(
        all_of(c("county_fips", "year", "aewr_iv_cluster_id")),
        ~ !is.na(.x)
      )
    )
}

estimate <- function(formula, data) {
  feols(
    as.formula(formula),
    data = data,
    vcov = cluster_vcov,
    warn = FALSE,
    notes = FALSE
  )
}

estimate_first_stage <- function(data, controls) {
  rhs <- if (is.null(controls)) "z" else paste("z", controls, sep = " + ")
  estimate(paste(endogenous, "~", rhs, "|", fixed_effects), data)
}

estimate_iv <- function(data, outcome, controls) {
  rhs <- if (is.null(controls)) "1" else controls
  estimate(
    paste(outcome, "~", rhs, "|", fixed_effects, "|", endogenous, "~ z"),
    data
  )
}

# The control terms of one declared column, or NULL without controls.
model_controls <- function(controls) {
  if (controls) control_terms else NULL
}

same_observations <- function(models) {
  observations <- map(models, fixest::obs)
  all(map_lgl(observations[-1], ~ identical(.x, observations[[1]])))
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

control_labels <- c(
  ln_pop_census_l1 = "Lagged log population",
  farm_emp_share_l1 = "Lagged farm-employment share",
  emp_pop_ratio_l1 = "Lagged employment/population",
  wage_p10_l1 = "Lagged real p10 wage",
  h2a_ppml_static_propensity_z = "Static PPML propensity (standardized)",
  year_centered = "Year minus 2011"
)
treatment_labels <- c(
  fit_aewr_ppi = "Real AEWR (2012 dollars)",
  aewr_ppi = "Real AEWR (2012 dollars)"
)
significance_codes <- c("***" = 0.01, "**" = 0.05, "*" = 0.10)
controls_yes_no <- if_else(model_specs$controls, "Yes", "No")
all_yes <- rep("Yes", model_count)
instrument_note <- paste(
  "Every column uses the wage-plus-seasonal-and-composition instrument, which",
  "aggregates the county-mapped OEWS-area Big-Six hourly-wage proxy using the",
  "calibrated county weights; QCEW supplies county employment, seasonal, and",
  "field/livestock features."
)
trend_note <- paste0(
  "The controlled column includes the standardized static PPML propensity ",
  "interacted with year minus 2011; ",
  "the selected PPML training cutoff is ",
  H2A_PREDICTION_CUTOFF_YEAR,
  "."
)

# First stage ----------------------------------------------------------------

first_stage_data <- complete_cases(panel)
first_stage_models <- map(
  model_specs$controls,
  ~ estimate_first_stage(first_stage_data, model_controls(.x))
)
if (!same_observations(first_stage_models)) {
  stop("Both first-stage columns must use the same observations.", call. = FALSE)
}
first_stage_sample <- first_stage_data[
  fixest::obs(first_stage_models[[1]]),
  ,
  drop = FALSE
]
if (
  min(first_stage_sample$year) != DISSIMILARITY_IV_POLICY_START_YEAR ||
    max(first_stage_sample$year) != policy_end_year
) {
  stop(
    "The first-stage sample must span ",
    DISSIMILARITY_IV_POLICY_START_YEAR,
    " through ",
    policy_end_year,
    ".",
    call. = FALSE
  )
}

first_stage_results <- map_dfr(seq_len(model_count), function(index) {
  model <- first_stage_models[[index]]
  z_result <- coefficient_row(model, "^z$")
  bind_cols(
    model_specs[index, ],
    tibble(
      observations = nobs(model),
      counties = n_distinct(first_stage_sample$county_fips),
      aewr_regions = n_distinct(first_stage_sample$aewr_region_id),
      inference_clusters = n_distinct(first_stage_sample$aewr_iv_cluster_id),
      minimum_year = min(first_stage_sample$year),
      maximum_year = max(first_stage_sample$year),
      r2 = model_r2(model),
      within_r2 = model_r2(model, "wr2")
    ),
    z_result,
    tibble(first_stage_f = z_result$t_statistic^2)
  )
})

write_csv(
  first_stage_results,
  path_tables("iv_preferred_first_stage_estimates.csv")
)

etable(
  first_stage_models,
  tex = TRUE,
  title = "First Stage: Real AEWR on the OEWS-Area Donor-Wage Instrument",
  label = "tab:iv_preferred_first_stage",
  headers = list("Specification" = model_specs$header),
  keep_raw = "^z$",
  dict = c(z = "Instrument $Z$", control_labels),
  fitstat = ~ n + r2,
  extralines = list(
    "_^Excluded-instrument F" = format_number(
      first_stage_results$first_stage_f
    ),
    "_Controls" = controls_yes_no,
    "_Static propensity differential trend" = controls_yes_no,
    "_Prediction training cutoff" = rep(
      as.character(H2A_PREDICTION_CUTOFF_YEAR),
      model_count
    ),
    "_County fixed effects" = all_yes,
    "_Year fixed effects" = all_yes,
    "_AEWR-subregion clustered SEs" = all_yes,
    "_Number of SE clusters" = as.character(
      first_stage_results$inference_clusters
    ),
    "_Common sample" = all_yes
  ),
  notes = c(
    paste0(
      "Both columns use the same complete-case county-year sample from ",
      DISSIMILARITY_IV_POLICY_START_YEAR,
      " through ",
      policy_end_year,
      "."
    ),
    paste0(
      "Column ",
      preferred_column,
      " is preferred. The excluded-instrument F is the squared ",
      "AEWR-region-by-subregion clustered t statistic."
    ),
    instrument_note,
    trend_note
  ),
  signif.code = significance_codes,
  file = path_tables("table_iv_preferred_first_stage.tex"),
  replace = TRUE
)

# 2SLS outcomes --------------------------------------------------------------

baseline_control_terms <- paste(
  DISSIMILARITY_IV_BASELINE_CONTROL_COLUMNS,
  collapse = " + "
)
observation_keys <- function(data, model) {
  selected <- data[fixest::obs(model), c("county_fips", "year"), drop = FALSE]
  paste(selected$county_fips, selected$year, sep = "_")
}

iv_result_rows <- list()
iv_sample_rows <- list()
iv_control_diagnostic_rows <- list()
h2a_adjustment_models <- list()

for (outcome_index in seq_len(nrow(outcomes))) {
  outcome_spec <- outcomes[outcome_index, ]
  outcome_name <- outcome_spec$outcome[[1]]
  outcome_data <- complete_cases(panel, outcome_name)

  iv_models <- map(
    model_specs$controls,
    ~ estimate_iv(outcome_data, outcome_name, model_controls(.x))
  )
  if (!same_observations(iv_models)) {
    stop(
      "The IV columns for ",
      outcome_name,
      " do not use the same observations.",
      call. = FALSE
    )
  }
  estimation_data <- outcome_data[
    fixest::obs(iv_models[[1]]),
    ,
    drop = FALSE
  ]
  preferred_iv <- iv_models[[preferred_column]]
  if (outcome_name %in% h2a_adjustment_outcomes) {
    h2a_adjustment_models[[outcome_name]] <- preferred_iv
  }

  first_stage_models_outcome <- map(
    model_specs$controls,
    ~ estimate_first_stage(estimation_data, model_controls(.x))
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
  four_control_iv <- estimate_iv(
    estimation_data,
    outcome_name,
    baseline_control_terms
  )
  four_control_first_stage <- estimate_first_stage(
    estimation_data,
    baseline_control_terms
  )
  trend_control_first_stage <- first_stage_models_outcome[[preferred_column]]
  diagnostic_observations <- list(
    four_control_iv = observation_keys(estimation_data, four_control_iv),
    trend_control_iv = observation_keys(outcome_data, preferred_iv),
    four_control_first_stage = observation_keys(
      estimation_data,
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
      length(diagnostic_observations[[1]]) != nrow(estimation_data)
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
    preferred_iv,
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
      observations = nobs(preferred_iv),
      counties = n_distinct(estimation_data$county_fips),
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
      trend_control_within_r2 = model_r2(preferred_iv, "wr2"),
      within_r2_change = trend_control_within_r2 - four_control_within_r2,
      four_control_first_stage_f =
        four_first_stage_result$t_statistic[[1]]^2,
      trend_control_first_stage_f =
        trend_first_stage_result$t_statistic[[1]]^2,
      first_stage_f_change =
        trend_control_first_stage_f - four_control_first_stage_f
    )

  for (model_index in seq_along(iv_models)) {
    iv_result_rows[[length(iv_result_rows) + 1L]] <- bind_cols(
      outcome_spec,
      model_specs[model_index, ],
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
      coefficient_row(iv_models[[model_index]], "^(fit_)?aewr_ppi$")
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

  etable(
    iv_models,
    tex = TRUE,
    title = paste0(
      "IV Effect of the Real AEWR on ",
      outcome_spec$outcome_label[[1]]
    ),
    label = paste0("tab:iv_", outcome_spec$table_stub[[1]]),
    headers = list("Specification" = model_specs$header),
    keep_raw = "^(fit_)?aewr_ppi$",
    dict = c(treatment_labels, control_labels),
    fitstat = ~ n + r2,
    extralines = list(
      "_^First-stage excluded-instrument F" = format_number(
        first_stage_f
      ),
      "_Controls" = controls_yes_no,
      "_Static propensity differential trend" = controls_yes_no,
      "_Prediction training cutoff" = rep(
        as.character(H2A_PREDICTION_CUTOFF_YEAR),
        model_count
      ),
      "_County fixed effects" = all_yes,
      "_Year fixed effects" = all_yes,
      "_AEWR-subregion clustered SEs" = all_yes,
      "_Number of SE clusters" = rep(
        as.character(inference_clusters),
        model_count
      ),
      "_Common outcome sample" = all_yes
    ),
    notes = c(
      instrument_note,
      paste0(
        "Both columns use the controlled specification's complete-case ",
        "county-year sample from ",
        DISSIMILARITY_IV_POLICY_START_YEAR,
        " through ",
        policy_end_year,
        ". Column ",
        preferred_column,
        " is preferred."
      ),
      trend_note
    ),
    signif.code = significance_codes,
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
    "county and year fixed effects, the wage-plus-seasonal-and-composition ",
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
  filter(column == preferred_column) %>%
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
  dict = c(treatment_labels, control_labels),
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
      "All columns use the wage-plus-seasonal-and-composition instrument and",
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
  signif.code = significance_codes,
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
  "h2a_linked_worksite_businesses_per_farm_worker_2011_start_year",
  "Linked worksite businesses / 2011 farm employment",
  "Linked worksite\nbusinesses /\n2011 farm emp.",
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

email_specs <- model_specs %>%
  mutate(
    file_stub = estimation_spec$models$file_stub,
    display_title = if_else(
      column == preferred_column,
      paste(header, "(preferred)"),
      header
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
  setNames(
    "OEWS-area hourly-wage plus seasonal/composition instrument",
    instrument
  ),
  control_labels,
  setNames(outcomes$outcome_label, outcomes$outcome)
)

summary_data <- panel %>%
  filter(
    year >= DISSIMILARITY_IV_POLICY_START_YEAR,
    year <= policy_end_year
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
    policy_end_year,
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
