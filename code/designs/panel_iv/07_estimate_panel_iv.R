# Purpose: Estimate the publication first stages, 2SLS outcomes, and summary
# statistics on the full county-year panel.
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
    )
  )

endogenous <- "aewr_ppi"
fixed_effects <- "county_fips + year"
cluster_vcov <- ~aewr_iv_cluster_id
control_terms <- paste(
  DISSIMILARITY_IV_CONTROL_COLUMNS,
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
  "nbr_employers_balanced_start_year"                                  ,
  "H-2A employers (balanced linkage)"                                  ,
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
  endogenous,
  "z_wage_only_real",
  "z_wage_seasonal_real",
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
    "z_wage_seasonal_real",
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
  "z_wage_seasonal_real"    ,
  "Wage + seasonal"         ,
  FALSE                     ,
  "Basic + alt. targets"    ,
  3L                        ,
  "z_wage_only_real"        ,
  "Wage only"               ,
  TRUE                      ,
  "Controls"                ,
  4L                        ,
  "z_wage_seasonal_real"    ,
  "Wage + seasonal"         ,
  TRUE                      ,
  "Controls + alt. targets"
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
  title = "First Stage: Real AEWR on Donor-Wage Instruments",
  label = "tab:iv_preferred_first_stage",
  headers = list("Specification" = first_stage_specs$header),
  keep_raw = "^z$",
  dict = c(
    z = "Instrument $Z$",
    ln_pop_census_l1 = "Lagged log population",
    farm_emp_share_l1 = "Lagged farm-employment share",
    emp_pop_ratio_l1 = "Lagged employment/population",
    wage_p10_l1 = "Lagged real p10 wage"
  ),
  fitstat = ~ n + r2,
  extralines = list(
    "_^Excluded-instrument F" = format_number(
      first_stage_results$first_stage_f
    ),
    "_Alternative seasonal targets" = c("No", "Yes", "No", "Yes"),
    "_Controls" = if_else(first_stage_specs$controls, "Yes", "No"),
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
  "z_wage_seasonal_real"    ,
  "Wage + seasonal"         ,
  FALSE                     ,
  "Alt. targets"            ,
  3L                        ,
  "z_wage_only_real"        ,
  "Wage only"               ,
  TRUE                      ,
  "Wage only + controls"    ,
  4L                        ,
  "z_wage_seasonal_real"    ,
  "Wage + seasonal"         ,
  TRUE                      ,
  "Alt. targets + controls"
)

iv_result_rows <- list()
iv_sample_rows <- list()
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
      "z_wage_seasonal_real",
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
      wage_p10_l1 = "Lagged real p10 wage"
    ),
    fitstat = ~ n + r2,
    extralines = list(
      "_^First-stage excluded-instrument F" = format_number(
        first_stage_f
      ),
      "_Alternative seasonal targets" = if_else(
        second_stage_specs$instrument == "z_wage_seasonal_real",
        "Yes",
        "No"
      ),
      "_Controls" = if_else(
        second_stage_specs$controls,
        "Yes",
        "No"
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
        "use the wage-plus-seasonal instrument."
      ),
      paste0(
        "All columns use the controlled specification's complete-case ",
        "county-year sample from ",
        DISSIMILARITY_IV_POLICY_START_YEAR,
        " through ",
        DISSIMILARITY_IV_POLICY_END_YEAR,
        ". Column 4 is preferred."
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
    wage_p10_l1 = "Lagged real p10 wage"
  ),
  fitstat = ~ n + r2,
  extralines = list(
    "_^First-stage excluded-instrument F" = format_number(
      h2a_adjustment_results$first_stage_f
    ),
    "_Conditioning" = h2a_adjustment_specs$conditioning,
    "_Controls" = rep("Yes", length(h2a_adjustment_outcomes)),
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
      "All columns use the preferred wage-plus-seasonal instrument and",
      "lagged controls."
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

# Summary statistics --------------------------------------------------------

summary_variable_labels <- c(
  aewr_ppi = "Real AEWR (2012 dollars)",
  z_wage_only_real = "Wage-only donor OEWS instrument",
  z_wage_seasonal_real = "Preferred wage-plus-seasonal donor OEWS instrument",
  ln_pop_census_l1 = "Lagged log population",
  farm_emp_share_l1 = "Lagged farm-employment share",
  emp_pop_ratio_l1 = "Lagged employment/population",
  wage_p10_l1 = "Lagged real p10 wage",
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
