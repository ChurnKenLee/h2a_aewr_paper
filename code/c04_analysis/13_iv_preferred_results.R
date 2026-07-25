# Purpose: Estimate publication first stages and preferred 2SLS outcome models.
# Input: data/processed/county_df_analysis_year_iv.parquet.
# Outputs: common-sample first-stage, 2SLS, and summary-statistics tables/CSVs.
# Run after: code/c03_iv/10_attach_instruments_to_panel.R and the C02 panel
# containing the preferred controls and Fisher quantity index.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("c00_shared", "iv_preferred_design.R"))
library(arrow)
library(tidyverse)
library(fixest)

dir.create(path_tables(), recursive = TRUE, showWarnings = FALSE)

county_df_iv <- read_parquet(path_processed(
  "county_df_analysis_year_iv.parquet"
))

iv_outcomes <- tribble(
  ~outcome, ~outcome_label, ~table_stub,
  "h2a_cert_share_farm_workers_2011_start_year",
  "H-2A certified workers / 2011 farm employment",
  "h2a_normalized",
  "fisher_index_ppi",
  "Real Fisher crop price index",
  "prices",
  "emp_farm",
  "Farm employment",
  "farm_employment",
  "share_farm_prodexp_cashandinc",
  "Farm production expenses / cash receipts and other income",
  "production_expense_share",
  "farm_cashandinc_ppi",
  "Real farm cash receipts and other income",
  "farm_income",
  "share_farm_laborexp_prodexp",
  "Hired-labor share of farm production expenses",
  "farm_labor_share",
  "fisher_quantity_index",
  "Fisher crop output-quantity index (2011 = 100)",
  "output_quantities"
)

required_columns <- unique(c(
  "year",
  "county_fe",
  "year_fe",
  "aewr_region_id",
  "aewr_ppi",
  IV_INFERENCE_CLUSTER_COLUMN,
  IV_WAGE_ONLY_INSTRUMENT,
  IV_AUXILIARY_INSTRUMENT,
  IV_CONTROL_COLUMNS,
  iv_outcomes$outcome
))

missing_columns <- setdiff(required_columns, names(county_df_iv))
if (length(missing_columns) > 0) {
  stop(
    "Preferred IV panel is missing: ",
    paste(missing_columns, collapse = ", ")
  )
}

finite_complete <- function(data, numeric_columns, id_columns) {
  data %>%
    filter(year >= IV_POLICY_START_YEAR) %>%
    filter(
      if_all(
        all_of(numeric_columns),
        ~ !is.na(.x) & is.finite(.x)
      ),
      if_all(all_of(id_columns), ~ !is.na(.x))
    )
}

cluster_vcov <- as.formula(paste0("~", IV_INFERENCE_CLUSTER_COLUMN))
fe_terms <- "county_fe + year_fe"
control_terms <- paste(IV_CONTROL_COLUMNS, collapse = " + ")

model_r2 <- function(model, type) {
  value <- tryCatch(
    as.numeric(r2(model, type)),
    error = function(e) NA_real_
  )
  if (length(value) == 0) NA_real_ else value[[1]]
}

coefficient_row <- function(model, pattern) {
  model_coeftable <- coeftable(model)
  coefficient_names <- rownames(model_coeftable)
  selected <- coefficient_names[str_detect(coefficient_names, pattern)]
  if (length(selected) != 1) {
    stop(
      "Expected one coefficient matching '",
      pattern,
      "'; found: ",
      paste(selected, collapse = ", ")
    )
  }
  row <- model_coeftable[selected, , drop = FALSE]
  tibble(
    coefficient_name = selected,
    estimate = unname(row[1, "Estimate"]),
    std_error = unname(row[1, "Std. Error"]),
    t_stat = unname(row[1, "t value"]),
    p_value = unname(row[1, "Pr(>|t|)"])
  )
}

format_table_number <- function(x, digits = 2) {
  ifelse(
    is.na(x),
    "",
    formatC(x, format = "f", digits = digits, big.mark = ",")
  )
}

# Four common-sample first stages -------------------------------------------

first_stage_numeric <- c(
  "aewr_ppi",
  IV_WAGE_ONLY_INSTRUMENT,
  IV_AUXILIARY_INSTRUMENT,
  IV_CONTROL_COLUMNS
)
first_stage_ids <- c(
  "county_fe",
  "year_fe",
  IV_INFERENCE_CLUSTER_COLUMN
)

first_stage_data <- finite_complete(
  county_df_iv,
  first_stage_numeric,
  first_stage_ids
)

first_stage_specs <- tribble(
  ~column, ~weight_spec, ~controls, ~header,
  1L, "Wage-only entropy weights", FALSE, "Basic",
  2L, "Exact seasonal FLS moments", FALSE, "Exact seasonal moments",
  3L, "Wage-only entropy weights", TRUE, "Basic + controls",
  4L, "Exact seasonal FLS moments", TRUE,
  "Exact seasonal moments + controls (preferred)"
)

first_stage_models <- vector("list", nrow(first_stage_specs))
first_stage_stats <- vector("list", nrow(first_stage_specs))

for (i in seq_len(nrow(first_stage_specs))) {
  spec <- first_stage_specs[i, ]
  instrument_column <- if (spec$weight_spec == "Wage-only entropy weights") {
    IV_WAGE_ONLY_INSTRUMENT
  } else {
    IV_AUXILIARY_INSTRUMENT
  }
  model_data <- first_stage_data %>%
    mutate(z = .data[[instrument_column]])
  rhs <- if (spec$controls) {
    paste("z", control_terms, sep = " + ")
  } else {
    "z"
  }
  model_formula <- as.formula(paste(
    "aewr_ppi ~",
    rhs,
    "|",
    fe_terms
  ))
  model <- feols(
    model_formula,
    data = model_data,
    vcov = cluster_vcov
  )
  z_row <- coefficient_row(model, "^z$")
  estimation_rows <- fixest::obs(model)
  cluster_count <- n_distinct(
    model_data[[IV_INFERENCE_CLUSTER_COLUMN]][estimation_rows]
  )

  first_stage_models[[i]] <- model
  first_stage_stats[[i]] <- bind_cols(
    spec,
    tibble(
      instrument = instrument_column,
      n = nobs(model),
      cluster_count = cluster_count,
      r2 = model_r2(model, "r2"),
      within_r2 = model_r2(model, "wr2")
    ),
    z_row,
    tibble(first_stage_f = z_row$t_stat^2)
  )
}

first_stage_stats <- bind_rows(first_stage_stats)
stopifnot(
  n_distinct(first_stage_stats$n) == 1,
  n_distinct(first_stage_stats$cluster_count) == 1
)

write_csv(
  first_stage_stats,
  path_tables("iv_preferred_first_stage_estimates.csv")
)

etable(
  first_stage_models,
  tex = TRUE,
  title = "First Stage: Real AEWR on the Preferred Donor-Wage Instruments",
  label = "tab:iv_preferred_first_stage",
  headers = first_stage_specs$header,
  keep_raw = "^z$",
  dict = c(
    "z" = "Instrument $Z$",
    "ln_pop_census_l1" = "Lagged log population",
    "farm_emp_share_l1" = "Lagged farm-employment share",
    "emp_pop_ratio_l1" = "Lagged employment/population",
    "wage_p10_l1" = "Lagged real p10 wage"
  ),
  fitstat = ~n + r2 + wr2,
  extralines = list(
    "_^Excluded-instrument F" = format_table_number(
      first_stage_stats$first_stage_f,
      2
    ),
    "_Controls" = if_else(first_stage_specs$controls, "Yes", "No"),
    "_CZ-clustered SEs" = rep(
      "Yes",
      nrow(first_stage_specs)
    ),
    "_Number of SE clusters" = as.character(
      first_stage_stats$cluster_count
    ),
    "_Common sample" = rep("Yes", nrow(first_stage_specs))
  ),
  notes = c(
    "All columns use county and year fixed effects and the same complete-case sample from 2011 onward.",
    "The preferred specification is column 4. The excluded-instrument F is the squared cluster-robust t statistic."
  ),
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
  file = path_tables("table_iv_preferred_first_stage.tex"),
  replace = TRUE
)

# Preferred 2SLS outcome models ---------------------------------------------

iv_model_rows <- list()
iv_sample_rows <- list()

for (i in seq_len(nrow(iv_outcomes))) {
  outcome_spec <- iv_outcomes[i, ]
  outcome_name <- outcome_spec$outcome
  outcome_data <- finite_complete(
    county_df_iv,
    c(
      outcome_name,
      "aewr_ppi",
      IV_AUXILIARY_INSTRUMENT,
      IV_CONTROL_COLUMNS
    ),
    first_stage_ids
  ) %>%
    mutate(z = .data[[IV_AUXILIARY_INSTRUMENT]])

  no_control_formula <- as.formula(paste(
    outcome_name,
    "~ 1 |",
    fe_terms,
    "| aewr_ppi ~ z"
  ))
  control_formula <- as.formula(paste(
    outcome_name,
    "~",
    control_terms,
    "|",
    fe_terms,
    "| aewr_ppi ~ z"
  ))

  iv_models <- list(
    feols(
      no_control_formula,
      data = outcome_data,
      vcov = cluster_vcov
    ),
    feols(
      control_formula,
      data = outcome_data,
      vcov = cluster_vcov
    )
  )
  estimation_rows <- fixest::obs(iv_models[[1]])
  stopifnot(identical(estimation_rows, fixest::obs(iv_models[[2]])))
  estimation_data <- outcome_data[estimation_rows, , drop = FALSE]

  first_stage_pair <- list(
    feols(
      as.formula(paste(
        "aewr_ppi ~ z |",
        fe_terms
      )),
      data = outcome_data,
      vcov = cluster_vcov
    ),
    feols(
      as.formula(paste(
        "aewr_ppi ~ z +",
        control_terms,
        "|",
        fe_terms
      )),
      data = outcome_data,
      vcov = cluster_vcov
    )
  )

  pair_f <- map_dbl(
    first_stage_pair,
    ~ coefficient_row(.x, "^z$")$t_stat^2
  )
  pair_clusters <- n_distinct(
    estimation_data[[IV_INFERENCE_CLUSTER_COLUMN]]
  )

  for (j in seq_along(iv_models)) {
    second_stage_row <- coefficient_row(
      iv_models[[j]],
      "^(fit_)?aewr_ppi$"
    )
    iv_model_rows[[length(iv_model_rows) + 1L]] <- bind_cols(
      outcome_spec,
      tibble(
        specification = if (j == 1) "No controls" else "Lagged controls",
        controls = j == 2,
        instrument = IV_AUXILIARY_INSTRUMENT,
        n = nobs(iv_models[[j]]),
        cluster_count = pair_clusters,
        r2 = model_r2(iv_models[[j]], "r2"),
        within_r2 = model_r2(iv_models[[j]], "wr2"),
        first_stage_f = pair_f[[j]]
      ),
      second_stage_row
    )
  }

  iv_sample_rows[[length(iv_sample_rows) + 1L]] <- tibble(
    outcome = outcome_name,
    outcome_label = outcome_spec$outcome_label,
    complete_case_n = nrow(outcome_data),
    singleton_observations_removed = nrow(outcome_data) -
      nrow(estimation_data),
    estimation_n = nrow(estimation_data),
    counties = n_distinct(estimation_data$county_fe),
    years = n_distinct(estimation_data$year),
    min_year = min(estimation_data$year),
    max_year = max(estimation_data$year),
    cluster_count = pair_clusters
  )

  stopifnot(nobs(iv_models[[1]]) == nobs(iv_models[[2]]))

  etable(
    iv_models,
    tex = TRUE,
    title = paste0(
      "IV Effect of the Real AEWR on ",
      outcome_spec$outcome_label
    ),
    label = paste0("tab:iv_", outcome_spec$table_stub),
    headers = c("No controls", "Lagged controls"),
    keep_raw = "^(fit_)?aewr_ppi$",
    dict = c(
      "fit_aewr_ppi" = "Real AEWR (2012 dollars)",
      "aewr_ppi" = "Real AEWR (2012 dollars)",
      "ln_pop_census_l1" = "Lagged log population",
      "farm_emp_share_l1" = "Lagged farm-employment share",
      "emp_pop_ratio_l1" = "Lagged employment/population",
      "wage_p10_l1" = "Lagged real p10 wage"
    ),
    fitstat = ~n + r2 + wr2,
    extralines = list(
      "_^First-stage excluded-instrument F" = format_table_number(
        pair_f,
        2
      ),
      "_Controls" = c("No", "Yes"),
      "_County fixed effects" = c("Yes", "Yes"),
      "_Year fixed effects" = c("Yes", "Yes"),
      "_CZ-clustered SEs" = c("Yes", "Yes"),
      "_Number of SE clusters" = rep(
        as.character(pair_clusters),
        2
      ),
      "_Common pair sample" = c("Yes", "Yes")
    ),
    notes = c(
      paste0(
        "Real AEWR is instrumented with the k = ",
        IV_PREFERRED_K,
        ", d = ",
        IV_PREFERRED_DONOR_COUNT,
        " donor OEWS wage using exact FLS wage and January/April/July ",
        "moment constraints with the BEA prior."
      ),
      "Both columns use the controlled specification's complete-case sample from 2011 onward."
    ),
    signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
    file = path_tables(paste0(
      "table_iv_",
      outcome_spec$table_stub,
      ".tex"
    )),
    replace = TRUE
  )
}

iv_model_results <- bind_rows(iv_model_rows)
iv_sample_audit <- bind_rows(iv_sample_rows)

stopifnot(
  all(
    iv_model_results %>%
      group_by(outcome) %>%
      summarise(n_values = n_distinct(n), .groups = "drop") %>%
      pull(n_values) == 1
  )
)

write_csv(
  iv_model_results,
  path_tables("iv_preferred_second_stage_estimates.csv")
)
write_csv(
  iv_sample_audit,
  path_tables("iv_preferred_second_stage_samples.csv")
)

# Summary statistics for every substantive regression variable -------------

summary_variable_labels <- c(
  "aewr_ppi" = "Real AEWR (2012 dollars)",
  setNames(
    "Wage-only donor OEWS instrument",
    IV_WAGE_ONLY_INSTRUMENT
  ),
  setNames(
    "Preferred exact-seasonal donor OEWS instrument",
    IV_AUXILIARY_INSTRUMENT
  ),
  "ln_pop_census_l1" = "Lagged log population",
  "farm_emp_share_l1" = "Lagged farm-employment share",
  "emp_pop_ratio_l1" = "Lagged employment/population",
  "wage_p10_l1" = "Lagged real p10 wage",
  setNames(iv_outcomes$outcome_label, iv_outcomes$outcome)
)

summary_data <- county_df_iv %>%
  filter(year >= IV_POLICY_START_YEAR)

summary_statistics <- imap_dfr(
  summary_variable_labels,
  function(label, variable) {
    values <- summary_data[[variable]]
    values <- values[is.finite(values)]
    tibble(
      variable,
      label,
      n = length(values),
      mean = mean(values),
      sd = sd(values),
      min = min(values),
      max = max(values)
    )
  }
)

write_csv(
  summary_statistics,
  path_tables("iv_preferred_summary_statistics.csv")
)

summary_tex_rows <- pmap_chr(
  summary_statistics,
  function(variable, label, n, mean, sd, min, max) {
    sprintf(
      "%s & %s & %.3f & %.3f & %.3f & %.3f \\\\",
      label,
      format(n, big.mark = ",", scientific = FALSE),
      mean,
      sd,
      min,
      max
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
  summary_tex_rows,
  "\\hline\\hline",
  "\\multicolumn{6}{p{0.95\\linewidth}}{\\footnotesize Notes: All valid county-year observations from 2011 onward. Identifiers, fixed-effect codes, and cluster codes are excluded because their numeric summaries are not substantively meaningful.} \\\\",
  "\\end{tabular}",
  "\\end{table}"
)

writeLines(
  summary_tex,
  path_tables("table_iv_preferred_summary_statistics.tex")
)

cat("Preferred first-stage sample N:", nrow(first_stage_data), "\n")
cat(
  "CZ inference clusters:",
  n_distinct(first_stage_data[[IV_INFERENCE_CLUSTER_COLUMN]]),
  "\n"
)
cat("Preferred IV result tables completed.\n")
