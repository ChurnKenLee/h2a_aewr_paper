# Purpose: Estimate first stages and 2SLS models that use the Kitagawa
# components of each retained instrument separately.
# Inputs: data/processed/panel_iv_county_year.parquet and
#   data/intermediate/panel_iv_instrument_decomposition_cluster_year.parquet.
# Outputs (outputs/tables):
#   iv_instrument_decomposition_first_stage.csv
#   iv_instrument_decomposition_variance.csv
#   iv_instrument_decomposition_second_stage.csv
#   iv_instrument_decomposition_overidentification.csv
#   table_iv_instrument_decomposition_first_stage.tex
#   table_iv_instrument_decomposition_second_stage.tex
#
# Diagnostic only: every model uses the controlled column-3/4 specification
# (controls, county and year FE, equal weights, target-subregion CRV1) on the
# subsample where the balanced decomposition exists. No result selects a model.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "specification.R"))
source(path_code("designs", "panel_iv", "decomposition.R"))
library(arrow)
library(dplyr)
library(fixest)
library(purrr)
library(readr)
library(tibble)

specification_path <- path_code("designs", "panel_iv", "estimation_spec.json")
estimation_spec <- panel_iv_read_specification(specification_path)
if (!identical(as.character(estimation_spec$controls), DISSIMILARITY_IV_CONTROL_TERMS)) {
  stop("Panel-IV control registry disagrees with design.R.", call. = FALSE)
}
endogenous <- estimation_spec$endogenous
fixed_effects <- paste(estimation_spec$fixed_effects, collapse = " + ")
inference_cluster <- estimation_spec$primary_inference_cluster
cluster_vcov <- stats::as.formula(paste("~", inference_cluster))
control_terms <- paste(estimation_spec$controls, collapse = " + ")
outcomes <- estimation_spec$outcomes

component_columns <- c(
  "z_dissimilarity_real", "z_balanced", "z_weight", "z_wage", "z_interaction",
  "z_weight_shapley", "z_wage_shapley"
)
component_labels <- c(
  z_dissimilarity_real = "Retained instrument",
  z_balanced = "Balanced-donor instrument",
  z_weight = "Weight component (wages fixed)",
  z_wage = "Wage component (weights fixed)",
  z_interaction = "Interaction component",
  z_weight_shapley = "Weight component + half interaction",
  z_wage_shapley = "Wage component + half interaction"
)

decomposition_specs <- estimation_spec$models %>%
  filter(controls) %>%
  transmute(
    column,
    retained_instrument = instrument,
    moment_targets,
    instrument_spec_label = if_else(
      instrument == "z_wage_only_real",
      DISSIMILARITY_IV_WAGE_ONLY_INSTRUMENT_LABEL,
      DISSIMILARITY_IV_PRIMARY_INSTRUMENT_LABEL
    )
  )
if (!identical(as.integer(decomposition_specs$column), 3:4)) {
  stop("The decomposition expects controlled columns 3 and 4.", call. = FALSE)
}

panel <- panel_iv_prepare_outcomes(
  read_parquet(path_processed("panel_iv_county_year.parquet")),
  estimation_spec
)
decomposition <- read_parquet(
  path_int("panel_iv_instrument_decomposition_cluster_year.parquet")
)
if (!identical(unique(decomposition$decomposition_version), IV_DECOMPOSITION_VERSION)) {
  stop("Stale instrument decomposition; rerun stage 10.", call. = FALSE)
}

finite_complete <- function(data, numeric_columns) {
  data %>%
    filter(
      year >= DISSIMILARITY_IV_POLICY_START_YEAR,
      year <= DISSIMILARITY_IV_POLICY_END_YEAR,
      if_all(all_of(numeric_columns), ~ !is.na(.x) & is.finite(.x)),
      !is.na(county_fips), !is.na(.data[[inference_cluster]])
    )
}

coefficient_rows <- function(model, terms) {
  table <- coeftable(model)
  if (!all(terms %in% rownames(table))) {
    stop("Missing coefficients: ", paste(setdiff(terms, rownames(table)), collapse = ", "),
      call. = FALSE)
  }
  tibble(
    term = terms,
    estimate = unname(table[terms, "Estimate"]),
    standard_error = unname(table[terms, "Std. Error"]),
    t_statistic = unname(table[terms, "t value"]),
    p_value = unname(table[terms, "Pr(>|t|)"])
  )
}

# Clustered Wald test of R b = 0 using the model's CRV1 covariance, referred
# to F(q, G - 1) as fixest does for clustered t statistics.
clustered_wald <- function(model, terms, restriction, clusters) {
  b <- coef(model)[terms]
  v <- vcov(model)[terms, terms, drop = FALSE]
  rb <- restriction %*% b
  statistic <- as.numeric(t(rb) %*% solve(restriction %*% v %*% t(restriction)) %*% rb) /
    nrow(restriction)
  list(
    statistic = statistic,
    p_value = stats::pf(statistic, nrow(restriction), clusters - 1L, lower.tail = FALSE)
  )
}

within_r2 <- function(model) {
  value <- tryCatch(as.numeric(r2(model, "wr2")), error = function(error) NA_real_)
  if (length(value)) value[[1]] else NA_real_
}

first_stage <- function(data, instruments) {
  feols(
    as.formula(paste(endogenous, "~", paste(c(instruments, control_terms), collapse = " + "),
      "|", fixed_effects)),
    data = data, vcov = cluster_vcov, warn = FALSE, notes = FALSE
  )
}

two_stage <- function(data, outcome, instruments) {
  feols(
    as.formula(paste(outcome, "~", control_terms, "|", fixed_effects, "|", endogenous, "~",
      paste(instruments, collapse = " + "))),
    data = data, vcov = cluster_vcov, warn = FALSE, notes = FALSE
  )
}

partial_out <- function(data, column) {
  model <- feols(
    as.formula(paste(column, "~", control_terms, "|", fixed_effects)),
    data = data, warn = FALSE, notes = FALSE
  )
  if (nobs(model) != nrow(data)) {
    stop("Partialling dropped observations for ", column, call. = FALSE)
  }
  as.numeric(resid(model))
}

first_stage_rows <- list()
variance_rows <- list()
second_stage_rows <- list()
overidentification_rows <- list()

for (spec_index in seq_len(nrow(decomposition_specs))) {
  spec <- decomposition_specs[spec_index, ]
  spec_components <- decomposition %>%
    filter(instrument_spec_label == spec$instrument_spec_label[[1]]) %>%
    transmute(
      aewr_region_id, target_cluster, year = policy_year,
      across(all_of(component_columns))
    )
  spec_panel <- panel %>%
    left_join(
      spec_components,
      by = c("aewr_region_id", "target_cluster", "year"),
      relationship = "many-to-one"
    )
  if (!isTRUE(all.equal(
    spec_panel$z_dissimilarity_real,
    spec_panel[[spec$retained_instrument[[1]]]]
  ))) {
    stop("Decomposition does not match the retained ", spec$retained_instrument[[1]],
      call. = FALSE)
  }
  base_columns <- c(
    endogenous, "z_wage_only_real", "z_wage_seasonal_composition_real",
    DISSIMILARITY_IV_CONTROL_COLUMNS
  )
  retained_sample_rows <- nrow(finite_complete(spec_panel, base_columns))

  # First stages --------------------------------------------------------------
  sample <- finite_complete(spec_panel, c(base_columns, component_columns))
  clusters <- n_distinct(sample[[inference_cluster]])
  model_terms <- list(
    retained = "z_dissimilarity_real",
    balanced = "z_balanced",
    weight = "z_weight",
    wage = "z_wage",
    interaction = "z_interaction",
    weight_shapley = "z_weight_shapley",
    wage_shapley = "z_wage_shapley",
    components_joint = c("z_weight", "z_wage", "z_interaction"),
    shapley_joint = c("z_weight_shapley", "z_wage_shapley")
  )
  for (model_name in names(model_terms)) {
    terms <- model_terms[[model_name]]
    model <- first_stage(sample, terms)
    if (nobs(model) != nrow(sample)) {
      stop("First stage ", model_name, " dropped observations.", call. = FALSE)
    }
    joint <- clustered_wald(model, terms, diag(length(terms)), clusters)
    equality <- if (length(terms) > 1L) {
      restriction <- cbind(diag(length(terms) - 1L), 0) -
        cbind(0, diag(length(terms) - 1L))
      clustered_wald(model, terms, restriction, clusters)
    } else {
      list(statistic = NA_real_, p_value = NA_real_)
    }
    first_stage_rows[[length(first_stage_rows) + 1L]] <- bind_cols(
      tibble(
        column = spec$column[[1]],
        moment_targets = spec$moment_targets[[1]],
        model = model_name
      ),
      coefficient_rows(model, terms),
      tibble(
        excluded_instrument_f = joint$statistic,
        excluded_instrument_p = joint$p_value,
        coefficient_equality_f = equality$statistic,
        coefficient_equality_p = equality$p_value,
        within_r2 = within_r2(model),
        observations = nobs(model),
        retained_sample_observations = retained_sample_rows,
        counties = n_distinct(sample$county_fips),
        inference_clusters = clusters
      )
    )
  }

  # Within variance decomposition after FE and controls ------------------------
  residuals <- sapply(
    c("z_dissimilarity_real", "z_balanced", "z_weight", "z_wage", "z_interaction"),
    function(column) partial_out(sample, column)
  )
  covariance <- crossprod(scale(residuals, scale = FALSE)) / nrow(residuals)
  balanced_variance <- covariance["z_balanced", "z_balanced"]
  pieces <- c("z_weight", "z_wage", "z_interaction")
  pairs <- as_tibble(expand.grid(left = pieces, right = pieces, stringsAsFactors = FALSE)) %>%
    filter(match(left, pieces) <= match(right, pieces))
  variance_table <- pairs %>%
    mutate(
      moment = if_else(left == right, "variance", "twice_covariance"),
      value = map2_dbl(left, right, ~ covariance[.x, .y]) * if_else(left == right, 1, 2),
      share_of_balanced_variance = value / balanced_variance,
      correlation = map2_dbl(
        left, right, ~ covariance[.x, .y] / sqrt(covariance[.x, .x] * covariance[.y, .y])
      )
    )
  if (abs(sum(variance_table$value) - balanced_variance) > 1e-8 * max(1, balanced_variance)) {
    stop("Residualized components do not reproduce the balanced variance.", call. = FALSE)
  }
  variance_rows[[spec_index]] <- bind_rows(
    variance_table,
    tibble(
      left = "z_balanced", right = "z_dissimilarity_real", moment = "retained_comparison",
      value = covariance["z_dissimilarity_real", "z_dissimilarity_real"],
      share_of_balanced_variance = value / balanced_variance,
      correlation = covariance["z_balanced", "z_dissimilarity_real"] /
        sqrt(balanced_variance * value)
    )
  ) %>%
    mutate(
      column = spec$column[[1]],
      moment_targets = spec$moment_targets[[1]],
      observations = nrow(sample),
      .before = 1
    )

  # Second stages --------------------------------------------------------------
  critical_value <- NULL
  for (outcome_index in seq_len(nrow(outcomes))) {
    outcome_name <- outcomes$outcome[[outcome_index]]
    outcome_data <- finite_complete(spec_panel, c(outcome_name, base_columns, component_columns))
    # As in stage 07, the estimation sample is the retained-instrument 2SLS
    # sample after fixest removes fixed-effect cells without variation.
    sample_model <- two_stage(outcome_data, outcome_name, "z_dissimilarity_real")
    outcome_data <- outcome_data[fixest::obs(sample_model), , drop = FALSE]
    outcome_clusters <- n_distinct(outcome_data[[inference_cluster]])
    critical_value <- stats::qt(0.975, outcome_clusters - 1L)^2
    outcome_data$ar_difference <- outcome_data[[outcome_name]] - outcome_data[[endogenous]]

    for (instrument in component_columns) {
      model <- two_stage(outcome_data, outcome_name, instrument)
      fs <- first_stage(outcome_data, instrument)
      reduced <- feols(
        as.formula(paste(outcome_name, "~", instrument, "+", control_terms, "|", fixed_effects)),
        data = outcome_data, vcov = cluster_vcov, warn = FALSE, notes = FALSE
      )
      difference <- feols(
        as.formula(paste("ar_difference ~", instrument, "+", control_terms, "|", fixed_effects)),
        data = outcome_data, vcov = cluster_vcov, warn = FALSE, notes = FALSE
      )
      if (length(unique(c(nobs(model), nobs(fs), nobs(reduced), nobs(difference),
        nrow(outcome_data)))) != 1L) {
        stop("Second-stage models for ", outcome_name, " use different samples.", call. = FALSE)
      }
      var_y <- vcov(reduced)[instrument, instrument]
      var_x <- vcov(fs)[instrument, instrument]
      cov_yx <- (var_y + var_x - vcov(difference)[instrument, instrument]) / 2
      ar <- iv_ar_confidence_set(
        coef(reduced)[[instrument]], coef(fs)[[instrument]], var_y, var_x, cov_yx,
        critical_value
      )
      fit <- coefficient_rows(model, paste0("fit_", endogenous))
      second_stage_rows[[length(second_stage_rows) + 1L]] <- tibble(
        column = spec$column[[1]],
        moment_targets = spec$moment_targets[[1]],
        outcome = outcome_name,
        outcome_label = outcomes$outcome_label[[outcome_index]],
        instrument = instrument,
        instrument_label = unname(component_labels[instrument]),
        estimate = fit$estimate,
        standard_error = fit$standard_error,
        p_value = fit$p_value,
        first_stage_f = coeftable(fs)[instrument, "t value"]^2,
        ar_lower = ar$lower,
        ar_upper = ar$upper,
        ar_set_type = ar$type,
        observations = nobs(model),
        counties = n_distinct(outcome_data$county_fips),
        inference_clusters = outcome_clusters
      )
    }

    # Overidentified 2SLS with the two Shapley components and cluster Hansen J.
    shapley <- c("z_weight_shapley", "z_wage_shapley")
    overid_model <- two_stage(outcome_data, outcome_name, shapley)
    partialled <- sapply(
      c(outcome_name, endogenous, shapley),
      function(column) partial_out(outcome_data, column)
    )
    hansen <- iv_cluster_hansen_j(
      partialled[, outcome_name], partialled[, endogenous], partialled[, shapley],
      outcome_data[[inference_cluster]]
    )
    overid_fit <- coefficient_rows(overid_model, paste0("fit_", endogenous))
    if (abs(hansen$beta_2sls - overid_fit$estimate) >
      1e-6 * max(1, abs(overid_fit$estimate))) {
      stop("Partialled 2SLS disagrees with fixest for ", outcome_name, call. = FALSE)
    }
    overidentification_rows[[length(overidentification_rows) + 1L]] <- tibble(
      column = spec$column[[1]],
      moment_targets = spec$moment_targets[[1]],
      outcome = outcome_name,
      outcome_label = outcomes$outcome_label[[outcome_index]],
      instruments = paste(shapley, collapse = "+"),
      estimate_2sls = overid_fit$estimate,
      standard_error_2sls = overid_fit$standard_error,
      p_value_2sls = overid_fit$p_value,
      estimate_two_step_gmm = hansen$beta_gmm,
      hansen_j = hansen$statistic,
      hansen_j_df = hansen$df,
      hansen_j_p = hansen$p_value,
      observations = nobs(overid_model),
      inference_clusters = outcome_clusters
    )
  }
}

first_stage_results <- bind_rows(first_stage_rows)
variance_results <- bind_rows(variance_rows)
second_stage_results <- bind_rows(second_stage_rows)
overidentification_results <- bind_rows(overidentification_rows)

write_csv(first_stage_results, path_tables("iv_instrument_decomposition_first_stage.csv"))
write_csv(variance_results, path_tables("iv_instrument_decomposition_variance.csv"))
write_csv(second_stage_results, path_tables("iv_instrument_decomposition_second_stage.csv"))
write_csv(
  overidentification_results,
  path_tables("iv_instrument_decomposition_overidentification.csv")
)

# LaTeX tables ----------------------------------------------------------------

format_value <- function(value, digits = 3L) {
  ifelse(is.finite(value), formatC(value, format = "f", digits = digits, big.mark = ","), "")
}
format_interval <- function(lower, upper, type) {
  case_when(
    type == "bounded" ~ paste0("[", format_value(lower), ", ", format_value(upper), "]"),
    type == "complement_of_interval" ~ paste0(
      "$(-\\infty$, ", format_value(lower), "] $\\cup$ [", format_value(upper), ", $\\infty)$"
    ),
    type == "unbounded" ~ "$(-\\infty, \\infty)$",
    TRUE ~ ""
  )
}
table_notes <- function(text) {
  paste0("\\begin{minipage}{0.98\\textwidth}\\footnotesize Notes: ", text, "\\end{minipage}")
}

first_stage_table_rows <- first_stage_results %>%
  mutate(
    model_label = case_when(
      model %in% c("components_joint", "shapley_joint") ~ paste0(
        if_else(model == "components_joint", "Joint components: ", "Joint Shapley: "),
        unname(component_labels[term])
      ),
      TRUE ~ unname(component_labels[term])
    ),
    row = sprintf(
      "%s & %s & %s & %s & %s & %s & %s \\\\",
      if_else(column == 3L, "Wage only", "Seasonal/composition"),
      model_label,
      format_value(estimate),
      format_value(standard_error),
      format_value(excluded_instrument_f, 2L),
      format_value(coefficient_equality_p),
      format_value(within_r2)
    )
  ) %>%
  pull(row)

first_stage_tex <- c(
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{First Stage by Kitagawa Instrument Component}",
  "\\label{tab:iv_instrument_decomposition_first_stage}",
  "\\resizebox{\\textwidth}{!}{%",
  "\\begin{tabular}{llrrrrr}",
  "\\hline\\hline",
  paste0(
    "Weights & Instrument & Coefficient & Clustered SE & Excluded-instrument F & ",
    "Equality $p$ & Within $R^2$ \\\\"
  ),
  "\\hline",
  first_stage_table_rows,
  "\\hline\\hline",
  "\\end{tabular}%",
  "}",
  table_notes(paste0(
    "Real AEWR on each component with the controlled column-3/4 specification, county ",
    "and year fixed effects and AEWR-subregion clustered standard errors on the ",
    "common balanced-donor sample (",
    format_value(first_stage_results$observations[[1]], 0L), " county-years; the ",
    "retained common sample has ",
    format_value(first_stage_results$retained_sample_observations[[1]], 0L),
    "). The weight component holds donor wages at their ",
    paste(range(IV_DECOMPOSITION_BASE_WAGE_YEARS), collapse = "--"),
    " real mean; the wage component holds calibrated weights at their source-year ",
    "mean. Calibrated weights include their intended response to wage features. ",
    "Joint rows report one F for all listed components and a clustered test of equal ",
    "coefficients."
  )),
  "\\end{table}"
)
writeLines(
  sub("[[:blank:]]+$", "", first_stage_tex),
  path_tables("table_iv_instrument_decomposition_first_stage.tex")
)

preferred_second <- second_stage_results %>%
  filter(column == 4L, instrument %in% c("z_dissimilarity_real", "z_weight", "z_wage"))
second_stage_table_rows <- map_chr(outcomes$outcome, function(outcome_name) {
  rows <- preferred_second %>% filter(outcome == outcome_name)
  cell <- function(selected) {
    row <- rows %>% filter(instrument == selected)
    paste0(format_value(row$estimate), " (", format_value(row$standard_error), ")")
  }
  ar_cell <- function(selected) {
    row <- rows %>% filter(instrument == selected)
    format_interval(row$ar_lower, row$ar_upper, row$ar_set_type)
  }
  overid <- overidentification_results %>% filter(column == 4L, outcome == outcome_name)
  sprintf(
    "%s & %s & %s & %s & %s & %s \\\\",
    rows$outcome_label[[1]],
    cell("z_dissimilarity_real"),
    cell("z_weight"),
    ar_cell("z_weight"),
    cell("z_wage"),
    format_value(overid$hansen_j_p)
  )
})
second_stage_tex <- c(
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{Preferred-Specification 2SLS by Kitagawa Instrument Component}",
  "\\label{tab:iv_instrument_decomposition_second_stage}",
  "\\resizebox{\\textwidth}{!}{%",
  "\\begin{tabular}{lrrrrr}",
  "\\hline\\hline",
  paste0(
    "Outcome & Retained $Z$ & Weight component & Weight AR 95\\% set & ",
    "Wage component & Hansen $J$ $p$ \\\\"
  ),
  "\\hline",
  second_stage_table_rows,
  "\\hline\\hline",
  "\\end{tabular}%",
  "}",
  table_notes(paste0(
    "Column-4 controls, fixed effects and AEWR-subregion clustered standard errors on ",
    "each outcome's balanced-donor sample. Each estimate is just identified by the ",
    "named instrument. The AR set inverts the clustered Anderson--Rubin test. ",
    "Hansen $J$ tests equality of the effects implied by the two Shapley components ",
    "(weight and wage components each plus half the interaction) using cluster-robust ",
    "two-step GMM."
  )),
  "\\end{table}"
)
writeLines(
  sub("[[:blank:]]+$", "", second_stage_tex),
  path_tables("table_iv_instrument_decomposition_second_stage.tex")
)

print(first_stage_results %>%
  select(column, model, term, estimate, standard_error, excluded_instrument_f,
    coefficient_equality_p, observations), n = Inf, width = Inf)
print(variance_results, n = Inf, width = Inf)
