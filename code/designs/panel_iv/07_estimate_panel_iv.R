# Purpose: Estimate the retained four-specification panel IV.
# Outputs: estimate and AR CSVs, four-column table, and first-stage figure.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "helpers.R"))
library(arrow)
library(dplyr)
library(fixest)
library(ggplot2)
library(purrr)
library(readr)
library(tibble)

panel <- read_parquet(
  path_processed("panel_iv_cluster_year.parquet")
)

specifications <- tibble(
  specification = c(
    "primary_levels",
    "levels_controls",
    "levels_no_border",
    "census_frame_benchmark"
  ),
  specification_label = c(
    "Real AEWR",
    "Real AEWR + lagged controls",
    "Real AEWR, no border CZs",
    "Real AEWR, Census-frame benchmark"
  ),
  instrument = c(
    "z_dissimilarity_real",
    "z_dissimilarity_real",
    "z_dissimilarity_real",
    "z_dissimilarity_census_frame_real"
  ),
  controls = list(
    character(),
    DISSIMILARITY_IV_CONTROL_COLUMNS,
    character(),
    character()
  ),
  no_border_only = c(FALSE, FALSE, TRUE, FALSE)
)

outcome <- "h2a_cert_share_farm_workers_2011"
endogenous <- "aewr_ppi"
fixed_effects <- "aewr_iv_cluster_id + policy_year"
cluster_vcov <- ~aewr_region_id

coefficient_row <- function(model, coefficient_pattern) {
  table <- coeftable(model)
  selected <- grep(
    coefficient_pattern,
    rownames(table),
    value = TRUE
  )
  if (length(selected) != 1L) {
    stop("Expected exactly one model coefficient.", call. = FALSE)
  }
  row <- table[selected, , drop = FALSE]
  tibble(
    estimate = unname(row[1, "Estimate"]),
    standard_error = unname(row[1, "Std. Error"]),
    t_statistic = unname(row[1, "t value"]),
    p_value = unname(row[1, "Pr(>|t|)"])
  )
}

make_rhs <- function(controls) {
  if (length(controls) == 0L) {
    return("1")
  }
  paste(controls, collapse = " + ")
}

residualize_variable <- function(data, variable, rhs) {
  formula <- as.formula(paste(
    variable,
    "~",
    rhs,
    "|",
    fixed_effects
  ))
  model <- feols(
    formula,
    data = data,
    warn = FALSE,
    notes = FALSE
  )
  as.numeric(resid(model))
}

complete_spec_data <- function(data, specification) {
  numeric_columns <- c(
    outcome,
    endogenous,
    specification$instrument[[1]],
    specification$controls[[1]]
  )
  result <- data |>
    filter(
      if_all(all_of(numeric_columns), is.finite),
      !is.na(aewr_region_id),
      !is.na(aewr_iv_cluster_id),
      !is.na(policy_year)
    )
  if (specification$no_border_only[[1]]) {
    result <- result |>
      filter(no_border_cluster)
  }
  result
}

model_rows <- vector("list", nrow(specifications))
ar_interval_rows <- vector("list", nrow(specifications))
iv_models <- list()
residual_plot_data <- NULL

for (index in seq_len(nrow(specifications))) {
  specification <- specifications[index, ]
  instrument <- specification$instrument[[1]]
  controls <- specification$controls[[1]]
  rhs <- make_rhs(controls)
  data <- complete_spec_data(panel, specification)

  first_stage <- feols(
    as.formula(paste(
      endogenous,
      "~",
      paste(instrument, rhs, sep = " + "),
      "|",
      fixed_effects
    )),
    data = data,
    vcov = cluster_vcov,
    warn = FALSE,
    notes = FALSE
  )
  reduced_form <- feols(
    as.formula(paste(
      outcome,
      "~",
      paste(instrument, rhs, sep = " + "),
      "|",
      fixed_effects
    )),
    data = data,
    vcov = cluster_vcov,
    warn = FALSE,
    notes = FALSE
  )
  iv_model <- feols(
    as.formula(paste(
      outcome,
      "~",
      rhs,
      "|",
      fixed_effects,
      "|",
      endogenous,
      "~",
      instrument
    )),
    data = data,
    vcov = cluster_vcov,
    warn = FALSE,
    notes = FALSE
  )

  estimation_data <- data[fixest::obs(iv_model), , drop = FALSE]
  first_stage_result <- coefficient_row(
    first_stage,
    paste0("^", instrument, "$")
  )
  reduced_form_result <- coefficient_row(
    reduced_form,
    paste0("^", instrument, "$")
  )
  iv_result <- coefficient_row(
    iv_model,
    paste0("^(fit_)?", endogenous, "$")
  )

  y_tilde <- residualize_variable(estimation_data, outcome, rhs)
  d_tilde <- residualize_variable(estimation_data, endogenous, rhs)
  z_tilde <- residualize_variable(
    estimation_data,
    instrument,
    rhs
  )
  inference_cluster <- estimation_data$aewr_region_id

  first_stage_wild <- wild_cluster_score_test(
    y = d_tilde,
    z = z_tilde,
    cluster = inference_cluster,
    bootstrap_reps = DISSIMILARITY_IV_BOOTSTRAP_REPS,
    seed = DISSIMILARITY_IV_BOOTSTRAP_SEED + index
  )
  reduced_form_wild <- wild_cluster_score_test(
    y = y_tilde,
    z = z_tilde,
    cluster = inference_cluster,
    bootstrap_reps = DISSIMILARITY_IV_BOOTSTRAP_REPS,
    seed = DISSIMILARITY_IV_BOOTSTRAP_SEED + 100L + index
  )

  beta_grid <- make_ar_beta_grid(
    y = y_tilde,
    endogenous = d_tilde,
    center = iv_result$estimate[[1]],
    points = DISSIMILARITY_IV_AR_GRID_POINTS
  )
  ar_grid <- anderson_rubin_grid(
    y = y_tilde,
    endogenous = d_tilde,
    instrument = z_tilde,
    cluster = inference_cluster,
    beta_grid = beta_grid,
    bootstrap_reps = DISSIMILARITY_IV_BOOTSTRAP_REPS,
    seed = DISSIMILARITY_IV_BOOTSTRAP_SEED + 200L + index
  )
  ar_intervals <- accepted_grid_intervals(
    ar_grid$beta_null,
    ar_grid$ar_bootstrap_p_value,
    level = DISSIMILARITY_IV_AR_LEVEL
  ) |>
    as_tibble() |>
    mutate(
      specification = specification$specification[[1]],
      confidence_level = DISSIMILARITY_IV_AR_LEVEL,
      .before = 1
    )

  model_rows[[index]] <- bind_cols(
    specification |>
      select(
        specification,
        specification_label,
        instrument,
        no_border_only
      ) |>
      mutate(controls = paste(controls, collapse = ",")),
    tibble(
      outcome = outcome,
      endogenous = endogenous,
      observations = nobs(iv_model),
      regions = n_distinct(inference_cluster),
      target_clusters = n_distinct(
        estimation_data$aewr_iv_cluster_id
      ),
      minimum_year = min(estimation_data$policy_year),
      maximum_year = max(estimation_data$policy_year),
      first_stage_estimate = first_stage_result$estimate,
      first_stage_standard_error = first_stage_result$standard_error,
      first_stage_t = first_stage_result$t_statistic,
      first_stage_f = first_stage_result$t_statistic^2,
      first_stage_p = first_stage_result$p_value,
      first_stage_wild_p = first_stage_wild$bootstrap_p_value,
      reduced_form_estimate = reduced_form_result$estimate,
      reduced_form_standard_error = reduced_form_result$standard_error,
      reduced_form_t = reduced_form_result$t_statistic,
      reduced_form_p = reduced_form_result$p_value,
      reduced_form_wild_p = reduced_form_wild$bootstrap_p_value,
      iv_estimate = iv_result$estimate,
      iv_standard_error = iv_result$standard_error,
      iv_t = iv_result$t_statistic,
      iv_p = iv_result$p_value,
      ar_interval_count = nrow(ar_intervals),
      ar_grid_lower = min(beta_grid),
      ar_grid_upper = max(beta_grid),
      ar_grid_boundary_accepted = any(
        ar_intervals$lower_hits_grid |
          ar_intervals$upper_hits_grid
      )
    )
  )
  ar_interval_rows[[index]] <- ar_intervals
  iv_models[[specification$specification[[1]]]] <- iv_model

  if (specification$specification[[1]] == "primary_levels") {
    residual_plot_data <- estimation_data |>
      transmute(
        aewr_region_id,
        residualized_instrument = z_tilde,
        residualized_aewr = d_tilde
      )
  }
}

model_results <- bind_rows(model_rows)
ar_interval_results <- bind_rows(ar_interval_rows)

write_csv(
  model_results,
  path_tables("iv_dissimilarity_model_estimates.csv")
)
write_csv(
  ar_interval_results,
  path_tables("iv_dissimilarity_ar_intervals.csv")
)

format_ar_set <- function(specification) {
  intervals <- ar_interval_results |>
    filter(.data$specification == .env$specification)
  if (nrow(intervals) == 0L) {
    return("empty on grid")
  }
  pieces <- map_chr(seq_len(nrow(intervals)), \(row) {
    lower <- paste0(
      if (intervals$lower_hits_grid[[row]]) "edge " else "",
      sprintf("%.2f", intervals$lower[[row]])
    )
    upper <- paste0(
      sprintf("%.2f", intervals$upper[[row]]),
      if (intervals$upper_hits_grid[[row]]) " edge" else ""
    )
    paste0("[", lower, ", ", upper, "]")
  })
  paste(pieces, collapse = "; ")
}

ar_set_labels <- map_chr(
  model_results$specification,
  format_ar_set
)

first_stage_figure <- ggplot(
  residual_plot_data,
  aes(
    x = residualized_instrument,
    y = residualized_aewr,
    color = aewr_region_id
  )
) +
  geom_point(alpha = 0.55, size = 1.5) +
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = FALSE,
    color = "black",
    linewidth = 0.8
  ) +
  labs(
    x = "Residualized donor OEWS wage",
    y = "Residualized real AEWR",
    color = "AEWR region",
    title = "Dissimilarity-cluster panel-IV first stage",
    subtitle = "Target-cluster and year fixed effects"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank()
  )

ggsave(
  path_figures("fig_iv_dissimilarity_first_stage.png"),
  first_stage_figure,
  width = 8,
  height = 5.5,
  dpi = 300
)

etable(
  iv_models,
  tex = TRUE,
  title = "Panel IV Using FLS-Moment-Weighted Dissimilar Donors",
  label = "tab:iv_dissimilarity_panel",
  headers = list(
    "Specification" = c(
      "Levels",
      "Controls",
      "No border",
      "Census frame"
    )
  ),
  dict = c(
    h2a_cert_share_farm_workers_2011 = "H-2A certifications / 2011 farm employment",
    aewr_ppi = "Real AEWR",
    aewr_iv_cluster_id = "Target-cluster fixed effects",
    policy_year = "Year fixed effects"
  ),
  keep_raw = "^(fit_)?aewr_ppi$",
  signif.code = NA,
  fontsize = "scriptsize",
  fitstat = ~n,
  extralines = list(
    "_^First-stage cluster-robust F statistic" = sprintf(
      "%.2f",
      model_results$first_stage_f
    ),
    "_First-stage wild-cluster p-value" = sprintf(
      "%.3f",
      model_results$first_stage_wild_p
    ),
    "_Reduced-form wild-cluster p-value" = sprintf(
      "%.3f",
      model_results$reduced_form_wild_p
    ),
    "_95% Anderson--Rubin set" = ar_set_labels,
    "_AEWR-region clustered SEs" = rep("Yes", 4),
    "_Number of AEWR regions" = as.character(model_results$regions)
  ),
  notes = c(
    paste(
      paste0(
        "Primary weights target published FLS worker-composition ",
        "and field-and-livestock wage moments;"
      ),
      "the final column uses the Census hired-worker frame."
    ),
    paste(
      "Webb six-point wild-cluster tests and Anderson--Rubin sets",
      "cluster at the AEWR-region level."
    )
  ),
  file = path_tables("table_iv_dissimilarity_panel.tex"),
  replace = TRUE
)
