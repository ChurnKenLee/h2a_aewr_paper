# Purpose: Promote the predeclared admissible primary specifications, compute
# their complete delta-method summaries, and render the specification curve.

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
library(fixest)
library(ggplot2)
library(readr)
library(tidyr)

fixest_threads <- mc_sp_configure_fixest_threads()

dir.create(path_tables(), recursive = TRUE, showWarnings = FALSE)
dir.create(path_figures(), recursive = TRUE, showWarnings = FALSE)

registry_bundle <- readRDS(
  path_int("mundlak_chamberlain_specification_registry.rds")
)
registry <- registry_bundle$registry
primary_selection <- read_csv(
  path_tables("mc_primary_specification_selection.csv"),
  show_col_types = FALSE
)
grid_effects <- read_csv(
  path_tables("mc_specification_effects.csv"),
  show_col_types = FALSE
)
panel_cache_directory <- file.path(
  path_int(),
  "mundlak_chamberlain_specification_panels"
)
checkpoint_directory <- file.path(
  path_int(),
  "mundlak_chamberlain_specification_checkpoints"
)

mc_sp_region_year_weights <- function(data) {
  cell <- interaction(
    data$aewr_region_id,
    data$year,
    drop = TRUE
  )
  1 / ave(rep(1, nrow(data)), cell, FUN = sum)
}

mc_sp_standardizations <- function(data, outcome_id) {
  result <- list(
    county_year_equal = rep(1, nrow(data)),
    region_year_equal = mc_sp_region_year_weights(data)
  )
  if (outcome_id %in% MC_FARM_EMPLOYMENT_SCALED_OUTCOME_IDS) {
    result$farm_employment_weighted <- data$emp_farm_2011
  } else if (identical(outcome_id, "positions_per_application")) {
    result$exposure_weighted <- data$mc_y_applications
  } else if (identical(outcome_id, "hours_per_position")) {
    result$exposure_weighted <- data$mc_y_certified_positions
  }
  result
}

mc_sp_primary_effect <- function(
  model,
  data,
  checkpoint,
  outcome_specification,
  treatment_column,
  estimand,
  dose_change = NULL,
  subset = NULL,
  weights = NULL,
  normalize = TRUE,
  standardization = "county_year_equal",
  scope = "overall",
  scope_value = "all"
) {
  gradient <- mc_sp_gradient(
    model = model,
    data = data,
    treatment_column = treatment_column,
    dose_change = dose_change,
    derivative = identical(
      estimand,
      "average_marginal_effect"
    ),
    subset = subset,
    weights = weights,
    normalize = normalize,
    causal_dictionary = checkpoint$causal_dictionary
  )
  projected <- mc_sp_project_effect(
    gradient = gradient,
    coefficients = checkpoint$coefficients,
    state_errors = checkpoint$state_errors,
    observations = checkpoint$diagnostics$observations[[1]],
    effective_parameters =
      checkpoint$diagnostics$effective_parameters[[1]]
  )
  multiplier <- if (
    identical(
      outcome_specification$effect_unit[[1]],
      "probability"
    )
  ) {
    100
  } else {
    1
  }
  data.frame(
    outcome_id = outcome_specification$outcome_id[[1]],
    outcome_label = outcome_specification$outcome_label[[1]],
    model_id = MC_PRIMARY_MODEL_ID,
    spec_id = checkpoint$spec_id,
    horizon = names(MC_DYNAMIC_HORIZONS)[
      match(treatment_column, unname(MC_DYNAMIC_HORIZONS))
    ],
    treatment_column = treatment_column,
    estimand = estimand,
    dose_change = if (is.null(dose_change)) {
      NA_real_
    } else {
      dose_change
    },
    standardization = standardization,
    scope = scope,
    scope_value = as.character(scope_value),
    estimate = projected$estimate * multiplier,
    raw_standard_error =
      projected$raw_standard_error * multiplier,
    standard_error =
      projected$standard_error * multiplier,
    conf_low = projected$conf_low * multiplier,
    conf_high = projected$conf_high * multiplier,
    reported_unit = if (multiplier == 100) {
      "percentage_points"
    } else {
      outcome_specification$effect_unit[[1]]
    },
    observations = sum(
      if (is.null(subset)) rep(TRUE, nrow(data)) else subset
    ),
    weight_sum = sum(
      if (is.null(weights)) {
        if (is.null(subset)) {
          rep(1, nrow(data))
        } else {
          as.numeric(subset)
        }
      } else if (is.null(subset)) {
        weights
      } else {
        weights * as.numeric(subset)
      }
    ),
    normalized = normalize,
    variance_method = MC_CCV_METHOD,
    reference_design = MC_CCV_REFERENCE_DESIGN,
    inference_df = MC_CCV_DF,
    df_multiplier = projected$df_multiplier,
    stringsAsFactors = FALSE
  )
}

finite_rows <- list()
ame_rows <- list()
year_rows <- list()
heterogeneity_rows <- list()
parameter_rows <- list()
gradient_audit_rows <- list()

for (selection_index in seq_len(nrow(primary_selection))) {
  selected <- primary_selection[
    selection_index,
    ,
    drop = FALSE
  ]
  outcome_specification <- MC_OUTCOMES[
    MC_OUTCOMES$outcome_id == selected$outcome_id[[1]],
    ,
    drop = FALSE
  ]
  registry_row <- registry[
    registry$spec_id == selected$spec_id[[1]],
    ,
    drop = FALSE
  ]
  specification <- mc_sp_specification(registry_row)
  panel_bundle <- readRDS(
    file.path(
      panel_cache_directory,
      paste0(specification$calendar_id, ".rds")
    )
  )
  checkpoint <- readRDS(
    file.path(
      checkpoint_directory,
      specification$spec_id,
      paste0(selected$outcome_id[[1]], ".rds")
    )
  )
  if (!identical(checkpoint$status, "estimated")) {
    stop("A selected primary checkpoint is not estimated.", call. = FALSE)
  }

  model_data <- mc_sp_apply_sample_rule(
    panel_bundle$panel,
    outcome_specification$sample_rule[[1]]
  )
  outcome_column <- outcome_specification$outcome_column[[1]]
  model_data <- model_data[
    is.finite(model_data[[outcome_column]]),
    ,
    drop = FALSE
  ]
  formula_bundle <- mc_sp_formula_bundle(
    outcome = outcome_column,
    specification = specification,
    metadata = panel_bundle$metadata,
    history_selection = checkpoint$history_selection,
    excluded_terms = checkpoint$compiler_pruned_terms
  )
  model <- mc_sp_fit_ols(formula_bundle$formula, model_data)
  estimation_data <- model_data[
    fixest::obs(model),
    ,
    drop = FALSE
  ]
  if (
    !identical(
      names(stats::coef(model)),
      names(checkpoint$coefficients)
    ) ||
      max(
        abs(
          stats::coef(model) -
            checkpoint$coefficients
        )
      ) > 1e-7
  ) {
    stop("Primary refit does not match its checkpoint.", call. = FALSE)
  }

  # The formula builder and delta method must resolve the same named causal
  # basis.  Audit a compact sample retaining every region-year cell; because
  # the fitted model is linear in its generated basis, the two gradients must
  # agree to machine precision for a finite dose change.
  validation_index <- !duplicated(
    paste(
      estimation_data$aewr_region_id,
      estimation_data$year,
      sep = ":"
    )
  )
  validation_data <- estimation_data[
    validation_index,
    ,
    drop = FALSE
  ]
  analytic_gradient <- mc_sp_gradient(
    model,
    validation_data,
    treatment_column = "mc_dose_current",
    dose_change = 5,
    causal_dictionary = formula_bundle$causal_dictionary
  )
  counterfactual_data <- mc_counterfactual_data(
    validation_data,
    "mc_dose_current",
    5
  )
  direct_gradient <- colMeans(
    mc_sp_model_matrix(model, counterfactual_data) -
      mc_sp_model_matrix(model, validation_data)
  )
  causal_names <- formula_bundle$causal_dictionary$coefficient_name
  gradient_audit_rows[[length(gradient_audit_rows) + 1L]] <-
    data.frame(
      outcome_id = selected$outcome_id[[1]],
      spec_id = checkpoint$spec_id,
      validation_rows = nrow(validation_data),
      causal_dictionary_terms = length(causal_names),
      resolved_causal_terms = sum(
        causal_names %in% names(stats::coef(model))
      ),
      unresolved_causal_terms = sum(
        !causal_names %in% names(stats::coef(model))
      ),
      unexpected_causal_terms = length(setdiff(
        grep("^year::", names(stats::coef(model)), value = TRUE),
        causal_names
      )),
      maximum_gradient_error = max(abs(
        analytic_gradient - direct_gradient
      )),
      estimate_error = abs(
        sum(analytic_gradient * stats::coef(model)) -
          sum(direct_gradient * stats::coef(model))
      ),
      stringsAsFactors = FALSE
    )

  standardizations <- mc_sp_standardizations(
    estimation_data,
    selected$outcome_id[[1]]
  )
  treatment_columns <- unname(MC_DYNAMIC_HORIZONS)[
    seq_len(specification$horizon_count)
  ]
  for (treatment_column in treatment_columns) {
    for (dose_change in MC_COUNTERFACTUAL_DOSES) {
      for (standardization in names(standardizations)) {
        finite_rows[[length(finite_rows) + 1L]] <-
          mc_sp_primary_effect(
            model,
            estimation_data,
            checkpoint,
            outcome_specification,
            treatment_column,
            "finite_dose_change",
            dose_change = dose_change,
            weights = standardizations[[standardization]],
            standardization = standardization
          )
      }
      if (selected$outcome_id[[1]] %in% MC_VOLUME_OUTCOME_IDS) {
        conversion_weights <- if (
          identical(selected$outcome_id[[1]], "employers")
        ) {
          rep(1, nrow(estimation_data))
        } else if (
          identical(selected$outcome_id[[1]], "certified_hours")
        ) {
          estimation_data$emp_farm_2011
        } else {
          estimation_data$emp_farm_2011 / 1000
        }
        total <- mc_sp_primary_effect(
          model,
          estimation_data,
          checkpoint,
          outcome_specification,
          treatment_column,
          "finite_dose_change",
          dose_change = dose_change,
          weights = conversion_weights,
          normalize = FALSE,
          standardization = "sample_period_total"
        )
        total$reported_unit <- if (
          identical(selected$outcome_id[[1]], "certified_hours")
        ) {
          "certified_hours"
        } else {
          sub(
            "_per_1000$",
            "",
            outcome_specification$effect_unit[[1]]
          )
        }
        finite_rows[[length(finite_rows) + 1L]] <- total
      }
    }

    for (standardization in names(standardizations)) {
      ame_rows[[length(ame_rows) + 1L]] <-
        mc_sp_primary_effect(
          model,
          estimation_data,
          checkpoint,
          outcome_specification,
          treatment_column,
          "average_marginal_effect",
          weights = standardizations[[standardization]],
          standardization = standardization
        )
    }
    for (effect_year in specification$analysis_years) {
      year_subset <- estimation_data$year == effect_year
      year_rows[[length(year_rows) + 1L]] <-
        mc_sp_primary_effect(
          model,
          estimation_data,
          checkpoint,
          outcome_specification,
          treatment_column,
          "finite_dose_change",
          dose_change = 5,
          subset = year_subset,
          standardization = "county_year_equal",
          scope = "year",
          scope_value = effect_year
        )
      ame_rows[[length(ame_rows) + 1L]] <-
        mc_sp_primary_effect(
          model,
          estimation_data,
          checkpoint,
          outcome_specification,
          treatment_column,
          "average_marginal_effect",
          subset = year_subset,
          standardization = "county_year_equal",
          scope = "year",
          scope_value = effect_year
        )
    }
    for (dimension in c(
      "mc_binding_quartile",
      "mc_baseline_h2a_quartile"
    )) {
      for (quartile in 1:4) {
        group_subset <-
          estimation_data[[dimension]] == quartile
        heterogeneity_rows[[length(heterogeneity_rows) + 1L]] <-
          mc_sp_primary_effect(
            model,
            estimation_data,
            checkpoint,
            outcome_specification,
            treatment_column,
            "finite_dose_change",
            dose_change = 5,
            subset = group_subset,
            standardization = "county_year_equal",
            scope = dimension,
            scope_value = quartile
          )
        ame_rows[[length(ame_rows) + 1L]] <-
          mc_sp_primary_effect(
            model,
            estimation_data,
            checkpoint,
            outcome_specification,
            treatment_column,
            "average_marginal_effect",
            subset = group_subset,
            standardization = "county_year_equal",
            scope = dimension,
            scope_value = quartile
          )
      }
    }
  }

  parameter_rows[[length(parameter_rows) + 1L]] <- data.frame(
    outcome_id = selected$outcome_id[[1]],
    model_id = MC_PRIMARY_MODEL_ID,
    spec_id = checkpoint$spec_id,
    variance_method = MC_CCV_METHOD,
    term = names(checkpoint$coefficients),
    estimate = unname(checkpoint$coefficients),
    raw_standard_error = unname(
      checkpoint$raw_parameter_standard_error
    ),
    standard_error = unname(
      checkpoint$parameter_standard_error
    ),
    statistic = unname(
      checkpoint$coefficients /
        checkpoint$parameter_standard_error
    ),
    p_value = 2 * stats::pt(
      -abs(
        checkpoint$coefficients /
          checkpoint$parameter_standard_error
      ),
      df = MC_CCV_DF
    ),
    conventional_cluster_standard_error =
      checkpoint$conventional_cluster_standard_error,
    stringsAsFactors = FALSE
  )
}

finite_effects <- bind_rows(finite_rows)
average_marginal_effects <- bind_rows(ame_rows)
year_effects <- bind_rows(year_rows)
heterogeneity_effects <- bind_rows(heterogeneity_rows)
primary_parameters <- bind_rows(parameter_rows)
gradient_audit <- bind_rows(gradient_audit_rows)

write_csv(
  finite_effects,
  path_tables("mc_finite_dose_effects.csv")
)
write_csv(
  average_marginal_effects,
  path_tables("mc_average_marginal_effects.csv")
)
write_csv(
  year_effects,
  path_tables("mc_year_effects.csv")
)
write_csv(
  heterogeneity_effects,
  path_tables("mc_heterogeneity_effects.csv")
)
write_csv(
  gradient_audit,
  path_tables("mc_delta_gradient_audit.csv")
)

parameter_path <- path_tables("mc_parameter_estimates.csv")
benchmark_parameters <- if (file.exists(parameter_path)) {
  read_csv(parameter_path, show_col_types = FALSE) |>
    filter(
      .data$model_id %in%
        c("twfe_benchmark", "mundlak_multilevel")
    )
} else {
  tibble()
}
write_csv(
  bind_rows(benchmark_parameters, primary_parameters),
  parameter_path
)

curve_data <- grid_effects |>
  filter(
    .data$estimand == "average_marginal_effect",
    .data$standardization == "sample_average"
  ) |>
  left_join(
    registry |>
      select(
        .data$spec_id,
        .data$horizon_count,
        .data$polynomial_degree,
        .data$richness_tier,
        .data$richness_label,
        .data$preperiod_start,
        .data$preperiod_end,
        .data$analysis_start
      ),
    by = "spec_id"
  ) |>
  left_join(
    primary_selection |>
      select(.data$outcome_id, primary_spec_id = .data$spec_id),
    by = "outcome_id"
  ) |>
  group_by(.data$outcome_id, .data$treatment_column) |>
  arrange(.data$estimate, .by_group = TRUE) |>
  mutate(
    specification_rank = dplyr::row_number(),
    primary = .data$spec_id == .data$primary_spec_id
  ) |>
  ungroup()

write_csv(
  curve_data,
  path_tables("mc_specification_curve_data.csv")
)
if (nrow(curve_data) > 0L) {
  curve_plot <- ggplot(
    curve_data,
    aes(
      x = specification_rank,
      y = estimate,
      color = factor(richness_tier)
    )
  ) +
    geom_hline(yintercept = 0, color = "grey75") +
    geom_linerange(
      aes(ymin = conf_low, ymax = conf_high),
      alpha = 0.16,
      linewidth = 0.2
    ) +
    geom_point(size = 0.7, alpha = 0.75) +
    geom_point(
      data = curve_data |> filter(.data$primary),
      shape = 21,
      fill = "gold",
      color = "black",
      size = 2.1,
      stroke = 0.5
    ) +
    facet_grid(
      outcome_id ~ horizon,
      scales = "free_y"
    ) +
    scale_color_viridis_d(
      name = "Richness tier",
      option = "C",
      end = 0.9
    ) +
    labs(
      x = "Admissible specifications ordered by AME",
      y = "Average marginal effect",
      title = "Mundlak–Chamberlain specification curve",
      subtitle = paste(
        "Adjusted finite-design CCV intervals;",
        "gold point is the predeclared primary"
      )
    ) +
    theme_minimal(base_size = 9) +
    theme(
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      strip.text.y = element_text(angle = 0)
    )
  ggsave(
    path_figures("fig_mc_specification_curve.png"),
    curve_plot,
    width = 13,
    height = 18,
    dpi = 300
  )
}

message(
  "Promoted ",
  nrow(primary_selection),
  " outcome-specific primary specifications."
)
