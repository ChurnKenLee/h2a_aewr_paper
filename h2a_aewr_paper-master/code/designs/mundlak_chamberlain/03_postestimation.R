# Purpose: Convert the rich master-model coefficient vector into interpretable
# finite dose effects and average marginal effects using the delta method.
# Inputs:
#   data/processed/mundlak_chamberlain_county_year.parquet
#   data/intermediate/mundlak_chamberlain_models.rds
# Outputs:
#   outputs/tables/mc_finite_dose_effects.csv
#   outputs/tables/mc_average_marginal_effects.csv
#   outputs/tables/mc_year_effects.csv
#   outputs/tables/mc_heterogeneity_effects.csv
#   outputs/tables/mc_ame_grid.csv
#   outputs/figures/fig_mc_ccv_coefficients.png
#   outputs/figures/fig_mc_dynamic_effects.png
#   outputs/figures/fig_mc_year_effects.png
#   outputs/figures/fig_mc_heterogeneity.png

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "mundlak_chamberlain", "design.R"))
source(path_code("designs", "mundlak_chamberlain", "helpers.R"))
library(arrow)
library(dplyr)
library(ggplot2)
library(purrr)
library(readr)
library(tibble)
library(tidyr)

dir.create(path_tables(), recursive = TRUE, showWarnings = FALSE)
dir.create(path_figures(), recursive = TRUE, showWarnings = FALSE)

panel <- read_parquet(
  path_processed("mundlak_chamberlain_county_year.parquet")
) %>%
  as.data.frame()
bundle <- readRDS(
  path_int("mundlak_chamberlain_models.rds")
)
metadata <- bundle$metadata

if (!identical(bundle$design_version, metadata$design_version)) {
  stop("MC model and metadata versions differ.", call. = FALSE)
}

mc_estimation_data <- function(outcome_id, model_id) {
  row_ids <- bundle$sample_row_ids[[outcome_id]][[model_id]]
  row_index <- match(row_ids, panel$mc_row_id)
  if (anyNA(row_index)) {
    stop("Stored MC sample rows are absent from the panel.", call. = FALSE)
  }
  panel[row_index, , drop = FALSE]
}

mc_region_year_weights <- function(data) {
  cell <- interaction(
    data$aewr_region_id,
    data$year,
    drop = TRUE
  )
  1 / ave(rep(1, nrow(data)), cell, FUN = sum)
}

mc_standardizations <- function(data, outcome_id) {
  standardizations <- list(
    county_year_equal = rep(1, nrow(data)),
    region_year_equal = mc_region_year_weights(data)
  )
  if (
    outcome_id %in% MC_FARM_EMPLOYMENT_SCALED_OUTCOME_IDS
  ) {
    standardizations$farm_employment_weighted <- data$emp_farm_2011
  } else if (identical(outcome_id, "positions_per_application")) {
    standardizations$exposure_weighted <- data$mc_y_applications
  } else if (identical(outcome_id, "hours_per_position")) {
    standardizations$exposure_weighted <-
      data$mc_y_certified_positions
  }
  standardizations
}

mc_scale_effect <- function(result, outcome_specification) {
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
  result %>%
    mutate(
      across(
        c(estimate, standard_error, conf_low, conf_high),
        ~ .x * multiplier
      ),
      reported_unit = if (multiplier == 100) {
        "percentage_points"
      } else {
        outcome_specification$effect_unit[[1]]
      }
    )
}

finite_rows <- list()
ame_rows <- list()
year_rows <- list()
heterogeneity_rows <- list()
grid_rows <- list()

treatment_cell_panel <- panel %>%
  distinct(
    aewr_region_id,
    year,
    across(all_of(unname(MC_DYNAMIC_HORIZONS)))
  )
treatment_quantiles <- lapply(
  unname(MC_DYNAMIC_HORIZONS),
  function(column) {
    stats::quantile(
      treatment_cell_panel[[column]],
      probs = c(0.10, 0.50, 0.90),
      na.rm = TRUE,
      names = FALSE
    )
  }
)
names(treatment_quantiles) <- unname(MC_DYNAMIC_HORIZONS)

for (outcome_index in seq_len(nrow(metadata$outcomes))) {
  outcome_specification <- metadata$outcomes[outcome_index, ]
  outcome_id <- outcome_specification$outcome_id[[1]]
  outcome_label <- outcome_specification$outcome_label[[1]]
  model <- bundle$models[[outcome_id]][[MC_PRIMARY_MODEL_ID]]
  estimation_data <- mc_estimation_data(
    outcome_id,
    MC_PRIMARY_MODEL_ID
  )
  standardizations <- mc_standardizations(
    estimation_data,
    outcome_id
  )

  for (horizon_name in names(MC_DYNAMIC_HORIZONS)) {
    treatment_column <- MC_DYNAMIC_HORIZONS[[horizon_name]]

    for (dose_change in MC_COUNTERFACTUAL_DOSES) {
      for (standardization in names(standardizations)) {
        result <- mc_master_sample_effect(
          model = model,
          data = estimation_data,
          treatment_column = treatment_column,
          dose_change = dose_change,
          weights = standardizations[[standardization]],
          normalize = TRUE,
          cluster_df = metadata$cluster_df
        ) %>%
          mc_scale_effect(outcome_specification) %>%
          mutate(
            outcome_id = outcome_id,
            outcome_label = outcome_label,
            horizon = horizon_name,
            treatment_column = treatment_column,
            standardization = standardization,
            .before = 1
          )
        finite_rows[[length(finite_rows) + 1L]] <- result
      }

      if (
        outcome_id %in% MC_VOLUME_OUTCOME_IDS
      ) {
        conversion_weights <- if (identical(outcome_id, "employers")) {
          rep(1, nrow(estimation_data))
        } else if (identical(outcome_id, "certified_hours")) {
          estimation_data$emp_farm_2011
        } else {
          estimation_data$emp_farm_2011 / 1000
        }
        total_result <- mc_master_sample_effect(
          model = model,
          data = estimation_data,
          treatment_column = treatment_column,
          dose_change = dose_change,
          weights = conversion_weights,
          normalize = FALSE,
          cluster_df = metadata$cluster_df
        ) %>%
          mutate(
            reported_unit = if (identical(outcome_id, "certified_hours")) {
              "certified_hours"
            } else {
              sub("_per_1000$", "", outcome_specification$effect_unit[[1]])
            },
            outcome_id = outcome_id,
            outcome_label = outcome_label,
            horizon = horizon_name,
            treatment_column = treatment_column,
            standardization = "sample_period_total",
            .before = 1
          )
        finite_rows[[length(finite_rows) + 1L]] <- total_result
      }
    }

    for (standardization in names(standardizations)) {
      ame_result <- mc_master_sample_effect(
        model = model,
        data = estimation_data,
        treatment_column = treatment_column,
        derivative = TRUE,
        weights = standardizations[[standardization]],
        normalize = TRUE,
        cluster_df = metadata$cluster_df
      ) %>%
        mc_scale_effect(outcome_specification) %>%
        mutate(
          outcome_id = outcome_id,
          outcome_label = outcome_label,
          horizon = horizon_name,
          treatment_column = treatment_column,
          standardization = standardization,
          scope = "overall",
          scope_value = "all",
          .before = 1
        )
      ame_rows[[length(ame_rows) + 1L]] <- ame_result
    }

    for (effect_year in metadata$analysis_years) {
      year_subset <- estimation_data$year == effect_year
      year_result <- mc_master_sample_effect(
        model = model,
        data = estimation_data,
        treatment_column = treatment_column,
        dose_change = 5,
        subset = year_subset,
        normalize = TRUE,
        cluster_df = metadata$cluster_df
      ) %>%
        mc_scale_effect(outcome_specification) %>%
        mutate(
          outcome_id = outcome_id,
          outcome_label = outcome_label,
          horizon = horizon_name,
          treatment_column = treatment_column,
          year = effect_year,
          standardization = "county_year_equal",
          .before = 1
        )
      year_rows[[length(year_rows) + 1L]] <- year_result

      year_ame <- mc_master_sample_effect(
        model = model,
        data = estimation_data,
        treatment_column = treatment_column,
        derivative = TRUE,
        subset = year_subset,
        normalize = TRUE,
        cluster_df = metadata$cluster_df
      ) %>%
        mc_scale_effect(outcome_specification) %>%
        mutate(
          outcome_id = outcome_id,
          outcome_label = outcome_label,
          horizon = horizon_name,
          treatment_column = treatment_column,
          standardization = "county_year_equal",
          scope = "year",
          scope_value = as.character(effect_year),
          .before = 1
        )
      ame_rows[[length(ame_rows) + 1L]] <- year_ame
    }

    for (heterogeneity_dimension in c(
      "mc_binding_quartile",
      "mc_baseline_h2a_quartile"
    )) {
      for (quartile in 1:4) {
        group_subset <-
          estimation_data[[heterogeneity_dimension]] == quartile
        group_result <- mc_master_sample_effect(
          model = model,
          data = estimation_data,
          treatment_column = treatment_column,
          dose_change = 5,
          subset = group_subset,
          normalize = TRUE,
          cluster_df = metadata$cluster_df
        ) %>%
          mc_scale_effect(outcome_specification) %>%
          mutate(
            outcome_id = outcome_id,
            outcome_label = outcome_label,
            horizon = horizon_name,
            treatment_column = treatment_column,
            heterogeneity_dimension = heterogeneity_dimension,
            quartile = quartile,
            standardization = "county_year_equal",
            .before = 1
          )
        heterogeneity_rows[[length(heterogeneity_rows) + 1L]] <-
          group_result

        group_ame <- mc_master_sample_effect(
          model = model,
          data = estimation_data,
          treatment_column = treatment_column,
          derivative = TRUE,
          subset = group_subset,
          normalize = TRUE,
          cluster_df = metadata$cluster_df
        ) %>%
          mc_scale_effect(outcome_specification) %>%
          mutate(
            outcome_id = outcome_id,
            outcome_label = outcome_label,
            horizon = horizon_name,
            treatment_column = treatment_column,
            standardization = "county_year_equal",
            scope = heterogeneity_dimension,
            scope_value = as.character(quartile),
            .before = 1
          )
        ame_rows[[length(ame_rows) + 1L]] <- group_ame
      }
    }

    dose_values <- treatment_quantiles[[treatment_column]]
    for (effect_year in metadata$analysis_years) {
      for (dose_index in seq_along(dose_values)) {
        for (z_value in c(-1, 0, 1)) {
          grid_result <- mc_master_index_ame(
            model = model,
            treatment_column = treatment_column,
            year = effect_year,
            dose = dose_values[[dose_index]],
            z = z_value,
            cluster_df = metadata$cluster_df
          ) %>%
            mc_scale_effect(outcome_specification) %>%
            mutate(
              outcome_id = outcome_id,
              outcome_label = outcome_label,
              horizon = horizon_name,
              dose_quantile = c("p10", "p50", "p90")[[dose_index]],
              z_label = c(
                `-1` = "one_sd_below",
                `0` = "mean",
                `1` = "one_sd_above"
              )[[as.character(z_value)]],
              .before = 1
            )
          grid_rows[[length(grid_rows) + 1L]] <- grid_result
        }
      }
    }
  }
}

finite_effects <- bind_rows(finite_rows)
average_marginal_effects <- bind_rows(ame_rows)
year_effects <- bind_rows(year_rows)
heterogeneity_effects <- bind_rows(heterogeneity_rows)
ame_grid <- bind_rows(grid_rows)

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
  ame_grid,
  path_tables("mc_ame_grid.csv")
)

# Coefficient graph for the transparent TWFE benchmark. The estimates are
# unchanged, while every interval uses the finite-design CCV covariance stored
# in the model bundle. The conventional clustered SE remains in the parameter
# CSV only as a comparison diagnostic and is not used in this graph.
parameter_estimates <- read_csv(
  path_tables("mc_parameter_estimates.csv"),
  show_col_types = FALSE
)
coefficient_plot <- parameter_estimates %>%
  filter(
    model_id == "twfe_benchmark",
    term %in% unname(MC_DYNAMIC_HORIZONS)
  ) %>%
  left_join(
    metadata$outcomes %>%
      select(outcome_id, outcome_label),
    by = "outcome_id"
  ) %>%
  mutate(
    horizon = recode(
      term,
      mc_dose_current = "Current",
      mc_dose_lag1 = "Lag 1",
      mc_dose_lag2 = "Lag 2"
    ),
    horizon = factor(
      horizon,
      levels = c("Current", "Lag 1", "Lag 2")
    ),
    critical = stats::qt(0.975, df = metadata$cluster_df),
    conf_low = estimate - critical * standard_error,
    conf_high = estimate + critical * standard_error
  )
coefficient_plot_object <- ggplot(
  coefficient_plot,
  aes(
    x = horizon,
    y = estimate,
    ymin = conf_low,
    ymax = conf_high
  )
) +
  geom_hline(yintercept = 0, color = "grey65", linewidth = 0.35) +
  geom_pointrange(color = "#7a3e00", linewidth = 0.45) +
  facet_wrap(vars(outcome_label), scales = "free_y", ncol = 2) +
  labs(
    x = NULL,
    y = "TWFE dose coefficient",
    title = "AEWR-growth coefficients with design-covariance CCV intervals",
    caption = paste0(
      "95% t(16) CCV intervals from the balanced 17-state\n",
      "AEWR-path reference design"
    )
  ) +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(hjust = 0)
  )
ggsave(
  filename = path_figures("fig_mc_ccv_coefficients.png"),
  plot = coefficient_plot_object,
  width = 10,
  height = 9,
  dpi = 300
)

dynamic_plot <- average_marginal_effects %>%
  filter(
    scope == "overall",
    scope_value == "all",
    standardization == "county_year_equal"
  ) %>%
  mutate(
    horizon = factor(
      horizon,
      levels = names(MC_DYNAMIC_HORIZONS),
      labels = c("Current", "Lag 1", "Lag 2")
    )
  )

dynamic_plot_object <- ggplot(
  dynamic_plot,
  aes(
    x = horizon,
    y = estimate,
    ymin = conf_low,
    ymax = conf_high
  )
) +
  geom_hline(yintercept = 0, color = "grey65", linewidth = 0.35) +
  geom_pointrange(color = "#1b4965", linewidth = 0.45) +
  facet_wrap(
    vars(outcome_label),
    scales = "free_y",
    ncol = 2
  ) +
  labs(
    x = NULL,
    y = "Average marginal effect per log point",
    caption = paste0(
      "95% t(16) design-covariance CCV intervals; balanced 17-state\n",
      "AEWR-path reference design"
    )
  ) +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(hjust = 0)
  )
ggsave(
  filename = path_figures("fig_mc_dynamic_effects.png"),
  plot = dynamic_plot_object,
  width = 10,
  height = 9,
  dpi = 300
)

year_plot <- average_marginal_effects %>%
  filter(
    outcome_id == "certified_positions",
    horizon == "contemporaneous",
    scope == "year"
  ) %>%
  mutate(year = as.integer(scope_value))
year_plot_object <- ggplot(
  year_plot,
  aes(
    x = year,
    y = estimate,
    ymin = conf_low,
    ymax = conf_high
  )
) +
  geom_hline(yintercept = 0, color = "grey65", linewidth = 0.35) +
  geom_pointrange(color = "#1b4965", linewidth = 0.45) +
  scale_x_continuous(breaks = metadata$analysis_years) +
  labs(
    x = NULL,
    y = "Certified positions per 1,000 baseline farm workers",
    title = "Year-specific marginal effect of AEWR growth",
    caption = "95% t(16) design-covariance CCV intervals"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank())
ggsave(
  filename = path_figures("fig_mc_year_effects.png"),
  plot = year_plot_object,
  width = 9,
  height = 5.5,
  dpi = 300
)

heterogeneity_plot <- average_marginal_effects %>%
  filter(
    outcome_id == "certified_positions",
    horizon == "contemporaneous",
    scope %in%
      c(
        "mc_binding_quartile",
        "mc_baseline_h2a_quartile"
      )
  ) %>%
  mutate(
    quartile = as.integer(scope_value),
    dimension = recode(
      scope,
      mc_binding_quartile = "Baseline AEWR bite",
      mc_baseline_h2a_quartile = "Baseline H-2A intensity"
    )
  )
heterogeneity_plot_object <- ggplot(
  heterogeneity_plot,
  aes(
    x = factor(quartile),
    y = estimate,
    ymin = conf_low,
    ymax = conf_high,
    group = dimension,
    color = dimension
  )
) +
  geom_hline(yintercept = 0, color = "grey65", linewidth = 0.35) +
  geom_pointrange(
    position = position_dodge(width = 0.35),
    linewidth = 0.45
  ) +
  labs(
    x = "Predetermined quartile",
    y = "Certified positions per 1,000 baseline farm workers",
    color = NULL,
    title = "Heterogeneous contemporaneous AEWR-growth effects",
    caption = "Average marginal effects with 95% t(16) CCV intervals"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank())
ggsave(
  filename = path_figures("fig_mc_heterogeneity.png"),
  plot = heterogeneity_plot_object,
  width = 9,
  height = 5.5,
  dpi = 300
)

message(
  "Postestimated ",
  nrow(finite_effects),
  " finite effects, ",
  nrow(average_marginal_effects),
  " sample AMEs, and ",
  nrow(ame_grid),
  " dose-by-Z grid AMEs."
)
