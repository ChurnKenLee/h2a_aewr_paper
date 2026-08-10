# Purpose: Diagnose the selected version-3 Mundlak-Chamberlain program.
#
# The diagnostics are deliberately separate from model selection.  They use
# the already selected primary records, inspect treatment support and
# moderator alignment, and estimate one restartable one-year-lead placebo per
# outcome.  No diagnostic is allowed to change a primary specification.

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
panel_cache_directory <- file.path(
  path_int(),
  "mundlak_chamberlain_specification_panels"
)
checkpoint_directory <- file.path(
  path_int(),
  "mundlak_chamberlain_specification_checkpoints"
)
placebo_directory <- file.path(
  path_int(),
  "mundlak_chamberlain_placebo_checkpoints"
)
dir.create(placebo_directory, recursive = TRUE, showWarnings = FALSE)

selected_registry <- primary_selection |>
  select(.data$outcome_id, .data$spec_id) |>
  left_join(registry, by = "spec_id")
if (
  nrow(selected_registry) != nrow(MC_OUTCOMES) ||
    anyNA(selected_registry$calendar_id)
) {
  stop("Primary selection is incomplete.", call. = FALSE)
}

target_calendar <- unique(selected_registry$calendar_id)
if (length(target_calendar) != 1L) {
  stop("All primary outcomes must use the target calendar.", call. = FALSE)
}
panel_bundle <- readRDS(
  file.path(panel_cache_directory, paste0(target_calendar, ".rds"))
)
panel <- panel_bundle$panel
metadata <- panel_bundle$metadata

# -------------------------------------------------------------------------
# Assignment-cell support and hierarchy
# -------------------------------------------------------------------------

treatment_columns <- c(
  unname(MC_DYNAMIC_HORIZONS),
  "mc_dose_lead1"
)
treatment_cells <- panel |>
  distinct(
    .data$aewr_region_id,
    .data$year,
    across(all_of(treatment_columns))
  )

treatment_support <- treatment_cells |>
  group_by(.data$year) |>
  summarise(
    assignment_cells = n(),
    mean = mean(.data$mc_dose_current),
    standard_deviation = sd(.data$mc_dose_current),
    minimum = min(.data$mc_dose_current),
    p10 = quantile(.data$mc_dose_current, 0.10),
    median = median(.data$mc_dose_current),
    p90 = quantile(.data$mc_dose_current, 0.90),
    maximum = max(.data$mc_dose_current),
    negative_share = mean(.data$mc_dose_current < 0),
    .groups = "drop"
  )
write_csv(treatment_support, path_tables("mc_treatment_support.csv"))

counterfactual_rows <- list()
for (horizon_name in names(MC_DYNAMIC_HORIZONS)) {
  treatment_column <- MC_DYNAMIC_HORIZONS[[horizon_name]]
  value <- treatment_cells[[treatment_column]]
  year_minimum <- ave(
    value,
    treatment_cells$year,
    FUN = function(x) min(x, na.rm = TRUE)
  )
  year_maximum <- ave(
    value,
    treatment_cells$year,
    FUN = function(x) max(x, na.rm = TRUE)
  )
  for (dose_change in MC_COUNTERFACTUAL_DOSES) {
    counterfactual <- value + dose_change
    counterfactual_rows[[length(counterfactual_rows) + 1L]] <-
      data.frame(
        horizon = horizon_name,
        treatment_column = treatment_column,
        dose_change = dose_change,
        assignment_cells = sum(is.finite(counterfactual)),
        outside_overall_support_share = mean(
          counterfactual < min(value, na.rm = TRUE) |
            counterfactual > max(value, na.rm = TRUE),
          na.rm = TRUE
        ),
        outside_same_year_support_share = mean(
          counterfactual < year_minimum |
            counterfactual > year_maximum,
          na.rm = TRUE
        ),
        counterfactual_minimum = min(counterfactual, na.rm = TRUE),
        counterfactual_maximum = max(counterfactual, na.rm = TRUE),
        stringsAsFactors = FALSE
      )
  }
}
write_csv(
  bind_rows(counterfactual_rows),
  path_tables("mc_counterfactual_support.csv")
)

hierarchy_units <- panel |>
  distinct(
    .data$county_fips,
    .data$mc_market_id,
    .data$state_fips,
    .data$aewr_region_id
  )
hierarchy_rows <- list(
  data.frame(
    level = "AEWR region",
    units = n_distinct(hierarchy_units$aewr_region_id),
    parent_level = NA_character_,
    minimum_children = NA_real_,
    median_children = NA_real_,
    maximum_children = NA_real_
  )
)
add_hierarchy <- function(level, child, parent, parent_label) {
  counts <- hierarchy_units |>
    distinct(.data[[parent]], .data[[child]]) |>
    count(.data[[parent]], name = "children")
  data.frame(
    level = level,
    units = n_distinct(hierarchy_units[[child]]),
    parent_level = parent_label,
    minimum_children = min(counts$children),
    median_children = median(counts$children),
    maximum_children = max(counts$children)
  )
}
hierarchy_rows[[2L]] <- add_hierarchy(
  "State", "state_fips", "aewr_region_id", "AEWR region"
)
hierarchy_rows[[3L]] <- add_hierarchy(
  "Market cell", "mc_market_id", "state_fips", "State"
)
hierarchy_rows[[4L]] <- add_hierarchy(
  "County", "county_fips", "mc_market_id", "Market cell"
)
write_csv(
  bind_rows(hierarchy_rows),
  path_tables("mc_hierarchy_counts.csv")
)

# -------------------------------------------------------------------------
# Within-year moderator alignment with the region-year dose basis
# -------------------------------------------------------------------------

mc_sp_alignment_blocks <- function(specification, metadata) {
  dictionary <- mc_sp_causal_dictionary(specification, metadata)
  moderators <- unique(stats::na.omit(dictionary$moderator_column))
  list(
    bite_level = intersect(moderators, metadata$bite_mean_column),
    county_market_means = intersect(
      moderators,
      metadata$mean_causal_columns
    ),
    bite_trajectories = intersect(
      moderators,
      setdiff(
        metadata$bite_trajectory_columns,
        metadata$bite_mean_column
      )
    ),
    county_market_trajectories = intersect(
      moderators,
      setdiff(
        metadata$trajectory_causal_columns,
        metadata$mean_causal_columns
      )
    ),
    bite_three_way = intersect(
      moderators,
      metadata$bite_three_way_columns
    )
  )
}

mc_sp_multiple_correlation <- function(
  response,
  moderators,
  controls = NULL
) {
  if (is.null(controls)) {
    controls <- matrix(numeric(), nrow = length(response), ncol = 0L)
  }
  finite <- is.finite(response) &
    apply(moderators, 1L, function(row) all(is.finite(row))) &
    if (ncol(controls) == 0L) {
      TRUE
    } else {
      apply(controls, 1L, function(row) all(is.finite(row)))
    }
  response <- response[finite]
  moderators <- moderators[finite, , drop = FALSE]
  controls <- controls[finite, , drop = FALSE]
  if (length(response) < 3L || ncol(moderators) == 0L) {
    return(c(
      rank = 0,
      r_squared = NA_real_,
      adjusted_r_squared = NA_real_,
      partial_r_squared = NA_real_
    ))
  }
  design <- cbind(intercept = 1, moderators)
  fit <- lm.fit(design, response, tol = 1e-9)
  rank <- fit$rank - 1L
  total <- sum((response - mean(response))^2)
  residual <- sum(fit$residuals^2)
  r_squared <- if (total > 0) 1 - residual / total else NA_real_
  adjusted <- if (
    is.finite(r_squared) && length(response) > rank + 1L
  ) {
    1 - (1 - r_squared) *
      (length(response) - 1L) /
      (length(response) - rank - 1L)
  } else {
    NA_real_
  }

  residualized <- if (ncol(controls) == 0L) {
    cbind(response, moderators)
  } else {
    control_design <- cbind(intercept = 1, controls)
    lm.fit(
      control_design,
      cbind(response, moderators),
      tol = 1e-9
    )$residuals
  }
  partial_response <- residualized[, 1L]
  partial_moderators <- residualized[, -1L, drop = FALSE]
  partial_total <- sum(partial_response^2)
  partial_fit <- lm.fit(
    partial_moderators,
    partial_response,
    tol = 1e-9
  )
  partial_r_squared <- if (partial_total > 0) {
    1 - sum(partial_fit$residuals^2) / partial_total
  } else {
    NA_real_
  }
  c(
    rank = rank,
    r_squared = r_squared,
    adjusted_r_squared = adjusted,
    partial_r_squared = partial_r_squared
  )
}

alignment_rows <- list()
for (selection_index in seq_len(nrow(selected_registry))) {
  selected <- selected_registry[selection_index, , drop = FALSE]
  specification <- mc_sp_specification(selected)
  checkpoint <- readRDS(
    file.path(
      checkpoint_directory,
      specification$spec_id,
      paste0(selected$outcome_id[[1]], ".rds")
    )
  )
  sample_key <- paste(
    checkpoint$estimation_keys$county_fips,
    checkpoint$estimation_keys$year,
    sep = ":"
  )
  panel_key <- paste(panel$county_fips, panel$year, sep = ":")
  sample_data <- panel[panel_key %in% sample_key, , drop = FALSE]
  blocks <- mc_sp_alignment_blocks(specification, metadata)
  all_moderator_columns <- unique(unlist(blocks, use.names = FALSE))
  dictionary <- mc_sp_causal_dictionary(specification, metadata)
  dose_bases <- unique(
    dictionary[c("treatment_column", "degree", "basis_column")]
  )
  for (block_name in names(blocks)) {
    block_columns <- blocks[[block_name]]
    if (length(block_columns) == 0L) {
      next
    }
    for (effect_year in specification$analysis_years) {
      year_data <- sample_data[
        sample_data$year == effect_year,
        ,
        drop = FALSE
      ]
      moderators <- as.matrix(
        year_data[, block_columns, drop = FALSE]
      )
      control_columns <- setdiff(
        all_moderator_columns,
        block_columns
      )
      controls <- as.matrix(
        year_data[, control_columns, drop = FALSE]
      )
      for (basis_index in seq_len(nrow(dose_bases))) {
        basis <- dose_bases[basis_index, , drop = FALSE]
        aligned <- mc_sp_multiple_correlation(
          year_data[[basis$basis_column[[1]]]],
          moderators,
          controls
        )
        alignment_rows[[length(alignment_rows) + 1L]] <- data.frame(
          outcome_id = selected$outcome_id[[1]],
          spec_id = specification$spec_id,
          richness_tier = specification$richness_tier,
          year = effect_year,
          moderator_block = block_name,
          declared_moderators = length(block_columns),
          active_rank = unname(aligned[["rank"]]),
          treatment_column = basis$treatment_column[[1]],
          degree = basis$degree[[1]],
          basis_column = basis$basis_column[[1]],
          multiple_correlation = sqrt(max(
            unname(aligned[["r_squared"]]),
            0,
            na.rm = TRUE
          )),
          r_squared = unname(aligned[["r_squared"]]),
          adjusted_r_squared =
            unname(aligned[["adjusted_r_squared"]]),
          partial_r_squared =
            unname(aligned[["partial_r_squared"]]),
          partial_correlation = sqrt(max(
            unname(aligned[["partial_r_squared"]]),
            0,
            na.rm = TRUE
          )),
          observations = nrow(year_data),
          stringsAsFactors = FALSE
        )
      }
    }
  }
}
write_csv(
  bind_rows(alignment_rows),
  path_tables("mc_moderator_alignment.csv")
)

# -------------------------------------------------------------------------
# Primary-only one-year-lead placebo, compiled under the same rank rules
# -------------------------------------------------------------------------

atomic_save_rds <- function(object, target) {
  dir.create(dirname(target), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(
    pattern = paste0(".", basename(target), "."),
    tmpdir = dirname(target)
  )
  on.exit(unlink(temporary), add = TRUE)
  saveRDS(object, temporary, compress = "gzip")
  if (!file.rename(temporary, target)) {
    stop("Could not install placebo checkpoint.", call. = FALSE)
  }
}

estimate_placebo <- function(selection_index) {
  selected <- selected_registry[selection_index, , drop = FALSE]
  outcome_id <- selected$outcome_id[[1]]
  target <- file.path(placebo_directory, paste0(outcome_id, ".rds"))
  if (file.exists(target)) {
    cached <- readRDS(target)
    if (
      identical(cached$design_version, MC_SPEC_PROGRAM_VERSION) &&
        identical(cached$primary_spec_id, selected$spec_id[[1]])
    ) {
      return(cached)
    }
  }
  outcome_specification <- MC_OUTCOMES[
    MC_OUTCOMES$outcome_id == outcome_id,
    ,
    drop = FALSE
  ]
  specification <- mc_sp_placebo_specification(
    mc_sp_specification(selected)
  )
  message("Estimating selected-primary lead placebo: ", outcome_id)
  result <- mc_sp_estimate_outcome(
    panel = panel,
    metadata = metadata,
    specification = specification,
    outcome_specification = outcome_specification
  )
  result$design_version <- MC_SPEC_PROGRAM_VERSION
  result$primary_spec_id <- selected$spec_id[[1]]
  atomic_save_rds(result, target)
  result
}

skip_placebo <- identical(Sys.getenv("MC_SKIP_PLACEBO", unset = "0"), "1")
placebo_results <- if (skip_placebo) {
  list()
} else {
  lapply(seq_len(nrow(selected_registry)), estimate_placebo)
}
if (length(placebo_results) > 0L) {
  placebo_diagnostics <- bind_rows(lapply(
    placebo_results,
    function(result) result$diagnostics
  ))
  write_csv(
    placebo_diagnostics,
    path_tables("mc_lead_placebo_diagnostics.csv")
  )
  rejected <- vapply(
    placebo_results,
    function(result) !identical(result$status, "estimated"),
    logical(1)
  )
  if (any(rejected)) {
    stop(
      "Selected-primary lead placebo was rejected for: ",
      paste(selected_registry$outcome_id[rejected], collapse = ", "),
      call. = FALSE
    )
  }
  lead_placebos <- bind_rows(lapply(
    placebo_results,
    function(result) {
      result$effects |>
        filter(
          .data$treatment_column == "mc_dose_lead1",
          .data$estimand == "finite_dose_change",
          .data$dose_change == 5
        ) |>
        mutate(
          model_id = "selected_primary_lead_test",
          primary_spec_id = result$primary_spec_id,
          t_statistic = .data$estimate / .data$standard_error,
          p_value = 2 * stats::pt(
            -abs(.data$t_statistic),
            df = MC_CCV_DF
          )
        )
    }
  ))
  write_csv(
    lead_placebos,
    path_tables("mc_lead_placebo_effects.csv")
  )
}

support_long <- treatment_cells |>
  select(
    .data$year,
    .data$aewr_region_id,
    all_of(unname(MC_DYNAMIC_HORIZONS))
  ) |>
  pivot_longer(
    cols = all_of(unname(MC_DYNAMIC_HORIZONS)),
    names_to = "treatment_column",
    values_to = "dose"
  ) |>
  mutate(
    horizon = recode(
      .data$treatment_column,
      !!!setNames(
        names(MC_DYNAMIC_HORIZONS),
        unname(MC_DYNAMIC_HORIZONS)
      )
    )
  )
support_plot <- ggplot(support_long, aes(x = .data$dose)) +
  geom_histogram(
    bins = 24,
    fill = "#1b4965",
    color = "white",
    linewidth = 0.25
  ) +
  facet_wrap(vars(.data$horizon), scales = "free_y", ncol = 1) +
  labs(
    x = "AEWR change (log percentage points)",
    y = "Region-year cells",
    title = "Observed support for current and lagged AEWR growth"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank())
ggsave(
  path_figures("fig_mc_treatment_support.png"),
  support_plot,
  width = 8,
  height = 8,
  dpi = 300
)

message(
  "Wrote support, hierarchy, moderator-alignment, and primary-placebo ",
  "diagnostics."
)
