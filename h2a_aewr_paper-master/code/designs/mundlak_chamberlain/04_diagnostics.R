# Purpose: Audit treatment support, hierarchy, rank restrictions, and future
# dose placebos for the rich Mundlak-Chamberlain master specification.
# Inputs:
#   data/processed/mundlak_chamberlain_county_year.parquet
#   data/intermediate/mundlak_chamberlain_models.rds
# Outputs:
#   outputs/tables/mc_treatment_support.csv
#   outputs/tables/mc_counterfactual_support.csv
#   outputs/tables/mc_hierarchy_counts.csv
#   outputs/tables/mc_identification_rank_audit.csv
#   outputs/tables/mc_lead_placebo_effects.csv
#   outputs/figures/fig_mc_treatment_support.png

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "mundlak_chamberlain", "design.R"))
source(path_code("designs", "mundlak_chamberlain", "helpers.R"))
library(arrow)
library(dplyr)
library(ggplot2)
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

treatment_cells <- panel %>%
  distinct(
    aewr_region_id,
    year,
    mc_dose_current,
    mc_dose_lag1,
    mc_dose_lag2,
    mc_dose_lead1
  )

treatment_support <- treatment_cells %>%
  group_by(year) %>%
  summarise(
    assignment_cells = n(),
    mean = mean(mc_dose_current),
    standard_deviation = sd(mc_dose_current),
    minimum = min(mc_dose_current),
    p10 = quantile(mc_dose_current, 0.10),
    median = median(mc_dose_current),
    p90 = quantile(mc_dose_current, 0.90),
    maximum = max(mc_dose_current),
    negative_share = mean(mc_dose_current < 0),
    .groups = "drop"
  )
write_csv(
  treatment_support,
  path_tables("mc_treatment_support.csv")
)

counterfactual_rows <- list()
for (horizon_name in names(MC_DYNAMIC_HORIZONS)) {
  treatment_column <- MC_DYNAMIC_HORIZONS[[horizon_name]]
  value <- treatment_cells[[treatment_column]]
  overall_minimum <- min(value, na.rm = TRUE)
  overall_maximum <- max(value, na.rm = TRUE)
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
    counterfactual_rows[[length(counterfactual_rows) + 1L]] <- tibble(
      horizon = horizon_name,
      treatment_column = treatment_column,
      dose_change = dose_change,
      assignment_cells = sum(is.finite(counterfactual)),
      outside_overall_support_share = mean(
        counterfactual < overall_minimum |
          counterfactual > overall_maximum,
        na.rm = TRUE
      ),
      outside_same_year_support_share = mean(
        counterfactual < year_minimum |
          counterfactual > year_maximum,
        na.rm = TRUE
      ),
      counterfactual_minimum = min(counterfactual, na.rm = TRUE),
      counterfactual_maximum = max(counterfactual, na.rm = TRUE)
    )
  }
}
counterfactual_support <- bind_rows(counterfactual_rows)
write_csv(
  counterfactual_support,
  path_tables("mc_counterfactual_support.csv")
)

hierarchy_counts <- tribble(
  ~level, ~units, ~parent_level, ~minimum_children, ~median_children, ~maximum_children,
  "AEWR region",
  n_distinct(panel$aewr_region_id),
  NA_character_,
  NA_real_,
  NA_real_,
  NA_real_,
  "State",
  n_distinct(panel$state_fips),
  "AEWR region",
  min(
    panel %>%
      distinct(aewr_region_id, state_fips) %>%
      count(aewr_region_id) %>%
      pull(n)
  ),
  median(
    panel %>%
      distinct(aewr_region_id, state_fips) %>%
      count(aewr_region_id) %>%
      pull(n)
  ),
  max(
    panel %>%
      distinct(aewr_region_id, state_fips) %>%
      count(aewr_region_id) %>%
      pull(n)
  ),
  "Market cell",
  n_distinct(panel$mc_market_id),
  "State",
  min(
    panel %>%
      distinct(state_fips, mc_market_id) %>%
      count(state_fips) %>%
      pull(n)
  ),
  median(
    panel %>%
      distinct(state_fips, mc_market_id) %>%
      count(state_fips) %>%
      pull(n)
  ),
  max(
    panel %>%
      distinct(state_fips, mc_market_id) %>%
      count(state_fips) %>%
      pull(n)
  ),
  "County",
  n_distinct(panel$county_fips),
  "Market cell",
  min(
    panel %>%
      distinct(mc_market_id, county_fips) %>%
      count(mc_market_id) %>%
      pull(n)
  ),
  median(
    panel %>%
      distinct(mc_market_id, county_fips) %>%
      count(mc_market_id) %>%
      pull(n)
  ),
  max(
    panel %>%
      distinct(mc_market_id, county_fips) %>%
      count(mc_market_id) %>%
      pull(n)
  )
)
write_csv(
  hierarchy_counts,
  path_tables("mc_hierarchy_counts.csv")
)

# The literal equation supplied by the researcher is audited before imposing
# the estimable leave-focal-out restriction.  At outcome year s, its
# W_{s-p} x d_s columns coincide with the r = s-p columns of the unstructured
# Chamberlain history projection.  Standardizing W_r only adds year and
# Z-by-year columns already in the model, so it does not resolve the rank
# failure.
rank_audit <- tribble(
  ~audit_item, ~count, ~available_assignment_cells, ~status, ~explanation,
  "Literal non-Z columns per outcome year",
  1L +
    length(MC_LAG_ORDERS) *
      length(MC_LITERAL_POLYNOMIAL_DEGREES) +
    length(MC_TREATMENT_HISTORY_YEARS),
  17L,
  "not_identified",
  paste(
    "Year intercept + 3 lags x 3 powers + all 12 history values",
    "exceeds the 17 AEWR-region cells."
  ),
  "Exact focal-history duplicates per outcome year",
  length(MC_LAG_ORDERS),
  17L,
  "not_identified",
  paste(
    "Each linear causal W_(s-p) term is collinear with the",
    "r=s-p Chamberlain-history term; the same is true after",
    "multiplication by Z."
  ),
  "Leave-focal-out cubic non-Z columns per outcome year",
  1L +
    length(MC_LAG_ORDERS) *
      length(MC_LITERAL_POLYNOMIAL_DEGREES) +
    (
      length(MC_TREATMENT_HISTORY_YEARS) -
        length(MC_LAG_ORDERS)
    ),
  17L,
  "not_identified",
  "Even after removing exact duplicates, a cubic leaves 19 columns for 17 cells.",
  "Implemented quadratic non-Z columns per outcome year",
  1L +
    length(MC_LAG_ORDERS) *
      length(MC_MASTER_POLYNOMIAL_DEGREES) +
    (
      length(MC_TREATMENT_HISTORY_YEARS) -
        length(MC_LAG_ORDERS)
    ),
  17L,
  "identified_if_full_rank",
  paste(
    "The implemented maximum is 16 columns for 17 assignment",
    "cells; all causal columns are separately checked after estimation."
  ),
  "Design-covariance CCV rank upper bound",
  MC_CLUSTER_DF,
  17L,
  "weak_inference_warning",
  paste(
    "With 17 finite reference states, the centered CCV coefficient-error",
    "covariance has rank at most 16 regardless of county-year sample size."
  )
)
write_csv(
  rank_audit,
  path_tables("mc_identification_rank_audit.csv")
)

lead_rows <- list()
for (outcome_index in seq_len(nrow(metadata$outcomes))) {
  outcome_specification <- metadata$outcomes[outcome_index, ]
  outcome_id <- outcome_specification$outcome_id[[1]]
  model <- bundle$models[[outcome_id]][["chamberlain_lead_test"]]
  row_ids <-
    bundle$sample_row_ids[[outcome_id]][["chamberlain_lead_test"]]
  estimation_data <- panel[
    match(row_ids, panel$mc_row_id),
    ,
    drop = FALSE
  ]

  derivative_result <- mc_master_sample_effect(
    model = model,
    data = estimation_data,
    treatment_column = "mc_dose_lead1",
    derivative = TRUE,
    normalize = TRUE,
    cluster_df = metadata$cluster_df
  )
  finite_result <- mc_master_sample_effect(
    model = model,
    data = estimation_data,
    treatment_column = "mc_dose_lead1",
    dose_change = 5,
    normalize = TRUE,
    cluster_df = metadata$cluster_df
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
  lead_rows[[length(lead_rows) + 1L]] <- bind_rows(
    derivative_result,
    finite_result
  ) %>%
    mutate(
      across(
        c(estimate, standard_error, conf_low, conf_high),
        ~ .x * multiplier
      ),
      outcome_id = outcome_id,
      outcome_label = outcome_specification$outcome_label[[1]],
      reported_unit = if (multiplier == 100) {
        "percentage_points"
      } else {
        outcome_specification$effect_unit[[1]]
      },
      t_statistic = estimate / standard_error,
      p_value = 2 * stats::pt(
        -abs(t_statistic),
        df = metadata$cluster_df
      ),
      .before = 1
    )
}
lead_placebos <- bind_rows(lead_rows)
write_csv(
  lead_placebos,
  path_tables("mc_lead_placebo_effects.csv")
)

support_long <- treatment_cells %>%
  select(
    year,
    aewr_region_id,
    all_of(unname(MC_DYNAMIC_HORIZONS))
  ) %>%
  pivot_longer(
    cols = all_of(unname(MC_DYNAMIC_HORIZONS)),
    names_to = "treatment_column",
    values_to = "dose"
  ) %>%
  mutate(
    horizon = recode(
      treatment_column,
      !!!setNames(
        names(MC_DYNAMIC_HORIZONS),
        unname(MC_DYNAMIC_HORIZONS)
      )
    )
  )

support_plot_object <- ggplot(
  support_long,
  aes(x = dose)
) +
  geom_histogram(
    bins = 24,
    fill = "#1b4965",
    color = "white",
    linewidth = 0.25
  ) +
  facet_wrap(
    vars(horizon),
    scales = "free_y",
    ncol = 1
  ) +
  labs(
    x = "AEWR change (log percentage points)",
    y = "Region-year cells",
    title = "Observed support for current and lagged AEWR growth"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank())
ggsave(
  filename = path_figures("fig_mc_treatment_support.png"),
  plot = support_plot_object,
  width = 8,
  height = 8,
  dpi = 300
)

message(
  "Audited ",
  nrow(treatment_cells),
  " region-year treatment cells and ",
  nrow(lead_placebos),
  " lead-placebo estimands."
)
