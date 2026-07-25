# Purpose: Diagnose calibration feasibility, concentration, and IV support.
# Inputs: cluster, wage/auxiliary calibration, and long instrument artifacts.
# Outputs: calibration/IV diagnostic CSV and support/tradeoff figures.
# Run after: code/c03_iv/09_construct_donor_instruments.R.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(tidyverse)

seasonal_imbalance_columns <- c(
  "seasonal_january_standardized_imbalance",
  "seasonal_april_standardized_imbalance",
  "seasonal_july_standardized_imbalance"
)

max_if_any <- function(x) {
  if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE)
}

median_if_any <- function(x) {
  if (all(is.na(x))) NA_real_ else median(x, na.rm = TRUE)
}

weight_spec_display <- function(
  weight_spec_label,
  soft_penalty,
  prior_spec
) {
  case_when(
    weight_spec_label == "wage_only_exact" ~ "Wage only (exact)",
    weight_spec_label == "wage_seasonal_exact" ~
      "Wage + seasonal (exact; BEA prior)",
    weight_spec_label == "wage_seasonal_qwi_duration_exact" ~
      "Wage + seasonal + QWI duration (exact)",
    weight_spec_label == "wage_seasonal_census_duration_exact" ~
      "Wage + seasonal + bridged Census duration (exact)",
    weight_spec_label == "wage_seasonal_interval" ~
      "Wage + seasonal (interval)",
    str_detect(weight_spec_label, "wage_seasonal_soft") ~
      paste0("Wage + seasonal (soft; rho = ", soft_penalty, ")"),
    str_detect(weight_spec_label, "_prior_") ~ paste0(
      "Wage + seasonal (exact; ",
      str_replace_all(prior_spec, "_", " "),
      " prior)"
    ),
    .default = weight_spec_label
  )
}

iv_clusters <- read_parquet(path_int("iv_cz_aewr_clusters.parquet"))
iv_cluster_diagnostics <- read_parquet(path_int(
  "iv_cluster_diagnostics.parquet"
))
iv_oews_long <- read_parquet(path_int("iv_oews_entropy_long.parquet")) %>%
  filter(near(gap_closure, 1))

iv_design_specs <- iv_oews_long %>%
  distinct(
    iv_k,
    donor_cluster_count,
    weight_spec_label,
    prior_spec,
    moment_spec,
    calibration_mode,
    duration_analogue,
    soft_penalty
  )

full_gap_wage_diagnostics <- read_parquet(path_int(
  "fls_wage_entropy_diagnostics.parquet"
)) %>%
  filter(near(gap_closure, 1))

wage_calibration_years <- full_gap_wage_diagnostics %>%
  filter(str_detect(calibration_status, "^calibrated")) %>%
  select(aewr_region_id, year) %>%
  distinct()

wage_target_years <- wage_calibration_years %>%
  transmute(
    aewr_region_id,
    year = year + 1L
  ) %>%
  distinct()

# Downstream support uses successful wage-only cells as a common denominator,
# making both calibration loss and donor-overlap loss visible.
expected_instrument_cells <- iv_clusters %>%
  select(cz_aewr_region_fe, aewr_region_id, iv_k) %>%
  inner_join(iv_design_specs, by = "iv_k", relationship = "many-to-many") %>%
  inner_join(
    wage_target_years,
    by = "aewr_region_id",
    relationship = "many-to-many"
  )

instrument_cell_support <- expected_instrument_cells %>%
  left_join(
    iv_oews_long %>%
      select(
        cz_aewr_region_fe,
        aewr_region_id,
        year,
        iv_k,
        donor_cluster_count,
        weight_spec_label,
        z_oews_entropy_agwage_l1,
        oews_iv_donor_weight,
        oews_iv_donor_clusters,
        oews_iv_donor_areas,
        oews_iv_donor_units,
        oews_iv_candidate_areas,
        oews_iv_overlap_areas_excluded
      ),
    by = c(
      "cz_aewr_region_fe",
      "aewr_region_id",
      "year",
      "iv_k",
      "donor_cluster_count",
      "weight_spec_label"
    ),
    relationship = "one-to-one"
  ) %>%
  group_by(
    weight_spec_label,
    prior_spec,
    moment_spec,
    calibration_mode,
    duration_analogue,
    soft_penalty,
    iv_k,
    donor_cluster_count
  ) %>%
  summarise(
    expected_instrument_cells = n(),
    observed_instrument_cells = sum(
      !is.na(z_oews_entropy_agwage_l1)
    ),
    instrument_coverage_rate = mean(
      !is.na(z_oews_entropy_agwage_l1)
    ),
    median_donor_weight = median_if_any(oews_iv_donor_weight),
    median_donor_clusters = median_if_any(oews_iv_donor_clusters),
    median_donor_areas = median_if_any(oews_iv_donor_areas),
    median_donor_units = median_if_any(oews_iv_donor_units),
    median_candidate_areas = median_if_any(oews_iv_candidate_areas),
    median_overlap_areas_excluded = median_if_any(
      oews_iv_overlap_areas_excluded
    ),
    .groups = "drop"
  )

# Evaluate seasonal targets under wage-only weights using exactly the imputed
# public features and scales used by the preferred exact-seasonal estimator.
seasonal_feature_scaffold <- read_parquet(path_int(
  "fls_oews_area_weight_soft_calibrated.parquet"
)) %>%
  filter(
    prior_spec == "bea",
    moment_spec == "wage_seasonal_exact",
    near(gap_closure, 1)
  ) %>%
  select(
    aewr_region_id,
    year,
    oews_area_code,
    fls_hired_worker_share_january,
    fls_hired_worker_share_april,
    fls_hired_worker_share_july,
    calibration_feature_seasonal_january,
    calibration_feature_seasonal_april,
    calibration_feature_seasonal_july,
    seasonal_january_feature_scale,
    seasonal_april_feature_scale,
    seasonal_july_feature_scale
  ) %>%
  distinct()

wage_only_seasonal_balance <- read_parquet(path_int(
  "fls_oews_area_weight_wage_calibrated.parquet"
)) %>%
  filter(
    near(gap_closure, 1),
    str_detect(calibration_status, "^calibrated")
  ) %>%
  select(
    aewr_region_id,
    year,
    oews_area_code,
    oews_area_weight_wage_calibrated
  ) %>%
  inner_join(
    seasonal_feature_scaffold,
    by = c("aewr_region_id", "year", "oews_area_code"),
    relationship = "one-to-one"
  ) %>%
  group_by(aewr_region_id, year) %>%
  summarise(
    seasonal_january_standardized_imbalance = (
      sum(
        oews_area_weight_wage_calibrated *
          calibration_feature_seasonal_january
      ) - first(fls_hired_worker_share_january)
    ) / first(seasonal_january_feature_scale),
    seasonal_april_standardized_imbalance = (
      sum(
        oews_area_weight_wage_calibrated *
          calibration_feature_seasonal_april
      ) - first(fls_hired_worker_share_april)
    ) / first(seasonal_april_feature_scale),
    seasonal_july_standardized_imbalance = (
      sum(
        oews_area_weight_wage_calibrated *
          calibration_feature_seasonal_july
      ) - first(fls_hired_worker_share_july)
    ) / first(seasonal_july_feature_scale),
    .groups = "drop"
  )

wage_only_calibration_cells <- full_gap_wage_diagnostics %>%
  transmute(
    aewr_region_id,
    year,
    weight_spec_label = "wage_only_exact",
    prior_spec = "bea",
    moment_spec = "wage_only_exact",
    calibration_mode = "exact",
    duration_analogue = NA_character_,
    soft_penalty = NA_real_,
    calibration_status,
    wage_moment_error = if_else(
      str_detect(calibration_status, "^calibrated"),
      oews_calibrated_weighted_wage - entropy_target_wage,
      NA_real_
    ),
    lp_feasibility_status = NA_integer_,
    exact_max_abs_residual = abs(wage_moment_error),
    interval_max_violation = NA_real_,
    minimum_active_observed_prior_mass = 1,
    entropy_kl_divergence,
    effective_area_count_ratio,
    maximum_weight_adjustment,
    duration_standardized_imbalance = NA_real_,
    seasonal_january_interval_slack = NA_real_,
    seasonal_april_interval_slack = NA_real_,
    seasonal_july_interval_slack = NA_real_
  ) %>%
  left_join(
    wage_only_seasonal_balance,
    by = c("aewr_region_id", "year"),
    relationship = "one-to-one"
  )

auxiliary_calibration_cells <- read_parquet(path_int(
  "fls_soft_entropy_diagnostics.parquet"
)) %>%
  filter(include_wage_target, near(gap_closure, 1)) %>%
  transmute(
    aewr_region_id,
    year,
    weight_spec_label,
    prior_spec,
    moment_spec,
    calibration_mode,
    duration_analogue,
    soft_penalty,
    calibration_status,
    wage_moment_error,
    lp_feasibility_status,
    exact_max_abs_residual,
    interval_max_violation,
    minimum_active_observed_prior_mass,
    entropy_kl_divergence,
    effective_area_count_ratio,
    maximum_weight_adjustment,
    duration_standardized_imbalance,
    seasonal_january_standardized_imbalance,
    seasonal_april_standardized_imbalance,
    seasonal_july_standardized_imbalance,
    seasonal_january_interval_slack,
    seasonal_april_interval_slack,
    seasonal_july_interval_slack
  )

calibration_summary <- bind_rows(
  wage_only_calibration_cells,
  auxiliary_calibration_cells
) %>%
  semi_join(
    wage_calibration_years,
    by = c("aewr_region_id", "year")
  ) %>%
  mutate(
    seasonal_balance_complete = if_all(
      all_of(seasonal_imbalance_columns),
      ~ !is.na(.x)
    ),
    seasonal_standardized_rmse_cell = if_else(
      seasonal_balance_complete,
      sqrt(rowMeans(pick(all_of(seasonal_imbalance_columns))^2)),
      NA_real_
    ),
    minimum_interval_slack = pmin(
      seasonal_january_interval_slack,
      seasonal_april_interval_slack,
      seasonal_july_interval_slack,
      na.rm = FALSE
    )
  ) %>%
  group_by(
    weight_spec_label,
    prior_spec,
    moment_spec,
    calibration_mode,
    duration_analogue,
    soft_penalty
  ) %>%
  summarise(
    calibration_cells = n(),
    calibrated_cells = sum(
      str_detect(calibration_status, "^calibrated")
    ),
    exact_infeasible_cells = sum(
      calibration_status == "exact_infeasible"
    ),
    interval_infeasible_cells = sum(
      calibration_status == "interval_infeasible"
    ),
    insufficient_coverage_cells = sum(
      calibration_status == "insufficient_auxiliary_coverage"
    ),
    calibration_coverage_rate = mean(
      str_detect(calibration_status, "^calibrated")
    ),
    maximum_absolute_wage_moment_error = max_if_any(
      abs(wage_moment_error)
    ),
    maximum_exact_standardized_residual = max_if_any(
      exact_max_abs_residual
    ),
    maximum_interval_violation = max_if_any(interval_max_violation),
    minimum_interval_slack = if (all(is.na(minimum_interval_slack))) {
      NA_real_
    } else {
      min(minimum_interval_slack, na.rm = TRUE)
    },
    minimum_auxiliary_observed_prior_mass = if (
      all(is.na(minimum_active_observed_prior_mass))
    ) {
      NA_real_
    } else {
      min(minimum_active_observed_prior_mass, na.rm = TRUE)
    },
    seasonal_balance_cells = sum(seasonal_balance_complete),
    seasonal_standardized_rmse = if (
      all(is.na(seasonal_standardized_rmse_cell))
    ) {
      NA_real_
    } else {
      sqrt(mean(
        seasonal_standardized_rmse_cell^2,
        na.rm = TRUE
      ))
    },
    median_absolute_duration_imbalance = median_if_any(
      abs(duration_standardized_imbalance)
    ),
    median_entropy_kl_divergence = median_if_any(
      entropy_kl_divergence
    ),
    median_effective_area_count_ratio = median_if_any(
      effective_area_count_ratio
    ),
    median_maximum_weight_adjustment = median_if_any(
      maximum_weight_adjustment
    ),
    .groups = "drop"
  ) %>%
  mutate(
    weight_spec_display = weight_spec_display(
      weight_spec_label,
      soft_penalty,
      prior_spec
    )
  )

cluster_count_support <- iv_cluster_diagnostics %>%
  group_by(iv_k) %>%
  summarise(
    aewr_regions = n_distinct(aewr_region_id),
    total_clusters = n(),
    minimum_cluster_units = min(cluster_units),
    median_cluster_units = median(cluster_units),
    minimum_cluster_feature_weight_share = min(
      cluster_feature_weight_share
    ),
    median_cluster_feature_weight_share = median(
      cluster_feature_weight_share
    ),
    .groups = "drop"
  )

iv_cluster_count_diagnostics <- instrument_cell_support %>%
  inner_join(
    cluster_count_support,
    by = "iv_k",
    relationship = "many-to-one"
  ) %>%
  inner_join(
    calibration_summary,
    by = c(
      "weight_spec_label",
      "prior_spec",
      "moment_spec",
      "calibration_mode",
      "duration_analogue",
      "soft_penalty"
    ),
    relationship = "many-to-one"
  ) %>%
  arrange(weight_spec_display, iv_k, donor_cluster_count)

cat("=== Calibration and IV support diagnostics ===\n")
print(
  iv_cluster_count_diagnostics %>%
    select(
      weight_spec_display,
      iv_k,
      donor_cluster_count,
      instrument_coverage_rate,
      calibration_coverage_rate,
      exact_infeasible_cells,
      seasonal_standardized_rmse,
      median_effective_area_count_ratio,
      median_donor_units
    ) %>%
    mutate(across(where(is.numeric), ~ round(.x, 4))),
  n = Inf,
  width = Inf
)

calibration_coverage_lines <- iv_cluster_count_diagnostics %>%
  distinct(weight_spec_display, calibration_coverage_rate)

iv_support_figure <- ggplot(
  iv_cluster_count_diagnostics,
  aes(
    x = donor_cluster_count,
    y = 100 * instrument_coverage_rate,
    color = factor(iv_k),
    group = iv_k
  )
) +
  geom_hline(
    data = calibration_coverage_lines,
    aes(yintercept = 100 * calibration_coverage_rate),
    inherit.aes = FALSE,
    color = "grey55",
    linetype = "dashed",
    linewidth = 0.5
  ) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.4) +
  facet_wrap(~weight_spec_display, ncol = 3) +
  scale_x_continuous(
    breaks = seq_len(max(iv_design_specs$donor_cluster_count))
  ) +
  labs(
    x = "Number of furthest donor clusters used",
    y = "Instrument-cell coverage (%)",
    color = "Clusters per AEWR\nregion (k)",
    title = "IV coverage by entropy-projection specification",
    subtitle = "Dashed lines show calibration-cell coverage"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom"
  )

moment_coverage_tradeoff <- iv_cluster_count_diagnostics %>%
  group_by(weight_spec_label, weight_spec_display) %>%
  summarise(
    seasonal_standardized_rmse = first(
      seasonal_standardized_rmse
    ),
    mean_instrument_coverage_rate = mean(instrument_coverage_rate),
    median_effective_area_count_ratio = first(
      median_effective_area_count_ratio
    ),
    .groups = "drop"
  )

moment_coverage_figure <- ggplot(
  moment_coverage_tradeoff,
  aes(
    x = seasonal_standardized_rmse,
    y = 100 * mean_instrument_coverage_rate,
    color = median_effective_area_count_ratio
  )
) +
  geom_point(size = 3.5) +
  geom_text(
    aes(label = weight_spec_label),
    nudge_y = 0.8,
    check_overlap = TRUE,
    size = 2.6
  ) +
  scale_color_viridis_c(option = "C", direction = -1) +
  labs(
    x = "Seasonal-moment standardized RMSE",
    y = "Mean instrument-cell coverage across k/d (%)",
    color = "Median effective-\narea ratio",
    title = "Moment balance, concentration, and IV coverage"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank())

dir.create(path_figures(), recursive = TRUE, showWarnings = FALSE)
dir.create(path_tables(), recursive = TRUE, showWarnings = FALSE)

ggsave(
  filename = path_figures("fig_iv_cluster_count_support.png"),
  plot = iv_support_figure,
  width = 13,
  height = 10,
  dpi = 300
)

ggsave(
  filename = path_figures(
    "fig_iv_auxiliary_moment_coverage_tradeoff.png"
  ),
  plot = moment_coverage_figure,
  width = 10,
  height = 6,
  dpi = 300
)

write_csv(
  iv_cluster_count_diagnostics,
  path_tables("iv_cluster_count_diagnostics.csv")
)
