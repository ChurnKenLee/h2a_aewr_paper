# Purpose: Diagnose full-gap wage-only IV support across cluster and donor counts.
# Inputs: cluster assignments/diagnostics, wage diagnostics, and long IV output.
# Outputs: donor-set support figure and diagnostic CSV.
# Run after: code/c03_iv/09_construct_donor_instruments.R.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
library(arrow)
library(tidyverse)

iv_clusters <- read_parquet(path_int("iv_cz_aewr_clusters.parquet"))
iv_cluster_diagnostics <- read_parquet(path_int(
  "iv_cluster_diagnostics.parquet"
))
iv_oews_long <- read_parquet(path_int("iv_oews_entropy_long.parquet")) %>%
  filter(near(gap_closure, 1), moment_spec == "wage_only")
iv_design_specs <- iv_oews_long %>%
  distinct(iv_k, donor_cluster_count) %>%
  arrange(iv_k, donor_cluster_count)

# A target unit-year is expected whenever its AEWR region has a successful
# full-gap wage calibration. The donor construction may lose that cell if the
# overlap exclusion leaves no eligible donor support.
full_gap_wage_diagnostics <- read_parquet(path_int(
  "fls_wage_entropy_diagnostics.parquet"
)) %>%
  filter(
    near(gap_closure, 1),
    str_detect(calibration_status, "^calibrated")
  )

wage_target_years <- full_gap_wage_diagnostics %>%
  transmute(
    aewr_region_num,
    year = as.integer(year) + 1L
  ) %>%
  distinct()

wage_balance_summary <- full_gap_wage_diagnostics %>%
  summarise(
    maximum_absolute_wage_moment_error = max(
      abs(oews_calibrated_weighted_wage - entropy_target_wage),
      na.rm = TRUE
    ),
    median_absolute_wage_moment_error = median(
      abs(oews_calibrated_weighted_wage - entropy_target_wage),
      na.rm = TRUE
    )
  )

expected_instrument_cells <- iv_clusters %>%
  select(
    cz_aewr_region_fe,
    aewr_region_num,
    iv_k
  ) %>%
  inner_join(
    iv_design_specs,
    by = "iv_k",
    relationship = "many-to-many"
  ) %>%
  inner_join(
    wage_target_years,
    by = "aewr_region_num",
    relationship = "many-to-many"
  )

instrument_cell_support <- expected_instrument_cells %>%
  left_join(
    iv_oews_long %>%
      select(
        cz_aewr_region_fe,
        aewr_region_num,
        year,
        iv_k,
        donor_cluster_count,
        z_oews_entropy_agwage_l1,
        oews_iv_donor_weight,
        oews_iv_donor_clusters,
        oews_iv_donor_areas,
        oews_iv_donor_units,
        oews_iv_candidate_donor_clusters,
        oews_iv_candidate_areas,
        oews_iv_candidate_units,
        oews_iv_overlap_areas_excluded,
        oews_iv_donor_cluster_distance,
        oews_iv_farthest_donor_cluster_distance
      ),
    by = c(
      "cz_aewr_region_fe",
      "aewr_region_num",
      "year",
      "iv_k",
      "donor_cluster_count"
    ),
    relationship = "one-to-one"
  ) %>%
  group_by(iv_k, donor_cluster_count) %>%
  summarise(
    expected_instrument_cells = n(),
    observed_instrument_cells = sum(
      !is.na(z_oews_entropy_agwage_l1)
    ),
    instrument_coverage_rate = mean(
      !is.na(z_oews_entropy_agwage_l1)
    ),
    median_donor_weight = median(
      oews_iv_donor_weight,
      na.rm = TRUE
    ),
    median_donor_clusters = median(
      oews_iv_donor_clusters,
      na.rm = TRUE
    ),
    median_donor_areas = median(
      oews_iv_donor_areas,
      na.rm = TRUE
    ),
    median_donor_units = median(
      oews_iv_donor_units,
      na.rm = TRUE
    ),
    median_candidate_areas = median(
      oews_iv_candidate_areas,
      na.rm = TRUE
    ),
    median_candidate_donor_clusters = median(
      oews_iv_candidate_donor_clusters,
      na.rm = TRUE
    ),
    median_candidate_units = median(
      oews_iv_candidate_units,
      na.rm = TRUE
    ),
    median_overlap_areas_excluded = median(
      oews_iv_overlap_areas_excluded,
      na.rm = TRUE
    ),
    median_donor_cluster_distance = median(
      oews_iv_donor_cluster_distance,
      na.rm = TRUE
    ),
    median_farthest_donor_cluster_distance = median(
      oews_iv_farthest_donor_cluster_distance,
      na.rm = TRUE
    ),
    .groups = "drop"
  )

cluster_count_support <- iv_cluster_diagnostics %>%
  group_by(iv_k) %>%
  summarise(
    aewr_regions = n_distinct(aewr_region_num),
    total_clusters = n(),
    minimum_cluster_units = min(cluster_units),
    median_cluster_units = median(cluster_units),
    minimum_cluster_feature_weight_share = min(
      cluster_feature_weight_share
    ),
    median_cluster_feature_weight_share = median(
      cluster_feature_weight_share
    ),
    median_region_min_cluster_units = median(
      region_min_cluster_units
    ),
    median_region_min_cluster_weight_share = median(
      region_min_cluster_weight_share
    ),
    .groups = "drop"
  )

iv_cluster_count_diagnostics <- cluster_count_support %>%
  inner_join(
    instrument_cell_support,
    by = "iv_k",
    relationship = "one-to-many"
  ) %>%
  mutate(
    maximum_absolute_wage_moment_error =
      wage_balance_summary$maximum_absolute_wage_moment_error,
    median_absolute_wage_moment_error =
      wage_balance_summary$median_absolute_wage_moment_error
  ) %>%
  arrange(iv_k, donor_cluster_count)

cat("=== Full-gap wage-only IV diagnostics by cluster and donor count ===\n")
print(
  iv_cluster_count_diagnostics %>%
    mutate(across(where(is.numeric), ~ round(.x, 4))),
  n = Inf,
  width = Inf
)

iv_cluster_count_figure_data <- bind_rows(
  iv_cluster_count_diagnostics %>%
    transmute(
      iv_k,
      donor_cluster_count,
      diagnostic = "Instrument coverage (%)",
      value = 100 * instrument_coverage_rate
    ),
  iv_cluster_count_diagnostics %>%
    transmute(
      iv_k,
      donor_cluster_count,
      diagnostic = "Median donor CZ-region units",
      value = median_donor_units
    ),
  iv_cluster_count_diagnostics %>%
    transmute(
      iv_k,
      donor_cluster_count,
      diagnostic = "Median donor OEWS areas",
      value = median_donor_areas
    )
) %>%
  mutate(
    diagnostic = factor(
      diagnostic,
      levels = c(
        "Instrument coverage (%)",
        "Median donor CZ-region units",
        "Median donor OEWS areas"
      )
    )
  )

iv_cluster_count_figure <- ggplot(
  iv_cluster_count_figure_data,
  aes(
    x = donor_cluster_count,
    y = value,
    color = factor(iv_k),
    group = iv_k
  )
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.8) +
  facet_wrap(~diagnostic, scales = "free_y", nrow = 1) +
  scale_x_continuous(
    breaks = seq_len(max(iv_design_specs$donor_cluster_count))
  ) +
  scale_color_manual(
    values = c(
      `2` = "#0072B2",
      `3` = "#D55E00",
      `4` = "#009E73",
      `5` = "#CC79A7"
    )
  ) +
  labs(
    x = "Number of furthest donor clusters used",
    y = NULL,
    color = "Clusters per AEWR\nregion (k)",
    title = "Full-gap wage-only IV support by donor-set size",
    subtitle = paste0(
      "100% wage target held fixed (max absolute moment error: ",
      format(
        wage_balance_summary$maximum_absolute_wage_moment_error,
        scientific = TRUE,
        digits = 2
      ),
      "); ",
      "each line uses a different donor partition"
    )
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom"
  )

dir.create(path_figures(), recursive = TRUE, showWarnings = FALSE)
dir.create(path_tables(), recursive = TRUE, showWarnings = FALSE)

ggsave(
  filename = path_figures("fig_iv_cluster_count_support.png"),
  plot = iv_cluster_count_figure,
  width = 10,
  height = 4.6,
  dpi = 300
)

write_csv(
  iv_cluster_count_diagnostics,
  path_tables("iv_cluster_count_diagnostics.csv")
)
