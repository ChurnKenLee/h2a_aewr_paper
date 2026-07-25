# Purpose: Produce publication descriptives for the preferred k=5, d=2 IV.
# Inputs: FLS/OEWS weights, CZ features/clusters, panel, and county shapes.
# Outputs: national wage, CZ comparison, weight-change, similarity, and map
# figures plus their plotting data.
# Run after: code/c03_iv/08_cluster_cz_donor_units.R,
# code/c03_iv/10_attach_instruments_to_panel.R.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("c00_shared", "analysis_helpers.R"))
source(path_code("c00_shared", "iv_preferred_design.R"))
library(arrow)
library(tidyverse)
library(sf)

dir.create(path_figures(), recursive = TRUE, showWarnings = FALSE)
dir.create(path_tables(), recursive = TRUE, showWarnings = FALSE)

preferred_caption <- paste0(
  "Preferred design: ",
  IV_PREFERRED_K,
  " clusters per AEWR region, ",
  IV_PREFERRED_DONOR_COUNT,
  " furthest donor clusters, exact FLS wage and January/April/July ",
  "moments, BEA prior."
)

safe_weighted_mean <- function(x, w) {
  keep <- is.finite(x) & is.finite(w) & w > 0
  if (!any(keep)) {
    return(NA_real_)
  }
  weighted.mean(x[keep], w[keep])
}

county_df_iv <- read_parquet(path_processed(
  "county_df_analysis_year_iv.parquet"
))
assert_geo_columns(
  county_df_iv,
  c("county_fips", "state_fips", "aewr_region_id", "cz_id")
)

required_panel_columns <- c(
  "county_fips",
  "year",
  "state_abbrev",
  "aewr_region_id",
  "cz_aewr_region_fe",
  "aewr_ppi",
  "ppi_2012",
  "emp_farm",
  IV_CLUSTER_ASSIGNMENT_COLUMN
)
stopifnot(all(required_panel_columns %in% names(county_df_iv)))

ppi_by_year <- county_df_iv %>%
  filter(!is.na(ppi_2012), ppi_2012 > 0) %>%
  group_by(year) %>%
  summarise(
    ppi_2012 = median(ppi_2012),
    .groups = "drop"
  )

# National AEWR and OEWS wage series ----------------------------------------

fls_source <- read_parquet(path_int("fls_region.parquet"))
assert_geo_columns(fls_source, "aewr_region_id")
fls_source <- fls_source %>%
  transmute(
    aewr_region_id,
    source_year = preliminary_year,
    policy_year = source_year + 1L,
    fls_wage_nominal = as.numeric(field_livestock_preliminary),
    fls_worker_weight = as.numeric(
      fls_hired_workers_reference_week_total
    )
  ) %>%
  filter(
    !is.na(aewr_region_id),
    !is.na(source_year),
    is.finite(fls_wage_nominal),
    fls_wage_nominal > 0,
    is.finite(fls_worker_weight),
    fls_worker_weight > 0
  ) %>%
  arrange(aewr_region_id, source_year) %>%
  distinct(aewr_region_id, source_year, .keep_all = TRUE)

wage_series_weight_specs <- c(
  "wage_only_exact",
  "wage_seasonal_exact",
  "wage_seasonal_interval",
  "wage_seasonal_soft_rho100"
)

donor_wages_by_spec <- read_parquet(path_int(
  "iv_oews_entropy_long.parquet"
))
assert_geo_columns(
  donor_wages_by_spec,
  "aewr_region_id"
)
donor_wages_by_spec <- donor_wages_by_spec %>%
  filter(
    iv_k == IV_PREFERRED_K,
    donor_cluster_count == IV_PREFERRED_DONOR_COUNT,
    near(gap_closure, IV_PREFERRED_GAP_CLOSURE),
    weight_spec_label %in% wage_series_weight_specs
  ) %>%
  transmute(
    cz_aewr_region_fe,
    aewr_region_id,
    policy_year = year,
    source_year = policy_year - 1L,
    weight_spec_label,
    oews_unweighted_nominal = z_oews_unweighted_agwage_l1,
    oews_entropy_nominal = z_oews_entropy_agwage_l1,
    unweighted_donor_units = oews_iv_unweighted_donor_units,
    unweighted_donor_counties = oews_iv_unweighted_donor_counties,
    entropy_donor_units = oews_iv_donor_units
  )

stopifnot(
  nrow(donor_wages_by_spec) ==
    nrow(
      distinct(
        donor_wages_by_spec,
        cz_aewr_region_fe,
        policy_year,
        weight_spec_label
      )
    )
)

preferred_donor_wages <- donor_wages_by_spec %>%
  filter(weight_spec_label == IV_PREFERRED_WEIGHT_SPEC)

oews_region_wages <- donor_wages_by_spec %>%
  group_by(
    aewr_region_id,
    source_year,
    policy_year,
    weight_spec_label
  ) %>%
  summarise(
    oews_unweighted_nominal = mean(
      oews_unweighted_nominal,
      na.rm = TRUE
    ),
    oews_entropy_nominal = mean(
      oews_entropy_nominal,
      na.rm = TRUE
    ),
    target_cz_region_units = n_distinct(cz_aewr_region_fe),
    .groups = "drop"
  )

aewr_region_policy <- county_df_iv %>%
  filter(
    !is.na(aewr_region_id),
    is.finite(aewr_ppi),
    aewr_ppi > 0
  ) %>%
  group_by(
    aewr_region_id,
    policy_year = year
  ) %>%
  summarise(
    # State agricultural-wage floors can differ within an AEWR region.
    # Collapse those effective wages to the region with farm employment.
    aewr_real = safe_weighted_mean(aewr_ppi, emp_farm),
    .groups = "drop"
  )

national_wage_region_cells <- oews_region_wages %>%
  inner_join(
    fls_source %>%
      select(
        aewr_region_id,
        source_year,
        policy_year,
        fls_wage_nominal,
        fls_worker_weight
    ),
    by = c("aewr_region_id", "source_year", "policy_year"),
    relationship = "many-to-one"
  ) %>%
  inner_join(
    aewr_region_policy,
    by = c("aewr_region_id", "policy_year"),
    relationship = "many-to-one"
  ) %>%
  inner_join(
    ppi_by_year %>% rename(policy_year = year),
    by = "policy_year",
    relationship = "many-to-one"
  ) %>%
  mutate(
    oews_unweighted_real = oews_unweighted_nominal / ppi_2012,
    oews_entropy_real = oews_entropy_nominal / ppi_2012,
    fls_wage_real_policy_dollars = fls_wage_nominal / ppi_2012
  )

national_wage_baseline_series <- national_wage_region_cells %>%
  filter(weight_spec_label == IV_PREFERRED_WEIGHT_SPEC) %>%
  group_by(policy_year) %>%
  summarise(
    `AEWR` = safe_weighted_mean(aewr_real, fls_worker_weight),
    `OEWS, equal-weighted donors` = safe_weighted_mean(
      oews_unweighted_real,
      fls_worker_weight
    ),
    regions = n_distinct(aewr_region_id),
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols = c(
      `AEWR`,
      `OEWS, equal-weighted donors`
    ),
    names_to = "series",
    values_to = "real_hourly_wage"
  )

national_wage_entropy_series <- national_wage_region_cells %>%
  group_by(policy_year, weight_spec_label) %>%
  summarise(
    real_hourly_wage = safe_weighted_mean(
      oews_entropy_real,
      fls_worker_weight
    ),
    regions = n_distinct(aewr_region_id),
    .groups = "drop"
  ) %>%
  mutate(
    series = case_when(
      weight_spec_label == "wage_only_exact" ~
        "OEWS, wage-only entropy weights",
      weight_spec_label == "wage_seasonal_exact" ~
        "OEWS, exact seasonal entropy weights",
      weight_spec_label == "wage_seasonal_interval" ~
        "OEWS, interval seasonal entropy weights",
      weight_spec_label == "wage_seasonal_soft_rho100" ~
        "OEWS, soft seasonal entropy weights (rho = 1)",
      .default = weight_spec_label
    )
  ) %>%
  select(policy_year, regions, series, real_hourly_wage)

national_wage_series <- bind_rows(
  national_wage_entropy_series,
  national_wage_baseline_series
) %>%
  filter(is.finite(real_hourly_wage)) %>%
  mutate(
    # Draw the entropy-weighted OEWS series first and the AEWR last so the
    # thinner dashed AEWR remains visible where the series overlap.
    series = factor(
      series,
      levels = c(
        "OEWS, wage-only entropy weights",
        "OEWS, soft seasonal entropy weights (rho = 1)",
        "OEWS, interval seasonal entropy weights",
        "OEWS, exact seasonal entropy weights",
        "OEWS, equal-weighted donors",
        "AEWR"
      )
    )
  ) %>%
  arrange(series, policy_year)

write_csv(
  national_wage_series,
  path_tables("iv_national_real_wage_series.csv")
)

wage_series_legend_breaks <- c(
  "AEWR",
  "OEWS, equal-weighted donors",
  "OEWS, wage-only entropy weights",
  "OEWS, soft seasonal entropy weights (rho = 1)",
  "OEWS, interval seasonal entropy weights",
  "OEWS, exact seasonal entropy weights"
)

national_wage_plot <- ggplot(
  national_wage_series,
  aes(
    x = policy_year,
    y = real_hourly_wage,
    color = series,
    linetype = series,
    linewidth = series,
    shape = series
  )
) +
  geom_line() +
  geom_point(size = 2) +
  scale_color_manual(
    name = NULL,
    values = c(
      "AEWR" = "black",
      "OEWS, equal-weighted donors" = "#0072B2",
      "OEWS, wage-only entropy weights" = "#56B4E9",
      "OEWS, soft seasonal entropy weights (rho = 1)" = "#009E73",
      "OEWS, interval seasonal entropy weights" = "#CC79A7",
      "OEWS, exact seasonal entropy weights" = "#D55E00"
    ),
    breaks = wage_series_legend_breaks
  ) +
  scale_linetype_manual(
    name = NULL,
    values = c(
      "AEWR" = "22",
      "OEWS, equal-weighted donors" = "solid",
      "OEWS, wage-only entropy weights" = "dotted",
      "OEWS, soft seasonal entropy weights (rho = 1)" = "longdash",
      "OEWS, interval seasonal entropy weights" = "dotdash",
      "OEWS, exact seasonal entropy weights" = "solid"
    ),
    breaks = wage_series_legend_breaks
  ) +
  scale_linewidth_manual(
    name = NULL,
    values = c(
      "AEWR" = 0.85,
      "OEWS, equal-weighted donors" = 1,
      "OEWS, wage-only entropy weights" = 0.8,
      "OEWS, soft seasonal entropy weights (rho = 1)" = 0.85,
      "OEWS, interval seasonal entropy weights" = 1,
      "OEWS, exact seasonal entropy weights" = 1.65
    ),
    breaks = wage_series_legend_breaks
  ) +
  scale_shape_manual(
    name = NULL,
    values = c(
      "AEWR" = 15,
      "OEWS, equal-weighted donors" = 17,
      "OEWS, wage-only entropy weights" = 3,
      "OEWS, soft seasonal entropy weights (rho = 1)" = 18,
      "OEWS, interval seasonal entropy weights" = 8,
      "OEWS, exact seasonal entropy weights" = 16
    ),
    breaks = wage_series_legend_breaks
  ) +
  scale_x_continuous(
    breaks = scales::breaks_pretty(n = 8)
  ) +
  scale_y_continuous(
    labels = scales::label_dollar(accuracy = 0.01)
  ) +
  labs(
    x = "Policy year",
    y = "Real hourly wage (2012 dollars)",
    color = NULL,
    title = "National AEWR and OEWS Agricultural Wages",
    subtitle = paste0(
      "OEWS and FLS source-year wages are aligned to the following AEWR ",
      "policy year"
    ),
    caption = str_wrap(
      paste0(
        "Within each target CZ-region, the unweighted OEWS series is the ",
        "simple mean of eligible donor CZ-region wages; the weighted series ",
        "uses entropy weights on the same donor support. Regions are weighted ",
        "by FLS hired-worker totals. The preferred entropy projection imposes ",
        "the wage and January/April/July moments exactly with the BEA prior."
      ),
      width = 120
    )
  ) +
  theme_minimal(base_size = 11) +
  guides(
    color = guide_legend(nrow = 2, byrow = TRUE)
  ) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    plot.caption = element_text(hjust = 0)
  )

ggsave(
  path_figures("fig_iv_national_real_wage_series.png"),
  national_wage_plot,
  width = 9,
  height = 5.5,
  dpi = 300,
  bg = "white"
)

# FLS versus target-CZ-region donor OEWS scatterplots -----------------------

preferred_county_weights <- read_parquet(path_int(
  "fls_county_weight_soft_calibrated.parquet"
))
assert_geo_columns(
  preferred_county_weights,
  c(
    "county_fips",
    "state_fips",
    "aewr_region_id",
    "cz_id"
  )
)
preferred_county_weights <- preferred_county_weights %>%
  filter(
    include_wage_target,
    near(gap_closure, IV_PREFERRED_GAP_CLOSURE),
    prior_spec == IV_PREFERRED_PRIOR_SPEC,
    weight_spec_label == IV_PREFERRED_WEIGHT_SPEC,
    str_detect(calibration_status, "^calibrated")
  ) %>%
  transmute(
    county_fips,
    source_year = year,
    aewr_region_id,
    cz_aewr_region_fe,
    entropy_county_weight = fls_county_weight_entropy_calibrated
  )

cz_wage_comparison <- preferred_donor_wages %>%
  inner_join(
    fls_source %>%
      select(
        aewr_region_id,
        source_year,
        fls_wage_nominal
      ),
    by = c("aewr_region_id", "source_year"),
    relationship = "many-to-one"
  ) %>%
  inner_join(
    ppi_by_year %>% rename(source_year = year),
    by = "source_year",
    relationship = "many-to-one"
  ) %>%
  transmute(
    aewr_region_id,
    cz_aewr_region_fe,
    source_year,
    fls_real_wage = fls_wage_nominal / ppi_2012,
    `Unweighted donor OEWS` = oews_unweighted_nominal / ppi_2012,
    `Entropy-weighted donor OEWS` = oews_entropy_nominal / ppi_2012
  ) %>%
  pivot_longer(
    cols = c(
      `Unweighted donor OEWS`,
      `Entropy-weighted donor OEWS`
    ),
    names_to = "panel",
    values_to = "oews_real_wage"
  ) %>%
  mutate(
    panel = factor(
      panel,
      levels = c(
        "Unweighted donor OEWS",
        "Entropy-weighted donor OEWS"
      )
    )
  ) %>%
  filter(
    is.finite(fls_real_wage),
    is.finite(oews_real_wage)
  )

write_csv(
  cz_wage_comparison,
  path_tables("iv_cz_region_fls_oews_wage_comparison.csv")
)

scatter_limits <- range(
  c(
    cz_wage_comparison$fls_real_wage,
    cz_wage_comparison$oews_real_wage
  ),
  na.rm = TRUE
)

cz_wage_scatter <- ggplot(
  cz_wage_comparison,
  aes(x = fls_real_wage, y = oews_real_wage)
) +
  geom_abline(
    slope = 1,
    intercept = 0,
    color = "grey35",
    linetype = "dashed",
    linewidth = 0.7
  ) +
  geom_point(alpha = 0.22, size = 0.8, color = "#0072B2") +
  facet_wrap(
    ~panel,
    nrow = 1,
    labeller = as_labeller(
      c(
        "Unweighted donor OEWS" =
          "Panel A: Unweighted donor OEWS",
        "Entropy-weighted donor OEWS" =
          "Panel B: Entropy-weighted donor OEWS"
      )
    )
  ) +
  coord_equal(xlim = scatter_limits, ylim = scatter_limits) +
  scale_x_continuous(labels = scales::label_dollar(accuracy = 1)) +
  scale_y_continuous(labels = scales::label_dollar(accuracy = 1)) +
  labs(
    x = "FLS field-and-livestock wage (2012 dollars)",
    y = "OEWS agricultural wage (2012 dollars)",
    title = "FLS and Donor OEWS Wages across Target CZ–Region Years",
    caption = str_wrap(
      paste0(
        "Panel A equally averages eligible donor CZ-region wages; Panel B ",
        "uses entropy weights over the identical donor support. Dashed line ",
        "is y = x. ",
        preferred_caption
      ),
      width = 125
    )
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0)
  )

ggsave(
  path_figures("fig_iv_fls_oews_cz_region_scatter.png"),
  cz_wage_scatter,
  width = 10,
  height = 5.2,
  dpi = 300,
  bg = "white"
)

# Year-to-year percentage-point changes in CZ entropy-weight shares ----------

cz_entropy_weights <- preferred_county_weights %>%
  group_by(
    aewr_region_id,
    cz_aewr_region_fe,
    source_year
  ) %>%
  summarise(
    cz_entropy_weight = sum(entropy_county_weight, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(cz_aewr_region_fe, source_year) %>%
  group_by(cz_aewr_region_fe) %>%
  mutate(
    prior_year = lag(source_year),
    prior_cz_entropy_weight = lag(cz_entropy_weight),
    entropy_weight_change_pp = if_else(
      prior_year == source_year - 1L,
      100 * (cz_entropy_weight - prior_cz_entropy_weight),
      NA_real_
    )
  ) %>%
  ungroup()

weight_sum_diagnostic <- cz_entropy_weights %>%
  group_by(aewr_region_id, source_year) %>%
  summarise(
    cz_weight_sum = sum(cz_entropy_weight),
    .groups = "drop"
  )

stopifnot(
  max(abs(weight_sum_diagnostic$cz_weight_sum - 1), na.rm = TRUE) <
    1e-8
)

weight_change_data <- cz_entropy_weights %>%
  filter(is.finite(entropy_weight_change_pp))

write_csv(
  weight_change_data,
  path_tables("iv_cz_entropy_weight_changes_pp.csv")
)

weight_change_plot <- ggplot(
  weight_change_data,
  aes(x = entropy_weight_change_pp)
) +
  geom_vline(
    xintercept = 0,
    color = "grey35",
    linetype = "dashed",
    linewidth = 0.6
  ) +
  geom_histogram(
    bins = 60,
    fill = "#0072B2",
    color = "white",
    linewidth = 0.15
  ) +
  scale_x_continuous(
    labels = scales::label_number(suffix = " pp", accuracy = 0.1)
  ) +
  labs(
    x = "Annual change in CZ share of AEWR-region entropy weight",
    y = "CZ–AEWR-region-year observations",
    title = "Year-to-Year Changes in Entropy-Balancing Weights",
    subtitle = "First differences are multiplied by 100 and pooled across CZ–AEWR-region cells",
    caption = preferred_caption
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    plot.caption = element_text(hjust = 0)
  )

ggsave(
  path_figures("fig_iv_cz_entropy_weight_changes_pp.png"),
  weight_change_plot,
  width = 8,
  height = 5.2,
  dpi = 300,
  bg = "white"
)

# Target-donor similarity diagnostics ---------------------------------------

selected_features <- c(
  share_cdl_fruit_tree_nuts = "Fruit/tree-nut acreage share",
  normal_cb_temp_tavg_b00 = "Temperature-normal basis component",
  aws0150wta = "Available water storage (0–150 cm)"
)

cz_features <- read_parquet(path_int(
  "iv_cz_aewr_features.parquet"
))
assert_geo_columns(cz_features, c("aewr_region_id", "cz_id"))
stopifnot(all(names(selected_features) %in% names(cz_features)))

iv_clusters <- read_parquet(path_int("iv_cz_aewr_clusters.parquet"))
assert_geo_columns(iv_clusters, "aewr_region_id")
iv_clusters <- iv_clusters %>%
  filter(iv_k == IV_PREFERRED_K) %>%
  select(
    cz_aewr_region_fe,
    aewr_region_id,
    target_cluster = iv_cluster
  )

donor_pairs <- read_parquet(path_int("iv_donor_clusters.parquet"))
assert_geo_columns(donor_pairs, "aewr_region_id")
donor_pairs <- donor_pairs %>%
  filter(
    iv_k == IV_PREFERRED_K,
    donor_cluster_count == IV_PREFERRED_DONOR_COUNT
  ) %>%
  select(
    aewr_region_id,
    target_cluster,
    donor_cluster,
    donor_rank,
    donor_cluster_distance
  )

standardized_features <- cz_features %>%
  select(
    cz_aewr_region_fe,
    aewr_region_id,
    all_of(names(selected_features))
  ) %>%
  group_by(aewr_region_id) %>%
  mutate(
    across(
      all_of(names(selected_features)),
      ~ {
        feature_sd <- sd(.x, na.rm = TRUE)
        if (is.finite(feature_sd) && feature_sd > 0) {
          (.x - mean(.x, na.rm = TRUE)) / feature_sd
        } else {
          0
        }
      }
    )
  ) %>%
  ungroup() %>%
  inner_join(
    iv_clusters,
    by = c("cz_aewr_region_fe", "aewr_region_id"),
    relationship = "one-to-one"
  )

cluster_feature_means <- standardized_features %>%
  group_by(aewr_region_id, target_cluster) %>%
  summarise(
    across(all_of(names(selected_features)), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols = all_of(names(selected_features)),
    names_to = "feature",
    values_to = "target_value"
  )

target_donor_features <- donor_pairs %>%
  inner_join(
    cluster_feature_means,
    by = c("aewr_region_id", "target_cluster"),
    relationship = "many-to-many"
  ) %>%
  inner_join(
    cluster_feature_means %>%
      rename(
        donor_cluster = target_cluster,
        donor_value = target_value
      ),
    by = c("aewr_region_id", "donor_cluster", "feature"),
    relationship = "many-to-one"
  ) %>%
  mutate(
    feature_label = recode(
      feature,
      !!!as.list(selected_features)
    ),
    standardized_gap = target_value - donor_value,
    absolute_standardized_gap = abs(standardized_gap)
  )

donor_mean_features <- target_donor_features %>%
  group_by(
    aewr_region_id,
    target_cluster,
    feature,
    feature_label
  ) %>%
  summarise(
    target_value = first(target_value),
    donor_mean_value = mean(donor_value),
    .groups = "drop"
  )

similarity_slopes <- donor_mean_features %>%
  group_by(aewr_region_id, feature, feature_label) %>%
  group_modify(~ {
    complete <- .x %>%
      filter(is.finite(target_value), is.finite(donor_mean_value))
    valid <- nrow(complete) >= 3 &&
      sd(complete$target_value) > 0 &&
      sd(complete$donor_mean_value) > 0
    if (!valid) {
      return(tibble(
        observations = nrow(complete),
        slope = NA_real_,
        intercept = NA_real_,
        r2 = NA_real_,
        regression_status = "undefined_zero_variance_or_support"
      ))
    }
    fit <- lm(target_value ~ donor_mean_value, data = complete)
    tibble(
      observations = nrow(complete),
      slope = unname(coef(fit)[["donor_mean_value"]]),
      intercept = unname(coef(fit)[["(Intercept)"]]),
      r2 = summary(fit)$r.squared,
      regression_status = "estimated"
    )
  }) %>%
  ungroup()

write_csv(
  target_donor_features,
  path_tables("iv_target_donor_standardized_feature_gaps.csv")
)
write_csv(
  similarity_slopes,
  path_tables("iv_target_donor_regional_bivariate_slopes.csv")
)

similarity_plot_data <- bind_rows(
  target_donor_features %>%
    transmute(
      feature_label,
      diagnostic = "Absolute standardized target–donor gap",
      value = absolute_standardized_gap
    ),
  similarity_slopes %>%
    filter(regression_status == "estimated") %>%
    transmute(
      feature_label,
      diagnostic = "AEWR-region bivariate slope",
      value = slope
    )
) %>%
  mutate(
    feature_label = factor(
      feature_label,
      levels = unname(selected_features)
    ),
    diagnostic = factor(
      diagnostic,
      levels = c(
        "Absolute standardized target–donor gap",
        "AEWR-region bivariate slope"
      )
    )
  )

similarity_plot <- ggplot(
  similarity_plot_data,
  aes(x = value)
) +
  geom_vline(
    data = tibble(
      diagnostic = factor(
        "AEWR-region bivariate slope",
        levels = levels(similarity_plot_data$diagnostic)
      ),
      reference = 1
    ),
    aes(xintercept = reference),
    inherit.aes = FALSE,
    color = "grey35",
    linetype = "dashed",
    linewidth = 0.5
  ) +
  geom_histogram(
    bins = 25,
    fill = "#009E73",
    color = "white",
    linewidth = 0.2
  ) +
  facet_grid(
    rows = vars(diagnostic),
    cols = vars(feature_label),
    scales = "free"
  ) +
  labs(
    x = NULL,
    y = "Observations",
    title = "Similarity between Target and Donor Clusters",
    subtitle = paste0(
      "Slopes regress five target-cluster means on their two-donor means ",
      "separately within each AEWR region"
    ),
    caption = preferred_caption
  ) +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0)
  )

ggsave(
  path_figures("fig_iv_target_donor_similarity.png"),
  similarity_plot,
  width = 12,
  height = 7,
  dpi = 300,
  bg = "white"
)

# Preferred-cluster maps and California example -----------------------------

county_map_full <- read_county_map(
  path_raw("county_shapefile", "tl_2020_us_county.zip"),
  simplify = FALSE
)
county_map <- st_simplify(
  county_map_full,
  preserveTopology = FALSE,
  dTolerance = 1000
)

county_assignments <- county_df_iv %>%
  distinct(
    county_fips,
    state_abbrev,
    aewr_region_id,
    cz_aewr_region_fe,
    iv_cluster_k5
  ) %>%
  mutate(county_fips = recode(county_fips, `46113` = "46102"))

county_cluster_map <- county_map %>%
  left_join(
    county_assignments,
    by = "county_fips",
    relationship = "one-to-one"
  ) %>%
  filter(!is.na(iv_cluster_k5)) %>%
  mutate(
    iv_cluster_label = factor(
      paste0("Cluster ", iv_cluster_k5),
      levels = paste0("Cluster ", seq_len(IV_PREFERRED_K))
    )
  )

aewr_boundaries <- county_map_full %>%
  inner_join(
    county_assignments %>%
      select(county_fips, aewr_region_id),
    by = "county_fips",
    relationship = "one-to-one"
  ) %>%
  group_by(aewr_region_id) %>%
  summarise(geometry = st_union(geometry), .groups = "drop") %>%
  st_simplify(preserveTopology = TRUE, dTolerance = 1000)

cluster_colors <- c(
  "Cluster 1" = "#1b9e77",
  "Cluster 2" = "#d95f02",
  "Cluster 3" = "#7570b3",
  "Cluster 4" = "#e7298a",
  "Cluster 5" = "#66a61e"
)

preferred_cluster_map <- ggplot() +
  geom_sf(
    data = county_cluster_map,
    aes(fill = iv_cluster_label),
    color = "white",
    linewidth = 0.04
  ) +
  geom_sf(
    data = aewr_boundaries,
    fill = NA,
    color = "grey10",
    linewidth = 0.55
  ) +
  scale_fill_manual(values = cluster_colors, name = NULL) +
  coord_sf(datum = NA) +
  labs(
    title = "Five Dissimilarity Clusters within Each AEWR Region",
    subtitle = "Black outlines denote the 17 AEWR wage-setting regions",
    caption = preferred_caption
  ) +
  theme_void(base_size = 11) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0),
    plot.background = element_rect(fill = "white", color = NA)
  )

ggsave(
  path_figures("map_iv_preferred_five_clusters.png"),
  preferred_cluster_map,
  width = 12,
  height = 7.5,
  dpi = 300,
  bg = "white"
)

ca_target <- county_df_iv %>%
  filter(year == 2011, state_abbrev == "CA") %>%
  group_by(aewr_region_id, cz_aewr_region_fe, iv_cluster_k5) %>%
  summarise(
    farm_employment_2011 = sum(emp_farm, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(farm_employment_2011), cz_aewr_region_fe) %>%
  slice_head(n = 1)

stopifnot(nrow(ca_target) == 1)

ca_donors <- donor_pairs %>%
  filter(
    aewr_region_id == ca_target$aewr_region_id,
    target_cluster == ca_target$iv_cluster_k5
  ) %>%
  arrange(donor_rank)

stopifnot(nrow(ca_donors) == IV_PREFERRED_DONOR_COUNT)

ca_role_lookup <- ca_donors %>%
  transmute(
    iv_cluster_k5 = donor_cluster,
    map_role = paste0("Donor cluster ", donor_rank)
  )

ca_map_data <- county_cluster_map %>%
  filter(aewr_region_id == ca_target$aewr_region_id) %>%
  mutate(
    map_role = case_when(
      cz_aewr_region_fe == ca_target$cz_aewr_region_fe ~ "Target CZ",
      iv_cluster_k5 == ca_target$iv_cluster_k5 ~ "Other target-cluster CZs",
      TRUE ~ "Unused cluster"
    )
  ) %>%
  left_join(
    ca_role_lookup,
    by = "iv_cluster_k5",
    suffix = c("", "_donor")
  ) %>%
  mutate(
    map_role = coalesce(map_role_donor, map_role),
    map_role = factor(
      map_role,
      levels = c(
        "Target CZ",
        "Other target-cluster CZs",
        "Donor cluster 1",
        "Donor cluster 2",
        "Unused cluster"
      )
    )
  )

ca_target_boundary <- ca_map_data %>%
  filter(cz_aewr_region_fe == ca_target$cz_aewr_region_fe) %>%
  summarise(geometry = st_union(geometry))

ca_target_label <- ca_target_boundary %>%
  st_point_on_surface() %>%
  mutate(label = "Target CZ")

ca_example_map <- ggplot() +
  geom_sf(
    data = ca_map_data,
    aes(fill = map_role),
    color = scales::alpha("white", 0.55),
    linewidth = 0.08
  ) +
  geom_sf(
    data = ca_target_boundary,
    fill = NA,
    color = "black",
    linewidth = 1.15
  ) +
  geom_sf_text(
    data = ca_target_label,
    aes(label = label),
    color = "white",
    size = 3.2,
    fontface = "bold"
  ) +
  scale_fill_manual(
    values = c(
      "Target CZ" = "#000000",
      "Other target-cluster CZs" = "#80cdc1",
      "Donor cluster 1" = "#d73027",
      "Donor cluster 2" = "#fc8d59",
      "Unused cluster" = "grey88"
    ),
    name = NULL,
    drop = FALSE
  ) +
  coord_sf(datum = NA) +
  labs(
    title = "California Example: Target CZ and Preferred Donor Clusters",
    subtitle = paste0(
      "AEWR region ",
      ca_target$aewr_region_id,
      "; target selected by largest California farm employment in 2011"
    ),
    caption = preferred_caption
  ) +
  theme_void(base_size = 11) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0),
    plot.background = element_rect(fill = "white", color = NA)
  )

ggsave(
  path_figures("map_iv_california_target_and_donors.png"),
  ca_example_map,
  width = 8.5,
  height = 7,
  dpi = 300,
  bg = "white"
)

write_csv(
  ca_target %>%
    mutate(
      donor_cluster_1 = ca_donors$donor_cluster[
        ca_donors$donor_rank == 1
      ],
      donor_cluster_2 = ca_donors$donor_cluster[
        ca_donors$donor_rank == 2
      ]
    ),
  path_tables("iv_california_target_donor_metadata.csv")
)

cat("Preferred IV descriptive exhibits completed.\n")
cat(
  "California target:",
  as.character(ca_target$cz_aewr_region_fe),
  "| AEWR region:",
  ca_target$aewr_region_id,
  "| target cluster:",
  ca_target$iv_cluster_k5,
  "| donor clusters:",
  paste(ca_donors$donor_cluster, collapse = ", "),
  "\n"
)
