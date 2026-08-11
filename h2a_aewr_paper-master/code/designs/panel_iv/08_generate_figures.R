# Purpose: Produce retained panel-IV diagnostic figures and plotting data.
# Inputs: fixed clusters, recovered primary entropy weights, OEWS/FLS wages,
# county-area shares, the shared county panel, PPI, and county geometry.
# Outputs: five diagnostic PNGs and five plotting-data CSVs.
rm(list = ls())
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "helpers.R"))
library(arrow)
library(dplyr)
library(ggplot2)
library(readr)
library(sf)
library(stringr)
library(tibble)
library(tidyr)

assert_unique_keys <- function(data, keys, label) {
  missing_keys <- setdiff(keys, names(data))
  if (length(missing_keys) > 0L) {
    stop(
      label,
      " is missing keys: ",
      paste(missing_keys, collapse = ", "),
      call. = FALSE
    )
  }
  duplicates <- data %>%
    count(across(all_of(keys)), name = "key_count") %>%
    filter(key_count != 1L)
  if (nrow(duplicates) > 0L) {
    stop(label, " does not have unique keys.", call. = FALSE)
  }
  invisible(data)
}

assert_close <- function(value, target, tolerance, label) {
  if (
    length(value) == 0L ||
      any(!is.finite(value)) ||
      any(abs(value - target) > tolerance)
  ) {
    stop(label, " failed its numerical tolerance.", call. = FALSE)
  }
  invisible(value)
}

assert_complete_cells <- function(data, expected, keys, label) {
  missing <- anti_join(expected, data, by = keys)
  extra <- anti_join(data, expected, by = keys)
  if (nrow(missing) > 0L || nrow(extra) > 0L) {
    stop(label, " does not have the expected cells.", call. = FALSE)
  }
  invisible(data)
}

first_nonmissing_character <- function(value) {
  value <- value[!is.na(value) & value != ""]
  if (length(value) == 0L) {
    return(NA_character_)
  }
  value[[1]]
}

source_years <- seq.int(
  DISSIMILARITY_IV_POLICY_START_YEAR - 1L,
  DISSIMILARITY_IV_POLICY_END_YEAR - 1L
)
policy_years <- seq.int(
  DISSIMILARITY_IV_POLICY_START_YEAR,
  DISSIMILARITY_IV_POLICY_END_YEAR
)

county_panel <- read_parquet(
  path_processed("county_year_panel.parquet")
) %>%
  mutate(
    panel_iv_target_unit_id = make_panel_iv_target_unit_id(
      cz_id,
      aewr_region_id
    )
  )

region_labels <- county_panel %>%
  distinct(aewr_region_id, state_abbrev) %>%
  filter(!is.na(aewr_region_id), !is.na(state_abbrev)) %>%
  arrange(as.integer(aewr_region_id), state_abbrev) %>%
  group_by(aewr_region_id) %>%
  summarise(
    aewr_region_states = paste(state_abbrev, collapse = ", "),
    .groups = "drop"
  ) %>%
  mutate(
    aewr_region_number = as.integer(aewr_region_id),
    aewr_region_label = paste0(
      "AEWR Region ",
      aewr_region_id,
      " (",
      aewr_region_states,
      ")"
    )
  ) %>%
  arrange(aewr_region_number)

if (
  nrow(region_labels) != 17L ||
    !identical(region_labels$aewr_region_number, seq_len(17L))
) {
  stop("Expected labels for all 17 AEWR regions.", call. = FALSE)
}
assert_unique_keys(region_labels, "aewr_region_id", "AEWR region labels")

expected_source_cells <- region_labels %>%
  select(aewr_region_id) %>%
  crossing(source_year = source_years)
expected_policy_cells <- region_labels %>%
  select(aewr_region_id) %>%
  crossing(policy_year = policy_years)

ppi <- read_parquet(path_int("ppi_2012.parquet")) %>%
  transmute(
    year = as.integer(year),
    ppi_2012 = as.numeric(ppi_2012)
  ) %>%
  filter(
    year >= min(source_years),
    year <= max(policy_years),
    is.finite(ppi_2012),
    ppi_2012 > 0
  ) %>%
  distinct()
assert_unique_keys(ppi, "year", "PPI")
if (!all(c(source_years, policy_years) %in% ppi$year)) {
  stop("PPI does not cover every figure year.", call. = FALSE)
}

primary_clusters <- read_parquet(
  path_int("panel_iv_target_clusters.parquet")
) %>%
  filter(iv_k == DISSIMILARITY_IV_PRIMARY_K) %>%
  transmute(
    panel_iv_target_unit_id,
    aewr_region_id,
    iv_k,
    target_cluster = iv_cluster
  ) %>%
  distinct()
assert_unique_keys(
  primary_clusters,
  c("panel_iv_target_unit_id", "aewr_region_id"),
  "Primary cluster assignments"
)

primary_donor_map <- read_parquet(
  path_int("panel_iv_donor_clusters.parquet")
) %>%
  filter(
    iv_k == DISSIMILARITY_IV_PRIMARY_K,
    donor_cluster_count == DISSIMILARITY_IV_PRIMARY_DONOR_COUNT
  ) %>%
  select(
    aewr_region_id,
    iv_k,
    target_cluster,
    donor_cluster,
    donor_rank,
    donor_cluster_distance,
    donor_cluster_count
  ) %>%
  distinct()
assert_unique_keys(
  primary_donor_map,
  c("aewr_region_id", "target_cluster", "donor_rank"),
  "Primary donor map"
)
donor_counts <- primary_donor_map %>%
  count(aewr_region_id, target_cluster, name = "selected_donors")
if (
  nrow(donor_counts) != 17L * DISSIMILARITY_IV_PRIMARY_K ||
    any(
      donor_counts$selected_donors != DISSIMILARITY_IV_PRIMARY_DONOR_COUNT
    )
) {
  stop("Every target cluster must have exactly two donors.", call. = FALSE)
}

county_units <- county_panel %>%
  distinct(
    county_fips,
    countyname,
    state_fips,
    state_abbrev,
    cz_id,
    aewr_region_id,
    panel_iv_target_unit_id
  ) %>%
  inner_join(
    primary_clusters,
    by = c("panel_iv_target_unit_id", "aewr_region_id"),
    relationship = "many-to-one"
  )
assert_unique_keys(county_units, "county_fips", "County-to-target-unit map")

primary_weights <- read_parquet(
  path_int("panel_iv_fls_geography_weight_summary.parquet")
) %>%
  filter(
    source_year %in% source_years,
    specification == DISSIMILARITY_IV_PRIMARY_WEIGHT_SPECIFICATION,
    weight_spec == DISSIMILARITY_IV_PRIMARY_WEIGHT_SPEC,
    moment_spec == DISSIMILARITY_IV_PRIMARY_MOMENT_SPEC,
    is_primary,
    wage_target_used,
    near(rho, DISSIMILARITY_IV_PRIMARY_RHO),
    near(
      kappa_multiplier,
      DISSIMILARITY_IV_PRIMARY_KAPPA_MULTIPLIER
    ),
    is.na(weight_draw_id),
    str_detect(center_solver_status, "^calibrated"),
    is.finite(calibrated_center_weight),
    calibrated_center_weight >= 0
  ) %>%
  transmute(
    aewr_region_id,
    source_year = as.integer(source_year),
    oews_area_code,
    oews_area_name,
    calibrated_center_weight
  )
assert_unique_keys(
  primary_weights,
  c("aewr_region_id", "source_year", "oews_area_code"),
  "Primary entropy area weights"
)
assert_complete_cells(
  primary_weights %>%
    distinct(aewr_region_id, source_year),
  expected_source_cells,
  c("aewr_region_id", "source_year"),
  "Primary entropy region-year support"
)
primary_weight_sums <- primary_weights %>%
  group_by(aewr_region_id, source_year) %>%
  summarise(
    entropy_area_weight_sum = sum(calibrated_center_weight),
    .groups = "drop"
  )
assert_close(
  primary_weight_sums$entropy_area_weight_sum,
  1,
  1e-8,
  "Primary entropy area-weight conservation"
)

wage_features <- read_parquet(
  path_int("panel_iv_fls_geography_wage_features.parquet")
) %>%
  filter(
    source_year %in% source_years,
    moment_spec == DISSIMILARITY_IV_PRIMARY_MOMENT_SPEC,
    wage_target_used
  ) %>%
  transmute(
    aewr_region_id,
    source_year = as.integer(source_year),
    oews_area_code,
    oews_area_name,
    oews_area_mean_hourly_wage,
    oews_area_wage_observed,
    fls_field_livestock_mean_hourly_wage
  )
assert_unique_keys(
  wage_features,
  c("aewr_region_id", "source_year", "oews_area_code"),
  "OEWS wage features"
)
assert_complete_cells(
  wage_features %>%
    distinct(aewr_region_id, source_year),
  expected_source_cells,
  c("aewr_region_id", "source_year"),
  "OEWS wage-feature region-year support"
)

area_wages_and_weights <- wage_features %>%
  inner_join(
    primary_weights %>%
      select(
        aewr_region_id,
        source_year,
        oews_area_code,
        calibrated_center_weight
      ),
    by = c("aewr_region_id", "source_year", "oews_area_code"),
    relationship = "one-to-one"
  )
if (nrow(area_wages_and_weights) != nrow(wage_features)) {
  stop("Primary weights and wage-feature support differ.", call. = FALSE)
}

# AEWR and OEWS real-wage series --------------------------------------------

aewr_policy_wages <- county_panel %>%
  filter(as.integer(year) %in% policy_years) %>%
  transmute(
    aewr_region_id,
    policy_year = as.integer(year),
    effective_aewr_nominal = as.numeric(aewr)
  ) %>%
  filter(is.finite(effective_aewr_nominal)) %>%
  distinct()
assert_unique_keys(
  aewr_policy_wages,
  c("aewr_region_id", "policy_year"),
  "Effective AEWR wages"
)
assert_complete_cells(
  aewr_policy_wages %>%
    select(aewr_region_id, policy_year),
  expected_policy_cells,
  c("aewr_region_id", "policy_year"),
  "Effective AEWR region-year support"
)

regional_oews_wages <- area_wages_and_weights %>%
  filter(
    oews_area_wage_observed,
    is.finite(oews_area_mean_hourly_wage),
    oews_area_mean_hourly_wage > 0
  ) %>%
  group_by(aewr_region_id, source_year) %>%
  summarise(
    oews_simple_mean_nominal = mean(oews_area_mean_hourly_wage),
    oews_entropy_mean_nominal = positive_weighted_mean(
      oews_area_mean_hourly_wage,
      calibrated_center_weight
    ),
    observed_oews_area_count = n_distinct(oews_area_code),
    observed_entropy_weight_mass = sum(calibrated_center_weight),
    .groups = "drop"
  )
assert_unique_keys(
  regional_oews_wages,
  c("aewr_region_id", "source_year"),
  "Regional OEWS wages"
)
assert_complete_cells(
  regional_oews_wages %>%
    select(aewr_region_id, source_year),
  expected_source_cells,
  c("aewr_region_id", "source_year"),
  "Regional OEWS wage support"
)

wage_series <- regional_oews_wages %>%
  mutate(policy_year = source_year + 1L) %>%
  inner_join(
    aewr_policy_wages,
    by = c("aewr_region_id", "policy_year"),
    relationship = "one-to-one"
  ) %>%
  inner_join(
    ppi %>%
      transmute(
        policy_year = year,
        policy_year_ppi_2012 = ppi_2012
      ),
    by = "policy_year",
    relationship = "many-to-one"
  ) %>%
  inner_join(
    region_labels,
    by = "aewr_region_id",
    relationship = "many-to-one"
  ) %>%
  pivot_longer(
    cols = c(
      effective_aewr_nominal,
      oews_simple_mean_nominal,
      oews_entropy_mean_nominal
    ),
    names_to = "wage_series",
    values_to = "nominal_hourly_wage"
  ) %>%
  mutate(
    wage_series_label = recode(
      wage_series,
      effective_aewr_nominal = "Effective AEWR",
      oews_simple_mean_nominal = "OEWS simple area mean",
      oews_entropy_mean_nominal = "OEWS entropy-weighted area mean"
    ),
    real_2012_hourly_wage = nominal_hourly_wage / policy_year_ppi_2012
  ) %>%
  arrange(aewr_region_number, policy_year, wage_series)

assert_unique_keys(
  wage_series,
  c("aewr_region_id", "policy_year", "wage_series"),
  "Real-wage plotting data"
)
if (
  nrow(wage_series) != 17L * length(policy_years) * 3L ||
    any(!is.finite(wage_series$real_2012_hourly_wage))
) {
  stop("The real-wage plotting grid is incomplete.", call. = FALSE)
}

write_csv(
  wage_series,
  path_tables("iv_aewr_region_real_wage_series.csv"),
  na = ""
)

wage_series_colors <- c(
  "Effective AEWR" = "#1B1B1B",
  "OEWS simple area mean" = "#0072B2",
  "OEWS entropy-weighted area mean" = "#D55E00"
)
wage_series_plot <- wage_series %>%
  mutate(
    aewr_region_label = factor(
      aewr_region_label,
      levels = region_labels$aewr_region_label
    ),
    wage_series_label = factor(
      wage_series_label,
      levels = names(wage_series_colors)
    )
  ) %>%
  ggplot(
    aes(
      x = policy_year,
      y = real_2012_hourly_wage,
      color = wage_series_label
    )
  ) +
  geom_line(linewidth = 0.55) +
  geom_point(size = 0.75) +
  facet_wrap(
    vars(aewr_region_label),
    ncol = 4,
    labeller = label_wrap_gen(width = 32)
  ) +
  scale_color_manual(values = wage_series_colors, drop = FALSE) +
  scale_x_continuous(
    breaks = c(min(policy_years), 2017L, max(policy_years))
  ) +
  scale_y_continuous(
    labels = scales::label_dollar(accuracy = 1)
  ) +
  labs(
    title = "Effective AEWR and Observed OEWS Wages by Region",
    subtitle = paste0(
      "OEWS source-year wages are aligned to the following AEWR policy year; ",
      "all series are in 2012 dollars."
    ),
    x = "AEWR policy year",
    y = "Real hourly wage (2012 dollars)",
    color = NULL,
    caption = paste0(
      "Simple OEWS means give each observed area equal weight. Entropy means ",
      "renormalize primary area weights over the same observed areas."
    )
  ) +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold", size = 8),
    plot.title = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0)
  )

ggsave(
  path_figures("fig_iv_aewr_region_real_wage_series.png"),
  wage_series_plot,
  width = 15,
  height = 12,
  dpi = 300,
  device = "png",
  bg = "white"
)

national_wage_series <- wage_series %>%
  group_by(wage_series_label, policy_year) %>%
  summarise(real_2012_hourly_wage = mean(real_2012_hourly_wage)) %>%
  ungroup()

national_wage_series_plot <- national_wage_series %>%
  mutate(
    wage_series_label = factor(
      wage_series_label,
      levels = names(wage_series_colors)
    )
  ) %>%
  ggplot(
    aes(
      x = policy_year,
      y = real_2012_hourly_wage,
      color = wage_series_label
    )
  ) +
  scale_color_manual(values = wage_series_colors, drop = FALSE) +
  geom_line(linewidth = 0.55) +
  geom_point(size = 0.75) +
  labs(
    title = "Effective AEWR and Observed OEWS Wages",
    subtitle = paste0(
      "OEWS source-year wages are aligned to the following AEWR policy year; ",
      "all series are in 2012 dollars."
    ),
    x = "AEWR policy year",
    y = "Real hourly wage (2012 dollars)",
    color = NULL,
    caption = paste0(
      "Simple OEWS means give each AEWR region equal weight. Entropy means ",
      "renormalize primary area weights over the same observed areas."
    )
  ) +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold", size = 8),
    plot.title = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0)
  )

ggsave(
  path_figures("fig_iv_national_real_wage_series.png"),
  national_wage_series_plot,
  width = 15,
  height = 12,
  dpi = 300,
  device = "png",
  bg = "white"
)

# FLS versus own-CZ observed OEWS wages -------------------------------------

county_area_prior <- read_parquet(
  path_int("panel_iv_fls_geography_county_area_prior.parquet")
) %>%
  filter(
    source_year %in% source_years,
    baseline_weight_spec == DISSIMILARITY_IV_FRAME_WEIGHT_SPEC,
    geographic_allocation_spec == DISSIMILARITY_IV_GEOGRAPHIC_ALLOCATION_SPEC
  ) %>%
  transmute(
    county_fips,
    aewr_region_id,
    source_year = as.integer(source_year),
    oews_area_code,
    baseline_county_conditional_within_area
  )
assert_unique_keys(
  county_area_prior,
  c(
    "county_fips",
    "aewr_region_id",
    "source_year",
    "oews_area_code"
  ),
  "County-area prior"
)
county_conditional_sums <- county_area_prior %>%
  group_by(aewr_region_id, source_year, oews_area_code) %>%
  summarise(
    county_conditional_sum = sum(baseline_county_conditional_within_area),
    .groups = "drop"
  )
assert_close(
  county_conditional_sums$county_conditional_sum,
  1,
  1e-8,
  "Within-area county-share conservation"
)

unmapped_county_areas <- county_area_prior %>%
  anti_join(
    county_units %>%
      select(county_fips, aewr_region_id),
    by = c("county_fips", "aewr_region_id")
  )
if (nrow(unmapped_county_areas) > 0L) {
  stop("County-area rows lack CZ x AEWR-region units.", call. = FALSE)
}

expanded_entropy_weights <- county_area_prior %>%
  inner_join(
    primary_weights %>%
      select(
        aewr_region_id,
        source_year,
        oews_area_code,
        calibrated_center_weight
      ),
    by = c("aewr_region_id", "source_year", "oews_area_code"),
    relationship = "many-to-one"
  ) %>%
  mutate(
    entropy_county_area_weight = calibrated_center_weight *
      baseline_county_conditional_within_area
  )

expanded_weight_sums <- expanded_entropy_weights %>%
  group_by(aewr_region_id, source_year) %>%
  summarise(
    expanded_entropy_weight_sum = sum(entropy_county_area_weight),
    .groups = "drop"
  )
assert_complete_cells(
  expanded_weight_sums %>%
    select(aewr_region_id, source_year),
  expected_source_cells,
  c("aewr_region_id", "source_year"),
  "Expanded entropy-weight region-year support"
)
assert_close(
  expanded_weight_sums$expanded_entropy_weight_sum,
  1,
  1e-8,
  "Expanded entropy-weight conservation"
)

cz_area_wages <- expanded_entropy_weights %>%
  inner_join(
    county_units %>%
      select(
        county_fips,
        aewr_region_id,
        cz_id,
        panel_iv_target_unit_id
      ),
    by = c("county_fips", "aewr_region_id"),
    relationship = "many-to-one"
  ) %>%
  inner_join(
    wage_features,
    by = c("aewr_region_id", "source_year", "oews_area_code"),
    relationship = "many-to-one"
  ) %>%
  group_by(
    panel_iv_target_unit_id,
    cz_id,
    aewr_region_id,
    source_year,
    oews_area_code
  ) %>%
  summarise(
    oews_area_name = first_nonmissing_character(oews_area_name),
    oews_area_mean_hourly_wage = first(oews_area_mean_hourly_wage),
    oews_area_wage_observed = first(oews_area_wage_observed),
    fls_field_livestock_mean_hourly_wage = first(
      fls_field_livestock_mean_hourly_wage
    ),
    cz_area_entropy_weight = sum(entropy_county_area_weight),
    mapped_counties_in_cz = n_distinct(county_fips),
    .groups = "drop"
  )
assert_unique_keys(
  cz_area_wages,
  c(
    "panel_iv_target_unit_id",
    "aewr_region_id",
    "source_year",
    "oews_area_code"
  ),
  "CZ-area wage support"
)

cz_wage_comparison_wide <- cz_area_wages %>%
  filter(
    oews_area_wage_observed,
    is.finite(oews_area_mean_hourly_wage),
    oews_area_mean_hourly_wage > 0,
    is.finite(fls_field_livestock_mean_hourly_wage),
    fls_field_livestock_mean_hourly_wage > 0
  ) %>%
  group_by(
    panel_iv_target_unit_id,
    cz_id,
    aewr_region_id,
    source_year
  ) %>%
  summarise(
    fls_field_livestock_nominal = first(fls_field_livestock_mean_hourly_wage),
    oews_simple_mean_nominal = mean(oews_area_mean_hourly_wage),
    oews_entropy_mean_nominal = positive_weighted_mean(
      oews_area_mean_hourly_wage,
      cz_area_entropy_weight
    ),
    observed_oews_area_count = n_distinct(oews_area_code),
    observed_cz_entropy_weight_mass = sum(cz_area_entropy_weight),
    .groups = "drop"
  ) %>%
  inner_join(
    ppi %>%
      transmute(
        source_year = year,
        source_year_ppi_2012 = ppi_2012
      ),
    by = "source_year",
    relationship = "many-to-one"
  ) %>%
  mutate(
    fls_field_livestock_real_2012 = fls_field_livestock_nominal /
      source_year_ppi_2012,
    oews_simple_mean_real_2012 = oews_simple_mean_nominal /
      source_year_ppi_2012,
    oews_entropy_mean_real_2012 = oews_entropy_mean_nominal /
      source_year_ppi_2012
  ) %>%
  filter(
    is.finite(fls_field_livestock_real_2012),
    is.finite(oews_simple_mean_real_2012),
    is.finite(oews_entropy_mean_real_2012)
  )
assert_unique_keys(
  cz_wage_comparison_wide,
  c("panel_iv_target_unit_id", "aewr_region_id", "source_year"),
  "CZ wage-comparison support"
)
if (nrow(cz_wage_comparison_wide) == 0L) {
  stop("The CZ wage comparison has no finite observations.", call. = FALSE)
}

cz_wage_comparison <- cz_wage_comparison_wide %>%
  pivot_longer(
    cols = c(
      oews_simple_mean_real_2012,
      oews_entropy_mean_real_2012
    ),
    names_to = "comparison_panel",
    values_to = "oews_real_2012"
  ) %>%
  mutate(
    comparison_panel_label = recode(
      comparison_panel,
      oews_simple_mean_real_2012 = "Panel A: Simple mean of observed OEWS areas",
      oews_entropy_mean_real_2012 = "Panel B: Entropy-weighted mean within CZ"
    )
  ) %>%
  inner_join(
    region_labels,
    by = "aewr_region_id",
    relationship = "many-to-one"
  ) %>%
  arrange(
    comparison_panel,
    aewr_region_number,
    source_year,
    as.integer(cz_id)
  )
assert_unique_keys(
  cz_wage_comparison,
  c(
    "comparison_panel",
    "panel_iv_target_unit_id",
    "aewr_region_id",
    "source_year"
  ),
  "CZ scatter plotting data"
)
scatter_panel_support <- cz_wage_comparison %>%
  count(
    panel_iv_target_unit_id,
    aewr_region_id,
    source_year,
    name = "panel_count"
  )
if (
  any(scatter_panel_support$panel_count != 2L) ||
    any(!is.finite(cz_wage_comparison$fls_field_livestock_real_2012)) ||
    any(!is.finite(cz_wage_comparison$oews_real_2012))
) {
  stop("Scatter panels do not use identical finite support.", call. = FALSE)
}

write_csv(
  cz_wage_comparison,
  path_tables("iv_fls_oews_cz_scatter.csv"),
  na = ""
)

scatter_limits <- range(
  c(
    cz_wage_comparison$fls_field_livestock_real_2012,
    cz_wage_comparison$oews_real_2012
  )
)
scatter_padding <- max(diff(scatter_limits) * 0.04, 0.25)
scatter_limits <- scatter_limits + c(-scatter_padding, scatter_padding)

cz_wage_scatter_plot <- cz_wage_comparison %>%
  mutate(
    comparison_panel_label = factor(
      comparison_panel_label,
      levels = c(
        "Panel A: Simple mean of observed OEWS areas",
        "Panel B: Entropy-weighted mean within CZ"
      )
    )
  ) %>%
  ggplot(
    aes(
      x = fls_field_livestock_real_2012,
      y = oews_real_2012
    )
  ) +
  geom_abline(
    intercept = 0,
    slope = 1,
    color = "grey35",
    linetype = "dashed",
    linewidth = 0.55
  ) +
  geom_point(
    color = "#0072B2",
    alpha = 0.35,
    size = 0.85
  ) +
  facet_wrap(vars(comparison_panel_label), nrow = 1) +
  coord_equal(
    xlim = scatter_limits,
    ylim = scatter_limits,
    expand = FALSE
  ) +
  scale_x_continuous(labels = scales::label_dollar(accuracy = 1)) +
  scale_y_continuous(labels = scales::label_dollar(accuracy = 1)) +
  labs(
    title = "FLS Wages versus Observed OEWS Wages in the Same CZ",
    subtitle = paste0(
      "CZ x AEWR-region units and source years 2011-2021; ",
      "the dashed line is equality."
    ),
    x = "FLS field-and-livestock hourly wage (2012 dollars)",
    y = "Own-CZ OEWS hourly wage (2012 dollars)",
    caption = str_wrap(
      paste0(
        "Panel A weights each unique observed area equally. Panel B expands ",
        "primary area weights with within-area county shares and renormalizes ",
        "over observed areas touching the CZ."
      ),
      width = 105
    )
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0)
  )

ggsave(
  path_figures("fig_iv_fls_oews_cz_scatter.png"),
  cz_wage_scatter_plot,
  width = 11,
  height = 5.8,
  dpi = 300,
  device = "png",
  bg = "white"
)

# Annual changes in CZ entropy weights --------------------------------------

cz_entropy_weights_observed <- expanded_entropy_weights %>%
  inner_join(
    county_units %>%
      select(
        county_fips,
        aewr_region_id,
        cz_id,
        panel_iv_target_unit_id
      ),
    by = c("county_fips", "aewr_region_id"),
    relationship = "many-to-one"
  ) %>%
  group_by(
    panel_iv_target_unit_id,
    cz_id,
    aewr_region_id,
    source_year
  ) %>%
  summarise(
    cz_entropy_weight = sum(entropy_county_area_weight),
    .groups = "drop"
  )
assert_unique_keys(
  cz_entropy_weights_observed,
  c("panel_iv_target_unit_id", "aewr_region_id", "source_year"),
  "Observed CZ entropy weights"
)

target_units <- primary_clusters %>%
  inner_join(
    county_units %>%
      distinct(
        panel_iv_target_unit_id,
        cz_id,
        aewr_region_id
      ),
    by = c("panel_iv_target_unit_id", "aewr_region_id"),
    relationship = "one-to-one"
  ) %>%
  select(
    panel_iv_target_unit_id,
    cz_id,
    aewr_region_id,
    target_cluster
  )
assert_unique_keys(
  target_units,
  c("panel_iv_target_unit_id", "aewr_region_id"),
  "Target units"
)

cz_entropy_weights <- target_units %>%
  crossing(source_year = source_years) %>%
  left_join(
    cz_entropy_weights_observed,
    by = c(
      "panel_iv_target_unit_id",
      "cz_id",
      "aewr_region_id",
      "source_year"
    ),
    relationship = "one-to-one"
  ) %>%
  mutate(cz_entropy_weight = replace_na(cz_entropy_weight, 0)) %>%
  group_by(panel_iv_target_unit_id, aewr_region_id) %>%
  arrange(source_year, .by_group = TRUE) %>%
  mutate(
    previous_source_year = lag(source_year),
    previous_cz_entropy_weight = lag(cz_entropy_weight),
    entropy_weight_change_pp = if_else(
      source_year - previous_source_year == 1L,
      100 * (cz_entropy_weight - previous_cz_entropy_weight),
      NA_real_
    )
  ) %>%
  ungroup()
assert_unique_keys(
  cz_entropy_weights,
  c("panel_iv_target_unit_id", "aewr_region_id", "source_year"),
  "CZ entropy-weight changes"
)

cz_weight_conservation <- cz_entropy_weights %>%
  group_by(aewr_region_id, source_year) %>%
  summarise(
    region_year_entropy_weight_sum = sum(cz_entropy_weight),
    .groups = "drop"
  )
assert_complete_cells(
  cz_weight_conservation %>%
    select(aewr_region_id, source_year),
  expected_source_cells,
  c("aewr_region_id", "source_year"),
  "CZ entropy-weight region-year support"
)
assert_close(
  cz_weight_conservation$region_year_entropy_weight_sum,
  1,
  1e-8,
  "CZ entropy-weight conservation"
)

cz_change_conservation <- cz_entropy_weights %>%
  filter(is.finite(entropy_weight_change_pp)) %>%
  group_by(aewr_region_id, source_year) %>%
  summarise(
    region_year_entropy_weight_change_pp = sum(entropy_weight_change_pp),
    .groups = "drop"
  )
assert_close(
  cz_change_conservation$region_year_entropy_weight_change_pp,
  0,
  1e-8,
  "Annual CZ entropy-weight-change conservation"
)

cz_entropy_weight_changes <- cz_entropy_weights %>%
  left_join(
    cz_weight_conservation,
    by = c("aewr_region_id", "source_year"),
    relationship = "many-to-one"
  ) %>%
  left_join(
    cz_change_conservation,
    by = c("aewr_region_id", "source_year"),
    relationship = "many-to-one"
  ) %>%
  inner_join(
    region_labels,
    by = "aewr_region_id",
    relationship = "many-to-one"
  ) %>%
  arrange(aewr_region_number, as.integer(cz_id), source_year)

write_csv(
  cz_entropy_weight_changes,
  path_tables("iv_cz_entropy_weight_changes_pp.csv"),
  na = ""
)

weight_change_plot_data <- cz_entropy_weight_changes %>%
  filter(is.finite(entropy_weight_change_pp))
if (nrow(weight_change_plot_data) == 0L) {
  stop("No valid annual CZ entropy-weight changes.", call. = FALSE)
}

weight_change_plot <- ggplot(
  weight_change_plot_data,
  aes(x = entropy_weight_change_pp)
) +
  geom_histogram(
    bins = 60,
    fill = "#0072B2",
    color = "white",
    linewidth = 0.2
  ) +
  geom_vline(
    xintercept = 0,
    color = "#D55E00",
    linetype = "dashed",
    linewidth = 0.7
  ) +
  scale_y_continuous(
    trans = scales::transform_log1p(),
    breaks = c(0, 1, 10, 100, 1000, 5000),
    labels = scales::label_comma()
  ) +
  labs(
    title = "Annual Changes in CZ Entropy Weights",
    subtitle = paste0(
      "Consecutive source-year changes pooled across CZ x AEWR-region units"
    ),
    x = "Change in CZ weight (percentage points)",
    y = "CZ-year changes (log1p scale)",
    caption = str_wrap(
      paste0(
        "Changes equal 100 \u00d7 (weight[t] - weight[t-1]); they are not ",
        "percentage growth rates. Changes sum to zero within every ",
        "region-year."
      ),
      width = 90
    )
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0)
  )

ggsave(
  path_figures("fig_iv_cz_entropy_weight_changes_pp.png"),
  weight_change_plot,
  width = 8,
  height = 5.5,
  dpi = 300,
  device = "png",
  bg = "white"
)

# California target and donor example --------------------------------------

california_region_id <- "17"
california_unit_employment <- county_panel %>%
  filter(
    as.integer(year) == 2011L,
    aewr_region_id == california_region_id
  ) %>%
  group_by(
    panel_iv_target_unit_id,
    cz_id,
    aewr_region_id
  ) %>%
  summarise(
    farm_employment_2011 = sum(emp_farm, na.rm = TRUE),
    counties = paste(sort(unique(countyname)), collapse = " | "),
    county_count = n_distinct(county_fips),
    .groups = "drop"
  ) %>%
  inner_join(
    primary_clusters %>%
      select(
        panel_iv_target_unit_id,
        aewr_region_id,
        target_cluster
      ),
    by = c("panel_iv_target_unit_id", "aewr_region_id"),
    relationship = "one-to-one"
  )
assert_unique_keys(
  california_unit_employment,
  c("panel_iv_target_unit_id", "aewr_region_id"),
  "California target units"
)

selected_california_target <- california_unit_employment %>%
  arrange(
    desc(farm_employment_2011),
    as.integer(cz_id),
    panel_iv_target_unit_id
  ) %>%
  slice(1L)
if (
  nrow(selected_california_target) != 1L ||
    selected_california_target$cz_id[[1]] != "62"
) {
  stop(
    "The deterministic California target must be CZ 62.",
    call. = FALSE
  )
}

california_target_cluster <-
  selected_california_target$target_cluster[[1]]
california_donors <- primary_donor_map %>%
  filter(
    aewr_region_id == california_region_id,
    target_cluster == california_target_cluster
  ) %>%
  arrange(donor_rank)
expected_california_donor_ranks <- seq_len(
  DISSIMILARITY_IV_PRIMARY_DONOR_COUNT
)
if (
  nrow(california_donors) != DISSIMILARITY_IV_PRIMARY_DONOR_COUNT ||
    !identical(
      california_donors$donor_rank,
      expected_california_donor_ranks
    ) ||
    any(california_donors$target_cluster != california_target_cluster) ||
    any(california_donors$donor_cluster == california_target_cluster) ||
    n_distinct(california_donors$donor_cluster) !=
      DISSIMILARITY_IV_PRIMARY_DONOR_COUNT
) {
  stop(
    "The California target cluster has an invalid donor ranking.",
    call. = FALSE
  )
}

selected_california_role <- paste0(
  "Selected target: CZ ",
  selected_california_target$cz_id[[1]]
)
california_target_cluster_role <- paste0(
  "Other CZs in target cluster ",
  california_target_cluster
)
california_donor_roles <- paste0(
  "Donor rank ",
  california_donors$donor_rank,
  ": cluster ",
  california_donors$donor_cluster
)
california_unused_role <- "Unused clusters"

california_plot_metadata <- california_unit_employment %>%
  left_join(
    california_donors %>%
      select(
        donor_cluster,
        donor_rank,
        donor_cluster_distance
      ),
    by = c("target_cluster" = "donor_cluster"),
    relationship = "many-to-one"
  ) %>%
  mutate(
    selected_target = panel_iv_target_unit_id ==
      selected_california_target$panel_iv_target_unit_id[[1]],
    selected_target_cz_id = selected_california_target$cz_id[[1]],
    selected_target_cluster = california_target_cluster,
    selection_role = case_when(
      selected_target ~ selected_california_role,
      target_cluster == california_target_cluster ~
        california_target_cluster_role,
      donor_rank == 1L ~ california_donor_roles[[1]],
      donor_rank == 2L ~ california_donor_roles[[2]],
      TRUE ~ california_unused_role
    )
  ) %>%
  arrange(
    desc(selected_target),
    donor_rank,
    target_cluster,
    as.integer(cz_id)
  )
if (
  sum(california_plot_metadata$selected_target) != 1L ||
    n_distinct(
      california_plot_metadata$target_cluster[
        !is.na(california_plot_metadata$donor_rank)
      ]
    ) !=
      2L
) {
  stop("California selection metadata is incomplete.", call. = FALSE)
}

write_csv(
  california_plot_metadata,
  path_tables("iv_california_target_and_donors.csv"),
  na = ""
)

california_counties <- county_units %>%
  filter(aewr_region_id == california_region_id) %>%
  select(
    county_fips,
    panel_iv_target_unit_id,
    cz_id,
    aewr_region_id
  ) %>%
  inner_join(
    california_plot_metadata %>%
      select(panel_iv_target_unit_id, selection_role),
    by = "panel_iv_target_unit_id",
    relationship = "many-to-one"
  )
assert_unique_keys(
  california_counties,
  "county_fips",
  "California county map metadata"
)

county_shape_zip <- path_raw("county_shapefile", "tl_2020_us_county.zip")
if (!file.exists(county_shape_zip)) {
  stop("The county shapefile archive is missing.", call. = FALSE)
}
shape_directory <- tempfile("panel-iv-figures-")
dir.create(shape_directory)
on.exit(
  unlink(shape_directory, recursive = TRUE, force = TRUE),
  add = TRUE
)
utils::unzip(county_shape_zip, exdir = shape_directory)
county_shape_path <- list.files(
  shape_directory,
  pattern = "tl_2020_us_county[.]shp$",
  full.names = TRUE
)
if (length(county_shape_path) != 1L) {
  stop("Expected exactly one county shapefile.", call. = FALSE)
}

california_map <- sf::st_read(
  county_shape_path[[1]],
  quiet = TRUE
) %>%
  mutate(
    state_fips = state_fips(STATEFP),
    county_fips = combine_county_fips(STATEFP, COUNTYFP)
  ) %>%
  filter(state_fips == "06") %>%
  sf::st_make_valid() %>%
  sf::st_transform(5070) %>%
  inner_join(
    california_counties,
    by = "county_fips",
    relationship = "one-to-one"
  )
if (nrow(california_map) != nrow(california_counties)) {
  stop(
    "California geometry does not cover every analysis county.",
    call. = FALSE
  )
}

california_unit_boundaries <- california_map %>%
  group_by(panel_iv_target_unit_id, selection_role) %>%
  summarise(geometry = sf::st_union(geometry), .groups = "drop")
california_target_boundary <- california_unit_boundaries %>%
  filter(
    panel_iv_target_unit_id ==
      selected_california_target$panel_iv_target_unit_id[[1]]
  )

california_role_levels <- c(
  selected_california_role,
  california_donor_roles,
  california_target_cluster_role,
  california_unused_role
)
california_role_colors <- setNames(
  c(
    "#D55E00",
    "#0072B2",
    "#56B4E9",
    "#E69F00",
    "#D9D9D9"
  ),
  california_role_levels
)

california_target_plot <- california_map %>%
  mutate(
    selection_role = factor(
      selection_role,
      levels = california_role_levels
    )
  ) %>%
  ggplot() +
  geom_sf(
    aes(fill = selection_role),
    color = "white",
    linewidth = 0.15
  ) +
  geom_sf(
    data = california_unit_boundaries,
    fill = NA,
    color = scales::alpha("grey20", 0.7),
    linewidth = 0.3
  ) +
  geom_sf(
    data = california_target_boundary,
    fill = NA,
    color = "black",
    linewidth = 1.1
  ) +
  scale_fill_manual(
    values = california_role_colors,
    breaks = california_role_levels,
    drop = FALSE,
    name = NULL
  ) +
  coord_sf(datum = NA) +
  labs(
    title = "California Target CZ and Its Dissimilar Donor Clusters",
    subtitle = str_wrap(
      paste0(
        "CZ 62 (Fresno and neighboring Central Valley counties) has the ",
        "largest 2011 farm employment."
      ),
      width = 75
    ),
    caption = str_wrap(
      paste0(
        "The black outline marks the selected target. Donor ranks are fixed ",
        "by distance between five cluster centroids within AEWR Region 17."
      ),
      width = 82
    )
  ) +
  theme_void(base_size = 11) +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.text = element_text(size = 9),
    plot.title = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0),
    plot.background = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA)
  ) +
  guides(fill = guide_legend(nrow = 3, byrow = TRUE))

ggsave(
  path_figures("fig_iv_california_target_and_donors.png"),
  california_target_plot,
  width = 8.5,
  height = 7,
  dpi = 300,
  device = "png",
  bg = "white"
)

# Target-donor covariate similarity -----------------------------------------

unit_features <- read_parquet(
  path_int("panel_iv_target_unit_features.parquet")
)
feature_names <- setdiff(
  names(unit_features),
  c(
    "cz_id",
    "aewr_region_id",
    "panel_iv_target_unit_id",
    "unit_feature_weight"
  )
)
soil_continuous_names <- c(
  "slope_r",
  "slopegradwta",
  "resdept_r",
  "aws025wta",
  "aws050wta",
  "aws0100wta",
  "aws0150wta",
  "wtdepannmin",
  "wtdepaprjunmin",
  "brockdepmin",
  "cropprodindex"
)
feature_blocks <- list(
  crops = feature_names[str_detect(feature_names, "^share_cdl_")],
  climate = feature_names[str_detect(feature_names, "^normal_cb_")],
  soil_continuous = intersect(feature_names, soil_continuous_names),
  soil_categorical = feature_names[str_detect(feature_names, "^share_soil_")]
)
similarity_covariates <- tribble(
  ~feature_name                        ,
  ~covariate_label                     ,
  ~feature_block                       ,
  "share_cdl_fruit_tree_nuts"          ,
  "Fruit/tree-nut acreage composition" ,
  "crops"                              ,
  "normal_cb_temp_tavg_b00"            ,
  "Temperature-normal basis b00"       ,
  "climate"                            ,
  "aws0150wta"                         ,
  "Available water storage, 0-150 cm"  ,
  "soil_continuous"
) %>%
  mutate(
    block_feature_count = vapply(
      feature_block,
      function(block_name) length(feature_blocks[[block_name]]),
      integer(1)
    )
  )
if (
  !all(similarity_covariates$feature_name %in% feature_names) ||
    any(similarity_covariates$block_feature_count <= 0L)
) {
  stop("Similarity covariates or feature blocks are missing.", call. = FALSE)
}

transformed_region_list <- list()
for (region_id in region_labels$aewr_region_id) {
  region_features <- unit_features %>%
    filter(aewr_region_id == region_id) %>%
    select(
      cz_id,
      aewr_region_id,
      panel_iv_target_unit_id,
      all_of(similarity_covariates$feature_name)
    )

  for (feature_index in seq_len(nrow(similarity_covariates))) {
    feature_name <- similarity_covariates$feature_name[[feature_index]]
    block_count <-
      similarity_covariates$block_feature_count[[feature_index]]
    value <- region_features[[feature_name]]
    value[is.nan(value)] <- NA_real_
    region_median <- median(value, na.rm = TRUE)
    if (is.na(region_median)) {
      region_median <- 0
    }
    value[is.na(value)] <- region_median
    region_sd <- sd(value)
    standardized_value <- if (
      is.finite(region_sd) &&
        region_sd > 0
    ) {
      (value - mean(value)) / region_sd
    } else {
      rep(0, length(value))
    }
    region_features[[feature_name]] <-
      standardized_value / sqrt(block_count)
  }

  transformed_region_list[[region_id]] <- region_features
}

transformed_similarity_features <- bind_rows(transformed_region_list) %>%
  inner_join(
    primary_clusters %>%
      select(
        panel_iv_target_unit_id,
        aewr_region_id,
        target_cluster
      ),
    by = c("panel_iv_target_unit_id", "aewr_region_id"),
    relationship = "one-to-one"
  ) %>%
  pivot_longer(
    cols = all_of(similarity_covariates$feature_name),
    names_to = "feature_name",
    values_to = "transformed_feature_value"
  ) %>%
  inner_join(
    similarity_covariates,
    by = "feature_name",
    relationship = "many-to-one"
  )
if (
  any(
    !is.finite(
      transformed_similarity_features$transformed_feature_value
    )
  )
) {
  stop("Clustering transformations produced nonfinite values.", call. = FALSE)
}

target_cluster_means <- transformed_similarity_features %>%
  group_by(
    aewr_region_id,
    feature_name,
    covariate_label,
    feature_block,
    block_feature_count,
    target_cluster
  ) %>%
  summarise(
    target_cluster_mean = mean(transformed_feature_value),
    target_cluster_units = n(),
    .groups = "drop"
  )
cluster_mean_counts <- target_cluster_means %>%
  count(
    aewr_region_id,
    feature_name,
    name = "target_cluster_count"
  )
if (
  nrow(cluster_mean_counts) != 17L * nrow(similarity_covariates) ||
    any(
      cluster_mean_counts$target_cluster_count != DISSIMILARITY_IV_PRIMARY_K
    )
) {
  stop(
    "Similarity inputs must have five cluster means per cell.",
    call. = FALSE
  )
}

donor_cluster_means <- target_cluster_means %>%
  transmute(
    aewr_region_id,
    feature_name,
    donor_cluster = target_cluster,
    donor_cluster_mean = target_cluster_mean
  )

target_donor_similarity_points <- target_cluster_means %>%
  inner_join(
    primary_donor_map %>%
      select(
        aewr_region_id,
        target_cluster,
        donor_cluster,
        donor_rank
      ),
    by = c("aewr_region_id", "target_cluster"),
    relationship = "many-to-many"
  ) %>%
  inner_join(
    donor_cluster_means,
    by = c("aewr_region_id", "feature_name", "donor_cluster"),
    relationship = "many-to-one"
  ) %>%
  group_by(
    aewr_region_id,
    feature_name,
    covariate_label,
    feature_block,
    block_feature_count,
    target_cluster,
    target_cluster_mean,
    target_cluster_units
  ) %>%
  summarise(
    donor_cluster_mean = mean(donor_cluster_mean),
    selected_donor_clusters = n_distinct(donor_cluster),
    selected_donor_ranks = n_distinct(donor_rank),
    .groups = "drop"
  )
if (
  any(
    target_donor_similarity_points$selected_donor_clusters !=
      DISSIMILARITY_IV_PRIMARY_DONOR_COUNT
  ) ||
    any(
      target_donor_similarity_points$selected_donor_ranks !=
        DISSIMILARITY_IV_PRIMARY_DONOR_COUNT
    )
) {
  stop(
    "Similarity points do not average exactly two donor clusters.",
    call. = FALSE
  )
}

fit_similarity_regression <- function(data) {
  complete <- data %>%
    filter(
      is.finite(target_cluster_mean),
      is.finite(donor_cluster_mean)
    )
  sample_size <- nrow(complete)
  target_clusters <- n_distinct(complete$target_cluster)
  donor_variance <- if (sample_size >= 2L) {
    stats::var(complete$donor_cluster_mean)
  } else {
    NA_real_
  }
  target_variance <- if (sample_size >= 2L) {
    stats::var(complete$target_cluster_mean)
  } else {
    NA_real_
  }

  if (
    sample_size != DISSIMILARITY_IV_PRIMARY_K ||
      target_clusters != DISSIMILARITY_IV_PRIMARY_K
  ) {
    return(tibble(
      intercept = NA_real_,
      slope = NA_real_,
      sample_size = sample_size,
      target_cluster_count = target_clusters,
      regression_status = "incomplete_five_cluster_support",
      r_squared = NA_real_,
      donor_mean_variance = donor_variance,
      target_mean_variance = target_variance
    ))
  }
  if (
    !is.finite(donor_variance) ||
      donor_variance <= .Machine$double.eps
  ) {
    return(tibble(
      intercept = NA_real_,
      slope = NA_real_,
      sample_size = sample_size,
      target_cluster_count = target_clusters,
      regression_status = "zero_variance_donor_mean",
      r_squared = NA_real_,
      donor_mean_variance = donor_variance,
      target_mean_variance = target_variance
    ))
  }
  if (
    !is.finite(target_variance) ||
      target_variance <= .Machine$double.eps
  ) {
    return(tibble(
      intercept = NA_real_,
      slope = NA_real_,
      sample_size = sample_size,
      target_cluster_count = target_clusters,
      regression_status = "zero_variance_target_mean",
      r_squared = NA_real_,
      donor_mean_variance = donor_variance,
      target_mean_variance = target_variance
    ))
  }

  fit <- stats::lm(
    target_cluster_mean ~ donor_cluster_mean,
    data = complete
  )
  coefficients <- stats::coef(fit)
  fit_summary <- summary(fit)
  status <- if (
    length(coefficients) == 2L &&
      all(is.finite(coefficients)) &&
      is.finite(fit_summary$r.squared)
  ) {
    "ok"
  } else {
    "nonfinite_regression_result"
  }
  tibble(
    intercept = if (status == "ok") unname(coefficients[[1]]) else NA_real_,
    slope = if (status == "ok") unname(coefficients[[2]]) else NA_real_,
    sample_size = sample_size,
    target_cluster_count = target_clusters,
    regression_status = status,
    r_squared = if (status == "ok") fit_summary$r.squared else NA_real_,
    donor_mean_variance = donor_variance,
    target_mean_variance = target_variance
  )
}

similarity_regressions <- target_donor_similarity_points %>%
  group_by(
    aewr_region_id,
    feature_name,
    covariate_label,
    feature_block,
    block_feature_count
  ) %>%
  group_modify(~ fit_similarity_regression(.x)) %>%
  ungroup() %>%
  inner_join(
    region_labels,
    by = "aewr_region_id",
    relationship = "many-to-one"
  ) %>%
  arrange(
    match(feature_name, similarity_covariates$feature_name),
    aewr_region_number
  )
assert_unique_keys(
  similarity_regressions,
  c("aewr_region_id", "feature_name"),
  "Similarity regression results"
)
if (
  nrow(similarity_regressions) != 17L * nrow(similarity_covariates) ||
    any(
      similarity_regressions$sample_size != DISSIMILARITY_IV_PRIMARY_K
    ) ||
    any(
      similarity_regressions$target_cluster_count != DISSIMILARITY_IV_PRIMARY_K
    )
) {
  stop(
    "Every similarity regression must use five cluster observations.",
    call. = FALSE
  )
}
recognized_similarity_statuses <- c(
  "ok",
  "incomplete_five_cluster_support",
  "zero_variance_donor_mean",
  "zero_variance_target_mean",
  "nonfinite_regression_result"
)
if (
  any(
    !similarity_regressions$regression_status %in%
      recognized_similarity_statuses
  )
) {
  stop("A similarity regression has an unrecorded status.", call. = FALSE)
}

write_csv(
  similarity_regressions,
  path_tables("iv_target_donor_similarity_regressions.csv"),
  na = ""
)

similarity_plot_data <- similarity_regressions %>%
  filter(regression_status == "ok", is.finite(slope))
if (nrow(similarity_plot_data) == 0L) {
  stop("No finite target-donor similarity slopes.", call. = FALSE)
}

similarity_slope_plot <- ggplot(
  similarity_plot_data,
  aes(x = slope)
) +
  geom_histogram(
    bins = 12,
    fill = "#009E73",
    color = "white",
    linewidth = 0.25
  ) +
  geom_vline(
    xintercept = 1,
    color = "#D55E00",
    linetype = "dashed",
    linewidth = 0.7
  ) +
  facet_wrap(
    vars(covariate_label),
    nrow = 1,
    labeller = label_wrap_gen(width = 30)
  ) +
  labs(
    title = "Target-Donor Similarity across AEWR Regions",
    subtitle = paste0(
      "Within-region slopes across five target clusters; ",
      "the dashed reference is slope one."
    ),
    x = "Target-cluster mean on mean of two donor-cluster means: slope",
    y = "AEWR regions",
    caption = paste0(
      "Covariates use the same within-region imputation, standardization, ",
      "and feature-block scaling as the clustering design."
    )
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0)
  )

ggsave(
  path_figures("fig_iv_target_donor_similarity_slopes.png"),
  similarity_slope_plot,
  width = 12,
  height = 5,
  dpi = 300,
  device = "png",
  bg = "white"
)
