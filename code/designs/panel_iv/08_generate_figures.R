# Purpose: Produce retained county-calibrated panel-IV diagnostics.
# Inputs: County calibration moments and weights plus cluster-year instruments.
# Outputs: Six diagnostic PNGs and their reproducible plotting-data CSVs.
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
library(tidyr)

obsolete_area_diagnostics <- c(
  path_figures("fig_iv_aewr_region_real_wage_series.png"),
  path_figures("fig_iv_national_real_wage_series.png"),
  path_figures("fig_iv_fls_oews_cz_scatter.png"),
  path_figures("fig_iv_cz_entropy_weight_changes_pp.png"),
  path_figures("fig_iv_california_target_and_donors.png"),
  path_figures("fig_iv_target_donor_similarity_slopes.png"),
  path_figures("fig_iv_donor_wage_source_shares.png"),
  path_tables("iv_aewr_region_real_wage_series.csv"),
  path_tables("iv_fls_oews_cz_scatter.csv"),
  path_tables("iv_cz_entropy_weight_changes_pp.csv"),
  path_tables("iv_california_target_and_donors.csv"),
  path_tables("iv_target_donor_similarity_regressions.csv"),
  path_tables("iv_donor_wage_source_shares.csv")
)
unlink(obsolete_area_diagnostics[file.exists(obsolete_area_diagnostics)])

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

finite_average <- function(value) {
  value <- value[is.finite(value)]
  if (length(value) == 0L) {
    return(NA_real_)
  }
  mean(value)
}

diagnostic_theme <- theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title.position = "plot",
    legend.position = "bottom"
  )

source_years <- seq.int(
  DISSIMILARITY_IV_POLICY_START_YEAR - 1L,
  DISSIMILARITY_IV_POLICY_END_YEAR - 1L
)

county_panel <- read_parquet(
  path_processed("county_year_panel.parquet")
)
region_labels <- county_panel %>%
  filter(!is.na(aewr_region_id), !is.na(state_abbrev)) %>%
  distinct(aewr_region_id, state_abbrev) %>%
  arrange(as.integer(aewr_region_id), state_abbrev) %>%
  group_by(aewr_region_id) %>%
  summarise(
    region_states = paste(state_abbrev, collapse = ", "),
    .groups = "drop"
  ) %>%
  mutate(
    region_number = as.integer(aewr_region_id),
    region_label = paste0("Region ", aewr_region_id, ": ", region_states)
  ) %>%
  arrange(region_number)
assert_geo_columns(region_labels, "aewr_region_id")
assert_unique_keys(region_labels, "aewr_region_id", "AEWR region labels")
if (!identical(region_labels$region_number, seq_len(17L))) {
  stop("Expected labels for all 17 AEWR regions.", call. = FALSE)
}

moment_diagnostics <- read_parquet(
  path_int("panel_iv_fls_county_moment_diagnostics.parquet")
)
weight_summary <- read_parquet(
  path_int("panel_iv_fls_county_weight_summary.parquet")
)
instrument_grid <- read_parquet(
  path_int("panel_iv_instrument_cluster_year.parquet")
)

assert_geo_columns(moment_diagnostics, "aewr_region_id")
assert_geo_columns(
  weight_summary,
  c("aewr_region_id", "county_fips")
)
assert_geo_columns(instrument_grid, "aewr_region_id")
assert_unique_keys(
  moment_diagnostics,
  c(
    "aewr_region_id",
    "source_year",
    "specification",
    "moment_id"
  ),
  "County moment diagnostics"
)
assert_unique_keys(
  weight_summary,
  c(
    "aewr_region_id",
    "source_year",
    "county_fips",
    "specification"
  ),
  "County weight summary"
)
assert_unique_keys(
  instrument_grid,
  c(
    "aewr_region_id",
    "target_cluster",
    "source_year",
    "instrument_spec_label"
  ),
  "Panel-IV cluster-year instrument grid"
)

primary_moments <- moment_diagnostics %>%
  filter(
    specification == DISSIMILARITY_IV_PRIMARY_WEIGHT_SPECIFICATION,
    source_year %in% source_years
  )
primary_weights <- weight_summary %>%
  filter(
    specification == DISSIMILARITY_IV_PRIMARY_WEIGHT_SPECIFICATION,
    source_year %in% source_years
  )
primary_instruments <- instrument_grid %>%
  filter(
    instrument_spec_label == DISSIMILARITY_IV_PRIMARY_INSTRUMENT_LABEL,
    source_year %in% source_years
  )

weight_sums <- primary_weights %>%
  group_by(aewr_region_id, source_year) %>%
  summarise(
    frame_prior_sum = sum(frame_prior_weight),
    calibrated_weight_sum = sum(calibrated_center_weight),
    .groups = "drop"
  )
if (
  nrow(weight_sums) != 17L * length(source_years) ||
    any(abs(weight_sums$frame_prior_sum - 1) > 1e-10) ||
    any(abs(weight_sums$calibrated_weight_sum - 1) > 1e-10)
) {
  stop("Preferred county weights do not conserve region-year mass.", call. = FALSE)
}
if (
  nrow(primary_instruments) !=
    17L * DISSIMILARITY_IV_PRIMARY_K * length(source_years) ||
    any(primary_instruments$policy_year != primary_instruments$source_year + 1L)
) {
  stop("Preferred instrument grid violates coverage or t-1 timing.", call. = FALSE)
}

# Annual FLS/OEWS calibration by AEWR region -------------------------------

annual_wage_calibration <- primary_moments %>%
  filter(moment_id == "annual_fls_oews_hourly_wage") %>%
  select(
    aewr_region_id,
    source_year,
    fls_target_hourly_wage = raw_target,
    prior_county_oews_hourly_wage = prior_raw_moment,
    calibrated_county_oews_hourly_wage = calibrated_raw_moment,
    observed_prior_mass,
    moment_active,
    moment_status
  ) %>%
  left_join(region_labels, by = "aewr_region_id", relationship = "many-to-one") %>%
  arrange(region_number, source_year)
if (nrow(annual_wage_calibration) != 17L * length(source_years)) {
  stop("Annual FLS/OEWS calibration cells are incomplete.", call. = FALSE)
}
write_csv(
  annual_wage_calibration,
  path_tables("iv_fls_county_wage_calibration.csv")
)

annual_wage_long <- annual_wage_calibration %>%
  select(
    aewr_region_id,
    source_year,
    region_number,
    region_label,
    fls_target_hourly_wage,
    prior_county_oews_hourly_wage,
    calibrated_county_oews_hourly_wage
  ) %>%
  pivot_longer(
    cols = ends_with("hourly_wage"),
    names_to = "series",
    values_to = "hourly_wage"
  ) %>%
  mutate(
    series = recode(
      series,
      fls_target_hourly_wage = "Published FLS target",
      prior_county_oews_hourly_wage = "County-prior OEWS moment",
      calibrated_county_oews_hourly_wage = "Calibrated county OEWS moment"
    )
  )

regional_wage_plot <- ggplot(
  annual_wage_long,
  aes(source_year, hourly_wage, colour = series)
) +
  geom_line(linewidth = 0.45, na.rm = TRUE) +
  geom_point(size = 0.7, na.rm = TRUE) +
  facet_wrap(vars(region_label), ncol = 3, scales = "free_y") +
  scale_colour_manual(
    values = c(
      "Published FLS target" = "#111111",
      "County-prior OEWS moment" = "#0072B2",
      "Calibrated county OEWS moment" = "#D55E00"
    )
  ) +
  labs(
    title = "Annual FLS Wage Targets and County-Calibrated OEWS Moments",
    subtitle = "One county-weight distribution per AEWR region and source year",
    x = "Source year",
    y = "Nominal hourly wage (dollars)",
    colour = NULL
  ) +
  diagnostic_theme
ggsave(
  path_figures("fig_iv_aewr_region_wage_calibration.png"),
  regional_wage_plot,
  width = 12,
  height = 16,
  dpi = 300
)

national_wage_calibration <- annual_wage_long %>%
  group_by(source_year, series) %>%
  summarise(
    equal_region_mean_hourly_wage = finite_average(hourly_wage),
    contributing_regions = sum(is.finite(hourly_wage)),
    .groups = "drop"
  )
write_csv(
  national_wage_calibration,
  path_tables("iv_national_wage_calibration.csv")
)
national_wage_plot <- ggplot(
  national_wage_calibration,
  aes(source_year, equal_region_mean_hourly_wage, colour = series)
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.4) +
  scale_colour_manual(
    values = c(
      "Published FLS target" = "#111111",
      "County-prior OEWS moment" = "#0072B2",
      "Calibrated county OEWS moment" = "#D55E00"
    )
  ) +
  labs(
    title = "National Summary of the Annual Wage Calibration",
    subtitle = "Equal-weight mean across the 17 AEWR regions",
    x = "Source year",
    y = "Nominal hourly wage (dollars)",
    colour = NULL
  ) +
  diagnostic_theme
ggsave(
  path_figures("fig_iv_national_wage_calibration.png"),
  national_wage_plot,
  width = 8,
  height = 5,
  dpi = 300
)

# FLS/QCEW moment residuals -------------------------------------------------

moment_residuals <- primary_moments %>%
  mutate(
    prior_standardized_residual = if_else(
      moment_active & is.finite(prior_scale) & prior_scale > 0,
      prior_raw_residual / prior_scale,
      NA_real_
    )
  ) %>%
  group_by(source_year, moment_family) %>%
  summarise(
    mean_absolute_prior_standardized_residual = finite_average(
      abs(prior_standardized_residual)
    ),
    mean_absolute_calibrated_standardized_residual = finite_average(
      abs(calibrated_standardized_residual)
    ),
    mean_observed_prior_mass = finite_average(observed_prior_mass),
    active_region_moments = sum(moment_active),
    inactive_region_moments = sum(!moment_active),
    .groups = "drop"
  ) %>%
  arrange(source_year, moment_family)
write_csv(
  moment_residuals,
  path_tables("iv_qcew_fls_moment_residuals.csv")
)

moment_residual_plot_data <- moment_residuals %>%
  pivot_longer(
    cols = c(
      mean_absolute_prior_standardized_residual,
      mean_absolute_calibrated_standardized_residual
    ),
    names_to = "weight_stage",
    values_to = "mean_absolute_standardized_residual"
  ) %>%
  mutate(
    weight_stage = recode(
      weight_stage,
      mean_absolute_prior_standardized_residual = "County prior",
      mean_absolute_calibrated_standardized_residual = "Calibrated county weights"
    ),
    moment_family = recode(
      moment_family,
      annual_wage = "Annual FLS/OEWS hourly wage",
      seasonal = "FLS/QCEW seasonal shares",
      composition = "FLS/QCEW field-livestock composition"
    )
  )
moment_residual_plot <- ggplot(
  moment_residual_plot_data,
  aes(source_year, mean_absolute_standardized_residual, colour = weight_stage)
) +
  geom_line(linewidth = 0.75, na.rm = TRUE) +
  geom_point(size = 1.2, na.rm = TRUE) +
  facet_wrap(vars(moment_family), ncol = 1, scales = "free_y") +
  scale_colour_manual(
    values = c(
      "County prior" = "#0072B2",
      "Calibrated county weights" = "#D55E00"
    )
  ) +
  labs(
    title = "Preferred FLS/QCEW Calibration Residuals",
    subtitle = "Absolute standardized residuals; inactive moments remain diagnosed",
    x = "Source year",
    y = "Mean absolute standardized residual",
    colour = NULL
  ) +
  diagnostic_theme
ggsave(
  path_figures("fig_iv_qcew_fls_moment_residuals.png"),
  moment_residual_plot,
  width = 8,
  height = 9,
  dpi = 300
)

# County-weight changes -----------------------------------------------------

county_weight_changes <- primary_weights %>%
  transmute(
    aewr_region_id,
    source_year,
    county_fips,
    frame_prior_weight,
    calibrated_center_weight,
    calibrated_minus_prior_percentage_points = 100 *
      (calibrated_center_weight - frame_prior_weight),
    active_moment_count,
    inactive_moment_count,
    calibrated_effective_county_count,
    maximum_calibrated_county_weight
  ) %>%
  arrange(source_year, aewr_region_id, county_fips)
write_csv(
  county_weight_changes,
  path_tables("iv_county_entropy_weight_changes_pp.csv")
)
county_weight_plot <- ggplot(
  county_weight_changes,
  aes(factor(source_year), calibrated_minus_prior_percentage_points)
) +
  geom_hline(yintercept = 0, colour = "grey55", linewidth = 0.35) +
  geom_boxplot(
    width = 0.65,
    outlier.alpha = 0.08,
    outlier.size = 0.35,
    fill = "#56B4E9",
    colour = "#1F4E79"
  ) +
  labs(
    title = "Preferred Calibration Changes to County Weights",
    subtitle = "Distribution across counties and AEWR regions",
    x = "Source year",
    y = "Calibrated minus prior weight (percentage points)"
  ) +
  diagnostic_theme
ggsave(
  path_figures("fig_iv_county_entropy_weight_changes_pp.png"),
  county_weight_plot,
  width = 9,
  height = 5,
  dpi = 300
)

# OEWS hourly-wage proxy and QCEW feature coverage -------------------------

wage_proxy_coverage <- primary_instruments %>%
  group_by(source_year, policy_year) %>%
  summarise(
    mean_oews_wage_proxy_coverage_weight = finite_average(
      oews_wage_proxy_coverage_weight
    ),
    mean_qcew_employment_coverage_weight = finite_average(
      qcew_employment_coverage_weight
    ),
    available_target_clusters = sum(instrument_available),
    .groups = "drop"
  ) %>%
  arrange(source_year)
write_csv(
  wage_proxy_coverage,
  path_tables("iv_oews_wage_proxy_coverage.csv")
)
wage_proxy_coverage_plot_data <- wage_proxy_coverage %>%
  pivot_longer(
    cols = c(
      mean_oews_wage_proxy_coverage_weight,
      mean_qcew_employment_coverage_weight
    ),
    names_to = "coverage_measure",
    values_to = "mean_calibrated_weight_coverage"
  ) %>%
  mutate(
    coverage_measure = recode(
      coverage_measure,
      mean_oews_wage_proxy_coverage_weight = "OEWS hourly-wage proxy",
      mean_qcew_employment_coverage_weight =
        "QCEW 111+112 employment features"
    )
  )
wage_proxy_coverage_plot <- ggplot(
  wage_proxy_coverage_plot_data,
  aes(
    policy_year,
    mean_calibrated_weight_coverage,
    colour = coverage_measure
  )
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  scale_y_continuous(labels = scales::label_percent(accuracy = 1)) +
  scale_colour_manual(
    values = c(
      "OEWS hourly-wage proxy" = "#0072B2",
      "QCEW 111+112 employment features" = "#009E73"
    )
  ) +
  labs(
    title = "County Feature and OEWS Wage-Proxy Coverage",
    subtitle = paste(
      "Mean nonoverlap calibrated-weight coverage; source year is",
      "policy year minus one"
    ),
    x = "Policy year",
    y = "Mean calibrated-weight coverage",
    colour = NULL
  ) +
  diagnostic_theme
ggsave(
  path_figures("fig_iv_oews_wage_proxy_coverage.png"),
  wage_proxy_coverage_plot,
  width = 9,
  height = 5,
  dpi = 300
)

donor_support <- primary_instruments %>%
  group_by(source_year, policy_year) %>%
  summarise(
    instrument_availability_rate = mean(instrument_available),
    mean_instrument_weight_ess = finite_average(instrument_weight_ess),
    mean_maximum_instrument_county_weight = finite_average(
      maximum_instrument_county_weight
    ),
    mean_eligible_calibrated_weight_mass = finite_average(
      eligible_calibrated_weight_mass
    ),
    mean_qcew_employment_coverage_weight = finite_average(
      qcew_employment_coverage_weight
    ),
    mean_oews_wage_proxy_coverage_weight = finite_average(
      oews_wage_proxy_coverage_weight
    ),
    mean_active_moment_count = finite_average(active_moment_count),
    .groups = "drop"
  ) %>%
  arrange(source_year)
write_csv(
  donor_support,
  path_tables("iv_target_donor_support.csv")
)
donor_support_plot_data <- donor_support %>%
  select(
    policy_year,
    instrument_availability_rate,
    mean_instrument_weight_ess,
    mean_eligible_calibrated_weight_mass,
    mean_qcew_employment_coverage_weight,
    mean_oews_wage_proxy_coverage_weight,
    mean_active_moment_count
  ) %>%
  pivot_longer(
    cols = -policy_year,
    names_to = "diagnostic",
    values_to = "value"
  ) %>%
  mutate(
    diagnostic = recode(
      diagnostic,
      instrument_availability_rate = "Available target-cluster share",
      mean_instrument_weight_ess = "Effective donor counties",
      mean_eligible_calibrated_weight_mass = "Eligible calibrated mass",
      mean_qcew_employment_coverage_weight =
        "QCEW employment-feature coverage",
      mean_oews_wage_proxy_coverage_weight = "OEWS wage-proxy coverage",
      mean_active_moment_count = "Active preferred moments"
    )
  )
donor_support_plot <- ggplot(
  donor_support_plot_data,
  aes(policy_year, value)
) +
  geom_line(colour = "#5E3C99", linewidth = 0.75, na.rm = TRUE) +
  geom_point(colour = "#5E3C99", size = 1.3, na.rm = TRUE) +
  facet_wrap(vars(diagnostic), scales = "free_y", ncol = 2) +
  labs(
    title = "Preferred Instrument Coverage and Donor Support",
    subtitle = "County-weight, feature, wage-proxy, and active-moment diagnostics",
    x = "Policy year",
    y = NULL
  ) +
  diagnostic_theme
ggsave(
  path_figures("fig_iv_target_donor_support.png"),
  donor_support_plot,
  width = 9,
  height = 8,
  dpi = 300
)
