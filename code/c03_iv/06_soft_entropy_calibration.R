# Purpose: Entropy-balance duration and seasonality moments, optionally with a wage target.
# Inputs: FLS targets, OEWS priors, and auxiliary moments.
# Outputs: area/county soft-calibrated weights and diagnostics.
# Run after: 03_oews_area_prior_weights.R and 04_auxiliary_moments.R.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)

gap_closure_values <- c(0, 0.25, 0.50, 0.75, 1)

# rho is the coefficient on one-half of the squared standardized imbalance.
# Larger values impose the public analogue more strongly. Retain a path rather
# than selecting rho using the downstream AEWR first stage.
soft_penalty_values <- c(1, 4, 16)

wage_moment_specs <- bind_rows(
  tibble(
    moment_spec = "wage_only",
    soft_penalty = 0
  ),
  crossing(
    moment_spec = c(
      "wage_duration",
      "wage_duration_seasonal"
    ),
    soft_penalty = soft_penalty_values
  )
)

# Wage calibration is optional. Wage-targeted specifications retain the full
# gap-closure path. Wage-free specifications use only the auxiliary FLS
# moments, so gap_closure and entropy_target_wage are intentionally missing.
calibration_specs <- bind_rows(
  wage_moment_specs %>%
    crossing(gap_closure = gap_closure_values) %>%
    mutate(include_wage_target = TRUE),
  crossing(
    moment_spec = c("duration", "duration_seasonal"),
    soft_penalty = soft_penalty_values
  ) %>%
    mutate(
      gap_closure = NA_real_,
      include_wage_target = FALSE
    )
)

duration_feature <- "census_hired_worker_150_plus_share"
seasonal_features <- c(
  "qcew_area_reference_month_employment_share_january",
  "qcew_area_reference_month_employment_share_april",
  "qcew_area_reference_month_employment_share_july"
)
duration_target <- "fls_hired_worker_150_plus_share"
seasonal_targets <- c(
  "fls_hired_worker_share_january",
  "fls_hired_worker_share_april",
  "fls_hired_worker_share_july"
)

feature_output_names <- c(
  setNames("calibration_feature_duration", duration_feature),
  setNames(
    c(
      "calibration_feature_seasonal_january",
      "calibration_feature_seasonal_april",
      "calibration_feature_seasonal_july"
    ),
    seasonal_features
  )
)

lambda_output_names <- c(
  setNames("entropy_lambda_duration", duration_feature),
  setNames(
    c(
      "entropy_lambda_seasonal_january",
      "entropy_lambda_seasonal_april",
      "entropy_lambda_seasonal_july"
    ),
    seasonal_features
  )
)

center_output_names <- c(
  setNames("duration_feature_prior_mean", duration_feature),
  setNames(
    c(
      "seasonal_january_feature_prior_mean",
      "seasonal_april_feature_prior_mean",
      "seasonal_july_feature_prior_mean"
    ),
    seasonal_features
  )
)

scale_output_names <- c(
  setNames("duration_feature_scale", duration_feature),
  setNames(
    c(
      "seasonal_january_feature_scale",
      "seasonal_april_feature_scale",
      "seasonal_july_feature_scale"
    ),
    seasonal_features
  )
)

observed_mass_output_names <- c(
  setNames("duration_feature_observed_prior_mass", duration_feature),
  setNames(
    c(
      "seasonal_january_feature_observed_prior_mass",
      "seasonal_april_feature_observed_prior_mass",
      "seasonal_july_feature_observed_prior_mass"
    ),
    seasonal_features
  )
)

source(path_code("c00_shared", "entropy_calibration.R"))

fls_target <- read_parquet(path_int("fls_region.parquet")) %>%
  transmute(
    aewr_region_num = as.integer(aewr_region_num),
    year = as.integer(preliminary_year),
    fls_target_wage = as.numeric(field_livestock_preliminary),
    fls_hired_worker_150_plus_share = as.numeric(
      fls_hired_worker_150_plus_share
    ),
    fls_hired_worker_share_january = as.numeric(
      fls_hired_worker_share_january
    ),
    fls_hired_worker_share_april = as.numeric(
      fls_hired_worker_share_april
    ),
    fls_hired_worker_share_july = as.numeric(
      fls_hired_worker_share_july
    )
  ) %>%
  filter(
    !is.na(aewr_region_num),
    !is.na(year)
  ) %>%
  distinct(aewr_region_num, year, .keep_all = TRUE)

fls_oews_area_prior <- read_parquet(path_int(
  "fls_oews_area_auxiliary_moments.parquet"
)) %>%
  mutate(
    aewr_region_num = as.integer(aewr_region_num),
    year = as.integer(year)
  ) %>%
  filter(
    oews_wage_observed,
    !is.na(oews_area_prior_weight),
    oews_area_prior_weight > 0
  ) %>%
  inner_join(
    fls_target,
    by = c("aewr_region_num", "year")
  ) %>%
  group_by(aewr_region_num, year) %>%
  mutate(
    oews_prior_weighted_wage = sum(
      oews_area_prior_weight * oews_area_mean_hourly_wage
    )
  ) %>%
  ungroup()

fls_oews_area_regularization_path <- fls_oews_area_prior %>%
  crossing(calibration_specs) %>%
  filter(
    !include_wage_target |
      (!is.na(fls_target_wage) & fls_target_wage > 0)
  ) %>%
  mutate(
    entropy_target_wage = if_else(
      include_wage_target,
      oews_prior_weighted_wage +
        gap_closure * (fls_target_wage - oews_prior_weighted_wage),
      NA_real_
    )
  )

fls_oews_area_soft_calibrated <- fls_oews_area_regularization_path %>%
  group_by(
    aewr_region_num,
    year,
    include_wage_target,
    gap_closure,
    moment_spec,
    soft_penalty
  ) %>%
  group_modify(
    ~ calibrate_soft_cell(
      .x,
      entropy_target_wage = first(.x$entropy_target_wage),
      include_wage_target = .y$include_wage_target[[1]],
      moment_spec = .y$moment_spec[[1]],
      soft_penalty = .y$soft_penalty[[1]]
    )
  ) %>%
  ungroup() %>%
  mutate(
    oews_area_weight_adjustment = oews_area_weight_soft_calibrated /
      oews_area_prior_weight,
    entropy_kl_divergence_component = if_else(
      !is.na(oews_area_weight_soft_calibrated) &
        oews_area_weight_soft_calibrated > 0,
      oews_area_weight_soft_calibrated *
        log(oews_area_weight_soft_calibrated / oews_area_prior_weight),
      0
    )
  )

fls_soft_entropy_diagnostics <- fls_oews_area_soft_calibrated %>%
  group_by(
    aewr_region_num,
    year,
    include_wage_target,
    gap_closure,
    moment_spec,
    soft_penalty
  ) %>%
  summarise(
    calibration_status = first(calibration_status),
    wage_target_active = first(include_wage_target),
    optimizer_convergence = first(optimizer_convergence),
    soft_moment_count = first(soft_moment_count),
    fls_target_wage = first(fls_target_wage),
    entropy_target_wage = first(entropy_target_wage),
    oews_prior_weighted_wage = first(oews_prior_weighted_wage),
    oews_calibrated_weighted_wage = weighted_sum_if_observed(
      oews_area_weight_soft_calibrated,
      oews_area_mean_hourly_wage
    ),
    fls_duration_target = first(fls_hired_worker_150_plus_share),
    prior_duration_moment = first(duration_feature_prior_mean),
    calibrated_duration_moment = weighted_sum_if_observed(
      oews_area_weight_soft_calibrated,
      calibration_feature_duration
    ),
    duration_standardized_imbalance = (calibrated_duration_moment -
      fls_duration_target) /
      first(duration_feature_scale),
    fls_seasonal_january_target = first(fls_hired_worker_share_january),
    prior_seasonal_january_moment = first(
      seasonal_january_feature_prior_mean
    ),
    calibrated_seasonal_january_moment = weighted_sum_if_observed(
      oews_area_weight_soft_calibrated,
      calibration_feature_seasonal_january
    ),
    seasonal_january_standardized_imbalance = (calibrated_seasonal_january_moment -
      fls_seasonal_january_target) /
      first(seasonal_january_feature_scale),
    fls_seasonal_april_target = first(fls_hired_worker_share_april),
    prior_seasonal_april_moment = first(
      seasonal_april_feature_prior_mean
    ),
    calibrated_seasonal_april_moment = weighted_sum_if_observed(
      oews_area_weight_soft_calibrated,
      calibration_feature_seasonal_april
    ),
    seasonal_april_standardized_imbalance = (calibrated_seasonal_april_moment -
      fls_seasonal_april_target) /
      first(seasonal_april_feature_scale),
    fls_seasonal_july_target = first(fls_hired_worker_share_july),
    prior_seasonal_july_moment = first(
      seasonal_july_feature_prior_mean
    ),
    calibrated_seasonal_july_moment = weighted_sum_if_observed(
      oews_area_weight_soft_calibrated,
      calibration_feature_seasonal_july
    ),
    seasonal_july_standardized_imbalance = (calibrated_seasonal_july_moment -
      fls_seasonal_july_target) /
      first(seasonal_july_feature_scale),
    duration_feature_observed_prior_mass = first(
      duration_feature_observed_prior_mass
    ),
    seasonal_january_feature_observed_prior_mass = first(
      seasonal_january_feature_observed_prior_mass
    ),
    seasonal_april_feature_observed_prior_mass = first(
      seasonal_april_feature_observed_prior_mass
    ),
    seasonal_july_feature_observed_prior_mass = first(
      seasonal_july_feature_observed_prior_mass
    ),
    entropy_lambda_wage = first(entropy_lambda_wage),
    entropy_lambda_duration = first(entropy_lambda_duration),
    entropy_lambda_seasonal_january = first(
      entropy_lambda_seasonal_january
    ),
    entropy_lambda_seasonal_april = first(entropy_lambda_seasonal_april),
    entropy_lambda_seasonal_july = first(entropy_lambda_seasonal_july),
    entropy_kl_divergence = if_else(
      all(is.na(oews_area_weight_soft_calibrated)),
      NA_real_,
      sum(entropy_kl_divergence_component)
    ),
    prior_effective_area_count = 1 / sum(oews_area_prior_weight^2),
    calibrated_effective_area_count = if_else(
      all(is.na(oews_area_weight_soft_calibrated)),
      NA_real_,
      1 / sum(oews_area_weight_soft_calibrated^2)
    ),
    effective_area_count_ratio = calibrated_effective_area_count /
      prior_effective_area_count,
    maximum_weight_adjustment = max_if_observed(
      oews_area_weight_adjustment
    ),
    maximum_calibrated_area_weight = max_if_observed(
      oews_area_weight_soft_calibrated
    ),
    .groups = "drop"
  )

fls_county_area_prior <- read_parquet(path_int(
  "fls_county_oews_area_prior_weight.parquet"
)) %>%
  mutate(
    aewr_region_num = as.integer(aewr_region_num),
    year = as.integer(year)
  )

fls_county_weight_soft_calibrated <- fls_county_area_prior %>%
  inner_join(
    fls_oews_area_soft_calibrated %>%
      select(
        aewr_region_num,
        year,
        include_wage_target,
        oews_area_code,
        gap_closure,
        moment_spec,
        soft_penalty,
        oews_area_prior_weight_all,
        oews_area_weight_soft_calibrated,
        calibration_status
      ),
    by = c("aewr_region_num", "year", "oews_area_code")
  ) %>%
  mutate(
    county_share_within_oews_area = if_else(
      oews_area_prior_weight_all > 0,
      county_area_prior_weight / oews_area_prior_weight_all,
      NA_real_
    ),
    county_area_weight_soft_calibrated = oews_area_weight_soft_calibrated *
      county_share_within_oews_area
  ) %>%
  group_by(
    countyfips,
    year,
    statefips,
    state_abbrev,
    aewr_region_num,
    cz_out10,
    cz_aewr_region_fe,
    fls_county_weight_prior,
    include_wage_target,
    gap_closure,
    moment_spec,
    soft_penalty,
    calibration_status
  ) %>%
  summarise(
    fls_county_weight_soft_calibrated = sum(
      county_area_weight_soft_calibrated,
      na.rm = FALSE
    ),
    .groups = "drop"
  ) %>%
  arrange(countyfips, year) %>%
  group_by(
    countyfips,
    include_wage_target,
    gap_closure,
    moment_spec,
    soft_penalty
  ) %>%
  mutate(
    fls_county_weight_soft_calibrated_l1 = if_else(
      lag(year) == year - 1L,
      lag(fls_county_weight_soft_calibrated),
      NA_real_
    )
  ) %>%
  ungroup()

write_parquet(
  fls_oews_area_soft_calibrated,
  path_int("fls_oews_area_weight_soft_calibrated.parquet")
)

write_parquet(
  fls_county_weight_soft_calibrated,
  path_int("fls_county_weight_soft_calibrated.parquet")
)

write_parquet(
  fls_soft_entropy_diagnostics,
  path_int("fls_soft_entropy_diagnostics.parquet")
)
