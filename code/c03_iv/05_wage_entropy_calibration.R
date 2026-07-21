# Purpose: Calibrate OEWS-area priors to the full regional FLS wage target.
# Inputs: FLS regional targets and OEWS-area prior weights.
# Outputs: area/county calibrated weights and wage-calibration diagnostics.
# Run after: 03_oews_area_prior_weights.R.

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

gap_closure_values <- 1

source(path_code("c00_shared", "entropy_calibration.R"))

fls_target <- read_parquet(path_int("fls_region.parquet")) %>%
  transmute(
    aewr_region_num = as.integer(aewr_region_num),
    year = as.integer(preliminary_year),
    fls_target_wage = as.numeric(field_livestock_preliminary)
  ) %>%
  filter(
    !is.na(aewr_region_num),
    !is.na(year),
    !is.na(fls_target_wage),
    fls_target_wage > 0
  ) %>%
  distinct(aewr_region_num, year, .keep_all = TRUE)

# Unjoined rows are just years outside of our target years
fls_oews_area_prior <- read_parquet(path_int(
  "fls_oews_area_prior_weight.parquet"
)) %>%
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

fls_oews_area_regularization_path <- tidyr::crossing(
  fls_oews_area_prior,
  gap_closure = gap_closure_values
) %>%
  mutate(
    entropy_target_wage = oews_prior_weighted_wage +
      gap_closure * (fls_target_wage - oews_prior_weighted_wage)
  )

fls_oews_area_calibrated <- fls_oews_area_regularization_path %>%
  group_by(aewr_region_num, year, gap_closure) %>%
  group_modify(
    ~ calibrate_wage_cell(
      .x,
      entropy_target_wage = first(.x$entropy_target_wage)
    )
  ) %>%
  ungroup() %>%
  mutate(
    oews_area_weight_adjustment = oews_area_weight_wage_calibrated /
      oews_area_prior_weight,
    entropy_kl_divergence_component = if_else(
      !is.na(oews_area_weight_wage_calibrated) &
        oews_area_weight_wage_calibrated > 0,
      oews_area_weight_wage_calibrated *
        log(
          oews_area_weight_wage_calibrated / oews_area_prior_weight
        ),
      0
    )
  )

fls_entropy_diagnostics <- fls_oews_area_calibrated %>%
  group_by(aewr_region_num, year, gap_closure) %>%
  summarise(
    calibration_status = first(calibration_status),
    fls_target_wage = first(fls_target_wage),
    entropy_target_wage = first(entropy_target_wage),
    oews_prior_weighted_wage = first(oews_prior_weighted_wage),
    oews_calibrated_weighted_wage = sum(
      oews_area_weight_wage_calibrated * oews_area_mean_hourly_wage
    ),
    oews_sampling_composition_shock = oews_calibrated_weighted_wage -
      oews_prior_weighted_wage,
    oews_fls_wage_gap_remaining = fls_target_wage -
      oews_calibrated_weighted_wage,
    achieved_gap_closure = if_else(
      abs(fls_target_wage - oews_prior_weighted_wage) > 1e-10,
      oews_sampling_composition_shock /
        (fls_target_wage - oews_prior_weighted_wage),
      1
    ),
    entropy_lambda = first(entropy_lambda),
    entropy_kl_divergence = if_else(
      all(is.na(oews_area_weight_wage_calibrated)),
      NA_real_,
      sum(entropy_kl_divergence_component)
    ),
    oews_minimum_wage = first(oews_minimum_wage),
    oews_maximum_wage = first(oews_maximum_wage),
    oews_area_count = n(),
    oews_distinct_wage_count = n_distinct(oews_area_mean_hourly_wage),
    oews_observed_prior_mass = first(oews_observed_prior_mass),
    prior_effective_area_count = 1 / sum(oews_area_prior_weight^2),
    calibrated_effective_area_count = if_else(
      all(is.na(oews_area_weight_wage_calibrated)),
      NA_real_,
      1 / sum(oews_area_weight_wage_calibrated^2)
    ),
    effective_area_count_ratio = calibrated_effective_area_count /
      prior_effective_area_count,
    maximum_weight_adjustment = max_if_observed(
      oews_area_weight_adjustment
    ),
    maximum_calibrated_area_weight = max_if_observed(
      oews_area_weight_wage_calibrated
    ),
    .groups = "drop"
  )

fls_county_area_prior <- read_parquet(path_int(
  "fls_county_oews_area_prior_weight.parquet"
))

fls_county_weight_wage_calibrated <- fls_county_area_prior %>%
  inner_join(
    fls_oews_area_calibrated %>%
      select(
        aewr_region_num,
        year,
        oews_area_code,
        gap_closure,
        oews_area_prior_weight_all,
        oews_area_weight_wage_calibrated,
        calibration_status
      ),
    by = c(
      "aewr_region_num",
      "year",
      "oews_area_code"
    )
  ) %>%
  mutate(
    county_share_within_oews_area = if_else(
      oews_area_prior_weight_all > 0,
      county_area_prior_weight / oews_area_prior_weight_all,
      NA_real_
    ),
    county_area_weight_wage_calibrated = oews_area_weight_wage_calibrated *
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
    gap_closure,
    calibration_status
  ) %>%
  summarise(
    fls_county_weight_wage_calibrated = sum(
      county_area_weight_wage_calibrated,
      na.rm = FALSE
    ),
    .groups = "drop"
  ) %>%
  arrange(countyfips, year) %>%
  group_by(countyfips, gap_closure) %>%
  mutate(
    fls_county_weight_wage_calibrated_l1 = if_else(
      lag(year) == year - 1L,
      lag(fls_county_weight_wage_calibrated),
      NA_real_
    )
  ) %>%
  ungroup()

write_parquet(
  fls_oews_area_calibrated,
  path_int("fls_oews_area_weight_wage_calibrated.parquet")
)

write_parquet(
  fls_county_weight_wage_calibrated,
  path_int("fls_county_weight_wage_calibrated.parquet")
)

write_parquet(
  fls_entropy_diagnostics,
  path_int("fls_wage_entropy_diagnostics.parquet")
)
