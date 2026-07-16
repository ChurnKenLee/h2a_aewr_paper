# Calibrate OEWS-area weights along a regional FLS wage regularization path
rm(list = ls())
if (!exists("path_code", mode = "function")) {
  source(file.path("code", "paths.R"))
}
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)

gap_closure_values <- c(0, 0.25, 0.50, 0.75, 1)

entropy_weights <- function(lambda, prior_weight, wage) {
  log_weight <- log(prior_weight) + lambda * wage
  log_weight <- log_weight - max(log_weight)
  weight <- exp(log_weight)
  weight / sum(weight)
}

entropy_mean <- function(lambda, prior_weight, wage) {
  weight <- entropy_weights(lambda, prior_weight, wage)
  sum(weight * wage)
}

calibrate_wage_cell <- function(
  data,
  entropy_target_wage,
  tolerance = 1e-10
) {
  prior_weight <- data$oews_area_prior_weight
  wage <- data$oews_area_mean_hourly_wage
  prior_mean <- sum(prior_weight * wage)
  minimum_wage <- min(wage)
  maximum_wage <- max(wage)

  data$oews_prior_weighted_wage <- prior_mean
  data$oews_minimum_wage <- minimum_wage
  data$oews_maximum_wage <- maximum_wage
  data$entropy_lambda <- NA_real_
  data$oews_area_weight_wage_calibrated <- NA_real_
  data$calibration_status <- "outside_support"

  if (
    entropy_target_wage < minimum_wage - tolerance ||
      entropy_target_wage > maximum_wage + tolerance
  ) {
    return(data)
  }

  if (abs(entropy_target_wage - prior_mean) <= tolerance) {
    data$entropy_lambda <- 0
    data$oews_area_weight_wage_calibrated <- prior_weight
    data$calibration_status <- "calibrated"
    return(data)
  }

  if (abs(maximum_wage - minimum_wage) <= tolerance) {
    data$calibration_status <- "no_wage_variation"
    return(data)
  }

  if (abs(entropy_target_wage - minimum_wage) <= tolerance) {
    keep <- abs(wage - minimum_wage) <= tolerance
    weight <- if_else(keep, prior_weight, 0)
    data$entropy_lambda <- -Inf
    data$oews_area_weight_wage_calibrated <- weight / sum(weight)
    data$calibration_status <- "calibrated_boundary"
    return(data)
  }

  if (abs(entropy_target_wage - maximum_wage) <= tolerance) {
    keep <- abs(wage - maximum_wage) <= tolerance
    weight <- if_else(keep, prior_weight, 0)
    data$entropy_lambda <- Inf
    data$oews_area_weight_wage_calibrated <- weight / sum(weight)
    data$calibration_status <- "calibrated_boundary"
    return(data)
  }

  lower <- -1
  while (entropy_mean(lower, prior_weight, wage) > entropy_target_wage) {
    lower <- lower * 2
  }

  upper <- 1
  while (entropy_mean(upper, prior_weight, wage) < entropy_target_wage) {
    upper <- upper * 2
  }

  lambda <- uniroot(
    function(value) {
      entropy_mean(value, prior_weight, wage) - entropy_target_wage
    },
    interval = c(lower, upper),
    tol = tolerance
  )$root

  data$entropy_lambda <- lambda
  data$oews_area_weight_wage_calibrated <- entropy_weights(
    lambda,
    prior_weight,
    wage
  )
  data$calibration_status <- "calibrated"
  data
}

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
    maximum_weight_adjustment = if_else(
      all(is.na(oews_area_weight_adjustment)),
      NA_real_,
      max(oews_area_weight_adjustment, na.rm = TRUE)
    ),
    maximum_calibrated_area_weight = if_else(
      all(is.na(oews_area_weight_wage_calibrated)),
      NA_real_,
      max(oews_area_weight_wage_calibrated, na.rm = TRUE)
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
