# Scratch-only helpers for the 2025-27 policy-episode pre-checks.
# Reuses explorations/identification/id_helpers.R (scratch guard, FY2025 read
# guard, CSV writer). Nothing here reads an FY2026 outcome or writes a
# retained artifact.
source("explorations/identification/id_helpers.R")

EP_FIRST_FY <- 2012L          # Track 1 window start
EP_LAST_PRE_FY <- ID_LAST_PRE_FY
EP_PLACEBO_FY <- 2018:2025    # placebo "post" years; each is contrasted with the year before
EP_EVENT1 <- as.Date("2025-10-02")
EP_EVENT2 <- as.Date("2026-08-03")

# Second argument: directory holding the outputs of
# explorations/identification_search/shared/01_outcomes.R.
ep_shared_dir <- function() {
  arg <- commandArgs(trailingOnly = TRUE)[2]
  if (is.na(arg) || !file.exists(file.path(arg, "outcomes_county_fy.parquet")))
    stop("Supply the shared-outcomes directory (output of identification_search/shared/01_outcomes.R) as argument 2")
  normalizePath(arg)
}

# One row per state: former AEWR region, the FY2025 regional rate, the
# statewide skill-I rates of both schedules, and the doses.
ep_state_frame <- function() {
  region <- arrow::open_dataset(path_processed("county_year_panel.parquet")) %>%
    filter(year == 2024L) %>% select(state_fips, aewr_region_id) %>% collect() %>% distinct() %>%
    mutate(aewr_region_id = as.character(aewr_region_id))
  stopifnot(!anyDuplicated(region$state_fips), is.character(region$state_fips))
  rate25 <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>%
    filter(effective_start <= as.Date("2025-04-01"), effective_end_exclusive > as.Date("2025-04-01")) %>%
    group_by(state_fips) %>% summarise(rate25 = unique(aewr), .groups = "drop")
  sw <- read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
    filter(soc_code == "FIELD_LIVESTOCK_COMBINED", skill_level == 1L, effective_start %in% c(EP_EVENT1, EP_EVENT2)) %>%
    transmute(state_fips, key = paste0(tolower(worker_category), ifelse(effective_start == EP_EVENT1, "_e1", "_e2")),
      rate = as.numeric(hourly_aewr)) %>% distinct()
  stopifnot(!anyDuplicated(sw[c("state_fips", "key")]))
  sw <- tidyr::pivot_wider(sw, names_from = key, values_from = rate)
  f <- region %>% inner_join(rate25, by = "state_fips") %>% inner_join(sw, by = "state_fips") %>%
    group_by(aewr_region_id) %>% mutate(region_states = n()) %>% ungroup() %>%
    mutate(multi_state_region = region_states > 1,
      dose1_us = us_e1 - rate25, dose1_h2a = h2a_e1 - rate25,
      dose2_us = us_e2 - us_e1, dose2_h2a = h2a_e2 - h2a_e1)
  stopifnot(!anyDuplicated(f$state_fips), !anyNA(f$dose1_us))
  f
}

# Weighted residual of x on a set of controls (data frame columns), with an
# optional factor absorbed. Used for the "dose left after controls" support SD.
# The fit can drop rows (for example singleton groups); ep_resid_fit keeps the
# kept rows' weights next to the residuals.
ep_resid_fit <- function(d, x, controls = character(), fe = NULL, w = "w") {
  rhs <- if (length(controls)) paste(controls, collapse = " + ") else "1"
  f <- as.formula(paste(x, "~", rhs, if (!is.null(fe)) paste("|", fe) else ""))
  m <- feols(f, d, weights = d[[w]])
  list(r = resid(m), w = d[[w]][obs(m)])
}
ep_resid <- function(d, x, controls = character(), fe = NULL, w = "w") {
  out <- ep_resid_fit(d, x, controls, fe, w)
  stopifnot(length(out$r) == nrow(d))
  out$r
}

# One placebo regression: outcome growth on the dose plus controls, weighted,
# heteroskedasticity-robust. Returns estimate, SE and the weighted SD of the
# dose left after the controls.
ep_fit <- function(d, y, dose, controls = character(), fe = NULL, w = "w") {
  rhs <- paste(c(dose, controls), collapse = " + ")
  f <- as.formula(paste(y, "~", rhs, if (!is.null(fe)) paste("|", fe) else ""))
  m <- feols(f, d, weights = d[[w]], vcov = "hetero")
  r <- ep_resid_fit(d, dose, controls, fe, w)
  tibble(estimate = unname(coef(m)[dose]), se = unname(se(m)[dose]), states = nobs(m),
    dose_sd_left = sqrt(sum(r$w * r$r^2) / sum(r$w)))
}

# Summary over placebo years. RMSE around zero is the design error: the true
# placebo effect is zero, so bias and noise both count.
ep_summarise <- function(res, mean_rate) {
  res %>% group_by(across(any_of(c("outcome", "weighting", "sample", "estimator")))) %>%
    summarise(placebo_years = n(), states = round(mean(states)), dose_sd_left = mean(dose_sd_left),
      mean_estimate = mean(estimate), sd_estimate = sd(estimate), rmse = sqrt(mean(estimate^2)),
      mean_se = mean(se), share_abs_t_over_2 = mean(abs(estimate / se) > 2), .groups = "drop") %>%
    mutate(mean_rate = mean_rate,
      bias_elasticity = mean_estimate * mean_rate,
      mde_elasticity_rmse = 2.8 * rmse * mean_rate,
      mde_elasticity_se = 2.8 * mean_se * mean_rate)
}
