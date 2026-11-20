# Pre-period check A: can the between-region part of the 2025 cut be used?
#
# The 2025-rule design ("Track 1") keeps only within-former-region dispersion of
# the cut, because cross-state shocks are large. This script asks, on
# FY2012-FY2025 outcomes only, whether modelling those shocks (lagged growth,
# state trends, factor loadings) makes the full cross-state dose usable.
#
# Method: for each placebo year tau in 2018..2025, regress the state's
# one-year log growth in certified hours (or positions), tau-1 to tau, on the
# ACTUAL 2025 dose. Nothing happened to the dose in those years, so the true
# coefficient is zero and the root mean squared placebo estimate is the design
# error (trend bias plus shock noise). Elasticity MDE = 2.8 x RMSE x mean rate.
#
# With county FE and a state-level regressor, two-period PPML on county counts
# depends on the data only through state totals, so a state panel loses nothing.
#
# Args: 1 = scratch output dir (outside the repo); 2 = shared-outcomes dir.
# Reads no FY2026 outcome.
source("explorations/episode_2025/ep_helpers.R")
out_dir <- id_output_dir()
shared <- ep_shared_dir()

frame <- ep_state_frame()
cty <- arrow::open_dataset(file.path(shared, "outcomes_county_fy.parquet")) %>%
  filter(year >= EP_FIRST_FY, year <= EP_LAST_PRE_FY) %>%
  select(county_fips, state_fips, year, cert_hours_dq13, cert_pos_dq13) %>% collect()
id_assert_pre(cty$year)
stopifnot(!anyDuplicated(cty[c("county_fips", "year")]), is.character(cty$state_fips))

st <- cty %>% group_by(state_fips, year) %>%
  summarise(hours = sum(cert_hours_dq13, na.rm = TRUE), positions = sum(cert_pos_dq13, na.rm = TRUE), .groups = "drop") %>%
  inner_join(frame, by = "state_fips")
stopifnot(!anyDuplicated(st[c("state_fips", "year")]))
# Log growth needs positive totals in every year; states with a zero year are dropped and counted.
keep <- st %>% group_by(state_fips) %>% summarise(ok = n() == EP_LAST_PRE_FY - EP_FIRST_FY + 1L && all(hours > 0 & positions > 0))
dropped <- keep$state_fips[!keep$ok]
st <- st %>% filter(state_fips %in% keep$state_fips[keep$ok]) %>% arrange(state_fips, year) %>% group_by(state_fips) %>%
  mutate(g_hours = log(hours) - log(lag(hours)), g_positions = log(positions) - log(lag(positions)),
    w_hours = lag(hours), w_positions = lag(positions)) %>% ungroup()

support <- frame %>% filter(state_fips %in% st$state_fips) %>%
  left_join(st %>% filter(year == EP_LAST_PRE_FY) %>% select(state_fips, hours25 = hours), by = "state_fips") %>%
  mutate(wr = ave(hours25, aewr_region_id, FUN = sum),
    dose_within = dose1_us - ave(dose1_us * hours25, aewr_region_id, FUN = sum) / wr)
wsd <- function(x, w) sqrt(sum(w * (x - sum(w * x) / sum(w))^2) / sum(w))
support_row <- function(d, label) tibble(sample = label, states = nrow(d), hours25_m = sum(d$hours25) / 1e6,
  mean_dose = mean(d$dose1_us), sd_dose = sd(d$dose1_us), hours_weighted_mean_dose = sum(d$hours25 * d$dose1_us) / sum(d$hours25),
  hours_weighted_sd_dose = wsd(d$dose1_us, d$hours25),
  hours_weighted_sd_dose_within_region = sqrt(sum(d$hours25 * d$dose_within^2) / sum(d$hours25)),
  mean_rate25_hours_weighted = sum(d$hours25 * d$rate25) / sum(d$hours25))
support_tab <- bind_rows(support_row(support %>% filter(multi_state_region), "multi_state_regions"),
  support_row(support, "all_states"))
id_write(support_tab, out_dir, "full_cut_support.csv")
id_write(tibble(dropped_state_fips = dropped), out_dir, "full_cut_dropped_states.csv")

# Pre-tau objects for one outcome: state trend (mean past growth), two lags,
# and factor loadings from the year-demeaned past growth matrix.
pre_objects <- function(d, g, tau, r_max = 2L) {
  past <- d %>% filter(year < tau, year > EP_FIRST_FY) %>% select(state_fips, year, g = all_of(g))
  wide <- tidyr::pivot_wider(past, names_from = year, values_from = g) %>% arrange(state_fips)
  M <- as.matrix(wide[, -1]); M <- sweep(M, 2, colMeans(M))
  sv <- svd(M)
  load <- sweep(sv$u[, seq_len(r_max), drop = FALSE], 2, sv$d[seq_len(r_max)], `*`)
  tibble(state_fips = wide$state_fips, trend = rowMeans(as.matrix(wide[, -1])),
    lag1 = wide[[as.character(tau - 1L)]], lag2 = wide[[as.character(tau - 2L)]],
    f1 = load[, 1], f2 = load[, 2],
    factor_share_1 = sv$d[1]^2 / sum(sv$d^2), factor_share_2 = sum(sv$d[1:2]^2) / sum(sv$d^2))
}

one_year <- function(d, outcome, tau, weighting) {
  g <- paste0("g_", outcome)
  x <- d %>% filter(year == tau) %>% inner_join(pre_objects(d, g, tau), by = "state_fips") %>%
    mutate(y = .data[[g]], y_detrended = y - trend, w = if (weighting == "size") .data[[paste0("w_", outcome)]] else 1)
  ms <- x %>% filter(multi_state_region)
  bind_rows(
    ep_fit(ms, "y", "dose1_us", fe = "aewr_region_id") %>% mutate(sample = "multi_state_regions", estimator = "within region (current design)"),
    ep_fit(ms, "y_detrended", "dose1_us", fe = "aewr_region_id") %>% mutate(sample = "multi_state_regions", estimator = "within region, state trend removed"),
    ep_fit(ms, "y", "dose1_us") %>% mutate(sample = "multi_state_regions", estimator = "full dose, no adjustment"),
    ep_fit(x, "y", "dose1_us") %>% mutate(sample = "all_states", estimator = "full dose, no adjustment"),
    ep_fit(x, "y_detrended", "dose1_us") %>% mutate(sample = "all_states", estimator = "full dose, state trend removed"),
    ep_fit(x, "y", "dose1_us", c("lag1", "lag2")) %>% mutate(sample = "all_states", estimator = "full dose, two lags of growth"),
    ep_fit(x, "y", "dose1_us", "f1") %>% mutate(sample = "all_states", estimator = "full dose, 1 factor loading"),
    ep_fit(x, "y", "dose1_us", c("f1", "f2")) %>% mutate(sample = "all_states", estimator = "full dose, 2 factor loadings"),
    ep_fit(x, "y_detrended", "dose1_us", c("f1", "f2")) %>% mutate(sample = "all_states", estimator = "full dose, trend removed + 2 loadings")
  ) %>% mutate(outcome = outcome, weighting = weighting, tau = tau,
    factor_share_1 = x$factor_share_1[1], factor_share_2 = x$factor_share_2[1])
}

res <- bind_rows(lapply(c("hours", "positions"), function(o) bind_rows(lapply(c("size", "equal"), function(wt)
  bind_rows(lapply(EP_PLACEBO_FY, function(tau) one_year(st, o, tau, wt)))))))
id_write(res, out_dir, "full_cut_placebo_by_year.csv")
mean_rate <- support_tab$mean_rate25_hours_weighted[support_tab$sample == "all_states"]
summ <- ep_summarise(res, mean_rate) %>% arrange(outcome, weighting, sample, mde_elasticity_rmse)
id_write(summ, out_dir, "full_cut_placebo_summary.csv")

# Validation against the reported Track 1 placebo (FY2024 vs FY2025, PPML): two-period state PPML.
ppml <- bind_rows(lapply(c("hours", "positions"), function(o) {
  d <- st %>% filter(year %in% 2024:2025, multi_state_region) %>% mutate(x = ifelse(year == 2025L, dose1_us, 0), rfy = paste(aewr_region_id, year))
  m <- fepois(as.formula(paste(o, "~ x | state_fips + rfy")), d, vcov = ~state_fips)
  tibble(outcome = o, ppml_semi_elasticity = unname(coef(m)["x"]), elasticity = unname(coef(m)["x"]) * mean_rate, states = n_distinct(d$state_fips))
}))
id_write(ppml, out_dir, "full_cut_ppml_validation_fy2024_fy2025.csv")

options(width = 220)
cat("states dropped for a zero year:", length(dropped), "\n\nSUPPORT\n"); print(as.data.frame(support_tab), digits = 4)
cat("\nshare of past growth variance on the first one / two factors (mean over placebo years):",
  round(mean(res$factor_share_1), 3), round(mean(res$factor_share_2), 3), "\n")
cat("\nPLACEBO SUMMARY (elasticity units; true effect is zero)\n")
print(as.data.frame(summ %>% select(outcome, weighting, sample, estimator, states, dose_sd_left, bias_elasticity,
  mde_elasticity_rmse, mde_elasticity_se, share_abs_t_over_2)), digits = 3)
cat("\nTrack 1 placebo reproduction (reported elasticities: hours -0.08, positions -0.11)\n"); print(as.data.frame(ppml), digits = 3)
