# Track 1 pre-period diagnostics declared in track1_registry.json (run after
# the registry was frozen; pre-FY2026 data only). Reads track1_pre_panel.csv
# and track1_treatment_schedule.csv written by 04_track1_power.R.
#   - event study: dose x FY, reference FY2025, FY2012-FY2025, county +
#     former-region x FY FE, state CRV1; joint pre-trend Wald; linear trend;
#   - warning check: dose vs FY2024-to-FY2025 outcome growth within former
#     region, compared with the same slope for every earlier adjacent FY pair.
source("explorations/identification/id_helpers.R")
out_dir <- id_output_dir()
pp <- read_csv(file.path(out_dir, "track1_pre_panel.csv"), show_col_types = FALSE,
  col_types = cols(county_fips = "c", state_fips = "c", aewr_region_id = "c"))
id_assert_pre(pp$year)
bite_check_keys(pp)
tr <- read_csv(file.path(out_dir, "track1_treatment_schedule.csv"), show_col_types = FALSE, col_types = cols(state_fips = "c"))
dose <- tr %>% filter(year %in% c(2025, 2026)) %>% select(state_fips, year, label, aewr) %>%
  tidyr::pivot_wider(names_from = year, values_from = aewr, names_prefix = "r") %>%
  mutate(dose = r2026 - r2025) %>% select(state_fips, label, dose) %>%
  tidyr::pivot_wider(names_from = label, values_from = dose, names_prefix = "dose_")
pp <- pp %>% left_join(dose, by = "state_fips", relationship = "many-to-one") %>%
  mutate(t = year - 2025L)
stopifnot(all(is.finite(pp$dose_us_skill_1)))
Y <- c(cert_q = "y_cert_q", hours_q = "y_hours_q", cert_full = "y_cert_full", hours_full = "y_hours_full", wage_q = "y_wage_q")
FE <- "county_fips + aewr_region_id^year"
ev <- tests <- trend <- growth <- list()
for (nm in names(Y)) for (dv in c("dose_us_skill_1", "dose_h2a_skill_1", "dose_us_skill_2")) {
  d <- pp %>% mutate(D = .data[[dv]], DT = D * t) %>% bite_complete(c(Y[[nm]], "D"))
  if (nm == "wage_q") d <- d %>% group_by(county_fips) %>% filter(n() == 14) %>% ungroup()  # balanced offers
  m <- feols(as.formula(paste(Y[[nm]], "~ i(year, D, ref = 2025) |", FE)), d, vcov = ~state_fips)
  ct <- coeftable(m)
  ev[[paste(nm, dv)]] <- tibble(outcome = nm, dose = dv, term = rownames(ct), year = as.integer(sub("year::([0-9]+):D", "\\1", rownames(ct))),
    estimate = ct[, 1], se = ct[, 2], p = ct[, 4], n = nobs(m), mean_y = mean(d[[Y[[nm]]]]))
  w <- wald(m, keep = "year::", print = FALSE)
  tests[[paste(nm, dv)]] <- tibble(outcome = nm, dose = dv, test = "joint FY2012-FY2024 dose x FY = 0", stat = w$stat, p = w$p,
    df1 = w$df1, df2 = w$df2, states = n_distinct(d$state_fips), counties = n_distinct(d$county_fips))
  ml <- feols(as.formula(paste(Y[[nm]], "~ DT |", FE)), d, vcov = ~state_fips, demeaned = TRUE)
  r <- id_rows(ml, d, "DT", tibble(outcome = nm, dose = dv, model = "linear dose x (FY-2025) trend"), c("state_fips", "aewr_region_id"), "D")
  wb <- if (nm != "wage_q" && dv == "dose_us_skill_1") id_wild(ml, d, "DT", "state_fips")$wild_p else NA_real_
  # If the linear drift continues, a static post-vs-pre comparison (county FE,
  # FY2012-FY2025 pre) picks up drift x (2026 - mean pre FY) per dollar of dose.
  trend[[paste(nm, dv)]] <- r %>% mutate(wild_p_state = wb, implied_drift_over_14y_per_dollar = 14 * estimate,
    static_did_drift_bias_fy2026 = estimate * (2026 - mean(2012:2025)),
    static_did_drift_bias_from_event_path = -mean(c(ev[[paste(nm, dv)]]$estimate, 0)) + estimate)
}
# Warning check: dose vs adjacent-year growth; FY2024->FY2025 is the one of interest.
for (nm in c("cert_q", "hours_q")) {
  g <- pp %>% arrange(county_fips, year) %>% group_by(county_fips) %>%
    mutate(dy = .data[[Y[[nm]]]] - lag(.data[[Y[[nm]]]])) %>% ungroup() %>% filter(!is.na(dy))
  for (yy in sort(unique(g$year))) {
    d <- g %>% filter(year == yy) %>% mutate(D = dose_us_skill_1)
    m <- feols(dy ~ D | aewr_region_id, d, vcov = ~state_fips)
    growth[[paste(nm, yy)]] <- tibble(outcome = nm, growth_to_fy = yy, estimate = coef(m)[["D"]], se = se(m)[["D"]],
      p = pvalue(m)[["D"]], mean_abs_dy = mean(abs(d$dy)), n = nobs(m))
  }
}
# Which power DGP is realistic? Share of pre-period residual variance (after
# county + former-region x FY FE) that is common to a state-FY cell. The
# simulated shared DGPs put 50% (state) or 25%+ (state, plus spatial) there.
state_share <- bind_rows(lapply(c("cert_q", "hours_q"), function(nm) bind_rows(lapply(list(2020:2025, 2012:2025), function(yy) {
  d <- pp %>% filter(year %in% yy)
  r <- resid(feols(as.formula(paste(Y[[nm]], "~ 1 |", FE)), d))
  cell <- paste(d$state_fips, d$year)
  tibble(outcome = nm, years = paste(range(yy), collapse = "-"),
    state_year_share = sum(ave(r, cell)^2) / sum(r^2),
    expected_share_if_no_state_shock = n_distinct(cell) / nrow(d))
}))))
id_write(state_share, out_dir, "track1_state_year_share.csv")
print(as.data.frame(state_share), digits = 3)
growth <- bind_rows(growth) %>% group_by(outcome) %>%
  mutate(rank_abs_t_of_fy2025 = rank(-abs(estimate / se))[growth_to_fy == 2025]) %>% ungroup()
id_write(bind_rows(ev), out_dir, "track1_pre_event_study.csv")
id_write(bind_rows(tests), out_dir, "track1_pre_trend_tests.csv")
id_write(bind_rows(trend), out_dir, "track1_pre_linear_trend.csv")
id_write(growth, out_dir, "track1_fy2025_growth_check.csv")
print(as.data.frame(bind_rows(tests)), digits = 3)
print(as.data.frame(bind_rows(trend) %>% filter(cluster == "state_fips") %>% select(outcome, dose, estimate, se, p, wild_p_state, mean_y, mean_price)), digits = 3)
print(as.data.frame(growth), digits = 3)
