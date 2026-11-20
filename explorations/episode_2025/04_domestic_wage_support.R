# Design-stage support for the domestic-wage design. Outcomes through 2024 only;
# no QCEW or QWI file after 2024 exists in the repository, so this design is blind.
#
#   change in log crop pay_c = beta * (S_c x cut_s) + gamma * S_c + state effects
#   S_c   = mean 2022-2024 certified positions per 2011 farm worker, winsorized at p99;
#   cut_s = 2025-10-02 U.S. skill-I rate minus the FY2025 regional rate.
# Placebo "event" years 2016..2024 use real pre-period pay changes with the
# actual dose, so the true coefficient is zero. MDE = 2.8 x max(mean clustered
# SE, SD of placebo estimates). Benchmark: crop pay moving with H-2A offers in
# proportion to S, 0.66 / 16.60 log points per share-dollar (first stage and
# mean offered wage of the 2025-rule design).
#
# Three outcome series:
#   qcew_annual    QCEW NAICS 111 annual pay per worker, year-over-year;
#   qwi_annual     QWI NAICS 111 stable-job monthly earnings, 4-quarter mean, year-over-year;
#   qwi_q4_q2      QWI, mean year-over-year change over Q4(t-1), Q1(t), Q2(t): the
#                  quarters event 1 was in force before the August 2026 schedule.
#   qcew_q4_q2     QCEW NAICS 111 average weekly wage, same window (only if argument 2 is given).
# Args: 1 = scratch output dir (outside the repo);
#       2 = optional qcew_crop_county_quarter_<first>_<last>.parquet from 06_qcew_quarterly_extract.R,
#           covering years no later than 2024.
source("explorations/episode_2025/ep_helpers.R")
out_dir <- id_output_dir()
LAST_OUTCOME_YEAR <- 2024L
BENCHMARK <- 0.66 / 16.60
frame <- ep_state_frame()

panel <- arrow::open_dataset(path_processed("county_year_panel.parquet")) %>% filter(year <= LAST_OUTCOME_YEAR, year >= 2012L) %>%
  select(county_fips, state_fips, year, h2a_cert_share_farm_workers_2011_start_year, qcew_crop_sector_disclosed,
    qcew_crop_sector_annual_avg_emplvl, qcew_crop_sector_total_annual_wages) %>% collect()
stopifnot(!anyDuplicated(panel[c("county_fips", "year")]), max(panel$year) <= LAST_OUTCOME_YEAR)
S <- panel %>% filter(year %in% 2022:2024) %>% group_by(county_fips, state_fips) %>%
  summarise(S = mean(h2a_cert_share_farm_workers_2011_start_year), .groups = "drop") %>% filter(is.finite(S)) %>%
  mutate(Sw = pmin(S, quantile(S, .99)))
exposure <- tibble(counties = nrow(S), p50 = median(S$S), p90 = quantile(S$S, .9), p99 = quantile(S$S, .99),
  sd_winsorized = sd(S$Sw), within_state_sd_winsorized = sd(S$Sw - ave(S$Sw, S$state_fips)))
id_write(exposure, out_dir, "domestic_wage_exposure_support.csv")

qcew_annual <- panel %>% filter(qcew_crop_sector_disclosed %in% TRUE, qcew_crop_sector_annual_avg_emplvl >= 10, qcew_crop_sector_total_annual_wages > 0) %>%
  transmute(county_fips, year, lw = log(qcew_crop_sector_total_annual_wages / qcew_crop_sector_annual_avg_emplvl), emp = qcew_crop_sector_annual_avg_emplvl) %>%
  arrange(county_fips, year) %>% group_by(county_fips) %>%
  mutate(dlw = lw - lag(lw), emp_l = lag(emp), ok = year - lag(year) == 1L) %>% ungroup() %>% filter(ok, is.finite(dlw)) %>%
  select(county_fips, tau = year, dlw, emp_l)

qwi <- read_parquet(path_int("qwi_county_ag_quarterly_earnings.parquet")) %>%
  filter(industry_code == "111", year >= 2012L, year <= LAST_OUTCOME_YEAR, qwi_stable_monthly_earnings_status == 1L,
    qwi_stable_monthly_earnings > 0, qwi_stable_employment >= 10) %>%
  transmute(county_fips, year, qtr, lw = log(qwi_stable_monthly_earnings), emp = qwi_stable_employment)
stopifnot(!anyDuplicated(qwi[c("county_fips", "year", "qtr")]))
qwi_annual <- qwi %>% group_by(county_fips, year) %>% summarise(nq = n(), lw = mean(lw), emp = mean(emp), .groups = "drop") %>% filter(nq == 4) %>%
  arrange(county_fips, year) %>% group_by(county_fips) %>%
  mutate(dlw = lw - lag(lw), emp_l = lag(emp), ok = year - lag(year) == 1L) %>% ungroup() %>% filter(ok, is.finite(dlw)) %>%
  select(county_fips, tau = year, dlw, emp_l)
# Year-over-year change by quarter, then the mean over Q4(tau-1), Q1(tau), Q2(tau).
yoy <- qwi %>% inner_join(qwi %>% transmute(county_fips, year = year + 1L, qtr, lw_l = lw, emp_l = emp), by = c("county_fips", "year", "qtr")) %>%
  mutate(d = lw - lw_l, tau = ifelse(qtr == 4L, year + 1L, year)) %>% filter(qtr %in% c(4L, 1L, 2L), tau <= LAST_OUTCOME_YEAR)
qwi_q4_q2 <- yoy %>% group_by(county_fips, tau) %>% summarise(nq = n(), dlw = mean(d), emp_l = mean(emp_l), .groups = "drop") %>%
  filter(nq == 3) %>% select(county_fips, tau, dlw, emp_l)

# The same window on QCEW average weekly wages, if the pre-period extract is supplied.
qcew_quarterly_path <- commandArgs(trailingOnly = TRUE)[2]
qcew_q4_q2 <- NULL
if (!is.na(qcew_quarterly_path)) {
  qq <- read_parquet(qcew_quarterly_path)
  stopifnot(max(qq$year) <= LAST_OUTCOME_YEAR, !anyDuplicated(qq[c("county_fips", "year", "qtr")]))
  qq <- qq %>% filter(disclosed, emp >= 10, is.finite(avg_wkly_wage), avg_wkly_wage > 0) %>%
    transmute(county_fips, year, qtr, lw = log(avg_wkly_wage), emp)
  qcew_q4_q2 <- qq %>% inner_join(qq %>% transmute(county_fips, year = year + 1L, qtr, lw_l = lw, emp_l = emp), by = c("county_fips", "year", "qtr")) %>%
    mutate(d = lw - lw_l, tau = ifelse(qtr == 4L, year + 1L, year)) %>% filter(qtr %in% c(4L, 1L, 2L), tau <= LAST_OUTCOME_YEAR) %>%
    group_by(county_fips, tau) %>% summarise(nq = n(), dlw = mean(d), emp_l = mean(emp_l), .groups = "drop") %>%
    filter(nq == 3) %>% select(county_fips, tau, dlw, emp_l)
}

hours24 <- arrow::open_dataset(path_processed("county_year_panel.parquet")) %>% filter(year == 2024L) %>%
  select(county_fips, man_hours_certified_start_year) %>% collect()

run <- function(o, label) {
  d <- o %>% inner_join(S, by = "county_fips") %>% inner_join(frame %>% select(state_fips, dose1_us), by = "state_fips") %>%
    group_by(state_fips, tau) %>% filter(n() >= 2) %>% ungroup() %>% mutate(D = Sw * dose1_us)
  last <- d %>% filter(tau == max(tau)) %>% left_join(hours24, by = "county_fips")
  cover <- tibble(series = label, last_placebo_year = max(d$tau), counties = nrow(last), states = n_distinct(last$state_fips),
    share_of_2024_certified_hours = sum(last$man_hours_certified_start_year, na.rm = TRUE) / sum(hours24$man_hours_certified_start_year, na.rm = TRUE),
    sd_change_after_state_fe = sd(resid(feols(dlw ~ 1 | state_fips, last))),
    sd_dose_after_state_fe = sd(resid(feols(D ~ 1 | state_fips, last))),
    sd_dose_after_state_fe_and_S = sd(resid(feols(D ~ Sw | state_fips, last))))
  res <- bind_rows(lapply(2016:max(d$tau), function(tau) {
    x <- d %>% filter(.data$tau == !!tau)
    bind_rows(lapply(c("equal", "employment"), function(wt) {
      w <- if (wt == "equal") NULL else ~emp_l
      a <- feols(dlw ~ D | state_fips, x, weights = w, vcov = ~state_fips)
      b <- feols(dlw ~ D + Sw | state_fips, x, weights = w, vcov = ~state_fips)
      tibble(series = label, tau, weighting = wt, spec = c("S x cut", "S x cut, controlling for S"),
        estimate = c(coef(a)["D"], coef(b)["D"]), se = c(se(a)["D"], se(b)["D"]), counties = nobs(a))
    }))
  }))
  summ <- res %>% group_by(series, weighting, spec) %>% summarise(placebo_years = n(), counties = round(mean(counties)),
    mean_estimate = mean(estimate), sd_estimate = sd(estimate), mean_se = mean(se), .groups = "drop") %>%
    mutate(mde = 2.8 * pmax(mean_se, sd_estimate), benchmark = BENCHMARK, mde_over_benchmark = mde / benchmark,
      mde_pct_pay_at_S0.3_cut4 = 100 * mde * 0.3 * 4)
  list(cover = cover, placebo = res, summary = summ)
}
r <- list(run(qcew_annual, "qcew_annual"), run(qwi_annual, "qwi_annual"), run(qwi_q4_q2, "qwi_q4_q2"))
if (!is.null(qcew_q4_q2)) r <- c(r, list(run(qcew_q4_q2, "qcew_q4_q2")))
cover <- bind_rows(lapply(r, `[[`, "cover")); summ <- bind_rows(lapply(r, `[[`, "summary"))
id_write(cover, out_dir, "domestic_wage_coverage.csv")
id_write(bind_rows(lapply(r, `[[`, "placebo")), out_dir, "domestic_wage_placebo_by_year.csv")
id_write(summ, out_dir, "domestic_wage_placebo_summary.csv")
options(width = 220)
print(as.data.frame(exposure), digits = 3); print(as.data.frame(cover), digits = 3)
print(as.data.frame(summ %>% select(series, weighting, spec, placebo_years, counties, mean_estimate, sd_estimate, mean_se, mde, mde_over_benchmark, mde_pct_pay_at_S0.3_cut4)), digits = 3)
