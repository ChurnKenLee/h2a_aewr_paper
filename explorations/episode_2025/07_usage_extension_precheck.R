# Pre-period check for extending the 2025-rule usage design to later windows.
# H-2A outcomes through FY2025 only. Size-weighted, within former region,
# multi-state regions: the design as the registry will declare it.
#
# Part 1, late-season windows of the FY2026 Q4 disclosure, by case receipt date:
#   jul1_aug2   received 1 Jul - 2 Aug of the FY (event-1 rates in force);
#   aug3_sep30  received 3 Aug - 30 Sep of the FY (event-2 rates in force; the ruling and backpay notice fall here).
#   Both require a decision inside the same FY, as the Q4 disclosure will.
# Part 2, a second season: FY2027 decisions October-June against FY2025, alone
#   and averaged with the first season.
#
# Placebo years use real pre-period growth with the actual doses; the true
# effect is zero. Growth is the symmetric rate 2(a-b)/(a+b), which is defined
# when a thin window has zero hours in one year; weights are (a+b)/2.
# Args: 1 = scratch output dir (outside the repo); 2 = shared-outcomes dir.
source("explorations/episode_2025/ep_helpers.R")
out_dir <- id_output_dir()
shared <- ep_shared_dir()
frame <- ep_state_frame() %>% filter(multi_state_region)

rows <- arrow::open_dataset(file.path(shared, "outcomes_case_rows.parquet")) %>%
  filter(fiscal_year >= EP_FIRST_FY, fiscal_year <= EP_LAST_PRE_FY) %>%
  select(fiscal_year, state_fips, received, in_dq13, cert_hours) %>% collect() %>% filter(state_fips != "00")
id_assert_pre(rows$fiscal_year)
rows <- rows %>% mutate(md = as.integer(format(received, "%m%d")), same_year = as.integer(format(received, "%Y")) == fiscal_year,
  window = case_when(in_dq13 ~ "oct_jun_decisions", same_year & md >= 701L & md <= 802L ~ "jul1_aug2",
    same_year & md >= 803L & md <= 930L ~ "aug3_sep30", TRUE ~ "other"))
shares <- rows %>% group_by(fiscal_year, window) %>% summarise(hours_m = sum(cert_hours, na.rm = TRUE) / 1e6, .groups = "drop") %>%
  group_by(fiscal_year) %>% mutate(share = hours_m / sum(hours_m)) %>% ungroup()
id_write(shares, out_dir, "extension_window_shares.csv")

st <- rows %>% filter(window != "other") %>% group_by(state_fips, year = fiscal_year, window) %>%
  summarise(h = sum(cert_hours, na.rm = TRUE), .groups = "drop") %>%
  tidyr::complete(state_fips = frame$state_fips, year = EP_FIRST_FY:EP_LAST_PRE_FY, window, fill = list(h = 0)) %>%
  inner_join(frame, by = "state_fips")
sym <- function(a, b) ifelse(a + b > 0, 2 * (a - b) / (a + b), NA_real_)

fit <- function(x, dose) {
  x <- x %>% filter(is.finite(y), w > 0)
  ep_fit(x, "y", dose, fe = "aewr_region_id")
}
part1 <- bind_rows(lapply(c("oct_jun_decisions", "jul1_aug2", "aug3_sep30"), function(wn) {
  d <- st %>% filter(window == wn) %>% arrange(state_fips, year) %>% group_by(state_fips) %>%
    mutate(y = sym(h, lag(h)), w = (h + lag(h)) / 2) %>% ungroup()
  dose <- if (wn == "aug3_sep30") "dose2_us" else "dose1_us"
  bind_rows(lapply(EP_PLACEBO_FY, function(tau) fit(d %>% filter(year == tau), dose) %>% mutate(tau = tau))) %>%
    mutate(estimator = paste0(wn, " on ", dose), outcome = "hours", weighting = "size", sample = "one-year contrast")
}))

wide <- st %>% filter(window == "oct_jun_decisions") %>% select(state_fips, year, h) %>% mutate(lh = log(h))
stopifnot(all(is.finite(wide$lh)))
second <- bind_rows(lapply(2018:(EP_LAST_PRE_FY - 1L), function(tau) {
  x <- frame %>% inner_join(wide %>% filter(year == tau - 1L) %>% transmute(state_fips, l0 = lh, w = h), by = "state_fips") %>%
    inner_join(wide %>% filter(year == tau) %>% transmute(state_fips, l1 = lh), by = "state_fips") %>%
    inner_join(wide %>% filter(year == tau + 1L) %>% transmute(state_fips, l2 = lh), by = "state_fips") %>%
    mutate(d_season2 = dose1_us + dose2_us, d_avg = (dose1_us + d_season2) / 2)
  bind_rows(
    ep_fit(x %>% mutate(y = l1 - l0), "y", "dose1_us", fe = "aewr_region_id") %>% mutate(estimator = "first season vs base year"),
    ep_fit(x %>% mutate(y = l2 - l0), "y", "d_season2", fe = "aewr_region_id") %>% mutate(estimator = "second season vs base year"),
    ep_fit(x %>% mutate(y = (l1 + l2) / 2 - l0), "y", "d_avg", fe = "aewr_region_id") %>% mutate(estimator = "mean of two seasons vs base year")
  ) %>% mutate(tau = tau)
})) %>% mutate(outcome = "hours", weighting = "size", sample = "second season (placebo years 2018-2024)")

res <- bind_rows(part1, second)
id_write(res, out_dir, "extension_placebo_by_year.csv")
mean_rate <- 17.28   # hours-weighted FY2025 rate, multi-state regions (full_cut_support.csv)
summ <- ep_summarise(res, mean_rate)
id_write(summ, out_dir, "extension_placebo_summary.csv")
options(width = 220)
cat("SHARE OF FY CERTIFIED HOURS BY WINDOW\n")
print(as.data.frame(shares %>% filter(fiscal_year %in% c(2019, 2023, 2024, 2025)) %>% select(-hours_m) %>%
  tidyr::pivot_wider(names_from = window, values_from = share)), digits = 3)
cat("\nPLACEBO SUMMARY (elasticity units; true effect is zero)\n")
print(as.data.frame(summ %>% select(sample, estimator, placebo_years, states, dose_sd_left, bias_elasticity, mde_elasticity_rmse, mde_elasticity_se)), digits = 3)
