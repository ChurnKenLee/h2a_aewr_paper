# Domestic-wage design: the declared estimate, and the same computation on a
# placebo year for checking the code path.
#
#   change_c = mean over Q4(tau-1), Q1(tau), Q2(tau) of the year-over-year change in
#              log QCEW average weekly wage, private crop production (NAICS 111);
#   change_c = beta * (S_c x cut_s) + gamma * S_c + state effects + error.
# S_c and cut_s come from the frozen exposure file, not from the panel.
#
# Args: 1 = scratch output dir (outside the repo);
#       2 = qcew_crop_county_quarter_<first>_<last>.parquet from 06_qcew_quarterly_extract.R;
#       3 = tau, the calendar year of Q1 and Q2 (2026 for the declared estimate; <= 2024 for a placebo run).
# For tau = 2026 the script refuses to run unless the registry and the exposure
# file match their recorded SHA-256 and all three quarters are present.
source("explorations/episode_2025/ep_helpers.R")
out_dir <- id_output_dir()
args <- commandArgs(trailingOnly = TRUE)
tau <- as.integer(args[3])
stopifnot(file.exists(args[2]), is.finite(tau))
here_dir <- "explorations/episode_2025"
sha <- function(f) unname(tools::sha256sum(f))
declared <- tau == 2026L
if (!declared && tau > 2024L) stop("Only tau = 2026 (declared) or tau <= 2024 (placebo) is allowed")
if (declared) {
  rec <- readLines(file.path(here_dir, "domestic_wage_registry.sha256"))
  want <- setNames(sub(" .*", "", rec), sub("^[0-9a-f]+ +", "", rec))
  for (f in names(want)) if (!identical(sha(file.path(here_dir, f)), unname(want[f]))) stop("Hash mismatch: ", f)
}

expo <- readr::read_csv(file.path(here_dir, "domestic_wage_exposure_frozen.csv"),
  col_types = readr::cols(county_fips = "c", state_fips = "c", cz_id = "c", .default = "d"))
stopifnot(!anyDuplicated(expo$county_fips), identical(expo$county_fips, harmonize_county_fips_2010(expo$county_fips)))

q <- read_parquet(args[2])
need <- tibble(year = c(tau - 2L, tau - 1L, tau - 1L, tau - 1L, tau, tau), qtr = c(4L, 1L, 2L, 4L, 1L, 2L))
have <- distinct(q, year, qtr)
if (nrow(anti_join(need, have, by = c("year", "qtr"))) > 0) stop("The extract lacks a quarter the window needs; nothing is estimated")
q <- q %>% semi_join(need, by = c("year", "qtr")) %>%
  filter(disclosed, emp >= 10, is.finite(avg_wkly_wage), avg_wkly_wage > 0) %>%
  transmute(county_fips, year, qtr, lw = log(avg_wkly_wage), emp)
yoy <- q %>% inner_join(q %>% transmute(county_fips, year = year + 1L, qtr, lw_l = lw, emp_l = emp), by = c("county_fips", "year", "qtr")) %>%
  filter((year == tau - 1L & qtr == 4L) | (year == tau & qtr %in% 1:2)) %>%
  group_by(county_fips) %>% summarise(nq = n(), change = mean(lw - lw_l), emp_l = mean(emp_l), .groups = "drop") %>% filter(nq == 3)
d <- yoy %>% inner_join(expo, by = "county_fips") %>% group_by(state_fips) %>% filter(n() >= 2) %>% ungroup() %>%
  mutate(D = S_winsorized * cut_us, D_h2a = S_winsorized * cut_h2a, D_cz = S_cz_winsorized * cut_us)
stopifnot(nrow(d) > 0, !anyDuplicated(d$county_fips))

jackknife <- function(f, data, term, w = NULL) {
  est <- unname(coef(feols(f, data, weights = w))[term])
  g <- unique(data$state_fips)
  loo <- vapply(g, function(s) unname(coef(feols(f, data[data$state_fips != s, ], weights = w))[term]), numeric(1))
  G <- length(g); se <- sqrt((G - 1) / G * sum((loo - mean(loo))^2))
  c(se = se, p = 2 * pt(-abs(est / se), G - 1), low = est - qt(.975, G - 1) * se, high = est + qt(.975, G - 1) * se, states = G)
}
one <- function(label, f, term, w = NULL) {
  m <- feols(f, d, weights = w, vcov = ~state_fips); jk <- jackknife(f, d, term, w)
  tibble(specification = label, term, estimate = unname(coef(m)[term]), se_state_crv1 = unname(se(m)[term]),
    se_jackknife = jk["se"], p_jackknife = jk["p"], low_jackknife = jk["low"], high_jackknife = jk["high"],
    counties = nobs(m), states = jk["states"])
}
res <- bind_rows(
  one("primary: S x cut, controlling for S, equal weights", change ~ D + S_winsorized | state_fips, "D"),
  one("secondary: employment weights", change ~ D + S_winsorized | state_fips, "D", ~emp_l),
  one("sensitivity: H-2A cash dose", change ~ D_h2a + S_winsorized | state_fips, "D_h2a"),
  one("sensitivity: commuting-zone intensity", change ~ D_cz + S_cz_winsorized | state_fips, "D_cz"),
  one("labeled sensitivity: no S control", change ~ D | state_fips, "D")
) %>% mutate(tau = tau, run = ifelse(declared, "declared", "placebo"), benchmark = 0.66 / 16.60,
  pct_pay_at_S0.3_cut4 = 100 * estimate * 0.3 * -4)
id_write(res, out_dir, sprintf("domestic_wage_estimate_tau%d.csv", tau))
options(width = 220); print(as.data.frame(res), digits = 3)
