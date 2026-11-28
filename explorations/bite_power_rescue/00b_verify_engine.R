# Self-test of Phase 1 blind calibration engine against known production benchmarks:
# 1. Exact 2-period binomial equivalence vs fixest::fepois on FY2024-2025 pre-policy placebo
# 2. Blind Phase 1 placebo calibration for policy_2025 Track 1 benchmark (FY2012-FY2025)
# 3. Blind Phase 1 FE-only calibration for code/designs/bite 1b-PPML and 1a-PPML benchmarks

source("explorations/bite_power_rescue/helpers_phase1.R")

cat("=== TEST 1: Exact 2-period PPML binomial equivalence (FY2024-FY2025 pre-policy) ===\n")
qp <- read_parquet(path_processed("policy_2025", "quantity_county_fy.parquet")) %>%
  mutate(year = as.integer(year))
d2425 <- qp %>% filter(year %in% c(2024L, 2025L))

m_fep <- fit_cv3j_fixest(
  d2425,
  hours_full_fy ~ x_placebo | county_fips + aewr_region_id^year,
  term = "x_placebo",
  cluster_col = "state_fips",
  estimator = "fepois"
)

# Collapse to 2-period wide by county
w2425 <- d2425 %>%
  select(county_fips, state_fips, aewr_region_id, year, hours_full_fy, x_placebo) %>%
  pivot_wider(names_from = year, values_from = c(hours_full_fy, x_placebo)) %>%
  mutate(dx = x_placebo_2025 - x_placebo_2024)

m_bin_cty <- fit_2period_ppml_cv3j(
  y0 = w2425$hours_full_fy_2024,
  y1 = w2425$hours_full_fy_2025,
  dx = w2425$dx,
  group_fe = w2425$aewr_region_id,
  cluster_id = w2425$state_fips
)

# Also test state-collapsed (46 rows!)
w2425_st <- w2425 %>%
  group_by(state_fips, aewr_region_id) %>%
  summarise(y0 = sum(hours_full_fy_2024), y1 = sum(hours_full_fy_2025), dx = first(dx), .groups = "drop")

m_bin_st <- fit_2period_ppml_cv3j(
  y0 = w2425_st$y0, y1 = w2425_st$y1, dx = w2425_st$dx,
  group_fe = w2425_st$aewr_region_id, cluster_id = w2425_st$state_fips
)

cat(sprintf("fixest::fepois : b = %.10f, se_cv3j = %.10f\n", m_fep$estimate, m_fep$se_cv3j))
cat(sprintf("binom (county) : b = %.10f, se_cv3j = %.10f (diff=%.2e, %.2e)\n",
  m_bin_cty$beta, m_bin_cty$se_cv3j, abs(m_bin_cty$beta - m_fep$estimate), abs(m_bin_cty$se_cv3j - m_fep$se_cv3j)))
cat(sprintf("binom (state)  : b = %.10f, se_cv3j = %.10f (diff=%.2e, %.2e)\n",
  m_bin_st$beta, m_bin_st$se_cv3j, abs(m_bin_st$beta - m_fep$estimate), abs(m_bin_st$se_cv3j - m_fep$se_cv3j)))
stopifnot(abs(m_bin_st$beta - m_fep$estimate) < 1e-9, abs(m_bin_st$se_cv3j - m_fep$se_cv3j) < 1e-9)

cat("\n=== TEST 2: Blind 2025 Track 1 Benchmark Calibration (650 real pre-policy draws, FY2012-2025) ===\n")
st_dose <- qp %>%
  filter(year %in% c(2025L, 2026L)) %>%
  distinct(state_fips, aewr_region_id, year, x_us_skill_1) %>%
  pivot_wider(names_from = year, values_from = x_us_skill_1, names_prefix = "x_") %>%
  mutate(dx_raw = x_2026 - x_2025, dx_ln = log(x_2026) - log(x_2025))
mean_p_2526 <- mean(c(st_dose$x_2025, st_dose$x_2026))

st_pre <- qp %>%
  filter(year <= 2025L) %>%
  group_by(state_fips, aewr_region_id, year) %>%
  summarise(hours = sum(hours_full_fy), cert = sum(cert_full_fy), .groups = "drop")

set.seed(SEED_PHASE1)
pre_pairs <- 2013L:2025L
B_per_yr <- 50L # 13 * 50 = 650 draws
b_h <- se_h <- b_c <- se_c <- b_lh <- se_lh <- numeric(length(pre_pairs) * B_per_yr)
idx <- 0L
for (yr in pre_pairs) {
  sub_p <- st_pre %>%
    filter(year %in% c(yr - 1L, yr)) %>%
    pivot_wider(names_from = year, values_from = c(hours, cert))
  y0h <- sub_p[[paste0("hours_", yr - 1L)]]; y1h <- sub_p[[paste0("hours_", yr)]]
  y0c <- sub_p[[paste0("cert_", yr - 1L)]];  y1c <- sub_p[[paste0("cert_", yr)]]
  # Align to st_dose order
  ord <- match(st_dose$state_fips, sub_p$state_fips)
  y0h <- y0h[ord]; y1h <- y1h[ord]; y0c <- y0c[ord]; y1c <- y1c[ord]
  for (k in seq_len(B_per_yr)) {
    idx <- idx + 1L
    perm_dx <- st_dose %>%
      group_by(aewr_region_id) %>%
      mutate(perm_ord = sample.int(n()), dx_p = dx_raw[perm_ord], dln_p = dx_ln[perm_ord]) %>%
      ungroup()
    rh <- fit_2period_ppml_cv3j(y0h, y1h, perm_dx$dx_p, st_dose$aewr_region_id, st_dose$state_fips)
    rc <- fit_2period_ppml_cv3j(y0c, y1c, perm_dx$dx_p, st_dose$aewr_region_id, st_dose$state_fips)
    rlh <- fit_2period_ppml_cv3j(y0h, y1h, perm_dx$dln_p, st_dose$aewr_region_id, st_dose$state_fips)
    b_h[[idx]] <- rh$beta * mean_p_2526; se_h[[idx]] <- rh$se_cv3j * mean_p_2526
    b_c[[idx]] <- rc$beta * mean_p_2526; se_c[[idx]] <- rc$se_cv3j * mean_p_2526
    b_lh[[idx]] <- rlh$beta;             se_lh[[idx]] <- rlh$se_cv3j
  }
}
eh <- compute_empirical_mde(b_h, se_h, df_t = 45L)
ec <- compute_empirical_mde(b_c, se_c, df_t = 45L)
elh <- compute_empirical_mde(b_lh, se_lh, df_t = 45L)
cat(sprintf("Benchmark Track 1 hours (raw $ -> el): size=%.3f [%.3f,%.3f], MDE=%.3f (analytic=%.3f), prod=0.72\n",
  eh$emp_size, eh$wilson_lo, eh$wilson_hi, eh$mde_upper, eh$analytic_mde))
cat(sprintf("Benchmark Track 1 cert  (raw $ -> el): size=%.3f [%.3f,%.3f], MDE=%.3f (analytic=%.3f), prod=0.52\n",
  ec$emp_size, ec$wilson_lo, ec$wilson_hi, ec$mde_upper, ec$analytic_mde))
cat(sprintf("Benchmark Track 1 hours (ln rate)    : size=%.3f [%.3f,%.3f], MDE=%.3f (analytic=%.3f), prod=0.54\n",
  elh$emp_size, elh$wilson_lo, elh$wilson_hi, elh$mde_upper, elh$analytic_mde))

cat("\n=== TEST 3: Blind 2011-2025 Bite Benchmark Calibration (FE-only, 999 draws) ===\n")
bp <- read_parquet(path_processed("bite_county_year_panel.parquet")) %>%
  filter(year >= 2011L, year <= 2025L)

cal_1b <- calibrate_multiperiod_ppml_blind(
  bp, y_col = "man_hours_certified_start_year", x_col = "E_p25_x_ln_aewr",
  fe_rhs = "county_fips + aewr_region_id^year"
)
cal_1a <- calibrate_multiperiod_ppml_blind(
  bp, y_col = "man_hours_certified_start_year", x_col = "E_p25_x_ln_aewr",
  fe_rhs = "county_fips + aewr_region_id^year",
  ctrl_rhs = "i(year, E_p25, ref = 2011)"
)
cat(sprintf("Benchmark 1b-PPML hours: G_eff=%.2f (prod=13.36), SD(X~)=%.4f (prod=0.1496), SY_ratio=%.2f (prod=2.85), size=%.3f, MDE_1SD=%.3f (prod=0.379), MDE_2SD=%.3f\n",
  cal_1b$g_eff, cal_1b$sd_x_tilde_wt, cal_1b$sy_ms_ratio, cal_1b$emp_size, cal_1b$mde_1sd, cal_1b$mde_2sd))
cat(sprintf("Benchmark 1a-PPML hours: G_eff=%.2f, SD(X~)=%.4f (prod=0.0277), ctrl_infl=%.2f, size=%.3f, MDE_1SD=%.3f (prod=2.106)\n",
  cal_1a$g_eff, cal_1a$sd_x_tilde_wt, cal_1a$ctrl_se_inflation, cal_1a$emp_size, cal_1a$mde_1sd))
