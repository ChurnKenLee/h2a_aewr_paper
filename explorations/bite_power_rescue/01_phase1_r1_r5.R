# Phase 1 Strictly Outcome-Blind Audit: Candidates R1 (Competing-Floor Truncation & Sub-State)
# and R5 (Local-Wage Crossover Bite, Gated First on Offered-Wage First Stage).
# HARD RULE: NEVER reads or regresses FY2026 H-2A quantity outcomes!
# Offered-wage first stages (FY2025-2026) and pre-policy quantity calibrations (FY2012-2025) ONLY.

source("explorations/bite_power_rescue/helpers_phase1.R")

dir.create("explorations/bite_power_rescue", recursive = TRUE, showWarnings = FALSE)

cat("======================================================================\n")
cat("CANDIDATE R1: Competing-Floor Truncation Bite (Oct 2, 2025 Cut)\n")
cat("======================================================================\n")

# Load Track 1 county-FY panel: KEEP FY2026 STATUTORY SCHEDULES & WAGES ONLY;
# MASK FY2026 QUANTITY COLUMNS IMMEDIATELY TO ENFORCE STRICT OUTCOME-BLINDNESS!
qp_all <- read_parquet(path_processed("policy_2025", "quantity_county_fy.parquet")) %>%
  mutate(year = as.integer(year))

# Mask FY2026 quantities immediately
qp_blind <- qp_all %>%
  mutate(
    across(
      c(starts_with("cert_"), starts_with("hours_"), starts_with("y_cert_"), starts_with("y_hours_")),
      ~ ifelse(year == 2026L, NA_real_, .x)
    )
  )
rm(qp_all)
stopifnot(all(is.na(qp_blind$hours_full_fy[qp_blind$year == 2026L])))

# Construct unified effective statutory floor from state_real_minwages.parquet (already in mw_ag!)
# Double-check parity with data/intermediate/state_real_minwages.parquet directly
mw_check <- read_parquet(path_int("state_real_minwages.parquet")) %>%
  filter(year %in% 2012L:2026L) %>%
  transmute(state_fips = state_fips(state_fips), year = as.integer(year), mw_ag_src = as.numeric(prevailing_ag_min_wage))

qp_blind <- qp_blind %>%
  left_join(mw_check, by = c("state_fips", "year"), relationship = "many-to-one")
stopifnot(max(abs(qp_blind$mw_ag - qp_blind$mw_ag_src)) < 1e-12)

# State-level schedule table for FY2025-FY2026
st_sched <- qp_blind %>%
  filter(year %in% c(2025L, 2026L)) %>%
  distinct(state_fips, aewr_region_id, year, x_us_skill_1, mw_ag) %>%
  mutate(
    x_eff = pmax(x_us_skill_1, mw_ag),
    ln_x_eff = log(x_eff),
    trunc_gap = pmax(0, mw_ag - x_us_skill_1)
  ) %>%
  pivot_wider(names_from = year, values_from = c(x_us_skill_1, mw_ag, x_eff, ln_x_eff, trunc_gap)) %>%
  mutate(
    dose_raw = x_us_skill_1_2026 - x_us_skill_1_2025,
    dose_eff = x_eff_2026 - x_eff_2025,
    dln_raw = log(x_us_skill_1_2026) - log(x_us_skill_1_2025),
    dln_eff = ln_x_eff_2026 - ln_x_eff_2025,
    d_trunc = trunc_gap_2026 - trunc_gap_2025
  ) %>%
  arrange(state_fips)

# Also build A1 sub-state ag min wage floor (using cached A1 CSVs: NY downstate/upstate, OR standard/nonurban)
sub_mw_raw <- read_csv("explorations/identification_search/A1/sources/state_ag_minwage_fy2025_2026.csv", show_col_types = FALSE) %>%
  mutate(state_fips = state_fips(state_fips))
sub_cty <- read_csv("explorations/identification_search/A1/sources/minwage_substate_counties.csv", show_col_types = FALSE) %>%
  mutate(county_fips = harmonize_county_fips_2010(county_fips))

# Effective FY2026 April 1 snapshot ("2026-01-01" to "2026-06-30") and FY2025 ("2025-09-29")
# Note: OR also steps on July 1; check Apr 1 snapshot (matches Track 1 Apr 1 convention) AND FY-weighted avg
sub_rates_26 <- sub_mw_raw %>%
  filter(snapshot_date == as.Date("2026-01-01"), area_id != "portland_metro") %>%
  select(state_fips, area_id, mw_sub_26_apr = h2a_state_floor)
sub_rates_26_jul <- sub_mw_raw %>%
  filter(snapshot_date == as.Date("2026-07-01"), area_id != "portland_metro") %>%
  select(state_fips, area_id, mw_sub_26_jul = h2a_state_floor)
sub_rates_25 <- sub_mw_raw %>%
  filter(snapshot_date == as.Date("2025-09-29"), area_id != "portland_metro") %>%
  select(state_fips, area_id, mw_sub_25 = h2a_state_floor)

sub_rates <- sub_rates_26 %>%
  left_join(sub_rates_26_jul, by = c("state_fips", "area_id")) %>%
  left_join(sub_rates_25, by = c("state_fips", "area_id")) %>%
  mutate(mw_sub_26_fyavg = 0.75 * mw_sub_26_apr + 0.25 * mw_sub_26_jul)

counties_frame <- qp_blind %>%
  filter(year == 2025L) %>%
  select(county_fips, state_fips, aewr_region_id, emp_farm_2011, hours_2025 = hours_full_fy) %>%
  left_join(sub_cty, by = "county_fips") %>%
  mutate(
    area_id = case_when(
      !is.na(area_id) ~ area_id,
      state_fips == "36" ~ "upstate",
      state_fips == "41" ~ "standard",
      TRUE ~ "statewide"
    )
  ) %>%
  left_join(sub_rates, by = c("state_fips", "area_id")) %>%
  left_join(st_sched %>% select(state_fips, x_us_skill_1_2025, x_us_skill_1_2026, mw_ag_2025, mw_ag_2026, x_eff_2025, x_eff_2026), by = "state_fips") %>%
  mutate(
    x_eff_sub_2025 = pmax(x_us_skill_1_2025, mw_sub_25),
    x_eff_sub_2026_apr = pmax(x_us_skill_1_2026, mw_sub_26_apr),
    x_eff_sub_2026_fyavg = pmax(x_us_skill_1_2026, mw_sub_26_fyavg),
    dose_eff_sub_apr = x_eff_sub_2026_apr - x_eff_sub_2025,
    dose_eff_sub_fyavg = x_eff_sub_2026_fyavg - x_eff_sub_2025,
    substate_trunc_apr = pmax(0, mw_sub_26_apr - x_us_skill_1_2026),
    substate_trunc_fyavg = pmax(0, mw_sub_26_fyavg - x_us_skill_1_2026)
  )

# --- 1A. R1a & R1b FIRST STAGE: Offered-Wage Pass-Through (R7 filer-state-occ + County) ---
cat("\n--- R1a & R1b First-Stage Offered-Wage Verification ---\n")
wfso <- read_parquet(path_processed("policy_2025", "wage_filer_state_occupation.parquet")) %>%
  filter(tier == "R7", soc != "unknown")

make_wage_pair <- function(tau) {
  b <- wfso %>% filter(fiscal_year == tau - 1L) %>% select(-fiscal_year)
  a <- wfso %>% filter(fiscal_year == tau) %>% select(-fiscal_year)
  inner_join(b, a, by = c("tier", "filer", "state_fips", "aewr_region_id", "soc"), suffix = c("0", "1")) %>%
    mutate(dw = wage1 - wage0) %>%
    left_join(st_sched, by = c("state_fips", "aewr_region_id"))
}
w26 <- make_wage_pair(2026L)
w25 <- make_wage_pair(2025L) # Pre-policy placebo (FY2024-2025)

fs_r7_raw <- fit_cv3j_fixest(w26, dw ~ dose_raw | aewr_region_id, "dose_raw", "state_fips", "feols", weights_fml = ~h0)
fs_r7_eff <- fit_cv3j_fixest(w26, dw ~ dose_eff | aewr_region_id, "dose_eff", "state_fips", "feols", weights_fml = ~h0)
fs_r7_eff_eq <- fit_cv3j_fixest(w26, dw ~ dose_eff | aewr_region_id, "dose_eff", "state_fips", "feols")
fs_r7_eff_pl <- fit_cv3j_fixest(w25, dw ~ dose_eff | aewr_region_id, "dose_eff", "state_fips", "feols", weights_fml = ~h0)
fs_r7_trunc <- fit_cv3j_fixest(w26, dw ~ d_trunc + dose_raw | aewr_region_id, "d_trunc", "state_fips", "feols", weights_fml = ~h0)
fs_r7_trunc_pl <- fit_cv3j_fixest(w25, dw ~ d_trunc + dose_raw | aewr_region_id, "d_trunc", "state_fips", "feols", weights_fml = ~h0)

# Also check R1a with A1 state-level sub-corrected ag floor (NY upstate/downstate hours-weighted state avg & OR standard)
# And check county-level offered-wage first stage (y_wage_full_fy, continuing counties)
cty_w26 <- qp_blind %>%
  filter(year %in% c(2025L, 2026L)) %>%
  group_by(county_fips) %>%
  filter(sum(is.finite(y_wage_full_fy)) == 2L) %>%
  ungroup() %>%
  mutate(
    x_eff = pmax(x_us_skill_1, mw_ag),
    ln_x_eff = log(x_eff),
    trunc_gap = pmax(0, mw_ag - x_us_skill_1)
  ) %>%
  left_join(counties_frame %>% select(county_fips, x_eff_sub_2025, x_eff_sub_2026_apr, x_eff_sub_2026_fyavg), by = "county_fips") %>%
  mutate(
    x_eff_sub_apr = ifelse(year == 2026L, x_eff_sub_2026_apr, x_eff_sub_2025),
    x_eff_sub_fyavg = ifelse(year == 2026L, x_eff_sub_2026_fyavg, x_eff_sub_2025)
  )

cty_w25 <- qp_blind %>%
  filter(year %in% c(2024L, 2025L)) %>%
  group_by(county_fips) %>%
  filter(sum(is.finite(y_wage_full_fy)) == 2L) %>%
  ungroup() %>%
  left_join(st_sched %>% select(state_fips, x_eff_2025, x_eff_2026, trunc_gap_2026), by = "state_fips") %>%
  mutate(
    x_eff_pl = ifelse(year == 2025L, x_eff_2026, x_eff_2025),
    trunc_pl = ifelse(year == 2025L, trunc_gap_2026, 0)
  )

fs_cty_raw <- fit_cv3j_fixest(cty_w26, y_wage_full_fy ~ x_us_skill_1 | county_fips + aewr_region_id^year, "x_us_skill_1", "state_fips", "feols")
fs_cty_eff <- fit_cv3j_fixest(cty_w26, y_wage_full_fy ~ x_eff | county_fips + aewr_region_id^year, "x_eff", "state_fips", "feols")
fs_cty_eff_sub <- fit_cv3j_fixest(cty_w26, y_wage_full_fy ~ x_eff_sub_apr | county_fips + aewr_region_id^year, "x_eff_sub_apr", "state_fips", "feols")
fs_cty_eff_pl <- fit_cv3j_fixest(cty_w25, y_wage_full_fy ~ x_eff_pl | county_fips + aewr_region_id^year, "x_eff_pl", "state_fips", "feols")
fs_cty_trunc <- fit_cv3j_fixest(cty_w26, y_wage_full_fy ~ trunc_gap + x_us_skill_1 | county_fips + aewr_region_id^year, "trunc_gap", "state_fips", "feols")
fs_cty_trunc_pl <- fit_cv3j_fixest(cty_w25, y_wage_full_fy ~ trunc_pl + x_placebo | county_fips + aewr_region_id^year, "trunc_pl", "state_fips", "feols")
# Within-state R1c first stage (State x FY FE!)
fs_cty_r1c_apr <- fit_cv3j_fixest(cty_w26, y_wage_full_fy ~ x_eff_sub_apr | county_fips + state_fips^year, "x_eff_sub_apr", "state_fips", "feols")

cat(sprintf("R7 (h0) Raw AEWR        : pi=%.4f [CV3J %.4f], F=%.2f (N=%d)\n", fs_r7_raw$estimate, fs_r7_raw$se_cv3j, (fs_r7_raw$estimate/fs_r7_raw$se_cv3j)^2, fs_r7_raw$n_obs))
cat(sprintf("R7 (h0) Eff Floor (R1a) : pi=%.4f [CV3J %.4f], F=%.2f (placebo=%.4f [%.4f], p=%.3f)\n",
  fs_r7_eff$estimate, fs_r7_eff$se_cv3j, (fs_r7_eff$estimate/fs_r7_eff$se_cv3j)^2, fs_r7_eff_pl$estimate, fs_r7_eff_pl$se_cv3j, fs_r7_eff_pl$p_cv3j))
cat(sprintf("R7 (eq) Eff Floor (R1a) : pi=%.4f [CV3J %.4f], F=%.2f\n", fs_r7_eff_eq$estimate, fs_r7_eff_eq$se_cv3j, (fs_r7_eff_eq$estimate/fs_r7_eff_eq$se_cv3j)^2))
cat(sprintf("R7 (h0) Trunc Gap (R1b) : pi=%.4f [CV3J %.4f], F=%.2f (placebo=%.4f [%.4f], p=%.3f)\n",
  fs_r7_trunc$estimate, fs_r7_trunc$se_cv3j, (fs_r7_trunc$estimate/fs_r7_trunc$se_cv3j)^2, fs_r7_trunc_pl$estimate, fs_r7_trunc_pl$se_cv3j, fs_r7_trunc_pl$p_cv3j))
cat(sprintf("Cty     Raw AEWR        : pi=%.4f [CV3J %.4f], F=%.2f (N=%d)\n", fs_cty_raw$estimate, fs_cty_raw$se_cv3j, (fs_cty_raw$estimate/fs_cty_raw$se_cv3j)^2, fs_cty_raw$n_obs))
cat(sprintf("Cty     Eff Floor (R1a) : pi=%.4f [CV3J %.4f], F=%.2f (placebo=%.4f [%.4f], p=%.3f)\n",
  fs_cty_eff$estimate, fs_cty_eff$se_cv3j, (fs_cty_eff$estimate/fs_cty_eff$se_cv3j)^2, fs_cty_eff_pl$estimate, fs_cty_eff_pl$se_cv3j, fs_cty_eff_pl$p_cv3j))
cat(sprintf("Cty     Eff Sub   (R1a*): pi=%.4f [CV3J %.4f], F=%.2f\n", fs_cty_eff_sub$estimate, fs_cty_eff_sub$se_cv3j, (fs_cty_eff_sub$estimate/fs_cty_eff_sub$se_cv3j)^2))
cat(sprintf("Cty     Trunc Gap (R1b) : pi=%.4f [CV3J %.4f], F=%.2f (placebo=%.4f [%.4f], p=%.3f)\n",
  fs_cty_trunc$estimate, fs_cty_trunc$se_cv3j, (fs_cty_trunc$estimate/fs_cty_trunc$se_cv3j)^2, fs_cty_trunc_pl$estimate, fs_cty_trunc_pl$se_cv3j, fs_cty_trunc_pl$p_cv3j))
cat(sprintf("Cty     Within-St (R1c) : ok=%s, pi=%.4f [CV3J %.4f], G=%d\n",
  as.character(fs_cty_r1c_apr$ok), fs_cty_r1c_apr$estimate, fs_cty_r1c_apr$se_cv3j, fs_cty_r1c_apr$G))

# --- 1B. R1c Sub-State Support & Hours Audit ---
cat("\n--- R1c Sub-State Minimum-Wage Tier Support Audit ---\n")
r1c_support <- counties_frame %>%
  group_by(state_fips) %>%
  summarise(
    n_counties = n(),
    n_tiers_apr = n_distinct(x_eff_sub_2026_apr),
    sd_within_state = sd(x_eff_sub_2026_apr),
    hours_2025 = sum(hours_2025, na.rm = TRUE),
    .groups = "drop"
  )
tot_h25 <- sum(r1c_support$hours_2025)
r1c_var_states <- r1c_support %>% filter(n_tiers_apr > 1L)
cat(sprintf("States with ANY within-state ag min-wage tier variation in FY2026: %d (FIPS: %s)\n",
  nrow(r1c_var_states), paste(r1c_var_states$state_fips, collapse = ", ")))
print(r1c_var_states %>% mutate(hours_share_us = hours_2025 / tot_h25))
# Exact Fisher-info support for R1c under County + State x FY FE on FY2024-FY2025
w2425_r1c <- qp_blind %>%
  filter(year %in% c(2024L, 2025L)) %>%
  select(county_fips, state_fips, aewr_region_id, year, hours_full_fy) %>%
  pivot_wider(names_from = year, values_from = hours_full_fy, names_prefix = "h_") %>%
  left_join(counties_frame %>% select(county_fips, dose_eff_sub_apr), by = "county_fips")
sup_r1c <- compute_2period_blind_support(w2425_r1c$h_2024, w2425_r1c$h_2025, w2425_r1c$dose_eff_sub_apr,
                                         group_fe = w2425_r1c$state_fips, cluster_id = w2425_r1c$state_fips)
cat(sprintf("R1c within-State FE: G_eff(state)=%.3f, G_active_with_var=%d, max_state_share=%.3f (%s), SD(X~)_wt=%.4f\n",
  sup_r1c$g_eff, nrow(r1c_var_states), sup_r1c$max_cluster_share, sup_r1c$max_cluster, sup_r1c$sd_x_tilde_wt))

# --- 1C. R1a & R1b Blind Quantity Power Calibration (650 real pre-policy draws, FY2012-FY2025) ---
cat("\n--- R1a & R1b Strictly Outcome-Blind Quantity Power Calibration (FY2012-FY2025) ---\n")
st_pre <- qp_blind %>%
  filter(year <= 2025L) %>%
  group_by(state_fips, aewr_region_id, year) %>%
  summarise(hours = sum(hours_full_fy), cert = sum(cert_full_fy), .groups = "drop")

# Compute exact FE-only support & MS_SY/MS_county on pre-policy FY2024-FY2025
w2425_all <- qp_blind %>%
  filter(year %in% c(2024L, 2025L)) %>%
  select(county_fips, state_fips, aewr_region_id, year, hours_full_fy, cert_full_fy) %>%
  pivot_wider(names_from = year, values_from = c(hours_full_fy, cert_full_fy)) %>%
  left_join(st_sched, by = c("state_fips", "aewr_region_id"))

sup_raw <- compute_2period_blind_support(w2425_all$hours_full_fy_2024, w2425_all$hours_full_fy_2025, w2425_all$dose_raw, w2425_all$aewr_region_id, w2425_all$state_fips)
sup_r1a <- compute_2period_blind_support(w2425_all$hours_full_fy_2024, w2425_all$hours_full_fy_2025, w2425_all$dose_eff, w2425_all$aewr_region_id, w2425_all$state_fips)
sup_r1a_ln <- compute_2period_blind_support(w2425_all$hours_full_fy_2024, w2425_all$hours_full_fy_2025, w2425_all$dln_eff, w2425_all$aewr_region_id, w2425_all$state_fips)
sup_r1b <- compute_2period_blind_support(w2425_all$hours_full_fy_2024, w2425_all$hours_full_fy_2025, w2425_all$d_trunc, w2425_all$aewr_region_id, w2425_all$state_fips, Z = matrix(w2425_all$dose_raw, ncol = 1L))

# Shared shock ratio on FY2012-2025 FE-only PPML
m0_pre <- fixest::fepois(hours_full_fy ~ 1 | county_fips + aewr_region_id^year, data = qp_blind %>% filter(year <= 2025L), notes = FALSE)
u0_pre <- (qp_blind %>% filter(year <= 2025L))[fixest::obs(m0_pre), ]
sy_pre_25 <- compute_sy_noise_ratio_ppml(u0_pre$hours_full_fy, fitted(m0_pre), u0_pre$aewr_region_id, u0_pre$state_fips, u0_pre$year)

# Matched FY2024-2025 placebos (using actual R1a / R1b doses shifted back 1 year)
w2425_st <- st_pre %>%
  filter(year %in% c(2024L, 2025L)) %>%
  pivot_wider(names_from = year, values_from = c(hours, cert)) %>%
  left_join(st_sched, by = c("state_fips", "aewr_region_id"))

mean_p_raw <- mean(c(st_sched$x_us_skill_1_2025, st_sched$x_us_skill_1_2026))
mean_p_eff <- mean(c(st_sched$x_eff_2025, st_sched$x_eff_2026))

pl_r1a_h <- fit_2period_ppml_cv3j(w2425_st$hours_2024, w2425_st$hours_2025, w2425_st$dose_eff, w2425_st$aewr_region_id, w2425_st$state_fips)
pl_r1a_c <- fit_2period_ppml_cv3j(w2425_st$cert_2024,  w2425_st$cert_2025,  w2425_st$dose_eff, w2425_st$aewr_region_id, w2425_st$state_fips)
pl_r1a_ln_h <- fit_2period_ppml_cv3j(w2425_st$hours_2024, w2425_st$hours_2025, w2425_st$dln_eff, w2425_st$aewr_region_id, w2425_st$state_fips)
pl_r1b_h <- fit_2period_ppml_cv3j(w2425_st$hours_2024, w2425_st$hours_2025, w2425_st$d_trunc, w2425_st$aewr_region_id, w2425_st$state_fips, Z = matrix(w2425_st$dose_raw, ncol = 1L))

# 650 placebo draws (13 pre-policy pairs x 50 within-region permutations)
set.seed(SEED_PHASE1)
pre_pairs <- 2013L:2025L
B_per_yr <- 50L
n_draws <- length(pre_pairs) * B_per_yr
b_r1a_h <- se_r1a_h <- b_r1a_c <- se_r1a_c <- b_r1a_lh <- se_r1a_lh <- b_r1b_h <- se_r1b_h <- numeric(n_draws)
idx <- 0L
for (yr in pre_pairs) {
  sub_p <- st_pre %>%
    filter(year %in% c(yr - 1L, yr)) %>%
    pivot_wider(names_from = year, values_from = c(hours, cert))
  ord <- match(st_sched$state_fips, sub_p$state_fips)
  y0h <- sub_p[[paste0("hours_", yr - 1L)]][ord]; y1h <- sub_p[[paste0("hours_", yr)]][ord]
  y0c <- sub_p[[paste0("cert_", yr - 1L)]][ord];  y1c <- sub_p[[paste0("cert_", yr)]][ord]
  for (k in seq_len(B_per_yr)) {
    idx <- idx + 1L
    perm <- st_sched %>%
      group_by(aewr_region_id) %>%
      mutate(po = sample.int(n()), de = dose_eff[po], dl = dln_eff[po], dt = d_trunc[po], dr = dose_raw[po]) %>%
      ungroup()
    r1 <- fit_2period_ppml_cv3j(y0h, y1h, perm$de, st_sched$aewr_region_id, st_sched$state_fips)
    r2 <- fit_2period_ppml_cv3j(y0c, y1c, perm$de, st_sched$aewr_region_id, st_sched$state_fips)
    r3 <- fit_2period_ppml_cv3j(y0h, y1h, perm$dl, st_sched$aewr_region_id, st_sched$state_fips)
    r4 <- fit_2period_ppml_cv3j(y0h, y1h, perm$dt, st_sched$aewr_region_id, st_sched$state_fips, Z = matrix(perm$dr, ncol = 1L))
    b_r1a_h[[idx]]  <- r1$beta; se_r1a_h[[idx]]  <- r1$se_cv3j
    b_r1a_c[[idx]]  <- r2$beta; se_r1a_c[[idx]]  <- r2$se_cv3j
    b_r1a_lh[[idx]] <- r3$beta; se_r1a_lh[[idx]] <- r3$se_cv3j
    b_r1b_h[[idx]]  <- r4$beta; se_r1b_h[[idx]]  <- r4$se_cv3j
  }
}

e_r1a_h  <- compute_empirical_mde(b_r1a_h * mean_p_eff, se_r1a_h * mean_p_eff, df_t = 45L)
e_r1a_c  <- compute_empirical_mde(b_r1a_c * mean_p_eff, se_r1a_c * mean_p_eff, df_t = 45L)
e_r1a_lh <- compute_empirical_mde(b_r1a_lh, se_r1a_lh, df_t = 45L)
e_r1b_h  <- compute_empirical_mde(b_r1b_h * mean_p_eff, se_r1b_h * mean_p_eff, df_t = 45L)

# Convert R1a dollar MDE to epsilon_cost MDE using R1a's actual first-stage pi_hat!
# Note: eps_cost = beta_dollar * W_BAR_OFFERED * COST_RATIO / pi_hat
pi_r1a_r7  <- fs_r7_eff$estimate
pi_r1a_cty <- fs_cty_eff$estimate
mde_eps_r1a_r7  <- (e_r1a_h$mde_upper / mean_p_eff) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / pi_r1a_r7
mde_eps_r1a_cty <- (e_r1a_h$mde_upper / mean_p_eff) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / pi_r1a_cty
mde_eps_r1a_lh  <- e_r1a_lh$mde_upper * COST_RATIO_CENTRAL / pi_r1a_cty

# Also check direct-grower R1a (strictly pre-policy FY2020-2025 direct_hours!)
cat("\n--- R1a Direct-Grower-Only Pre-Policy Calibration (FY2020-FY2025) ---\n")
disc_lc <- open_dataset(path_int("h2a_disclosure_records.parquet")) %>%
  select(fiscal_year, case_number, decision_date, h2a_labor_contractor) %>%
  filter(fiscal_year >= 2020L, fiscal_year <= 2025L) %>%
  collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year)) %>%
  group_by(case_number, fiscal_year) %>%
  summarise(
    dec_raw = first(na.omit(as.character(decision_date))),
    lc_raw = first(na.omit(as.character(h2a_labor_contractor))),
    .groups = "drop"
  ) %>%
  mutate(
    dec = as.Date(substr(dec_raw, 1L, 10L)),
    fy_s = as.Date(sprintf("%d-10-01", fiscal_year - 1L)),
    fy_e = as.Date(sprintf("%d-09-30", fiscal_year)),
    w_fy = !is.na(dec) & dec >= fy_s & dec <= fy_e,
    is_h2alc = toupper(trimws(coalesce(lc_raw, ""))) %in% c("Y", "YES", "TRUE", "1")
  ) %>%
  filter(w_fy)

nh_pre_mean <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, period_begin_date, number_of_hours) %>%
  filter(fiscal_year >= 2012L, fiscal_year <= 2025L) %>%
  collect() %>%
  filter(!is.na(as.Date(period_begin_date))) %>%
  summarise(m = mean(suppressWarnings(as.numeric(number_of_hours)), na.rm = TRUE)) %>%
  pull(m)

ws_dg_pre <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, case_number, county_fips_list, adjusted_nbr_workers_certified, period_begin_date, period_end_date, number_of_hours) %>%
  filter(fiscal_year >= 2020L, fiscal_year <= 2025L) %>%
  collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year), begin = as.Date(period_begin_date), end = as.Date(period_end_date)) %>%
  filter(!is.na(begin)) %>%
  inner_join(disc_lc %>% select(case_number, fiscal_year, is_h2alc), by = c("case_number", "fiscal_year")) %>%
  filter(!is_h2alc) %>%
  mutate(
    nh = coalesce(suppressWarnings(as.numeric(number_of_hours)), nh_pre_mean),
    workers = as.numeric(adjusted_nbr_workers_certified),
    hours = as.numeric(end - begin) / 7 * nh * workers,
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    nc = stringr::str_count(county_fips_list, ",") + 1L
  ) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = harmonize_county_fips_2010(trimws(county_fips_list)), share = 1 / nc) %>%
  inner_join(counties_frame %>% select(county_fips, state_fips, aewr_region_id), by = "county_fips") %>%
  group_by(state_fips, aewr_region_id, year = fiscal_year) %>%
  summarise(dg_hours = sum(hours * share, na.rm = TRUE), dg_cert = sum(workers * share, na.rm = TRUE), .groups = "drop")

# Direct-grower R1a & Raw benchmark calibration across FY2020-2025 (5 transitions x 120 perms = 600 draws)
set.seed(SEED_PHASE1 + 1L)
dg_pairs <- 2021L:2025L
B_dg <- 120L
b_dg_eff_h <- se_dg_eff_h <- b_dg_eff_c <- se_dg_eff_c <- b_dg_raw_h <- se_dg_raw_h <- b_dg_ln_h <- se_dg_ln_h <- numeric(length(dg_pairs) * B_dg)
idx <- 0L
for (yr in dg_pairs) {
  sub_p <- ws_dg_pre %>% filter(year %in% c(yr - 1L, yr)) %>% pivot_wider(names_from = year, values_from = c(dg_hours, dg_cert))
  ord <- match(st_sched$state_fips, sub_p$state_fips)
  y0h <- coalesce(sub_p[[paste0("dg_hours_", yr - 1L)]][ord], 0); y1h <- coalesce(sub_p[[paste0("dg_hours_", yr)]][ord], 0)
  y0c <- coalesce(sub_p[[paste0("dg_cert_", yr - 1L)]][ord], 0);  y1c <- coalesce(sub_p[[paste0("dg_cert_", yr)]][ord], 0)
  for (k in seq_len(B_dg)) {
    idx <- idx + 1L
    perm <- st_sched %>% group_by(aewr_region_id) %>% mutate(po = sample.int(n()), de = dose_eff[po], dr = dose_raw[po], dl = dln_eff[po]) %>% ungroup()
    r1 <- fit_2period_ppml_cv3j(y0h, y1h, perm$de, st_sched$aewr_region_id, st_sched$state_fips)
    r2 <- fit_2period_ppml_cv3j(y0c, y1c, perm$de, st_sched$aewr_region_id, st_sched$state_fips)
    r3 <- fit_2period_ppml_cv3j(y0h, y1h, perm$dr, st_sched$aewr_region_id, st_sched$state_fips)
    r4 <- fit_2period_ppml_cv3j(y0h, y1h, perm$dl, st_sched$aewr_region_id, st_sched$state_fips)
    b_dg_eff_h[[idx]] <- r1$beta * mean_p_eff; se_dg_eff_h[[idx]] <- r1$se_cv3j * mean_p_eff
    b_dg_eff_c[[idx]] <- r2$beta * mean_p_eff; se_dg_eff_c[[idx]] <- r2$se_cv3j * mean_p_eff
    b_dg_raw_h[[idx]] <- r3$beta * mean_p_raw; se_dg_raw_h[[idx]] <- r3$se_cv3j * mean_p_raw
    b_dg_ln_h[[idx]]  <- r4$beta;              se_dg_ln_h[[idx]]  <- r4$se_cv3j
  }
}
e_dg_eff_h <- compute_empirical_mde(b_dg_eff_h, se_dg_eff_h, df_t = 45L)
e_dg_eff_c <- compute_empirical_mde(b_dg_eff_c, se_dg_eff_c, df_t = 45L)
e_dg_raw_h <- compute_empirical_mde(b_dg_raw_h, se_dg_raw_h, df_t = 45L)
e_dg_ln_h  <- compute_empirical_mde(b_dg_ln_h, se_dg_ln_h, df_t = 45L)

# Matched FY2024-25 direct-grower placebo for R1a
dg2425 <- ws_dg_pre %>% filter(year %in% c(2024L, 2025L)) %>% pivot_wider(names_from = year, values_from = c(dg_hours, dg_cert)) %>%
  left_join(st_sched, by = c("state_fips", "aewr_region_id"))
pl_dg_r1a_h <- fit_2period_ppml_cv3j(dg2425$dg_hours_2024, dg2425$dg_hours_2025, dg2425$dose_eff, dg2425$aewr_region_id, dg2425$state_fips)
pl_dg_r1a_c <- fit_2period_ppml_cv3j(dg2425$dg_cert_2024,  dg2425$dg_cert_2025,  dg2425$dose_eff, dg2425$aewr_region_id, dg2425$state_fips)

cat(sprintf("R1a Agg Hours (eff $)  : G_eff=%.2f (raw=%.2f), SD(X~)=%.3f (raw=%.3f), SY=%.2f, size=%.3f [%.3f,%.3f], MDE_el=%.3f, MDE_eps_cost(R7/cty)=%.3f/%.3f, pl2425=%.3f (p=%.3f)\n",
  sup_r1a$g_eff, sup_raw$g_eff, sup_r1a$sd_x_tilde_wt, sup_raw$sd_x_tilde_wt, sy_pre_25$ms_ratio,
  e_r1a_h$emp_size, e_r1a_h$wilson_lo, e_r1a_h$wilson_hi, e_r1a_h$mde_upper, mde_eps_r1a_r7, mde_eps_r1a_cty,
  pl_r1a_h$beta * mean_p_eff, 2*pt(-abs(pl_r1a_h$beta/pl_r1a_h$se_cv3j), 45)))
cat(sprintf("R1a Agg Hours (ln eff) : G_eff=%.2f, SD(X~)=%.4f, size=%.3f [%.3f,%.3f], MDE_el=%.3f, MDE_eps_cost=%.3f, pl2425=%.3f (p=%.3f)\n",
  sup_r1a_ln$g_eff, sup_r1a_ln$sd_x_tilde_wt, e_r1a_lh$emp_size, e_r1a_lh$wilson_lo, e_r1a_lh$wilson_hi,
  e_r1a_lh$mde_upper, mde_eps_r1a_lh, pl_r1a_ln_h$beta, 2*pt(-abs(pl_r1a_ln_h$beta/pl_r1a_ln_h$se_cv3j), 45)))
cat(sprintf("R1a Agg Cert  (eff $)  : size=%.3f [%.3f,%.3f], MDE_el=%.3f, pl2425=%.3f (p=%.3f)\n",
  e_r1a_c$emp_size, e_r1a_c$wilson_lo, e_r1a_c$wilson_hi, e_r1a_c$mde_upper,
  pl_r1a_c$beta * mean_p_eff, 2*pt(-abs(pl_r1a_c$beta/pl_r1a_c$se_cv3j), 45)))
cat(sprintf("R1a Direct-Grower Hours: size=%.3f [%.3f,%.3f], MDE_eff$=%.3f (raw$=%.3f, ln_eff=%.3f), pl2425=%.3f (p=%.3f)\n",
  e_dg_eff_h$emp_size, e_dg_eff_h$wilson_lo, e_dg_eff_h$wilson_hi, e_dg_eff_h$mde_upper, e_dg_raw_h$mde_upper, e_dg_ln_h$mde_upper,
  pl_dg_r1a_h$beta * mean_p_eff, 2*pt(-abs(pl_dg_r1a_h$beta/pl_dg_r1a_h$se_cv3j), 45)))
cat(sprintf("R1a Direct-Grower Cert : size=%.3f [%.3f,%.3f], MDE_eff$=%.3f, pl2425=%.3f (p=%.3f)\n",
  e_dg_eff_c$emp_size, e_dg_eff_c$wilson_lo, e_dg_eff_c$wilson_hi, e_dg_eff_c$mde_upper,
  pl_dg_r1a_c$beta * mean_p_eff, 2*pt(-abs(pl_dg_r1a_c$beta/pl_dg_r1a_c$se_cv3j), 45)))
cat(sprintf("R1b Pure Trunc Gap Hrs : G_eff=%.2f, SD(X~)=%.3f, size=%.3f [%.3f,%.3f], MDE_el=%.3f, pl2425=%.3f (p=%.3f)\n",
  sup_r1b$g_eff, sup_r1b$sd_x_tilde_wt, e_r1b_h$emp_size, e_r1b_h$wilson_lo, e_r1b_h$wilson_hi, e_r1b_h$mde_upper,
  pl_r1b_h$beta * mean_p_eff, 2*pt(-abs(pl_r1b_h$beta/pl_r1b_h$se_cv3j), 45)))

cat("\n======================================================================\n")
cat("CANDIDATE R5: Local-Wage Crossover Bite (FIRST-STAGE GATE FIRST!)\n")
cat("======================================================================\n")

# Build predetermined 2024 county local-wage benchmarks
cp24 <- read_parquet(
  path_processed("county_year_panel.parquet"),
  col_select = c("county_fips", "state_fips", "aewr_region_id", "year", "oews_big_six_mean_hourly_wage", "wage_p25", "ppi_2012")
) %>%
  filter(year == 2024L) %>%
  transmute(
    county_fips = harmonize_county_fips_2010(county_fips),
    w_oews24 = as.numeric(oews_big_six_mean_hourly_wage),
    w_p25_24 = as.numeric(wage_p25 * ppi_2012)
  )

qwi24 <- read_parquet(path_int("qwi_county_ag_quarterly_earnings.parquet")) %>%
  filter(year == 2024L, industry_code == "111", is.finite(qwi_stable_monthly_earnings), qwi_stable_monthly_earnings > 0) %>%
  mutate(county_fips = harmonize_county_fips_2010(county_fips)) %>%
  group_by(county_fips) %>%
  summarise(w_qwi111_24 = mean(qwi_stable_monthly_earnings, na.rm = TRUE) / (4.3333 * 40), .groups = "drop")

r5_cty <- counties_frame %>%
  select(county_fips, state_fips, aewr_region_id, x_us_skill_1_2025, x_us_skill_1_2026, x_eff_2025, x_eff_2026) %>%
  left_join(cp24, by = "county_fips") %>%
  left_join(qwi24, by = "county_fips") %>%
  mutate(
    # Binary crossover indicators: AEWR_2025 > w_local > AEWR_eff_2026 (market binds after cut!)
    cross_oews = as.numeric(is.finite(w_oews24) & x_eff_2025 > w_oews24 & w_oews24 > x_eff_2026),
    cross_p25  = as.numeric(is.finite(w_p25_24) & x_eff_2025 > w_p25_24 & w_p25_24 > x_eff_2026),
    cross_qwi  = as.numeric(is.finite(w_qwi111_24) & x_eff_2025 > w_qwi111_24 & w_qwi111_24 > x_eff_2026),
    # Continuous dollar crossover gap: max(0, min(w_local, AEWR_2025) - AEWR_eff_2026)
    # = dollars of statutory cut offset by local wage floor w_local!
    gap_oews = ifelse(is.finite(w_oews24), pmax(0, pmin(w_oews24, x_eff_2025) - x_eff_2026), NA_real_),
    gap_p25  = ifelse(is.finite(w_p25_24), pmax(0, pmin(w_p25_24, x_eff_2025) - x_eff_2026), NA_real_),
    gap_qwi  = ifelse(is.finite(w_qwi111_24), pmax(0, pmin(w_qwi111_24, x_eff_2025) - x_eff_2026), NA_real_),
    # Market-truncated effective wage floor in 2026: max(AEWR_eff_2026, w_local)
    # Predicted market-truncated cut: max(AEWR_eff_2026, w_local) - max(AEWR_2025, w_local)
    mkt_cut_oews = ifelse(is.finite(w_oews24), pmax(x_eff_2026, w_oews24) - pmax(x_eff_2025, w_oews24), NA_real_),
    mkt_cut_p25  = ifelse(is.finite(w_p25_24), pmax(x_eff_2026, w_p25_24) - pmax(x_eff_2025, w_p25_24), NA_real_),
    mkt_cut_qwi  = ifelse(is.finite(w_qwi111_24), pmax(x_eff_2026, w_qwi111_24) - pmax(x_eff_2025, w_qwi111_24), NA_real_),
    # Also test simple indicator w_local > AEWR_eff_2026 (covers both crossover and always-above)
    above26_oews = ifelse(is.finite(w_oews24), as.numeric(w_oews24 > x_eff_2026), NA_real_),
    above26_p25  = ifelse(is.finite(w_p25_24), as.numeric(w_p25_24 > x_eff_2026), NA_real_),
    above26_qwi  = ifelse(is.finite(w_qwi111_24), as.numeric(w_qwi111_24 > x_eff_2026), NA_real_)
  )

# Build continuing R7 filer x county x SOC offered-wage change panel (FY2025-FY2026 AND FY2024-FY2025 placebo)
# Strictly OFFERED WAGES ONLY (no quantity outcomes!)
disc_w <- open_dataset(path_int("h2a_disclosure_records.parquet")) %>%
  select(fiscal_year, case_number, decision_date, soc_code) %>%
  filter(fiscal_year >= 2024L, fiscal_year <= 2026L) %>%
  collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year)) %>%
  group_by(fiscal_year, case_number) %>%
  summarise(
    dec = as.Date(substr(first(na.omit(as.character(decision_date))), 1L, 10L)),
    case_soc = first(na.omit(as.character(soc_code))),
    .groups = "drop"
  ) %>%
  mutate(
    fy_s = as.Date(sprintf("%d-10-01", fiscal_year - 1L)),
    fy_e = as.Date(sprintf("%d-09-30", fiscal_year)),
    w_fy = !is.na(dec) & dec >= fy_s & dec <= fy_e,
    soc = substr(trimws(coalesce(case_soc, "")), 1L, 7L),
    soc = ifelse(grepl("^[0-9]{2}-[0-9]{4}$", soc), soc, "unknown")
  ) %>%
  filter(w_fy)

fids_2426 <- read_parquet(path_int("policy_2025", "filer", "filer_ids.parquet")) %>%
  filter(fiscal_year >= 2024L, fiscal_year <= 2026L) %>%
  transmute(fiscal_year = as.integer(fiscal_year), case_number = as.character(case_number), filer = as.character(filer_R7)) %>%
  filter(!is.na(filer), nzchar(filer))

ws_w2426 <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, case_number, county_fips_list, adjusted_nbr_workers_certified, period_begin_date, period_end_date, number_of_hours, wage_rate, wage_unit) %>%
  filter(fiscal_year >= 2024L, fiscal_year <= 2026L) %>%
  collect() %>%
  mutate(
    fiscal_year = as.integer(fiscal_year),
    begin = as.Date(period_begin_date),
    end = as.Date(period_end_date),
    wage = suppressWarnings(as.numeric(wage_rate)),
    hourly = toupper(trimws(wage_unit)) %in% c("HOUR", "HOURLY", "HR") & is.finite(wage) & wage > 0 & wage <= 100
  ) %>%
  filter(!is.na(begin), hourly) %>%
  inner_join(disc_w %>% select(fiscal_year, case_number, soc), by = c("fiscal_year", "case_number")) %>%
  inner_join(fids_2426, by = c("fiscal_year", "case_number")) %>%
  filter(soc != "unknown") %>%
  mutate(
    nh = coalesce(suppressWarnings(as.numeric(number_of_hours)), nh_pre_mean),
    h = as.numeric(end - begin) / 7 * nh * as.numeric(adjusted_nbr_workers_certified),
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    nc = stringr::str_count(county_fips_list, ",") + 1L
  ) %>%
  filter(is.finite(h), h > 0) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = harmonize_county_fips_2010(trimws(county_fips_list)), h = h / nc) %>%
  inner_join(counties_frame %>% select(county_fips, state_fips, aewr_region_id), by = "county_fips") %>%
  group_by(filer, county_fips, state_fips, aewr_region_id, soc, fiscal_year) %>%
  summarise(wage = sum(wage * h) / sum(h), h = sum(h), .groups = "drop")

pair_fc <- function(tau) {
  b <- ws_w2426 %>% filter(fiscal_year == tau - 1L) %>% select(-fiscal_year)
  a <- ws_w2426 %>% filter(fiscal_year == tau) %>% select(-fiscal_year)
  inner_join(b, a, by = c("filer", "county_fips", "state_fips", "aewr_region_id", "soc"), suffix = c("0", "1")) %>%
    mutate(dw = wage1 - wage0) %>%
    left_join(r5_cty, by = c("county_fips", "state_fips", "aewr_region_id")) %>%
    left_join(st_sched %>% select(state_fips, dose_raw, dose_eff), by = "state_fips")
}

fc26 <- pair_fc(2026L)
fc25 <- pair_fc(2025L)

# Also merge onto county-level continuing offered wage change (FY2025-26 & FY2024-25)
cw26_diff <- cty_w26 %>%
  select(county_fips, state_fips, aewr_region_id, year, y_wage_full_fy, hhours_full_fy) %>%
  pivot_wider(names_from = year, values_from = c(y_wage_full_fy, hhours_full_fy)) %>%
  mutate(dw = y_wage_full_fy_2026 - y_wage_full_fy_2025, h0 = hhours_full_fy_2025) %>%
  left_join(r5_cty, by = c("county_fips", "state_fips", "aewr_region_id")) %>%
  left_join(st_sched %>% select(state_fips, dose_raw, dose_eff), by = "state_fips")

cw25_diff <- cty_w25 %>%
  select(county_fips, state_fips, aewr_region_id, year, y_wage_full_fy, hhours_full_fy) %>%
  pivot_wider(names_from = year, values_from = c(y_wage_full_fy, hhours_full_fy)) %>%
  mutate(dw = y_wage_full_fy_2025 - y_wage_full_fy_2024, h0 = hhours_full_fy_2024) %>%
  left_join(r5_cty, by = c("county_fips", "state_fips", "aewr_region_id")) %>%
  left_join(st_sched %>% select(state_fips, dose_raw, dose_eff), by = "state_fips")

r5_tests <- list(
  # Within-state (State FE): continuous dollar crossover gap (expected +pi if local wage props up offers!)
  list(id = "R5_fc_gap_oews_stfe",     d = fc26, dp = fc25, fml = "dw ~ gap_oews | state_fips",      term = "gap_oews",     wt = ~h0, desc = "Filer-Cty-SOC h0, State FE: OEWS crossover gap ($)"),
  list(id = "R5_fc_gap_p25_stfe",      d = fc26, dp = fc25, fml = "dw ~ gap_p25 | state_fips",       term = "gap_p25",      wt = ~h0, desc = "Filer-Cty-SOC h0, State FE: ACS p25 crossover gap ($)"),
  list(id = "R5_fc_gap_qwi_stfe",      d = fc26, dp = fc25, fml = "dw ~ gap_qwi | state_fips",       term = "gap_qwi",      wt = ~h0, desc = "Filer-Cty-SOC h0, State FE: QWI 111 crossover gap ($)"),
  # Within-state (State FE): binary crossover x dose_eff (expected negative/positive attenuation of pass-through)
  list(id = "R5_fc_cross_oews_stfe",   d = fc26, dp = fc25, fml = "dw ~ cross_oews:dose_eff + cross_oews | state_fips", term = "cross_oews:dose_eff", wt = ~h0, desc = "Filer-Cty-SOC h0, State FE: OEWS binary crossover x dose_eff"),
  list(id = "R5_fc_cross_p25_stfe",    d = fc26, dp = fc25, fml = "dw ~ cross_p25:dose_eff + cross_p25 | state_fips",   term = "cross_p25:dose_eff",  wt = ~h0, desc = "Filer-Cty-SOC h0, State FE: ACS p25 binary crossover x dose_eff"),
  list(id = "R5_fc_cross_qwi_stfe",    d = fc26, dp = fc25, fml = "dw ~ cross_qwi:dose_eff + cross_qwi | state_fips",   term = "cross_qwi:dose_eff",  wt = ~h0, desc = "Filer-Cty-SOC h0, State FE: QWI 111 binary crossover x dose_eff"),
  # Within-state (State FE): market-truncated cut mkt_cut (expected +pi if market wage truncates cut!)
  list(id = "R5_fc_mktcut_oews_stfe",  d = fc26, dp = fc25, fml = "dw ~ mkt_cut_oews | state_fips",  term = "mkt_cut_oews", wt = ~h0, desc = "Filer-Cty-SOC h0, State FE: OEWS market-truncated cut"),
  list(id = "R5_fc_mktcut_p25_stfe",   d = fc26, dp = fc25, fml = "dw ~ mkt_cut_p25 | state_fips",   term = "mkt_cut_p25",  wt = ~h0, desc = "Filer-Cty-SOC h0, State FE: ACS p25 market-truncated cut"),
  list(id = "R5_fc_mktcut_qwi_stfe",   d = fc26, dp = fc25, fml = "dw ~ mkt_cut_qwi | state_fips",   term = "mkt_cut_qwi",  wt = ~h0, desc = "Filer-Cty-SOC h0, State FE: QWI 111 market-truncated cut"),
  # Equal-weighted county panel (State FE)
  list(id = "R5_cty_gap_oews_stfe",    d = cw26_diff, dp = cw25_diff, fml = "dw ~ gap_oews | state_fips", term = "gap_oews", wt = NULL, desc = "County eq-wt, State FE: OEWS crossover gap ($)"),
  list(id = "R5_cty_gap_p25_stfe",     d = cw26_diff, dp = cw25_diff, fml = "dw ~ gap_p25 | state_fips",  term = "gap_p25",  wt = NULL, desc = "County eq-wt, State FE: ACS p25 crossover gap ($)"),
  list(id = "R5_cty_gap_qwi_stfe",     d = cw26_diff, dp = cw25_diff, fml = "dw ~ gap_qwi | state_fips",  term = "gap_qwi",  wt = NULL, desc = "County eq-wt, State FE: QWI 111 crossover gap ($)"),
  list(id = "R5_cty_mktcut_p25_stfe",  d = cw26_diff, dp = cw25_diff, fml = "dw ~ mkt_cut_p25 | state_fips", term = "mkt_cut_p25", wt = NULL, desc = "County eq-wt, State FE: ACS p25 market-truncated cut"),
  list(id = "R5_cty_mktcut_oews_stfe", d = cw26_diff, dp = cw25_diff, fml = "dw ~ mkt_cut_oews | state_fips", term = "mkt_cut_oews", wt = NULL, desc = "County eq-wt, State FE: OEWS market-truncated cut")
)

r5_rows <- lapply(r5_tests, function(tt) {
  f26 <- fit_cv3j_fixest(tt$d, tt$fml, tt$term, "state_fips", "feols", weights_fml = tt$wt)
  f25 <- fit_cv3j_fixest(tt$dp, tt$fml, tt$term, "state_fips", "feols", weights_fml = tt$wt)
  fstat <- if (f26$ok && f26$se_cv3j > 0) (f26$estimate / f26$se_cv3j)^2 else NA_real_
  pass_fs <- isTRUE(f26$ok) && is.finite(fstat) && fstat >= 10 && abs(f26$estimate) >= 0.25 && is.finite(f25$p_cv3j) && f25$p_cv3j > 0.05
  cat(sprintf("%-24s | pi=%.4f [CV3J %.4f], F=%.2f, p=%.3f | pl2425=%.4f (p=%.3f) | N=%d, G=%d | PASS_FS=%s\n",
    tt$id, f26$estimate, f26$se_cv3j, fstat, f26$p_cv3j, f25$estimate, f25$p_cv3j, f26$n_obs, f26$G, as.character(pass_fs)))
  tibble(
    spec_id = tt$id, desc = tt$desc, n_obs = f26$n_obs, G = f26$G,
    pi_hat = f26$estimate, se_cv3j = f26$se_cv3j, f_cv3j = fstat, p_cv3j = f26$p_cv3j,
    pl2425_est = f25$estimate, pl2425_se = f25$se_cv3j, pl2425_p = f25$p_cv3j,
    pass_fs_gate = pass_fs
  )
})
r5_df <- bind_rows(r5_rows)
write_csv(r5_df, "explorations/bite_power_rescue/r5_first_stage_audit.csv")

# Save R1 & R5 summary rows for phase1_triage.csv
r1_r5_triage <- bind_rows(
  tibble(
    candidate = "R1a_agg_hours_eff_dollar",
    family = "R1",
    estimand = "aggregate_effective_floor_hours_elasticity",
    unit = "county_fy",
    n_obs = 5842L,
    n_units = 2921L,
    us_hours_share = 0.785,
    G = 46L,
    G_eff = sup_r1a$g_eff,
    sd_x_tilde = sup_r1a$sd_x_tilde_wt,
    r2_fe = sup_r1a$r2_fe_wt,
    first_stage_pi = fs_cty_eff$estimate,
    first_stage_se = fs_cty_eff$se_cv3j,
    first_stage_F = (fs_cty_eff$estimate / fs_cty_eff$se_cv3j)^2,
    sy_ms_ratio = sy_pre_25$ms_ratio,
    n_placebo_draws = e_r1a_h$n_valid,
    emp_size_alpha05 = e_r1a_h$emp_size,
    size_wilson_hi = e_r1a_h$wilson_hi,
    mde_primary = e_r1a_h$mde_upper,
    mde_unit = "statutory_eff_hours_elasticity",
    mde_eps_cost = mde_eps_r1a_cty,
    mde_eps_cost_r7 = mde_eps_r1a_r7,
    placebo_est = pl_r1a_h$beta * mean_p_eff,
    placebo_se = pl_r1a_h$se_cv3j * mean_p_eff,
    placebo_p = 2 * pt(-abs(pl_r1a_h$beta / pl_r1a_h$se_cv3j), 45L),
    notes = sprintf("R7 pi=%.3f (F=%.1f), cty pi=%.3f (F=%.1f); unified state_real_minwages; 7/46 states truncated in 2026",
      fs_r7_eff$estimate, (fs_r7_eff$estimate/fs_r7_eff$se_cv3j)^2, fs_cty_eff$estimate, (fs_cty_eff$estimate/fs_cty_eff$se_cv3j)^2)
  ),
  tibble(
    candidate = "R1a_agg_hours_ln_eff",
    family = "R1",
    estimand = "aggregate_effective_floor_log_hours_elasticity",
    unit = "county_fy",
    n_obs = 5842L,
    n_units = 2921L,
    us_hours_share = 0.785,
    G = 46L,
    G_eff = sup_r1a_ln$g_eff,
    sd_x_tilde = sup_r1a_ln$sd_x_tilde_wt,
    r2_fe = sup_r1a_ln$r2_fe_wt,
    first_stage_pi = fs_cty_eff$estimate,
    first_stage_se = fs_cty_eff$se_cv3j,
    first_stage_F = (fs_cty_eff$estimate / fs_cty_eff$se_cv3j)^2,
    sy_ms_ratio = sy_pre_25$ms_ratio,
    n_placebo_draws = e_r1a_lh$n_valid,
    emp_size_alpha05 = e_r1a_lh$emp_size,
    size_wilson_hi = e_r1a_lh$wilson_hi,
    mde_primary = e_r1a_lh$mde_upper,
    mde_unit = "ln_eff_hours_elasticity",
    mde_eps_cost = mde_eps_r1a_lh,
    mde_eps_cost_r7 = e_r1a_lh$mde_upper * COST_RATIO_CENTRAL / fs_r7_eff$estimate,
    placebo_est = pl_r1a_ln_h$beta,
    placebo_se = pl_r1a_ln_h$se_cv3j,
    placebo_p = 2 * pt(-abs(pl_r1a_ln_h$beta / pl_r1a_ln_h$se_cv3j), 45L),
    notes = "Direct ln(max(AEWR_US1, mw_ag)) FE-PPML"
  ),
  tibble(
    candidate = "R1a_agg_cert_eff_dollar",
    family = "R1",
    estimand = "aggregate_effective_floor_positions_elasticity",
    unit = "county_fy",
    n_obs = 5842L,
    n_units = 2921L,
    us_hours_share = 0.785,
    G = 46L,
    G_eff = sup_r1a$g_eff,
    sd_x_tilde = sup_r1a$sd_x_tilde_wt,
    r2_fe = sup_r1a$r2_fe_wt,
    first_stage_pi = fs_cty_eff$estimate,
    first_stage_se = fs_cty_eff$se_cv3j,
    first_stage_F = (fs_cty_eff$estimate / fs_cty_eff$se_cv3j)^2,
    sy_ms_ratio = sy_pre_25$ms_ratio,
    n_placebo_draws = e_r1a_c$n_valid,
    emp_size_alpha05 = e_r1a_c$emp_size,
    size_wilson_hi = e_r1a_c$wilson_hi,
    mde_primary = e_r1a_c$mde_upper,
    mde_unit = "statutory_eff_positions_elasticity",
    mde_eps_cost = (e_r1a_c$mde_upper / mean_p_eff) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / pi_r1a_cty,
    mde_eps_cost_r7 = (e_r1a_c$mde_upper / mean_p_eff) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / pi_r1a_r7,
    placebo_est = pl_r1a_c$beta * mean_p_eff,
    placebo_se = pl_r1a_c$se_cv3j * mean_p_eff,
    placebo_p = 2 * pt(-abs(pl_r1a_c$beta / pl_r1a_c$se_cv3j), 45L),
    notes = "Effective floor on certified positions"
  ),
  tibble(
    candidate = "R1a_direct_grower_hours_eff",
    family = "R1",
    estimand = "direct_grower_effective_floor_hours_elasticity",
    unit = "county_fy_direct",
    n_obs = 5842L,
    n_units = 2921L,
    us_hours_share = 0.587,
    G = 46L,
    G_eff = sup_r1a$g_eff,
    sd_x_tilde = sup_r1a$sd_x_tilde_wt,
    r2_fe = sup_r1a$r2_fe_wt,
    first_stage_pi = fs_cty_eff$estimate,
    first_stage_se = fs_cty_eff$se_cv3j,
    first_stage_F = (fs_cty_eff$estimate / fs_cty_eff$se_cv3j)^2,
    sy_ms_ratio = NA_real_,
    n_placebo_draws = e_dg_eff_h$n_valid,
    emp_size_alpha05 = e_dg_eff_h$emp_size,
    size_wilson_hi = e_dg_eff_h$wilson_hi,
    mde_primary = e_dg_eff_h$mde_upper,
    mde_unit = "direct_grower_eff_hours_elasticity",
    mde_eps_cost = (e_dg_eff_h$mde_upper / mean_p_eff) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / pi_r1a_cty,
    mde_eps_cost_r7 = (e_dg_eff_h$mde_upper / mean_p_eff) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / pi_r1a_r7,
    placebo_est = pl_dg_r1a_h$beta * mean_p_eff,
    placebo_se = pl_dg_r1a_h$se_cv3j * mean_p_eff,
    placebo_p = 2 * pt(-abs(pl_dg_r1a_h$beta / pl_dg_r1a_h$se_cv3j), 45L),
    notes = sprintf("Direct-grower hours calibrated on FY2020-25 (600 draws); ln_eff MDE=%.3f; raw AEWR MDE=%.3f", e_dg_ln_h$mde_upper, e_dg_raw_h$mde_upper)
  ),
  tibble(
    candidate = "R1a_direct_grower_cert_eff",
    family = "R1",
    estimand = "direct_grower_effective_floor_positions_elasticity",
    unit = "county_fy_direct",
    n_obs = 5842L,
    n_units = 2921L,
    us_hours_share = 0.587,
    G = 46L,
    G_eff = sup_r1a$g_eff,
    sd_x_tilde = sup_r1a$sd_x_tilde_wt,
    r2_fe = sup_r1a$r2_fe_wt,
    first_stage_pi = fs_cty_eff$estimate,
    first_stage_se = fs_cty_eff$se_cv3j,
    first_stage_F = (fs_cty_eff$estimate / fs_cty_eff$se_cv3j)^2,
    sy_ms_ratio = NA_real_,
    n_placebo_draws = e_dg_eff_c$n_valid,
    emp_size_alpha05 = e_dg_eff_c$emp_size,
    size_wilson_hi = e_dg_eff_c$wilson_hi,
    mde_primary = e_dg_eff_c$mde_upper,
    mde_unit = "direct_grower_eff_positions_elasticity",
    mde_eps_cost = (e_dg_eff_c$mde_upper / mean_p_eff) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / pi_r1a_cty,
    mde_eps_cost_r7 = (e_dg_eff_c$mde_upper / mean_p_eff) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / pi_r1a_r7,
    placebo_est = pl_dg_r1a_c$beta * mean_p_eff,
    placebo_se = pl_dg_r1a_c$se_cv3j * mean_p_eff,
    placebo_p = 2 * pt(-abs(pl_dg_r1a_c$beta / pl_dg_r1a_c$se_cv3j), 45L),
    notes = "Direct-grower positions calibrated on FY2020-25 (600 draws)"
  ),
  tibble(
    candidate = "R1b_pure_truncation_gap_hours",
    family = "R1",
    estimand = "pure_minwage_truncation_bite_hours_elasticity",
    unit = "county_fy",
    n_obs = 5842L,
    n_units = 2921L,
    us_hours_share = 0.785,
    G = 46L,
    G_eff = sup_r1b$g_eff,
    sd_x_tilde = sup_r1b$sd_x_tilde_wt,
    r2_fe = sup_r1b$r2_fe_wt,
    first_stage_pi = fs_r7_trunc$estimate,
    first_stage_se = fs_r7_trunc$se_cv3j,
    first_stage_F = (fs_r7_trunc$estimate / fs_r7_trunc$se_cv3j)^2,
    sy_ms_ratio = sy_pre_25$ms_ratio,
    n_placebo_draws = e_r1b_h$n_valid,
    emp_size_alpha05 = e_r1b_h$emp_size,
    size_wilson_hi = e_r1b_h$wilson_hi,
    mde_primary = e_r1b_h$mde_upper,
    mde_unit = "truncation_hours_elasticity",
    mde_eps_cost = ifelse(is.finite(fs_r7_trunc$estimate) & abs(fs_r7_trunc$estimate) > 0.05,
      (e_r1b_h$mde_upper / mean_p_eff) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / abs(fs_r7_trunc$estimate), NA_real_),
    mde_eps_cost_r7 = NA_real_,
    placebo_est = pl_r1b_h$beta * mean_p_eff,
    placebo_se = pl_r1b_h$se_cv3j * mean_p_eff,
    placebo_p = 2 * pt(-abs(pl_r1b_h$beta / pl_r1b_h$se_cv3j), 45L),
    notes = sprintf("Only 7/46 states have positive trunc_gap; cty FS pi=%.3f (F=%.2f)", fs_cty_trunc$estimate, (fs_cty_trunc$estimate/fs_cty_trunc$se_cv3j)^2)
  ),
  tibble(
    candidate = "R1c_substate_tiers_within_state",
    family = "R1",
    estimand = "within_state_substate_minwage_tier_elasticity",
    unit = "county_fy",
    n_obs = 5842L,
    n_units = 2921L,
    us_hours_share = sum(r1c_var_states$hours_2025) / tot_h25,
    G = nrow(r1c_var_states),
    G_eff = sup_r1c$g_eff,
    sd_x_tilde = sup_r1c$sd_x_tilde_wt,
    r2_fe = sup_r1c$r2_fe_wt,
    first_stage_pi = fs_cty_r1c_apr$estimate,
    first_stage_se = fs_cty_r1c_apr$se_cv3j,
    first_stage_F = if (isTRUE(fs_cty_r1c_apr$ok) && fs_cty_r1c_apr$se_cv3j > 0) (fs_cty_r1c_apr$estimate/fs_cty_r1c_apr$se_cv3j)^2 else NA_real_,
    sy_ms_ratio = 1.0,
    n_placebo_draws = 0L,
    emp_size_alpha05 = NA_real_,
    size_wilson_hi = NA_real_,
    mde_primary = NA_real_,
    mde_unit = "within_state_tier_elasticity",
    mde_eps_cost = NA_real_,
    mde_eps_cost_r7 = NA_real_,
    placebo_est = NA_real_,
    placebo_se = NA_real_,
    placebo_p = NA_real_,
    notes = sprintf("KILLED IN PHASE 1: only %d states (NY, OR) have sub-state ag min-wage variation; G_eff=%.2f", nrow(r1c_var_states), sup_r1c$g_eff)
  )
)

# Check if ANY R5 spec passed the first-stage gate; if yes, calibrate its quantity power; if no, record clean Phase 1 FS kill
best_r5 <- r5_df %>% arrange(desc(f_cv3j)) %>% slice(1L)
r5_triage_row <- tibble(
  candidate = paste0("R5_best_", best_r5$spec_id),
  family = "R5",
  estimand = "within_state_local_wage_crossover_bite",
  unit = "county_or_filer_fy",
  n_obs = as.integer(best_r5$n_obs),
  n_units = NA_integer_,
  us_hours_share = 0.785,
  G = as.integer(best_r5$G),
  G_eff = NA_real_,
  sd_x_tilde = NA_real_,
  r2_fe = NA_real_,
  first_stage_pi = best_r5$pi_hat,
  first_stage_se = best_r5$se_cv3j,
  first_stage_F = best_r5$f_cv3j,
  sy_ms_ratio = 1.0,
  n_placebo_draws = 0L,
  emp_size_alpha05 = NA_real_,
  size_wilson_hi = NA_real_,
  mde_primary = NA_real_,
  mde_unit = "crossover_differential_elasticity",
  mde_eps_cost = NA_real_,
  mde_eps_cost_r7 = NA_real_,
  placebo_est = best_r5$pl2425_est,
  placebo_se = best_r5$pl2425_se,
  placebo_p = best_r5$pl2425_p,
  notes = sprintf("Best of 14 R5 first-stage specs (%s): pi=%.4f, F=%.2f, pass_fs=%s", best_r5$spec_id, best_r5$pi_hat, best_r5$f_cv3j, as.character(best_r5$pass_fs_gate))
)

write_csv(bind_rows(r1_r5_triage, r5_triage_row), "explorations/bite_power_rescue/phase1_r1_r5_triage.csv")
cat("\nWrote explorations/bite_power_rescue/phase1_r1_r5_triage.csv and r5_first_stage_audit.csv\n")
