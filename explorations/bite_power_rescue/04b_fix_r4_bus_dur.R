# Corrected audit for R4a Business-Level Duration Bite (fixing dplyr summarise shadowing bug in coh_bin)
# + Referee diagnostics on R4a Bite H-2A Duration (2011-2025): 1a-PPML, hours share, structural eps_cost scaling,
#   and E_p25 benchmark on the identical 1,623 baseline-H2A county subsample!

source("explorations/bite_power_rescue/helpers_phase1.R")

FIXED_COST_F <- 2500.0
std_within <- function(x, grp) {
  out <- rep(NA_real_, length(x))
  for (g in unique(grp[!is.na(grp)])) {
    idx <- which(grp == g & is.finite(x))
    if (length(idx) >= 2L) {
      s <- stats::sd(x[idx])
      if (is.finite(s) && s > 0) out[idx] <- (x[idx] - mean(x[idx])) / s
    }
  }
  out
}

cat("=== 1. Referee Check on R4a Bite H-2A Duration (2011-2025) ===\n")
bp_full <- read_parquet(path_processed("bite_county_year_panel.parquet")) %>% mutate(year = as.integer(year))
bp_est  <- bp_full %>% filter(year >= 2011L, year <= 2025L)

ws_0810_dur <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, county_fips_list, adjusted_nbr_workers_certified, period_begin_date, period_end_date) %>%
  filter(fiscal_year %in% c(2008L, 2009L, 2010L)) %>%
  collect() %>%
  mutate(
    weeks = as.numeric(as.Date(period_end_date) - as.Date(period_begin_date)) / 7,
    workers = as.numeric(adjusted_nbr_workers_certified),
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    nc = stringr::str_count(county_fips_list, ",") + 1L
  ) %>%
  filter(is.finite(weeks), weeks > 0, weeks <= 60, is.finite(workers), workers > 0) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = harmonize_county_fips_2010(trimws(county_fips_list)), w = workers / nc) %>%
  group_by(county_fips) %>%
  summarise(mean_weeks_0810 = sum(weeks * w) / sum(w), .groups = "drop")

aewr_0810 <- bp_full %>% filter(year %in% c(2008L, 2010L)) %>%
  group_by(county_fips, aewr_region_id) %>%
  summarise(aewr_0810 = mean(aewr, na.rm = TRUE), .groups = "drop") %>%
  left_join(ws_0810_dur, by = "county_fips") %>%
  mutate(
    s_wage_0810 = ifelse(is.finite(mean_weeks_0810), (aewr_0810 * 40 * mean_weeks_0810) / (aewr_0810 * 40 * mean_weeks_0810 + FIXED_COST_F), NA_real_),
    E_dur_h2a_0810 = std_within(s_wage_0810, aewr_region_id)
  )

bp_sub <- bp_est %>%
  left_join(aewr_0810 %>% select(county_fips, s_wage_0810, E_dur_h2a_0810), by = "county_fips") %>%
  filter(is.finite(E_dur_h2a_0810)) %>%
  mutate(x_dur_h2a = E_dur_h2a_0810 * ln_aewr, x_swage_raw = s_wage_0810 * ln_aewr)

us_sh_1623 <- sum(bp_sub$man_hours_certified_start_year) / sum(bp_est$man_hours_certified_start_year)
sd_swage_wt <- sqrt(sum(bp_sub$man_hours_certified_start_year * (bp_sub$s_wage_0810 - weighted.mean(bp_sub$s_wage_0810, bp_sub$man_hours_certified_start_year))^2) / sum(bp_sub$man_hours_certified_start_year))
cat(sprintf("1,623 baseline-H2A counties cover %.1f%% of 2011-2025 US H-2A hours; hours-weighted SD(s_wage)=%.4f (unw=%.4f)\n",
  100 * us_sh_1623, sd_swage_wt, sd(aewr_0810$s_wage_0810, na.rm = TRUE)))

# Check 1a-PPML (+ E_dur x Year), raw s_wage structural cost-elasticity MDE, AND E_p25 on the SAME 1,623 counties!
cal_dur_1a   <- calibrate_multiperiod_ppml_blind(bp_sub, "man_hours_certified_start_year", "x_dur_h2a", "county_fips + aewr_region_id^year", ctrl_rhs = "i(year, E_dur_h2a_0810, ref = 2011)")
cal_dur_raw  <- calibrate_multiperiod_ppml_blind(bp_sub, "man_hours_certified_start_year", "x_swage_raw", "county_fips + aewr_region_id^year")
cal_ep25_sub <- calibrate_multiperiod_ppml_blind(bp_sub, "man_hours_certified_start_year", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year")
cal_dur_pos  <- calibrate_multiperiod_ppml_blind(bp_sub, "nbr_workers_certified_start_year", "x_dur_h2a", "county_fips + aewr_region_id^year")

cat(sprintf("R4a Dur 1a-PPML (+E_dur x Yr) : G_eff=%.2f, SD(X~)=%.4f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_dur_1a$g_eff, cal_dur_1a$sd_x_tilde_wt, cal_dur_1a$emp_size, cal_dur_1a$mde_1sd, cal_dur_1a$mde_2sd))
cat(sprintf("R4a Dur Raw s_wage (eps_cost!): G_eff=%.2f, SD(X~)=%.4f, size=%.3f, MDE_eps_cost=%.3f\n",
  cal_dur_raw$g_eff, cal_dur_raw$sd_x_tilde_wt, cal_dur_raw$emp_size, cal_dur_raw$mde_1sd))
cat(sprintf("R4a Dur Positions (1b-PPML)   : G_eff=%.2f, SD(X~)=%.4f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_dur_pos$g_eff, cal_dur_pos$sd_x_tilde_wt, cal_dur_pos$emp_size, cal_dur_pos$mde_1sd, cal_dur_pos$mde_2sd))
cat(sprintf("Comparator E_p25 on SAME 1623 : G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_ep25_sub$g_eff, cal_ep25_sub$sd_x_tilde_wt, cal_ep25_sub$sy_ms_ratio, cal_ep25_sub$emp_size, cal_ep25_sub$mde_1sd, cal_ep25_sub$mde_2sd))

cat("\n=== 2. Fixed R4a Business-Level Duration Bite (State x FY FE, FY2020-2025) ===\n")
qp_pre <- read_parquet(path_processed("policy_2025", "quantity_county_fy.parquet")) %>% mutate(year = as.integer(year)) %>% filter(year <= 2025L)
st_sched_46 <- read_parquet(path_processed("policy_2025", "quantity_county_fy.parquet"),
  col_select = c("state_fips", "aewr_region_id", "year", "x_us_skill_1")) %>%
  filter(year %in% c(2025L, 2026L)) %>% distinct() %>%
  pivot_wider(names_from = year, values_from = x_us_skill_1, names_prefix = "x_") %>%
  mutate(dose_raw = x_2026 - x_2025) %>% arrange(state_fips)
mean_p_46 <- mean(c(st_sched_46$x_2025, st_sched_46$x_2026))
t1_cty_ids <- qp_pre %>% filter(year == 2025L) %>% select(county_fips, state_fips_t1 = state_fips, aewr_region_id)

disc_2025 <- open_dataset(path_int("h2a_disclosure_records.parquet")) %>%
  select(fiscal_year, case_number, decision_date, h2a_labor_contractor) %>%
  filter(fiscal_year >= 2020L, fiscal_year <= 2025L) %>% collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year)) %>%
  group_by(case_number, fiscal_year) %>%
  summarise(dec = as.Date(substr(first(na.omit(as.character(decision_date))), 1L, 10L)),
            lc = first(na.omit(as.character(h2a_labor_contractor))), .groups = "drop") %>%
  filter(!is.na(dec), dec >= as.Date(sprintf("%d-10-01", fiscal_year - 1L)), dec <= as.Date(sprintf("%d-09-30", fiscal_year))) %>%
  mutate(is_h2alc = toupper(trimws(coalesce(lc, ""))) %in% c("Y", "YES", "TRUE", "1"))

ws_bus_dur <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, case_number, county_fips_list, adjusted_nbr_workers_certified, period_begin_date, period_end_date, number_of_hours, employer_is_worksite_business, employer_id_balanced) %>%
  filter(fiscal_year >= 2020L, fiscal_year <= 2025L) %>% collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year), begin = as.Date(period_begin_date), end = as.Date(period_end_date)) %>%
  filter(!is.na(begin), !is.na(end)) %>%
  inner_join(disc_2025 %>% filter(!is_h2alc) %>% select(case_number, fiscal_year), by = c("case_number", "fiscal_year")) %>%
  filter(!is.na(employer_is_worksite_business), as.logical(employer_is_worksite_business), !is.na(employer_id_balanced), nzchar(trimws(as.character(employer_id_balanced)))) %>%
  mutate(
    bus_id = trimws(as.character(employer_id_balanced)),
    weeks = pmax(as.numeric(end - begin) / 7, 0),
    nh = coalesce(suppressWarnings(as.numeric(number_of_hours)), 40),
    workers = as.numeric(adjusted_nbr_workers_certified),
    hours = weeks * nh * workers,
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    nc = stringr::str_count(county_fips_list, ",") + 1L
  ) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = harmonize_county_fips_2010(trimws(county_fips_list)), h = hours / nc, w = workers / nc) %>%
  inner_join(t1_cty_ids, by = "county_fips") %>%
  group_by(bus_id, state_fips = state_fips_t1, aewr_region_id, year = fiscal_year) %>%
  summarise(hours = sum(h, na.rm = TRUE), cert = sum(w, na.rm = TRUE), .groups = "drop")

set.seed(SEED_PHASE1 + 310L)
b_1b <- se_1b <- b_1a <- se_1a <- numeric(500L)
pl_1b <- pl_1a <- sup_1b <- sup_1a <- NULL
idx <- 0L
for (yr in 2021L:2025L) {
  b0 <- ws_bus_dur %>%
    filter(year == yr - 1L, hours > 0, cert > 0) %>%
    group_by(bus_id) %>%
    arrange(desc(hours), state_fips, .by_group = TRUE) %>%
    summarise(state_fips = first(state_fips), aewr_region_id = first(aewr_region_id), h0 = sum(hours), hpw0 = sum(hours) / sum(cert), .groups = "drop") %>%
    left_join(st_sched_46 %>% select(state_fips, x_2025), by = "state_fips") %>%
    mutate(s_wage = (x_2025 * hpw0) / (x_2025 * hpw0 + FIXED_COST_F), E_dur = std_within(s_wage, state_fips)) %>%
    filter(is.finite(E_dur))
  b1 <- ws_bus_dur %>% filter(year == yr) %>% group_by(bus_id) %>% summarise(h1 = sum(hours), .groups = "drop")
  coh <- b0 %>% left_join(b1, by = "bus_id") %>% mutate(h1 = coalesce(h1, 0), w_tot = h0 + h1)
  # FIXED binning: compute weighted.mean(E_dur, w_tot) BEFORE shadowing h0, h1!
  coh_bin <- coh %>%
    group_by(state_fips) %>%
    mutate(ebin = ntile(E_dur, 20L)) %>%
    group_by(state_fips, ebin) %>%
    summarise(E_dur_m = sum(E_dur * w_tot) / sum(w_tot), h0_s = sum(h0), h1_s = sum(h1), .groups = "drop") %>%
    rename(E_dur = E_dur_m, h0 = h0_s, h1 = h1_s)

  if (yr == 2025L) {
    dx_full <- coh$E_dur * st_sched_46$dose_raw[match(coh$state_fips, st_sched_46$state_fips)]
    dx_bin  <- coh_bin$E_dur * st_sched_46$dose_raw[match(coh_bin$state_fips, st_sched_46$state_fips)]
    sup_1b <- compute_2period_blind_support(coh$h0, coh$h1, dx_full, coh$state_fips, coh$state_fips)
    sup_1a <- compute_2period_blind_support(coh$h0, coh$h1, dx_full, coh$state_fips, coh$state_fips, Z = matrix(coh$E_dur, ncol = 1L))
    pl_1b  <- fit_2period_ppml_cv3j(coh$h0, coh$h1, dx_full, coh$state_fips, coh$state_fips)
    pl_1a  <- fit_2period_ppml_cv3j(coh$h0, coh$h1, dx_full, coh$state_fips, coh$state_fips, Z = matrix(coh$E_dur, ncol = 1L))
    pl_bin <- fit_2period_ppml_cv3j(coh_bin$h0, coh_bin$h1, dx_bin, coh_bin$state_fips, coh_bin$state_fips, Z = matrix(coh_bin$E_dur, ncol = 1L))
    cat(sprintf("Verify bin vs unbinned pl2425 (+Emain): full=%.4f [%.4f] vs bin=%.4f [%.4f]\n",
      pl_1a$beta * mean_p_46, pl_1a$se_cv3j * mean_p_46, pl_bin$beta * mean_p_46, pl_bin$se_cv3j * mean_p_46))
  }
  for (k in seq_len(100L)) {
    idx <- idx + 1L
    perm <- st_sched_46 %>% group_by(aewr_region_id) %>% mutate(d = dose_raw[sample.int(n())]) %>% ungroup()
    dx_p <- coh_bin$E_dur * perm$d[match(coh_bin$state_fips, perm$state_fips)]
    r1 <- fit_2period_ppml_cv3j(coh_bin$h0, coh_bin$h1, dx_p, coh_bin$state_fips, coh_bin$state_fips)
    r2 <- fit_2period_ppml_cv3j(coh_bin$h0, coh_bin$h1, dx_p, coh_bin$state_fips, coh_bin$state_fips, Z = matrix(coh_bin$E_dur, ncol = 1L))
    b_1b[[idx]] <- r1$beta * mean_p_46; se_1b[[idx]] <- r1$se_cv3j * mean_p_46
    b_1a[[idx]] <- r2$beta * mean_p_46; se_1a[[idx]] <- r2$se_cv3j * mean_p_46
  }
}
e_1b <- compute_empirical_mde(b_1b, se_1b, 45L)
e_1a <- compute_empirical_mde(b_1a, se_1a, 45L)
cat(sprintf("FIXED R4a Bus-Dur (StFE,noEmain): G_eff=%.2f, SD(X~)=%.3f, size=%.3f [%.3f,%.3f], MDE_1SD=%.3f, MDE_2SD=%.3f, pl2425=%.3f [%.3f] (p=%.3f)\n",
  sup_1b$g_eff, sup_1b$sd_x_tilde_wt, e_1b$emp_size, e_1b$wilson_lo, e_1b$wilson_hi,
  e_1b$mde_upper, 2 * e_1b$mde_upper, pl_1b$beta * mean_p_46, pl_1b$se_cv3j * mean_p_46, 2*pt(-abs(pl_1b$beta/pl_1b$se_cv3j), 45L)))
cat(sprintf("FIXED R4a Bus-Dur (StFE,+Emain) : G_eff=%.2f, SD(X~)=%.3f, size=%.3f [%.3f,%.3f], MDE_1SD=%.3f, MDE_2SD=%.3f, pl2425=%.3f [%.3f] (p=%.3f)\n",
  sup_1a$g_eff, sup_1a$sd_x_tilde_wt, e_1a$emp_size, e_1a$wilson_lo, e_1a$wilson_hi,
  e_1a$mde_upper, 2 * e_1a$mde_upper, pl_1a$beta * mean_p_46, pl_1a$se_cv3j * mean_p_46, 2*pt(-abs(pl_1a$beta/pl_1a$se_cv3j), 45L)))

# Patch the two corrected business-duration rows in phase1_r4_r6_triage.csv + append the referee rows
r46 <- read_csv("explorations/bite_power_rescue/phase1_r4_r6_triage.csv", show_col_types = FALSE) %>%
  filter(!candidate %in% c("R4a_2025_bus_dur_stfe_1b", "R4a_2025_bus_dur_stfe_1a"))

add_rows <- bind_rows(
  tibble(
    candidate = "R4a_2025_bus_dur_stfe_1b", family = "R4", estimand = "business_duration_amortization_2025_stfe_1b", unit = "business_fy",
    n_obs = as.integer(2L * nrow(coh)), n_units = as.integer(nrow(coh)), us_hours_share = 0.456, G = 46L,
    G_eff = sup_1b$g_eff, sd_x_tilde = sup_1b$sd_x_tilde_wt, r2_fe = sup_1b$r2_fe_wt,
    first_stage_pi = NA_real_, first_stage_se = NA_real_, first_stage_F = NA_real_, sy_ms_ratio = 1.0,
    n_placebo_draws = e_1b$n_valid, emp_size_alpha05 = e_1b$emp_size, size_wilson_hi = e_1b$wilson_hi,
    mde_primary = 2 * e_1b$mde_upper, mde_unit = "2SD_differential_bite_elasticity",
    mde_eps_cost = NA_real_, mde_eps_cost_r7 = NA_real_,
    placebo_est = pl_1b$beta * mean_p_46, placebo_se = pl_1b$se_cv3j * mean_p_46,
    placebo_p = 2 * pt(-abs(pl_1b$beta / pl_1b$se_cv3j), 45L),
    notes = sprintf("1SD_MDE=%.3f; direct-grower business duration x dose, State x FY FE (no E main)", e_1b$mde_upper)
  ),
  tibble(
    candidate = "R4a_2025_bus_dur_stfe_1a", family = "R4", estimand = "business_duration_amortization_2025_stfe_1a", unit = "business_fy",
    n_obs = as.integer(2L * nrow(coh)), n_units = as.integer(nrow(coh)), us_hours_share = 0.456, G = 46L,
    G_eff = sup_1a$g_eff, sd_x_tilde = sup_1a$sd_x_tilde_wt, r2_fe = sup_1a$r2_fe_wt,
    first_stage_pi = NA_real_, first_stage_se = NA_real_, first_stage_F = NA_real_, sy_ms_ratio = 1.0,
    n_placebo_draws = e_1a$n_valid, emp_size_alpha05 = e_1a$emp_size, size_wilson_hi = e_1a$wilson_hi,
    mde_primary = 2 * e_1a$mde_upper, mde_unit = "2SD_differential_bite_elasticity",
    mde_eps_cost = NA_real_, mde_eps_cost_r7 = NA_real_,
    placebo_est = pl_1a$beta * mean_p_46, placebo_se = pl_1a$se_cv3j * mean_p_46,
    placebo_p = 2 * pt(-abs(pl_1a$beta / pl_1a$se_cv3j), 45L),
    notes = sprintf("1SD_MDE=%.3f; direct-grower business duration x dose + E main, State x FY FE", e_1a$mde_upper)
  ),
  tibble(
    candidate = "R4a_bite_h2a_duration_1a", family = "R4", estimand = "duration_amortization_bite_2011_25_1a", unit = "county_year_bite",
    n_obs = as.integer(cal_dur_1a$n_obs), n_units = as.integer(cal_dur_1a$n_counties), us_hours_share = us_sh_1623,
    G = 17L, G_eff = cal_dur_1a$g_eff, sd_x_tilde = cal_dur_1a$sd_x_tilde_wt, r2_fe = cal_dur_1a$r2_x_wt,
    first_stage_pi = NA_real_, first_stage_se = NA_real_, first_stage_F = NA_real_, sy_ms_ratio = cal_dur_1a$sy_ms_ratio,
    n_placebo_draws = 999L, emp_size_alpha05 = cal_dur_1a$emp_size, size_wilson_hi = cal_dur_1a$wilson_hi,
    mde_primary = cal_dur_1a$mde_1sd, mde_unit = "1SD_1a_PPML_elasticity",
    mde_eps_cost = cal_dur_raw$mde_1sd, mde_eps_cost_r7 = NA_real_,
    placebo_est = NA_real_, placebo_se = NA_real_, placebo_p = NA_real_,
    notes = sprintf("1a-PPML (+E_dur x Yr); raw s_wage structural eps_cost MDE=%.2f; E_p25 on SAME 1623 counties: 1SD=%.3f / 2SD=%.3f",
      cal_dur_raw$mde_1sd, cal_ep25_sub$mde_1sd, cal_ep25_sub$mde_2sd)
  )
)

write_csv(bind_rows(r46, add_rows), "explorations/bite_power_rescue/phase1_r4_r6_triage.csv")
cat("Updated explorations/bite_power_rescue/phase1_r4_r6_triage.csv\n")
