# Phase 1 Strictly Outcome-Blind Audit: Candidate R4 (Structural Cost-Share Bite:
# Contract-Duration Amortization & Farm Labor-Cost Share) and Candidate R6 (Stacked
# 2009-10 OES-to-FLS Reversal + 2025 Statewide Cut on Certified Positions).
# HARD RULE: NEVER reads or regresses FY2026 H-2A quantity outcomes!

source("explorations/bite_power_rescue/helpers_phase1.R")

cat("======================================================================\n")
cat("CANDIDATE R4: Structural Cost-Share Bite (Duration Amortization & Labor Share)\n")
cat("======================================================================\n")

# --- Load base panels (strictly masking FY2026 quantities!) ---
bp_full <- read_parquet(path_processed("bite_county_year_panel.parquet")) %>%
  mutate(year = as.integer(year))
bp_est <- bp_full %>% filter(year >= 2011L, year <= 2025L)

qp_pre <- read_parquet(path_processed("policy_2025", "quantity_county_fy.parquet")) %>%
  mutate(year = as.integer(year)) %>%
  filter(year <= 2025L)

st_sched_46 <- read_parquet(
  path_processed("policy_2025", "quantity_county_fy.parquet"),
  col_select = c("state_fips", "aewr_region_id", "year", "x_us_skill_1", "mw_ag")
) %>%
  filter(year %in% c(2025L, 2026L)) %>%
  distinct() %>%
  pivot_wider(names_from = year, values_from = c(x_us_skill_1, mw_ag)) %>%
  mutate(dose_raw = x_us_skill_1_2026 - x_us_skill_1_2025) %>%
  arrange(state_fips)
mean_p_46 <- mean(c(st_sched_46$x_us_skill_1_2025, st_sched_46$x_us_skill_1_2026))

# Helper: standardize within group
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

# --- 4.1 Build Frozen Predetermined Duration-Amortization & Labor-Cost-Share Exposures ---
# Fixed cost F = $2,500 per worker per contract (transport + visa + filing + housing setup).
# Cash wage share s_wage = (w * H_contract) / (w * H_contract + F), where H_contract = 40 * weeks.
FIXED_COST_F <- 2500.0

# (A) For 2011-2025 bite:
#   - H-2A baseline duration (2008-2010 mean contract weeks per worker from h2a_allocated_worksites)
#   - Census of Ag 2007 hired-worker >=150 days share (full-county coverage!)
#   - BEA 2008-2010 farm labor-cost share (share_farm_laborexp_prodexp, 3,041 counties!)
ws_0810_dur <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, county_fips_list, adjusted_nbr_workers_certified, period_begin_date, period_end_date) %>%
  filter(fiscal_year %in% c(2008L, 2009L, 2010L)) %>%
  collect() %>%
  mutate(
    begin = as.Date(period_begin_date),
    end = as.Date(period_end_date),
    weeks = as.numeric(end - begin) / 7,
    workers = as.numeric(adjusted_nbr_workers_certified),
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    nc = stringr::str_count(county_fips_list, ",") + 1L
  ) %>%
  filter(is.finite(weeks), weeks > 0, weeks <= 60, is.finite(workers), workers > 0) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = harmonize_county_fips_2010(trimws(county_fips_list)), w = workers / nc) %>%
  group_by(county_fips) %>%
  summarise(mean_weeks_0810 = sum(weeks * w) / sum(w), .groups = "drop")

aewr_0810 <- bp_full %>%
  filter(year %in% c(2008L, 2010L)) %>%
  group_by(county_fips, aewr_region_id, state_fips) %>%
  summarise(
    aewr_0810 = mean(aewr, na.rm = TRUE),
    labsh_0810 = mean(share_farm_laborexp_prodexp, na.rm = TRUE),
    .groups = "drop"
  )

cens_dur <- read_parquet(path_int("census_ag_hired_worker_duration_county.parquet")) %>%
  mutate(county_fips = harmonize_county_fips_2010(county_fips), year = as.integer(year)) %>%
  filter(year == 2007L) %>%
  transmute(county_fips, cens_150p_2007 = as.numeric(census_hired_worker_150_plus_share))

r4_bite_cty <- aewr_0810 %>%
  left_join(ws_0810_dur, by = "county_fips") %>%
  left_join(cens_dur, by = "county_fips") %>%
  mutate(
    # Cash wage share in total H-2A contract cost
    s_wage_0810 = ifelse(is.finite(mean_weeks_0810), (aewr_0810 * 40 * mean_weeks_0810) / (aewr_0810 * 40 * mean_weeks_0810 + FIXED_COST_F), NA_real_),
    E_dur_h2a_0810 = std_within(s_wage_0810, aewr_region_id),
    E_dur_cens_07  = std_within(cens_150p_2007, aewr_region_id),
    E_labsh_0810   = std_within(labsh_0810, aewr_region_id)
  )

cat(sprintf("R4 Bite 2008-10 exposures: H2A dur finite=%d (SD raw s_wage=%.3f), Census 150+ finite=%d, BEA labsh finite=%d (SD raw=%.3f)\n",
  sum(is.finite(r4_bite_cty$E_dur_h2a_0810)), sd(r4_bite_cty$s_wage_0810, na.rm = TRUE),
  sum(is.finite(r4_bite_cty$E_dur_cens_07)), sum(is.finite(r4_bite_cty$E_labsh_0810)), sd(r4_bite_cty$labsh_0810, na.rm = TRUE)))

bp_r4 <- bp_est %>%
  left_join(r4_bite_cty %>% select(county_fips, E_dur_h2a_0810, E_dur_cens_07, E_labsh_0810), by = "county_fips") %>%
  mutate(
    x_dur_h2a = E_dur_h2a_0810 * ln_aewr,
    x_dur_cens = E_dur_cens_07 * ln_aewr,
    x_labsh = E_labsh_0810 * ln_aewr
  )

# Calibrate R4 on 2011-2025 Bite (FE-only, strictly blind!) under BOTH Region x Year AND State x Year FE
cal_r4_dur_ry  <- calibrate_multiperiod_ppml_blind(bp_r4, "man_hours_certified_start_year", "x_dur_h2a", "county_fips + aewr_region_id^year")
cal_r4_dur_sy  <- calibrate_multiperiod_ppml_blind(bp_r4, "man_hours_certified_start_year", "x_dur_h2a", "county_fips + state_fips^year")
cal_r4_cens_ry <- calibrate_multiperiod_ppml_blind(bp_r4, "man_hours_certified_start_year", "x_dur_cens", "county_fips + aewr_region_id^year")
cal_r4_cens_sy <- calibrate_multiperiod_ppml_blind(bp_r4, "man_hours_certified_start_year", "x_dur_cens", "county_fips + state_fips^year")
cal_r4_lab_ry  <- calibrate_multiperiod_ppml_blind(bp_r4, "man_hours_certified_start_year", "x_labsh", "county_fips + aewr_region_id^year")
cal_r4_lab_sy  <- calibrate_multiperiod_ppml_blind(bp_r4, "man_hours_certified_start_year", "x_labsh", "county_fips + state_fips^year")

# Also check pre-estimation 2008-2010 vs 2011 pre-trend F-test on 2008-2011 data ONLY (strictly pre-2012!)
bp_0811 <- bp_full %>%
  filter(year %in% 2008L:2011L) %>%
  left_join(r4_bite_cty %>% select(county_fips, E_dur_h2a_0810, E_dur_cens_07, E_labsh_0810), by = "county_fips")
check_pretrend_0811 <- function(e_col) {
  d_sub <- bp_0811 %>% filter(is.finite(.data[[e_col]]))
  fml <- stats::as.formula(paste0("nbr_workers_certified_start_year ~ i(year, ", e_col, ", ref = 2011) | county_fips + aewr_region_id^year"))
  m <- fixest::fepois(fml, data = d_sub, vcov = ~aewr_region_id, notes = FALSE)
  # Use year 2008 vs 2011 t-stat & joint CRV1 Wald on 2008-2010
  trms <- paste0("year::", c(2008L, 2009L, 2010L), ":", e_col)
  b <- coef(m)[trms]; V <- vcov(m)[trms, trms, drop = FALSE]
  fval <- as.numeric(t(b) %*% solve(V, b)) / 3
  pf(fval, 3L, 16L, lower.tail = FALSE)
}
p_pre_dur  <- check_pretrend_0811("E_dur_h2a_0810")
p_pre_cens <- check_pretrend_0811("E_dur_cens_07")
p_pre_lab  <- check_pretrend_0811("E_labsh_0810")

cat(sprintf("R4a Bite H2A Dur (C+RY)  : G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f, pre0810_p=%.3f\n",
  cal_r4_dur_ry$g_eff, cal_r4_dur_ry$sd_x_tilde_wt, cal_r4_dur_ry$sy_ms_ratio, cal_r4_dur_ry$emp_size, cal_r4_dur_ry$mde_1sd, cal_r4_dur_ry$mde_2sd, p_pre_dur))
cat(sprintf("R4a Bite H2A Dur (C+SY)  : G_eff=%.2f, SD(X~)=%.4f, SY=1.00, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_r4_dur_sy$g_eff, cal_r4_dur_sy$sd_x_tilde_wt, cal_r4_dur_sy$emp_size, cal_r4_dur_sy$mde_1sd, cal_r4_dur_sy$mde_2sd))
cat(sprintf("R4a Bite Cens150+(C+RY)  : G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f, pre0810_p=%.3f\n",
  cal_r4_cens_ry$g_eff, cal_r4_cens_ry$sd_x_tilde_wt, cal_r4_cens_ry$sy_ms_ratio, cal_r4_cens_ry$emp_size, cal_r4_cens_ry$mde_1sd, cal_r4_cens_ry$mde_2sd, p_pre_cens))
cat(sprintf("R4a Bite Cens150+(C+SY)  : G_eff=%.2f, SD(X~)=%.4f, SY=1.00, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_r4_cens_sy$g_eff, cal_r4_cens_sy$sd_x_tilde_wt, cal_r4_cens_sy$emp_size, cal_r4_cens_sy$mde_1sd, cal_r4_cens_sy$mde_2sd))
cat(sprintf("R4b Bite LabShare(C+RY)  : G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f, pre0810_p=%.3f\n",
  cal_r4_lab_ry$g_eff, cal_r4_lab_ry$sd_x_tilde_wt, cal_r4_lab_ry$sy_ms_ratio, cal_r4_lab_ry$emp_size, cal_r4_lab_ry$mde_1sd, cal_r4_lab_ry$mde_2sd, p_pre_lab))
cat(sprintf("R4b Bite LabShare(C+SY)  : G_eff=%.2f, SD(X~)=%.4f, SY=1.00, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_r4_lab_sy$g_eff, cal_r4_lab_sy$sd_x_tilde_wt, cal_r4_lab_sy$emp_size, cal_r4_lab_sy$mde_1sd, cal_r4_lab_sy$mde_2sd))

# --- 4.2 R4 on the Oct 2, 2025 Rule (Within-State State x FY FE! Eliminates 2.1x State-Year Shock!) ---
cat("\n--- R4 on 2025 Rule (County + State x FY FE & Direct-Grower Business + State x FY FE) ---\n")
# County-level predetermined FY2022-FY2024 H-2A contract duration & 2020-2022 BEA labor cost share
ws_2224_dur <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, county_fips_list, adjusted_nbr_workers_certified, period_begin_date, period_end_date, number_of_hours) %>%
  filter(fiscal_year %in% c(2022L, 2023L, 2024L)) %>%
  collect() %>%
  mutate(
    begin = as.Date(period_begin_date), end = as.Date(period_end_date),
    weeks = as.numeric(end - begin) / 7,
    nh = coalesce(suppressWarnings(as.numeric(number_of_hours)), 40),
    h_per_w = weeks * nh,
    workers = as.numeric(adjusted_nbr_workers_certified),
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    nc = stringr::str_count(county_fips_list, ",") + 1L
  ) %>%
  filter(is.finite(h_per_w), h_per_w > 0, weeks <= 60, is.finite(workers), workers > 0) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = harmonize_county_fips_2010(trimws(county_fips_list)), w = workers / nc) %>%
  group_by(county_fips) %>%
  summarise(mean_h_per_w_2224 = sum(h_per_w * w) / sum(w), .groups = "drop")

labsh_2022 <- bp_full %>%
  filter(year %in% c(2020L, 2021L, 2022L)) %>%
  group_by(county_fips) %>%
  summarise(labsh_2022 = mean(share_farm_laborexp_prodexp, na.rm = TRUE), .groups = "drop")

cty_25_r4 <- qp_pre %>%
  filter(year == 2025L) %>%
  select(county_fips, state_fips, aewr_region_id) %>%
  left_join(st_sched_46 %>% select(state_fips, x_us_skill_1_2025, dose_raw), by = "state_fips") %>%
  left_join(ws_2224_dur, by = "county_fips") %>%
  left_join(labsh_2022, by = "county_fips") %>%
  mutate(
    s_wage_2224 = ifelse(is.finite(mean_h_per_w_2224), (x_us_skill_1_2025 * mean_h_per_w_2224) / (x_us_skill_1_2025 * mean_h_per_w_2224 + FIXED_COST_F), NA_real_),
    E_dur_st   = std_within(s_wage_2224, state_fips),
    E_labsh_st = std_within(labsh_2022, state_fips)
  )

# Calibrate within-State-FE differential bite (E_c * dose_s | county + state x FY FE, controlling for E_c x post!)
# Wait: with State x FY FE, E_c * dose_s is identified across states with different dose_s,
# while E_c * post absorbs national differential trend in E_c!
# Test BOTH: (a) without E_c x post (1b-style), and (b) WITH E_c x post (1a-style, pure dose interaction!)
cal_r4_2025_cty <- function(e_col, include_e_main, label_tag) {
  set.seed(SEED_PHASE1 + 300L)
  pairs <- 2013L:2025L; B_yr <- 40L # 520 draws
  n_d <- length(pairs) * B_yr
  b_v <- se_v <- numeric(n_d)
  pl_obj <- sup_obj <- NULL
  idx <- 0L
  for (yr in pairs) {
    w <- qp_pre %>%
      filter(year %in% c(yr - 1L, yr)) %>%
      select(county_fips, state_fips, aewr_region_id, year, hours_full_fy) %>%
      pivot_wider(names_from = year, values_from = hours_full_fy, names_prefix = "h_") %>%
      left_join(cty_25_r4 %>% select(county_fips, E = all_of(e_col)), by = "county_fips") %>%
      filter(is.finite(E))
    y0 <- w[[paste0("h_", yr - 1L)]]; y1 <- w[[paste0("h_", yr)]]
    Z_e <- if (include_e_main) matrix(w$E, ncol = 1L) else NULL
    if (yr == 2025L) {
      dx_a <- w$E * st_sched_46$dose_raw[match(w$state_fips, st_sched_46$state_fips)]
      sup_obj <- compute_2period_blind_support(y0, y1, dx_a, group_fe = w$state_fips, cluster_id = w$state_fips, Z = Z_e)
      pl_obj  <- fit_2period_ppml_cv3j(y0, y1, dx_a, group_fe = w$state_fips, cluster_id = w$state_fips, Z = Z_e)
    }
    for (k in seq_len(B_yr)) {
      idx <- idx + 1L
      perm <- st_sched_46 %>% group_by(aewr_region_id) %>% mutate(d = dose_raw[sample.int(n())]) %>% ungroup()
      dx_p <- w$E * perm$d[match(w$state_fips, perm$state_fips)]
      r <- fit_2period_ppml_cv3j(y0, y1, dx_p, group_fe = w$state_fips, cluster_id = w$state_fips, Z = Z_e)
      b_v[[idx]] <- r$beta * mean_p_46; se_v[[idx]] <- r$se_cv3j * mean_p_46
    }
  }
  em <- compute_empirical_mde(b_v, se_v, df_t = 45L)
  pl_el <- pl_obj$beta * mean_p_46; pl_se <- pl_obj$se_cv3j * mean_p_46
  pl_p <- 2 * pt(-abs(pl_el / pl_se), 45L)
  cat(sprintf("R4-2025 %-22s | G_eff=%.2f, SD(X~)=%.3f, size=%.3f [%.3f,%.3f] | MDE_1SD=%.3f, MDE_2SD=%.3f | pl2425=%.3f (p=%.3f)\n",
    label_tag, sup_obj$g_eff, sup_obj$sd_x_tilde_wt, em$emp_size, em$wilson_lo, em$wilson_hi, em$mde_upper, 2 * em$mde_upper, pl_el, pl_p))
  list(em = em, sup = sup_obj, pl = list(est = pl_el, se = pl_se, p = pl_p), n_u = sum(is.finite(cty_25_r4[[e_col]])))
}

r4_25_dur_1b  <- cal_r4_2025_cty("E_dur_st",   FALSE, "dur_stfe_noEmain")
r4_25_dur_1a  <- cal_r4_2025_cty("E_dur_st",   TRUE,  "dur_stfe_withEmain")
r4_25_lab_1b  <- cal_r4_2025_cty("E_labsh_st", FALSE, "labsh_stfe_noEmain")
r4_25_lab_1a  <- cal_r4_2025_cty("E_labsh_st", TRUE,  "labsh_stfe_withEmain")

# Also test R4a at DIRECT-GROWER BUSINESS LEVEL (FY2020-2025, business baseline duration x dose_s | business + state x FY FE!)
cat("\n--- R4a at Direct-Grower Business Level (State x FY FE, FY2020-2025) ---\n")
disc_2025 <- open_dataset(path_int("h2a_disclosure_records.parquet")) %>%
  select(fiscal_year, case_number, decision_date, h2a_labor_contractor) %>%
  filter(fiscal_year >= 2020L, fiscal_year <= 2025L) %>%
  collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year)) %>%
  group_by(case_number, fiscal_year) %>%
  summarise(
    dec = as.Date(substr(first(na.omit(as.character(decision_date))), 1L, 10L)),
    lc = first(na.omit(as.character(h2a_labor_contractor))), .groups = "drop"
  ) %>%
  filter(!is.na(dec), dec >= as.Date(sprintf("%d-10-01", fiscal_year - 1L)), dec <= as.Date(sprintf("%d-09-30", fiscal_year))) %>%
  mutate(is_h2alc = toupper(trimws(coalesce(lc, ""))) %in% c("Y", "YES", "TRUE", "1"))

t1_cty_ids <- qp_pre %>% filter(year == 2025L) %>% select(county_fips, state_fips_t1 = state_fips, aewr_region_id)
ws_bus_dur <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, case_number, county_fips_list, adjusted_nbr_workers_certified, period_begin_date, period_end_date, number_of_hours, employer_is_worksite_business, employer_id_balanced) %>%
  filter(fiscal_year >= 2020L, fiscal_year <= 2025L) %>%
  collect() %>%
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
  summarise(hours = sum(h, na.rm = TRUE), cert = sum(w, na.rm = TRUE), mean_hpw = ifelse(sum(w) > 0, sum(h) / sum(w), NA_real_), .groups = "drop")

# Calibrate business-level duration-amortization bite under State x FY FE across FY2021-2025 (5 x 100 = 500 draws!)
set.seed(SEED_PHASE1 + 310L)
b_bdur_1b <- se_bdur_1b <- b_bdur_1a <- se_bdur_1a <- numeric(500L)
pl_bdur_1b <- pl_bdur_1a <- sup_bdur_1b <- sup_bdur_1a <- NULL
idx <- 0L
for (yr in 2021L:2025L) {
  b0 <- ws_bus_dur %>%
    filter(year == yr - 1L, hours > 0, is.finite(mean_hpw), mean_hpw > 0) %>%
    group_by(bus_id) %>%
    arrange(desc(hours), state_fips, .by_group = TRUE) %>%
    summarise(state_fips = first(state_fips), aewr_region_id = first(aewr_region_id), h0 = sum(hours), hpw0 = sum(hours) / sum(cert), .groups = "drop") %>%
    left_join(st_sched_46 %>% select(state_fips, x_us_skill_1_2025), by = "state_fips") %>%
    mutate(s_wage = (x_us_skill_1_2025 * hpw0) / (x_us_skill_1_2025 * hpw0 + FIXED_COST_F), E_dur = std_within(s_wage, state_fips)) %>%
    filter(is.finite(E_dur))
  b1 <- ws_bus_dur %>% filter(year == yr) %>% group_by(bus_id) %>% summarise(h1 = sum(hours), .groups = "drop")
  coh <- b0 %>% left_join(b1, by = "bus_id") %>% mutate(h1 = coalesce(h1, 0))
  # Collapse by state x E_dur quantile? No, exact business-level binomial logit with 46 state FEs is fast!
  if (yr == 2025L) {
    dx_a <- coh$E_dur * st_sched_46$dose_raw[match(coh$state_fips, st_sched_46$state_fips)]
    sup_bdur_1b <- compute_2period_blind_support(coh$h0, coh$h1, dx_a, group_fe = coh$state_fips, cluster_id = coh$state_fips)
    sup_bdur_1a <- compute_2period_blind_support(coh$h0, coh$h1, dx_a, group_fe = coh$state_fips, cluster_id = coh$state_fips, Z = matrix(coh$E_dur, ncol = 1L))
    pl_bdur_1b <- fit_2period_ppml_cv3j(coh$h0, coh$h1, dx_a, group_fe = coh$state_fips, cluster_id = coh$state_fips)
    pl_bdur_1a <- fit_2period_ppml_cv3j(coh$h0, coh$h1, dx_a, group_fe = coh$state_fips, cluster_id = coh$state_fips, Z = matrix(coh$E_dur, ncol = 1L))
  }
  # Note: For speed over 500 draws x 15k businesses, bin E_dur into 20 within-state vigintiles (exact sufficient stats for 2-period PPML
  # when E_dur is discretized, or keep 20 bins per state = 920 rows — spot-check accuracy vs full 15k!)
  coh_bin <- coh %>%
    group_by(state_fips) %>%
    mutate(ebin = ntile(E_dur, 20L)) %>%
    group_by(state_fips, ebin) %>%
    summarise(h0 = sum(h0), h1 = sum(h1), E_dur = sum(E_dur * (h0 + h1)) / sum(h0 + h1), .groups = "drop")
  for (k in seq_len(100L)) {
    idx <- idx + 1L
    perm <- st_sched_46 %>% group_by(aewr_region_id) %>% mutate(d = dose_raw[sample.int(n())]) %>% ungroup()
    dx_p <- coh_bin$E_dur * perm$d[match(coh_bin$state_fips, perm$state_fips)]
    r1 <- fit_2period_ppml_cv3j(coh_bin$h0, coh_bin$h1, dx_p, group_fe = coh_bin$state_fips, cluster_id = coh_bin$state_fips)
    r2 <- fit_2period_ppml_cv3j(coh_bin$h0, coh_bin$h1, dx_p, group_fe = coh_bin$state_fips, cluster_id = coh_bin$state_fips, Z = matrix(coh_bin$E_dur, ncol = 1L))
    b_bdur_1b[[idx]] <- r1$beta * mean_p_46; se_bdur_1b[[idx]] <- r1$se_cv3j * mean_p_46
    b_bdur_1a[[idx]] <- r2$beta * mean_p_46; se_bdur_1a[[idx]] <- r2$se_cv3j * mean_p_46
  }
}
e_bdur_1b <- compute_empirical_mde(b_bdur_1b, se_bdur_1b, 45L)
e_bdur_1a <- compute_empirical_mde(b_bdur_1a, se_bdur_1a, 45L)
cat(sprintf("R4a Bus-Dur (StFE,noEmain): G_eff=%.2f, SD(X~)=%.3f, size=%.3f [%.3f,%.3f], MDE_1SD=%.3f, MDE_2SD=%.3f, pl2425=%.3f (p=%.3f)\n",
  sup_bdur_1b$g_eff, sup_bdur_1b$sd_x_tilde_wt, e_bdur_1b$emp_size, e_bdur_1b$wilson_lo, e_bdur_1b$wilson_hi,
  e_bdur_1b$mde_upper, 2 * e_bdur_1b$mde_upper, pl_bdur_1b$beta * mean_p_46, 2*pt(-abs(pl_bdur_1b$beta/pl_bdur_1b$se_cv3j), 45L)))
cat(sprintf("R4a Bus-Dur (StFE,+Emain) : G_eff=%.2f, SD(X~)=%.3f, size=%.3f [%.3f,%.3f], MDE_1SD=%.3f, MDE_2SD=%.3f, pl2425=%.3f (p=%.3f)\n",
  sup_bdur_1a$g_eff, sup_bdur_1a$sd_x_tilde_wt, e_bdur_1a$emp_size, e_bdur_1a$wilson_lo, e_bdur_1a$wilson_hi,
  e_bdur_1a$mde_upper, 2 * e_bdur_1a$mde_upper, pl_bdur_1a$beta * mean_p_46, 2*pt(-abs(pl_bdur_1a$beta/pl_bdur_1a$se_cv3j), 45L)))

cat("\n======================================================================\n")
cat("CANDIDATE R6: Stacked Multi-Event Statutory-Reversal Design (2009-10 + 2025)\n")
cat("======================================================================\n")

# Step 1: Construct exact 2009 OES -> 2010 FLS statutory county effective-floor schedule
# Strictly obeys AEWR-001 (NEVER uses aewr_nonstatutory_application_wage_2009!)
WY09 <- "2008-07-01/2009-06-30"; WY10 <- "2009-07-01/2010-06-30"
sch_09 <- read_csv("explorations/identification_search/sources/aewr_2009_oes_schedule.csv", col_types = cols(.default = "c")) %>%
  filter(skill_level == "I", soc_code %in% c("45-2092", "45-2099", "45-2091")) %>%
  transmute(wage_year, state_fips = state_fips(state_fips), oes_area_code, soc_code, w = suppressWarnings(as.numeric(owl_hourly_wage))) %>%
  pivot_wider(names_from = soc_code, values_from = w) %>%
  group_by(state_fips, oes_area_code) %>%
  mutate(l1_other = ifelse(wage_year == WY09, `45-2092`[wage_year == WY10][1], `45-2092`[wage_year == WY09][1])) %>%
  ungroup() %>%
  mutate(l1_fill = coalesce(`45-2092`, l1_other, `45-2099`, `45-2091`))

xw_09 <- read_csv("explorations/identification_search/sources/oes_area_county_crosswalk_2009.csv", col_types = cols(.default = "c")) %>%
  filter(!is.na(county_fips)) %>%
  mutate(county_fips = harmonize_county_fips_2010(county_fips), state_fips = state_fips(state_fips)) %>%
  count(wage_year, county_fips, state_fips, oes_area_code, name = "towns") %>%
  group_by(wage_year, county_fips) %>%
  arrange(desc(towns), oes_area_code, .by_group = TRUE) %>%
  slice(1L) %>%
  ungroup()

fls_0811 <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>%
  filter(occupation_scope == "nonrange_reference", reference_year %in% 2008L:2011L) %>%
  transmute(state_fips = state_fips(state_fips), ref_yr = as.integer(reference_year), fls = as.numeric(aewr),
            start = as.Date(effective_start), end_x = as.Date(effective_end_exclusive))

mw_0811 <- read_parquet(path_int("state_real_minwages.parquet")) %>%
  filter(year %in% 2008L:2011L) %>%
  transmute(state_fips = state_fips(state_fips), year = as.integer(year), mw = as.numeric(prevailing_ag_min_wage))

# Build county-level statutory floor for OES regime (FY2009 OES window: 2009-01-17 to 2010-03-14)
# vs restored FLS regime (2010-03-15 onward: 2010 FLS AEWR)
fls_2010_st <- fls_0811 %>%
  filter(start <= as.Date("2010-04-01"), is.na(end_x) | end_x > as.Date("2010-04-01")) %>%
  select(state_fips, fls_2010 = fls) %>%
  distinct()
fls_2008_st <- fls_0811 %>%
  filter(start <= as.Date("2008-04-01"), is.na(end_x) | end_x > as.Date("2008-04-01")) %>%
  select(state_fips, fls_2008 = fls) %>%
  distinct()

# County OES floor (average of WY09 & WY10 Level I, floored at max(7.25, state ag min wage 2009))
cty_oes_floor <- xw_09 %>%
  left_join(sch_09 %>% select(wage_year, state_fips, oes_area_code, l1_fill), by = c("wage_year", "state_fips", "oes_area_code")) %>%
  left_join(mw_0811 %>% filter(year == 2009L) %>% select(state_fips, mw09 = mw), by = "state_fips") %>%
  left_join(mw_0811 %>% filter(year == 2010L) %>% select(state_fips, mw10 = mw), by = "state_fips") %>%
  mutate(
    floor_wy = ifelse(wage_year == WY09, pmax(l1_fill, 7.25, mw09, na.rm = TRUE), pmax(l1_fill, 7.25, mw10, na.rm = TRUE)),
    trunc_wy = ifelse(wage_year == WY09, pmax(7.25, mw09, na.rm = TRUE) > l1_fill, pmax(7.25, mw10, na.rm = TRUE) > l1_fill)
  ) %>%
  group_by(county_fips, state_fips) %>%
  summarise(
    x_oes_09 = mean(floor_wy, na.rm = TRUE),
    x_oes_wy09 = first(floor_wy[wage_year == WY09]),
    x_oes_wy10 = first(floor_wy[wage_year == WY10]),
    trunc_rate_09 = mean(trunc_wy, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  inner_join(t1_cty_ids %>% select(county_fips, aewr_region_id), by = "county_fips") %>%
  left_join(fls_2008_st, by = "state_fips") %>%
  left_join(fls_2010_st, by = "state_fips") %>%
  left_join(mw_0811 %>% filter(year == 2010L) %>% select(state_fips, mw10 = mw), by = "state_fips") %>%
  mutate(
    x_fls_10 = pmax(fls_2010, mw10, na.rm = TRUE),
    dose_09_to_10 = x_fls_10 - x_oes_09,
    dln_09_to_10 = log(x_fls_10) - log(x_oes_09)
  )

cat(sprintf("2009 OES -> 2010 FLS statutory schedule: %d Track-1 counties, mean dose=+$%.2f/hr, SD=$%.2f/hr, trunc_rate=%.1f%%\n",
  nrow(cty_oes_floor), mean(cty_oes_floor$dose_09_to_10, na.rm = TRUE), sd(cty_oes_floor$dose_09_to_10, na.rm = TRUE),
  100 * mean(cty_oes_floor$trunc_rate_09, na.rm = TRUE)))

# Step 2: Verify 2009 -> 2010 OFFERED-WAGE FIRST STAGE (strictly required before stacking!)
# Use allocated worksites FY2008-FY2011 (observed hourly wage offers & receipt dates; non-monthly!)
disc_0811 <- open_dataset(path_int("h2a_disclosure_records.parquet")) %>%
  select(fiscal_year, case_number, case_received_date, decision_date, case_status) %>%
  filter(fiscal_year >= 2008L, fiscal_year <= 2012L) %>%
  collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year)) %>%
  group_by(case_number, fiscal_year) %>%
  summarise(
    recv = as.Date(substr(first(na.omit(as.character(case_received_date))), 1L, 10L)),
    dec  = as.Date(substr(first(na.omit(as.character(decision_date))), 1L, 10L)),
    .groups = "drop"
  )

ws_0811 <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, case_number, county_fips_list, adjusted_nbr_workers_certified, wage_rate, wage_unit, period_begin_date) %>%
  filter(fiscal_year >= 2008L, fiscal_year <= 2012L) %>%
  collect() %>%
  mutate(
    fiscal_year = as.integer(fiscal_year),
    begin = as.Date(period_begin_date),
    wage = suppressWarnings(as.numeric(wage_rate)),
    unit_u = toupper(trimws(coalesce(as.character(wage_unit), ""))),
    hourly = unit_u %in% c("HOUR", "HOURLY", "HR") & is.finite(wage) & wage > 0 & wage <= 100,
    monthly = unit_u %in% c("MTH", "MONTHLY", "MONTH"),
    workers = as.numeric(adjusted_nbr_workers_certified)
  ) %>%
  left_join(disc_0811, by = c("case_number", "fiscal_year")) %>%
  mutate(
    # Exact regime period by receipt date:
    # pre_fls : 2008-01-17 .. 2009-01-16 (12 mos before 2008 rule)
    # oes_09  : 2009-01-17 .. 2010-03-14 (14 mos under 2008 OES rule)
    # post_fls: 2010-03-15 .. 2011-03-14 (12 mos after 2010 return to FLS)
    regime = case_when(
      !is.na(recv) & recv >= as.Date("2008-01-17") & recv < as.Date("2009-01-17") ~ "pre08",
      !is.na(recv) & recv >= as.Date("2009-01-17") & recv < as.Date("2010-03-15") ~ "oes09",
      !is.na(recv) & recv >= as.Date("2010-03-15") & recv < as.Date("2011-03-15") ~ "fls10",
      TRUE ~ NA_character_
    ),
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    nc = stringr::str_count(county_fips_list, ",") + 1L
  ) %>%
  filter(!is.na(regime), !monthly, is.finite(workers), workers > 0) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = harmonize_county_fips_2010(trimws(county_fips_list)), pos = workers / nc) %>%
  inner_join(cty_oes_floor, by = "county_fips")

# County x regime positions & offered wage (strictly non-monthly positions, ZERO imputed hours!)
r6_cty_reg <- ws_0811 %>%
  group_by(county_fips, state_fips, aewr_region_id, regime) %>%
  summarise(
    cert_pos = sum(pos, na.rm = TRUE),
    w_mean = ifelse(sum(pos[hourly], na.rm = TRUE) > 0, sum(wage[hourly] * pos[hourly], na.rm = TRUE) / sum(pos[hourly], na.rm = TRUE), NA_real_),
    pos_hourly = sum(pos[hourly], na.rm = TRUE),
    .groups = "drop"
  )

# Build balanced county x {oes09, fls10} panel for 2009->2010 reversal
r6_ep1 <- tidyr::crossing(
  cty_oes_floor %>% select(county_fips, state_fips, aewr_region_id, x_oes_09, x_fls_10, fls_2008, dose_09_to_10),
  regime = c("oes09", "fls10")
) %>%
  left_join(r6_cty_reg, by = c("county_fips", "state_fips", "aewr_region_id", "regime")) %>%
  mutate(
    cert_pos = coalesce(cert_pos, 0),
    period = as.integer(regime == "fls10"),
    x_stat = ifelse(period == 0L, x_oes_09, x_fls_10)
  )

# First stage on continuing counties (oes09 -> fls10) under BOTH:
# (a) County + Former-Region x Period FE (cross-state + sub-state reversal!)
# (b) County + State x Period FE (pure within-state OES variation, A2 benchmark)
r6_fs_df <- r6_ep1 %>%
  group_by(county_fips) %>%
  filter(sum(is.finite(w_mean)) == 2L) %>%
  ungroup()

fs_r6_reg  <- fit_cv3j_fixest(r6_fs_df, w_mean ~ x_stat | county_fips + aewr_region_id^period, "x_stat", "state_fips", "feols")
fs_r6_regw <- fit_cv3j_fixest(r6_fs_df, w_mean ~ x_stat | county_fips + aewr_region_id^period, "x_stat", "state_fips", "feols", weights_fml = ~pos_hourly)
fs_r6_st   <- fit_cv3j_fixest(r6_fs_df, w_mean ~ x_stat | county_fips + state_fips^period, "x_stat", "state_fips", "feols")
fs_r6_stw  <- fit_cv3j_fixest(r6_fs_df, w_mean ~ x_stat | county_fips + state_fips^period, "x_stat", "state_fips", "feols", weights_fml = ~pos_hourly)

cat(sprintf("R6 2009->10 FS (County + RegxPer, eq-wt) : pi=%.4f [CV3J %.4f], F=%.2f, N=%d, G=%d\n",
  fs_r6_reg$estimate, fs_r6_reg$se_cv3j, (fs_r6_reg$estimate/fs_r6_reg$se_cv3j)^2, fs_r6_reg$n_obs, fs_r6_reg$G))
cat(sprintf("R6 2009->10 FS (County + RegxPer, pos-wt): pi=%.4f [CV3J %.4f], F=%.2f\n",
  fs_r6_regw$estimate, fs_r6_regw$se_cv3j, (fs_r6_regw$estimate/fs_r6_regw$se_cv3j)^2))
cat(sprintf("R6 2009->10 FS (County + StxPer,  eq-wt) : pi=%.4f [CV3J %.4f], F=%.2f\n",
  fs_r6_st$estimate, fs_r6_st$se_cv3j, (fs_r6_st$estimate/fs_r6_st$se_cv3j)^2))
cat(sprintf("R6 2009->10 FS (County + StxPer,  pos-wt): pi=%.4f [CV3J %.4f], F=%.2f\n",
  fs_r6_stw$estimate, fs_r6_stw$se_cv3j, (fs_r6_stw$estimate/fs_r6_stw$se_cv3j)^2))

# Step 3: Exact Fisher-information share check!
# How much of the stacked (event 1: 2009->10 + event 2: 2025->26) FE-PPML positions Fisher information
# comes from event 1 (2009->10) vs event 2 (2025->26)?
# Note: Use strictly pre-policy FY2024->25 positions for event 2's null weights (outcome-blind!)
# and check BOTH raw positions-weighted PPML AND normalized (equal-event-weight!) stacked PPML!
w_ep1 <- r6_ep1 %>%
  select(county_fips, state_fips, aewr_region_id, period, cert_pos, x_stat) %>%
  pivot_wider(names_from = period, values_from = c(cert_pos, x_stat)) %>%
  mutate(dx = x_stat_1 - x_stat_0)

w_ep2 <- qp_pre %>%
  filter(year %in% c(2024L, 2025L)) %>%
  select(county_fips, state_fips, aewr_region_id, year, cert_full_fy) %>%
  pivot_wider(names_from = year, values_from = cert_full_fy, names_prefix = "c_") %>%
  left_join(st_sched_46 %>% select(state_fips, dx = dose_raw), by = "state_fips")

sup_ep1_reg <- compute_2period_blind_support(w_ep1$cert_pos_0, w_ep1$cert_pos_1, w_ep1$dx, w_ep1$aewr_region_id, w_ep1$state_fips)
sup_ep1_st  <- compute_2period_blind_support(w_ep1$cert_pos_0, w_ep1$cert_pos_1, w_ep1$dx, w_ep1$state_fips, w_ep1$state_fips)
sup_ep2_reg <- compute_2period_blind_support(w_ep2$c_2024, w_ep2$c_2025, w_ep2$dx, w_ep2$aewr_region_id, w_ep2$state_fips)

# Note: In dollar units, total Fisher info I_ep1 vs I_ep2; ALSO check in LOG-RATE units (since 2009 AEWR was ~$9 vs 2025 ~$17!)
w_ep1_ln <- w_ep1 %>% mutate(dln = log(x_stat_1) - log(x_stat_0))
st_dln_46 <- st_sched_46 %>% mutate(dln = log(x_us_skill_1_2026) - log(x_us_skill_1_2025))
w_ep2_ln <- w_ep2 %>% select(-dx) %>% left_join(st_dln_46 %>% select(state_fips, dln), by = "state_fips")

sup_ep1_ln <- compute_2period_blind_support(w_ep1_ln$cert_pos_0, w_ep1_ln$cert_pos_1, w_ep1_ln$dln, w_ep1_ln$aewr_region_id, w_ep1_ln$state_fips)
sup_ep2_ln <- compute_2period_blind_support(w_ep2_ln$c_2024, w_ep2_ln$c_2025, w_ep2_ln$dln, w_ep2_ln$aewr_region_id, w_ep2_ln$state_fips)

info_sh_ep1_dollar <- sup_ep1_reg$total_fisher_info / (sup_ep1_reg$total_fisher_info + sup_ep2_reg$total_fisher_info)
info_sh_ep1_log    <- sup_ep1_ln$total_fisher_info / (sup_ep1_ln$total_fisher_info + sup_ep2_ln$total_fisher_info)
# Also check event-normalized PPML (scaling each event's positions so sum(positions) = 1 in each event, removing 4x secular H-2A growth!)
scale_ep1 <- sum(w_ep2$c_2024 + w_ep2$c_2025) / sum(w_ep1$cert_pos_0 + w_ep1$cert_pos_1)
info_sh_ep1_norm_log <- (sup_ep1_ln$total_fisher_info * scale_ep1) / (sup_ep1_ln$total_fisher_info * scale_ep1 + sup_ep2_ln$total_fisher_info)

cat(sprintf("R6 Event 1 (2009->10, Reg FE): G_eff=%.2f, SD(dx$)=%.3f, SD(dln)=%.4f, pos_total=%.0f (vs Ep2=%.0f, ratio=%.2fx)\n",
  sup_ep1_reg$g_eff, sup_ep1_reg$sd_x_tilde_wt, sup_ep1_ln$sd_x_tilde_wt,
  sum(w_ep1$cert_pos_0 + w_ep1$cert_pos_1)/2, sum(w_ep2$c_2024 + w_ep2$c_2025)/2, scale_ep1))
cat(sprintf("R6 Event 1 (2009->10, St  FE): G_eff=%.2f, SD(dx$)=%.3f\n", sup_ep1_st$g_eff, sup_ep1_st$sd_x_tilde_wt))
cat(sprintf("R6 Fisher-info share from 2009->10: raw_dollar=%.1f%%, raw_log=%.1f%%, event-size-normalized_log=%.1f%%\n",
  100 * info_sh_ep1_dollar, 100 * info_sh_ep1_log, 100 * info_sh_ep1_norm_log))

# Step 4: Strictly outcome-blind stacked positions placebo calibration (500 draws, exact 46-state CV3J!)
# Stack two independent pre-policy 2-year transitions per draw:
#   - Event 1 placebo: random pre-policy transition from FY2012-FY2025 (or pre08->09) with permuted 2009->10 doses
#   - Event 2 placebo: different pre-policy transition from FY2012-FY2025 with permuted 2025->26 doses
# Wait: for realistic 2009->10 noise scale, also check actual 2008->2009 / 2010->2011 transitions AND 2012-2025 transitions!
# Compute BOTH raw stacked PPML (log-rate) AND event-size-normalized stacked PPML (log-rate, where Ep1 has weight scale_ep1!)
st_ep1_ln <- w_ep1_ln %>%
  group_by(state_fips, aewr_region_id) %>%
  summarise(dln_ep1 = weighted.mean(dln, pmax(cert_pos_0 + cert_pos_1, 1e-6)), .groups = "drop")

# Use actual pre-policy county positions from qp_pre (FY2012-2025) to calibrate stacked 2-episode positions MDE
# (using 500 draws of pairs of distinct pre-policy transitions, permuting Ep1 and Ep2 doses within former regions!)
set.seed(SEED_PHASE1 + 400L)
pre_yrs_r6 <- 2013L:2025L
n_r6_draws <- 500L
b_stk_raw <- se_stk_raw <- b_stk_norm <- se_stk_norm <- b_ep2_only <- se_ep2_only <- numeric(n_r6_draws)

# Pre-extract county positions by year for fast lookup
pos_by_yr <- lapply(2012L:2025L, function(yy) {
  d <- qp_pre %>% filter(year == yy) %>% select(county_fips, state_fips, aewr_region_id, cert_full_fy)
  d[match(w_ep1_ln$county_fips, d$county_fips), ]
})
names(pos_by_yr) <- as.character(2012L:2025L)

for (b in seq_len(n_r6_draws)) {
  yrs2 <- sample(pre_yrs_r6, 2L, replace = FALSE)
  y1_a <- pos_by_yr[[as.character(yrs2[1] - 1L)]]$cert_full_fy
  y1_b <- pos_by_yr[[as.character(yrs2[1])]]$cert_full_fy
  y2_a <- pos_by_yr[[as.character(yrs2[2] - 1L)]]$cert_full_fy
  y2_b <- pos_by_yr[[as.character(yrs2[2])]]$cert_full_fy

  # Scale Event 1 down by 1/scale_ep1 for raw stacked (matching 2009-10 ~80k vs 2025-26 ~320k positions!),
  # OR keep equal scale for event-normalized stacked!
  y1_a_raw <- y1_a / scale_ep1; y1_b_raw <- y1_b / scale_ep1

  # Permute state doses within former AEWR region independently for Ep1 and Ep2 (preserving county OES deviations within state for Ep1!)
  perm1 <- st_dln_46 %>% group_by(aewr_region_id) %>% mutate(src_st = state_fips[sample.int(n())]) %>% ungroup()
  perm2 <- st_dln_46 %>% group_by(aewr_region_id) %>% mutate(dln2 = dln[sample.int(n())]) %>% ungroup()
  # Map Ep1 county-level dln by shifting state-level mean via perm1 while preserving within-state OES spread
  st_m_ep1 <- st_ep1_ln$dln_ep1[match(w_ep1_ln$state_fips, st_ep1_ln$state_fips)]
  st_p_ep1 <- st_ep1_ln$dln_ep1[match(perm1$src_st[match(w_ep1_ln$state_fips, perm1$state_fips)], st_ep1_ln$state_fips)]
  dln1_cty <- (w_ep1_ln$dln - st_m_ep1) + st_p_ep1
  dln2_cty <- perm2$dln2[match(w_ep1_ln$state_fips, perm2$state_fips)]

  # Stack Event 1 + Event 2 with event x region FE and state clustering!
  grp_stk <- c(paste0("E1_", w_ep1_ln$aewr_region_id), paste0("E2_", w_ep1_ln$aewr_region_id))
  st_stk  <- c(w_ep1_ln$state_fips, w_ep1_ln$state_fips)
  dx_stk  <- c(dln1_cty, dln2_cty)

  r_raw  <- fit_2period_ppml_cv3j(c(y1_a_raw, y2_a), c(y1_b_raw, y2_b), dx_stk, grp_stk, st_stk)
  r_norm <- fit_2period_ppml_cv3j(c(y1_a, y2_a),     c(y1_b, y2_b),     dx_stk, grp_stk, st_stk)
  r_e2   <- fit_2period_ppml_cv3j(y2_a, y2_b, dln2_cty, w_ep1_ln$aewr_region_id, w_ep1_ln$state_fips)

  b_stk_raw[[b]]  <- r_raw$beta;  se_stk_raw[[b]]  <- r_raw$se_cv3j
  b_stk_norm[[b]] <- r_norm$beta; se_stk_norm[[b]] <- r_norm$se_cv3j
  b_ep2_only[[b]] <- r_e2$beta;   se_ep2_only[[b]] <- r_e2$se_cv3j
}

e_stk_raw  <- compute_empirical_mde(b_stk_raw, se_stk_raw, 45L)
e_stk_norm <- compute_empirical_mde(b_stk_norm, se_stk_norm, 45L)
e_ep2_only <- compute_empirical_mde(b_ep2_only, se_ep2_only, 45L)

# Compute stacked G_eff
sup_stk_raw  <- compute_2period_blind_support(c(w_ep1_ln$cert_pos_0, w_ep2_ln$c_2024), c(w_ep1_ln$cert_pos_1, w_ep2_ln$c_2025),
                                              c(w_ep1_ln$dln, w_ep2_ln$dln),
                                              c(paste0("E1_", w_ep1_ln$aewr_region_id), paste0("E2_", w_ep2_ln$aewr_region_id)),
                                              c(w_ep1_ln$state_fips, w_ep2_ln$state_fips))
sup_stk_norm <- compute_2period_blind_support(c(w_ep1_ln$cert_pos_0 * scale_ep1, w_ep2_ln$c_2024), c(w_ep1_ln$cert_pos_1 * scale_ep1, w_ep2_ln$c_2025),
                                              c(w_ep1_ln$dln, w_ep2_ln$dln),
                                              c(paste0("E1_", w_ep1_ln$aewr_region_id), paste0("E2_", w_ep2_ln$aewr_region_id)),
                                              c(w_ep1_ln$state_fips, w_ep2_ln$state_fips))

cat(sprintf("R6 Ep2-only (ln rate, pos)  : size=%.3f [%.3f,%.3f] | MDE=%.3f (analytic=%.3f)\n",
  e_ep2_only$emp_size, e_ep2_only$wilson_lo, e_ep2_only$wilson_hi, e_ep2_only$mde_upper, e_ep2_only$analytic_mde))
cat(sprintf("R6 Stacked Raw (ln, pos)    : G_eff=%.2f, Ep1_info=%.1f%%, size=%.3f [%.3f,%.3f] | MDE=%.3f (analytic=%.3f)\n",
  sup_stk_raw$g_eff, 100 * info_sh_ep1_log, e_stk_raw$emp_size, e_stk_raw$wilson_lo, e_stk_raw$wilson_hi, e_stk_raw$mde_upper, e_stk_raw$analytic_mde))
cat(sprintf("R6 Stacked Norm (ln, pos)   : G_eff=%.2f, Ep1_info=%.1f%%, size=%.3f [%.3f,%.3f] | MDE=%.3f (analytic=%.3f)\n",
  sup_stk_norm$g_eff, 100 * info_sh_ep1_norm_log, e_stk_norm$emp_size, e_stk_norm$wilson_lo, e_stk_norm$wilson_hi, e_stk_norm$mde_upper, e_stk_norm$analytic_mde))

# Save R4 & R6 triage table
make_r4_bite_row <- function(cand, est_lbl, cal, p_pre, note_txt) {
  tibble(
    candidate = cand, family = "R4", estimand = est_lbl, unit = "county_year_bite",
    n_obs = as.integer(cal$n_obs), n_units = as.integer(cal$n_counties), us_hours_share = 1.0,
    G = as.integer(cal$G), G_eff = cal$g_eff, sd_x_tilde = cal$sd_x_tilde_wt, r2_fe = cal$r2_x_wt,
    first_stage_pi = NA_real_, first_stage_se = NA_real_, first_stage_F = NA_real_,
    sy_ms_ratio = cal$sy_ms_ratio, n_placebo_draws = 999L,
    emp_size_alpha05 = cal$emp_size, size_wilson_hi = cal$wilson_hi,
    mde_primary = cal$mde_2sd, mde_unit = "2SD_differential_bite_elasticity",
    mde_eps_cost = NA_real_, mde_eps_cost_r7 = NA_real_,
    placebo_est = NA_real_, placebo_se = NA_real_, placebo_p = p_pre,
    notes = sprintf("1SD_MDE=%.3f; 2008-10 pretrend p=%.3f; %s", cal$mde_1sd, p_pre, note_txt)
  )
}

make_r4_25_row <- function(cand, est_lbl, obj, note_txt) {
  tibble(
    candidate = cand, family = "R4", estimand = est_lbl, unit = "county_fy",
    n_obs = as.integer(2L * obj$n_u), n_units = as.integer(obj$n_u), us_hours_share = 0.785,
    G = 46L, G_eff = obj$sup$g_eff, sd_x_tilde = obj$sup$sd_x_tilde_wt, r2_fe = obj$sup$r2_fe_wt,
    first_stage_pi = NA_real_, first_stage_se = NA_real_, first_stage_F = NA_real_,
    sy_ms_ratio = 1.0, n_placebo_draws = obj$em$n_valid,
    emp_size_alpha05 = obj$em$emp_size, size_wilson_hi = obj$em$wilson_hi,
    mde_primary = 2 * obj$em$mde_upper, mde_unit = "2SD_differential_bite_elasticity",
    mde_eps_cost = NA_real_, mde_eps_cost_r7 = NA_real_,
    placebo_est = obj$pl$est, placebo_se = obj$pl$se, placebo_p = obj$pl$p,
    notes = sprintf("1SD_MDE=%.3f; %s", obj$em$mde_upper, note_txt)
  )
}

r4_r6_triage <- bind_rows(
  make_r4_bite_row("R4a_bite_h2a_duration_ry", "duration_amortization_bite_2011_25_ry", cal_r4_dur_ry, p_pre_dur, "2008-10 H-2A contract duration share, County+RegxYr"),
  make_r4_bite_row("R4a_bite_h2a_duration_sy", "duration_amortization_bite_2011_25_sy", cal_r4_dur_sy, p_pre_dur, "2008-10 H-2A contract duration share, County+StxYr"),
  make_r4_bite_row("R4a_bite_census150_ry",    "census_duration_bite_2011_25_ry",       cal_r4_cens_ry, p_pre_cens, "2007 Census >=150d share, County+RegxYr"),
  make_r4_bite_row("R4a_bite_census150_sy",    "census_duration_bite_2011_25_sy",       cal_r4_cens_sy, p_pre_cens, "2007 Census >=150d share, County+StxYr"),
  make_r4_bite_row("R4b_bite_laborshare_ry",   "farm_labor_share_bite_2011_25_ry",      cal_r4_lab_ry, p_pre_lab, "2008-10 BEA farm labor cost share, County+RegxYr"),
  make_r4_bite_row("R4b_bite_laborshare_sy",   "farm_labor_share_bite_2011_25_sy",      cal_r4_lab_sy, p_pre_lab, "2008-10 BEA farm labor cost share, County+StxYr"),
  make_r4_25_row("R4a_2025_cty_dur_stfe_1b",   "duration_amortization_2025_stfe_1b",    r4_25_dur_1b, "2022-24 H-2A duration x dose, County+StatexFY FE (no E main)"),
  make_r4_25_row("R4a_2025_cty_dur_stfe_1a",   "duration_amortization_2025_stfe_1a",    r4_25_dur_1a, "2022-24 H-2A duration x dose + E main, County+StatexFY FE"),
  make_r4_25_row("R4b_2025_cty_labsh_stfe_1b", "farm_labor_share_2025_stfe_1b",         r4_25_lab_1b, "2020-22 BEA labor share x dose, County+StatexFY FE (no E main)"),
  make_r4_25_row("R4b_2025_cty_labsh_stfe_1a", "farm_labor_share_2025_stfe_1a",         r4_25_lab_1a, "2020-22 BEA labor share x dose + E main, County+StatexFY FE"),
  tibble(
    candidate = "R4a_2025_bus_dur_stfe_1b", family = "R4", estimand = "business_duration_amortization_2025_stfe_1b", unit = "business_fy",
    n_obs = NA_integer_, n_units = NA_integer_, us_hours_share = 0.587, G = 46L,
    G_eff = sup_bdur_1b$g_eff, sd_x_tilde = sup_bdur_1b$sd_x_tilde_wt, r2_fe = sup_bdur_1b$r2_fe_wt,
    first_stage_pi = NA_real_, first_stage_se = NA_real_, first_stage_F = NA_real_, sy_ms_ratio = 1.0,
    n_placebo_draws = e_bdur_1b$n_valid, emp_size_alpha05 = e_bdur_1b$emp_size, size_wilson_hi = e_bdur_1b$wilson_hi,
    mde_primary = 2 * e_bdur_1b$mde_upper, mde_unit = "2SD_differential_bite_elasticity",
    mde_eps_cost = NA_real_, mde_eps_cost_r7 = NA_real_,
    placebo_est = pl_bdur_1b$beta * mean_p_46, placebo_se = pl_bdur_1b$se_cv3j * mean_p_46,
    placebo_p = 2 * pt(-abs(pl_bdur_1b$beta / pl_bdur_1b$se_cv3j), 45L),
    notes = sprintf("1SD_MDE=%.3f; direct-grower business duration x dose, State x FY FE (no E main)", e_bdur_1b$mde_upper)
  ),
  tibble(
    candidate = "R4a_2025_bus_dur_stfe_1a", family = "R4", estimand = "business_duration_amortization_2025_stfe_1a", unit = "business_fy",
    n_obs = NA_integer_, n_units = NA_integer_, us_hours_share = 0.587, G = 46L,
    G_eff = sup_bdur_1a$g_eff, sd_x_tilde = sup_bdur_1a$sd_x_tilde_wt, r2_fe = sup_bdur_1a$r2_fe_wt,
    first_stage_pi = NA_real_, first_stage_se = NA_real_, first_stage_F = NA_real_, sy_ms_ratio = 1.0,
    n_placebo_draws = e_bdur_1a$n_valid, emp_size_alpha05 = e_bdur_1a$emp_size, size_wilson_hi = e_bdur_1a$wilson_hi,
    mde_primary = 2 * e_bdur_1a$mde_upper, mde_unit = "2SD_differential_bite_elasticity",
    mde_eps_cost = NA_real_, mde_eps_cost_r7 = NA_real_,
    placebo_est = pl_bdur_1a$beta * mean_p_46, placebo_se = pl_bdur_1a$se_cv3j * mean_p_46,
    placebo_p = 2 * pt(-abs(pl_bdur_1a$beta / pl_bdur_1a$se_cv3j), 45L),
    notes = sprintf("1SD_MDE=%.3f; direct-grower business duration x dose + E main, State x FY FE", e_bdur_1a$mde_upper)
  ),
  tibble(
    candidate = "R6_stacked_raw_positions_ln", family = "R6", estimand = "stacked_2009_10_plus_2025_positions_elasticity", unit = "event_county_period",
    n_obs = 11684L, n_units = 2921L, us_hours_share = 0.785, G = 46L,
    G_eff = sup_stk_raw$g_eff, sd_x_tilde = sup_stk_raw$sd_x_tilde_wt, r2_fe = sup_stk_raw$r2_fe_wt,
    first_stage_pi = fs_r6_reg$estimate, first_stage_se = fs_r6_reg$se_cv3j, first_stage_F = (fs_r6_reg$estimate / fs_r6_reg$se_cv3j)^2,
    sy_ms_ratio = NA_real_, n_placebo_draws = e_stk_raw$n_valid,
    emp_size_alpha05 = e_stk_raw$emp_size, size_wilson_hi = e_stk_raw$wilson_hi,
    mde_primary = e_stk_raw$mde_upper, mde_unit = "stacked_positions_ln_elasticity",
    mde_eps_cost = e_stk_raw$mde_upper * COST_RATIO_CENTRAL / 0.6714, mde_eps_cost_r7 = NA_real_,
    placebo_est = NA_real_, placebo_se = NA_real_, placebo_p = NA_real_,
    notes = sprintf("2009-10 Fisher info share=%.1f%% (raw_log), %.1f%% (raw_dollar); 2009-10 pos-wt FS pi=%.3f (F=%.1f); Ep2-only MDE=%.3f",
      100 * info_sh_ep1_log, 100 * info_sh_ep1_dollar, fs_r6_regw$estimate, (fs_r6_regw$estimate/fs_r6_regw$se_cv3j)^2, e_ep2_only$mde_upper)
  ),
  tibble(
    candidate = "R6_stacked_norm_positions_ln", family = "R6", estimand = "stacked_equal_event_wt_positions_elasticity", unit = "event_county_period",
    n_obs = 11684L, n_units = 2921L, us_hours_share = 0.785, G = 46L,
    G_eff = sup_stk_norm$g_eff, sd_x_tilde = sup_stk_norm$sd_x_tilde_wt, r2_fe = sup_stk_norm$r2_fe_wt,
    first_stage_pi = fs_r6_reg$estimate, first_stage_se = fs_r6_reg$se_cv3j, first_stage_F = (fs_r6_reg$estimate / fs_r6_reg$se_cv3j)^2,
    sy_ms_ratio = NA_real_, n_placebo_draws = e_stk_norm$n_valid,
    emp_size_alpha05 = e_stk_norm$emp_size, size_wilson_hi = e_stk_norm$wilson_hi,
    mde_primary = e_stk_norm$mde_upper, mde_unit = "stacked_positions_ln_elasticity",
    mde_eps_cost = e_stk_norm$mde_upper * COST_RATIO_CENTRAL / 0.6714, mde_eps_cost_r7 = NA_real_,
    placebo_est = NA_real_, placebo_se = NA_real_, placebo_p = NA_real_,
    notes = sprintf("Event-size-normalized 2009-10 Fisher info share=%.1f%%; Ep2-only MDE=%.3f", 100 * info_sh_ep1_norm_log, e_ep2_only$mde_upper)
  )
)

write_csv(r4_r6_triage, "explorations/bite_power_rescue/phase1_r4_r6_triage.csv")
cat("\nWrote explorations/bite_power_rescue/phase1_r4_r6_triage.csv\n")
