# Phase 1 Strictly Outcome-Blind Audit: Candidate R3
# (R3a: Balanced Fixed-Cohort Direct-Grower Worksite-Business Panel, FY2020-2025 calibration;
#  R3b: Balanced Within-Multi-State-Employer Cross-State Reallocation, FY2020-2025 calibration)
# HARD RULE: NEVER reads or regresses FY2026 H-2A quantity outcomes!
# Offered-wage first stage for R3b (FY2025-2026) is audited strictly on wage_filer_state_occupation.

source("explorations/bite_power_rescue/helpers_phase1.R")

cat("======================================================================\n")
cat("CANDIDATE R3: Fixed-Cohort Direct-Grower Businesses & Multi-State Filers\n")
cat("======================================================================\n")

# Load statutory 2025 -> 2026 schedules (ALL 48 contiguous states + 46 Track 1 states)
cp_st <- read_parquet(
  path_processed("county_year_panel.parquet"),
  col_select = c("county_fips", "state_fips", "aewr_region_id", "emp_farm_2011", "year")
) %>%
  filter(year == 2011L, !state_fips %in% c("02", "15", "11")) %>%
  mutate(
    county_fips = harmonize_county_fips_2010(county_fips),
    state_fips = state_fips(state_fips),
    aewr_region_id = aewr_region_id(aewr_region_id)
  )

t1_counties <- cp_st %>%
  filter(is.finite(emp_farm_2011), emp_farm_2011 > 0) %>%
  group_by(aewr_region_id) %>%
  filter(n_distinct(state_fips) > 1L) %>%
  ungroup()

st_48_meta <- cp_st %>% distinct(state_fips, aewr_region_id)
st_46_ids <- sort(unique(t1_counties$state_fips))

# Build exact April 1 FY2025 & FY2026 statutory rates for all 48 contiguous states
reg_sched <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>%
  filter(occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference"), worker_category == "covered_workers") %>%
  transmute(state_fips = state_fips(state_fips), aewr = as.numeric(aewr), start = as.Date(effective_start), end = as.Date(effective_end_exclusive))

sw_sched <- read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
  filter(soc_code == "FIELD_LIVESTOCK_COMBINED", trimws(source_sheet) == "Statewide 655.120(b)(1)(i)",
         toupper(trimws(worker_category)) == "US", as.character(skill_level) == "1") %>%
  transmute(state_fips = state_fips(state_fips), aewr = as.numeric(hourly_aewr), start = as.Date(effective_start), end = as.Date(effective_end_exclusive)) %>%
  distinct()

mw_2526 <- read_parquet(path_int("state_real_minwages.parquet")) %>%
  filter(year %in% c(2025L, 2026L)) %>%
  transmute(state_fips = state_fips(state_fips), year = as.integer(year), mw_ag = as.numeric(prevailing_ag_min_wage))

r25_48 <- reg_sched %>% filter(start <= as.Date("2025-04-01"), is.na(end) | end > as.Date("2025-04-01")) %>% select(state_fips, x_2025 = aewr)
r26_48 <- sw_sched  %>% filter(start <= as.Date("2026-04-01"), is.na(end) | end > as.Date("2026-04-01")) %>% select(state_fips, x_2026 = aewr)

st_sched_48 <- st_48_meta %>%
  left_join(r25_48, by = "state_fips") %>%
  left_join(r26_48, by = "state_fips") %>%
  left_join(mw_2526 %>% filter(year == 2025L) %>% select(state_fips, mw25 = mw_ag), by = "state_fips") %>%
  left_join(mw_2526 %>% filter(year == 2026L) %>% select(state_fips, mw26 = mw_ag), by = "state_fips") %>%
  mutate(
    dose_raw = x_2026 - x_2025,
    x_eff_2025 = pmax(x_2025, mw25),
    x_eff_2026 = pmax(x_2026, mw26),
    dose_eff = x_eff_2026 - x_eff_2025,
    dln_raw = log(x_2026) - log(x_2025),
    in_46 = state_fips %in% st_46_ids
  ) %>%
  arrange(state_fips)
stopifnot(nrow(st_sched_48) == 48L, all(is.finite(st_sched_48$dose_raw)))

st_sched_46 <- st_sched_48 %>% filter(in_46)
mean_p_46 <- mean(c(st_sched_46$x_2025, st_sched_46$x_2026))
mean_p_48 <- mean(c(st_sched_48$x_2025, st_sched_48$x_2026))

# Load strictly PRE-POLICY FY2020-FY2025 disclosures, worksites & filer IDs (ZERO FY2026 quantity rows!)
disc_2025 <- open_dataset(path_int("h2a_disclosure_records.parquet")) %>%
  select(fiscal_year, case_number, decision_date, h2a_labor_contractor) %>%
  filter(fiscal_year >= 2020L, fiscal_year <= 2025L) %>%
  collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year)) %>%
  group_by(case_number, fiscal_year) %>%
  summarise(
    dec = as.Date(substr(first(na.omit(as.character(decision_date))), 1L, 10L)),
    lc = first(na.omit(as.character(h2a_labor_contractor))),
    .groups = "drop"
  ) %>%
  filter(!is.na(dec), dec >= as.Date(sprintf("%d-10-01", fiscal_year - 1L)), dec <= as.Date(sprintf("%d-09-30", fiscal_year))) %>%
  mutate(is_h2alc = toupper(trimws(coalesce(lc, ""))) %in% c("Y", "YES", "TRUE", "1"))

nh_pre_mean <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, period_begin_date, number_of_hours) %>%
  filter(fiscal_year >= 2012L, fiscal_year <= 2025L) %>%
  collect() %>%
  filter(!is.na(as.Date(period_begin_date))) %>%
  summarise(m = mean(suppressWarnings(as.numeric(number_of_hours)), na.rm = TRUE)) %>%
  pull(m)

fids_2025 <- read_parquet(path_int("policy_2025", "filer", "filer_ids.parquet")) %>%
  filter(fiscal_year >= 2020L, fiscal_year <= 2025L) %>%
  transmute(fiscal_year = as.integer(fiscal_year), case_number = as.character(case_number), filer_R7 = as.character(filer_R7))

ws_2025 <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(
    fiscal_year, case_number, county_fips_list, adjusted_nbr_workers_certified,
    period_begin_date, period_end_date, number_of_hours,
    employer_is_worksite_business, employer_id_balanced
  ) %>%
  filter(fiscal_year >= 2020L, fiscal_year <= 2025L) %>%
  collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year), begin = as.Date(period_begin_date), end = as.Date(period_end_date)) %>%
  filter(!is.na(begin), !is.na(end)) %>%
  inner_join(disc_2025 %>% select(case_number, fiscal_year, is_h2alc), by = c("case_number", "fiscal_year")) %>%
  left_join(fids_2025, by = c("case_number", "fiscal_year")) %>%
  mutate(
    nh = coalesce(suppressWarnings(as.numeric(number_of_hours)), nh_pre_mean),
    workers = as.numeric(adjusted_nbr_workers_certified),
    hours = pmax(as.numeric(end - begin) / 7, 0) * nh * workers,
    bus_id = ifelse(
      !is.na(employer_is_worksite_business) & as.logical(employer_is_worksite_business) &
        !is.na(employer_id_balanced) & nzchar(trimws(as.character(employer_id_balanced))),
      trimws(as.character(employer_id_balanced)), NA_character_
    ),
    r7_id = ifelse(!is.na(filer_R7) & nzchar(trimws(filer_R7)), trimws(filer_R7), NA_character_),
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    nc = stringr::str_count(county_fips_list, ",") + 1L
  ) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(
    county_fips = harmonize_county_fips_2010(trimws(county_fips_list)),
    state_fips = substr(county_fips, 1L, 2L),
    h = hours / nc,
    w = workers / nc
  )

tot_us_h25 <- sum(ws_2025$h[ws_2025$fiscal_year == 2025L & ws_2025$state_fips %in% st_sched_48$state_fips], na.rm = TRUE)

# ==============================================================================
# PART A: R3a — Balanced Fixed-Cohort Direct-Grower Worksite-Business Panel
# ==============================================================================
cat("\n--- R3a: Fixed-Cohort Direct-Grower Business Panel (FY2020-FY2025) ---\n")

# Restrict to Track 1 multi-state counties (46 states) & worksite businesses & direct growers
wb_dg_yr <- ws_2025 %>%
  filter(!is_h2alc, !is.na(bus_id)) %>%
  inner_join(t1_counties %>% select(county_fips, state_fips_t1 = state_fips, aewr_region_id), by = "county_fips") %>%
  group_by(bus_id, state_fips = state_fips_t1, aewr_region_id, year = fiscal_year) %>%
  summarise(hours = sum(h, na.rm = TRUE), cert = sum(w, na.rm = TRUE), .groups = "drop")

# Assign each business x transition (yr-1 -> yr) to a strictly predetermined baseline cohort:
# Cohort definition: businesses active in yr-2 AND/OR yr-1 (specifically active in yr-1; also test active in BOTH yr-2 & yr-1!).
# Primary baseline state = state with largest hours in yr-1; retain explicit Y = 0 in yr for exits!
build_r3a_cohort <- function(target_yr, require_2yr_base = FALSE) {
  b1 <- wb_dg_yr %>% filter(year == target_yr - 1L, hours > 0)
  if (require_2yr_base) {
    b2_ids <- unique(wb_dg_yr$bus_id[wb_dg_yr$year == target_yr - 2L & wb_dg_yr$hours > 0])
    b1 <- b1 %>% filter(bus_id %in% b2_ids)
  }
  # Predetermined primary state & total baseline hours/positions per business
  base_bus <- b1 %>%
    group_by(bus_id) %>%
    arrange(desc(hours), state_fips, .by_group = TRUE) %>%
    summarise(
      state_fips = first(state_fips),
      aewr_region_id = first(aewr_region_id),
      h0 = sum(hours),
      c0 = sum(cert),
      n_st0 = n_distinct(state_fips),
      .groups = "drop"
    )
  # Year target_yr outcomes for that exact fixed cohort (summed in baseline state, or total firm; check single-state share!)
  y1_bus <- wb_dg_yr %>%
    filter(year == target_yr) %>%
    group_by(bus_id) %>%
    summarise(h1 = sum(hours), c1 = sum(cert), .groups = "drop")

  base_bus %>%
    left_join(y1_bus, by = "bus_id") %>%
    mutate(
      h1 = coalesce(h1, 0),
      c1 = coalesce(c1, 0),
      stay = as.numeric(h1 > 0),
      # Normalized proportional ratio (winsorized at p99 of baseline to prevent micro-farm explosion, or exact ratio)
      ratio_h = h1 / h0,
      ratio_c = c1 / c0,
      # Symmetric Davis-Haltiwanger-Schuh (DHS) growth rate in [-2, +2], handles exact exits (-2) cleanly!
      dhs_h = 2 * (h1 - h0) / (h1 + h0),
      dhs_c = 2 * (c1 - c0) / (c1 + c0)
    )
}

c2425_1yr <- build_r3a_cohort(2025L, require_2yr_base = FALSE)
c2425_2yr <- build_r3a_cohort(2025L, require_2yr_base = TRUE)
cat(sprintf("FY2024->25 R3a 1yr-base cohort: N=%d businesses, single-state=%.1f%%, exit rate=%.1f%%, US hours share=%.3f\n",
  nrow(c2425_1yr), 100 * mean(c2425_1yr$n_st0 == 1L), 100 * mean(c2425_1yr$h1 == 0), sum(c2425_1yr$h1) / tot_us_h25))
cat(sprintf("FY2024->25 R3a 2yr-base cohort: N=%d businesses, single-state=%.1f%%, exit rate=%.1f%%, US hours share=%.3f\n",
  nrow(c2425_2yr), 100 * mean(c2425_2yr$n_st0 == 1L), 100 * mean(c2425_2yr$h1 == 0), sum(c2425_2yr$h1) / tot_us_h25))

# Compute state-year shared-shock ratio at business level on FY2024-25 cohort DHS growth & PPML deviance
sy_r3a_dhs <- compute_sy_noise_ratio_resid(c2425_1yr$dhs_h, c2425_1yr$aewr_region_id, c2425_1yr$state_fips, rep("2025", nrow(c2425_1yr)))
# PPML deviance on 2-period business cohort
g_p <- tapply(c2425_1yr$h1, c2425_1yr$aewr_region_id, sum) / tapply(c2425_1yr$h0 + c2425_1yr$h1, c2425_1yr$aewr_region_id, sum)
mu_b1 <- (c2425_1yr$h0 + c2425_1yr$h1) * as.numeric(g_p[c2425_1yr$aewr_region_id])
sy_r3a_ppml <- compute_sy_noise_ratio_ppml(c2425_1yr$h1, mu_b1, c2425_1yr$aewr_region_id, c2425_1yr$state_fips, rep("2025", nrow(c2425_1yr)))
cat(sprintf("R3a SY shared-shock ratio (business level): PPML=%.2f, DHS=%.2f\n", sy_r3a_ppml$ms_ratio, sy_r3a_dhs$ms_ratio))

# Calibrate R3a across all pre-policy transitions:
# 1yr-base: FY2021..2025 (5 transitions x 120 perms = 600 draws)
# 2yr-base: FY2022..2025 (4 transitions x 150 perms = 600 draws)
cal_r3a <- function(targets, B_per_yr, req_2yr, label_tag) {
  set.seed(SEED_PHASE1 + if (req_2yr) 102L else 101L)
  n_d <- length(targets) * B_per_yr
  b_ph <- se_ph <- b_pc <- se_pc <- b_dhs_h <- se_dhs_h <- b_dhs_c <- se_dhs_c <- b_stay <- se_stay <- numeric(n_d)
  pl_out <- list()
  sup_ph <- sup_dhs <- NULL
  idx <- 0L
  for (yr in targets) {
    coh <- build_r3a_cohort(yr, require_2yr_base = req_2yr)
    # Collapse by state for exact PPML (mathematically identical!) AND exact state-mean OLS for equal-weighted DHS/stay!
    # Wait: for equal-weighted business OLS (y_b ~ dx_s | region_fe), exact algebra gives WLS on state means weighted by N_s!
    st_c <- coh %>%
      group_by(state_fips, aewr_region_id) %>%
      summarise(
        h0 = sum(h0), h1 = sum(h1), c0 = sum(c0), c1 = sum(c1),
        mean_dhs_h = mean(dhs_h), mean_dhs_c = mean(dhs_c), mean_stay = mean(stay),
        n_bus = n(), .groups = "drop"
      )
    ord <- match(st_sched_46$state_fips, st_c$state_fips)
    h0 <- st_c$h0[ord]; h1 <- st_c$h1[ord]; c0 <- st_c$c0[ord]; c1 <- st_c$c1[ord]
    mdh <- st_c$mean_dhs_h[ord]; mdc <- st_c$mean_dhs_c[ord]; ms <- st_c$mean_stay[ord]; nb <- st_c$n_bus[ord]

    if (yr == 2025L) {
      dx_a <- st_sched_46$dose_raw
      pl_out$ppml_h <- fit_2period_ppml_cv3j(h0, h1, dx_a, st_sched_46$aewr_region_id, st_sched_46$state_fips)
      pl_out$ppml_c <- fit_2period_ppml_cv3j(c0, c1, dx_a, st_sched_46$aewr_region_id, st_sched_46$state_fips)
      pl_out$dhs_h  <- fit_state_wls_cv3j(mdh, dx_a, st_sched_46$aewr_region_id, st_sched_46$state_fips, nb)
      pl_out$dhs_c  <- fit_state_wls_cv3j(mdc, dx_a, st_sched_46$aewr_region_id, st_sched_46$state_fips, nb)
      pl_out$stay   <- fit_state_wls_cv3j(ms,  dx_a, st_sched_46$aewr_region_id, st_sched_46$state_fips, nb)
      sup_ph <- compute_2period_blind_support(h0, h1, dx_a, st_sched_46$aewr_region_id, st_sched_46$state_fips)
      # Support for equal-weighted business DHS (weights = n_bus)
      f_d <- fixest::feols(dx ~ 1 | reg, data = data.frame(dx = dx_a, reg = factor(st_sched_46$aewr_region_id), w = nb), weights = ~w, notes = FALSE)
      xt <- resid(f_d); info <- nb * xt^2; sh <- info / sum(info)
      sup_dhs <- list(g_eff = 1 / sum(sh^2), sd_x_tilde_wt = sqrt(sum(nb * xt^2) / sum(nb)))
    }

    for (k in seq_len(B_per_yr)) {
      idx <- idx + 1L
      dx_p <- (st_sched_46 %>% group_by(aewr_region_id) %>% mutate(d = dose_raw[sample.int(n())]) %>% ungroup())$d
      r1 <- fit_2period_ppml_cv3j(h0, h1, dx_p, st_sched_46$aewr_region_id, st_sched_46$state_fips)
      r2 <- fit_2period_ppml_cv3j(c0, c1, dx_p, st_sched_46$aewr_region_id, st_sched_46$state_fips)
      r3 <- fit_state_wls_cv3j(mdh, dx_p, st_sched_46$aewr_region_id, st_sched_46$state_fips, nb)
      r4 <- fit_state_wls_cv3j(mdc, dx_p, st_sched_46$aewr_region_id, st_sched_46$state_fips, nb)
      r5 <- fit_state_wls_cv3j(ms,  dx_p, st_sched_46$aewr_region_id, st_sched_46$state_fips, nb)
      b_ph[[idx]]    <- r1$beta * mean_p_46; se_ph[[idx]]    <- r1$se_cv3j * mean_p_46
      b_pc[[idx]]    <- r2$beta * mean_p_46; se_pc[[idx]]    <- r2$se_cv3j * mean_p_46
      b_dhs_h[[idx]] <- r3$beta * mean_p_46; se_dhs_h[[idx]] <- r3$se_cv3j * mean_p_46
      b_dhs_c[[idx]] <- r4$beta * mean_p_46; se_dhs_c[[idx]] <- r4$se_cv3j * mean_p_46
      b_stay[[idx]]  <- r5$beta * mean_p_46; se_stay[[idx]]  <- r5$se_cv3j * mean_p_46
    }
  }
  out <- list(
    ppml_h = compute_empirical_mde(b_ph, se_ph, 45L),
    ppml_c = compute_empirical_mde(b_pc, se_pc, 45L),
    dhs_h  = compute_empirical_mde(b_dhs_h, se_dhs_h, 45L),
    dhs_c  = compute_empirical_mde(b_dhs_c, se_dhs_c, 45L),
    stay   = compute_empirical_mde(b_stay, se_stay, 45L),
    pl = pl_out, sup_ph = sup_ph, sup_dhs = sup_dhs, tag = label_tag
  )
  for (nm in c("ppml_h", "ppml_c", "dhs_h", "dhs_c", "stay")) {
    e <- out[[nm]]; p <- pl_out[[nm]]
    pel <- p$beta * mean_p_46; pse <- p$se_cv3j * mean_p_46; pp <- 2 * pt(-abs(pel / pse), 45L)
    cat(sprintf("R3a %-8s %-7s | size=%.3f [%.3f,%.3f] | MDE=%.3f (analytic=%.3f) | pl2425=%.3f (p=%.3f)\n",
      label_tag, nm, e$emp_size, e$wilson_lo, e$wilson_hi, e$mde_upper, e$analytic_mde, pel, pp))
  }
  out
}

# Ultra-fast vectorized state WLS + CV3J
fit_state_wls_cv3j <- function(y, dx, reg, st, w) {
  fit_wls_fe_cv3j_fast(y = y, dx = dx, grp = reg, st = st, w = w)
}

r3a_1yr <- cal_r3a(2021L:2025L, 120L, FALSE, "1yr_base")
r3a_2yr <- cal_r3a(2022L:2025L, 150L, TRUE,  "2yr_base")

# ==============================================================================
# PART B: R3b — Balanced Multi-State R7 Filer Cross-State Reallocation
# ==============================================================================
cat("\n--- R3b: Within-Multi-State-Employer Cross-State Reallocation ---\n")

# First verify R3b offered-wage first stage on policy_2025 wage_filer_state_occupation.parquet
# (All multi-state R7 filers, Direct-grower multi-state R7 filers, H-2ALC multi-state R7 filers)
wfso <- read_parquet(path_processed("policy_2025", "wage_filer_state_occupation.parquet")) %>%
  filter(tier == "R7") %>%
  group_by(filer, state_fips, aewr_region_id, fiscal_year) %>%
  summarise(wage = sum(wage * h) / sum(h), contractor = sum(contractor * h, na.rm = TRUE) / sum(h), h = sum(h), .groups = "drop")

pair_r3b_w <- function(tau) {
  b <- wfso %>% filter(fiscal_year == tau - 1L) %>% select(-fiscal_year)
  a <- wfso %>% filter(fiscal_year == tau) %>% select(-fiscal_year)
  inner_join(b, a, by = c("filer", "state_fips", "aewr_region_id"), suffix = c("0", "1")) %>%
    mutate(dw = wage1 - wage0) %>%
    group_by(filer) %>% filter(n_distinct(state_fips) > 1L) %>% ungroup() %>%
    left_join(st_sched_48 %>% select(state_fips, dose_raw, dose_eff), by = "state_fips")
}
r3b_w26 <- pair_r3b_w(2026L)
r3b_w25 <- pair_r3b_w(2025L)

fs_r3b_all    <- fit_cv3j_fixest(r3b_w26, dw ~ dose_raw | filer + aewr_region_id, "dose_raw", "state_fips", "feols", weights_fml = ~h0)
fs_r3b_all_nf <- fit_cv3j_fixest(r3b_w26, dw ~ dose_raw | filer, "dose_raw", "state_fips", "feols", weights_fml = ~h0)
fs_r3b_pl     <- fit_cv3j_fixest(r3b_w25, dw ~ dose_raw | filer + aewr_region_id, "dose_raw", "state_fips", "feols", weights_fml = ~h0)
fs_r3b_dg     <- fit_cv3j_fixest(r3b_w26 %>% filter(contractor0 < 0.5), dw ~ dose_raw | filer + aewr_region_id, "dose_raw", "state_fips", "feols", weights_fml = ~h0)
fs_r3b_lc     <- fit_cv3j_fixest(r3b_w26 %>% filter(contractor0 >= 0.5), dw ~ dose_raw | filer + aewr_region_id, "dose_raw", "state_fips", "feols", weights_fml = ~h0)

cat(sprintf("R3b FS All (filer+reg, h0): pi=%.4f [CV3J %.4f], F=%.2f, N=%d, G=%d (pl2425=%.4f, p=%.3f)\n",
  fs_r3b_all$estimate, fs_r3b_all$se_cv3j, (fs_r3b_all$estimate/fs_r3b_all$se_cv3j)^2, fs_r3b_all$n_obs, fs_r3b_all$G, fs_r3b_pl$estimate, fs_r3b_pl$p_cv3j))
cat(sprintf("R3b FS All (filer only,h0): pi=%.4f [CV3J %.4f], F=%.2f\n",
  fs_r3b_all_nf$estimate, fs_r3b_all_nf$se_cv3j, (fs_r3b_all_nf$estimate/fs_r3b_all_nf$se_cv3j)^2))
cat(sprintf("R3b FS Direct-grower  (h0): pi=%.4f [CV3J %.4f], F=%.2f, N=%d, G=%d\n",
  fs_r3b_dg$estimate, fs_r3b_dg$se_cv3j, (fs_r3b_dg$estimate/fs_r3b_dg$se_cv3j)^2, fs_r3b_dg$n_obs, fs_r3b_dg$G))
cat(sprintf("R3b FS H-2ALC         (h0): pi=%.4f [CV3J %.4f], F=%.2f, N=%d, G=%d\n",
  fs_r3b_lc$estimate, fs_r3b_lc$se_cv3j, (fs_r3b_lc$estimate/fs_r3b_lc$se_cv3j)^2, fs_r3b_lc$n_obs, fs_r3b_lc$G))

# Build balanced multi-state R7 filer x state x FY panel strictly on FY2020-FY2025!
r7_fs_yr <- ws_2025 %>%
  filter(!is.na(r7_id), state_fips %in% st_sched_48$state_fips) %>%
  left_join(st_48_meta, by = "state_fips") %>%
  group_by(r7_id, state_fips, aewr_region_id, year = fiscal_year) %>%
  summarise(
    hours = sum(h, na.rm = TRUE),
    cert = sum(w, na.rm = TRUE),
    h2alc_share = sum(h * is_h2alc, na.rm = TRUE) / pmax(sum(h, na.rm = TRUE), 1e-12),
    .groups = "drop"
  )

build_r3b_balanced <- function(target_yr, states_allowed, direct_only = FALSE) {
  d_sub <- r7_fs_yr %>% filter(state_fips %in% states_allowed)
  # Predetermined baseline set: filers active in >=2 states in target_yr - 1
  b0 <- d_sub %>% filter(year == target_yr - 1L, hours > 0)
  if (direct_only) {
    # Classify filer as direct grower strictly at baseline target_yr - 1
    dg_filers <- b0 %>% group_by(r7_id) %>% summarise(lc_sh = weighted.mean(h2alc_share, hours), .groups = "drop") %>% filter(lc_sh < 0.5) %>% pull(r7_id)
    b0 <- b0 %>% filter(r7_id %in% dg_filers)
  }
  multi_filers <- b0 %>% group_by(r7_id) %>% summarise(ns = n_distinct(state_fips), .groups = "drop") %>% filter(ns >= 2L) %>% pull(r7_id)
  # Balanced cells = all baseline (r7_id, state_fips) pairs of those multi-state filers;
  # ALSO include any state entered in target_yr? NO! Entering a new state in target_yr is post-treatment;
  # Wait: check BOTH (a) baseline active states (exits retained as Y=0!) and (b) ever-active states in (yr-2, yr-1) (predetermined footprint!)
  base_cells <- d_sub %>%
    filter(r7_id %in% multi_filers, year %in% c(target_yr - 1L)) %>%
    select(r7_id, state_fips, aewr_region_id, h0 = hours, c0 = cert)
  y1_cells <- d_sub %>%
    filter(r7_id %in% multi_filers, year == target_yr) %>%
    select(r7_id, state_fips, h1 = hours, c1 = cert)
  base_cells %>%
    left_join(y1_cells, by = c("r7_id", "state_fips")) %>%
    mutate(
      h1 = coalesce(h1, 0),
      c1 = coalesce(c1, 0),
      dhs_h = 2 * (h1 - h0) / (h1 + h0)
    ) %>%
    # Drop filers whose total h1 across all baseline states is 0 (they contribute 0 information to filer x FY FE)
    group_by(r7_id) %>%
    filter(sum(h1) > 0) %>%
    ungroup()
}

# Verify exact equivalence of our binomial-logit solver with group_fe = r7_id against fixest::fepois(hours ~ x | r7_id^state_fips + r7_id^year)
r3b_2425_46 <- build_r3b_balanced(2025L, st_46_ids, direct_only = FALSE)
r3b_2425_48 <- build_r3b_balanced(2025L, st_sched_48$state_fips, direct_only = FALSE)
r3b_2425_dg <- build_r3b_balanced(2025L, st_46_ids, direct_only = TRUE)

cat(sprintf("FY2024->25 R3b (46st, all)   : %d cells, %d multi-state filers, %d states, US hours share=%.3f\n",
  nrow(r3b_2425_46), n_distinct(r3b_2425_46$r7_id), n_distinct(r3b_2425_46$state_fips), sum(r3b_2425_46$h1) / tot_us_h25))
cat(sprintf("FY2024->25 R3b (48st, all)   : %d cells, %d multi-state filers, %d states, US hours share=%.3f\n",
  nrow(r3b_2425_48), n_distinct(r3b_2425_48$r7_id), n_distinct(r3b_2425_48$state_fips), sum(r3b_2425_48$h1) / tot_us_h25))
cat(sprintf("FY2024->25 R3b (46st, direct): %d cells, %d multi-state filers, %d states, US hours share=%.3f\n",
  nrow(r3b_2425_dg), n_distinct(r3b_2425_dg$r7_id), n_distinct(r3b_2425_dg$state_fips), sum(r3b_2425_dg$h1) / tot_us_h25))

# Spot-check exact equivalence against fixest::fepois on FY2024-25 R3b (46st)
long_chk <- bind_rows(
  r3b_2425_46 %>% transmute(r7_id, state_fips, year = 2024L, hours = h0, x = 0),
  r3b_2425_46 %>% left_join(st_sched_46 %>% select(state_fips, dose_raw), by = "state_fips") %>%
    transmute(r7_id, state_fips, year = 2025L, hours = h1, x = dose_raw)
) %>% mutate(cell_id = paste(r7_id, state_fips, sep = "_"))
m_chk_fep <- fixest::fepois(hours ~ x | cell_id + r7_id^year, data = long_chk, vcov = ~state_fips, notes = FALSE)
dx_chk <- st_sched_46$dose_raw[match(r3b_2425_46$state_fips, st_sched_46$state_fips)]
b_chk_bin <- fit_2period_ppml_binom(r3b_2425_46$h0, r3b_2425_46$h1, dx_chk, group_fe = r3b_2425_46$r7_id)
cat(sprintf("R3b exact PPML check: fixest=%.10f vs binom=%.10f (diff=%.2e)\n", coef(m_chk_fep)[["x"]], b_chk_bin, abs(coef(m_chk_fep)[["x"]] - b_chk_bin)))
stopifnot(abs(coef(m_chk_fep)[["x"]] - b_chk_bin) < 1e-8)

# Compute state-year shared-shock ratio AFTER absorbing filer x FY FE!
g_r3b <- tapply(r3b_2425_46$h1, r3b_2425_46$r7_id, sum) / tapply(r3b_2425_46$h0 + r3b_2425_46$h1, r3b_2425_46$r7_id, sum)
mu_r3b <- (r3b_2425_46$h0 + r3b_2425_46$h1) * as.numeric(g_r3b[r3b_2425_46$r7_id])
sy_r3b_46 <- compute_sy_noise_ratio_ppml(r3b_2425_46$h1, mu_r3b, r3b_2425_46$aewr_region_id, r3b_2425_46$state_fips, rep("2025", nrow(r3b_2425_46)))
cat(sprintf("R3b (46st) SY shared-shock ratio AFTER filer x FY FE: %.2f (p=%.3g)\n", sy_r3b_46$ms_ratio, sy_r3b_46$p_val))

# Also check two-way (state + filer) CRV1 vs state CV3J on FY2024-25 placebo
se_twoway_chk <- fixest::se(summary(m_chk_fep, vcov = ~state_fips + r7_id))[["x"]]
se_stcrv1_chk <- fixest::se(summary(m_chk_fep, vcov = ~state_fips))[["x"]]

# Calibrate R3b across all 5 pre-policy transitions FY2020-21 .. FY2024-25 (5 x 100 perms = 500 draws!)
cal_r3b <- function(states_allowed, sched_df, direct_only, label_tag, B_per_yr = 100L) {
  set.seed(SEED_PHASE1 + if (direct_only) 203L else if (nrow(sched_df) == 48L) 202L else 201L)
  targets <- 2021L:2025L
  n_d <- length(targets) * B_per_yr
  mp <- mean(c(sched_df$x_2025, sched_df$x_2026))
  b_h <- se_h <- b_c <- se_c <- b_dhs <- se_dhs <- numeric(n_d)
  pl_h <- pl_c <- pl_dhs <- sup_h <- NULL
  idx <- 0L
  for (yr in targets) {
    bal <- build_r3b_balanced(yr, states_allowed, direct_only = direct_only)
    if (yr == 2025L) {
      dx_a <- sched_df$dose_raw[match(bal$state_fips, sched_df$state_fips)]
      sup_h <- compute_2period_blind_support(bal$h0, bal$h1, dx_a, group_fe = bal$r7_id, cluster_id = bal$state_fips)
      pl_h  <- fit_2period_ppml_cv3j(bal$h0, bal$h1, dx_a, group_fe = bal$r7_id, cluster_id = bal$state_fips)
      pl_c  <- fit_2period_ppml_cv3j(bal$c0, bal$c1, dx_a, group_fe = bal$r7_id, cluster_id = bal$state_fips)
      # DHS equal-weighted cell OLS with filer FE
      df_dhs <- data.frame(y = bal$dhs_h, dx = dx_a, f = bal$r7_id, st = bal$state_fips)
      pl_dhs <- fit_cv3j_fixest(df_dhs, y ~ dx | f, "dx", "st", "feols")
    }
    for (k in seq_len(B_per_yr)) {
      idx <- idx + 1L
      # Permute state doses within former AEWR regions (for 48st, CA & FL are single-state regions so permute across ALL 48 states
      # OR within region; wait: for 48st without Region FE, permuting across all 48 states tests cross-region randomization;
      # for 46st, permute within the 15 former regions!)
      if (nrow(sched_df) == 46L) {
        perm <- sched_df %>% group_by(aewr_region_id) %>% mutate(d = dose_raw[sample.int(n())]) %>% ungroup()
      } else {
        perm <- sched_df %>% mutate(d = dose_raw[sample.int(n())])
      }
      dx_p <- perm$d[match(bal$state_fips, perm$state_fips)]
      rh <- fit_2period_ppml_cv3j(bal$h0, bal$h1, dx_p, group_fe = bal$r7_id, cluster_id = bal$state_fips)
      rc <- fit_2period_ppml_cv3j(bal$c0, bal$c1, dx_p, group_fe = bal$r7_id, cluster_id = bal$state_fips)
      b_h[[idx]] <- rh$beta * mp; se_h[[idx]] <- rh$se_cv3j * mp
      b_c[[idx]] <- rc$beta * mp; se_c[[idx]] <- rc$se_cv3j * mp
    }
  }
  df_t <- nrow(sched_df) - 1L
  eh <- compute_empirical_mde(b_h, se_h, df_t = df_t)
  ec <- compute_empirical_mde(b_c, se_c, df_t = df_t)
  pl_h_el <- pl_h$beta * mp; pl_h_se <- pl_h$se_cv3j * mp; pl_h_p <- 2 * pt(-abs(pl_h_el / pl_h_se), df_t)
  pl_c_el <- pl_c$beta * mp; pl_c_se <- pl_c$se_cv3j * mp; pl_c_p <- 2 * pt(-abs(pl_c_el / pl_c_se), df_t)
  cat(sprintf("R3b %-12s Hours | G_eff=%.2f, SD(X~)=%.3f, size=%.3f [%.3f,%.3f] | MDE=%.3f (analytic=%.3f) | pl2425=%.3f (p=%.3f)\n",
    label_tag, sup_h$g_eff, sup_h$sd_x_tilde_wt, eh$emp_size, eh$wilson_lo, eh$wilson_hi, eh$mde_upper, eh$analytic_mde, pl_h_el, pl_h_p))
  cat(sprintf("R3b %-12s Cert  | size=%.3f [%.3f,%.3f] | MDE=%.3f (analytic=%.3f) | pl2425=%.3f (p=%.3f)\n",
    label_tag, ec$emp_size, ec$wilson_lo, ec$wilson_hi, ec$mde_upper, ec$analytic_mde, pl_c_el, pl_c_p))
  list(eh = eh, ec = ec, sup = sup_h, pl_h = list(est = pl_h_el, se = pl_h_se, p = pl_h_p),
       pl_c = list(est = pl_c_el, se = pl_c_se, p = pl_c_p), pl_dhs = pl_dhs, mp = mp, G = nrow(sched_df))
}

r3b_46_all <- cal_r3b(st_46_ids, st_sched_46, FALSE, "46st_all", B_per_yr = 100L)
r3b_48_all <- cal_r3b(st_sched_48$state_fips, st_sched_48, FALSE, "48st_all", B_per_yr = 100L)
r3b_46_dg  <- cal_r3b(st_46_ids, st_sched_46, TRUE,  "46st_direct", B_per_yr = 100L)

# Also check R3b (46st_all) WITH Former-Region x FY FE alongside filer x FY FE (strictly within former region AND within filer!)
cat("\n--- R3b Sensitivity: Adding Former-Region x FY FE alongside Filer x FY FE ---\n")
# Quick check on FY2024-25 support & placebo via fixest
m_r3b_reg <- fit_cv3j_fixest(long_chk %>% left_join(st_48_meta, by = "state_fips"), hours ~ x | cell_id + r7_id^year + aewr_region_id^year, "x", "state_fips", "fepois")
cat(sprintf("R3b 46st + RegionxFY FE pl2425: el=%.3f [CV3J %.3f], p=%.3f (2.8*SE=%.3f)\n",
  m_r3b_reg$estimate * mean_p_46, m_r3b_reg$se_cv3j * mean_p_46, m_r3b_reg$p_cv3j, 2.8 * m_r3b_reg$se_cv3j * mean_p_46))
cat(sprintf("R3b 46st two-way (state+filer) CRV1 SE=%.4f vs state CRV1 SE=%.4f vs state CV3J SE=%.4f\n",
  se_twoway_chk * mean_p_46, se_stcrv1_chk * mean_p_46, r3b_46_all$pl_h$se))

# Save R3 triage table
make_r3a_row <- function(obj, nm, est_lbl, mde_u, us_sh, n_u) {
  e <- obj[[nm]]; p <- obj$pl[[nm]]
  pel <- p$beta * mean_p_46; pse <- p$se_cv3j * mean_p_46; pp <- 2 * pt(-abs(pel / pse), 45L)
  sup <- if (grepl("ppml", nm)) obj$sup_ph else obj$sup_dhs
  tibble(
    candidate = paste0("R3a_", obj$tag, "_", nm), family = "R3", estimand = est_lbl, unit = "business_fy_cohort",
    n_obs = as.integer(2L * n_u), n_units = as.integer(n_u), us_hours_share = us_sh,
    G = 46L, G_eff = sup$g_eff, sd_x_tilde = sup$sd_x_tilde_wt, r2_fe = NA_real_,
    first_stage_pi = 0.6714, first_stage_se = 0.0988, first_stage_F = 46.18,
    sy_ms_ratio = if (grepl("ppml", nm)) sy_r3a_ppml$ms_ratio else sy_r3a_dhs$ms_ratio,
    n_placebo_draws = e$n_valid, emp_size_alpha05 = e$emp_size, size_wilson_hi = e$wilson_hi,
    mde_primary = e$mde_upper, mde_unit = mde_u,
    mde_eps_cost = (e$mde_upper / mean_p_46) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / 0.6714,
    mde_eps_cost_r7 = (e$mde_upper / mean_p_46) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / 0.6229,
    placebo_est = pel, placebo_se = pse, placebo_p = pp,
    notes = sprintf("Balanced fixed-cohort direct-grower businesses (exits=0); analytic_MDE=%.3f", e$analytic_mde)
  )
}

make_r3b_row <- function(obj, cand, outcome_tag, fs_obj, us_sh, n_cells) {
  e <- if (outcome_tag == "hours") obj$eh else obj$ec
  p <- if (outcome_tag == "hours") obj$pl_h else obj$pl_c
  pi_v <- fs_obj$estimate
  tibble(
    candidate = cand, family = "R3", estimand = paste0("within_multistate_filer_reallocation_", outcome_tag),
    unit = "filer_state_fy", n_obs = as.integer(2L * n_cells), n_units = as.integer(n_cells), us_hours_share = us_sh,
    G = as.integer(obj$G), G_eff = obj$sup$g_eff, sd_x_tilde = obj$sup$sd_x_tilde_wt, r2_fe = obj$sup$r2_fe_wt,
    first_stage_pi = pi_v, first_stage_se = fs_obj$se_cv3j, first_stage_F = (pi_v / fs_obj$se_cv3j)^2,
    sy_ms_ratio = sy_r3b_46$ms_ratio, n_placebo_draws = e$n_valid,
    emp_size_alpha05 = e$emp_size, size_wilson_hi = e$wilson_hi,
    mde_primary = e$mde_upper, mde_unit = paste0("within_firm_reallocation_", outcome_tag, "_elasticity"),
    mde_eps_cost = (e$mde_upper / obj$mp) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / pi_v,
    mde_eps_cost_r7 = (e$mde_upper / obj$mp) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / pi_v,
    placebo_est = p$est, placebo_se = p$se, placebo_p = p$p,
    notes = sprintf("Within-filer cross-state substitution elasticity (NOT aggregate!); filer x FY + filer x state FE; analytic_MDE=%.3f", e$analytic_mde)
  )
}

r3_triage <- bind_rows(
  make_r3a_row(r3a_1yr, "ppml_h", "incumbent_direct_business_hours_ppml", "direct_grower_hours_elasticity", sum(c2425_1yr$h1)/tot_us_h25, nrow(c2425_1yr)),
  make_r3a_row(r3a_1yr, "ppml_c", "incumbent_direct_business_cert_ppml",  "direct_grower_positions_elasticity", sum(c2425_1yr$h1)/tot_us_h25, nrow(c2425_1yr)),
  make_r3a_row(r3a_1yr, "dhs_h",  "incumbent_direct_business_hours_dhs_eqwt", "direct_grower_hours_dhs_elasticity", sum(c2425_1yr$h1)/tot_us_h25, nrow(c2425_1yr)),
  make_r3a_row(r3a_1yr, "dhs_c",  "incumbent_direct_business_cert_dhs_eqwt",  "direct_grower_positions_dhs_elasticity", sum(c2425_1yr$h1)/tot_us_h25, nrow(c2425_1yr)),
  make_r3a_row(r3a_2yr, "ppml_h", "incumbent_2yr_direct_business_hours_ppml", "direct_grower_hours_elasticity", sum(c2425_2yr$h1)/tot_us_h25, nrow(c2425_2yr)),
  make_r3a_row(r3a_2yr, "ppml_c", "incumbent_2yr_direct_business_cert_ppml",  "direct_grower_positions_elasticity", sum(c2425_2yr$h1)/tot_us_h25, nrow(c2425_2yr)),
  make_r3a_row(r3a_2yr, "dhs_h",  "incumbent_2yr_direct_business_hours_dhs_eqwt", "direct_grower_hours_dhs_elasticity", sum(c2425_2yr$h1)/tot_us_h25, nrow(c2425_2yr)),
  make_r3b_row(r3b_46_all, "R3b_46st_all_hours", "hours", fs_r3b_all, sum(r3b_2425_46$h1)/tot_us_h25, nrow(r3b_2425_46)),
  make_r3b_row(r3b_46_all, "R3b_46st_all_cert",  "cert",  fs_r3b_all, sum(r3b_2425_46$h1)/tot_us_h25, nrow(r3b_2425_46)),
  make_r3b_row(r3b_48_all, "R3b_48st_all_hours", "hours", fs_r3b_all_nf, sum(r3b_2425_48$h1)/tot_us_h25, nrow(r3b_2425_48)),
  make_r3b_row(r3b_48_all, "R3b_48st_all_cert",  "cert",  fs_r3b_all_nf, sum(r3b_2425_48$h1)/tot_us_h25, nrow(r3b_2425_48)),
  make_r3b_row(r3b_46_dg,  "R3b_46st_direct_hours", "hours", fs_r3b_dg, sum(r3b_2425_dg$h1)/tot_us_h25, nrow(r3b_2425_dg)),
  make_r3b_row(r3b_46_dg,  "R3b_46st_direct_cert",  "cert",  fs_r3b_dg, sum(r3b_2425_dg$h1)/tot_us_h25, nrow(r3b_2425_dg))
)

write_csv(r3_triage, "explorations/bite_power_rescue/phase1_r3_triage.csv")
cat("\nWrote explorations/bite_power_rescue/phase1_r3_triage.csv\n")
