# Strictly outcome-blind Phase 1 deep verification of the 3 borderline differential-bite candidates:
#   (A) R2a Direct-Grower (2020-2025) 1b-PPML and 1a-PPML (+ E_p25 x Year FE)
#   (B) R2c Bite IFE (r=1, 2011-2025) with honest cluster-resampled & iterative Bai-IFE CV3J
#   (C) R4a Bite H-2A Duration Amortization (2011-2025) with exact nonlinear CV3J,
#       +CDL/X_pre controls, baseline balance, and 2008-2010 CV3J pre-trend test.
# HARD RULE: NEVER estimates the actual post-2011 or post-2020 treatment slope!

source("explorations/bite_power_rescue/helpers_phase1.R")

bp_full <- read_parquet(path_processed("bite_county_year_panel.parquet")) %>% mutate(year = as.integer(year))
bp_est  <- bp_full %>% filter(year >= 2011L, year <= 2025L)

# ==============================================================================
# A. BORDERLINE CANDIDATE A: R2a Direct-Grower (2020-2025) 1b & 1a-PPML
# ==============================================================================
cat("======================================================================\n")
cat("DEEP BLIND AUDIT A: R2a Direct-Grower 2020-2025 (1b & 1a-PPML)\n")
cat("======================================================================\n")

# Rebuild start-year direct-grower hours 2013-2025 (strictly <= 2025)
disc_h2alc <- open_dataset(path_int("h2a_disclosure_records.parquet")) %>%
  select(fiscal_year, case_number, h2a_labor_contractor, organization_flag, type_of_employer_application) %>%
  filter(fiscal_year >= 2012L, fiscal_year <= 2026L) %>% collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year)) %>%
  group_by(case_number, fiscal_year) %>%
  summarise(
    lc_raw = first(na.omit(as.character(h2a_labor_contractor))),
    org_raw = first(na.omit(as.character(organization_flag))),
    typ_raw = first(na.omit(as.character(type_of_employer_application))), .groups = "drop"
  ) %>%
  mutate(
    is_h2alc = (toupper(trimws(coalesce(lc_raw, ""))) %in% c("Y", "YES", "TRUE", "1")) |
      grepl("LABOR CONTRACTOR|JOB CONTRACTOR", toupper(coalesce(org_raw, ""))) |
      grepl("LABOR CONTRACTOR|JOB CONTRACTOR", toupper(coalesce(typ_raw, "")))
  ) %>%
  select(case_number, fiscal_year, is_h2alc)

nh_pre_mean <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, period_begin_date, number_of_hours) %>%
  filter(fiscal_year >= 2012L, fiscal_year <= 2025L) %>% collect() %>%
  filter(!is.na(as.Date(period_begin_date))) %>%
  summarise(m = mean(suppressWarnings(as.numeric(number_of_hours)), na.rm = TRUE)) %>% pull(m)

ws_sy <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, case_number, county_fips_list, adjusted_nbr_workers_certified, period_begin_date, period_end_date, number_of_hours) %>%
  filter(fiscal_year >= 2012L, fiscal_year <= 2026L) %>% collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year), begin = as.Date(period_begin_date), end = as.Date(period_end_date),
         start_year = as.integer(format(begin, "%Y"))) %>%
  filter(!is.na(begin), !is.na(end), start_year >= 2013L, start_year <= 2025L) %>%
  left_join(disc_h2alc, by = c("case_number", "fiscal_year")) %>%
  mutate(
    is_h2alc = coalesce(is_h2alc, FALSE),
    nh = coalesce(suppressWarnings(as.numeric(number_of_hours)), nh_pre_mean),
    workers = as.numeric(adjusted_nbr_workers_certified),
    hours = pmax(as.numeric(end - begin) / 7, 0) * nh * workers,
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    nc = stringr::str_count(county_fips_list, ",") + 1L
  ) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = harmonize_county_fips_2010(trimws(county_fips_list)), share = 1 / nc) %>%
  group_by(county_fips, year = start_year) %>%
  summarise(
    all_hours_sy = sum(hours * share, na.rm = TRUE),
    direct_hours_sy = sum(hours * share * (!is_h2alc), na.rm = TRUE),
    direct_workers_sy = sum(workers * share * (!is_h2alc), na.rm = TRUE),
    .groups = "drop"
  )

bp_1325 <- bp_est %>% filter(year >= 2013L) %>% left_join(ws_sy, by = c("county_fips", "year")) %>%
  mutate(across(c(all_hours_sy, direct_hours_sy, direct_workers_sy), ~ coalesce(.x, 0)))

bp_2025 <- bp_1325 %>% filter(year >= 2020L)
bp_1319 <- bp_1325 %>% filter(year <= 2019L)

# 1. Check ALL (2020-25) 1a-PPML vs Direct (2020-25) 1a-PPML (is 1a-PPML tighter just because it's 2020-25?)
cal_all20_1a <- calibrate_multiperiod_ppml_blind(bp_2025, "all_hours_sy", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year", ctrl_rhs = "i(year, E_p25, ref = 2020)")
cal_dg20_1a  <- calibrate_multiperiod_ppml_blind(bp_2025, "direct_hours_sy", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year", ctrl_rhs = "i(year, E_p25, ref = 2020)")
cal_dg20_w1b <- calibrate_multiperiod_ppml_blind(bp_2025, "direct_workers_sy", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year")
cal_dg20_w1a <- calibrate_multiperiod_ppml_blind(bp_2025, "direct_workers_sy", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year", ctrl_rhs = "i(year, E_p25, ref = 2020)")
cat(sprintf("All    (2020-25) 1a-PPML (blind lin): G_eff=%.2f, SD(X~)=%.4f, size=%.3f, MDE_1SD=%.3f\n",
  cal_all20_1a$g_eff, cal_all20_1a$sd_x_tilde_wt, cal_all20_1a$emp_size, cal_all20_1a$mde_1sd))
cat(sprintf("Direct (2020-25) 1a-PPML (blind lin): G_eff=%.2f, SD(X~)=%.4f, size=%.3f, MDE_1SD=%.3f\n",
  cal_dg20_1a$g_eff, cal_dg20_1a$sd_x_tilde_wt, cal_dg20_1a$emp_size, cal_dg20_1a$mde_1sd))
cat(sprintf("Direct (2020-25) Pos 1b  (blind lin): G_eff=%.2f, SD(X~)=%.4f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_dg20_w1b$g_eff, cal_dg20_w1b$sd_x_tilde_wt, cal_dg20_w1b$emp_size, cal_dg20_w1b$mde_1sd, cal_dg20_w1b$mde_2sd))
cat(sprintf("Direct (2020-25) Pos 1a  (blind lin): G_eff=%.2f, SD(X~)=%.4f, size=%.3f, MDE_1SD=%.3f\n",
  cal_dg20_w1a$g_eff, cal_dg20_w1a$sd_x_tilde_wt, cal_dg20_w1a$emp_size, cal_dg20_w1a$mde_1sd))

# 2. EXACT NONLINEAR fixest::fepois + EXACT 17-REGION CV3J PLACEBO PERMUTATION CHECK on 2020-2025!
# Strictly outcome-blind: permute the 17 regional ln_aewr paths across regions AND sign-flip E_p25 per region!
# 80 exact nonlinear draws x 18 fits (1 full + 17 LOO) = 1,440 fast 6-yr fepois fits (~15 sec) -> reveals exact nonlinear CV3J SE ratio vs linearized!
reg_paths_2025 <- bp_2025 %>% distinct(aewr_region_id, year, ln_aewr) %>% arrange(aewr_region_id, year)
regs_17 <- sort(unique(bp_2025$aewr_region_id))

# Pre-filter active ever-positive direct-grower counties in 2020-2025
pos_cty_20 <- bp_2025 %>% group_by(county_fips) %>% summarise(s = sum(direct_hours_sy), .groups = "drop") %>% filter(s > 0) %>% pull(county_fips)
d20_act <- bp_2025 %>% filter(county_fips %in% pos_cty_20) %>%
  select(county_fips, aewr_region_id, state_fips, year, direct_hours_sy, E_p25) %>% as.data.frame()

set.seed(SEED_PHASE1 + 701L)
B_nl <- 80L
b_nl_1b <- se_nl_1b <- b_nl_1a <- se_nl_1a <- numeric(B_nl)
for (b in seq_len(B_nl)) {
  # Random derangement/permutation of 17 regional ln_aewr paths + random region sign flip on E_p25
  perm_map <- setNames(sample(regs_17), regs_17)
  sgn_map  <- setNames(sample(c(-1, 1), length(regs_17), replace = TRUE), regs_17)
  rp_b <- reg_paths_2025 %>% mutate(target_reg = names(perm_map)[match(aewr_region_id, perm_map)]) %>%
    select(aewr_region_id = target_reg, year, ln_aewr_p = ln_aewr)
  db <- d20_act %>%
    left_join(rp_b, by = c("aewr_region_id", "year")) %>%
    mutate(Ep = E_p25 * unname(sgn_map[aewr_region_id]), xp = Ep * ln_aewr_p)
  f1b <- fit_cv3j_fixest(db, direct_hours_sy ~ xp | county_fips + aewr_region_id^year, "xp", "aewr_region_id", "fepois")
  f1a <- fit_cv3j_fixest(db, direct_hours_sy ~ xp + i(year, Ep, ref = 2020) | county_fips + aewr_region_id^year, "xp", "aewr_region_id", "fepois")
  b_nl_1b[[b]] <- f1b$estimate; se_nl_1b[[b]] <- f1b$se_cv3j
  b_nl_1a[[b]] <- f1a$estimate; se_nl_1a[[b]] <- f1a$se_cv3j
}
em_nl_1b <- compute_empirical_mde(b_nl_1b, se_nl_1b, 16L)
em_nl_1a <- compute_empirical_mde(b_nl_1a, se_nl_1a, 16L)
cat(sprintf("EXACT NONLINEAR Direct (2020-25) 1b (80 draws): size=%.3f, se_med=%.3f, beta_sd=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  em_nl_1b$emp_size, em_nl_1b$se_median, em_nl_1b$beta_sd, em_nl_1b$mde_upper, 2 * em_nl_1b$mde_upper))
cat(sprintf("EXACT NONLINEAR Direct (2020-25) 1a (80 draws): size=%.3f, se_med=%.3f, beta_sd=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  em_nl_1a$emp_size, em_nl_1a$se_median, em_nl_1a$beta_sd, em_nl_1a$mde_upper, 2 * em_nl_1a$mde_upper))

# 3. Pre-period (2013-2019) event-study pre-trend check for 2020-2025 direct-grower E_p25!
# Strictly on 2013-2019 pre-2020 data (ref = 2019)!
pre_es_20 <- fit_cv3j_fixest(bp_1319, direct_hours_sy ~ i(year, E_p25, ref = 2019) | county_fips + aewr_region_id^year,
                             term = "year::2013:E_p25", cluster_col = "aewr_region_id", estimator = "fepois")
m_es19 <- pre_es_20$fit
trms_1318 <- paste0("year::", 2013L:2018L, ":E_p25")
b_1318 <- coef(m_es19)[trms_1318]; V_1318 <- vcov(m_es19)[trms_1318, trms_1318]
f_1318 <- as.numeric(t(b_1318) %*% solve(V_1318, b_1318)) / 6
cat(sprintf("Direct (2013-2019 pre-2020) E_p25 pre-trend: 2013 vs 2019 = %.3f [CV3J %.3f], p=%.3f; joint 2013-18 CRV1 F(6,16)=%.2f (p=%.3f)\n",
  pre_es_20$estimate, pre_es_20$se_cv3j, pre_es_20$p_cv3j, f_1318, pf(f_1318, 6L, 16L, lower.tail = FALSE)))

# ==============================================================================
# B. BORDERLINE CANDIDATE B: R2c Bite IFE (r=1, 2011-2025) — Honest Size & Attenuation Test
# ==============================================================================
cat("\n======================================================================\n")
cat("DEEP BLIND AUDIT B: R2c Bite IFE (r=1, 2011-2025) — Honest Cluster-Resampled & Iterative Bai IFE\n")
cat("======================================================================\n")

m0_b <- fixest::fepois(man_hours_certified_start_year ~ 1 | county_fips + aewr_region_id^year, data = bp_est, notes = FALSE)
u_b <- bp_est[fixest::obs(m0_b), ]
mu0_b <- pmax(as.numeric(fitted(m0_b)), 1e-12)
e0_b <- (u_b$man_hours_certified_start_year - mu0_b) / mu0_b
u_b$.w_mu0 <- mu0_b

ctys_b <- sort(unique(u_b$county_fips)); yrs_b <- 2011L:2025L
R_mat <- matrix(0, nrow = length(ctys_b), ncol = length(yrs_b), dimnames = list(ctys_b, as.character(yrs_b)))
ci_idx <- match(u_b$county_fips, ctys_b); yi_idx <- match(u_b$year, yrs_b)
R_mat[cbind(ci_idx, yi_idx)] <- sqrt(mu0_b) * e0_b

fx_b <- fixest::feols(E_p25_x_ln_aewr ~ 1 | county_fips + aewr_region_id^year, data = u_b, weights = ~.w_mu0, notes = FALSE)
xt_b <- as.numeric(resid(fx_b))
X_mat <- matrix(0, nrow = length(ctys_b), ncol = length(yrs_b))
X_mat[cbind(ci_idx, yi_idx)] <- sqrt(mu0_b) * xt_b

reg_of_cty <- as.character(u_b$aewr_region_id[match(ctys_b, u_b$county_fips)])
cls_b <- sort(unique(reg_of_cty)); Gb <- length(cls_b)
reg_rows <- split(seq_along(ctys_b), reg_of_cty)[cls_b]

# Exact iterative Bai (2009) r=1 IFE solver on (Y_mat, X_mat):
# Alternates: (1) F = top-1 SVD of (Y - beta*X), (2) beta = <X, Y - F> / <X, X> (with exact tangent projection or ALS!)
# Wait: Bai (2009) minimizes ||Y - beta*X - L F'||_F^2 over (beta, rank-r M):
# Alternating least squares (ALS): given beta, M = SVD_r(Y - beta*X); given M, beta = sum(X*(Y-M)) / sum(X^2)!
# For exact convergence to the true Bai (2009) global minimizer, run 25 ALS iterations from beta=0.
bai_ife_r1 <- function(Y_m, X_m, max_it = 30L, tol = 1e-7) {
  beta <- 0
  sxx <- sum(X_m^2)
  for (it in seq_len(max_it)) {
    sv <- svd(Y_m - beta * X_m, nu = 1L, nv = 1L)
    M1 <- (sv$d[1L] * sv$u[, 1L]) %o% sv$v[, 1L]
    beta_new <- sum(X_m * (Y_m - M1)) / sxx
    if (abs(beta_new - beta) < tol) { beta <- beta_new; break }
    beta <- beta_new
  }
  beta
}

# Exact delete-one-region CV3J of iterative Bai (2009) r=1 IFE!
bai_ife_r1_cv3j <- function(Y_m, X_m) {
  b_full <- bai_ife_r1(Y_m, X_m)
  loo <- vapply(seq_len(Gb), function(j) {
    keep_r <- setdiff(seq_len(nrow(Y_m)), reg_rows[[j]])
    bai_ife_r1(Y_m[keep_r, , drop = FALSE], X_m[keep_r, , drop = FALSE])
  }, numeric(1))
  se_jk <- sqrt(((Gb - 1) / Gb) * sum((loo - mean(loo))^2))
  list(beta = b_full, se_cv3j = se_jk)
}

# Test HONEST Empirical Size & True-Signal Power (injecting delta = 0 AND delta = -0.25!)
# Draw 200 placebo panels by:
#   - Permuting the 17 regional ln_aewr paths + random region sign-flip on E_p25 (REAL Y residuals R_mat untouched, zero real X*Y!)
#   - Running EXACT iterative Bai (2009) r=1 IFE + EXACT delete-one-region CV3J (re-fitting SVD inside every LOO replicate!)
reg_paths_1125 <- bp_est %>% distinct(aewr_region_id, year, ln_aewr) %>% arrange(aewr_region_id, year)
cty_ep25 <- bp_est$E_p25[match(ctys_b, bp_est$county_fips)]
mu0_mat <- matrix(0, nrow = length(ctys_b), ncol = length(yrs_b))
mu0_mat[cbind(ci_idx, yi_idx)] <- mu0_b

set.seed(SEED_PHASE1 + 702L)
B_ife <- 120L # 120 draws x 18 iterative Bai SVD fits = 2,160 SVDs (~15-20s!)
b_ife_0 <- se_ife_0 <- b_ife_sig <- se_ife_sig <- numeric(B_ife)
TRUE_SIG <- -0.25 # 1-SD effect = -0.25 -> 2-SD spread = -0.50!
for (b in seq_len(B_ife)) {
  perm_map <- setNames(sample(cls_b), cls_b)
  sgn_map  <- setNames(sample(c(-1, 1), Gb, replace = TRUE), cls_b)
  # Build placebo X_p matrix (demeaned by county + region x year weighted by mu0!)
  rp <- reg_paths_1125 %>% mutate(tr = names(perm_map)[match(aewr_region_id, perm_map)]) %>%
    select(aewr_region_id = tr, year, ln_aewr_p = ln_aewr)
  u_tmp <- u_b %>% select(county_fips, aewr_region_id, year, E_p25, .w_mu0) %>%
    left_join(rp, by = c("aewr_region_id", "year")) %>%
    mutate(xp = E_p25 * unname(sgn_map[as.character(aewr_region_id)]) * ln_aewr_p)
  xt_p <- as.numeric(resid(fixest::feols(xp ~ 1 | county_fips + aewr_region_id^year, data = u_tmp, weights = ~.w_mu0, notes = FALSE)))
  Xp_mat <- matrix(0, nrow = length(ctys_b), ncol = length(yrs_b))
  Xp_mat[cbind(ci_idx, yi_idx)] <- sqrt(mu0_b) * xt_p

  # Null draw (delta = 0)
  r0 <- bai_ife_r1_cv3j(R_mat, Xp_mat)
  # True-signal draw (delta = -0.25 -> checks unbiasedness & true 80% power under iterative Bai IFE!)
  rs <- bai_ife_r1_cv3j(R_mat + TRUE_SIG * Xp_mat, Xp_mat)
  b_ife_0[[b]] <- r0$beta; se_ife_0[[b]] <- r0$se_cv3j
  b_ife_sig[[b]] <- rs$beta; se_ife_sig[[b]] <- rs$se_cv3j
}
em_bai <- compute_empirical_mde(b_ife_0, se_ife_0, 16L)
t_c16 <- qt(0.975, 16L)
pow_at_025 <- mean(abs(b_ife_sig / se_ife_sig) >= t_c16)
cat(sprintf("EXACT ITERATIVE BAI IFE (r=1, 120 draws, exact LOO SVD CV3J):\n"))
cat(sprintf("  Null size=%.3f [%.3f,%.3f], beta_sd=%.3f, se_med=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  em_bai$emp_size, em_bai$wilson_lo, em_bai$wilson_hi, em_bai$beta_sd, em_bai$se_median, em_bai$mde_upper, 2 * em_bai$mde_upper))
cat(sprintf("  Injected true delta=-0.25 (2SD=-0.50): mean(beta_hat)=%.4f (bias=%.4f), actual power=%.3f\n",
  mean(b_ife_sig), mean(b_ife_sig) - TRUE_SIG, pow_at_025))

# ==============================================================================
# C. BORDERLINE CANDIDATE C: R4a Bite H-2A Duration Amortization (2011-2025)
# ==============================================================================
cat("\n======================================================================\n")
cat("DEEP BLIND AUDIT C: R4a Bite H-2A Duration Amortization (2011-2025)\n")
cat("======================================================================\n")

FIXED_COST_F <- 2500.0
std_within <- function(x, grp) {
  out <- rep(NA_real_, length(x))
  for (g in unique(grp[!is.na(grp)])) {
    idx <- which(grp == g & is.finite(x))
    if (length(idx) >= 2L) { s <- stats::sd(x[idx]); if (is.finite(s) && s > 0) out[idx] <- (x[idx] - mean(x[idx])) / s }
  }
  out
}

ws_0810_dur <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, county_fips_list, adjusted_nbr_workers_certified, period_begin_date, period_end_date) %>%
  filter(fiscal_year %in% c(2008L, 2009L, 2010L)) %>% collect() %>%
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

cdl_0810 <- read_parquet(path_int("croplandcros_county_crop_type_acres.parquet")) %>%
  mutate(county_fips = harmonize_county_fips_2010(county_fips), year = as.integer(year), crop_grp = tolower(trimws(as.character(crop_type_label))), acres = as.numeric(acres)) %>%
  filter(year %in% 2008L:2010L, crop_grp != "non-crop") %>%
  group_by(county_fips) %>%
  summarise(
    tot = sum(acres, na.rm = TRUE),
    cdl_fruit = ifelse(tot > 0, sum(acres[crop_grp == "fruit & tree nuts"], na.rm = TRUE) / tot, 0),
    cdl_veg   = ifelse(tot > 0, sum(acres[crop_grp == "vegetables"], na.rm = TRUE) / tot, 0),
    cdl_field = ifelse(tot > 0, sum(acres[crop_grp == "field crops"], na.rm = TRUE) / tot, 0),
    cdl_hort  = ifelse(tot > 0, sum(acres[crop_grp == "horticulture"], na.rm = TRUE) / tot, 0),
    .groups = "drop"
  )

cty_r4c <- bp_full %>% filter(year %in% c(2008L, 2010L)) %>%
  group_by(county_fips, aewr_region_id, state_fips) %>%
  summarise(
    aewr_0810 = mean(aewr, na.rm = TRUE),
    E_p25 = first(E_p25),
    ln_pop_2010 = first(ln_pop_2010),
    farm_emp_share_0810 = first(farm_emp_share_0810),
    ln_emp_farm_0810 = first(ln_emp_farm_0810),
    labsh_0810 = mean(share_farm_laborexp_prodexp, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  left_join(ws_0810_dur, by = "county_fips") %>%
  left_join(cdl_0810, by = "county_fips") %>%
  mutate(
    s_wage_0810 = ifelse(is.finite(mean_weeks_0810), (aewr_0810 * 40 * mean_weeks_0810) / (aewr_0810 * 40 * mean_weeks_0810 + FIXED_COST_F), NA_real_),
    E_dur = std_within(s_wage_0810, aewr_region_id)
  ) %>%
  filter(is.finite(E_dur))

# 1. Baseline Covariate Balance Check for E_dur (strictly predetermined 2008-2010 covariates!)
cat("Baseline 2008-2010 Covariate Balance for E_dur (correlations & within-region t-stats):\n")
for (cov in c("E_p25", "ln_pop_2010", "farm_emp_share_0810", "ln_emp_farm_0810", "labsh_0810", "cdl_fruit", "cdl_veg", "cdl_field", "cdl_hort")) {
  sub_c <- cty_r4c %>% filter(is.finite(.data[[cov]]))
  r_cor <- cor(sub_c$E_dur, sub_c[[cov]])
  f_bal <- fit_cv3j_fixest(sub_c, stats::as.formula(paste0(cov, " ~ E_dur | aewr_region_id")), "E_dur", "aewr_region_id", "feols")
  cat(sprintf("  %-22s: corr=%+.3f, within-reg slope=%+.4f [CV3J %.4f], p=%.3f\n", cov, r_cor, f_bal$estimate, f_bal$se_cv3j, f_bal$p_cv3j))
}

# 2. Exact 2008-2010 vs 2011 CV3J pre-trend check (strictly on 2008-2011 data ONLY!)
bp_0811_dur <- bp_full %>% filter(year %in% 2008L:2011L) %>%
  inner_join(cty_r4c %>% select(county_fips, E_dur), by = "county_fips")
for (outc in c("man_hours_certified_start_year", "nbr_workers_certified_start_year")) {
  m_pt <- fixest::fepois(stats::as.formula(paste0(outc, " ~ i(year, E_dur, ref = 2011) | county_fips + aewr_region_id^year")),
                         data = bp_0811_dur, vcov = ~aewr_region_id, notes = FALSE)
  trms <- paste0("year::", c(2008L, 2009L, 2010L), ":E_dur")
  # Exact CV3J VCOV on 2008-2010 pre-trend vector!
  used_pt <- bp_0811_dur[fixest::obs(m_pt), ]
  cls_pt <- sort(unique(as.character(used_pt$aewr_region_id))); Gpt <- length(cls_pt)
  loo_m <- matrix(NA_real_, Gpt, 3L)
  for (i in seq_len(Gpt)) {
    mm <- fixest::fepois(stats::as.formula(paste0(outc, " ~ i(year, E_dur, ref = 2011) | county_fips + aewr_region_id^year")),
                         data = bp_0811_dur[bp_0811_dur$aewr_region_id != cls_pt[[i]], ], notes = FALSE)
    loo_m[i, ] <- unname(coef(mm)[trms])
  }
  V_jk <- ((Gpt - 1) / Gpt) * crossprod(sweep(loo_m, 2L, colMeans(loo_m), "-"))
  b_pt <- unname(coef(m_pt)[trms])
  f_jk <- as.numeric(t(b_pt) %*% solve(V_jk, b_pt)) / 3
  cat(sprintf("Pre-trend 2008-2010 (%s): b=[%.3f, %.3f, %.3f], F_CV3J(3,16)=%.3f (p=%.3f)\n",
    outc, b_pt[1], b_pt[2], b_pt[3], f_jk, pf(f_jk, 3L, 16L, lower.tail = FALSE)))
}

# 3. EXACT NONLINEAR fixest::fepois CV3J placebo permutation check for R4a Bite H-2A Duration (80 draws!)
d_dur_act <- bp_est %>%
  inner_join(cty_r4c %>% select(county_fips, E_dur, cdl_fruit, cdl_veg, cdl_field, cdl_hort), by = "county_fips") %>%
  select(county_fips, aewr_region_id, state_fips, year, man_hours_certified_start_year, E_dur, ln_pop_2010, farm_emp_share_0810, ln_emp_farm_0810, cdl_fruit, cdl_veg, cdl_field, cdl_hort) %>%
  as.data.frame()

set.seed(SEED_PHASE1 + 703L)
B_dur_nl <- 80L
b_dur_ry <- se_dur_ry <- b_dur_sy <- se_dur_sy <- numeric(B_dur_nl)
for (b in seq_len(B_dur_nl)) {
  perm_map <- setNames(sample(cls_b), cls_b)
  sgn_map  <- setNames(sample(c(-1, 1), Gb, replace = TRUE), cls_b)
  rp <- reg_paths_1125 %>% mutate(tr = names(perm_map)[match(aewr_region_id, perm_map)]) %>%
    select(aewr_region_id = tr, year, ln_aewr_p = ln_aewr)
  db <- d_dur_act %>% left_join(rp, by = c("aewr_region_id", "year")) %>%
    mutate(xp = E_dur * unname(sgn_map[as.character(aewr_region_id)]) * ln_aewr_p)
  fry <- fit_cv3j_fixest(db, man_hours_certified_start_year ~ xp | county_fips + aewr_region_id^year, "xp", "aewr_region_id", "fepois")
  fsy <- fit_cv3j_fixest(db, man_hours_certified_start_year ~ xp | county_fips + state_fips^year, "xp", "aewr_region_id", "fepois")
  b_dur_ry[[b]] <- fry$estimate; se_dur_ry[[b]] <- fry$se_cv3j
  b_dur_sy[[b]] <- fsy$estimate; se_dur_sy[[b]] <- fsy$se_cv3j
}
em_dur_ry <- compute_empirical_mde(b_dur_ry, se_dur_ry, 16L)
em_dur_sy <- compute_empirical_mde(b_dur_sy, se_dur_sy, 16L)
cat(sprintf("EXACT NONLINEAR R4a Dur (C+RY, 80 draws): size=%.3f, se_med=%.3f, beta_sd=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  em_dur_ry$emp_size, em_dur_ry$se_median, em_dur_ry$beta_sd, em_dur_ry$mde_upper, 2 * em_dur_ry$mde_upper))
cat(sprintf("EXACT NONLINEAR R4a Dur (C+SY, 80 draws): size=%.3f, se_med=%.3f, beta_sd=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  em_dur_sy$emp_size, em_dur_sy$se_median, em_dur_sy$beta_sd, em_dur_sy$mde_upper, 2 * em_dur_sy$mde_upper))
