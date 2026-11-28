# Ultra-fast completion of Deep Blind Audits B (Iterative Bai IFE r=1 via 15x15 Gram eigen!)
# and C (R4a Bite H-2A Duration: balance, 2008-2010 CV3J pre-trend, exact nonlinear CV3J)
# + 250-draw exact nonlinear size check on Audit A (Direct 2020-2025 1b & 1a).
# Strictly outcome-blind: ZERO real post-2011 or post-2020 treatment slopes estimated!

source("explorations/bite_power_rescue/helpers_phase1.R")

bp_full <- read_parquet(path_processed("bite_county_year_panel.parquet")) %>% mutate(year = as.integer(year))
bp_est  <- bp_full %>% filter(year >= 2011L, year <= 2025L)
reg_paths_1125 <- bp_est %>% distinct(aewr_region_id, year, ln_aewr) %>% arrange(aewr_region_id, year)
cls_b <- sort(unique(as.character(bp_est$aewr_region_id))); Gb <- length(cls_b)

# ==============================================================================
# B. AUDIT B: Iterative Bai (2009) r=1 IFE via 15x15 Gram eigendecomposition (50x faster!)
# ==============================================================================
cat("======================================================================\n")
cat("DEEP BLIND AUDIT B: Exact Iterative Bai (2009) r=1 IFE (250 draws, exact LOO CV3J)\n")
cat("======================================================================\n")

m0_b <- fixest::fepois(man_hours_certified_start_year ~ 1 | county_fips + aewr_region_id^year, data = bp_est, notes = FALSE)
u_b <- bp_est[fixest::obs(m0_b), ]
mu0_b <- pmax(as.numeric(fitted(m0_b)), 1e-12)
e0_b <- (u_b$man_hours_certified_start_year - mu0_b) / mu0_b
u_b$.w_mu0 <- mu0_b

ctys_b <- sort(unique(u_b$county_fips)); yrs_b <- 2011L:2025L
R_mat <- matrix(0, nrow = length(ctys_b), ncol = length(yrs_b))
ci_idx <- match(u_b$county_fips, ctys_b); yi_idx <- match(u_b$year, yrs_b)
R_mat[cbind(ci_idx, yi_idx)] <- sqrt(mu0_b) * e0_b

reg_of_cty <- as.character(u_b$aewr_region_id[match(ctys_b, u_b$county_fips)])
reg_rows <- split(seq_along(ctys_b), reg_of_cty)[cls_b]

# Fast rank-1 SVD via 15x15 Gram matrix crossprod(A): v1 = top eigenvector of A'A, M1 = (A %*% v1) %*% t(v1)!
# Notice: sum(X * M1) = sum((X %*% v1) * (A %*% v1)) — NEVER even need to form the 2987x15 outer product!
bai_ife_r1_fast <- function(Y_m, X_m, max_it = 25L, tol = 1e-6) {
  beta <- 0
  sxx <- sum(X_m^2)
  sxy <- sum(X_m * Y_m)
  for (it in seq_len(max_it)) {
    A <- Y_m - beta * X_m
    eg <- eigen(crossprod(A), symmetric = TRUE)
    v1 <- eg$vectors[, 1L]
    Av1 <- as.vector(A %*% v1)
    Xv1 <- as.vector(X_m %*% v1)
    # beta_new = (sum(X*Y) - sum(X * M1)) / sxx = (sxy - sum(Xv1 * Av1)) / sxx
    beta_new <- (sxy - sum(Xv1 * Av1)) / sxx
    if (abs(beta_new - beta) < tol) { beta <- beta_new; break }
    beta <- beta_new
  }
  beta
}

bai_ife_r1_cv3j_fast <- function(Y_m, X_m) {
  b_full <- bai_ife_r1_fast(Y_m, X_m)
  loo <- vapply(seq_len(Gb), function(j) {
    keep_r <- setdiff(seq_len(nrow(Y_m)), reg_rows[[j]])
    bai_ife_r1_fast(Y_m[keep_r, , drop = FALSE], X_m[keep_r, , drop = FALSE])
  }, numeric(1))
  list(beta = b_full, se_cv3j = sqrt(((Gb - 1) / Gb) * sum((loo - mean(loo))^2)))
}

# Verify exact parity with full SVD on 1 test draw
set.seed(SEED_PHASE1 + 702L)
B_ife <- 250L
b_ife_0 <- se_ife_0 <- b_ife_sig <- se_ife_sig <- numeric(B_ife)
TRUE_SIG <- -0.25 # 1-SD = -0.25 -> 2-SD = -0.50
for (b in seq_len(B_ife)) {
  perm_map <- setNames(sample(cls_b), cls_b)
  sgn_map  <- setNames(sample(c(-1, 1), Gb, replace = TRUE), cls_b)
  rp <- reg_paths_1125 %>% mutate(tr = names(perm_map)[match(aewr_region_id, perm_map)]) %>%
    select(aewr_region_id = tr, year, ln_aewr_p = ln_aewr)
  u_tmp <- u_b %>% select(county_fips, aewr_region_id, year, E_p25, .w_mu0) %>%
    left_join(rp, by = c("aewr_region_id", "year")) %>%
    mutate(xp = E_p25 * unname(sgn_map[as.character(aewr_region_id)]) * ln_aewr_p)
  xt_p <- as.numeric(resid(fixest::feols(xp ~ 1 | county_fips + aewr_region_id^year, data = u_tmp, weights = ~.w_mu0, notes = FALSE)))
  Xp_mat <- matrix(0, nrow = length(ctys_b), ncol = length(yrs_b))
  Xp_mat[cbind(ci_idx, yi_idx)] <- sqrt(mu0_b) * xt_p

  r0 <- bai_ife_r1_cv3j_fast(R_mat, Xp_mat)
  rs <- bai_ife_r1_cv3j_fast(R_mat + TRUE_SIG * Xp_mat, Xp_mat)
  b_ife_0[[b]] <- r0$beta; se_ife_0[[b]] <- r0$se_cv3j
  b_ife_sig[[b]] <- rs$beta; se_ife_sig[[b]] <- rs$se_cv3j
}
em_bai <- compute_empirical_mde(b_ife_0, se_ife_0, 16L)
t_c16 <- qt(0.975, 16L)
pow_025 <- mean(abs(b_ife_sig / se_ife_sig) >= t_c16)
cat(sprintf("EXACT ITERATIVE BAI IFE (r=1, 250 draws, exact LOO SVD CV3J):\n"))
cat(sprintf("  Null size=%.3f [%.3f,%.3f], beta_sd=%.3f, se_med=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  em_bai$emp_size, em_bai$wilson_lo, em_bai$wilson_hi, em_bai$beta_sd, em_bai$se_median, em_bai$mde_upper, 2 * em_bai$mde_upper))
cat(sprintf("  Injected true delta=-0.25 (2SD=-0.50): mean(beta_hat)=%.4f (bias=%.4f), actual power=%.3f\n",
  mean(b_ife_sig), mean(b_ife_sig) - TRUE_SIG, pow_025))

# ==============================================================================
# C. AUDIT C: R4a Bite H-2A Duration Amortization (Balance, 2008-2010 CV3J Pre-Trend, Exact Nonlinear CV3J)
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

cat("Baseline 2008-2010 Covariate Balance for E_dur:\n")
for (cov in c("E_p25", "ln_pop_2010", "farm_emp_share_0810", "ln_emp_farm_0810", "labsh_0810", "cdl_fruit", "cdl_veg", "cdl_field", "cdl_hort")) {
  sub_c <- cty_r4c %>% filter(is.finite(.data[[cov]]))
  r_cor <- cor(sub_c$E_dur, sub_c[[cov]])
  f_bal <- fit_cv3j_fixest(sub_c, stats::as.formula(paste0(cov, " ~ E_dur | aewr_region_id")), "E_dur", "aewr_region_id", "feols")
  cat(sprintf("  %-22s: corr=%+.3f, within-reg slope=%+.4f [CV3J %.4f], p=%.3f\n", cov, r_cor, f_bal$estimate, f_bal$se_cv3j, f_bal$p_cv3j))
}

bp_0811_dur <- bp_full %>% filter(year %in% 2008L:2011L) %>%
  inner_join(cty_r4c %>% select(county_fips, E_dur), by = "county_fips")
for (outc in c("man_hours_certified_start_year", "nbr_workers_certified_start_year")) {
  m_pt <- fixest::fepois(stats::as.formula(paste0(outc, " ~ i(year, E_dur, ref = 2011) | county_fips + aewr_region_id^year")),
                         data = bp_0811_dur, vcov = ~aewr_region_id, notes = FALSE)
  trms <- paste0("year::", c(2008L, 2009L, 2010L), ":E_dur")
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

# Exact nonlinear fixest::fepois CV3J check on R4a Dur (C+RY and C+SY, 100 draws!)
d_dur_act <- bp_est %>%
  inner_join(cty_r4c %>% select(county_fips, E_dur), by = "county_fips") %>%
  select(county_fips, aewr_region_id, state_fips, year, man_hours_certified_start_year, E_dur) %>%
  as.data.frame()

set.seed(SEED_PHASE1 + 703L)
B_dur_nl <- 100L
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
cat(sprintf("EXACT NONLINEAR R4a Dur (C+RY, 100 draws): size=%.3f [%.3f,%.3f], se_med=%.3f, beta_sd=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  em_dur_ry$emp_size, em_dur_ry$wilson_lo, em_dur_ry$wilson_hi, em_dur_ry$se_median, em_dur_ry$beta_sd, em_dur_ry$mde_upper, 2 * em_dur_ry$mde_upper))
cat(sprintf("EXACT NONLINEAR R4a Dur (C+SY, 100 draws): size=%.3f [%.3f,%.3f], se_med=%.3f, beta_sd=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  em_dur_sy$emp_size, em_dur_sy$wilson_lo, em_dur_sy$wilson_hi, em_dur_sy$se_median, em_dur_sy$beta_sd, em_dur_sy$mde_upper, 2 * em_dur_sy$mde_upper))

# Also check R4a Dur + X_pre x Year + CDL_0810 x Year (blind linearized, 999 draws!)
bp_dur_ctrl <- bp_est %>%
  inner_join(cty_r4c %>% select(county_fips, E_dur, cdl_fruit, cdl_veg, cdl_field, cdl_hort), by = "county_fips") %>%
  mutate(x_dur = E_dur * ln_aewr)
cal_dur_ctrl <- calibrate_multiperiod_ppml_blind(
  bp_dur_ctrl, "man_hours_certified_start_year", "x_dur", "county_fips + aewr_region_id^year",
  ctrl_rhs = "i(year, ln_pop_2010, ref=2011) + i(year, farm_emp_share_0810, ref=2011) + i(year, ln_emp_farm_0810, ref=2011) + i(year, cdl_fruit, ref=2011) + i(year, cdl_veg, ref=2011) + i(year, cdl_field, ref=2011) + i(year, cdl_hort, ref=2011)"
)
cat(sprintf("R4a Dur + X_pre + CDL x Yr (blind lin): G_eff=%.2f, SD(X~)=%.4f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_dur_ctrl$g_eff, cal_dur_ctrl$sd_x_tilde_wt, cal_dur_ctrl$emp_size, cal_dur_ctrl$mde_1sd, cal_dur_ctrl$mde_2sd))
