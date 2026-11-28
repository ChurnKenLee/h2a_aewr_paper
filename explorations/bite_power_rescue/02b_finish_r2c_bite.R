# Fast completion of R2c Bite (ANCOVA rho / long-run elasticity translation + IFE r=1,2 honest size check)
# and assembly of phase1_r2_triage.csv from R2 results. Strictly outcome-blind!

source("explorations/bite_power_rescue/helpers_phase1.R")

bp_full <- read_parquet(path_processed("bite_county_year_panel.parquet")) %>% mutate(year = as.integer(year))
bp_est  <- bp_full %>% filter(year >= 2011L, year <= 2025L)

cat("=== 1. R2c Bite ANCOVA: Exact Persistence rho & Long-Run Elasticity MDE ===\n")
bp_lag <- bp_full %>%
  arrange(county_fips, year) %>%
  group_by(county_fips) %>%
  mutate(ln_h_l1 = log1p(dplyr::lag(man_hours_certified_start_year, 1L))) %>%
  ungroup() %>%
  filter(year >= 2011L, year <= 2025L)

# Strictly FE-only (ZERO treatment regressor!) to check persistence coefficient rho on ln(1 + H_{t-1})
m0_anc <- fixest::fepois(man_hours_certified_start_year ~ ln_h_l1 | county_fips + aewr_region_id^year, data = bp_lag, notes = FALSE)
rho_hat <- unname(coef(m0_anc)[["ln_h_l1"]])
cat(sprintf("FE-only PPML persistence rho on ln(1 + H_{t-1}) = %.4f (1 - rho = %.4f)\n", rho_hat, 1 - rho_hat))

cal_anc_1b <- calibrate_multiperiod_ppml_blind(bp_lag, "man_hours_certified_start_year", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year", ctrl_rhs = "ln_h_l1")
cal_anc_1a <- calibrate_multiperiod_ppml_blind(bp_lag, "man_hours_certified_start_year", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year", ctrl_rhs = "ln_h_l1 + i(year, E_p25, ref = 2011)")
cat(sprintf("R2c Bite ANCOVA 1b: SR MDE_1SD=%.3f / MDE_2SD=%.3f -> Implied LR MDE_1SD=%.3f / MDE_2SD=%.3f (size=%.3f, SY=%.2f)\n",
  cal_anc_1b$mde_1sd, cal_anc_1b$mde_2sd, cal_anc_1b$mde_1sd / (1 - rho_hat), cal_anc_1b$mde_2sd / (1 - rho_hat), cal_anc_1b$emp_size, cal_anc_1b$sy_ms_ratio))
cat(sprintf("R2c Bite ANCOVA 1a: SR MDE_1SD=%.3f -> Implied LR MDE_1SD=%.3f (size=%.3f)\n",
  cal_anc_1a$mde_1sd, cal_anc_1a$mde_1sd / (1 - rho_hat), cal_anc_1a$emp_size))

cat("\n=== 2. R2c Bite IFE (r=1, r=2): Residual Absorption & Honest Overfitting Check ===\n")
m0_b <- fixest::fepois(man_hours_certified_start_year ~ 1 | county_fips + aewr_region_id^year, data = bp_est, notes = FALSE)
u_b <- bp_est[fixest::obs(m0_b), ]
mu0_b <- pmax(as.numeric(fitted(m0_b)), 1e-12)
e0_b <- (u_b$man_hours_certified_start_year - mu0_b) / mu0_b
u_b$.w_mu0 <- mu0_b

ctys_b <- sort(unique(u_b$county_fips)); yrs_b <- 2011L:2025L
R_mat <- matrix(0, nrow = length(ctys_b), ncol = length(yrs_b), dimnames = list(ctys_b, as.character(yrs_b)))
ci_idx <- match(u_b$county_fips, ctys_b); yi_idx <- match(u_b$year, yrs_b)
R_mat[cbind(ci_idx, yi_idx)] <- sqrt(mu0_b) * e0_b
sv_b <- svd(R_mat)
cat(sprintf("SVD singular values share of FE-only working-residual variance: r=1: %.1f%%, r=1..2: %.1f%%\n",
  100 * sv_b$d[1]^2 / sum(sv_b$d^2), 100 * sum(sv_b$d[1:2]^2) / sum(sv_b$d^2)))

fx_b <- fixest::feols(E_p25_x_ln_aewr ~ 1 | county_fips + aewr_region_id^year, data = u_b, weights = ~.w_mu0, notes = FALSE)
xt_b <- as.numeric(resid(fx_b))
X_mat <- matrix(0, nrow = length(ctys_b), ncol = length(yrs_b))
X_mat[cbind(ci_idx, yi_idx)] <- sqrt(mu0_b) * xt_b

ife_rows <- list()
for (r_fac in c(1L, 2L)) {
  U_r <- sv_b$u[, seq_len(r_fac), drop = FALSE]
  V_r <- sv_b$v[, seq_len(r_fac), drop = FALSE]
  approx_r <- U_r %*% diag(sv_b$d[seq_len(r_fac)], nrow = r_fac) %*% t(V_r)
  e_ife <- e0_b - approx_r[cbind(ci_idx, yi_idx)] / sqrt(mu0_b)
  sy_ife <- compute_sy_noise_ratio_resid(e_ife * sqrt(mu0_b), u_b$aewr_region_id, u_b$state_fips, u_b$year)

  # Exact Bai (2009) tangent-space projection of X orthogonal to U_r and V_r
  X_proj <- X_mat - U_r %*% crossprod(U_r, X_mat) - (X_mat %*% V_r) %*% t(V_r) + U_r %*% (crossprod(U_r, X_mat) %*% V_r) %*% t(V_r)
  xt_ife <- X_proj[cbind(ci_idx, yi_idx)] / sqrt(mu0_b)
  sd_xt_ife <- sqrt(sum(mu0_b * xt_ife^2) / sum(mu0_b))

  cl_b <- as.character(u_b$aewr_region_id); cls_b <- sort(unique(cl_b)); Gb <- length(cls_b)
  H_g <- as.numeric(tapply(mu0_b * xt_ife^2, cl_b, sum)[cls_b]); H_tot <- sum(H_g)
  S_g <- as.numeric(tapply(mu0_b * xt_ife * e_ife, cl_b, sum)[cls_b])
  z_g <- S_g / sqrt(pmax(H_g, 1e-12))

  # (A) Naive fixed-factor cluster bootstrap (treats U_r, V_r as known!)
  set.seed(SEED_PHASE1 + 50L + r_fac)
  bd <- sd_v <- numeric(999L)
  for (b in seq_len(999L)) {
    ib <- sample.int(Gb, Gb, replace = TRUE); xi <- sample(c(-1, 1), Gb, replace = TRUE)
    Ss <- sqrt(H_g) * xi * z_g[ib]
    bd[[b]] <- sum(Ss) / H_tot
    bl <- (sum(Ss) - Ss) / (H_tot - H_g)
    sd_v[[b]] <- sqrt(((Gb - 1) / Gb) * sum((bl - mean(bl))^2))
  }
  em_naive <- compute_empirical_mde(bd, sd_v, df_t = Gb - 1L)

  # (B) HONEST re-estimated IFE size & MDE check: re-fit SVD r factors INSIDE each region-sign-flip draw!
  # Generate null residual matrix R_star by randomly sign-flipping each region's entire 15-year residual block,
  # re-extract top r SVD factors from R_star, compute Bai IFE slope & naive fixed-factor CV3J SE!
  # This directly tests whether IFE with G=17 regions over-fits & distorts empirical size!
  set.seed(SEED_PHASE1 + 60L + r_fac)
  B_hon <- 200L
  reg_of_cty <- u_b$aewr_region_id[match(ctys_b, u_b$county_fips)]
  bd_hon <- sd_hon <- numeric(B_hon)
  for (b in seq_len(B_hon)) {
    xi_reg <- stats::setNames(sample(c(-1, 1), Gb, replace = TRUE), cls_b)
    row_sgn <- as.numeric(xi_reg[as.character(reg_of_cty)])
    R_star <- R_mat * row_sgn
    sv_s <- svd(R_star, nu = r_fac, nv = r_fac)
    Us <- sv_s$u; Vs <- sv_s$v
    Res_s <- R_star - Us %*% diag(sv_s$d[seq_len(r_fac)], nrow = r_fac) %*% t(Vs)
    Xp_s  <- X_mat - Us %*% crossprod(Us, X_mat) - (X_mat %*% Vs) %*% t(Vs) + Us %*% (crossprod(Us, X_mat) %*% Vs) %*% t(Vs)
    h_s <- as.numeric(tapply(rowSums(Xp_s^2), reg_of_cty, sum)[cls_b])
    s_s <- as.numeric(tapply(rowSums(Xp_s * Res_s), reg_of_cty, sum)[cls_b])
    bd_hon[[b]] <- sum(s_s) / sum(h_s)
    bl_s <- (sum(s_s) - s_s) / (sum(h_s) - h_s)
    sd_hon[[b]] <- sqrt(((Gb - 1) / Gb) * sum((bl_s - mean(bl_s))^2))
  }
  em_hon <- compute_empirical_mde(bd_hon, sd_hon, df_t = Gb - 1L)
  g_eff_ife <- 1 / sum((H_g / H_tot)^2)

  cat(sprintf("R2c Bite IFE (r=%d) fixed-factor: G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
    r_fac, g_eff_ife, sd_xt_ife, sy_ife$ms_ratio, em_naive$emp_size, em_naive$mde_upper, 2 * em_naive$mde_upper))
  cat(sprintf("R2c Bite IFE (r=%d) HONEST refit: size=%.3f [%.3f,%.3f], beta_sd=%.3f, se_med=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
    r_fac, em_hon$emp_size, em_hon$wilson_lo, em_hon$wilson_hi, em_hon$beta_sd, em_hon$se_median, em_hon$mde_upper, 2 * em_hon$mde_upper))
  ife_rows[[r_fac]] <- tibble(
    candidate = sprintf("R2c_bite_ife_r%d", r_fac), family = "R2", estimand = sprintf("differential_bite_ife_r%d", r_fac),
    unit = "county_year_bite", n_obs = nrow(u_b), n_units = length(ctys_b), us_hours_share = 1.0,
    G = Gb, G_eff = g_eff_ife, sd_x_tilde = sd_xt_ife, r2_fe = NA_real_,
    first_stage_pi = NA_real_, first_stage_se = NA_real_, first_stage_F = NA_real_,
    sy_ms_ratio = sy_ife$ms_ratio, n_placebo_draws = B_hon,
    emp_size_alpha05 = em_hon$emp_size, size_wilson_hi = em_hon$wilson_hi,
    mde_primary = 2 * em_hon$mde_upper, mde_unit = "2SD_differential_bite_elasticity",
    mde_eps_cost = NA_real_, mde_eps_cost_r7 = NA_real_,
    placebo_est = NA_real_, placebo_se = NA_real_, placebo_p = NA_real_,
    notes = sprintf("Honest SVD-refit 1SD=%.3f/2SD=%.3f (fixed-factor 1SD=%.3f/2SD=%.3f, size=%.3f); SVD captures %.1f%% residual var",
      em_hon$mde_upper, 2 * em_hon$mde_upper, em_naive$mde_upper, 2 * em_naive$mde_upper, em_naive$emp_size,
      100 * sum(sv_b$d[seq_len(r_fac)]^2) / sum(sv_b$d^2))
  )
}

write_csv(bind_rows(ife_rows), "explorations/bite_power_rescue/r2c_bite_ife_audit.csv")
cat("Wrote explorations/bite_power_rescue/r2c_bite_ife_audit.csv\n")
