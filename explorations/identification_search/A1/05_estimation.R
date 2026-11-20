# Identification search, strategy A1 (Phase 5): the declared first stage,
# primary estimate, lead secondary, secondaries, secondary designs SD1-SD6 and
# sensitivities S1-S10 of registry.json, with the declared inference (state
# CRV1 primary by the design-stage rule; state jackknife and former-region CRV1
# alternates) and the declared conversion to epsilon_cost (point (eta / pi_U) x
# C_bar, ITT bound, Fieller/AR set from the leave-one-state-out joint
# covariance of (eta, pi_U), jackknife SE of the ratio).
#
# Usage (repository root, after 04_falsification.R):
#   Rscript --vanilla explorations/identification_search/A1/05_estimation.R "$S"
# Writes $S/A1/a1_estimation_rows.csv, a1_first_stage_frame.csv, a1_conversion.csv,
# a1_decomposition.csv, a1_jackknife_primary.csv, a1_estimation.rds.
source("explorations/identification_search/A1/03_helpers.R")
a1_check_registry()
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- a1_out_dir(S)
FE_W <- c("county_fips", "state_year"); FE_X <- c("county_fips", "region_year")

doses <- a1_doses(S) %>% left_join(a1_ppml_weights(S), by = "county_fips") %>% mutate(ppml_w = coalesce(ppml_w, 0))
cv <- a1_fy2025_covariates(S, doses)
doses <- doses %>% left_join(cv$cov %>% select(-state_fips), by = "county_fips")
ycols <- c("cert_hours_rq13d", "cert_pos_rq13d", "cert_posweeks_rq13d", "req_pos_rq13d", "req_hours_rq13d", "any_cert_rq13d", "cert_hours_dq13")
# Requested counts must be available (non-NA) for every county-year present in the shared panel.
chk <- a1_outcomes(S, 2025:2026, ycols) %>% filter(county_fips %in% doses$county_fips)
stopifnot(!anyNA(chk$req_pos_rq13d), !anyNA(chk$req_hours_rq13d), !anyNA(chk$cert_hours_dq13))
p <- a1_panel(S, doses, 2025:2026, ycols) %>% mutate(post = as.numeric(year == 2026),
  x = e25 + post * d1_h, xU = e25 + post * d1_u, any_cert = as.numeric(any_cert_rq13d))
rows <- list(); keep <- list()
add <- function(r, key = NULL) { rows[[length(rows) + 1]] <<- r$rows; if (!is.null(key)) keep[[key]] <<- r; invisible(r) }

# ---- First stage: offered wage on the U.S.-rate binding dose ------------------------
cr <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"),
  col_select = c("fy_receipt", "in_rq13d", "received", "county_fips", "cert_hours", "hourly_wage")) %>%
  filter(in_rq13d, fy_receipt %in% 2025:2026, cert_hours > 0, is.finite(hourly_wage), hourly_wage > 0, hourly_wage <= 100,
    county_fips %in% doses$county_fips, fy_receipt == 2025 | received >= as.Date("2025-10-02"))
om <- cr %>% group_by(county_fips, year = fy_receipt) %>% summarise(omega = sum(hourly_wage * cert_hours) / sum(cert_hours),
  wage_hours = sum(cert_hours), .groups = "drop")
fs <- p %>% inner_join(om, by = c("county_fips", "year")) %>% group_by(county_fips) %>% filter(n() == 2, all(ppml_w > 0)) %>% ungroup()
a1_write(fs %>% select(county_fips, state_fips, year, omega, wage_hours, ppml_w, x, xU), out, "a1_first_stage_frame.csv")
fsU <- add(a1_est("FS_first_stage_us_rate_dose", fs, "omega", "xU", FE_W, method = "ols", wvar = "ppml_w",
  role = "first_stage", note = "pi_hat_U: offered hourly wage (U.S.-worker rate; FY2026 receipts >= 2025-10-02) on x^U; weights FY2020-25 mean rq13d hours"), "FS_U")
fsH <- add(a1_est("FS_first_stage_h2a_dose", fs, "omega", "x", FE_W, method = "ols", wvar = "ppml_w",
  role = "first_stage_secondary", note = "same model on the primary H-2A cash dose x^H (transparency)"), "FS_H")
pi_U <- fsU$rows$estimate; F_fs <- (pi_U / fsU$rows$se)^2
message(sprintf("pi_hat_U = %.4f (CRV1 se %.4f, jk se %.4f), F = %.1f", pi_U, fsU$rows$se, fsU$rows$se_jackknife, F_fs))

# ---- Primary and lead secondary --------------------------------------------------
pr <- add(a1_est("P_primary", p, "cert_hours_rq13d", "x", FE_W, role = "primary"), "P")
ls <- add(a1_est("LS_req_pos", p, "req_pos_rq13d", "x", FE_W, role = "lead_secondary"), "LS")

# Conversion with the leave-one-state-out joint covariance of (eta, pi_U)
convert <- function(r, label) {
  st <- rownames(r$B); Bfs <- fsU$B
  piv <- setNames(rep(pi_U, length(st)), st); hit <- intersect(st, rownames(Bfs)); piv[hit] <- Bfs[hit, 1]
  J <- cbind(eta = r$B[, 1], pi = piv); J <- J[complete.cases(J), , drop = FALSE]; G <- nrow(J)
  V <- (G - 1) / G * crossprod(sweep(J, 2, colMeans(J)))
  eta <- r$rows$estimate; tq <- qt(0.975, n_distinct(r$dd$state_fips) - 1)
  fl <- a1_fieller(eta, pi_U, V, tq)
  rat <- J[, "eta"] / J[, "pi"]; se_rat <- sqrt((G - 1) / G * sum((rat - mean(rat))^2))
  beta <- eta / pi_U
  tibble(spec_id = label, eta = eta, se_eta_crv1 = r$rows$se, p_eta_crv1 = r$rows$p, se_eta_jk = sqrt(V[1, 1]),
    pi_U = pi_U, se_pi_crv1 = fsU$rows$se, se_pi_jk = sqrt(V[2, 2]), cov_eta_pi_jk = V[1, 2], first_stage_F_crv1 = F_fs,
    weak_first_stage = F_fs < 10, beta = beta, se_beta_jk = se_rat,
    eps_cost = beta * CBAR[["central"]], se_eps_cost_jk = se_rat * CBAR[["central"]],
    fieller_type = fl$type, eps_fieller_low = fl$low * CBAR[["central"]], eps_fieller_high = fl$high * CBAR[["central"]],
    p_ar_beta0_jk = pval_t(eta, sqrt(V[1, 1]), G - 1),
    eps_itt = eta * CBAR[["central"]], eps_itt_ci_low = r$rows$ci_low * CBAR[["central"]], eps_itt_ci_high = r$rows$ci_high * CBAR[["central"]],
    eps_bound_low = min(eta * CBAR[["central"]], beta * CBAR[["central"]]), eps_bound_high = max(eta * CBAR[["central"]], beta * CBAR[["central"]]),
    eps_w = beta * W_BAR, eps_cost_ratio_low = beta * CBAR[["low"]], eps_cost_ratio_high = beta * CBAR[["high"]],
    eps_cost_ratio_extreme = beta * CBAR[["extreme"]],
    realized_mde_eps_cost_jk = 2.8 * se_rat * CBAR[["central"]],
    realized_mde_eps_cost_crv1_point_pi = 2.8 * r$rows$se * CBAR[["central"]] / pi_U,
    realized_mde_eps_itt_crv1 = 2.8 * r$rows$se * CBAR[["central"]], jk_states = G, tq = tq)
}
conv <- bind_rows(convert(pr, "P_primary"), convert(ls, "LS_req_pos"))
a1_write(conv, out, "a1_conversion.csv"); print(as.data.frame(t(conv)))
a1_write(tibble(state_fips = rownames(pr$B), eta_loo = pr$B[, 1]) %>%
  left_join(tibble(state_fips = rownames(fsU$B), pi_U_loo = fsU$B[, 1]), by = "state_fips") %>%
  left_join(tibble(state_fips = rownames(ls$B), eta_req_pos_loo = ls$B[, 1]), by = "state_fips"), out, "a1_jackknife_primary.csv")

# ---- Secondaries ------------------------------------------------------------------
dpos <- add(a1_est("SEC_cert_pos", p, "cert_pos_rq13d", "x", FE_W, role = "secondary",
  note = "also the positions term of the decomposition"), "SEC_pos")
add(a1_est("SEC_req_hours", p, "req_hours_rq13d", "x", FE_W, role = "secondary"))
dpw <- add(a1_est("SEC_decomp_cert_posweeks", p, "cert_posweeks_rq13d", "x", FE_W, role = "secondary"), "D_pw")
dh <- pr  # decomposition hours term = primary fit (P_primary)
# Joint state-CRV1 covariance of the three PPML coefficients: stacked PPML with
# outcome-interacted FE (point estimates identical to the separate fits).
stk <- bind_rows(lapply(c(pos = "cert_pos_rq13d", pw = "cert_posweeks_rq13d", h = "cert_hours_rq13d"), function(v)
  p %>% transmute(county_fips, state_fips, aewr_region_id, state_year, yv = .data[[v]], outc = v, x)), .id = "k") %>%
  mutate(x_pos = x * (k == "pos"), x_pw = x * (k == "pw"), x_h = x * (k == "h"))
ms <- fepois(yv ~ x_pos + x_pw + x_h | county_fips^outc + state_year^outc, stk, vcov = ~state_fips)
Vs <- vcov(ms); bs <- coef(ms); Gs <- n_distinct(stk$state_fips[obs(ms)])
dec <- tibble(spec_id = c("SEC_decomp_log_weeks", "SEC_decomp_log_weekly_hours"),
  estimate = c(bs[["x_pw"]] - bs[["x_pos"]], bs[["x_h"]] - bs[["x_pw"]]),
  se = c(sqrt(Vs["x_pw", "x_pw"] + Vs["x_pos", "x_pos"] - 2 * Vs["x_pw", "x_pos"]), sqrt(Vs["x_h", "x_h"] + Vs["x_pw", "x_pw"] - 2 * Vs["x_h", "x_pw"])),
  se_jackknife = c({ dd <- dpw$B[, 1] - dpos$B[, 1]; G <- length(dd); sqrt((G - 1) / G * sum((dd - mean(dd))^2)) },
    { dd <- dh$B[, 1] - dpw$B[, 1]; G <- length(dd); sqrt((G - 1) / G * sum((dd - mean(dd))^2)) }),
  stacked_check_max_abs_diff = max(abs(c(bs[["x_pos"]] - dpos$rows$estimate, bs[["x_pw"]] - dpw$rows$estimate, bs[["x_h"]] - dh$rows$estimate))),
  clusters = Gs) %>% mutate(p = pval_t(estimate, se, clusters - 1), p_jackknife = pval_t(estimate, se_jackknife, clusters - 1),
    ci_low = estimate - qt(.975, clusters - 1) * se, ci_high = estimate + qt(.975, clusters - 1) * se)
a1_write(dec, out, "a1_decomposition.csv"); print(dec, width = 200)
stopifnot(all(dec$stacked_check_max_abs_diff < 1e-6))
rows[[length(rows) + 1]] <- dec %>% transmute(spec_id, role = "secondary", declared = TRUE,
  outcome = c("log weeks = posweeks/pos", "log weekly hours = hours/posweeks"), term = "x (difference of PPML coefficients)",
  model = "difference of PPML coefficients; joint state CRV1 from stacked PPML with outcome-interacted FE", estimate, se, ci_low, ci_high, p,
  inference_method = "state_fips_CRV1_t(G-1)", se_jackknife, p_jackknife, se_region_crv1 = NA_real_, p_region_crv1 = NA_real_,
  n = dpos$rows$n, units = dpos$rows$units, clusters, region_clusters = NA_integer_, jk_failed_refits = 0L, mean_y = NA_real_,
  loglik = NA_real_, note = "registry decomposition")
ext <- p %>% filter(ppml_w > 0)
add(a1_est("SEC_extensive_any_cert_lpm", ext, "any_cert", "x", FE_W, method = "ols", role = "secondary",
  note = "LPM, counties with any certified rq13d hours FY2020-FY2025, unweighted"))
eq <- p %>% filter(is.finite(emp_farm_2011), emp_farm_2011 > 0) %>% mutate(y_ratio = cert_hours_rq13d / emp_farm_2011)
add(a1_est("SEC_equal_county_ratio", eq, "y_ratio", "x", FE_W, method = "ols", role = "secondary",
  note = "cert_hours_rq13d per 2011 farm worker, OLS levels, unweighted"))

# ---- Secondary designs -------------------------------------------------------------
pm <- p %>% filter(multi_region)
add(a1_est("SD1_cross_state_d1", pm, "cert_hours_rq13d", "x", FE_X, role = "secondary_design", note = "46 states, county + region x FY"))
add(a1_est("SD2_cross_state_effective_floor", pm %>% mutate(xF = ifelse(year == 2025, e25, FH)), "cert_hours_rq13d", "xF", FE_X,
  role = "secondary_design", note = "x = e25 (FY2025), F_c (FY2026); RI by state within region not run: F_c varies within NY and OR (minimum-wage areas), as at the design stage"))
add(a1_est("SD3_cross_state_statutory", pm %>% mutate(xS = ifelse(year == 2025, a25, H2A_s1)), "cert_hours_rq13d", "xS", FE_X,
  role = "secondary_design", note = "x = AEWR25 (FY2025), H-2A skill I (FY2026)"))
add(a1_est("SD4_truncation", pm %>% mutate(xT = post * trunc_resid), "cert_hours_rq13d", "xT", FE_X,
  role = "secondary_design", note = "x = residualized truncation (FY2026), 0 (FY2025)"))
add(a1_est("SD6_us_rate_dose", p, "cert_hours_rq13d", "xU", FE_W, role = "secondary_design", note = "primary model with x^U"))
# SD5 employer x state
eb <- read_parquet(file.path(out, "a1_employer_doses.parquet"))
ce <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"),
  col_select = c("fy_receipt", "in_rq13d", "state_fips", "employer_id_balanced", "cert_hours")) %>%
  filter(in_rq13d, fy_receipt %in% 2025:2026, !is.na(employer_id_balanced)) %>%
  group_by(unit = paste(employer_id_balanced, state_fips), year = fy_receipt) %>% summarise(cert_hours_rq13d = sum(cert_hours, na.rm = TRUE), .groups = "drop")
pe <- tidyr::crossing(eb %>% select(unit, state_fips, aewr_region_id, e25, lvl_d1), year = 2025:2026) %>%
  left_join(ce, by = c("unit", "year")) %>% mutate(cert_hours_rq13d = coalesce(cert_hours_rq13d, 0),
    x = ifelse(year == 2025, e25, lvl_d1), state_year = paste(state_fips, year))
stopifnot(all(pe$cert_hours_rq13d[pe$year == 2025] > 0))
add(a1_est("SD5_employer_within_state", pe, "cert_hours_rq13d", "x", c("unit", "state_year"), role = "secondary_design",
  note = "employer x state units with FY2025 rq13d hours; FY2026 zeros (exit); incumbent response"))

# ---- Sensitivities -----------------------------------------------------------------
add(a1_est("S1_dq13", p %>% mutate(xq = e25 + post * d1_h_dq13), "cert_hours_dq13", "xq", FE_W, role = "sensitivity"))
add(a1_est("S2_b_fy2023_24", p %>% mutate(x2 = e25 + post * d1_h_2324), "cert_hours_rq13d", "x2", FE_W, role = "sensitivity"))
add(a1_est("S3_drop_imputed_b", p %>% filter(b_known), "cert_hours_rq13d", "x", FE_W, role = "sensitivity"))
add(a1_est("S4_mw_blend", p %>% mutate(x4 = e25 + post * d1_h_blend), "cert_hours_rq13d", "x4", FE_W, role = "sensitivity"))
add(a1_est("S5_mw_contract_days", p %>% mutate(x5 = e25 + post * d1_h_cd), "cert_hours_rq13d", "x5", FE_W, role = "sensitivity"))
add(a1_est("S6_46_states", pm, "cert_hours_rq13d", "x", FE_W, role = "sensitivity"))
p7 <- p %>% mutate(g = pmax(H2A_s2, mw_20260101) - pmax(H2A_s1, mw_20260101), tier_post = exp3_share_imp * g * post)
add(a1_est("S7_tier_covariate", p7, "cert_hours_rq13d", "x", FE_W, ctrl = "tier_post", role = "sensitivity",
  note = "control s_c x g_c x 1(FY2026); s_c FY2025 hours share with WORK_EXPERIENCE_MONTHS >= 3 (state share imputed where no FY2025 hours)"))
h23 <- a1_outcomes(S, c(2023, 2025), "cert_hours_rq13d") %>% tidyr::pivot_wider(names_from = year, values_from = cert_hours_rq13d, names_prefix = "h")
p8 <- p %>% left_join(h23, by = "county_fips") %>%
  mutate(pre_ch_post = (log1p(coalesce(h2025, 0)) - log1p(coalesce(h2023, 0))) * post)
add(a1_est("S8_pretrend_adjust", p8, "cert_hours_rq13d", "x", FE_W, ctrl = "pre_ch_post", role = "sensitivity"))
w25 <- a1_outcomes(S, 2025, c("cert_pos_rq13d", "cert_posweeks_rq13d", "cert_hours_rq13d")) %>%
  transmute(county_fips, wk = ifelse(cert_pos_rq13d > 0, cert_posweeks_rq13d / cert_pos_rq13d, NA_real_), h = cert_hours_rq13d)
p9 <- p %>% left_join(w25, by = "county_fips") %>% group_by(state_fips) %>%
  mutate(wk_imp = ifelse(is.finite(wk), wk, sum((h * wk)[is.finite(wk)], na.rm = TRUE) / sum(h[is.finite(wk)], na.rm = TRUE))) %>% ungroup() %>%
  mutate(field_post = field_share_imp * post, weeks_post = wk_imp * post)
add(a1_est("S9_soc_crop", p9, "cert_hours_rq13d", "x", FE_W, ctrl = c("field_post", "weeks_post"), role = "sensitivity",
  note = "controls FY2025 field-SOC hours share x 1(FY2026), FY2025 mean contract weeks x 1(FY2026); state FY2025-hours-weighted values where no FY2025 hours"))
c10 <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"),
  col_select = c("fy_receipt", "in_rq13d", "received", "county_fips", "cert_hours")) %>%
  filter(in_rq13d, fy_receipt %in% 2025:2026, county_fips != "00000")
rebuild <- c10 %>% group_by(county_fips, year = fy_receipt) %>% summarise(h_all = sum(cert_hours, na.rm = TRUE),
  h_no_oct1 = sum(cert_hours[format(received, "%m-%d") != "10-01"], na.rm = TRUE), .groups = "drop")
p10 <- p %>% left_join(rebuild, by = c("county_fips", "year")) %>% mutate(h_all = coalesce(h_all, 0), h_no_oct1 = coalesce(h_no_oct1, 0))
s10chk <- p10 %>% summarise(max_abs_diff_rebuild = max(abs(h_all - cert_hours_rq13d)),
  oct1_hours_2025 = sum((h_all - h_no_oct1)[year == 2025]), oct1_hours_2026 = sum((h_all - h_no_oct1)[year == 2026]))
print(s10chk); stopifnot(s10chk$max_abs_diff_rebuild < 1e-4)
add(a1_est("S10_drop_oct1_receipts", p10, "h_no_oct1", "x", FE_W, role = "sensitivity",
  note = sprintf("outcome rebuilt from case rows excluding receipts dated 1 October (FY2025 %.0f h, FY2026 %.0f h dropped)", s10chk$oct1_hours_2025, s10chk$oct1_hours_2026)))

est <- bind_rows(rows)
a1_write(est, out, "a1_estimation_rows.csv")
print(est %>% select(spec_id, outcome, term, estimate, se, p, se_jackknife, p_jackknife, p_region_crv1, n, units, clusters), n = 60, width = 220)
saveRDS(list(rows = est, conv = conv, pi_U = pi_U, F_fs = F_fs, dec = dec, s10 = s10chk, cov_match = cv$match), file.path(out, "a1_estimation.rds"))
message("done ", format(Sys.time()))
