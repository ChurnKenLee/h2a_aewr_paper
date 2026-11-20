# Identification search, strategy A1 (Phase 4): the falsification tests declared
# in registry.json (falsification.gating F1-F2; falsification.reported_non_gating
# F3-F6), exactly as declared. Gating rule: primary inference (state CRV1,
# t(G - 1)) two-sided p < 0.05 fails. A failure is reported, not repaired.
#
# Usage (repository root, after 01_support.R):
#   Rscript --vanilla explorations/identification_search/A1/04_falsification.R "$S"
# Writes $S/A1/a1_falsification.csv, a1_event_study.csv, a1_event_study_wald.csv,
# a1_balance.csv, a1_f5b_fit.csv, a1_fy2025_covariate_match.csv, a1_falsification.rds.
source("explorations/identification_search/A1/03_helpers.R")
a1_check_registry()
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- a1_out_dir(S)

doses <- a1_doses(S) %>% left_join(a1_ppml_weights(S), by = "county_fips") %>% mutate(ppml_w = coalesce(ppml_w, 0))
cv <- a1_fy2025_covariates(S, doses)
a1_write(cv$match, out, "a1_fy2025_covariate_match.csv"); print(cv$match)
doses <- doses %>% left_join(cv$cov %>% select(-state_fips), by = "county_fips")
rows <- list(); keep <- list()
add <- function(r, key = NULL) { rows[[length(rows) + 1]] <<- r$rows; if (!is.null(key)) keep[[key]] <<- r; invisible(r) }

# ---- F1 gating: FY2024-FY2025 placebo with the FY2024 binding share ----------------
# x = e (state constant, absorbed by state x FY) in FY2024; e + b2024 x (F - e25) in FY2025.
p1 <- a1_panel(S, doses, 2024:2025, "cert_hours_rq13d") %>% mutate(x = e25 + ifelse(year == 2025, d1_h_pl2024, 0))
add(a1_est("F1_placebo_fy2024_fy2025", p1, "cert_hours_rq13d", "x", c("county_fips", "state_year"),
  role = "falsification_gating", note = "placebo dose b2024 x (F - e25) in FY2025"), "F1")

# ---- F2 gating: FY2020-FY2025 linear drift on D_c ----------------------------------
p2 <- a1_panel(S, doses, CAL, "cert_hours_rq13d") %>% mutate(drift = d1_h * (year - 2025))
add(a1_est("F2_pretrend_linear_drift", p2, "cert_hours_rq13d", "drift", c("county_fips", "state_year"),
  role = "falsification_gating", note = "regressor D_c x (FY - 2025), D_c = b_c x (F_c - e25_s)"), "F2")

# ---- F3 event study FY2020-FY2025, reference 2025 ----------------------------------
ev_terms <- paste0("D", 2020:2024)
for (t in 2020:2024) p2[[paste0("D", t)]] <- p2$d1_h * (p2$year == t)
f3 <- add(a1_est("F3_event_study", p2, "cert_hours_rq13d", ev_terms, c("county_fips", "state_year"),
  role = "falsification_reported", note = "D_c x 1(FY = t), reference FY2025"), "F3")
b3 <- coef(f3$m)[ev_terms]; G3 <- n_distinct(f3$dd$state_fips)
Wj <- drop(t(b3) %*% solve(f3$Vj) %*% b3)
Vc <- vcov(f3$m, cluster = ~state_fips)[ev_terms, ev_terms]; Wc <- drop(t(b3) %*% solve(Vc) %*% b3)
ev_wald <- tibble(test = c("joint_jackknife_wald", "joint_state_crv1_wald"), restrictions = 5, wald = c(Wj, Wc),
  p_F = pf(c(Wj, Wc) / 5, 5, G3 - 1, lower.tail = FALSE), p_chi2 = pchisq(c(Wj, Wc), 5, lower.tail = FALSE), clusters = G3,
  note = c("primary joint test (registry): W/5 ~ F(5, G-1)", "CRV1 joint Wald with 5 restrictions, noted as prone to over-reject"))
a1_write(f3$rows %>% mutate(year = as.integer(sub("D", "", term))), out, "a1_event_study.csv")
a1_write(ev_wald, out, "a1_event_study_wald.csv"); print(ev_wald)

# ---- F4 role switch / truncation ---------------------------------------------------
p4 <- a1_panel(S, doses, 2025:2026, "cert_hours_rq13d") %>%
  mutate(post = as.numeric(year == 2026), x = e25 + post * d1_h, btrunc_post = b * trunc * post)
add(a1_est("F4a_truncation_within_state", p4, "cert_hours_rq13d", c("x", "btrunc_post"), c("county_fips", "state_year"),
  role = "falsification_reported", note = "primary model adding b_c x trunc_c x 1(FY2026), trunc = max(0, MW(2026-01-01) - H2A skill I); expected 0"))
p4b <- a1_panel(S, doses %>% filter(multi_region), 2024:2025, "cert_hours_rq13d") %>% mutate(x = trunc_resid * (year == 2025))
add(a1_est("F4b_role_switch_preperiod", p4b, "cert_hours_rq13d", "x", c("county_fips", "region_year"),
  role = "falsification_reported", note = "FY2024-FY2025 placebo, residualized truncation dose in FY2025, county + region x FY, 46 states; expected 0"))

# ---- F5 D2 placebo groups ---------------------------------------------------------
p5a <- a1_panel(S, doses %>% filter(multi_region), 2025:2026, "cert_hours_rq13d") %>%
  mutate(post = as.numeric(year == 2026), d1_post = d1_h * post, nonbind_post = (1 - b) * eff_cut_h * post)
add(a1_est("F5a_above_aewr_cross_state", p5a, "cert_hours_rq13d", c("d1_post", "nonbind_post"), c("county_fips", "region_year"),
  role = "falsification_reported", note = "county + region x FY, 46 states; expected coefficient on (1 - b) x (F - e25) = 0"))
p5 <- p4 %>% mutate(d1_post = d1_h * post, floored = state_fips %in% MW_FLOORED, d1_post_floored = d1_post * floored,
  stat_post = b * stat_cut_h * post)
add(a1_est("F5b_mw_floored_interaction", p5, "cert_hours_rq13d", c("d1_post", "d1_post_floored"), c("county_fips", "state_year"),
  role = "falsification_reported", note = "D_c x 1(MW-floored state: CA, CT, FL, ME, MO, NY, RI, WA); expected interaction 0"))
f5e <- a1_est("F5b_fit_effective_floor_dose", p5, "cert_hours_rq13d", "d1_post", c("county_fips", "state_year"), jk = FALSE,
  role = "falsification_reported", note = "fit comparison: effective-floor dose b x (F - e25)")
f5s <- a1_est("F5b_fit_statutory_dose", p5, "cert_hours_rq13d", "stat_post", c("county_fips", "state_year"), jk = FALSE,
  role = "falsification_reported", note = "fit comparison: statutory dose b x (H-2A skill I - AEWR25)")
add(f5s)
fitcmp <- tibble(spec = c("effective_floor_dose", "statutory_dose"), loglik = c(f5e$rows$loglik, f5s$rows$loglik),
  deviance = c(deviance(f5e$m), deviance(f5s$m)), n = c(f5e$rows$n, f5s$rows$n)) %>%
  mutate(statutory_fits_worse = loglik[spec == "statutory_dose"] < loglik[spec == "effective_floor_dose"])
a1_write(fitcmp, out, "a1_f5b_fit.csv"); print(fitcmp)
wmed <- function(x, w) { o <- order(x); x <- x[o]; w <- w[o]; x[which(cumsum(w) >= sum(w) / 2)[1]] }
cut_pr <- with(doses %>% filter(piece_share_known, h25 > 0), wmed(piece_share, h25))
p5c <- p5 %>% mutate(piece_hi = piece_share_imp > cut_pr, d1_post_piecehi = d1_post * piece_hi)
add(a1_est("F5c_piece_rate_heterogeneity", p5c, "cert_hours_rq13d", c("d1_post", "d1_post_piecehi"), c("county_fips", "state_year"),
  role = "falsification_reported", note = sprintf("interaction with FY2025 piece-rate positions share above the FY2025-hours-weighted median (%.4f); expected attenuation (negative-sign dose: interaction opposite sign to main)", cut_pr)))

# ---- F6 balance: within-state weighted correlation of D_c with pre covariates --------
h <- a1_outcomes(S, c(2020, 2025), c("cert_hours_rq13d", "cert_pos_rq13d", "cert_posweeks_rq13d")) %>%
  tidyr::pivot_wider(names_from = year, values_from = c(cert_hours_rq13d, cert_pos_rq13d, cert_posweeks_rq13d))
bal <- doses %>% filter(ppml_w > 0) %>% left_join(h, by = "county_fips") %>% mutate(
  hours_growth_2020_2025 = log1p(coalesce(cert_hours_rq13d_2025, 0)) - log1p(coalesce(cert_hours_rq13d_2020, 0)),
  contract_weeks_2025 = ifelse(coalesce(cert_pos_rq13d_2025, 0) > 0, cert_posweeks_rq13d_2025 / cert_pos_rq13d_2025, NA_real_),
  field_soc_share_2025 = ifelse(field_share_known, field_share, NA_real_),
  piece_rate_share_2025 = ifelse(piece_share_known, piece_share, NA_real_),
  exp3_share_2025 = ifelse(exp3_share_known, exp3_share, NA_real_),
  log_emp_farm_2011 = log1p(emp_farm_2011))
bal_rows <- lapply(c("hours_growth_2020_2025", "contract_weeks_2025", "field_soc_share_2025", "piece_rate_share_2025",
  "exp3_share_2025", "log_emp_farm_2011"), function(v) {
  z <- bal %>% filter(is.finite(.data[[v]]), is.finite(d1_h))
  fl <- list(factor(z$state_fips))
  r <- fixest::demean(cbind(z[[v]], z$d1_h), f = fl, weights = z$ppml_w, nthreads = 1L)
  w <- z$ppml_w; cr <- sum(w * r[, 1] * r[, 2]) / sqrt(sum(w * r[, 1]^2) * sum(w * r[, 2]^2))
  z$zs <- (z[[v]] - weighted.mean(z[[v]], w)) / sqrt(sum(w * (z[[v]] - weighted.mean(z[[v]], w))^2) / sum(w))
  m <- feols(zs ~ d1_h | state_fips, z, weights = ~ppml_w, vcov = ~state_fips)
  G <- n_distinct(z$state_fips)
  tibble(covariate = v, counties = nrow(z), states = G, within_state_weighted_corr = cr,
    slope_sd_per_dollar = coef(m)[["d1_h"]], se_state_crv1 = se(m)[["d1_h"]], p_state_crv1 = pval_t(coef(m)[["d1_h"]], se(m)[["d1_h"]], G - 1))
})
bal_tab <- bind_rows(bal_rows)
a1_write(bal_tab, out, "a1_balance.csv"); print(bal_tab, width = 200)

fals <- bind_rows(rows) %>% mutate(
  gating = role == "falsification_gating",
  falsification_flag = case_when(gating & p < 0.05 ~ "fail", gating ~ "pass", TRUE ~ "reported"))
a1_write(fals, out, "a1_falsification.csv")
print(fals %>% select(spec_id, term, estimate, se, p, se_jackknife, p_jackknife, n, clusters, falsification_flag), width = 200)
saveRDS(list(rows = fals, ev_wald = ev_wald, balance = bal_tab, f5b_fit = fitcmp, piece_cut = cut_pr), file.path(out, "a1_falsification.rds"))
message("done ", format(Sys.time()))
