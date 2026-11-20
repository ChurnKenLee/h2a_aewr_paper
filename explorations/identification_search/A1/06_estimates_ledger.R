# Identification search, strategy A1 (Phases 4-5): assemble $S/A1/estimates.csv
# (one row per declared quantity) and write the estimation fields of
# explorations/identification_search/A1/ledger_row.json (declarations.ledger).
#
# Usage (repository root, after 04_falsification.R, 05_estimation.R and 07_*):
#   Rscript --vanilla explorations/identification_search/A1/06_estimates_ledger.R "$S"
source("explorations/identification_search/A1/03_helpers.R")
a1_check_registry()
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- a1_out_dir(S)
F <- readRDS(file.path(out, "a1_falsification.rds")); E <- readRDS(file.path(out, "a1_estimation.rds"))
pi_U <- E$pi_U; cv <- E$conv; Cc <- CBAR[["central"]]
weak <- E$F_fs < 10

# ---- Rows from fits ---------------------------------------------------------------
base <- bind_rows(F$rows, E$rows %>% mutate(gating = FALSE, falsification_flag = NA_character_))
ppml_doses <- c("x", "xU", "xq", "x2", "x4", "x5")  # binding-weighted floor doses (A1 dose family)
conv_rows <- base %>% mutate(
  ppml = grepl("^PPML", model),
  levels = grepl("^(OLS|WLS)", model) & !grepl("^FS_", spec_id),
  semi = case_when(ppml ~ estimate, levels ~ estimate / mean_y, TRUE ~ NA_real_),
  semi_lo = case_when(ppml ~ ci_low, levels ~ ci_low / mean_y, TRUE ~ NA_real_),
  semi_hi = case_when(ppml ~ ci_high, levels ~ ci_high / mean_y, TRUE ~ NA_real_),
  in_family = role %in% c("primary", "lead_secondary", "secondary", "sensitivity") |
    spec_id %in% c("SD1_cross_state_d1", "SD5_employer_within_state", "SD6_us_rate_dose"),
  eps_itt = ifelse(role %in% c("falsification_gating", "falsification_reported"), NA_real_, semi * Cc),
  eps_itt_ci_low = ifelse(is.na(eps_itt), NA_real_, semi_lo * Cc), eps_itt_ci_high = ifelse(is.na(eps_itt), NA_real_, semi_hi * Cc),
  eps_cost_point = ifelse(in_family & !is.na(eps_itt), eps_itt / pi_U, NA_real_),
  conversion = case_when(
    grepl("^FS_", spec_id) ~ "first stage (pi); no conversion",
    role %in% c("falsification_gating", "falsification_reported") ~ "falsification; no conversion",
    in_family & ppml ~ "eps_itt = eta x C_bar (pi = 1); eps_cost_point = (eta / pi_U) x C_bar, C_bar = 1.2401 x 15.2607 = 18.924",
    in_family & levels ~ "eps_itt = (beta / y_bar) x C_bar (y_bar = estimation-sample mean, both FY); eps_cost_point = eps_itt / pi_U",
    spec_id %in% c("SEC_decomp_log_weeks", "SEC_decomp_log_weekly_hours") ~ "eps_itt = difference x C_bar; eps_cost_point = eps_itt / pi_U",
    ppml ~ "eps_itt = eta x C_bar per dollar of this design's dose (pass-through for this dose not estimated: pi = 1 bound only)",
    TRUE ~ NA_character_),
  eps_itt = ifelse(spec_id %in% c("SEC_decomp_log_weeks", "SEC_decomp_log_weekly_hours"), estimate * Cc, eps_itt),
  eps_itt_ci_low = ifelse(spec_id %in% c("SEC_decomp_log_weeks", "SEC_decomp_log_weekly_hours"), ci_low * Cc, eps_itt_ci_low),
  eps_itt_ci_high = ifelse(spec_id %in% c("SEC_decomp_log_weeks", "SEC_decomp_log_weekly_hours"), ci_high * Cc, eps_itt_ci_high),
  eps_cost_point = ifelse(spec_id %in% c("SEC_decomp_log_weeks", "SEC_decomp_log_weekly_hours"), eps_itt / pi_U, eps_cost_point),
  eps_cost_ci_low = NA_real_, eps_cost_ci_high = NA_real_,
  eps_cost_headline = ifelse(is.na(eps_cost_point), NA, !weak)) %>%
  select(-ppml, -levels, -semi, -semi_lo, -semi_hi, -in_family)

# ---- Conversion rows (primary and lead secondary) -------------------------------------
crow <- function(c, base_id, outc, role) {
  r <- function(id, est, se, lo, hi, p, meth, conv, note = "") tibble(spec_id = paste0(base_id, id), role = role, declared = TRUE,
    outcome = outc, term = "epsilon_cost", model = "conversion of the PPML semi-elasticity", estimate = est, se = se, ci_low = lo, ci_high = hi,
    p = p, inference_method = meth, conversion = conv, n = NA_integer_, clusters = c$jk_states, note = note,
    gating = FALSE, falsification_flag = NA_character_, eps_cost_headline = NA)
  bind_rows(
    r("_eps_cost", c$eps_cost, c$se_eps_cost_jk, c$eps_fieller_low, c$eps_fieller_high, c$p_ar_beta0_jk,
      sprintf("Fieller/AR set (%s) from leave-one-state-out joint covariance of (eta, pi_U), t(0.975, G-1); SE = jackknife SE of the ratio x C_bar; p = AR test of beta = 0", c$fieller_type),
      "(eta / pi_U) x C_bar, central ratio 1.2401",
      if (c$weak_first_stage) sprintf("WEAK FIRST STAGE (F = %.2f < 10): registry rule -> epsilon_cost reported only with the Fieller/AR set; point value not a headline", c$first_stage_F_crv1) else ""),
    r("_eps_itt_bound", c$eps_itt, c$se_eta_crv1 * Cc, c$eps_itt_ci_low, c$eps_itt_ci_high, c$p_eta_crv1, "state_fips_CRV1_t(G-1) on eta, scaled by C_bar",
      "ITT bound eta x C_bar (pi = 1)"),
    r("_eps_cost_interval_itt_to_iv", NA_real_, NA_real_, c$eps_bound_low, c$eps_bound_high, NA_real_, "point interval [eta x C_bar, eta x C_bar / pi_hat]",
      "registry: interval between the ITT bound and the pass-through-adjusted point"),
    r("_eps_w", c$eps_w, c$se_beta_jk * W_BAR, NA_real_, NA_real_, NA_real_, "jackknife SE", "epsilon_w = epsilon_cost / ratio_A1 = (eta / pi_U) x w_bar"),
    r("_eps_cost_ratio_low", c$eps_cost_ratio_low, c$se_beta_jk * CBAR[["low"]], ifelse(is.finite(c$eps_fieller_low), c$eps_fieller_low / Cc * CBAR[["low"]], -Inf),
      ifelse(is.finite(c$eps_fieller_high), c$eps_fieller_high / Cc * CBAR[["low"]], Inf), c$p_ar_beta0_jk, "Fieller/AR set scaled", "(eta / pi_U) x 1.2145 x w_bar"),
    r("_eps_cost_ratio_high", c$eps_cost_ratio_high, c$se_beta_jk * CBAR[["high"]], ifelse(is.finite(c$eps_fieller_low), c$eps_fieller_low / Cc * CBAR[["high"]], -Inf),
      ifelse(is.finite(c$eps_fieller_high), c$eps_fieller_high / Cc * CBAR[["high"]], Inf), c$p_ar_beta0_jk, "Fieller/AR set scaled", "(eta / pi_U) x 1.3834 x w_bar"),
    r("_eps_cost_ratio_extreme", c$eps_cost_ratio_extreme, c$se_beta_jk * CBAR[["extreme"]], ifelse(is.finite(c$eps_fieller_low), c$eps_fieller_low / Cc * CBAR[["extreme"]], -Inf),
      ifelse(is.finite(c$eps_fieller_high), c$eps_fieller_high / Cc * CBAR[["extreme"]], Inf), c$p_ar_beta0_jk, "Fieller/AR set scaled", "(eta / pi_U) x 1.76 x w_bar (extreme sensitivity)"),
    r("_realized_mde", c$realized_mde_eps_cost_jk, NA_real_, NA_real_, NA_real_, NA_real_, "2.8 x jackknife SE of epsilon_cost",
      "realized MDE in epsilon_cost units", sprintf("also: 2.8 x CRV1 SE(eta) x C_bar / pi_hat = %.3f; ITT (pi = 1) 2.8 x CRV1 SE(eta) x C_bar = %.3f",
        c$realized_mde_eps_cost_crv1_point_pi, c$realized_mde_eps_itt_crv1)))
}
cr_p <- crow(cv %>% filter(spec_id == "P_primary"), "P_primary", "cert_hours_rq13d", "primary_conversion")
cr_l <- crow(cv %>% filter(spec_id == "LS_req_pos"), "LS_req_pos", "req_pos_rq13d", "lead_secondary_conversion")
fsF <- tibble(spec_id = "FS_first_stage_F", role = "first_stage", declared = TRUE, outcome = "omega", term = "xU",
  model = "first-stage F from the primary inference method (state CRV1): (pi / se)^2", estimate = E$F_fs, clusters = 48L,
  inference_method = "state_fips_CRV1", conversion = "weak-first-stage rule: F < 10 -> epsilon_cost only with the Fieller/AR set",
  note = ifelse(weak, "F < 10: weak-first-stage rule triggered", "F >= 10"), gating = FALSE)

# ---- Falsification extras ---------------------------------------------------------
f3w <- F$ev_wald %>% transmute(spec_id = paste0("F3_", test), role = "falsification_reported", declared = TRUE, outcome = "cert_hours_rq13d",
  term = "D2020..D2024 joint", model = "Wald, 5 restrictions", estimate = wald, p = p_F, clusters,
  inference_method = ifelse(grepl("jackknife", test), "state jackknife covariance; W/5 ~ F(5, G-1)", "state CRV1 covariance; W/5 ~ F(5, G-1)"),
  note = note, gating = FALSE, falsification_flag = "reported")
f5f <- F$f5b_fit %>% transmute(spec_id = paste0("F5b_fit_loglik_", spec), role = "falsification_reported", declared = TRUE, outcome = "cert_hours_rq13d",
  term = "log-likelihood", model = "PPML, county + state x FY FE", estimate = loglik, n,
  note = paste0("statutory dose fits worse: ", statutory_fits_worse), gating = FALSE, falsification_flag = "reported")
f6 <- F$balance %>% transmute(spec_id = paste0("F6_balance_", covariate), role = "falsification_reported", declared = TRUE, outcome = covariate,
  term = "within-state weighted correlation with D_c", model = "state FE, PPML weights", estimate = within_state_weighted_corr,
  p = p_state_crv1, n = counties, clusters = states,
  inference_method = "p: slope of standardized covariate on D_c, state FE, PPML weights, state CRV1 t(G-1)",
  note = sprintf("slope %.4f SD per dollar (se %.4f)", slope_sd_per_dollar, se_state_crv1), gating = FALSE, falsification_flag = "reported")

est <- bind_rows(conv_rows, cr_p, cr_l, fsF, f3w, f5f, f6) %>%
  mutate(declared = ifelse(declared %in% TRUE, "declared", "POST-OUTCOME"),
    order = match(role, c("falsification_gating", "falsification_reported", "first_stage", "first_stage_secondary", "primary", "primary_conversion",
      "lead_secondary", "lead_secondary_conversion", "secondary", "secondary_design", "sensitivity"))) %>%
  arrange(order) %>%
  select(spec_id, role, declared, outcome, term, model, estimate, se, ci_low, ci_high, p, inference_method,
    se_jackknife, p_jackknife, se_region_crv1, p_region_crv1, n, units, clusters, region_clusters, mean_y,
    conversion, eps_itt, eps_itt_ci_low, eps_itt_ci_high, eps_cost_point, eps_cost_headline, gating, falsification_flag, note)
a1_write(est, out, "estimates.csv")
message("estimates.csv rows: ", nrow(est))

# ---- Ledger row --------------------------------------------------------------------
lp <- file.path(A1_DIR, "ledger_row.json")
L <- jsonlite::read_json(lp, simplifyVector = FALSE)
P <- cv %>% filter(spec_id == "P_primary"); fr <- F$rows
g <- fr %>% filter(gating)
fal_pass <- all(g$falsification_flag == "pass")
fmt <- function(x, d = 4) formatC(x, digits = d, format = "fg", flag = "#")
L$implementer <- "Claude Opus 5.5 (Phase 3 A1 designer; Phase 4-5 estimator)"
L$phase_reached <- "estimation"
L$status <- "estimated"
L$conversion$first_stage_pass_through <- pi_U
L$conversion$first_stage_se <- P$se_pi_crv1
L$conversion$first_stage_detail <- sprintf("pi_hat_U = %s (state CRV1 se %s, jackknife se %s), F = %s < 10: declared weak-first-stage rule triggered. Secondary first stage on x^H: %s (se %s). 2,832 counties, 48 states.",
  fmt(pi_U), fmt(P$se_pi_crv1), fmt(P$se_pi_jk), fmt(E$F_fs, 3), fmt(E$rows$estimate[E$rows$spec_id == "FS_first_stage_h2a_dose"]),
  fmt(E$rows$se[E$rows$spec_id == "FS_first_stage_h2a_dose"]))
L$falsification_result <- sprintf(paste0("%s. Gating: F1 FY2024-25 placebo eta = %s (state CRV1 se %s, p = %s); F2 FY2020-25 linear drift = %s (se %s, p = %s). ",
  "Reported: F3 event study joint jackknife Wald p = %s (CRV1 p = %s); F4a truncation b x trunc = %s (p = %s); F4b pre-period role switch = %s (p = %s); ",
  "F5a (1 - b) term = %s (p = %s); F5b MW-floored interaction = %s (p = %s), statutory dose fits worse (loglik %s vs %s); F5c piece-rate interaction = %s (p = %s); ",
  "F6 balance: within-state weighted corr of D_c with FY2025 field-SOC share = %s (p < 0.001), other covariates |corr| <= %s."),
  ifelse(fal_pass, "pass", "FAIL"),
  fmt(g$estimate[1]), fmt(g$se[1]), fmt(g$p[1], 3), fmt(g$estimate[2]), fmt(g$se[2]), fmt(g$p[2], 3),
  fmt(F$ev_wald$p_F[1], 3), fmt(F$ev_wald$p_F[2], 3),
  fmt(fr$estimate[fr$term == "btrunc_post"]), fmt(fr$p[fr$term == "btrunc_post"], 3),
  fmt(fr$estimate[fr$spec_id == "F4b_role_switch_preperiod"]), fmt(fr$p[fr$spec_id == "F4b_role_switch_preperiod"], 3),
  fmt(fr$estimate[fr$term == "nonbind_post"]), fmt(fr$p[fr$term == "nonbind_post"], 3),
  fmt(fr$estimate[fr$term == "d1_post_floored"]), fmt(fr$p[fr$term == "d1_post_floored"], 3),
  fmt(F$f5b_fit$loglik[2], 8), fmt(F$f5b_fit$loglik[1], 8),
  fmt(fr$estimate[fr$term == "d1_post_piecehi"]), fmt(fr$p[fr$term == "d1_post_piecehi"], 3),
  fmt(F$balance$within_state_weighted_corr[F$balance$covariate == "field_soc_share_2025"], 3),
  fmt(max(abs(F$balance$within_state_weighted_corr[F$balance$covariate != "field_soc_share_2025"])), 2))
L$estimate_native <- P$eta
L["estimate_elasticity_cost"] <- list(NULL)
L$se_elasticity_cost <- P$se_eps_cost_jk
L$ci95_elasticity_cost <- list(ifelse(is.finite(P$eps_fieller_low), P$eps_fieller_low, "-Inf"), ifelse(is.finite(P$eps_fieller_high), P$eps_fieller_high, "Inf"))
L$elasticity_cost_low_high_ratio <- list(fieller_set = sprintf("%s at every ratio", P$fieller_type),
  point_not_headline = list(P$eps_cost_ratio_low, P$eps_cost_ratio_high),
  itt_bound_pi1 = list(P$eta * CBAR[["low"]], P$eta * CBAR[["high"]]))
L$p_unadjusted <- P$p_eta_crv1
L$p_size_valid <- P$p_eta_crv1
L$realized_mde_elasticity <- P$realized_mde_eps_cost_jk
L$pooling_eligible <- fal_pass && isTRUE(L$pooling_eligible)
if (!fal_pass) L$pooling_exclusion_reason <- "gating falsification failure (registry falsification.rule)"
L$post_outcome_analyses <- list(
  "07_post_outcome_first_stage_descriptive.R -> a1_post_outcome_first_stage_descriptive.csv: PPML-weighted mean FY2025-to-FY2026 change in county offered wage by within-state binding half (written after the first-stage F was seen; descriptive only)")
L$estimates_path <- file.path(out, "estimates.csv")
sec <- E$rows
gv <- function(id, col = "estimate") sec[[col]][sec$spec_id == id][1]
L$notes <- paste0(L$notes, " | ESTIMATION (Phase 4-5, 2026-11-05): registry hash verified; gating falsification ", ifelse(fal_pass, "passed", "FAILED"),
  sprintf(". Primary eta = %s per dollar (state CRV1 se %s, 95%% CI [%s, %s], p = %s; jackknife se %s; region CRV1 p = %s); 2,923 counties, 48 states.",
    fmt(P$eta), fmt(P$se_eta_crv1), fmt(gv("P_primary", "ci_low")), fmt(gv("P_primary", "ci_high")), fmt(P$p_eta_crv1, 3), fmt(P$se_eta_jk),
    fmt(gv("P_primary", "p_region_crv1"), 3)),
  sprintf(" First stage pi_U = %s (se %s), F = %s: WEAK; per the registry epsilon_cost is reported only with the Fieller/AR set, which is the whole real line (point (eta/pi) x C_bar = %s is not a headline; jackknife SE %s; realized MDE 2.8 x jackknife SE = %s).",
    fmt(pi_U), fmt(P$se_pi_crv1), fmt(E$F_fs, 3), fmt(P$eps_cost, 3), fmt(P$se_eps_cost_jk, 3), fmt(P$realized_mde_eps_cost_jk, 3)),
  sprintf(" ITT bound (pi = 1): eta x C_bar = %s, 95%% CI [%s, %s] (realized ITT MDE %s vs design 0.82).",
    fmt(P$eps_itt, 3), fmt(P$eps_itt_ci_low, 3), fmt(P$eps_itt_ci_high, 3), fmt(P$realized_mde_eps_itt_crv1, 3)),
  sprintf(" Lead secondary requested positions eta = %s (se %s, p = %s; ITT eps %s [%s, %s]).",
    fmt(gv("LS_req_pos")), fmt(gv("LS_req_pos", "se")), fmt(gv("LS_req_pos", "p"), 3),
    fmt(gv("LS_req_pos") * Cc, 3), fmt(gv("LS_req_pos", "ci_low") * Cc, 3), fmt(gv("LS_req_pos", "ci_high") * Cc, 3)),
  " Pooling: eligibility is left as set by the design-stage and falsification rules; the synthesis lead should note that the declared conversion yields an unbounded epsilon_cost set (inclusion criterion 'converts to the primary estimand'), and that the jackknife SE of epsilon_cost gives negligible inverse-variance weight.",
  " Implementation choices within the registry text: CRV1 p-values use t(G-1) (harness convention); FY2025 SOC and piece rate from h2a_disclosure_records by case_number (case-row soc7 is blank for 59% of FY2025 hours); counties without FY2025 rows take the state FY2025-hours-weighted share for S7/S9 covariates and F5c; F5c median weighted by FY2025 rq13d hours; SD2 RI by state within region not run because F_c varies within NY and OR (as at the design stage).")
L$updated <- "2026-11-05"
jsonlite::write_json(L, lp, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null")
message("ledger_row.json updated")
