# Identification search, strategy F1 (Phases 4-5): apply the registry's gating and
# interpretation rules mechanically, build the welfare-model feed, and assemble
# $S/F1/estimates.csv (= f1_estimates.csv, one row per declared quantity),
# $S/F1/f1_validation.csv and the estimation fields of
# explorations/identification_search/F1/ledger_row.json (declarations.json ledger).
#
# Usage (after 04-10): R_LIBS_USER=... Rscript --vanilla explorations/identification_search/F1/11_estimates_ledger.R "$S"
source("explorations/identification_search/F1/03_est_helpers.R")
f1_check_registry()
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- f1_out_dir(S)
rd <- function(f) read_csv(file.path(out, f), show_col_types = FALSE, guess_max = 10000)
P1 <- readRDS(file.path(out, "f1_p1.rds")); PP <- readRDS(file.path(out, "f1_p2p3.rds")); P4 <- readRDS(file.path(out, "f1_p4.rds"))
sens <- rd("f1_sensitivities.csv"); g2 <- rd("f1_g2.csv"); g3 <- rd("f1_g3.csv"); po1 <- rd("f1_post_outcome_p1_frame.csv")
tun <- rd("f1_p4_tuning.csv"); ch <- fromJSON(file.path(out, "f1_p4_choice.json"))
g <- PP$g

# ---- Gating ---------------------------------------------------------------------------------
p1 <- P1$rows
fals1 <- p1 %>% filter(item == "FALS1_placebo"); fals1_fail <- fals1$p_theta0 < 0.05
th <- if (fals1_fail) p1 %>% filter(item == "P1_theta_T") else p1 %>% filter(item == "P1_theta")
v1 <- PP$p3 %>% filter(scope == "primary: M26 vs M25", grid == "m=1")
v1_reject <- v1$ci_excludes_0
inc0 <- th$ci_low <= 0 & th$ci_high >= 0; inc1 <- th$ci_low <= 1 & th$ci_high >= 1
reading <- case_when(
  !inc0 & inc1 ~ "M2_supported",
  inc0 & !inc1 & !v1_reject ~ "M1_supported",
  inc0 & !inc1 & v1_reject ~ "conflict (P1 supports M1, P3 rejects stability)",
  !inc0 & !inc1 ~ "partial_indexation",
  TRUE ~ "indeterminate")
message(sprintf("theta source %s: %.3f [%.3f, %.3f]; V1 Delta_S %.4f [%.4f, %.4f] (reject: %s) -> reading: %s",
  th$item, th$estimate, th$ci_low, th$ci_high, v1$estimate, v1$ci_low, v1$ci_high, v1_reject, reading))
p2_label <- switch(sub(" .*", "", reading),
  M1_supported = "counterfactual binding-share function (M1 reading)",
  M2_supported = "distribution of FY2026 offers relative to the floor (offer premia over the floor); NOT market wages (M2 reading)",
  partial_indexation = "floor-conditional offer distribution (partial indexation)",
  indeterminate = "both readings (indeterminate)",
  conflict = "M1 reading only on the balanced-incumbent P3 version (conflict)")
p4_label <- paste("EXTRAPOLATION;", switch(sub(" .*", "", reading),
  M1_supported = "latent market-wage distribution", M2_supported = "distribution of offer premia over the floor, NOT market wages",
  partial_indexation = "floor-conditional offer distribution", indeterminate = "both readings", conflict = "latent distribution (conflict)"))

# ---- Rows ----------------------------------------------------------------------------------
row <- function(item, scope, estimate, se = NA, ci_low = NA, ci_high = NA, inference = NA, sample = NA, n_orders = NA, n_units = NA, n_states = NA,
  declared = TRUE, label = NA, role = NA, gating = FALSE, gate_result = NA, p_theta0 = NA, p_theta1 = NA, note = NA)
  tibble(item, scope, estimate, se, ci_low, ci_high, p_theta0, p_theta1, inference, sample, n_orders, n_units, n_states, declared, role, gating, gate_result, label, note)
E <- list()
E$p1 <- p1 %>% transmute(item, scope = ifelse(item %in% c("FALS1_placebo", "FALS2_theta_pre"), "INC_placebo", "INC"), estimate, se, ci_low, ci_high,
  p_theta0, p_theta1, inference, sample = paste0(sample, "; regressor ", regressor, "; weights ", weights), n_orders = NA, n_units, n_states,
  declared = TRUE, role, gating = item %in% c("FALS1_placebo", "P1_theta"),
  gate_result = case_when(item == "FALS1_placebo" ~ ifelse(fals1_fail, "fail", "pass"),
    item == "P1_theta" ~ paste("decides reading:", reading), TRUE ~ NA_character_),
  label = "MODEL-BASED (not an elasticity)",
  note = ifelse(item == "P1_theta_T", "reported pass-through only if FALS1 fails", ifelse(is.na(se_crv1), NA,
    sprintf("state CRV1 alternate: SE %.4f, CI [%.4f, %.4f], p(theta=0) %.4f, p(theta=1) %.4g", se_crv1, ci_low_crv1, ci_high_crv1, p_theta0_crv1, p_theta1_crv1))))
E$p2 <- PP$p2 %>% transmute(item = paste0("P2_b26[", grid, "]"), scope, estimate, se, ci_low, ci_high, inference,
  sample = "M26 (FY2026 rq13d, received >= 2025-10-02, hourly non-piece, classes above/at_floor/at_old_aewr)", n_orders, n_units = NA, n_states,
  declared = TRUE, role = "primary", gating = FALSE, gate_result = NA, label = p2_label, note = sprintf("S26 = %.4f; above-floor orders %d", S26, n_above))
E$p3 <- PP$p3 %>% transmute(item = ifelse(grepl("^S6", scope), "S6_P3_Delta_S", paste0("P3_Delta_S[", grid, "]")), scope, estimate, se, ci_low, ci_high, inference,
  sample = "M26 vs M25", n_orders, n_units = NA, n_states, declared = TRUE,
  role = ifelse(grepl("^S6", scope), "sensitivity", ifelse(grid == "m=1" & grepl("^primary", scope), "primary validation (V1)", "primary (also)")),
  gating = grid == "m=1" & grepl("^primary", scope),
  gate_result = ifelse(grid == "m=1" & grepl("^primary", scope), ifelse(ci_excludes_0, "fail (M1 stability rejected)", "pass"), NA),
  label = "nonparametric", note = sprintf("S26 %.5f [%.5f, %.5f]; S25 %.5f [%.5f, %.5f]; g = %.4f", S26, S26_ci_low, S26_ci_high, S25, S25_ci_low, S25_ci_high, g))
E$p4 <- P4$p4 %>% transmute(item, scope, estimate, se, ci_low, ci_high, inference = "999-draw state-cluster bootstrap over the 32 confirmation states (refit each draw), percentile",
  sample = "M26 confirmation states", n_orders, n_units = NA, n_states, declared = TRUE, role = "secondary", gating = FALSE, gate_result = NA,
  label = p4_label, note = ifelse(is.na(band_min), sprintf("censored hours share %.3f", censored_hours_share),
    sprintf("model-uncertainty band (4 candidates on confirmation states) [%.4f, %.4f]; censored hours share %.3f", band_min, band_max, censored_hours_share)))
E$tun <- tun %>% filter(level == "total") %>% transmute(item = paste0("P4_tuning_LOSO_ll[", model, "]"), scope = "16 exploration states", estimate = loso_ll,
  inference = "leave-one-state-out predictive log-likelihood", sample = "M26 exploration states", n_states = 16, declared = TRUE, role = "tuning",
  gate_result = ifelse(chosen, "chosen", NA), label = "P4 tuning", note = NA_character_) %>% mutate(se = NA_real_, ci_low = NA_real_, ci_high = NA_real_)
E$v2 <- P4$v2 %>% transmute(item = paste0(item, ifelse(declared, "", "[post-outcome: obs incl. at_old_aewr]")), scope = observed, estimate, se, ci_low, ci_high,
  inference = "WLS (state FY2026 hours) of observed on LOSO-predicted state share; 999-draw state bootstrap, percentile", sample = "32 confirmation states",
  n_states, declared, role = ifelse(declared, "validation (reported)", "post-outcome"), gating = FALSE, gate_result = ifelse(pass, "pass", "fail"),
  label = "V2", note = "pass if slope CI includes 1 and intercept CI includes 0")
E$v3 <- P4$v3 %>% transmute(item, scope = "confirmation states", estimate, se, ci_low, ci_high,
  inference = "predicted minus observed; 999-draw state-cluster bootstrap refitting on each draw, percentile", sample = "fit M25 conf. states; predict M26 conf. states",
  n_orders = n_orders_fy26, n_states, declared = TRUE, role = "validation (reported)", gating = FALSE,
  gate_result = ifelse(ci_includes_0, "CI includes 0", "CI excludes 0"), label = "V3",
  note = sprintf("predicted %.4f, observed %.4f%s", predicted, observed, ifelse(item == "V3_share_above_Fapp", "; not blind (seen as support)", "")))
E$v4 <- PP$v4 %>% transmute(item, scope, estimate, se, ci_low, ci_high, inference = "999-draw state-cluster bootstrap, percentile", sample = scope,
  n_orders, n_states, declared = TRUE, role = "validation (reported)", gating = FALSE, gate_result = NA, label = "V4 (S3 refits in sensitivities)",
  note = sprintf("orders at shared values: %d", n_orders_mp))
E$sens <- sens %>% transmute(item = paste0(sub(" .*", "", sensitivity), "_", item), scope = sensitivity, estimate, se, ci_low, ci_high, inference,
  sample = ifelse(grepl("^P4", item), "M26 confirmation states", "M26 / M25, 48 states"), n_orders, n_states, declared = TRUE, role = "sensitivity",
  gating = FALSE, gate_result = NA, label = ifelse(grepl("^P4", item), p4_label, p2_label), note = NA_character_)
E$s7 <- P4$s7 %>% transmute(item = paste0("S7_", item), scope, estimate, se, ci_low, ci_high, inference = "999-draw state-cluster bootstrap over 48 states, percentile",
  sample = "M26, 48 states", n_orders, n_states, declared = TRUE, role = "sensitivity", gating = FALSE, gate_result = NA, label = p4_label, note = NA_character_)
E$g1 <- PP$g1 %>% transmute(item, scope, estimate, se, ci_low, ci_high, inference = "999-draw state-cluster bootstrap, percentile", sample = scope, n_orders,
  declared = TRUE, role = "gap-model implication", gating = FALSE, gate_result = NA,
  label = ifelse(item == "G1_b26(a25)" & reading != "M1_supported", "descriptive only: the at-a25 reading requires M1, not supported", "descriptive"), note = NA_character_)
E$g2 <- g2 %>% transmute(item, scope = "states with >= 10 above-floor M26 orders", estimate, se, ci_low, ci_high,
  inference = "hours-weighted correlation; 999-draw state bootstrap, percentile", sample = "M26 states; ACS 2024 state ag mean hourly wage", n_states,
  declared = TRUE, role = "gap-model implication", gating = FALSE, gate_result = ifelse(premise_supported, "premise supported", "premise not supported"),
  label = ifelse(grepl("p4", item), p4_label, "nonparametric"), note = ifelse(grepl("p4", item), "state shortfalls fixed at the single confirmation fit", NA))
E$g3 <- g3 %>% transmute(item, scope = era, estimate, se, ci_low, ci_high, inference, sample = "eras, 48 states, hourly non-piece, classes above/at_floor",
  n_orders, n_states, declared = TRUE, role = "descriptive (G3)", gating = FALSE, gate_result = NA, label, note = NA_character_)

# ---- Welfare feed ------------------------------------------------------------------------------
xa <- PP$p2 %>% filter(grepl("^national|^state|^other", scope)) %>%
  transmute(scope, grid, b26 = estimate, x = b26 + th$estimate * (1 - b26), x_lo = b26 + th$ci_low * (1 - b26), x_hi = b26 + th$ci_high * (1 - b26))
feed_x <- reading == "M2_supported"
E$wf <- bind_rows(
  row("WF_dwda_above_floor_theta_hat", "national", th$estimate, th$se, th$ci_low, th$ci_high, inference = th$inference, sample = th$item,
    n_units = th$n_units, n_states = th$n_states, role = "welfare feed (i)", label = "MODEL-BASED",
    note = sprintf("source: %s; reading %s%s", th$item, reading, ifelse(feed_x, "; under M2 the declared feed uses dw/da = 1 for all hours", ""))),
  row("WF_dwda_feed_value", "national", if (feed_x) 1 else if (reading == "indeterminate") NA_real_ else th$estimate,
    inference = "registry interpretation_rule", role = "welfare feed (i)", label = reading,
    note = if (feed_x) "M2: dw/da = 1, X_m = all hours of hourly orders" else if (reading == "indeterminate") "polar scenarios only (0 and 1)" else "theta-hat above the floor"),
  xa %>% transmute(item = paste0("WF_x[", grid, "]"), scope, estimate = x, se = NA_real_, ci_low = x_lo, ci_high = x_hi,
    inference = "x(a) = b26(a) + theta-hat (1 - b26(a)); interval = theta CI endpoints at the b26 point estimate", sample = "M26",
    declared = TRUE, role = "welfare feed (ii)", gating = FALSE, gate_result = NA,
    label = ifelse(feed_x, "computed as declared; NOT the feed under the M2 reading (feed uses x = 1)", "welfare feed"), note = sprintf("b26 = %.4f", b26)),
  P4$p4 %>% filter(grepl("^P4_b_", item)) %>% mutate(chosen_est = estimate) %>% transmute(item = paste0("WF_P4_scenario_band_", sub("^P4_", "", item)), scope = "confirmation states",
    estimate = NA_real_, ci_low = band_min, ci_high = band_max, inference = "min-max over the 4 candidate models (scenario band, not a point input)",
    sample = "M26 confirmation states", declared = TRUE, role = "welfare feed (iii)", gating = FALSE, gate_result = NA, label = p4_label,
    note = sprintf("chosen model %s: %.4f", ch$chosen, chosen_est)))
E$interp <- row("interpretation_rule", "P1 + P3", NA_real_, inference = "registry interpretation_rule (mechanical)", role = "interpretation",
  gate_result = reading, label = reading,
  note = sprintf("theta (%s) %.3f CI [%.3f, %.3f]: includes 0 %s, includes 1 %s; Delta_S CI [%.4f, %.4f] includes 0 %s; FALS1 %s",
    th$item, th$estimate, th$ci_low, th$ci_high, inc0, inc1, v1$ci_low, v1$ci_high, !v1_reject, ifelse(fals1_fail, "fail", "pass")))
E$po <- po1 %>% transmute(item, scope = "INC design frame (352 units)", estimate, se, ci_low, ci_high, p_theta0, p_theta1, inference, sample, n_units, n_states,
  declared = FALSE, role = "post-outcome", gating = FALSE, gate_result = NA, label = "POST-OUTCOME", note)
est <- bind_rows(E) %>% mutate(registry_sha256 = REGISTRY_SHA)
write_csv(est, file.path(out, "estimates.csv")); write_csv(est, file.path(out, "f1_estimates.csv"))
message("estimates rows: ", nrow(est))

# ---- Validation table ------------------------------------------------------------------------------
v2d <- P4$v2 %>% filter(declared)
v3 <- P4$v3
val <- tibble(
  id = c("V1", "FALS1", "FALS2", "V2", "V3", "V4"),
  role = c("gating (M1 reading)", "gating (P1)", "reported", "reported", "reported", "reported"),
  statistic = c(sprintf("Delta_S(m=1) = %.4f [%.4f, %.4f]", v1$estimate, v1$ci_low, v1$ci_high),
    sprintf("placebo theta = %.3f [%.3f, %.3f], p = %.3f", fals1$estimate, fals1$ci_low, fals1$ci_high, fals1$p_theta0),
    with(p1 %>% filter(item == "FALS2_theta_pre"), sprintf("theta_pre = %.3f [%.3f, %.3f], p(0) = %.3f, p(1) = %.3f", estimate, ci_low, ci_high, p_theta0, p_theta1)),
    sprintf("slope %.3f [%.3f, %.3f]; intercept %.4f [%.4f, %.4f]", v2d$estimate[2], v2d$ci_low[2], v2d$ci_high[2], v2d$estimate[1], v2d$ci_low[1], v2d$ci_high[1]),
    paste(sprintf("%s pred-obs %.4f [%.4f, %.4f]", sub("V3_", "", v3$item), v3$estimate, v3$ci_low, v3$ci_high), collapse = "; "),
    with(PP$v4, paste(sprintf("%s %.3f [%.3f, %.3f]", scope, estimate, ci_low, ci_high), collapse = "; "))),
  result = c(ifelse(v1_reject, "fail (M1 stability rejected)", "pass"), ifelse(fals1_fail, "fail", "pass"), "reported",
    ifelse(v2d$pass[1], "pass", "fail"), ifelse(all(v3$ci_includes_0), "all CIs include 0", paste("CI excludes 0:", paste(sub("V3_", "", v3$item[!v3$ci_includes_0]), collapse = ", "))),
    "reported"))
print(val, width = 250)
write_csv(val, file.path(out, "f1_validation.csv"))

# ---- Ledger --------------------------------------------------------------------------------------
lp <- file.path(F1_DIR, "ledger_row.json"); L <- fromJSON(lp, simplifyVector = FALSE)
p1r <- p1 %>% filter(item == "P1_theta")
L$phase_reached <- "estimation"; L$status <- "estimated"
L$falsification_result <- sprintf("%s (FALS1 gating placebo: theta = %.3f, jackknife p = %.3f). V1 (gates the M1 reading): %s, Delta_S = %.4f [%.4f, %.4f]. Reported: FALS2 theta_pre = %.3f [%.3f, %.3f]; V2 %s; V3 %s; V4 mass-point share of M26 field above-floor hours %.3f.",
  ifelse(fals1_fail, "fail", "pass"), fals1$estimate, fals1$p_theta0, ifelse(v1_reject, "fail (M1 stability rejected)", "pass"), v1$estimate, v1$ci_low, v1$ci_high,
  with(p1 %>% filter(item == "FALS2_theta_pre"), estimate), with(p1 %>% filter(item == "FALS2_theta_pre"), ci_low), with(p1 %>% filter(item == "FALS2_theta_pre"), ci_high),
  val$result[val$id == "V2"], val$result[val$id == "V3"], PP$v4$estimate[PP$v4$scope == "M26 field"])
L$estimate_native <- sprintf("P1 theta = %.3f (state CV3J SE %.3f, 95%% CI [%.3f, %.3f], t(%d); p(theta=0) = %.4f, p(theta=1) = %.4f; %d units, %d states). Reading under the registry rule: %s. P2 b26(F_app) = %.3f, b26(a25) = %.3f, b26(a25(1+g)) = %.3f. P3 Delta_S = %.4f [%.4f, %.4f]. P4 (%s, EXTRAPOLATION): b(H-2A skill-I rate) = %.3f, band [%.3f, %.3f].",
  p1r$estimate, p1r$se, p1r$ci_low, p1r$ci_high, p1r$n_states - 1L, p1r$p_theta0, p1r$p_theta1, p1r$n_units, p1r$n_states, reading,
  PP$p2$estimate[PP$p2$scope == "national (48 states), field" & PP$p2$grid == "F_app"], PP$p2$estimate[PP$p2$scope == "national (48 states), field" & PP$p2$grid == "a25"],
  PP$p2$estimate[PP$p2$scope == "national (48 states), field" & PP$p2$grid == "a25(1+g)"], v1$estimate, v1$ci_low, v1$ci_high, ch$chosen,
  P4$p4$estimate[P4$p4$item == "P4_b_H2A_s1"], P4$p4$band_min[P4$p4$item == "P4_b_H2A_s1"], P4$p4$band_max[P4$p4$item == "P4_b_H2A_s1"])
L$estimate_elasticity_cost <- NULL; L["estimate_elasticity_cost"] <- list(NULL)
L$p_unadjusted <- p1r$p_theta0; L$p_size_valid <- p1r$p_theta0
L$post_outcome_analyses <- list(
  sprintf("PO1: P1 on the 352-unit design-stage frame (presence on any new-rule field non-piece FY2026 order; the declared INC requires an M26-class order and has %d units): theta = %.3f [%.3f, %.3f] (jackknife), p(0) = %.3f, p(1) = %.4f. Under the rule this frame would read M1 (CI includes 0, excludes 1); declared P1 reads M2. Labeled post-outcome; does not replace P1.",
    p1r$n_units, po1$estimate, po1$ci_low, po1$ci_high, po1$p_theta0, po1$p_theta1),
  sprintf("PO2: V2 with observed = hours share of exact observations above F_app + 0.05 (incl. at_old_aewr; model-consistent reading): slope %.3f [%.3f, %.3f], intercept %.4f [%.4f, %.4f], %s.",
    P4$v2$estimate[!P4$v2$declared][2], P4$v2$ci_low[!P4$v2$declared][2], P4$v2$ci_high[!P4$v2$declared][2], P4$v2$estimate[!P4$v2$declared][1],
    P4$v2$ci_low[!P4$v2$declared][1], P4$v2$ci_high[!P4$v2$declared][1], ifelse(P4$v2$pass[!P4$v2$declared][1], "pass", "fail")))
L$estimates_path <- file.path(out, "estimates.csv")
L$notes <- paste0(sub(" ESTIMATION \\(Phases 4-5.*$", "", L$notes), sprintf(" ESTIMATION (Phases 4-5, registry SHA-256 and design-script hashes verified): FALS1 %s; P1 theta = %.3f [%.3f, %.3f] -> %s; V1 %s. P2/P4 therefore reported as %s. Welfare feed: %s. Scripts 03_est_helpers.R-11_estimates_ledger.R; outputs %s.",
  ifelse(fals1_fail, "fails", "passes"), th$estimate, th$ci_low, th$ci_high, reading, ifelse(v1_reject, "rejects M1 stability", "does not reject M1 stability"), p2_label,
  ifelse(feed_x, "dw/da = 1 for all hours (X_m = all hours of hourly orders)", "see estimates.csv WF_ rows"), out))
L$updated <- "2026-11-05"
write(toJSON(L, auto_unbox = TRUE, pretty = TRUE, null = "null", digits = NA), lp)
message("ledger updated; reading: ", reading)
