# Identification search, Phase 7 synthesis, step 1.
#
# 1. Verifies the SHA-256 of shared/declarations.json and of every frozen
#    strategy registry against the recorded hashes.
# 2. Applies the declared pooling rule (declarations.json pooling_rule) to every
#    strategy, criterion by criterion, from ledger_row.json, review.md and the
#    strategy outputs. Computes a pooled estimate only if at least one strategy
#    is eligible (random-effects DerSimonian-Laird, Q, I^2, tau^2, HKSJ).
# 3. Applies the declared multiplicity rule (declarations.json multiplicity) to
#    every estimated primary hypothesis: Romano-Wolf if joint resampling is
#    feasible, otherwise Holm on the size-valid p-values.
# 4. Restates Track 1 (pre-search, explorations/identification/08_track1_estimate.R)
#    in per-dollar semi-elasticities and, as a labeled post-hoc conversion, in
#    epsilon_cost units, for the welfare range. Nothing here re-estimates Track 1.
#
# Reads only files; computes no new outcome contrast.
#
# Usage (repository root):
#   Rscript --vanilla explorations/identification_search/synthesis/01_pooling_multiplicity.R "$S" "$S_T1"
#   $S    = identification_search scratch root (strategy outputs)
#   $S_T1 = identification scratch root (Track 1 outputs: track1_post_estimates.csv)
suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
S <- normalizePath(args[1])
S_T1 <- normalizePath(args[2])
out <- file.path(S, "synthesis")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
root <- "explorations/identification_search"
stopifnot(dir.exists(root))

sha <- function(path) {
  # sha256 of a file via the system tool (identical to `sha256sum`)
  sub(" .*$", "", system2("sha256sum", shQuote(path), stdout = TRUE))
}
recorded <- function(path) sub("[[:space:]].*$", "", readLines(path, warn = FALSE)[1])

# ---- 1. Hash checks ----------------------------------------------------------
hash_rows <- list(data.table(
  file = file.path(root, "shared/declarations.json"),
  sha256_actual = sha(file.path(root, "shared/declarations.json")),
  sha256_recorded = recorded(file.path(root, "shared/declarations.sha256"))
))
for (id in c("A1", "F1", "A2", "A3", "N1")) {
  reg <- file.path(root, id, "registry.json")
  lr <- fromJSON(file.path(root, id, "ledger_row.json"), simplifyVector = FALSE)
  hash_rows[[id]] <- data.table(
    file = reg,
    sha256_actual = sha(reg),
    sha256_recorded = recorded(file.path(root, id, "registry.sha256")),
    sha256_ledger_row = lr$registry_sha256
  )
}
hashes <- rbindlist(hash_rows, fill = TRUE)
hashes[, match_recorded := sha256_actual == sha256_recorded]
hashes[, match_ledger_row := is.na(sha256_ledger_row) | sha256_actual == sha256_ledger_row]
fwrite(hashes, file.path(out, "hash_check.csv"))
print(hashes[, .(file, match_recorded, match_ledger_row)])
stopifnot(all(hashes$match_recorded), all(hashes$match_ledger_row))

decl <- fromJSON(file.path(root, "shared/declarations.json"), simplifyVector = FALSE)

# ---- 2. Pooling rule ---------------------------------------------------------
review_verdict <- function(id) {
  p <- file.path(root, id, "review.md")
  if (!file.exists(p)) return(NA_character_)
  v <- grep("^## Verdict", readLines(p, warn = FALSE), value = TRUE)[1]
  tolower(gsub("^## Verdict: \\*\\*|\\*\\*.*$", "", v))
}
lr <- lapply(setNames(nm = c("A1", "F1", "A2", "A3", "N1", "D2", "A6")),
             function(id) fromJSON(file.path(root, id, "ledger_row.json"), simplifyVector = FALSE))
nz <- function(x) if (is.null(x)) NA else x

a1_conv <- fread(file.path(S, "A1/a1_conversion.csv"))[spec_id == "P_primary"]
a1_fals <- lr$A1$falsification_result

# One row per strategy that holds a ledger_row.json; criteria from
# declarations.json pooling_rule.inclusion, in order.
pool <- rbindlist(list(
  data.table(
    strategy_id = "A1",
    c1_frozen_registry = TRUE,
    c2_design_mde_le_1 = nz(lr$A1$design_mde_elasticity) <= 1.0,
    c2_note = sprintf("design MDE %.2f at pi = 1 (design premise); realized first stage pi_U = %.3f (F = %.2f)",
                      lr$A1$design_mde_elasticity, a1_conv$pi_U, a1_conv$first_stage_F_crv1),
    c3_falsification_pass = grepl("^pass", a1_fals),
    c4_review_stands = review_verdict("A1") %in% c("stands", "stands with caveats"),
    c4_note = paste0("review verdict: ", review_verdict("A1")),
    c5_converts_to_eps_cost = !(a1_conv$fieller_type == "whole_line"),
    c5_note = sprintf("Fieller/AR set for epsilon_cost: %s; jackknife SE %.1f", a1_conv$fieller_type, a1_conv$se_eps_cost_jk),
    c6_distinct_assumptions = TRUE,
    c6_note = "statutory_2025_statewide family (counts once with any other A1 variant)",
    excluded_from_headline_class = FALSE
  ),
  data.table(
    strategy_id = "F1",
    c1_frozen_registry = TRUE,
    c2_design_mde_le_1 = NA,
    c2_note = "no elasticity MDE (model-based)",
    c3_falsification_pass = grepl("^pass", lr$F1$falsification_result),
    c4_review_stands = review_verdict("F1") %in% c("stands", "stands with caveats"),
    c4_note = paste0("review verdict: ", review_verdict("F1")),
    c5_converts_to_eps_cost = FALSE,
    c5_note = "not converted to epsilon_cost (registry; declarations excluded_from_headline: F-family)",
    c6_distinct_assumptions = TRUE,
    c6_note = "structural_market_wage_censoring",
    excluded_from_headline_class = TRUE
  ),
  rbindlist(lapply(c("A2", "A3", "N1"), function(id) data.table(
    strategy_id = id,
    c1_frozen_registry = TRUE,
    c2_design_mde_le_1 = FALSE,
    c2_note = paste0("killed at design: ", substr(lr[[id]]$kill_reason, 1, 160)),
    c3_falsification_pass = NA,
    c4_review_stands = NA,
    c4_note = "not estimated, not reviewed",
    c5_converts_to_eps_cost = NA,
    c5_note = "not estimated",
    c6_distinct_assumptions = TRUE,
    c6_note = lr[[id]]$assumption_family,
    excluded_from_headline_class = identical(substr(as.character(nz(lr[[id]]$external_validity_flag)), 1, 4), "true")
  ))),
  rbindlist(lapply(c("D2", "A6"), function(id) data.table(
    strategy_id = id,
    c1_frozen_registry = FALSE,
    c2_design_mde_le_1 = NA,
    c2_note = paste0("not run: ", lr[[id]]$kill_reason),
    c3_falsification_pass = NA, c4_review_stands = NA, c4_note = "not run",
    c5_converts_to_eps_cost = NA, c5_note = "not run",
    c6_distinct_assumptions = NA, c6_note = NA_character_,
    excluded_from_headline_class = FALSE
  )))
), fill = TRUE)
crit <- c("c1_frozen_registry", "c2_design_mde_le_1", "c3_falsification_pass",
          "c4_review_stands", "c5_converts_to_eps_cost", "c6_distinct_assumptions")
pool[, pooling_eligible := Reduce(`&`, lapply(.SD, function(x) !is.na(x) & x)), .SDcols = crit]
pool[, headline_pool := pooling_eligible & !excluded_from_headline_class]
pool[, failed_criteria := apply(.SD, 1, function(r) {
  f <- names(r)[is.na(r) | !as.logical(r)]
  if (length(f) == 0) "none" else paste(f, collapse = "; ")
}), .SDcols = crit]
fwrite(pool, file.path(out, "pooling_decision.csv"))
print(pool[, .(strategy_id, pooling_eligible, headline_pool, failed_criteria)])

n_headline <- sum(pool$headline_pool)
n_sens <- sum(pool$pooling_eligible)
pooled <- data.table(
  pool = c("headline", "sensitivity (adds external-validity-flagged)", "model-based (F-family, reported separately)"),
  n_included = c(n_headline, n_sens, 0L),
  estimate = NA_real_, se = NA_real_, ci_low = NA_real_, ci_high = NA_real_,
  Q = NA_real_, I2 = NA_real_, tau2 = NA_real_,
  result = c(
    if (n_headline == 0) "not computed: no strategy meets every inclusion criterion" else "see code",
    if (n_sens == 0) "not computed: no strategy meets every inclusion criterion" else "see code",
    "not computed: F1 has no epsilon_cost estimate and failed review"
  )
)
if (n_headline >= 1) stop("An eligible strategy exists: implement the DerSimonian-Laird pool here before reporting.")
fwrite(pooled, file.path(out, "pooled_estimate.csv"))

# ---- 3. Multiplicity ---------------------------------------------------------
f1 <- fread(file.path(S, "F1/estimates.csv"))
f1_p1 <- f1[item == "P1_theta" & role == "primary"]
f1_p3 <- f1[item == "P3_Delta_S[m=1]" & role == "primary validation (V1)"]
stopifnot(nrow(f1_p1) == 1, nrow(f1_p3) == 1)
# P3's declared inference is a percentile state-cluster bootstrap CI; no p-value
# and no draws were saved. Use the normal approximation with the bootstrap SE
# (the same approximation as the F1 review, $S/F1/review/r01_holm.csv).
p3_p <- 2 * pnorm(-abs(f1_p3$estimate / f1_p3$se))

hyp <- data.table(
  strategy_id = c("A1", "F1", "F1", "F1"),
  hypothesis = c("A1 primary: H0 epsilon_cost = 0 (PPML eta = 0, cert_hours_rq13d, county + state x FY FE)",
                 "F1 P1 H0a: theta = 0 (M1, fixed market wage)",
                 "F1 P1 H0b: theta = 1 (M2, fixed premium)",
                 "F1 P3 (V1): Delta_S[m = 1] = 0 (M1 stability)"),
  in_declared_eps_cost_family = c(TRUE, FALSE, FALSE, FALSE),
  estimate = c(a1_conv$eta, f1_p1$estimate, f1_p1$estimate, f1_p3$estimate),
  se = c(a1_conv$se_eta_crv1, f1_p1$se, f1_p1$se, f1_p3$se),
  p_unadjusted = c(a1_conv$p_eta_crv1, f1_p1$p_theta0, f1_p1$p_theta1, p3_p),
  p_size_valid = c(a1_conv$p_eta_crv1, f1_p1$p_theta0, f1_p1$p_theta1, p3_p),
  p_source = c("state CRV1 t(G-1) on the PPML fit (A1 declared primary inference; AR/jackknife p 0.719)",
               "state CV3J jackknife t(G-1) (F1 declared primary inference)",
               "state CV3J jackknife t(G-1) (F1 declared primary inference)",
               "normal approximation with the 999-draw state-bootstrap SE (declared inference is a percentile CI only)")
)
# Declared family: H0 epsilon_cost = 0 of every strategy reaching estimation
# (plus co-primary requested-position hypotheses; A1 declares none). F1 has no
# epsilon_cost hypothesis. Extended family (synthesis brief): adds F1's three
# declared gating primaries.
hyp[in_declared_eps_cost_family == TRUE, p_holm_declared_family := p.adjust(p_size_valid, "holm")]
hyp[, p_holm_extended_family := p.adjust(p_size_valid, "holm")]
hyp[, method := "Holm"]
hyp[, romano_wolf_feasible := FALSE]
hyp[, rw_infeasible_reason := paste(
  "Joint resampling is infeasible on the declared terms: F1 P3's declared inference is a percentile bootstrap",
  "with no analytic SE, so studentizing it inside each of 9,999 joint draws needs a nested bootstrap;",
  "F1 P1's size-valid inference is the state jackknife, not a bootstrap; A1's size-valid inference is",
  "state CRV1 on a PPML fit whose state bootstrap was never size-tested. Declared fallback: Holm on the",
  "size-valid p-values for the whole family.")]
fwrite(hyp, file.path(out, "multiplicity.csv"))
print(hyp[, .(hypothesis, p_size_valid, p_holm_declared_family, p_holm_extended_family)])

# ---- 4. Track 1 reference (pre-search; not part of the family or the pool) ----
t1 <- fread(file.path(S_T1, "track1_post_estimates.csv"))
t1 <- t1[spec %in% c("B_primary", "B_primary_first_stage", "placebo_fy2024_fy2025")]
fs <- t1[spec == "B_primary_first_stage" & outcome == "y_wage_q"]
pi_t1 <- fs$estimate
w_bar <- fs$mean_y  # hours-weighted mean offered wage, FY2025-26 frame
ratio <- 1.2401     # declared FY2025-26 central cost ratio (A1 and N1 ledger rows)
t1r <- t1[spec == "B_primary" & outcome %in% c("y_cert_q", "y_hours_q", "cert_q", "hours_q")]
t1r[, semi_per_stat_dollar := ifelse(estimator == "PPML", estimate, estimate / mean_y)]
t1r[, semi_low := elasticity_low_jk / mean_price]
t1r[, semi_high := elasticity_high_jk / mean_price]
t1r[, `:=`(
  eps_statutory = elasticity, eps_statutory_low = elasticity_low_jk, eps_statutory_high = elasticity_high_jk,
  mde_eps_statutory = mde_elasticity_jk,
  conv_factor_eps_cost = ratio * w_bar / pi_t1,
  eps_cost_posthoc = semi_per_stat_dollar * ratio * w_bar / pi_t1,
  eps_cost_posthoc_low = semi_low * ratio * w_bar / pi_t1,
  eps_cost_posthoc_high = semi_high * ratio * w_bar / pi_t1,
  mde_eps_cost_posthoc = mde_elasticity_jk / mean_price * ratio * w_bar / pi_t1,
  pi_first_stage = pi_t1, w_bar_offered = w_bar, cost_ratio = ratio,
  label = paste("Track 1 B_primary (explorations/identification/08_track1_estimate.R).",
                "eps_statutory: elasticity w.r.t. the statutory U.S. skill-I rate (as reported).",
                "eps_cost_posthoc: SYNTHESIS POST-HOC conversion (eta / pi) x ratio x w_bar,",
                "pi = Track 1 first stage at its point value (its CI [0.47, 0.85] is ignored), not declared by any registry.")
)]
keep <- c("spec", "outcome", "estimator", "estimate", "se_jackknife", "p_jackknife", "mean_y", "mean_price",
          "semi_per_stat_dollar", "semi_low", "semi_high", "eps_statutory", "eps_statutory_low",
          "eps_statutory_high", "mde_eps_statutory", "pi_first_stage", "w_bar_offered", "cost_ratio",
          "conv_factor_eps_cost", "eps_cost_posthoc", "eps_cost_posthoc_low", "eps_cost_posthoc_high",
          "mde_eps_cost_posthoc", "label")
fwrite(t1r[, ..keep], file.path(out, "track1_reference.csv"))
print(t1r[, .(outcome, estimator, eps_statutory, eps_statutory_low, eps_statutory_high, mde_eps_statutory,
              eps_cost_posthoc, eps_cost_posthoc_low, eps_cost_posthoc_high, mde_eps_cost_posthoc)])
cat("Wrote", out, "\n")
