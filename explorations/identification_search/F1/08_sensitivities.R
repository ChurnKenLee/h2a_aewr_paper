# Identification search, strategy F1 (Phase 5): sensitivities S1-S5 (registry
# sensitivities; S6 is in 05_p2_p3.R and S7 in 07_p4_confirmation.R).
#   S1 piece-rate orders included (both FY2025 and FY2026 samples)
#   S2 at_old_aewr offers (a) censored at the old AEWR, (b) excluded
#   S3 'above' offers at a state x SOC-group x FY value offered by >= 3 employers censored there
#   S4 tolerance $0.10 (orders reclassified with f1_classify, TOL = 0.10; P2/P3 indicators use 0.10)
#   S5 non-field SOC orders added (P2, P4; P4 with a non-field indicator)
# For each: P2 national grid, P3 Delta_S at m = 1 (S1-S4), and P4 with the chosen model on the
# confirmation states (999-draw state bootstrap, seed 20261102). P2 is computed on offers, so S2a and
# S3 leave it unchanged by construction (reported anyway).
#
# Usage: R_LIBS_USER=... Rscript --vanilla explorations/identification_search/F1/08_sensitivities.R "$S"
source("explorations/identification_search/F1/03_est_helpers.R")
f1_check_registry()
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- f1_out_dir(S)
ch <- fromJSON(file.path(out, "f1_p4_choice.json")); CH <- P4_CANDS %>% filter(model == ch$chosen)
o0 <- read_parquet(file.path(out, "f1_orders.parquet"))
doses <- f1_doses(S); ST48 <- sort(unique(doses$state_fips)); g <- f1_g_frozen(out)
o <- f1_attach_doses(o0, doses)
set.seed(SEED); cnt48 <- boot_counts(ST48)

p4_boot <- function(D, nonfield = FALSE) {   # at_old_aewr handling is applied to D by run_sens
  md <- p4_data(D, nonfield = nonfield); fit <- p4_fit(md, CH$fam, CH$scale_k); est <- p4_outputs(fit, md)
  states <- sort(unique(md$d$state_fips)); set.seed(SEED); cnt <- boot_counts(states)
  sidx <- split(seq_len(nrow(md$d)), md$d$state_fips)[states]
  dr <- t(vapply(seq_len(NBOOT), function(b) {
    idx <- unlist(rep(sidx, cnt[, b]), use.names = FALSE)
    mb <- p4_data(md$d[idx, ], center = md$center, nonfield = nonfield)
    p4_outputs(p4_fit(mb, CH$fam, CH$scale_k, start = fit$par), mb)
  }, numeric(length(est))))
  tibble(item = paste0("P4_", names(est)), estimate = unname(est), se = apply(dr, 2, sd),
    ci_low = apply(dr, 2, function(x) pci(x)[1]), ci_high = apply(dr, 2, function(x) pci(x)[2]),
    n_orders = nrow(md$d), n_states = length(states), censored_hours_share = sum(md$d$hours[md$cens]) / sum(md$d$hours))
}
run_sens <- function(id, M26, M25, tol = TOL, nonfield = FALSE, old = "exact", do_p3 = TRUE) {
  t0 <- Sys.time()
  if (old == "excluded") M26 <- M26 %>% filter(cls != "at_old_aewr")
  r2 <- p2_rows(M26, g, ST48, cnt48, "national", tol = tol)
  r3 <- if (do_p3) p3_rows(M26, M25, g, ST48, cnt48, "national", ms = 1, tol = tol) %>% mutate(item = "P3_Delta_S[m=1]") else NULL
  D4 <- M26 %>% filter(part == "confirmation")
  if (old == "censored") D4 <- f1_old_value(D4) %>%
    mutate(c_floor = ifelse(cls == "at_old_aewr", c_old, c_floor), cls = ifelse(cls == "at_old_aewr", "at_floor", cls))
  r4 <- p4_boot(D4, nonfield = nonfield)
  message(id, " done in ", round(difftime(Sys.time(), t0, units = "mins"), 1), " min")
  bind_rows(r2 %>% mutate(item = paste0("P2_b26[", grid, "]")), r3, r4) %>% mutate(sensitivity = id)
}
res <- list()
# S1
res$S1 <- run_sens("S1 piece-rate orders included", f1_M26(o, ST48, piece_ok = TRUE), f1_M25(o, ST48, piece_ok = TRUE))
# S2
M26 <- f1_M26(o, ST48); M25 <- f1_M25(o, ST48)
res$S2a <- run_sens("S2a at_old_aewr censored at the old AEWR", M26, M25, old = "censored")
res$S2b <- run_sens("S2b at_old_aewr excluded", M26, M25, old = "excluded")
# S3
o3 <- f1_apply_S3(o)
message("S3: 'above' orders recoded as censored (all FY, non-piece pool): ", sum(o3$s3))
res$S3 <- run_sens("S3 shared wage values (>= 3 employers) censored", f1_M26(o3, ST48), f1_M25(o3, ST48))
# S4: reclassify at tolerance 0.10
base_cols <- c("case_number", "fiscal_year", "state_fips", "mw_area", "fy", "received", "decided", "begin", "end", "wage", "employer_id",
  "cert_pos", "hours", "piece", "flc", "exp_months", "soc7", "soc_grp", "skill", "exp_class", "new_rule")
TOL <- 0.10
o4 <- f1_classify(o0[, base_cols]) %>% select(-any_of("old_rates")) %>%
  left_join(o0 %>% distinct(state_fips, region, part), by = "state_fips")
TOL <- 0.05
o4 <- f1_attach_doses(o4, doses)
cls_tab <- bind_rows(o %>% filter(fy == 2026, new_rule, !piece, soc_grp == "field", state_fips %in% ST48) %>% count(cls) %>% mutate(tol = 0.05),
  o4 %>% filter(fy == 2026, new_rule, !piece, soc_grp == "field", state_fips %in% ST48) %>% count(cls) %>% mutate(tol = 0.10))
print(cls_tab)
res$S4 <- run_sens("S4 tolerance 0.10", f1_M26(o4, ST48), f1_M25(o4, ST48), tol = 0.10)
# S5
res$S5 <- run_sens("S5 non-field SOC orders added (P2, P4)", f1_M26(o, ST48, soc = c("field", "nonfield")), NULL, nonfield = TRUE, do_p3 = FALSE)
res <- bind_rows(res) %>% mutate(inference = "999-draw state-cluster bootstrap, percentile")
print(res %>% select(sensitivity, item, estimate, ci_low, ci_high, n_orders), n = 200, width = 200)
write_csv(res, file.path(out, "f1_sensitivities.csv")); write_csv(cls_tab, file.path(out, "f1_s4_class_counts.csv"))
message("done ", format(Sys.time()))
