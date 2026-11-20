# Identification search, strategy F1 (Phases 4-5): P2 binding-share function,
# P3 stability test (= V1, gating the M1 reading), S6 (P3 drift), V4 (mass points),
# G1 (at-floor shares). Registry: estimands.P2/P3, validation V1/V4, sensitivities S6,
# gap_model_implications G1.
#
# Usage (repository root):
#   R_LIBS_USER=$(ls -d $PWD/renv/library/*/R-*/* | head -1) Rscript --vanilla \
#     explorations/identification_search/F1/05_p2_p3.R "$S"
# Writes $S/F1/f1_p2.csv, f1_p3.csv, f1_v4.csv, f1_g1.csv, f1_p2p3.rds.
#
# Inference: 999-draw state-cluster bootstrap over the 48 dose-file states (seed
# 20261102), percentile 95% CI; one set of draws shared by all P2/P3 statistics.
# Single-state P2 rows have no state-cluster CI (one cluster): CI reported as NA.
# g = 0.0376 as frozen by 02_design_precision.R (read from f1_design_shares.csv).
source("explorations/identification_search/F1/03_est_helpers.R")
f1_check_registry()
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- f1_out_dir(S)
o <- read_parquet(file.path(out, "f1_orders.parquet"))
doses <- f1_doses(S); ST48 <- sort(unique(doses$state_fips))
g <- f1_g_frozen(out); message("g = ", g)
o <- f1_attach_doses(o, doses)
M26 <- f1_M26(o, ST48); M25 <- f1_M25(o, ST48); N26 <- f1_M26(o, ST48, soc = "nonfield")
stopifnot(!anyNA(M26$a25d), !anyNA(M25$a25d), all(abs(M26$a25d - M26$a25) < 1e-9))
set.seed(SEED); cnt <- boot_counts(ST48)

# ---- P2 -------------------------------------------------------------------------------
k <- M26 %>% group_by(state_fips) %>% summarise(n_above = sum(cls == "above"), .groups = "drop")
big <- k$state_fips[k$n_above >= 10]
p2 <- bind_rows(
  p2_rows(M26, g, ST48, cnt, "national (48 states), field"),
  p2_rows(M26 %>% filter(skill == "I"), g, ST48, cnt, "skill I, field"),
  p2_rows(M26 %>% filter(skill == "II"), g, ST48, cnt, "skill II, field"),
  p2_rows(N26, g, ST48, cnt, "non-field group (separate)"),
  p2_rows(M26 %>% filter(!state_fips %in% big), g, ST48, cnt, "other states pooled (< 10 above-floor orders), field"))
for (s in big) {
  r <- p2_rows(M26 %>% filter(state_fips == s), g, ST48, cnt, paste0("state ", s, ", field"))
  p2 <- bind_rows(p2, r %>% mutate(se = NA_real_, ci_low = NA_real_, ci_high = NA_real_))
}
p2 <- p2 %>% mutate(S26 = 1 - estimate, inference = ifelse(is.na(ci_low), "point only (single state: no state-cluster CI)",
  "999-draw state-cluster bootstrap, percentile"))
print(p2 %>% filter(!grepl("^state", scope)) %>% select(scope, grid, estimate, ci_low, ci_high, n_orders, n_above), n = 40, width = 200)
message("states with >= 10 above-floor M26 orders: ", length(big))

# ---- P3 / V1 and S6 ---------------------------------------------------------------------
# Balanced incumbents: employer x state units with M25 and M26 orders
bal <- inner_join(M25 %>% filter(!is.na(employer_id)) %>% distinct(employer_id, state_fips),
  M26 %>% filter(!is.na(employer_id)) %>% distinct(employer_id, state_fips), by = c("employer_id", "state_fips"))
B26 <- M26 %>% semi_join(bal, by = c("employer_id", "state_fips")); B25 <- M25 %>% semi_join(bal, by = c("employer_id", "state_fips"))
p3 <- bind_rows(
  p3_rows(M26, M25, g, ST48, cnt, "primary: M26 vs M25"),
  p3_rows(B26, B25, g, ST48, cnt, "balanced incumbents (employer x state in M25 and M26)"),
  p3_rows(M26, M25, 0, ST48, cnt, "S6: g = 0", ms = 1),
  p3_rows(M26, M25, 2 * g, ST48, cnt, "S6: g = 2 x 0.0376", ms = 1)) %>%
  mutate(ci_excludes_0 = !(ci_low <= 0 & ci_high >= 0), inference = "999-draw state-cluster bootstrap (states jointly for both years), percentile")
print(p3 %>% select(scope, grid, S26, S25, estimate, ci_low, ci_high, ci_excludes_0), width = 200)
v1 <- p3 %>% filter(scope == "primary: M26 vs M25", grid == "m=1")
message(sprintf("V1: Delta_S = %.4f [%.4f, %.4f] -> M1 stability %s", v1$estimate, v1$ci_low, v1$ci_high,
  ifelse(v1$ci_excludes_0, "REJECTED", "not rejected")))
message("balanced units: ", nrow(bal), "; orders FY26 ", nrow(B26), ", FY25 ", nrow(B25))

# ---- V4: mass points among 'above' hours ----------------------------------------------------
mpind <- function(d, pool) {   # pool: non-piece 'above' offers of the FY defining shared values
  kk <- pool %>% filter(cls == "above") %>% mutate(wr = round(wage, 2)) %>% group_by(state_fips, soc_grp, wr) %>%
    summarise(n_emp = n_distinct(employer_id), .groups = "drop")
  d %>% mutate(wr = round(wage, 2)) %>% left_join(kk, by = c("state_fips", "soc_grp", "wr")) %>% mutate(mp = coalesce(n_emp, 0L) >= 3L)
}
pool26 <- o %>% filter(fy == 2026, new_rule, !piece); pool25 <- o %>% filter(fy == 2025, !piece)
v4 <- list()
for (nm in c("M26 field", "M26 field + non-field", "M25 field")) {
  d <- switch(nm, `M26 field` = M26, `M26 field + non-field` = bind_rows(M26, N26), `M25 field` = M25) %>% filter(cls == "above")
  d <- mpind(d, if (grepl("M26", nm)) pool26 else pool25)
  sb <- share_boot(d, cbind(mp = d$mp * 1), ST48, cnt)
  v4[[nm]] <- tibble(item = "V4_masspoint_share", scope = nm, estimate = sb$est, se = sd(sb$draws), ci_low = pci(sb$draws)[1], ci_high = pci(sb$draws)[2],
    n_orders = nrow(d), n_orders_mp = sum(d$mp), n_states = n_distinct(d$state_fips))
}
v4 <- bind_rows(v4); print(v4)

# ---- G1 --------------------------------------------------------------------------------------
sb <- share_boot(M25, cbind(at_floor = (M25$cls == "at_floor") * 1), ST48, cnt)
g1 <- bind_rows(
  p2 %>% filter(scope == "national (48 states), field", grid %in% c("F_app", "a25")) %>%
    transmute(item = paste0("G1_b26(", grid, ")"), scope, estimate, se, ci_low, ci_high, n_orders),
  tibble(item = "G1_M25_at_floor_share", scope = "M25 field (FY2025 seen)", estimate = sb$est, se = sd(sb$draws),
    ci_low = pci(sb$draws)[1], ci_high = pci(sb$draws)[2], n_orders = nrow(M25)))
print(g1)

write_csv(p2, file.path(out, "f1_p2.csv")); write_csv(p3, file.path(out, "f1_p3.csv"))
write_csv(v4, file.path(out, "f1_v4.csv")); write_csv(g1, file.path(out, "f1_g1.csv"))
saveRDS(list(p2 = p2, p3 = p3, v4 = v4, g1 = g1, g = g, big = big, n_bal = nrow(bal)), file.path(out, "f1_p2p3.rds"))
message("done ", format(Sys.time()))
