# F1 blind review (Phase 6), POST-OUTCOME check r03: what drives V1 / P3.
#  (a) Orders and hours behind S26(a25(1+g)) and S25(a25).
#  (b) Delta_S(m=1) under alternative treatments of suspected unseen floors:
#      primary; S3 (all shared values censored); only FY2025 offers equal to a prior
#      reference AEWR of the state (stale floors, 2019+) censored; S3 restricted to
#      non-round values (round dollar or quarter values kept as market wages);
#      and g in {0, 0.0376} for each.
#  (c) Balanced employer x state units above a25 in FY2025: where are they in FY2026?
# Same bootstrap machinery as 05_p2_p3.R (999 state draws, seed 20261102).
# Usage: R_LIBS_USER=... Rscript --vanilla explorations/identification_search/F1/review/r03_p3_v1.R "$S"
source("explorations/identification_search/F1/03_est_helpers.R")
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "F1", "review"); dir.create(out, showWarnings = FALSE)
o <- read_parquet(file.path(S, "F1", "f1_orders.parquet"))
doses <- f1_doses(S); ST48 <- sort(unique(doses$state_fips)); g <- f1_g_frozen(file.path(S, "F1"))
o <- f1_attach_doses(o, doses)
set.seed(SEED); cnt <- boot_counts(ST48)

# (a)
M26 <- f1_M26(o, ST48); M25 <- f1_M25(o, ST48)
a <- tibble(stat = c("FY2026 orders above a25(1+g)", "FY2025 orders above a25"),
  orders = c(sum(M26$cls == "above" & M26$wage > M26$a25d * (1 + g) + TOL), sum(M25$cls == "above" & M25$wage > M25$a25d + TOL)),
  employers = c(n_distinct(M26$employer_id[M26$cls == "above" & M26$wage > M26$a25d * (1 + g) + TOL]), n_distinct(M25$employer_id[M25$cls == "above" & M25$wage > M25$a25d + TOL])),
  states = c(n_distinct(M26$state_fips[M26$cls == "above" & M26$wage > M26$a25d * (1 + g) + TOL]), n_distinct(M25$state_fips[M25$cls == "above" & M25$wage > M25$a25d + TOL])),
  hours = c(sum(M26$hours[M26$cls == "above" & M26$wage > M26$a25d * (1 + g) + TOL]), sum(M25$hours[M25$cls == "above" & M25$wage > M25$a25d + TOL])),
  total_hours = c(sum(M26$hours), sum(M25$hours)))
print(a); write_csv(a, file.path(out, "r03a_counts.csv"))

# (b) variants
ref <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>% filter(occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference"),
  worker_category == "covered_workers", as.Date(effective_start) >= as.Date("2019-01-01")) %>% transmute(state_fips, v = as.numeric(aewr)) %>% distinct()
stale <- function(d) {   # FY2025 'above' offers equal (within TOL) to any 2019+ reference AEWR of the state -> censored
  m <- d %>% filter(cls == "above") %>% select(oid, state_fips, wage) %>% inner_join(ref, by = "state_fips", relationship = "many-to-many") %>%
    filter(abs(wage - v) <= TOL) %>% distinct(oid)
  d %>% mutate(cls = ifelse(oid %in% m$oid, "at_floor", cls))
}
s3_nonround <- function(oo) {
  k <- oo %>% filter(cls == "above", !piece) %>% mutate(wr = round(wage, 2)) %>% group_by(fy, state_fips, soc_grp, wr) %>%
    summarise(n_emp = n_distinct(employer_id), .groups = "drop")
  oo %>% mutate(wr = round(wage, 2)) %>% left_join(k, by = c("fy", "state_fips", "soc_grp", "wr")) %>%
    mutate(s3 = cls == "above" & coalesce(n_emp, 0L) >= 3L & abs(wr * 4 - round(wr * 4)) > 0.005, cls = ifelse(s3, "at_floor", cls)) %>% select(-wr, -n_emp, -s3)
}
o3 <- f1_apply_S3(o); o3n <- s3_nonround(o)
V <- list(primary = list(M26, M25), S3 = list(f1_M26(o3, ST48), f1_M25(o3, ST48)),
  `FY2025 stale-AEWR offers censored` = list(M26, stale(M25)),
  `S3 non-round values only` = list(f1_M26(o3n, ST48), f1_M25(o3n, ST48)),
  `S3 FY2025 only` = list(M26, f1_M25(o3, ST48)), `S3 FY2026 only` = list(f1_M26(o3, ST48), M25))
res <- bind_rows(lapply(names(V), function(nm) bind_rows(
  p3_rows(V[[nm]][[1]], V[[nm]][[2]], g, ST48, cnt, nm, ms = 1) %>% mutate(g_used = g),
  p3_rows(V[[nm]][[1]], V[[nm]][[2]], 0, ST48, cnt, nm, ms = 1) %>% mutate(g_used = 0))))
res <- res %>% mutate(ci_excludes_0 = !(ci_low <= 0 & ci_high >= 0))
print(res %>% select(scope, g_used, S26, S25, estimate, ci_low, ci_high, ci_excludes_0), width = 200)
write_csv(res, file.path(out, "r03b_delta_s_variants.csv"))

# M2-implied Delta_S under S3 (same construction as 02_design_precision.R, on the S3-censored FY2025 data)
f25s3 <- f1_M25(o3, ST48)
m2s3 <- sum(f25s3$hours[f25s3$cls == "above" & f25s3$wage > f25s3$a25d + (f25s3$a25d * (1 + g) - f25s3$FU) + TOL]) / sum(f25s3$hours)
s25s3 <- sum(f25s3$hours[f25s3$cls == "above" & f25s3$wage > f25s3$a25d + TOL]) / sum(f25s3$hours)
message(sprintf("S3: M2-implied S26 %.5f, S25(a25) %.5f, M2-implied Delta_S %.5f", m2s3, s25s3, m2s3 - s25s3))

# (c) balanced units above a25 in FY2025
u25 <- M25 %>% filter(!is.na(employer_id)) %>% group_by(employer_id, state_fips) %>%
  summarise(h25 = sum(hours), sh_above_a25 = sum(hours[cls == "above" & wage > a25d + TOL]) / sum(hours), .groups = "drop") %>% filter(sh_above_a25 >= 0.9999)
u26 <- o %>% filter(fy == 2026, new_rule, !piece, soc_grp == "field", state_fips %in% ST48, !is.na(employer_id)) %>% group_by(employer_id, state_fips) %>%
  summarise(sh_above_a25g = sum(hours[cls == "above" & wage > a25d * (1 + g) + TOL]) / sum(hours), sh_at_floor = sum(hours[cls == "at_floor"]) / sum(hours),
    sh_old = sum(hours[cls == "at_old_aewr"]) / sum(hours), sh_above_any = sum(hours[cls == "above"]) / sum(hours), sh_below = sum(hours[cls %in% c("below_floor", "below_skill2")]) / sum(hours), .groups = "drop")
cc <- u25 %>% left_join(u26, by = c("employer_id", "state_fips")) %>% mutate(present = !is.na(sh_at_floor))
ccs <- cc %>% summarise(units = n(), n_present = sum(present), h25_total = sum(h25), sh_present_h = sum(h25[present]) / sum(h25),
  among_present_still_above_a25g = sum((h25 * sh_above_a25g)[present]) / sum(h25[present]),
  among_present_above_floor_below_a25g = sum((h25 * (sh_above_any - sh_above_a25g))[present]) / sum(h25[present]),
  among_present_at_floor = sum((h25 * sh_at_floor)[present]) / sum(h25[present]), among_present_old = sum((h25 * sh_old)[present]) / sum(h25[present]),
  among_present_below = sum((h25 * sh_below)[present]) / sum(h25[present]))
# (d) absolute hours above the cutoffs, all units and balanced employer x state units (denominator-free version of P3)
bal <- inner_join(M25 %>% filter(!is.na(employer_id)) %>% distinct(employer_id, state_fips), M26 %>% filter(!is.na(employer_id)) %>% distinct(employer_id, state_fips), by = c("employer_id", "state_fips"))
B26 <- M26 %>% semi_join(bal, by = c("employer_id", "state_fips")); B25 <- M25 %>% semi_join(bal, by = c("employer_id", "state_fips"))
absh <- function(d26, d25, lab) tibble(sample = lab, hours26_total = sum(d26$hours), hours25_total = sum(d25$hours),
  hours26_above_a25g = sum(d26$hours[d26$cls == "above" & d26$wage > d26$a25d * (1 + g) + TOL]), hours26_above_a25 = sum(d26$hours[d26$cls == "above" & d26$wage > d26$a25d + TOL]),
  hours25_above_a25 = sum(d25$hours[d25$cls == "above" & d25$wage > d25$a25d + TOL]))
ab <- bind_rows(absh(M26, M25, "all M26 / M25"), absh(B26, B25, "balanced employer x state")) %>% mutate(ratio_total = hours26_total / hours25_total,
  ratio_above_g = hours26_above_a25g / hours25_above_a25, ratio_above_g0 = hours26_above_a25 / hours25_above_a25,
  M1_entry_only_S26 = hours25_above_a25 / hours26_total)
print(ab, width = 250); write_csv(ab, file.path(out, "r03d_absolute_hours.csv"))
print(ccs, width = 250); write_csv(ccs, file.path(out, "r03c_above_a25_units.csv"))
