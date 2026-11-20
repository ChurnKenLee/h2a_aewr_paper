# F1 blind review (Phase 6), POST-OUTCOME check r05: multi-state job orders.
# F1 classifies each case x state row against that state's floors. A multi-state
# itinerary order (e.g. custom harvest) offers one wage, which may be the floor of
# ANOTHER state on its itinerary (highest applicable rate). Such rows are classified
# 'above' although they sit at a floor. Counts 'above' hours (FY2025 M25, FY2026 M26,
# 48 states) whose case spans >1 state and whose wage is within $0.05 of a floor of
# another itinerary state: FY2025 = that state's reference AEWR in the FY2025 window;
# FY2026 = that state's US/H-2A skill I/II rates or MW values. Re-runs P1 (declared
# frame and 352 frame) and P3 with those rows reclassified as at_floor (censored).
# Usage: R_LIBS_USER=... Rscript --vanilla explorations/identification_search/F1/review/r05_multistate.R "$S"
source("explorations/identification_search/F1/03_est_helpers.R")
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "F1", "review"); dir.create(out, showWarnings = FALSE)
o <- read_parquet(file.path(S, "F1", "f1_orders.parquet"))
doses <- f1_doses(S); ST48 <- sort(unique(doses$state_fips)); g <- f1_g_frozen(file.path(S, "F1"))
o <- f1_attach_doses(o, doses)
nst <- o %>% group_by(case_number, fiscal_year) %>% summarise(n_states = n_distinct(state_fips), .groups = "drop")
o <- o %>% left_join(nst, by = c("case_number", "fiscal_year"))
old <- f1_old_aewr() %>% select(state_fips, old_rates) %>% tidyr::unnest(old_rates) %>% rename(v = old_rates)
sr <- f1_state_skill_rates() %>% pivot_longer(-state_fips, values_to = "v") %>% select(state_fips, v)
mwv <- f1_minwage() %>% select(state_fips, v = mw)
newv <- bind_rows(sr, mwv) %>% distinct()
other_floor <- function(d, tab) {
  it <- o %>% filter(fiscal_year %in% unique(d$fiscal_year)) %>% distinct(case_number, fiscal_year, it_state = state_fips)
  m <- d %>% filter(cls == "above", n_states > 1) %>% select(oid, case_number, fiscal_year, state_fips, wage) %>%
    inner_join(it, by = c("case_number", "fiscal_year"), relationship = "many-to-many") %>% filter(it_state != state_fips) %>%
    inner_join(tab, by = c("it_state" = "state_fips"), relationship = "many-to-many") %>% filter(abs(wage - v) <= TOL) %>% distinct(oid, v) %>%
    group_by(oid) %>% summarise(v = max(v), .groups = "drop")
  d %>% left_join(m, by = "oid") %>% mutate(ms_floor = !is.na(v), c_floor = ifelse(ms_floor, v, c_floor), cls = ifelse(ms_floor, "at_floor", cls)) %>% select(-v)
}
o25 <- other_floor(o %>% filter(fy == 2025), old); o26 <- other_floor(o %>% filter(fy == 2026, new_rule), newv)
sumtab <- bind_rows(
  o25 %>% filter(state_fips %in% ST48, !piece, soc_grp == "field") %>% summarise(set = "FY2025 field non-piece", above_hours = sum(hours[cls == "above" | ms_floor]),
    multistate_above_hours = sum(hours[(cls == "above" | ms_floor) & n_states > 1]), reclassified_hours = sum(hours[ms_floor]), reclassified_orders = sum(ms_floor)),
  o26 %>% filter(state_fips %in% ST48, !piece, soc_grp == "field") %>% summarise(set = "FY2026 field non-piece", above_hours = sum(hours[cls == "above" | ms_floor]),
    multistate_above_hours = sum(hours[(cls == "above" | ms_floor) & n_states > 1]), reclassified_hours = sum(hours[ms_floor]), reclassified_orders = sum(ms_floor))) %>%
  mutate(reclassified_share_of_above = reclassified_hours / above_hours)
print(sumtab, width = 250); write_csv(sumtab, file.path(out, "r05_multistate_counts.csv"))
# top reclassified FY2025 values
print(o25 %>% filter(ms_floor, state_fips %in% ST48, !piece, soc_grp == "field") %>% count(state_fips, wage = round(wage, 2), wt = hours, sort = TRUE) %>% head(10))
# P3 and P1 with multistate rows censored
oo <- bind_rows(o %>% filter(!(fy %in% c(2025, 2026))), o %>% filter(fy == 2026, !new_rule), o25, o26)
set.seed(SEED); cnt <- boot_counts(ST48)
p3 <- bind_rows(p3_rows(f1_M26(oo, ST48), f1_M25(oo, ST48), g, ST48, cnt, "multistate floors censored", ms = 1),
  p3_rows(f1_M26(oo, ST48), f1_M25(oo, ST48), 0, ST48, cnt, "multistate floors censored, g = 0", ms = 1))
print(p3 %>% select(scope, S26, S25, estimate, ci_low, ci_high))
q <- oo %>% filter(state_fips %in% ST48, !piece, soc_grp == "field", !is.na(employer_id))
b <- q %>% filter(fy == 2025) %>% group_by(employer_id, state_fips) %>% summarise(h = sum(hours), sh_above = sum(hours[cls == "above"]) / sum(hours),
  any_unknown = any(cls == "floor_unknown"), mw_area = mw_area[which.max(hours)], w_base = sum(wage * hours) / sum(hours), .groups = "drop") %>% filter(!any_unknown, sh_above >= 0.9999)
nx <- q %>% filter(fy == 2026, new_rule) %>% group_by(employer_id, state_fips) %>% summarise(w_all = sum(wage * hours) / sum(hours),
  any_m26 = any(cls %in% c("above", "at_floor", "at_old_aewr")),
  w_m26 = sum((wage * hours)[cls %in% c("above", "at_floor", "at_old_aewr")]) / sum(hours[cls %in% c("above", "at_floor", "at_old_aewr")]), .groups = "drop")
d <- b %>% inner_join(nx, by = c("employer_id", "state_fips")) %>% left_join(doses, by = c("state_fips", "mw_area")) %>% mutate(dF = eff_cut_u)
p1 <- bind_rows(p1_fit(d$dF[d$any_m26], (d$w_m26 - d$w_base)[d$any_m26], d$h[d$any_m26], d$state_fips[d$any_m26]) %>% mutate(frame = "declared-type frame, multistate floors censored"),
  p1_fit(d$dF, d$w_all - d$w_base, d$h, d$state_fips) %>% mutate(frame = "352-type frame, multistate floors censored"))
print(p1 %>% select(frame, estimate, ci_low, ci_high, p_theta0, p_theta1, n_units, n_states))
write_csv(bind_rows(p3 %>% mutate(item = "P3"), p1 %>% mutate(item = "P1")), file.path(out, "r05_multistate_p1_p3.csv"))

# Chance benchmark: the same matching with every wage shifted by +$0.37 / -$0.37
bench <- function(d, tab, shift) {
  it <- o %>% filter(fiscal_year %in% unique(d$fiscal_year)) %>% distinct(case_number, fiscal_year, it_state = state_fips)
  m <- d %>% filter(cls == "above", n_states > 1) %>% mutate(wage = wage + shift) %>% select(oid, case_number, fiscal_year, state_fips, wage, hours) %>%
    inner_join(it, by = c("case_number", "fiscal_year"), relationship = "many-to-many") %>% filter(it_state != state_fips) %>%
    inner_join(tab, by = c("it_state" = "state_fips"), relationship = "many-to-many") %>% filter(abs(wage - v) <= TOL) %>% distinct(oid, hours)
  sum(m$hours)
}
d25 <- o %>% filter(fy == 2025, state_fips %in% ST48, !piece, soc_grp == "field"); d26 <- o %>% filter(fy == 2026, new_rule, state_fips %in% ST48, !piece, soc_grp == "field")
bt <- tibble(set = c("FY2025", "FY2026"), actual = c(bench(d25, old, 0), bench(d26, newv, 0)),
  shift_plus = c(bench(d25, old, 0.37), bench(d26, newv, 0.37)), shift_minus = c(bench(d25, old, -0.37), bench(d26, newv, -0.37)))
print(bt); write_csv(bt, file.path(out, "r05_chance_benchmark.csv"))
# Multi-state share among the 67 units the declared INC drops (original classification)
q0 <- o %>% filter(state_fips %in% ST48, !piece, soc_grp == "field", !is.na(employer_id))
b0 <- q0 %>% filter(fy == 2025) %>% group_by(employer_id, state_fips) %>% summarise(sh_above = sum(hours[cls == "above"]) / sum(hours),
  any_unknown = any(cls == "floor_unknown"), ms25 = sum(hours[n_states > 1]) / sum(hours), .groups = "drop") %>% filter(!any_unknown, sh_above >= 0.9999)
n0 <- q0 %>% filter(fy == 2026, new_rule) %>% group_by(employer_id, state_fips) %>% summarise(any_m26 = any(cls %in% c("above", "at_floor", "at_old_aewr")),
  ms26 = sum(hours[n_states > 1]) / sum(hours), .groups = "drop")
print(b0 %>% inner_join(n0, by = c("employer_id", "state_fips")) %>% group_by(any_m26) %>% summarise(units = n(), mean_ms25 = mean(ms25), mean_ms26 = mean(ms26)))

# After the multi-state correction: FY2026 above share; INC FY2026 position; theta_T (04 construction)
M26c <- f1_M26(oo, ST48)
message(sprintf("FY2026 M26 field share above every floor: original %.4f, multistate-corrected %.4f",
  sum(f1_M26(o, ST48)$hours[f1_M26(o, ST48)$cls == "above"]) / sum(f1_M26(o, ST48)$hours), sum(M26c$hours[M26c$cls == "above"]) / sum(M26c$hours)))
pos <- q %>% filter(fy == 2026, new_rule) %>% group_by(employer_id, state_fips) %>% summarise(f = sum(hours[cls == "at_floor"]) / sum(hours),
  a = sum(hours[cls == "above"]) / sum(hours), .groups = "drop") %>% inner_join(b, by = c("employer_id", "state_fips"))
message(sprintf("corrected INC-type units: %d; FY2025-hours-weighted share of FY2026 hours at a floor %.3f, above %.3f", nrow(pos), sum(pos$h * pos$f) / sum(pos$h), sum(pos$h * pos$a) / sum(pos$h)))
inc <- d %>% filter(any_m26) %>% mutate(dw = w_m26 - w_base)
pl_b <- q %>% filter(fy == 2024) %>% group_by(employer_id, state_fips) %>% summarise(h = sum(hours), sh_above = sum(hours[cls == "above"]) / sum(hours),
  any_unknown = any(cls == "floor_unknown"), mw_area = mw_area[which.max(hours)], w_base = sum(wage * hours) / sum(hours), .groups = "drop") %>% filter(!any_unknown, sh_above >= 0.9999)
pl <- pl_b %>% inner_join(q %>% filter(fy == 2025) %>% group_by(employer_id, state_fips) %>% summarise(w_next = sum(wage * hours) / sum(hours), .groups = "drop"),
  by = c("employer_id", "state_fips")) %>% left_join(doses, by = c("state_fips", "mw_area")) %>% mutate(dw = w_next - w_base, dF_pre = a25 - a24)
thT <- function(A, B) (sum(A$h * A$dw) / sum(A$h) - sum(B$h * B$dw) / sum(B$h)) / (sum(A$h * A$dF) / sum(A$h) - sum(B$h * B$dF_pre) / sum(B$h))
stu <- sort(union(inc$state_fips, pl$state_fips)); set.seed(SEED); cc <- boot_counts(stu)
dr <- apply(cc, 2, function(k) { w <- setNames(k, stu); A <- inc %>% mutate(h = h * w[state_fips]); B <- pl %>% mutate(h = h * w[state_fips]); thT(A, B) })
message(sprintf("theta_T multistate-corrected: %.3f [%.3f, %.3f] (INC %d units, placebo %d units)", thT(inc, pl), quantile(dr, 0.025), quantile(dr, 0.975), nrow(inc), nrow(pl)))
