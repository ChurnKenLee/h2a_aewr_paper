# Identification search, strategy F1 (Phase 3 design, step 1): support for
# counterfactual market-wage recovery.
#
# Usage (repository root):
#   R_LIBS_USER=$(ls -d $PWD/renv/library/*/R-*/* | head -1) Rscript --vanilla \
#     explorations/identification_search/F1/01_support.R "$S"
# Reads $S/shared/outcomes_case_rows.parquet, $S/triage/cred_job_orders_fy2018_2025.parquet,
# data/intermediate AEWR schedules and disclosure records, the raw FY2026 Q3
# disclosure (experience months), the A1 sourced minimum-wage table and
# $S/shared/split_state_by_former_region.csv.
# Writes $S/F1/f1_orders.parquet (classified orders, all periods) and
# f1_support_*.csv.
#
# Integrity: support statistics only. Counts and hours-weighted shares of
# orders by floor class (how much of the offered-wage distribution is observed
# above every floor, i.e. uncensored). No wage level, wage ratio or wage change
# of above-floor offers is computed for any year from FY2024 on, and no model
# is fitted. The FY2026 class shares computed here are recorded as prior
# exposure in the registry.
source("explorations/identification_search/F1/00_helpers.R")
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- f1_out_dir(S)
FYS <- c(2008:2026)

o <- f1_orders(S, FYS)
o <- f1_classify(o)
split <- read_csv(file.path(S, "shared", "split_state_by_former_region.csv"), col_types = cols(.default = "c"), progress = FALSE) %>%
  transmute(state_fips = unit, region = stratum, part)
o <- o %>% left_join(split, by = "state_fips")
write_parquet(o %>% select(-any_of(c("old_rates"))), file.path(out, "f1_orders.parquet"))
message("orders: ", nrow(o), "; FY2026 new-rule orders: ", sum(o$fy == 2026 & o$new_rule))

# ---- 1. By period: every FY, old vs new rule, SOC group ---------------------------
per <- f1_shares(o %>% filter(fy != 2026 | new_rule), fy, new_rule, soc_grp) %>% arrange(fy, soc_grp)
write_csv(per, file.path(out, "f1_support_by_period.csv"))
per_np <- f1_shares(o %>% filter(fy != 2026 | new_rule, !piece), fy, new_rule, soc_grp) %>% mutate(sample = "non-piece")
write_csv(per_np, file.path(out, "f1_support_by_period_nonpiece.csv"))
print(per_np %>% filter(soc_grp != "nonfield") %>% select(fy, soc_grp, orders, hours, orders_above, sh_above, sh_at_floor, sh_between, sh_below, sh_unknown), n = 30, width = 200)

# ---- 2. FY2026 new rule ------------------------------------------------------------
n26 <- o %>% filter(fy == 2026, new_rule)
write_csv(f1_shares(n26, soc_grp, piece, skill, exp_class), file.path(out, "f1_support_fy2026_cells.csv"))
print(f1_shares(n26, soc_grp, piece), width = 200)
ft <- n26 %>% filter(cls == "at_floor") %>% group_by(soc_grp, piece, floor_types) %>% summarise(orders = n(), hours = sum(hours), .groups = "drop") %>%
  group_by(soc_grp, piece) %>% mutate(share_of_at_floor = hours / sum(hours)) %>% ungroup() %>% arrange(soc_grp, piece, desc(hours))
write_csv(ft, file.path(out, "f1_support_fy2026_floor_types.csv"))
print(ft %>% filter(!piece), n = 20)

# Model sample (declared primary): hourly, non-piece, SOC known, floor known
ms <- n26 %>% filter(!piece, soc_grp != "unknown")
st <- f1_shares(ms, state_fips, part) %>%
  left_join(ms %>% group_by(state_fips) %>% summarise(employers_above = n_distinct(employer_id[cls == "above"]),
    mw_floored = any(mw_max > us_s1_ref + TOL, na.rm = TRUE), .groups = "drop"), by = "state_fips") %>% arrange(desc(hours))
write_csv(st, file.path(out, "f1_support_fy2026_state.csv"))
print(st %>% select(state_fips, part, orders, hours, orders_above, employers_above, sh_above, sh_at_floor, sh_at_old_aewr, sh_below, mw_floored), n = 60)
nat <- bind_rows(
  f1_shares(ms %>% mutate(scope = "all states"), scope),
  f1_shares(ms %>% mutate(scope = paste("part:", part)), scope),
  f1_shares(ms %>% mutate(scope = paste("soc:", soc_grp)), scope),
  f1_shares(ms %>% mutate(scope = paste("skill:", skill)), scope),
  f1_shares(ms %>% mutate(scope = paste("mw_floored:", mw_max > us_s1_ref + TOL)), scope),
  f1_shares(n26 %>% filter(piece, soc_grp != "unknown") %>% mutate(scope = "piece-rate orders (excluded)"), scope),
  f1_shares(n26 %>% filter(soc_grp == "unknown") %>% mutate(scope = "SOC unknown (excluded)"), scope)) %>%
  mutate(states_with_10plus_above = NA_integer_)
cnt <- ms %>% group_by(part, state_fips) %>% summarise(k = sum(cls == "above"), .groups = "drop")
nat$states_with_10plus_above[nat$scope == "all states"] <- sum(cnt$k >= 10)
nat$states_with_10plus_above[nat$scope == "part: confirmation"] <- sum(cnt$k[cnt$part %in% "confirmation"] >= 10)
nat$states_with_10plus_above[nat$scope == "part: exploration"] <- sum(cnt$k[cnt$part %in% "exploration"] >= 10)
write_csv(nat, file.path(out, "f1_support_fy2026_summary.csv"))
print(nat, width = 250)

# Suspected unseen floors: above-floor hours at a state x SOC-group wage value
# (rounded to the cent) offered by >= 3 distinct employers (registry sensitivity S3).
mp <- function(d) {
  a <- d %>% filter(cls == "above") %>% mutate(wr = round(wage, 2))
  k <- a %>% group_by(state_fips, soc_grp, wr) %>% summarise(n_emp = n_distinct(employer_id), .groups = "drop")
  a %>% left_join(k, by = c("state_fips", "soc_grp", "wr")) %>%
    summarise(above_orders = n(), above_hours = sum(hours), share_hours_at_shared_value_3emp = sum(hours[n_emp >= 3]) / sum(hours),
      share_orders_at_shared_value_3emp = mean(n_emp >= 3))
}
mpt <- bind_rows(mp(ms) %>% mutate(period = "FY2026 new rule (model sample)"),
  mp(o %>% filter(fy == 2025, !piece, soc_grp == "field")) %>% mutate(period = "FY2025 field non-piece"),
  mp(o %>% filter(fy %in% 2011:2015, soc_grp == "field")) %>% mutate(period = "FY2011-15 field"))
write_csv(mpt, file.path(out, "f1_support_masspoints.csv")); print(mpt)

# ---- 3. Incumbent support for the pass-through test (theta) -------------------------
# Unit: employer x state, field SOC, hourly non-piece orders. Above-floor in the
# base FY if all (primary) or >= half (sensitivity) of its base-FY hours are 'above'
# and none are floor-unknown; present in the next FY if it has any such order then.
inc <- function(base, nxt, new_only) {
  b <- o %>% filter(fy == base, !piece, soc_grp == "field", !is.na(employer_id)) %>%
    group_by(employer_id, state_fips) %>% summarise(h = sum(hours), sh_above = sum(hours[cls == "above"]) / h,
      any_unknown = any(cls == "floor_unknown"), n_orders = n(), .groups = "drop") %>% filter(!any_unknown)
  n <- o %>% filter(fy == nxt, new_rule | !new_only, !piece, soc_grp == "field", !is.na(employer_id)) %>%
    distinct(employer_id, state_fips) %>% mutate(present = TRUE)
  b %>% left_join(n, by = c("employer_id", "state_fips")) %>% mutate(present = coalesce(present, FALSE)) %>%
    left_join(split, by = "state_fips")
}
incs <- list(
  `FY2025->FY2026 (estimation)` = inc(2025, 2026, TRUE),
  `FY2024->FY2025 (placebo)` = inc(2024, 2025, FALSE),
  `FY2023->FY2024 (design calibration)` = inc(2023, 2024, FALSE))
isum <- bind_rows(lapply(names(incs), function(nm) {
  x <- incs[[nm]]
  bind_rows(
    x %>% filter(sh_above >= 0.9999) %>% summarise(units_base = n(), units_present = sum(present), hours_base_present = sum(h[present]),
      states_present = n_distinct(state_fips[present]), states_5plus = sum(table(state_fips[present]) >= 5),
      conf_units_present = sum(present & part %in% "confirmation"), conf_states_present = n_distinct(state_fips[present & part %in% "confirmation"])) %>%
      mutate(definition = "all base hours above"),
    x %>% filter(sh_above >= 0.5) %>% summarise(units_base = n(), units_present = sum(present), hours_base_present = sum(h[present]),
      states_present = n_distinct(state_fips[present]), states_5plus = sum(table(state_fips[present]) >= 5),
      conf_units_present = sum(present & part %in% "confirmation"), conf_states_present = n_distinct(state_fips[present & part %in% "confirmation"])) %>%
      mutate(definition = ">= half of base hours above"),
    x %>% summarise(units_base = n(), units_present = sum(present), hours_base_present = sum(h[present]),
      states_present = n_distinct(state_fips[present]), states_5plus = NA_integer_, conf_units_present = NA_integer_, conf_states_present = NA_integer_) %>%
      mutate(definition = "all field non-piece units (reference)")) %>% mutate(transition = nm)
}))
write_csv(isum, file.path(out, "f1_support_incumbents.csv")); print(isum, width = 250)
ist <- incs[[1]] %>% filter(sh_above >= 0.9999, present) %>% count(state_fips, part, name = "units") %>% arrange(desc(units))
write_csv(ist, file.path(out, "f1_support_incumbents_fy2026_by_state.csv"))
message("done ", format(Sys.time()))
