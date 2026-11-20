# Identification search, strategy F1: POST-OUTCOME (not declared in the registry).
# P1 on the design-stage incumbent frame. The registry's INC requires >= 1 new-rule
# M26 order in FY2026 (classes above / at_floor / at_old_aewr); the design stage
# counted presence on any new-rule field non-piece order (352 units). 67 of those
# units have only below_skill2 / below_floor FY2026 orders and drop out of the declared
# INC (285 units), i.e. presence is conditioned on the FY2026 offer class. This script
# re-estimates theta on the 352-unit frame (FY2026 unit wage over all its new-rule
# field non-piece orders), labeled post-outcome; it does not replace P1.
#
# Usage: R_LIBS_USER=... Rscript --vanilla explorations/identification_search/F1/10_post_outcome_p1_frame.R "$S"
source("explorations/identification_search/F1/03_est_helpers.R")
f1_check_registry()
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- f1_out_dir(S)
o <- read_parquet(file.path(out, "f1_orders.parquet"))
doses <- f1_doses(S); ST48 <- unique(doses$state_fips)
q <- o %>% filter(state_fips %in% ST48, !piece, soc_grp == "field", !is.na(employer_id))
b <- q %>% filter(fy == 2025) %>% group_by(employer_id, state_fips) %>%
  summarise(h = sum(hours), sh_above = sum(hours[cls == "above"]) / sum(hours), any_unknown = any(cls == "floor_unknown"),
    mw_area = mw_area[which.max(hours)], w_base = sum(wage * hours) / sum(hours), .groups = "drop") %>% filter(!any_unknown, sh_above >= 0.9999)
nx <- q %>% filter(fy == 2026, new_rule) %>% group_by(employer_id, state_fips) %>%
  summarise(w_next = sum(wage * hours) / sum(hours), only_below = all(cls %in% c("below_floor", "below_skill2")), .groups = "drop")
d <- b %>% inner_join(nx, by = c("employer_id", "state_fips")) %>% left_join(doses, by = c("state_fips", "mw_area")) %>%
  mutate(dw = w_next - w_base, dF = eff_cut_u)
stopifnot(!anyNA(d$dF))
r <- p1_fit(d$dF, d$dw, d$h, d$state_fips) %>% mutate(item = "PO1_theta_design_frame", role = "post-outcome",
  sample = "352-unit design frame: presence on any new-rule field non-piece FY2026 order", regressor = "eff_cut_u",
  weights = "unit FY2025 certified hours", inference = "state CV3J jackknife, t(G-1); alternate state CRV1",
  note = sprintf("units %d, of which %d have only below_skill2/below_floor FY2026 orders (hours-weighted mean dw of those: %.3f vs %.3f for the rest)",
    nrow(d), sum(d$only_below), with(d[d$only_below, ], sum(h * dw) / sum(h)), with(d[!d$only_below, ], sum(h * dw) / sum(h))))
print(r %>% select(item, estimate, se, ci_low, ci_high, p_theta0, p_theta1, n_units, n_states, note), width = 250)
write_csv(r, file.path(out, "f1_post_outcome_p1_frame.csv"))
