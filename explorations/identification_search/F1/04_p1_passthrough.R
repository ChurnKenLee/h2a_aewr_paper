# Identification search, strategy F1 (Phases 4-5): P1 pass-through theta and its
# declared secondaries, FALS1 (gating placebo) and FALS2 (pre-period pass-through).
# Registry: estimands.P1_theta, validation_and_falsification FALS1/FALS2.
#
# Usage (repository root, after 01_support.R):
#   R_LIBS_USER=$(ls -d $PWD/renv/library/*/R-*/* | head -1) Rscript --vanilla \
#     explorations/identification_search/F1/04_p1_passthrough.R "$S"
# Writes $S/F1/f1_p1.csv and f1_p1.rds.
#
# Implementation readings (registry text silent; fixed here before running):
#   - INC presence and FY2026 unit wage use the unit's M26 orders (new rule, field,
#     non-piece, classes above / at_floor / at_old_aewr), as the registry's INC text says.
#   - INC_placebo FY2025 unit wage uses every FY2025 rq13d field non-piece hourly
#     order of the unit ("any FY2025 rq13d order"), any floor class.
#   - '>= half above' variant: FY2025 unit wage over its M25-class orders (above, at_floor).
#   - Universe: the 48 dose-file states. Unit MW area = area of its largest base-FY order.
source("explorations/identification_search/F1/03_est_helpers.R")
f1_check_registry()
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- f1_out_dir(S)
o <- read_parquet(file.path(out, "f1_orders.parquet"))
doses <- f1_doses(S); ST48 <- unique(doses$state_fips); stopifnot(length(ST48) == 48)
q <- o %>% filter(state_fips %in% ST48, !piece, soc_grp == "field", !is.na(employer_id))

base_units <- function(base, def) {
  q %>% filter(fy == base) %>% group_by(employer_id, state_fips) %>%
    summarise(h_all = sum(hours), sh_above = sum(hours[cls == "above"]) / sum(hours), any_unknown = any(cls == "floor_unknown"),
      mw_area = mw_area[which.max(hours)],
      sh_exp3 = sum(hours[!is.na(exp_months) & exp_months >= 3]) / sum(hours),
      h = sum(hours[cls %in% c("above", "at_floor")]),
      w_base = sum((wage * hours)[cls %in% c("above", "at_floor")]) / sum(hours[cls %in% c("above", "at_floor")]), .groups = "drop") %>%
    filter(!any_unknown, if (def == "all") sh_above >= 0.9999 else sh_above >= 0.5)
}
# Primary: weight = unit base-FY certified hours (all base hours are 'above' in the primary definition)
inc26 <- function(def = "all") {
  nx <- f1_M26(o, ST48) %>% filter(!is.na(employer_id)) %>% group_by(employer_id, state_fips) %>%
    summarise(w_next = sum(wage * hours) / sum(hours), .groups = "drop")
  base_units(2025, def) %>% inner_join(nx, by = c("employer_id", "state_fips")) %>%
    left_join(doses, by = c("state_fips", "mw_area")) %>%
    mutate(dw = w_next - w_base, dF = eff_cut_u, dF_skill = ifelse(sh_exp3 > 0.5, pmax(US_s2, mw_20260101) - e25, eff_cut_u),
      dF_pre = a25 - a24)
}
inc_placebo <- function() {
  nx <- q %>% filter(fy == 2025) %>% group_by(employer_id, state_fips) %>% summarise(w_next = sum(wage * hours) / sum(hours), .groups = "drop")
  base_units(2024, "all") %>% inner_join(nx, by = c("employer_id", "state_fips")) %>%
    left_join(doses, by = c("state_fips", "mw_area")) %>% mutate(dw = w_next - w_base, dF = eff_cut_u, dF_pre = a25 - a24)
}
INC <- inc26("all"); INCh <- inc26("half"); PL <- inc_placebo()
stopifnot(!anyNA(INC$dF), !anyNA(INC$dw), !anyNA(PL$dF), !anyNA(PL$dF_pre), !anyNA(INCh$dF))
message(sprintf("INC units %d (states %d); INC half %d; INC_placebo %d (states %d)", nrow(INC), n_distinct(INC$state_fips),
  nrow(INCh), nrow(PL), n_distinct(PL$state_fips)))

rows <- bind_rows(
  p1_fit(INC$dF, INC$dw, INC$h, INC$state_fips) %>% mutate(item = "P1_theta", role = "primary", sample = "INC FY2025->FY2026, all FY2025 hours above",
    regressor = "eff_cut_u", weights = "unit FY2025 certified hours"),
  p1_fit(INC$dF_skill, INC$dw, INC$h, INC$state_fips) %>% mutate(item = "P1_theta_skill_floor", role = "secondary", sample = "INC",
    regressor = "max(US_s2, MW 2026-01-01) - e25 for units with majority FY2025 hours exp >= 3 months, else eff_cut_u", weights = "unit FY2025 certified hours"),
  p1_fit(INCh$dF, INCh$dw, INCh$h, INCh$state_fips) %>% mutate(item = "P1_theta_half_above", role = "secondary", sample = "INC, >= half of FY2025 hours above",
    regressor = "eff_cut_u", weights = "unit FY2025 M25-class certified hours"),
  p1_fit(INC$dF, INC$dw, rep(1, nrow(INC)), INC$state_fips) %>% mutate(item = "P1_theta_unweighted", role = "secondary", sample = "INC",
    regressor = "eff_cut_u", weights = "none"),
  p1_fit(PL$dF, PL$dw, PL$h, PL$state_fips) %>% mutate(item = "FALS1_placebo", role = "falsification_gating",
    sample = "INC_placebo FY2024->FY2025, all FY2024 hours above", regressor = "eff_cut_u (FY2026 floor change)", weights = "unit FY2024 certified hours"),
  p1_fit(PL$dF_pre, PL$dw, PL$h, PL$state_fips) %>% mutate(item = "FALS2_theta_pre", role = "falsification_reported",
    sample = "INC_placebo FY2024->FY2025", regressor = "a25 - a24 (actual FY2025 floor change)", weights = "unit FY2024 certified hours"))
rows <- rows %>% mutate(inference = "state CV3J jackknife, t(G-1); alternate state CRV1 t(G-1)")
fals1_fail <- rows$p_theta0[rows$item == "FALS1_placebo"] < 0.05
message("FALS1 p = ", signif(rows$p_theta0[rows$item == "FALS1_placebo"], 4), " -> ", ifelse(fals1_fail, "FAIL", "pass"))

# theta_T: (mean dw INC - mean dw INC_placebo) / (mean dF FY25->26 - mean dF FY24->25), hours-weighted
# 999-draw state-cluster bootstrap resampling states jointly across both transitions (seed 20261102), percentile CI.
ss <- function(d, dFcol) d %>% group_by(state_fips) %>% summarise(hdw = sum(h * dw), hdF = sum(h * .data[[dFcol]]), h = sum(h), .groups = "drop")
a <- ss(INC, "dF"); b <- ss(PL, "dF_pre")
stu <- sort(union(a$state_fips, b$state_fips))
mat <- function(x, col) { v <- setNames(rep(0, length(stu)), stu); v[x$state_fips] <- x[[col]]; v }
A <- sapply(c("hdw", "hdF", "h"), function(cn) mat(a, cn)); B <- sapply(c("hdw", "hdF", "h"), function(cn) mat(b, cn))
thT <- function(cA, cB) {
  num <- sum(cA * A[, "hdw"]) / sum(cA * A[, "h"]) - sum(cB * B[, "hdw"]) / sum(cB * B[, "h"])
  den <- sum(cA * A[, "hdF"]) / sum(cA * A[, "h"]) - sum(cB * B[, "hdF"]) / sum(cB * B[, "h"])
  c(theta_T = num / den, num = num, den = den)
}
est_T <- thT(rep(1, length(stu)), rep(1, length(stu)))
set.seed(SEED); cnt <- boot_counts(stu)
dr <- t(apply(cnt, 2, function(cc) thT(cc, cc)))
ciT <- pci(dr[, "theta_T"])
rows <- bind_rows(rows, tibble(item = "P1_theta_T", role = "secondary", estimate = est_T[["theta_T"]], se = sd(dr[, "theta_T"]),
  ci_low = ciT[1], ci_high = ciT[2], p_theta0 = NA_real_, p_theta1 = NA_real_,
  n_units = nrow(INC) + nrow(PL), n_states = length(stu), sample = "INC and INC_placebo",
  regressor = sprintf("mean dw diff %.4f / mean dF diff %.4f", est_T[["num"]], est_T[["den"]]), weights = "unit base-FY certified hours",
  inference = "999-draw state-cluster bootstrap (joint over transitions), percentile CI; SE = bootstrap SD",
  boot_p_ci_excl0 = !(ciT[1] <= 0 & ciT[2] >= 0), boot_p_ci_excl1 = !(ciT[1] <= 1 & ciT[2] >= 1)))
# descriptive means used in theta_T
desc <- tibble(item = c("INC mean dw", "INC mean dF", "INC_placebo mean dw", "INC_placebo mean dF_pre"),
  value = c(sum(INC$h * INC$dw) / sum(INC$h), sum(INC$h * INC$dF) / sum(INC$h), sum(PL$h * PL$dw) / sum(PL$h), sum(PL$h * PL$dF_pre) / sum(PL$h)))
print(desc)
print(rows %>% select(item, estimate, se, ci_low, ci_high, p_theta0, p_theta1, se_crv1, n_units, n_states), width = 200)
write_csv(rows, file.path(out, "f1_p1.csv"))
saveRDS(list(rows = rows, INC = INC, PL = PL, INCh = INCh, fals1_fail = fals1_fail, theta_T_draws = dr, desc = desc), file.path(out, "f1_p1.rds"))
message("done ", format(Sys.time()))
