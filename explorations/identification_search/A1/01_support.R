# Identification search, strategy A1 (Phase 3 design, step 1): dose
# construction and support statistics for every candidate dose and level.
#
# Usage (repository root):
#   Rscript --vanilla explorations/identification_search/A1/01_support.R "$S"
# Reads $S/shared outcome panels, data/intermediate AEWR schedules and
# explorations/identification_search/A1/sources/state_ag_minwage_fy2025_2026.csv.
# Writes $S/A1/a1_state_doses.csv, a1_county_doses.parquet,
# a1_employer_doses.parquet, a1_support.csv, a1_node_share.csv,
# a1_contract_calendar.csv, a1_binding_share_summary.csv.
#
# Integrity: no outcome is regressed on any dose and no FY2026 outcome or
# FY2026 offered wage is read. Outcome use is limited to (i) FE-only residual
# variance of FY2020-FY2025 calibration outcomes and (ii) predetermined FY2025
# (and FY2023-24) composition used to build the dose: the at-AEWR hours share
# and the calendar distribution of FY2025 contract days. All support statistics
# are on the two-period FY2025-FY2026 design frame with PPML weights (the
# unit's FY2020-FY2025 mean count), resolving triage disagreement 2.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(tidyr); library(readr)})
source("explorations/identification_search/shared/power_harness.R")
setFixest_nthreads(1)
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "A1"); dir.create(out, showWarnings = FALSE)
if (startsWith(paste0(S, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
CAL <- 2020:2025

# ---- State doses ----------------------------------------------------------------
mw <- read_csv("explorations/identification_search/A1/sources/state_ag_minwage_fy2025_2026.csv",
  col_types = cols(state_fips = "c", .default = col_guess()))
sub <- read_csv("explorations/identification_search/A1/sources/minwage_substate_counties.csv", col_types = cols(.default = "c"))
regional <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>%
  filter(occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference"), worker_category == "covered_workers") %>%
  transmute(state_fips, aewr = as.numeric(aewr), start = as.Date(effective_start), end = as.Date(effective_end_exclusive))
rate_on <- function(date) { r <- regional %>% filter(start <= date, is.na(end) | end > date) %>% select(state_fips, aewr)
  stopifnot(!anyDuplicated(r$state_fips)); r }
a24 <- rate_on(as.Date("2024-04-01")) %>% rename(a24 = aewr)
a25 <- rate_on(as.Date("2025-04-01")) %>% rename(a25 = aewr)
a25_lo <- regional %>% filter(start <= as.Date("2025-06-30"), is.na(end) | end > as.Date("2024-10-01")) %>%
  group_by(state_fips) %>% summarise(a25_min_in_window = min(aewr), .groups = "drop")
sk <- read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
  filter(trimws(source_sheet) == "Statewide 655.120(b)(1)(i)", as.Date(effective_start) == as.Date("2025-10-02")) %>%
  transmute(state_fips, k = paste0(worker_category, "_s", skill_level), r = as.numeric(hourly_aewr)) %>%
  pivot_wider(names_from = k, values_from = r)

cp <- read_parquet(path_processed("county_year_panel.parquet"),
  col_select = c("county_fips", "state_fips", "aewr_region_id", "emp_farm_2011", "year")) %>% filter(year == 2011) %>% select(-year)
stopifnot(is.character(cp$county_fips), !anyDuplicated(cp$county_fips))
cty <- cp %>% filter(!state_fips %in% c("02", "15", "11")) %>%
  left_join(sub, by = "county_fips") %>%
  mutate(area_id = case_when(state_fips == "36" & area_id %in% "downstate" ~ "downstate", state_fips == "36" ~ "upstate",
    state_fips == "41" & area_id %in% "nonurban" ~ "nonurban", state_fips == "41" ~ "standard", TRUE ~ "statewide"))
mwc <- mw %>% select(state_fips, area_id, snapshot_date, h2a_state_floor) %>%
  pivot_wider(names_from = snapshot_date, values_from = h2a_state_floor, names_prefix = "mw_")
names(mwc) <- gsub("-", "", names(mwc))
cty <- cty %>% left_join(mwc, by = c("state_fips", "area_id"))
stopifnot(!anyNA(cty$mw_20260101))

# Contract-day calendar of FY2025 rq13d certified rows (predetermined timing
# used for the contract-day-weighted minimum-wage sensitivity). Shift +1 year.
cr25 <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"),
  col_select = c("case_number", "fy_receipt", "in_rq13d", "state_fips", "county_fips", "begin", "end", "cert_pos", "cert_hours")) %>%
  filter(fy_receipt == 2025, in_rq13d, cert_hours > 0, !is.na(begin), !is.na(end), end >= begin)
seg <- tibble(seg = c("s20251001", "s20260101", "s20260701", "s20260930", "after"),
  lo = as.Date(c("2025-10-01", "2026-01-01", "2026-07-01", "2026-09-30", "2027-01-01")) - 365,
  hi = as.Date(c("2025-12-31", "2026-06-30", "2026-09-29", "2026-12-31", "2099-12-31")) - 365)
overlap <- function(b, e, lo, hi) pmax(0, as.numeric(pmin(e, hi) - pmax(b, lo)) + 1)
for (i in seq_len(nrow(seg))) cr25[[seg$seg[i]]] <- cr25$cert_pos * overlap(cr25$begin, cr25$end, seg$lo[i], seg$hi[i])
cal_nat <- cr25 %>% summarise(across(all_of(seg$seg), sum)) %>% mutate(tot = rowSums(across(everything())))
cal_nat <- cal_nat %>% mutate(across(all_of(seg$seg), ~ .x / tot)) %>% mutate(scope = "national", before_2025_10_01 = NA_real_)
pre_days <- sum(cr25$cert_pos * overlap(cr25$begin, cr25$end, as.Date("1900-01-01"), as.Date("2024-09-30")))
cal_nat$before_2025_10_01 <- pre_days / (pre_days + cal_nat$tot)
cal_st <- cr25 %>% group_by(state_fips) %>% summarise(across(all_of(seg$seg), sum), .groups = "drop") %>%
  mutate(tot = s20251001 + s20260101 + s20260701 + s20260930 + after)
write_csv(bind_rows(cal_nat, cal_st %>% mutate(across(all_of(seg$seg), ~ .x / tot), scope = "state")),
  file.path(out, "a1_contract_calendar.csv"))
message("FY2025 rq13d position-days shifted +1y (national): "); print(cal_nat, width = 200)
# Contract-day-weighted MW (days after 2026-12-31 carry the 2026-09-30 value)
cw <- cal_st %>% transmute(state_fips, w1 = s20251001 / tot, w2 = s20260101 / tot, w3 = s20260701 / tot, w4 = (s20260930 + after) / tot)
cty <- cty %>% left_join(cw, by = "state_fips") %>%
  mutate(across(c(w1, w2, w3, w4), ~ coalesce(.x, c(w1 = cal_nat$s20251001, w2 = cal_nat$s20260101, w3 = cal_nat$s20260701,
    w4 = cal_nat$s20260930 + cal_nat$after)[cur_column()])),
    mw_cd = w1 * mw_20251001 + w2 * mw_20260101 + w3 * mw_20260701 + w4 * mw_20260930,
    mw_blend = mw_20251001 / 3 + 2 * mw_20260101 / 3)

cty <- cty %>% left_join(a25, by = "state_fips") %>% left_join(a24, by = "state_fips") %>% left_join(a25_lo, by = "state_fips") %>%
  left_join(sk, by = "state_fips")
stopifnot(!anyNA(cty$a25), !anyNA(cty$H2A_s1), !anyNA(cty$US_s1))
# FY2025 floor: the minimum wage never exceeded the FY2025 AEWR (upper-bound check)
chk <- cty %>% filter(mw_20250929 >= a25_min_in_window)
if (nrow(chk)) stop("Minimum wage reaches the FY2025 AEWR in: ", paste(unique(chk$state_fips), collapse = ","))
cty <- cty %>% mutate(e25 = pmax(a25, mw_20250929),
  FH = pmax(H2A_s1, mw_20260101), FU = pmax(US_s1, mw_20260101),
  FH_blend = pmax(H2A_s1, mw_blend), FH_cd = pmax(H2A_s1, mw_cd),
  stat_cut_h = H2A_s1 - a25, stat_cut_u = US_s1 - a25,
  eff_cut_h = FH - e25, eff_cut_u = FU - e25, eff_cut_h_blend = FH_blend - e25, eff_cut_h_cd = FH_cd - e25,
  trunc = pmax(0, mw_20260101 - H2A_s1), mw_binds_h2a = mw_20260101 > H2A_s1, mw_binds_us = mw_20260101 > US_s1,
  housing_adj = US_s1 - H2A_s1, multi_region = aewr_region_id %in% (cp %>% group_by(aewr_region_id) %>%
    filter(n_distinct(state_fips) > 1) %>% pull(aewr_region_id) %>% unique()))
# Residualize truncation on the statutory cut within former region (state level)
st <- cty %>% distinct(state_fips, area_id, aewr_region_id, a24, a25, e25, H2A_s1, H2A_s2, US_s1, US_s2, housing_adj,
  mw_20250929, mw_20251001, mw_20260101, mw_20260701, mw_20260930, mw_blend, mw_cd, FH, FU, FH_blend, FH_cd,
  stat_cut_h, stat_cut_u, eff_cut_h, eff_cut_u, eff_cut_h_blend, eff_cut_h_cd, trunc, mw_binds_h2a, mw_binds_us, multi_region)
write_csv(st, file.path(out, "a1_state_doses.csv"))
st1 <- st %>% filter(area_id %in% c("statewide", "upstate", "standard"))
message("States: ", nrow(st1), "; MW > H-2A skill I: ", sum(st1$mw_binds_h2a), " (", paste(sort(st1$state_fips[st1$mw_binds_h2a]), collapse = ","),
  "); MW > U.S. skill I: ", sum(st1$mw_binds_us), " (", paste(sort(st1$state_fips[st1$mw_binds_us]), collapse = ","), ")")
tr_fit <- lm(trunc ~ stat_cut_h + factor(aewr_region_id), data = st1)
cty$trunc_resid <- cty$trunc - predict(tr_fit, newdata = cty)

# ---- Binding share b (FY2025, predetermined) -----------------------------------
oc <- read_parquet(file.path(S, "shared", "outcomes_county_fy.parquet"),
  col_select = c("county_fips", "year", "cert_hours_rq13d", "cert_hours_dq13", "req_pos_rq13d",
    "at_aewr_hours_rq13d", "hourly_aewr_known_hours_rq13d", "at_aewr_hours_dq13", "hourly_aewr_known_hours_dq13")) %>%
  filter(year <= 2025)  # no FY2026 outcome is read
bshare <- function(years, at, kn) {
  s <- oc %>% filter(year %in% years) %>% group_by(county_fips) %>%
    summarise(at = sum(.data[[at]], na.rm = TRUE), kn = sum(.data[[kn]], na.rm = TRUE), .groups = "drop")
  cty %>% select(county_fips, state_fips) %>% left_join(s, by = "county_fips") %>% group_by(state_fips) %>%
    mutate(st_share = sum(at, na.rm = TRUE) / sum(kn, na.rm = TRUE)) %>% ungroup() %>%
    transmute(county_fips, b = pmin(pmax(ifelse(is.finite(kn) & kn > 0, at / kn, st_share), 0), 1), b_known = is.finite(kn) & kn > 0, b_kn = coalesce(kn, 0))
}
b25 <- bshare(2025, "at_aewr_hours_rq13d", "hourly_aewr_known_hours_rq13d")
cty <- cty %>% left_join(b25 %>% rename(b = b, b_known = b_known, b_kn = b_kn), by = "county_fips") %>%
  left_join(bshare(2025, "at_aewr_hours_dq13", "hourly_aewr_known_hours_dq13") %>% select(county_fips, b_dq13 = b), by = "county_fips") %>%
  left_join(bshare(2023:2024, "at_aewr_hours_rq13d", "hourly_aewr_known_hours_rq13d") %>% select(county_fips, b_2324 = b), by = "county_fips") %>%
  left_join(bshare(2024, "at_aewr_hours_rq13d", "hourly_aewr_known_hours_rq13d") %>% select(county_fips, b_2024 = b), by = "county_fips")
# State x binding-half node (hours-weighted within-state median of b; stress DGP)
wmed <- function(x, w) { o <- order(x); x <- x[o]; w <- w[o]; if (sum(w) <= 0) return(median(x)); x[which(cumsum(w) >= sum(w) / 2)[1]] }
cty <- cty %>% group_by(state_fips) %>% mutate(b_med = wmed(b, b_kn), b_half = ifelse(b >= b_med, "hi", "lo")) %>% ungroup() %>%
  mutate(state_bhalf = paste(state_fips, b_half))
bs <- cty %>% summarise(counties = n(), counties_b_known = sum(b_known), hw_mean_b = sum(b_kn * b) / sum(b_kn),
  hw_sd_b = sqrt(sum(b_kn * (b - sum(b_kn * b) / sum(b_kn))^2) / sum(b_kn)),
  hw_within_state_sd_b = sqrt(sum(b_kn * (b - ave(b * b_kn, state_fips, FUN = sum) / pmax(ave(b_kn, state_fips, FUN = sum), 1e-9))^2) / sum(b_kn)),
  share_hours_b_below_0.8 = sum(b_kn[b < 0.8]) / sum(b_kn), share_hours_b_eq_1 = sum(b_kn[b >= 0.9999]) / sum(b_kn),
  cor_b_b2324 = cor(b[b_known], b_2324[b_known]), cor_b_bdq13 = cor(b[b_known], b_dq13[b_known]))
write_csv(bs, file.path(out, "a1_binding_share_summary.csv")); print(bs, width = 200)

doses <- cty %>% mutate(
  d1_h = b * eff_cut_h, d1_u = b * eff_cut_u, d1_h_dq13 = b_dq13 * eff_cut_h, d1_h_2324 = b_2324 * eff_cut_h,
  d1_h_blend = b * eff_cut_h_blend, d1_h_cd = b * eff_cut_h_cd, d1_h_pl2024 = b_2024 * eff_cut_h)
write_parquet(doses, file.path(out, "a1_county_doses.parquet"))

# ---- Frames ------------------------------------------------------------------
pre_c <- tidyr::crossing(doses %>% select(county_fips, state_fips, aewr_region_id, state_bhalf, multi_region), year = CAL) %>%
  left_join(oc %>% select(county_fips, year, cert_hours_rq13d, req_pos_rq13d), by = c("county_fips", "year")) %>%
  mutate(cert_hours_rq13d = coalesce(cert_hours_rq13d, 0), req_pos_rq13d = coalesce(req_pos_rq13d, 0),
    region_year = paste(aewr_region_id, year), state_year = paste(state_fips, year), sbh_year = paste(state_bhalf, year))
mk_pol <- function(x25, x26) {
  tidyr::crossing(doses, year = 2025:2026) %>%
    mutate(x = ifelse(year == 2025, .data[[x25]], .data[[x26]]),
      region_year = paste(aewr_region_id, year), state_year = paste(state_fips, year), sbh_year = paste(state_bhalf, year))
}
doses <- doses %>% mutate(zero = 0, lvl_stat_h = H2A_s1, lvl_FH = FH, lvl_d1 = e25 + d1_h, lvl_d1_u = e25 + d1_u,
  lvl_tr = trunc_resid)

support_one <- function(tag, policy, pre, ycount, unit, fe, clusters, jk, nodes = c("state_fips")) {
  lin <- ish_ppml_linearize(pre, policy, ycount, unit)
  pol <- lin$policy[is.finite(lin$policy$x), , drop = FALSE]; pr <- lin$pre[lin$pre[[unit]] %in% pol[[unit]], , drop = FALSE]
  pp <- ish_prep(pol, unit, "year", fe, "state_fips", NULL, lin$weights, x = "x", clusters = clusters, jk_cluster = jk)
  lev <- rowsum(pp$xw^2, match(pp$data[[jk]], pp$jk))[, 1]
  prr <- ish_prep(pr, unit, "year", fe, "state_fips", NULL, lin$weights)
  res <- drop(ish_within(cbind(prr$data[[lin$y]]), prr)) / prr$sw
  ns <- bind_rows(lapply(nodes, function(nd) ish_node_period_share(pr, lin$y, unit, "year", fe, nd, weights = lin$weights) %>% mutate(node = nd)))
  w <- pol$ppml_weight
  sup <- tibble(candidate = tag, outcome = ycount, units = length(pp$units), states = length(pp$jk),
    clusters_with_dose = sum(lev > 1e-12 * pp$sxx), dose_sd_after_fe = sqrt(pp$sxx / sum(pp$w)),
    effective_clusters = pp$sxx^2 / sum(lev^2), max_leverage_share = max(lev) / pp$sxx,
    max_leverage_cluster = pp$jk[as.integer(names(lev))[which.max(lev)]],
    second_leverage_share = sort(lev, decreasing = TRUE)[2] / pp$sxx,
    preperiod_residual_sd = sqrt(sum(prr$w * res^2) / sum(prr$w)),
    weighted_mean_dose_level = sum(w * pol$x) / sum(w), fe = paste(fe, collapse = "+"), calib_years = paste(range(pr$year), collapse = "-"))
  list(support = sup, node = ns %>% mutate(candidate = tag, outcome = ycount, fe = paste(fe, collapse = "+")))
}

cands <- list(
  C1_statutory_h2a_region = list(x25 = "a25", x26 = "lvl_stat_h", fe = "region_year", sample = "multi"),
  C2_effective_floor_region = list(x25 = "e25", x26 = "lvl_FH", fe = "region_year", sample = "multi"),
  C3_d1_cross_state_region = list(x25 = "e25", x26 = "lvl_d1", fe = "region_year", sample = "multi"),
  C4_d1_within_state_48 = list(x25 = "e25", x26 = "lvl_d1", fe = "state_year", sample = "all"),
  C4b_d1_within_state_46 = list(x25 = "e25", x26 = "lvl_d1", fe = "state_year", sample = "multi"),
  C4u_d1_within_state_usrate_48 = list(x25 = "e25", x26 = "lvl_d1_u", fe = "state_year", sample = "all"),
  C6_truncation_resid_region = list(x25 = "zero", x26 = "lvl_tr", fe = "region_year", sample = "multi"))
sup <- list(); nsh <- list()
for (nm in names(cands)) for (yv in c("cert_hours_rq13d", "req_pos_rq13d")) {
  cc <- cands[[nm]]
  pol <- mk_pol(cc$x25, cc$x26); pre <- pre_c
  if (cc$sample == "multi") { pol <- pol %>% filter(multi_region); pre <- pre %>% filter(multi_region) }
  nodes <- if (cc$fe == "state_year") c("state_fips", "state_bhalf") else "state_fips"
  r <- support_one(nm, pol, pre, yv, "county_fips", c("county_fips", cc$fe), c("state_fips", "aewr_region_id"), "state_fips", nodes)
  sup[[length(sup) + 1]] <- r$support; nsh[[length(nsh) + 1]] <- r$node
  print(r$support, width = 250); print(r$node)
}

# ---- Employer x state level (within-state D1) -----------------------------------
cre <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"),
  col_select = c("fy_receipt", "in_rq13d", "state_fips", "employer_id_balanced", "cert_hours", "hourly_wage", "aewr_known", "at_aewr")) %>%
  filter(in_rq13d, fy_receipt %in% CAL, !is.na(employer_id_balanced), !is.na(state_fips), state_fips %in% doses$state_fips)
emp <- cre %>% mutate(hourly = is.finite(hourly_wage) & hourly_wage > 0 & hourly_wage <= 100) %>%
  group_by(employer_id_balanced, state_fips, year = fy_receipt) %>%
  summarise(cert_hours_rq13d = sum(cert_hours, na.rm = TRUE), kn = sum(cert_hours[hourly & aewr_known %in% TRUE], na.rm = TRUE),
    at = sum(cert_hours[hourly & aewr_known %in% TRUE & at_aewr %in% TRUE], na.rm = TRUE), .groups = "drop") %>%
  mutate(unit = paste(employer_id_balanced, state_fips))
inc <- emp %>% filter(year == 2025, cert_hours_rq13d > 0)
st_b <- inc %>% group_by(state_fips) %>% summarise(st_share = sum(at) / sum(kn), .groups = "drop")
eb <- inc %>% left_join(st_b, by = "state_fips") %>% transmute(unit, employer_id_balanced, state_fips, b = ifelse(kn > 0, at / kn, st_share))
stv <- st %>% filter(area_id %in% c("statewide", "upstate", "standard")) %>% select(state_fips, e25, eff_cut_h, aewr_region_id)
# NY/OR employer units: state-level (upstate / standard) floor; sub-state MW areas not assigned to employers.
eb <- eb %>% left_join(stv, by = "state_fips") %>% mutate(lvl_d1 = e25 + b * eff_cut_h)
write_parquet(eb, file.path(out, "a1_employer_doses.parquet"))
pre_e <- tidyr::crossing(eb %>% select(unit, state_fips, aewr_region_id), year = CAL) %>%
  left_join(emp %>% select(unit, year, cert_hours_rq13d), by = c("unit", "year")) %>%
  mutate(cert_hours_rq13d = coalesce(cert_hours_rq13d, 0), state_year = paste(state_fips, year))
pol_e <- tidyr::crossing(eb, year = 2025:2026) %>% mutate(x = ifelse(year == 2025, e25, lvl_d1), state_year = paste(state_fips, year))
r <- support_one("C5_d1_within_state_employer", pol_e, pre_e, "cert_hours_rq13d", "unit", c("unit", "state_year"), c("state_fips", "aewr_region_id"), "state_fips")
sup[[length(sup) + 1]] <- r$support; nsh[[length(nsh) + 1]] <- r$node; print(r$support, width = 250); print(r$node)

write_csv(bind_rows(sup), file.path(out, "a1_support.csv"))
write_csv(bind_rows(nsh) %>% mutate(ratio = observed_node_period_share / pure_unit_noise_share,
  gate_dgp_by_rule = ifelse(observed_node_period_share <= 1.5 * pure_unit_noise_share, "county_serial", "max_shared")),
  file.path(out, "a1_node_share.csv"))
message("done ", format(Sys.time()))
