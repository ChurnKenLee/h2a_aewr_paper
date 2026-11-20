# Identification search, strategy N1 (Phase 3 design, step 1b): predetermined
# skill-tier dose, units, support statistics, node-period share for the gate
# rule, and pre-period pass-through mechanics.
#
# Usage (repository root, after 01_experience_extract.R):
#   Rscript --vanilla explorations/identification_search/N1/02_dose_support.R "$S"
# Reads $S/N1/n1_experience_extract.parquet, $S/shared/outcomes_case_rows.parquet,
# data/intermediate/aewr_occupation_skill_rates.parquet, aewr_rate_schedule.parquet,
# data/processed/county_year_panel.parquet (state -> former AEWR region) and the
# verified minimum-wage table explorations/identification_search/A1/sources/.
# Writes $S/N1/n1_state_rates.csv, n1_order_floors_fy{2024,2025}.parquet,
# n1_units_{employer,county}.parquet, n1_calibration_{employer,county}.parquet,
# n1_support.csv, n1_node_share.csv, n1_passthrough_mechanics.csv,
# n1_zero_dose_states.csv, n1_dose_summary.csv.
#
# Integrity: no outcome is regressed on any dose. Outcomes enter only as
# (i) FY2020-FY2025 calibration counts (FE-only residual variance, PPML
# weights) and (ii) FY2024-FY2025 certified hours used as composition weights
# for the predetermined tier share. No FY2026 row of the case-row file or of
# the experience extract is read.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(tidyr); library(readr)})
source("explorations/identification_search/shared/power_harness.R")
setFixest_nthreads(1)
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "N1"); dir.create(out, showWarnings = FALSE)
if (startsWith(paste0(S, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
CAL <- 2020:2025
FIELD5 <- c("45-2041", "45-2091", "45-2092", "45-2093", "53-7064")  # 655.120(b)(1)(i) field and livestock (combined)

# ---- State rates (2025-10-02 schedule) and FY2025 reference AEWR ---------------
sk_all <- read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
  filter(as.Date(effective_start) == as.Date("2025-10-02")) %>% mutate(sheet = trimws(source_sheet))
sk <- sk_all %>% filter(sheet == "Statewide 655.120(b)(1)(i)") %>%
  transmute(state_fips, k = paste0(worker_category, "_s", skill_level), r = as.numeric(hourly_aewr)) %>%
  pivot_wider(names_from = k, values_from = r) %>%
  mutate(adj = US_s1 - H2A_s1, adj_s2 = US_s2 - H2A_s2, g_h2a = H2A_s2 - H2A_s1, g_us = US_s2 - US_s1)
stopifnot(max(abs(sk$adj - sk$adj_s2)) < 1e-9)  # 655.120(b)(3): one statewide adjustment for both skill levels
soc_rates <- sk_all %>% filter(sheet == "Statewide 655.120(b)(1)(ii)") %>%
  transmute(state_fips, soc = soc_code, k = paste0(worker_category, "_s", skill_level), r = as.numeric(hourly_aewr)) %>%
  distinct() %>% group_by(state_fips, soc, k) %>% summarise(r = max(r), .groups = "drop") %>%
  pivot_wider(names_from = k, values_from = r, names_prefix = "soc_")
regional <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>%
  filter(occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference"), worker_category == "covered_workers") %>%
  transmute(state_fips, aewr = as.numeric(aewr), start = as.Date(effective_start), end = as.Date(effective_end_exclusive))
rate_on <- function(date) { r <- regional %>% filter(start <= date, is.na(end) | end > date) %>% select(state_fips, aewr)
  stopifnot(!anyDuplicated(r$state_fips)); r }
a25 <- rate_on(as.Date("2025-04-01")) %>% rename(a25 = aewr)
a24 <- rate_on(as.Date("2024-04-01")) %>% rename(a24 = aewr)

mw <- read_csv("explorations/identification_search/A1/sources/state_ag_minwage_fy2025_2026.csv",
  col_types = cols(state_fips = "c", .default = col_guess()))
sub <- read_csv("explorations/identification_search/A1/sources/minwage_substate_counties.csv", col_types = cols(.default = "c"))
mwc <- mw %>% select(state_fips, area_id, snapshot_date, h2a_state_floor) %>%
  mutate(snapshot_date = paste0("mw_", gsub("-", "", as.character(snapshot_date)))) %>%
  pivot_wider(names_from = snapshot_date, values_from = h2a_state_floor)
reg <- read_parquet(path_processed("county_year_panel.parquet"), col_select = c("state_fips", "aewr_region_id")) %>%
  distinct() %>% filter(!is.na(aewr_region_id)) %>% group_by(state_fips) %>% summarise(aewr_region_id = as.character(first(aewr_region_id)), .groups = "drop")

st <- sk %>% left_join(a25, by = "state_fips") %>% left_join(a24, by = "state_fips") %>% left_join(reg, by = "state_fips") %>%
  mutate(aewr_region_id = coalesce(aewr_region_id, paste0("st", state_fips)))
area_of <- function(county_fips, state_fips) case_when(
  state_fips == "36" & county_fips %in% sub$county_fips[sub$area_id == "downstate"] ~ "downstate", state_fips == "36" ~ "upstate",
  state_fips == "41" & county_fips %in% sub$county_fips[sub$area_id == "nonurban"] ~ "nonurban", state_fips == "41" ~ "standard",
  TRUE ~ "statewide")

# ---- Rows -> orders -> floors ------------------------------------------------------
ex <- read_parquet(file.path(out, "n1_experience_extract.parquet")) %>% filter(fiscal_year <= 2025) %>%
  mutate(soc = substr(gsub("[^0-9-]", "", soc_raw), 1, 7))
cr <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"),
  col_select = c("case_number", "fiscal_year", "fy_receipt", "in_rq13d", "state_fips", "county_fips", "employer_id_balanced",
    "cert_hours", "cert_pos", "req_pos", "begin", "end", "hourly_wage", "aewr_known", "at_aewr")) %>%
  filter(fy_receipt %in% CAL, in_rq13d)
stopifnot(is.character(cr$county_fips), is.character(cr$state_fips))

order_floors <- function(base) {
  r <- cr %>% filter(fy_receipt == base, cert_hours > 0) %>%
    left_join(ex %>% select(case_number, fiscal_year, exp_months, tier2, soc, piece_rate, flc, naics), by = c("case_number", "fiscal_year")) %>%
    mutate(area_id = area_of(county_fips, state_fips), field = is.na(soc) | soc %in% FIELD5 | !nzchar(soc)) %>%
    left_join(st %>% select(state_fips, H2A_s1, H2A_s2, US_s1, US_s2, adj, a25, a24), by = "state_fips") %>%
    left_join(soc_rates, by = c("state_fips", "soc")) %>%
    left_join(mwc, by = c("state_fips", "area_id")) %>%
    mutate(nonfield_rate = !field & !is.na(soc_H2A_s1) & !is.na(soc_H2A_s2),
      r1 = ifelse(nonfield_rate, soc_H2A_s1, H2A_s1), r2 = ifelse(nonfield_rate, soc_H2A_s2, H2A_s2),
      u1 = ifelse(nonfield_rate, soc_US_s1, US_s1), u2 = ifelse(nonfield_rate, soc_US_s2, US_s2),
      F1 = pmax(r1, mw_20260101), F2 = pmax(r2, mw_20260101), FU1 = pmax(u1, mw_20260101), FU2 = pmax(u2, mw_20260101),
      e_base = pmax(if (base == 2025) a25 else a24, if (base == 2025) mw_20250929 else 0))
  # Order (case) floors across all itinerary states/areas (F1 review lesson): max over the order's rows.
  o <- r %>% group_by(case_number, fiscal_year) %>% summarise(F1_o = max(F1), F2_o = max(F2), FU1_o = max(FU1), FU2_o = max(FU2),
    e_o = max(e_base), n_states_o = n_distinct(state_fips), .groups = "drop")
  r %>% left_join(o, by = c("case_number", "fiscal_year")) %>%
    mutate(gap_o = F2_o - F1_o, gapU_o = FU2_o - FU1_o, lvl_o = F1_o - e_o, base_year = base)
}
rows <- bind_rows(order_floors(2025), order_floors(2024))
miss_rates <- rows %>% filter(is.na(F1) | is.na(F2)) %>% count(base_year, state_fips, wt = cert_hours)
message("hours with no rate (dropped): "); print(miss_rates)
rows <- rows %>% filter(!is.na(F1_o), !is.na(F2_o), !is.na(e_o))
write_parquet(rows %>% select(base_year, case_number, fiscal_year, state_fips, county_fips, employer_id_balanced, cert_hours,
  exp_months, tier2, soc, field, nonfield_rate, F1_o, F2_o, FU1_o, FU2_o, e_o, gap_o, gapU_o, lvl_o, n_states_o),
  file.path(out, "n1_order_floors.parquet"))

# ---- Unit doses --------------------------------------------------------------------
unit_dose <- function(r, keys) r %>% group_by(across(all_of(keys))) %>% summarise(
  h_base = sum(cert_hours), h_exp_known = sum(cert_hours[!is.na(tier2)]),
  s = sum(cert_hours[tier2 %in% TRUE]) / h_exp_known,
  s_exact3 = sum(cert_hours[exp_months %in% 3]) / h_exp_known,
  s_1_2 = sum(cert_hours[exp_months %in% c(1, 2)]) / h_exp_known,
  s_0 = sum(cert_hours[exp_months %in% 0]) / h_exp_known,
  near_threshold_only = all(exp_months[!is.na(exp_months)] %in% 1:3),
  T = sum((cert_hours * tier2 * gap_o)[!is.na(tier2)]) / h_exp_known,
  TU = sum((cert_hours * tier2 * gapU_o)[!is.na(tier2)]) / h_exp_known,
  G = sum((cert_hours * gap_o)[!is.na(tier2)]) / h_exp_known,
  L = sum((cert_hours * lvl_o)[!is.na(tier2)]) / h_exp_known,
  F1u = sum((cert_hours * F1_o)[!is.na(tier2)]) / h_exp_known, e_u = sum((cert_hours * e_o)[!is.na(tier2)]) / h_exp_known,
  nonfield_share = sum(cert_hours[!field]) / h_base, multistate_share = sum(cert_hours[n_states_o > 1]) / h_base,
  piece_share = sum(cert_hours[piece_rate > 0 & !is.na(piece_rate)]) / h_base,
  flc_share = sum(cert_hours[flc %in% "Y"]) / h_base,
  dom_soc = { hs <- tapply(cert_hours, coalesce(soc, "na"), sum); names(hs)[which.max(hs)] },
  naics3 = { hn <- tapply(cert_hours, coalesce(substr(naics, 1, 3), "na"), sum); names(hn)[which.max(hn)] },
  hourly_hours = sum(cert_hours[aewr_known %in% TRUE]), at_aewr_hours = sum(cert_hours[at_aewr %in% TRUE & aewr_known %in% TRUE]),
  .groups = "drop") %>%
  filter(h_exp_known > 0) %>% mutate(b = ifelse(hourly_hours > 0, at_aewr_hours / hourly_hours, NA_real_))

# Employer x state units (primary). Hours without an employer id are excluded (report share).
noemp <- rows %>% group_by(base_year) %>% summarise(share_no_employer = sum(cert_hours[is.na(employer_id_balanced)]) / sum(cert_hours))
print(noemp)
ue <- rows %>% filter(!is.na(employer_id_balanced)) %>% mutate(unit = paste(employer_id_balanced, state_fips, sep = "|")) %>%
  unit_dose(c("base_year", "unit", "employer_id_balanced", "state_fips")) %>%
  left_join(st %>% select(state_fips, aewr_region_id, g_h2a, g_us), by = "state_fips")
uc <- rows %>% unit_dose(c("base_year", "county_fips", "state_fips")) %>%
  left_join(st %>% select(state_fips, aewr_region_id, g_h2a, g_us), by = "state_fips")
write_parquet(ue, file.path(out, "n1_units_employer.parquet")); write_parquet(uc, file.path(out, "n1_units_county.parquet"))

# Zero-dose states: H-2A skill-II rate at or below the applicable minimum wage, so both tiers are floored by it.
zd <- rows %>% filter(base_year == 2025) %>% group_by(state_fips) %>%
  summarise(hours_fy2025 = sum(cert_hours), mean_gap_o = weighted.mean(gap_o, cert_hours), share_gap_zero = sum(cert_hours[gap_o < 0.005]) / sum(cert_hours),
    .groups = "drop") %>% left_join(st %>% select(state_fips, H2A_s1, H2A_s2, g_h2a), by = "state_fips") %>%
  left_join(mwc %>% filter(area_id %in% c("statewide", "upstate", "standard")) %>% select(state_fips, mw_20260101), by = "state_fips") %>%
  mutate(zero_dose_state = mw_20260101 >= H2A_s2, partial_truncation = mw_20260101 > H2A_s1 & mw_20260101 < H2A_s2) %>% arrange(desc(hours_fy2025))
write_csv(zd, file.path(out, "n1_zero_dose_states.csv"))
write_csv(st %>% left_join(mwc %>% filter(area_id %in% c("statewide", "upstate", "standard")) %>% select(state_fips, mw_20250929, mw_20260101), by = "state_fips"),
  file.path(out, "n1_state_rates.csv"))

# ---- Calibration panels (FY2020-FY2025, rq13d by receipt FY) ----------------------
cal_e <- cr %>% filter(!is.na(employer_id_balanced)) %>% mutate(unit = paste(employer_id_balanced, state_fips, sep = "|")) %>%
  group_by(unit, year = fy_receipt) %>% summarise(cert_hours_rq13d = sum(cert_hours, na.rm = TRUE), req_pos_rq13d = sum(req_pos, na.rm = TRUE), .groups = "drop")
cal_c <- cr %>% group_by(county_fips, year = fy_receipt) %>%
  summarise(cert_hours_rq13d = sum(cert_hours, na.rm = TRUE), req_pos_rq13d = sum(req_pos, na.rm = TRUE), .groups = "drop")
write_parquet(cal_e, file.path(out, "n1_calibration_employer.parquet")); write_parquet(cal_c, file.path(out, "n1_calibration_county.parquet"))

# ---- Design frames ------------------------------------------------------------------
# Identifying dose: tier component T_i (= sum over FY-base orders of h x 1{exp >= 3} x (F2_o - F1_o) / h),
# residualized (PPML-weighted, within state) on the level component L_i = mean(F1_o - e_o), which is
# nonzero within state only through multi-state itinerary orders and sub-state minimum-wage areas.
frames <- function(u, unitcol, base, ycol = "cert_hours_rq13d", cal_years = CAL, dose = "T", resid = TRUE, cal = NULL) {
  u <- u %>% filter(base_year == base)
  pre <- tidyr::crossing(u %>% select(all_of(c(unitcol, "state_fips", "aewr_region_id"))), year = cal_years) %>%
    left_join(cal %>% select(all_of(c(unitcol, "year", ycol))), by = c(unitcol, "year")) %>%
    mutate(across(all_of(ycol), ~ coalesce(.x, 0)), state_year = paste(state_fips, year), region_year = paste(aewr_region_id, year))
  lin <- ish_ppml_linearize(as.data.frame(pre), as.data.frame(u), ycol, unitcol)
  uu <- lin$policy
  if (resid) {
    f <- fixest::feols(as.formula(paste(dose, "~ L | state_fips")), data = uu, weights = ~ppml_weight)
    uu$x_id <- uu[[dose]] - coef(f)[["L"]] * uu$L  # remove the within-state L slope; state means are absorbed by state x FY FE
  } else uu$x_id <- uu[[dose]]
  pol <- bind_rows(uu %>% mutate(year = base), uu %>% mutate(year = base + 1L)) %>%
    mutate(x = ifelse(year == base, 0, x_id), state_year = paste(state_fips, year), region_year = paste(aewr_region_id, year),
      soc_year = paste(dom_soc, year), naics_year = paste(naics3, year))
  list(pre = lin$pre, policy = pol, y = lin$y, w = lin$weights, units = uu)
}

support_one <- function(tag, fr, unitcol, fe = c(unitcol, "state_year")) {
  pp <- ish_prep(as.data.frame(fr$policy), unitcol, "year", fe, "state_fips", "aewr_region_id", fr$w, x = "x",
    clusters = c("state_fips", "aewr_region_id"), jk_cluster = "state_fips")
  lev <- rowsum(pp$xw^2, match(pp$data$state_fips, pp$jk))[, 1]
  pre_pp <- ish_prep(as.data.frame(fr$pre), unitcol, "year", fe, "state_fips", NULL, fr$w)
  res <- drop(ish_within(cbind(pre_pp$data[[fr$y]]), pre_pp))
  ns <- ish_node_period_share(as.data.frame(fr$pre), fr$y, unitcol, "year", fe, "state_fips", fr$w)
  wu <- fr$units$ppml_weight
  tibble(design = tag, units = length(pp$units), states = length(pp$jk), states_with_dose = sum(lev > 1e-12 * pp$sxx),
    dose_sd_after_fe = sqrt(pp$sxx / sum(pp$w)), effective_clusters = pp$sxx^2 / sum(lev^2),
    max_leverage_share = max(lev) / pp$sxx, max_leverage_state = pp$jk[which.max(lev)],
    preperiod_residual_sd = sqrt(sum(pre_pp$w * res^2) / sum(pre_pp$w)),
    observed_node_period_share = ns$observed_node_period_share, pure_unit_noise_share = ns$pure_unit_noise_share,
    node_share_ratio = ns$observed_node_period_share / ns$pure_unit_noise_share,
    gate_dgp = ifelse(ns$observed_node_period_share <= 1.5 * ns$pure_unit_noise_share, "county_serial", "max_shared"),
    mean_s_w = weighted.mean(fr$units$s, wu), sd_s_w = sqrt(weighted.mean((fr$units$s - weighted.mean(fr$units$s, wu))^2, wu)),
    mean_T_w = weighted.mean(fr$units$T, wu), mean_x_w = weighted.mean(fr$units$x_id, wu),
    wbar_h2a_floor = weighted.mean((fr$units$e_u + fr$units$F1u + fr$units$T) / 2, wu))
}
designs <- list(
  E_primary = list(u = ue, unit = "unit", base = 2025, dose = "T", resid = TRUE, cal = cal_e),
  E_no_level_control = list(u = ue, unit = "unit", base = 2025, dose = "T", resid = FALSE, cal = cal_e),
  E_us_rate_dose = list(u = ue, unit = "unit", base = 2025, dose = "TU", resid = TRUE, cal = cal_e),
  E_placebo_fy2024 = list(u = ue, unit = "unit", base = 2024, dose = "T", resid = TRUE, cal = cal_e, cal_years = 2020:2024),
  E_near_threshold = list(u = ue %>% filter(near_threshold_only), unit = "unit", base = 2025, dose = "T", resid = TRUE, cal = cal_e),
  C_county = list(u = uc, unit = "county_fips", base = 2025, dose = "T", resid = TRUE, cal = cal_c))
sup <- list()
for (nm in names(designs)) {
  d <- designs[[nm]]
  fr <- frames(d$u, d$unit, d$base, dose = d$dose, resid = d$resid, cal = d$cal, cal_years = if (is.null(d$cal_years)) CAL else d$cal_years)
  sup[[nm]] <- support_one(nm, fr, d$unit); print(sup[[nm]], width = 300)
  if (nm == "E_primary") {
    # Support after the declared sensitivity S2 fixed effects (dominant FY2025 SOC x FY, NAICS-3 x FY): the A3 lesson
    # (occupation-mix controls absorbed most of A3's dose). Policy-frame dose SD only; calibration unchanged.
    for (ff in list(c("unit", "state_year", "soc_year"), c("unit", "state_year", "naics_year"), c("unit", "state_year", "soc_year", "naics_year"), c("unit", "region_year"))) {
      pp <- ish_prep(as.data.frame(fr$policy), "unit", "year", ff, "state_fips", NULL, fr$w, x = "x", clusters = "state_fips", jk_cluster = "state_fips")
      lev <- rowsum(pp$xw^2, match(pp$data$state_fips, pp$jk))[, 1]
      sup[[paste0("E_primary_fe_", paste(ff[-1], collapse = "+"))]] <- tibble(design = paste0("E_primary_fe_", paste(ff[-1], collapse = "+")),
        units = length(pp$units), states = length(pp$jk), states_with_dose = sum(lev > 1e-12 * pp$sxx),
        dose_sd_after_fe = sqrt(pp$sxx / sum(pp$w)), effective_clusters = pp$sxx^2 / sum(lev^2), max_leverage_share = max(lev) / pp$sxx,
        max_leverage_state = pp$jk[which.max(lev)])
      print(sup[[length(sup)]], width = 300)
    }
  }
}
# Requested positions (lead secondary): node share on its own calibration outcome.
fr_req <- frames(ue, "unit", 2025, ycol = "req_pos_rq13d", cal = cal_e)
sup[["E_primary_req_pos"]] <- support_one("E_primary_req_pos", fr_req, "unit")
supp <- bind_rows(sup)
write_csv(supp, file.path(out, "n1_support.csv"))

# ---- Dose summary (treatment side) ----------------------------------------------------
u25 <- ue %>% filter(base_year == 2025)
dsum <- tibble(stat = c("units_fy2025", "employers", "states", "hours_fy2025_rq13d", "hours_share_exp_known",
  "hours_share_tier2", "hours_share_exact3", "hours_share_1_2", "hours_share_0", "units_s_0", "units_s_1", "units_s_interior",
  "hours_share_units_s_interior", "hours_share_multistate_orders", "hours_share_nonfield", "hours_share_nonfield_with_soc_rate",
  "mean_gap_h2a_states", "sd_gap_h2a_states", "min_gap_h2a_states", "max_gap_h2a_states", "hours_weighted_gap_o",
  "near_threshold_units", "near_threshold_hours_share"),
  value = c(nrow(u25), n_distinct(u25$employer_id_balanced), n_distinct(u25$state_fips), sum(u25$h_base), sum(u25$h_exp_known) / sum(u25$h_base),
    with(rows %>% filter(base_year == 2025, !is.na(tier2)), sum(cert_hours[tier2]) / sum(cert_hours)),
    with(rows %>% filter(base_year == 2025, !is.na(tier2)), sum(cert_hours[exp_months == 3]) / sum(cert_hours)),
    with(rows %>% filter(base_year == 2025, !is.na(tier2)), sum(cert_hours[exp_months %in% 1:2]) / sum(cert_hours)),
    with(rows %>% filter(base_year == 2025, !is.na(tier2)), sum(cert_hours[exp_months == 0]) / sum(cert_hours)),
    sum(u25$s == 0), sum(u25$s == 1), sum(u25$s > 0 & u25$s < 1), sum(u25$h_base[u25$s > 0 & u25$s < 1]) / sum(u25$h_base),
    with(rows %>% filter(base_year == 2025), sum(cert_hours[n_states_o > 1]) / sum(cert_hours)),
    with(rows %>% filter(base_year == 2025), sum(cert_hours[!field]) / sum(cert_hours)),
    with(rows %>% filter(base_year == 2025), sum(cert_hours[nonfield_rate]) / sum(cert_hours)),
    mean(st$g_h2a), sd(st$g_h2a), min(st$g_h2a), max(st$g_h2a),
    with(rows %>% filter(base_year == 2025), weighted.mean(gap_o, cert_hours)),
    sum(u25$near_threshold_only), sum(u25$h_base[u25$near_threshold_only]) / sum(u25$h_base)))
write_csv(dsum, file.path(out, "n1_dose_summary.csv")); print(as.data.frame(dsum))

# ---- Pass-through mechanics from pre-period data and the rule's published projection ----
# (1) Persistence: within-state slope of the unit's FY2025 tier share on its FY2024 share (units in both years,
#     weights = FY2024 hours). With no price incentive, this bounds the FY2026 assignment pass-through from above
#     (no downgrading, every tier-II order assigned skill II, floor binding).
p <- ue %>% filter(base_year == 2024) %>% select(unit, state_fips, s24 = s, h24 = h_base) %>%
  inner_join(ue %>% filter(base_year == 2025) %>% select(unit, s25 = s, h25 = h_base), by = "unit")
pf <- fixest::feols(s25 ~ s24 | state_fips, data = p, weights = ~h24, cluster = ~state_fips)
pf_rev <- fixest::feols(s24 ~ s25 | state_fips, data = p, weights = ~h25, cluster = ~state_fips)
# (2) DOL's projection in the IFR RIA (90 FR 47914, Exhibit 8): wage transfers assume 92% of H-2A workers at the
#     entry (Skill I) wage and 8% at the experience (Skill II) wage. With the FY2025 tier-II hours share s_bar,
#     the implied assignment pass-through is 0.08 / s_bar (assuming no FY2025 tier-I order is assigned skill II).
sbar <- with(rows %>% filter(base_year == 2025, !is.na(tier2)), sum(cert_hours[tier2]) / sum(cert_hours))
mech <- tibble(quantity = c("persistence_slope_s25_on_s24_within_state", "persistence_se_state", "persistence_slope_reverse",
  "units_in_both_years", "s_bar_fy2025_hours", "dol_ria_skill2_share", "pi_dol_implied"),
  value = c(coef(pf)[["s24"]], se(pf)[["s24"]], coef(pf_rev)[["s25"]], nrow(p), sbar, 0.08, 0.08 / sbar))
write_csv(mech, file.path(out, "n1_passthrough_mechanics.csv")); print(as.data.frame(mech))
message("done ", format(Sys.time()))
