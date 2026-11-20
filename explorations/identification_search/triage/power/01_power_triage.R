# Identification search, Phase 1 triage (power-first angle): treatment-side
# support statistics and rough design-stage MDEs for catalogue entries and new
# strategies.
#
# Usage (repository root):
#   Rscript --vanilla explorations/identification_search/triage/power/01_power_triage.R "$S" [blocks]
# `blocks` is an optional comma list (A1,D1,A3,NP3,A2,C2,C1). Outputs go to
# "$S/triage/power_*.csv".
#
# Integrity: no outcome is regressed on any dose. The only outcome use is the
# FE-only residual variance of calibration-year outcomes inside ish_power()
# (pre-period for A1/D1/A3/NP1/NP3; off-event years for A2 and the C family,
# labeled in the output). Doses are statutory or engineered from public
# schedules; approximations are labeled in the `notes` column.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(tidyr); library(readr)})
source(path_code("designs", "border_discontinuity", "power.R"))
source("explorations/identification_search/shared/power_harness.R")
setFixest_nthreads(1)
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "triage"); dl <- file.path(out, "power_dl")
if (startsWith(paste0(S, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
blocks <- if (length(args) > 1) strsplit(args[2], ",")[[1]] else c("A1", "D1", "A3", "NP3", "A2", "C2", "C1")
settings <- ish_power_settings()
RATIO <- 1.23  # declared default cost/wage ratio (declarations.json cost_unit.declared_ratio)
SIX <- c("45-2041", "45-2091", "45-2092", "45-2093", "53-7064", "45-2099")

# ---- Common frames -------------------------------------------------------------
cp <- read_parquet(path_processed("county_year_panel.parquet"),
  col_select = c("county_fips", "state_fips", "aewr_region_id", "emp_farm_2011", "year")) %>% filter(year == 2011) %>% select(-year)
base <- cp %>% filter(!state_fips %in% c("02", "15", "11"), is.finite(emp_farm_2011), emp_farm_2011 > 0)
multi <- base %>% group_by(aewr_region_id) %>% filter(n_distinct(state_fips) > 1) %>% ungroup()
oc <- read_parquet(file.path(S, "shared", "outcomes_county_fy.parquet"),
  col_select = c("county_fips", "year", "cert_hours_dq13", "cert_hours_fy", "at_aewr_hours_dq13", "hourly_aewr_known_hours_dq13"))
regional <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>%
  filter(occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference"), worker_category == "covered_workers") %>%
  transmute(state_fips, aewr = as.numeric(aewr), start = as.Date(effective_start), end = as.Date(effective_end_exclusive))
rates <- read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>% filter(as.Date(effective_start) == as.Date("2025-10-02"), skill_level == 1)
sw <- rates %>% filter(trimws(source_sheet) == "Statewide 655.120(b)(1)(i)") %>%
  select(state_fips, worker_category, r = hourly_aewr) %>% mutate(r = as.numeric(r)) %>%
  pivot_wider(names_from = worker_category, values_from = r, names_prefix = "sw_")
rate_on <- function(date) { r <- regional %>% filter(start <= date, is.na(end) | end > date) %>% select(state_fips, aewr)
  stopifnot(!anyDuplicated(r$state_fips)); r }
reg_fy <- bind_rows(lapply(c(2008, 2010:2025), function(fy) rate_on(as.Date(sprintf("%d-04-01", fy))) %>% mutate(year = fy)))
adj <- read_parquet(path_int("county_adjacency2010.parquet")) %>%
  transmute(a = substr(harmonize_county_fips_2010(county_fips), 1, 2), b = substr(harmonize_county_fips_2010(neighbor_county_fips), 1, 2)) %>%
  filter(a != b) %>% distinct()

res_all <- list(); notes <- list()
save_res <- function(tag, r, extra) {
  for (k in c("support", "summary", "null", "calibration")) {
    f <- file.path(out, paste0("power_harness_", k, ".csv"))
    d <- r[[k]] %>% mutate(design = tag, .before = 1)
    if (k == "support") d <- bind_cols(d, extra)
    write_csv(d, f, append = file.exists(f))
  }
}
# PPML-linearized run; MDE in epsilon_cost units: beta_e1 = pi / Cbar.
pw_run <- function(tag, policy, pre, ycount, unit, fe, clusters, jk, node, group = NULL, pi = 1, dgps,
  ri = NULL, loading = NULL, cbar = NULL, note = "") {
  message("== ", tag, " ", format(Sys.time()))
  bad <- unique(policy[[unit]][!is.finite(policy$x)])
  if (length(bad)) { message("dropping units with missing dose: ", length(bad)); policy <- policy[!policy[[unit]] %in% bad, , drop = FALSE]; pre <- pre[!pre[[unit]] %in% bad, , drop = FALSE] }
  lin <- ish_ppml_linearize(pre, policy, ycount, unit)
  pol <- lin$policy; pr <- lin$pre
  stopifnot(nrow(pol) > 0, nrow(pr) > 0)
  mp <- sum(pol$ppml_weight * pol$x) / sum(pol$ppml_weight)
  cb <- if (is.null(cbar)) mp * RATIO else cbar
  ns <- ish_node_period_share(pr, lin$y, unit, "year", fe, node, weights = lin$weights)
  t0 <- Sys.time()
  r <- ish_power(pol, pr, y = lin$y, x = "x", unit = unit, period = "year", fe = fe, clusters = clusters, jk_cluster = jk,
    node = node, group = group, weights = lin$weights, beta_e1 = pi / cb, loading = loading, dgps = dgps,
    settings = settings, ri = ri, label = list())
  gate_dgp <- if (ns$observed_node_period_share <= 1.5 * ns$pure_unit_noise_share) "county_serial" else "max_shared"
  sv <- r$summary
  gate_mde <- if (gate_dgp == "county_serial") sv$size_valid_mde_elasticity[sv$dgp_id == "county_serial"] else
    suppressWarnings(max(sv$size_valid_mde_elasticity[sv$dgp_id != "county_serial"], na.rm = FALSE))
  extra <- tibble(pi = pi, cbar = cb, mean_dose = mp, observed_node_period_share = ns$observed_node_period_share,
    pure_unit_noise_share = ns$pure_unit_noise_share, gate_dgp = gate_dgp, gate_mde_eps_cost = gate_mde,
    calib_years = paste(range(pr$year), collapse = "-"), policy_years = paste(range(pol$year), collapse = "-"),
    seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")), note = note)
  save_res(tag, r, extra)
  print(bind_cols(r$support, extra), width = 250); print(r$summary, width = 200)
  invisible(r)
}

# ---- A1: 2025 statewide rule, FY2025 vs FY2026, PPML hours ---------------------
if (any(c("A1", "D1", "NP2", "NP4", "NPC") %in% blocks)) {
  dose25 <- reg_fy %>% filter(year == 2025) %>% select(state_fips, a25 = aewr) %>% inner_join(sw, by = "state_fips")
  pol_a1 <- tidyr::crossing(multi, year = 2025:2026) %>% left_join(dose25, by = "state_fips") %>%
    mutate(x = ifelse(year == 2025, a25, sw_H2A), region_year = paste(aewr_region_id, year), state_year = paste(state_fips, year))
  stopifnot(all(is.finite(pol_a1$x)))
  pre_a1 <- tidyr::crossing(multi, year = 2020:2025) %>% left_join(oc, by = c("county_fips", "year")) %>%
    mutate(cert_hours_dq13 = coalesce(cert_hours_dq13, 0), region_year = paste(aewr_region_id, year), state_year = paste(state_fips, year))
  loading <- ish_spatial_loading(unique(multi$state_fips), adj, settings$spatial_graph_coefficient)
}
if ("A1" %in% blocks) {
  pw_run("A1_ppml_hours_h2a_skill1", pol_a1, pre_a1, "cert_hours_dq13", "county_fips", c("county_fips", "region_year"),
    c("state_fips", "aewr_region_id"), "state_fips", "state_fips", "aewr_region_id", pi = 0.65, dgps = settings$dgp_ids,
    ri = list(assign = "state_fips", strata = "aewr_region_id", P = 199), loading = loading,
    note = "Dose: FY2025 regional AEWR -> FY2026 statewide H2A skill-I cash rate; pi = 0.65 (Track 1 S3 first stage); calibration FY2020-25 dq13 hours")
}

# ---- NP4: predetermined skill tier (FY2025 job-order experience requirement) ----
# 90 FR 47914 (2025-10-02), preamble: Skill Level I ~ "up to 2 months" related
# experience; Skill Level II ~ experience "normal for the occupation" (e.g. 3
# months). FY2025 (pre-rule) WORK_EXPERIENCE_MONTHS >= 3 predicts Skill II.
if (any(c("NP4", "NPC") %in% blocks)) {
  xl <- readxl::read_excel(file.path(path_root(), "data", "raw", "h2a", "H-2A_Disclosure_Data_FY2025_Q4.xlsx"), guess_max = 50000)
  xl <- xl[, c("CASE_NUMBER", "WORK_EXPERIENCE_MONTHS", "SUPERVISE_OTHER_EMP")]
  ex <- tibble(case_number = trimws(as.character(xl$CASE_NUMBER)), exp_m = suppressWarnings(as.numeric(xl$WORK_EXPERIENCE_MONTHS)),
    sup = toupper(substr(trimws(as.character(xl$SUPERVISE_OTHER_EMP)), 1, 1)) == "Y") %>% distinct(case_number, .keep_all = TRUE) %>%
    mutate(tier2 = coalesce(exp_m >= 3, FALSE))
  cr25 <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"), col_select = c("case_number", "fiscal_year", "county_fips", "state_fips", "in_dq13", "cert_hours")) %>%
    filter(fiscal_year == 2025, in_dq13, cert_hours > 0) %>% left_join(ex, by = "case_number")
  sk <- read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
    filter(trimws(source_sheet) == "Statewide 655.120(b)(1)(i)", as.Date(effective_start) == as.Date("2025-10-02"), worker_category == "H2A") %>%
    transmute(state_fips, lev = paste0("sk", skill_level), r = as.numeric(hourly_aewr)) %>% pivot_wider(names_from = lev, values_from = r)
  sc <- cr25 %>% group_by(county_fips) %>% summarise(h = sum(cert_hours), h2 = sum(cert_hours[tier2 %in% TRUE]), matched = sum(cert_hours[!is.na(exp_m)]), .groups = "drop")
  notes$NP4_support <- cr25 %>% summarise(hours = sum(cert_hours), share_hours_matched = sum(cert_hours[!is.na(exp_m)]) / hours,
    share_hours_exp_ge3 = sum(cert_hours[tier2 %in% TRUE]) / hours, share_hours_exp_ge1 = sum(cert_hours[coalesce(exp_m >= 1, FALSE)]) / hours,
    share_hours_supervise = sum(cert_hours[sup %in% TRUE]) / hours) %>%
    bind_cols(sk %>% semi_join(multi, by = "state_fips") %>% summarise(mean_sk2_minus_sk1 = mean(sk2 - sk1), sd_sk2_minus_sk1 = sd(sk2 - sk1)))
  m4 <- multi %>% left_join(sc, by = "county_fips") %>% group_by(state_fips) %>% mutate(st_s = sum(h2, na.rm = TRUE) / sum(h, na.rm = TRUE)) %>% ungroup() %>%
    mutate(s = ifelse(is.finite(h) & h > 0, h2 / h, st_s), s = coalesce(s, 0))
  ww <- coalesce(m4$h, 0)
  notes$NP4_support <- notes$NP4_support %>% mutate(county_hours_weighted_sd_s = sqrt(sum(ww * (m4$s - sum(ww * m4$s) / sum(ww))^2) / sum(ww)),
    within_state_hours_weighted_sd_s = sqrt(sum(ww * (m4$s - ave(m4$s * ww, m4$state_fips, FUN = sum) / pmax(ave(ww, m4$state_fips, FUN = sum), 1e-9))^2) / sum(ww)))
  print(notes$NP4_support, width = 250)
  pol4 <- pol_a1 %>% left_join(m4 %>% select(county_fips, s), by = "county_fips") %>% left_join(sk, by = "state_fips") %>%
    mutate(x = ifelse(year == 2025, a25, a25 + (1 - s) * (sk1 - a25) + s * (sk2 - a25)))
  pw_run("NP4_skill_tier_state_fe", pol4, pre_a1, "cert_hours_dq13", "county_fips", c("county_fips", "state_year"),
    c("state_fips", "aewr_region_id"), "state_fips", "state_fips", "aewr_region_id", pi = 0.65,
    dgps = c("county_serial", "shared_region_serial", "shared_region_and_pair_serial"),
    note = "Dose: AEWR25 + (1-s)(H2A skill I - AEWR25) + s(H2A skill II - AEWR25), s = FY2025 county share of dq13 certified hours on job orders requiring >= 3 months experience (predetermined); county + state x FY FE; pi = 0.65 assumed")
  pw_run("NP4_skill_tier_region_fe", pol4, pre_a1, "cert_hours_dq13", "county_fips", c("county_fips", "region_year"),
    c("state_fips", "aewr_region_id"), "state_fips", "state_fips", "aewr_region_id", pi = 0.65,
    dgps = c("county_serial", "shared_region_serial", "shared_region_and_pair_serial"),
    note = "As NP4 with county + former region x FY FE (adds A1's cross-state contrast)")
}

# ---- NPC: combined ex-ante effective dose (D1 binding share x NP4 tier x NP2 MW) ---
# Requires NP4 and NP2 in the same call (uses pol4 and eff).
if ("NPC" %in% blocks) {
  stopifnot(exists("pol4"))
  mwa <- read_parquet(path_int("state_real_minwages.parquet")) %>% filter(year %in% 2025:2026) %>%
    select(state_fips, year, m = prevailing_ag_min_wage) %>% pivot_wider(names_from = year, values_from = m, names_prefix = "mw") %>%
    mutate(mw2025 = coalesce(mw2025, 7.25), mw2026 = coalesce(mw2026, mw2025), mw_dq13 = mw2025 / 3 + 2 * mw2026 / 3)
  sh <- oc %>% filter(year == 2025) %>% select(county_fips, at = at_aewr_hours_dq13, kn = hourly_aewr_known_hours_dq13)
  shs <- multi %>% left_join(sh, by = "county_fips") %>% group_by(state_fips) %>% mutate(st_share = sum(at, na.rm = TRUE) / sum(kn, na.rm = TRUE)) %>% ungroup() %>%
    mutate(share = ifelse(is.finite(kn) & kn > 0, at / kn, st_share), share = pmin(pmax(share, 0), 1)) %>% select(county_fips, share)
  polc <- pol4 %>% left_join(shs, by = "county_fips") %>% left_join(mwa %>% select(state_fips, mw2025, mw_dq13), by = "state_fips") %>%
    mutate(floor26 = pmax((1 - s) * sk1 + s * sk2, mw_dq13), e25 = pmax(a25, mw2025),
      x = ifelse(year == 2025, e25, e25 + share * (floor26 - e25)))
  for (fe2 in c("state_year", "region_year")) {
    pw_run(paste0("NPC_combined_", fe2), polc, pre_a1, "cert_hours_dq13", "county_fips", c("county_fips", fe2),
      c("state_fips", "aewr_region_id"), "state_fips", "state_fips", "aewr_region_id", pi = 0.8,
      dgps = c("county_serial", "shared_region_serial", "shared_region_and_pair_serial"),
      note = "Dose: e25 + share_at_AEWR_2025 x (max(tier-weighted H2A skill rate, ag MW) - e25); pi = 0.8 ASSUMED; MDE scales with 1/pi")
  }
}

# ---- NP2: FY2026 effective binding floor (statutory cut truncated by state ag minimum wage)
if ("NP2" %in% blocks) {
  mwa <- read_parquet(path_int("state_real_minwages.parquet")) %>% filter(year %in% 2025:2026) %>%
    select(state_fips, year, m = prevailing_ag_min_wage) %>% pivot_wider(names_from = year, values_from = m, names_prefix = "mw")
  eff <- dose25 %>% left_join(mwa, by = "state_fips") %>%
    mutate(mw2025 = coalesce(mw2025, 7.25), mw2026 = coalesce(mw2026, mw2025), mw_dq13 = mw2025 / 3 + 2 * mw2026 / 3,
      e25 = pmax(a25, mw2025), e26 = pmax(sw_H2A, mw_dq13), stat_cut = sw_H2A - a25, eff_cut = e26 - e25)
  st_in <- eff %>% semi_join(multi, by = "state_fips")
  notes$NP2_dose <- st_in %>% summarise(states = n(), states_mw_binds_2026 = sum(mw_dq13 > sw_H2A), mean_stat_cut = mean(stat_cut), mean_eff_cut = mean(eff_cut),
    sd_stat_cut = sd(stat_cut), sd_eff_cut = sd(eff_cut), cor_stat_eff = cor(stat_cut, eff_cut))
  write_csv(eff, file.path(out, "power_note_NP2_state_doses.csv"))
  print(notes$NP2_dose, width = 200)
  pol_np2 <- pol_a1 %>% left_join(eff %>% select(state_fips, e25, e26), by = "state_fips") %>% mutate(x = ifelse(year == 2025, e25, e26))
  pw_run("NP2_effective_floor_pi0.9", pol_np2, pre_a1, "cert_hours_dq13", "county_fips", c("county_fips", "region_year"),
    c("state_fips", "aewr_region_id"), "state_fips", "state_fips", "aewr_region_id", pi = 0.9, dgps = settings$dgp_ids,
    ri = list(assign = "state_fips", strata = "aewr_region_id", P = 199), loading = loading,
    note = "Dose: max(H2A skill I, applicable ag minimum wage blended 1/3 2025 + 2/3 2026) vs max(AEWR25, ag MW 2025); pi = 0.9 ASSUMED (to be estimated); MDE scales with 1/pi")
}

# ---- D1: revealed binding x statutory change -----------------------------------
if ("D1" %in% blocks) {
  sh <- oc %>% filter(year == 2025) %>% select(county_fips, at = at_aewr_hours_dq13, kn = hourly_aewr_known_hours_dq13)
  shs <- multi %>% left_join(sh, by = "county_fips") %>% group_by(state_fips) %>%
    mutate(st_share = sum(at, na.rm = TRUE) / sum(kn, na.rm = TRUE)) %>% ungroup() %>%
    mutate(share = ifelse(is.finite(kn) & kn > 0, at / kn, st_share), share = pmin(pmax(share, 0), 1))
  w25 <- shs$kn; w25[!is.finite(w25)] <- 0
  notes$D1_share <- tibble(counties = nrow(shs), counties_with_hourly_2025 = sum(w25 > 0),
    hours_weighted_mean_share = sum(w25 * shs$share) / sum(w25), hours_weighted_sd_share = sqrt(sum(w25 * (shs$share - sum(w25 * shs$share) / sum(w25))^2) / sum(w25)),
    share_hours_in_counties_below_0.8 = sum(w25[shs$share < 0.8]) / sum(w25), unweighted_sd_share = sd(shs$share, na.rm = TRUE))
  print(notes$D1_share)
  pol_d1 <- pol_a1 %>% left_join(shs %>% select(county_fips, share), by = "county_fips") %>%
    mutate(x = ifelse(year == 2025, a25, a25 + share * (sw_H2A - a25)))
  pw_run("D1_region_fe", pol_d1, pre_a1, "cert_hours_dq13", "county_fips", c("county_fips", "region_year"),
    c("state_fips", "aewr_region_id"), "state_fips", "state_fips", "aewr_region_id", pi = 1,
    dgps = c("county_serial", "shared_region_serial", "shared_region_and_pair_serial"),
    note = "Dose: AEWR25 + FY2025 county at-AEWR hours share x (H2A skill I 2026 - AEWR25); pi = 1 assumed (binding-weighted dose)")
  pw_run("D1_state_fe", pol_d1, pre_a1, "cert_hours_dq13", "county_fips", c("county_fips", "state_year"),
    c("state_fips", "aewr_region_id"), "state_fips", "state_fips", "aewr_region_id", pi = 1,
    dgps = c("county_serial", "shared_region_serial", "shared_region_and_pair_serial"),
    note = "As D1_region_fe with state x FY FE: identification only from within-state dispersion of the FY2025 binding share")
}

# ---- Case rows + SOC (A3, NP1, NP3) --------------------------------------------
if (any(c("A3", "NP3") %in% blocks)) {
  cr <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"),
    col_select = c("case_number", "fiscal_year", "state_fips", "employer_id_balanced", "fy_receipt", "in_rq13d", "in_dq13", "cert_hours")) %>%
    filter(fiscal_year >= 2020 | fy_receipt >= 2020)
  soc <- read_parquet(path_int("h2a_disclosure_records.parquet"), col_select = c("case_number", "fiscal_year", "soc_code")) %>%
    filter(fiscal_year >= 2019) %>% mutate(soc7 = substr(trimws(soc_code), 1, 7)) %>% filter(nchar(soc7) == 7) %>%
    distinct(case_number, fiscal_year, soc7) %>% group_by(case_number, fiscal_year) %>% slice(1) %>% ungroup()
  cr <- cr %>% left_join(soc, by = c("case_number", "fiscal_year")) %>% filter(!is.na(employer_id_balanced), !is.na(state_fips), cert_hours > 0)
  message("case rows with SOC: ", mean(!is.na(cr$soc7)))
}
if ("A3" %in% blocks) {
  # State OEWS mean wage by SOC x release year, employment-weighted over areas (approximation to the statewide OEWS AEWR).
  st_map <- rates %>% distinct(state_name, state_fips) %>% inner_join(tibble(state_name = c(state.name, "District of Columbia"), abb = c(state.abb, "DC")), by = "state_name")
  ow <- read_parquet(path_int("oews.parquet"), col_select = c("area", "area_name", "occ_code", "year", "tot_emp", "h_mean")) %>%
    filter(year %in% 2021:2023) %>% mutate(h = suppressWarnings(as.numeric(h_mean)), w = coalesce(tot_emp, 1))
  ow <- ow %>% filter(is.finite(h)) %>%
    mutate(abb = ifelse(nchar(area) == 7, NA_character_, sub("^([A-Z]{2}).*$", "\\1", trimws(sub("^.*,", "", area_name)))),
      st7 = ifelse(nchar(area) == 7, substr(area, 1, 2), NA_character_)) %>%
    left_join(st_map %>% select(abb, st_abb = state_fips), by = "abb") %>% mutate(state_fips = coalesce(st7, st_abb)) %>% filter(!is.na(state_fips))
  ow_st <- ow %>% group_by(state_fips, soc7 = occ_code, oy = year) %>% summarise(oews = sum(w * h) / sum(w), .groups = "drop")
  ow_nat <- ow %>% group_by(soc7 = occ_code, oy = year) %>% summarise(oews_nat = sum(w * h) / sum(w), .groups = "drop")
  r26 <- rates %>% filter(trimws(source_sheet) == "Statewide 655.120(b)(1)(ii)", worker_category == "H2A") %>%
    transmute(state_fips, soc7 = substr(soc_code, 1, 7), r26 = as.numeric(hourly_aewr)) %>% distinct(state_fips, soc7, .keep_all = TRUE)

  a3 <- cr %>% filter(in_rq13d, fy_receipt %in% 2020:2026) %>% mutate(grp = ifelse(is.na(soc7) | soc7 %in% SIX, "field", "nonfield"))
  # Dominant non-field SOC per employer x state cell in FY2020-22 (frozen)
  domsoc <- a3 %>% filter(grp == "nonfield", fy_receipt <= 2022) %>% group_by(employer_id_balanced, state_fips, soc7) %>%
    summarise(h = sum(cert_hours), .groups = "drop") %>% group_by(employer_id_balanced, state_fips) %>% slice_max(h, n = 1, with_ties = FALSE) %>%
    ungroup() %>% select(employer_id_balanced, state_fips, dsoc = soc7)
  cells <- a3 %>% group_by(employer_id_balanced, state_fips, grp, year = fy_receipt) %>% summarise(hours = sum(cert_hours), .groups = "drop")
  keep_emp <- cells %>% filter(year <= 2022) %>% distinct(employer_id_balanced, grp) %>% count(employer_id_balanced) %>% filter(n == 2) %>% pull(employer_id_balanced)
  cell_ids <- cells %>% filter(year <= 2022) %>% distinct(employer_id_balanced, state_fips, grp)
  frame <- tidyr::crossing(cell_ids, year = 2020:2026) %>% left_join(cells, by = c("employer_id_balanced", "state_fips", "grp", "year")) %>%
    mutate(hours = coalesce(hours, 0)) %>% left_join(domsoc, by = c("employer_id_balanced", "state_fips")) %>%
    left_join(cp %>% distinct(state_fips, aewr_region_id), by = "state_fips") %>% filter(!is.na(aewr_region_id)) %>%
    left_join(reg_fy %>% rename(aewr_t = aewr), by = c("state_fips", "year")) %>% left_join(sw %>% select(state_fips, sw_H2A), by = "state_fips")
  ow_lookup <- function(f, y) f %>% left_join(ow_st %>% filter(oy == y) %>% select(state_fips, dsoc = soc7, o = oews), by = c("state_fips", "dsoc")) %>%
    left_join(ow_nat %>% filter(oy == y) %>% select(dsoc = soc7, on = oews_nat), by = "dsoc") %>% mutate(o = coalesce(o, on)) %>% pull(o)
  frame$o21 <- ow_lookup(frame, 2021); frame$o22 <- ow_lookup(frame, 2022); frame$o23 <- ow_lookup(frame, 2023)
  frame <- frame %>% left_join(r26 %>% rename(dsoc = soc7), by = c("state_fips", "dsoc")) %>%
    mutate(aewr_t = ifelse(year == 2026, NA_real_, aewr_t),
      field_rate = ifelse(year == 2026, sw_H2A, aewr_t),
      nf_rate = case_when(year <= 2022 ~ aewr_t, year == 2023 ~ (6 * aewr_t + 3 * coalesce(o21, aewr_t)) / 9,
        year == 2024 ~ coalesce(o22, aewr_t), year == 2025 ~ coalesce(o23, aewr_t), year == 2026 ~ coalesce(r26, sw_H2A)),
      x = ifelse(grp == "field", field_rate, nf_rate),
      cell = paste(employer_id_balanced, state_fips, grp), emp_year = paste(employer_id_balanced, year),
      state_year = paste(state_fips, year), region_year = paste(aewr_region_id, year))
  frame <- frame %>% filter(is.finite(x) | year > 2025)
  sup <- frame %>% filter(year == 2024, grp == "nonfield") %>% summarise(nonfield_cells = n(), oews_found = mean(is.finite(o22)), mean_nf_rate_2024 = mean(nf_rate, na.rm = TRUE), mean_aewr_2024 = mean(aewr_t, na.rm = TRUE))
  sup2 <- frame %>% filter(year <= 2022) %>% group_by(grp) %>% summarise(cells = n_distinct(cell), employers = n_distinct(employer_id_balanced), hours = sum(hours), .groups = "drop")
  notes$A3_support <- bind_cols(sup, tibble(employers_with_both = length(keep_emp)), tibble(field_hours_2020_22 = sup2$hours[sup2$grp == "field"], nonfield_hours_2020_22 = sup2$hours[sup2$grp == "nonfield"], nonfield_cells_all = sup2$cells[sup2$grp == "nonfield"], nonfield_employers_all = sup2$employers[sup2$grp == "nonfield"]))
  print(notes$A3_support, width = 200)
  gap <- frame %>% filter(grp == "nonfield", year == 2024) %>% mutate(g = nf_rate - aewr_t)
  notes$A3_gap <- tibble(mean_gap_2024 = mean(gap$g, na.rm = TRUE), sd_gap_2024 = sd(gap$g, na.rm = TRUE), p10 = quantile(gap$g, .1, na.rm = TRUE), p90 = quantile(gap$g, .9, na.rm = TRUE))
  g26 <- frame %>% filter(year == 2026) %>% group_by(cell) %>% summarise(x26 = first(x), grp = first(grp)) %>% left_join(frame %>% filter(year == 2025) %>% select(cell, x25 = x), by = "cell") %>% mutate(d = x26 - x25)
  notes$NP1_dose2026 <- g26 %>% group_by(grp) %>% summarise(mean_change = mean(d, na.rm = TRUE), sd_change = sd(d, na.rm = TRUE), .groups = "drop")
  print(notes$A3_gap); print(notes$NP1_dose2026)
  fe_emp <- frame %>% filter(employer_id_balanced %in% keep_emp)
  dg <- c("county_serial", "shared_region_serial")
  pw_run("A3_emp_year_fe_FY2020_25", fe_emp %>% filter(year <= 2025), fe_emp %>% filter(year <= 2022),
    "hours", "cell", c("cell", "emp_year"), c("state_fips", "employer_id_balanced"), "state_fips", "state_fips", pi = 1, dgps = dg,
    note = "Employer x state x {field,nonfield} cells, receipt FY (rq13d); employers with both groups in FY2020-22; dose: 2023 non-field statewide OEWS mean (approximated from area OEWS, vintage FY-2) vs FLS AEWR; pi = 1 assumed")
  pw_run("A3_state_year_fe_FY2020_25", frame %>% filter(year <= 2025), frame %>% filter(year <= 2022),
    "hours", "cell", c("cell", "state_year"), c("state_fips", "employer_id_balanced"), "state_fips", "state_fips", pi = 1, dgps = dg,
    note = "As A3 with cell + state x FY FE (all cells; field cells in the same state as controls)")
  pw_run("NP1_emp_year_fe_FY2020_26", fe_emp, fe_emp %>% filter(year <= 2022), "hours", "cell", c("cell", "emp_year"),
    c("state_fips", "employer_id_balanced"), "state_fips", "state_fips", pi = 1, dgps = dg,
    note = "NP1: A3 plus the FY2026 reversal (field -> (b)(1)(i) H2A skill I; non-field -> (b)(1)(ii) H2A skill I for the cell's SOC); pi = 1 assumed")
  pw_run("NP1_state_year_fe_FY2020_26", frame, frame %>% filter(year <= 2022), "hours", "cell", c("cell", "state_year"),
    c("state_fips", "employer_id_balanced"), "state_fips", "state_fips", pi = 1, dgps = dg,
    note = "NP1 with cell + state x FY FE")
}

# ---- NP3: within-employer cross-state reallocation under the 2025 rule ---------
if ("NP3" %in% blocks) {
  dose25 <- reg_fy %>% filter(year == 2025) %>% select(state_fips, a25 = aewr) %>% inner_join(sw, by = "state_fips")
  e3 <- cr %>% filter(in_dq13, fiscal_year %in% 2021:2026) %>% group_by(employer_id_balanced, state_fips, year = fiscal_year) %>%
    summarise(hours = sum(cert_hours), .groups = "drop")
  multi_emp <- e3 %>% filter(year <= 2025) %>% distinct(employer_id_balanced, state_fips) %>% count(employer_id_balanced) %>% filter(n >= 2)
  ids <- e3 %>% filter(year <= 2025, employer_id_balanced %in% multi_emp$employer_id_balanced) %>% distinct(employer_id_balanced, state_fips)
  st <- cp %>% distinct(state_fips, aewr_region_id)
  fr3 <- tidyr::crossing(ids, year = 2021:2026) %>% left_join(e3, by = c("employer_id_balanced", "state_fips", "year")) %>%
    mutate(hours = coalesce(hours, 0)) %>% inner_join(st, by = "state_fips") %>% inner_join(dose25, by = "state_fips") %>%
    mutate(x = ifelse(year == 2026, sw_H2A, a25), cell = paste(employer_id_balanced, state_fips), emp_year = paste(employer_id_balanced, year),
      region_year = paste(aewr_region_id, year))
  tot <- e3 %>% filter(year == 2025) %>% summarise(h = sum(hours)) %>% pull(h)
  notes$NP3_support <- tibble(multi_state_employers = n_distinct(fr3$employer_id_balanced), cells = n_distinct(fr3$cell),
    share_fy2025_hours = sum(fr3$hours[fr3$year == 2025]) / tot)
  print(notes$NP3_support)
  pw_run("NP3_emp_state_cells", fr3 %>% filter(year >= 2025), fr3 %>% filter(year <= 2025), "hours", "cell",
    c("cell", "emp_year", "region_year"), c("state_fips", "employer_id_balanced"), "state_fips", "state_fips", pi = 0.65,
    dgps = c("county_serial", "shared_region_serial"), ri = list(assign = "state_fips", strata = NULL, P = 199),
    note = "Employer x state cells of multi-state employers (dq13); FE cell + employer x FY + former region x FY; dose as A1; pi = 0.65")
}

# ---- A2: 2009 OES switch / 2010 FLS return ---------------------------------------
if ("A2" %in% blocks) {
  mw <- read_parquet(path_int("fred_state_minwages.parquet")) %>% filter(year == 2009) %>% transmute(state_fips, smw = as.numeric(state_min_wage))
  ad <- read_parquet(path_int("oews_area_definitions.parquet")) %>% filter(year == 2008) %>% distinct(county_fips, state_fips, oews_area_code) %>%
    mutate(key = ifelse(nchar(oews_area_code) == 6, paste0(substr(oews_area_code, 1, 2), "0", substr(oews_area_code, 3, 6)), oews_area_code))
  o8 <- read_parquet(path_int("oews.parquet"), col_select = c("area", "occ_code", "year", "h_pct10", "h_pct25")) %>%
    filter(year == 2008, occ_code == "45-2092") %>% transmute(key = area, l1 = (suppressWarnings(as.numeric(h_pct10)) + suppressWarnings(as.numeric(h_pct25))) / 2) %>%
    filter(is.finite(l1)) %>% distinct(key, .keep_all = TRUE)
  cmap <- base %>% left_join(ad %>% select(county_fips, key), by = "county_fips") %>% group_by(county_fips) %>% slice(1) %>% ungroup() %>%
    left_join(o8, by = "key") %>% group_by(state_fips) %>% mutate(l1_state_med = median(l1, na.rm = TRUE)) %>% ungroup() %>%
    mutate(matched = is.finite(l1), l1 = coalesce(l1, l1_state_med)) %>% left_join(mw, by = "state_fips") %>%
    mutate(floor09 = pmax(l1, 7.25, coalesce(smw, 0), na.rm = TRUE), area = paste(coalesce(key, "none"), state_fips))
  r8 <- reg_fy %>% filter(year == 2008) %>% select(state_fips, a08 = aewr); r10 <- reg_fy %>% filter(year == 2010) %>% select(state_fips, a10 = aewr)
  r11 <- reg_fy %>% filter(year == 2011) %>% select(state_fips, a11 = aewr)
  cmap <- cmap %>% left_join(r8, by = "state_fips") %>% left_join(r10, by = "state_fips") %>% left_join(r11, by = "state_fips")
  notes$A2_support <- cmap %>% summarise(counties = n(), oes_area_matched = mean(matched), areas = n_distinct(area),
    mean_l1 = mean(l1, na.rm = TRUE), share_floor_binding_mw = mean(floor09 > l1 + 1e-9, na.rm = TRUE),
    mean_cut_vs_aewr08 = mean(floor09 - a08, na.rm = TRUE), sd_floor09 = sd(floor09, na.rm = TRUE),
    within_state_sd_floor09 = sqrt(mean((floor09 - ave(floor09, state_fips))^2, na.rm = TRUE)))
  print(notes$A2_support, width = 200)
  pol2 <- tidyr::crossing(cmap %>% select(county_fips, state_fips, aewr_region_id, area, floor09, a08, a10, a11), year = 2008:2011) %>%
    mutate(x = case_when(year == 2008 ~ a08, year == 2009 ~ 0.3 * a08 + 0.7 * floor09, year == 2010 ~ 0.55 * floor09 + 0.45 * a10, TRUE ~ a11),
      state_year = paste(state_fips, year), region_year = paste(aewr_region_id, year))
  pre2 <- tidyr::crossing(cmap %>% select(county_fips, state_fips, aewr_region_id, area), year = 2011:2014) %>% left_join(oc, by = c("county_fips", "year")) %>%
    mutate(cert_hours_fy = coalesce(cert_hours_fy, 0), state_year = paste(state_fips, year), region_year = paste(aewr_region_id, year))
  pw_run("A2_substate_state_fe", pol2, pre2, "cert_hours_fy", "county_fips", c("county_fips", "state_year"), c("state_fips", "area"),
    "state_fips", "state_fips", pi = 0.6, dgps = c("county_serial", "shared_region_serial"),
    ri = list(assign = "area", strata = "state_fips", P = 199),
    note = "Dose proxy: OES May-2008 45-2092 area (p10+p25)/2 floored at max(7.25, state MW 2009); FY2009 weight 0.7, FY2010 0.55; county + state x FY FE; pi = 0.6 assumed; calibration FY2011-14 (off-event, not pre-period)")
  t1 <- read_parquet(path_int("aewr_2009_nonstatutory_wages.parquet")) %>% transmute(state_fips, w09 = as.numeric(aewr_nonstatutory_application_wage_2009))
  pol2s <- pol2 %>% left_join(t1, by = "state_fips") %>% filter(is.finite(w09)) %>%
    mutate(x = case_when(year == 2008 ~ a08, year == 2009 ~ 0.3 * a08 + 0.7 * w09, year == 2010 ~ 0.55 * w09 + 0.45 * a10, TRUE ~ a11))
  pre2s <- pre2 %>% semi_join(pol2s, by = "county_fips")
  pw_run("A2_state_table1_region_fe", pol2s, pre2s, "cert_hours_fy", "county_fips", c("county_fips", "region_year"), c("state_fips", "aewr_region_id"),
    "state_fips", "state_fips", "aewr_region_id", pi = 1, dgps = c("county_serial", "shared_region_serial", "shared_region_and_pair_serial"),
    ri = list(assign = "state_fips", strata = "aewr_region_id", P = 199),
    note = "Dose: 2010 rule Table 1 state mean certified 2009 application wage (realized, so pi = 1); county + former region x FY FE; calibration FY2011-14 (off-event)")
}

# ---- C2: transportation (distance x diesel) --------------------------------------
if ("C2" %in% blocks) {
  dbf <- foreign::read.dbf(file.path(dl, "tl_2020_us_county.dbf"), as.is = TRUE)
  cc <- tibble(county_fips = county_fips(dbf$GEOID), lat = as.numeric(dbf$INTPTLAT), lon = as.numeric(dbf$INTPTLON))
  hav <- function(la1, lo1, la2, lo2) { r <- pi / 180; a <- sin((la2 - la1) * r / 2)^2 + cos(la1 * r) * cos(la2 * r) * sin((lo2 - lo1) * r / 2)^2; 2 * 3958.8 * asin(sqrt(a)) }
  cc$dist <- 1.2 * hav(cc$lat, cc$lon, 25.67, -100.31)  # Monterrey consulate, road factor 1.2 (assumption)
  dz <- read_csv(file.path(dl, "diesel_gasdesw.csv"), show_col_types = FALSE) %>% transmute(d = as.Date(observation_date), p = as.numeric(GASDESW)) %>%
    filter(is.finite(p)) %>% mutate(year = as.integer(format(d, "%Y")) + (as.integer(format(d, "%m")) >= 10)) %>% group_by(year) %>% summarise(diesel = mean(p))
  dz <- dz %>% filter(year %in% 2012:2025) %>% mutate(rel = diesel / mean(diesel) - 1)
  cb <- reg_fy %>% filter(year %in% 2012:2025) %>% summarise(m = mean(aewr)) %>% pull(m) * RATIO
  c2 <- base %>% inner_join(cc, by = "county_fips") %>% mutate(dist_rel = dist / mean(dist))
  pol_c2 <- tidyr::crossing(c2, dz) %>% mutate(x = 1050 * 0.25 * rel * dist_rel / 1000, region_year = paste(aewr_region_id, year), state_year = paste(state_fips, year))
  notes$C2_support <- tibble(diesel_fy_min = min(dz$diesel), diesel_fy_max = max(dz$diesel), dist_mean_miles = mean(c2$dist), dist_sd_miles = sd(c2$dist),
    dose_sd_raw = sd(pol_c2$x), cbar = cb)
  print(notes$C2_support)
  pre_c2 <- pol_c2 %>% select(-x) %>% left_join(oc, by = c("county_fips", "year")) %>% mutate(cert_hours_fy = coalesce(cert_hours_fy, 0))
  pw_run("C2_region_fe", pol_c2, pre_c2, "cert_hours_fy", "county_fips", c("county_fips", "region_year"), c("state_fips", "aewr_region_id"),
    "state_fips", "state_fips", pi = 1, dgps = c("county_serial", "shared_region_serial"), cbar = cb,
    note = "Dose ($/hour): $1,050 round trip x 0.25 fuel share x (FY diesel/mean - 1) x (road distance to Monterrey / mean) / 1,000 contract hours (all assumptions); calibration FY2012-25 FE-only residuals (no pre-period exists)")
}

# ---- C1: housing (assumed dose distribution) -----------------------------------
if ("C1" %in% blocks) {
  set.seed(20261105)
  cb <- reg_fy %>% filter(year %in% 2012:2025) %>% summarise(m = mean(aewr)) %>% pull(m) * RATIO
  innov <- base %>% select(county_fips) %>% tidyr::crossing(year = 2012:2025) %>% arrange(county_fips, year) %>%
    mutate(e = rnorm(n(), 0, 0.03)) %>% group_by(county_fips) %>% mutate(x = cumsum(e)) %>% ungroup() %>% select(-e)
  pol_c1 <- base %>% inner_join(innov, by = "county_fips") %>% mutate(state_year = paste(state_fips, year))
  pre_c1 <- pol_c1 %>% select(-x) %>% left_join(oc, by = c("county_fips", "year")) %>% mutate(cert_hours_fy = coalesce(cert_hours_fy, 0))
  pw_run("C1_assumed_rw003", pol_c1, pre_c1, "cert_hours_fy", "county_fips", c("county_fips", "state_year"), c("state_fips"),
    "state_fips", "state_fips", pi = 1, dgps = c("county_serial", "shared_region_serial"), cbar = cb,
    note = "ASSUMED dose: county random walk in housing $/hour with innovation SD $0.03/yr (about 2% of a $1.5-1.9 housing cost; HUD FMR idiosyncratic growth assumed); FE county + state x FY; full pass-through and cost equivalence")
}

for (k in names(notes)) write_csv(notes[[k]], file.path(out, paste0("power_note_", k, ".csv")))
message("done ", format(Sys.time()))
