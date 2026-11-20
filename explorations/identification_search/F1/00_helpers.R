# Identification search, strategy F1 (counterfactual market-wage recovery):
# shared helpers. Sourced by 01_support.R and 02_design_precision.R (and by
# the Phase 4-5 scripts that implement explorations/identification_search/F1/registry.json).
#
# What lives here:
#   f1_orders(S, fys)    job-order x state x MW-area table of hourly H-2A offers
#                        (receipt window rq13d), with SOC group, experience
#                        class, piece-rate flag, FLC flag and certified hours.
#   f1_classify(o)       F1's own floor classification (not the shared at-AEWR
#                        flag; see the A1 review): every applicable floor known
#                        to the repository is a candidate censoring point.
#
# Floor classification (declared in the registry, section "floor_classification"):
#   New rule (receipt >= 2025-10-02): candidate floors = statewide (b)(1)(i)
#     U.S. and H-2A skill I and II rates; SOC-specific (b)(1)(ii) U.S./H-2A skill
#     I/II rates for the order's SOC; every state agricultural minimum wage value
#     (A1 sourced table, all sub-state areas of the state) in force on any day of
#     [received, end]; the federal $7.25. Applicable floor F_app = max(U.S. rate
#     at the order's skill level for its SOC group, the order area's maximum MW
#     over [received, end], 7.25). Skill II if WORK_EXPERIENCE_MONTHS >= 3.
#   Old rule (receipt < 2025-10-02): candidate floors = the reference AEWR in
#     force on the receipt, decision or period-begin date (plus the prior rate
#     before an alternative start), as in shared/01_outcomes.R; F_app = max of
#     those values. Non-field SOCs received 2023-03-30 to 2025-10-01 have an
#     unknown floor (2023-rule OEWS AEWRs not in the repository); rows with no
#     reference rate (2009-01-17 to 2010-03-15) have an unknown floor. State
#     minimum wages are not candidates before 2025-10-02 (they exceeded the
#     AEWR in one state-year, AZ 2018, by $0.04; credibility triage fact 1).
#   Classes (tolerance $0.05, b01 AEWR_OFFER_TOLERANCE):
#     at_floor      within 0.05 of a candidate floor (censored at that value)
#     at_old_aewr   new rule only: within 0.05 of the state's FY2025 AEWR
#                   (19.97-type rates in force 2024-12-16 to 2025-10-01) and not
#                   at a candidate floor; not a legal floor after 2025-10-02
#     above         > F_app + 0.05 and at no candidate floor (exact observation)
#     between_rates old rule only: between two candidate AEWRs, at neither
#     below_skill2  new rule only: experience >= 3 months and between the skill-I
#                   and skill-II floors, at neither
#     below_floor   < F_app - 0.05 otherwise (data or schedule mismatch, unseen
#                   lower rate); excluded from the model, counted
#     floor_unknown no applicable floor known
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(tidyr); library(readr)})

F1_DIR <- "explorations/identification_search/F1"
TOL <- 0.05
NEW_RULE_START <- as.Date("2025-10-02")
RULE2023_START <- as.Date("2023-03-30")
# Field and livestock SOCs (2023 AEWR rule list; triage and A1 convention)
FIELD_SOC <- c("45-2041", "45-2091", "45-2092", "45-2093", "53-7064", "45-2099")

f1_out_dir <- function(S) {
  S <- normalizePath(S); out <- file.path(S, "F1")
  if (startsWith(paste0(S, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
  dir.create(out, showWarnings = FALSE); out
}

# ---- Floors ------------------------------------------------------------------
f1_reference_aewr <- function() {
  read_parquet(path_int("aewr_rate_schedule.parquet")) %>%
    filter(occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference"), worker_category == "covered_workers") %>%
    transmute(state_fips, aewr = as.numeric(aewr), start = as.Date(effective_start), end = as.Date(effective_end_exclusive),
      alt_start = as.Date(alternative_effective_start)) %>% arrange(state_fips, start) %>%
    group_by(state_fips) %>% mutate(prev_aewr = lag(aewr)) %>% ungroup()
}

# 2025-10-02 statewide (b)(1)(i) rates, one row per state: US_s1, US_s2, H2A_s1, H2A_s2
f1_state_skill_rates <- function() {
  read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
    filter(trimws(source_sheet) == "Statewide 655.120(b)(1)(i)", as.Date(effective_start) == NEW_RULE_START) %>%
    transmute(state_fips, k = paste0(worker_category, "_s", skill_level), r = as.numeric(hourly_aewr)) %>%
    pivot_wider(names_from = k, values_from = r)
}

# 2025-10-02 SOC-specific (b)(1)(ii) rates, one row per state x soc7
f1_soc_skill_rates <- function() {
  read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
    filter(trimws(source_sheet) == "Statewide 655.120(b)(1)(ii)", as.Date(effective_start) == NEW_RULE_START) %>%
    transmute(state_fips, soc7 = substr(soc_code, 1, 7), k = paste0("soc_", worker_category, "_s", skill_level), r = as.numeric(hourly_aewr)) %>%
    distinct(state_fips, soc7, k, .keep_all = TRUE) %>%
    pivot_wider(names_from = k, values_from = r)
}

# State agricultural minimum wage segments (A1 sourced table), with MW areas.
f1_minwage <- function() {
  mw <- read_csv(file.path("explorations/identification_search/A1/sources/state_ag_minwage_fy2025_2026.csv"),
    col_types = cols(state_fips = "c", .default = col_guess()), progress = FALSE)
  mw %>% transmute(state_fips, area_id, seg_start = as.Date(snapshot_date), seg_end = as.Date(valid_through), mw = as.numeric(h2a_state_floor))
}
f1_county_area <- function() {
  read_csv("explorations/identification_search/A1/sources/minwage_substate_counties.csv", col_types = cols(.default = "c"), progress = FALSE)
}
f1_area_of <- function(state_fips, county_fips) {
  sub <- f1_county_area()
  a <- sub$area_id[match(county_fips, sub$county_fips)]
  case_when(state_fips == "36" & a %in% "downstate" ~ "downstate", state_fips == "36" ~ "upstate",
    state_fips == "41" & a %in% "nonurban" ~ "nonurban", state_fips == "41" ~ "standard", TRUE ~ "statewide")
}

# FY2025 AEWR on 2025-04-01 and the minimum over the FY2025 window (old-AEWR mass point)
f1_old_aewr <- function() {
  ref <- f1_reference_aewr()
  on <- ref %>% filter(start <= as.Date("2025-04-01"), is.na(end) | end > as.Date("2025-04-01")) %>% select(state_fips, a25 = aewr)
  win <- ref %>% filter(start <= as.Date("2025-09-30"), is.na(end) | end > as.Date("2024-10-01")) %>%
    group_by(state_fips) %>% summarise(old_rates = list(unique(aewr)), .groups = "drop")
  on %>% left_join(win, by = "state_fips")
}

# ---- Orders --------------------------------------------------------------------
# Experience months by case: FY2018-FY2025 from the credibility triage extract of
# the raw disclosures; FY2026 from the raw FY2026 Q3 disclosure.
f1_experience <- function(S, fys) {
  out <- list()
  if (any(fys %in% 2018:2025)) {
    out[[1]] <- read_parquet(file.path(S, "triage", "cred_job_orders_fy2018_2025.parquet"), col_select = c("fy", "case_number", "exp_months")) %>%
      filter(fy %in% fys, !is.na(case_number)) %>% distinct(case_number, .keep_all = TRUE) %>% select(case_number, exp_months)
  }
  if (2026 %in% fys) {
    xl <- readxl::read_excel(path_raw("h2a", "H-2A_Disclosure_Data_FY2026_Q3.xlsx"), col_types = "text")
    out[[2]] <- tibble(case_number = xl$CASE_NUMBER, exp_months = suppressWarnings(as.numeric(xl$WORK_EXPERIENCE_MONTHS))) %>%
      filter(!is.na(case_number)) %>% distinct(case_number, .keep_all = TRUE)
  }
  bind_rows(out) %>% distinct(case_number, .keep_all = TRUE)
}

# One row per (case_number, fiscal_year, state_fips, mw_area): receipt-window rq13d
# certified rows with an hourly offer (0 < wage <= 100). Weights: certified hours.
f1_orders <- function(S, fys) {
  cr <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"),
    col_select = c("case_number", "fiscal_year", "fy_receipt", "in_rq13d", "received", "decided", "begin", "end",
      "state_fips", "county_fips", "employer_id_balanced", "cert_pos", "cert_hours", "hourly_wage", "soc7")) %>%
    filter(fy_receipt %in% fys, in_rq13d, is.finite(cert_hours), cert_hours > 0, !is.na(state_fips), state_fips != "00",
      is.finite(hourly_wage), hourly_wage > 0, hourly_wage <= 100)
  stopifnot(is.character(cr$state_fips), is.character(cr$county_fips))
  cr <- cr %>% mutate(mw_area = f1_area_of(state_fips, county_fips))
  o <- cr %>% group_by(case_number, fiscal_year, state_fips, mw_area) %>%
    summarise(fy = first(fy_receipt), received = first(received), decided = first(decided), begin = min(begin), end = max(end),
      wage = first(hourly_wage), wage_sd = sd(hourly_wage), soc7_row = first(soc7[!is.na(soc7) & soc7 != ""], default = NA_character_),
      employer_id = { h <- tapply(cert_hours, coalesce(employer_id_balanced, "NA"), sum); names(h)[which.max(h)] },
      cert_pos = sum(cert_pos), hours = sum(cert_hours), .groups = "drop") %>%
    mutate(employer_id = ifelse(employer_id == "NA", NA_character_, employer_id))
  stopifnot(all(is.na(o$wage_sd) | o$wage_sd < 1e-9))
  disc <- open_dataset(path_int("h2a_disclosure_records.parquet")) %>% filter(fiscal_year %in% !!unique(o$fiscal_year)) %>%
    select(case_number, fiscal_year, soc_code, piece_rate_offer, h2a_labor_contractor) %>% collect() %>%
    mutate(soc7_d = substr(gsub("[^0-9-]", "", coalesce(soc_code, "")), 1, 7),
      piece = coalesce(suppressWarnings(as.numeric(piece_rate_offer)), 0) > 0,
      flc = toupper(substr(trimws(coalesce(h2a_labor_contractor, "")), 1, 1)) == "Y") %>%
    distinct(case_number, fiscal_year, .keep_all = TRUE) %>% select(case_number, fiscal_year, soc7_d, piece, flc)
  o <- o %>% left_join(disc, by = c("case_number", "fiscal_year")) %>%
    left_join(f1_experience(S, fys), by = "case_number") %>%
    mutate(soc7 = ifelse(!is.na(soc7_d) & nchar(soc7_d) == 7, soc7_d, soc7_row),
      soc_grp = case_when(is.na(soc7) | soc7 == "" ~ "unknown", soc7 %in% FIELD_SOC ~ "field", TRUE ~ "nonfield"),
      skill = ifelse(!is.na(exp_months) & exp_months >= 3, "II", "I"),
      exp_class = case_when(is.na(exp_months) ~ "NA", exp_months == 0 ~ "0", exp_months < 3 ~ "1-2", exp_months == 3 ~ "3", TRUE ~ "4+"),
      piece = coalesce(piece, FALSE), flc = coalesce(flc, FALSE),
      new_rule = !is.na(received) & received >= NEW_RULE_START) %>% select(-soc7_d, -soc7_row, -wage_sd)
  stopifnot(!anyDuplicated(o[c("case_number", "fiscal_year", "state_fips", "mw_area")]))
  o
}

# ---- Classification --------------------------------------------------------------
f1_classify <- function(o) {
  ref <- f1_reference_aewr(); sr <- f1_state_skill_rates(); so <- f1_soc_skill_rates(); mw <- f1_minwage(); old <- f1_old_aewr()
  o <- o %>% mutate(oid = row_number())
  # Old rule: reference AEWR at receipt, decision, begin (+ prior before alternative start)
  od <- o %>% filter(!new_rule)
  long <- bind_rows(lapply(c("received", "decided", "begin"), function(v) od %>% transmute(oid, state_fips, d = .data[[v]]))) %>%
    filter(!is.na(d)) %>% distinct()
  lk <- long %>% inner_join(ref, by = "state_fips", relationship = "many-to-many") %>%
    filter(d >= start, is.na(end) | d < end) %>%
    mutate(prev = ifelse(!is.na(alt_start) & d < alt_start, prev_aewr, NA_real_))
  cand_old <- bind_rows(lk %>% select(oid, v = aewr), lk %>% filter(!is.na(prev)) %>% select(oid, v = prev)) %>%
    distinct() %>% mutate(ftype = "aewr")
  # New rule candidates
  nd <- o %>% filter(new_rule) %>% left_join(sr, by = "state_fips") %>%
    left_join(so, by = c("state_fips", "soc7"))
  mwseg <- nd %>% select(oid, state_fips, mw_area, received, end) %>%
    inner_join(mw, by = "state_fips", relationship = "many-to-many") %>%
    filter(seg_start <= coalesce(end, received), is.na(seg_end) | seg_end >= received)
  mw_own <- mwseg %>% filter(area_id == mw_area | (area_id == "statewide")) %>% group_by(oid) %>% summarise(mw_max = max(mw), .groups = "drop")
  cand_new <- bind_rows(
    nd %>% select(oid, US_s1, US_s2, H2A_s1, H2A_s2) %>% pivot_longer(-oid, names_to = "ftype", values_to = "v") %>%
      mutate(ftype = ifelse(grepl("^US", ftype), paste0("us_", tolower(sub("US_", "", ftype))), "h2a_rate")),
    nd %>% select(oid, any_of(c("soc_US_s1", "soc_US_s2", "soc_H2A_s1", "soc_H2A_s2"))) %>%
      pivot_longer(-oid, names_to = "ftype", values_to = "v") %>% mutate(ftype = "soc_rate"),
    mwseg %>% transmute(oid, ftype = "minwage", v = mw),
    nd %>% transmute(oid, ftype = "federal_mw", v = 7.25)) %>% filter(is.finite(v)) %>% distinct()
  cand <- bind_rows(cand_old, cand_new)
  # Applicable floor
  fapp_old <- cand_old %>% group_by(oid) %>% summarise(F_app = max(v), F_min = min(v), .groups = "drop")
  fapp_new <- nd %>% left_join(mw_own, by = "oid") %>%
    mutate(us_k_field = ifelse(skill == "II", US_s2, US_s1),
      us_k_soc = ifelse(skill == "II", soc_US_s2, soc_US_s1),
      us_k = case_when(soc_grp == "nonfield" ~ us_k_soc, TRUE ~ us_k_field),
      F_app = pmax(us_k, coalesce(mw_max, 7.25), 7.25),
      F_s1 = pmax(case_when(soc_grp == "nonfield" ~ soc_US_s1, TRUE ~ US_s1), coalesce(mw_max, 7.25), 7.25)) %>%
    transmute(oid, F_app, F_min = F_s1, mw_max, us_s1_ref = US_s1)
  # Match offers to candidates
  hit <- o %>% select(oid, wage) %>% inner_join(cand, by = "oid", relationship = "many-to-many") %>%
    filter(abs(wage - v) <= TOL) %>% group_by(oid) %>%
    summarise(c_floor = max(v), floor_types = paste(sort(unique(ftype)), collapse = "+"), .groups = "drop")
  oldhit <- o %>% filter(new_rule) %>% select(oid, state_fips, wage) %>% left_join(old, by = "state_fips") %>%
    mutate(at_old = mapply(function(w, r) length(r) > 0 && any(abs(w - r) <= TOL), wage, old_rates)) %>% select(oid, at_old, a25)
  o <- o %>% left_join(bind_rows(fapp_old, fapp_new), by = "oid") %>% left_join(hit, by = "oid") %>% left_join(oldhit, by = "oid") %>%
    mutate(at_old = coalesce(at_old, FALSE),
      floor_unknown = !is.finite(F_app) | (!new_rule & soc_grp == "nonfield" & received >= RULE2023_START),
      cls = case_when(
        floor_unknown ~ "floor_unknown",
        !is.na(c_floor) ~ "at_floor",
        new_rule & at_old ~ "at_old_aewr",
        wage > F_app + TOL ~ "above",
        !new_rule & wage >= F_min - TOL ~ "between_rates",
        new_rule & skill == "II" & wage > F_min + TOL ~ "below_skill2",
        TRUE ~ "below_floor"),
      # Reference floor for ratios: new rule = state U.S. skill-I field rate; old rule = F_app
      F_ref = ifelse(new_rule, us_s1_ref, F_app))
  o
}

# Hours-weighted class shares
f1_shares <- function(d, ...) {
  d %>% mutate(h_ = hours) %>% group_by(...) %>% summarise(orders = n(),
    orders_above = sum(cls == "above"), hours_above = sum(h_[cls == "above"]),
    sh_above = sum(h_[cls == "above"]) / sum(h_), sh_at_floor = sum(h_[cls == "at_floor"]) / sum(h_),
    sh_at_old_aewr = sum(h_[cls == "at_old_aewr"]) / sum(h_), sh_between = sum(h_[cls == "between_rates"]) / sum(h_),
    sh_below_skill2 = sum(h_[cls == "below_skill2"]) / sum(h_), sh_below = sum(h_[cls == "below_floor"]) / sum(h_),
    sh_unknown = sum(h_[cls == "floor_unknown"]) / sum(h_), hours = sum(h_), .groups = "drop")
}
