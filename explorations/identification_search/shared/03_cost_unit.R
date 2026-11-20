# Identification search, Phase 0, step 2: the common cost unit.
#
# H-2A cost per hour C = binding wage floor w + housing per hour h
#                        + transportation per hour t + fees per hour f.
# Writes the national FY series of C / AEWR (low, central, high, extreme) and
# a state x FY cost-per-hour series to the scratch directory. Sources are in
# explorations/identification_search/shared/sources/cost_unit_sources.csv.
#
# Usage (after 01_outcomes.R):
#   Rscript --vanilla explorations/identification_search/shared/03_cost_unit.R "$S/shared"
#
# Declared imputations (also in declarations.json):
# - h: DOL's own valuation of employer-provided housing (90 FR 47914, 2025
#   IFR). National h/AEWR ratio from RIA Exhibit 6 for FY2014-FY2024, held at
#   the FY2014 ratio before FY2014 and at the FY2024 ratio in FY2025; FY2026
#   uses the state adjustments in force (US minus H2A skill-I statewide rate).
#   High: DOL's 0-bedroom alternative ($3.54 vs AEWR $16.66 in 2024).
# - t: inbound + outbound = 2 x $400-650 per worker (USDA), per in-data
#   certified contract hour per position of the same FY. Nominal, all years.
# - f: consular $190 + border $6 + DOL/DHS $15.60 + recruitment $100-250 per
#   worker + agent $1,500-3,500 per application / positions per application.
# - Extreme: $10,000 minimum non-wage cost per worker (Castillo, Martin and
#   Rutledge 2024) per contract hour.
# - w: the reference AEWR in effect on April 1 (hours-weighted across states);
#   FY2026 uses the statewide skill-I rates (US for U.S.-worker comparison,
#   H2A cash rate for H-2A workers). Prevailing, minimum and CBA wages are
#   not modelled in the series; job-order strategies use the observed hourly
#   offer as w.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(tidyr); library(readr)})
out_dir <- normalizePath(commandArgs(trailingOnly = TRUE)[1])
if (startsWith(paste0(out_dir, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
src <- read_csv("explorations/identification_search/shared/sources/cost_unit_sources.csv", show_col_types = FALSE)
val <- function(id, col = "value_central") src[[col]][src$component_id == id]

# DOL IFR RIA Exhibit 6 (90 FR 47914): hourly housing and baseline national AEWR, $2025.
exhibit6 <- tibble(year = 2014:2024,
  housing = c(1.07, 1.12, 1.17, 1.24, 1.29, 1.35, 1.43, 1.48, 1.54, 1.70, 1.89),
  aewr_exhibit = c(10.54, 10.83, 11.32, 11.73, 11.99, 12.58, 13.25, 13.79, 14.63, 15.81, 16.66)) %>%
  mutate(h_ratio = housing / aewr_exhibit)
stopifnot(abs(exhibit6$housing[exhibit6$year == 2024] - val("housing_dol_ifr_2014_2024")) < 1e-9)
alt_ratio_2024 <- val("housing_dol_alt_0br") / exhibit6$aewr_exhibit[exhibit6$year == 2024]

# In-data contract hours per certified position and positions per application.
county <- read_parquet(file.path(out_dir, "outcomes_county_fy.parquet"),
  col_select = c("county_fips", "state_fips", "year", "cert_pos_fy", "cert_hours_fy", "apps_fy", "cert_pos_dq13", "cert_hours_dq13", "apps_dq13"))
use_fy <- function(d) d %>% mutate(cp = ifelse(year == 2026, cert_pos_dq13, cert_pos_fy), ch = ifelse(year == 2026, cert_hours_dq13, cert_hours_fy),
  ap = ifelse(year == 2026, apps_dq13, apps_fy))
state_fy <- county %>% use_fy() %>% group_by(state_fips, year) %>% summarise(cp = sum(cp), ch = sum(ch), ap = sum(ap), .groups = "drop")
nat <- state_fy %>% group_by(year) %>% summarise(cert_pos = sum(cp), cert_hours = sum(ch), apps = sum(ap), .groups = "drop") %>%
  mutate(hours_per_position = cert_hours / cert_pos, positions_per_application = cert_pos / apps)

# Wage floor: reference AEWR on April 1 of each FY; FY2026 statewide skill I.
ref <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>%
  filter(occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference"), worker_category == "covered_workers") %>%
  transmute(state_fips, aewr = as.numeric(aewr), start = as.Date(effective_start), end = as.Date(effective_end_exclusive))
sw <- read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
  filter(trimws(source_sheet) == "Statewide 655.120(b)(1)(i)", skill_level == 1) %>%
  transmute(state_fips, worker_category, aewr = as.numeric(hourly_aewr), start = as.Date(effective_start), end = as.Date(effective_end_exclusive))
rate_on <- function(s, date) s %>% filter(start <= date, is.na(end) | end > date)
sw26 <- rate_on(sw, as.Date("2026-04-01")) %>% select(state_fips, worker_category, aewr) %>%
  pivot_wider(names_from = worker_category, values_from = aewr) %>% rename(aewr_us = US, aewr_h2a = H2A) %>%
  mutate(housing_adj_2026 = aewr_us - aewr_h2a)
stopifnot(!anyDuplicated(sw26$state_fips), all(sw26$housing_adj_2026 >= 0, na.rm = TRUE))
floor_sf <- bind_rows(lapply(2008:2025, function(fy) rate_on(ref, as.Date(sprintf("%d-04-01", fy))) %>%
  transmute(state_fips, year = fy, aewr_us = aewr, aewr_h2a = aewr))) %>%
  bind_rows(sw26 %>% transmute(state_fips, year = 2026L, aewr_us, aewr_h2a))
# FY2009 April 1 falls in the OES period (no reference rate): flagged NA.
sfy <- state_fy %>% left_join(floor_sf, by = c("state_fips", "year"))
nat_aewr <- sfy %>% filter(is.finite(aewr_us), ch > 0) %>% group_by(year) %>%
  summarise(aewr_us = sum(aewr_us * ch) / sum(ch), aewr_h2a = sum(aewr_h2a * ch) / sum(ch),
    hours_with_rate_share = sum(ch) / sum(sfy$ch[sfy$year == first(year)]), .groups = "drop")
h26 <- sfy %>% filter(year == 2026) %>% inner_join(sw26 %>% select(state_fips, housing_adj_2026), by = "state_fips") %>%
  summarise(h = sum(housing_adj_2026 * ch) / sum(ch), lo = min(housing_adj_2026), hi = max(housing_adj_2026))

per_worker <- function(level) {
  col <- c(low = "value_low", central = "value_central", high = "value_high")[[level]]
  list(transport = 2 * val("transport_inbound", col),
    fees_worker = val("fee_consular", col) + val("fee_border", col) + val("fee_dol_dhs", col) + val("fee_recruitment", col),
    agent_app = val("fee_agent", col))
}
series <- nat %>% left_join(nat_aewr, by = "year") %>% left_join(exhibit6 %>% select(year, h_ratio), by = "year") %>%
  mutate(h_ratio = case_when(year < 2014 ~ exhibit6$h_ratio[1], year == 2025 ~ exhibit6$h_ratio[nrow(exhibit6)], TRUE ~ h_ratio),
    w = aewr_h2a, housing_central = ifelse(year == 2026, h26$h, h_ratio * w), housing_low = housing_central,
    housing_high = ifelse(year == 2026, h26$h * alt_ratio_2024 / exhibit6$h_ratio[nrow(exhibit6)], alt_ratio_2024 * w))
for (lv in c("low", "central", "high")) {
  pw <- per_worker(lv)
  series[[paste0("transport_", lv)]] <- pw$transport / series$hours_per_position
  series[[paste0("fees_", lv)]] <- (pw$fees_worker + pw$agent_app / series$positions_per_application) / series$hours_per_position
  series[[paste0("nonwage_", lv)]] <- series[[paste0("housing_", lv)]] + series[[paste0("transport_", lv)]] + series[[paste0("fees_", lv)]]
  series[[paste0("cost_ratio_", lv)]] <- 1 + series[[paste0("nonwage_", lv)]] / series$w
}
# w is the H-2A cash floor (= the AEWR through FY2025; the H2A skill-I cash
# rate in FY2026, which already nets out the housing adjustment), so C = w + h + t + f
# in every year. In FY2026, C / (U.S. skill-I rate) is also reported.
series <- series %>% mutate(nonwage_extreme = val("nonwage_total_cmr") / hours_per_position, cost_ratio_extreme = 1 + nonwage_extreme / w,
  cost_central = w + nonwage_central, cost_over_us_rate_central = cost_central / aewr_us)
write_csv(series, file.path(out_dir, "cost_unit_by_fy.csv"))

# Pooled declared values: certified-hours-weighted over FY2008-FY2025 (2009 has no April-1 rate).
pooled <- series %>% filter(year <= 2025, is.finite(w)) %>%
  summarise(across(c(cost_ratio_low, cost_ratio_central, cost_ratio_high, cost_ratio_extreme,
    housing_central, transport_central, fees_central, nonwage_central, w, hours_per_position), ~ sum(.x * cert_hours) / sum(cert_hours)))
recent <- series %>% filter(year %in% 2020:2025) %>%
  summarise(across(c(cost_ratio_low, cost_ratio_central, cost_ratio_high, cost_ratio_extreme), ~ sum(.x * cert_hours) / sum(cert_hours)))
summary_tbl <- bind_rows(pooled %>% mutate(scope = "FY2008-FY2025 hours-weighted"), recent %>% mutate(scope = "FY2020-FY2025 hours-weighted"),
  series %>% filter(year == 2024) %>% select(starts_with("cost_ratio_"), housing_central, transport_central, fees_central, nonwage_central, w, hours_per_position) %>% mutate(scope = "FY2024"),
  series %>% filter(year == 2026) %>% select(starts_with("cost_ratio_"), housing_central, transport_central, fees_central, nonwage_central, w, hours_per_position) %>% mutate(scope = "FY2026 (Q1-Q3)")) %>%
  mutate(fy2026_state_housing_adj_min = h26$lo, fy2026_state_housing_adj_max = h26$hi) %>% relocate(scope)
write_csv(summary_tbl, file.path(out_dir, "cost_unit_summary.csv"))

# State x FY cost per hour (central). State housing: FY2026 state adjustment
# scaled by the national housing path (housing_central_t / housing_central_2026).
nat_scale <- series %>% transmute(year, h_scale = housing_central / h26$h)
state_cost <- sfy %>% left_join(sw26 %>% select(state_fips, housing_adj_2026), by = "state_fips") %>%
  left_join(nat_scale, by = "year") %>% left_join(nat %>% select(year, positions_per_application), by = "year") %>%
  mutate(hours_per_position = ifelse(cp > 0, ch / cp, NA_real_),
    housing_per_hour = housing_adj_2026 * h_scale,
    transport_per_hour = per_worker("central")$transport / hours_per_position,
    fees_per_hour = (per_worker("central")$fees_worker + per_worker("central")$agent_app / positions_per_application) / hours_per_position,
    cost_per_hour_central = aewr_h2a + housing_per_hour + transport_per_hour + fees_per_hour,
    cost_ratio_central = cost_per_hour_central / aewr_us) %>%
  select(state_fips, year, cert_pos = cp, cert_hours = ch, aewr_us, aewr_h2a, housing_per_hour, transport_per_hour, fees_per_hour,
    hours_per_position, cost_per_hour_central, cost_ratio_central)
stopifnot(!anyDuplicated(state_cost[c("state_fips", "year")]), is.character(state_cost$state_fips))
write_parquet(state_cost, file.path(out_dir, "cost_per_hour_state_fy.parquet"))
print(as.data.frame(series %>% select(year, w, hours_per_position, positions_per_application, housing_central, transport_central, fees_central,
  cost_ratio_low, cost_ratio_central, cost_ratio_high, cost_ratio_extreme)), digits = 4)
print(as.data.frame(summary_tbl), digits = 4)
