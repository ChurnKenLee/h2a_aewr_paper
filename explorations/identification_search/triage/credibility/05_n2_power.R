# Triage (credibility-first): rough design-stage power for new strategy N2
# (competing state minimum-wage floors under the 2025 rule) next to A1 doses,
# on the Track 1 county frame (multi-state former regions), PPML-linearized
# certified hours (window dq13), FY2020-FY2025 calibration only. No dose
# regression on any outcome. Statutory-ITT units: beta_e1 = 1 / mean dose price.
#   Rscript --vanilla explorations/identification_search/triage/credibility/05_n2_power.R "$S/triage"
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(tidyr); library(readr)})
source(path_code("designs", "border_discontinuity", "power.R"))
source("explorations/identification_search/shared/power_harness.R")
setFixest_nthreads(1)
out <- commandArgs(trailingOnly = TRUE)[1]; S <- dirname(out)
registry <- jsonlite::read_json("explorations/identification/track1_registry.json", simplifyVector = TRUE)
settings <- ish_power_settings()

cp <- read_parquet(path_processed("county_year_panel.parquet"),
  col_select = c("county_fips", "state_fips", "aewr_region_id", "emp_farm_2011", "year")) %>% filter(year == 2011) %>% select(-year)
counties <- cp %>% filter(!state_fips %in% registry$geography$excluded_states, is.finite(emp_farm_2011), emp_farm_2011 > 0) %>%
  group_by(aewr_region_id) %>% filter(n_distinct(state_fips) > 1) %>% ungroup() %>% arrange(aewr_region_id, state_fips, county_fips)
st <- read_csv(file.path(out, "cred_state_dose_table.csv"), col_types = cols(state_fips = "c")) %>%
  filter(state_fips %in% counties$state_fips) %>%
  mutate(mw_fy26 = mw2025 / 3 + 2 * mw2026 / 3,
    us_s1 = US_s1, h2a_s1 = H2A_s1, eff_floor = pmax(H2A_s1, mw_fy26), trunc_fy26 = pmax(0, mw_fy26 - H2A_s1))
# truncation residualized on the statutory H-2A cut within former region (FWL at state level)
rr <- st %>% left_join(counties %>% distinct(state_fips, aewr_region_id), by = "state_fips")
rr$trunc_resid <- resid(lm(trunc_fy26 ~ stat_cut_h2a + factor(aewr_region_id), data = rr))
st <- st %>% left_join(rr %>% select(state_fips, trunc_resid), by = "state_fips")
regional <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>%
  filter(occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference"), worker_category == "covered_workers") %>%
  transmute(state_fips, aewr = as.numeric(aewr), start = as.Date(effective_start), end = as.Date(effective_end_exclusive))
rate_on <- function(s, date) s %>% filter(start <= date, is.na(end) | end > date) %>% select(state_fips, aewr)
treat <- bind_rows(lapply(2012:2026, function(fy) tibble(state_fips = unique(counties$state_fips), year = fy) %>%
  left_join(rate_on(regional, as.Date(sprintf("%d-04-01", min(fy, 2025)))), by = "state_fips"))) %>%
  left_join(st %>% select(state_fips, us_s1, h2a_s1, eff_floor, trunc_resid), by = "state_fips") %>%
  mutate(x_us = ifelse(year == 2026, us_s1, aewr), x_h2a = ifelse(year == 2026, h2a_s1, aewr),
    x_eff = ifelse(year == 2026, eff_floor, aewr), x_trunc = ifelse(year == 2026, trunc_resid, 0))
policy <- tidyr::crossing(counties, year = 2012:2026) %>% left_join(treat, by = c("state_fips", "year")) %>%
  mutate(region_year = paste(aewr_region_id, year))
stopifnot(all(is.finite(policy$x_eff)))
oc <- read_parquet(file.path(S, "shared", "outcomes_county_fy.parquet"), col_select = c("county_fips", "year", "cert_hours_dq13")) %>%
  filter(year >= 2020, year <= 2025)
pre <- tidyr::crossing(counties, year = 2020:2025) %>% left_join(oc, by = c("county_fips", "year")) %>%
  mutate(cert_hours_dq13 = coalesce(cert_hours_dq13, 0), region_year = paste(aewr_region_id, year))
lin <- ish_ppml_linearize(pre, policy, "cert_hours_dq13", "county_fips")
res <- list()
for (xv in c("x_us", "x_h2a", "x_eff", "x_trunc")) {
  pol <- lin$policy
  mean_p <- if (xv == "x_trunc") sum(pol$ppml_weight * pol$x_eff) / sum(pol$ppml_weight) else sum(pol$ppml_weight * pol[[xv]]) / sum(pol$ppml_weight)
  message(xv)
  r <- ish_power(pol, lin$pre, y = lin$y, x = xv, unit = "county_fips", period = "year", fe = c("county_fips", "region_year"),
    clusters = c("state_fips", "aewr_region_id"), jk_cluster = "state_fips", node = "state_fips", group = "aewr_region_id",
    weights = lin$weights, beta_e1 = 1 / mean_p, dgps = c("county_serial", "shared_region_serial"), settings = settings,
    label = list(dose = xv, mean_price = mean_p))
  res[[xv]] <- r
}
write_csv(bind_rows(lapply(res, `[[`, "support")), file.path(out, "n2_power_support.csv"))
write_csv(bind_rows(lapply(res, `[[`, "summary")), file.path(out, "n2_power_summary.csv"))
write_csv(bind_rows(lapply(res, `[[`, "mde")), file.path(out, "n2_power_mde.csv"))
print(bind_rows(lapply(res, `[[`, "support")) %>% select(dose, mean_price, dose_sd_after_fe, effective_clusters, max_leverage_share), width = 200)
print(bind_rows(lapply(res, `[[`, "summary")), width = 200)
