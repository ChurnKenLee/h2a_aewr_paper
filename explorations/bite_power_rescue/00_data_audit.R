# Strictly outcome-blind metadata & pre-policy coverage audit for R1-R6.
# NEVER reads FY2026 H-2A quantity outcomes.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(readr)
  library(tidyr)
})

cat("=== 1. R1: Statutory 2025 vs 2026 AEWR & Ag Min Wage ===\n")
qp_sched <- read_parquet(
  path_processed("policy_2025", "quantity_county_fy.parquet"),
  col_select = c("county_fips", "state_fips", "aewr_region_id", "year", "x_us_skill_1", "mw_ag", "mw_general")
) %>%
  filter(year %in% c(2025L, 2026L)) %>%
  distinct(state_fips, aewr_region_id, year, x_us_skill_1, mw_ag, mw_general)

r1_st <- qp_sched %>%
  pivot_wider(names_from = year, values_from = c(x_us_skill_1, mw_ag, mw_general)) %>%
  mutate(
    raw_cut = x_us_skill_1_2026 - x_us_skill_1_2025,
    eff_2025 = pmax(x_us_skill_1_2025, mw_ag_2025),
    eff_2026 = pmax(x_us_skill_1_2026, mw_ag_2026),
    eff_cut = eff_2026 - eff_2025,
    trunc_gap_2026 = pmax(0, mw_ag_2026 - x_us_skill_1_2026),
    binds_2025 = mw_ag_2025 > x_us_skill_1_2025,
    binds_2026 = mw_ag_2026 > x_us_skill_1_2026
  )

cat(sprintf(
  "States (46): binds_2025=%d, binds_2026=%d, max trunc_gap_2026=$%.2f\n",
  sum(r1_st$binds_2025), sum(r1_st$binds_2026), max(r1_st$trunc_gap_2026)
))
print(
  r1_st %>%
    filter(binds_2026) %>%
    select(state_fips, aewr_region_id, x_us_skill_1_2025, x_us_skill_1_2026, mw_ag_2026, raw_cut, eff_cut, trunc_gap_2026),
  n = 30
)

sub_mw <- read_csv("explorations/identification_search/A1/sources/state_ag_minwage_fy2025_2026.csv", show_col_types = FALSE)
sub_areas <- sub_mw %>%
  filter(area_id != "statewide") %>%
  distinct(state_abbrev, state_fips, area_id, snapshot_date, h2a_state_floor)
cat("\nSub-state ag min wage rows in cached A1 source:\n")
print(sub_areas, n = 40)

cat("\n=== 2. R2: H-2ALC flag, CDL years, NOAA climate years ===\n")
disc_h2alc <- open_dataset(path_int("h2a_disclosure_records.parquet")) %>%
  select(fiscal_year, h2a_labor_contractor) %>%
  filter(fiscal_year >= 2008L, fiscal_year <= 2025L) %>%
  collect() %>%
  mutate(
    fy = as.integer(fiscal_year),
    h_up = toupper(trimws(coalesce(as.character(h2a_labor_contractor), ""))),
    is_y = h_up %in% c("Y", "YES", "TRUE", "1"),
    is_n = h_up %in% c("N", "NO", "FALSE", "0")
  ) %>%
  group_by(fy) %>%
  summarise(n = n(), n_y = sum(is_y), n_n = sum(is_n), n_na = sum(!is_y & !is_n), .groups = "drop")
cat("h2a_labor_contractor by FY (2008-2025):\n")
print(disc_h2alc, n = 20)

cdl_yrs <- open_dataset(path_int("croplandcros_county_crop_type_acres.parquet")) %>%
  select(year, crop_type_label, county_fips) %>%
  collect()
cat(sprintf(
  "CDL years: %s | labels: %s\n",
  paste(sort(unique(cdl_yrs$year)), collapse = ","),
  paste(sort(unique(cdl_yrs$crop_type_label)), collapse = ", ")
))
cdl_cov <- cdl_yrs %>% group_by(year) %>% summarise(n_cty = n_distinct(county_fips), .groups = "drop")
print(cdl_cov, n = 25)

noaa_yrs <- open_dataset(path_int("county_h2a_prediction_climate_basis_annual.parquet")) %>%
  select(year, county_fips) %>%
  collect() %>%
  group_by(year) %>%
  summarise(n_cty = n_distinct(county_fips), .groups = "drop")
cat("NOAA climate basis years (2008+):\n")
print(noaa_yrs %>% filter(year >= 2008L), n = 25)

cat("\n=== 3. R3: Post-FY2020 Worksite Business & Multi-State R7 Filers (Pre-FY2026) ===\n")
ws_pre <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, case_number, employer_is_worksite_business, employer_id_balanced, adjusted_nbr_workers_certified) %>%
  filter(fiscal_year >= 2020L, fiscal_year <= 2025L) %>%
  collect() %>%
  group_by(fiscal_year) %>%
  summarise(
    rows = n(),
    wb_rows = sum(!is.na(employer_is_worksite_business) & as.logical(employer_is_worksite_business)),
    n_wb = n_distinct(employer_id_balanced[!is.na(employer_is_worksite_business) & as.logical(employer_is_worksite_business)]),
    .groups = "drop"
  )
print(ws_pre)

wfso <- read_parquet(path_processed("policy_2025", "wage_filer_state_occupation.parquet")) %>%
  filter(tier == "R7", fiscal_year <= 2025L) %>%
  group_by(fiscal_year, filer) %>%
  summarise(n_states = n_distinct(state_fips), h = sum(h), .groups = "drop") %>%
  group_by(fiscal_year) %>%
  summarise(
    n_filers = n(),
    n_multistate = sum(n_states > 1L),
    hours_share_multistate = sum(h[n_states > 1L]) / sum(h),
    .groups = "drop"
  )
cat("R7 filers in wage_filer_state_occupation (FY2023-FY2025):\n")
print(wfso)

cat("\n=== 4. R4 & R5: Baseline Duration, Farm Labor Share, Local Wage Crossover ===\n")
cp <- read_parquet(
  path_processed("county_year_panel.parquet"),
  col_select = c(
    "county_fips", "state_fips", "aewr_region_id", "year",
    "aewr", "oews_big_six_mean_hourly_wage", "wage_p25", "ppi_2012",
    "share_farm_laborexp_prodexp", "h2a_cert_hours_per_position_start_year",
    "man_hours_certified_start_year", "nbr_workers_certified_start_year"
  )
) %>% mutate(year = as.integer(year))

cat("share_farm_laborexp_prodexp finite count by year:\n")
print(cp %>% group_by(year) %>% summarise(
  n_labsh = sum(is.finite(share_farm_laborexp_prodexp)),
  n_oews = sum(is.finite(oews_big_six_mean_hourly_wage)),
  n_p25 = sum(is.finite(wage_p25)),
  .groups = "drop"
), n = 20)

qwi <- read_parquet(path_int("qwi_county_ag_quarterly_earnings.parquet"))
cat("QWI years & industries:\n")
print(qwi %>% group_by(year, industry_code) %>% summarise(
  n_obs = n(),
  n_finite_earn = sum(is.finite(qwi_stable_monthly_earnings)),
  n_cty = n_distinct(county_fips[is.finite(qwi_stable_monthly_earnings)]),
  .groups = "drop"
) %>% filter(year >= 2021L), n = 30)

# Check 2024 local wage vs AEWR_2025 and AEWR_2026 crossover rates
cp24 <- cp %>%
  filter(year == 2024L) %>%
  mutate(
    wage_p25_nom = wage_p25 * ppi_2012
  ) %>%
  inner_join(r1_st %>% select(state_fips, x_us_skill_1_2025, x_us_skill_1_2026, mw_ag_2026), by = "state_fips")

qwi24 <- qwi %>%
  filter(year == 2024L, industry_code == "111", is.finite(qwi_stable_monthly_earnings), qwi_stable_monthly_earnings > 0) %>%
  group_by(county_fips) %>%
  summarise(qwi111_hourly_approx = mean(qwi_stable_monthly_earnings, na.rm = TRUE) / (4.333 * 40), .groups = "drop")

cp24 <- cp24 %>% left_join(qwi24, by = "county_fips")
cat("\n2024 Local Wage Crossover vs Oct 2, 2025 AEWR cut (46 states, 2,921+ counties):\n")
cat(sprintf(
  "OEWS Big-Six (n=%d): >AEWR26=%.1f%%, crossover(AEWR25>w>AEWR26)=%.1f%%, >eff26=%.1f%%\n",
  sum(is.finite(cp24$oews_big_six_mean_hourly_wage)),
  100 * mean(cp24$oews_big_six_mean_hourly_wage > cp24$x_us_skill_1_2026, na.rm = TRUE),
  100 * mean(cp24$x_us_skill_1_2025 > cp24$oews_big_six_mean_hourly_wage & cp24$oews_big_six_mean_hourly_wage > cp24$x_us_skill_1_2026, na.rm = TRUE),
  100 * mean(cp24$oews_big_six_mean_hourly_wage > pmax(cp24$x_us_skill_1_2026, cp24$mw_ag_2026), na.rm = TRUE)
))
cat(sprintf(
  "ACS p25 nom (n=%d): >AEWR26=%.1f%%, crossover(AEWR25>w>AEWR26)=%.1f%%, >eff26=%.1f%%\n",
  sum(is.finite(cp24$wage_p25_nom)),
  100 * mean(cp24$wage_p25_nom > cp24$x_us_skill_1_2026, na.rm = TRUE),
  100 * mean(cp24$x_us_skill_1_2025 > cp24$wage_p25_nom & cp24$wage_p25_nom > cp24$x_us_skill_1_2026, na.rm = TRUE),
  100 * mean(cp24$wage_p25_nom > pmax(cp24$x_us_skill_1_2026, cp24$mw_ag_2026), na.rm = TRUE)
))
cat(sprintf(
  "QWI 111 hourly (n=%d): >AEWR26=%.1f%%, crossover(AEWR25>w>AEWR26)=%.1f%%, >eff26=%.1f%%\n",
  sum(is.finite(cp24$qwi111_hourly_approx)),
  100 * mean(cp24$qwi111_hourly_approx > cp24$x_us_skill_1_2026, na.rm = TRUE),
  100 * mean(cp24$x_us_skill_1_2025 > cp24$qwi111_hourly_approx & cp24$qwi111_hourly_approx > cp24$x_us_skill_1_2026, na.rm = TRUE),
  100 * mean(cp24$qwi111_hourly_approx > pmax(cp24$x_us_skill_1_2026, cp24$mw_ag_2026), na.rm = TRUE)
))

cat("\n=== 5. Check R6 cached files ===\n")
cat("aewr_2009_oes_schedule exists:", file.exists("explorations/identification_search/sources/aewr_2009_oes_schedule.csv"), "\n")
cat("oes_area_county_crosswalk_2009 exists:", file.exists("explorations/identification_search/sources/oes_area_county_crosswalk_2009.csv"), "\n")
cat("h2a_disclosures_fy2006_2007 exists:", file.exists("explorations/identification_search/sources/h2a_disclosures_fy2006_2007.csv"), "\n")
