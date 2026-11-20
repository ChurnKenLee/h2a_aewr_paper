# Track 1 support (§4c): identifying variation of the 2025 statewide rule
# after county and former-region x FY fixed effects. Reads only rate
# schedules, the shared panel's fixed denominators and pre-FY2026 case
# records (decision-date coverage). No FY2026 outcome, wage or count is read.
source("explorations/identification/id_helpers.R")
out_dir <- id_output_dir()
EXCLUDED <- c("02", "15", "11")  # as statewide_design.json: AK, HI, DC

cp <- read_parquet(path_processed("county_year_panel.parquet"),
  col_select = c("county_fips", "state_fips", "aewr_region_id", "emp_farm_2011", "year")) %>%
  filter(year == 2011) %>% select(-year)
bite_check_keys(cp %>% mutate(year = 2011L))
sr <- cp %>% distinct(state_fips, aewr_region_id)
stopifnot(!anyDuplicated(sr$state_fips))

sw <- read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
  filter(trimws(source_sheet) == "Statewide 655.120(b)(1)(i)") %>%
  transmute(state_fips, worker_category, skill_level = as.character(skill_level),
    rate = as.numeric(hourly_aewr), start = as.Date(effective_start))
reg <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>%
  filter(occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference"), worker_category == "covered_workers") %>%
  transmute(state_fips, aewr = as.numeric(aewr), start = as.Date(effective_start), end = as.Date(effective_end_exclusive))
on <- function(date) reg %>% filter(start <= date, is.na(end) | end > date) %>% select(state_fips, aewr)
fy25 <- on(as.Date("2025-04-01")) %>% rename(regional_fy2025 = aewr)
stopifnot(!anyDuplicated(fy25$state_fips))

variants <- tibble(label = c("us_skill_1", "h2a_skill_1", "us_skill_2"),
  worker_category = c("US", "H2A", "US"), skill_level = c("1", "1", "2"))
states <- sr %>% filter(!state_fips %in% EXCLUDED) %>% inner_join(fy25, by = "state_fips")
dose <- bind_rows(lapply(seq_len(nrow(variants)), function(i) {
  v <- variants[i, ]
  bind_rows(lapply(c(fy2026 = "2025-10-02", fy2027 = "2026-08-03"), function(st) {
    sw %>% filter(worker_category == v$worker_category, skill_level == v$skill_level, start == as.Date(st)) %>%
      select(state_fips, rate) %>% mutate(schedule_start = st)
  })) %>% mutate(label = v$label)
})) %>% inner_join(states, by = "state_fips") %>%
  mutate(dose = rate - regional_fy2025)
stopifnot(!anyDuplicated(dose[c("state_fips", "label", "schedule_start")]))

region_sizes <- states %>% count(aewr_region_id, name = "states_in_region")
county_counts <- cp %>% filter(!state_fips %in% EXCLUDED, is.finite(emp_farm_2011), emp_farm_2011 > 0) %>%
  count(state_fips, name = "counties")
dose <- dose %>% left_join(region_sizes, by = "aewr_region_id") %>% left_join(county_counts, by = "state_fips") %>%
  group_by(label, schedule_start, aewr_region_id) %>%
  mutate(within_region_dev = rate - mean(rate), county_weighted_dev = rate - weighted.mean(rate, counties)) %>% ungroup()

support <- dose %>% group_by(label, schedule_start) %>% summarise(
  states = n(), regions = n_distinct(aewr_region_id),
  single_state_regions = paste(sort(unique(aewr_region_id[states_in_region == 1])), collapse = ";"),
  contributing_states = sum(states_in_region > 1), contributing_regions = n_distinct(aewr_region_id[states_in_region > 1]),
  contributing_counties = sum(counties[states_in_region > 1], na.rm = TRUE),
  mean_dose = mean(dose), sd_dose = sd(dose), min_dose = min(dose), max_dose = max(dose),
  within_region_sd_state = sqrt(sum(within_region_dev[states_in_region > 1]^2) / (sum(states_in_region > 1) - contributing_regions)),
  within_region_sd_county_weighted = sqrt(sum(counties * county_weighted_dev^2, na.rm = TRUE) / sum(counties[states_in_region > 1], na.rm = TRUE)),
  within_region_range_max = max(tapply(rate, aewr_region_id, function(z) diff(range(z)))),
  .groups = "drop")
by_region <- dose %>% filter(label == "us_skill_1", schedule_start == "2025-10-02") %>%
  group_by(aewr_region_id) %>% summarise(states = n(), counties = sum(counties, na.rm = TRUE),
    regional_fy2025 = first(regional_fy2025), min_rate = min(rate), max_rate = max(rate),
    sd_rate = if (n() > 1) sd(rate) else NA_real_, state_list = paste(state_fips, collapse = ";"), .groups = "drop")

# Decision-date coverage of pre-period cases (FY2012-FY2025) for the Q1-Q3 outcome.
cases <- id_read_disclosures(c("case_number", "decision_date", "case_received_date")) %>%
  filter(fiscal_year >= 2012) %>%
  group_by(case_number, fiscal_year) %>%
  summarise(decision_date = suppressWarnings(as.Date(first(na.omit(ifelse(nzchar(decision_date), decision_date, NA))))),
    received = suppressWarnings(as.Date(first(na.omit(ifelse(nzchar(case_received_date), case_received_date, NA))))), .groups = "drop") %>%
  mutate(fy_start = as.Date(sprintf("%d-10-01", fiscal_year - 1L)), fy_end = as.Date(sprintf("%d-09-30", fiscal_year)),
    decision_in_fy = !is.na(decision_date) & decision_date >= fy_start & decision_date <= fy_end,
    q1_q3 = decision_in_fy & decision_date <= as.Date(sprintf("%d-06-30", fiscal_year)))
coverage <- cases %>% group_by(fiscal_year) %>% summarise(cases = n(),
  decision_date_present = mean(!is.na(decision_date)), decision_within_fy = mean(decision_in_fy),
  share_q1_q3_of_dated = sum(q1_q3) / sum(decision_in_fy), .groups = "drop")

id_write(dose, out_dir, "track1_dose_by_state.csv")
id_write(support, out_dir, "track1_support.csv")
id_write(by_region, out_dir, "track1_support_by_region.csv")
id_write(coverage, out_dir, "track1_decision_date_coverage.csv")
print(as.data.frame(support), digits = 3); print(as.data.frame(by_region), digits = 3); print(as.data.frame(coverage), digits = 3)
