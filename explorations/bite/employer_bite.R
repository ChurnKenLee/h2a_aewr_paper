# Scratch diagnostic: design 4, employer-level AEWR bindingness.
# Cohort: v2 balanced worksite businesses with hourly certified hours in start
# years 2011-2012 and a single baseline AEWR region. Bound = at least half of
# baseline hourly certified hours offered at the AEWR (b01 rule: within $0.05
# of the start-year or prior-year state AEWR). Writes only to argument 1.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressMessages({library(arrow); library(dplyr); library(tidyr); library(fixest); library(readr); library(lubridate)})
out_dir <- commandArgs(trailingOnly = TRUE)[1]
TOL <- 0.05

rates <- read_parquet(path_int("aewr.parquet")) %>%
  transmute(state = state_fips(state_fips), year = as.integer(year), aewr = as.numeric(aewr)) %>%
  filter(year <= 2022L) %>% distinct() %>% arrange(state, year) %>%
  group_by(state) %>% mutate(aewr_lag = lag(aewr)) %>% ungroup()
cp <- read_parquet(path_processed("county_year_panel.parquet"),
  col_select = c("state_fips", "year", "aewr_region_id", "aewr", "ppi_2012"))
state_region <- cp %>% distinct(state = state_fips, aewr_region_id) %>%
  mutate(aewr_region_id = as.character(aewr_region_id))
stopifnot(!anyDuplicated(state_region$state))
region_aewr <- cp %>% mutate(aewr_region_id = as.character(aewr_region_id)) %>%
  group_by(aewr_region_id, year) %>% summarise(aewr_real = mean(aewr / ppi_2012), .groups = "drop")

w <- read_parquet(path_int("h2a_allocated_worksites.parquet"),
  col_select = c("employer_id_balanced", "employer_is_worksite_business", "adjusted_nbr_workers_certified",
    "period_begin_date", "period_end_date", "number_of_hours", "wage_rate", "wage_unit", "county_fips_list")) %>%
  filter(employer_is_worksite_business, !is.na(employer_id_balanced), employer_id_balanced != "") %>%
  mutate(begin = as.Date(period_begin_date), end = as.Date(period_end_date), year = year(begin),
    workers = coalesce(as.numeric(adjusted_nbr_workers_certified), 0),
    hours = pmax(as.numeric(end - begin), 0) / 7 * as.numeric(number_of_hours) * workers,
    wage = suppressWarnings(as.numeric(wage_rate)),
    hourly = toupper(trimws(wage_unit)) %in% c("HOUR", "HOURLY", "HR") & is.finite(wage) & wage > 0 & wage <= 100,
    state = substr(trimws(sub(",.*", "", county_fips_list)), 1, 2)) %>%
  filter(year >= 2008, year <= 2024)

# Baseline classification ------------------------------------------------------
base <- w %>% filter(year %in% 2011:2012, workers > 0, hourly, is.finite(hours), hours > 0) %>%
  left_join(rates, by = c("state", "year")) %>%
  mutate(at = (is.finite(aewr) & abs(wage - aewr) <= TOL) | (is.finite(aewr_lag) & abs(wage - aewr_lag) <= TOL),
    rel = wage / aewr) %>%
  left_join(state_region, by = "state")
cohort <- base %>% group_by(employer_id_balanced) %>%
  summarise(state = names(which.max(tapply(hours, state, sum))),
    share_at = sum(hours * at) / sum(hours),
    rel_wage = sum(hours * rel, na.rm = TRUE) / sum(hours * is.finite(rel)),
    regions = n_distinct(aewr_region_id), region = first(aewr_region_id),
    workers = sum(workers), hours = sum(hours), .groups = "drop") %>%
  filter(regions == 1, !is.na(region)) %>%
  mutate(bound = as.integer(share_at >= 0.5))
cat("Cohort employers:", nrow(cohort), " bound:", sum(cohort$bound), " unbound:", sum(1 - cohort$bound),
  " share of baseline hours unbound:", round(sum(cohort$hours[cohort$bound == 0]) / sum(cohort$hours), 3), "\n")
print(cohort %>% group_by(bound) %>% summarise(n = n(), median_workers = median(workers),
  median_rel_wage = median(rel_wage, na.rm = TRUE), p90_rel = quantile(rel_wage, .9, na.rm = TRUE)))
print(cohort %>% filter(bound == 0) %>% count(state, sort = TRUE) %>% head(8))
cat("Unbound share by region:\n"); print(cohort %>% group_by(region) %>%
  summarise(n = n(), unbound = round(mean(1 - bound), 2)) %>% arrange(desc(unbound)), n = 17)

# Employer-year panel ------------------------------------------------------------
ey <- w %>% semi_join(cohort, by = "employer_id_balanced") %>%
  group_by(employer_id_balanced, year) %>%
  summarise(workers = sum(workers), hours = sum(hours, na.rm = TRUE), .groups = "drop")
panel <- cohort %>% select(employer_id_balanced, bound, share_at, region, state) %>%
  crossing(year = 2008:2024) %>%
  left_join(ey, by = c("employer_id_balanced", "year")) %>%
  mutate(workers = coalesce(workers, 0), hours = coalesce(hours, 0),
    any = as.integer(workers > 0), lworkers = log1p(workers), unbound_share = 1 - share_at) %>%
  left_join(region_aewr, by = c("region" = "aewr_region_id", "year"))
write_parquet(panel, file.path(out_dir, "employer_bite_panel.parquet"))
print(panel %>% group_by(year, bound) %>% summarise(any = round(mean(any), 3), .groups = "drop") %>%
  pivot_wider(names_from = bound, values_from = any, names_prefix = "any_bound"), n = 20)

# Estimates ---------------------------------------------------------------------
rows <- list(); es_rows <- list()
for (y in c("any", "lworkers")) {
  dd <- panel %>% filter(year >= 2011, is.finite(aewr_real)) %>% mutate(X = bound * aewr_real)
  b <- feols(as.formula(paste(y, "~ X | employer_id_balanced + region^year")), dd, vcov = ~state)
  a <- feols(as.formula(paste(y, "~ X | employer_id_balanced + region^year + year[bound]")), dd, vcov = ~state)
  dd19 <- dd %>% filter(year <= 2019)
  b19 <- feols(as.formula(paste(y, "~ X | employer_id_balanced + region^year")), dd19, vcov = ~state)
  a19 <- feols(as.formula(paste(y, "~ X | employer_id_balanced + region^year + year[bound]")), dd19, vcov = ~state)
  for (m in list(list("1b-analog 2011-24", b), list("1a-analog 2011-24", a),
    list("1b-analog 2011-19", b19), list("1a-analog 2011-19", a19))) {
    rows[[length(rows) + 1]] <- tibble::tibble(outcome = y, model = m[[1]], coef = coef(m[[2]])[["X"]],
      se_state = se(m[[2]])[["X"]], se_employer = se(summary(m[[2]], vcov = ~employer_id_balanced))[["X"]],
      se_region = se(summary(m[[2]], vcov = ~region))[["X"]], nobs = nobs(m[[2]]))
  }
  es <- feols(as.formula(paste(y, "~ i(year, bound, ref = 2012) | employer_id_balanced + region^year")),
    panel, vcov = ~state)
  ct <- coeftable(es)
  es_rows[[y]] <- tibble::tibble(outcome = y, year = as.integer(sub(".*::(\\d+):bound", "\\1", rownames(ct))),
    coef = ct[, 1], se = ct[, 2])
}
res <- bind_rows(rows) %>% mutate(t = coef / se_state)
write_csv(res, file.path(out_dir, "employer_bite_estimates.csv"))
print(as.data.frame(res), digits = 3)
es <- bind_rows(es_rows)
write_csv(es, file.path(out_dir, "employer_bite_event_study.csv"))
print(as.data.frame(es), digits = 3)
