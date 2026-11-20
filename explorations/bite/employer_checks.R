# Scratch diagnostic: design 4 checks. Employer type from baseline job titles,
# within-type estimates, lead placebo, and alternative baseline cohorts.
# Writes only to argument 1.
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
region_aewr <- cp %>% mutate(aewr_region_id = as.character(aewr_region_id)) %>%
  group_by(aewr_region_id, year) %>% summarise(aewr_real = mean(aewr / ppi_2012), .groups = "drop") %>%
  arrange(aewr_region_id, year) %>% group_by(aewr_region_id) %>%
  mutate(aewr_real_lead = lead(aewr_real)) %>% ungroup()

titles <- read_parquet(path_int("h2a_with_fips.parquet"), col_select = c("case_number", "fiscal_year", "job_title")) %>%
  distinct(case_number, fiscal_year, .keep_all = TRUE) %>%
  mutate(t = toupper(job_title),
    type = case_when(
      grepl("EQUIP|MACHIN|OPERATOR|COMBIN|TRUCK|DRIVER|TRACTOR|CUSTOM|MECHANIC", t) ~ "equipment",
      grepl("ANIMAL|LIVESTOCK|RANCH|DAIRY|HERD|CATTLE|SHEEP|GOAT|HOG|SWINE|POULTRY|BEE|HORSE|MILK", t) ~ "livestock",
      TRUE ~ "crop")) %>% select(case_number, fiscal_year, type)

w <- read_parquet(path_int("h2a_allocated_worksites.parquet"),
  col_select = c("case_number", "fiscal_year", "employer_id_balanced", "employer_is_worksite_business",
    "adjusted_nbr_workers_certified", "period_begin_date", "period_end_date", "number_of_hours",
    "wage_rate", "wage_unit", "county_fips_list")) %>%
  filter(employer_is_worksite_business, !is.na(employer_id_balanced), employer_id_balanced != "") %>%
  mutate(begin = as.Date(period_begin_date), end = as.Date(period_end_date), year = year(begin),
    workers = coalesce(as.numeric(adjusted_nbr_workers_certified), 0),
    hours = pmax(as.numeric(end - begin), 0) / 7 * as.numeric(number_of_hours) * workers,
    wage = suppressWarnings(as.numeric(wage_rate)),
    hourly = toupper(trimws(wage_unit)) %in% c("HOUR", "HOURLY", "HR") & is.finite(wage) & wage > 0 & wage <= 100,
    state = substr(trimws(sub(",.*", "", county_fips_list)), 1, 2)) %>%
  filter(year >= 2008, year <= 2024) %>%
  left_join(titles, by = c("case_number", "fiscal_year"))

build <- function(base_years, years) {
  base <- w %>% filter(year %in% base_years, workers > 0, hourly, is.finite(hours), hours > 0) %>%
    left_join(rates, by = c("state", "year")) %>%
    mutate(at = (is.finite(aewr) & abs(wage - aewr) <= TOL) | (is.finite(aewr_lag) & abs(wage - aewr_lag) <= TOL)) %>%
    left_join(state_region, by = "state")
  cohort <- base %>% group_by(employer_id_balanced) %>%
    summarise(state = names(which.max(tapply(hours, state, sum))),
      type = names(which.max(tapply(hours, coalesce(type, "crop"), sum))),
      share_at = sum(hours * at) / sum(hours), regions = n_distinct(aewr_region_id),
      region = first(aewr_region_id), .groups = "drop") %>%
    filter(regions == 1, !is.na(region)) %>% mutate(bound = as.integer(share_at >= 0.5))
  ey <- w %>% semi_join(cohort, by = "employer_id_balanced") %>% group_by(employer_id_balanced, year) %>%
    summarise(workers = sum(workers), .groups = "drop")
  cohort %>% select(employer_id_balanced, bound, type, region, state) %>% crossing(year = years) %>%
    left_join(ey, by = c("employer_id_balanced", "year")) %>%
    mutate(workers = coalesce(workers, 0), any = as.integer(workers > 0), lworkers = log1p(workers)) %>%
    left_join(region_aewr, by = c("region" = "aewr_region_id", "year")) %>%
    filter(is.finite(aewr_real)) %>% mutate(X = bound * aewr_real, Xlead = bound * aewr_real_lead)
}
est <- function(label, d, rhs, fe) {
  out <- list()
  for (y in c("any", "lworkers")) {
    m <- feols(as.formula(paste(y, "~", rhs, "|", fe)), d, vcov = ~state)
    ct <- coeftable(m)
    for (term in rownames(ct)) out[[length(out) + 1]] <- tibble::tibble(check = label, outcome = y, term = term,
      coef = ct[term, 1], se = ct[term, 2], se_region = se(summary(m, vcov = ~region))[[term]], nobs = nobs(m))
  }
  bind_rows(out)
}
FE_1A <- "employer_id_balanced + region^year + year[bound]"
FE_TYPE <- "employer_id_balanced + region^year^type + bound^type^year"

p11 <- build(2011:2012, 2011:2019)
comp <- p11 %>% distinct(employer_id_balanced, bound, type, region) %>% count(type, bound) %>%
  pivot_wider(names_from = bound, values_from = n, names_prefix = "bound")
print(comp)
cat("Mixed regions within crop type:\n")
print(p11 %>% filter(type == "crop") %>% distinct(employer_id_balanced, bound, region) %>% count(region, bound) %>%
  pivot_wider(names_from = bound, values_from = n, names_prefix = "bound") %>% filter(bound0 >= 10), n = 17)

res <- bind_rows(
  est("A. baseline 1(a)-analog, 2011-12 cohort, 2011-19", p11, "X", FE_1A),
  est("B. crop-titled employers only", p11 %>% filter(type == "crop"), "X", FE_1A),
  est("C. equipment-titled employers only", p11 %>% filter(type == "equipment"), "X", FE_1A),
  est("D. all, type x region x year FE and bound x type x year FE", p11, "X", FE_TYPE),
  est("E. lead placebo (adds bound x next-year AEWR), 2011-18", p11 %>% filter(is.finite(Xlead)), "X + Xlead", FE_1A),
  est("F. 2014-15 cohort, 2014-19", build(2014:2015, 2014:2019), "X", FE_1A),
  est("G. 2020-21 cohort, 2020-24", build(2020:2021, 2020:2024), "X", FE_1A)
) %>% mutate(t = coef / se)
write_csv(res, file.path(out_dir, "employer_checks.csv"))
print(as.data.frame(res), digits = 3)
cohorts <- list(c2014 = build(2014:2015, 2014:2019), c2020 = build(2020:2021, 2020:2024))
for (nm in names(cohorts)) {
  x <- cohorts[[nm]] %>% distinct(employer_id_balanced, bound, type)
  cat(nm, ": employers", nrow(x), " unbound", sum(x$bound == 0), " unbound equipment share",
    round(mean(x$type[x$bound == 0] == "equipment"), 2), "\n")
}
