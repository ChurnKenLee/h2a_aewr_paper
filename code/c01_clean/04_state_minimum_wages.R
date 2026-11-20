# Purpose: Construct real state and agricultural minimum-wage measures.
# Inputs: fred_state_minwages.parquet, ppi_2012.parquet, and the hand-maintained
#   dated coverage table code/a01_sources/state_ag_minwage_coverage.csv.
# Output: data/intermediate/state_real_minwages.parquet.
# Run after: 03_producer_price_index.R; the FRED parquet is an upstream input.
#
# Agricultural floor: the coverage spell in effect on 1 July of each year sets
# applicable_floor. state_minimum -> max(state, federal); federal_minimum ->
# federal; ag_specific_rate -> max(ag_min_wage, federal). Years before the
# first spell (the table starts 2008-01-01) take the first spell and are
# flagged ag_coverage_backcast.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
library(arrow)
library(dplyr)

state_minwages <- read_parquet(path_int("fred_state_minwages.parquet"))
ppi_data <- read_parquet(path_int("ppi_2012.parquet"))

coverage <- read.csv(path_code("a01_sources", "state_ag_minwage_coverage.csv"),
  colClasses = "character", na.strings = character()) %>%
  transmute(
    state_fips,
    start = as.Date(effective_start),
    end = as.Date(ifelse(effective_end == "", "9999-12-31", effective_end)),
    applicable_floor,
    ag_min_wage = suppressWarnings(as.numeric(ag_min_wage))
  )
stopifnot(
  nrow(coverage) > 0,
  all(nchar(coverage$state_fips) == 2),
  !anyNA(coverage$start), !anyNA(coverage$end), all(coverage$end >= coverage$start),
  all(coverage$applicable_floor %in% c("state_minimum", "federal_minimum", "ag_specific_rate")),
  all(is.finite(coverage$ag_min_wage[coverage$applicable_floor == "ag_specific_rate"]))
)
# Spells must not overlap within a state.
overlap <- coverage %>% arrange(state_fips, start) %>% group_by(state_fips) %>%
  filter(start <= lag(end, default = as.Date("1000-01-01"))) %>% ungroup()
if (nrow(overlap)) stop("Overlapping coverage spells: ", paste(unique(overlap$state_fips), collapse = ", "))
first_start <- coverage %>% group_by(state_fips) %>% summarise(first_start = min(start), .groups = "drop")

state_years <- state_minwages %>% distinct(state_fips, year) %>%
  left_join(first_start, by = "state_fips", relationship = "many-to-one") %>%
  mutate(
    ag_coverage_backcast = as.Date(sprintf("%d-07-01", year)) < first_start,
    on = pmax(as.Date(sprintf("%d-07-01", year)), first_start)
  )
missing_states <- setdiff(unique(state_years$state_fips), coverage$state_fips)
if (length(missing_states)) stop("States without coverage spells: ", paste(missing_states, collapse = ", "))
ag_coverage <- state_years %>%
  inner_join(coverage, by = "state_fips", relationship = "many-to-many") %>%
  filter(start <= on, end >= on) %>%
  select(state_fips, year, applicable_floor, ag_min_wage, ag_coverage_backcast)
stopifnot(
  !anyDuplicated(ag_coverage[c("state_fips", "year")]),
  nrow(ag_coverage) == nrow(state_years)
)

state_minwage_ppi <- state_minwages %>%
  left_join(ppi_data, by = "year", relationship = "many-to-one") %>%
  left_join(ag_coverage, by = c("state_fips", "year"), relationship = "many-to-one") %>%
  mutate(
    agriculture_exemption = applicable_floor == "federal_minimum",
    state_min_wage = coalesce(state_min_wage, federal_min_wage),
    prevailing_ag_min_wage = case_when(
      applicable_floor == "federal_minimum" ~ federal_min_wage,
      applicable_floor == "ag_specific_rate" ~ pmax(ag_min_wage, federal_min_wage, na.rm = TRUE),
      TRUE ~ pmax(state_min_wage, federal_min_wage, na.rm = TRUE)
    ),
    prevailing_min_wage_ppi = prevailing_min_wage / ppi_2012,
    prevailing_ag_min_wage_ppi = prevailing_ag_min_wage / ppi_2012,
    federal_min_wage_ppi = federal_min_wage / ppi_2012,
    state_min_wage_ppi = state_min_wage / ppi_2012
  ) %>%
  rename(ag_floor_rule = applicable_floor, ag_specific_min_wage = ag_min_wage) %>%
  select(-ppi_2012)
stopifnot(nrow(state_minwage_ppi) > 0, !anyDuplicated(state_minwage_ppi[c("state_fips", "year")]))

write_parquet(
  state_minwage_ppi,
  path_int("state_real_minwages.parquet")
)
