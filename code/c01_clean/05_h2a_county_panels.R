# Purpose: Build census-period, annual, and national H-2A panels.
# Inputs: h2a_aggregated.parquet and the elastic-net H-2A prediction.
# Outputs: h2a_predict.parquet, h2a_data_year.parquet, and processed H-2A panels.
# Run after: code/b01_derived/01_h2a_aggregation_nodupes.R and
# code/b01_derived/07_h2a_prediction_elastic_net.py.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(dplyr)
library(stringr)

h2a_data <- read_parquet(
  file = path_int("h2a_aggregated.parquet")
)
h2a_predict <- read_parquet(
  file = path_int("h2a_prediction_using_elastic_net_continuous_basis.parquet")
)
assert_geo_columns(
  h2a_data,
  c("state_fips", "county_code", "county_fips")
)
assert_geo_columns(h2a_predict, "county_fips")
write_parquet(h2a_predict, path_int("h2a_predict.parquet"))

# census period
h2a_data <- h2a_data %>%
  mutate(
    census_period = ifelse(
      year >= 2008 & year < 2012,
      2012,
      ifelse(
        year >= 2012 & year < 2017,
        2017,
        ifelse(year >= 2017 & year < 2022, 2022, NA)
      )
    )
  )


h2a_data <- h2a_data %>%
  filter(!is.na(census_period) & !is.na(county_fips)) %>%
  select(
    -state_fips,
    -county_code
  )

# collapse by period, county (county and state fips)
h2a_prediction <- h2a_data %>%
  select(county_fips)

h2a_data <- h2a_data %>%
  group_by(census_period, county_fips) %>%
  summarise(
    nbr_workers_requested_all_years = sum(
      nbr_workers_requested_all_years,
      na.rm = T
    ),
    nbr_workers_certified_all_years = sum(
      nbr_workers_certified_all_years,
      na.rm = T
    ),
    man_hours_requested_all_years = sum(
      man_hours_requested_all_years,
      na.rm = T
    ),
    man_hours_certified_all_years = sum(
      man_hours_certified_all_years,
      na.rm = T
    ),
    nbr_applications_all_years = sum(nbr_applications_all_years, na.rm = T),
    nbr_workers_requested_start_year = sum(
      nbr_workers_requested_start_year,
      na.rm = T
    ),
    nbr_workers_certified_start_year = sum(
      nbr_workers_certified_start_year,
      na.rm = T
    ),
    man_hours_requested_start_year = sum(
      man_hours_requested_start_year,
      na.rm = T
    ),
    man_hours_certified_start_year = sum(
      man_hours_certified_start_year,
      na.rm = T
    ),
    nbr_applications_start_year = sum(nbr_applications_start_year, na.rm = T),
    nbr_workers_requested_fiscal_year = sum(
      nbr_workers_requested_fiscal_year,
      na.rm = T
    ),
    nbr_workers_certified_fiscal_year = sum(
      nbr_workers_certified_fiscal_year,
      na.rm = T
    ),
    man_hours_requested_fiscal_year = sum(
      man_hours_requested_fiscal_year,
      na.rm = T
    ),
    man_hours_certified_fiscal_year = sum(
      man_hours_certified_fiscal_year,
      na.rm = T
    ),
    nbr_applications_fiscal_year = sum(nbr_applications_fiscal_year, na.rm = T)
  )


assert_geo_columns(h2a_data, "county_fips")
write_parquet(h2a_data, path_processed("h2a_data.parquet"))

# yearly, for TS

h2a_data_ts <- read_parquet(
  path_int("h2a_aggregated.parquet"),
  stringsAsFactors = F
)
assert_geo_columns(
  h2a_data_ts,
  c("state_fips", "county_code", "county_fips")
)

h2a_data_ts <- h2a_data_ts %>%
  group_by(year) %>%
  summarise(
    h2a_man_hours_certified = sum(man_hours_certified_start_year, na.rm = T),
    h2a_man_hours_requested = sum(man_hours_requested_start_year, na.rm = T),
    h2a_nbr_workers_certified = sum(
      nbr_workers_certified_start_year,
      na.rm = T
    ),
    h2a_nbr_workers_requested = sum(
      nbr_workers_requested_start_year,
      na.rm = T
    ),
    n_applications = sum(nbr_applications_start_year, na.rm = T)
  )


h2a_data_ts <- h2a_data_ts %>%
  rename(case_year = year)

write_parquet(h2a_data_ts, path_processed("h2a_data_ts.parquet"))

h2a_data <- read_parquet(
  path_int("h2a_aggregated.parquet"),
  stringsAsFactors = F
)
assert_geo_columns(
  h2a_data,
  c("state_fips", "county_code", "county_fips")
)

h2a_data <- h2a_data %>%
  filter(year <= 2022 & year > 2007 & !is.na(county_fips)) %>%
  select(
    -state_fips,
    -county_code
  )

# collapse by period, county (county and state fips)

## Handle NAs?

assert_geo_columns(h2a_data, "county_fips")
write_parquet(h2a_data, path_int("h2a_data_year.parquet"))
cat("h2a_data_year:", nrow(h2a_data), "rows,", ncol(h2a_data), "cols\n")
