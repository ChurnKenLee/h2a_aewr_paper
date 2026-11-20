# Purpose: Assemble annual county population estimates on the project's county vintage.
# Inputs: three Census population vintages (2009, 2019, 2025).
# Output: data/intermediate/census_pop_ests_year.parquet.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(dplyr)
library(readr)
library(tidyr)

census_pop_2000 <- read_csv(
  path_raw("census_population_estimates", "co-est2009-alldata.csv"),
  show_col_types = FALSE
)
census_pop_2010 <- read_csv(
  path_raw("census_population_estimates", "co-est2019-alldata.csv"),
  show_col_types = FALSE
)
census_pop_2020 <- read_csv(
  path_raw("census_population_estimates", "co-est2025-alldata.csv"),
  show_col_types = FALSE
)

# Connecticut reports planning regions from 2020 on. Project its 2010-vintage
# counties with statewide growth: 2020 bridges the 2019 vintage to the 2025
# vintage, and later years use the 2025 vintage alone.
ct_state_2019 <- census_pop_2010 %>%
  filter(as.integer(SUMLEV) == 40, as.integer(STATE) == 9L) %>%
  pull(POPESTIMATE2019)
ct_state_2020 <- census_pop_2020 %>%
  filter(as.integer(SUMLEV) == 40, as.integer(STATE) == 9L) %>%
  select(starts_with("POPESTIMATE"))
ct_state_population <- c(ct_state_2019, unlist(ct_state_2020[1, ]))
ct_popgrth <- data.frame(
  year = c(2019L, as.integer(sub("POPESTIMATE", "", names(ct_state_2020)))),
  grt = c(NA_real_, ct_state_population[-1] / head(ct_state_population, -1))
)
if (length(ct_state_2019) != 1L || nrow(ct_state_2020) != 1L) {
  stop("Connecticut state population rows are missing.", call. = FALSE)
}

census_pop_2000 <- census_pop_2000 %>%
  filter(as.integer(SUMLEV) == 50) %>%
  select(STATE, COUNTY, 10:19)
census_pop_2010 <- census_pop_2010 %>%
  filter(as.integer(SUMLEV) == 50) %>%
  select(STATE, COUNTY, 10:19)
census_pop_2020 <- census_pop_2020 %>%
  filter(as.integer(SUMLEV) == 50) %>%
  select(STATE, COUNTY, starts_with("POPESTIMATE"))

# fix CT county to region nonsense.
# https://www.ctdata.org/blog/geographic-resources-for-connecticuts-new-county-equivalent-geography
# project using state growth rates

census_pop_ests <- census_pop_2000 %>%
  full_join(
    census_pop_2010,
    by = c("STATE", "COUNTY"),
    relationship = "one-to-one"
  ) %>%
  full_join(
    census_pop_2020,
    by = c("STATE", "COUNTY"),
    relationship = "one-to-one"
  )

census_pop_ests <- census_pop_ests %>%
  mutate(
    county_fips = combine_county_fips(STATE, COUNTY)
  )

census_pop_ests <- census_pop_ests %>%
  select(-STATE, -COUNTY)

census_pop_ests <- census_pop_ests %>%
  pivot_longer(
    cols = starts_with("POPESTIMATE"),
    names_to = "year",
    names_prefix = "POPESTIMATE",
    values_to = "pop_census",
    values_drop_na = FALSE
  )

census_pop_ests <- census_pop_ests %>%
  filter(year >= SHARED_PANEL_START_YEAR & year <= SHARED_PANEL_END_YEAR)


# fix CT

census_pop_ests <- census_pop_ests %>%
  mutate(
    ct_region = ifelse(
      as.integer(county_fips) >= 9000 &
        as.integer(county_fips) <= 9999 &
        year >= 2020,
      1,
      0
    )
  ) # remove CT in 2020 on

ct_base <- census_pop_ests %>%
  filter(
    as.integer(county_fips) >= 9000 &
      as.integer(county_fips) <= 9999 &
      year == 2019
  )

census_pop_ests <- census_pop_ests %>%
  filter(ct_region == 0) %>%
  select(-ct_region)

census_pop_ests <- census_pop_ests %>%
  filter(as.integer(county_fips) < 9100 | as.integer(county_fips) > 9199) # get rid of the regions

ct_base <- ct_base %>%
  filter(as.integer(county_fips) < 9100) %>%
  select(county_fips, pop_census) %>%
  cross_join(
    ct_popgrth %>%
      filter(year >= 2020L, year <= SHARED_PANEL_END_YEAR) %>%
      arrange(year) %>%
      mutate(cumulative_growth = cumprod(grt))
  ) %>%
  transmute(
    county_fips,
    year = as.character(year),
    pop_census = pop_census * cumulative_growth
  )

ct_fill <- ct_base


census_pop_ests <- bind_rows(census_pop_ests, ct_fill)

# fix SD county change

census_pop_ests$county_fips <- harmonize_county_fips_2010(
  census_pop_ests$county_fips
)

census_pop_ests <- census_pop_ests %>%
  filter(!is.na(pop_census))

write_parquet(
  census_pop_ests,
  path_int("census_pop_ests_year.parquet")
)
