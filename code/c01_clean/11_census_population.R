# Purpose: Assemble annual county population estimates on the project's county vintage.
# Inputs: three Census population vintages and the Connecticut growth crosswalk.
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
  path_raw("census_population_estimates", "co-est2022-alldata.csv"),
  show_col_types = FALSE
)

ct_popgrth <- read_csv(
  path_raw("geographic_crosswalks", "phil", "ct_pop_grth.csv"),
  show_col_types = FALSE
)

census_pop_2000 <- census_pop_2000 |>
  filter(as.integer(SUMLEV) == 50) |>
  select(STATE, COUNTY, 10:19)
census_pop_2010 <- census_pop_2010 |>
  filter(as.integer(SUMLEV) == 50) |>
  select(STATE, COUNTY, 10:19)
census_pop_2020 <- census_pop_2020 |>
  filter(as.integer(SUMLEV) == 50) |>
  select(STATE, COUNTY, 9:11)

# fix CT county to region nonsense.
# https://www.ctdata.org/blog/geographic-resources-for-connecticuts-new-county-equivalent-geography
# project using state growth rates

census_pop_ests <- census_pop_2000 |>
  full_join(
    census_pop_2010,
    by = c("STATE", "COUNTY"),
    relationship = "one-to-one"
  ) |>
  full_join(
    census_pop_2020,
    by = c("STATE", "COUNTY"),
    relationship = "one-to-one"
  )

census_pop_ests <- census_pop_ests |>
  mutate(
    county_fips = combine_county_fips(STATE, COUNTY)
  )

census_pop_ests <- census_pop_ests |>
  select(-STATE, -COUNTY)

census_pop_ests <- census_pop_ests |>
  pivot_longer(
    cols = starts_with("POPESTIMATE"),
    names_to = "year",
    names_prefix = "POPESTIMATE",
    values_to = "pop_census",
    values_drop_na = FALSE
  )

census_pop_ests <- census_pop_ests |>
  filter(year > 2007 & year <= 2022)


# fix CT

census_pop_ests <- census_pop_ests |>
  mutate(
    ct_region = ifelse(
      as.integer(county_fips) >= 9000 &
        as.integer(county_fips) <= 9999 &
        year >= 2020,
      1,
      0
    )
  ) # remove CT in 2020 on

ct_base <- census_pop_ests |>
  filter(
    as.integer(county_fips) >= 9000 &
      as.integer(county_fips) <= 9999 &
      year == 2019
  )

census_pop_ests <- census_pop_ests |>
  filter(ct_region == 0) |>
  select(-ct_region)

census_pop_ests <- census_pop_ests |>
  filter(as.integer(county_fips) < 9100 | as.integer(county_fips) > 9199) # get rid of the regions

ct_base <- ct_base |>
  filter(as.integer(county_fips) < 9100) |>
  select(county_fips, pop_census) |>
  mutate(
    pop_census2020 = pop_census * ct_popgrth$grt[ct_popgrth$year == 2020],
    pop_census2021 = pop_census2020 * ct_popgrth$grt[ct_popgrth$year == 2021],
    pop_census2022 = pop_census2021 * ct_popgrth$grt[ct_popgrth$year == 2022]
  )

ct_base <- ct_base |>
  select(-pop_census)

ct_fill <- ct_base |>
  pivot_longer(
    cols = starts_with("pop_census"),
    names_to = "year",
    names_prefix = "pop_census",
    values_to = "pop_census",
    values_drop_na = FALSE
  )


census_pop_ests <- bind_rows(census_pop_ests, ct_fill)

# fix SD county change

census_pop_ests$county_fips <- harmonize_county_fips_2010(
  census_pop_ests$county_fips
)

census_pop_ests <- census_pop_ests |>
  filter(!is.na(pop_census))

write_parquet(
  census_pop_ests,
  path_int("census_pop_ests_year.parquet")
)
