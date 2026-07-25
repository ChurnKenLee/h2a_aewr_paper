# Purpose: Build annual and 2007-baseline Census of Agriculture cropland panels.
# Input: data/intermediate/qs_census_selected_obs.parquet.
# Outputs: census_ag_cropland_year.parquet and census_ag_cropland_2007_year.parquet.
# Run after: code/a01_sources/03_02_nass_select_quickstats_obs.py.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(dplyr)
library(tidyr)

options(arrow.skip_nul = TRUE)
census_of_agriculture <- read_parquet(path_int(
  "qs_census_selected_obs.parquet"
))
assert_geo_columns(
  census_of_agriculture,
  "county_fips",
  allow_na = "county_fips"
)

# general cleaning #

ag_census_data_items <- census_of_agriculture %>%
  group_by(commodity_desc) %>%
  tally()

# want: FARM OPERATIONS, AG LAND

census_of_agriculture_trim <- census_of_agriculture %>%
  filter(commodity_desc == "FARM OPERATIONS" | commodity_desc == "AG LAND")

census_of_agriculture_trim_items <- census_of_agriculture_trim %>%
  group_by(short_desc) %>%
  tally()

census_of_agriculture_trim <- census_of_agriculture_trim %>%
  filter(short_desc == "AG LAND, CROPLAND - ACRES")

census_of_agriculture_trim <- census_of_agriculture_trim %>%
  arrange(county_fips, year)

census_of_agriculture_trim <- census_of_agriculture_trim %>%
  mutate(label = "cropland_acr") %>%
  select(year, county_fips, value, label)

census_of_agriculture_trim <- census_of_agriculture_trim %>%
  filter(!is.na(county_fips))

census_of_agriculture_cropland <- census_of_agriculture_trim %>%
  pivot_wider(names_from = "label", values_from = "value")

assert_geo_columns(census_of_agriculture_cropland, "county_fips")
census_of_agriculture_cropland %>%
  write_parquet(path_int("census_ag_cropland_year.parquet"))


census_of_agriculture_cropland_base <- census_of_agriculture_cropland %>%
  filter(year == 2007) %>%
  rename(cropland_acr_2007 = cropland_acr) %>%
  select(-year)

census_of_agriculture_cropland_base %>%
  write_parquet(path_int("census_ag_cropland_2007_year.parquet"))
