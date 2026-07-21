# Purpose: Build annual and 2007-baseline Census of Agriculture cropland panels.
# Input: data/intermediate/qs_census_selected_obs.parquet.
# Outputs: census_ag_cropland_year.parquet and census_ag_cropland_2007_year.parquet.
# Run after: code/a01_sources/03_02_nass_select_quickstats_obs.py.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
source(path_code("c00_shared", "fips.R"))
library(arrow)
library(dplyr)
library(tidyr)

options(arrow.skip_nul = TRUE)
census_of_agriculture <- read_parquet(path_int(
  "qs_census_selected_obs.parquet"
))

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

# fix fips

census_of_agriculture_trim <- census_of_agriculture_trim %>%
  mutate(
    countyfips = combine_county_fips(state_fips_code, county_code)
  )

census_of_agriculture_trim <- census_of_agriculture_trim %>%
  arrange(countyfips, year)

census_of_agriculture_trim <- census_of_agriculture_trim %>%
  mutate(label = "cropland_acr") %>%
  select(year, countyfips, value, label)

census_of_agriculture_trim <- census_of_agriculture_trim %>%
  filter(!is.na(countyfips))

census_of_agriculture_cropland <- census_of_agriculture_trim %>%
  pivot_wider(names_from = "label", values_from = "value")


census_of_agriculture_cropland %>%
  write_parquet(path_int("census_ag_cropland_year.parquet"))


census_of_agriculture_cropland_base <- census_of_agriculture_cropland %>%
  filter(year == 2007) %>%
  rename(cropland_acr_2007 = cropland_acr) %>%
  select(-year)

census_of_agriculture_cropland_base %>%
  write_parquet(path_int("census_ag_cropland_2007_year.parquet"))
