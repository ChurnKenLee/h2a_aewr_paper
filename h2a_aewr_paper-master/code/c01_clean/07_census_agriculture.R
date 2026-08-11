# Purpose: Build annual and 2007-baseline Census of Agriculture cropland panels.
# Input: data/intermediate/qs_census_selected_obs.parquet.
# Outputs: census_ag_cropland_year.parquet and census_ag_cropland_2007_year.parquet.
# Run after: code/a01_sources/03_02_nass_select_quickstats_obs.py.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
library(arrow)
library(dplyr)
library(tidyr)

options(arrow.skip_nul = TRUE)
census_of_agriculture <- read_parquet(path_int(
  "qs_census_selected_obs.parquet"
))

census_of_agriculture_trim <- suppressWarnings(
  census_of_agriculture %>%
    filter(
      commodity_desc %in% c("FARM OPERATIONS", "AG LAND"),
      short_desc == "AG LAND, CROPLAND - ACRES",
      !is.na(county_fips)
    ) %>%
    arrange(county_fips, year) %>%
    transmute(year, county_fips, value, label = "cropland_acr")
)

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
