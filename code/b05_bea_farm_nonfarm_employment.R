library(here)
library(arrow)
library(tidyverse)
library(dplyr)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)
library(foreign)
library(tidycensus)
library(lubridate)
library(foreign)
library(haven)
library(sf)
library(ggplot2)
library(ggnewscale)
library(tidycensus)
options(tigris_use_cache = TRUE)

rm(list = ls())

# BEA
bea <- read_csv(here("Data", "bea", "CAEMP25N__ALL_AREAS_2001_2022.csv"))

# We only want total farm and nonfarm employment
bea_farm <- bea %>%
  filter(LineCode == "70") %>%
  pivot_longer(
    cols = starts_with("20"),
    names_to = "year",
    values_to = "bea_farm_emp",
  ) %>%
  mutate(bea_farm_emp = as.numeric(bea_farm_emp)) %>%
  select(GeoFIPS, year, bea_farm_emp)

bea_nonfarm <- bea %>%
  filter(LineCode == "80") %>%
  pivot_longer(
    cols = starts_with("20"),
    names_to = "year",
    values_to = "bea_nonfarm_emp",
  ) %>%
  mutate(bea_nonfarm_emp = as.numeric(bea_nonfarm_emp)) %>%
  select(GeoFIPS, year, bea_nonfarm_emp)

bea_farm_nonfarm <- bea_farm %>%
  left_join(bea_nonfarm) %>%
  rename(county_fips = GeoFIPS) %>%
  mutate(year = as.numeric(year))

# Aggregate to relevant ACS years and save
bea_farm_nonfarm <- bea_farm_nonfarm %>%
  mutate(
    acs_year = case_when(
      between(year, 2000, 2004) ~ 2004,
      between(year, 2005, 2007) ~ 2007,
      between(year, 2008, 2012) ~ 2012,
      between(year, 2013, 2017) ~ 2017,
      between(year, 2018, 2022) ~ 2022
    )
  ) %>%
  group_by(county_fips, acs_year) %>%
  summarize(
    bea_farm_emp = mean(bea_farm_emp),
    bea_nonfarm_emp = mean(bea_nonfarm_emp)
  ) %>%
  ungroup()

bea_farm_nonfarm %>%
  write_parquet(here("files_for_phil", "bea_farm_nonfarm_emp.parquet")) %>%
  write_parquet(here("binaries", "bea_farm_nonfarm_emp.parquet"))
