library(here)
library(arrow)
library(tidyverse)
library(dplyr)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)
library(foreign)
library(tidycensus)

rm(list = ls())

# Import NAWSPAD data from CSV
nawspad1_df <- read.csv(here("Data", "nawspad", "NAWS_A2E197.csv")) %>% 
  clean_names()
nawspad2_df <- read.csv(here("Data", "nawspad", "NAWS_F2Y197.csv")) %>% 
  clean_names()

# Combine NAWSPAD data
nawspad_df <- nawspad1_df %>% 
  full_join(nawspad2_df, by = c("fwid"), suffix = c("", ".y"))

# Keep only variables we need to calculate work hours by region, crop type, seasonality
nawspad_df <- nawspad_df %>% 
  select(fwid, pwtycrd, fy, region6, fwweeks, fwrdays, c10, d01, d04, d08, crop)

# Rename variables to more informative names
nawspad_df <- nawspad_df %>% 
  rename(
    id = fwid,
    weight = pwtycrd,
    year = fy,
    farm_work_weeks = fwweeks,
    farm_work_days = fwrdays,
    farm_nonfarm_work_days = c10,
    farm_work_months = d01,
    farm_work_hours = d04,
    farm_nonfarm_work_hours = d08,
    crop_type = crop
  )

# Calculate farm work hours by seasonal vs full-time in 5 year bands around each census year
census_years = c(2002, 2007, 2012, 2017)
nawspad_df <- nawspad_df %>% 
  mutate(seasonal = farm_work_weeks < 21.4) %>% 
  mutate(year = as.numeric(year)) %>% 
  rowwise() %>% 
  mutate(census_year = census_years[order(abs(year - census_years))][1]) %>% 
  ungroup() %>%
  mutate(farm_work_hours_annual = farm_work_weeks*farm_work_hours)

# Add labels
nawspad_df <- nawspad_df %>% 
  mutate(crop_type = case_when(
    crop_type == 1 ~ "Field crops",
    crop_type == 2 ~ "Fruits & nuts",
    crop_type == 3 ~ "Horticulture",
    crop_type == 4 ~ "Vegetables",
    crop_type == 5 ~ "Misc/mult"
  )) %>% 
  mutate(seasonality = case_when(
    seasonal == TRUE ~ "Seasonal",
    seasonal == FALSE ~ "Full-time"
  ))

# Weighted average for different crosstabs
nawspad_df_seasonal <- nawspad_df %>% 
  group_by(region6, census_year, seasonality) %>%
  summarise(mean_annual_farm_work_hours = weighted.mean(farm_work_hours_annual, weight, na.rm = TRUE)) %>% 
  ungroup() %>% 
  filter(!is.na(seasonality)) %>% 
  mutate(crop_type = "All")

nawspad_df_crop <- nawspad_df %>%
  group_by(region6, census_year, crop_type) %>% 
  summarise(mean_annual_farm_work_hours = weighted.mean(farm_work_hours_annual, weight, na.rm = TRUE)) %>%
  ungroup %>% 
  filter(!is.na(crop_type)) %>% 
  mutate(seasonality = "All")

nawspad_df_crop_seasonal <- nawspad_df %>%
  group_by(region6, census_year, crop_type, seasonality) %>% 
  summarise(mean_annual_farm_work_hours = weighted.mean(farm_work_hours_annual, weight, na.rm = TRUE)) %>%
  filter(!is.na(crop_type)) %>% 
  filter(!is.na(seasonality)) %>% 
  ungroup()

# Add states and FIPS code
region_state_df <- read.csv(here("Data", "nawspad", "nawspad_region6.csv")) %>% 
  clean_names() %>% 
  separate_longer_delim(states, delim = ", ")

nawspad_agg_df <- nawspad_df_seasonal %>% 
  bind_rows(nawspad_df_crop) %>% 
  bind_rows(nawspad_df_crop_seasonal)

nawspad_state_df <- nawspad_agg_df %>% 
  cross_join(region_state_df) %>% 
  filter(region6 == region)

# Get FIPS codes from Census API
data(fips_codes)
state_fips <- fips_codes %>% 
  distinct(state_code, state_name)

# Add FIPS codes
nawspad_state_df <- nawspad_state_df %>% 
  left_join(state_fips, by = c("states" = "state_name")) %>% 
  select(-region, -region_name, -region6, -states) %>% 
  rename(
    state_fips_code = state_code,
    year = census_year
    )

# Export
nawspad_state_df %>% 
  write_parquet(here("files_for_phil", "nawspad.parquet")) %>% 
  write_parquet(here("binaries", "nawspad.parquet"))

