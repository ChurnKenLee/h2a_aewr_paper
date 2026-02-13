library(here)
library(arrow)
library(tidyverse)
library(dplyr)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)
library(tidycensus)
library(fixest)

rm(list = ls())

# Load tables for common crop name and crop type
census_table <- read_parquet(here("binaries", "census_common_name_table.parquet"))

croplandcros_cdf_table <- read_parquet(here("binaries", "croplandcros_cdl_common_name_table.parquet"))

# Aggregate for census
census_df <- read_parquet(here("binaries", "census_of_agriculture.parquet")) %>% 
  clean_names() %>% 
  arrange(state_name, county_name, sector_desc, group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc)

census_df <- census_df %>% 
  filter(sector_desc == "CROPS") %>% 
  filter(unit_desc != "OPERATIONS") %>% 
  filter(domain_desc == "TOTAL") %>% 
  filter((unit_desc == "ACRES") | (unit_desc == "SQ FT"))

# Convert values to numeric, drop suppressed observations, join with common name table
census_df <- census_df %>% 
  mutate(across(value, as.numeric)) %>% 
  filter(!is.na(value)) %>% 
  inner_join(census_table)

# Convert units to acres, to, harmonize, then aggregate by county
census_df <- census_df %>% 
  mutate(harmonized_value = case_when(
    unit_desc == "ACRES" ~ value,
    unit_desc == "SQ FT" ~ value/43560
  )) %>% 
  rename(county_fips_code = county_code)

# Aggregate both by crop, and crop type
census_crop_agg <- census_df %>% 
  group_by(year, state_fips_code, county_fips_code, common_name) %>% 
  summarize(census_acreage = sum(harmonized_value)) %>% 
  ungroup()

census_crop_type_agg <- census_df %>% 
  group_by(year, state_fips_code, county_fips_code, crop_type) %>% 
  summarize(census_acreage = sum(harmonized_value)) %>% 
  ungroup()

# Now aggregate for CroplandCROS
croplandcros_df <- read_parquet(here("binaries", "croplandcros_county_crop_acres.parquet"))

# Need to deal with double cropping
# As I set up double cropping entries with both types of crops, can just cross-join
croplandcros_df <- croplandcros_df %>% 
  full_join(croplandcros_cdf_table) %>% 
  filter(!is.na(crop_type))

# Aggregate by both crop and crop type
croplandcros_crop_agg <- croplandcros_df %>% 
  group_by(year, state_fips_code, county_fips_code, common_name) %>% 
  summarize(croplandcros_acreage = sum(acres)) %>% 
  ungroup()

croplandcros_crop_type_agg <- croplandcros_df %>% 
  group_by(year, state_fips_code, county_fips_code, crop_type) %>% 
  summarize(croplandcros_acreage = sum(acres)) %>% 
  ungroup()

# Merge census and CroplandCROS
crop_agg <- census_crop_agg %>% 
  inner_join(croplandcros_crop_agg)

crop_type_agg <- census_crop_type_agg %>% 
  inner_join(croplandcros_crop_type_agg)

# Run regression
model <- feols(croplandcros_acreage ~ census_acreage - 1, data = crop_agg)
summary(model)

model <- feols(croplandcros_acreage ~ census_acreage - 1, data = crop_type_agg)
summary(model)

# Fit is TERRIBLE; maybe try state level aggregates?
census_state_crop_agg <- census_df %>% 
  filter(is.na(county_fips_code)) %>% 
  group_by(year, state_fips_code, common_name) %>% 
  summarize(census_acreage = sum(harmonized_value)) %>% 
  ungroup()

census_state_crop_type_agg <- census_df %>% 
  filter(is.na(county_fips_code)) %>% 
  group_by(year, state_fips_code, crop_type) %>% 
  summarize(census_acreage = sum(harmonized_value)) %>% 
  ungroup()

croplandcros_state_crop_agg <- croplandcros_df %>% 
  group_by(year, state_fips_code, common_name) %>% 
  summarize(croplandcros_acreage = sum(acres)) %>% 
  ungroup()

croplandcros_state_crop_type_agg <- croplandcros_df %>% 
  group_by(year, state_fips_code, crop_type) %>% 
  summarize(croplandcros_acreage = sum(acres)) %>% 
  ungroup()

state_crop_agg <- census_state_crop_agg %>% 
  inner_join(croplandcros_state_crop_agg)

state_crop_type_agg <- census_state_crop_type_agg %>% 
  inner_join(croplandcros_state_crop_type_agg)

model <- feols(croplandcros_acreage ~ census_acreage - 1, data = crop_agg)
summary(model)

model <- feols(croplandcros_acreage ~ census_acreage - 1, data = crop_type_agg)
summary(model)

# Plot scatterplot of crop type acreage of Census and CroplandCROS




