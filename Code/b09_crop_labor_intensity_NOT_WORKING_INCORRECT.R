library(here)
library(arrow)
library(tidyverse)
library(dplyr)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)
library(fixest)

rm(list = ls())

# Load files
census_df <- read_parquet((here("binaries", "census_of_agriculture.parquet")))

yield_df <- read_parquet(here("binaries", "crop_yield.parquet"))

census_crop_names <- read_parquet((here("binaries", "census_crop_names.parquet")))

# Create separate dataframes for crop revenue, animal revenue, and labor cost
# Crop revenues have to be imputed via yields ($ per acre)
crop_revenue_df <- census_df %>% 
  filter(sector_desc == "CROPS") %>% 
  left_join(census_crop_names) %>% 
  left_join(yield_df)

crop_revenue_df <- crop_revenue_df %>% 
  mutate(harmonized_value = as.numeric(gsub(",", "", value))) %>% 
  mutate(harmonized_acreage = case_when(
    unit_desc == "ACRES" ~ harmonized_value,
    unit_desc == "SQ FT" ~ harmonized_value/43560
  )) %>% 
  mutate(revenue = harmonized_acreage*national_yield) %>% 
  mutate(revenue = if_else(is.na(revenue), 0, revenue))

# Aggregate to county-year-crop level
crop_revenue_df <- crop_revenue_df %>% 
  group_by(state_alpha, state_fips_code, county_code, year, common_name) %>% 
  summarize(revenue = sum(revenue, na.rm = TRUE)) %>% 
  ungroup() %>% 
  filter(!is.na(common_name)) %>% 
  mutate(common_name = gsub("[ -]", "_", common_name))

# Animal revenue
animal_revenue_df <- census_df %>% 
  filter(sector_desc == "ANIMALS & PRODUCTS") %>% 
  filter(unit_desc == "$") %>% 
  mutate(revenue = as.numeric(gsub(",", "", value))) %>% 
  mutate(revenue = if_else(is.na(revenue), 0, revenue))

# Aggregate to county-year-animal level
animal_revenue_df <- animal_revenue_df %>% 
  mutate(common_name = tolower(commodity_desc)) %>% 
  mutate(common_name = gsub("[ -&]", "_", common_name)) %>% 
  group_by(state_alpha, state_fips_code, county_code, year, common_name) %>% 
  summarize(revenue = sum(revenue, na.rm = TRUE)) %>% 
  ungroup() %>% 
  filter(!is.na(common_name)) %>% 
  mutate(common_name = gsub("[ -]", "_", common_name))

revenue_df <- crop_revenue_df %>% 
  rbind(animal_revenue_df)

# List of all commodity revenue to put in regression formula
revenue_vars <- revenue_df %>% 
  distinct(common_name) %>% 
  pull(common_name)

# Reshape to run regression
revenue_df <- revenue_df %>% 
  pivot_wider(
    id_cols = c(state_alpha, state_fips_code, county_code, year),
    names_from = common_name,
    values_from = revenue
  ) %>% 
  mutate_all(~replace(., is.na(.), 0))

# Labor costs
labor_cost_df <- census_df %>% 
  filter(sector_desc == "ECONOMICS") %>% 
  filter(group_desc == "EXPENSES") %>% 
  filter(commodity_desc == "LABOR") %>% 
  filter(unit_desc == "$")

labor_cost_df <- labor_cost_df %>% 
  mutate(harmonized_value = as.numeric(gsub(",", "", value))) %>% 
  group_by(state_alpha, state_fips_code, county_code, year) %>% 
  summarize(labor_cost = sum(harmonized_value, na.rm = TRUE)) %>% 
  ungroup()

# We should drop 2007 as ERS only started imputing cash receipts by crop-state in 2008
reg_df <- revenue_df %>% 
  left_join(labor_cost_df) %>% 
  filter(year != 2007) %>% 
  filter(county_code != "0") # Drop states

# Define formula and run regression
formula <- reformulate(revenue_vars, response = "labor_cost")

model <- feols(formula, data = reg_df, fixef = c("state_alpha"))

commodity_coeffs <- coefficients(model)

# Create dataframe of coefficients
labor_intensity <- data.frame(as.list(commodity_coeffs))

labor_intensity <- labor_intensity %>% 
  pivot_longer(
    cols = everything(),
    names_to = "commodity",
    values_to = "coeff"
  ) %>% 
  arrange(coeff)


                              