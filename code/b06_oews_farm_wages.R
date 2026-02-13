library(here)
library(arrow)
library(tidyverse)
library(dplyr)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)

rm(list = ls())

# Load crosswalks
oews_area_definitions_df <- read_parquet(here("binaries", "oews_area_definitions.parquet"))

# Load OEWS data
oews_df <- read_parquet(here("binaries", "oews.parquet"))

# Keep only the "Big Six" SOCs used to calculate the AEWR, as detailed in the DOL OFLC final rule for AEWR determination:
# https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/2023%20AEWR%20Rule%20FAQ%20-%20Round%202%20-%207-11-2023.pdf
# 45-2041 Graders and Sorters, Agricultural Products
# 45-2091 Agricultural Equipment Operators
# 45-2092 Farmworkers and Laborers, Crop, Nursery and Greenhouse
# 45-2093 Farmworkers, Farm, Ranch, and Aquacultural Animals
# 53-7064 Packers and Packagers, Hand
# 45-2099 Agricultural Workers – Other
# For years 1997 and 1998, we have to use the BLS' own codes: 
# https://www.bls.gov/oes/special-requests/oesdic_98.pdf
# 79011 GRADERS AND SORTERS, AGRICULTURAL PRODUCTS
# 79021 FARM EQUIPMENT OPERATORS
# 79856 FARMWORKERS, FOOD AND FIBER CROPS
# 79858 FARMWORKERS, FARM AND RANCH ANIMALS
# 98902 HAND PACKERS AND PACKAGERS
oews_df <- oews_df %>% 
  filter(
    occ_code == "45-2041" | 
      occ_code == "45-2091" | 
      occ_code == "45-2092" | 
      occ_code == "45-2093" | 
      occ_code == "53-7064" | 
      occ_code == "45-2099" | 
      occ_code == "79011" |  
      occ_code == "79021" | 
      occ_code == "79856" | 
      occ_code == "79858" | 
      occ_code == "98902") 

oews_df <- oews_df %>% 
  mutate(tot_emp = as.numeric(tot_emp)) %>% 
  mutate(mean_hourly_wage = as.numeric(h_mean)) %>% 
  mutate(mean_annual_wage = as.numeric(a_mean)) %>% 
  mutate(oews_area_code = area)

# Drop occ-area-year without average wage or total employment data
# oews_df <- oews_df %>% filter(!is.na(h_mean) & !is.na(tot_emp)) 

# Combine with OEWS area definitons to calculate county-level wages
oews_df <- oews_df %>% 
  select(oews_area_code, area_name, occ_code, occ_title, tot_emp, mean_hourly_wage, mean_annual_wage, year)

oews_area_year_df <- oews_area_definitions_df %>% 
  left_join(oews_df, by = c("oews_area_code", "year"))

# Collapse into county-years-occupations
oews_county_year_df <- oews_area_year_df %>% 
  group_by(oews_state_fips, oews_county_fips, year, occ_code, occ_title) %>% 
  summarize(
    tot_emp = sum(tot_emp, na.rm = TRUE),
    tot_hourly_wage = sum(mean_hourly_wage*tot_emp, na.rm = TRUE),
    tot_annual_wage = sum(mean_annual_wage*tot_emp, na.rm = TRUE),
  ) %>% 
  ungroup() %>% 
  mutate(
    mean_hourly_wage = tot_hourly_wage/tot_emp,
    mean_annual_wage = tot_annual_wage/tot_emp
  ) %>% 
  select(-tot_hourly_wage, -tot_annual_wage)

# Collapse oocupations as well to get AEWR equivalent
oews_aewr_df <- oews_area_year_df %>% 
  group_by(oews_state_fips, oews_county_fips, year) %>% 
  summarize(
    tot_emp = sum(tot_emp, na.rm = TRUE),
    tot_hourly_wage = sum(mean_hourly_wage*tot_emp, na.rm = TRUE),
    tot_annual_wage = sum(mean_annual_wage*tot_emp, na.rm = TRUE),
  ) %>% 
  ungroup() %>% 
  mutate(
    mean_hourly_wage = tot_hourly_wage/tot_emp,
    mean_annual_wage = tot_annual_wage/tot_emp
  ) %>% 
  select(-tot_hourly_wage, -tot_annual_wage) %>% 
  mutate(
    occ_code = "AEWR",
    occ_title = "Aggregated occupation for AEWR equivalent"
    )

# Combine
oews_combined <- rbind(oews_county_year_df, oews_aewr_df) %>% 
  arrange(oews_state_fips, oews_county_fips, year, occ_code)

# Harmonize variable names
oews_combined <- oews_combined %>% 
  rename(
    state_fips_code = oews_state_fips,
    county_fips_code = oews_county_fips,
    oews_mean_hourly_wage = mean_hourly_wage,
    oews_mean_annual_wage = mean_annual_wage,
    oews_tot_emp = tot_emp
  )

# Export
oews_combined %>% 
  write_parquet(here("files_for_phil", "oews_county_aggregated.parquet")) %>% 
  write_parquet(here("binaries", "oews_county_aggregated.parquet"))
