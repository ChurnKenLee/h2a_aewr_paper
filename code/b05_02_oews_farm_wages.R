rm(list = ls())
if (!exists("path_code", mode = "function")) {
  source(file.path("code", "paths.R"))
}
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(janitor)

# Load crosswalks
oews_area_definitions_df <- read_parquet(path_int(
  "oews_area_definitions.parquet"
))

# Load OEWS data
oews_df <- read_parquet(path_int("oews.parquet"))

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
      occ_code == "98902"
  )

oews_df <- oews_df %>%
  mutate(tot_emp = as.numeric(tot_emp)) %>%
  mutate(mean_hourly_wage = as.numeric(h_mean)) %>%
  mutate(mean_annual_wage = as.numeric(a_mean)) %>%
  mutate(oews_area_code = str_trim(as.character(area)))

# Drop occ-area-year without average wage or total employment data
# oews_df <- oews_df %>% filter(!is.na(h_mean) & !is.na(tot_emp))

# Combine with OEWS area definitons to calculate county-level wages
oews_df <- oews_df %>%
  select(
    oews_area_code,
    area_name,
    occ_code,
    occ_title,
    tot_emp,
    mean_hourly_wage,
    mean_annual_wage,
    year
  ) %>%
  mutate(
    hourly_wage_bill = mean_hourly_wage * tot_emp,
    annual_wage_bill = mean_annual_wage * tot_emp,
    hourly_wage_emp = if_else(!is.na(mean_hourly_wage), tot_emp, NA_real_),
    annual_wage_emp = if_else(!is.na(mean_annual_wage), tot_emp, NA_real_)
  )

oews_area_year_df <- oews_area_definitions_df %>%
  left_join(oews_df, by = c("oews_area_code", "year"))

# Collapse into county-years-occupations
oews_county_year_occ <- oews_area_year_df %>%
  group_by(oews_state_fips, oews_county_fips, year, occ_code, occ_title) %>%
  summarize(
    tot_emp = sum(tot_emp, na.rm = TRUE),
    hourly_wage_emp = sum(hourly_wage_emp, na.rm = TRUE),
    annual_wage_emp = sum(annual_wage_emp, na.rm = TRUE),
    tot_hourly_wage = sum(hourly_wage_bill, na.rm = TRUE),
    tot_annual_wage = sum(annual_wage_bill, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    mean_hourly_wage = if_else(
      hourly_wage_emp > 0,
      tot_hourly_wage / hourly_wage_emp,
      NA_real_
    ),
    mean_annual_wage = if_else(
      annual_wage_emp > 0,
      tot_annual_wage / annual_wage_emp,
      NA_real_
    )
  ) %>%
  select(
    -hourly_wage_emp,
    -annual_wage_emp,
    -tot_hourly_wage,
    -tot_annual_wage
  )

# Collapse occupations as well to get AEWR equivalent
oews_county_year_aewr <- oews_area_year_df %>%
  group_by(oews_state_fips, oews_county_fips, year) %>%
  summarize(
    tot_emp = sum(tot_emp, na.rm = TRUE),
    hourly_wage_emp = sum(hourly_wage_emp, na.rm = TRUE),
    annual_wage_emp = sum(annual_wage_emp, na.rm = TRUE),
    tot_hourly_wage = sum(hourly_wage_bill, na.rm = TRUE),
    tot_annual_wage = sum(annual_wage_bill, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    mean_hourly_wage = if_else(
      hourly_wage_emp > 0,
      tot_hourly_wage / hourly_wage_emp,
      NA_real_
    ),
    mean_annual_wage = if_else(
      annual_wage_emp > 0,
      tot_annual_wage / annual_wage_emp,
      NA_real_
    )
  ) %>%
  select(
    -hourly_wage_emp,
    -annual_wage_emp,
    -tot_hourly_wage,
    -tot_annual_wage
  ) %>%
  mutate(
    occ_code = "AEWR",
    occ_title = "Aggregated occupation for AEWR equivalent"
  )

# Combine
oews_county_year <- rbind(oews_county_year_occ, oews_county_year_aewr) %>%
  arrange(oews_state_fips, oews_county_fips, year, occ_code)

# Harmonize variable names
oews_county_year <- oews_county_year %>%
  rename(
    state_fips_code = oews_state_fips,
    county_fips_code = oews_county_fips,
    oews_mean_hourly_wage = mean_hourly_wage,
    oews_mean_annual_wage = mean_annual_wage,
    oews_tot_emp = tot_emp
  )

# Export
oews_county_year %>%
  write_parquet(path_int("oews_county_aggregated.parquet"))

# Collapse into state-year as well for comparison with FLS
oews_state_year_occ <- oews_area_year_df %>%
  group_by(oews_state_fips, year, occ_code, occ_title) %>%
  summarize(
    tot_emp = sum(tot_emp, na.rm = TRUE),
    hourly_wage_emp = sum(hourly_wage_emp, na.rm = TRUE),
    annual_wage_emp = sum(annual_wage_emp, na.rm = TRUE),
    tot_hourly_wage = sum(hourly_wage_bill, na.rm = TRUE),
    tot_annual_wage = sum(annual_wage_bill, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    mean_hourly_wage = if_else(
      hourly_wage_emp > 0,
      tot_hourly_wage / hourly_wage_emp,
      NA_real_
    ),
    mean_annual_wage = if_else(
      annual_wage_emp > 0,
      tot_annual_wage / annual_wage_emp,
      NA_real_
    )
  ) %>%
  select(
    -hourly_wage_emp,
    -annual_wage_emp,
    -tot_hourly_wage,
    -tot_annual_wage
  )

oews_state_year_aewr <- oews_area_year_df %>%
  group_by(oews_state_fips, year) %>%
  summarize(
    tot_emp = sum(tot_emp, na.rm = TRUE),
    hourly_wage_emp = sum(hourly_wage_emp, na.rm = TRUE),
    annual_wage_emp = sum(annual_wage_emp, na.rm = TRUE),
    tot_hourly_wage = sum(hourly_wage_bill, na.rm = TRUE),
    tot_annual_wage = sum(annual_wage_bill, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    mean_hourly_wage = if_else(
      hourly_wage_emp > 0,
      tot_hourly_wage / hourly_wage_emp,
      NA_real_
    ),
    mean_annual_wage = if_else(
      annual_wage_emp > 0,
      tot_annual_wage / annual_wage_emp,
      NA_real_
    )
  ) %>%
  select(
    -hourly_wage_emp,
    -annual_wage_emp,
    -tot_hourly_wage,
    -tot_annual_wage
  ) %>%
  mutate(
    occ_code = "AEWR",
    occ_title = "Aggregated occupation for AEWR equivalent"
  )

# Combine
oews_state_year <- rbind(oews_state_year_occ, oews_state_year_aewr) %>%
  arrange(oews_state_fips, year, occ_code)

# Harmonize variable names
oews_state_year <- oews_state_year %>%
  rename(
    state_fips_code = oews_state_fips,
    oews_mean_hourly_wage = mean_hourly_wage,
    oews_mean_annual_wage = mean_annual_wage,
    oews_tot_emp = tot_emp
  )

oews_state_year %>%
  write_parquet(path_int("oews_state_aggregated.parquet"))
