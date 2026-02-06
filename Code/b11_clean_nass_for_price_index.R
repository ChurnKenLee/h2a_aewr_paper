library(here)
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(janitor)

rm(list = ls())

#### Extract and combine variables needed for computing production quantities/prices ####
quickstats_census <- read_parquet(here("binaries", "qs_census_selected_obs.parquet"))
quickstats_survey <- read_parquet(here("binaries", "qs_survey_selected_obs.parquet"))

# We only have state-level prices
qs_survey_price <- quickstats_survey %>%
  filter(is_price) %>%
  filter(agg_level_desc == "STATE")

# Production (or area) has to be at the county level
qs_census_production <- quickstats_census %>%
  filter(is_production) %>%
  filter(agg_level_desc == "COUNTY")

qs_survey_production <- quickstats_survey %>%
  filter(is_production) %>%
  filter(agg_level_desc == "COUNTY")

# We can impute production = area*yield
# Yield units can be at the county, state, or national level
qs_survey_yield <- quickstats_survey %>%
  filter(is_yield)

# Area
qs_census_area <- quickstats_census %>%
  filter(is_area) %>%
  filter(agg_level_desc == "COUNTY")

qs_survey_area <- quickstats_survey %>%
  filter(is_area) %>%
  filter(agg_level_desc == "COUNTY")

# We want to do a full join across all relevant variables, so we can then filter to get consistent units
# This happens because some crops are reported with multiple units
# Keep only variables we need to do full joins
qs_survey_price <- qs_survey_price %>% 
  select(
    state_name, state_fips_code,
    group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
    year,
    unit_desc, reference_period_desc,
    value
    )

qs_survey_production <- qs_survey_production %>% 
  select(
    state_name, state_fips_code, county_name, county_code,
    group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
    year,
    unit_desc, reference_period_desc,
    value
  )

qs_census_production <- qs_census_production %>% 
  select(
    state_name, state_fips_code, county_name, county_code,
    group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
    year,
    unit_desc, reference_period_desc,
    value
  )

qs_survey_production <- qs_survey_production %>% 
  mutate(production_source = "qs_survey")
qs_census_production <- qs_census_production %>% 
  mutate(production_source = "qs_census")

qs_survey_area <- qs_survey_area %>% 
  select(
    state_name, state_fips_code, county_name, county_code,
    group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
    year,
    unit_desc, reference_period_desc,
    value
  )

qs_census_area <- qs_census_area %>% 
  select(
    state_name, state_fips_code, county_name, county_code,
    group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
    year,
    unit_desc, reference_period_desc,
    value
  )

qs_survey_area <- qs_survey_area %>% 
  mutate(area_source = "qs_survey")
qs_census_area <- qs_census_area %>% 
  mutate(area_source = "qs_census")

# # Create a set of all combinations of county-crop-year, then impute production and acreage
# # Impute production
# production_county_crop_years <- bind_rows(qs_survey_production, qs_census_production)
# production_county_crop_years <- production_county_crop_years %>% 
#   expand(
#     nesting(
#       state_name, state_fips_code, county_name, county_code,
#       group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc
#       ),
#     year
#   )
# 
# qs_survey_production <- full_join(production_county_crop_years, qs_survey_production)
# qs_census_production <- full_join(production_county_crop_years, qs_census_production)

production <- bind_rows(qs_survey_production, qs_census_production)
production <- production %>% 
  rename(
    production_ref_period = reference_period_desc,
    production_unit = unit_desc,
    production = value
  )

# # Impute acreage
# area_county_crop_years <- bind_rows(qs_survey_area, qs_census_area)
# area_county_crop_years <- area_county_crop_years %>% 
#   expand(
#     nesting(
#       state_name, state_fips_code, county_name, county_code,
#       group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc
#     ),
#     year
#   )
# 
# qs_survey_area <- full_join(area_county_crop_years, qs_survey_area)
# qs_census_area <- full_join(area_county_crop_years, qs_census_area)

area <- bind_rows(qs_survey_area, qs_census_area)
area <- area %>% 
  rename(
    area_ref_period = reference_period_desc,
    area_unit = unit_desc,
    area = value
  )

# # Fill in missing values
# production <- production %>% 
#   mutate(data_source = case_when(
#     !is.na(production) ~ "production observed",
#     is.na(production) ~ "production filled"
#   )) %>% 
#   arrange(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year
#   )
# 
# production <- production %>% 
#   fill(production_ref_period, production_unit, production, .direction = "downup")
# 
# area <- area %>% 
#   mutate(data_source = case_when(
#     !is.na(area) ~ "acreage observed",
#     is.na(area) ~ "acreage filled"
#   )) %>% 
#   arrange(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year
#   )
# 
# area <- area %>% 
#   fill(area_ref_period, area_unit, area, .direction = "downup")

# Add yields to acreage
# Yields are available at the national, state, and county level; have to split to merge with area
qs_survey_yield <- qs_survey_yield %>% 
  filter(reference_period_desc == "YEAR") %>% 
  select(
    state_name, state_fips_code, county_name, county_code,
    group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
    year,
    unit_desc, value,
    agg_level_desc
  )

yield_county <- qs_survey_yield %>% 
  filter(agg_level_desc == "COUNTY") %>% 
  rename(county_yield = value) %>% 
  rename(county_yield_unit = unit_desc) %>% 
  select(-agg_level_desc)

yield_state <- qs_survey_yield %>% 
  filter(agg_level_desc == "STATE") %>% 
  rename(state_yield = value) %>% 
  rename(state_yield_unit = unit_desc) %>% 
  select(-agg_level_desc, -county_name, -county_code)

yield_national <- qs_survey_yield %>% 
  filter(agg_level_desc == "NATIONAL") %>% 
  rename(national_yield = value) %>% 
  rename(national_yield_unit = unit_desc) %>% 
  select(-agg_level_desc, -state_name, -state_fips_code, -county_name, -county_code)

area_yield <- area %>% 
  left_join(yield_county)
area_yield <- area_yield %>% 
  left_join(yield_state)
area_yield <- area_yield %>% 
  left_join(yield_national)

# All yield units are in terms of acres, so we can just directly multiply to get imputed production
area_yield <- area_yield %>% 
  mutate(yield = case_when(
    !is.na(county_yield) ~ county_yield,
    is.na(county_yield) & !is.na(state_yield) ~ state_yield,
    is.na(county_yield) & is.na(state_yield) ~ national_yield
  )) %>% 
  mutate(yield_unit = case_when(
    !is.na(county_yield) ~ county_yield_unit,
    is.na(county_yield) & !is.na(state_yield) ~ state_yield_unit,
    is.na(county_yield) & is.na(state_yield) ~ national_yield_unit
  ))

# # Fill in yield for observations with no yield information
# area_yield <- area_yield %>% 
#   mutate(data_source = case_when(
#     !is.na(yield) ~ "state/national yield observed",
#     is.na(yield) ~ "yield filled",
#     !is.na(area) ~ "area observed",
#     is.na(area) ~ "area filled",
#   )) %>% 
#   arrange(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year
#   )
# 
# area_yield <- area_yield %>% 
#   fill(yield, yield_unit, area, area_unit, .direction = "downup")

#### Harmonize units between price, production, and imputed production ####
# We only want specialty crops
production <- production %>% 
  filter(group_desc %in% c("FRUIT & TREE NUTS", "VEGETABLES", "HORTICULTURE"))
area_yield <- area_yield %>% 
  filter(group_desc %in% c("FRUIT & TREE NUTS", "VEGETABLES", "HORTICULTURE"))

# Full set of available production estimates
production_area_yield <- production %>% 
  full_join(area_yield)

# Add whatever price data we have from NASS surveys
price <- qs_survey_price %>% 
  mutate(price_unit = unit_desc) %>% 
  mutate(price = value) %>% 
  mutate(price_reference_period = reference_period_desc) %>% 
  select(-unit_desc, -value, -reference_period_desc)

combined <- production_area_yield %>% 
  full_join(price)

# Since all yield units are in terms of acres, we can just directly multiply with acreage
test <- combined  %>% 
  distinct(production_unit, area_unit, yield_unit)

combined  <- combined  %>%
  mutate(imputed_production = area*yield)

# Add units for imputed production
combined  <- combined  %>%
  mutate(imputed_production_unit = str_extract(yield_unit, '^[A-Z]*')) %>%
  mutate(imputed_production_unit = case_when(
    yield_unit == "TONS / ACRE, DRY BASIS" ~ "TONS, DRY BASIS",
    yield_unit == "TONS / ACRE, FRESH BASIS" ~ "TONS, FRESH BASIS",
    yield_unit == "LB / ACRE, CHERRY BASIS" ~ "LB, CHERRY BASIS",
    .default = imputed_production_unit
  ))

# Keep only rows that have at least observed price
combined  <- combined  %>% 
  filter(!is.na(price))

# Choose desired final production sources
combined <- combined  %>% 
  mutate(impute_production_here = (is.na(production) & !is.na(imputed_production))) %>% 
  mutate(production = if_else(impute_production_here, imputed_production, production)) %>% 
  mutate(production_unit = if_else(impute_production_here, imputed_production_unit, production_unit))

# Harmonize price and production
# See list of all units in data
production_price_units <- combined %>% 
  distinct(price_unit, production_unit, .keep_all = TRUE) %>% 
  filter(!is.na(price_unit) & !is.na(production_unit)) %>% 
  select(commodity_desc, class_desc, prodn_practice_desc, util_practice_desc, price_unit, production_unit) %>% 
  arrange(price_unit, production_unit, commodity_desc, class_desc)

# Adjust production units to match price units, when conversion is just between weights
combined <- combined %>% 
  mutate(
    production = case_when(
      price_unit == "$ / LB" & production_unit == "CWT" ~ production*112, # cwt to lb
      price_unit == "$ / LB" & production_unit == "TONS" ~ production*2000, # tons to lb
      price_unit == "$ / CWT" & production_unit == "LB" ~ production/112, # lb to cwt
      price_unit == "$ / CWT" & production_unit == "TONS" ~ production*2000/112, # tons to cwt
      price_unit == "$ / TON" & production_unit == "LB" ~ production/2000, # lb to tons,
      .default = production
      )
    ) %>% 
  mutate(
    production_unit = case_when(
      price_unit == "$ / LB" & production_unit == "CWT" ~ "LB", # cwt to lb
      price_unit == "$ / LB" & production_unit == "TONS" ~ "LB", # tons to lb
      price_unit == "$ / CWT" & production_unit == "LB" ~ "CWT", # lb to cwt
      price_unit == "$ / CWT" & production_unit == "TONS" ~ "CWT", # tons to cwt
      price_unit == "$ / TON" & production_unit == "LB" ~ "TONS", # lb to tons,
      .default = production_unit
      )
    )

# Convert from bushels to weight when needed
combined <- combined %>% 
  mutate(
    production = case_when(
      # Cranberries: 1 barrel = 100 lb; 1 bushel = 32 lbs
      commodity_desc == "CRANBERRIES" & price_unit == "$ / BARREL" & production_unit == "BU" ~ production*32/100, 
      commodity_desc == "CRANBERRIES" & price_unit == "$ / BARREL" & production_unit == "LB" ~ production/100,
      commodity_desc == "CRANBERRIES" & price_unit == "$ / BARREL" & production_unit == "CWT" ~ production*112/32,
      commodity_desc == "CRANBERRIES" & price_unit == "$ / BARREL" & production_unit == "TONS" ~ production*2000/100,
      # Potatoes: 1 bushel = 60 lbs
      commodity_desc == "POTATOES" & price_unit == "$ / CWT" & production_unit == "BU" ~ production*60/112,
      # Sweet potatoes: 1 bushel = 55 lbs
      commodity_desc == "SWEET POTATOES" & price_unit == "$ / CWT" & production_unit == "BU" ~ production*55/112,
      # Apples: 1 bushel = 40 lbs
      commodity_desc == "APPLES" & price_unit == "$ / LB" & production_unit == "BU" ~ production*40,
      # Peaches: 1 bushel = 48 lbs
      commodity_desc == "PEACHES" & price_unit == "$ / TON" & production_unit == "BU" ~ production*48/2000,
      # Chile peppers: 1 bushel = 25-30 lbs
      commodity_desc == "PEPPERS" & class_desc == "CHILE" & price_unit == "$ / CWT" & production_unit == "BU" ~ production*27.5/112,
      # Green peas: 1 bushel = 28 lbs
      commodity_desc == "PEAS" & class_desc == "GREEN" & price_unit == "$ / TON" & production_unit == "BU" ~ production*28/2000,
      .default = production
      )
    ) %>% 
  mutate(
    production_unit = case_when(
      # Cranberries: 1 barrel = 100 lb; 1 bushel = 32 lbs
      commodity_desc == "CRANBERRIES" & price_unit == "$ / BARREL" & production_unit == "BU" ~ "BARRELS", 
      commodity_desc == "CRANBERRIES" & price_unit == "$ / BARREL" & production_unit == "LB" ~ "BARRELS",
      commodity_desc == "CRANBERRIES" & price_unit == "$ / BARREL" & production_unit == "CWT" ~ "BARRELS",
      commodity_desc == "CRANBERRIES" & price_unit == "$ / BARREL" & production_unit == "TONS" ~ "BARRELS",
      # Potatoes: 1 bushel = 60 lbs
      commodity_desc == "POTATOES" & price_unit == "$ / CWT" & production_unit == "BU" ~ "CWT",
      # Sweet potatoes: 1 bushel = 50 lbs
      commodity_desc == "SWEET POTATOES" & price_unit == "$ / CWT" & production_unit == "BU" ~ "CWT",
      # Apples: 1 bushel = 40 lbs
      commodity_desc == "APPLES" & price_unit == "$ / LB" & production_unit == "BU" ~ "LB",
      # Peaches: 1 bushel = 48 lbs
      commodity_desc == "PEACHES" & price_unit == "$ / TON" & production_unit == "BU" ~ "TONS",
      # Chile peppers: 1 bushel = 25-30 lbs
      commodity_desc == "PEPPERS" & class_desc == "CHILE" & price_unit == "$ / CWT" & production_unit == "BU" ~ "CWT",
      # Green peas: 1 bushel = 28 lbs
      commodity_desc == "PEAS" & class_desc == "GREEN" & price_unit == "$ / TON" & production_unit == "BU" ~ 'TONS',
      .default = production_unit
      )
    )

# Drop observations with inconsistent unharmonizable units
combined <- combined %>% 
  filter(!(price_unit == "$ / CWT" & production_unit == "GALLONS")) %>% 
  filter(!(price_unit == "$ / LB" & production_unit == "BOXES")) %>% 
  filter(!(price_unit == "$ / LB" & production_unit == "BU")) %>% 
  filter(!(price_unit == "$ / TON" & production_unit == "BARRELS")) %>% 
  filter(!(price_unit == "$ / LB" & production_unit == "TONS, DRY BASIS")) %>% 
  filter(!(price_unit == "$ / LB" & production_unit == "TONS, FRESH BASIS")) %>% 
  filter(!(price_unit == "$ / TON" & production_unit == "BOXES")) %>% 
  filter(!(price_unit == "$ / TON" & production_unit == "BU")) %>% 
  filter(!(price_unit == "$ / TON" & production_unit == "TONS, DRY BASIS")) %>% 
  filter(!(price_unit == "$ / TON, FRESH BASIS" & production_unit == "TONS, DRY BASIS")) %>% 
  filter(!(price_unit == "$ / TON, DRY BASIS" & production_unit == "TONS")) %>% 
  filter(!(price_unit == "$ / LB, CHERRY BASIS" & production_unit == "LB")) %>% 
  filter(!(price_unit == "$ / LB, CHERRY BASIS" & production_unit == "TONS")) %>% 
  filter(!(price_unit == "$ / LB, GREEN BASIS" & production_unit == "LB")) %>% 
  filter(!(price_unit == "$ / LB, GREEN BASIS" & production_unit == "TONS"))

# Check units again
production_price_units <- combined %>% 
  distinct(price_unit, production_unit, .keep_all = TRUE) %>% 
  filter(!is.na(price_unit) & !is.na(production_unit)) %>% 
  select(commodity_desc, class_desc, prodn_practice_desc, util_practice_desc, price_unit, production_unit) %>% 
  arrange(price_unit, production_unit, commodity_desc, class_desc)

# Fill in missing production when needed
combined <- combined %>%
  mutate(data_source = case_when(
    !is.na(production) ~ "production observed",
    is.na(production) ~ "production filled"
  )) %>%
  arrange(
    state_name, state_fips_code, county_name, county_code,
    group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
    year
  )

combined <- combined %>%
  fill(production_unit, production, .direction = "downup")

# Export
combined %>%
  write_parquet(here("files_for_phil", "county_price_output.parquet")) %>%
  write_parquet(here("binaries", "county_price_output.parquet"))
