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

























# #### OLD CODE ####
# 
# production_price <- production_price %>% 
#   mutate(production = case_when(
#     commodity_desc == "SORGHUM" & price_unit == "$ / CWT" & production_unit == "BU" ~ production*56/112, # USDA uses 1 bushel = 56 lb for sorghum
#     price_unit == "$ / TON" & production_unit == "LB" ~ production/2000, # lb to ton
#     price_unit == "$ / CWT" & production_unit == "LB" ~ production/112, # lb to cwt
#     .default = production
#   )) %>% 
#   mutate(production_unit = case_when(
#     commodity_desc == "SORGHUM" & price_unit == "$ / CWT" & production_unit == "BU" ~ "CWT", # USDA uses 1 bushel = 56 lb for sorghum
#     price_unit == "$ / TON" & production_unit == "LB" ~ "TON", # lb to ton
#     price_unit == "$ / TON" & production_unit == "LB" ~ "CWT", # lb to cwt
#     .default = production_unit
# 
# # Adjust units for consistency
# # Adjust for actual production values
# production_price <- production_price %>%
#   mutate(adjusted_production_value = case_when(
#     county_production_unit == "LB" & state_price_unit == "$ / CWT" ~ county_production_value/112, # lb to cwt
#     county_production_unit == "LB" & state_price_unit == "$ / TON" ~ county_production_value/2000, # lb to ton,
#     county_production_unit == "BALES" & state_price_unit == "$ / LB" & commodity_desc == "COTTON" ~ county_production_value*480, # bale of cotton to lb
#     .default = county_production_value
#   )) %>%
#   mutate(adjusted_production_unit = case_when(
#     county_production_unit == "LB" & state_price_unit == "$ / CWT" ~ "CWT", # lb to cwt
#     county_production_unit == "LB" & state_price_unit == "$ / TON" ~ "TON", # lb to ton,
#     county_production_unit == "BALES" & state_price_unit == "$ / LB" & commodity_desc == "COTTON" ~ "LB", # bale of cotton to lb
#     .default = county_production_unit
#   ))
# 
# # Adjust for imputed production values
# imputed_production_price_units <- combined %>%
#   distinct(imputed_production_unit, state_price_unit)
# 
# combined <- combined %>%
#   mutate(adjusted_imputed_production_value = case_when(
#     imputed_production_unit == "LB" & state_price_unit == "$ / CWT" ~ imputed_production_value/112, # lb to cwt
#     imputed_production_unit == "TONS" & state_price_unit == "$ / LB" ~ imputed_production_value*2000, # ton to lb
#     imputed_production_unit == "BU" & state_price_unit == "$ / LB" & commodity_desc == "APPLES" ~ imputed_production_value*40, # BU to LB for apples
#     imputed_production_unit == "BU" & state_price_unit == "$ / LB" & commodity_desc == "PEACHES" ~ imputed_production_value*50, # BU to LB for peaches
#     imputed_production_unit == "BU" & state_price_unit == "$ / TON" & commodity_desc == "PEACHES" ~ imputed_production_value*0.025, # BU to TON for peaches
#     .default = imputed_production_value
#   )) %>%
#   mutate(adjusted_imputed_production_unit = case_when(
#     imputed_production_unit == "LB" & state_price_unit == "$ / CWT" ~ "CWT", # lb to cwt
#     imputed_production_unit == "TONS" & state_price_unit == "$ / LB" ~ "LB", # ton to lb
#     imputed_production_unit == "BU" & state_price_unit == "$ / LB" & commodity_desc == "APPLES" ~ "LB", # BU to LB for apples
#     imputed_production_unit == "BU" & state_price_unit == "$ / LB" & commodity_desc == "PEACHES" ~ "LB", # BU to LB for peaches
#     imputed_production_unit == "BU" & state_price_unit == "$ / TON" & commodity_desc == "PEACHES" ~ "TON", # BU to TON for peaches
#     .default = imputed_production_unit
#   ))
# 
# # Check consistency of units again
# production_price_units <- combined %>%
#   distinct(adjusted_production_unit, state_price_unit)
# 
# imputed_production_price_units <- combined %>%
#   distinct(adjusted_imputed_production_unit, state_price_unit)
# 
# #### Merge price and production
# production <- bind_rows(qs_survey_production, qs_census_production, census_production)
# 
# production <- production %>% 
#   rename(
#     production_value = value,
#     production_unit = unit_desc,
#     production_ref_period = reference_period_desc
#     )
# 
# qs_survey_price <- qs_survey_price %>% 
#   rename(
#     price_value = value,
#     price_unit = unit_desc,
#     price_ref_period = reference_period_desc
#   )
# 
# production_price <- production %>% 
#   full_join(qs_survey_price)
# 
# # Drop obs with no price or production
# production_price <- production_price %>% 
#   filter(!is.na(production_unit) & !is.na(price_unit))
# 
# production_price <- production_price %>% 
#   mutate(price_weight = str_extract(price_unit, "(?<=/ ).*"))
# 
# # Check units
# test <- production_price %>% 
#   distinct(production_unit, price_unit)
# 
# # Now select an observation for each county-crop-year
# production_source_priority <- c(
#   "qs_survey",
#   "qs_census",
#   "census"
# )
# 
# selected_production_price <- production_price %>% 
#   group_by(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year) %>%
#   arrange(
#     match(production_source, production_source_priority),
#     production_ref_period == price_ref_period,
#   ) %>%
#   slice_head(n = 1) %>%
#   ungroup()
# 
# test <- selected_production_price %>% 
#   group_by(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year) %>% 
#   mutate(n_prices = n()) %>% 
#   ungroup() %>% 
#   arrange(n_prices, group_desc, commodity_desc, class_desc)
# 
# # Now impute production using acreage*yield
# qs_survey_area <- qs_survey_area %>% 
#   select(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year,
#     unit_desc, reference_period_desc,
#     value
#   )
# 
# qs_census_area <- qs_census_area %>% 
#   select(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year,
#     unit_desc, reference_period_desc,
#     value
#   )
# 
# census_area <- census_area %>% 
#   select(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year,
#     unit_desc, reference_period_desc,
#     value
#   )
# 
# qs_survey_yield <- qs_survey_yield %>% 
#   select(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year,
#     unit_desc, reference_period_desc,
#     value
#   )
# 
# qs_survey_area <- qs_survey_area %>% 
#   mutate(area_source = "qs_survey")
# 
# qs_census_area <- qs_census_area %>% 
#   mutate(area_source = "qs_census")
# 
# census_area <- census_area %>% 
#   mutate(area_source = "census")
# 
# area <- bind_rows(qs_survey_area, qs_census_area, census_area)
# 
# area <- area %>% 
#   rename(
#     area_value = value,
#     area_unit = unit_desc,
#     area_ref_period = reference_period_desc
#   )
# 
# qs_survey_yield <- qs_survey_yield %>% 
#   rename(
#     yield_value = value,
#     yield_unit = unit_desc,
#     yield_ref_period = reference_period_desc
#   )
# 
# area_yield <- area %>% 
#   full_join(qs_survey_yield)
# 
# # Drop obs with no price or or production
# production_price <- production_price %>% 
#   filter(!is.na(production_unit) & !is.na(price_unit))
# 
# production_price <- production_price %>% 
#   mutate(price_weight = str_extract(price_unit, "(?<=/ ).*"))
# 
# # Check units
# test <- production_price %>% 
#   distinct(production_unit, price_unit)
# 
# # Check number of price observations per crop
# test <- qs_survey_price %>% 
#   distinct(group_desc, commodity_desc, class_desc, unit_desc, reference_period_desc) %>% 
#   group_by(group_desc, commodity_desc, class_desc) %>% 
#   mutate(n_price_units = n()) %>% 
#   ungroup() %>% 
#   arrange(n_price_units, group_desc, commodity_desc, class_desc)
# 
# test <- qs_survey_price %>% 
#   group_by(state_name, state_fips_code, 
#            group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc, 
#            year) %>% 
#   arrange(unit_desc) %>% 
#   mutate(nth = 1:n()) %>% 
#   ungroup() %>% 
#   arrange(state_name, state_fips_code, 
#           group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc, 
#           year, unit_desc)
# 
# # Define selection priority for reference_period_desc and unit_desc, to select a unique obs for each county-crop-year
# reference_period_priority <- c(
#   "YEAR",
#   "MARKETING YEAR"
# )
# 
# crop_price_units <- qs_survey_price %>% 
#   distinct(group_desc, commodity_desc, class_desc, unit_desc)
# 
# # Unique prices
# price_unit_priority <- c(
#   "$ / BASKET",
#   "$ / BLOOM",
#   "$ / BOX, PHD EQUIV",
#   "$ / BOX, ON TREE EQUIV",
#   "$ / BUNCH",
#   "$ / FLAT",
#   "$ / LB",
#   "$ / LB, CHERRY BASIS",
#   "$ / POT",
#   "$ / SPIKE",
#   "$ / STEM",
#   "$ / TON",
#   "$ / TON, FRESH BASIS",
#   "CENTS / BLOOM",
#   "CENTS / SPIKE",
#   "CENTS / STEM"
# )
# 
# selected_survey_price_obs <- qs_survey_price %>%
#   group_by(
#     state_name, state_fips_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year) %>%
#   arrange(
#     match(reference_period_desc, reference_period_priority),
#     match(unit_desc, price_unit_priority)
#   ) %>%
#   slice_head(n = 1) %>%
#   ungroup()
# 
# # Unique production
# survey_production_unit_priority <- c(
#   "LB",
#   "CWT",
#   "TONS",
#   "BU"
# )
# 
# census_production_unit_priority <- c(
#   "LB",
#   "CWT",
#   "TONS",
#   "GALLONS",
#   "BU",
#   "BALES"
# )
# 
# selected_survey_production_obs <- qs_survey_production %>%
#   group_by(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year) %>%
#   arrange(
#     match(reference_period_desc, reference_period_priority),
#     match(unit_desc, survey_production_unit_priority)
#   ) %>%
#   slice_head(n = 1) %>%
#   ungroup()
# 
# selected_census_production_obs <- census_production %>%
#   group_by(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year) %>%
#   arrange(
#     match(unit_desc, census_production_unit_priority)
#   ) %>%
#   slice_head(n = 1) %>%
#   ungroup()
# 
# selected_qs_census_production_obs <- qs_census_production %>%
#   group_by(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year) %>%
#   arrange(
#     match(unit_desc, census_production_unit_priority)
#   ) %>%
#   slice_head(n = 1) %>%
#   ungroup()
# 
# # Unique area
# survey_area_stat_priority <- c(
#   "AREA PLANTED",
#   "AREA BEARING",
#   "AREA HARVESTED",
#   "AREA PLANTED, NET"
# )
# 
# census_area_stat_priority <- c(
#   "AREA BEARING & NON-BEARING",
#   "AREA GROWN",
#   "AREA BEARING",
#   "AREA HARVESTED"
# )
# 
# selected_survey_area_obs <- qs_survey_area %>%
#   group_by(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year) %>%
#   arrange(
#     match(statisticcat_desc, survey_area_stat_priority)
#   ) %>%
#   slice_head(n = 1) %>%
#   ungroup()
# 
# selected_qs_census_area_obs <- qs_census_area %>%
#   filter(statisticcat_desc != "AREA NOT HARVESTED") %>%
#   group_by(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year) %>%
#   arrange(
#     match(statisticcat_desc, census_area_stat_priority)
#   ) %>%
#   slice_head(n = 1) %>%
#   ungroup()
# 
# selected_census_area_obs <- census_area %>%
#   filter(statisticcat_desc != "AREA NOT HARVESTED") %>%
#   group_by(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year) %>%
#   arrange(
#     match(statisticcat_desc, census_area_stat_priority)
#   ) %>%
#   slice_head(n = 1) %>%
#   ungroup()
# 
# # Unique yield
# yield_reference_period_priority <- c(
#   "YEAR",
#   "YEAR - DEC FORECAST",
#   "YEAR - NOV FORECAST",
#   "YEAR - OCT FORECAST",
#   "YEAR - SEP FORECAST",
#   "YEAR - AUG FORECAST",
#   "YEAR - JUL FORECAST",
#   "YEAR - JUN FORECAST",
#   "YEAR - FEB FORECAST",
#   "YEAR - JAN FORECAST"
# )
# 
# yield_unit_priority <- c(
#   "BU / ACRE",
#   "BU / PLANTED ACRE",
#   "LB / ACRE",
#   "TONS / ACRE",
#   "TONS / ACRE, FRESH BASIS"
# )
# 
# selected_survey_national_yield_obs <- qs_survey_yield %>%
#   filter(agg_level_desc == "NATIONAL") %>%
#   group_by(
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year) %>%
#   arrange(
#     match(reference_period_desc, yield_reference_period_priority),
#     match(unit_desc, yield_unit_priority),
#   ) %>%
#   slice_head(n = 1) %>%
#   ungroup()
# 
# selected_survey_state_yield_obs <- qs_survey_yield %>%
#   filter(agg_level_desc == "STATE") %>%
#   group_by(
#     state_name, state_fips_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year) %>%
#   arrange(
#     match(reference_period_desc, yield_reference_period_priority),
#     match(unit_desc, yield_unit_priority),
#   ) %>%
#   slice_head(n = 1) %>%
#   ungroup()
# 
# selected_survey_county_yield_obs <- qs_survey_yield %>%
#   filter(agg_level_desc == "COUNTY") %>%
#   group_by(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year) %>%
#   arrange(
#     match(reference_period_desc, yield_reference_period_priority),
#     match(unit_desc, yield_unit_priority),
#   ) %>%
#   slice_head(n = 1) %>%
#   ungroup()
# 
# # Check uniqueness of obs
# test <- selected_survey_price_obs %>%
#   group_by(
#     state_name, state_fips_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year
#   ) %>%
#   mutate(nobs = n()) %>%
#   ungroup() %>%
#   filter(nobs > 1)
# 
# test <- selected_survey_production_obs %>%
#   group_by(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year
#   ) %>%
#   mutate(nobs = n()) %>%
#   ungroup() %>%
#   filter(nobs > 1)
# 
# test <- selected_qs_census_production_obs %>%
#   group_by(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year
#   ) %>%
#   mutate(nobs = n()) %>%
#   ungroup() %>%
#   filter(nobs > 1)
# 
# test <- selected_census_production_obs %>%
#   group_by(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year
#   ) %>%
#   mutate(nobs = n()) %>%
#   ungroup() %>%
#   filter(nobs > 1)
# 
# test <- selected_survey_area_obs %>%
#   group_by(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year
#   ) %>%
#   mutate(nobs = n()) %>%
#   ungroup() %>%
#   filter(nobs > 1)
# 
# test <- selected_qs_census_area_obs %>%
#   group_by(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year
#   ) %>%
#   mutate(nobs = n()) %>%
#   ungroup() %>%
#   filter(nobs > 1)
# 
# test <- selected_census_area_obs %>%
#   group_by(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year
#   ) %>%
#   mutate(nobs = n()) %>%
#   ungroup() %>%
#   filter(nobs > 1)
# 
# test <- selected_survey_county_yield_obs %>%
#   group_by(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year
#   ) %>%
#   mutate(nobs = n()) %>%
#   ungroup() %>%
#   filter(nobs > 1)
# 
# test <- selected_survey_state_yield_obs %>%
#   group_by(
#     state_name, state_fips_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year
#   ) %>%
#   mutate(nobs = n()) %>%
#   ungroup() %>%
#   filter(nobs > 1)
# 
# test <- selected_survey_national_yield_obs %>%
#   group_by(
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year
#   ) %>%
#   mutate(nobs = n()) %>%
#   ungroup() %>%
#   filter(nobs > 1)
# 
# # All obs are unique
# 
# #### Combine price, production, area, yield ####
# # Get set of county-crop-years
# qs_survey_county_crop_years <- quickstats_survey %>%
#   distinct(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year
#     )
# 
# qs_census_county_crop_years <- quickstats_census %>%
#   distinct(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year
#   )
# 
# census_county_crop_years <- census %>%
#   distinct(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year
#   )
# 
# county_crop_years <- bind_rows(qs_survey_county_crop_years, qs_census_county_crop_years, census_county_crop_years) %>%
#   distinct() %>%
#   mutate(across(as.character())) %>%
#   arrange(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year
#     )
# 
# test <- county_crop_years %>% distinct(state_name)
# 
# # Rename variables for merging
# price_data <- selected_survey_price_obs %>%
#   mutate(
#     price = value,
#     price_unit = unit_desc
#   ) %>%
#   select(
#     state_name, state_fips_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year,
#     price, price_unit
#   )
# 
# qs_survey_production_data <- selected_survey_production_obs %>%
#   mutate(
#     qs_survey_production = value,
#     qs_survey_production_unit = unit_desc
#   ) %>%
#   select(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year,
#     qs_survey_production, qs_survey_production_unit
#   )
# 
# qs_census_production_data <- selected_qs_census_production_obs %>%
#   mutate(
#     qs_census_production = value,
#     qs_census_production_unit = unit_desc
#   ) %>%
#   select(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year,
#     qs_census_production, qs_census_production_unit
#   )
# 
# census_production_data <- selected_census_production_obs %>%
#   mutate(
#     census_production = value,
#     census_production_unit = unit_desc
#   ) %>%
#   select(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year,
#     census_production, census_production_unit
#   )
# 
# qs_survey_area_data <- selected_survey_area_obs %>%
#   mutate(
#     qs_survey_area = value,
#     qs_survey_area_unit = unit_desc
#   ) %>%
#   select(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year,
#     qs_survey_area, qs_survey_area_unit
#   )
# 
# qs_census_area_data <- selected_qs_census_area_obs %>%
#   mutate(
#     qs_census_area = value,
#     qs_census_area_unit = unit_desc
#   ) %>%
#   select(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year,
#     qs_census_area, qs_census_area_unit
#   )
# 
# census_area_data <- selected_census_area_obs %>%
#   mutate(
#     census_area = value,
#     census_area_unit = unit_desc
#   ) %>%
#   select(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year,
#     census_area, census_area_unit
#   )
# 
# national_yield_data <- selected_survey_national_yield_obs %>%
#   mutate(
#     national_yield = value,
#     national_yield_unit = unit_desc
#   ) %>%
#   select(
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year,
#     national_yield, national_yield_unit
#   )
# 
# state_yield_data <- selected_survey_state_yield_obs %>%
#   mutate(
#     state_yield = value,
#     state_yield_unit = unit_desc
#   ) %>%
#   select(
#     state_name, state_fips_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year,
#     state_yield, state_yield_unit
#   )
# 
# county_yield_data <- selected_survey_county_yield_obs %>%
#   mutate(
#     county_yield = value,
#     county_yield_unit = unit_desc
#   ) %>%
#   select(
#     state_name, state_fips_code, county_name, county_code,
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc,
#     year,
#     county_yield, county_yield_unit
#   )
# 
# # Merge
# combined <- county_crop_years %>%
#   full_join(price_data)
# combined <- combined %>%
#   full_join(qs_survey_production_data)
# combined <- combined %>%
#   full_join(qs_census_production_data)
# combined <- combined %>%
#   full_join(census_production_data)
# combined <- combined %>%
#   full_join(qs_survey_area_data)
# combined <- combined %>%
#   full_join(qs_census_area_data)
# combined <- combined %>%
#   full_join(census_area_data)
# combined <- combined %>%
#   full_join(national_yield_data)
# combined <- combined %>%
#   full_join(state_yield_data)
# combined <- combined %>%
#   full_join(county_yield_data)
# 
# combined %>% write_parquet(here("combined.parquet"))
# 
# #### Harmonize combined data ####
# # Construct maximum coverage of price, production, acreage, yield
# combined <- read_parquet(here("combined.parquet"))
# 
# combined <- combined %>% 
#   mutate(production = case_when(
#     !is.na(qs_survey_production) ~ qs_survey_production,
#     is.na(qs_survey_production) & !is.na(qs_census_production) ~ qs_census_production,
#     is.na(qs_survey_production) & is.na(qs_census_production) & !is.na(census_production) ~ census_production
#   )) %>% 
#   mutate(production_unit = case_when(
#     !is.na(qs_survey_production) ~ qs_survey_production_unit,
#     is.na(qs_survey_production) & !is.na(qs_census_production) ~ qs_census_production_unit,
#     is.na(qs_survey_production) & is.na(qs_census_production) & !is.na(census_production) ~ census_production_unit
#   ))
# 
# combined <- combined %>% 
#   mutate(area = case_when(
#     !is.na(qs_survey_area) ~ qs_survey_area,
#     is.na(qs_survey_area) & !is.na(qs_census_area) ~ qs_census_area,
#     is.na(qs_survey_area) & is.na(qs_census_area) & !is.na(census_area) ~ census_area
#   )) %>% 
#   mutate(area_unit = case_when(
#     !is.na(qs_survey_area) ~ qs_survey_area_unit,
#     is.na(qs_survey_area) & !is.na(qs_census_area) ~ qs_census_area_unit,
#     is.na(qs_survey_area) & is.na(qs_census_area) & !is.na(census_area) ~ census_area_unit
#   ))
# 
# combined <- combined %>% 
#   mutate(yield = case_when(
#     !is.na(county_yield) ~ county_yield,
#     is.na(county_yield) & !is.na(state_yield) ~ state_yield,
#     is.na(county_yield) & is.na(state_yield) & !is.na(national_yield) ~ national_yield
#   )) %>% 
#   mutate(yield_unit = case_when(
#     !is.na(county_yield) ~ county_yield_unit,
#     is.na(county_yield) & !is.na(state_yield) ~ state_yield_unit,
#     is.na(county_yield) & is.na(state_yield) & !is.na(national_yield) ~ national_yield_unit
#   ))
# 
# # Acreage is measured only every five years from the ag census
# # Impute missing years for county acreage
# years <- combined %>% distinct(year)
# 
# # Define the groups we want to impute within, i.e., each county-crop
# combined <- combined %>%
#   group_by(
#     state_fips_code, state_name, county_code, county_name, 
#     group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc) %>%
#   mutate(county_crop_group_id = cur_group_id()) %>%
#   ungroup()
# 
# combined <- combined %>%
#   mutate(acreage_is_imputed = is.na(area)) %>%
#   group_by(county_crop_group_id) %>%
#   fill(
#     area, area_unit, .direction = "downup") %>%
#   ungroup()
# 
# # Harmonize price and production
# price_production_units <- combined %>% 
#   distinct(price_unit, production_unit, .keep_all = TRUE) %>% 
#   filter(!is.na(price_unit) & !is.na(production_unit)) %>% 
#   select(commodity_desc, class_desc, price_unit, production_unit)
# 
# # Adjust production units to match price units
# combined <- combined %>% 
#   mutate(production = case_when(
#     commodity_desc == "SORGHUM" & price_unit == "$ / CWT" & production_unit == "BU" ~ production*56/112, # USDA uses 1 bushel = 56 lb for sorghum
#     price_unit == "$ / TON" & production_unit == "LB" ~ production/2000, # lb to ton
#     price_unit == "$ / CWT" & production_unit == "LB" ~ production/112, # lb to cwt
#     .default = production
#   )) %>% 
#   mutate(production_unit = case_when(
#     commodity_desc == "SORGHUM" & price_unit == "$ / CWT" & production_unit == "BU" ~ "CWT", # USDA uses 1 bushel = 56 lb for sorghum
#     price_unit == "$ / TON" & production_unit == "LB" ~ "TON", # lb to ton
#     price_unit == "$ / TON" & production_unit == "LB" ~ "CWT", # lb to cwt
#     .default = production_unit
#   ))
# 
# # Impute production = area*yield
# # Harmonize units if necessary
# # Check acreage and yield units
# area_yield_units <- combined %>% 
#   distinct(area_unit, yield_unit, .keep_all = TRUE) %>% 
#   filter(!is.na(area_unit) & !is.na(yield_unit)) %>% 
#   select(commodity_desc, class_desc, area_unit, yield_unit)
# 
# 
# # All yield units are in terms of acre, so we can just directly multiply to get imputed production
# combined <- combined %>%
#   mutate(imputed_production = area*yield)
# 
# # Add units for imputed production
# combined <- combined %>%
#   mutate(imputed_production_unit = str_extract(yield_unit, '^[A-Z]*')) %>%
#   mutate(imputed_production_unit = case_when(
#     yield_unit == "TONS / ACRE, DRY BASIS" ~ "TONS, DRY BASIS",
#     yield_unit == "TONS / ACRE, FRESH BASIS" ~ "TONS, FRESH BASIS",
#     .default = imputed_production_unit
#   ))
# 
# # Check consistency of price and imputed production units
# price_imputed_production_units <- combined %>%
#   distinct(price_unit, imputed_production_unit, .keep_all = TRUE) %>% 
#   filter(!is.na(price_unit) & !is.na(imputed_production_unit))
# 
# 
# 
# %>% 
#   select(commodity_desc, class_desc, price_unit, imputed_production_unit)
# 
# # Adjust imputed production units to match price units
# combined <- combined %>% 
#   mutate(imputed_production = case_when(
#     commodity_desc == "PEACHES" & price_unit == "$ / TON" & imputed_production_unit == "BU" ~ production*50/2000, # USDA uses 1 bushel = 50 lb for peaches
#     commodity_desc == "SORGHUM" & price_unit == "$ / CWT" & imputed_production_unit == "BU" ~ production*56/112, # USDA uses 1 bushel = 56 lb for sorghum
#     .default = production
#   )) %>% 
#   mutate(imputed_production_unit = case_when(
#     commodity_desc == "PEACHES" & price_unit == "$ / TON" & imputed_production_unit == "BU" ~ "TON", # USDA uses 1 bushel = 50 lb for peaches
#     commodity_desc == "SORGHUM" & price_unit == "$ / CWT" & imputed_production_unit == "BU" ~ "CWT", # USDA uses 1 bushel = 56 lb for sorghum
#     .default = production_unit
#   ))
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# # Harmonize units if necessary
# # Check acreage and yield units
# acreage_yield_units <- combined %>%
#   distinct(county_acreage_unit, yield_unit)
# 
# # All yield units are in terms of acre, so we can just directly multiply to get imputed production
# combined <- combined %>%
#   mutate(imputed_production_value = county_acreage_value*yield_value)
# 
# # Add units for imputed production
# combined <- combined %>%
#   mutate(imputed_production_unit = str_extract(yield_unit, '^[A-Z]*')) %>%
#   mutate(imputed_production_unit = case_when(
#     yield_unit == "TONS / ACRE, DRY BASIS" ~ "TONS, DRY BASIS",
#     .default = imputed_production_unit
#   ))
# 
# # Check consistency of price and quantity units
# production_price_units <- combined %>%
#   distinct(county_production_unit, state_price_unit)
# 
# # Adjust units for consistency
# # Adjust for actual production values
# combined <- combined %>%
#   mutate(adjusted_production_value = case_when(
#     county_production_unit == "LB" & state_price_unit == "$ / CWT" ~ county_production_value/112, # lb to cwt
#     county_production_unit == "LB" & state_price_unit == "$ / TON" ~ county_production_value/2000, # lb to ton,
#     county_production_unit == "BALES" & state_price_unit == "$ / LB" & commodity_desc == "COTTON" ~ county_production_value*480, # bale of cotton to lb
#     .default = county_production_value
#   )) %>%
#   mutate(adjusted_production_unit = case_when(
#     county_production_unit == "LB" & state_price_unit == "$ / CWT" ~ "CWT", # lb to cwt
#     county_production_unit == "LB" & state_price_unit == "$ / TON" ~ "TON", # lb to ton,
#     county_production_unit == "BALES" & state_price_unit == "$ / LB" & commodity_desc == "COTTON" ~ "LB", # bale of cotton to lb
#     .default = county_production_unit
#   ))
# 
# # Adjust for imputed production values
# imputed_production_price_units <- combined %>%
#   distinct(imputed_production_unit, state_price_unit)
# 
# combined <- combined %>%
#   mutate(adjusted_imputed_production_value = case_when(
#     imputed_production_unit == "LB" & state_price_unit == "$ / CWT" ~ imputed_production_value/112, # lb to cwt
#     imputed_production_unit == "TONS" & state_price_unit == "$ / LB" ~ imputed_production_value*2000, # ton to lb
#     imputed_production_unit == "BU" & state_price_unit == "$ / LB" & commodity_desc == "APPLES" ~ imputed_production_value*40, # BU to LB for apples
#     imputed_production_unit == "BU" & state_price_unit == "$ / LB" & commodity_desc == "PEACHES" ~ imputed_production_value*50, # BU to LB for peaches
#     imputed_production_unit == "BU" & state_price_unit == "$ / TON" & commodity_desc == "PEACHES" ~ imputed_production_value*0.025, # BU to TON for peaches
#     .default = imputed_production_value
#   )) %>%
#   mutate(adjusted_imputed_production_unit = case_when(
#     imputed_production_unit == "LB" & state_price_unit == "$ / CWT" ~ "CWT", # lb to cwt
#     imputed_production_unit == "TONS" & state_price_unit == "$ / LB" ~ "LB", # ton to lb
#     imputed_production_unit == "BU" & state_price_unit == "$ / LB" & commodity_desc == "APPLES" ~ "LB", # BU to LB for apples
#     imputed_production_unit == "BU" & state_price_unit == "$ / LB" & commodity_desc == "PEACHES" ~ "LB", # BU to LB for peaches
#     imputed_production_unit == "BU" & state_price_unit == "$ / TON" & commodity_desc == "PEACHES" ~ "TON", # BU to TON for peaches
#     .default = imputed_production_unit
#   ))
# 
# # Check consistency of units again
# production_price_units <- combined %>%
#   distinct(adjusted_production_unit, state_price_unit)
# 
# imputed_production_price_units <- combined %>%
#   distinct(adjusted_imputed_production_unit, state_price_unit)
# 
# # Now finalize dataset and export
# county_production_prices <- combined %>%
#   select(state_fips_code, state_name, county_code, county_name,
#          year,
#          group_desc, commodity_desc, class_desc, util_practice_desc, prodn_practice_desc,
#          state_price_unit, state_price_value,
#          adjusted_production_unit, adjusted_production_value,
#          adjusted_imputed_production_unit, adjusted_imputed_production_value,
#          yield_source, acreage_is_imputed)
# 
# # Clean to share with Phil
# # Harmonize name of FIPS code variable
# county_production_prices <- county_production_prices %>%
#   rename(
#     county_fips_code = county_code,
#     price_unit = state_price_unit,
#     price = state_price_value,
#     output_unit = adjusted_production_unit,
#     output = adjusted_production_value,
#     imputed_output_unit = adjusted_imputed_production_unit,
#     imputed_output = adjusted_imputed_production_value
#     )
# 
# county_production_prices %>%
#   write_parquet(here("files_for_phil", "county_price_output.parquet")) %>%
#   write_parquet(here("binaries", "county_price_output.parquet"))
