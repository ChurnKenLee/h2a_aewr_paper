library(here)
library(arrow)
library(janitor)
library(tidyverse)
library(tidylog)

rm(list = ls())

#### Extract non-duplicative crop observations ####
# A crop is a combination of group-commodity-class-util-prodn
# Get list of all crop entries in Census and Quick Stats surveys
census_years_list <- c(2002, 2007, 2012, 2017, 2022)

qs_census_crops <- read_parquet(here("binaries", "qs_census_crops.parquet")) %>%
  clean_names()

qs_survey_crops <- read_parquet(here("binaries", "qs_survey_crops.parquet")) %>% 
  clean_names()

qs_census_crops <- qs_census_crops %>% 
  filter(freq_desc == "ANNUAL") %>%
  filter(agg_level_desc %in% c("NATIONAL", "STATE", "COUNTY")) %>%
  filter(unit_desc != "OPERATIONS")

qs_survey_crops <- qs_survey_crops %>% 
  filter(freq_desc == "ANNUAL") %>%
  filter(agg_level_desc %in% c("NATIONAL", "STATE", "COUNTY")) %>%
  filter(unit_desc != "OPERATIONS")

qs_census_crops <- qs_census_crops %>% 
  mutate(numeric_value = str_replace_all(value, pattern = ",", replacement = "")) %>%
  mutate(numeric_value = as.numeric(numeric_value)) %>%
  mutate(numeric_value = case_when(
    value == "(Z)" ~ 0, # This is for when value is less than half the rounding value
    .default = numeric_value
  )) %>%
  select(-value) %>%
  rename(value = numeric_value) %>%
  filter(!is.na(value))

qs_survey_crops <- qs_survey_crops %>% 
  mutate(numeric_value = str_replace_all(value, pattern = ",", replacement = "")) %>%
  mutate(numeric_value = as.numeric(numeric_value)) %>%
  mutate(numeric_value = case_when(
    value == "(Z)" ~ 0, # This is for when value is less than half the rounding value
    .default = numeric_value
  )) %>%
  select(-value) %>%
  rename(value = numeric_value) %>%
  filter(!is.na(value))
  
census_stats <- qs_census_crops %>%
  distinct(statisticcat_desc, unit_desc)

survey_stats <- qs_survey_crops %>%
  distinct(statisticcat_desc, unit_desc)

# Classify each observation into type of observation
qs_census_crops <- qs_census_crops %>%
  mutate(is_area = (unit_desc == "ACRES") & (statisticcat_desc != "AREA NON-BEARING")) %>%
  mutate(is_production = (statisticcat_desc == "PRODUCTION")) %>%
  mutate(is_yield = (statisticcat_desc == "YIELD"))

qs_survey_crops <- qs_survey_crops %>%
  mutate(is_area = (unit_desc == "ACRES") | (unit_desc == "ACRES WHERE TAPS SET")) %>%
  mutate(is_price = (statisticcat_desc == "PRICE RECEIVED")) %>%
  mutate(is_production = (statisticcat_desc == "PRODUCTION")) %>%
  mutate(is_yield = (statisticcat_desc == "YIELD"))

# Count number of observations available by type
qs_census_obs <- qs_census_crops %>%
  group_by(group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc) %>%
  summarize(
    n_area_obs_census = sum(is_area),
    n_prod_obs_census = sum(is_production),
    n_yield_obs_census = sum(is_yield)
  ) %>%
  ungroup()

qs_survey_obs <- qs_survey_crops %>%
  group_by(group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc) %>%
  summarize(
    n_area_obs_survey = sum(is_area),
    n_prod_obs_survey = sum(is_production),
    n_yield_obs_survey = sum(is_yield),
    n_price_obs_survey = sum(is_price)
  ) %>%
  ungroup()

# Choose crop cats/subcats to extract by number of observations
crops_obs <- qs_survey_obs %>% 
  full_join(qs_census_obs) %>% 
  arrange(group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc) %>% 
  mutate(across(where(is.numeric), ~replace_na(.x, 0)))

# Sum of observations across categories in census and survey
crops_obs <- crops_obs %>% 
  mutate(n_census_obs = n_area_obs_census + n_prod_obs_census + n_yield_obs_census) %>% 
  mutate(n_survey_obs = n_area_obs_survey + n_prod_obs_survey + n_yield_obs_survey + n_price_obs_survey)

# We drop crops that have 0 census observations
crops_obs <- crops_obs %>% 
  filter(n_census_obs > 0)

# Within each commodity, look at crop with highest census obs
# If that crop is also universal, use that as selected obs for that commodity
commodity_max <- crops_obs %>% 
  group_by(group_desc, commodity_desc) %>% 
  mutate(commodity_max = max(n_census_obs)) %>% 
  ungroup() %>% 
  filter(n_census_obs == commodity_max) %>% 
  filter(class_desc == "ALL CLASSES") %>% 
  filter(prodn_practice_desc == "ALL PRODUCTION PRACTICES") %>% 
  filter(util_practice_desc == "ALL UTILIZATION PRACTICES") %>% 
  mutate(commodity_max_is_universal = TRUE) %>% 
  select(group_desc, commodity_desc, commodity_max_is_universal)

crops_obs <- crops_obs %>% 
  left_join(commodity_max)

# For remainder, choose crop with max obs within class, filter further after
class_max <- crops_obs %>% 
  filter(is.na(commodity_max_is_universal)) %>% 
  group_by(group_desc, commodity_desc, class_desc) %>% 
  mutate(class_max = max(n_census_obs)) %>% 
  ungroup() %>% 
  filter(n_census_obs == class_max)

# Drop commodity-class'es that are duplicative
# For non-HORTICULTURE, we only want to include IN THE OPEN and ALL PRODUCTION PRACTICES
# This drops most of the observations from the organics supplementary census
class_max <- class_max %>% 
  mutate(keep_this = case_when(
    group_desc != "HORTICULTURE" & prodn_practice_desc == "ALL PRODUCTION PRACTICES" ~ TRUE,
    group_desc != "HORTICULTURE" & prodn_practice_desc == "IN THE OPEN" ~ TRUE,
    group_desc == "HORTICULTURE" ~ TRUE,
    .default = FALSE
  )) %>% 
  filter(keep_this)

# Further perform adjustment
# HAY has best ALL CLASSES
# MINT has valid ALL CLASSES for OIL util
# BLUEBERRIES has much worse ALL CLASSES coverage; use subclasses
# HORTICULTURE CABBAGE HEAD has worse coverage than HEAD subclasses
# CUT FLOWERS has best ALL CLASSES
# FLOWER SEEDS has best ALL CLASSES
# FLOWERING PLANTS, POTTED has best coverage with INDOOR USE class
# HORTICULTURE ONIONS with ALL UTILIZATION CLASSES has best coverage
# HORTICULTURE PEPPERS ALL CLASSES has best coverage
class_max <- class_max %>% 
  mutate(keep_this = case_when(
    commodity_desc == "HAY" & class_desc != "ALL CLASSES" ~ FALSE,
    commodity_desc == "MINT" & class_desc != "ALL CLASSES" & util_practice_desc == "OIL" ~ FALSE,
    commodity_desc == "BLUEBERRIES" & class_desc == "ALL CLASSES" ~ FALSE,
    group_desc == "HORTICULTURE" & commodity_desc == "CABBAGE" & class_desc == "HEAD" ~ FALSE,
    commodity_desc == "CUT FLOWERS" & class_desc != "ALL CLASSES" ~ FALSE,
    commodity_desc == "FLOWER SEEDS" & class_desc != "ALL CLASSES" ~ FALSE,
    commodity_desc == "FLOWERING PLANTS, POTTED" & class_desc != "INDOOR USE" ~ FALSE,
    group_desc == "HORTICULTURE" & class_desc == "ONIONS" & util_practice_desc != "ALL UTILIZATION PRACTICES" ~ FALSE,
    group_desc == "HORTICULTURE" & commodity_desc == "PEPPERS" & class_desc != "ALL CLASSES" ~ FALSE,
    .default = keep_this
  )) %>% 
  filter(keep_this)

# Now define set of crops we want to keep from QuickStats database
selected_crops_obs <- crops_obs %>% 
  filter(commodity_max_is_universal) %>% 
  filter(class_desc == "ALL CLASSES") %>% 
  filter(prodn_practice_desc == "ALL PRODUCTION PRACTICES") %>% 
  filter(util_practice_desc == "ALL UTILIZATION PRACTICES")

selected_crops_obs <- selected_crops_obs %>% 
  bind_rows(class_max) %>% 
  arrange(group_desc, commodity_desc, class_desc) %>% 
  select(group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc)

selected_crops_obs %>% write_parquet(here("binaries", "qs_selected_crops_obs.parquet"))

# Pull obs we want from QuickStats data and export binary
qs_census_crops <- qs_census_crops %>% 
  inner_join(selected_crops_obs)

qs_survey_crops <- qs_survey_crops %>% 
  inner_join(selected_crops_obs) 
# Coverage is naturally worse in survey data as selection criteria is based on census obs counts

#### Extract cropland asset data ####
qs_census_economics <- read_parquet(here("binaries", "qs_census_economics.parquet")) %>% 
  clean_names()

qs_census_economics <- qs_census_economics %>% 
  filter(freq_desc == "ANNUAL") %>%
  filter(agg_level_desc %in% c("NATIONAL", "STATE", "COUNTY"))

qs_census_economics <- qs_census_economics %>% 
  mutate(numeric_value = str_replace_all(value, pattern = ",", replacement = "")) %>%
  mutate(numeric_value = as.numeric(numeric_value)) %>%
  mutate(numeric_value = case_when(
    value == "(Z)" ~ 0, # This is for when value is less than half the rounding value
    .default = numeric_value
  )) %>%
  select(-value) %>%
  rename(value = numeric_value) %>%
  filter(!is.na(value))

qs_census_cropland <- qs_census_economics %>% 
  filter(commodity_desc == "FARM OPERATIONS" | commodity_desc == "AG LAND") %>% 
  filter(short_desc == "AG LAND, CROPLAND - ACRES") %>% 
  filter(domain_desc == "TOTAL") # Want county totals


#### Export all selected obs
qs_census_selected_obs <- qs_census_crops %>% 
  bind_rows(qs_census_cropland)
qs_survey_selected_obs <- qs_survey_crops

qs_census_selected_obs %>% write_parquet(here("binaries", "qs_census_selected_obs.parquet"))
qs_survey_selected_obs %>% write_parquet(here("binaries", "qs_survey_selected_obs.parquet"))

