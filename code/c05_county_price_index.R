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

# # Load reports; note large size, may want to pre-filter rows in Python
# crops_df <- read_parquet(here("binaries", "nass_quickstats_crops.parquet"))
# 
# survey_df <- crops_df %>%
#   filter(SOURCE_DESC == "SURVEY")
# 
# # Maybe we can use the survey to calculate yield $/acre
# # This gives us acreage by crop
# acreage_df <- survey_df %>%
#   filter(GROUP_DESC == "FRUIT & TREE NUTS" | GROUP_DESC == "VEGETABLES") %>%
#   filter(PRODN_PRACTICE_DESC == "ALL PRODUCTION PRACTICES") %>% # We are not interested in specialty farming
#   filter(UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES") %>%
#   filter(!grepl('TOTALS', COMMODITY_DESC)) %>% # We do not want aggregated totals
#   filter(UNIT_DESC == "ACRES") %>%
#   filter(DOMAINCAT_DESC == "NOT SPECIFIED") %>% # Removes counts of operators by characteristics
#   arrange(STATE_NAME, COMMODITY_DESC, YEAR)
# 
# acreage_df %>% write_parquet(here("binaries", "nass_quickstats_acreage.parquet"))
# 
# # We want farm gate prices, which is denoted PRICE RECEIVED by NASS
# price_received_df <- survey_df %>%
#   filter(GROUP_DESC == "FRUIT & TREE NUTS" | GROUP_DESC == "VEGETABLES") %>%
#   filter(STATISTICCAT_DESC == "PRICE RECEIVED") %>%
#   arrange(STATE_NAME, COMMODITY_DESC, YEAR)
# 
# price_received_df %>% write_parquet(here("binaries", "nass_quickstats_price_received.parquet"))
# 
# # These are state-level production for specialty crops
# production_df <- survey_df %>%
#   filter(GROUP_DESC == "FRUIT & TREE NUTS" | GROUP_DESC == "VEGETABLES") %>%
#   filter(STATISTICCAT_DESC == "PRODUCTION") %>%
#   filter(PRODN_PRACTICE_DESC == "ALL PRODUCTION PRACTICES") %>% # We don't want production in separate categories of usage
#   filter(UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES") %>%
#   filter(!grepl('TOTALS', COMMODITY_DESC)) %>% # We do not want aggregated totals
#   arrange(STATE_NAME, COMMODITY_DESC, YEAR)
# 
# production_df %>% write_parquet(here("binaries", "nass_quickstats_production.parquet"))
# 
# # We can compare with yield numbers given by the survey
# yield_df <- survey_df %>%
#   filter(GROUP_DESC == "FRUIT & TREE NUTS" | GROUP_DESC == "VEGETABLES") %>%
#   filter(STATISTICCAT_DESC == "YIELD") %>%
#   filter(PRODN_PRACTICE_DESC == "ALL PRODUCTION PRACTICES") %>% # We don't want production in separate categories of usage
#   filter(UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES") %>%
#   arrange(STATE_NAME, COMMODITY_DESC, YEAR)
# 
# yield_df %>% write_parquet(here("binaries", "nass_quickstats_yield.parquet"))
# 
# # Construct wide table to check agreement of numbers
# price_received_df <- read_parquet(here("binaries", "nass_quickstats_price_received.parquet"))
# production_df <- read_parquet(here("binaries", "nass_quickstats_production.parquet"))
# acreage_df <- read_parquet(here("binaries", "nass_quickstats_acreage.parquet"))
# yield_df <- read_parquet(here("binaries", "nass_quickstats_yield.parquet"))
# 
# # Join horizontally; select identifying variables to do so
# price_received_df <- price_received_df %>%
#   filter(as.numeric(YEAR) >= 2000) %>%
#   mutate(price_received_value = VALUE) %>%
#   mutate(price_received_unit = UNIT_DESC) %>%
#   select(COMMODITY_DESC, CLASS_DESC, PRODN_PRACTICE_DESC, UTIL_PRACTICE_DESC, STATE_FIPS_CODE, STATE_NAME, YEAR, price_received_value, price_received_unit)
# 
# production_df <- production_df %>%
#   filter(as.numeric(YEAR) >= 2000) %>%
#   mutate(production_value = VALUE) %>%
#   mutate(production_unit = UNIT_DESC) %>%
#   select(COMMODITY_DESC, CLASS_DESC, PRODN_PRACTICE_DESC, UTIL_PRACTICE_DESC, STATE_FIPS_CODE, STATE_NAME, YEAR, production_value, production_unit)
# 
# acreage_df <- acreage_df %>%
#   filter(as.numeric(YEAR) >= 2000) %>%
#   mutate(acreage_value = VALUE) %>%
#   mutate(acreage_unit = UNIT_DESC) %>%
#   select(COMMODITY_DESC, CLASS_DESC, PRODN_PRACTICE_DESC, UTIL_PRACTICE_DESC, STATE_FIPS_CODE, STATE_NAME, YEAR, acreage_value, acreage_unit)
# 
# yield_df <- yield_df %>%
#   filter(as.numeric(YEAR) >= 2000) %>%
#   mutate(yield_value = VALUE) %>%
#   mutate(yield_unit = UNIT_DESC) %>%
#   select(COMMODITY_DESC, CLASS_DESC, PRODN_PRACTICE_DESC, UTIL_PRACTICE_DESC, STATE_FIPS_CODE, STATE_NAME, YEAR, yield_value, yield_unit)
# 
# joined_df <- full_join(price_received_df, production_df) %>%
#   full_join(acreage_df) %>%
#   full_join(yield_df)

# Match crops between ERS yearbooks and CroplandCROS CDL using previously defined common names
cdl_crop_acreage_df <- read_parquet(here("binaries", "croplandcros_county_crop_acres.parquet"))
croplandcros_df <- read_parquet(here("binaries", "croplandcros_county_crop_acres.parquet"))
croplandcros_cdl_table <- read_parquet(here("binaries", "croplandcros_cdl_common_name_table.parquet"))








# Match crops between NASS surveys and CroplandCROS CDL
croplandcros_df <- read_parquet(here("binaries", "croplandcros_county_crop_acres.parquet"))
croplandcros_cdl_table <- read_parquet(here("binaries", "croplandcros_cdl_common_name_table.parquet"))

# Get list of NASS crops
# We only want state level prices
nass_crops <- price_received_df %>% 
  distinct(COMMODITY_DESC, CLASS_DESC, PRODN_PRACTICE_DESC, UTIL_PRACTICE_DESC, .keep_all = TRUE) %>% 
  arrange(COMMODITY_DESC, CLASS_DESC)

cdl_crops <- croplandcros_df %>% 
  select(crop_name) %>% 
  distinct() %>% 
  arrange(crop_name)

# Cross-join table
nass_cdl_crop_joiner <- nass_crops %>% 
  select("COMMODITY_DESC", "CLASS_DESC", "UTIL_PRACTICE_DESC") %>% 
  mutate(cdl_crop_name = case_when(
    COMMODITY_DESC == "APRICOTS" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Apricots",
    COMMODITY_DESC == "ASPARAGUS" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Asparagus",
    COMMODITY_DESC == "AVOCADOS" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Avocados",
    COMMODITY_DESC == "BLUEBERRIES" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Blueberries",
    COMMODITY_DESC == "BROCCOLI" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Broccoli",
    COMMODITY_DESC == "CABBAGE" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Cabbage",
    COMMODITY_DESC == "CARROTS" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Carrots",
    COMMODITY_DESC == "CAUIFLOWER" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Cauliflower",
    COMMODITY_DESC == "CELERY" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Celery",
    COMMODITY_DESC == "CHERRIES" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Cherries",
    COMMODITY_DESC == "CRANBERRIES" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Cranberries",
    COMMODITY_DESC == "CUCUMBERS" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Cucumbers",
    COMMODITY_DESC == "EGGPLANT" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Eggplants",
    COMMODITY_DESC == "GARLIC" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Garlic",
    COMMODITY_DESC == "GRAPES" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Grapes",
    COMMODITY_DESC == "GREENS" & CLASS_DESC == "COLLARD" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Greens", # CDL does not have mustard but has greens
    COMMODITY_DESC == "GREENS" & CLASS_DESC == "KALE" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Greens", # CDL does not have kale but has greens
    COMMODITY_DESC == "GREENS" & CLASS_DESC == "MUSTARD" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Mustard",
    COMMODITY_DESC == "GREENS" & CLASS_DESC == "TURNIP" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Turnips",
    COMMODITY_DESC == "LETTUCE" & CLASS_DESC == "HEAD" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Lettuce", # CDL does not distinguish lettuce
    COMMODITY_DESC == "LETTUCE" & CLASS_DESC == "LEAF" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Lettuce",
    COMMODITY_DESC == "LETTUCE" & CLASS_DESC == "ROMAINE" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Lettuce",
    COMMODITY_DESC == "MELONS" & CLASS_DESC == "CANTALOUP" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Cantaloupes",
    COMMODITY_DESC == "MELONS" & CLASS_DESC == "HONEYDEW" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Honeydew Melons",
    COMMODITY_DESC == "MELONS" & CLASS_DESC == "WATERMELON" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Watermelons",
    COMMODITY_DESC == "NECTARINES" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Nectarines",
    COMMODITY_DESC == "OLIVES" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Olives",
    COMMODITY_DESC == "ONIONS" & CLASS_DESC == "DRY" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Onions", # CDL does not distinguish onion usages
    COMMODITY_DESC == "ONIONS" & CLASS_DESC == "DRY, SPRING" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Onions",
    COMMODITY_DESC == "ONIONS" & CLASS_DESC == "DRY, SUMMER, STORAGE" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Onions",
    COMMODITY_DESC == "ONIONS" & CLASS_DESC == "DRY, SUMMER, NON-STORAGE" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Onions",
    COMMODITY_DESC == "ONIONS" & CLASS_DESC == "DRY, SUMMER" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Onions",
    COMMODITY_DESC == "ORANGES" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Oranges",
    COMMODITY_DESC == "PEACHES" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Peaches",
    COMMODITY_DESC == "PEARS" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Pears",
    COMMODITY_DESC == "PEAS" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Peas",
    COMMODITY_DESC == "PECANS" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Pecans",
    COMMODITY_DESC == "PEPPERS" & CLASS_DESC == "BELL" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Peppers", # CDL does not distinguish between bell and chile peppers
    COMMODITY_DESC == "PEPPERS" & CLASS_DESC == "CHILE" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Peppers",
    COMMODITY_DESC == "PISTACHIOS" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Pistachios",
    COMMODITY_DESC == "PLUMS" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Plums ",
    COMMODITY_DESC == "PRUNES" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Prunes",
    COMMODITY_DESC == "PUMPKINS" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Pumpkins",
    COMMODITY_DESC == "RADISHES" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Radishes",
    COMMODITY_DESC == "SQUASH" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Squash",
    COMMODITY_DESC == "STRAWBERRIES" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Strawberries",
    COMMODITY_DESC == "SWEET CORN" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Sweet Corn",
    COMMODITY_DESC == "SWEET POTATOES" & CLASS_DESC == "ALL CLASSES" & UTIL_PRACTICE_DESC == "ALL UTILIZATION PRACTICES" ~ "Sweet Potatoes"
  ))












