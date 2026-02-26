library(here)
library(arrow)
library(tidyverse)
library(dplyr)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)
library(tidycensus)

rm(list = ls())

# Census data
census_crops_df <- read_parquet((here("files_for_phil", "census_of_agriculture.parquet"))) %>% 
  filter(sector_desc == "CROPS")

census_animals_df <- read_parquet((here("files_for_phil", "census_of_agriculture.parquet"))) %>% 
  filter(sector_desc == "ANIMALS & PRODUCTS")

# These are all the crops in Census data
census_crop_types <- census_crops_df %>% 
  distinct(short_desc, .keep_all = TRUE) %>% 
  mutate(short_desc = as.character(short_desc)) %>% 
  arrange(short_desc)

# NASS state-level Farm Income and Wealth Statistics data
nass_cash_receipts_df <- read_csv(here("Data", "farm_income_and_wealth_statistics", "FarmIncome_WealthStatisticsData_November2023.csv")) %>% 
  clean_names()

# Keep only cash receipts
nass_cash_receipts_df <- nass_cash_receipts_df %>% 
  filter(grepl('Cash receipt', variable_description_total))

# These are all the crop types in NASS data
nass_crop_types <- nass_cash_receipts_df %>% 
  distinct(variable_description_total, .keep_all = TRUE) %>% 
  arrange(variable_description_total)

# Define common crop names for merging
nass_crop_types <- nass_crop_types %>% 
  mutate(
    common_name = case_when(
      variable_description_total == "Cash receipts value, almonds , all" ~ "almoonds",
      variable_description_total == "Cash receipts value, apples , all" ~ "apples",
      variable_description_total == "Cash receipts value, apricots , all" ~ "apricots",
      variable_description_total == "Cash receipts value, artichokes , all" ~ "artichokes",
      variable_description_total == "Cash receipts value, asparagus , all" ~ "asparagus",
      variable_description_total == "Cash receipts value, avocados , all" ~ "avocadoes",
      variable_description_total == "Cash receipts value, bananas , all" ~ "bananas",
      variable_description_total == "Cash receipts value, barley , all" ~ "barley",
      variable_description_total == "Cash receipts value, beans, green lima , all" ~ "lima beans",
      variable_description_total == "Cash receipts value, beans, snap , all" ~ "snap beans",
      variable_description_total == "Cash receipts value, blackberry group , all" ~ "blackberry group", # blackberry group = blackberries + boysenberries + loganberries
      variable_description_total == "Cash receipts value, blueberries , all" ~ "blueberries",
      variable_description_total == "Cash receipts value, broccoli , all" ~ "broccoli",
      variable_description_total == "Cash receipts value, cabbage , all" ~ "cabbage",
      variable_description_total == "Cash receipts value, cane for sugar , all" ~ "sugarcane", # cane has sugar and seeds
      variable_description_total == "Cash receipts value, canola , all" ~ "canola",
      variable_description_total == "Cash receipts value, cantaloups , all" ~ "cantaloupes",
      variable_description_total == "Cash receipts value, carrots , all" ~ "carrots",
      variable_description_total == "Cash receipts value, cauliflower , all" ~ "cauliflower",
      variable_description_total == "Cash receipts value, celery , all" ~ "celery",
      variable_description_total == "Cash receipts value, cherries , all" ~ "cherries",
      variable_description_total == "Cash receipts value, coffee , all" ~ "coffee",
      variable_description_total == "Cash receipts value, corn , all" ~ "corn", 
      variable_description_total == "Cash receipts value, cotton , all" ~ "cotton", # cotton = cotton lint + cottonseed
      variable_description_total == "Cash receipts value, cranberries , all" ~ "cranberries",
      variable_description_total == "Cash receipts value, cucumbers , all" ~ "cucumbers",
      variable_description_total == "Cash receipts value, dates , all" ~ 'dates',
      variable_description_total == "Cash receipts value, dry beans , all" ~ "dry beans", # includes dry lima beans, excludes green lima and snap beans; not sure about chickpeas
      variable_description_total == "Cash receipts value, dry peas , all" ~ "dry peas", # dry peas = field peas, includes Austrian Winter, Edible, and maybe Cowpeas; excludes green and Chinese peas
      variable_description_total == "Cash receipts value, figs , all" ~ "figs",
      variable_description_total == "Cash receipts value, flaxseed , all" ~ "flaxseed",
      variable_description_total == "Cash receipts value, garlic , all" ~ "garlic",
      variable_description_total == "Cash receipts value, ginger root , all" ~ "ginger root",
      variable_description_total == "Cash receipts value, grapefruit , all" ~ "grapefruit",
      variable_description_total == "Cash receipts value, grapes , all" ~ "grapefruit",
      variable_description_total == "Cash receipts value, greenhouse/nursery , floriculture" ~ "floriculture", # not sure if includes non-flori horticulture
      variable_description_total == "Cash receipts value, guavas , all" ~ "guavas",
      variable_description_total == "Cash receipts value, hay , all" ~ "hay",
      variable_description_total == "Cash receipts value, hazelnuts , all" ~ "hazelnuts",
      variable_description_total == "Cash receipts value, honeydews , all" ~ "honeydews",
      variable_description_total == "Cash receipts value, hops , all" ~ "hops",
      variable_description_total == "Cash receipts value, kiwifruit , all" ~ "kiwifruit",
      variable_description_total == "Cash receipts value, lemons , all" ~ "lemons",
      variable_description_total == "Cash receipts value, lentils (beans) , all" ~ "lentils",
      variable_description_total == "Cash receipts value, lettuce , all" ~  "lettuce", # Already includes head, leaf, and romaine
      variable_description_total == "Cash receipts value, macadamia nuts , all" ~ "macadamias",
      variable_description_total == "Cash receipts value, millet, proso, all" ~ "proso millet",
      variable_description_total == "Cash receipts value, mint , all" ~ "mint",
      variable_description_total == "Cash receipts value, mushrooms , all" ~ "mushrooms",
      variable_description_total == "Cash receipts value, mustardseed, all" ~ "mustard seed",
      variable_description_total == "Cash receipts value, nectarines , all" ~ "nectarines",
      variable_description_total == "Cash receipts value, oats , all" ~ "oats",
      variable_description_total == "Cash receipts value, olives , all" ~ "olives",
      variable_description_total == "Cash receipts value, onions , all" ~ "onions",
      variable_description_total == "Cash receipts value, oranges , all" ~ "oranges",
      variable_description_total == "Cash receipts value, papayas , all" ~ "papayas",
      variable_description_total == "Cash receipts value, peaches , all" ~ "peaches",
      variable_description_total == "Cash receipts value, peanuts , all" ~ "peanuts",
      variable_description_total == "Cash receipts value, pears , all" ~ "pears",
      variable_description_total == "Cash receipts value, peas, green , all" ~ "green peas", # Includes Chinese peas
      variable_description_total == "Cash receipts value, pecans , all" ~ "pecans",
      variable_description_total == "Cash receipt value, peppers, bell, all" ~ "bell peppers",
      variable_description_total == "Cash receipts value, peppers, chile , all" ~ "chile peppers",
      variable_description_total == "Cash receipts value, pistachios , all" ~ "pistachios",
      variable_description_total == "Cash receipts value, plums and prunes , all" ~ "plums and prunes",
      variable_description_total == "Cash receipts value, potatoes , all" ~ "potatoes",
      variable_description_total == "Cash receipts value, pumpkins , all" ~ "pumpkins",
      variable_description_total == "Cash receipts value, rapeseed , all" ~ "rapeseed",
      variable_description_total == "Cash receipts value, raspberries , all" ~ "raspberries",
      variable_description_total == "Cash receipts value, rice , all" ~ "rice",
      variable_description_total == "Cash receipts value, rye , all" ~ "rye",
      variable_description_total == "Cash receipts value, safflower , all" ~ "safflower",
      variable_description_total == "Cash receipts value, sorghum grain, all" ~ "sorghum",
      variable_description_total == "Cash receipts value, soybeans , all" ~ "soybeans",
      variable_description_total == "Cash receipts value, spinach , all" ~ "spinach",
      variable_description_total == "Cash receipts value, squash , all" ~ "squash",
      variable_description_total == "Cash receipts value, strawberries , all" ~ "strawberries",
      variable_description_total == "Cash receipts value, sugar beets , all" ~ "sugar beets", # Not sure if this is all beets
      variable_description_total == "Cash receipts value, sunflower , all" ~ "sunflower",
      variable_description_total == "Cash receipts value, corn , sweet corn, all" ~ "sweet corn",
      variable_description_total == "Cash receipts value, sweet potatoes , all" ~ "sweet potatoes",
      variable_description_total == "Cash receipts value, tangelos , all" ~ "tangelos",
      variable_description_total == "Cash receipts value, tangerines , all" ~ "tangerines",
      variable_description_total == "Cash receipts value, taro , all" ~ "taro",
      variable_description_total == "Cash receipts value, tobacco , all" ~ "tobacco",
      variable_description_total == "Cash receipts value, tomatoes , all" ~ "tomatoes",
      variable_description_total == "Cash receipts value, walnuts , all" ~ "walnuts",
      variable_description_total == "Cash receipts value, watermelons , all" ~ "watermelons",
      variable_description_total == "Cash receipts value, wheat , all" ~ "wheat"
    )
  ) %>% 
  arrange(common_name)

census_crop_types <- census_crop_types %>% 
  mutate(common_name = case_when(
    short_desc == "ALMONDS - ACRES BEARING & NON-BEARING" ~ "almonds",
    short_desc == "APPLES - ACRES BEARING & NON-BEARING" ~ "apples",
    short_desc == "APRICOTS - ACRES BEARING & NON-BEARING" ~ "apricots",
    short_desc == "ARONIA BERRIES - ACRES GROWN" ~ "aronia berries",
    short_desc == "ARTICHOKES - ACRES HARVESTED" ~ "artichokes",
    short_desc == "ASPARAGUS - ACRES HARVESTED" ~ "asparagus",
    short_desc == "AVOCADOS - ACRES BEARING & NON-BEARING" ~ "avocados",
    short_desc == "BANANAS - ACRES BEARING & NON-BEARING" ~ "bananas",
    short_desc == "BARLEY - ACRES HARVESTED" ~ "barley",
    short_desc == "BEANS, DRY EDIBLE, (EXCL CHICKPEAS & LIMA) - ACRES HARVESTED" ~ "dry beans",
    short_desc == "BEANS, DRY EDIBLE, LIMA - ACRES HARVESTED" ~ "dry beans",
    short_desc == "CHICKPEAS - ACRES HARVESTED" ~ "dry beans",
    short_desc == "BEANS, GREEN, LIMA - ACRES HARVESTED" ~ "lima beans",
    short_desc == "BEANS, SNAP - ACRES HARVESTED" ~ "snap beans",
    short_desc == "BEETS - ACRES HARVESTED" ~ "beets",
    short_desc == "BLACKBERRIES, INCL DEWBERRIES & MARIONBERRIES - ACRES GROWN" ~ "blackberry group",
    short_desc == "BOYSENBERRIES - ACRES GROWN" ~ "blackberry group",
    short_desc == "LOGANBERRIES - ACRES GROWN" ~ "blackberry group",
    short_desc == "BLUEBERRIES - ACRES GROWN" ~ "blueberries",
    short_desc == "BROCCOLI - ACRES HARVESTED" ~ "broccoli",
    short_desc == "BRUSSELS SPROUTS - ACRES HARVESTED" ~ "brussels sprouts",
    short_desc == "BUCKWHEAT - ACRES HARVESTED" ~ "buckwheat",
    short_desc == "CABBAGE, CHINESE - ACRES HARVESTED" ~ "cabbage",
    short_desc == "CABBAGE, HEAD - ACRES HARVESTED" ~ "cabbage",
    short_desc == "CABBAGE, MUSTARD - ACRES HARVESTED" ~ "cabbage",
    short_desc == "CAMELINA - ACRES HARVESTED" ~ "camelina",
    short_desc == "CANOLA - ACRES HARVESTED" ~ "canola",
    short_desc == "CARROTS - ACRES HARVESTED" ~ "carrots",
    short_desc == "CAULIFLOWER - ACRES HARVESTED" ~ "cauliflower",
    short_desc == "CELERY - ACRES HARVESTED" ~ "celery",
    short_desc == "CHERIMOYAS - ACRES BEARING & NON-BEARING" ~ "cherimoyas",
    short_desc == "CHERRIES, SWEET - ACRES BEARING & NON-BEARING" ~ "cherries",
    short_desc == "CHERRIES, TART - ACRES BEARING & NON-BEARING" ~ "cherries",
    short_desc == "CHESTNUTS - ACRES BEARING & NON-BEARING" ~ "chestnuts",
    short_desc == "CHICORY - ACRES HARVESTED" ~ "chicory",
    short_desc == "COFFEE - ACRES BEARING & NON-BEARING" ~ "coffee",
    short_desc == "CORN, GRAIN - ACRES HARVESTED" ~ "corn",
    short_desc == "CORN, SILAGE - ACRES HARVESTED" ~ "corn",
    short_desc == "CORN, TRADITIONAL OR INDIAN - ACRES HARVESTED" ~ "corn",
    short_desc == "COTTON - ACRES HARVESTED" ~ "cotton",
    short_desc == "CRANBERRIES - ACRES GROWN" ~ "cranberries",
    short_desc == "CUCUMBERS - ACRES HARVESTED" ~ "cucumbers",
    short_desc == "CURRANTS - ACRES GROWN" ~ "currants",
    short_desc == "DAIKON - ACRES HARVESTED" ~ "daikon",
    short_desc == "DATES - ACRES BEARING & NON-BEARING" ~ "dates",
    short_desc == "DILL, OIL - ACRES HARVESTED" ~ "dill",
    short_desc == "EGGPLANT - ACRES HARVESTED" ~ "eggplant",
    short_desc == "ELDERBERRIES - ACRES GROWN" ~ "elderberries",
    short_desc == "EMMER & SPELT - ACRES HARVESTED" ~ "emmer and spelt",
    short_desc == "ESCAROLE & ENDIVE - ACRES HARVESTED" ~ "escarole and endive",
    short_desc == "FIGS - ACRES BEARING & NON-BEARING" ~ "figs",
    short_desc == "FLAXSEED - ACRES HARVESTED" ~ "flaxseed",
    short_desc == "FLORICULTURE TOTALS, IN THE OPEN - ACRES IN PRODUCTION" ~ "floriculture",
    short_desc == "FLORICULTURE TOTALS, UNDER PROTECTION - SQ FT IN PRODUCTION" ~ "floriculture",
    short_desc == "FLORICULTURE, OTHER, IN THE OPEN - ACRES IN PRODUCTION" ~ "floriculture",
    short_desc == "FLORICULTURE, OTHER, UNDER PROTECTION - SQ FT IN PRODUCTION" ~ "floriculture",
    short_desc == "GARLIC - ACRES HARVESTED" ~ "garlic",
    short_desc == "GINGER ROOT - ACRES HARVESTED" ~ "ginger root",
    short_desc == "GINSENG - ACRES HARVESTED" ~ "ginseng",
    short_desc == "GRAPEFRUIT - ACRES BEARING & NON-BEARING" ~ "grapefruit",
    short_desc == "GRAPES - ACRES BEARING & NON-BEARING" ~ "grapes",
    short_desc == "GREENS, COLLARD - ACRES HARVESTED" ~ "collard",
    short_desc == "GREENS, KALE - ACRES HARVESTED" ~ "kale",
    short_desc == "GREENS, MUSTARD - ACRES HARVESTED" ~ "mustard",
    short_desc == "MUSTARD, SEED - ACRES HARVESTED" ~ "mustard seed",
    short_desc == "GREENS, TURNIP - ACRES HARVESTED" ~ "turnip",
    short_desc == "GUAR - ACRES HARVESTED" ~ "guar",
    short_desc == "GUAVAS - ACRES BEARING & NON-BEARING" ~ "guavas",
    short_desc == "HAY - ACRES HARVESTED" ~ "hay",
    short_desc == "HAYLAGE - ACRES HARVESTED" ~ "hay",
    short_desc == "HAZELNUTS - ACRES BEARING & NON-BEARING" ~ "hazelnuts",
    short_desc == "HERBS, DRY - ACRES HARVESTED" ~ "herbs",
    short_desc == "HERBS, FRESH CUT - ACRES HARVESTED" ~ "herbs",
    short_desc == "HOPS - ACRES HARVESTED" ~ "hops",
    short_desc == "HORSERADISH - ACRES HARVESTED" ~ "horseradish",
    short_desc == "JOJOBA - ACRES HARVESTED" ~ "jojoba",
    short_desc == "KIWIFRUIT - ACRES BEARING & NON-BEARING" ~ "kiwifruit",
    short_desc == "KUMQUATS - ACRES BEARING & NON-BEARING" ~ "kumquats",
    short_desc == "LEMONS - ACRES BEARING & NON-BEARING" ~ "lemons",
    short_desc == "LENTILS - ACRES HARVESTED" ~ "lentils",
    short_desc == "LETTUCE - ACRES HARVESTED" ~ "lettuce",
    short_desc == "LIMES - ACRES BEARING & NON-BEARING" ~ "limes",
    short_desc == "MACADAMIAS - ACRES BEARING & NON-BEARING" ~ "macadamias",
    short_desc == "MANGOES - ACRES BEARING & NON-BEARING" ~ "mangoes",
    short_desc == "MELONS, CANTALOUP - ACRES HARVESTED" ~ "cantaloupes",
    short_desc == "MELONS, HONEYDEW - ACRES HARVESTED" ~ "honeydews",
    short_desc == "MELONS, WATERMELON - ACRES HARVESTED" ~ "watermelon",
    short_desc == "MILLET, PROSO - ACRES HARVESTED" ~ "proso millet",
    short_desc == "MINT, OIL - ACRES HARVESTED" ~ "mint",
    short_desc == "MINT, TEA LEAVES - ACRES HARVESTED" ~ "mint",
    short_desc == "MISCANTHUS - ACRES HARVESTED" ~ "miscanthus",
    short_desc == "MUSHROOMS - SQ FT IN PRODUCTION" ~ "mushroom",
    short_desc == "NECTARINES - ACRES BEARING & NON-BEARING" ~ "nectarines",
    short_desc == "NURSERY TOTALS, IN THE OPEN - ACRES IN PRODUCTION" ~ "nursery",
    short_desc == "NURSERY TOTALS, UNDER PROTECTION - SQ FT IN PRODUCTION" ~ "nursery",
    short_desc == "OATS - ACRES HARVESTED" ~ "oats",
    short_desc == "OKRA - ACRES HARVESTED" ~ "okra",
    short_desc == "OLIVES - ACRES BEARING & NON-BEARING" ~ "olives",
    short_desc == "ONIONS, DRY - ACRES HARVESTED" ~ "onions",
    short_desc == "ONIONS, GREEN - ACRES HARVESTED" ~ "onions",
    short_desc == "ORANGES - ACRES BEARING & NON-BEARING" ~ "oranges",
    short_desc == "PAPAYAS - ACRES BEARING & NON-BEARING" ~ "papayas",
    short_desc == "PARSLEY - ACRES HARVESTED" ~ "parsley",
    short_desc == "PASSION FRUIT - ACRES BEARING & NON-BEARING" ~ "passion fruit",
    short_desc == "PEACHES - ACRES BEARING & NON-BEARING" ~ "peaches",
    short_desc == "PEANUTS - ACRES HARVESTED" ~ "peanuts",
    short_desc == "PEARS - ACRES BEARING & NON-BEARING" ~ "pears",
    short_desc == "PEAS, AUSTRIAN WINTER - ACRES HARVESTED" ~ "dry peas",
    short_desc == "PEAS, DRY EDIBLE - ACRES HARVESTED" ~ "dry peas",
    short_desc == "PEAS, DRY, SOUTHERN (COWPEAS) - ACRES HARVESTED" ~ "dry peas",
    short_desc == "PEAS, CHINESE (SUGAR & SNOW) - ACRES HARVESTED" ~ "green peas",
    short_desc == "PEAS, GREEN, (EXCL SOUTHERN) - ACRES HARVESTED" ~ "green peas",
    short_desc == "PEAS, GREEN, SOUTHERN (COWPEAS) - ACRES HARVESTED" ~ "green peas",
    short_desc == "PECANS - ACRES BEARING & NON-BEARING" ~ "pecans",
    short_desc == "PEPPERS, BELL - ACRES HARVESTED" ~ "bell peppers",
    short_desc == "PEPPERS, CHILE - ACRES HARVESTED" ~ "chile peppers",
    short_desc == "PERSIMMONS - ACRES BEARING & NON-BEARING" ~ "persimmons",
    short_desc == "PINEAPPLES - ACRES BEARING & NON-BEARING" ~ "pineapples",
    short_desc == "PISTACHIOS - ACRES BEARING & NON-BEARING" ~ "pistachios",
    short_desc == "PLUM-APRICOT HYBRIDS, INCL PLUMCOTS & PLUOTS - ACRES BEARING & NON-BEARING" ~ "plum-apricot hybrids",
    short_desc == "PLUMS - ACRES BEARING & NON-BEARING" ~ "plums and prunes",
    short_desc == "PRUNES - ACRES BEARING & NON-BEARING" ~ "plums and prunes",
    short_desc == "POMEGRANATES - ACRES BEARING & NON-BEARING" ~ "pomegranates",
    short_desc == "POPCORN, SHELLED - ACRES HARVESTED" ~ "popcorn",
    short_desc == "POTATOES - ACRES HARVESTED" ~ "potatoes",
    short_desc == "PUMPKINS - ACRES HARVESTED" ~ "pumpkins",
    short_desc == "RADISHES - ACRES HARVESTED" ~ "radishes",
    short_desc == "RAPESEED - ACRES HARVESTED" ~ "rapeseed",
    short_desc == "RASPBERRIES - ACRES GROWN" ~ "raspberries",
    short_desc == "RHUBARB - ACRES HARVESTED" ~ "rhubard",
    short_desc == "RICE - ACRES HARVESTED" ~ "rice",
    short_desc == "RYE - ACRES HARVESTED" ~ "rye",
    short_desc == "SAFFLOWER - ACRES HARVESTED" ~ "safflower",
    short_desc == "SESAME - ACRES HARVESTED" ~ "sesame",
    short_desc == "SOD - ACRES HARVESTED" ~ "sod",
    short_desc == "SORGHUM, GRAIN - ACRES HARVESTED" ~ "sorghum",
    short_desc == "SORGHUM, SILAGE - ACRES HARVESTED" ~ "sorghum",
    short_desc == "SORGHUM, SYRUP - ACRES HARVESTED" ~ "sorghum",
    short_desc == "SOYBEANS - ACRES HARVESTED" ~ "soybeans",
    short_desc == "SPINACH - ACRES HARVESTED" ~ "spinach",
    short_desc == "SQUASH - ACRES HARVESTED" ~ "squash",
    short_desc == "STRAWBERRIES - ACRES GROWN" ~ "strawberries",
    short_desc == "SUGARBEETS - ACRES HARVESTED" ~ "sugar beets",
    short_desc == "SUGARCANE, SEED - ACRES HARVESTED" ~ "sugarcane",
    short_desc == "SUGARCANE, SUGAR - ACRES HARVESTED" ~ "sugarcane",
    short_desc == "SUGARCANE, SUGAR & SEED - ACRES HARVESTED" ~ "sugarcane",
    short_desc == "SUNFLOWER - ACRES HARVESTED" ~ "sunflower",
    short_desc == "SWEET CORN - ACRES HARVESTED" ~ "sweet corn",
    short_desc == "SWEET POTATOES - ACRES HARVESTED" ~ "sweet potatoes",
    short_desc == "SWITCHGRASS - ACRES HARVESTED" ~ "swithgrass",
    short_desc == "TANGELOS - ACRES BEARING & NON-BEARING" ~ "tangelos",
    short_desc == "TANGERINES - ACRES BEARING & NON-BEARING" ~ "tangerines",
    short_desc == "TARO - ACRES HARVESTED" ~ "taro",
    short_desc == "TOBACCO - ACRES HARVESTED" ~ "tobacco",
    short_desc == "TOMATOES, IN THE OPEN - ACRES HARVESTED" ~ "tomatoes",
    short_desc == "TOMATOES, UNDER PROTECTION - SQ FT IN PRODUCTION" ~ "tomatoes",
    short_desc == "TRITICALE - ACRES HARVESTED" ~ "triticale",
    short_desc == "TURNIPS - ACRES HARVESTED" ~ "turnips",
    short_desc == "WALNUTS, ENGLISH - ACRES BEARING & NON-BEARING" ~ "walnuts",
    short_desc == "WATERCRESS - ACRES HARVESTED" ~ "watercress",
    short_desc == "WHEAT - ACRES HARVESTED" ~ "wheat",
    short_desc == "WILD RICE - ACRES HARVESTED" ~ "wild rice"
  )
  ) %>% 
  arrange(common_name)

# Common names to merge back in
census_common_names <- census_crop_types %>% 
  select(short_desc, common_name)

census_crops_df <- census_crops_df %>% 
  left_join(census_common_names)

nass_common_names <- nass_crop_types %>% 
  select(variable_description_total, common_name)

nass_cash_receipts_df <- nass_cash_receipts_df %>% 
  left_join(nass_common_names)

# Aggregate into common crop types
# Adjust from sq ft to acres for some crops
census_crops_df <- census_crops_df %>% 
  mutate(destringed_value = as.numeric(gsub(",", "", value))) %>% 
  mutate(harmonized_acreage = case_when(
    unit_desc == "ACRES" ~ destringed_value,
    unit_desc == "SQ FT" ~ destringed_value/43560
  ))

# We want just state-level acreage
census_state_df <- census_crops_df %>% 
  filter(agg_level_desc == "STATE") %>% 
  group_by(state_alpha, state_name, state_fips_code, year, common_name) %>% 
  summarize(acres = sum(harmonized_acreage, na.rm = TRUE)) %>% 
  ungroup()

# Add FIPS code to NASS data and aggregate
state_fips_codes <- get(data(fips_codes)) %>% 
  select(state, state_code) %>% 
  rename(state_fips_code = state_code) %>% 
  distinct(state_fips_code, .keep_all = TRUE)

nass_state_df <- nass_cash_receipts_df %>% 
  mutate(amount = amount*1000) %>% 
  left_join(state_fips_codes) %>% 
  group_by(state_fips_code, year, common_name) %>% 
  summarize(cash_receipt = sum(amount, na.rm = TRUE)) %>% 
  ungroup()

# Add cash receipts to Census acreage and calculate yield
state_yield_df <- census_state_df %>% 
  left_join(nass_state_df) %>% 
  mutate(state_yield = cash_receipt/acres) %>% 
  select(state_alpha, state_fips_code, year, common_name, state_yield)

# Calculate national yield
nass_national_df <- nass_cash_receipts_df %>% 
  mutate(amount = amount*1000) %>% 
  filter(state == "US") %>% 
  group_by(year, common_name) %>% 
  summarize(cash_receipt = sum(amount, na.rm = TRUE)) %>% 
  ungroup()

census_national_df <- census_crops_df %>% 
  filter(agg_level_desc == "NATIONAL") %>% 
  group_by(year, common_name) %>% 
  summarize(acres = sum(harmonized_acreage, na.rm = TRUE)) %>% 
  ungroup()

national_yield_df <- census_national_df %>% 
  left_join(nass_national_df) %>% 
  mutate(national_yield = cash_receipt/acres) %>% 
  select(year, common_name, national_yield)

# Combine and export
state_yield_df %>% 
  left_join(national_yield_df) %>% 
  write_parquet(here("binaries", "crop_yield.parquet"))

# Export common names
census_crop_types %>% 
  select(short_desc, common_name) %>% 
  write_parquet(here("binaries", "census_crop_names.parquet")) %>% 
  write_csv(here("binaries", "census_crop_names.csv"))

nass_crop_types %>% 
  select(variable_description_total, common_name) %>% 
  write_parquet(here("binaries", "nass_crop_names.parquet"))