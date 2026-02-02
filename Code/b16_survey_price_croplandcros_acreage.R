library(here)
library(arrow)
library(tidyverse)
library(dplyr)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)
library(tidycensus)
library(PriceIndices)
library(IndexNumR)

rm(list = ls())

# Get list of crops in census data
i <- 2022
file_name <- paste0("census_of_agriculture_", i, ".parquet")
df <- read_parquet(here("binaries", file_name)) %>% 
  clean_names() %>% 
  arrange(state_name, county_name, sector_desc, group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc)

census_crops_df <- df %>% 
  filter(sector_desc == "CROPS") %>% 
  filter(unit_desc != "OPERATIONS") %>% 
  filter(domain_desc == "TOTAL") %>% 
  filter((unit_desc == "ACRES") | (unit_desc == "SQ FT"))

census_crops_list_df <- census_crops_df %>% 
  distinct(short_desc, .keep_all = TRUE) %>% 
  mutate(short_desc = as.character(short_desc)) %>% 
  ungroup() %>% 
  arrange(short_desc)

# Select observations we want
census_crops_list_df <- census_crops_list_df %>% 
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
    short_desc == "BLACKBERRIES, INCL DEWBERRIES & MARIONBERRIES - ACRES GROWN" ~ "caneberries",
    short_desc == "BOYSENBERRIES - ACRES GROWN" ~ "caneberries",
    short_desc == "LOGANBERRIES - ACRES GROWN" ~ "caneberries",
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
    short_desc == "CUT CHRISTMAS TREES - ACRES IN PRODUCTION" ~ "christmas trees",
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
    short_desc == "MUSTARD, SEED - ACRES HARVESTED" ~ "mustard",
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
    short_desc == "MELONS, HONEYDEW - ACRES HARVESTED" ~ "honeydew",
    short_desc == "MELONS, WATERMELON - ACRES HARVESTED" ~ "watermelons",
    short_desc == "MILLET, PROSO - ACRES HARVESTED" ~ "millet",
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
    short_desc == "SWITCHGRASS - ACRES HARVESTED" ~ "switchgrass",
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
  ))

# Keep only variables we need
census_crops_list_df <- census_crops_list_df %>% 
  select(group_desc, short_desc, common_name) %>% 
  filter(!is.na(common_name)) %>% 
  arrange(common_name)

# Define common crop names for CroplandCROS CDL data
# Get list of crops in CroplandCROS CDL
file_name <- paste0("county_crop_acres.parquet")
croplandcros_cdl_county_acreage <- read_parquet(here("binaries", file_name))

croplandcros_cdl_crops_list <- croplandcros_cdl_county_acreage %>% 
  distinct(crop_name)

# Add common name to CroplandCROS CDL crops
croplandcros_cdl_crops_list <- croplandcros_cdl_crops_list %>% 
  mutate(common_name = case_when(
    crop_name == "Alfalfa" ~ "hay",
    crop_name == "Almonds" ~ "almonds",
    crop_name == "Apples" ~ "apples",
    crop_name == "Apricots" ~ "apricots",
    crop_name == "Asparagus" ~ "asparagus",
    crop_name == "Avocados" ~ "avocados",
    crop_name == "Barley" ~ "barley",
    crop_name == "Blueberries" ~ "blueberries",
    crop_name == "Broccoli" ~ "broccoli",
    crop_name == "Buckwheat" ~ "buckwheat",
    crop_name == "Cabbage" ~ "cabbage",
    crop_name == "Camelina" ~ "camelina",
    crop_name == "Caneberries" ~ "caneberries",
    crop_name == "Canola" ~ "canola",
    crop_name == "Cantaloupes" ~ "cantaloupes",
    crop_name == "Carrots" ~ "carrots",
    crop_name == "Cauliflower" ~ "cauliflower",
    crop_name == "Celery" ~ "celery",
    crop_name == "Cherries" ~ "cherries",
    crop_name == "Chick Peas" ~ "dry beans",
    crop_name == "Christmas Trees" ~ "christmas trees",
    crop_name == "Citrus" ~ "citrus",
    crop_name == "Corn" ~ "corn",
    crop_name == "Cotton" ~ "cotton",
    crop_name == "Cranberries" ~ "cranberries",
    crop_name == "Cucumbers" ~ "cucumbers",
    crop_name == "Dry Beans" ~ "dry beans",
    crop_name == "Durum Wheat" ~ "wheat",
    crop_name == "Eggplants" ~ "eggplant",
    crop_name == "Flaxseed" ~ "flaxseed",
    crop_name == "Garlic" ~ "garlic",
    crop_name == "Gourds" ~ "gourds",
    crop_name == "Grapes" ~ "grapes",
    crop_name == "Greens" ~ "greens",
    crop_name == "Herbs" ~ "herbs",
    crop_name == "Honeydew Melons" ~ "honeydew",
    crop_name == "Hops" ~ "hops",
    crop_name == "Lentils" ~ "lentils",
    crop_name == "Lettuce" ~ "lettuce",
    crop_name == "Millet" ~ "millet",
    crop_name == "Mint" ~ "mint",
    crop_name == "Misc Vegs & Fruits" ~ "misc vegs and fruits",
    crop_name == "Mustard" ~ "mustard",
    crop_name == "Nectarines" ~ "nectarines",
    crop_name == "Oats" ~ "oats",
    crop_name == "Olives" ~ "olives",
    crop_name == "Onions" ~ "onions",
    crop_name == "Oranges" ~ "oranges",
    crop_name == "Other Hay/Non Alfalfa" ~ "hay",
    crop_name == "Peaches" ~ "peaches",
    crop_name == "Peanuts" ~ "peanuts",
    crop_name == "Pears" ~ "pears",
    crop_name == "Peas" ~ "peas",
    crop_name == "Pecans" ~ "pecans",
    crop_name == "Peppers" ~ "peppers",
    crop_name == "Pistachios" ~ "pistachios",
    crop_name == "Plums" ~ "plums and prunes",
    crop_name == "Pomegranates" ~ "pomegranates",
    crop_name == "Pop or Orn Corn" ~ "popcorn",
    crop_name == "Potatoes" ~ "potatoes",
    crop_name == "Prunes" ~ "plums and prunes",
    crop_name == "Pumpkins" ~ "pumpkins",
    crop_name == "Radishes" ~ "radishes",
    crop_name == "Rape Seed" ~ "rapeseed",
    crop_name == "Rice" ~ "rice",
    crop_name == "Rye" ~ "rye",
    crop_name == "Safflower" ~ "safflower",
    crop_name == "Sod/Grass Seed" ~ "sod",
    crop_name == "Sorghum" ~ "sorghum",
    crop_name == "Soybeans" ~ "soybeans",
    crop_name == "Speltz" ~ "emmer and spelt",
    crop_name == "Spring Wheat" ~ "wheat",
    crop_name == "Squash" ~ "squash",
    crop_name == "Strawberries" ~ "strawberries",
    crop_name == "Sugarbeets" ~ "sugar beets",
    crop_name == "Sugarcane" ~ "sugarcane",
    crop_name == "Sunflower" ~ "sunflower",
    crop_name == "Sweet Corn" ~ "sweet corn",
    crop_name == "Sweet Potatoes" ~ "sweet potatoes",
    crop_name == "Switchgrass" ~ "switchgrass",
    crop_name == "Tobacco" ~ "tobacco",
    crop_name == "Tomatoes" ~ "tomatoes",
    crop_name == "Triticale" ~ "triticale",
    crop_name == "Turnips" ~ "turnips",
    crop_name == "Walnuts" ~ "walnuts",
    crop_name == "Watermelons" ~ "watermelons",
    crop_name == "Winter Wheat" ~ "wheat"
  ))

# For double cropping, create entry for each individual type per double-cropped type
croplandcros_cdl_crops_list <- croplandcros_cdl_crops_list %>% 
  mutate(common_name = case_when(
    crop_name == "Dbl Crop Barley/Corn" ~ "barley",
    crop_name == "Dbl Crop Barley/Sorghum" ~ "barley",
    crop_name == "Dbl Crop Barley/Soybeans" ~ "barley",
    crop_name == "Dbl Crop Corn/Soybeans" ~ "corn",
    crop_name == "Dbl Crop Durum Wht/Sorghum" ~ "wheat",
    crop_name == "Dbl Crop Lettuce/Barley" ~ "lettuce",
    crop_name == "Dbl Crop Lettuce/Cantaloupe" ~ "lettuce",
    crop_name == "Dbl Crop Lettuce/Cotton" ~ "lettuce",
    crop_name == "Dbl Crop Lettuce/Durum Wht" ~ "lettuce",
    crop_name == "Dbl Crop Oats/Corn" ~ "oats",
    crop_name == "Dbl Crop Soybeans/Cotton" ~ "soybeans",
    crop_name == "Dbl Crop Soybeans/Oats" ~ "soybeans",
    crop_name == "Dbl Crop Triticale/Corn" ~ "triticale",
    crop_name == "Dbl Crop WinWht/Corn" ~ "wheat",
    crop_name == "Dbl Crop WinWht/Cotton" ~ "wheat", 
    crop_name == "Dbl Crop WinWht/Sorghum" ~ "wheat",
    crop_name == "Dbl Crop WinWht/Soybeans" ~ "wheat",
    .default = common_name
  ))

# Generate doubled entry so acreage-by-crop is correct when doing a cross-join
croplandcros_cdl_crops_list <- croplandcros_cdl_crops_list %>% 
  add_row(crop_name = "Dbl Crop Barley/Corn", common_name = "corn") %>% 
  add_row(crop_name = "Dbl Crop Barley/Sorghum", common_name = "sorghum") %>% 
  add_row(crop_name = "Dbl Crop Barley/Soybeans", common_name = "soybens") %>% 
  add_row(crop_name = "Dbl Crop Corn/Soybeans", common_name = "soybeans") %>% 
  add_row(crop_name = "Dbl Crop Durum Wht/Sorghum", common_name = "sorghum") %>% 
  add_row(crop_name = "Dbl Crop Lettuce/Barley", common_name = "barley") %>% 
  add_row(crop_name = "Dbl Crop Lettuce/Cantaloupe", common_name = "cantaloupes") %>% 
  add_row(crop_name = "Dbl Crop Lettuce/Cotton", common_name = "cotton") %>% 
  add_row(crop_name = "Dbl Crop Lettuce/Durum Wht", common_name = "wheat") %>% 
  add_row(crop_name = "Dbl Crop Oats/Corn", common_name = "corn") %>% 
  add_row(crop_name = "Dbl Crop Soybeans/Cotton", common_name = "cotton") %>% 
  add_row(crop_name = "Dbl Crop Soybeans/Oats", common_name = "oats") %>% 
  add_row(crop_name = "Dbl Crop Triticale/Corn", common_name = "corn") %>% 
  add_row(crop_name = "Dbl Crop WinWht/Corn", common_name = "corn") %>% 
  add_row(crop_name = "Dbl Crop WinWht/Cotton", common_name = "cotton") %>% 
  add_row(crop_name = "Dbl Crop WinWht/Sorghum", common_name = "sorghum") %>% 
  add_row(crop_name = "Dbl Crop WinWht/Soybeans", common_name = "soybeans") 

croplandcros_cdl_crops_list <- croplandcros_cdl_crops_list %>% 
  arrange(common_name, crop_name)




























# Add common name, this also doubles acreage for land that is double-cropped
croplandcros_cdl_county_acreage <- croplandcros_cdl_county_acreage %>% 
  full_join(croplandcros_cdl_crops_list)

# Aggregate by county-year-crop using common crop names
croplandcros_cdl_county_acreage <- croplandcros_cdl_county_acreage %>% 
  select(-crop_code, -crop_name) %>% 
  group_by(state_fips_code, county_fips_code, year, common_name) %>% 
  summarize(pixel_count = sum(pixel_count), acres = sum(acres)) %>% 
  ungroup()

# Load previously extracted NASS binaries
nass_price_received <- read_parquet(here("binaries", "nass_quickstats_price_received.parquet"))
nass_production <- read_parquet(here("binaries", "nass_quickstats_production.parquet"))
nass_acreage <- read_parquet(here("binaries", "nass_quickstats_acreage.parquet"))
nass_yield <- read_parquet(here("binaries", "nass_quickstats_yield.parquet"))
nass_census_acreage <- read_parquet(here("binaries", "nass_quickstats_census_acreage.parquet"))

# Get lists of NASS crops
price_received_crops <- nass_price_received %>% 
  distinct(group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc) %>% 
  mutate_all(as.character) %>% 
  arrange(commodity_desc, class_desc) %>% 
  mutate(source = "price_received")

production_crops <- nass_production %>% 
  distinct(group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc) %>% 
  mutate_all(as.character) %>% 
  arrange(commodity_desc, class_desc) %>% 
  mutate(source = "production")

yield_crops <- nass_yield %>% 
  distinct(group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc) %>% 
  mutate_all(as.character) %>% 
  arrange(commodity_desc, class_desc) %>% 
  mutate(source = "yield")

nass_crops <- bind_rows(price_received_crops, production_crops, yield_crops) %>% 
  group_by(group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc) %>% 
  summarize(sources = paste0(source, collapse = ", ")) %>% 
  ungroup() %>% 
  distinct() %>% 
  arrange(commodity_desc, class_desc, prodn_practice_desc, util_practice_desc)

test <- nass_production %>% 
  filter(commodity_desc == "ALMONDS")

%>% 
  filter(util_practice_desc == "ALL UTILIZATION PRACTICES" | util_practice_desc == "UTILIZED")



# Now do the same for crops in the NASS survey data

