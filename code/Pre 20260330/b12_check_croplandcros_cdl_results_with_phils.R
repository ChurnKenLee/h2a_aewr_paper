library(here)
library(arrow)
library(tidyverse)
library(dplyr)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)
library(tidycensus)
library(fixest)
library(ggplot2)

rm(list = ls())

# Load my crop data
ken_crop <- read_parquet(here("binaries", "croplandcros_county_crop_acres.parquet"))

phil_crop <- readRDS(here("For Ken 20241223", "County Outcomes", "county_cdl_outcomes_all.rds"))

# Harmonize data for comparison
ken_crop <- ken_crop %>% 
  mutate(county_code = paste(state_fips_code, county_fips_code, sep = ""))

croplandcros_cdl_crops_list_df <- ken_crop %>% 
  distinct(crop_name, .keep_all = TRUE) %>% 
  select(crop_name)

# Add common name to CroplandCROS CDL crops
croplandcros_cdl_crops_list_df <- croplandcros_cdl_crops_list_df %>% 
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
croplandcros_cdl_crops_list_df <- croplandcros_cdl_crops_list_df %>% 
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
croplandcros_cdl_crops_list_df <- croplandcros_cdl_crops_list_df %>% 
  add_row(crop_name = "Dbl Crop Barley/Corn", common_name = "corn") %>% 
  add_row(crop_name = "Dbl Crop Barley/Sorghum", common_name = "sorghum") %>% 
  add_row(crop_name = "Dbl Crop Barley/Soybeans", common_name = "soybeans") %>% 
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

# Create doubled entries
ken_crop <- ken_crop %>% 
  inner_join(croplandcros_cdl_crops_list_df)

# Drop double-cropped land for comparison, as Phil does not include double-cropped land
ken_crop <- ken_crop %>% 
  filter(!grepl("Dbl Crop", crop_name))

# Aggregate crops in my dataset
ken_crop <- ken_crop %>% 
  mutate(agg_name = case_when(
    common_name == "corn" ~ "corn",
    common_name == "sweet corn" ~ "corn",
    common_name == "cotton" ~ "cotton",
    common_name == "rice" ~ "rice",
    common_name == "soybeans" ~ "soybeans",
    common_name == "sugar beets" ~ "sugar",
    common_name == "sugarcane" ~ "sugar",
    common_name == "wheat" ~ "wheat"
    )) %>% 
  filter(!is.na(agg_name))

ken_crop <- ken_crop %>% 
  group_by(county_code, year, agg_name) %>% 
  summarize(ken_acres = sum(acres), ken_pixels = sum(pixel_count)) %>% 
  ungroup()

# Create 0 acres for crops not observed in any given county-year
ken_crop <- ken_crop %>% 
  complete(county_code, year, agg_name, fill = list(ken_acres = 0, ken_pixels = 0))

# Reshape Phil's data into long for merging
phil_crop <- phil_crop %>% 
  select(county_code, year, corn, cotton, rice, soybeans, sugar, wheat)

phil_crop <- phil_crop %>% 
  pivot_longer(cols = -one_of(c("county_code", "year")), names_to = "agg_name", values_to = "phil_area")

# Merge with Phil dataset
both_set <- ken_crop %>% 
  full_join(phil_crop)

# Check if they are the same
model <- feols(phil_area ~ ken_pixels, data = both_set)
summary(model)

both_set %>% 
  ggplot(
  aes(x = ken_pixels, y = phil_area)
  ) + 
  geom_point()



