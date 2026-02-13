library(here)
library(arrow)
library(tidyverse)
library(dplyr)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)

rm(list = ls())

# Clean variable names
# df <- read_parquet(here("binaries", "census_of_agriculture_2002.parquet"))
# df <- df %>% clean_names()
# write_parquet(df, here("binaries", "census_of_agriculture_2002.parquet"))
# 
# df <- read_parquet(here("binaries", "census_of_agriculture_2007.parquet"))
# df <- df %>% clean_names()
# write_parquet(df, here("binaries", "census_of_agriculture_2007.parquet"))
# 
# df <- read_parquet(here("binaries", "census_of_agriculture_2012.parquet"))
# df <- df %>% clean_names()
# write_parquet(df, here("binaries", "census_of_agriculture_2012.parquet"))
# 
# df <- read_parquet(here("binaries", "census_of_agriculture_2017.parquet"))
# df <- df %>% clean_names()
# write_parquet(df, here("binaries", "census_of_agriculture_2017.parquet"))

# Create county aggregates for heads of animals
census_years_list <- c(2007, 2012, 2017)
county_animal_df_list <- list()

i <- 2017
file_name <- paste0("census_of_agriculture_", i, ".parquet")

df <- read_parquet(here("binaries", file_name)) %>% 
  clean_names() %>% 
  arrange(state_name, county_name, sector_desc, group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc)

# Animal totals
animal_df <- df %>% 
  filter(agg_level_desc == "COUNTY") %>% 
  filter(sector_desc == "ANIMALS & PRODUCTS") %>% 
  filter(unit_desc != "OPERATIONS") %>% # Don't need farm count
  filter(domaincat_desc == "NOT SPECIFIED")  # Removes distributional counts
  
test <- animal_df %>% distinct(short_desc, .keep_all = TRUE) %>% 
  filter((statisticcat_desc == "INVENTORY") | (statisticcat_desc == "SALES & DISTRIBUTION")) # Aquaculture only has sales

animal_short_desc_df <- animal_df %>% 
  distinct(short_desc, .keep_all=TRUE) %>% 
  arrange(group_desc, commodity_desc, class_desc, statisticcat_desc)

# Define indicators for aggregating animal heads
animal_heads_df <- animal_short_desc_df %>% 
  mutate(head_indicator = FALSE)

# Cattle
animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "CATTLE, INCL CALVES - INVENTORY", TRUE, head_indicator))

# Goats
animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "GOATS - INVENTORY", TRUE, head_indicator))

# Hogs
animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "HOGS - INVENTORY", TRUE, head_indicator))

# Sheep
animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "SHEEP, INCL LAMBS - INVENTORY", TRUE, head_indicator))

# Chickens (3 kinds: broilers, layers, pullets)
animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(commodity_desc == "CHICKENS", TRUE, head_indicator))

# Other poultry
animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "DUCKS - INVENTORY", TRUE, head_indicator))

animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "EMUS - INVENTORY", TRUE, head_indicator))

animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "GEESE - INVENTORY", TRUE, head_indicator))

animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "PHEASANTS - INVENTORY", TRUE, head_indicator))

animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "PIGEONS & SQUAB - INVENTORY", TRUE, head_indicator))

animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "POULTRY, OTHER - INVENTORY", TRUE, head_indicator))

animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "QUAIL - INVENTORY", TRUE, head_indicator))

animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "TURKEY - INVENTORY", TRUE, head_indicator))

animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "OSTRICHES - INVENTORY", TRUE, head_indicator))

# Specialty animals
# Alpacas
animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "ALPACAS - INVENTORY", TRUE, head_indicator))

# Bison
animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "BISON - INVENTORY", TRUE, head_indicator))

# Deer
animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "DEER - INVENTORY", TRUE, head_indicator))

# Elk
animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "ELK - INVENTORY", TRUE, head_indicator))

# Equine
animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "EQUINE, HORSES & PONIES - INVENTORY", TRUE, head_indicator))

animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "EQUINE, MULES & BURROS & DONKEYS - INVENTORY", TRUE, head_indicator))

# Bees
animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "HONEY, BEE COLONIES - INVENTORY, MEASURED IN COLONIES", TRUE, head_indicator))

# Llamas
animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "LLAMAS - INVENTORY", TRUE, head_indicator))

# Rabbits
animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "RABBITS, LIVE AND PELTS - INVENTORY", TRUE, head_indicator))

# Mink
animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else(short_desc == "MINK, LIVE AND PELTS - INVENTORY", TRUE, head_indicator))

# Aquaculture (in lbs)
animal_heads_df <- animal_heads_df %>% 
  mutate(head_indicator = if_else((group_desc == "AQUACULTURE") & (unit_desc == "LB"), TRUE, head_indicator))

aquaculture_df <- animal_df %>% 
  filter(group_desc == "AQUACULTURE")

head_indicator_df <- animal_heads_df %>% 
  select(commodity_desc, class_desc, short_desc, head_indicator) %>% 
  rename(aggregation_indicator = head_indicator)

# Merge aggregation indicators back into original df
county_animal_df <- animal_df %>% 
  left_join(head_indicator_df)

# Remove commas
county_animal_df <- county_animal_df %>% 
  mutate(harmonized_unit = as.numeric(gsub(",", "", value)))

# Aggregate into product types
# Drop values that we don't need to add to aggregate
county_animal_df <- county_animal_df %>% 
  filter(aggregation_indicator == TRUE)

# Aggregate by county x sector-group-commodity-class
county_animal_df <- county_animal_df %>% 
  mutate(harmonized_unit = if_else(is.na(harmonized_unit), 0, harmonized_unit)) %>% 
  group_by(state_fips_code, county_code, state_name, county_name, sector_desc, group_desc, commodity_desc, class_desc, unit_desc) %>% 
  summarize(harmonized_value = sum(harmonized_unit)) %>% 
  filter(!is.na(county_name)) %>% 
  mutate(year = i)

county_animal_df_list[[as.character(i)]] <- county_animal_df


for (i in census_years_list) {
  file_name <- paste0("census_of_agriculture_", i, ".parquet")
  df <- read_parquet(here("binaries", file_name)) %>% 
    arrange(state_name, county_name, sector_desc, group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc)
  
  # Animal totals
  animal_df <- df %>% 
    filter(agg_level_desc == "COUNTY") %>% 
    filter(sector_desc == "ANIMALS & PRODUCTS") %>% 
    filter(unit_desc != "OPERATIONS") %>% # Don't need farm count
    filter(domaincat_desc == "NOT SPECIFIED") %>%  # Removes distributional counts
    filter((statisticcat_desc == "INVENTORY") | (statisticcat_desc == "SALES & DISTRIBUTION")) # Aquaculture only has sales
  
  animal_short_desc_df <- animal_df %>% 
    distinct(short_desc, .keep_all=TRUE) %>% 
    arrange(group_desc, commodity_desc, class_desc, statisticcat_desc)
  
  # Define indicators for aggregating animal heads
  animal_heads_df <- animal_short_desc_df %>% 
    mutate(head_indicator = FALSE)
  
  # Cattle
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "CATTLE, INCL CALVES - INVENTORY", TRUE, head_indicator))
  
  # Goats
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "GOATS - INVENTORY", TRUE, head_indicator))
  
  # Hogs
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "HOGS - INVENTORY", TRUE, head_indicator))
  
  # Sheep
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "SHEEP, INCL LAMBS - INVENTORY", TRUE, head_indicator))
  
  # Chickens (3 kinds: broilers, layers, pullets)
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(commodity_desc == "CHICKENS", TRUE, head_indicator))
  
  # Other poultry
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "DUCKS - INVENTORY", TRUE, head_indicator))
  
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "EMUS - INVENTORY", TRUE, head_indicator))
  
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "GEESE - INVENTORY", TRUE, head_indicator))
  
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "PHEASANTS - INVENTORY", TRUE, head_indicator))
  
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "PIGEONS & SQUAB - INVENTORY", TRUE, head_indicator))
  
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "POULTRY, OTHER - INVENTORY", TRUE, head_indicator))
  
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "QUAIL - INVENTORY", TRUE, head_indicator))
  
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "TURKEY - INVENTORY", TRUE, head_indicator))
  
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "OSTRICHES - INVENTORY", TRUE, head_indicator))
  
  # Specialty animals
  # Alpacas
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "ALPACAS - INVENTORY", TRUE, head_indicator))
  
  # Bison
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "BISON - INVENTORY", TRUE, head_indicator))
  
  # Deer
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "DEER - INVENTORY", TRUE, head_indicator))
  
  # Elk
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "ELK - INVENTORY", TRUE, head_indicator))
  
  # Equine
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "EQUINE, HORSES & PONIES - INVENTORY", TRUE, head_indicator))
  
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "EQUINE, MULES & BURROS & DONKEYS - INVENTORY", TRUE, head_indicator))
  
  # Bees
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "HONEY, BEE COLONIES - INVENTORY, MEASURED IN COLONIES", TRUE, head_indicator))
  
  # Llamas
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "LLAMAS - INVENTORY", TRUE, head_indicator))
  
  # Rabbits
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "RABBITS, LIVE AND PELTS - INVENTORY", TRUE, head_indicator))
  
  # Mink
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else(short_desc == "MINK, LIVE AND PELTS - INVENTORY", TRUE, head_indicator))
  
  # Aquaculture (in lbs)
  animal_heads_df <- animal_heads_df %>% 
    mutate(head_indicator = if_else((group_desc == "AQUACULTURE") & (unit_desc == "LB"), TRUE, head_indicator))
  
  aquaculture_df <- animal_df %>% 
    filter(group_desc == "AQUACULTURE")
  
  head_indicator_df <- animal_heads_df %>% 
    select(commodity_desc, class_desc, short_desc, head_indicator) %>% 
    rename(aggregation_indicator = head_indicator)
  
  # Merge aggregation indicators back into original df
  county_animal_df <- animal_df %>% 
    left_join(head_indicator_df)
  
  # Remove commas
  county_animal_df <- county_animal_df %>% 
    mutate(harmonized_unit = as.numeric(gsub(",", "", value)))
  
  # Aggregate into product types
  # Drop values that we don't need to add to aggregate
  county_animal_df <- county_animal_df %>% 
    filter(aggregation_indicator == TRUE)
  
  # Aggregate by county x sector-group-commodity-class
  county_animal_df <- county_animal_df %>% 
    mutate(harmonized_unit = if_else(is.na(harmonized_unit), 0, harmonized_unit)) %>% 
    group_by(state_fips_code, county_code, state_name, county_name, sector_desc, group_desc, commodity_desc, class_desc, unit_desc) %>% 
    summarize(harmonized_value = sum(harmonized_unit)) %>% 
    filter(!is.na(county_name)) %>% 
    mutate(year = i)
  
  county_animal_df_list[[as.character(i)]] <- county_animal_df
}

# Create county aggregates for acres of crops
county_crop_df_list <- list()
for (i in census_years_list) {
  file_name <- paste0("census_of_agriculture_", i, ".parquet")
  df <- read_parquet(here("binaries", file_name)) %>% 
    arrange(state_name, county_name, sector_desc, group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc)
  
  # Plant totals
  crop_df <- df %>% 
    filter(agg_level_desc == "COUNTY") %>% 
    filter(sector_desc == "CROPS") %>% 
    filter(unit_desc != "OPERATIONS") %>% 
    filter(domaincat_desc == "NOT SPECIFIED") %>% # Removes distribution counts
    filter((unit_desc == "ACRES") | (unit_desc == "SQ FT")) %>% # We don't need dollar values
    filter(!grepl('TOTALS', commodity_desc))
  
  crop_short_desc_df <- crop_df %>% 
    distinct(short_desc, .keep_all=TRUE) %>% 
    arrange(commodity_desc, unit_desc)
  
  # Define indicators for aggregating crop acreage
  # Check for presence of universal measure within each crop type
  # Group by crop type
  acreage_count_df <- crop_short_desc_df %>% 
    group_by(commodity_desc, class_desc) %>% 
    mutate(universal_production = prodn_practice_desc == "ALL PRODUCTION PRACTICES") %>% 
    mutate(universal_utilization = util_practice_desc == "ALL UTILIZATION PRACTICES") %>% 
    mutate(area_measure = (statisticcat_desc == "AREA HARVESTED") | (statisticcat_desc == "AREA BEARING & NON-BEARING") | (statisticcat_desc == "AREA IN PRODUCTION"))
  
  # Fill in crops that do not have universal area measure
  # Denote universal measure for plants that have only 1 specific use
  # Grasses, legumes, grasses & legumes, other, and mustard acreage are used for seeds only
  acreage_count_df <- acreage_count_df %>% 
    mutate(universal_utilization = if_else((commodity_desc == "GRASSES" | commodity_desc == "LEGUMES" | commodity_desc == "MUSTARD" | commodity_desc == "GRASSES & LEGUMES, OTHER") & util_practice_desc == "SEED", TRUE, universal_utilization))
  
  # Dill is used for oils only
  acreage_count_df <- acreage_count_df %>% 
    mutate(universal_utilization = if_else((commodity_desc == "DILL") & util_practice_desc == "OIL", TRUE, universal_utilization))
  
  # Mint is used for oil only
  acreage_count_df <- acreage_count_df %>% 
    mutate(universal_utilization = if_else((commodity_desc == "MINT") & util_practice_desc == "OIL", TRUE, universal_utilization))
  
  # Popcorn are all shelled
  acreage_count_df <- acreage_count_df %>% 
    mutate(universal_utilization = if_else((commodity_desc == "POPCORN") & util_practice_desc == "SHELLED", TRUE, universal_utilization))
  
  # Now we deal with plants with multiple production practices or uses
  # Plants that are split into "IN THE OPEN" and "UNDER PROTECTION"
  acreage_count_df <- acreage_count_df %>% 
    mutate(split_production = if_else((prodn_practice_desc == "IN THE OPEN") | (prodn_practice_desc == "UNDER PROTECTION"), TRUE, FALSE))
  
  # Now we deal with plants with split uses
  # Corn is used for grain and silage
  acreage_count_df <- acreage_count_df %>% 
    mutate(split_utilization = if_else((commodity_desc == "CORN") & ((util_practice_desc == "GRAIN") | (util_practice_desc == "SILAGE")), TRUE, FALSE))
  
  # Sorghum is used for grain, silage, and syrup
  acreage_count_df <- acreage_count_df %>% 
    mutate(split_utilization = if_else((commodity_desc == "SORGHUM") & ((util_practice_desc == "GRAIN") | (util_practice_desc == "SILAGE") | (util_practice_desc == "SYRUP")), TRUE, split_utilization))
  
  # Sugarcane is used for seed and sugar
  acreage_count_df <- acreage_count_df %>% 
    mutate(split_utilization = if_else((commodity_desc == "SUGARCANE") & ((util_practice_desc == "SEED") | (util_practice_desc == "SUGAR")), TRUE, split_utilization))
  
  # Some crops may have multiple universal acre measure
  # Short term woody crop has 2 universal measures: area in production and area harvested
  # Short rotation woody crops have minimal maintenance requirements, so use area harvested
  acreage_count_df <- acreage_count_df %>% 
    mutate(universal_production = if_else((commodity_desc == "SHORT TERM WOODY CROPS") & (statisticcat_desc == "AREA IN PRODUCTION"), FALSE, universal_production))
  
  # Summarize each crop
  # Universal production and utilization practices
  acreage_count_df <- acreage_count_df %>% 
    mutate(universal_measure = universal_production & universal_utilization & area_measure) %>% 
    mutate(universal_measure_count = sum(universal_measure))
  
  # Split production practices, universal utilization practices
  acreage_count_df <- acreage_count_df %>% 
    mutate(split_production_measure = split_production & universal_utilization & area_measure) %>% 
    mutate(split_production_measure_count = sum(split_production_measure))
  
  # Split utilization practices, universal production practices
  acreage_count_df <- acreage_count_df %>% 
    mutate(split_utilization_measure = split_utilization & universal_production & area_measure) %>% 
    mutate(split_utilization_measure_count = sum(split_utilization_measure))
  
  acreage_count_df <- acreage_count_df %>% 
    arrange(split_utilization_measure_count, split_production_measure_count, universal_measure_count, commodity_desc, class_desc)
  
  # Define acreage indicator
  acreage_count_df <- acreage_count_df %>% 
    mutate(acreage_indicator = (universal_measure + split_production_measure + split_utilization_measure)>=1)
  
  # Some crops have duplicate observations due to additional observations that into classes, e.g., peaches-all and clingstone and freestone
  # We want to drop these duplicate observations, doing so by setting the aggregation indicators to FALSE
  # Cotton has upland and pima types
  acreage_count_df <- acreage_count_df %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "COTTON") & (class_desc == "UPLAND"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "COTTON") & (class_desc == "PIMA"), FALSE, acreage_indicator))
  
  # Hay has alfalfa, small grain, tame, (excl alfalfa & small grain), wild
  acreage_count_df <- acreage_count_df %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "HAY") & (class_desc == "ALFALFA"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "HAY") & (class_desc == "SMALL GRAIN"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "HAY") & (class_desc == "TAME, (EXCL ALFALFA & SMALL GRAIN)"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "HAY") & (class_desc == "WILD"), FALSE, acreage_indicator))
  
  # Haylage has (excl alfalfa), alfalfa
  acreage_count_df <- acreage_count_df %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "HAYLAGE") & (class_desc == "(EXCL ALFALFA)"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "HAYLAGE") & (class_desc == "ALFALFA"), FALSE, acreage_indicator))
  
  # There is also a combined hay & haylage type
  acreage_count_df <- acreage_count_df %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "HAY & HAYLAGE"), FALSE, acreage_indicator))
  
  # Lettuce has head, Romaine, and leaf
  acreage_count_df <- acreage_count_df %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "LETTUCE") & (class_desc == "HEAD"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "LETTUCE") & (class_desc == "LEAF"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "LETTUCE") & (class_desc == "ROMAINE"), FALSE, acreage_indicator))
  
  # Mint has peppermint, spearmint, tea leaves
  acreage_count_df <- acreage_count_df %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "MINT") & (class_desc == "PEPPERMINT"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "MINT") & (class_desc == "SPEARMINT"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "MINT") & (class_desc == "TEA LEAVES"), FALSE, acreage_indicator))
  
  # Peaches has Clingstone and Freestone
  acreage_count_df <- acreage_count_df %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "PEACHES") & (class_desc == "FREESTONE"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "PEACHES") & (class_desc == "CLINGSTONE"), FALSE, acreage_indicator))
  
  # Pears has Bartlett and (excl Bartlett)
  acreage_count_df <- acreage_count_df %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "PEARS") & (class_desc == "BARTLETT"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "PEARS") & (class_desc == "(EXCL BARTLETT)"), FALSE, acreage_indicator))
  
  # Pecans has Improved and native & seedling
  acreage_count_df <- acreage_count_df %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "PECANS") & (class_desc == "IMPROVED"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "PECANS") & (class_desc == "NATIVE & SEEDLING"), FALSE, acreage_indicator))
  
  # Raspberries has red and black
  acreage_count_df <- acreage_count_df %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "RASPBERRIES") & (class_desc == "BLACK"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "RASPBERRIES") & (class_desc == "RED"), FALSE, acreage_indicator))
  
  # Squash has summer and winter, we want to keep these separate?
  acreage_count_df <- acreage_count_df %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "SQUASH") & (class_desc == "ALL CLASSES"), FALSE, acreage_indicator))
  
  # Sunflower has oil type and non-oil type
  acreage_count_df <- acreage_count_df %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "SUNFLOWER") & (class_desc == "NON-OIL TYPE"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "SUNFLOWER") & (class_desc == "OIL TYPE"), FALSE, acreage_indicator))
  
  # Wheat has winter, spring, durum, spring, (excl durum)
  acreage_count_df <- acreage_count_df %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "WHEAT") & (class_desc == "WINTER"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "WHEAT") & (class_desc == "SPRING"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "WHEAT") & (class_desc == "SPRING, DURUM"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "WHEAT") & (class_desc == "SPRING, (EXCL DURUM)"), FALSE, acreage_indicator))
  
  # Oranges has mid & navel, valencia
  acreage_count_df <- acreage_count_df %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "ORANGES") & (class_desc == "MID & NAVEL"), FALSE, acreage_indicator)) %>% 
    mutate(acreage_indicator = if_else((commodity_desc == "ORANGES") & (class_desc == "VALENCIA"), FALSE, acreage_indicator))
  
  # Now to create county-crop table with indicator for obs that will be used for aggregate measures
  # Keep only relevant columns
  acreage_indicator_df <- acreage_count_df %>% 
    ungroup() %>% 
    select(commodity_desc, class_desc, short_desc, acreage_indicator) %>% 
    rename(aggregation_indicator = acreage_indicator)
  
  # Merge aggregation indicators back into original df
  county_crop_df <- crop_df %>% 
    left_join(acreage_indicator_df)
  
  # Harmonize unit measures, converting from sq ft to acres
  county_crop_df <- county_crop_df %>% 
    mutate(harmonized_unit = as.numeric(gsub(",", "", value))) %>% 
    mutate(harmonized_unit = if_else(unit_desc == "SQ FT", harmonized_unit/43560, harmonized_unit))
  
  # Aggregate into product types
  # Drop values that we don't need to add to aggregate
  county_crop_df <- county_crop_df %>% 
    filter(aggregation_indicator == TRUE)
  
  # Aggregate by county x sector-group-commodity-class
  county_crop_df <- county_crop_df %>% 
    mutate(harmonized_unit = if_else(is.na(harmonized_unit), 0, harmonized_unit)) %>% 
    group_by(state_fips_code, county_code, state_name, county_name, sector_desc, group_desc, commodity_desc, class_desc) %>% 
    summarize(harmonized_value = sum(harmonized_unit)) %>% 
    filter(!is.na(county_name)) %>% 
    mutate(unit_desc = "ACRES") %>% 
    mutate(year = i)
  
  county_crop_df_list[[as.character(i)]] <- county_crop_df
}

# Create county aggregates for labor
county_labor_df_list <- list()

for (i in census_years_list) {
  # Load file
  file_name <- paste0("census_of_agriculture_", i, ".parquet")
  df <- read_parquet(here("binaries", file_name)) %>% 
    arrange(state_name, county_name, sector_desc, group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc)
  
  # We only want worker counts
  labor_df <- df %>% 
    filter(agg_level_desc == "COUNTY") %>% 
    filter(sector_desc == "ECONOMICS") %>% 
    filter(commodity_desc == "LABOR") %>% 
    filter(unit_desc == "NUMBER") %>% 
    filter(domaincat_desc == "NOT SPECIFIED")
  
  # Keep only information we want
  labor_df <- labor_df %>% 
    mutate(harmonized_value = as.numeric(gsub(",", "", value))) %>% 
    select(state_fips_code, county_code, state_name, county_name, sector_desc, group_desc, commodity_desc, class_desc, unit_desc, harmonized_value) %>% 
    mutate(year = i)
  
  county_labor_df_list[[as.character(i)]] <- labor_df
}

# Revenue
county_income_df_list <- list()

for (i in census_years_list) {
  # Load file
  file_name <- paste0("census_of_agriculture_", i, ".parquet")
  df <- read_parquet(here("binaries", file_name)) %>% 
    arrange(state_name, county_name, sector_desc, group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc)
  
  # We only want income variables
  income_df <- df %>% 
    filter(agg_level_desc == "COUNTY") %>% 
    filter(sector_desc == "ECONOMICS") %>% 
    filter(group_desc == "INCOME") %>% 
    filter(unit_desc == "$") %>% 
    filter(domaincat_desc == "NOT SPECIFIED") %>% 
    filter(
      short_desc == "COMMODITY TOTALS - SALES, MEASURED IN $" |
        short_desc == "GOVT PROGRAMS, FEDERAL - RECEIPTS, MEASURED IN $" |
        short_desc == "INCOME, FARM-RELATED - RECEIPTS, MEASURED IN $" |
        short_desc == "CCC LOANS - RECEIPTS, MEASURED IN $"
    )
  
  # We want to keep only the information we want
  income_df <- income_df %>% 
    mutate(harmonized_value = as.numeric(gsub(",", "", value))) %>% 
    select(state_fips_code, county_code, state_name, county_name, sector_desc, group_desc, commodity_desc, class_desc, unit_desc, harmonized_value) %>% 
    mutate(year = i)
  
  county_income_df_list[[as.character(i)]] <- income_df
}

# Expenses
county_expenses_df_list <- list()

for (i in census_years_list) {
  #Load file
  file_name <- paste0("census_of_agriculture_", i, ".parquet")
  df <- read_parquet(here("binaries", file_name)) %>% 
    arrange(state_name, county_name, sector_desc, group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc)
  
  # We only want expense variables
  expenses_df <- df %>% 
    filter(agg_level_desc == "COUNTY") %>% 
    filter(sector_desc == "ECONOMICS") %>% 
    filter(group_desc == "EXPENSES") %>% 
    filter(unit_desc == "$") %>% 
    filter(domaincat_desc == "NOT SPECIFIED")
  
  # We want to keep only the information we want
  expenses_df <- expenses_df %>% 
    mutate(harmonized_value = as.numeric(gsub(",", "", value))) %>% 
    select(state_fips_code, county_code, state_name, county_name, sector_desc, group_desc, commodity_desc, class_desc, unit_desc, harmonized_value) %>% 
    mutate(year = i)
  
  county_expenses_df_list[[as.character(i)]] <- expenses_df
}

# Assets
county_assets_df_list <- list()

for (i in census_years_list) {
  #Load file
  file_name <- paste0("census_of_agriculture_", i, ".parquet")
  df <- read_parquet(here("binaries", file_name)) %>% 
    arrange(state_name, county_name, sector_desc, group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc)
  
  # We only want asset variables
  assets_df <- df %>% 
    filter(agg_level_desc == "COUNTY") %>% 
    filter(sector_desc == "ECONOMICS") %>% 
    filter(group_desc == "FARMS & LAND & ASSETS") %>% 
    filter(unit_desc != "OPERATIONS") %>% 
    filter(domaincat_desc == "NOT SPECIFIED") %>% 
    filter(prodn_practice_desc == "ALL PRODUCTION PRACTICES") %>% 
    filter(unit_desc == "$" | unit_desc == "ACRES" | unit_desc == "NUMBER")
  
  # We want to keep only the information we want
  assets_df <- assets_df %>% 
    mutate(harmonized_value = as.numeric(gsub(",", "", value))) %>% 
    select(state_fips_code, county_code, state_name, county_name, sector_desc, group_desc, commodity_desc, class_desc, unit_desc, harmonized_value) %>% 
    mutate(year = i)
  
  county_assets_df_list[[as.character(i)]] <- assets_df
}

# Combine all dataframes
county_crop_df <- bind_rows(county_crop_df_list, .id = NULL)
county_animal_df <- bind_rows(county_animal_df_list, .id = NULL)
county_labor_df <- bind_rows(county_labor_df_list, .id = NULL)
county_income_df <- bind_rows(county_income_df_list, .id = NULL)
county_expenses_df <- bind_rows(county_expenses_df_list, .id = NULL)
county_assets_df <- bind_rows(county_assets_df_list, .id = NULL)

census_df <- bind_rows(
  county_crop_df,
  county_animal_df,
  county_labor_df,
  county_income_df,
  county_expenses_df,
  county_assets_df,
  .id = NULL
) %>% 
  rename(county_fips_code = county_code)

# Export
census_df %>% 
  write_parquet(here("files_for_phil", "census_of_agriculture.parquet"))
