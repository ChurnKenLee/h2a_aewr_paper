# Purpose: Aggregate CDL county acreage to project crop-type groups.
# Inputs: cdl_crop_type.parquet and croplandcros_county_crop_acres.parquet.
# Output: data/intermediate/croplandcros_county_crop_type_acres.parquet.
# Run after: the corresponding a-stage CDL scripts.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
source(path_code("c00_shared", "fips.R"))
library(arrow)
library(dplyr)

cdl_codes <- read_parquet(path_int("cdl_crop_type.parquet")) %>%
  select(cdl_code, crop_type_label) %>%
  rename(crop_code = cdl_code)

cdl_data <- read_parquet(path_int("croplandcros_county_crop_acres.parquet"))
cdl_data <- cdl_data %>%
  left_join(cdl_codes, by = c("crop_code"))

cdl_data_collapse <- cdl_data %>%
  group_by(crop_type_label, year, fips) %>%
  summarise(
    acres = sum(acres, na.rm = T),
    crop_count = n()
  ) %>%
  ungroup()

cdl_data_collapse <- cdl_data_collapse %>%
  ungroup() %>%
  mutate(countyfips = county_fips(fips)) %>%
  dplyr::select(-fips)

write_parquet(
  cdl_data_collapse,
  path_int("croplandcros_county_crop_type_acres.parquet")
)
