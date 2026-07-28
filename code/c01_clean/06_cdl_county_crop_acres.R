# Purpose: Aggregate CDL county acreage to project crop-type groups.
# Inputs: cdl_crop_type.parquet and croplandcros_county_crop_acres.parquet.
# Output: data/intermediate/croplandcros_county_crop_type_acres.parquet.
# Run after: the corresponding a-stage CDL scripts.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
library(arrow)
library(dplyr)

cdl_codes <- read_parquet(path_int("cdl_crop_type.parquet")) |>
  select(cdl_code, crop_type_label) |>
  rename(crop_code = cdl_code)

cdl_data <- read_parquet(path_int("croplandcros_county_crop_acres.parquet"))
cdl_data <- cdl_data |>
  left_join(cdl_codes, by = c("crop_code"))

cdl_data_collapse <- cdl_data |>
  group_by(crop_type_label, year, county_fips) |>
  summarise(
    acres = sum(acres, na.rm = TRUE),
    crop_count = n(),
    .groups = "drop"
  )

write_parquet(
  cdl_data_collapse,
  path_int("croplandcros_county_crop_type_acres.parquet")
)
