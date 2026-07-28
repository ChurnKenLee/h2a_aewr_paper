# Purpose: Build fixed 2008-2011 county features for panel-IV clustering.
# Inputs: county crop-type acreage, climate basis, and gNATSGO soil-cell parquets.
# Output: data/intermediate/panel_iv_county_features.parquet.
# Run after: code/c01_clean/06_cdl_county_crop_acres.R and the H-2A prediction inputs.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("designs", "panel_iv", "design.R"))
library(arrow)
library(dplyr)
library(janitor)
library(purrr)
library(tibble)

cdl <- read_parquet(path_int(
  "croplandcros_county_crop_type_acres.parquet"
)) |>
  filter(!is.na(crop_type_label), crop_type_label != "non-crop")

climate <- read_parquet(path_int(
  "county_h2a_prediction_climate_basis_annual.parquet"
))

soil <- read_parquet(path_int(
  "county_h2a_prediction_gnatsgo_soil_cells.parquet"
))
# Fixed 2008-2011 county primitives

crop_names <- cdl |>
  distinct(crop_type_label) |>
  mutate(crop_var = paste0("share_cdl_", make_clean_names(crop_type_label)))

crop_features <- cdl |>
  filter(
    year >= DISSIMILARITY_IV_FEATURE_START_YEAR,
    year <= DISSIMILARITY_IV_FEATURE_END_YEAR
  ) |>
  left_join(crop_names, by = "crop_type_label") |>
  group_by(county_fips, year, crop_var) |>
  summarise(acres = sum(acres, na.rm = TRUE), .groups = "drop") |>
  group_by(county_fips, year) |>
  mutate(crop_share = acres / sum(acres, na.rm = TRUE)) |>
  ungroup() |>
  group_by(county_fips, crop_var) |>
  summarise(crop_share = mean(crop_share, na.rm = TRUE), .groups = "drop")

crop_features <- xtabs(
  crop_share ~ county_fips + crop_var,
  data = crop_features
) |>
  as.data.frame.matrix() |>
  rownames_to_column("county_fips") |>
  as_tibble()

climate_features <- climate |>
  filter(
    year >= DISSIMILARITY_IV_FEATURE_START_YEAR,
    year <= DISSIMILARITY_IV_FEATURE_END_YEAR
  ) |>
  select(county_fips, starts_with("normal_cb_")) |>
  group_by(county_fips) |>
  summarise(
    across(starts_with("normal_cb_"), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )

soil_vars <- c(
  "slope_r",
  "slopegradwta",
  "resdept_r",
  "aws025wta",
  "aws050wta",
  "aws0100wta",
  "aws0150wta",
  "wtdepannmin",
  "wtdepaprjunmin",
  "brockdepmin",
  "cropprodindex"
)

soil_cat_vars <- c("taxorder", "drainagecl", "hydgrp", "nirrcapcl")

soil_cont_features <- soil |>
  group_by(county_fips) |>
  summarise(
    across(all_of(soil_vars), ~ weighted.mean(.x, total_acres, na.rm = TRUE)),
    .groups = "drop"
  )

# This constructs a set of dataframes inside the list soil_cat_vars
# Each dataframe contains a categorical variable
# This is needed because each categorical variable can take multiple values
# Each column is the share of a particular value within the county
soil_cat_list <- list()
for (v in soil_cat_vars) {
  soil_value <- soil[[v]]
  soil_value[is.na(soil_value)] <- "missing"
  soil_value_names <- data.frame(
    soil_value = unique(soil_value),
    soil_value_clean = make_clean_names(unique(soil_value))
  )

  temp <- data.frame(
    county_fips = soil$county_fips,
    soil_value = soil_value,
    total_acres = soil$total_acres
  )
  temp <- temp |>
    left_join(
      soil_value_names,
      by = "soil_value",
      relationship = "many-to-one"
    )
  temp$soil_feature <- paste0("share_soil_", v, "_", temp$soil_value_clean)
  temp <- aggregate(
    total_acres ~ county_fips + soil_feature,
    temp,
    sum,
    na.rm = TRUE
  )
  temp$soil_share <- temp$total_acres /
    ave(
      temp$total_acres,
      temp$county_fips,
      FUN = sum
    )

  soil_cat_list[[v]] <- xtabs(
    soil_share ~ county_fips + soil_feature,
    data = temp
  ) |>
    as.data.frame.matrix() |>
    rownames_to_column("county_fips") |>
    as_tibble()
}

soil_cat_features <- reduce(soil_cat_list, full_join, by = "county_fips")

soil_features <- soil_cont_features |>
  full_join(soil_cat_features, by = "county_fips")

county_features <- crop_features |>
  full_join(climate_features, by = "county_fips") |>
  full_join(soil_features, by = "county_fips")

write_parquet(county_features, path_int("panel_iv_county_features.parquet"))
