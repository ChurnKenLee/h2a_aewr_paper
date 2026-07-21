# Purpose: Build fixed 2008-2011 county crop, climate, and soil features for IV clustering.
# Inputs: county crop-type acreage, climate basis, and gNATSGO soil-cell parquets.
# Output: data/intermediate/iv_county_features.parquet.
# Run after: code/c01_clean/06_cdl_county_crop_acres.R and the H-2A prediction inputs.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
library(arrow)
library(tidyverse)
library(janitor)

cdl <- read_parquet(path_int(
  "croplandcros_county_crop_type_acres.parquet"
)) %>%
  clean_names() %>%
  filter(!is.na(crop_type_label), crop_type_label != "non-crop")

climate <- read_parquet(path_int(
  "county_h2a_prediction_climate_basis_annual.parquet"
)) %>%
  clean_names()

soil <- read_parquet(path_int(
  "county_h2a_prediction_gnatsgo_soil_cells.parquet"
)) %>%
  clean_names()
# Fixed 2008-2011 county primitives

crop_names <- cdl %>%
  distinct(crop_type_label) %>%
  mutate(crop_var = paste0("share_cdl_", make_clean_names(crop_type_label)))

crop_features <- cdl %>%
  filter(year >= 2008, year <= 2011) %>%
  mutate(
    county_ansi = countyfips
  ) %>%
  left_join(crop_names, by = "crop_type_label") %>%
  group_by(county_ansi, year, crop_var) %>%
  summarise(acres = sum(acres, na.rm = TRUE), .groups = "drop") %>%
  group_by(county_ansi, year) %>%
  mutate(crop_share = acres / sum(acres, na.rm = TRUE)) %>%
  ungroup() %>%
  group_by(county_ansi, crop_var) %>%
  summarise(crop_share = mean(crop_share, na.rm = TRUE), .groups = "drop")

crop_features <- xtabs(
  crop_share ~ county_ansi + crop_var,
  data = crop_features
) %>%
  as.data.frame.matrix() %>%
  rownames_to_column("county_ansi") %>%
  as_tibble()

climate_features <- climate %>%
  mutate(
    county_ansi = fips
  ) %>%
  filter(year >= 2008, year <= 2011) %>%
  select(county_ansi, starts_with("normal_cb_")) %>%
  group_by(county_ansi) %>%
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

soil_cont_features <- soil %>%
  group_by(county_ansi) %>%
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
    county_ansi = soil$county_ansi,
    soil_value = soil_value,
    total_acres = soil$total_acres
  )
  temp <- merge(
    temp,
    soil_value_names,
    by = "soil_value",
    all.x = TRUE,
    all.y = FALSE
  )
  temp$soil_feature <- paste0("share_soil_", v, "_", temp$soil_value_clean)
  temp <- aggregate(
    total_acres ~ county_ansi + soil_feature,
    temp,
    sum,
    na.rm = TRUE
  )
  temp$soil_share <- temp$total_acres /
    ave(
      temp$total_acres,
      temp$county_ansi,
      FUN = sum
    )

  soil_cat_list[[v]] <- xtabs(
    soil_share ~ county_ansi + soil_feature,
    data = temp
  ) %>%
    as.data.frame.matrix() %>%
    rownames_to_column("county_ansi") %>%
    as_tibble()
}

soil_cat_features <- reduce(soil_cat_list, full_join, by = "county_ansi")

soil_features <- soil_cont_features %>%
  full_join(soil_cat_features, by = "county_ansi")

county_features <- crop_features %>%
  full_join(climate_features, by = "county_ansi") %>%
  full_join(soil_features, by = "county_ansi")

county_feature_names <- setdiff(names(county_features), "county_ansi")

write_parquet(county_features, path_int("iv_county_features.parquet"))
