rm(list = ls())

if (file.exists("paths.R")) {
  source("paths.R")
} else {
  source(file.path("code", "paths.R"))
}

library(arrow)
library(tidyverse)

## CZ-level dissimilar-donor AEWR instrument ----------------------------------

baseline_years <- 2008:2011
scale_year <- 2011
donor_top_n <- 5

id_cols <- c("aewr_region_num", "cz_out10")
wage_cols <- paste0("wage_p", c(10, 25, 50, 75, 90))

soil_cont_cols <- c(
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

soil_cat_cols <- c(
  "taxorder",
  "taxsuborder",
  "taxgrtgroup",
  "drainagecl",
  "hydgrp",
  "nirrcapcl"
)

cz_ids <- function(.data) {
  .data %>%
    mutate(
      aewr_region_num = as.integer(aewr_region_num),
      cz_out10 = as.character(cz_out10)
    )
}

positive_weight <- function(x) {
  if_else(!is.na(x) & x > 0, x, 1)
}

wtd_mean <- function(x, w) {
  keep <- !is.na(x) & !is.na(w) & w > 0

  if (!any(keep)) {
    return(NA_real_)
  }

  weighted.mean(x[keep], w[keep])
}

safe_feature_names <- function(x) {
  x %>%
    str_replace_all("[^0-9A-Za-z]+", "_") %>%
    str_replace_all("^_+|_+$", "") %>%
    str_to_lower() %>%
    na_if("") %>%
    replace_na("feature") %>%
    make.unique(sep = "_")
}

standardize_features <- function(df, feature_map) {
  feature_cols <- feature_map$feature

  missing_cols <- setdiff(feature_cols, names(df))
  if (length(missing_cols) > 0) {
    stop(
      "Missing feature columns: ",
      str_c(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }

  values <- df %>%
    mutate(row_id = row_number()) %>%
    select(row_id, all_of(feature_cols)) %>%
    pivot_longer(
      -row_id,
      names_to = "feature",
      values_to = "value",
      values_transform = list(value = as.numeric)
    )

  impute_stats <- values %>%
    group_by(feature) %>%
    summarise(
      imputation_median = median(value[is.finite(value)], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      imputation_median = if_else(
        is.finite(imputation_median),
        imputation_median,
        0
      )
    )

  values <- values %>%
    left_join(impute_stats, by = "feature") %>%
    mutate(value = if_else(is.finite(value), value, imputation_median))

  scale_stats <- values %>%
    group_by(feature) %>%
    summarise(
      standardization_mean = mean(value),
      standardization_sd = sd(value),
      .groups = "drop"
    ) %>%
    mutate(
      standardization_sd = if_else(
        is.finite(standardization_sd) & standardization_sd > 0,
        standardization_sd,
        1
      )
    )

  feature_stats <- feature_map %>%
    left_join(impute_stats, by = "feature") %>%
    left_join(scale_stats, by = "feature")

  imputed_features <- values %>%
    select(row_id, feature, value) %>%
    pivot_wider(names_from = feature, values_from = value) %>%
    arrange(row_id) %>%
    select(all_of(feature_cols))

  z_feature_cols <- paste0("z_", feature_cols)
  z_features <- values %>%
    left_join(feature_stats, by = "feature") %>%
    transmute(
      row_id,
      feature = paste0("z_", feature),
      value = ((value - standardization_mean) / standardization_sd) *
        feature_group_weight
    ) %>%
    pivot_wider(names_from = feature, values_from = value) %>%
    arrange(row_id) %>%
    select(all_of(z_feature_cols))

  list(
    data = df %>%
      select(-all_of(feature_cols)) %>%
      bind_cols(imputed_features, z_features),
    stats = feature_stats,
    z_cols = z_feature_cols
  )
}

pairwise_region_distances <- function(region_df, z_cols) {
  if (nrow(region_df) < 2) {
    return(tibble(
      target_cz_out10 = character(),
      donor_cz_out10 = character(),
      composite_distance = numeric()
    ))
  }

  z_matrix <- region_df %>%
    select(all_of(z_cols)) %>%
    as.matrix()
  rownames(z_matrix) <- region_df$cz_out10

  z_matrix %>%
    dist(method = "euclidean") %>%
    as.matrix() %>%
    as_tibble(rownames = "target_cz_out10") %>%
    pivot_longer(
      -target_cz_out10,
      names_to = "donor_cz_out10",
      values_to = "composite_distance"
    ) %>%
    filter(target_cz_out10 != donor_cz_out10)
}

## County and CZ baseline features --------------------------------------------

county_panel <- read_parquet(
  path_processed("county_df_analysis_year.parquet"),
  col_select = c(
    "countyfips",
    "year",
    "cz_out10",
    "aewr_region_num",
    "any_cropland_2007",
    "emp_farm",
    "cropland_acr_2007",
    all_of(wage_cols),
    "ppi_2012"
  )
) %>%
  distinct(countyfips, year, cz_out10, aewr_region_num, .keep_all = TRUE) %>%
  cz_ids() %>%
  mutate(
    countyfips = as.character(countyfips),
    year = as.integer(year)
  ) %>%
  filter(!is.na(aewr_region_num), !is.na(cz_out10))

baseline_counties <- county_panel %>%
  filter(
    year %in% baseline_years,
    any_cropland_2007 == 1
  )

scale_counties <- baseline_counties %>%
  filter(year == scale_year)

cz_scale_features <- scale_counties %>%
  group_by(across(all_of(id_cols))) %>%
  summarise(
    emp_farm_2011 = sum(emp_farm, na.rm = TRUE),
    cropland_acr_2007 = sum(cropland_acr_2007, na.rm = TRUE),
    n_counties_2011 = n(),
    .groups = "drop"
  )

cz_wage_features <- baseline_counties %>%
  group_by(across(all_of(id_cols))) %>%
  summarise(
    across(all_of(wage_cols), ~ mean(.x, na.rm = TRUE), .names = "{.col}_pre"),
    .groups = "drop"
  ) %>%
  mutate(
    wage_spread_p90_p10_pre = wage_p90_pre - wage_p10_pre,
    wage_spread_p75_p25_pre = wage_p75_pre - wage_p25_pre
  )

labor_feature_cols <- cz_wage_features %>%
  select(-all_of(id_cols)) %>%
  names()

cz_county_weights <- scale_counties %>%
  distinct(countyfips, aewr_region_num, cz_out10, cropland_acr_2007) %>%
  mutate(county_weight = positive_weight(cropland_acr_2007))

## Climate, soil, and crop features -------------------------------------------

climate_county <- read_parquet(path_int(
  "county_h2a_prediction_climate_basis_annual.parquet"
))

climate_cols <- names(climate_county) %>%
  keep(~ str_starts(.x, "normal_cb_"))
climate_feature_cols <- paste0("climate_", climate_cols)

cz_climate_features <- climate_county %>%
  select(countyfips = fips, all_of(climate_cols)) %>%
  distinct(countyfips, .keep_all = TRUE) %>%
  rename_with(~ paste0("climate_", .x), all_of(climate_cols)) %>%
  inner_join(cz_county_weights, by = "countyfips") %>%
  group_by(across(all_of(id_cols))) %>%
  summarise(
    across(
      all_of(climate_feature_cols),
      ~ wtd_mean(.x, county_weight)
    ),
    .groups = "drop"
  )

soil_cells <- read_parquet(path_int(
  "county_h2a_prediction_gnatsgo_soil_cells.parquet"
)) %>%
  select(
    county_ansi,
    total_acres,
    all_of(soil_cont_cols),
    all_of(soil_cat_cols)
  ) %>%
  inner_join(
    cz_county_weights %>% select(countyfips, all_of(id_cols)),
    by = c("county_ansi" = "countyfips")
  ) %>%
  mutate(total_acres = as.numeric(total_acres))

soil_cont_feature_cols <- paste0("soil_", soil_cont_cols)

cz_soil_cont_features <- soil_cells %>%
  group_by(across(all_of(id_cols))) %>%
  summarise(
    across(
      all_of(soil_cont_cols),
      ~ wtd_mean(.x, total_acres),
      .names = "soil_{.col}"
    ),
    .groups = "drop"
  )

cz_soil_cat_features <- soil_cells %>%
  select(all_of(id_cols), total_acres, all_of(soil_cat_cols)) %>%
  pivot_longer(
    all_of(soil_cat_cols),
    names_to = "soil_variable",
    values_to = "soil_value"
  ) %>%
  mutate(
    soil_value = replace_na(as.character(soil_value), "missing"),
    feature = str_c("soil_share_", soil_variable, "_", soil_value)
  ) %>%
  group_by(across(all_of(c(id_cols, "soil_variable", "feature")))) %>%
  summarise(acres = sum(total_acres, na.rm = TRUE), .groups = "drop_last") %>%
  mutate(share = acres / sum(acres, na.rm = TRUE)) %>%
  ungroup() %>%
  select(all_of(id_cols), feature, share) %>%
  pivot_wider(
    names_from = feature,
    values_from = share,
    values_fill = 0,
    values_fn = sum
  ) %>%
  rename_with(safe_feature_names, -all_of(id_cols))

soil_cat_feature_cols <- setdiff(names(cz_soil_cat_features), id_cols)

cz_soil_features <- cz_soil_cont_features %>%
  left_join(cz_soil_cat_features, by = id_cols) %>%
  mutate(across(all_of(soil_cat_feature_cols), ~ replace_na(.x, 0)))

cdl_county <- read_parquet(path_processed("cdl_cropshares.parquet"))
cdl_acre_cols <- names(cdl_county) %>%
  keep(~ str_starts(.x, "acres_") && .x != "acres_NA")
cdl_feature_cols <- safe_feature_names(
  paste0("cdl_share_", str_remove(cdl_acre_cols, "^acres_"))
)

cz_cdl_features <- cdl_county %>%
  filter(year %in% baseline_years) %>%
  select(countyfips, year, all_of(cdl_acre_cols)) %>%
  inner_join(
    baseline_counties %>%
      distinct(countyfips, year, aewr_region_num, cz_out10),
    by = c("countyfips", "year")
  ) %>%
  group_by(across(all_of(id_cols))) %>%
  summarise(
    across(all_of(cdl_acre_cols), ~ sum(.x, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(
    cdl_total_acres = rowSums(across(all_of(cdl_acre_cols)), na.rm = TRUE)
  ) %>%
  mutate(
    across(
      all_of(cdl_acre_cols),
      ~ if_else(cdl_total_acres > 0, .x / cdl_total_acres, 0)
    )
  ) %>%
  rename(!!!set_names(cdl_acre_cols, cdl_feature_cols)) %>%
  select(all_of(id_cols), cdl_total_acres, all_of(cdl_feature_cols))

## Standardized feature space --------------------------------------------------

feature_map <- list(
  climate_weather = climate_feature_cols,
  soil_continuous = soil_cont_feature_cols,
  soil_categorical = soil_cat_feature_cols,
  crop_types = cdl_feature_cols,
  labor_markets = labor_feature_cols
) %>%
  enframe(name = "feature_group", value = "feature") %>%
  unnest_longer(feature) %>%
  group_by(feature_group) %>%
  mutate(feature_group_weight = 1 / sqrt(n())) %>%
  ungroup()

cz_features <- cz_wage_features %>%
  left_join(cz_scale_features, by = id_cols) %>%
  left_join(cz_climate_features, by = id_cols) %>%
  left_join(cz_soil_features, by = id_cols) %>%
  left_join(cz_cdl_features, by = id_cols)

standardized <- standardize_features(cz_features, feature_map)
standardized_cz_features <- standardized$data
feature_stats <- standardized$stats
z_feature_cols <- standardized$z_cols

write_parquet(
  standardized_cz_features,
  path_processed("cz_dissimilar_features.parquet")
)
write_parquet(
  feature_stats,
  path_processed("cz_dissimilar_feature_stats.parquet")
)

## Dissimilar donor links ------------------------------------------------------

donor_attributes <- standardized_cz_features %>%
  select(
    aewr_region_num,
    donor_cz_out10 = cz_out10,
    donor_emp_farm_2011 = emp_farm_2011,
    donor_cropland_acr_2007 = cropland_acr_2007
  )

pairwise_distances <- standardized_cz_features %>%
  group_by(aewr_region_num) %>%
  group_modify(~ pairwise_region_distances(.x, z_feature_cols)) %>%
  ungroup() %>%
  left_join(donor_attributes, by = c("aewr_region_num", "donor_cz_out10")) %>%
  group_by(aewr_region_num, target_cz_out10) %>%
  arrange(desc(composite_distance), donor_cz_out10, .by_group = TRUE) %>%
  mutate(distance_rank_desc = row_number()) %>%
  ungroup()

donor_links <- pairwise_distances %>%
  filter(distance_rank_desc <= donor_top_n) %>%
  mutate(raw_donor_weight = positive_weight(donor_emp_farm_2011)) %>%
  group_by(aewr_region_num, target_cz_out10) %>%
  mutate(
    donor_weight = raw_donor_weight / sum(raw_donor_weight, na.rm = TRUE)
  ) %>%
  ungroup()

write_parquet(
  pairwise_distances,
  path_processed("cz_dissimilar_pairwise_distances.parquet")
)
write_parquet(
  donor_links,
  path_processed("cz_dissimilar_donor_links.parquet")
)

## Donor wage instrument -------------------------------------------------------

oews_ag_wages <- read_parquet(path_int("oews_county_aggregated.parquet")) %>%
  filter(occ_code == "AEWR") %>%
  mutate(
    countyfips = str_c(
      str_pad(as.character(state_fips_code), 2, side = "left", pad = "0"),
      str_pad(as.character(county_fips_code), 3, side = "left", pad = "0")
    ),
    year = as.integer(year),
    oews_tot_emp = as.numeric(oews_tot_emp),
    oews_mean_hourly_wage = as.numeric(oews_mean_hourly_wage)
  ) %>%
  select(countyfips, year, oews_tot_emp, oews_mean_hourly_wage)

oews_cz_year <- county_panel %>%
  distinct(countyfips, year, aewr_region_num, cz_out10, ppi_2012) %>%
  left_join(oews_ag_wages, by = c("countyfips", "year")) %>%
  mutate(oews_mean_hourly_wage_ppi = oews_mean_hourly_wage / ppi_2012) %>%
  group_by(aewr_region_num, cz_out10, year) %>%
  summarise(
    oews_ag_wage = wtd_mean(oews_mean_hourly_wage, oews_tot_emp),
    oews_ag_wage_ppi = wtd_mean(oews_mean_hourly_wage_ppi, oews_tot_emp),
    oews_ag_emp = sum(oews_tot_emp, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    oews_ag_wage = if_else(oews_ag_emp > 0, oews_ag_wage, NA_real_),
    oews_ag_wage_ppi = if_else(oews_ag_emp > 0, oews_ag_wage_ppi, NA_real_)
  )

donor_oews_links <- donor_links %>%
  left_join(
    oews_cz_year %>%
      rename(
        donor_cz_out10 = cz_out10,
        donor_oews_ag_wage = oews_ag_wage,
        donor_oews_ag_wage_ppi = oews_ag_wage_ppi,
        donor_oews_ag_emp = oews_ag_emp
      ),
    by = c("aewr_region_num", "donor_cz_out10"),
    relationship = "many-to-many"
  ) %>%
  mutate(
    valid_nominal_weight = if_else(!is.na(donor_oews_ag_wage), donor_weight, 0),
    valid_real_weight = if_else(!is.na(donor_oews_ag_wage_ppi), donor_weight, 0)
  )

cz_dissimilar_oews_instruments <- donor_oews_links %>%
  group_by(aewr_region_num, target_cz_out10, year) %>%
  summarise(
    dissimilar_donor_oews_ag_wage = wtd_mean(
      donor_oews_ag_wage,
      donor_weight
    ),
    dissimilar_donor_oews_ag_wage_ppi = wtd_mean(
      donor_oews_ag_wage_ppi,
      donor_weight
    ),
    valid_nominal_donor_weight = sum(valid_nominal_weight, na.rm = TRUE),
    valid_real_donor_weight = sum(valid_real_weight, na.rm = TRUE),
    n_selected_donors = n(),
    selected_donor_oews_ag_emp = sum(donor_oews_ag_emp, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  rename(cz_out10 = target_cz_out10) %>%
  arrange(aewr_region_num, cz_out10, year) %>%
  group_by(aewr_region_num, cz_out10) %>%
  mutate(
    across(
      c(dissimilar_donor_oews_ag_wage, dissimilar_donor_oews_ag_wage_ppi),
      list(l1 = ~ lag(.x), d1 = ~ .x - lag(.x)),
      .names = "{.col}_{.fn}"
    )
  ) %>%
  ungroup()

write_parquet(
  donor_oews_links,
  path_processed("cz_dissimilar_donor_oews_links.parquet")
)
write_parquet(
  cz_dissimilar_oews_instruments,
  path_processed("cz_dissimilar_oews_instruments.parquet")
)

output_summary <- tibble(
  object = c(
    "cz_region_units",
    "primitive_features",
    "pairwise_distances",
    "selected_donor_links",
    "instrument_rows"
  ),
  n = c(
    nrow(standardized_cz_features),
    nrow(feature_map),
    nrow(pairwise_distances),
    nrow(donor_links),
    nrow(cz_dissimilar_oews_instruments)
  )
)

print(output_summary)
