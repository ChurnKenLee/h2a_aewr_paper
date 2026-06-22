rm(list = ls())

if (file.exists("paths.R")) {
  source("paths.R")
} else {
  source(file.path("code", "paths.R"))
}

library(arrow)
library(tidyverse)

## CZ-level leave-agro-cluster-out AEWR instrument -----------------------------

baseline_years <- 2008:2011
crop_baseline_year <- 2008
scale_year <- 2011
climate_normal_years <- 2000:2007

min_cluster_size <- 4
cluster_target_size <- 12
max_clusters <- 6

write_outputs <- !tolower(Sys.getenv("CZD_WRITE_OUTPUTS", "true")) %in%
  c("0", "false", "no")

id_cols <- c("aewr_region_num", "cz_out10")
wage_cols <- paste0("wage_p", c(10, 25, 50, 75, 90))

soil_cont_cols <- c(
  "slope_r",
  "slopegradwta",
  "resdept_r",
  "aws050wta",
  "aws0100wta",
  "aws0150wta",
  "wtdepannmin",
  "wtdepaprjunmin",
  "brockdepmin",
  "cropprodindex"
)

soil_cat_cols <- c(
  "drainagecl",
  "hydgrp",
  "nirrcapcl"
)

climate_primitive_cols <- c(
  "climate_tavg_mean",
  "climate_tavg_growing_mean",
  "climate_tavg_sd",
  "climate_gdd10_annual",
  "climate_growing_gdd10_annual",
  "climate_heat_dd29_annual",
  "climate_heat_days32_annual",
  "climate_frost_days_annual",
  "climate_prcp_annual",
  "climate_growing_prcp_annual",
  "climate_wet_days_annual",
  "climate_growing_wet_days_annual",
  "climate_wet_day_prcp_intensity"
)

cz_ids <- function(.data) {
  .data %>%
    mutate(
      aewr_region_num = as.integer(aewr_region_num),
      cz_out10 = as.character(cz_out10)
    )
}

positive_weight <- function(x) {
  if_else(!is.na(x) & is.finite(x) & x > 0, x, 1)
}

wtd_mean <- function(x, w) {
  keep <- !is.na(x) & is.finite(x) & !is.na(w) & is.finite(w) & w > 0

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

write_output_parquet <- function(x, path) {
  if (write_outputs) {
    write_parquet(x, path)
  }

  invisible(x)
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

climate_file_index <- function(raw_dir, years) {
  files <- list.files(raw_dir, pattern = "^[0-9]{6}[.]parquet$", full.names = TRUE)

  tibble(path = files) %>%
    mutate(
      ym = str_remove(basename(path), "[.]parquet$"),
      year = as.integer(str_sub(ym, 1, 4)),
      month = as.integer(str_sub(ym, 5, 6))
    ) %>%
    filter(year %in% years) %>%
    arrange(year, month)
}

read_climate_month_summary <- function(path, year, month) {
  growing_month <- month %in% 4:9

  read_parquet(
    path,
    col_select = c("fips", "tmax", "tmin", "tavg", "prcp")
  ) %>%
    transmute(
      countyfips = as.character(fips),
      year = year,
      tmax = suppressWarnings(as.numeric(tmax)),
      tmin = suppressWarnings(as.numeric(tmin)),
      tavg = suppressWarnings(as.numeric(tavg)),
      prcp = suppressWarnings(as.numeric(prcp)),
      growing = growing_month
    ) %>%
    group_by(countyfips, year) %>%
    summarise(
      n_days = sum(!is.na(tavg)),
      tavg_sum = sum(tavg, na.rm = TRUE),
      tavg_sq_sum = sum(tavg^2, na.rm = TRUE),
      growing_n_days = sum(growing & !is.na(tavg)),
      growing_tavg_sum = sum(if_else(growing, tavg, NA_real_), na.rm = TRUE),
      gdd10 = sum(pmax(tavg - 10, 0), na.rm = TRUE),
      growing_gdd10 = sum(if_else(growing, pmax(tavg - 10, 0), NA_real_), na.rm = TRUE),
      heat_dd29 = sum(pmax(tavg - 29, 0), na.rm = TRUE),
      heat_days32 = sum(tmax >= 32.2, na.rm = TRUE),
      frost_days = sum(tmin < 0, na.rm = TRUE),
      prcp_sum = sum(pmax(prcp, 0), na.rm = TRUE),
      growing_prcp_sum = sum(if_else(growing, pmax(prcp, 0), NA_real_), na.rm = TRUE),
      wet_days = sum(prcp >= 1, na.rm = TRUE),
      growing_wet_days = sum(growing & prcp >= 1, na.rm = TRUE),
      .groups = "drop"
    )
}

build_county_climate_primitives <- function(raw_dir, years) {
  file_index <- climate_file_index(raw_dir, years)

  if (nrow(file_index) == 0) {
    stop(
      "No raw EpiNOAA monthly parquets found for years ",
      str_c(range(years), collapse = "-"),
      " in ",
      raw_dir,
      ". Run code/a09_01_h2a_prediction_pull_noaa.py first.",
      call. = FALSE
    )
  }

  month_summaries <- pmap_dfr(
    file_index,
    function(path, ym, year, month) {
      read_climate_month_summary(path, year, month)
    }
  )

  month_summaries %>%
    group_by(countyfips) %>%
    summarise(
      climate_normal_years = n_distinct(year),
      total_days = sum(n_days, na.rm = TRUE),
      growing_days = sum(growing_n_days, na.rm = TRUE),
      climate_tavg_mean = sum(tavg_sum, na.rm = TRUE) / total_days,
      climate_tavg_growing_mean =
        sum(growing_tavg_sum, na.rm = TRUE) / growing_days,
      climate_tavg_sd = sqrt(pmax(
        sum(tavg_sq_sum, na.rm = TRUE) / total_days - climate_tavg_mean^2,
        0
      )),
      climate_gdd10_annual = sum(gdd10, na.rm = TRUE) / climate_normal_years,
      climate_growing_gdd10_annual =
        sum(growing_gdd10, na.rm = TRUE) / climate_normal_years,
      climate_heat_dd29_annual =
        sum(heat_dd29, na.rm = TRUE) / climate_normal_years,
      climate_heat_days32_annual =
        sum(heat_days32, na.rm = TRUE) / climate_normal_years,
      climate_frost_days_annual =
        sum(frost_days, na.rm = TRUE) / climate_normal_years,
      climate_prcp_annual = sum(prcp_sum, na.rm = TRUE) / climate_normal_years,
      climate_growing_prcp_annual =
        sum(growing_prcp_sum, na.rm = TRUE) / climate_normal_years,
      climate_wet_days_annual = sum(wet_days, na.rm = TRUE) / climate_normal_years,
      climate_growing_wet_days_annual =
        sum(growing_wet_days, na.rm = TRUE) / climate_normal_years,
      climate_wet_day_prcp_intensity =
        sum(prcp_sum, na.rm = TRUE) / sum(wet_days, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      climate_wet_day_prcp_intensity = if_else(
        is.finite(climate_wet_day_prcp_intensity),
        climate_wet_day_prcp_intensity,
        NA_real_
      )
    )
}

choose_cluster_count <- function(n) {
  if (n < 2) {
    return(1L)
  }

  max_by_min_size <- max(2L, floor(n / min_cluster_size))
  desired <- max(2L, round(n / cluster_target_size))

  as.integer(min(max_clusters, max_by_min_size, desired))
}

balanced_order_clusters <- function(hc, n, k) {
  ordered_group <- ceiling(seq_len(n) * k / n)
  clusters <- integer(n)
  clusters[hc$order] <- ordered_group
  clusters
}

assign_region_clusters <- function(region_df, z_cols) {
  n <- nrow(region_df)

  if (n == 1) {
    return(region_df %>%
      mutate(
        agro_cluster_num = 1L,
        agro_cluster_id = str_c(aewr_region_num, "_", agro_cluster_num),
        agro_cluster_size = 1L,
        region_cluster_count = 1L
      ))
  }

  z_matrix <- region_df %>%
    select(all_of(z_cols)) %>%
    as.matrix()

  dist_obj <- dist(z_matrix, method = "euclidean")
  hc <- hclust(dist_obj, method = "ward.D2")
  k_start <- choose_cluster_count(n)
  clusters <- cutree(hc, k = k_start)
  chosen_k <- k_start
  cluster_method <- "ward_cut"
  found_min_size_cut <- min(tabulate(clusters)) >= min_cluster_size

  if (k_start > 2) {
    for (k in seq(k_start, 2)) {
      candidate <- cutree(hc, k = k)
      if (min(tabulate(candidate)) >= min_cluster_size || k == 2) {
        clusters <- candidate
        chosen_k <- k
        found_min_size_cut <- min(tabulate(candidate)) >= min_cluster_size
        break
      }
    }
  }

  if (!found_min_size_cut && n >= 2 * min_cluster_size) {
    chosen_k <- 2L
    clusters <- balanced_order_clusters(hc, n, chosen_k)
    cluster_method <- "balanced_dendrogram_order"
  }

  cluster_sizes <- tabulate(clusters)

  region_df %>%
    mutate(
      agro_cluster_num = as.integer(clusters),
      agro_cluster_id = str_c(aewr_region_num, "_", agro_cluster_num),
      agro_cluster_size = as.integer(cluster_sizes[agro_cluster_num]),
      region_cluster_count = chosen_k,
      agro_cluster_method = cluster_method
    )
}

pairwise_region_distances <- function(region_df, z_cols) {
  if (nrow(region_df) < 2) {
    return(tibble(
      aewr_region_num = integer(),
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
    filter(target_cz_out10 != donor_cz_out10) %>%
    mutate(aewr_region_num = first(region_df$aewr_region_num), .before = 1)
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

cz_labor_balance_features <- baseline_counties %>%
  group_by(across(all_of(id_cols))) %>%
  summarise(
    across(all_of(wage_cols), ~ mean(.x, na.rm = TRUE), .names = "{.col}_pre"),
    .groups = "drop"
  ) %>%
  mutate(
    wage_spread_p90_p10_pre = wage_p90_pre - wage_p10_pre,
    wage_spread_p75_p25_pre = wage_p75_pre - wage_p25_pre
  )

cz_county_weights <- scale_counties %>%
  distinct(countyfips, aewr_region_num, cz_out10, cropland_acr_2007) %>%
  mutate(county_weight = positive_weight(cropland_acr_2007))

## Climate, soil, and crop primitives -----------------------------------------

county_climate_features <- build_county_climate_primitives(
  path_raw("epinoaa_nclimgrid"),
  climate_normal_years
)

cz_climate_features <- county_climate_features %>%
  inner_join(cz_county_weights, by = "countyfips") %>%
  group_by(across(all_of(id_cols))) %>%
  summarise(
    across(
      all_of(climate_primitive_cols),
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
    soil_value = str_to_lower(soil_value),
    soil_value = str_replace_all(soil_value, "[^0-9a-z]+", "_"),
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

cz_cdl_features <- cdl_county %>%
  filter(year == crop_baseline_year) %>%
  select(countyfips, all_of(cdl_acre_cols)) %>%
  pivot_longer(
    all_of(cdl_acre_cols),
    names_to = "raw_crop_group",
    values_to = "acres"
  ) %>%
  mutate(
    crop_group = raw_crop_group %>%
      str_remove("^acres_") %>%
      str_replace_all("[^0-9A-Za-z]+", "_") %>%
      str_replace_all("^_+|_+$", "") %>%
      str_to_lower(),
    crop_group = if_else(is.na(crop_group) | crop_group == "", "unknown", crop_group),
    acres = as.numeric(acres)
  ) %>%
  group_by(countyfips, crop_group) %>%
  summarise(acres = sum(acres, na.rm = TRUE), .groups = "drop") %>%
  inner_join(
    baseline_counties %>%
      filter(year == crop_baseline_year) %>%
      distinct(countyfips, aewr_region_num, cz_out10),
    by = "countyfips"
  ) %>%
  group_by(across(all_of(c(id_cols, "crop_group")))) %>%
  summarise(acres = sum(acres, na.rm = TRUE), .groups = "drop") %>%
  group_by(across(all_of(id_cols))) %>%
  mutate(
    cdl_total_acres = sum(acres, na.rm = TRUE),
    share = if_else(cdl_total_acres > 0, acres / cdl_total_acres, 0),
    feature = paste0("cdl_share_", crop_group)
  ) %>%
  ungroup() %>%
  select(all_of(id_cols), cdl_total_acres, feature, share) %>%
  pivot_wider(
    names_from = feature,
    values_from = share,
    values_fill = 0,
    values_fn = sum
  )

cdl_feature_cols <- setdiff(names(cz_cdl_features), c(id_cols, "cdl_total_acres"))

## Standardized agro-ecological feature spaces --------------------------------

feature_groups <- list(
  crop_types = cdl_feature_cols,
  climate_weather = climate_primitive_cols,
  soil_continuous = soil_cont_feature_cols,
  soil_categorical = soil_cat_feature_cols
)

variant_specs <- tibble::tribble(
  ~donor_variant, ~selection_rule, ~crop_types, ~climate_weather, ~soil_continuous, ~soil_categorical,
  "cluster_baseline", "leave_cluster_out_ward", 1.35, 1.00, 1.00, 0.75,
  "cluster_crop_heavy", "leave_cluster_out_ward", 2.25, 0.75, 0.75, 0.25,
  "cluster_climate_crop", "leave_cluster_out_ward", 1.50, 1.50, 0.50, 0.25,
  "distance_tail_top25", "distance_top_tail", 1.35, 1.00, 1.00, 0.75,
  "distance_ring_p50_p90", "distance_middle_high_ring", 1.35, 1.00, 1.00, 0.75
)

make_feature_map <- function(feature_groups, group_importance) {
  feature_groups %>%
    enframe(name = "feature_group", value = "feature") %>%
    unnest_longer(feature) %>%
    group_by(feature_group) %>%
    mutate(
      feature_group_importance = group_importance[feature_group],
      feature_group_weight = feature_group_importance / sqrt(n())
    ) %>%
    ungroup()
}

prepare_variant_features <- function(
    donor_variant,
    selection_rule,
    crop_types,
    climate_weather,
    soil_continuous,
    soil_categorical) {
  feature_map <- make_feature_map(
    feature_groups,
    c(
      crop_types = crop_types,
      climate_weather = climate_weather,
      soil_continuous = soil_continuous,
      soil_categorical = soil_categorical
    )
  )

  standardized <- standardize_features(cz_features, feature_map)

  variant_features <- standardized$data
  if (selection_rule == "leave_cluster_out_ward") {
    variant_features <- variant_features %>%
      group_split(aewr_region_num) %>%
      map_dfr(assign_region_clusters, z_cols = standardized$z_cols)
  } else {
    variant_features <- variant_features %>%
      mutate(
        agro_cluster_num = NA_integer_,
        agro_cluster_id = NA_character_,
        agro_cluster_size = NA_integer_,
        region_cluster_count = NA_integer_,
        agro_cluster_method = NA_character_
      )
  }

  list(
    donor_variant = donor_variant,
    selection_rule = selection_rule,
    data = variant_features %>%
      mutate(
        donor_variant = donor_variant,
        selection_rule = selection_rule,
        .before = 1
      ),
    stats = standardized$stats %>%
      mutate(
        donor_variant = donor_variant,
        selection_rule = selection_rule,
        .before = 1
      ),
    feature_map = feature_map %>%
      mutate(
        donor_variant = donor_variant,
        selection_rule = selection_rule,
        .before = 1
      ),
    z_cols = standardized$z_cols
  )
}

cz_features <- cz_scale_features %>%
  left_join(cz_labor_balance_features, by = id_cols) %>%
  left_join(cz_climate_features, by = id_cols) %>%
  left_join(cz_soil_features, by = id_cols) %>%
  left_join(cz_cdl_features, by = id_cols)

variant_objects <- pmap(
  variant_specs,
  prepare_variant_features
)
names(variant_objects) <- variant_specs$donor_variant

variant_cz_features <- map_dfr(variant_objects, "data") %>%
  arrange(donor_variant, aewr_region_num, cz_out10)
feature_stats <- map_dfr(variant_objects, "stats")
feature_map <- map_dfr(variant_objects, "feature_map")
z_feature_cols <- unique(unlist(map(variant_objects, "z_cols")))
primary_variant <- "cluster_baseline"

standardized_cz_features <- variant_cz_features %>%
  filter(donor_variant == primary_variant) %>%
  select(-donor_variant, -selection_rule)

cluster_summary <- variant_cz_features %>%
  filter(!is.na(agro_cluster_id)) %>%
  group_by(
    donor_variant,
    selection_rule,
    aewr_region_num,
    agro_cluster_id,
    agro_cluster_num,
    agro_cluster_method
  ) %>%
  summarise(
    n_cz = n(),
    emp_farm_2011 = sum(emp_farm_2011, na.rm = TRUE),
    cropland_acr_2007 = sum(cropland_acr_2007, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(donor_variant, aewr_region_num, agro_cluster_num)

write_output_parquet(
  standardized_cz_features,
  path_processed("cz_dissimilar_features.parquet")
)
write_output_parquet(
  variant_cz_features,
  path_processed("cz_dissimilar_variant_features.parquet")
)
write_output_parquet(
  feature_stats,
  path_processed("cz_dissimilar_feature_stats.parquet")
)
write_output_parquet(
  cluster_summary,
  path_processed("cz_dissimilar_cluster_summary.parquet")
)

## Donor links under alternative dissimilarity rules ---------------------------

target_attributes <- variant_cz_features %>%
  select(
    donor_variant,
    selection_rule,
    aewr_region_num,
    target_cz_out10 = cz_out10,
    target_agro_cluster_id = agro_cluster_id,
    target_agro_cluster_num = agro_cluster_num,
    target_agro_cluster_size = agro_cluster_size
  )

donor_attributes <- variant_cz_features %>%
  select(
    donor_variant,
    selection_rule,
    aewr_region_num,
    donor_cz_out10 = cz_out10,
    donor_agro_cluster_id = agro_cluster_id,
    donor_agro_cluster_num = agro_cluster_num,
    donor_emp_farm_2011 = emp_farm_2011,
    donor_cropland_acr_2007 = cropland_acr_2007
  )

pairwise_distances <- imap_dfr(variant_objects, function(variant_object, variant_name) {
  variant_object$data %>%
    split(.$aewr_region_num) %>%
    map_dfr(pairwise_region_distances, z_cols = variant_object$z_cols) %>%
    mutate(
      donor_variant = variant_object$donor_variant,
      selection_rule = variant_object$selection_rule,
      .before = 1
    )
}) %>%
  left_join(
    target_attributes,
    by = c("donor_variant", "selection_rule", "aewr_region_num", "target_cz_out10")
  ) %>%
  left_join(
    donor_attributes,
    by = c("donor_variant", "selection_rule", "aewr_region_num", "donor_cz_out10")
  ) %>%
  group_by(donor_variant, aewr_region_num, target_cz_out10) %>%
  arrange(desc(composite_distance), donor_cz_out10, .by_group = TRUE) %>%
  mutate(
    distance_rank_desc = row_number(),
    n_possible_donors = n(),
    distance_percentile = percent_rank(composite_distance)
  ) %>%
  ungroup()

donor_links <- pairwise_distances %>%
  mutate(
    include_donor = case_when(
      selection_rule == "leave_cluster_out_ward" ~
        target_agro_cluster_id != donor_agro_cluster_id,
      selection_rule == "distance_top_tail" ~
        distance_rank_desc <= pmax(5L, ceiling(0.25 * n_possible_donors)),
      selection_rule == "distance_middle_high_ring" ~
        distance_percentile >= 0.50 & distance_percentile <= 0.90,
      TRUE ~ FALSE
    )
  ) %>%
  filter(include_donor) %>%
  mutate(raw_donor_weight = positive_weight(donor_emp_farm_2011)) %>%
  group_by(donor_variant, aewr_region_num, target_cz_out10) %>%
  mutate(
    donor_weight = raw_donor_weight / sum(raw_donor_weight, na.rm = TRUE),
    n_selected_donors = n(),
    donor_cluster_count = if_else(
      all(is.na(donor_agro_cluster_id)),
      NA_integer_,
      n_distinct(donor_agro_cluster_id)
    ),
    effective_n_donors = 1 / sum(donor_weight^2, na.rm = TRUE),
    mean_selected_donor_distance =
      weighted.mean(composite_distance, donor_weight, na.rm = TRUE)
  ) %>%
  ungroup()

donor_feature_balance <- donor_links %>%
  select(donor_variant, aewr_region_num, target_cz_out10, donor_cz_out10, donor_weight) %>%
  left_join(
    variant_cz_features %>%
      select(
        donor_variant,
        aewr_region_num,
        target_cz_out10 = cz_out10,
        all_of(z_feature_cols)
      ) %>%
      pivot_longer(
        all_of(z_feature_cols),
        names_to = "z_feature",
        values_to = "target_z"
      ),
    by = c("donor_variant", "aewr_region_num", "target_cz_out10"),
    relationship = "many-to-many"
  ) %>%
  left_join(
    variant_cz_features %>%
      select(
        donor_variant,
        aewr_region_num,
        donor_cz_out10 = cz_out10,
        all_of(z_feature_cols)
      ) %>%
      pivot_longer(
        all_of(z_feature_cols),
        names_to = "z_feature",
        values_to = "donor_z"
      ),
    by = c("donor_variant", "aewr_region_num", "donor_cz_out10", "z_feature"),
    relationship = "many-to-many"
  ) %>%
  mutate(feature = str_remove(z_feature, "^z_")) %>%
  left_join(
    feature_map %>% select(donor_variant, feature, feature_group),
    by = c("donor_variant", "feature")
  ) %>%
  group_by(donor_variant, aewr_region_num, target_cz_out10, feature_group, feature) %>%
  summarise(
    target_z = first(target_z),
    donor_z = weighted.mean(donor_z, donor_weight, na.rm = TRUE),
    donor_minus_target_z = donor_z - target_z,
    .groups = "drop"
  )

write_output_parquet(
  pairwise_distances,
  path_processed("cz_dissimilar_pairwise_distances.parquet")
)
write_output_parquet(
  donor_links,
  path_processed("cz_dissimilar_donor_links.parquet")
)
write_output_parquet(
  donor_feature_balance,
  path_processed("cz_dissimilar_donor_feature_balance.parquet")
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
  group_by(donor_variant, selection_rule, aewr_region_num, target_cz_out10, year) %>%
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
    n_selected_donors = first(n_selected_donors),
    n_valid_nominal_donors = sum(!is.na(donor_oews_ag_wage)),
    n_valid_real_donors = sum(!is.na(donor_oews_ag_wage_ppi)),
    donor_cluster_count = first(donor_cluster_count),
    effective_n_donors = first(effective_n_donors),
    mean_selected_donor_distance = first(mean_selected_donor_distance),
    selected_donor_oews_ag_emp = sum(donor_oews_ag_emp, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  rename(cz_out10 = target_cz_out10) %>%
  arrange(donor_variant, aewr_region_num, cz_out10, year) %>%
  group_by(donor_variant, aewr_region_num, cz_out10) %>%
  mutate(
    across(
      c(dissimilar_donor_oews_ag_wage, dissimilar_donor_oews_ag_wage_ppi),
      list(l1 = ~ lag(.x), d1 = ~ .x - lag(.x)),
      .names = "{.col}_{.fn}"
    )
  ) %>%
  ungroup()

write_output_parquet(
  donor_oews_links,
  path_processed("cz_dissimilar_donor_oews_links.parquet")
)
write_output_parquet(
  cz_dissimilar_oews_instruments,
  path_processed("cz_dissimilar_oews_instruments.parquet")
)

output_summary <- tibble(
  object = c(
    "donor_variants",
    "cz_region_units",
    "donor_assignment_features",
    "agro_clusters",
    "pairwise_distances",
    "selected_donor_links",
    "instrument_rows",
    "median_effective_donors"
  ),
  n = c(
    n_distinct(variant_specs$donor_variant),
    nrow(standardized_cz_features),
    nrow(feature_map %>% distinct(feature)),
    nrow(cluster_summary),
    nrow(pairwise_distances),
    nrow(donor_links),
    nrow(cz_dissimilar_oews_instruments),
    median(donor_links$effective_n_donors, na.rm = TRUE)
  )
)

print(output_summary)
