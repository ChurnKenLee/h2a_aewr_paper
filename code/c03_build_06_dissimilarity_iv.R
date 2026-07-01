# Construct CZ-level AEWR IV using same-AEWR-region CZs
rm(list = ls())
if (!exists("path_code", mode = "function")) {
  source(file.path("code", "paths.R"))
}
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(foreign)

cdl <- read_parquet(path_int(
  "croplandcros_county_crop_acres.parquet"
)) %>%
  clean_names()

cdl_crop_names <- read_csv(path_raw(
  "croplandcros_cdl",
  "croplandcros_cdl_crop_category.csv"
)) %>%
  clean_names() %>%
  filter(!is.na(crop_type)) %>%
  distinct(crop_name)

climate <- read_parquet(path_int(
  "county_h2a_prediction_climate_basis_annual.parquet"
)) %>%
  clean_names()

soil <- read_parquet(path_int(
  "county_h2a_prediction_gnatsgo_soil_cells.parquet"
)) %>%
  clean_names()

county_df <- read_parquet(path_processed(
  "county_df_analysis_year.parquet"
)) %>%
  clean_names()

oews <- read_parquet(path_int(
  "oews_county_aggregated.parquet"
)) %>%
  clean_names()

qcew <- read_parquet(path_int(
  "qcew_state_ag_wage.parquet"
)) %>%
  clean_names()

# Fixed 2008-2011 county primitives ------------------------------------------

crop_names <- cdl %>%
  semi_join(cdl_crop_names, by = "crop_name") %>%
  distinct(crop_name) %>%
  mutate(crop_var = paste0("share_cdl_", make_clean_names(crop_name)))

crop_features <- cdl %>%
  semi_join(cdl_crop_names, by = "crop_name") %>%
  filter(year >= 2008, year <= 2011) %>%
  mutate(
    county_ansi = fips
  ) %>%
  left_join(crop_names, by = "crop_name") %>%
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

# Cluster CZ x AEWR-region units within AEWR regions --------------------------

unit_xwalk <- county_df %>%
  mutate(county_ansi = countyfips) %>%
  distinct(county_ansi, cz_out10, aewr_region_num, cz_aewr_region_fe)

county_feature_weights <- county_df %>%
  mutate(county_ansi = countyfips) %>%
  filter(year >= 2008, year <= 2011) %>%
  group_by(county_ansi, cz_out10, aewr_region_num, cz_aewr_region_fe) %>%
  summarise(feature_weight = mean(emp_farm, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    feature_weight = if_else(
      is.nan(feature_weight) | is.na(feature_weight) | feature_weight <= 0,
      1,
      feature_weight
    )
  )

unit_features <- unit_xwalk %>%
  left_join(
    county_feature_weights,
    by = c("county_ansi", "cz_out10", "aewr_region_num", "cz_aewr_region_fe")
  ) %>%
  mutate(feature_weight = replace_na(feature_weight, 1)) %>%
  left_join(county_features, by = "county_ansi") %>%
  group_by(cz_out10, aewr_region_num, cz_aewr_region_fe) %>%
  summarise(
    unit_feature_weight = sum(feature_weight, na.rm = TRUE),
    across(
      all_of(county_feature_names),
      ~ weighted.mean(.x, w = feature_weight, na.rm = TRUE)
    ),
    .groups = "drop"
  )

feature_names <- setdiff(
  names(unit_features),
  c("cz_out10", "aewr_region_num", "cz_aewr_region_fe", "unit_feature_weight")
)

share_feature_names <- feature_names[
  str_detect(feature_names, "^share_cdl_|^share_soil_")
]

unit_features <- unit_features %>%
  mutate(
    across(
      all_of(share_feature_names),
      # For crop/soil shares, Euclidean distance on sqrt(shares) is the
      # Hellinger distance for compositional variables. It keeps large-acreage
      # crops important while reducing dominance by a few very large shares.
      ~ sqrt(pmax(.x, 0))
    )
  )

# Rescale each feature block to roughly equal weight
feature_blocks <- list(
  crops = feature_names[str_detect(feature_names, "^share_cdl_")],
  climate = feature_names[str_detect(feature_names, "^normal_cb_")],
  soil_continuous = intersect(feature_names, soil_vars),
  soil_categorical = feature_names[str_detect(feature_names, "^share_soil_")]
)

iv_k <- 2

cluster_list <- list()
cluster_diagnostic_list <- list()
donor_cluster_list <- list()
for (r in sort(unique(unit_features$aewr_region_num))) {
  d <- unit_features %>% filter(aewr_region_num == r)

  for (v in feature_names) {
    x <- d[[v]]
    x[is.nan(x)] <- NA_real_
    med <- median(x, na.rm = TRUE)
    if (is.na(med)) {
      med <- 0
    }
    x[is.na(x)] <- med
    sx <- sd(x)
    d[[v]] <- if (!is.na(sx) && sx > 0) {
      (x - mean(x)) / sx
    } else {
      0
    }
  }

  for (block_name in names(feature_blocks)) {
    block_cols <- feature_blocks[[block_name]]
    if (length(block_cols) > 0) {
      d[block_cols] <- d[block_cols] / sqrt(length(block_cols))
    }
  }

  x <- as.matrix(d[, feature_names])
  hclust_fit <- hclust(dist(x), method = "ward.D2")
  selected_cluster <- cutree(hclust_fit, k = iv_k)

  cluster_list[[as.character(r)]] <- d %>%
    select(cz_aewr_region_fe, aewr_region_num) %>%
    mutate(iv_cluster = selected_cluster, iv_k = iv_k)

  cluster_diagnostic_list[[as.character(r)]] <- tibble(
    aewr_region_num = r,
    iv_cluster = selected_cluster,
    unit_feature_weight = d$unit_feature_weight
  ) %>%
    group_by(aewr_region_num, iv_cluster) %>%
    summarise(
      iv_k = iv_k,
      cluster_units = n(),
      cluster_feature_weight = sum(unit_feature_weight, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      cluster_feature_weight_share = cluster_feature_weight /
        sum(cluster_feature_weight, na.rm = TRUE)
    )

  cluster_centroids <- as_tibble(x) %>%
    mutate(iv_cluster = selected_cluster) %>%
    group_by(iv_cluster) %>%
    summarise(across(all_of(feature_names), ~ mean(.x)), .groups = "drop")

  centroid_matrix <- as.matrix(cluster_centroids[, feature_names])
  rownames(centroid_matrix) <- cluster_centroids$iv_cluster
  centroid_distance <- as.matrix(dist(centroid_matrix))
  cluster_ids <- sort(unique(selected_cluster))
  donor_pair_list <- list()

  for (target_cluster_id in cluster_ids) {
    donor_pair_list[[as.character(target_cluster_id)]] <- tibble(
      aewr_region_num = r,
      target_cluster = target_cluster_id,
      donor_cluster = cluster_ids,
      donor_cluster_distance = centroid_distance[
        as.character(target_cluster_id),
        as.character(cluster_ids)
      ]
    ) %>%
      filter(donor_cluster != target_cluster) %>%
      arrange(desc(donor_cluster_distance)) %>%
      slice_head(n = 1)
  }

  donor_cluster_list[[as.character(r)]] <- bind_rows(donor_pair_list)
}

iv_clusters <- bind_rows(cluster_list)
iv_cluster_diagnostics <- bind_rows(cluster_diagnostic_list) %>%
  group_by(aewr_region_num) %>%
  mutate(
    region_min_cluster_units = min(cluster_units),
    region_min_cluster_weight_share = min(cluster_feature_weight_share)
  ) %>%
  ungroup()
iv_donor_clusters <- bind_rows(donor_cluster_list)

# Donor wage instrument --------------------------------------------------------

county_year_units <- county_df %>%
  mutate(county_ansi = countyfips) %>%
  distinct(
    county_ansi,
    state_fips_code = statefips,
    year,
    cz_aewr_region_fe,
    aewr_region_num,
    emp_farm
  )

unit_oews_wages <- oews %>%
  filter(
    occ_code == "AEWR",
    !is.na(oews_mean_hourly_wage),
    oews_tot_emp > 0
  ) %>%
  mutate(
    county_ansi = paste0(
      sprintf("%02d", as.integer(state_fips_code)),
      sprintf("%03d", as.integer(county_fips_code))
    )
  ) %>%
  inner_join(county_year_units, by = c("county_ansi", "year")) %>%
  group_by(cz_aewr_region_fe, aewr_region_num, year) %>%
  summarise(
    ag_wage = weighted.mean(oews_mean_hourly_wage, oews_tot_emp, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  # In outcome year t, use the proxy change from t - 2 to t - 1.
  arrange(cz_aewr_region_fe, year) %>%
  group_by(cz_aewr_region_fe) %>%
  mutate(
    ag_wage_l1 = lag(ag_wage),
    ag_wage_l2 = lag(ag_wage, 2),
    year_l1 = lag(year),
    year_l2 = lag(year, 2),
    dln_ag_wage_l1 = if_else(
      year_l1 == year - 1 & year_l2 == year - 2,
      log(ag_wage_l1) - log(ag_wage_l2),
      NA_real_
    )
  ) %>%
  select(-ag_wage_l2, -year_l1, -year_l2) %>%
  ungroup() %>%
  inner_join(iv_clusters, by = c("cz_aewr_region_fe", "aewr_region_num"))

unit_qcew_wages <- county_year_units %>%
  filter(!is.na(emp_farm), emp_farm > 0) %>%
  group_by(cz_aewr_region_fe, aewr_region_num, state_fips_code, year) %>%
  summarise(
    unit_state_emp_farm = sum(emp_farm, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  inner_join(
    qcew %>%
      filter(
        !is.na(qcew_ag_mean_hourly_wage_40h),
        qcew_ag_workers > 0
      ) %>%
      select(
        state_fips_code,
        year,
        qcew_ag_mean_hourly_wage_40h
      ),
    by = c("state_fips_code", "year")
  ) %>%
  group_by(cz_aewr_region_fe, aewr_region_num, year) %>%
  summarise(
    ag_wage = weighted.mean(
      qcew_ag_mean_hourly_wage_40h,
      unit_state_emp_farm,
      na.rm = TRUE
    ),
    .groups = "drop"
  ) %>%
  # In outcome year t, use the proxy change from t - 2 to t - 1.
  arrange(cz_aewr_region_fe, year) %>%
  group_by(cz_aewr_region_fe) %>%
  mutate(
    ag_wage_l1 = lag(ag_wage),
    ag_wage_l2 = lag(ag_wage, 2),
    year_l1 = lag(year),
    year_l2 = lag(year, 2),
    dln_ag_wage_l1 = if_else(
      year_l1 == year - 1 & year_l2 == year - 2,
      log(ag_wage_l1) - log(ag_wage_l2),
      NA_real_
    )
  ) %>%
  select(-ag_wage_l2, -year_l1, -year_l2) %>%
  ungroup() %>%
  inner_join(iv_clusters, by = c("cz_aewr_region_fe", "aewr_region_num"))

iv_oews <- unit_oews_wages %>%
  select(
    target_unit = cz_aewr_region_fe,
    aewr_region_num,
    year,
    target_cluster = iv_cluster
  ) %>%
  inner_join(
    unit_oews_wages %>%
      select(
        donor_unit = cz_aewr_region_fe,
        aewr_region_num,
        year,
        donor_cluster = iv_cluster,
        donor_ag_wage_l1 = ag_wage_l1,
        donor_dln_ag_wage_l1 = dln_ag_wage_l1
      ),
    by = c("aewr_region_num", "year")
  ) %>%
  inner_join(
    iv_donor_clusters,
    by = c("aewr_region_num", "target_cluster", "donor_cluster")
  ) %>%
  group_by(cz_aewr_region_fe = target_unit, aewr_region_num, year) %>%
  summarise(
    z_oews_agwage_l1 = mean(donor_ag_wage_l1, na.rm = TRUE),
    z_oews_dln_agwage_l1 = mean(donor_dln_ag_wage_l1, na.rm = TRUE),
    oews_iv_donor_units = n_distinct(donor_unit),
    oews_iv_donor_clusters = n_distinct(donor_cluster),
    oews_iv_donor_cluster_distance = mean(donor_cluster_distance),
    .groups = "drop"
  ) %>%
  mutate(
    across(starts_with("z_oews_"), ~ if_else(is.nan(.x), NA_real_, .x))
  )

iv_qcew <- unit_qcew_wages %>%
  select(
    target_unit = cz_aewr_region_fe,
    aewr_region_num,
    year,
    target_cluster = iv_cluster
  ) %>%
  inner_join(
    unit_qcew_wages %>%
      select(
        donor_unit = cz_aewr_region_fe,
        aewr_region_num,
        year,
        donor_cluster = iv_cluster,
        donor_ag_wage_l1 = ag_wage_l1,
        donor_dln_ag_wage_l1 = dln_ag_wage_l1
      ),
    by = c("aewr_region_num", "year")
  ) %>%
  inner_join(
    iv_donor_clusters,
    by = c("aewr_region_num", "target_cluster", "donor_cluster")
  ) %>%
  group_by(cz_aewr_region_fe = target_unit, aewr_region_num, year) %>%
  summarise(
    z_qcew_agwage_l1 = mean(donor_ag_wage_l1, na.rm = TRUE),
    z_qcew_dln_agwage_l1 = mean(donor_dln_ag_wage_l1, na.rm = TRUE),
    qcew_iv_donor_units = n_distinct(donor_unit),
    qcew_iv_donor_clusters = n_distinct(donor_cluster),
    qcew_iv_donor_cluster_distance = mean(donor_cluster_distance),
    .groups = "drop"
  ) %>%
  mutate(
    across(starts_with("z_qcew_"), ~ if_else(is.nan(.x), NA_real_, .x))
  )

county_df_iv <- county_df %>%
  mutate(
    dln_aewr = log(aewr) - log(aewr_l1),
    dln_aewr_ppi = log(aewr_ppi) - log(aewr_ppi_l1)
  ) %>%
  left_join(iv_clusters, by = c("cz_aewr_region_fe", "aewr_region_num")) %>%
  left_join(iv_oews, by = c("cz_aewr_region_fe", "aewr_region_num", "year")) %>%
  left_join(iv_qcew, by = c("cz_aewr_region_fe", "aewr_region_num", "year"))

write_parquet(
  county_df_iv,
  path_int("county_df_analysis_year_iv.parquet")
)

cat(
  "county_df_analysis_year_iv:",
  nrow(county_df_iv),
  "rows,",
  ncol(county_df_iv),
  "cols\n"
)
cat(
  "Nonmissing OEWS level IV rows:",
  sum(!is.na(county_df_iv$z_oews_agwage_l1)),
  "\n"
)
cat(
  "Nonmissing OEWS change IV rows:",
  sum(!is.na(county_df_iv$z_oews_dln_agwage_l1)),
  "\n"
)
cat(
  "Nonmissing QCEW level IV rows:",
  sum(!is.na(county_df_iv$z_qcew_agwage_l1)),
  "\n"
)
cat(
  "Nonmissing QCEW change IV rows:",
  sum(!is.na(county_df_iv$z_qcew_dln_agwage_l1)),
  "\n"
)
