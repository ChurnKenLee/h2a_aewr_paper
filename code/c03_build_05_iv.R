# Construct CZ-level AEWR IV using same-AEWR-region CZs
rm(list = ls())
if (file.exists("paths.R")) {
  source("paths.R")
} else {
  source(file.path("code", "paths.R"))
}
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(foreign)

crops <- read_parquet(path_processed(
  "cdl_cropshares.parquet"
)) %>%
  clean_names()

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

# Fixed 2008-2011 county primitives ------------------------------------------

crop_features <- crops %>%
  mutate(countyfips = str_pad(countyfips, 5, side = "left", pad = "0")) %>%
  filter(year >= 2008, year <= 2011) %>%
  select(countyfips, starts_with("acres_"), -any_of("acres_na")) %>%
  group_by(countyfips) %>%
  summarise(across(starts_with("acres_"), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
  mutate(crop_acres = rowSums(across(starts_with("acres_")), na.rm = TRUE)) %>%
  mutate(across(starts_with("acres_"), ~ .x / crop_acres, .names = "share_{.col}")) %>%
  select(countyfips, starts_with("share_acres_"))

climate_features <- climate %>%
  mutate(countyfips = str_pad(as.character(fips), 5, side = "left", pad = "0")) %>%
  filter(year >= 2008, year <= 2011) %>%
  select(countyfips, starts_with("normal_cb_")) %>%
  group_by(countyfips) %>%
  summarise(across(starts_with("normal_cb_"), ~ mean(.x, na.rm = TRUE)), .groups = "drop")

soil_vars <- c(
  "slope_r", "slopegradwta", "resdept_r", "aws025wta", "aws050wta",
  "aws0100wta", "aws0150wta", "wtdepannmin", "wtdepaprjunmin",
  "brockdepmin", "cropprodindex"
)

soil_features <- soil %>%
  mutate(countyfips = str_pad(county_ansi, 5, side = "left", pad = "0")) %>%
  group_by(countyfips) %>%
  summarise(
    across(all_of(soil_vars), ~ weighted.mean(.x, total_acres, na.rm = TRUE)),
    .groups = "drop"
  )

county_features <- crop_features %>%
  full_join(climate_features, by = "countyfips") %>%
  full_join(soil_features, by = "countyfips")

# Cluster CZ x AEWR-region units within AEWR regions --------------------------

unit_xwalk <- county_df %>%
  distinct(countyfips, cz_out10, aewr_region_num, cz_aewr_region_fe)

unit_features <- unit_xwalk %>%
  left_join(county_features, by = "countyfips") %>%
  group_by(cz_out10, aewr_region_num, cz_aewr_region_fe) %>%
  summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)), .groups = "drop")

feature_names <- setdiff(
  names(unit_features),
  c("cz_out10", "aewr_region_num", "cz_aewr_region_fe")
)

unit_features_scaled <- unit_features
for (v in feature_names) {
  x <- unit_features_scaled[[v]]
  x[is.nan(x)] <- NA_real_
  med <- median(x, na.rm = TRUE)
  if (is.na(med)) med <- 0
  x[is.na(x)] <- med
  sx <- sd(x)
  unit_features_scaled[[v]] <- if (!is.na(sx) && sx > 0) (x - mean(x)) / sx else 0
}

cluster_list <- list()
for (r in sort(unique(unit_features_scaled$aewr_region_num))) {
  d <- unit_features_scaled %>% filter(aewr_region_num == r)
  x <- as.matrix(d[, feature_names])
  cluster_list[[as.character(r)]] <- d %>%
    select(cz_aewr_region_fe, aewr_region_num) %>%
    mutate(iv_cluster = cutree(hclust(dist(x), method = "ward.D2"), k = 2))
}

iv_clusters <- bind_rows(cluster_list)

# Donor wage instrument --------------------------------------------------------

county_year_units <- county_df %>%
  distinct(countyfips, year, cz_aewr_region_fe, aewr_region_num)

unit_oews_wages <- oews %>%
  filter(occ_code == "AEWR", !is.na(oews_mean_hourly_wage), oews_tot_emp > 0) %>%
  mutate(
    countyfips = paste0(
      sprintf("%02d", as.integer(state_fips_code)),
      sprintf("%03d", as.integer(county_fips_code))
    )
  ) %>%
  inner_join(county_year_units, by = c("countyfips", "year")) %>%
  group_by(cz_aewr_region_fe, aewr_region_num, year) %>%
  summarise(
    ag_wage = weighted.mean(oews_mean_hourly_wage, oews_tot_emp, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(cz_aewr_region_fe, year) %>%
  group_by(cz_aewr_region_fe) %>%
  mutate(ag_wage_l1 = lag(ag_wage)) %>%
  ungroup() %>%
  inner_join(iv_clusters, by = c("cz_aewr_region_fe", "aewr_region_num"))

iv <- unit_oews_wages %>%
  select(target_unit = cz_aewr_region_fe, aewr_region_num, year, target_cluster = iv_cluster) %>%
  inner_join(
    unit_oews_wages %>%
      select(
        donor_unit = cz_aewr_region_fe,
        aewr_region_num,
        year,
        donor_cluster = iv_cluster,
        donor_ag_wage_l1 = ag_wage_l1
      ),
    by = c("aewr_region_num", "year")
  ) %>%
  filter(target_cluster != donor_cluster, !is.na(donor_ag_wage_l1)) %>%
  group_by(cz_aewr_region_fe = target_unit, aewr_region_num, year) %>%
  summarise(
    z_oews_agwage_l1 = mean(donor_ag_wage_l1),
    iv_donor_units = n_distinct(donor_unit),
    .groups = "drop"
  )

county_df_iv <- county_df %>%
  left_join(iv_clusters, by = c("cz_aewr_region_fe", "aewr_region_num")) %>%
  left_join(iv, by = c("cz_aewr_region_fe", "aewr_region_num", "year"))

write_parquet(county_df_iv, path_processed("county_df_analysis_year_iv.parquet"))
write_parquet(iv_clusters, path_processed("aewr_iv_clusters.parquet"))

cat("county_df_analysis_year_iv:", nrow(county_df_iv), "rows,", ncol(county_df_iv), "cols\n")
cat("Nonmissing IV rows:", sum(!is.na(county_df_iv$z_oews_agwage_l1)), "\n")
