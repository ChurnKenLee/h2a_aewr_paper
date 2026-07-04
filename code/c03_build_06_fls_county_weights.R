# Construct county weights that proxy FLS regional wage contribution
rm(list = ls())
if (!exists("path_code", mode = "function")) {
  source(file.path("code", "paths.R"))
}
ensure_project_dirs()

library(arrow)
library(dplyr)

# Minimal baseline:
#   90 percent BEA hired farm employment share within AEWR region-year
#   10 percent CDL ag land share within AEWR region-year
# AEWR/FLS wages are intentionally not used to estimate the weights.
fls_list_frame_share <- 0.75

county_year <- read_parquet(path_processed(
  "county_df_analysis_year.parquet"
)) %>%
  transmute(
    countyfips,
    year = as.integer(year),
    statefips,
    state_abbrev,
    aewr_region_num = as.integer(aewr_region_num),
    cz_out10,
    cz_aewr_region_fe = as.character(cz_aewr_region_fe),
    bea_farm_emp = as.numeric(emp_farm)
  ) %>%
  distinct() %>%
  filter(!is.na(aewr_region_num))

county_year_duplicates <- county_year %>%
  count(countyfips, year) %>%
  filter(n > 1)

if (nrow(county_year_duplicates) > 0) {
  stop("county_year has duplicated county-year rows after distinct().")
}

cdl_ag_land <- read_parquet(path_int(
  "croplandcros_county_crop_type_acres.parquet"
)) %>%
  mutate(
    year = as.integer(year),
    acres = as.numeric(acres),
    cdl_is_ag = !is.na(crop_type_label) & crop_type_label != "non-crop"
  ) %>%
  group_by(countyfips, year) %>%
  summarise(
    cdl_ag_acres = sum(if_else(cdl_is_ag, acres, 0), na.rm = TRUE),
    cdl_non_ag_acres = sum(if_else(!cdl_is_ag, acres, 0), na.rm = TRUE),
    cdl_total_acres = sum(acres, na.rm = TRUE),
    cdl_ag_share_county = if_else(
      cdl_total_acres > 0,
      cdl_ag_acres / cdl_total_acres,
      NA_real_
    ),
    .groups = "drop"
  )

fls_county_weight <- county_year %>%
  left_join(cdl_ag_land, by = c("countyfips", "year")) %>%
  mutate(
    bea_farm_emp_weight_base = if_else(
      !is.na(bea_farm_emp) & bea_farm_emp > 0,
      bea_farm_emp,
      0
    ),
    cdl_ag_acres_weight_base = if_else(
      !is.na(cdl_ag_acres) & cdl_ag_acres > 0,
      cdl_ag_acres,
      0
    )
  ) %>%
  group_by(aewr_region_num, year) %>%
  mutate(
    aewr_region_bea_farm_emp = sum(bea_farm_emp_weight_base, na.rm = TRUE),
    aewr_region_cdl_ag_acres = sum(cdl_ag_acres_weight_base, na.rm = TRUE),
    fls_emp_share = if_else(
      aewr_region_bea_farm_emp > 0,
      bea_farm_emp_weight_base / aewr_region_bea_farm_emp,
      NA_real_
    ),
    fls_area_share = if_else(
      aewr_region_cdl_ag_acres > 0,
      cdl_ag_acres_weight_base / aewr_region_cdl_ag_acres,
      NA_real_
    ),
    fls_weight_raw = fls_list_frame_share *
      coalesce(fls_emp_share, 0) +
      (1 - fls_list_frame_share) * coalesce(fls_area_share, 0),
    fls_weight_raw_region_total = sum(fls_weight_raw, na.rm = TRUE),
    fls_county_weight = if_else(
      fls_weight_raw_region_total > 0,
      fls_weight_raw / fls_weight_raw_region_total,
      NA_real_
    )
  ) %>%
  ungroup() %>%
  arrange(countyfips, year) %>%
  group_by(countyfips) %>%
  mutate(
    fls_county_weight_l1 = if_else(
      lag(year) == year - 1L,
      lag(fls_county_weight),
      NA_real_
    )
  ) %>%
  ungroup() %>%
  select(
    countyfips,
    year,
    statefips,
    state_abbrev,
    aewr_region_num,
    cz_out10,
    cz_aewr_region_fe,
    bea_farm_emp,
    cdl_ag_acres,
    cdl_non_ag_acres,
    cdl_total_acres,
    cdl_ag_share_county,
    aewr_region_bea_farm_emp,
    aewr_region_cdl_ag_acres,
    fls_emp_share,
    fls_area_share,
    fls_weight_raw,
    fls_county_weight,
    fls_county_weight_l1
  )

fls_county_weight_summary <- fls_county_weight %>%
  group_by(year, aewr_region_num) %>%
  summarise(
    counties = n(),
    weight_sum = sum(fls_county_weight, na.rm = TRUE),
    emp_share_sum = sum(fls_emp_share, na.rm = TRUE),
    area_share_sum = sum(fls_area_share, na.rm = TRUE),
    max_weight = max(fls_county_weight, na.rm = TRUE),
    effective_counties = 1 / sum(fls_county_weight^2, na.rm = TRUE),
    .groups = "drop"
  )

write_parquet(
  fls_county_weight,
  path_int("fls_county_weight.parquet")
)

cat(
  "fls_county_weight:",
  nrow(fls_county_weight),
  "rows,",
  ncol(fls_county_weight),
  "cols\n"
)
cat(
  "FLS county weights written to",
  path_int("fls_county_weight.parquet"),
  "\n"
)
print(summary(fls_county_weight_summary$weight_sum))
