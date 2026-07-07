# Construct county weights that proxy FLS regional wage contribution
rm(list = ls())
if (!exists("path_code", mode = "function")) {
  source(file.path("code", "paths.R"))
}
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)

# Current baseline:
# 75 percent BEA hired farm employment share within AEWR region-year
# 25 percent CDL ag land share within AEWR region-year
fls_list_frame_share <- 1

county_year <- read_parquet(path_processed(
  "county_df_analysis_year.parquet"
)) %>%
  select(
    countyfips,
    year,
    statefips,
    state_abbrev,
    aewr_region_num,
    cz_out10,
    cz_aewr_region_fe,
    emp_farm
  ) %>%
  rename(bea_farm_emp = emp_farm) %>%
  distinct() %>%
  filter(!is.na(aewr_region_num))

# Count acreage of crop land within county
cdl_ag_land <- read_parquet(path_int(
  "croplandcros_county_crop_type_acres.parquet"
))
crop_type_list <- c(
  "field crops",
  "fruit & tree nuts",
  "horticulture",
  "mixed crops",
  "mixed field crops/horticulture",
  "mixed field crops/horticulture/vegetables",
  "mixed field crops/vegetables",
  "mixed fruit & tree nuts/vegetables",
  "vegetables"
)
county_crop_acreage <- cdl_ag_land %>%
  mutate(cdl_is_ag = crop_type_label %in% crop_type_list) %>%
  group_by(countyfips, year) %>%
  summarise(
    cdl_ag_acres = sum(if_else(cdl_is_ag, acres, 0), na.rm = TRUE),
    cdl_non_ag_acres = sum(if_else(!cdl_is_ag, acres, 0), na.rm = TRUE)
  ) %>%
  ungroup()

# Join CDL crop acreage to county farm emp to get weights
# Note county-year dataframe ends 2022, while CDL goes to 2025
fls_county_weight <- county_year %>%
  left_join(county_crop_acreage, by = c("countyfips", "year")) %>%
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
    aewr_region_cdl_ag_acres = sum(cdl_ag_acres_weight_base, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  mutate(
    fls_emp_share = bea_farm_emp_weight_base / aewr_region_bea_farm_emp,
    fls_area_share = cdl_ag_acres_weight_base / aewr_region_cdl_ag_acres
  ) %>%
  mutate(
    fls_weight_raw = fls_list_frame_share *
      fls_emp_share +
      (1 - fls_list_frame_share) * fls_area_share
  ) %>%
  group_by(aewr_region_num, year) %>%
  mutate(fls_weight_raw_region_total = sum(fls_weight_raw)) %>%
  ungroup() %>%
  mutate(fls_county_weight = fls_weight_raw / fls_weight_raw_region_total) %>%
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
    aewr_region_bea_farm_emp,
    aewr_region_cdl_ag_acres,
    fls_emp_share,
    fls_area_share,
    fls_weight_raw,
    fls_county_weight,
    fls_county_weight_l1
  )

write_parquet(
  fls_county_weight,
  path_int("fls_county_weight.parquet")
)
