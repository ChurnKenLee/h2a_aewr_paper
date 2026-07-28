# Purpose: Merge normalized source panels onto the county-year backbone.
# Output: data/intermediate/county_year_merged.parquet.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(dplyr)
library(readr)

read_year_panel <- function(filename) {
  read_parquet(path_int(filename)) |>
    mutate(year = as.integer(year))
}

aewr_regions <- read_csv(
  path_raw("geographic_crosswalks", "phil", "aewr_regions.csv"),
  show_col_types = FALSE
) |>
  rename(aewr_region_id = aewr_region_num) |>
  mutate(aewr_region_id = aewr_region_id(aewr_region_id))

fips_codes <- read_csv(
  path_raw("geographic_crosswalks", "phil", "fips_codes.csv"),
  show_col_types = FALSE
) |>
  transmute(
    state_fips = state_fips(fips),
    across(-fips)
  )

state_minimum_wages <- read_parquet(
  path_int("state_real_minwages.parquet")
) |>
  mutate(year = as.integer(year)) |>
  arrange(state_fips, year) |>
  group_by(state_fips) |>
  mutate(
    across(
      -year,
      lag,
      .names = "{.col}_l1"
    )
  ) |>
  ungroup()

ppi <- read_parquet(path_int("ppi_2012.parquet"))
wage_quantiles <- read_parquet(
  path_int("acs_czone_wage_quantiles.parquet")
) |>
  rename(year = YEAR) |>
  filter(year >= 2005) |>
  left_join(ppi, by = "year", relationship = "many-to-one") |>
  mutate(
    across(
      c(wage_p10, wage_p25, wage_p50, wage_p75, wage_p90),
      \(value) value / ppi_2012
    )
  ) |>
  select(-ppi_2012) |>
  arrange(county_fips, year) |>
  group_by(county_fips) |>
  mutate(
    across(
      starts_with("wage_p"),
      lag,
      .names = "{.col}_l1"
    )
  ) |>
  ungroup()

county_panel <- read_parquet(path_int("county_df_year.parquet")) |>
  mutate(state_fips = state_from_county_fips(county_fips)) |>
  left_join(fips_codes, by = "state_fips", relationship = "many-to-one") |>
  inner_join(
    aewr_regions,
    by = "state_abbrev",
    relationship = "many-to-one"
  ) |>
  left_join(
    read_year_panel("aewr_data_year.parquet"),
    by = c("year", "state_fips", "state_abbrev"),
    relationship = "many-to-one"
  ) |>
  left_join(
    read_parquet(path_int("cz_file_2010_small.parquet")),
    by = "county_fips",
    relationship = "many-to-one"
  ) |>
  left_join(
    read_year_panel("bea_caemp25n_data_year.parquet"),
    by = c("county_fips", "year"),
    relationship = "many-to-one"
  ) |>
  left_join(
    read_year_panel("bea_cainc45_data_year.parquet"),
    by = c("county_fips", "year"),
    relationship = "many-to-one"
  ) |>
  left_join(
    read_year_panel("h2a_data_year.parquet"),
    by = c("county_fips", "year"),
    relationship = "many-to-one"
  ) |>
  left_join(
    read_year_panel("census_pop_ests_year.parquet"),
    by = c("county_fips", "year"),
    relationship = "many-to-one"
  ) |>
  left_join(
    read_year_panel("census_ag_cropland_year.parquet"),
    by = c("county_fips", "year"),
    relationship = "many-to-one"
  ) |>
  left_join(
    read_year_panel("nass_fisher_price_index.parquet"),
    by = c("county_fips", "year"),
    relationship = "many-to-one"
  ) |>
  left_join(
    read_parquet(path_int("h2a_predict.parquet")),
    by = "county_fips",
    relationship = "many-to-one"
  ) |>
  left_join(
    read_parquet(path_int("census_ag_cropland_2007_year.parquet")),
    by = "county_fips",
    relationship = "many-to-one"
  ) |>
  left_join(
    state_minimum_wages,
    by = c("state_fips", "year"),
    relationship = "many-to-one"
  ) |>
  left_join(
    wage_quantiles,
    by = c("county_fips", "year", "cz_id"),
    relationship = "many-to-one"
  ) |>
  mutate(fisher_index_ppi = fisher_index / ppi_2012)

if (
  nrow(county_panel) == 0L ||
    anyDuplicated(county_panel[c("county_fips", "year")]) > 0L
) {
  stop("county_year_merged must have unique county-year keys.", call. = FALSE)
}

write_parquet(
  county_panel,
  path_int("county_year_merged.parquet")
)
