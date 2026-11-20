# Purpose: Reshape and harmonize annual BEA county farm and total earnings.
# Inputs: bea_CAINC5N_trim.parquet, county_adjacency2010.parquet, and the BEA
# FIPS crosswalk.
# Output: data/intermediate/bea_cainc5n_data_year.parquet.
# Run after: code/a01_sources/08_bea_farm_nonfarm_emp.py.
#
# CAINC5N continues after BEA discontinued the CAEMP25N employment and CAINC45
# farm-income tables, whose last year is 2022.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("c00_shared", "bea_county_crosswalk.R"))
library(arrow)
library(dplyr)
library(readr)
library(tidyr)

bea_fips_xwalk <- read_csv(
  path_raw("geographic_crosswalks", "phil", "bea_fips_xwalk.csv"),
  show_col_types = FALSE
) %>%
  prepare_bea_county_crosswalk()

canonical_counties <- read_parquet(
  path_int("county_adjacency2010.parquet")
) %>%
  distinct(county_fips)
assert_geo_columns(canonical_counties, "county_fips")

bea_cainc5n_categories <- c(
  "35" = "earnings_place_of_work",
  "81" = "farm_earnings"
)
bea_cainc5n_descriptions <- c(
  earnings_place_of_work = "BEA CAINC5N line 35 earnings by place of work, nominal thousands of dollars",
  farm_earnings = "BEA CAINC5N line 81 farm earnings (may be negative), nominal thousands of dollars"
)
bea_cainc5n_line_codes <- setNames(
  names(bea_cainc5n_categories),
  unname(bea_cainc5n_categories)
)

bea_cainc5n_source <- read_parquet(path_int("bea_CAINC5N_trim.parquet"))
source_years <- as.integer(sub("^y", "", grep(
  "^y[0-9]{4}$", names(bea_cainc5n_source), value = TRUE
)))
expected_years <- intersect(
  SHARED_PANEL_START_YEAR:SHARED_PANEL_END_YEAR, source_years
)

bea_cainc5n_data <- bea_cainc5n_source %>%
  filter(as.character(LineCode) %in% names(bea_cainc5n_categories)) %>%
  mutate(
    county_fips = county_fips(GeoFIPS),
    category = unname(bea_cainc5n_categories[as.character(LineCode)])
  ) %>%
  select(county_fips, category, starts_with("y")) %>%
  pivot_longer(
    cols = starts_with("y"),
    names_to = "year",
    names_prefix = "y",
    values_to = "temp",
    values_drop_na = TRUE
  ) %>%
  mutate(
    year = as.integer(year),
    earnings = suppressWarnings(as.numeric(temp))
  ) %>%
  select(-temp) %>%
  pivot_wider(names_from = "category", values_from = "earnings") %>%
  filter(year >= SHARED_PANEL_START_YEAR, year <= SHARED_PANEL_END_YEAR) %>%
  apply_bea_county_crosswalk(bea_fips_xwalk) %>%
  semi_join(canonical_counties, by = "county_fips") %>%
  arrange(county_fips, year)

missing_output_columns <- setdiff(
  c("county_fips", "year", names(bea_cainc5n_descriptions)),
  names(bea_cainc5n_data)
)
if (length(missing_output_columns) > 0L) {
  stop(
    "BEA CAINC5N output is missing required columns: ",
    paste(missing_output_columns, collapse = ", "),
    call. = FALSE
  )
}

for (column in names(bea_cainc5n_descriptions)) {
  attr(bea_cainc5n_data[[column]], "description") <-
    bea_cainc5n_descriptions[[column]]
  attr(bea_cainc5n_data[[column]], "level_of_aggregation") <- "county-year"
  attr(bea_cainc5n_data[[column]], "source_table") <- "CAINC5N"
  attr(bea_cainc5n_data[[column]], "source_line_code") <-
    bea_cainc5n_line_codes[[column]]
  attr(bea_cainc5n_data[[column]], "unit") <- "Thousands of current dollars"
}

assert_geo_columns(bea_cainc5n_data, "county_fips")
if (
  nrow(bea_cainc5n_data) == 0L ||
    anyNA(bea_cainc5n_data$year) ||
    anyDuplicated(bea_cainc5n_data[c("county_fips", "year")]) > 0L ||
    !setequal(unique(bea_cainc5n_data$year), expected_years)
) {
  stop(
    "bea_cainc5n_data_year must have unique county-year keys covering ",
    "the observed source years within the shared calendar.",
    call. = FALSE
  )
}

write_parquet(
  bea_cainc5n_data,
  path_int("bea_cainc5n_data_year.parquet")
)
