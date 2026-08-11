# Purpose: Reshape and harmonize annual BEA county employment measures.
# Inputs: bea_CAEMP25N_trim.parquet, county_adjacency2010.parquet, and
# bea_fips_xwalk.csv.
# Output: data/intermediate/bea_caemp25n_data_year.parquet.
# Run after: code/a01_sources/08_bea_farm_nonfarm_emp.py.

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
if (
  nrow(canonical_counties) == 0L ||
    anyDuplicated(canonical_counties$county_fips) > 0L
) {
  stop("The 2010 county universe must be nonempty and unique.", call. = FALSE)
}

bea_caemp25n_categories <- c(
  "10" = "emp_tot",
  "20" = "emp_wage_salary",
  "50" = "emp_farm_propr",
  "70" = "emp_farm",
  "80" = "emp_nonfarm",
  "90" = "emp_privatenonfarm"
)

bea_caemp25n_descriptions <- c(
  emp_tot = "BEA CAEMP25N line 10 total employment, measured as number of jobs",
  emp_wage_salary = "BEA CAEMP25N line 20 wage-and-salary employment, measured as number of jobs",
  emp_farm_propr = "BEA CAEMP25N line 50 farm proprietors employment, measured as number of jobs",
  emp_farm = "BEA CAEMP25N line 70 farm employment, measured as number of jobs",
  emp_nonfarm = "BEA CAEMP25N line 80 nonfarm employment, measured as number of jobs",
  emp_privatenonfarm = "BEA CAEMP25N line 90 private nonfarm employment, measured as number of jobs"
)

bea_caemp25n_line_codes <- setNames(
  names(bea_caemp25n_categories),
  unname(bea_caemp25n_categories)
)

bea_caemp25n_data <- read_parquet(
  path_int("bea_CAEMP25N_trim.parquet")
) %>%
  filter(as.character(LineCode) %in% names(bea_caemp25n_categories)) %>%
  mutate(
    county_fips = county_fips(GeoFIPS),
    category = unname(bea_caemp25n_categories[as.character(LineCode)])
  ) %>%
  select(county_fips, category, starts_with("y")) %>%
  pivot_longer(
    cols = starts_with("y"),
    names_to = "year",
    names_prefix = "y",
    values_to = "temp",
    values_drop_na = FALSE
  ) %>%
  mutate(
    year = as.integer(year),
    emp = suppressWarnings(as.numeric(temp))
  ) %>%
  select(-temp) %>%
  pivot_wider(names_from = "category", values_from = "emp") %>%
  filter(year > 2007L, year <= 2022L) %>%
  apply_bea_county_crosswalk(bea_fips_xwalk) %>%
  semi_join(canonical_counties, by = "county_fips") %>%
  arrange(county_fips, year)

missing_output_columns <- setdiff(
  c("county_fips", "year", names(bea_caemp25n_descriptions)),
  names(bea_caemp25n_data)
)
if (length(missing_output_columns) > 0L) {
  stop(
    "BEA CAEMP25N output is missing required columns: ",
    paste(missing_output_columns, collapse = ", "),
    call. = FALSE
  )
}

for (column in names(bea_caemp25n_descriptions)) {
  attr(bea_caemp25n_data[[column]], "description") <-
    bea_caemp25n_descriptions[[column]]
  attr(bea_caemp25n_data[[column]], "level_of_aggregation") <- "county-year"
  attr(bea_caemp25n_data[[column]], "source_table") <- "CAEMP25N"
  attr(bea_caemp25n_data[[column]], "source_line_code") <-
    bea_caemp25n_line_codes[[column]]
  attr(bea_caemp25n_data[[column]], "unit") <- "Number of jobs"
}

assert_geo_columns(bea_caemp25n_data, "county_fips")
if (
  nrow(bea_caemp25n_data) == 0L ||
    anyNA(bea_caemp25n_data$year) ||
    anyDuplicated(bea_caemp25n_data[c("county_fips", "year")]) > 0L
) {
  stop(
    "bea_caemp25n_data_year must have unique, nonmissing county-year keys.",
    call. = FALSE
  )
}

bea_employment_values <- unlist(
  bea_caemp25n_data[names(bea_caemp25n_descriptions)]
)
if (
  any(
    !is.na(bea_employment_values) &
      (!is.finite(bea_employment_values) | bea_employment_values < 0)
  )
) {
  stop("BEA employment measures must be finite and nonnegative.", call. = FALSE)
}

if (all(is.na(bea_caemp25n_data$emp_wage_salary))) {
  stop("BEA wage-and-salary employment has no county-year coverage.", call. = FALSE)
}

write_parquet(
  bea_caemp25n_data,
  path_int("bea_caemp25n_data_year.parquet")
)
