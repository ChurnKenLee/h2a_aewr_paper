# Purpose: Reshape, deflate, and harmonize annual BEA county farm-income and
# farm-employee wage measures.
# Inputs: bea_CAINC45_trim.parquet, ppi_2012.parquet,
# county_adjacency2010.parquet, and the BEA FIPS crosswalk.
# Output: data/intermediate/bea_cainc45_data_year.parquet.
# Run after: 03_producer_price_index.R and
# code/a01_sources/08_bea_farm_nonfarm_emp.py.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("c00_shared", "bea_county_crosswalk.R"))
library(arrow)
library(dplyr)
library(readr)
library(tidyr)

ppi_data <- read_parquet(path_int("ppi_2012.parquet")) %>%
  mutate(year = as.integer(year))
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

bea_cainc45_categories <- c(
  "20" = "farm_cashanimal",
  "60" = "farm_cashcrops",
  "130" = "farm_govpayments",
  "150" = "farm_prodexp",
  "210" = "farm_laborexpense",
  "270" = "farm_cashandinc",
  "350" = "farm_wages_salaries",
  "360" = "farm_wage_supplements"
)

bea_cainc45_nominal_descriptions <- c(
  farm_cashanimal = "BEA CAINC45 line 20 livestock cash receipts, nominal thousands of dollars",
  farm_cashcrops = "BEA CAINC45 line 60 crop cash receipts, nominal thousands of dollars",
  farm_govpayments = "BEA CAINC45 line 130 government payments to farms, nominal thousands of dollars",
  farm_prodexp = "BEA CAINC45 line 150 farm production expenses, nominal thousands of dollars",
  farm_laborexpense = "BEA CAINC45 line 210 hired farm labor expenses, nominal thousands of dollars",
  farm_cashandinc = "BEA CAINC45 line 270 farm cash receipts and other income, nominal thousands of dollars",
  farm_wages_salaries = "BEA CAINC45 line 350 farm wages and salaries, nominal thousands of dollars",
  farm_wage_supplements = "BEA CAINC45 line 360 farm supplements to wages and salaries, nominal thousands of dollars"
)

bea_cainc45_line_codes <- setNames(
  names(bea_cainc45_categories),
  unname(bea_cainc45_categories)
)

bea_cainc45_data <- read_parquet(path_int("bea_CAINC45_trim.parquet")) %>%
  filter(as.character(LineCode) %in% names(bea_cainc45_categories)) %>%
  mutate(
    county_fips = county_fips(GeoFIPS),
    category = unname(bea_cainc45_categories[as.character(LineCode)])
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
    fin = suppressWarnings(as.numeric(temp))
  ) %>%
  select(-temp) %>%
  pivot_wider(names_from = "category", values_from = "fin") %>%
  left_join(ppi_data, by = "year", relationship = "many-to-one") %>%
  mutate(
    farm_cashanimal_ppi = farm_cashanimal / ppi_2012,
    farm_cashcrops_ppi = farm_cashcrops / ppi_2012,
    farm_govpayments_ppi = farm_govpayments / ppi_2012,
    farm_prodexp_ppi = farm_prodexp / ppi_2012,
    farm_laborexpense_ppi = farm_laborexpense / ppi_2012,
    farm_cashandinc_ppi = farm_cashandinc / ppi_2012,
    farm_wages_salaries_ppi = farm_wages_salaries / ppi_2012,
    farm_wage_supplements_ppi = farm_wage_supplements / ppi_2012
  ) %>%
  filter(year > 2007L, year <= 2022L) %>%
  apply_bea_county_crosswalk(bea_fips_xwalk) %>%
  semi_join(canonical_counties, by = "county_fips") %>%
  arrange(county_fips, year)

bea_cainc45_real_descriptions <- setNames(
  sub(
    "nominal thousands of dollars",
    "thousands of 2012 dollars using ppi_2012",
    bea_cainc45_nominal_descriptions,
    fixed = TRUE
  ),
  paste0(names(bea_cainc45_nominal_descriptions), "_ppi")
)
bea_cainc45_descriptions <- c(
  bea_cainc45_nominal_descriptions,
  bea_cainc45_real_descriptions
)
bea_cainc45_output_line_codes <- c(
  bea_cainc45_line_codes,
  setNames(
    unname(bea_cainc45_line_codes),
    paste0(names(bea_cainc45_line_codes), "_ppi")
  )
)

missing_output_columns <- setdiff(
  c("county_fips", "year", "ppi_2012", names(bea_cainc45_descriptions)),
  names(bea_cainc45_data)
)
if (length(missing_output_columns) > 0L) {
  stop(
    "BEA CAINC45 output is missing required columns: ",
    paste(missing_output_columns, collapse = ", "),
    call. = FALSE
  )
}

for (column in names(bea_cainc45_descriptions)) {
  attr(bea_cainc45_data[[column]], "description") <-
    bea_cainc45_descriptions[[column]]
  attr(bea_cainc45_data[[column]], "level_of_aggregation") <- "county-year"
  attr(bea_cainc45_data[[column]], "source_table") <- "CAINC45"
  attr(bea_cainc45_data[[column]], "source_line_code") <-
    bea_cainc45_output_line_codes[[column]]
  attr(bea_cainc45_data[[column]], "unit") <- if (
    endsWith(column, "_ppi")
  ) {
    "Thousands of 2012 dollars"
  } else {
    "Thousands of current dollars"
  }
}

assert_geo_columns(bea_cainc45_data, "county_fips")
if (
  nrow(bea_cainc45_data) == 0L ||
    anyNA(bea_cainc45_data$year) ||
    anyDuplicated(bea_cainc45_data[c("county_fips", "year")]) > 0L
) {
  stop(
    "bea_cainc45_data_year must have unique, nonmissing county-year keys.",
    call. = FALSE
  )
}

if (
  any(
    !is.finite(bea_cainc45_data$ppi_2012) |
      bea_cainc45_data$ppi_2012 <= 0
  )
) {
  stop("BEA county-years require a finite, positive ppi_2012.", call. = FALSE)
}

bea_financial_values <- unlist(
  bea_cainc45_data[names(bea_cainc45_descriptions)]
)
if (
  any(
    !is.na(bea_financial_values) &
      (!is.finite(bea_financial_values) | bea_financial_values < 0)
  )
) {
  stop(
    "BEA farm financial measures must be finite and nonnegative.",
    call. = FALSE
  )
}

if (
  all(is.na(bea_cainc45_data$farm_wages_salaries)) ||
    all(is.na(bea_cainc45_data$farm_wage_supplements))
) {
  stop("BEA farm wage measures have no county-year coverage.", call. = FALSE)
}

write_parquet(
  bea_cainc45_data,
  path_int("bea_cainc45_data_year.parquet")
)
