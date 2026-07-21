# Purpose: Normalize the county Fisher price index for the analysis window.
# Input: data/intermediate/price_index_fisher_county_year.parquet.
# Output: data/intermediate/nass_fisher_price_index.parquet.
# Run after: code/b01_derived/02_price_index_nass_synthetic_cdl.py.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
source(path_code("c00_shared", "fips.R"))
library(arrow)
library(dplyr)

price_data <- read_parquet(path_int(
  "price_index_fisher_county_year.parquet"
)) %>%
  mutate(
    countyfips = county_fips(fips),
    year = as.integer(year)
  ) %>%
  select(countyfips, year, fisher_index) %>%
  filter(year >= 2008 & year <= 2022)

write_parquet(
  price_data,
  path_int("nass_fisher_price_index.parquet")
)
