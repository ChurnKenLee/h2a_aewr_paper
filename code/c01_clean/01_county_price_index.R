# Purpose: Normalize county Fisher price and quantity indexes for analysis.
# Input: data/intermediate/price_index_fisher_county_year.parquet.
# Output: data/intermediate/nass_fisher_price_index.parquet.
# Run after: code/b01_derived/02_price_index_nass_synthetic_cdl.py.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
library(arrow)
library(dplyr)

price_data <- read_parquet(path_int(
  "price_index_fisher_county_year.parquet"
))
price_data <- price_data |>
  select(
    county_fips,
    year,
    fisher_index,
    fisher_quantity_index
  ) |>
  filter(year >= 2008 & year <= 2022)

write_parquet(
  price_data,
  path_int("nass_fisher_price_index.parquet")
)
