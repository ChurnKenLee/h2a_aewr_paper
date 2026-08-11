# Purpose: Build the state-year AEWR panel in real and nominal terms.
# Inputs: aewr.parquet, ppi_2012.parquet, and fips_codes.csv.
# Output: data/intermediate/aewr_data_year.parquet.
# Run after: 03_producer_price_index.R and the a-stage AEWR extraction.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(dplyr)
library(readr)

fips_codes <- read_csv(
  path_raw("geographic_crosswalks", "phil", "fips_codes.csv"),
  show_col_types = FALSE
) %>%
  transmute(
    state_fips = state_fips(fips),
    across(-fips)
  )
ppi_data <- read_parquet(path_int("ppi_2012.parquet"))

aewr_data <- read_parquet(
  path_int("aewr.parquet")
) %>%
  left_join(ppi_data, by = "year", relationship = "many-to-one") %>%
  mutate(
    aewr = as.numeric(aewr),
    aewr_ppi = aewr / ppi_2012
  ) %>%
  arrange(state_fips, year) %>%
  group_by(state_fips) %>%
  mutate(
    aewr_ppi_l1 = lag(aewr_ppi),
    aewr_l1 = lag(aewr),
    aewr_ppi_l2 = lag(aewr_ppi, n = 2),
    aewr_l2 = lag(aewr, n = 2)
  ) %>%
  ungroup() %>%
  filter(year > 2007, year <= 2022) %>%
  left_join(fips_codes, by = "state_fips", relationship = "many-to-one") %>%
  select(
    aewr,
    aewr_ppi,
    aewr_l1,
    aewr_ppi_l1,
    aewr_l2,
    aewr_ppi_l2,
    year,
    state_fips,
    state_abbrev
  )

write_parquet(aewr_data, path_int("aewr_data_year.parquet"))
