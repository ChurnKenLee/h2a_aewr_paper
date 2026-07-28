# Purpose: Reshape and harmonize annual BEA county employment measures.
# Inputs: bea_CAEMP25N_trim.parquet and bea_fips_xwalk.csv.
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
) |>
  prepare_bea_county_crosswalk()

bea_caemp25n_data <- read_parquet(path_int("bea_CAEMP25N_trim.parquet"))
# Retain total, proprietor, farm, nonfarm, and private-nonfarm employment.

bea_caemp25n_data <- bea_caemp25n_data |>
  filter(
    LineCode == 10 |
      LineCode == 50 |
      LineCode == 70 |
      LineCode == 80 |
      LineCode == 90
  )

bea_caemp25n_data <- bea_caemp25n_data |>
  mutate(
    county_fips = county_fips(GeoFIPS), # remove quotes
    category = ifelse(
      LineCode == 10,
      "emp_tot",
      ifelse(
        LineCode == 50,
        "emp_farm_propr", # farm proprietors
        ifelse(
          LineCode == 70,
          "emp_farm",
          ifelse(
            LineCode == 80,
            "emp_nonfarm",
            ifelse(LineCode == 90, "emp_privatenonfarm", NA)
          )
        )
      )
    )
  )

bea_caemp25n_data <- bea_caemp25n_data |>
  select(9:32)

bea_caemp25n_data <- bea_caemp25n_data |>
  pivot_longer(
    cols = starts_with("y"),
    names_to = "year",
    names_prefix = "y",
    values_to = "temp",
    values_drop_na = FALSE
  )

bea_caemp25n_data <- bea_caemp25n_data |>
  mutate(
    year = as.integer(year),
    emp = suppressWarnings(as.numeric(temp))
  ) |>
  select(-temp)

bea_caemp25n_data <- bea_caemp25n_data |>
  pivot_wider(names_from = "category", values_from = "emp")

bea_caemp25n_data <- bea_caemp25n_data |>
  filter(year > 2007 & year <= 2022)

bea_caemp25n_data <- apply_bea_county_crosswalk(
  bea_caemp25n_data,
  bea_fips_xwalk
)

write_parquet(
  bea_caemp25n_data,
  path_int("bea_caemp25n_data_year.parquet")
)
