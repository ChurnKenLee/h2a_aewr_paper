# Purpose: Build the annual county H-2A panel and normalize its prediction.
# Inputs: h2a_aggregated.parquet and the elastic-net H-2A prediction.
# Outputs: h2a_predict.parquet and h2a_data_year.parquet.
# Run after: code/b01_derived/01_h2a_aggregation_nodupes.R and
# code/b01_derived/07_h2a_prediction_elastic_net.py.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
library(arrow)
library(dplyr)

h2a_data <- read_parquet(
  file = path_int("h2a_aggregated.parquet")
)
h2a_predict <- read_parquet(
  file = path_int("h2a_prediction_using_elastic_net_continuous_basis.parquet")
)
write_parquet(h2a_predict, path_int("h2a_predict.parquet"))

h2a_data <- h2a_data |>
  filter(year > 2007, year <= 2022, !is.na(county_fips)) |>
  select(-state_fips, -county_code)

write_parquet(h2a_data, path_int("h2a_data_year.parquet"))
