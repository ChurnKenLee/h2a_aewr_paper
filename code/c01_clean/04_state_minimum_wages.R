# Purpose: Construct real state and agricultural minimum-wage measures.
# Inputs: fred_state_minwages.parquet, state_year_min_wage.parquet, and ppi_2012.parquet.
# Output: data/intermediate/state_real_minwages.parquet.
# Run after: 03_producer_price_index.R; the FRED parquet is an upstream input.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
library(arrow)
library(dplyr)

state_minwages <- read_parquet(path_int("fred_state_minwages.parquet"))

state_min_alt <- read_parquet(path_int("state_year_min_wage.parquet"))
ppi_data <- read_parquet(path_int("ppi_2012.parquet"))
state_min_alt <- state_min_alt %>%
  filter(year == 2024) %>%
  select(state_fips, agriculture_exemption)

state_minwage_ppi <- state_minwages %>%
  left_join(ppi_data, by = "year", relationship = "many-to-one") %>%
  left_join(state_min_alt, by = "state_fips", relationship = "many-to-one") %>%
  mutate(
    agriculture_exemption = coalesce(agriculture_exemption, TRUE),
    state_min_wage = coalesce(state_min_wage, federal_min_wage),
    prevailing_ag_min_wage = if_else(
      agriculture_exemption,
      federal_min_wage,
      pmax(state_min_wage, federal_min_wage, na.rm = TRUE)
    ),
    prevailing_min_wage_ppi = prevailing_min_wage / ppi_2012,
    prevailing_ag_min_wage_ppi = prevailing_ag_min_wage / ppi_2012,
    federal_min_wage_ppi = federal_min_wage / ppi_2012,
    state_min_wage_ppi = state_min_wage / ppi_2012
  ) %>%
  select(-ppi_2012)

write_parquet(
  state_minwage_ppi,
  path_int("state_real_minwages.parquet")
)
