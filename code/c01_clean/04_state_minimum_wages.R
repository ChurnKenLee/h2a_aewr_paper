# Purpose: Construct real state and agricultural minimum-wage measures.
# Inputs: fred_state_minwages.parquet, state_year_min_wage.parquet, and ppi_2012.parquet.
# Output: data/intermediate/state_real_minwages.parquet.
# Run after: 03_producer_price_index.R; the FRED parquet is an upstream input.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(dplyr)

state_minwages <- read_parquet(path_int("fred_state_minwages.parquet"))
assert_geo_columns(state_minwages, "state_fips")

## Alt Min Wage from Ken -------------------------------

state_min_alt <- read_parquet(path_int("state_year_min_wage.parquet"))
ppi_data <- read_parquet(path_int("ppi_2012.parquet"))
assert_geo_columns(state_min_alt, "state_fips")
state_min_alt <- state_min_alt %>%
  filter(year == 2024) %>% # these are stable, so ignore them
  select(state_fips, agriculture_exemption)

state_minwage_ppi <- merge(
  x = state_minwages,
  y = ppi_data,
  by = "year",
  all.x = T,
  all.y = F
)

state_minwage_ppi <- merge(
  x = state_minwage_ppi,
  y = state_min_alt,
  by = "state_fips",
  all.x = T,
  all.y = F
)

state_minwage_ppi$agriculture_exemption[is.na(
  state_minwage_ppi$agriculture_exemption
)] <- TRUE

# make the prevailing ag min wage #

# fill in state min with federal if missing #

state_minwage_ppi <- state_minwage_ppi %>%
  transform(
    state_min_wage = ifelse(
      !is.na(state_min_wage),
      state_min_wage,
      federal_min_wage
    )
  )

state_minwage_ppi <- state_minwage_ppi %>%
  mutate(
    prevailing_ag_min_wage = ifelse(
      agriculture_exemption == T,
      federal_min_wage,
      pmax(state_min_wage, federal_min_wage, na.rm = TRUE)
    )
  )

state_minwage_ppi <- state_minwage_ppi %>%
  mutate(
    prevailing_min_wage_ppi = prevailing_min_wage / ppi_2012,
    prevailing_ag_min_wage_ppi = prevailing_ag_min_wage / ppi_2012,
    federal_min_wage_ppi = federal_min_wage / ppi_2012,
    state_min_wage_ppi = state_min_wage / ppi_2012
  ) %>%
  dplyr::select(-ppi_2012)


assert_geo_columns(state_minwage_ppi, "state_fips")
write_parquet(
  state_minwage_ppi,
  path_int("state_real_minwages.parquet")
)
