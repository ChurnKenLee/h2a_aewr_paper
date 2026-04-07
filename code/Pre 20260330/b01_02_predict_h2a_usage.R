rm(list = ls())
library(here)
library(arrow)
library(tidyverse)
library(tlverse)
library(tidycensus)
library(usmap)

#### Prediction target ####
h2a <- read_parquet(here("binaries", "h2a_aggregated.parquet"))
bea <- read_parquet(here("Data Int", "bea_caemp25n_data.parquet"))
counties <- read_parquet(here("Data Int", "county_df_year.parquet"))

# Merge. Restrict year and geography
h2a <- h2a %>%
  mutate(county_ansi = as.integer(paste0(state_fips_code, county_fips_code)))

bea <- bea %>%
  rename(county_ansi = countyfips)

h2a_share <- counties %>%
  rename(county_ansi = fipscounty) %>%
  mutate(
    census_period = ifelse(
      year > 2002 & year <= 2007,
      2007,
      ifelse(
        year > 2007 & year <= 2012,
        2012,
        ifelse(
          year > 2012 & year <= 2017,
          2017,
          ifelse(year > 2017 & year <= 2022, 2022, NA)
        )
      )
    )
  )

h2a_share <- h2a_share %>%
  full_join(
    bea,
    by = c("census_period", "county_ansi")
  )

h2a_share <- h2a_share %>%
  full_join(
    h2a,
    by = c("year", "county_ansi")
  )

# Drop Alaska, Hawaii, DC, territories
h2a_share <- h2a_share %>%
  filter(
    county_ansi < 2000 | county_ansi > 2999, # Alaska
    county_ansi < 11000 | county_ansi > 11999, # DC
    county_ansi < 15000 | county_ansi > 15999, # Hawaii
    county_ansi < 57000 # Territories
  )

# Prediction on 2008-2011
# Independent cities in VA don't have BEA farm emp
h2a_sample <- h2a_share %>%
  filter(year >= 2008 & year <= 2011) %>%
  filter(!is.na(emp_farm)) %>%
  mutate(
    nbr_workers_certified_start_year = replace_na(
      nbr_workers_certified_start_year,
      0
    )
  ) %>%
  select(year, county_ansi, nbr_workers_certified_start_year, emp_farm)

#### Prediction inputs ####
noaa <- read_parquet(here(
  "binaries",
  "county_h2a_prediction_climate_gdd_annual.parquet"
))
soil <- read_parquet(here(
  "binaries",
  "county_h2a_prediction_soil_shares.parquet"
))
slope <- read_parquet(here(
  "binaries",
  "county_h2a_prediction_ssurgo_slopes.parquet"
))

# Reshape soil and slope to be one row per county (these are static)
slope_sample <- slope %>%
  select(areasymbol, mean_slope_pct, steep_share)

test <- soil %>%
  distinct(taxorder, taxsuborder, taxgrtgroup, compname)

soil_sample <- soil %>%
  unite(soil_tax, taxorder, taxsuborder, taxgrtgroup, compname, na.rm = TRUE)
soil_sample <- soil_sample %>%
  pivot_wider(
    id_cols = areasymbol,
    names_from = soil_tax,
    values_from = soil_share,
    values_fill = 0
  )

# Add ANSI to slope and soil data
slope <- slope %>%
  mutate(state_abbrev = substr(areasymbol, 0, 2))

soil <- soil %>%
  mutate(state_abbrev = substr(areasymbol, 0, 2))

state_abbrev <- slope %>%
  distinct(state_abbrev) %>%
  mutate(state_ansi = fips(state_abbrev))
