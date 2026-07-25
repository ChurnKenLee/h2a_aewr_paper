# Purpose: Merge cleaned source panels onto the balanced county-year backbone.
# Inputs: cleaned C01 artifacts plus upstream wage and price parquets.
# Output: data/intermediate/county_df_build_merge.parquet.
# Run after: all required scripts in code/c01_clean.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))

## H2A Build Dataset
## Phil Hoxie
## 1/31/24
library(tidyverse)
library(arrow)
library(tidylog, warn.conflicts = FALSE)

## Yearly Full County Dataset ------------------------------------------------

## ------- Full County ----------------------------------------------------------------------
## Yearly Dataset ## -----------------------------------------------------------
## -------- Full County -----------------------------------------------------------

## Load Data -------------------------------------------------------------------

# yearly versions #
aewr_data <- read_parquet(path_int("aewr_data_year.parquet"))
aewr_regions <- read_csv(
  file = path_raw("geographic_crosswalks", "phil", "aewr_regions.csv")
) %>%
  rename(aewr_region_id = aewr_region_num) %>%
  mutate(aewr_region_id = aewr_region_id(aewr_region_id))
assert_geo_columns(aewr_regions, "aewr_region_id")
bea_caemp25n_data <- read_parquet(path_int("bea_caemp25n_data_year.parquet"))
bea_cainc45_data <- read_parquet(path_int("bea_cainc45_data_year.parquet"))
fips_codes <- read_csv(
  file = path_raw("geographic_crosswalks", "phil", "fips_codes.csv")
)
fips_codes <- fips_codes %>%
  transmute(
    state_fips = state_fips(fips),
    across(-fips)
  )
h2a_data <- read_parquet(path_int("h2a_data_year.parquet"))
h2a_predict <- read_parquet(path_int("h2a_predict.parquet"))
census_of_agriculture_cropland <- read_parquet(path_int(
  "census_ag_cropland_year.parquet"
))

census_pop_ests <- read_parquet(path_int("census_pop_ests_year.parquet"))

census_of_agriculture_cropland_base <- read_parquet(path_int(
  "census_ag_cropland_2007_year.parquet"
))

state_min <- read_parquet(path_int("state_real_minwages.parquet"))

cz_wage_quantiles <- read_parquet(path_int(
  "acs_czone_wage_quantiles.parquet"
))

ppi_annual <- read_parquet(path_int("ppi_2012.parquet"))

nass_price_index <- read_parquet(path_int("nass_fisher_price_index.parquet"))

# base for full county dataset

county_df <- read_parquet(path_int("county_df_year.parquet"))

# CZ

cz_file_small <- read_parquet(path_int("cz_file_2010_small.parquet"))

assert_geo_columns(aewr_data, "state_fips")
assert_geo_columns(bea_caemp25n_data, "county_fips")
assert_geo_columns(bea_cainc45_data, "county_fips")
assert_geo_columns(h2a_data, "county_fips")
assert_geo_columns(h2a_predict, "county_fips")
assert_geo_columns(census_of_agriculture_cropland, "county_fips")
assert_geo_columns(census_pop_ests, "county_fips")
assert_geo_columns(
  census_of_agriculture_cropland_base,
  "county_fips"
)
assert_geo_columns(state_min, "state_fips")
assert_geo_columns(
  cz_wage_quantiles,
  c("county_fips", "cz_id")
)
assert_geo_columns(county_df, "county_fips")
assert_geo_columns(cz_file_small, c("county_fips", "cz_id"))

county_df %>% group_by(year) %>% tally()

# merge in for each side:

# by state and year

# by state

# merge for only one side

# by
# loop ?

# by county and census_period

## a tad of prep ---------------------------------------------------------------

class(county_df$county_fips)

# make state fips

county_df <- county_df %>%
  mutate(
    state_fips = state_from_county_fips(county_fips)
  )

hist(as.integer(county_df$state_fips)) # it worked!

## both sides merge ----------------------------------------------------------

# fips first

county_df <- merge(
  x = county_df,
  y = fips_codes,
  by = "state_fips",
  all.x = T,
  all.y = F
)

non_aewr_states <- county_df %>%
  distinct(state_abbrev) %>%
  anti_join(
    aewr_regions %>% distinct(state_abbrev),
    by = "state_abbrev"
  ) %>%
  pull(state_abbrev)

expected_non_aewr_states <- c(
  "AK",
  "AS",
  "DC",
  "GU",
  "HI",
  "MP",
  "PR",
  "VI"
)
unexpected_non_aewr_states <- setdiff(
  non_aewr_states,
  expected_non_aewr_states
)
if (length(unexpected_non_aewr_states) > 0L) {
  stop(
    "States unexpectedly missing from the AEWR-region crosswalk: ",
    paste(sort(unexpected_non_aewr_states), collapse = ", "),
    call. = FALSE
  )
}
message(
  "Restricting the county panel to states covered by AEWR regions; ",
  "excluding ",
  paste(sort(non_aewr_states), collapse = ", "),
  "."
)
county_df <- county_df %>%
  semi_join(
    aewr_regions %>% distinct(state_abbrev),
    by = "state_abbrev"
  )

# first side

county_df <- merge(
  x = county_df,
  y = aewr_data,
  by = c("year", "state_fips", "state_abbrev"),
  all.x = T,
  all.y = F
)


# no need to rename these

county_df <- merge(
  x = county_df,
  y = aewr_regions,
  by = c("state_abbrev"),
  all.x = F,
  all.y = F
)


county_df <- county_df %>%
  left_join(cz_file_small, by = "county_fips")

if (any(is.na(county_df$cz_id))) {
  stop("Some panel counties do not match the Penn 2010 county-to-CZ file.")
}

datasets <- c(
  "bea_caemp25n_data",
  "bea_cainc45_data",
  "h2a_data",
  "census_pop_ests",
  "census_of_agriculture_cropland",
  "nass_price_index"
)

for (i in 1:length(datasets)) {
  print(paste0("Rep ", i))
  temp <- get(datasets[i])
  print(dim(county_df))
  print(dim(temp))
  county_df <- merge(
    x = county_df,
    y = temp,
    by = c("year", "county_fips"),
    all.x = T,
    all.y = F
  )
  rm(temp)
}

county_df <- county_df %>%
  left_join(h2a_predict, by = "county_fips")


# county only #

county_df <- merge(
  x = county_df,
  y = census_of_agriculture_cropland_base,
  by = "county_fips",
  all.x = T,
  all.y = F
)

county_df %>%
  group_by(year) %>%
  tally()

county_df %>%
  filter(year == 2008 & !is.na(cropland_acr_2007)) %>%
  count()

# add in minimum wages

# make a few lags

state_min <- state_min %>%
  arrange(state_fips, year) %>%
  group_by(state_fips) %>%
  mutate(across(names(state_min)[3:11], lag, .names = "{.col}_l1"))

county_df <- merge(
  x = county_df,
  y = state_min,
  by = c("state_fips", "year"),
  all.x = T,
  all.y = F
)

# add in wage quantiles
# annual data starts 2005
cz_wage_quantiles <- cz_wage_quantiles %>%
  rename(year = YEAR) %>%
  filter(year >= 2005)

# Deflate wage percentiles to real 2012 terms using PPI WPU01 (rebased 2012=100)
# aewr_ppi is real; wage_p* must also be real before computing bite variables
cz_wage_quantiles <- cz_wage_quantiles %>%
  left_join(ppi_annual, by = "year") %>%
  mutate(across(
    c(wage_p10, wage_p25, wage_p50, wage_p75, wage_p90),
    ~ . / ppi_2012
  )) %>%
  select(-ppi_2012)

# Add lags as in state minimum wages
cz_wage_quantiles <- cz_wage_quantiles %>%
  group_by(county_fips) %>%
  mutate(across(
    starts_with("wage_p"),
    ~ lag(.x, order_by = year),
    .names = "{.col}_l1"
  ))

county_df <- merge(
  x = county_df,
  y = cz_wage_quantiles,
  by = c("county_fips", "year", "cz_id"),
  all.x = T,
  all.y = F
)

# Deflate the Fisher price index to real 2012 terms. The companion Fisher
# quantity index is unit-free and stays normalized to 2011 = 100.
# ppi_2012 is already present in county_df via bea_cainc45_data_year merge
county_df <- county_df %>%
  mutate(fisher_index_ppi = fisher_index / ppi_2012)

assert_geo_columns(
  county_df,
  c("state_fips", "county_fips", "cz_id", "aewr_region_id")
)
write_parquet(county_df, path_int("county_df_build_merge.parquet"))
cat(
  "county_df_build_merge:",
  nrow(county_df),
  "rows,",
  ncol(county_df),
  "cols\n"
)
