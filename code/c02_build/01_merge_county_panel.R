# Purpose: Merge cleaned source panels onto the balanced county-year backbone.
# Inputs: cleaned C01 artifacts plus upstream wage and price parquets.
# Output: data/intermediate/county_df_build_merge.parquet.
# Run after: all required scripts in code/c01_clean.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
source(path_code("c00_shared", "fips.R"))

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
)
bea_caemp25n_data <- read_parquet(path_int("bea_caemp25n_data_year.parquet"))
bea_cainc45_data <- read_parquet(path_int("bea_cainc45_data_year.parquet"))
fips_codes <- read_csv(
  file = path_raw("geographic_crosswalks", "phil", "fips_codes.csv")
)
fips_codes <- fips_codes %>%
  mutate(fips = state_fips(fips))
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


county_df %>% group_by(year) %>% tally()

# merge in for each side:

# by state and year

# by state

# merge for only one side

# by
# loop ?

# by county and census_period

## a tad of prep ---------------------------------------------------------------

class(county_df$fipscounty)

# make state fips

county_df <- county_df %>%
  mutate(
    fipscounty = county_fips(fipscounty),
    statefips = state_from_county_fips(fipscounty)
  )

hist(as.integer(county_df$statefips)) # it worked!

## both sides merge ----------------------------------------------------------

# fips first

county_df <- merge(
  x = county_df,
  y = fips_codes,
  by.x = "statefips",
  by.y = "fips",
  all.x = T,
  all.y = F
)

county_df <- county_df %>%
  filter(as.integer(statefips) <= 56) # only states

# first side

county_df <- merge(
  x = county_df,
  y = aewr_data,
  by = c("year", "state_abbrev"),
  all.x = T,
  all.y = F
)


# no need to rename these

county_df <- merge(
  x = county_df,
  y = aewr_regions,
  by = c("state_abbrev"),
  all.x = T,
  all.y = F
)


county_df <- county_df %>%
  rename(countyfips = fipscounty) %>%
  left_join(cz_file_small, by = "countyfips")

if (any(is.na(county_df$cz_out10))) {
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
    by = c("year", "countyfips"),
    all.x = T,
    all.y = F
  )
  rm(temp)
}

county_df <- county_df %>%
  left_join(h2a_predict, by = "countyfips")


# county only #

county_df <- merge(
  x = county_df,
  y = census_of_agriculture_cropland_base,
  by = "countyfips",
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

state_min <- state_min %>%
  rename(statefips = fips)

# make a few lags

state_min <- state_min %>%
  arrange(statefips, year) %>%
  group_by(statefips) %>%
  mutate(across(names(state_min)[3:11], lag, .names = "{.col}_l1"))

county_df <- merge(
  x = county_df,
  y = state_min,
  by = c("statefips", "year"),
  all.x = T,
  all.y = F
)

# add in wage quantiles
# annual data starts 2005
cz_wage_quantiles <- cz_wage_quantiles %>%
  rename(year = YEAR) %>%
  mutate(countyfips = county_fips(county_ansi)) %>%
  filter(year >= 2005) %>%
  select(-county_ansi)

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
  group_by(countyfips) %>%
  mutate(across(
    starts_with("wage_p"),
    lag,
    .names = "{.col}_l1",
    order_by = year
  ))

county_df <- merge(
  x = county_df,
  y = cz_wage_quantiles,
  by = c("countyfips", "year", "cz_out10"),
  all.x = T,
  all.y = F
)

# deflate fisher price index to real 2012 terms
# ppi_2012 is already present in county_df via bea_cainc45_data_year merge
county_df <- county_df %>%
  mutate(fisher_index_ppi = fisher_index / ppi_2012)

write_parquet(county_df, path_int("county_df_build_merge.parquet"))
cat(
  "county_df_build_merge:",
  nrow(county_df),
  "rows,",
  ncol(county_df),
  "cols\n"
)
