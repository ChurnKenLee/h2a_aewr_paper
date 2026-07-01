## H2A Build Dataset
## Phil Hoxie
## 1/31/24
rm(list = ls())
if (!exists("path_code", mode = "function")) {
  source(file.path("code", "paths.R"))
}
ensure_project_dirs()
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
aewr_regions <- read.csv(
  file = path_raw("geographic_crosswalks", "phil", "aewr_regions.csv"),
  stringsAsFactors = F
)
bea_caemp25n_data <- read_parquet(path_int("bea_caemp25n_data_year.parquet"))
bea_cainc45_data <- read_parquet(path_int("bea_cainc45_data_year.parquet"))
fips_codes <- read.csv(
  file = path_raw("geographic_crosswalks", "phil", "fips_codes.csv"),
  stringsAsFactors = F
)
h2a_data <- read_parquet(path_int("h2a_data_year.parquet"))
h2a_predict <- read_parquet(path_int("h2a_predict.parquet"))
census_of_agriculture_cropland <- read_parquet(path_int("census_ag_cropland_year.parquet"))

census_pop_ests <- read_parquet(path_int("census_pop_ests_year.parquet"))

census_of_agriculture_cropland_base <- read_parquet(path_int("census_ag_cropland_2007_year.parquet"))

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

head(county_df)

county_df %>% group_by(year) %>% tally()

# merge in for each side:

# by state and year
head(aewr_data)

# by state
head(aewr_regions)

# merge for only one side

# by
# loop ?

# by county and census_period

head(bea_caemp25n_data)
head(bea_cainc45_data)
head(h2a_data)
head(census_pop_ests)

## a tad of prep ---------------------------------------------------------------

class(county_df$fipscounty)

# make state fips

county_df <- county_df %>%
  mutate(fipscounty = as.numeric(fipscounty)) %>%
  mutate(statefips = floor(fipscounty / 1000))

hist(county_df$statefips) # it worked!

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
  filter(statefips <= 56) # only states

# first side

names(aewr_data)

dim(county_df)

county_df <- merge(
  x = county_df,
  y = aewr_data,
  by = c("year", "state_abbrev"),
  all.x = T,
  all.y = F
)

dim(county_df) # DC missing

head(county_df)
# no need to rename these

names(aewr_regions)
dim(county_df)

county_df <- merge(
  x = county_df,
  y = aewr_regions,
  by = c("state_abbrev"),
  all.x = T,
  all.y = F
)

dim(county_df)

county_df <- county_df %>%
  rename(countyfips = fipscounty)

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

dim(county_df)
head(county_df)

# county only #

dim(census_of_agriculture_cropland_base)

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

head(state_min)

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
  rename(year = YEAR, cz_1990 = czone) %>%
  mutate(countyfips = as.numeric(county_ansi)) %>%
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
    names(cz_wage_quantiles)[3:7],
    lag,
    .names = "{.col}_l1",
    order_by = year
  ))

county_df <- merge(
  x = county_df,
  y = cz_wage_quantiles,
  by = c("countyfips", "year"),
  all.x = T,
  all.y = F
)

# deflate fisher price index to real 2012 terms
# ppi_2012 is already present in county_df via bea_cainc45_data_year merge
county_df <- county_df %>%
  mutate(fisher_index_ppi = fisher_index / ppi_2012)

## Variable cleaning ## -----------------

# AEWR vs ag min diff #

county_df <- county_df %>%
  mutate(
    aewr_state_ag_ppi_l1 = aewr_ppi_l1 - prevailing_ag_min_wage_ppi_l1,
    aewr_state_ag_ppi = aewr_ppi - prevailing_ag_min_wage_ppi,
    # new CZ-distribution-based bites (non-lagged):
    aewr_cz_p10 = aewr_ppi - wage_p10,
    aewr_cz_p25 = aewr_ppi - wage_p25,
    aewr_cz_p50 = aewr_ppi - wage_p50,
    # lagged CZ bites:
    aewr_cz_p10_l1 = aewr_ppi_l1 - wage_p10_l1,
    aewr_cz_p25_l1 = aewr_ppi_l1 - wage_p25_l1,
    aewr_cz_p50_l1 = aewr_ppi_l1 - wage_p50_l1
  )

summary(county_df$aewr_ppi_l1)
summary(county_df$aewr_cz_p10_l1)

check <- county_df %>%
  filter(is.na(wage_p10_l1)) %>%
  group_by(state_abbrev, countyfips, countyname) %>%
  tally()

county_df <- county_df %>%
  mutate(
    ln_aewr_state_ag_ppi = log(aewr_state_ag_ppi),
    ln_aewr_state_ag_ppi_l1 = log(aewr_state_ag_ppi_l1)
  )

test <- county_df %>%
  filter(is.na(ln_aewr_state_ag_ppi))

# AEWR vs state min diff #

# H2A NAs to zero

county_df$nbr_workers_requested_all_years[is.na(
  county_df$nbr_workers_requested_all_years
)] <- 0
county_df$nbr_workers_certified_all_years[is.na(
  county_df$nbr_workers_certified_all_years
)] <- 0
county_df$man_hours_requested_all_years[is.na(
  county_df$man_hours_requested_all_years
)] <- 0
county_df$man_hours_certified_all_years[is.na(
  county_df$man_hours_certified_all_years
)] <- 0
county_df$nbr_applications_all_years[is.na(
  county_df$nbr_applications_all_years
)] <- 0

county_df$nbr_workers_requested_start_year[is.na(
  county_df$nbr_workers_requested_start_year
)] <- 0
county_df$nbr_workers_certified_start_year[is.na(
  county_df$nbr_workers_certified_start_year
)] <- 0
county_df$man_hours_requested_start_year[is.na(
  county_df$man_hours_requested_start_year
)] <- 0
county_df$man_hours_certified_start_year[is.na(
  county_df$man_hours_certified_start_year
)] <- 0
county_df$nbr_applications_start_year[is.na(
  county_df$nbr_applications_start_year
)] <- 0

county_df$nbr_workers_requested_fiscal_year[is.na(
  county_df$nbr_workers_requested_fiscal_year
)] <- 0
county_df$nbr_workers_certified_fiscal_year[is.na(
  county_df$nbr_workers_certified_fiscal_year
)] <- 0
county_df$man_hours_requested_fiscal_year[is.na(
  county_df$man_hours_requested_fiscal_year
)] <- 0
county_df$man_hours_certified_fiscal_year[is.na(
  county_df$man_hours_certified_fiscal_year
)] <- 0
county_df$nbr_applications_fiscal_year[is.na(
  county_df$nbr_applications_fiscal_year
)] <- 0

head(county_df)

# cropland zeros

county_df$cropland_acr[is.na(county_df$cropland_acr)] <- 0
county_df$cropland_acr_2007[is.na(county_df$cropland_acr_2007)] <- 0

# emp pop ratio

county_df <- county_df %>%
  mutate(emp_pop_ratio = emp_tot / pop_census)

# logs of some vars

county_df <- county_df %>%
  mutate(
    ln_aewr = log(aewr),
    ln_aewr_ppi = log(aewr_ppi),
    ln_aewr_l1 = log(aewr_l1),
    ln_aewr_ppi_l1 = log(aewr_ppi_l1),
    ln_aewr_l2 = log(aewr_l2),
    ln_aewr_ppi_l2 = log(aewr_ppi_l2),
    ln_aewr_cz_p10 = log(aewr_cz_p10),
    ln_aewr_cz_p25 = log(aewr_cz_p25),
    ln_aewr_cz_p50 = log(aewr_cz_p50),
    ln_aewr_cz_p10_l1 = log(aewr_cz_p10_l1),
    ln_aewr_cz_p25_l1 = log(aewr_cz_p25_l1),
    ln_aewr_cz_p50_l1 = log(aewr_cz_p50_l1),
    ln_emp_farm_propr = log(emp_farm_propr),
    ln_emp_farm = log(emp_farm),
    ln_emp_nonfarm = log(emp_nonfarm),
    ln_emp_privatenonfarm = log(emp_privatenonfarm),
    ln_pop_census = log(pop_census),
    ln_cropland_acr = log(cropland_acr),
    ln_cropland_acr_2007 = log(cropland_acr_2007),
    ln_nbr_workers_requested_start_years = log(
      nbr_workers_requested_start_year
    ),
    ln_nbr_workers_requested_all_years = log(nbr_workers_requested_all_years),
    ln_nbr_applications_fiscal_year = log(nbr_applications_fiscal_year),
    ln1_nbr_workers_requested_start_years = log(
      nbr_workers_requested_start_year + 1
    ),
    ln1_nbr_workers_requested_all_years = log(
      nbr_workers_requested_all_years + 1
    ),
    ln1_nbr_applications_fiscal_year = log(nbr_applications_fiscal_year + 1)
  )

# budget shares

county_df <- county_df %>%
  mutate(
    share_farm_laborexp_prodexp = farm_laborexpense / farm_prodexp,
    share_farm_crop_cashandinc = farm_cashcrops / farm_cashandinc,
    share_farm_animal_cashandinc = farm_cashanimal / farm_cashandinc,
    share_farm_govt_cashandinc = farm_govpayments / farm_cashandinc
  )

# H2A outcome variables

county_df_farmempbase <- county_df %>%
  filter(year == 2011) %>%
  select(countyfips, emp_farm) %>%
  rename(emp_farm_2011 = emp_farm)

county_df <- merge(
  x = county_df,
  y = county_df_farmempbase,
  by = "countyfips",
  all.x = T,
  all.y = F
)

county_df <- county_df %>%
  mutate(
    h2a_req_share_farm_workers_start_year = nbr_workers_requested_start_year /
      emp_farm,
    h2a_cert_share_farm_workers_start_year = nbr_workers_certified_start_year /
      emp_farm,
    h2a_req_share_farm_workers_2011_start_year = nbr_workers_requested_start_year /
      emp_farm_2011,
    h2a_cert_share_farm_workers_2011_start_year = nbr_workers_certified_start_year /
      emp_farm_2011,
    h2a_predicted_share_2011 = predicted_h2a_count / emp_farm_2011
  ) # add h2a apps per farm later

county_df$h2a_req_share_farm_workers_start_year[is.infinite(
  county_df$h2a_req_share_farm_workers_start_year
)] <- NA
county_df$h2a_cert_share_farm_workers_start_year[is.infinite(
  county_df$h2a_cert_share_farm_workers_start_year
)] <- NA

county_df$h2a_req_share_farm_workers_2011_start_year[is.infinite(
  county_df$h2a_req_share_farm_workers_2011_start_year
)] <- NA
county_df$h2a_cert_share_farm_workers_2011_start_year[is.infinite(
  county_df$h2a_cert_share_farm_workers_2011_start_year
)] <- NA

county_df$h2a_predicted_share_2011[is.infinite(
  county_df$h2a_predicted_share_2011
)] <- NA

# add in CZs

county_df <- merge(
  x = county_df,
  y = cz_file_small,
  by = "countyfips",
  all.x = T,
  all.y = F
)

# fixed effects

# period

county_df$year_fe <- with(county_df, as.factor(year)) # treats counties in separate clusters as separate

# county

county_df$county_fe <- with(county_df, as.factor(countyfips)) # treats counties in separate clusters as separate

# state

county_df$state_fe <- with(county_df, as.factor(state_abbrev)) # treats counties in separate clusters as separate

# AEWR Region

county_df$aewr_region_fe <- with(county_df, as.factor(aewr_region_num)) # treats counties in separate clusters as separate

# cZ
county_df$cz_fe <- with(county_df, as.factor(cz_out10)) # treats counties in separate clusters as separate


# CZ x time

county_df$cztime_fe <- with(county_df, interaction(as.factor(cz_out10), year))

levels(county_df$cztime_fe) <- c(levels(county_df$cztime_fe), "0.0")

unique(county_df$year)

county_df$cztime_fe[county_df$year == 2008] <- "0.0" # first period is 0

# aewr region  x time

county_df$aewrregtime_fe <- with(
  county_df,
  interaction(as.factor(aewr_region_num), year)
)

levels(county_df$aewrregtime_fe) <- c(levels(county_df$aewrregtime_fe), "0.0")

unique(county_df$year)

county_df$aewrregtime_fe[county_df$year == 2008] <- "0.0" # first period is 0

# CZ x AEWR region FE — each (CZ, AEWR region) pair is a distinct FE level.
# CZs that span AEWR region borders are split: counties on each side
# get separate levels, matching the clustering unit for the main regressions.

county_df$cz_aewr_region_fe <- with(
  county_df,
  interaction(as.factor(cz_out10), as.factor(aewr_region_num))
)


# sample restriction to counties with cropland ------------------------------------

# we want to drop counties with no cropland in 2007

county_df <- county_df %>%
  mutate(
    any_cropland_2007 = ifelse(
      cropland_acr_2007 != 0 & !is.na(cropland_acr_2007),
      1,
      0
    )
  )

county_df %>%
  group_by(any_cropland_2007, year) %>%
  tally()

# remove HI, AK, and DC

county_df <- county_df %>%
  filter(!is.na(aewr) & !is.na(aewr_region_num))

# check for NAs in key vars ------------------------------------

# use sum(is.na(x))

NA_df <- NULL

dim(county_df)

for (i in 1:length(names(county_df))) {
  temp_na_cnt <- sum(is.na(county_df[, i]))
  temp_varname <- paste0(names(county_df)[i])

  temp <- data.frame(var = temp_varname, full_na_cnt = temp_na_cnt)

  NA_df <- bind_rows(NA_df, temp)
  rm(temp, temp_na_cnt, temp_varname)
}

# easiest / most important to fix
fixcounty <- county_df %>%
  filter(is.na(pop_census))

fixcounty %>%
  group_by(state_abbrev) %>%
  tally() # BEA issue for VA counties. We need to drop bristol city. It will be ok.

## lags of h2a variables

county_df <- county_df %>%
  arrange(county_fe, year) %>%
  group_by(county_fe) %>%
  mutate(
    nbr_workers_requested_all_years_l1 = lag(
      nbr_workers_requested_all_years,
      n = 1,
      order_by = county_fe
    ),
    nbr_workers_requested_all_years_l2 = lag(
      nbr_workers_requested_all_years,
      n = 2,
      order_by = county_fe
    ),
    nbr_workers_certified_all_years_l1 = lag(
      nbr_workers_certified_all_years,
      n = 1,
      order_by = county_fe
    ),
    nbr_workers_certified_all_years_l2 = lag(
      nbr_workers_certified_all_years,
      n = 2,
      order_by = county_fe
    ),
    man_hours_requested_all_years_l1 = lag(
      man_hours_requested_all_years,
      n = 1,
      order_by = county_fe
    ),
    man_hours_requested_all_years_l2 = lag(
      man_hours_requested_all_years,
      n = 2,
      order_by = county_fe
    ),
    man_hours_certified_all_years_l1 = lag(
      man_hours_certified_all_years,
      n = 1,
      order_by = county_fe
    ),
    man_hours_certified_all_years_l2 = lag(
      man_hours_certified_all_years,
      n = 2,
      order_by = county_fe
    ),
    nbr_applications_all_years_l1 = lag(
      nbr_applications_all_years,
      n = 1,
      order_by = county_fe
    ),
    nbr_applications_all_years_l2 = lag(
      nbr_applications_all_years,
      n = 2,
      order_by = county_fe
    ),
    nbr_workers_requested_start_year_l1 = lag(
      nbr_workers_requested_start_year,
      n = 1,
      order_by = county_fe
    ),
    nbr_workers_requested_start_year_l2 = lag(
      nbr_workers_requested_start_year,
      n = 2,
      order_by = county_fe
    ),
    nbr_workers_certified_start_year_l1 = lag(
      nbr_workers_certified_start_year,
      n = 1,
      order_by = county_fe
    ),
    nbr_workers_certified_start_year_l2 = lag(
      nbr_workers_certified_start_year,
      n = 2,
      order_by = county_fe
    ),
    man_hours_requested_start_year_l1 = lag(
      man_hours_requested_start_year,
      n = 1,
      order_by = county_fe
    ),
    man_hours_requested_start_year_l2 = lag(
      man_hours_requested_start_year,
      n = 2,
      order_by = county_fe
    ),
    man_hours_certified_start_year_l1 = lag(
      man_hours_certified_start_year,
      n = 1,
      order_by = county_fe
    ),
    man_hours_certified_start_year_l2 = lag(
      man_hours_certified_start_year,
      n = 2,
      order_by = county_fe
    ),
    nbr_applications_start_year_l1 = lag(
      nbr_applications_start_year,
      n = 1,
      order_by = county_fe
    ),
    nbr_applications_start_year_l2 = lag(
      nbr_applications_start_year,
      n = 2,
      order_by = county_fe
    ),
    nbr_workers_requested_fiscal_year_l1 = lag(
      nbr_workers_requested_fiscal_year,
      n = 1,
      order_by = county_fe
    ),
    nbr_workers_requested_fiscal_year_l2 = lag(
      nbr_workers_requested_fiscal_year,
      n = 2,
      order_by = county_fe
    ),
    nbr_workers_certified_fiscal_year_l1 = lag(
      nbr_workers_certified_fiscal_year,
      n = 1,
      order_by = county_fe
    ),
    nbr_workers_certified_fiscal_year_l2 = lag(
      nbr_workers_certified_fiscal_year,
      n = 2,
      order_by = county_fe
    ),
    man_hours_requested_fiscal_year_l1 = lag(
      man_hours_requested_fiscal_year,
      n = 1,
      order_by = county_fe
    ),
    man_hours_requested_fiscal_year_l2 = lag(
      man_hours_requested_fiscal_year,
      n = 2,
      order_by = county_fe
    ),
    man_hours_certified_fiscal_year_l1 = lag(
      man_hours_certified_fiscal_year,
      n = 1,
      order_by = county_fe
    ),
    man_hours_certified_fiscal_year_l2 = lag(
      man_hours_certified_fiscal_year,
      n = 2,
      order_by = county_fe
    ),
    nbr_applications_fiscal_year_l1 = lag(
      nbr_applications_fiscal_year,
      n = 1,
      order_by = county_fe
    ),
    nbr_applications_fiscal_year_l2 = lag(
      nbr_applications_fiscal_year,
      n = 2,
      order_by = county_fe
    )
  )

# new variables

names(county_df)


# Cut using pred X actual use rates in 2008
true_share_cutoff <- 0.01
pred_share_cutoff <- 0.01

county_type_classification <- county_df %>%
  filter(year == 2008) %>%
  mutate(
    county_treatment_group_classification = case_when(
      (h2a_predicted_share_2011 > pred_share_cutoff) &
        (h2a_cert_share_farm_workers_2011_start_year >
          true_share_cutoff) ~ "always takers",
      (h2a_predicted_share_2011 > pred_share_cutoff) &
        (h2a_cert_share_farm_workers_2011_start_year <
          true_share_cutoff) ~ "adopters",
      (h2a_predicted_share_2011 < pred_share_cutoff) &
        (h2a_cert_share_farm_workers_2011_start_year >
          true_share_cutoff) ~ "defiers",
      (h2a_predicted_share_2011 < pred_share_cutoff) &
        (h2a_cert_share_farm_workers_2011_start_year <
          true_share_cutoff) ~ "never takers",
    ),
    county_simple_treatment_groups = case_when(
      (county_treatment_group_classification ==
        "always takers") ~ "always takers",
      (county_treatment_group_classification !=
        "always takers") ~ "exposed adoptors"
    )
  ) %>%
  select(
    countyfips,
    county_treatment_group_classification,
    county_simple_treatment_groups
  )


# cuts by 2008 h2a usage
h2a_use_df <- county_df %>%
  ungroup() %>%
  filter(year == 2008)

dim(h2a_use_df)

hist(h2a_use_df$nbr_workers_requested_start_year)
summary(h2a_use_df$nbr_workers_requested_start_year)
summary(h2a_use_df$h2a_req_share_farm_workers_start_year)

# cut by count

count_cuts <- quantile(
  h2a_use_df$nbr_workers_requested_start_year,
  probs = c(.5, .66, .75)
)

# cut by share

share_cuts <- quantile(
  h2a_use_df$h2a_cert_share_farm_workers_start_year,
  probs = c(.5, .66, .75),
  na.rm = T
)


h2a_use_df <- h2a_use_df %>%
  mutate(
    high_h2a_count_50 = ifelse(
      nbr_workers_certified_start_year > count_cuts[1],
      1,
      0
    ),
    high_h2a_count_66 = ifelse(
      nbr_workers_certified_start_year > count_cuts[2],
      1,
      0
    ),
    high_h2a_count_75 = ifelse(
      nbr_workers_certified_start_year > count_cuts[3],
      1,
      0
    ),
    high_h2a_share_50 = ifelse(
      h2a_cert_share_farm_workers_start_year > share_cuts[1] &
        !is.na(h2a_cert_share_farm_workers_start_year),
      1,
      0
    ),
    high_h2a_share_66 = ifelse(
      h2a_cert_share_farm_workers_start_year > share_cuts[2] &
        !is.na(h2a_cert_share_farm_workers_start_year),
      1,
      0
    ),
    high_h2a_share_75 = ifelse(
      h2a_cert_share_farm_workers_start_year > share_cuts[3] &
        !is.na(h2a_cert_share_farm_workers_start_year),
      1,
      0
    )
  )

h2a_use_df %>%
  group_by(high_h2a_count_50) %>%
  tally()

h2a_use_df <- h2a_use_df %>%
  select(
    countyfips,
    high_h2a_count_50,
    high_h2a_count_66,
    high_h2a_count_75,
    high_h2a_share_50,
    high_h2a_share_66,
    high_h2a_share_75
  )

head(h2a_use_df)

dim(h2a_use_df)

county_df <- merge(
  x = county_df,
  y = h2a_use_df,
  by = "countyfips",
  all.x = T,
  all.y = F
)

county_type_classification <- county_type_classification %>%
  ungroup() %>%
  select(-county_fe)

county_df <- county_df %>%
  left_join(county_type_classification, by = "countyfips")

names(county_df) #check
# year dummys

summary(county_df$year)

county_df <- county_df %>%
  mutate(
    yeardummy_2008 = ifelse(year == 2008, 1, 0),
    yeardummy_2009 = ifelse(year == 2009, 1, 0),
    yeardummy_2010 = ifelse(year == 2010, 1, 0),
    yeardummy_2011 = ifelse(year == 2011, 1, 0),
    yeardummy_2012 = ifelse(year == 2012, 1, 0),
    yeardummy_2013 = ifelse(year == 2013, 1, 0),
    yeardummy_2014 = ifelse(year == 2014, 1, 0),
    yeardummy_2015 = ifelse(year == 2015, 1, 0),
    yeardummy_2016 = ifelse(year == 2016, 1, 0),
    yeardummy_2017 = ifelse(year == 2017, 1, 0),
    yeardummy_2018 = ifelse(year == 2018, 1, 0),
    yeardummy_2019 = ifelse(year == 2019, 1, 0),
    yeardummy_2020 = ifelse(year == 2020, 1, 0),
    yeardummy_2021 = ifelse(year == 2021, 1, 0),
    yeardummy_2022 = ifelse(year == 2022, 1, 0)
  )


# ID border CZs

cz_borders <- county_df %>%
  group_by(cz_out10) %>%
  summarise(
    AEWRregmin = min(aewr_region_num, na.rm = T),
    AEWRregmax = max(aewr_region_num, na.rm = T)
  )

cz_borders <- cz_borders %>%
  mutate(border_cz = ifelse(AEWRregmin != AEWRregmax, 1, 0))

cz_borders %>% group_by(border_cz) %>% tally()

county_df <- merge(
  x = county_df,
  y = cz_borders,
  by = "cz_out10",
  all.x = T,
  all.y = F
)

names(county_df) #check


# pre post dummy

county_df <- county_df %>%
  mutate(postdummy = ifelse(year > 2011, 1, 0))

# low use dummy

county_df <- county_df %>%
  mutate(
    high_h2a_share_75_inverse = ifelse(high_h2a_share_75 == 0, 1, 0),
    high_h2a_share_66_inverse = ifelse(high_h2a_share_66 == 0, 1, 0),
    high_h2a_share_50_inverse = ifelse(high_h2a_share_50 == 0, 1, 0),
    high_h2a_count_75_inverse = ifelse(high_h2a_count_75 == 0, 1, 0),
    high_h2a_count_66_inverse = ifelse(high_h2a_count_66 == 0, 1, 0),
    high_h2a_count_50_inverse = ifelse(high_h2a_count_50 == 0, 1, 0)
  )

county_df %>%
  write_parquet(path_processed("county_df_analysis_year.parquet"))

# --- Diagnostic: county_df_analysis_year ---
cat("county_df rows:", nrow(county_df), " | cols:", ncol(county_df), "\n")
stopifnot(nrow(county_df) > 10000)
stopifnot(all(
  c(
    "aewr_state_ag_ppi_l1",
    "high_h2a_share_75",
    "high_h2a_share_75_inverse",
    "h2a_cert_share_farm_workers_2011_start_year",
    "county_fe",
    "year_fe",
    "statefips",
    "ln_pop_census",
    "emp_pop_ratio"
  ) %in%
    names(county_df)
))
cat("Diagnostic passed: county_df_analysis_year\n")

## Remove files ## -------------------------------------------------------------

str_detect(ls(), "folder_")

objects <- data.frame(name = ls(), keep = str_detect(ls(), "folder_")) %>%
  filter(keep == F)

rm(list = objects[, 1])
gc()
rm(objects)
