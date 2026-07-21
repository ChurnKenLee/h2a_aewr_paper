# Purpose: Derive analysis variables, real measures, shares, fixed effects, and lags.
# Input: county_df_build_merge.parquet.
# Output: data/intermediate/county_df_variable_cleaned_year.parquet.
# Run after: 01_merge_county_panel.R.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
source(path_code("c00_shared", "fips.R"))
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)

county_df <- read_parquet(path_int("county_df_build_merge.parquet"))

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

# H2A and cropland NAs to zero

h2a_zero_vars <- c(
  "nbr_workers_requested_all_years",
  "nbr_workers_certified_all_years",
  "man_hours_requested_all_years",
  "man_hours_certified_all_years",
  "nbr_applications_all_years",
  "nbr_workers_requested_start_year",
  "nbr_workers_certified_start_year",
  "man_hours_requested_start_year",
  "man_hours_certified_start_year",
  "nbr_applications_start_year",
  "nbr_workers_requested_fiscal_year",
  "nbr_workers_certified_fiscal_year",
  "man_hours_requested_fiscal_year",
  "man_hours_certified_fiscal_year",
  "nbr_applications_fiscal_year"
)

county_df <- county_df %>%
  mutate(
    across(all_of(h2a_zero_vars), ~ replace_na(.x, 0)),
    across(c(cropland_acr, cropland_acr_2007), ~ replace_na(.x, 0))
  )


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

h2a_lag_vars <- c(
  "nbr_workers_requested_all_years",
  "nbr_workers_certified_all_years",
  "man_hours_requested_all_years",
  "man_hours_certified_all_years",
  "nbr_applications_all_years",
  "nbr_workers_requested_start_year",
  "nbr_workers_certified_start_year",
  "man_hours_requested_start_year",
  "man_hours_certified_start_year",
  "nbr_applications_start_year",
  "nbr_workers_requested_fiscal_year",
  "nbr_workers_certified_fiscal_year",
  "man_hours_requested_fiscal_year",
  "man_hours_certified_fiscal_year",
  "nbr_applications_fiscal_year"
)

county_df <- county_df %>%
  group_by(county_fe) %>%
  arrange(year, .by_group = TRUE) %>%
  mutate(
    across(
      all_of(h2a_lag_vars),
      list(l1 = ~ lag(.x, n = 1), l2 = ~ lag(.x, n = 2)),
      .names = "{.col}_{.fn}"
    )
  ) %>%
  ungroup()


write_parquet(county_df, path_int("county_df_variable_cleaned_year.parquet"))
