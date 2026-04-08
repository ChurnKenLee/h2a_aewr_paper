# Harmonize county ANSI from disparate sources to 2010 standard
# List of sources we have as inputs:
# H-2A usage aggregated and predicted usage
# NASS quickstats
# OEWS
# QCEW
# CroplandCROS CDL
# Predicted H-2A usage
# ACS immigrant imputation
# ACS wage quantiles
# BEA farm employment
# NAWSPAD

library(here)
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)
library(foreign)
library(haven)

rm(list = ls())

h2a <- read_parquet(here("binaries", "h2a_aggregated.parquet"))
h2a_predict <- read_parquet(here(
    "binaries",
    "h2a_prediction_using_elastic_net.parquet"
))
cz_wage_quantiles <- read_parquet(here(
    "binaries",
    "acs_czone_wage_quantiles.parquet"
))
qs_census <- read_parquet(here(
    "binaries",
    "qs_census_selected_obs.parquet"
))
cdl_acreage <- read_parquet(here(
    "binaries",
    "croplandcros_county_crop_acres.parquet"
))
acs_qcew <- read_parquet(here("binaries", "acs_qcew.parquet"))
oews_agg <- read_parquet(here("binaries", "oews_county_aggregated.parquet"))
acs_immigrant_imputed <- read_parquet(here(
    "binaries",
    "acs_immigrant_imputed.parquet"
))
bea_farm_nonfarm <- read_parquet(here(
    "binaries",
    "bea_farm_nonfarm_emp.parquet"
))

#### Adjust BEA FIPS to match Phil's 2010 county crosswalk ####
# Construct FIPS crosswalk following Phil
# This is just adjusting for BEA's Virginia coding
bea_fips_xwalk <- read_csv(
    here("Data Int", "bea_fips_xwalk.csv")
)
full_county_set <- read_csv(
    here("Data Int", "county_adjacency2010.csv")
)
county_list <- unique(select(full_county_set, fipscounty, countyname)) %>% # all county fips and names
    mutate(indata = 1)
bea_fips_xwalk <- merge(
    x = bea_fips_xwalk,
    y = county_list,
    by.x = "realfips",
    by.y = "fipscounty",
    all.x = T,
    all.y = F
)

# Keep counties when conflict
bea_fips_xwalk <- bea_fips_xwalk %>%
    filter(county == 1) %>%
    select(realfips, beafips) %>%
    mutate(across(everything(), as.character))

bea_harmonized <- merge(
    x = bea_farm_nonfarm,
    y = bea_fips_xwalk,
    by.x = "county_fips",
    by.y = "beafips",
    all.x = T,
    all.y = F
)

# Fix Virginia FIPS
bea_harmonized <- bea_harmonized %>%
    rename(oldfips = county_fips) %>%
    mutate(county_fips = ifelse(!is.na(realfips), realfips, oldfips))

bea_harmonized <- bea_harmonized %>%
    select(-oldfips, -realfips)

# South Dakota Oglala Lakota county to Shannon county
bea_harmonized <- bea_harmonized %>%
    mutate(
        county_fips = case_when(
            county_fips == "46102" ~ "46113",
            .default = county_fips
        )
    )
# Save harmonized BEA emp
bea_harmonized %>%
    write_parquet(here("binaries", "bea_farm_nonfarm_emp.parquet")) %>%
    write_parquet(here("Data Int", "bea_farm_nonfarm_emp.parquet"))
test <- bea_harmonized %>% filter(county_fips == "46102")

#### 2010 county spine ####
county_list_unique <- unique(select(full_county_set, fipscounty, countyname))
years <- seq(2008, 2022)
county_df <- NULL
for (i in 1:length(years)) {
    temp <- county_list_unique %>%
        mutate(year = years[i])
    county_df <- bind_rows(county_df, temp)
    rm(temp)
}
county_df %>%
    group_by(year) %>%
    tally()

#### Check H-2A county mapping
county_vec <- county_list_unique %>% pull(fipscounty) %>% as.vector()

# H-2A data
# SD Oglala Lakota to Shannon county
h2a <- h2a %>%
    mutate(
        county_fips_code = case_when(
            state_fips_code == "46" & county_fips_code == "102" ~ "113",
            .default = county_fips_code
        )
    )
# Check
h2a <- h2a %>%
    mutate(county_ansi = paste0(state_fips_code, county_fips_code))
unmatched_h2a_predict <- h2a_predict %>% filter(!county_ansi %in% county_vec)
unmatched_h2a <- h2a %>% filter(!county_ansi %in% county_vec)
# Only US aggregate not matched, match ok

# Save H-2A data
h2a %>%
    write_parquet(here("binaries", "h2a_aggregated.parquet")) %>%
    write_parquet(here("Data Int", "h2a_aggregated.parquet"))
h2a_predict %>%
    write_parquet(here(
        "binaries",
        "h2a_prediction_using_elastic_net.parquet"
    )) %>%
    write_parquet(here("Data Int", "h2a_prediction_using_elastic_net.parquet"))

#### Commuting Zone wage quantiles ####
unmatched_cz <- cz_wage_quantiles %>% filter(!county_ansi %in% county_vec)
# CZ okay
cz_wage_quantiles %>%
    write_parquet(here("binaries", "acs_czone_wage_quantiles.parquet")) %>%
    write_parquet(here("Data Int", "acs_czone_wage_quantiles.parquet"))

#### USDA QuickStats Census and Survey ####
qs_census_counties <- qs_census %>%
    distinct(state_fips_code, county_code) %>%
    mutate(county_ansi = paste0(state_fips_code, county_code))
unmatched_qs <- qs_census_counties %>% filter(!county_ansi %in% county_vec)
# Only unmatched county is in Alaska, we are good
qs_census %>%
    write_parquet(here("binaries", "qs_census_selected_obs.parquet")) %>%
    write_parquet(here("Data Int", "qs_census_selected_obs.parquet"))

#### CroplandCROS CDL ####
cdl_counties <- cdl_acreage %>% distinct(fips)
unmatched_cdl <- cdl_counties %>% filter(!fips %in% county_vec)
# CDL OK
cdl_acreage %>%
    write_parquet(here(
        "binaries",
        "croplandcros_county_crop_acres.parquet"
    )) %>%
    write_parquet(here(
        "Data Int",
        "croplandcros_county_crop_acres.parquet"
    ))

#### ACS QCEW ####
# SD Oglala Lakota to Shannon county
acs_qcew <- acs_qcew %>%
    mutate(
        county_fips_code = case_when(
            state_fips_code == "46" & county_fips_code == "102" ~ "113",
            .default = county_fips_code
        )
    )

acs_qcew_counties <- acs_qcew %>%
    distinct(state_fips_code, county_fips_code) %>%
    mutate(county_ansi = paste0(state_fips_code, county_fips_code))
unmatched_acs_qcew <- acs_qcew_counties %>% filter(!county_ansi %in% county_vec)
# Only unmatched location is in Alaska, ACS-QCEW ok
acs_qcew %>%
    write_parquet(here("binaries", "acs_qcew.parquet")) %>%
    write_parquet(here("Data Int", "acs_qcew.parquet"))

#### OEWS ####
# FL Dade to Miami-Dade
# SD Oglala Lakota to Shannon
oews_counties <- oews_agg %>%
    distinct(state_fips_code, county_fips_code) %>%
    mutate(county_ansi = paste0(state_fips_code, county_fips_code))
unmatched_oews <- oews_counties %>%
    filter(!county_ansi %in% county_vec) %>%
    arrange()
# Only unmatched counties are in Alaska and Puerto Rico, OEWS OK
oews_agg %>%
    write_parquet(here("binaries", "oews_county_aggregated.parquet")) %>%
    write_parquet(here("Data Int", "oews_county_aggregated.parquet"))
