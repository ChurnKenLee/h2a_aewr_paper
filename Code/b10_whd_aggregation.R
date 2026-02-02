library(here)
library(arrow)
library(tidyverse)
library(dplyr)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)
library(foreign)
library(tidycensus)
library(lubridate)
library(foreign)
library(haven)

rm(list = ls())

Mode <- function(x, na.rm = FALSE) {
  if(na.rm){
    x = x[!is.na(x)]
  }
  
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

whd_df <- read_parquet(here("binaries", "whd_with_fips.parquet")) %>% 
  clean_names() %>% 
  distinct(case_id, .keep_all = TRUE)

# Rename relevant variables to be more readable
whd_df <- whd_df %>% 
  rename(
    mspa_violation_count = mspa_violtn_cnt,
    mspa_back_wages_assessed_total_penalties_amount = mspa_bw_atp_amt,
    mspa_employee_assessed_total_penalties_count = mspa_ee_atp_cnt,
    mspa_civil_monetary_penalties_assessed_amount = mspa_cmp_assd_amt,
    h2a_violation_count = h2a_violtn_cnt,
    h2a_back_wages_assessed_total_penalties_amount = h2a_bw_atp_amt,
    h2a_employee_assessed_total_penalties_count = h2a_ee_atp_cnt,
    h2a_civil_monetary_penalties_assessed_amount = h2a_cmp_assd_amt
  )

# Clean up unmatched counties
whd_df <- whd_df %>% 
  mutate(fips = if_else(is.na(fips), "00000", fips)) %>% 
  mutate(fips = if_else(fips == "", "00000", fips))

# Set time to POSIX time
whd_df <- whd_df %>% 
  mutate(posix_start_date = as.POSIXct(findings_start_date, tz = "UTC", format = "%Y-%m-%d")) %>% 
  mutate(posix_end_date = as.POSIXct(findings_end_date, tz = "UTC", format = "%Y-%m-%d"))

# Drop entries with missing dates
whd_df <- whd_df %>% 
  filter(!is.na(posix_start_date)) %>% 
  filter(!is.na(posix_end_date))

# Remove cases with end date before start date
whd_df <- whd_df %>% 
  filter(posix_end_date >= posix_start_date)

# We take the end date of violations (and likely the start of investigation) as the relevant date for affecting farmer decisions
whd_df <- whd_df %>% 
  mutate(year = year(posix_end_date))

# For multi-county entries, equally split across all counties equally
whd_df <- whd_df %>% 
  mutate(n_counties = str_count(fips, ",") + 1) %>%
  separate_rows(fips, sep = ",")

whd_df <- whd_df %>% 
  mutate(across(starts_with("h2a_") | starts_with("mspa_"), ~ .x/n_counties))

# Aggregate into county-years
# Collapse by county-year
whd_aggregated_df <- whd_df %>% 
  group_by(fips, year) %>%
  summarise(
    across(
      starts_with("h2a_") | starts_with("mspa_"),
      sum,
      .names = "{.col}"
    )
  ) %>% 
  ungroup()

# Harmonize name of FIPS code variable
whd_aggregated_df <- whd_aggregated_df %>% 
  mutate(state_fips_code = substr(fips, 1, 2)) %>% 
  mutate(county_fips_code = substr(fips, 3, 5)) %>% 
  select(-fips)

# Export
whd_aggregated_df %>% 
  write_parquet(here("files_for_phil", "whd_aggregated.parquet")) %>% 
  write_parquet(here("binaries", "whd_aggregated.parquet"))


