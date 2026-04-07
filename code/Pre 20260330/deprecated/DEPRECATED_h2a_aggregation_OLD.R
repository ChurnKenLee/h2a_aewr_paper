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

h2a_df <- read_parquet(here("binaries", "h2a_with_fips.parquet")) %>% 
  clean_names() %>% 
  distinct(.keep_all = TRUE)

h2a_2021_2022_df <- read_parquet(here("binaries", "h2a_2021_2022_with_fips.parquet")) %>% 
  clean_names() %>% 
  mutate(fips_api = fips_nonapi)


# # Replace 2021-2022 data with my own data from DOL disclosure raw files
h2a_df <- h2a_df %>%
  filter(case_year < 2021) %>%
  bind_rows(h2a_2021_2022_df) %>%
  distinct(.keep_all = TRUE)

# Clean up unmatched counties
h2a_df <- h2a_df %>% 
  mutate(fips_nonapi = if_else(is.na(fips_nonapi), "00000", fips_api)) %>% 
  mutate(fips_nonapi = if_else(fips_nonapi == "", "00000", fips_api)) %>% 
  mutate(fips_api= if_else(is.na(fips_api), "00000", fips_api)) %>% 
  mutate(fips_api = if_else(fips_api == "", "00000", fips_api))

# Add application status
h2a_df <- h2a_df %>% 
  mutate(case_status_harmonized = if_else(grepl('CERTIFICATION|Certification|Certified', case_status), "certified", "")) %>% 
  mutate(case_status_harmonized = if_else(grepl('DENIED|Denied', case_status), "denied", case_status_harmonized)) %>% 
  mutate(case_status_harmonized = if_else(grepl('WITHDRAWN|Withdrawn', case_status), "withdrawn", case_status_harmonized)) %>% 
  mutate(case_status_harmonized = if_else(grepl('PARTIAL|Partial', case_status), "partial_certification", case_status_harmonized)) %>% 
  arrange(case_status_harmonized)

# Fill in missing worker counts for full certification cases
h2a_df <- h2a_df %>% 
  mutate(nbr_workers_requested = if_else(case_status_harmonized == "certified" & is.na(nbr_workers_requested), nbr_workers_certified, nbr_workers_requested)) %>% 
  mutate(nbr_workers_certified = if_else(case_status_harmonized == "certified" & is.na(nbr_workers_certified), nbr_workers_requested, nbr_workers_certified))

# For partial certification cases, we assume that the number of workers requested is at least number of workers certified
h2a_df <- h2a_df %>% 
  mutate(nbr_workers_requested = if_else(case_status_harmonized == "partial_certification" & is.na(nbr_workers_requested), nbr_workers_certified, nbr_workers_requested))

# There are 2 cases where number certified is not reported but number requested is for partial certification; assume equal
h2a_df <- h2a_df %>% 
  mutate(nbr_workers_certified = if_else(case_status_harmonized == "partial_certification" & is.na(nbr_workers_certified), nbr_workers_requested, nbr_workers_certified))

# For withdrawn and denied cases, number certified is 0
h2a_df <- h2a_df %>% 
  mutate(nbr_workers_certified = if_else(case_status_harmonized == "withdrawn" | case_status_harmonized == "denied", 0, nbr_workers_certified))

# Incorrect dates due to typos
h2a_df <- h2a_df %>% 
  mutate(certification_begin_date = if_else(certification_begin_date == "2031-07-01 00:00:00", "2013-07-01 00:00:00", certification_begin_date)) %>% 
  mutate(certification_begin_date = if_else(certification_begin_date == "1011-12-19 00:00:00", "2022-12-19 00:00:00", certification_begin_date))

h2a_df <- h2a_df %>% 
  mutate(certification_end_date = if_else(certification_end_date == "2105-11-14 00:00:00", "2015-11-14 00:00:00", certification_end_date)) %>% 
  mutate(certification_end_date = if_else(certification_end_date == "2104-12-10 00:00:00", "2014-12-10 00:00:00", certification_end_date)) %>%
  mutate(certification_end_date = if_else(certification_end_date == "2201-11-09 00:00:00", "2013-11-09 00:00:00", certification_end_date)) %>% 
  mutate(certification_end_date = if_else(certification_end_date == "2101-11-09 00:00:00", "2013-11-09 00:00:00", certification_end_date))

h2a_df <- h2a_df %>% 
  mutate(job_start_date = if_else(job_start_date == "2029-01-22 00:00:00", "2019-01-22 00:00:00", job_start_date)) %>% 
  mutate(job_start_date = if_else(job_start_date == "2029-03-08 00:00:00", "2019-03-08 00:00:00", job_start_date)) %>% 
  mutate(job_start_date = if_else(job_start_date == "2109-03-31 00:00:00", "2019-03-31 00:00:00", job_start_date)) %>% 
  mutate(job_start_date = if_else(job_start_date == "3019-08-23 00:00:00", "2019-08-23 00:00:00", job_start_date)) %>% 
  mutate(job_start_date = if_else(job_start_date == "2028-11-21 00:00:00", "2018-11-21 00:00:00", job_start_date)) %>% 
  mutate(job_start_date = if_else(job_start_date == "2108-05-22 00:00:00", "2018-05-22 00:00:00", job_start_date)) %>% 
  mutate(job_start_date = if_else(job_start_date == "3016-03-25 00:00:00", "2016-03-25 00:00:00", job_start_date))

h2a_df <- h2a_df %>%
  mutate(job_end_date = if_else(job_end_date == "2106-07-21 00:00:00", "2016-07-21 00:00:00", job_end_date)) %>%
  mutate(job_end_date = if_else(job_end_date == "2049-08-17 00:00:00", "2019-08-17 00:00:00", job_end_date))

# Some entries have swapped begin and end dates
h2a_df <- h2a_df %>% 
  mutate(temp_end = certification_end_date, temp_start = certification_begin_date) %>%
  mutate(certification_end_date = if_else(as.Date(temp_end) < as.Date(temp_start), temp_start, certification_end_date)) %>%
  mutate(certification_begin_date = if_else(as.Date(temp_end) < as.Date(temp_start), temp_end, certification_begin_date)) %>% 
  mutate(temp_end = job_end_date, temp_start = job_start_date) %>%
  mutate(job_end_date = if_else(as.Date(temp_end) < as.Date(temp_start), temp_start, job_end_date)) %>%
  mutate(job_start_date = if_else(as.Date(temp_end) < as.Date(temp_start), temp_end, job_start_date)) %>% 
  select(-temp_start, -temp_end)

# Create harmonized dates
h2a_df <- h2a_df %>% 
  mutate(start_date = if_else(!is.na(certification_begin_date), certification_begin_date, job_start_date)) %>%
  mutate(end_date = if_else(!is.na(certification_begin_date), certification_end_date, job_end_date)) %>% 
  mutate(start_date = as.POSIXct(start_date, tz = "UTC")) %>% 
  mutate(end_date = as.POSIXct(end_date, tz = "UTC"))

# Drop applications with no date
h2a_df <- h2a_df %>% 
  filter(!is.na(start_date)) %>% 
  filter(!is.na(end_date))

# Calculate man-hours employed
# We impute the number of hours worked per week for those missing it with the mean for that year
h2a_df <- h2a_df %>% 
  mutate(mean_basic_number_of_hours = mean(basic_number_of_hours, na.rm = TRUE))

h2a_df <- h2a_df %>% 
  mutate(basic_number_of_hours = if_else(is.na(basic_number_of_hours), mean_basic_number_of_hours, basic_number_of_hours)) %>%
  mutate(man_hours_certified = (as.numeric(as.Date(end_date) - as.Date(start_date))/7) * basic_number_of_hours * nbr_workers_certified) %>% 
  mutate(man_hours_requested = (as.numeric(as.Date(end_date) - as.Date(start_date))/7) * basic_number_of_hours * nbr_workers_requested)

# Keep only variables we care about
h2a_df <- h2a_df %>% 
  select(case_number, fiscal_year, fips_api, fips_nonapi, start_date, end_date, nbr_workers_requested, nbr_workers_certified, man_hours_requested, man_hours_certified)

# We now calculate the number of days within each year for each application, to apportion the workers and man hours to the appropriate year
# Start and end year for each application
h2a_all_years_df <- h2a_df %>%
  mutate(date_int = interval(start_date, end_date)) %>% 
  mutate(year = map2(year(start_date), year(end_date), seq)) 

# Number of days in each application
h2a_all_years_df <- h2a_all_years_df %>% 
  mutate(total_days = as.numeric(as.Date(end_date) - as.Date(start_date)) + 1)

# Explode each entry by calendar years spanned by each entry
h2a_all_years_df <- h2a_all_years_df %>% 
  unnest(year)

# Calculate the number of days in each year spanned by each entry by intersection with the calendar year
h2a_all_years_df <- h2a_all_years_df %>% 
  mutate(year_int = interval(as.Date(paste0(year, '-01-01')), as.Date(paste0(year, '-12-31')))) %>% 
  mutate(year_sect = intersect(date_int, year_int)) %>% # intersecting dates within each calendar year
  mutate(start_new = as.Date(int_start(year_sect))) %>% # start date of intersection
  mutate(end_new = as.Date(int_end(year_sect))) %>% # end date of intersection
  mutate(year = year(start_new)) %>% # year of intersection
  mutate(days_in_year = as.numeric(end_new - start_new) + 1) # number of days in intersection
         
# Calculate year-intersection weighted workers and man-hours
h2a_all_years_df <- h2a_all_years_df %>% 
  mutate(year_weighted_nbr_workers_requested = (days_in_year/total_days)*nbr_workers_requested) %>% 
  mutate(year_weighted_nbr_workers_certified = (days_in_year/total_days)*nbr_workers_certified) %>% 
  mutate(year_weighted_man_hours_requested = (days_in_year/total_days)*man_hours_requested) %>% 
  mutate(year_weighted_man_hours_certified = (days_in_year/total_days)*man_hours_certified) %>% 
  mutate(year_weighted_nbr_applications = days_in_year/365)

# Now split into non-api and api county-year numbers
h2a_all_years_nonapi_df <- h2a_all_years_df %>% 
  select(-fips_api)

h2a_all_years_api_df <- h2a_all_years_df %>% 
  select(-fips_nonapi)

# For multi-county entries, equally split across all counties equally
h2a_all_years_nonapi_df <- h2a_all_years_nonapi_df %>% 
  mutate(n_counties = str_count(fips_nonapi, ",") + 1) %>%
  separate_rows(fips_nonapi, sep = ",") %>% 
  mutate(county_year_weighted_nbr_workers_requested = year_weighted_nbr_workers_requested/n_counties) %>% 
  mutate(county_year_weighted_nbr_workers_certified = year_weighted_nbr_workers_certified/n_counties) %>% 
  mutate(county_year_weighted_man_hours_requested = year_weighted_man_hours_requested/n_counties) %>% 
  mutate(county_year_weighted_man_hours_certified = year_weighted_man_hours_certified/n_counties) %>% 
  mutate(county_year_weighted_nbr_applications = year_weighted_nbr_applications/n_counties)

h2a_all_years_api_df <- h2a_all_years_api_df %>% 
  mutate(n_counties = str_count(fips_api, ",") + 1) %>%
  separate_rows(fips_api, sep = ",") %>% 
  mutate(county_year_weighted_nbr_workers_requested = year_weighted_nbr_workers_requested/n_counties) %>% 
  mutate(county_year_weighted_nbr_workers_certified = year_weighted_nbr_workers_certified/n_counties) %>% 
  mutate(county_year_weighted_man_hours_requested = year_weighted_man_hours_requested/n_counties) %>% 
  mutate(county_year_weighted_man_hours_certified = year_weighted_man_hours_certified/n_counties) %>% 
  mutate(county_year_weighted_nbr_applications = year_weighted_nbr_applications/n_counties)

# Collapse by county-year
h2a_all_years_nonapi_aggregated_df <- h2a_all_years_nonapi_df %>% 
  group_by(fips_nonapi, year) %>%
  summarise(
    nbr_workers_requested_all_years_nonapi = sum(county_year_weighted_nbr_workers_requested, na.rm = TRUE),
    nbr_workers_certified_all_years_nonapi = sum(county_year_weighted_nbr_workers_certified, na.rm = TRUE),
    man_hours_requested_all_years_nonapi = sum(county_year_weighted_man_hours_requested, na.rm = TRUE),
    man_hours_certified_all_years_nonapi = sum(county_year_weighted_man_hours_certified, na.rm = TRUE),
    nbr_applications_all_years_nonapi = sum(county_year_weighted_nbr_applications, na.rm = TRUE),
    ) %>% 
  ungroup()

h2a_all_years_api_aggregated_df <- h2a_all_years_api_df %>% 
  group_by(fips_api, year) %>%
  summarise(
    nbr_workers_requested_all_years_api = sum(county_year_weighted_nbr_workers_requested, na.rm = TRUE),
    nbr_workers_certified_all_years_api = sum(county_year_weighted_nbr_workers_certified, na.rm = TRUE),
    man_hours_requested_all_years_api = sum(county_year_weighted_man_hours_requested, na.rm = TRUE),
    man_hours_certified_all_years_api = sum(county_year_weighted_man_hours_certified, na.rm = TRUE),
    nbr_applications_all_years_api = sum(county_year_weighted_nbr_applications, na.rm = TRUE),
  ) %>% 
  ungroup()

# Calculate same variables, but front loaded into the start year of each case
h2a_start_year_df <- h2a_df %>% 
  mutate(start_year = year(start_date))

# Split into non-api and api county-year numbers
h2a_start_year_nonapi_df <- h2a_start_year_df %>% 
  select(-fips_api)

h2a_start_year_api_df <- h2a_start_year_df %>% 
  select(-fips_nonapi)

# For multi-county entries, equally split workers and man-hours across all counties equally
h2a_start_year_nonapi_df <- h2a_start_year_nonapi_df %>% 
  mutate(n_counties = str_count(fips_nonapi, ",") + 1) %>%
  separate_rows(fips_nonapi, sep = ",") %>% 
  mutate(county_nbr_workers_requested = nbr_workers_requested/n_counties) %>% 
  mutate(county_nbr_workers_certified = nbr_workers_certified/n_counties) %>% 
  mutate(county_man_hours_requested = man_hours_requested/n_counties) %>% 
  mutate(county_man_hours_certified = man_hours_certified/n_counties) %>% 
  mutate(county_nbr_applications = 1/n_counties)

h2a_start_year_api_df <- h2a_start_year_api_df %>% 
  mutate(n_counties = str_count(fips_api, ",") + 1) %>%
  separate_rows(fips_api, sep = ",") %>% 
  mutate(county_nbr_workers_requested = nbr_workers_requested/n_counties) %>% 
  mutate(county_nbr_workers_certified = nbr_workers_certified/n_counties) %>% 
  mutate(county_man_hours_requested = man_hours_requested/n_counties) %>% 
  mutate(county_man_hours_certified = man_hours_certified/n_counties) %>% 
  mutate(county_nbr_applications = 1/n_counties)

# Collapse by county-year
h2a_start_year_nonapi_aggregated_df <- h2a_start_year_nonapi_df %>% 
  group_by(fips_nonapi, start_year) %>%
  summarise(
    nbr_workers_requested_start_year_nonapi = sum(county_nbr_workers_requested, na.rm = TRUE),
    nbr_workers_certified_start_year_nonapi = sum(county_nbr_workers_certified, na.rm = TRUE),
    man_hours_requested_start_year_nonapi = sum(county_man_hours_requested, na.rm = TRUE),
    man_hours_certified_start_year_nonapi = sum(county_man_hours_certified, na.rm = TRUE),
    nbr_applications_start_year_nonapi = sum(county_nbr_applications, na.rm = TRUE),
  ) %>% 
  ungroup()

h2a_start_year_api_aggregated_df <- h2a_start_year_api_df %>% 
  group_by(fips_api, start_year) %>%
  summarise(
    nbr_workers_requested_start_year_api = sum(county_nbr_workers_requested, na.rm = TRUE),
    nbr_workers_certified_start_year_api = sum(county_nbr_workers_certified, na.rm = TRUE),
    man_hours_requested_start_year_api = sum(county_man_hours_requested, na.rm = TRUE),
    man_hours_certified_start_year_api = sum(county_man_hours_certified, na.rm = TRUE),
    nbr_applications_start_year_api = sum(county_nbr_applications, na.rm = TRUE),
  ) %>% 
  ungroup()

# Calculate same variables, but aggregated into case fiscal year
h2a_fiscal_year_df <- h2a_df %>% 
  mutate(fiscal_year = as.numeric(fiscal_year))

# Split into non-api and api county-year numbers
h2a_fiscal_year_nonapi_df <- h2a_fiscal_year_df %>% 
  select(-fips_api)

h2a_fiscal_year_api_df <- h2a_fiscal_year_df %>% 
  select(-fips_nonapi)

# For multi-county entries, equally split workers and man-hours across all counties equally
h2a_fiscal_year_nonapi_df <- h2a_fiscal_year_nonapi_df %>% 
  mutate(n_counties = str_count(fips_nonapi, ",") + 1) %>%
  separate_rows(fips_nonapi, sep = ",") %>% 
  mutate(county_nbr_workers_requested = nbr_workers_requested/n_counties) %>% 
  mutate(county_nbr_workers_certified = nbr_workers_certified/n_counties) %>% 
  mutate(county_man_hours_requested = man_hours_requested/n_counties) %>% 
  mutate(county_man_hours_certified = man_hours_certified/n_counties) %>% 
  mutate(county_nbr_applications = 1/n_counties)

h2a_fiscal_year_api_df <- h2a_fiscal_year_api_df %>% 
  mutate(n_counties = str_count(fips_api, ",") + 1) %>%
  separate_rows(fips_api, sep = ",") %>% 
  mutate(county_nbr_workers_requested = nbr_workers_requested/n_counties) %>% 
  mutate(county_nbr_workers_certified = nbr_workers_certified/n_counties) %>% 
  mutate(county_man_hours_requested = man_hours_requested/n_counties) %>% 
  mutate(county_man_hours_certified = man_hours_certified/n_counties) %>% 
  mutate(county_nbr_applications = 1/n_counties)

# Collapse by county-fiscal-year
h2a_fiscal_year_nonapi_aggregated_df <- h2a_fiscal_year_nonapi_df %>% 
  group_by(fips_nonapi, fiscal_year) %>%
  summarise(
    nbr_workers_requested_fiscal_year_nonapi = sum(county_nbr_workers_requested, na.rm = TRUE),
    nbr_workers_certified_fiscal_year_nonapi = sum(county_nbr_workers_certified, na.rm = TRUE),
    man_hours_requested_fiscal_year_nonapi = sum(county_man_hours_requested, na.rm = TRUE),
    man_hours_certified_fiscal_year_nonapi = sum(county_man_hours_certified, na.rm = TRUE),
    nbr_applications_fiscal_year_nonapi = sum(county_nbr_applications, na.rm = TRUE),
  ) %>% 
  ungroup()

h2a_fiscal_year_api_aggregated_df <- h2a_fiscal_year_api_df %>% 
  group_by(fips_api, fiscal_year) %>%
  summarise(
    nbr_workers_requested_fiscal_year_api = sum(county_nbr_workers_requested, na.rm = TRUE),
    nbr_workers_certified_fiscal_year_api = sum(county_nbr_workers_certified, na.rm = TRUE),
    man_hours_requested_fiscal_year_api = sum(county_man_hours_requested, na.rm = TRUE),
    man_hours_certified_fiscal_year_api = sum(county_man_hours_certified, na.rm = TRUE),
    nbr_applications_fiscal_year_api = sum(county_nbr_applications, na.rm = TRUE),
  ) %>% 
  ungroup()

# Combine
# Harmonize joining variable name before merging
h2a_all_years_nonapi_aggregated_df <- h2a_all_years_nonapi_aggregated_df %>% 
  rename(fips = fips_nonapi) %>% 
  mutate(fips = if_else(fips == "", "00000", fips))

h2a_all_years_api_aggregated_df <- h2a_all_years_api_aggregated_df %>% 
  rename(fips = fips_api) %>% 
  mutate(fips = if_else(fips == "", "00000", fips))

h2a_start_year_nonapi_aggregated_df <- h2a_start_year_nonapi_aggregated_df %>% 
  rename(fips = fips_nonapi) %>% 
  rename(year = start_year) %>% 
  mutate(fips = if_else(fips == "", "00000", fips))

h2a_start_year_api_aggregated_df <- h2a_start_year_api_aggregated_df %>% 
  rename(fips = fips_api) %>% 
  rename(year = start_year) %>% 
  mutate(fips = if_else(fips == "", "00000", fips))

h2a_fiscal_year_nonapi_aggregated_df <- h2a_fiscal_year_nonapi_aggregated_df %>% 
  rename(fips = fips_nonapi) %>% 
  rename(year = fiscal_year) %>% 
  mutate(fips = if_else(fips == "", "00000", fips))

h2a_fiscal_year_api_aggregated_df <- h2a_fiscal_year_api_aggregated_df %>% 
  rename(fips = fips_api) %>% 
  rename(year = fiscal_year) %>% 
  mutate(fips = if_else(fips == "", "00000", fips))

# Merge
h2a_aggregated_df <- h2a_all_years_nonapi_aggregated_df %>% 
  full_join(h2a_all_years_api_aggregated_df) %>% 
  full_join(h2a_start_year_nonapi_aggregated_df) %>% 
  full_join(h2a_start_year_api_aggregated_df) %>% 
  full_join(h2a_fiscal_year_nonapi_aggregated_df) %>% 
  full_join(h2a_fiscal_year_api_aggregated_df)

# Harmonize name of FIPS code variable
h2a_aggregated_df <- h2a_aggregated_df %>% 
  mutate(state_fips_code = substr(fips, 1, 2)) %>% 
  mutate(county_fips_code = substr(fips, 3, 5)) %>% 
  select(-fips)

# Export
h2a_aggregated_df %>% 
  write_parquet(here("files_for_phil", "h2a_aggregated.parquet")) %>% 
  write_parquet(here("binaries", "h2a_aggregated.parquet"))

h2a_ts_df <- h2a_aggregated_df %>% 
  filter(state_fips_code != "00") %>% 
  group_by(year) %>% 
  summarize(nbr_workers_certified = sum(nbr_workers_certified_fiscal_year_api, na.rm = TRUE)) %>% 
  filter(year > 2007 & year < 2023)

tsplot <- ggplot(h2a_ts_df, aes(x=year, y=nbr_workers_certified)) +
  geom_line() + 
  xlab("")
tsplot
