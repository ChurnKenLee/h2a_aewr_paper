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
library(haven)

rm(list = ls())

Mode <- function(x, na.rm = FALSE) {
  if(na.rm){
    x = x[!is.na(x)]
  }
  
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

#### Produce workers requested and certified counts for each worksite entry ####
h2a_df <- read_parquet(here("binaries", "h2a_with_fips.parquet")) %>%
  clean_names() %>%
  distinct(.keep_all = TRUE)

addendum_b <- read_parquet(here("binaries", "h2a_addendum_b_with_fips.parquet"))%>%
  clean_names()

# Handle master entries that include worker counts in all sub entries
# Split cases into those with duplicated worker counts in sub-entries and those without
h2a_df <- h2a_df %>%
  mutate(nbr_workers_certified = as.numeric(nbr_workers_certified)) %>%
  mutate(nbr_workers_requested = as.numeric(nbr_workers_requested)) %>% 
  mutate(nbr_workers_needed = as.numeric(nbr_workers_needed))

# Entries with 0 workers certified indicate worker lodgings
h2a_df <- h2a_df %>% 
  filter(nbr_workers_certified != 0)

# Remove addendum B cases with no corresponding master entry in the H-2A master file
# We don't know the case status, we don't have the certified worker count otherwise
addendum_b_cases <- addendum_b %>%
  distinct(case_number, fiscal_year) %>%
  mutate(has_addendum = TRUE)

post_2019_cases <- h2a_df %>% 
  distinct(case_number, fiscal_year) %>% 
  mutate(has_master = TRUE)

addendum_b <- addendum_b %>% 
  left_join(post_2019_cases) %>% 
  filter(has_master) %>% 
  select(-has_master)

# Add indicator for entries that have corresponding addendum B entries
h2a_df <- h2a_df %>%
  left_join(addendum_b_cases) %>%
  mutate(has_addendum = if_else(is.na(has_addendum), FALSE, has_addendum))

h2a_df <- h2a_df %>%
  group_by(case_number, fiscal_year) %>%
  mutate(n_entries = n()) %>%
  ungroup()

h2a_unduped <- h2a_df %>%
  filter((n_entries == 1 & fiscal_year < 2020) | (!has_addendum & fiscal_year > 2019))

h2a_duped_pre2020 <- h2a_df %>%
  filter(n_entries > 1 & fiscal_year < 2020)

h2a_duped_post2020 <- h2a_df %>%
  filter(has_addendum & fiscal_year > 2019)

# First deal with years prior to 2020
# Only the master entry has an organization flag
# From 2016 to 2019, we can also use the primary_sub indicator to identify master and sub entries
h2a_duped_pre2020 <- h2a_duped_pre2020 %>%
  mutate(entry_type = case_when(
    (fiscal_year < 2016) & (organization_flag != '') ~ "master",
    (fiscal_year > 2015) & (fiscal_year < 2020) & (primary_sub == "PRI") ~ "master",
    .default = "sub"
  ))

sanity_check_total <- h2a_duped_pre2020 %>% 
  filter(entry_type == "master")

sanity_check_total <- sanity_check_total %>% 
  bind_rows(h2a_duped_post2020) %>% 
  bind_rows(h2a_unduped)

sanity_check_total <- sanity_check_total %>% 
  group_by(fiscal_year) %>% 
  summarize(
    nbr_workers_certified = sum(nbr_workers_certified, na.rm = TRUE),
    nbr_workers_requested = sum(nbr_workers_requested, na.rm = TRUE),
    nbr_workers_needed = sum(nbr_workers_needed, na.rm = TRUE)
    )

# Check master count; all cases have only 1 master record, which is good
h2a_duped_pre2020 <- h2a_duped_pre2020 %>%
  group_by(case_number, fiscal_year) %>%
  mutate(master_count = sum(entry_type == "master")) %>%
  ungroup() %>%
  arrange(master_count, case_number)

# Separate into master and sub entries
# Master entry worker count
master_entries <- h2a_duped_pre2020 %>%
  filter(entry_type == "master") %>%
  mutate(master_n_workers_certified = nbr_workers_certified) %>% 
  mutate(master_n_workers_requested = nbr_workers_requested) %>% 
  select(case_number, fiscal_year, master_n_workers_certified, master_n_workers_requested, organization_flag) %>% 
  rename(application_type = organization_flag)

# Sub entry worker count and types of sub entries in each case
sub_entries <- h2a_duped_pre2020 %>%
  filter(entry_type == "sub") %>%
  group_by(case_number, fiscal_year) %>%
  mutate(sub_n_workers_certified = sum(nbr_workers_certified, na.rm = TRUE)) %>% 
  mutate(sub_n_workers_requested = sum(nbr_workers_requested, na.rm = TRUE)) %>% 
  ungroup() %>%
  select(case_number, fiscal_year, sub_n_workers_certified, sub_n_workers_requested) %>%
  distinct(case_number, fiscal_year, .keep_all = TRUE)

# Add information back
h2a_duped_pre2020 <- h2a_duped_pre2020 %>% 
  left_join(master_entries) %>% 
  left_join(sub_entries)

# Master entry has the true count of the number of workers certified
# There are two types of multi-entry applications: association, and joint filings
# For applications by associations, we want to drop the master record, as this is just the headquarters of the association, e.g., VASS, NC for NCGA
# For joint applications, we want to spread the workers evenly across all entries, including the master entry
# Note that sub entries with 0 workers certified indicate housing for workers, which we already removed
application_types <- h2a_duped_pre2020 %>% 
  distinct(application_type)

# These are associational applications
drop_master_list <- c(
  "A", 
  "ASSOCIATION - JOINT EMPLOYER (H-2A ONLY)", 
  "ASSOCIATION - SOLE EMPLOYER (H-2A ONLY)", 
  "H-2A LABOR CONTRACTOR OR JOB CONTRACTOR", 
  "ASSOCIATION - FILING AS AGENT (H-2A ONLY)"
  )

# Decide on how we want to perform adjustment
h2a_duped_pre2020 <- h2a_duped_pre2020 %>% 
  mutate(adjustment_method_certified = case_when(
    application_type %in% drop_master_list & sub_n_workers_certified > 0 ~ "inflate_sub_workers_drop_master",
    application_type %in% drop_master_list & sub_n_workers_certified == 0 ~ "distribute_master_workers_across_sub_entries_drop_master",
    !(application_type %in% drop_master_list) & sub_n_workers_certified == 0 ~ "distribute_master_workers_across_all_entries_evenly", # all sub entries are NA
    !(application_type %in% drop_master_list) & sub_n_workers_certified > 0 ~ "distribute_master_workers_across_all_entries_fractionally" # distribute master worker count fractionally across entries
  ))

# Same adjustment for requested worker counts
h2a_duped_pre2020 <- h2a_duped_pre2020 %>% 
  mutate(adjustment_method_requested = case_when(
    application_type %in% drop_master_list & sub_n_workers_requested > 0 ~ "inflate_sub_workers_drop_master",
    application_type %in% drop_master_list & sub_n_workers_requested == 0 ~ "distribute_master_workers_across_sub_entries_drop_master",
    !(application_type %in% drop_master_list) & sub_n_workers_requested == 0 ~ "distribute_master_workers_across_all_entries_evenly",
    !(application_type %in% drop_master_list) & sub_n_workers_requested > 0 ~ "distribute_master_workers_across_all_entries_fractionally"
  ))

# Perform adjustment
h2a_duped_pre2020 <- h2a_duped_pre2020 %>% 
  mutate(master_sub_ratio_certified = master_n_workers_certified/sub_n_workers_certified) %>% 
  mutate(adjusted_nbr_workers_certified = case_when(
    adjustment_method_certified == "inflate_sub_workers_drop_master" ~ nbr_workers_certified*master_sub_ratio_certified,
    adjustment_method_certified == "distribute_master_workers_across_sub_entries_drop_master" ~ master_n_workers_certified/(n_entries-1),
    adjustment_method_certified == "distribute_master_workers_across_all_entries_evenly" ~ master_n_workers_certified/n_entries,
    adjustment_method_certified == "distribute_master_workers_across_all_entries_fractionally" ~ master_n_workers_certified*(nbr_workers_certified/(master_n_workers_certified + sub_n_workers_certified))
  ))

h2a_duped_pre2020 <- h2a_duped_pre2020 %>% 
  mutate(master_sub_ratio_requested = master_n_workers_requested/sub_n_workers_requested) %>% 
  mutate(adjusted_nbr_workers_requested = case_when(
    adjustment_method_requested == "inflate_sub_workers_drop_master" ~ nbr_workers_requested*master_sub_ratio_requested,
    adjustment_method_requested == "distribute_master_workers_across_sub_entries_drop_master" ~ master_n_workers_requested/(n_entries-1),
    adjustment_method_requested == "distribute_master_workers_across_all_entries_evenly" ~ master_n_workers_requested/n_entries,
    adjustment_method_requested == "distribute_master_workers_across_all_entries_fractionally" ~ master_n_workers_requested*(nbr_workers_requested/(master_n_workers_requested + sub_n_workers_requested))
  ))

# Discard unwanted master entries
h2a_duped_pre2020 <- h2a_duped_pre2020 %>%
  filter(!((application_type %in% drop_master_list) & (entry_type == "master")))

# Now deal with post 2020 years
# We may want to keep or discard the master entries depending on the application type
# Define unified worker requested count for each entry
h2a_duped_post2020 <- h2a_duped_post2020 %>%
  mutate(nbr_workers_requested = as.numeric(nbr_workers_requested)) %>% 
  mutate(entry_type = "master")

addendum_b <- addendum_b %>%
  mutate(nbr_workers_requested = as.numeric(total_h2a_workers_requested)) %>%
  select(-total_h2a_workers_requested) %>% 
  mutate(entry_type = "sub")

# For entries that were denied, number certified is NA, fill these in with 0
h2a_duped_post2020 <- h2a_duped_post2020 %>%
  mutate(nbr_workers_certified = if_else(is.na(nbr_workers_certified), 0, nbr_workers_certified))

# Since the master entries have the correct sums, we use those as the case totals
h2a_duped_post2020 <- h2a_duped_post2020 %>%
  mutate(case_nbr_workers_certified = nbr_workers_certified) %>%
  mutate(case_nbr_workers_requested = nbr_workers_requested) %>%
  bind_rows(addendum_b) %>%
  group_by(case_number, fiscal_year) %>%
  fill(case_nbr_workers_requested, case_nbr_workers_certified, type_of_employer_application) %>%
  ungroup() %>% 
  arrange(fiscal_year, case_number)

# For the sub-entries, fill in the application status, dates, wage rates, wage units, and hours of work, which is the same as in the master entry
h2a_duped_post2020 <- h2a_duped_post2020 %>%
  group_by(case_number, fiscal_year) %>% 
  fill(case_status, number_of_hours, wage_rate, wage_unit, job_begin_date, job_end_date) %>%
  ungroup()

# Use addendum B requested worker counts as weights
# Total weight depends on whether we keep master entry or not
application_types <- h2a_duped_post2020 %>% 
  distinct(type_of_employer_application)

drop_master_list <- c(
  "ASSOCIATION - AGENT", 
  "ASSOCIATION - SOLE EMPLOYER", 
  "ASSOCIATION - JOINT EMPLOYER"
)

# We use the number of workers requested for each worksite as the weights
# Have to fill in requested worker count for addendum B entries with missing counts
# This is because all addendum B entries are worksites, since housing is in separate addendum B housing record file
# Use the mean of the available entries
weights <- h2a_duped_post2020 %>% 
  filter(!((type_of_employer_application %in% drop_master_list) & (entry_type == "master"))) %>% 
  group_by(case_number, fiscal_year) %>% 
  mutate(mean_nbr_workers_requested_excl_drop_master = mean(nbr_workers_requested, na.rm = TRUE)) %>% 
  ungroup() %>% 
  mutate(weight_excl_drop_master = if_else(!is.na(nbr_workers_requested), nbr_workers_requested, mean_nbr_workers_requested_excl_drop_master)) %>% 
  group_by(case_number, fiscal_year) %>% 
  mutate(total_weight_excl_drop_master = sum(weight_excl_drop_master)) %>% 
  ungroup() %>% 
  select(case_number, fiscal_year, mean_nbr_workers_requested_excl_drop_master, total_weight_excl_drop_master) %>% 
  distinct()

# Add total weights (calculated excluding master where appropriate) back to original dataset
# Calculate final weights to use
h2a_duped_post2020 <- h2a_duped_post2020 %>% 
  left_join(weights) %>% 
  mutate(weight = if_else(!is.na(nbr_workers_requested), nbr_workers_requested, mean_nbr_workers_requested_excl_drop_master))

# There are still entries with missing weights
# These are sub-entries with all NA sub-entries but with master entries that have to be dropped
# Fill in the missing weight using the weight in the master entry, i.e., equal sub-entry weights
h2a_duped_post2020 <- h2a_duped_post2020 %>% 
  fill(weight, .direction="down")

# Check weights are correct
h2a_duped_post2020 <- h2a_duped_post2020 %>% 
  filter(!((type_of_employer_application %in% drop_master_list) & (entry_type == "master"))) %>% 
  mutate(weight = if_else(is.na(weight), mean_nbr_workers_requested_excl_drop_master, weight)) %>% 
  group_by(case_number, fiscal_year) %>% 
  mutate(total_weight = sum(weight, na.rm = TRUE)) %>% 
  ungroup() %>% 
  mutate(total_weight = if_else(is.na(total_weight_excl_drop_master), total_weight, total_weight_excl_drop_master)) %>% 
  mutate(weight_ratio = weight/total_weight) %>% 
  group_by(case_number, fiscal_year) %>% 
  mutate(total_weight_ratio_check = sum(weight_ratio)) %>% 
  ungroup() %>% 
  arrange(total_weight_ratio_check, fiscal_year, case_number)

# Define adjusted worker counts
h2a_duped_post2020 <- h2a_duped_post2020 %>% 
  mutate(adjusted_nbr_workers_certified = case_nbr_workers_certified*weight_ratio) %>% 
  mutate(adjusted_nbr_workers_requested = case_nbr_workers_requested*weight_ratio)

# Combine all of the H-2A subsets back together
# No adjustment needed for unduped entries
h2a_unduped <- h2a_unduped %>% 
  mutate(adjusted_nbr_workers_certified = nbr_workers_certified) %>% 
  mutate(adjusted_nbr_workers_requested = nbr_workers_requested)

h2a_combined <- h2a_unduped %>%
  bind_rows(h2a_duped_pre2020) %>%
  bind_rows(h2a_duped_post2020) %>%
  arrange(fiscal_year, case_number)

# Sanity check totals
sanity_check_total_2 <- h2a_combined %>% 
  group_by(fiscal_year) %>% 
  summarize(
    nbr_workers_certified = sum(adjusted_nbr_workers_certified, na.rm = TRUE),
    nbr_workers_requested = sum(adjusted_nbr_workers_requested, na.rm = TRUE),
  )
# Sanity check numbers match, we are good

#### Clean before aggregating into county years ####
h2a_combined <- h2a_combined %>%
  mutate(fips = if_else(is.na(fips), "00000", fips)) %>%
  mutate(fips = if_else(fips == "", "00000", fips))

# Add application status
h2a_combined <- h2a_combined %>%
  mutate(case_status_harmonized = if_else(grepl('CERTIFICATION|CERTIFICATION|CERTIFIED', case_status), "certified", "")) %>%
  mutate(case_status_harmonized = if_else(grepl('DENIED', case_status), "denied", case_status_harmonized)) %>%
  mutate(case_status_harmonized = if_else(grepl('WITHDRAWN', case_status), "withdrawn", case_status_harmonized)) %>%
  mutate(case_status_harmonized = if_else(grepl('PARTIAL', case_status), "partial_certification", case_status_harmonized)) %>%
  arrange(case_status_harmonized)

# For withdrawn and denied cases, number certified is 0
h2a_combined <- h2a_combined %>%
  mutate(adjusted_nbr_workers_certified = if_else(case_status_harmonized == "withdrawn" | case_status_harmonized == "denied", 0, adjusted_nbr_workers_certified))

# Create harmonized dates
h2a_combined <- h2a_combined %>%
  mutate(begin_date = if_else(is.na(certification_begin_date) | certification_begin_date == '', job_begin_date, certification_begin_date)) %>%
  mutate(end_date = if_else(is.na(certification_begin_date) | certification_end_date == '', job_end_date, certification_end_date)) %>%
  filter(!is.na(begin_date) & !is.na(end_date))

# Case dates
h2a_combined <- h2a_combined %>%
  group_by(case_number, fiscal_year) %>%
  mutate(group_begin_date = Mode(begin_date, na.rm = TRUE)) %>%
  mutate(group_end_date = Mode(end_date, na.rm = TRUE)) %>%
  ungroup()

# Correct typos using group begin and end dates
h2a_combined <- h2a_combined %>%
  mutate(begin_date_error = (begin_date != group_begin_date)) %>%
  mutate(end_date_error = (end_date != group_end_date)) %>%
  mutate(begin_date = case_when(
    begin_date_error & n_entries > 2 ~ group_begin_date,
    .default = begin_date
  )) %>%
  mutate(end_date = case_when(
    end_date_error & n_entries > 2 ~ group_end_date,
    .default = end_date
  ))

# Set time to POSIX time to calculate weighting within and across years
h2a_combined <- h2a_combined %>%
  mutate(posix_begin_date = as.POSIXct(begin_date, tz = "UTC", format = "%Y-%m-%d")) %>%
  mutate(posix_end_date = as.POSIXct(end_date, tz = "UTC", format = "%Y-%m-%d"))

# Drop entries with missing dates
h2a_combined <- h2a_combined %>%
  filter(!is.na(posix_begin_date)) %>%
  filter(!is.na(posix_end_date))

# Check for obvious transposing errors
errors <- h2a_combined %>% 
  mutate(begin_year = year(posix_begin_date)) %>%
  mutate(begin_date_error = abs(as.numeric(begin_year) - as.numeric(fiscal_year))) %>%
  mutate(end_year = year(posix_end_date)) %>%
  mutate(end_date_error = abs(as.numeric(end_year) - as.numeric(fiscal_year)))

begin_date_error <- errors %>% 
  select(case_number, fiscal_year, begin_date, end_date, begin_date_error, end_date_error) %>% 
  filter(begin_date_error > 1) %>% 
  arrange(begin_date_error)

# Many of these are simple year transposition errors
# Fix these manually
h2a_combined <- h2a_combined %>% 
  mutate(begin_date = case_when(
    case_number == "H-300-18325-715722" & begin_date == "2028-11-21" ~ "2018-11-21",
    case_number == "H-300-19017-003849" & begin_date == "2029-01-22" ~ "2019-01-22",
    case_number == "H-300-19044-363875" & begin_date == "2029-03-08" ~ "2019-03-08",
    case_number == "H-300-18229-549625" & begin_date == "2008-09-13" ~ "2018-09-13",
    case_number == "H-300-13135-069256" & begin_date == "2031-07-01" ~ "2013-07-31",
    case_number == "H-300-18141-758942" & begin_date == "2108-05-22" ~ "2018-05-22",
    case_number == "H-300-19090-126428" & begin_date == "2109-03-31" ~ "2019-03-31",
    case_number == "H-300-16082-567927" & begin_date == "3016-03-25" ~ "2017-03-25",
    case_number == "H-300-19207-740828" & begin_date == "3019-08-23" ~ "2019-08-23", 
    .default = begin_date
  ))

end_date_error <- errors %>% 
  select(case_number, fiscal_year, begin_date, end_date, begin_date_error, end_date_error) %>% 
  filter(end_date_error > 1) %>% 
  arrange(end_date_error)

h2a_combined <- h2a_combined %>% 
  mutate(end_date = case_when(
    case_number == "H-300-19057-031168" & end_date == "2010-08-01" ~ "2020-08-01",
    case_number == "H-300-19034-745829" & end_date == "2049-08-17" ~ "2019-08-17",
    case_number == "H-300-16011-375074" & end_date == "2106-07-21" ~ "2016-07-21",
    case_number == "H-300-24194-193164" & end_date == "3025-06-20" ~ "2025-06-20",
    .default = end_date
  ))

# Set data variable again with errors manually corrected
h2a_combined <- h2a_combined %>%
  mutate(posix_begin_date = as.POSIXct(begin_date, tz = "UTC", format = "%Y-%m-%d")) %>%
  mutate(posix_end_date = as.POSIXct(end_date, tz = "UTC", format = "%Y-%m-%d"))

# These 214 cases have a start date after the end date; drop these
h2a_combined <- h2a_combined %>%
  filter(posix_end_date >= posix_begin_date)

# Export cleaned H-2A dataset for further analysis
h2a_combined <- h2a_combined %>% 
  mutate(
    nbr_workers_certified = adjusted_nbr_workers_certified,
    nbr_workers_requested = adjusted_nbr_workers_requested
  ) %>% 
  select(-adjusted_nbr_workers_certified, -adjusted_nbr_workers_requested)

sanity_check_total_3 <- h2a_combined %>% 
  group_by(fiscal_year) %>% 
  summarize(
    nbr_workers_certified = sum(nbr_workers_certified, na.rm = TRUE),
    nbr_workers_requested = sum(nbr_workers_requested, na.rm = TRUE),
  )
# Sanity check numbers mostly match (we dropped entries with date errors)

h2a_combined %>% write_parquet(here("binaries", "h2a_cleaned.parquet"))

#### Aggregate to county-year level ####
# Calculate man-hours employed
# We impute the number of hours worked per week for those missing it with the mean, which is 40.1
h2a_cleaned_df <- read_parquet(here("binaries", "h2a_cleaned.parquet"))

h2a_cleaned_df <- h2a_cleaned_df %>% 
  mutate(number_of_hours = as.numeric(number_of_hours)) %>% 
  mutate(mean_number_of_hours = mean(number_of_hours, na.rm = TRUE)) %>% 
  mutate(number_of_hours = if_else(is.na(number_of_hours), mean_number_of_hours, number_of_hours)) %>%
  mutate(man_hours_certified = (as.numeric(as.Date(end_date) - as.Date(begin_date))/7) * number_of_hours * nbr_workers_certified) %>% 
  mutate(man_hours_requested = (as.numeric(as.Date(end_date) - as.Date(begin_date))/7) * number_of_hours * nbr_workers_requested)

# Keep only variables we care about
h2a_cleaned_df <- h2a_cleaned_df %>% 
  select(case_number, fiscal_year, fips, begin_date, end_date, nbr_workers_requested, nbr_workers_certified, man_hours_requested, man_hours_certified, number_of_hours, wage_rate, wage_unit)

# We want to calculate the average wage rate of H-2A workers within each county-year
# We need to select a cutoff for distinguishing hourly wage values from non-hourly
# Note that unit of pay does not actually correspond to the rate of pay, even though they should
# Assume that those with reported wage rates above $100 are non-hourly wages instead
h2a_cleaned_df <- h2a_cleaned_df %>% 
  mutate(wage_rate = as.numeric(wage_rate))

h2a_cleaned_df <- h2a_cleaned_df %>% 
  mutate(hourly_wage = case_when(
    wage_rate <= 100 ~ wage_rate,
    .default = NaN
  ))

# Each case-number count as one application, but may be split across multiple entries
# Split across all entries equally
h2a_cleaned_df <- h2a_cleaned_df %>% 
  group_by(case_number, fiscal_year) %>% 
  mutate(n_entries = n()) %>% 
  ungroup()

silly_employer <- h2a_cleaned_df %>% filter(wage_rate == 0)

# We now calculate the number of days within each year for each application, to apportion the workers and man hours to the appropriate year
# Start and end year for each application
h2a_all_years_df <- h2a_cleaned_df %>%
  mutate(date_int = interval(begin_date, end_date)) %>% 
  mutate(year = map2(year(begin_date), year(end_date), seq))

# Number of days in each application
h2a_all_years_df <- h2a_all_years_df %>% 
  mutate(total_days = as.numeric(as.Date(end_date) - as.Date(begin_date)) + 1)

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
  mutate(year_weighted_n_applications = (days_in_year/total_days)*1/n_entries)

# For multi-county entries, equally split across all counties equally
h2a_all_years_df <- h2a_all_years_df %>% 
  mutate(n_counties = str_count(fips, ",") + 1) %>%
  separate_rows(fips, sep = ",") %>% 
  mutate(county_year_weighted_nbr_workers_requested = year_weighted_nbr_workers_requested/n_counties) %>% 
  mutate(county_year_weighted_nbr_workers_certified = year_weighted_nbr_workers_certified/n_counties) %>% 
  mutate(county_year_weighted_man_hours_requested = year_weighted_man_hours_requested/n_counties) %>% 
  mutate(county_year_weighted_man_hours_certified = year_weighted_man_hours_certified/n_counties) %>% 
  mutate(county_year_weighted_n_applications = year_weighted_n_applications/n_counties)

# We have to manually calculate weighted average of hourly wage because R cannot handle NAs in weights
h2a_all_years_df <- h2a_all_years_df %>% 
  mutate(hourly_wage_X_nbr_workers = hourly_wage*county_year_weighted_nbr_workers_certified)

# Collapse by county-year
h2a_all_years_aggregated_df <- h2a_all_years_df %>% 
  group_by(fips, year) %>%
  summarise(
    nbr_workers_requested_all_years = sum(county_year_weighted_nbr_workers_requested, na.rm = TRUE),
    nbr_workers_certified_all_years = sum(county_year_weighted_nbr_workers_certified, na.rm = TRUE),
    man_hours_requested_all_years = sum(county_year_weighted_man_hours_requested, na.rm = TRUE),
    man_hours_certified_all_years = sum(county_year_weighted_man_hours_certified, na.rm = TRUE),
    nbr_applications_all_years = sum(county_year_weighted_n_applications, na.rm = TRUE),
    total_hourly_wage_X_nbr_workers = sum(hourly_wage_X_nbr_workers, na.rm = TRUE)
  ) %>% 
  ungroup() %>% 
  mutate(mean_hourly_wage_all_years = total_hourly_wage_X_nbr_workers/nbr_workers_certified_all_years) %>% 
  select(-total_hourly_wage_X_nbr_workers)

test <- h2a_all_years_aggregated_df %>% 
  arrange(year)

# Calculate same variables, but front loaded into the start year of each case
h2a_start_year_df <- h2a_cleaned_df %>% 
  mutate(start_year = year(begin_date))

# For multi-county entries, equally split workers and man-hours across all counties equally
h2a_start_year_df <- h2a_start_year_df %>% 
  mutate(n_counties = str_count(fips, ",") + 1) %>%
  separate_rows(fips, sep = ",") %>% 
  mutate(county_nbr_workers_requested = nbr_workers_requested/n_counties) %>% 
  mutate(county_nbr_workers_certified = nbr_workers_certified/n_counties) %>% 
  mutate(county_man_hours_requested = man_hours_requested/n_counties) %>% 
  mutate(county_man_hours_certified = man_hours_certified/n_counties) %>% 
  mutate(county_n_applications = (1/n_entries)/n_counties)

# We have to manually calculate weighted average of hourly wage because R cannot handle NAs in weights
h2a_start_year_df <- h2a_start_year_df %>% 
  mutate(hourly_wage_X_nbr_workers = hourly_wage*county_nbr_workers_certified)

# Collapse by county-year
h2a_start_year_aggregated_df <- h2a_start_year_df %>% 
  group_by(fips, start_year) %>%
  summarise(
    nbr_workers_requested_start_year = sum(county_nbr_workers_requested, na.rm = TRUE),
    nbr_workers_certified_start_year = sum(county_nbr_workers_certified, na.rm = TRUE),
    man_hours_requested_start_year = sum(county_man_hours_requested, na.rm = TRUE),
    man_hours_certified_start_year = sum(county_man_hours_certified, na.rm = TRUE),
    nbr_applications_start_year = sum(county_n_applications, na.rm = TRUE),
    total_hourly_wage_X_nbr_workers = sum(hourly_wage_X_nbr_workers, na.rm = TRUE)
  ) %>% 
  ungroup() %>% 
  mutate(mean_hourly_wage_start_year = total_hourly_wage_X_nbr_workers/ nbr_workers_certified_start_year) %>% 
  select(-total_hourly_wage_X_nbr_workers)

# Calculate same variables, but aggregated into case fiscal year
h2a_fiscal_year_df <- h2a_cleaned_df %>% 
  mutate(fiscal_year = as.numeric(fiscal_year))

# For multi-county entries, equally split workers and man-hours across all counties equally
h2a_fiscal_year_df <- h2a_fiscal_year_df %>% 
  mutate(n_counties = str_count(fips, ",") + 1) %>%
  separate_rows(fips, sep = ",") %>% 
  mutate(county_nbr_workers_requested = nbr_workers_requested/n_counties) %>% 
  mutate(county_nbr_workers_certified = nbr_workers_certified/n_counties) %>% 
  mutate(county_man_hours_requested = man_hours_requested/n_counties) %>% 
  mutate(county_man_hours_certified = man_hours_certified/n_counties) %>% 
  mutate(county_n_applications = (1/n_entries)/n_counties)

# We have to manually calculate weighted average of hourly wage because R cannot handle NAs in weights
h2a_fiscal_year_df <- h2a_fiscal_year_df %>% 
  mutate(hourly_wage_X_nbr_workers = hourly_wage*county_nbr_workers_certified)

# Collapse by county-fiscal-year
h2a_fiscal_year_aggregated_df <- h2a_fiscal_year_df %>% 
  group_by(fips, fiscal_year) %>%
  summarise(
    nbr_workers_requested_fiscal_year = sum(county_nbr_workers_requested, na.rm = TRUE),
    nbr_workers_certified_fiscal_year = sum(county_nbr_workers_certified, na.rm = TRUE),
    man_hours_requested_fiscal_year = sum(county_man_hours_requested, na.rm = TRUE),
    man_hours_certified_fiscal_year = sum(county_man_hours_certified, na.rm = TRUE),
    nbr_applications_fiscal_year = sum(county_n_applications, na.rm = TRUE),
    total_hourly_wage_X_nbr_workers = sum(hourly_wage_X_nbr_workers, na.rm = TRUE)
  ) %>% 
  ungroup() %>% 
  mutate(mean_hourly_wage_fiscal_year = total_hourly_wage_X_nbr_workers/nbr_workers_certified_fiscal_year) %>% 
  select(-total_hourly_wage_X_nbr_workers)

# Combine
# Harmonize variable names
h2a_start_year_aggregated_df <- h2a_start_year_aggregated_df %>% 
  rename(year = start_year)

h2a_fiscal_year_aggregated_df <- h2a_fiscal_year_aggregated_df %>% 
  rename(year = fiscal_year)

# Harmonize missing FIPS
h2a_all_years_aggregated_df <- h2a_all_years_aggregated_df %>% 
  mutate(fips_harmonized = if_else((fips == "" | fips == "00000"), "00000", fips)) %>% 
  select(-fips) %>% 
  group_by(fips_harmonized, year) %>% 
  summarise_all(sum, na.rm = TRUE) %>% 
  ungroup()

h2a_start_year_aggregated_df <- h2a_start_year_aggregated_df %>% 
  mutate(fips_harmonized = if_else((fips == "" | fips == "00000"), "00000", fips)) %>% 
  select(-fips) %>% 
  group_by(fips_harmonized, year) %>% 
  summarise_all(sum, na.rm = TRUE) %>% 
  ungroup()

h2a_fiscal_year_aggregated_df <- h2a_fiscal_year_aggregated_df %>% 
  mutate(fips_harmonized = if_else((fips == "" | fips == "00000"), "00000", fips)) %>% 
  select(-fips) %>% 
  group_by(fips_harmonized, year) %>% 
  summarise_all(sum, na.rm = TRUE) %>% 
  ungroup()

# Merge
h2a_aggregated_df <- h2a_all_years_aggregated_df %>% 
  full_join(h2a_start_year_aggregated_df) %>% 
  full_join(h2a_fiscal_year_aggregated_df)

# Harmonize name of FIPS code variable
h2a_aggregated_df <- h2a_aggregated_df %>% 
  mutate(state_fips_code = substr(fips_harmonized, 1, 2)) %>% 
  mutate(county_fips_code = substr(fips_harmonized, 3, 5)) %>% 
  select(-fips_harmonized)

# Export
sanity_check_total_4 <- h2a_aggregated_df %>% 
  group_by(year) %>% 
  summarize(
    nbr_workers_certified = sum(nbr_workers_certified_fiscal_year, na.rm = TRUE),
  ) %>% 
  filter(nbr_workers_certified > 0)
# Numbers match, we are good

h2a_aggregated_df %>% 
  write_parquet(here("files_for_phil", "h2a_aggregated.parquet")) %>% 
  write_parquet(here("binaries", "h2a_aggregated.parquet"))

h2a_ts_df <- h2a_aggregated_df %>% 
  filter(state_fips_code != "00") %>% 
  group_by(year) %>% 
  summarise_if(is.numeric, sum, na.rm = TRUE) %>% 
  filter(year > 2007 & year < 2023)

tsplot <- ggplot(h2a_ts_df, aes(x=year, y=nbr_applications_fiscal_year)) +
  geom_line() + 
  xlab("")
tsplot

