# Purpose: Aggregate matched H-2A worksite records without duplicating case totals.
# Inputs: h2a_with_fips.parquet and h2a_addendum_b_with_fips.parquet; the AEWR
#         annual reference, 2023-rule occupation rates and 2025-rule skill-tier
#         rates for the wage-offer exposure flags.
# Outputs: data/intermediate/h2a_aggregated.parquet.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(janitor)

AEWR_OFFER_TOLERANCE <- 0.05
# A job order is filed before its contract starts, so a dated rate counts if it
# was in force at any time in the year ending on the contract start date. This
# mirrors the annual rule, which accepts the start-year rate or its lag.
AEWR_LOOKBACK_DAYS <- 365L

# A requested total is unknown if any contributing allocation is unknown.
# In particular, FY2008--2010 and FY2013--2014 lack requested worker counts.
# Do not present either an all-missing group as zero or a partial sum as a total.
SumRequested <- function(x) {
  if (any(!is.finite(x))) NA_real_ else sum(x)
}

FirstNonempty <- function(x) {
  x <- x[!is.na(x) & trimws(x) != ""]
  if (length(x) == 0L) {
    return(NA_character_)
  }
  x[[1]]
}

PlausibleCaseDate <- function(value, fiscal_year) {
  value_year <- as.integer(format(value, "%Y"))
  value[!is.na(value_year) & abs(value_year - fiscal_year) > 1L] <- as.Date(NA)
  value
}

# One row per distinct state, occupation, contract start date and hourly offer
# for which a dated AEWR existed, with whether the offer equals one of them.
# `soc_rates` apply to one occupation; `state_rates` apply to every job.
DatedAewrMatch <- function(jobs, soc_rates, state_rates, lookback_days, tolerance) {
  keys <- jobs %>%
    filter(
      !is.na(hourly_offered_wage),
      !is.na(begin_date),
      !is.na(state_fips),
      begin_date >= min(soc_rates$rate_start, state_rates$rate_start)
    ) %>%
    distinct(state_fips, soc7, begin_date, hourly_offered_wage)
  InForce <- function(candidates) {
    candidates %>%
      filter(
        rate_start <= begin_date,
        is.na(rate_end) | rate_end > begin_date - lookback_days
      )
  }
  bind_rows(
    keys %>%
      inner_join(
        soc_rates,
        by = c("state_fips", "soc7" = "soc_code"),
        relationship = "many-to-many"
      ) %>%
      InForce(),
    keys %>%
      inner_join(state_rates, by = "state_fips", relationship = "many-to-many") %>%
      InForce()
  ) %>%
    group_by(state_fips, soc7, begin_date, hourly_offered_wage) %>%
    summarise(
      dated_aewr_known = TRUE,
      at_dated_aewr = any(abs(hourly_offered_wage - dated_aewr) <= tolerance),
      .groups = "drop"
    )
}

AssertNestedHours <- function(total, hourly, aewr_observed, at_aewr, label) {
  tolerance <- 1e-8 * pmax(1, abs(total))
  checks <- cbind(
    negative_total = total < -tolerance,
    negative_hourly = hourly < -tolerance,
    negative_aewr_observed = aewr_observed < -tolerance,
    negative_at_aewr = at_aewr < -tolerance,
    hourly_exceeds_total = hourly > total + tolerance,
    observed_exceeds_hourly = aewr_observed > hourly + tolerance,
    at_aewr_exceeds_observed = at_aewr > aewr_observed + tolerance
  )
  checks[is.na(checks)] <- FALSE
  invalid <- rowSums(checks) > 0
  invalid[is.na(invalid)] <- FALSE
  if (any(invalid)) {
    failure_counts <- colSums(checks)
    failure_counts <- failure_counts[failure_counts > 0]
    stop(
      "Invalid certified-hour exposure accounting in ",
      label,
      ": ",
      paste(names(failure_counts), failure_counts, sep = "=", collapse = ", "),
      call. = FALSE
    )
  }
}

# ---- Produce workers requested and certified counts for each worksite entry ----
h2a_df <- read_parquet(path_int("h2a_with_fips.parquet")) %>%
  clean_names() %>%
  distinct(.keep_all = TRUE) %>%
  mutate(
    allocation_source_dataset = "main",
    allocation_source_row = row_number(),
    source_number_of_hours = number_of_hours
  )

addendum_b <- read_parquet(path_int(
  "h2a_addendum_b_with_fips.parquet"
)) %>%
  clean_names() %>%
  mutate(
    allocation_source_dataset = "addendum_b",
    allocation_source_row = row_number(),
    source_worksite_workers_requested = total_h2a_workers_requested
  )

employer_membership <- read_parquet(path_int(
  "h2a_employer_record_membership.parquet"
)) %>%
  clean_names()

employer_linkage_metadata_columns <- c(
  "employer_linkage_unit",
  "employer_linkage_version",
  "employer_linkage_input_max_year",
  "employer_linkage_review_status",
  "employer_linkage_review_report_hash"
)

missing_linkage_metadata <- setdiff(
  employer_linkage_metadata_columns,
  names(employer_membership)
)
if (length(missing_linkage_metadata) > 0L) {
  stop(
    "Employer membership is missing v2 provenance: ",
    paste(missing_linkage_metadata, collapse = ", "),
    call. = FALSE
  )
}

employer_linkage_contract <- employer_membership %>%
  distinct(across(all_of(employer_linkage_metadata_columns)))
if (
  nrow(employer_linkage_contract) != 1L ||
    employer_linkage_contract$employer_linkage_unit[[1]] !=
      "worksite_business" ||
    employer_linkage_contract$employer_linkage_version[[1]] != "v2" ||
    employer_linkage_contract$employer_linkage_input_max_year[[1]] != 2026L ||
    !employer_linkage_contract$employer_linkage_review_status[[1]] %in%
      c("complete", "pending") ||
    !grepl(
      "^[0-9a-f]{64}$",
      employer_linkage_contract$employer_linkage_review_report_hash[[1]]
    )
) {
  stop(
    "Employer membership does not declare one reviewed worksite-business v2 contract.",
    call. = FALSE
  )
}


# Crosswalk join keys: input column = crosswalk column
h2a_employer_keys <- c(
  "case_number" = "case_number",
  "fiscal_year" = "fiscal_year",
  "employer_name" = "source_name_raw",
  "trade_name_dba" = "source_trade_name_raw",
  "employer_address_1" = "source_address_1_raw",
  "employer_address_2" = "source_address_2_raw",
  "employer_city" = "source_city_raw",
  "employer_state" = "source_state_raw",
  "employer_postal_code" = "source_postal_code_raw",
  "employer_phone" = "source_phone_raw",
  "employer_fein" = "source_fein_raw",
  "primary_sub" = "source_primary_sub_raw",
  "organization_flag" = "source_organization_flag_raw",
  "type_of_employer_application" =
    "source_type_of_employer_application_raw",
  "ag_association_or_agency" = "source_ag_association_or_agency_raw",
  "h2a_labor_contractor" = "source_h2a_labor_contractor_raw"
)

addendum_employer_keys <- c(
  "case_number" = "case_number",
  "fiscal_year" = "fiscal_year",
  "business_name" = "source_name_raw",
  "worksite_address_1" = "source_address_1_raw",
  "worksite_address_2" = "source_address_2_raw",
  "worksite_city" = "source_city_raw",
  "worksite_state" = "source_state_raw",
  "worksite_zip" = "source_postal_code_raw"
)

employer_id_columns <- c(
  "employer_record_id",
  "employer_id_conservative",
  "employer_id_balanced",
  "employer_id_high_recall",
  "employer_record_role",
  "employer_is_worksite_business"
)

# The crosswalk represents missing source values as empty strings.
EmptyIfMissing <- function(x) {
  replace_na(as.character(x), "")
}

h2a_employer_membership <- employer_membership %>%
  filter(source_dataset == "h2a_with_fips") %>%
  select(
    all_of(unname(h2a_employer_keys)),
    all_of(employer_id_columns)
  )

addendum_employer_membership <- employer_membership %>%
  filter(source_dataset == "h2a_addendum_b_with_fips") %>%
  select(
    all_of(unname(addendum_employer_keys)),
    all_of(employer_id_columns)
  )

# Canonicalize missing values exactly as the Python notebook does.
h2a_df <- h2a_df %>%
  mutate(across(
    all_of(setdiff(names(h2a_employer_keys), "fiscal_year")),
    EmptyIfMissing
  )) %>%
  left_join(
    h2a_employer_membership,
    by = h2a_employer_keys,
    relationship = "many-to-one"
  )

addendum_b <- addendum_b %>%
  mutate(across(
    all_of(setdiff(names(addendum_employer_keys), "fiscal_year")),
    EmptyIfMissing
  )) %>%
  left_join(
    addendum_employer_membership,
    by = addendum_employer_keys,
    relationship = "many-to-one"
  )

if (
  any(is.na(h2a_df$employer_record_role)) ||
    any(is.na(addendum_b$employer_record_role))
) {
  stop(
    "The v2 role map does not cover every H-2A and Addendum B source row.",
    call. = FALSE
  )
}
for (column in c(
  "employer_id_conservative",
  "employer_id_balanced",
  "employer_id_high_recall"
)) {
  if (
    any(
      h2a_df$employer_is_worksite_business & is.na(h2a_df[[column]])
    ) ||
      any(
        addendum_b$employer_is_worksite_business &
          is.na(addendum_b[[column]])
      )
  ) {
    stop(
      "Named included worksite businesses are missing ",
      column,
      ".",
      call. = FALSE
    )
  }
}


# The annual reference is the rate for every non-range job until the rule of
# 2023-03-30, and for the common field-and-livestock occupations afterwards.
aewr_rates <- read_parquet(path_int("aewr.parquet")) %>%
  transmute(
    state_fips = state_fips(state_fips),
    year = as.integer(year),
    nominal_aewr = as.numeric(aewr)
  ) %>%
  distinct() %>%
  arrange(state_fips, year) %>%
  group_by(state_fips) %>%
  mutate(lag_nominal_aewr = lag(nominal_aewr)) %>%
  ungroup()

if (anyDuplicated(aewr_rates[c("state_fips", "year")]) > 0L) {
  stop("AEWR rates must have unique state-year keys.", call. = FALSE)
}

# Dated rates that the annual reference does not carry:
#   2023 rule: statewide rates for occupations outside the common group
#     (2023-03-30 until the vacatur was implemented on 2025-08-29);
#   2025 rule: skill-tier rates offered to U.S. workers, for the combined
#     field-and-livestock group and for single occupations.
# Disclosed offers are U.S.-worker rates, so H-2A cash rates are not candidates.
aewr_skill_rates <- read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
  filter(worker_category == "US", !is.na(hourly_aewr)) %>%
  transmute(
    state_fips = state_fips(state_fips),
    soc_code,
    dated_aewr = as.numeric(hourly_aewr),
    rate_start = as.Date(effective_start),
    rate_end = as.Date(effective_end_exclusive)
  )
aewr_dated_soc_rates <- read_parquet(path_int("aewr_oews_occupation_rates.parquet")) %>%
  filter(!is.na(hourly_aewr)) %>%
  transmute(
    state_fips = state_fips(state_fips),
    soc_code,
    dated_aewr = as.numeric(hourly_aewr),
    rate_start = as.Date(effective_start),
    rate_end = as.Date(effective_end_exclusive)
  ) %>%
  bind_rows(aewr_skill_rates %>% filter(soc_code != "FIELD_LIVESTOCK_COMBINED")) %>%
  distinct()
aewr_dated_state_rates <- aewr_skill_rates %>%
  filter(soc_code == "FIELD_LIVESTOCK_COMBINED") %>%
  select(-soc_code) %>%
  distinct()

if (
  nrow(aewr_dated_soc_rates) == 0L ||
    nrow(aewr_dated_state_rates) == 0L ||
    anyNA(aewr_dated_soc_rates$rate_start) ||
    anyNA(aewr_dated_state_rates$rate_start)
) {
  stop("Dated AEWR rate tables are empty or lack effective dates.", call. = FALSE)
}

# Handle master entries that include worker counts in all sub entries
# Split cases into those with duplicated worker counts in sub-entries and those without
h2a_df <- h2a_df %>%
  mutate(nbr_workers_certified = as.numeric(nbr_workers_certified)) %>%
  mutate(nbr_workers_requested = as.numeric(nbr_workers_requested)) %>%
  mutate(nbr_workers_needed = as.numeric(nbr_workers_needed))

# Preserve every application before the worksite branch drops zero-certified
# records. This branch is collapsed to county-year moments later; it is not
# exported as a separate case-level dataset.
h2a_case_source <- h2a_df

# One occupation code per case, from the case record. Addendum B worksite rows
# carry none of their own, so the wage-offer flags read it from here.
case_soc <- h2a_case_source %>%
  transmute(
    case_number,
    fiscal_year,
    soc7 = substr(gsub("[^0-9-]", "", coalesce(soc_code, "")), 1, 7)
  ) %>%
  filter(soc7 != "") %>%
  distinct()

if (anyDuplicated(case_soc[c("case_number", "fiscal_year")]) > 0L) {
  stop("A case carries more than one occupation code.", call. = FALSE)
}

# Source coverage includes applications absent from the positive-certified
# allocation branch. This audit is not a second producer of case allocations.
write_parquet(
  h2a_case_source %>% select(
    case_number, fiscal_year, allocation_source_row, primary_sub,
    organization_flag, nbr_workers_certified, nbr_workers_requested,
    county_fips_list, case_status, certification_begin_date,
    certification_end_date, employment_begin_date, employment_end_date,
    job_order_begin_date, job_order_end_date, requested_begin_date, requested_end_date,
    period_begin_date, period_end_date, period_date_basis,
    application_begin_date, application_date_basis
  ),
  path_int("h2a_case_allocation_source.parquet")
)

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
  filter(
    (n_entries == 1 & fiscal_year < 2020) | (!has_addendum & fiscal_year > 2019)
  )

h2a_duped_pre2020 <- h2a_df %>%
  filter(n_entries > 1 & fiscal_year < 2020)

h2a_duped_post2020 <- h2a_df %>%
  filter(has_addendum & fiscal_year > 2019)

# First deal with years prior to 2020
# Only the master entry has an organization flag
# From 2016 to 2019, we can also use the primary_sub indicator to identify master and sub entries
h2a_duped_pre2020 <- h2a_duped_pre2020 %>%
  mutate(
    entry_type = case_when(
      (fiscal_year < 2016) & (organization_flag != '') ~ "master",
      (fiscal_year > 2015) &
        (fiscal_year < 2020) &
        (primary_sub == "PRI") ~ "master",
      .default = "sub"
    )
  )

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
  select(
    case_number,
    fiscal_year,
    master_n_workers_certified,
    master_n_workers_requested,
    organization_flag
  ) %>%
  rename(application_type = organization_flag)

# Sub entry worker count and types of sub entries in each case
sub_entries <- h2a_duped_pre2020 %>%
  filter(entry_type == "sub") %>%
  group_by(case_number, fiscal_year) %>%
  mutate(sub_n_workers_certified = sum(nbr_workers_certified, na.rm = TRUE)) %>%
  mutate(sub_n_workers_requested = sum(nbr_workers_requested, na.rm = TRUE)) %>%
  ungroup() %>%
  select(
    case_number,
    fiscal_year,
    sub_n_workers_certified,
    sub_n_workers_requested
  ) %>%
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
  "Association - Joint Employer (H-2A Only)",
  "Association - Sole Employer (H-2A Only)",
  "H-2A Labor Contractor or Job Contractor",
  "Association - Filing as Agent (H-2A Only)"
)

# Decide on how we want to perform adjustment
h2a_duped_pre2020 <- h2a_duped_pre2020 %>%
  mutate(
    adjustment_method_certified = case_when(
      application_type %in%
        drop_master_list &
        sub_n_workers_certified > 0 ~ "inflate_sub_workers_drop_master",
      application_type %in%
        drop_master_list &
        sub_n_workers_certified ==
          0 ~ "distribute_master_workers_across_sub_entries_drop_master",
      !(application_type %in% drop_master_list) &
        sub_n_workers_certified ==
          0 ~ "distribute_master_workers_across_all_entries_evenly", # all sub entries are NA
      !(application_type %in% drop_master_list) &
        sub_n_workers_certified >
          0 ~ "distribute_master_workers_across_all_entries_fractionally" # distribute master worker count fractionally across entries
    )
  )

# Same adjustment for requested worker counts
h2a_duped_pre2020 <- h2a_duped_pre2020 %>%
  mutate(
    adjustment_method_requested = case_when(
      application_type %in%
        drop_master_list &
        sub_n_workers_requested > 0 ~ "inflate_sub_workers_drop_master",
      application_type %in%
        drop_master_list &
        sub_n_workers_requested ==
          0 ~ "distribute_master_workers_across_sub_entries_drop_master",
      !(application_type %in% drop_master_list) &
        sub_n_workers_requested ==
          0 ~ "distribute_master_workers_across_all_entries_evenly",
      !(application_type %in% drop_master_list) &
        sub_n_workers_requested >
          0 ~ "distribute_master_workers_across_all_entries_fractionally"
    )
  )

# Perform adjustment
h2a_duped_pre2020 <- h2a_duped_pre2020 %>%
  mutate(
    master_sub_ratio_certified = master_n_workers_certified /
      sub_n_workers_certified
  ) %>%
  mutate(
    adjusted_nbr_workers_certified = case_when(
      adjustment_method_certified ==
        "inflate_sub_workers_drop_master" ~ nbr_workers_certified *
        master_sub_ratio_certified,
      adjustment_method_certified ==
        "distribute_master_workers_across_sub_entries_drop_master" ~ master_n_workers_certified /
        (n_entries - 1),
      adjustment_method_certified ==
        "distribute_master_workers_across_all_entries_evenly" ~ master_n_workers_certified /
        n_entries,
      adjustment_method_certified ==
        "distribute_master_workers_across_all_entries_fractionally" ~ master_n_workers_certified *
        (nbr_workers_certified /
          (master_n_workers_certified + sub_n_workers_certified))
    )
  )

h2a_duped_pre2020 <- h2a_duped_pre2020 %>%
  mutate(
    master_sub_ratio_requested = master_n_workers_requested /
      sub_n_workers_requested
  ) %>%
  mutate(
    adjusted_nbr_workers_requested = case_when(
      adjustment_method_requested ==
        "inflate_sub_workers_drop_master" ~ nbr_workers_requested *
        master_sub_ratio_requested,
      adjustment_method_requested ==
        "distribute_master_workers_across_sub_entries_drop_master" ~ master_n_workers_requested /
        (n_entries - 1),
      adjustment_method_requested ==
        "distribute_master_workers_across_all_entries_evenly" ~ master_n_workers_requested /
        n_entries,
      adjustment_method_requested ==
        "distribute_master_workers_across_all_entries_fractionally" ~ master_n_workers_requested *
        (nbr_workers_requested /
          (master_n_workers_requested + sub_n_workers_requested))
    )
  )

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
  mutate(
    nbr_workers_requested = as.numeric(total_h2a_workers_requested),
    # Nine FY2022--2023 addendum rows contain impossible negative counts.
    # Treat them like the other missing worksite weights and impute below.
    nbr_workers_requested = if_else(
      nbr_workers_requested < 0,
      NA_real_,
      nbr_workers_requested
    )
  ) %>%
  select(-total_h2a_workers_requested) %>%
  mutate(entry_type = "sub")

# For entries that were denied, number certified is NA, fill these in with 0
h2a_duped_post2020 <- h2a_duped_post2020 %>%
  mutate(
    nbr_workers_certified = if_else(
      is.na(nbr_workers_certified),
      0,
      nbr_workers_certified
    )
  )

# Since the master entries have the correct sums, we use those as the case totals
h2a_duped_post2020 <- h2a_duped_post2020 %>%
  mutate(case_nbr_workers_certified = nbr_workers_certified) %>%
  mutate(case_nbr_workers_requested = nbr_workers_requested) %>%
  bind_rows(addendum_b) %>%
  group_by(case_number, fiscal_year) %>%
  fill(
    case_nbr_workers_requested,
    case_nbr_workers_certified,
    type_of_employer_application
  ) %>%
  ungroup() %>%
  arrange(fiscal_year, case_number)

# For the sub-entries, fill in the application status, dates, wage rates, wage units, and hours of work, which is the same as in the master entry
h2a_duped_post2020 <- h2a_duped_post2020 %>%
  group_by(case_number, fiscal_year) %>%
  fill(
    case_status,
    number_of_hours,
    wage_rate,
    wage_unit,
    certification_begin_date, certification_end_date,
    employment_begin_date, employment_end_date,
    job_order_begin_date, job_order_end_date,
    requested_begin_date, requested_end_date,
    period_begin_date, period_end_date, period_date_basis
  ) %>%
  ungroup()

# Use addendum B requested worker counts as weights
# Total weight depends on whether we keep master entry or not
application_types <- h2a_duped_post2020 %>%
  distinct(type_of_employer_application)

drop_master_list <- c(
  "Association - Agent",
  "Association - Sole Employer",
  "Association - Joint Employer",
  "Association - Filing as Agent (H-2A Only)",
  "Association - Sole Employer (H-2A Only)",
  "Association - Joint Employer (H-2A Only)"
)

# We use the number of workers requested for each worksite as the weights
# Have to fill in requested worker count for addendum B entries with missing counts
# This is because all addendum B entries are worksites, since housing is in separate addendum B housing record file
# Use the mean of the available entries
weights <- h2a_duped_post2020 %>%
  filter(
    !((type_of_employer_application %in% drop_master_list) &
      (entry_type == "master"))
  ) %>%
  group_by(case_number, fiscal_year) %>%
  mutate(
    mean_nbr_workers_requested_excl_drop_master = mean(
      nbr_workers_requested,
      na.rm = TRUE
    )
  ) %>%
  ungroup() %>%
  mutate(
    weight_excl_drop_master = if_else(
      !is.na(nbr_workers_requested),
      nbr_workers_requested,
      mean_nbr_workers_requested_excl_drop_master
    )
  ) %>%
  group_by(case_number, fiscal_year) %>%
  mutate(total_weight_excl_drop_master = sum(weight_excl_drop_master)) %>%
  ungroup() %>%
  select(
    case_number,
    fiscal_year,
    mean_nbr_workers_requested_excl_drop_master,
    total_weight_excl_drop_master
  ) %>%
  distinct()

# Add total weights (calculated excluding master where appropriate) back to original dataset
# Calculate final weights to use
h2a_duped_post2020 <- h2a_duped_post2020 %>%
  left_join(weights) %>%
  mutate(
    weight = if_else(
      !is.na(nbr_workers_requested),
      nbr_workers_requested,
      mean_nbr_workers_requested_excl_drop_master
    )
  )

# There are still entries with missing weights
# These are sub-entries with all NA sub-entries but with master entries that have to be dropped
# Fill in the missing weight using the weight in the master entry, i.e., equal sub-entry weights
h2a_duped_post2020 <- h2a_duped_post2020 %>%
  fill(weight, .direction = "down")

# Check weights are correct
h2a_duped_post2020 <- h2a_duped_post2020 %>%
  filter(
    !((type_of_employer_application %in% drop_master_list) &
      (entry_type == "master"))
  ) %>%
  mutate(
    weight = if_else(
      is.na(weight),
      mean_nbr_workers_requested_excl_drop_master,
      weight
    )
  ) %>%
  group_by(case_number, fiscal_year) %>%
  mutate(total_weight = sum(weight, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(
    total_weight = if_else(
      is.na(total_weight_excl_drop_master),
      total_weight,
      total_weight_excl_drop_master
    )
  ) %>%
  mutate(weight_ratio = weight / total_weight) %>%
  group_by(case_number, fiscal_year) %>%
  mutate(total_weight_ratio_check = sum(weight_ratio)) %>%
  ungroup() %>%
  arrange(total_weight_ratio_check, fiscal_year, case_number)

# Define adjusted worker counts
h2a_duped_post2020 <- h2a_duped_post2020 %>%
  mutate(
    adjusted_nbr_workers_certified = case_nbr_workers_certified * weight_ratio
  ) %>%
  mutate(
    adjusted_nbr_workers_requested = case_nbr_workers_requested * weight_ratio
  )

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

# ---- Construct case-level moments before county aggregation ----
# For applications represented in the allocated worksite data, use the same
# geography produced by the existing master/sub-entry logic. For applications
# absent from that branch (principally denied/withdrawn or otherwise
# zero-certified cases), fall back to their disclosure-file geography.
case_geography_from_allocated <- h2a_combined %>%
  select(case_number, fiscal_year, county_fips_list)

cases_with_allocated_geography <- case_geography_from_allocated %>%
  distinct(case_number, fiscal_year)

case_geography_fallback <- h2a_case_source %>%
  anti_join(
    cases_with_allocated_geography,
    by = c("case_number", "fiscal_year")
  ) %>%
  select(case_number, fiscal_year, county_fips_list)

h2a_case_geography <- bind_rows(
  case_geography_from_allocated,
  case_geography_fallback
) %>%
  mutate(
    county_fips_list = if_else(
      is.na(county_fips_list) | county_fips_list == "",
      "00000",
      county_fips_list
    )
  ) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(
    county_fips_list = trimws(county_fips_list),
    county_fips_list = if_else(
      county_fips_list == "",
      "00000",
      county_fips_list
    ),
    county_fips = harmonize_county_fips_2010(county_fips_list)
  ) %>%
  distinct(case_number, fiscal_year, county_fips) %>%
  group_by(case_number, fiscal_year) %>%
  mutate(case_county_weight = 1 / n()) %>%
  ungroup()

h2a_case_df <- h2a_case_source %>%
  group_by(case_number, fiscal_year) %>%
  summarise(
    case_status = FirstNonempty(case_status),
    case_received_date = FirstNonempty(case_received_date),
    decision_date = FirstNonempty(decision_date),
    application_begin_date = FirstNonempty(application_begin_date),
    emergency_filing = FirstNonempty(emergency_filing),
    .groups = "drop"
  ) %>%
  mutate(
    across(
      c(
        case_received_date,
        decision_date,
        application_begin_date
      ),
      \(value) as.Date(value)
    ),
    across(
      c(
        case_received_date,
        decision_date,
        application_begin_date
      ),
      \(value) PlausibleCaseDate(value, fiscal_year)
    ),
    case_begin_date = application_begin_date,
    case_start_year = as.integer(format(case_begin_date, "%Y")),
    case_start_year = coalesce(case_start_year, as.integer(fiscal_year)),
    case_status_upper = str_to_upper(case_status),
    case_status_harmonized = case_when(
      str_detect(case_status_upper, "PARTIAL") ~ "partial_certification",
      str_detect(case_status_upper, "DENIED") ~ "denied",
      str_detect(case_status_upper, "WITHDRAWN") ~ "withdrawn",
      str_detect(case_status_upper, "CERTIF") ~ "certified",
      TRUE ~ "other"
    ),
    emergency_application = case_when(
      str_to_upper(emergency_filing) == "Y" ~ TRUE,
      str_to_upper(emergency_filing) == "N" ~ FALSE,
      TRUE ~ NA
    ),
    receipt_to_start_days = as.numeric(
      case_begin_date - case_received_date
    ),
    decision_to_start_days = as.numeric(
      case_begin_date - decision_date
    )
  ) %>%
  select(
    case_number,
    fiscal_year,
    case_start_year,
    case_status_harmonized,
    emergency_application,
    receipt_to_start_days,
    decision_to_start_days
  )

h2a_case_start_year_aggregated_df <- h2a_case_df %>%
  inner_join(
    h2a_case_geography,
    by = c("case_number", "fiscal_year"),
    relationship = "one-to-many"
  ) %>%
  mutate(
    receipt_timing_weight = if_else(
      is.na(receipt_to_start_days),
      0,
      case_county_weight
    ),
    decision_timing_weight = if_else(
      is.na(decision_to_start_days),
      0,
      case_county_weight
    ),
    emergency_observed_weight = if_else(
      is.na(emergency_application),
      0,
      case_county_weight
    )
  ) %>%
  group_by(county_fips, year = case_start_year) %>%
  summarise(
    nbr_applications_case_start_year = sum(case_county_weight),
    nbr_applications_certified_start_year = sum(
      case_county_weight *
        (case_status_harmonized %in%
          c("certified", "partial_certification"))
    ),
    nbr_applications_partial_start_year = sum(
      case_county_weight *
        (case_status_harmonized == "partial_certification")
    ),
    nbr_applications_denied_start_year = sum(
      case_county_weight * (case_status_harmonized == "denied")
    ),
    nbr_applications_withdrawn_start_year = sum(
      case_county_weight * (case_status_harmonized == "withdrawn")
    ),
    nbr_applications_emergency_start_year_observed = sum(
      case_county_weight * coalesce(emergency_application, FALSE)
    ),
    emergency_observed_weight = sum(emergency_observed_weight),
    receipt_to_start_days_weighted = sum(
      case_county_weight * coalesce(receipt_to_start_days, 0)
    ),
    receipt_timing_weight = sum(receipt_timing_weight),
    decision_to_start_days_weighted = sum(
      case_county_weight * coalesce(decision_to_start_days, 0)
    ),
    decision_timing_weight = sum(decision_timing_weight),
    .groups = "drop"
  ) %>%
  mutate(
    nbr_applications_emergency_start_year = if_else(
      year >= 2021 &
        near(
          emergency_observed_weight,
          nbr_applications_case_start_year
        ),
      nbr_applications_emergency_start_year_observed,
      NA_real_
    ),
    mean_receipt_to_start_days_start_year = if_else(
      receipt_timing_weight > 0,
      receipt_to_start_days_weighted / receipt_timing_weight,
      NA_real_
    ),
    mean_decision_to_start_days_start_year = if_else(
      decision_timing_weight > 0,
      decision_to_start_days_weighted / decision_timing_weight,
      NA_real_
    )
  ) %>%
  select(
    -nbr_applications_emergency_start_year_observed,
    -emergency_observed_weight,
    -receipt_to_start_days_weighted,
    -receipt_timing_weight,
    -decision_to_start_days_weighted,
    -decision_timing_weight
  )

# ---- Clean before aggregating into county years ----
h2a_combined <- h2a_combined %>%
  mutate(
    county_fips_list = if_else(
      is.na(county_fips_list),
      "00000",
      county_fips_list
    )
  ) %>%
  mutate(
    county_fips_list = if_else(
      county_fips_list == "",
      "00000",
      county_fips_list
    )
  )

# Add application status
h2a_combined <- h2a_combined %>%
  mutate(
    case_status_harmonized = if_else(
      grepl('CERTIFICATION|CERTIFICATION|CERTIFIED', case_status),
      "certified",
      ""
    )
  ) %>%
  mutate(
    case_status_harmonized = if_else(
      grepl('DENIED', case_status),
      "denied",
      case_status_harmonized
    )
  ) %>%
  mutate(
    case_status_harmonized = if_else(
      grepl('WITHDRAWN', case_status),
      "withdrawn",
      case_status_harmonized
    )
  ) %>%
  mutate(
    case_status_harmonized = if_else(
      grepl('PARTIAL', case_status),
      "partial_certification",
      case_status_harmonized
    )
  ) %>%
  arrange(case_status_harmonized)

# For withdrawn and denied cases, number certified is 0
h2a_combined <- h2a_combined %>%
  mutate(
    adjusted_nbr_workers_certified = if_else(
      case_status_harmonized == "withdrawn" |
        case_status_harmonized == "denied",
      0,
      adjusted_nbr_workers_certified
    )
  )

# docs-ground:start h2a-worksite-allocation-audit
# Preserve allocation and the source-owned employment-period date basis before
# date exclusion, hours imputation, or geographic splitting. The separate
# worksite-month producer prefers site dates and labels its case-date fallback.
write_parquet(
  h2a_combined %>%
    mutate(
      allocation_id = paste0(allocation_source_dataset, ":", allocation_source_row),
      allocation_method = case_when(
        !is.na(adjustment_method_certified) ~ adjustment_method_certified,
        !is.na(weight_ratio) & is.na(nbr_workers_requested) ~
          "case_certifications_requested_staffing_imputed",
        !is.na(weight_ratio) ~ "case_certifications_requested_staffing_observed",
        TRUE ~ "single_disclosed_record"
      )
    ) %>%
    select(
      allocation_id, allocation_source_dataset, allocation_source_row,
      case_number, fiscal_year, county_fips_list, allocation_method,
      adjusted_nbr_workers_certified, adjusted_nbr_workers_requested,
      nbr_workers_requested, source_worksite_workers_requested,
      case_nbr_workers_certified, master_n_workers_certified,
      certification_begin_date, certification_end_date,
      employment_begin_date, employment_end_date,
      job_order_begin_date, job_order_end_date, requested_begin_date, requested_end_date,
      period_begin_date, period_end_date, period_date_basis,
      worksite_requested_begin_date, worksite_requested_end_date,
      number_of_hours, source_number_of_hours, wage_rate, wage_unit,
      soc_code, case_status_harmonized, employer_record_role,
      employer_is_worksite_business, employer_id_balanced,
      worksite_address_1, worksite_address_2, worksite_city,
      worksite_state, worksite_zip
    ),
  path_int("h2a_allocated_worksites.parquet")
)
# docs-ground:end h2a-worksite-allocation-audit

# Consume the source-owned whole-pair selection. No job-order dates, modal
# replacements, or hand repairs enter employment timing. Start-only rows count
# toward start-year positions/businesses; duration measures require an end date.
h2a_combined <- h2a_combined %>%
  mutate(
    begin_date = as.Date(period_begin_date),
    end_date = as.Date(period_end_date)
  ) %>%
  filter(!is.na(begin_date))

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

# ---- Aggregate to county-year level ----
# Calculate man-hours employed
# We impute the number of hours worked per week for those missing it with the mean, which is 40.1

h2a_cleaned_df <- h2a_combined %>%
  mutate(number_of_hours = as.numeric(number_of_hours)) %>%
  mutate(mean_number_of_hours = mean(number_of_hours, na.rm = TRUE)) %>%
  mutate(
    number_of_hours = if_else(
      is.na(number_of_hours),
      mean_number_of_hours,
      number_of_hours
    )
  ) %>%
  mutate(
    man_hours_certified = (as.numeric(as.Date(end_date) - as.Date(begin_date)) /
      7) *
      number_of_hours *
      nbr_workers_certified
  ) %>%
  mutate(
    man_hours_requested = (as.numeric(as.Date(end_date) - as.Date(begin_date)) /
      7) *
      number_of_hours *
      nbr_workers_requested
  )

# Keep only variables we care about
h2a_cleaned_df <- h2a_cleaned_df %>%
  select(
    case_number,
    fiscal_year,
    county_fips_list,
    begin_date,
    end_date,
    nbr_workers_requested,
    nbr_workers_certified,
    man_hours_requested,
    man_hours_certified,
    number_of_hours,
    wage_rate,
    wage_unit,
    all_of(employer_id_columns)
  )

# We want to calculate the average wage rate of H-2A workers within each county-year
# We need to select a cutoff for distinguishing hourly wage values from non-hourly
# Note that unit of pay does not actually correspond to the rate of pay, even though they should
# Assume that those with reported wage rates above $100 are non-hourly wages instead
# Both wage definitions share the (0, 100] plausibility range; hourly_offered_wage
# additionally requires an hourly pay unit
HOURLY_WAGE_MAX <- 100
h2a_cleaned_df <- h2a_cleaned_df %>%
  mutate(wage_rate = as.numeric(wage_rate))

h2a_cleaned_df <- h2a_cleaned_df %>%
  mutate(
    hourly_wage = case_when(
      wage_rate > 0 & wage_rate <= HOURLY_WAGE_MAX ~ wage_rate,
      .default = NaN
    ),
    wage_unit_harmonized = str_to_upper(str_trim(wage_unit)),
    hourly_offered_wage = case_when(
      wage_unit_harmonized %in%
        c("HOUR", "HOURLY", "HR") &
        is.finite(wage_rate) &
        wage_rate > 0 &
        wage_rate <= HOURLY_WAGE_MAX ~ wage_rate,
      TRUE ~ NA_real_
    )
  )

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
  filter(!is.na(end_date)) %>%
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
  mutate(
    year_int = interval(
      as.Date(paste0(year, '-01-01')),
      as.Date(paste0(year, '-12-31'))
    )
  ) %>%
  mutate(year_sect = intersect(date_int, year_int)) %>% # intersecting dates within each calendar year
  mutate(start_new = as.Date(int_start(year_sect))) %>% # start date of intersection
  mutate(end_new = as.Date(int_end(year_sect))) %>% # end date of intersection
  mutate(year = year(start_new)) %>% # year of intersection
  mutate(days_in_year = as.numeric(end_new - start_new) + 1) # number of days in intersection

# Calculate year-intersection weighted workers and man-hours
h2a_all_years_df <- h2a_all_years_df %>%
  mutate(
    year_weighted_nbr_workers_requested = (days_in_year / total_days) *
      nbr_workers_requested
  ) %>%
  mutate(
    year_weighted_nbr_workers_certified = (days_in_year / total_days) *
      nbr_workers_certified
  ) %>%
  mutate(
    year_weighted_man_hours_requested = (days_in_year / total_days) *
      man_hours_requested
  ) %>%
  mutate(
    year_weighted_man_hours_certified = (days_in_year / total_days) *
      man_hours_certified
  ) %>%
  mutate(
    year_weighted_n_applications = (days_in_year / total_days) * 1 / n_entries
  )

# For multi-county entries, equally split across all counties equally
h2a_all_years_df <- h2a_all_years_df %>%
  mutate(n_counties = str_count(county_fips_list, ",") + 1) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(
    county_year_weighted_nbr_workers_requested = year_weighted_nbr_workers_requested /
      n_counties
  ) %>%
  mutate(
    county_year_weighted_nbr_workers_certified = year_weighted_nbr_workers_certified /
      n_counties
  ) %>%
  mutate(
    county_year_weighted_man_hours_requested = year_weighted_man_hours_requested /
      n_counties
  ) %>%
  mutate(
    county_year_weighted_man_hours_certified = year_weighted_man_hours_certified /
      n_counties
  ) %>%
  mutate(
    county_year_weighted_n_applications = year_weighted_n_applications /
      n_counties
  )

# We have to manually calculate weighted average of hourly wage because R cannot handle NAs in weights
h2a_all_years_df <- h2a_all_years_df %>%
  mutate(
    hourly_wage_X_nbr_workers = hourly_wage *
      county_year_weighted_nbr_workers_certified,
    nbr_workers_with_hourly_wage = if_else(
      !is.na(hourly_wage),
      county_year_weighted_nbr_workers_certified,
      0
    )
  )

# Collapse by county-year
h2a_all_years_aggregated_df <- h2a_all_years_df %>%
  group_by(county_fips_list, year) %>%
  summarise(
    nbr_workers_requested_all_years = SumRequested(
      county_year_weighted_nbr_workers_requested
    ),
    nbr_workers_certified_all_years = sum(
      county_year_weighted_nbr_workers_certified,
      na.rm = TRUE
    ),
    man_hours_requested_all_years = SumRequested(
      county_year_weighted_man_hours_requested
    ),
    man_hours_certified_all_years = sum(
      county_year_weighted_man_hours_certified,
      na.rm = TRUE
    ),
    nbr_applications_all_years = sum(
      county_year_weighted_n_applications,
      na.rm = TRUE
    ),
    total_hourly_wage_X_nbr_workers = sum(
      hourly_wage_X_nbr_workers,
      na.rm = TRUE
    ),
    total_nbr_workers_with_hourly_wage = sum(
      nbr_workers_with_hourly_wage,
      na.rm = TRUE
    )
  ) %>%
  ungroup() %>%
  mutate(
    # Average over certified workers with a valid hourly wage only
    mean_hourly_wage_all_years = if_else(
      total_nbr_workers_with_hourly_wage > 0,
      total_hourly_wage_X_nbr_workers / total_nbr_workers_with_hourly_wage,
      NA_real_
    )
  ) %>%
  select(-total_hourly_wage_X_nbr_workers, -total_nbr_workers_with_hourly_wage)

test <- h2a_all_years_aggregated_df %>%
  arrange(year)

# Calculate same variables, but front loaded into the start year of each case
h2a_start_year_df <- h2a_cleaned_df %>%
  mutate(start_year = year(begin_date))

# For multi-county entries, equally split workers and man-hours across all counties equally
h2a_start_year_df <- h2a_start_year_df %>%
  mutate(n_counties = str_count(county_fips_list, ",") + 1) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(
    county_fips_list = trimws(county_fips_list),
    state_fips = state_from_county_fips(county_fips_list)
  ) %>%
  left_join(
    case_soc,
    by = c("case_number", "fiscal_year"),
    relationship = "many-to-one"
  ) %>%
  mutate(soc7 = coalesce(soc7, "")) %>%
  left_join(
    aewr_rates,
    by = c("state_fips", "start_year" = "year"),
    relationship = "many-to-one"
  )

dated_aewr_match <- DatedAewrMatch(
  h2a_start_year_df,
  aewr_dated_soc_rates,
  aewr_dated_state_rates,
  AEWR_LOOKBACK_DAYS,
  AEWR_OFFER_TOLERANCE
)

h2a_start_year_df <- h2a_start_year_df %>%
  left_join(
    dated_aewr_match,
    by = c("state_fips", "soc7", "begin_date", "hourly_offered_wage"),
    relationship = "many-to-one"
  ) %>%
  mutate(
    dated_aewr_known = coalesce(dated_aewr_known, FALSE),
    at_dated_aewr = coalesce(at_dated_aewr, FALSE)
  ) %>%
  mutate(county_nbr_workers_requested = nbr_workers_requested / n_counties) %>%
  mutate(county_nbr_workers_certified = nbr_workers_certified / n_counties) %>%
  mutate(county_man_hours_requested = man_hours_requested / n_counties) %>%
  mutate(county_man_hours_certified = man_hours_certified / n_counties) %>%
  mutate(
    county_n_applications = (1 / n_entries) / n_counties,
    # Job orders can be filed before the annual rate update, so an offer at
    # either the contract start-year AEWR or its lag is AEWR-exposed. From
    # 2023-03-30 an offer at a dated occupation or skill-tier rate in force
    # for the job is AEWR-exposed as well. No dated rate exists before then,
    # so earlier years depend on the annual rule alone.
    cert_hours_with_hourly_wage = if_else(
      !is.na(hourly_offered_wage),
      county_man_hours_certified,
      0
    ),
    cert_hours_with_hourly_wage_and_aewr = if_else(
      !is.na(hourly_offered_wage) &
        (!is.na(nominal_aewr) | !is.na(lag_nominal_aewr) | dated_aewr_known),
      county_man_hours_certified,
      0
    ),
    cert_hours_at_aewr = if_else(
      !is.na(hourly_offered_wage) &
        ((!is.na(nominal_aewr) &
          abs(hourly_offered_wage - nominal_aewr) <= AEWR_OFFER_TOLERANCE) |
          (!is.na(lag_nominal_aewr) &
            abs(hourly_offered_wage - lag_nominal_aewr) <=
              AEWR_OFFER_TOLERANCE) |
          at_dated_aewr),
      county_man_hours_certified,
      0
    ),
    nominal_offered_wage_bill_certified = coalesce(
      hourly_offered_wage * county_man_hours_certified,
      0
    )
  )

AssertNestedHours(
  h2a_start_year_df$county_man_hours_certified,
  h2a_start_year_df$cert_hours_with_hourly_wage,
  h2a_start_year_df$cert_hours_with_hourly_wage_and_aewr,
  h2a_start_year_df$cert_hours_at_aewr,
  "allocated worksite rows"
)

# Construct employer counts separately from the additive worker totals.
h2a_start_year_employer_counts <- h2a_start_year_df %>%
  mutate(
    county_fips = harmonize_county_fips_2010(if_else(
      county_fips_list == "" | county_fips_list == "00000",
      "00000",
      county_fips_list
    )),
    year = as.integer(start_year)
  ) %>%
  filter(
    employer_is_worksite_business,
    is.finite(county_nbr_workers_certified),
    county_nbr_workers_certified > 0
  ) %>%
  group_by(county_fips, year) %>%
  summarise(
    nbr_employers_conservative_start_year = n_distinct(
      employer_id_conservative,
      na.rm = TRUE
    ),
    nbr_employers_balanced_start_year = n_distinct(
      employer_id_balanced,
      na.rm = TRUE
    ),
    nbr_employers_high_recall_start_year = n_distinct(
      employer_id_high_recall,
      na.rm = TRUE
    ),
    .groups = "drop"
  )

# The matching tiers are nested, so these inequalities must hold.
stopifnot(
  all(
    h2a_start_year_employer_counts$nbr_employers_conservative_start_year >=
      h2a_start_year_employer_counts$nbr_employers_balanced_start_year
  ),
  all(
    h2a_start_year_employer_counts$nbr_employers_balanced_start_year >=
      h2a_start_year_employer_counts$nbr_employers_high_recall_start_year
  )
)

# We have to manually calculate weighted average of hourly wage because R cannot handle NAs in weights
h2a_start_year_df <- h2a_start_year_df %>%
  mutate(
    hourly_wage_X_nbr_workers = hourly_wage * county_nbr_workers_certified,
    nbr_workers_with_hourly_wage = if_else(
      !is.na(hourly_wage),
      county_nbr_workers_certified,
      0
    )
  )

# Collapse by county-year
h2a_start_year_aggregated_df <- h2a_start_year_df %>%
  group_by(county_fips_list, start_year) %>%
  summarise(
    nbr_workers_requested_start_year = SumRequested(
      county_nbr_workers_requested
    ),
    nbr_workers_certified_start_year = sum(
      county_nbr_workers_certified,
      na.rm = TRUE
    ),
    man_hours_requested_start_year = SumRequested(
      county_man_hours_requested
    ),
    man_hours_certified_start_year = sum(
      county_man_hours_certified,
      na.rm = TRUE
    ),
    cert_hours_with_hourly_wage_start_year = sum(
      cert_hours_with_hourly_wage,
      na.rm = TRUE
    ),
    cert_hours_with_hourly_wage_and_aewr_start_year = sum(
      cert_hours_with_hourly_wage_and_aewr,
      na.rm = TRUE
    ),
    cert_hours_at_aewr_start_year = sum(
      cert_hours_at_aewr,
      na.rm = TRUE
    ),
    nominal_offered_wage_bill_certified_start_year = sum(
      nominal_offered_wage_bill_certified,
      na.rm = TRUE
    ),
    nbr_applications_start_year = sum(county_n_applications, na.rm = TRUE),
    total_hourly_wage_X_nbr_workers = sum(
      hourly_wage_X_nbr_workers,
      na.rm = TRUE
    ),
    total_nbr_workers_with_hourly_wage = sum(
      nbr_workers_with_hourly_wage,
      na.rm = TRUE
    )
  ) %>%
  ungroup() %>%
  mutate(
    # Average over certified workers with a valid hourly wage only
    mean_hourly_wage_start_year = if_else(
      total_nbr_workers_with_hourly_wage > 0,
      total_hourly_wage_X_nbr_workers / total_nbr_workers_with_hourly_wage,
      NA_real_
    ),
    share_cert_hours_with_hourly_wage_start_year = if_else(
      man_hours_certified_start_year > 0,
      cert_hours_with_hourly_wage_start_year /
        man_hours_certified_start_year,
      NA_real_
    ),
    share_cert_hours_at_aewr_start_year = if_else(
      cert_hours_with_hourly_wage_and_aewr_start_year > 0,
      cert_hours_at_aewr_start_year /
        cert_hours_with_hourly_wage_and_aewr_start_year,
      NA_real_
    )
  ) %>%
  select(-total_hourly_wage_X_nbr_workers, -total_nbr_workers_with_hourly_wage)

AssertNestedHours(
  h2a_start_year_aggregated_df$man_hours_certified_start_year,
  h2a_start_year_aggregated_df$cert_hours_with_hourly_wage_start_year,
  h2a_start_year_aggregated_df$cert_hours_with_hourly_wage_and_aewr_start_year,
  h2a_start_year_aggregated_df$cert_hours_at_aewr_start_year,
  "county start-year rows"
)

# Calculate same variables, but aggregated into case fiscal year
h2a_fiscal_year_df <- h2a_cleaned_df %>%
  mutate(fiscal_year = as.numeric(fiscal_year))

# For multi-county entries, equally split workers and man-hours across all counties equally
h2a_fiscal_year_df <- h2a_fiscal_year_df %>%
  mutate(n_counties = str_count(county_fips_list, ",") + 1) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_nbr_workers_requested = nbr_workers_requested / n_counties) %>%
  mutate(county_nbr_workers_certified = nbr_workers_certified / n_counties) %>%
  mutate(county_man_hours_requested = man_hours_requested / n_counties) %>%
  mutate(county_man_hours_certified = man_hours_certified / n_counties) %>%
  mutate(county_n_applications = (1 / n_entries) / n_counties)

# We have to manually calculate weighted average of hourly wage because R cannot handle NAs in weights
h2a_fiscal_year_df <- h2a_fiscal_year_df %>%
  mutate(
    hourly_wage_X_nbr_workers = hourly_wage * county_nbr_workers_certified,
    nbr_workers_with_hourly_wage = if_else(
      !is.na(hourly_wage),
      county_nbr_workers_certified,
      0
    )
  )

# Collapse by county-fiscal-year
h2a_fiscal_year_aggregated_df <- h2a_fiscal_year_df %>%
  group_by(county_fips_list, fiscal_year) %>%
  summarise(
    nbr_workers_requested_fiscal_year = SumRequested(
      county_nbr_workers_requested
    ),
    nbr_workers_certified_fiscal_year = sum(
      county_nbr_workers_certified,
      na.rm = TRUE
    ),
    man_hours_requested_fiscal_year = SumRequested(
      county_man_hours_requested
    ),
    man_hours_certified_fiscal_year = sum(
      county_man_hours_certified,
      na.rm = TRUE
    ),
    nbr_applications_fiscal_year = sum(county_n_applications, na.rm = TRUE),
    total_hourly_wage_X_nbr_workers = sum(
      hourly_wage_X_nbr_workers,
      na.rm = TRUE
    ),
    total_nbr_workers_with_hourly_wage = sum(
      nbr_workers_with_hourly_wage,
      na.rm = TRUE
    )
  ) %>%
  ungroup() %>%
  mutate(
    # Average over certified workers with a valid hourly wage only
    mean_hourly_wage_fiscal_year = if_else(
      total_nbr_workers_with_hourly_wage > 0,
      total_hourly_wage_X_nbr_workers / total_nbr_workers_with_hourly_wage,
      NA_real_
    )
  ) %>%
  select(-total_hourly_wage_X_nbr_workers, -total_nbr_workers_with_hourly_wage)

# Combine
# Harmonize variable names
h2a_start_year_aggregated_df <- h2a_start_year_aggregated_df %>%
  rename(year = start_year)

h2a_fiscal_year_aggregated_df <- h2a_fiscal_year_aggregated_df %>%
  rename(year = fiscal_year)

# Harmonize missing FIPS
h2a_all_years_aggregated_df <- h2a_all_years_aggregated_df %>%
  mutate(
    county_fips = harmonize_county_fips_2010(if_else(
      county_fips_list == "" | county_fips_list == "00000",
      "00000",
      county_fips_list
    ))
  ) %>%
  select(-county_fips_list) %>%
  group_by(county_fips, year) %>%
  summarise(
    across(matches("^(nbr_workers|man_hours)_requested_"), SumRequested),
    across(-matches("^(nbr_workers|man_hours)_requested_"), ~ sum(.x, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  ungroup()

h2a_start_year_aggregated_df <- h2a_start_year_aggregated_df %>%
  mutate(
    county_fips = harmonize_county_fips_2010(if_else(
      county_fips_list == "" | county_fips_list == "00000",
      "00000",
      county_fips_list
    ))
  ) %>%
  select(-county_fips_list) %>%
  group_by(county_fips, year) %>%
  summarise(
    across(matches("^(nbr_workers|man_hours)_requested_"), SumRequested),
    across(-matches("^(nbr_workers|man_hours)_requested_"), ~ sum(.x, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  ungroup() %>%
  mutate(
    share_cert_hours_with_hourly_wage_start_year = if_else(
      man_hours_certified_start_year > 0,
      cert_hours_with_hourly_wage_start_year /
        man_hours_certified_start_year,
      NA_real_
    ),
    share_cert_hours_at_aewr_start_year = if_else(
      cert_hours_with_hourly_wage_and_aewr_start_year > 0,
      cert_hours_at_aewr_start_year /
        cert_hours_with_hourly_wage_and_aewr_start_year,
      NA_real_
    )
  )

AssertNestedHours(
  h2a_start_year_aggregated_df$man_hours_certified_start_year,
  h2a_start_year_aggregated_df$cert_hours_with_hourly_wage_start_year,
  h2a_start_year_aggregated_df$cert_hours_with_hourly_wage_and_aewr_start_year,
  h2a_start_year_aggregated_df$cert_hours_at_aewr_start_year,
  "harmonized county start-year rows"
)

# Add county-start-year employer count
h2a_start_year_aggregated_df <- h2a_start_year_aggregated_df %>%
  left_join(
    h2a_start_year_employer_counts,
    by = c("county_fips", "year"),
    relationship = "one-to-one"
  ) %>%
  mutate(
    across(
      starts_with("nbr_employers_"),
      ~ replace_na(.x, 0L)
    )
  )

h2a_fiscal_year_aggregated_df <- h2a_fiscal_year_aggregated_df %>%
  mutate(
    county_fips = harmonize_county_fips_2010(if_else(
      county_fips_list == "" | county_fips_list == "00000",
      "00000",
      county_fips_list
    ))
  ) %>%
  select(-county_fips_list) %>%
  group_by(county_fips, year) %>%
  summarise(
    across(matches("^(nbr_workers|man_hours)_requested_"), SumRequested),
    across(-matches("^(nbr_workers|man_hours)_requested_"), ~ sum(.x, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  ungroup()

# Merge
# Track absence separately for each calendar. A county-year without allocated
# rows has zero requested positions/hours; a contributing row with an unknown
# requested amount must retain NA. These join markers are temporary only.
h2a_aggregated_df <- h2a_all_years_aggregated_df %>%
  mutate(has_all_years = TRUE) %>%
  full_join(h2a_start_year_aggregated_df %>% mutate(has_start_year = TRUE),
    by = c("county_fips", "year")) %>%
  full_join(h2a_fiscal_year_aggregated_df %>% mutate(has_fiscal_year = TRUE),
    by = c("county_fips", "year")) %>%
  full_join(
    h2a_case_start_year_aggregated_df,
    by = c("county_fips", "year")
  ) %>%
  mutate(
    across(matches("^(nbr_workers|man_hours)_requested_all_years$"),
      ~ if_else(is.na(has_all_years), 0, .x)),
    across(matches("^(nbr_workers|man_hours)_requested_start_year$"),
      ~ if_else(is.na(has_start_year), 0, .x)),
    across(matches("^(nbr_workers|man_hours)_requested_fiscal_year$"),
      ~ if_else(is.na(has_fiscal_year), 0, .x)),
    nbr_applications_start_year = coalesce(
      nbr_applications_case_start_year,
      0
    )
  ) %>%
  select(-nbr_applications_case_start_year, -has_all_years, -has_start_year,
    -has_fiscal_year)

# Harmonize name of FIPS code variable
h2a_aggregated_df <- h2a_aggregated_df %>%
  mutate(
    state_fips = state_from_county_fips(county_fips),
    county_code = county_code_from_county_fips(county_fips),
    employer_linkage_unit =
      employer_linkage_contract$employer_linkage_unit[[1]],
    employer_linkage_version =
      employer_linkage_contract$employer_linkage_version[[1]],
    employer_linkage_input_max_year =
      employer_linkage_contract$employer_linkage_input_max_year[[1]],
    employer_linkage_review_status =
      employer_linkage_contract$employer_linkage_review_status[[1]],
    employer_linkage_review_report_hash =
      employer_linkage_contract$employer_linkage_review_report_hash[[1]]
  )

# Export
sanity_check_total_4 <- h2a_aggregated_df %>%
  group_by(year) %>%
  summarize(
    nbr_workers_certified = sum(
      nbr_workers_certified_fiscal_year,
      na.rm = TRUE
    ),
  ) %>%
  filter(nbr_workers_certified > 0)
# Numbers match, we are good

assert_geo_columns(
  h2a_aggregated_df,
  c("state_fips", "county_code", "county_fips")
)
h2a_aggregated_df %>%
  write_parquet(path_int("h2a_aggregated.parquet"))

h2a_ts_df <- h2a_aggregated_df %>%
  filter(state_fips != "00") %>%
  group_by(year) %>%
  summarise_if(is.numeric, sum, na.rm = TRUE) %>%
  filter(year > 2007 & year < 2023)

tsplot <- ggplot(h2a_ts_df, aes(x = year, y = nbr_applications_fiscal_year)) +
  geom_line() +
  xlab("")
tsplot
