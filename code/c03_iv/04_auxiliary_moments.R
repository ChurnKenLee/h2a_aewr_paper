# Purpose: Build OEWS-area Census-duration and QCEW-seasonality calibration moments.
# Inputs: OEWS priors, hired-worker duration, and quarterly QCEW employment.
# Outputs: auxiliary-moment data and diagnostics parquets.
# Run after: 03_oews_area_prior_weights.R and the relevant a-stage extractors.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)

reference_months <- c("january", "april", "july", "october")

sum_if_observed <- function(x) {
  if (all(is.na(x))) {
    return(NA_real_)
  }
  sum(x, na.rm = TRUE)
}

interpolate_inside <- function(year, value) {
  observed <- !is.na(year) & !is.na(value)
  output <- rep(NA_real_, length(year))
  observed_year <- year[observed]
  observed_value <- value[observed]

  if (length(observed_year) == 0) {
    return(output)
  }

  if (length(observed_year) == 1) {
    output[year == observed_year] <- observed_value
    return(output)
  }

  approx(
    x = observed_year,
    y = observed_value,
    xout = year,
    method = "linear",
    rule = 1,
    ties = "ordered"
  )$y
}

fls_county_area_prior <- read_parquet(path_int(
  "fls_county_oews_area_prior_weight.parquet"
)) %>%
  mutate(
    countyfips = as.character(countyfips),
    year = as.integer(year),
    aewr_region_num = as.integer(aewr_region_num)
  )

fls_oews_area_prior <- read_parquet(path_int(
  "fls_oews_area_prior_weight.parquet"
)) %>%
  mutate(
    year = as.integer(year),
    aewr_region_num = as.integer(aewr_region_num)
  )

# Census duration shares are measured in 2007, 2012, 2017, and 2022. Allocate
# counties that intersect multiple OEWS areas using the same allocation as the
# area prior, aggregate counts, and interpolate only between observed Census
# years. Values outside the observed range remain missing.
census_county_duration <- read_parquet(path_int(
  "census_ag_hired_worker_duration_county.parquet"
)) %>%
  transmute(
    countyfips = as.character(countyfips),
    year = as.integer(year),
    census_hired_workers_150_days_or_more = as.numeric(
      census_hired_workers_150_days_or_more
    ),
    census_hired_workers_less_than_150_days = as.numeric(
      census_hired_workers_less_than_150_days
    ),
    census_hired_workers_duration_total = as.numeric(
      census_hired_workers_duration_total
    ),
    census_hired_worker_duration_complete = as.logical(
      census_hired_worker_duration_complete
    )
  )

census_area_mapping <- fls_county_area_prior %>%
  transmute(
    countyfips,
    mapping_year = year,
    aewr_region_num,
    oews_area_code,
    county_area_allocation,
    county_area_prior_weight
  )

census_area_at_census_year <- census_county_duration %>%
  mutate(
    mapping_year = pmax(year, min(census_area_mapping$mapping_year))
  ) %>%
  inner_join(census_area_mapping, by = c("countyfips", "mapping_year")) %>%
  mutate(
    census_area_workers_150_plus = census_hired_workers_150_days_or_more *
      county_area_allocation,
    census_area_workers_less_than_150 = census_hired_workers_less_than_150_days *
      county_area_allocation,
    census_area_workers_duration_total = census_hired_workers_duration_total *
      county_area_allocation
  ) %>%
  group_by(aewr_region_num, year, oews_area_code) %>%
  summarise(
    census_area_workers_150_plus = sum_if_observed(
      census_area_workers_150_plus
    ),
    census_area_workers_less_than_150 = sum_if_observed(
      census_area_workers_less_than_150
    ),
    census_area_workers_duration_total = sum_if_observed(
      census_area_workers_duration_total
    ),
    census_duration_observed_prior_mass = sum(
      county_area_prior_weight[
        !is.na(census_hired_worker_duration_complete) &
          census_hired_worker_duration_complete
      ],
      na.rm = TRUE
    ),
    census_area_prior_mass = sum(county_area_prior_weight, na.rm = TRUE),
    census_duration_counties_observed = n_distinct(
      countyfips[
        !is.na(census_hired_worker_duration_complete) &
          census_hired_worker_duration_complete
      ]
    ),
    census_duration_counties_total = n_distinct(countyfips),
    .groups = "drop"
  ) %>%
  mutate(
    census_hired_worker_150_plus_share_census_year = if_else(
      census_area_workers_duration_total > 0,
      census_area_workers_150_plus / census_area_workers_duration_total,
      NA_real_
    ),
    census_duration_observed_prior_share = if_else(
      census_area_prior_mass > 0,
      census_duration_observed_prior_mass / census_area_prior_mass,
      NA_real_
    )
  )

census_area_year_skeleton <- fls_oews_area_prior %>%
  select(aewr_region_num, year, oews_area_code) %>%

  distinct()

census_area_year <- census_area_year_skeleton %>%
  full_join(
    census_area_at_census_year,
    by = c("aewr_region_num", "year", "oews_area_code")
  ) %>%
  group_by(aewr_region_num, oews_area_code) %>%
  arrange(year, .by_group = TRUE) %>%
  mutate(
    census_hired_worker_150_plus_share = interpolate_inside(
      year,
      census_hired_worker_150_plus_share_census_year
    ),
    census_duration_share_source = case_when(
      !is.na(census_hired_worker_150_plus_share_census_year) ~ "census_year",
      !is.na(census_hired_worker_150_plus_share) ~ "linear_interpolation",
      .default = NA_character_
    )
  ) %>%
  ungroup() %>%
  semi_join(
    census_area_year_skeleton,
    by = c("aewr_region_num", "year", "oews_area_code")
  )

# QCEW month-one employment in quarters 1--4 corresponds to the FLS January,
# April, July, and October reference weeks. Suppressed industry cells have
# already been changed to missing by the extractor; sum only disclosed private
# NAICS 111 and 112 cells and retain disclosure coverage diagnostics.
qcew_county_quarter <- read_parquet(path_int(
  "qcew_county_ag_quarterly_employment.parquet"
)) %>%
  mutate(
    countyfips = as.character(countyfips),
    year = as.integer(year)
  ) %>%
  group_by(countyfips, year, qtr, reference_month) %>%
  summarise(
    qcew_ag_reference_month_employment = sum_if_observed(
      qcew_reference_month_emplvl
    ),
    qcew_disclosed_industry_cells = sum(
      qcew_employment_disclosed,
      na.rm = TRUE
    ),
    qcew_industry_cells = n(),
    .groups = "drop"
  )

qcew_area_quarter <- qcew_county_quarter %>%
  inner_join(
    fls_county_area_prior %>%
      select(
        countyfips,
        year,
        aewr_region_num,
        oews_area_code,
        county_area_allocation,
        county_area_prior_weight
      ),
    by = c("countyfips", "year")
  ) %>%
  mutate(
    qcew_area_reference_month_employment = qcew_ag_reference_month_employment *
      county_area_allocation
  ) %>%
  group_by(
    aewr_region_num,
    year,
    oews_area_code,
    qtr,
    reference_month
  ) %>%
  summarise(
    qcew_area_reference_month_employment = sum_if_observed(
      qcew_area_reference_month_employment
    ),
    qcew_disclosed_industry_cells = sum(
      qcew_disclosed_industry_cells,
      na.rm = TRUE
    ),
    qcew_industry_cells = sum(qcew_industry_cells, na.rm = TRUE),
    qcew_counties_observed = n_distinct(
      countyfips[!is.na(qcew_ag_reference_month_employment)]
    ),
    qcew_counties_present = n_distinct(countyfips),
    .groups = "drop"
  )

qcew_area_year <- qcew_area_quarter %>%
  select(
    aewr_region_num,
    year,
    oews_area_code,
    reference_month,
    qcew_area_reference_month_employment,
    qcew_disclosed_industry_cells,
    qcew_industry_cells,
    qcew_counties_observed,
    qcew_counties_present
  ) %>%
  pivot_wider(
    names_from = reference_month,
    values_from = c(
      qcew_area_reference_month_employment,
      qcew_disclosed_industry_cells,
      qcew_industry_cells,
      qcew_counties_observed,
      qcew_counties_present
    ),
    names_glue = "{.value}_{reference_month}"
  )

for (month in reference_months) {
  employment_column <- paste0(
    "qcew_area_reference_month_employment_",
    month
  )
  if (!employment_column %in% names(qcew_area_year)) {
    qcew_area_year[[employment_column]] <- NA_real_
  }
}

qcew_area_year <- qcew_area_year %>%
  mutate(
    qcew_quarters_observed = rowSums(
      !is.na(pick(all_of(paste0(
        "qcew_area_reference_month_employment_",
        reference_months
      ))))
    ),
    qcew_seasonal_employment_complete = qcew_quarters_observed == 4,
    qcew_area_reference_month_employment_total = if_else(
      qcew_seasonal_employment_complete,
      rowSums(across(all_of(paste0(
        "qcew_area_reference_month_employment_",
        reference_months
      )))),
      NA_real_
    )
  ) %>%
  mutate(
    qcew_area_reference_month_employment_share_january = if_else(
      qcew_area_reference_month_employment_total > 0,
      qcew_area_reference_month_employment_january /
        qcew_area_reference_month_employment_total,
      NA_real_
    ),
    qcew_area_reference_month_employment_share_april = if_else(
      qcew_area_reference_month_employment_total > 0,
      qcew_area_reference_month_employment_april /
        qcew_area_reference_month_employment_total,
      NA_real_
    ),
    qcew_area_reference_month_employment_share_july = if_else(
      qcew_area_reference_month_employment_total > 0,
      qcew_area_reference_month_employment_july /
        qcew_area_reference_month_employment_total,
      NA_real_
    ),
    qcew_area_reference_month_employment_share_october = if_else(
      qcew_area_reference_month_employment_total > 0,
      qcew_area_reference_month_employment_october /
        qcew_area_reference_month_employment_total,
      NA_real_
    )
  )

fls_oews_area_auxiliary_moments <- fls_oews_area_prior %>%
  left_join(
    census_area_year,
    by = c("aewr_region_num", "year", "oews_area_code")
  ) %>%
  left_join(
    qcew_area_year,
    by = c("aewr_region_num", "year", "oews_area_code")
  ) %>%
  arrange(aewr_region_num, year, oews_area_code)

fls_oews_area_auxiliary_moment_diagnostics <-
  fls_oews_area_auxiliary_moments %>%
  group_by(aewr_region_num, year) %>%
  summarise(
    oews_area_count = n(),
    duration_feature_area_count = sum(
      !is.na(census_hired_worker_150_plus_share)
    ),
    duration_feature_prior_mass = sum(
      if_else(
        !is.na(census_hired_worker_150_plus_share),
        oews_area_prior_weight_all,
        0
      ),
      na.rm = TRUE
    ),
    seasonal_feature_area_count = sum(qcew_seasonal_employment_complete),
    seasonal_feature_prior_mass = sum(
      if_else(
        qcew_seasonal_employment_complete,
        oews_area_prior_weight_all,
        0
      ),
      na.rm = TRUE
    ),
    .groups = "drop"
  )

write_parquet(
  fls_oews_area_auxiliary_moments,
  path_int("fls_oews_area_auxiliary_moments.parquet")
)

write_parquet(
  fls_oews_area_auxiliary_moment_diagnostics,
  path_int("fls_oews_area_auxiliary_moment_diagnostics.parquet")
)
