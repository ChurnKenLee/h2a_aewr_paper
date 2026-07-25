# Purpose: Build public OEWS-area seasonal and duration calibration analogues.
# Inputs: OEWS priors, Census hired labor, QCEW, and optional QWI employment.
# Outputs: auxiliary moments, public bridge, interval bands, and diagnostics.
# Run after: 03_oews_area_prior_weights.R and the a-stage source extractors.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
source(path_code("c00_shared", "auxiliary_moment_helpers.R"))
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)

reference_months <- c("january", "april", "july", "october")
reference_month_lookup <- c(
  "1" = "january",
  "2" = "april",
  "3" = "july",
  "4" = "october"
)

sum_if_observed <- function(x) {
  if (all(is.na(x))) {
    return(NA_real_)
  }
  sum(x, na.rm = TRUE)
}

interpolate_inside <- function(year, value) {
  observed <- !is.na(year) & is.finite(value)
  output <- rep(NA_real_, length(year))
  if (!any(observed)) {
    return(output)
  }
  if (sum(observed) == 1L) {
    output[year == year[observed][[1]]] <- value[observed][[1]]
    return(output)
  }
  approx(
    x = year[observed],
    y = value[observed],
    xout = year,
    method = "linear",
    rule = 1,
    ties = "ordered"
  )$y
}

ensure_month_columns <- function(data, prefixes) {
  for (prefix in prefixes) {
    for (month in reference_months) {
      column <- paste0(prefix, "_", month)
      if (!column %in% names(data)) {
        data[[column]] <- NA_real_
      }
    }
  }
  data
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

# Census duration ------------------------------------------------------------

# The Census share is a distinct-worker count and is not put directly into the
# primary specification. It is retained raw and transformed below using only
# matched QWI and Census cells.
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

census_year_mapping <- census_county_duration %>%
  distinct(year) %>%
  mutate(
    mapping_year = pmax(year, min(census_area_mapping$mapping_year))
  ) %>%
  inner_join(census_area_mapping, by = "mapping_year")

# Begin from every county in the applicable prior vintage. A county absent
# from the Census extract must reduce measured source coverage rather than
# disappear from its denominator.
census_area_at_census_year <- census_year_mapping %>%
  left_join(
    census_county_duration,
    by = c("countyfips", "year"),
    relationship = "many-to-one"
  ) %>%
  mutate(
    census_area_workers_150_plus =
      census_hired_workers_150_days_or_more * county_area_allocation,
    census_area_workers_less_than_150 =
      census_hired_workers_less_than_150_days * county_area_allocation,
    census_area_workers_duration_total =
      census_hired_workers_duration_total * county_area_allocation
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
      census_area_workers_150_plus /
        census_area_workers_duration_total,
      NA_real_
    ),
    census_duration_observed_prior_share_census_year = if_else(
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
    census_duration_observed_prior_share = interpolate_inside(
      year,
      census_duration_observed_prior_share_census_year
    ),
    census_duration_share_source = case_when(
      !is.na(census_hired_worker_150_plus_share_census_year) ~
        "census_year",
      !is.na(census_hired_worker_150_plus_share) ~
        "linear_interpolation",
      .default = NA_character_
    )
  ) %>%
  ungroup() %>%
  semi_join(
    census_area_year_skeleton,
    by = c("aewr_region_num", "year", "oews_area_code")
  )

# QCEW and QWI seasonality ----------------------------------------------------

qcew_county_quarter <- read_parquet(path_int(
  "qcew_county_ag_quarterly_employment.parquet"
)) %>%
  mutate(
    countyfips = as.character(countyfips),
    year = as.integer(year),
    qtr = as.integer(qtr)
  ) %>%
  group_by(countyfips, year, qtr, reference_month) %>%
  summarise(
    qcew_ag_reference_month_employment_partial = sum_if_observed(
      qcew_reference_month_emplvl
    ),
    qcew_disclosed_industry_cells = sum(
      qcew_employment_disclosed,
      na.rm = TRUE
    ),
    qcew_industry_cells = n_distinct(industry_code),
    .groups = "drop"
  ) %>%
  mutate(
    qcew_ag_reference_month_employment = if_else(
      qcew_industry_cells == 2L &
        qcew_disclosed_industry_cells == 2L,
      qcew_ag_reference_month_employment_partial,
      NA_real_
    ),
    qcew_employment_complete =
      !is.na(qcew_ag_reference_month_employment)
  )

qwi_path <- path_int("qwi_county_ag_quarterly_employment.parquet")
qwi_available <- file.exists(qwi_path)
qwi_county_quarter <- if (qwi_available) {
  read_parquet(qwi_path) %>%
    mutate(
      countyfips = as.character(countyfips),
      year = as.integer(year),
      qtr = as.integer(qtr)
    ) %>%
    group_by(countyfips, year, qtr) %>%
    summarise(
      qwi_beginning_quarter_employment_partial = sum_if_observed(
        qwi_beginning_quarter_employment
      ),
      qwi_stable_employment_partial = sum_if_observed(
        qwi_stable_employment
      ),
      qwi_industry_cells = n_distinct(industry_code),
      qwi_beginning_industry_cells_observed = sum(
        !is.na(qwi_beginning_quarter_employment)
      ),
      qwi_stable_industry_cells_observed = sum(
        !is.na(qwi_stable_employment)
      ),
      .groups = "drop"
    ) %>%
    mutate(
      qwi_beginning_quarter_employment = if_else(
        qwi_industry_cells == 2L &
          qwi_beginning_industry_cells_observed == 2L,
        qwi_beginning_quarter_employment_partial,
        NA_real_
      ),
      qwi_stable_employment = if_else(
        qwi_industry_cells == 2L &
          qwi_stable_industry_cells_observed == 2L,
        qwi_stable_employment_partial,
        NA_real_
      )
    )
} else {
  warning(
    paste0(
      "QWI input is absent; primary QCEW-only moments will be built, ",
      "but QWI fills, duration analogues, and interval bands will be missing."
    ),
    call. = FALSE
  )
  tibble(
    countyfips = character(),
    year = integer(),
    qtr = integer(),
    qwi_beginning_quarter_employment = numeric(),
    qwi_stable_employment = numeric()
  )
}

county_quarter_skeleton <- fls_county_area_prior %>%
  select(countyfips, year) %>%
  distinct() %>%
  crossing(qtr = seq_along(reference_months)) %>%
  mutate(
    reference_month = unname(
      reference_month_lookup[as.character(qtr)]
    )
  )

# Starting from the full prior skeleton makes absent source rows explicit.
# This keeps the 90-percent gate in units of total prior mass rather than only
# the subset of counties represented in QCEW or QWI.
county_quarter_public <- county_quarter_skeleton %>%
  left_join(
    qcew_county_quarter %>% select(-reference_month),
    by = c("countyfips", "year", "qtr"),
    relationship = "one-to-one"
  ) %>%
  left_join(
    qwi_county_quarter,
    by = c("countyfips", "year", "qtr"),
    relationship = "one-to-one"
  ) %>%
  mutate(
    qcew_employment_complete = coalesce(
      qcew_employment_complete,
      FALSE
    ),
    qwi_employment_complete =
      !is.na(qwi_beginning_quarter_employment),
    public_reference_month_employment = coalesce(
      qcew_ag_reference_month_employment,
      qwi_beginning_quarter_employment
    ),
    public_employment_source = case_when(
      qcew_employment_complete ~ "qcew",
      !qcew_employment_complete & qwi_employment_complete ~ "qwi_fill",
      .default = NA_character_
    )
  )

area_quarter_public <- county_quarter_public %>%
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
    public_area_reference_month_employment =
      public_reference_month_employment * county_area_allocation,
    qcew_area_reference_month_employment =
      qcew_ag_reference_month_employment * county_area_allocation,
    qwi_area_reference_month_employment =
      qwi_beginning_quarter_employment * county_area_allocation,
    qwi_area_stable_employment =
      qwi_stable_employment * county_area_allocation
  ) %>%
  group_by(
    aewr_region_num,
    year,
    oews_area_code,
    qtr,
    reference_month
  ) %>%
  summarise(
    public_area_reference_month_employment = sum_if_observed(
      public_area_reference_month_employment
    ),
    qcew_area_reference_month_employment = sum_if_observed(
      qcew_area_reference_month_employment
    ),
    qwi_area_reference_month_employment = sum_if_observed(
      qwi_area_reference_month_employment
    ),
    qwi_area_stable_employment = sum_if_observed(
      qwi_area_stable_employment
    ),
    public_observed_prior_mass = sum(
      county_area_prior_weight[
        !is.na(public_reference_month_employment)
      ],
      na.rm = TRUE
    ),
    qcew_observed_prior_mass = sum(
      county_area_prior_weight[qcew_employment_complete],
      na.rm = TRUE
    ),
    qwi_fill_prior_mass = sum(
      county_area_prior_weight[public_employment_source == "qwi_fill"],
      na.rm = TRUE
    ),
    qwi_observed_prior_mass = sum(
      county_area_prior_weight[qwi_employment_complete],
      na.rm = TRUE
    ),
    area_prior_mass = sum(county_area_prior_weight, na.rm = TRUE),
    public_counties_observed = n_distinct(
      countyfips[!is.na(public_reference_month_employment)]
    ),
    public_counties_present = n_distinct(countyfips),
    .groups = "drop"
  ) %>%
  mutate(
    public_observed_prior_share = public_observed_prior_mass /
      area_prior_mass,
    qcew_observed_prior_share = qcew_observed_prior_mass /
      area_prior_mass,
    qwi_fill_prior_share = qwi_fill_prior_mass / area_prior_mass,
    qwi_observed_prior_share = qwi_observed_prior_mass /
      area_prior_mass
  )

area_year_public <- area_quarter_public %>%
  select(
    aewr_region_num,
    year,
    oews_area_code,
    reference_month,
    public_area_reference_month_employment,
    qcew_area_reference_month_employment,
    qwi_area_reference_month_employment,
    qwi_area_stable_employment,
    public_observed_prior_share,
    qcew_observed_prior_share,
    qwi_fill_prior_share,
    qwi_observed_prior_share,
    public_counties_observed,
    public_counties_present
  ) %>%
  pivot_wider(
    names_from = reference_month,
    values_from = c(
      public_area_reference_month_employment,
      qcew_area_reference_month_employment,
      qwi_area_reference_month_employment,
      qwi_area_stable_employment,
      public_observed_prior_share,
      qcew_observed_prior_share,
      qwi_fill_prior_share,
      qwi_observed_prior_share,
      public_counties_observed,
      public_counties_present
    ),
    names_glue = "{.value}_{reference_month}"
  ) %>%
  ensure_month_columns(
    c(
      "public_area_reference_month_employment",
      "qcew_area_reference_month_employment",
      "qwi_area_reference_month_employment",
      "qwi_area_stable_employment",
      "public_observed_prior_share",
      "qcew_observed_prior_share",
      "qwi_fill_prior_share",
      "qwi_observed_prior_share"
    )
  )

public_employment_columns <- paste0(
  "public_area_reference_month_employment_",
  reference_months
)
qcew_employment_columns <- paste0(
  "qcew_area_reference_month_employment_",
  reference_months
)
qwi_employment_columns <- paste0(
  "qwi_area_reference_month_employment_",
  reference_months
)
qwi_stable_columns <- paste0(
  "qwi_area_stable_employment_",
  reference_months
)

area_year_public <- area_year_public %>%
  mutate(
    public_quarters_observed = rowSums(
      !is.na(pick(all_of(public_employment_columns)))
    ),
    qcew_quarters_observed = rowSums(
      !is.na(pick(all_of(qcew_employment_columns)))
    ),
    qwi_quarters_observed = rowSums(
      !is.na(pick(all_of(qwi_employment_columns)))
    ),
    qwi_stable_quarters_observed = rowSums(
      !is.na(pick(all_of(qwi_stable_columns)))
    ),
    public_seasonal_employment_complete =
      public_quarters_observed == 4L,
    qcew_seasonal_employment_complete =
      qcew_quarters_observed == 4L,
    qwi_seasonal_employment_complete =
      qwi_quarters_observed == 4L,
    public_area_reference_month_employment_total = if_else(
      public_seasonal_employment_complete,
      rowSums(pick(all_of(public_employment_columns))),
      NA_real_
    ),
    qcew_area_reference_month_employment_total = if_else(
      qcew_seasonal_employment_complete,
      rowSums(pick(all_of(qcew_employment_columns))),
      NA_real_
    ),
    qwi_area_reference_month_employment_total = if_else(
      qwi_seasonal_employment_complete,
      rowSums(pick(all_of(qwi_employment_columns))),
      NA_real_
    ),
    qwi_area_stable_employment_total = if_else(
      qwi_stable_quarters_observed == 4L,
      rowSums(pick(all_of(qwi_stable_columns))),
      NA_real_
    ),
    seasonal_feature_observed_share = pmin(
      !!!syms(paste0(
        "public_observed_prior_share_",
        reference_months
      )),
      na.rm = FALSE
    ),
    seasonal_qwi_fill_share = pmax(
      !!!syms(paste0("qwi_fill_prior_share_", reference_months)),
      na.rm = FALSE
    ),
    qcew_seasonal_observed_share = pmin(
      !!!syms(paste0(
        "qcew_observed_prior_share_",
        reference_months
      )),
      na.rm = FALSE
    ),
    qwi_duration_observed_share = pmin(
      !!!syms(paste0(
        "qwi_observed_prior_share_",
        reference_months
      )),
      na.rm = FALSE
    ),
    qwi_area_stable_employment_share = qwi_persistence_share(
      qwi_area_stable_employment_total,
      qwi_area_reference_month_employment_total
    ),
    seasonal_feature_source = case_when(
      !public_seasonal_employment_complete ~ NA_character_,
      seasonal_qwi_fill_share > 0 ~ "qcew_with_qwi_fill",
      .default = "qcew"
    )
  )

for (month in reference_months) {
  public_employment <- paste0(
    "public_area_reference_month_employment_",
    month
  )
  qcew_employment <- paste0(
    "qcew_area_reference_month_employment_",
    month
  )
  qwi_employment <- paste0(
    "qwi_area_reference_month_employment_",
    month
  )
  area_year_public[[paste0(
    "qcew_qwi_employment_share_",
    month
  )]] <- seasonal_employment_share(
    area_year_public[[public_employment]],
    area_year_public$public_area_reference_month_employment_total
  )
  area_year_public[[paste0(
    "qcew_direct_employment_share_",
    month
  )]] <- seasonal_employment_share(
    area_year_public[[qcew_employment]],
    area_year_public$qcew_area_reference_month_employment_total
  )
  area_year_public[[paste0(
    "qwi_employment_share_",
    month
  )]] <- seasonal_employment_share(
    area_year_public[[qwi_employment]],
    area_year_public$qwi_area_reference_month_employment_total
  )
}

# Public-only duration bridge -------------------------------------------------

bridge_matched_cells <- census_area_at_census_year %>%
  select(
    aewr_region_num,
    year,
    oews_area_code,
    census_hired_worker_150_plus_share_census_year,
    census_duration_observed_prior_share_census_year
  ) %>%
  inner_join(
    area_year_public %>%
      select(
        aewr_region_num,
        year,
        oews_area_code,
        qwi_area_stable_employment_share,
        qwi_duration_observed_share
      ),
    by = c("aewr_region_num", "year", "oews_area_code")
  ) %>%
  inner_join(
    fls_oews_area_prior %>%
      select(
        aewr_region_num,
        year,
        oews_area_code,
        oews_area_prior_weight_all
      ),
    by = c("aewr_region_num", "year", "oews_area_code")
  ) %>%
  filter(
    census_duration_observed_prior_share_census_year >= 0.90,
    qwi_duration_observed_share >= 0.90
  )

public_duration_odds_bridge_ratio <- estimate_public_odds_bridge(
  census_share =
    bridge_matched_cells$census_hired_worker_150_plus_share_census_year,
  qwi_persistence_share =
    bridge_matched_cells$qwi_area_stable_employment_share,
  weight = bridge_matched_cells$oews_area_prior_weight_all
)

census_area_year <- census_area_year %>%
  mutate(
    census_hired_worker_150_plus_share_bridged =
      apply_public_odds_bridge(
        census_hired_worker_150_plus_share,
        public_duration_odds_bridge_ratio
      ),
    census_duration_odds_bridge_ratio =
      public_duration_odds_bridge_ratio
  )

public_duration_bridge_diagnostics <- tibble(
  bridge_method = "weighted_median_qwi_to_census_odds",
  bridge_ratio = public_duration_odds_bridge_ratio,
  matched_area_year_cells = nrow(bridge_matched_cells),
  uses_fls_data = FALSE
)

# Fixed, source-discrepancy interval bands -----------------------------------

area_year_with_prior <- area_year_public %>%
  inner_join(
    fls_oews_area_prior %>%
      select(
        aewr_region_num,
        year,
        oews_area_code,
        oews_area_prior_weight_all
      ),
    by = c("aewr_region_num", "year", "oews_area_code")
  )

seasonal_band_rows <- map_dfr(
  c("january", "april", "july"),
  function(month) {
    band <- fixed_standardized_discrepancy_band(
      primary_value = if_else(
        area_year_with_prior$qcew_seasonal_observed_share >= 0.90,
        area_year_with_prior[[paste0(
          "qcew_direct_employment_share_",
          month
        )]],
        NA_real_
      ),
      comparison_value = if_else(
        area_year_with_prior$qwi_duration_observed_share >= 0.90,
        area_year_with_prior[[paste0(
          "qwi_employment_share_",
          month
        )]],
        NA_real_
      ),
      weight = area_year_with_prior$oews_area_prior_weight_all,
      probability = 0.90
    )
    tibble(
      moment_label = paste0("seasonal_", month),
      interval_half_width = band$half_width,
      discrepancy_scale = band$scale,
      matched_area_year_cells = band$matched_count,
      band_quantile = 0.90,
      source_comparison = "QCEW versus QWI"
    )
  }
)

duration_band_data <- census_area_year %>%
  select(
    aewr_region_num,
    year,
    oews_area_code,
    census_hired_worker_150_plus_share_bridged,
    census_duration_observed_prior_share
  ) %>%
  inner_join(
    area_year_with_prior %>%
      select(
        aewr_region_num,
        year,
        oews_area_code,
        qwi_area_stable_employment_share,
        qwi_duration_observed_share,
        oews_area_prior_weight_all
      ),
    by = c("aewr_region_num", "year", "oews_area_code")
  ) %>%
  filter(
    census_duration_observed_prior_share >= 0.90,
    qwi_duration_observed_share >= 0.90
  )
duration_band <- fixed_standardized_discrepancy_band(
  primary_value =
    duration_band_data$census_hired_worker_150_plus_share_bridged,
  comparison_value =
    duration_band_data$qwi_area_stable_employment_share,
  weight = duration_band_data$oews_area_prior_weight_all,
  probability = 0.90
)

auxiliary_interval_bands <- bind_rows(
  seasonal_band_rows,
  tibble(
    moment_label = "duration",
    interval_half_width = duration_band$half_width,
    discrepancy_scale = duration_band$scale,
    matched_area_year_cells = duration_band$matched_count,
    band_quantile = 0.90,
    source_comparison = "bridged Census versus QWI"
  )
)

# Assemble the calibration features -----------------------------------------

fls_oews_area_auxiliary_moments <- fls_oews_area_prior %>%
  left_join(
    census_area_year,
    by = c("aewr_region_num", "year", "oews_area_code")
  ) %>%
  left_join(
    area_year_public,
    by = c("aewr_region_num", "year", "oews_area_code")
  ) %>%
  arrange(aewr_region_num, year, oews_area_code)

fls_oews_area_auxiliary_moment_diagnostics <-
  fls_oews_area_auxiliary_moments %>%
  group_by(aewr_region_num, year) %>%
  summarise(
    oews_area_count = n(),
    raw_census_duration_feature_prior_mass = sum(
      oews_area_prior_weight_all *
        coalesce(census_duration_observed_prior_share, 0),
      na.rm = TRUE
    ),
    bridged_census_duration_feature_prior_mass = sum(
      oews_area_prior_weight_all *
        coalesce(census_duration_observed_prior_share, 0),
      na.rm = TRUE
    ),
    qwi_duration_feature_prior_mass = sum(
      oews_area_prior_weight_all *
        coalesce(qwi_duration_observed_share, 0),
      na.rm = TRUE
    ),
    seasonal_feature_area_count = sum(
      !is.na(qcew_qwi_employment_share_january) &
        !is.na(qcew_qwi_employment_share_april) &
        !is.na(qcew_qwi_employment_share_july)
    ),
    seasonal_feature_prior_mass = sum(
      oews_area_prior_weight_all *
        coalesce(seasonal_feature_observed_share, 0),
      na.rm = TRUE
    ),
    seasonal_qwi_fill_prior_mass = sum(
      oews_area_prior_weight_all *
        coalesce(seasonal_qwi_fill_share, 0),
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

write_parquet(
  public_duration_bridge_diagnostics,
  path_int("public_duration_odds_bridge_diagnostics.parquet")
)

write_parquet(
  auxiliary_interval_bands,
  path_int("fls_auxiliary_interval_bands.parquet")
)
