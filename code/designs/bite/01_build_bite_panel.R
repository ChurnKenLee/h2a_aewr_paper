# Purpose: Build the standalone full-panel PPML bite analysis panel from the
# design-neutral shared county-year panel.
# Input:  data/processed/county_year_panel.parquet
# Output: data/processed/bite_county_year_panel.parquet

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("designs", "bite", "helpers.R"))

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
})

panel <- read_parquet(path_processed("county_year_panel.parquet")) %>%
  mutate(year = as.integer(year))

# --- Strict shared-panel contract checks ---
assert_geo_columns(
  panel,
  c("county_fips", "state_fips", "cz_id", "aewr_region_id")
)

expected_years <- seq.int(SHARED_PANEL_START_YEAR, SHARED_PANEL_END_YEAR)
if (
  nrow(panel) != 55944L ||
    anyDuplicated(panel[c("county_fips", "year")]) > 0L ||
    n_distinct(panel$county_fips) != 3108L ||
    !identical(sort(unique(panel$year)), expected_years) ||
    n_distinct(panel$aewr_region_id) != 17L ||
    n_distinct(panel$state_fips) != 48L
) {
  stop(
    "Shared county_year_panel.parquet failed balanced 3,108 x 18 (2008-2025) contract.",
    call. = FALSE
  )
}

# AEWR-001: canonical 2009 statutory AEWR must remain strictly NULL.
rows_2009 <- panel$year == 2009L
if (!all(is.na(panel$aewr[rows_2009])) || !all(is.na(panel$aewr_ppi[rows_2009]))) {
  stop("AEWR-001 violation: 2009 aewr / aewr_ppi must be strictly NA.", call. = FALSE)
}

# Verify complete baseline inputs in 2008 & 2010 and estimation inputs in 2011-2025.
base_rows <- panel$year %in% BITE_BASELINE_YEARS
est_rows <- panel$year >= BITE_ESTIMATION_START_YEAR & panel$year <= BITE_ESTIMATION_END_YEAR
stopifnot(
  all(is.finite(panel$aewr[base_rows]) & panel$aewr[base_rows] > 0),
  all(is.finite(panel$aewr_ppi[base_rows]) & panel$aewr_ppi[base_rows] > 0),
  all(is.finite(panel$ppi_2012[base_rows]) & panel$ppi_2012[base_rows] > 0),
  all(is.finite(panel$prevailing_ag_min_wage_ppi[base_rows]) & panel$prevailing_ag_min_wage_ppi[base_rows] > 0),
  all(is.finite(panel$wage_p10[base_rows]) & panel$wage_p10[base_rows] > 0),
  all(is.finite(panel$wage_p25[base_rows]) & panel$wage_p25[base_rows] > 0),
  all(is.finite(panel$wage_p50[base_rows]) & panel$wage_p50[base_rows] > 0),
  all(is.finite(panel$wage_p75[base_rows]) & panel$wage_p75[base_rows] > 0),
  all(is.finite(panel$wage_p90[base_rows]) & panel$wage_p90[base_rows] > 0),
  all(is.finite(panel$aewr[est_rows]) & panel$aewr[est_rows] > 0),
  all(is.finite(panel$ppi_2012[est_rows]) & panel$ppi_2012[est_rows] > 0),
  all(is.finite(panel$prevailing_ag_min_wage[est_rows]) & panel$prevailing_ag_min_wage[est_rows] > 0),
  all(is.finite(panel$man_hours_certified_start_year) & panel$man_hours_certified_start_year >= 0),
  all(is.finite(panel$nbr_workers_certified_start_year) & panel$nbr_workers_certified_start_year >= 0)
)

# --- Construct frozen 2008/2010 county exposures (excluding missing 2009 AEWR) ---
baseline_rows_df <- panel %>%
  filter(year %in% BITE_BASELINE_YEARS)
baseline_rows_df$shortfall_log_real <- bite_compute_shortfall_log(baseline_rows_df)

baseline_2008_2010 <- baseline_rows_df %>%
  mutate(
    ln_aewr_over_p25 = log(aewr_ppi / wage_p25),
    ln_aewr_over_oews = if_else(
      is.finite(oews_big_six_mean_hourly_wage) & oews_big_six_mean_hourly_wage > 0,
      log(aewr / oews_big_six_mean_hourly_wage),
      NA_real_
    )
  ) %>%
  group_by(county_fips, aewr_region_id) %>%
  summarise(
    raw_E_p25_0810 = bite_safe_mean(ln_aewr_over_p25),
    raw_E_oews_0810 = bite_safe_mean(ln_aewr_over_oews),
    raw_E_sf_log_0810 = bite_safe_mean(shortfall_log_real),
    .groups = "drop"
  ) %>%
  mutate(
    E_p25 = bite_std_within_region(raw_E_p25_0810, aewr_region_id),
    E_oews = bite_std_within_region(raw_E_oews_0810, aewr_region_id),
    E_sf_log = bite_std_within_region(raw_E_sf_log_0810, aewr_region_id),
    E_p25_quartile = bite_make_quartiles(E_p25)
  ) %>%
  select(-aewr_region_id)

# --- Construct predetermined 2008-2010 baseline covariates (no time-varying BEA) ---
pop_2010 <- panel %>%
  filter(year == 2010L) %>%
  transmute(county_fips, ln_pop_2010 = ln_pop_census)

farm_0810 <- panel %>%
  filter(year %in% BITE_COVARIATE_YEARS) %>%
  group_by(county_fips) %>%
  summarise(
    farm_emp_share_0810 = bite_safe_mean(farm_emp_share),
    emp_farm_0810 = bite_safe_mean(emp_farm),
    .groups = "drop"
  ) %>%
  mutate(
    ln_emp_farm_0810 = if_else(
      is.finite(emp_farm_0810) & emp_farm_0810 >= 0,
      log1p(emp_farm_0810),
      NA_real_
    )
  )

county_frozen <- baseline_2008_2010 %>%
  left_join(pop_2010, by = "county_fips", relationship = "one-to-one") %>%
  left_join(farm_0810, by = "county_fips", relationship = "one-to-one")

stopifnot(
  nrow(county_frozen) == 3108L,
  !anyDuplicated(county_frozen$county_fips),
  all(is.finite(county_frozen$E_p25)),
  all(is.finite(county_frozen$E_sf_log)),
  !anyNA(county_frozen$E_p25_quartile)
)

# --- Merge frozen exposures & build design variables ---
bite_panel <- panel %>%
  left_join(county_frozen, by = "county_fips", relationship = "many-to-one") %>%
  mutate(
    ln_aewr = if_else(is.finite(aewr) & aewr > 0, log(aewr), NA_real_),
    ln_ppi_2012 = if_else(is.finite(ppi_2012) & ppi_2012 > 0, log(ppi_2012), NA_real_),
    ln_prevailing_ag_min_wage = if_else(
      is.finite(prevailing_ag_min_wage) & prevailing_ag_min_wage > 0,
      log(prevailing_ag_min_wage),
      NA_real_
    ),
    E_p25_x_ln_aewr = E_p25 * ln_aewr,
    E_oews_x_ln_aewr = E_oews * ln_aewr,
    E_sf_log_x_ln_aewr = E_sf_log * ln_aewr,
    E_p25_x_ln_ppi_2012 = E_p25 * ln_ppi_2012,
    any_h2a_gt0 = as.numeric(nbr_workers_certified_start_year > 0),
    any_h2a_ge1 = as.numeric(nbr_workers_certified_start_year >= 1),
    any_h2a_ge5 = as.numeric(nbr_workers_certified_start_year >= 5)
  )

assert_geo_columns(
  bite_panel,
  c("county_fips", "state_fips", "cz_id", "aewr_region_id")
)
stopifnot(
  nrow(bite_panel) == 55944L,
  !anyDuplicated(bite_panel[c("county_fips", "year")]),
  all(is.na(bite_panel$ln_aewr[bite_panel$year == 2009L])),
  all(is.finite(bite_panel$ln_aewr[bite_panel$year >= 2011L])),
  all(is.finite(bite_panel$ln_prevailing_ag_min_wage[bite_panel$year >= 2011L])),
  all(bite_panel$any_h2a_ge5 <= bite_panel$any_h2a_ge1),
  all(bite_panel$any_h2a_ge1 <= bite_panel$any_h2a_gt0)
)

write_parquet(bite_panel, path_processed("bite_county_year_panel.parquet"))

cat(
  sprintf(
    "Built bite_county_year_panel.parquet: %d rows (%d counties x %d years); finite E_p25=%d, E_oews=%d, E_sf_log=%d\n",
    nrow(bite_panel),
    n_distinct(bite_panel$county_fips),
    n_distinct(bite_panel$year),
    sum(is.finite(county_frozen$E_p25)),
    sum(is.finite(county_frozen$E_oews)),
    sum(is.finite(county_frozen$E_sf_log))
  )
)
