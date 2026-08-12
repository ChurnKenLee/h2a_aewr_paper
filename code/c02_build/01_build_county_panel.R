# Purpose: Build the design-neutral county-year analysis panel.
# Input: data/intermediate/county_year_merged.parquet.
# Output: data/processed/county_year_panel.parquet.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
library(arrow)
library(dplyr)
library(tidyr)

# docs-ground:start shared-panel-construction
h2a_zero_columns <- c(
  "nbr_workers_requested_all_years",
  "nbr_workers_certified_all_years",
  "man_hours_requested_all_years",
  "man_hours_certified_all_years",
  "nbr_applications_all_years",
  "nbr_workers_requested_start_year",
  "nbr_workers_certified_start_year",
  "man_hours_requested_start_year",
  "man_hours_certified_start_year",
  "cert_hours_with_hourly_wage_start_year",
  "cert_hours_with_hourly_wage_and_aewr_start_year",
  "cert_hours_at_aewr_start_year",
  "nominal_offered_wage_bill_certified_start_year",
  "nbr_applications_start_year",
  "nbr_employers_conservative_start_year",
  "nbr_employers_balanced_start_year",
  "nbr_employers_high_recall_start_year",
  "nbr_applications_certified_start_year",
  "nbr_applications_partial_start_year",
  "nbr_applications_denied_start_year",
  "nbr_applications_withdrawn_start_year",
  "nbr_workers_requested_fiscal_year",
  "nbr_workers_certified_fiscal_year",
  "man_hours_requested_fiscal_year",
  "man_hours_certified_fiscal_year",
  "nbr_applications_fiscal_year"
)

county_panel <- read_parquet(path_int("county_year_merged.parquet")) %>%
  mutate(
    across(all_of(h2a_zero_columns), \(value) replace_na(value, 0)),
    nbr_applications_emergency_start_year = case_when(
      year < 2021L ~ NA_real_,
      nbr_applications_start_year == 0 ~ 0,
      TRUE ~ nbr_applications_emergency_start_year
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
    ),
    aewr_cz_p25 = aewr_ppi - wage_p25,
    aewr_cz_p25_l1 = aewr_ppi_l1 - wage_p25_l1,
    emp_pop_ratio = if_else(
      is.finite(pop_census) & pop_census > 0,
      emp_tot / pop_census,
      NA_real_
    ),
    farm_emp_share = if_else(
      is.finite(emp_tot) & emp_tot > 0,
      emp_farm / emp_tot,
      NA_real_
    ),
    ln_aewr_ppi = if_else(aewr_ppi > 0, log(aewr_ppi), NA_real_),
    ln_pop_census = if_else(pop_census > 0, log(pop_census), NA_real_),
    share_farm_laborexp_prodexp = if_else(
      is.finite(farm_prodexp) & farm_prodexp > 0,
      farm_laborexpense / farm_prodexp,
      NA_real_
    ),
    share_farm_prodexp_cashandinc = if_else(
      is.finite(farm_cashandinc_ppi) & farm_cashandinc_ppi > 0,
      farm_prodexp_ppi / farm_cashandinc_ppi,
      NA_real_
    ),
    share_farm_crop_cashandinc = if_else(
      is.finite(farm_cashandinc) & farm_cashandinc > 0,
      farm_cashcrops / farm_cashandinc,
      NA_real_
    ),
    share_farm_animal_cashandinc = if_else(
      is.finite(farm_cashandinc) & farm_cashandinc > 0,
      farm_cashanimal / farm_cashandinc,
      NA_real_
    ),
    share_farm_govt_cashandinc = if_else(
      is.finite(farm_cashandinc) & farm_cashandinc > 0,
      farm_govpayments / farm_cashandinc,
      NA_real_
    ),
    any_cropland_2007 = census_cropland_2007 > 0
  ) %>%
  filter(!is.na(aewr), !is.na(aewr_region_id)) %>%
  arrange(county_fips, year) %>%
  group_by(county_fips) %>%
  mutate(
    ln_pop_census_l1 = if_else(
      lag(year) == year - 1L,
      lag(ln_pop_census),
      NA_real_
    ),
    farm_emp_share_l1 = if_else(
      lag(year) == year - 1L,
      lag(farm_emp_share),
      NA_real_
    ),
    emp_pop_ratio_l1 = if_else(
      lag(year) == year - 1L,
      lag(emp_pop_ratio),
      NA_real_
    )
  ) %>%
  ungroup()

baseline_farm_employment <- county_panel %>%
  filter(year == 2011L) %>%
  select(county_fips, emp_farm_2011 = emp_farm)

border_commuting_zones <- county_panel %>%
  distinct(cz_id, aewr_region_id) %>%
  count(cz_id, name = "aewr_region_count") %>%
  transmute(cz_id, border_cz = aewr_region_count > 1L)

county_panel <- county_panel %>%
  left_join(
    baseline_farm_employment,
    by = "county_fips",
    relationship = "many-to-one"
  ) %>%
  left_join(
    border_commuting_zones,
    by = "cz_id",
    relationship = "many-to-one"
  ) %>%
  mutate(
    h2a_req_share_farm_workers_start_year = if_else(
      is.finite(emp_farm) & emp_farm > 0,
      nbr_workers_requested_start_year / emp_farm,
      NA_real_
    ),
    h2a_cert_share_farm_workers_start_year = if_else(
      is.finite(emp_farm) & emp_farm > 0,
      nbr_workers_certified_start_year / emp_farm,
      NA_real_
    ),
    h2a_req_share_farm_workers_2011_start_year = if_else(
      is.finite(emp_farm_2011) & emp_farm_2011 > 0,
      nbr_workers_requested_start_year / emp_farm_2011,
      NA_real_
    ),
    h2a_cert_share_farm_workers_2011_start_year = if_else(
      is.finite(emp_farm_2011) & emp_farm_2011 > 0,
      nbr_workers_certified_start_year / emp_farm_2011,
      NA_real_
    ),
    h2a_cert_hours_per_farm_worker_2011_start_year = if_else(
      is.finite(emp_farm_2011) & emp_farm_2011 > 0,
      man_hours_certified_start_year / emp_farm_2011,
      NA_real_
    ),
    h2a_applications_per_farm_worker_2011_start_year = if_else(
      is.finite(emp_farm_2011) & emp_farm_2011 > 0,
      nbr_applications_start_year / emp_farm_2011,
      NA_real_
    ),
    h2a_cert_positions_per_application_start_year = if_else(
      is.finite(nbr_applications_start_year) &
        nbr_applications_start_year > 0,
      nbr_workers_certified_start_year / nbr_applications_start_year,
      NA_real_
    ),
    h2a_cert_hours_per_position_start_year = if_else(
      is.finite(nbr_workers_certified_start_year) &
        nbr_workers_certified_start_year > 0,
      man_hours_certified_start_year / nbr_workers_certified_start_year,
      NA_real_
    ),
    # Certified applications include partial certifications.
    h2a_application_certification_rate_start_year = if_else(
      is.finite(nbr_applications_start_year) &
        nbr_applications_start_year > 0,
      nbr_applications_certified_start_year /
        nbr_applications_start_year,
      NA_real_
    ),
    h2a_application_partial_certification_rate_start_year = if_else(
      is.finite(nbr_applications_start_year) &
        nbr_applications_start_year > 0,
      nbr_applications_partial_start_year /
        nbr_applications_start_year,
      NA_real_
    ),
    h2a_application_denial_rate_start_year = if_else(
      is.finite(nbr_applications_start_year) &
        nbr_applications_start_year > 0,
      nbr_applications_denied_start_year / nbr_applications_start_year,
      NA_real_
    ),
    h2a_application_withdrawal_rate_start_year = if_else(
      is.finite(nbr_applications_start_year) &
        nbr_applications_start_year > 0,
      nbr_applications_withdrawn_start_year /
        nbr_applications_start_year,
      NA_real_
    )
  )
# docs-ground:end shared-panel-construction

# docs-ground:start shared-panel-contract
prediction_contract <- county_panel %>%
  filter(!is.na(h2a_prediction_cutoff_year)) %>%
  select(
    county_fips,
    h2a_prediction_cutoff_year,
    h2a_prediction_model_spec,
    predicted_h2a_count,
    bea_farm_emp_2011,
    h2a_predicted_share_2011,
    emp_farm_2011
  ) %>%
  distinct()

if (
  nrow(prediction_contract) == 0L ||
    anyDuplicated(prediction_contract$county_fips) > 0L ||
    any(
      prediction_contract$h2a_prediction_cutoff_year !=
        H2A_PREDICTION_CUTOFF_YEAR
    ) ||
    !identical(
      unique(prediction_contract$h2a_prediction_model_spec),
      H2A_PREDICTION_MODEL_SPEC
    ) ||
    any(
      !is.finite(prediction_contract$predicted_h2a_count) |
        prediction_contract$predicted_h2a_count < 0
    ) ||
    any(
      !is.finite(prediction_contract$emp_farm_2011) |
        prediction_contract$emp_farm_2011 <= 0
    ) ||
    any(
      !is.finite(prediction_contract$bea_farm_emp_2011) |
        prediction_contract$bea_farm_emp_2011 <= 0
    ) ||
    any(
      abs(
        prediction_contract$bea_farm_emp_2011 -
          prediction_contract$emp_farm_2011
      ) >
        1e-8,
      na.rm = TRUE
    ) ||
    any(
      !is.finite(prediction_contract$h2a_predicted_share_2011) |
        prediction_contract$h2a_predicted_share_2011 < 0
    ) ||
    !isTRUE(all.equal(
      prediction_contract$h2a_predicted_share_2011,
      prediction_contract$predicted_h2a_count /
        prediction_contract$bea_farm_emp_2011,
      tolerance = 2e-6,
      check.attributes = FALSE
    ))
) {
  stop(
    "The shared panel must use one valid canonical static prediction per county.",
    call. = FALSE
  )
}

if (
  nrow(county_panel) == 0L ||
    anyDuplicated(county_panel[c("county_fips", "year")]) > 0L
) {
  stop("county_year_panel must have unique county-year keys.", call. = FALSE)
}
# docs-ground:end shared-panel-contract

qcew_shared_series <- c("crop_sector", "animal_sector", "all_sectors")
for (series in qcew_shared_series) {
  disclosed <- county_panel[[paste0("qcew_", series, "_disclosed")]]
  employment <- county_panel[[
    paste0("qcew_", series, "_annual_avg_emplvl")
  ]]
  wages <- county_panel[[paste0("qcew_", series, "_total_annual_wages")]]
  observed <- !is.na(disclosed)
  invalid <- observed & (
    (disclosed &
      (!is.finite(employment) | employment < 0 |
        !is.finite(wages) | wages < 0)) |
      (!disclosed & (!is.na(employment) | !is.na(wages)))
  )
  if (any(invalid)) {
    stop(
      "Shared-panel QCEW disclosure semantics are invalid for ",
      series,
      ".",
      call. = FALSE
    )
  }
}

# docs-ground:start shared-panel-output
write_parquet(
  county_panel,
  path_processed("county_year_panel.parquet")
)
# docs-ground:end shared-panel-output
