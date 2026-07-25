# Purpose: Audit rebuilt geographic artifacts without mutating them.
# Inputs: county-bearing A/B intermediate artifacts.
# Output: console diagnostics; exits nonzero on any contract violation.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)

contracts <- list(
  acs_1year_for_wages.parquet = "state_fips",
  acs_5year_for_immigrant_status_imputation.parquet = "state_fips",
  county_adjacency2010.parquet = c(
    "county_fips",
    "neighbor_county_fips"
  ),
  census_ag_hired_worker_duration_county.parquet = c(
    "state_fips",
    "county_code",
    "county_fips"
  ),
  qcew_county_ag_quarterly_employment.parquet = "county_fips",
  qwi_county_ag_quarterly_employment.parquet = "county_fips",
  oews_area_definitions.parquet = c(
    "state_fips",
    "county_code",
    "county_fips",
    "oews_area_code"
  ),
  croplandcros_county_crop_acres.parquet = "county_fips",
  price_index_fisher_county_year.parquet = "county_fips",
  bea_farm_nonfarm_emp.parquet = "county_fips",
  county_h2a_prediction_climate_basis_annual.parquet = c(
    "state_fips",
    "county_code",
    "county_fips"
  ),
  county_h2a_prediction_gnatsgo_soil_cells.parquet = "county_fips",
  h2a_aggregated.parquet = c(
    "state_fips",
    "county_code",
    "county_fips"
  ),
  h2a_prediction_using_elastic_net_continuous_basis.parquet = "county_fips",
  acs_czone_wage_quantiles.parquet = c("county_fips", "cz_id"),
  acs_state_ag_wage.parquet = "state_fips",
  oews_state_aggregated.parquet = "state_fips",
  qcew_state_ag_wage.parquet = "state_fips",
  aewr.parquet = "state_fips",
  fred_state_minwages.parquet = "state_fips",
  state_year_min_wage.parquet = "state_fips"
)

failures <- character()
for (artifact in names(contracts)) {
  artifact_path <- path_int(artifact)
  if (!file.exists(artifact_path)) {
    message("SKIP missing artifact: ", artifact)
    next
  }

  result <- tryCatch(
    {
      required <- contracts[[artifact]]
      geography <- dplyr::collect(
        dplyr::distinct(
          read_parquet(
            artifact_path,
            col_select = tidyselect::all_of(required),
            as_data_frame = FALSE
          )
        )
      )
      assert_geo_columns(geography, required)
      message("PASS ", artifact)
      NULL
    },
    error = function(error) {
      paste0(artifact, ": ", conditionMessage(error))
    }
  )

  if (!is.null(result)) {
    failures <- c(failures, result)
  }
}

if (length(failures) > 0L) {
  stop(
    "Geographic contract failures:\n- ",
    paste(failures, collapse = "\n- "),
    call. = FALSE
  )
}
