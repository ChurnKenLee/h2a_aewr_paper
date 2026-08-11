# Purpose: Build the annual county H-2A panel and select its canonical prediction.
# Inputs: h2a_aggregated.parquet and cutoff-specific elastic-net predictions.
# Outputs: h2a_predict.parquet and h2a_data_year.parquet.
# Run after: code/b01_derived/01_h2a_aggregation_nodupes.R and
# code/b01_derived/08_h2a_prediction_from_estimated_weights.py.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
library(arrow)
library(dplyr)

h2a_data <- read_parquet(
  file = path_int("h2a_aggregated.parquet")
)
h2a_predict <- read_parquet(
  file = path_int("h2a_prediction_using_elastic_net_by_cutoff.parquet")
) %>%
  filter(cutoff_year == H2A_PREDICTION_CUTOFF_YEAR) %>%
  transmute(
    county_fips,
    h2a_prediction_cutoff_year = as.integer(cutoff_year),
    h2a_prediction_model_spec = model_spec,
    predicted_h2a_count,
    bea_farm_emp_2011,
    h2a_predicted_share_2011 = predicted_h2a_share_2011
  )

if (
  nrow(h2a_predict) == 0L ||
    anyDuplicated(h2a_predict$county_fips) > 0L ||
    any(
      h2a_predict$h2a_prediction_cutoff_year !=
        H2A_PREDICTION_CUTOFF_YEAR
    ) ||
    !identical(
      unique(h2a_predict$h2a_prediction_model_spec),
      H2A_PREDICTION_MODEL_SPEC
    ) ||
    any(
      !is.finite(h2a_predict$predicted_h2a_count) |
        h2a_predict$predicted_h2a_count < 0
    ) ||
    any(
      !is.finite(h2a_predict$bea_farm_emp_2011) |
        h2a_predict$bea_farm_emp_2011 <= 0
    ) ||
    any(
      !is.finite(h2a_predict$h2a_predicted_share_2011) |
        h2a_predict$h2a_predicted_share_2011 < 0
    ) ||
    !isTRUE(all.equal(
      h2a_predict$h2a_predicted_share_2011,
      h2a_predict$predicted_h2a_count /
        h2a_predict$bea_farm_emp_2011,
      tolerance = 2e-6,
      check.attributes = FALSE
    ))
) {
  stop(
    "The canonical H-2A prediction violates its static county contract.",
    call. = FALSE
  )
}
write_parquet(h2a_predict, path_int("h2a_predict.parquet"))

h2a_data <- h2a_data %>%
  filter(year > 2007, year <= 2022, !is.na(county_fips)) %>%
  select(-state_fips, -county_code)

write_parquet(h2a_data, path_int("h2a_data_year.parquet"))
