# Purpose: Add the treatment classification and post period used by the DiD.
# Output: data/processed/did_county_year_panel.parquet.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
library(arrow)
library(dplyr)

# docs-ground:start did-treatment-inputs
true_share_cutoff <- 0.01
predicted_share_cutoff <- 0.01

county_panel <- read_parquet(
  path_processed("county_year_panel.parquet")
)

treatment_groups <- county_panel %>%
  # The PPML propensity is static; 2008 selects the observed H-2A baseline
  # used with it to define time-invariant treatment groups.
  filter(year == 2008L) %>%
  transmute(
    county_fips,
    county_treatment_group_classification = case_when(
      h2a_predicted_share_2011 > predicted_share_cutoff &
        h2a_cert_share_farm_workers_2011_start_year >
          true_share_cutoff ~ "always takers",
      h2a_predicted_share_2011 > predicted_share_cutoff &
        h2a_cert_share_farm_workers_2011_start_year <
          true_share_cutoff ~ "adopters",
      h2a_predicted_share_2011 < predicted_share_cutoff &
        h2a_cert_share_farm_workers_2011_start_year >
          true_share_cutoff ~ "defiers",
      h2a_predicted_share_2011 < predicted_share_cutoff &
        h2a_cert_share_farm_workers_2011_start_year <
          true_share_cutoff ~ "never takers"
    ),
    county_simple_treatment_groups = if_else(
      county_treatment_group_classification == "always takers",
      "always takers",
      "exposed adopters",
      missing = NA_character_
    )
  )

did_panel <- county_panel %>%
  left_join(
    treatment_groups,
    by = "county_fips",
    relationship = "many-to-one"
  ) %>%
  mutate(post = year > 2011L)
# docs-ground:end did-treatment-inputs

prediction_contract <- did_panel %>%
  filter(!is.na(h2a_predicted_share_2011)) %>%
  distinct(county_fips, h2a_predicted_share_2011)

if (
  nrow(did_panel) == 0L ||
    anyDuplicated(did_panel[c("county_fips", "year")]) > 0L ||
    anyDuplicated(treatment_groups$county_fips) > 0L ||
    anyDuplicated(prediction_contract$county_fips) > 0L ||
    any(
      !is.na(did_panel$h2a_prediction_cutoff_year) &
        did_panel$h2a_prediction_cutoff_year !=
          H2A_PREDICTION_CUTOFF_YEAR
    ) ||
    !identical(
      unique(did_panel$h2a_prediction_model_spec[
        !is.na(did_panel$h2a_prediction_model_spec)
      ]),
      H2A_PREDICTION_MODEL_SPEC
    )
) {
  stop(
    "did_county_year_panel must have unique county-year keys.",
    call. = FALSE
  )
}

write_parquet(
  did_panel,
  path_processed("did_county_year_panel.parquet")
)
