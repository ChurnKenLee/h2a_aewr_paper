# Purpose: Attach the two publication instruments to the full county panel.
# Output: data/processed/panel_iv_county_year.parquet.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "panel_iv", "design.R"))
library(arrow)
library(dplyr)

county_panel <- read_parquet(
  path_processed("county_year_panel.parquet")
) %>%
  mutate(
    year = as.integer(year),
    panel_iv_target_unit_id = make_panel_iv_target_unit_id(
      cz_id,
      aewr_region_id
    )
  )

cluster_assignments <- read_parquet(
  path_int("panel_iv_target_clusters.parquet")
) %>%
  filter(iv_k == DISSIMILARITY_IV_PRIMARY_K) %>%
  transmute(
    panel_iv_target_unit_id,
    aewr_region_id,
    target_cluster = iv_cluster,
    aewr_iv_cluster_id = make_dissimilarity_cluster_id(
      aewr_region_id,
      target_cluster
    )
  ) %>%
  distinct()

cluster_contract <- cluster_assignments %>%
  distinct(aewr_region_id, target_cluster, aewr_iv_cluster_id) %>%
  count(aewr_region_id, name = "subregions")
if (
  nrow(cluster_contract) != 17L ||
    any(cluster_contract$subregions != DISSIMILARITY_IV_PRIMARY_K) ||
    n_distinct(cluster_assignments$aewr_iv_cluster_id) != 85L
) {
  stop(
    "The publication design requires five subregions in each of 17 AEWR regions.",
    call. = FALSE
  )
}

instrument_long <- read_parquet(
  path_int("panel_iv_instrument_cluster_year.parquet")
) %>%
  filter(
    instrument_spec_label %in%
      c(
        DISSIMILARITY_IV_WAGE_ONLY_INSTRUMENT_LABEL,
        DISSIMILARITY_IV_PRIMARY_INSTRUMENT_LABEL
      ),
    is.na(weight_draw_id)
  )

instrument_keys <- c(
  "aewr_region_id",
  "target_cluster",
  "source_year",
  "policy_year"
)

wage_only_instrument <- instrument_long %>%
  filter(
    instrument_spec_label == DISSIMILARITY_IV_WAGE_ONLY_INSTRUMENT_LABEL
  ) %>%
  transmute(
    across(all_of(instrument_keys)),
    z_wage_only_real = z_dissimilarity_real,
    wage_only_instrument_available = instrument_available
  )

wage_seasonal_composition_instrument <- instrument_long %>%
  filter(
    instrument_spec_label == DISSIMILARITY_IV_PRIMARY_INSTRUMENT_LABEL
  ) %>%
  transmute(
    across(all_of(instrument_keys)),
    z_wage_seasonal_composition_real = z_dissimilarity_real,
    wage_seasonal_composition_instrument_available = instrument_available
  )

instrument_pair <- wage_only_instrument %>%
  inner_join(
    wage_seasonal_composition_instrument,
    by = instrument_keys,
    relationship = "one-to-one"
  )

if (
  nrow(instrument_pair) !=
    85L *
      (DISSIMILARITY_IV_POLICY_END_YEAR -
        DISSIMILARITY_IV_POLICY_START_YEAR +
        1L) ||
    any(instrument_pair$source_year != instrument_pair$policy_year - 1L)
) {
  stop(
    "Instrument grid must contain all 85 subregions and use source year t-1.",
    call. = FALSE
  )
}

panel_iv <- county_panel %>%
  filter(
    year >= DISSIMILARITY_IV_POLICY_START_YEAR,
    year <= DISSIMILARITY_IV_POLICY_END_YEAR
  ) %>%
  inner_join(
    cluster_assignments,
    by = c("panel_iv_target_unit_id", "aewr_region_id"),
    relationship = "many-to-one"
  ) %>%
  left_join(
    instrument_pair,
    by = c(
      "aewr_region_id",
      "target_cluster",
      "year" = "policy_year"
    ),
    relationship = "many-to-one"
  )

prediction_contract <- panel_iv %>%
  filter(!is.na(h2a_predicted_share_2011)) %>%
  select(
    county_fips,
    h2a_prediction_cutoff_year,
    h2a_prediction_model_spec,
    predicted_h2a_count,
    bea_farm_emp_2011,
    h2a_predicted_share_2011
  ) %>%
  distinct()

if (
  nrow(prediction_contract) == 0L ||
    anyDuplicated(prediction_contract$county_fips) > 0L ||
    any(is.na(prediction_contract$h2a_prediction_cutoff_year)) ||
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
      !is.finite(prediction_contract$bea_farm_emp_2011) |
        prediction_contract$bea_farm_emp_2011 <= 0
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
    "The IV panel must use one canonical static H-2A prediction per county.",
    call. = FALSE
  )
}

propensity_mean <- mean(prediction_contract$h2a_predicted_share_2011)
propensity_sd <- sd(prediction_contract$h2a_predicted_share_2011)
if (!is.finite(propensity_mean) || !is.finite(propensity_sd) || propensity_sd <= 0) {
  stop("The eligible IV-county PPML propensity must have positive variation.",
    call. = FALSE)
}

propensity_by_county <- prediction_contract %>%
  transmute(
    county_fips,
    h2a_ppml_static_propensity_z =
      (h2a_predicted_share_2011 - propensity_mean) / propensity_sd
  )

if (
  abs(mean(propensity_by_county$h2a_ppml_static_propensity_z)) > 1e-12 ||
    abs(sd(propensity_by_county$h2a_ppml_static_propensity_z) - 1) > 1e-12
) {
  stop("Static PPML standardization must weight each eligible IV county equally.",
    call. = FALSE)
}

panel_iv <- panel_iv %>%
  left_join(
    propensity_by_county,
    by = "county_fips",
    relationship = "many-to-one"
  ) %>%
  mutate(year_centered = as.integer(year) - 2011L)

if (
  nrow(panel_iv) == 0L ||
    anyDuplicated(panel_iv[c("county_fips", "year")]) > 0L
) {
  stop(
    "panel_iv_county_year must have unique county-year keys.",
    call. = FALSE
  )
}

if (
  n_distinct(panel_iv$aewr_iv_cluster_id) != 85L ||
    !identical(
      sort(unique(panel_iv$year)),
      seq.int(
        DISSIMILARITY_IV_POLICY_START_YEAR,
        DISSIMILARITY_IV_POLICY_END_YEAR
      )
    )
) {
  stop(
    "The county IV panel must retain all 85 subregions and policy years.",
    call. = FALSE
  )
}

legacy_panel <- path_processed("panel_iv_cluster_year.parquet")
if (file.exists(legacy_panel)) {
  unlink(legacy_panel)
}

write_parquet(
  panel_iv,
  path_processed("panel_iv_county_year.parquet")
)
