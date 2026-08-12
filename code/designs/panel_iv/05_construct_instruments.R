# Purpose: Construct county-weighted Panel-IV OEWS hourly-wage instruments.
# Inputs: fixed clusters, pseudo-FLS county entropy weights, shared county
# source fields, the Census/QCEW/QWI/BEA employment frame, and PPI.
# Outputs:
#   data/intermediate/panel_iv_county_donor_frame.parquet
#   data/intermediate/panel_iv_instrument_cluster_year.parquet

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "helpers.R"))
library(arrow)
library(dplyr)
library(stringr)
library(tidyr)

source_years <- seq.int(
  DISSIMILARITY_IV_POLICY_START_YEAR - 1L,
  DISSIMILARITY_IV_POLICY_END_YEAR - 1L
)

publication_specs <- tibble::tribble(
  ~weight_specification, ~moment_spec, ~is_primary, ~instrument_spec_label,
  DISSIMILARITY_IV_WAGE_ONLY_WEIGHT_SPECIFICATION,
  DISSIMILARITY_IV_WAGE_ONLY_MOMENT_SPEC,
  FALSE,
  DISSIMILARITY_IV_WAGE_ONLY_INSTRUMENT_LABEL,
  DISSIMILARITY_IV_PRIMARY_WEIGHT_SPECIFICATION,
  DISSIMILARITY_IV_PRIMARY_MOMENT_SPEC,
  TRUE,
  DISSIMILARITY_IV_PRIMARY_INSTRUMENT_LABEL
)

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

required_wage_columns <- c(
  "oews_big_six_mean_hourly_wage",
  "oews_wage_observed",
  "oews_wage_covered_occupation_count",
  "oews_mapped_area_count",
  "oews_primary_area_share",
  "oews_wage_observed_mapping_share"
)
missing_wage_columns <- setdiff(required_wage_columns, names(county_panel))
if (length(missing_wage_columns) > 0L) {
  stop(
    "The shared panel is missing donor-wage fields: ",
    paste(missing_wage_columns, collapse = ", "),
    call. = FALSE
  )
}

iv_clusters <- read_parquet(path_int("panel_iv_target_clusters.parquet"))
iv_donor_clusters <- read_parquet(path_int("panel_iv_donor_clusters.parquet"))
frame_employment <- read_parquet(path_int("panel_iv_fls_frame.parquet")) %>%
  filter(
    source_year %in% source_years,
    weight_spec == DISSIMILARITY_IV_FRAME_WEIGHT_SPEC,
    annual_update_spec == DISSIMILARITY_IV_ANNUAL_UPDATE_SPEC,
    is.na(weight_draw_id)
  ) %>%
  select(
    county_fips,
    source_year,
    frame_employment_mass,
    state_rake_factor,
    annual_update_source,
    annual_growth_source,
    qcew_strict_complete,
    qwi_annual_fallback_used,
    bea_annual_fallback_used,
    quality_flags
  )

recovered_weights <- read_parquet(path_int(
  "panel_iv_fls_county_weight_summary.parquet"
)) %>%
  filter(
    source_year %in% source_years,
    weight_spec == DISSIMILARITY_IV_PRIMARY_WEIGHT_SPEC,
    baseline_weight_spec == DISSIMILARITY_IV_FRAME_WEIGHT_SPEC,
    near(rho, DISSIMILARITY_IV_PRIMARY_RHO),
    near(
      kappa_multiplier,
      DISSIMILARITY_IV_PRIMARY_KAPPA_MULTIPLIER
    ),
    is.na(weight_draw_id),
    str_detect(center_solver_status, "^calibrated"),
    is.finite(calibrated_center_weight),
    calibrated_center_weight >= 0
  ) %>%
  inner_join(
    publication_specs,
    by = c(
      "specification" = "weight_specification",
      "moment_spec",
      "is_primary"
    ),
    relationship = "many-to-one"
  ) %>%
  transmute(
    county_fips,
    aewr_region_id,
    source_year = as.integer(source_year),
    weight_specification = specification,
    moment_spec,
    is_primary,
    instrument_spec_label,
    frame_prior_weight,
    calibrated_center_weight,
    active_moment_count,
    inactive_moment_count,
    calibrated_effective_county_count,
    maximum_calibrated_county_weight
  )

weight_sums <- recovered_weights %>%
  group_by(aewr_region_id, source_year, weight_specification) %>%
  summarise(
    prior_sum = sum(frame_prior_weight),
    calibrated_sum = sum(calibrated_center_weight),
    .groups = "drop"
  )
if (
  nrow(weight_sums) != 17L * length(source_years) * 2L ||
    any(abs(weight_sums$prior_sum - 1) > 1e-10) ||
    any(abs(weight_sums$calibrated_sum - 1) > 1e-10)
) {
  stop("Recovered county weights must sum to one by region-year-specification.",
    call. = FALSE)
}

primary_clusters <- iv_clusters %>%
  filter(iv_k == DISSIMILARITY_IV_PRIMARY_K) %>%
  transmute(
    panel_iv_target_unit_id,
    aewr_region_id,
    iv_k,
    donor_cluster = iv_cluster
  )

county_clusters <- county_panel %>%
  distinct(county_fips, aewr_region_id, panel_iv_target_unit_id) %>%
  inner_join(
    primary_clusters,
    by = c("panel_iv_target_unit_id", "aewr_region_id"),
    relationship = "many-to-one"
  )

primary_donor_map <- iv_donor_clusters %>%
  filter(
    iv_k == DISSIMILARITY_IV_PRIMARY_K,
    donor_cluster_count == DISSIMILARITY_IV_PRIMARY_DONOR_COUNT
  )

ppi <- read_parquet(path_int("ppi_2012.parquet")) %>%
  transmute(
    source_year = as.integer(year),
    source_year_ppi_2012 = as.numeric(ppi_2012)
  )

county_source <- county_panel %>%
  filter(year %in% source_years) %>%
  transmute(
    county_fips,
    aewr_region_id,
    source_year = year,
    oews_area_code,
    across(all_of(required_wage_columns))
  ) %>%
  left_join(
    frame_employment,
    by = c("county_fips", "source_year"),
    relationship = "one-to-one"
  ) %>%
  left_join(ppi, by = "source_year", relationship = "many-to-one")

wage_selection <- select_panel_iv_donor_wage(
  county_source$oews_big_six_mean_hourly_wage,
  county_source$oews_wage_observed
)

county_source <- bind_cols(county_source, wage_selection) %>%
  mutate(
    donor_real_hourly_wage = if_else(
      is.finite(donor_nominal_hourly_wage) &
        donor_nominal_hourly_wage > 0 &
        is.finite(source_year_ppi_2012) &
        source_year_ppi_2012 > 0,
      donor_nominal_hourly_wage / source_year_ppi_2012,
      NA_real_
    ),
    donor_wage_available = is.finite(donor_real_hourly_wage) &
      donor_real_hourly_wage > 0,
    donor_wage_spec = DISSIMILARITY_IV_DONOR_WAGE_SPEC,
    donor_wage_geography = DISSIMILARITY_IV_DONOR_WAGE_GEOGRAPHY
  ) %>%
  inner_join(
    county_clusters,
    by = c("county_fips", "aewr_region_id"),
    relationship = "many-to-one"
  )

target_area_codes <- county_source %>%
  filter(!is.na(oews_area_code)) %>%
  transmute(
    aewr_region_id,
    source_year,
    target_cluster = donor_cluster,
    oews_area_code,
    target_area_overlap = TRUE
  ) %>%
  distinct()

candidate_counties <- county_source %>%
  inner_join(
    primary_donor_map,
    by = c("aewr_region_id", "iv_k", "donor_cluster"),
    relationship = "many-to-many"
  ) %>%
  left_join(
    target_area_codes,
    by = c(
      "aewr_region_id",
      "source_year",
      "target_cluster",
      "oews_area_code"
    ),
    relationship = "many-to-one"
  ) %>%
  mutate(target_area_overlap = coalesce(target_area_overlap, FALSE))

candidate_support <- candidate_counties %>%
  group_by(aewr_region_id, target_cluster, source_year) %>%
  summarise(
    candidate_donor_clusters = n_distinct(donor_cluster),
    candidate_donor_units = n_distinct(panel_iv_target_unit_id),
    candidate_donor_counties = n_distinct(county_fips),
    target_overlap_counties_excluded = n_distinct(
      county_fips[target_area_overlap]
    ),
    target_overlap_areas_excluded = n_distinct(
      oews_area_code[target_area_overlap & !is.na(oews_area_code)]
    ),
    .groups = "drop"
  )

county_donor_frame <- candidate_counties %>%
  inner_join(
    recovered_weights,
    by = c("county_fips", "aewr_region_id", "source_year"),
    relationship = "many-to-many"
  ) %>%
  mutate(
    calibrated_weight_valid = is.finite(calibrated_center_weight) &
      calibrated_center_weight > 0,
    donor_eligible = calibrated_weight_valid &
      !target_area_overlap &
      donor_wage_available
  ) %>%
  group_by(
    aewr_region_id,
    target_cluster,
    source_year,
    weight_specification,
    moment_spec,
    is_primary,
    instrument_spec_label
  ) %>%
  mutate(
    candidate_calibrated_weight_mass = sum(
      if_else(calibrated_weight_valid, calibrated_center_weight, 0),
      na.rm = TRUE
    ),
    nonoverlap_calibrated_weight_mass = sum(
      if_else(
        calibrated_weight_valid & !target_area_overlap,
        calibrated_center_weight,
        0
      ),
      na.rm = TRUE
    ),
    eligible_calibrated_weight_mass = sum(
      if_else(donor_eligible, calibrated_center_weight, 0),
      na.rm = TRUE
    ),
    instrument_weight = if_else(
      donor_eligible & eligible_calibrated_weight_mass > 0,
      calibrated_center_weight / eligible_calibrated_weight_mass,
      NA_real_
    )
  ) %>%
  ungroup() %>%
  mutate(
    policy_year = source_year + 1L,
    aewr_iv_cluster_id = make_dissimilarity_cluster_id(
      aewr_region_id,
      target_cluster
    ),
    weight_spec = DISSIMILARITY_IV_PRIMARY_WEIGHT_SPEC,
    baseline_weight_spec = DISSIMILARITY_IV_FRAME_WEIGHT_SPEC,
    weight_component = DISSIMILARITY_IV_PRIMARY_WEIGHT_COMPONENT,
    wage_target_used = TRUE,
    rho = DISSIMILARITY_IV_PRIMARY_RHO,
    kappa_multiplier = DISSIMILARITY_IV_PRIMARY_KAPPA_MULTIPLIER,
    weight_draw_id = NA_integer_,
    instrument_family = DISSIMILARITY_IV_INSTRUMENT_FAMILY,
    aggregation_spec = DISSIMILARITY_IV_AGGREGATION_SPEC,
    baseline_frame_proxy = DISSIMILARITY_IV_BASELINE_FRAME_PROXY,
    annual_update_spec = DISSIMILARITY_IV_ANNUAL_UPDATE_SPEC,
    geographic_allocation_spec = DISSIMILARITY_IV_GEOGRAPHIC_ALLOCATION_SPEC,
    cluster_size_rule = DISSIMILARITY_IV_CLUSTER_SIZE_RULE,
    donor_cluster_count = DISSIMILARITY_IV_PRIMARY_DONOR_COUNT
  ) %>%
  arrange(
    aewr_region_id,
    target_cluster,
    source_year,
    desc(is_primary),
    donor_cluster,
    county_fips
  )

assert_geo_columns(
  county_donor_frame,
  c("county_fips", "aewr_region_id"),
  allow_na = character()
)
if (
  nrow(county_donor_frame) == 0L ||
    anyDuplicated(county_donor_frame[c(
      "aewr_region_id",
      "target_cluster",
      "source_year",
      "weight_specification",
      "county_fips"
    )]) > 0L
) {
  stop("The county donor frame must be nonempty and unique by target/spec/county.",
    call. = FALSE)
}

observed_instruments <- county_donor_frame %>%
  group_by(
    aewr_region_id,
    target_cluster,
    source_year,
    instrument_spec_label
  ) %>%
  summarise(
    z_dissimilarity_nominal_raw = sum(
      instrument_weight * donor_nominal_hourly_wage,
      na.rm = TRUE
    ),
    z_dissimilarity_real_raw = sum(
      instrument_weight * donor_real_hourly_wage,
      na.rm = TRUE
    ),
    instrument_weight_sum = sum(instrument_weight, na.rm = TRUE),
    observed_donor_clusters = n_distinct(donor_cluster[donor_eligible]),
    observed_donor_units = n_distinct(
      panel_iv_target_unit_id[donor_eligible]
    ),
    observed_donor_counties = n_distinct(county_fips[donor_eligible]),
    observed_oews_areas = n_distinct(
      oews_area_code[donor_eligible & !is.na(oews_area_code)]
    ),
    instrument_weight_ess = {
      finite_weights <- instrument_weight[
        is.finite(instrument_weight) & instrument_weight > 0
      ]
      if (length(finite_weights) > 0L) {
        1 / sum(finite_weights^2)
      } else {
        NA_real_
      }
    },
    maximum_instrument_county_weight = {
      finite_weights <- instrument_weight[
        is.finite(instrument_weight) & instrument_weight > 0
      ]
      if (length(finite_weights) > 0L) {
        max(finite_weights)
      } else {
        NA_real_
      }
    },
    oews_wage_proxy_coverage_weight = if_else(
      nonoverlap_calibrated_weight_mass[[1]] > 0,
      eligible_calibrated_weight_mass[[1]] /
        nonoverlap_calibrated_weight_mass[[1]],
      NA_real_
    ),
    qcew_employment_coverage_weight = if_else(
      nonoverlap_calibrated_weight_mass[[1]] > 0,
      sum(
        calibrated_center_weight[
          calibrated_weight_valid &
            !target_area_overlap &
            qcew_strict_complete
        ],
        na.rm = TRUE
      ) / nonoverlap_calibrated_weight_mass[[1]],
      NA_real_
    ),
    candidate_calibrated_weight_mass = first(
      candidate_calibrated_weight_mass
    ),
    nonoverlap_calibrated_weight_mass = first(
      nonoverlap_calibrated_weight_mass
    ),
    eligible_calibrated_weight_mass = first(
      eligible_calibrated_weight_mass
    ),
    active_moment_count = first(active_moment_count),
    inactive_moment_count = first(inactive_moment_count),
    source_year_ppi_2012 = first(source_year_ppi_2012),
    source_annual_update_methods = paste(
      sort(unique(annual_update_source)),
      collapse = "|"
    ),
    donor_wage_spec = first(donor_wage_spec),
    donor_wage_geography = first(donor_wage_geography),
    .groups = "drop"
  )

fixed_cluster_counts <- primary_clusters %>%
  count(
    aewr_region_id,
    donor_cluster,
    name = "fixed_donor_cluster_units"
  )

target_geometry <- primary_donor_map %>%
  left_join(
    fixed_cluster_counts,
    by = c("aewr_region_id", "donor_cluster"),
    relationship = "many-to-one"
  ) %>%
  group_by(aewr_region_id, target_cluster, iv_k) %>%
  summarise(
    donor_cluster_count = first(donor_cluster_count),
    selected_donor_clusters = n_distinct(donor_cluster),
    selected_donor_cluster_ids = paste(
      paste0("c", donor_cluster[order(donor_rank)]),
      collapse = "|"
    ),
    minimum_fixed_donor_cluster_units = min(
      fixed_donor_cluster_units
    ),
    fixed_donor_units = sum(fixed_donor_cluster_units),
    nearest_selected_donor_distance = min(donor_cluster_distance),
    farthest_selected_donor_distance = max(donor_cluster_distance),
    .groups = "drop"
  )

instrument_grid <- target_geometry %>%
  crossing(source_year = source_years) %>%
  crossing(publication_specs) %>%
  left_join(
    observed_instruments,
    by = c(
      "aewr_region_id",
      "target_cluster",
      "source_year",
      "instrument_spec_label"
    )
  ) %>%
  left_join(
    candidate_support,
    by = c("aewr_region_id", "target_cluster", "source_year")
  ) %>%
  mutate(
    across(
      c(
        observed_donor_clusters,
        observed_donor_units,
        observed_donor_counties,
        observed_oews_areas,
        candidate_donor_clusters,
        candidate_donor_units,
        candidate_donor_counties,
        target_overlap_counties_excluded,
        target_overlap_areas_excluded
      ),
      ~ replace_na(as.integer(.x), 0L)
    ),
    instrument_available =
      observed_donor_clusters ==
        DISSIMILARITY_IV_MIN_OBSERVED_DONOR_CLUSTERS &
        is.finite(instrument_weight_sum) &
        abs(instrument_weight_sum - 1) <= 1e-10 &
        is.finite(z_dissimilarity_real_raw) &
        z_dissimilarity_real_raw > 0,
    z_dissimilarity_nominal = if_else(
      instrument_available,
      z_dissimilarity_nominal_raw,
      NA_real_
    ),
    z_dissimilarity_real = if_else(
      instrument_available,
      z_dissimilarity_real_raw,
      NA_real_
    ),
    policy_year = source_year + 1L,
    aewr_iv_cluster_id = make_dissimilarity_cluster_id(
      aewr_region_id,
      target_cluster
    ),
    donor_unit_coverage = if_else(
      fixed_donor_units > 0,
      observed_donor_units / fixed_donor_units,
      NA_real_
    ),
    weight_spec = DISSIMILARITY_IV_PRIMARY_WEIGHT_SPEC,
    baseline_weight_spec = DISSIMILARITY_IV_FRAME_WEIGHT_SPEC,
    weight_component = DISSIMILARITY_IV_PRIMARY_WEIGHT_COMPONENT,
    wage_target_used = TRUE,
    rho = DISSIMILARITY_IV_PRIMARY_RHO,
    kappa_multiplier = DISSIMILARITY_IV_PRIMARY_KAPPA_MULTIPLIER,
    weight_draw_id = NA_integer_,
    instrument_family = DISSIMILARITY_IV_INSTRUMENT_FAMILY,
    aggregation_spec = DISSIMILARITY_IV_AGGREGATION_SPEC,
    baseline_frame_proxy = DISSIMILARITY_IV_BASELINE_FRAME_PROXY,
    annual_update_spec = DISSIMILARITY_IV_ANNUAL_UPDATE_SPEC,
    geographic_allocation_spec = DISSIMILARITY_IV_GEOGRAPHIC_ALLOCATION_SPEC,
    cluster_size_rule = DISSIMILARITY_IV_CLUSTER_SIZE_RULE
  ) %>%
  select(
    aewr_region_id,
    target_cluster,
    aewr_iv_cluster_id,
    source_year,
    policy_year,
    iv_k,
    donor_cluster_count,
    instrument_family,
    aggregation_spec,
    donor_wage_spec,
    donor_wage_geography,
    baseline_frame_proxy,
    annual_update_spec,
    geographic_allocation_spec,
    weight_spec,
    baseline_weight_spec,
    weight_component,
    weight_specification,
    moment_spec,
    wage_target_used,
    rho,
    kappa_multiplier,
    is_primary,
    cluster_size_rule,
    weight_draw_id,
    instrument_spec_label,
    z_dissimilarity_nominal,
    z_dissimilarity_real,
    instrument_available,
    selected_donor_clusters,
    selected_donor_cluster_ids,
    minimum_fixed_donor_cluster_units,
    fixed_donor_units,
    observed_donor_clusters,
    observed_donor_units,
    donor_unit_coverage,
    observed_donor_counties,
    observed_oews_areas,
    candidate_donor_clusters,
    candidate_donor_units,
    candidate_donor_counties,
    target_overlap_counties_excluded,
    target_overlap_areas_excluded,
    candidate_calibrated_weight_mass,
    nonoverlap_calibrated_weight_mass,
    eligible_calibrated_weight_mass,
    qcew_employment_coverage_weight,
    oews_wage_proxy_coverage_weight,
    active_moment_count,
    inactive_moment_count,
    instrument_weight_ess,
    maximum_instrument_county_weight,
    nearest_selected_donor_distance,
    farthest_selected_donor_distance,
    source_annual_update_methods,
    source_year_ppi_2012
  ) %>%
  arrange(
    aewr_region_id,
    target_cluster,
    policy_year,
    desc(is_primary)
  )

if (
  nrow(instrument_grid) !=
    85L * length(source_years) * nrow(publication_specs) ||
    any(instrument_grid$source_year != instrument_grid$policy_year - 1L) ||
    anyDuplicated(instrument_grid[c(
      "aewr_region_id",
      "target_cluster",
      "source_year",
      "instrument_spec_label"
    )]) > 0L
) {
  stop("The Panel-IV instrument grid violates its declared keys or t-1 timing.",
    call. = FALSE)
}

write_parquet(
  county_donor_frame,
  path_int("panel_iv_county_donor_frame.parquet")
)
write_parquet(
  instrument_grid,
  path_int("panel_iv_instrument_cluster_year.parquet")
)
