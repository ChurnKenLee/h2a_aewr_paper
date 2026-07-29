# Purpose: Construct the wage-only and wage-plus-seasonal panel-IV instruments.
# Inputs: fixed clusters/donor map, Census frame analog, recovered entropy
# centers, OEWS geography, and PPI.
# Outputs:
#   data/intermediate/panel_iv_area_frame.parquet
#   data/intermediate/panel_iv_instrument_cluster_year.parquet
# Run after: 02_cluster_target_units.R, 03_build_fls_frame.py,
# 04_recover_fls_geography.py, and the shared county panel.
#
# The two specifications share the same Census frame prior and soft wage
# target.  The preferred specification adds quarterly FLS worker shares.
# First-stage estimates and outcomes cannot select either specification.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "helpers.R"))
library(arrow)
library(dplyr)
library(stringr)
library(tidyr)

first_nonmissing <- function(value) {
  value <- value[!is.na(value) & value != ""]
  if (length(value) == 0L) {
    return(NA_character_)
  }
  value[[1]]
}

county_panel <- read_parquet(
  path_processed("county_year_panel.parquet")
) %>%
  mutate(
    panel_iv_target_unit_id = make_panel_iv_target_unit_id(
      cz_id,
      aewr_region_id
    )
  )
iv_clusters <- read_parquet(path_int("panel_iv_target_clusters.parquet"))
iv_donor_clusters <- read_parquet(path_int("panel_iv_donor_clusters.parquet"))
oews_source <- read_parquet(path_int("oews.parquet"))
frame_employment <- read_parquet(path_int(
  "panel_iv_fls_frame.parquet"
))
realized_weight_summary <- read_parquet(path_int(
  "panel_iv_fls_geography_weight_summary.parquet"
))
realized_county_area_prior <- read_parquet(path_int(
  "panel_iv_fls_geography_county_area_prior.parquet"
))
ppi <- read_parquet(path_int("ppi_2012.parquet"))


source_years <- seq.int(
  DISSIMILARITY_IV_POLICY_START_YEAR - 1L,
  DISSIMILARITY_IV_POLICY_END_YEAR - 1L
)

frame_employment <- frame_employment %>%
  filter(
    source_year %in% source_years,
    weight_spec == DISSIMILARITY_IV_FRAME_WEIGHT_SPEC,
    annual_update_spec == DISSIMILARITY_IV_ANNUAL_UPDATE_SPEC,
    is.na(weight_draw_id)
  )

publication_specs <- tibble::tribble(
  ~weight_specification, ~moment_spec, ~is_primary,
  ~instrument_spec_label,
  DISSIMILARITY_IV_WAGE_ONLY_WEIGHT_SPECIFICATION,
  DISSIMILARITY_IV_WAGE_ONLY_MOMENT_SPEC,
  FALSE,
  DISSIMILARITY_IV_WAGE_ONLY_INSTRUMENT_LABEL,
  DISSIMILARITY_IV_PRIMARY_WEIGHT_SPECIFICATION,
  DISSIMILARITY_IV_PRIMARY_MOMENT_SPEC,
  TRUE,
  DISSIMILARITY_IV_PRIMARY_INSTRUMENT_LABEL
)

publication_realized_weights <- realized_weight_summary %>%
  filter(
    source_year %in% source_years,
    weight_spec == DISSIMILARITY_IV_PRIMARY_WEIGHT_SPEC,
    wage_target_used,
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
    aewr_region_id,
    source_year = as.integer(source_year),
    oews_area_code,
    weight_specification = specification,
    moment_spec,
    is_primary,
    instrument_spec_label,
    calibrated_center_weight
  )

realized_county_area_weights <- realized_county_area_prior %>%
  filter(
    source_year %in% source_years,
    baseline_weight_spec == DISSIMILARITY_IV_FRAME_WEIGHT_SPEC
  ) %>%
  inner_join(
    publication_realized_weights,
    by = c(
      "aewr_region_id",
      "source_year",
      "oews_area_code"
    ),
    relationship = "many-to-many"
  ) %>%
  transmute(
    county_fips,
    aewr_region_id,
    source_year = as.integer(source_year),
    oews_area_code,
    weight_specification,
    moment_spec,
    is_primary,
    instrument_spec_label,
    recovered_baseline_county_area_mass = baseline_county_area_mass,
    baseline_county_conditional_within_area,
    entropy_county_area_weight = calibrated_center_weight *
      baseline_county_conditional_within_area
  )

primary_clusters <- iv_clusters %>%
  filter(iv_k == DISSIMILARITY_IV_PRIMARY_K) %>%
  transmute(
    panel_iv_target_unit_id,
    aewr_region_id,
    iv_k,
    target_cluster = iv_cluster
  ) %>%
  distinct()


primary_donor_map <- iv_donor_clusters %>%
  filter(
    iv_k == DISSIMILARITY_IV_PRIMARY_K,
    donor_cluster_count == DISSIMILARITY_IV_PRIMARY_DONOR_COUNT
  ) %>%
  select(
    aewr_region_id,
    iv_k,
    target_cluster,
    donor_cluster,
    donor_rank,
    donor_cluster_distance,
    donor_cluster_count
  ) %>%
  distinct()

county_units <- county_panel %>%
  distinct(
    county_fips,
    state_fips,
    state_abbrev,
    cz_id,
    aewr_region_id,
    panel_iv_target_unit_id
  ) %>%
  inner_join(
    primary_clusters,
    by = c("panel_iv_target_unit_id", "aewr_region_id"),
    relationship = "many-to-one"
  )


# OEWS employment is used only to combine covered occupations within an OEWS
# area wage. It never determines mass across counties or areas.
oews_area_wages <- oews_source %>%
  filter(
    year %in% source_years,
    occ_code %in% DISSIMILARITY_IV_BIG_SIX_OCC_CODES
  ) %>%
  transmute(
    oews_area_code = oews_area_code(area),
    oews_area_name_data = area_name,
    source_year = as.integer(year),
    oews_tot_emp = suppressWarnings(as.numeric(tot_emp)),
    oews_mean_hourly_wage = suppressWarnings(as.numeric(h_mean))
  ) %>%
  mutate(
    usable_wage = is.finite(oews_tot_emp) &
      oews_tot_emp > 0 &
      is.finite(oews_mean_hourly_wage) &
      oews_mean_hourly_wage > 0,
    oews_occupation_wage_weight = if_else(
      usable_wage,
      oews_tot_emp,
      0
    ),
    oews_hourly_wage_bill = if_else(
      usable_wage,
      oews_tot_emp * oews_mean_hourly_wage,
      0
    )
  ) %>%
  group_by(oews_area_code, source_year) %>%
  summarise(
    oews_area_name_data = first_nonmissing(oews_area_name_data),
    oews_area_wage_covered_employment = sum(
      oews_occupation_wage_weight,
      na.rm = TRUE
    ),
    oews_area_hourly_wage_bill = sum(
      oews_hourly_wage_bill,
      na.rm = TRUE
    ),
    oews_occupation_count = sum(usable_wage),
    .groups = "drop"
  ) %>%
  mutate(
    oews_area_mean_hourly_wage = if_else(
      oews_area_wage_covered_employment > 0,
      oews_area_hourly_wage_bill /
        oews_area_wage_covered_employment,
      NA_real_
    )
  )

# Reuse the audited county-to-area allocation from the recovery step.  This
# carries the explicit nearest-vintage fallback used for the 2010 Orange
# County, New York OEWS definition gap.
county_area_allocation <- realized_county_area_prior %>%
  filter(
    source_year %in% source_years,
    baseline_weight_spec == DISSIMILARITY_IV_FRAME_WEIGHT_SPEC
  ) %>%
  transmute(
    county_fips,
    aewr_region_id,
    source_year,
    oews_area_code,
    oews_area_name_crosswalk = oews_area_name,
    oews_area_mapped_townships,
    county_mapped_townships,
    county_oews_area_share,
    oews_area_allocated_frame_employment = baseline_county_area_mass,
    mapping_source_year,
    mapping_vintage_fallback
  ) %>%
  left_join(
    frame_employment %>%
      select(
        county_fips,
        source_year,
        frame_employment_mass,
        state_rake_factor,
        annual_update_source,
        annual_growth_source,
        quality_flags
      ),
    by = c("county_fips", "source_year"),
    relationship = "many-to-one"
  )

if (
  anyDuplicated(
    county_area_allocation[
      c("county_fips", "source_year", "oews_area_code")
    ]
  ) > 0L
) {
  stop("County-area allocation keys must be unique.", call. = FALSE)
}

if (
  any(
    !is.finite(county_area_allocation$oews_area_allocated_frame_employment) |
      county_area_allocation$oews_area_allocated_frame_employment < 0
  )
) {
  stop("County-area frame mass must be finite and nonnegative.", call. = FALSE)
}

county_area_wages <- county_units %>%
  inner_join(
    county_area_allocation,
    by = c("county_fips", "aewr_region_id"),
    relationship = "one-to-many"
  ) %>%
  left_join(
    oews_area_wages,
    by = c("oews_area_code", "source_year")
  ) %>%
  mutate(
    oews_area_name = coalesce(
      oews_area_name_data,
      oews_area_name_crosswalk
    )
  )

target_cluster_areas <- county_area_wages %>%
  transmute(
    aewr_region_id,
    source_year,
    target_cluster,
    oews_area_code
  ) %>%
  distinct()

# Join every donor county-area record to each target for which its cluster is
# selected. Donor membership is fixed by baseline features; only wage
# availability can vary over time.
donor_area_candidates <- county_area_wages %>%
  rename(donor_cluster = target_cluster) %>%
  inner_join(
    primary_donor_map,
    by = c(
      "aewr_region_id",
      "iv_k",
      "donor_cluster"
    ),
    relationship = "many-to-many"
  )

candidate_support <- donor_area_candidates %>%
  group_by(aewr_region_id, target_cluster, source_year) %>%
  summarise(
    candidate_donor_clusters = n_distinct(donor_cluster),
    candidate_donor_units = n_distinct(panel_iv_target_unit_id),
    candidate_donor_counties = n_distinct(county_fips),
    candidate_donor_areas = n_distinct(oews_area_code),
    candidate_wage_areas = n_distinct(
      oews_area_code[is.finite(oews_area_mean_hourly_wage)]
    ),
    candidate_frame_areas = n_distinct(
      oews_area_code[
        is.finite(oews_area_allocated_frame_employment) &
          oews_area_allocated_frame_employment > 0
      ]
    ),
    .groups = "drop"
  )

# If an OEWS area touches the target cluster, exclude that entire area from
# the corresponding donor set. Retain the county/unit mappings as support
# diagnostics, but never attach a county's full frame mass to each area.
eligible_donor_area_mappings <- donor_area_candidates %>%
  anti_join(
    target_cluster_areas,
    by = c(
      "aewr_region_id",
      "source_year",
      "target_cluster",
      "oews_area_code"
    )
  )

excluded_overlap_support <- donor_area_candidates %>%
  semi_join(
    target_cluster_areas,
    by = c(
      "aewr_region_id",
      "source_year",
      "target_cluster",
      "oews_area_code"
    )
  ) %>%
  group_by(aewr_region_id, target_cluster, source_year) %>%
  summarise(
    oews_overlap_areas_excluded = n_distinct(oews_area_code),
    .groups = "drop"
  )

eligible_observed_mappings <- eligible_donor_area_mappings %>%
  filter(
    is.finite(oews_area_mean_hourly_wage),
    oews_area_mean_hourly_wage > 0,
    is.finite(oews_area_allocated_frame_employment),
    oews_area_allocated_frame_employment > 0
  )

eligible_support <- eligible_observed_mappings %>%
  group_by(aewr_region_id, target_cluster, source_year) %>%
  summarise(
    eligible_donor_clusters = n_distinct(donor_cluster),
    eligible_donor_units = n_distinct(panel_iv_target_unit_id),
    eligible_donor_counties = n_distinct(county_fips),
    eligible_donor_areas = n_distinct(oews_area_code),
    .groups = "drop"
  )

# An OEWS area can map to many counties, donor units, and even both selected
# donor clusters. Collapse those mappings first. The entropy-center mass is
# expanded to counties with the fixed within-area frame shares before donor
# selection, then re-aggregated here.
area_support <- eligible_donor_area_mappings %>%
  group_by(
    aewr_region_id,
    target_cluster,
    source_year,
    iv_k,
    oews_area_code
  ) %>%
  summarise(
    oews_area_name = first_nonmissing(oews_area_name),
    oews_area_mean_hourly_wage = first(
      oews_area_mean_hourly_wage
    ),
    area_selected_frame_employment = sum(
      oews_area_allocated_frame_employment,
      na.rm = TRUE
    ),
    selected_county_area_share_sum = sum(county_oews_area_share),
    selected_mapped_townships = sum(oews_area_mapped_townships),
    oews_area_wage_covered_employment = first(
      oews_area_wage_covered_employment
    ),
    oews_occupation_count = first(oews_occupation_count),
    minimum_source_state_rake_factor = min(state_rake_factor),
    maximum_source_state_rake_factor = max(state_rake_factor),
    source_annual_update_methods = paste(
      sort(unique(annual_update_source)),
      collapse = "|"
    ),
    source_annual_growth_methods = paste(
      sort(unique(annual_growth_source)),
      collapse = "|"
    ),
    mapped_donor_clusters = n_distinct(donor_cluster),
    mapped_donor_cluster_ids = paste(
      paste0("c", sort(unique(donor_cluster))),
      collapse = "|"
    ),
    mapped_donor_units = n_distinct(panel_iv_target_unit_id),
    mapped_donor_counties = n_distinct(county_fips),
    mapping_rows = n(),
    .groups = "drop"
  ) %>%
  mutate(
    frame_employment_observed = is.finite(area_selected_frame_employment) &
      area_selected_frame_employment > 0,
    oews_wage_observed = is.finite(oews_area_mean_hourly_wage) &
      oews_area_mean_hourly_wage > 0
  ) %>%
  group_by(aewr_region_id, target_cluster, source_year) %>%
  mutate(
    baseline_frame_employment = sum(
      if_else(
        frame_employment_observed,
        area_selected_frame_employment,
        0
      ),
      na.rm = TRUE
    ),
    instrument_frame_employment = sum(
      if_else(
        oews_wage_observed,
        area_selected_frame_employment,
        0
      ),
      na.rm = TRUE
    ),
    missing_wage_frame_share = if_else(
      baseline_frame_employment > 0,
      1 -
        instrument_frame_employment /
          baseline_frame_employment,
      NA_real_
    )
  ) %>%
  ungroup()

selected_entropy_area_weights <- eligible_donor_area_mappings %>%
  select(
    county_fips,
    aewr_region_id,
    target_cluster,
    source_year,
    iv_k,
    oews_area_code
  ) %>%
  inner_join(
    realized_county_area_weights,
    by = c(
      "county_fips",
      "aewr_region_id",
      "source_year",
      "oews_area_code"
    ),
    relationship = "many-to-many"
  ) %>%
  group_by(
    aewr_region_id,
    target_cluster,
    source_year,
    iv_k,
    oews_area_code,
    weight_specification,
    moment_spec,
    is_primary,
    instrument_spec_label
  ) %>%
  summarise(
    area_selected_entropy_weight = sum(
      entropy_county_area_weight,
      na.rm = TRUE
    ),
    .groups = "drop"
  )

make_area_weight_spec <- function(
  data,
  selected_weight_column,
  weight_spec,
  baseline_weight_spec,
  weight_component,
  weight_specification,
  moment_spec,
  wage_target_used,
  rho,
  kappa_multiplier,
  is_primary,
  instrument_spec_label
) {
  data %>%
    mutate(
      selected_weight_mass = .data[[selected_weight_column]],
      weight_mass_observed = is.finite(selected_weight_mass) &
        selected_weight_mass > 0,
      weight_wage_observed = weight_mass_observed & oews_wage_observed
    ) %>%
    group_by(aewr_region_id, target_cluster, source_year) %>%
    mutate(
      eligible_weight_mass = sum(
        if_else(weight_mass_observed, selected_weight_mass, 0),
        na.rm = TRUE
      ),
      observed_wage_weight_mass = sum(
        if_else(weight_wage_observed, selected_weight_mass, 0),
        na.rm = TRUE
      ),
      missing_wage_weight_share = if_else(
        eligible_weight_mass > 0,
        1 - observed_wage_weight_mass / eligible_weight_mass,
        NA_real_
      ),
      area_weight = if_else(
        weight_mass_observed & eligible_weight_mass > 0,
        selected_weight_mass / eligible_weight_mass,
        NA_real_
      ),
      instrument_weight = if_else(
        weight_wage_observed & observed_wage_weight_mass > 0,
        selected_weight_mass / observed_wage_weight_mass,
        NA_real_
      )
    ) %>%
    ungroup() %>%
    mutate(
      weight_spec = .env$weight_spec,
      baseline_weight_spec = .env$baseline_weight_spec,
      weight_component = .env$weight_component,
      weight_specification = .env$weight_specification,
      moment_spec = .env$moment_spec,
      wage_target_used = .env$wage_target_used,
      rho = .env$rho,
      kappa_multiplier = .env$kappa_multiplier,
      is_primary = .env$is_primary,
      weight_draw_id = NA_integer_,
      instrument_spec_label = .env$instrument_spec_label
    )
}

entropy_area_frames <- lapply(
  seq_len(nrow(publication_specs)),
  function(index) {
    selected_spec <- publication_specs[index, ]
    spec_weights <- selected_entropy_area_weights %>%
      filter(
        weight_specification ==
          selected_spec$weight_specification[[1]],
        moment_spec == selected_spec$moment_spec[[1]],
        is_primary == selected_spec$is_primary[[1]],
        instrument_spec_label ==
          selected_spec$instrument_spec_label[[1]]
      ) %>%
      select(
        aewr_region_id,
        target_cluster,
        source_year,
        iv_k,
        oews_area_code,
        area_selected_entropy_weight
      )

    make_area_weight_spec(
      area_support %>%
        inner_join(
          spec_weights,
          by = c(
            "aewr_region_id",
            "target_cluster",
            "source_year",
            "iv_k",
            "oews_area_code"
          ),
          relationship = "one-to-one"
        ),
      selected_weight_column = "area_selected_entropy_weight",
      weight_spec = DISSIMILARITY_IV_PRIMARY_WEIGHT_SPEC,
      baseline_weight_spec = DISSIMILARITY_IV_FRAME_WEIGHT_SPEC,
      weight_component = DISSIMILARITY_IV_PRIMARY_WEIGHT_COMPONENT,
      weight_specification =
        selected_spec$weight_specification[[1]],
      moment_spec = selected_spec$moment_spec[[1]],
      wage_target_used = TRUE,
      rho = DISSIMILARITY_IV_PRIMARY_RHO,
      kappa_multiplier = DISSIMILARITY_IV_PRIMARY_KAPPA_MULTIPLIER,
      is_primary = selected_spec$is_primary[[1]],
      instrument_spec_label =
        selected_spec$instrument_spec_label[[1]]
    )
  }
)

area_frame <- bind_rows(entropy_area_frames) %>%
  mutate(
    policy_year = source_year + 1L,
    aewr_iv_cluster_id = make_dissimilarity_cluster_id(
      aewr_region_id,
      target_cluster
    ),
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
    oews_area_code
  )

observed_instruments <- area_frame %>%
  filter(weight_wage_observed) %>%
  group_by(
    aewr_region_id,
    target_cluster,
    source_year,
    instrument_spec_label
  ) %>%
  summarise(
    z_dissimilarity_nominal_raw = sum(
      instrument_weight *
        oews_area_mean_hourly_wage
    ),
    baseline_frame_employment = first(
      baseline_frame_employment
    ),
    instrument_frame_employment = first(
      instrument_frame_employment
    ),
    missing_wage_frame_share = first(
      missing_wage_frame_share
    ),
    eligible_weight_mass = first(eligible_weight_mass),
    observed_wage_weight_mass = first(
      observed_wage_weight_mass
    ),
    missing_wage_weight_share = first(
      missing_wage_weight_share
    ),
    observed_oews_areas = n_distinct(oews_area_code),
    instrument_weight_ess = 1 /
      sum(
        instrument_weight^2
      ),
    maximum_instrument_area_weight = max(
      instrument_weight
    ),
    weighted_oews_occupation_count = sum(
      instrument_weight * oews_occupation_count
    ),
    multi_county_oews_areas = sum(mapped_donor_counties > 1L),
    multi_unit_oews_areas = sum(mapped_donor_units > 1L),
    multi_cluster_oews_areas = sum(mapped_donor_clusters > 1L),
    .groups = "drop"
  )

write_parquet(
  area_frame,
  path_int("panel_iv_area_frame.parquet")
)

fixed_cluster_counts <- primary_clusters %>%
  count(
    aewr_region_id,
    donor_cluster = target_cluster,
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

instrument_weight_specs <- area_frame %>%
  distinct(
    weight_spec,
    baseline_weight_spec,
    weight_component,
    weight_specification,
    moment_spec,
    wage_target_used,
    rho,
    kappa_multiplier,
    is_primary,
    weight_draw_id,
    instrument_spec_label
  )

instrument_grid <- target_geometry %>%
  crossing(source_year = source_years) %>%
  crossing(instrument_weight_specs) %>%
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
  left_join(
    eligible_support,
    by = c("aewr_region_id", "target_cluster", "source_year")
  ) %>%
  left_join(
    excluded_overlap_support,
    by = c("aewr_region_id", "target_cluster", "source_year")
  ) %>%
  mutate(
    observed_donor_clusters = eligible_donor_clusters,
    observed_donor_units = eligible_donor_units,
    observed_donor_counties = eligible_donor_counties,
    across(
      c(
        observed_donor_clusters,
        observed_donor_units,
        observed_donor_counties,
        observed_oews_areas,
        multi_county_oews_areas,
        multi_unit_oews_areas,
        multi_cluster_oews_areas,
        candidate_donor_clusters,
        candidate_donor_units,
        candidate_donor_counties,
        candidate_donor_areas,
        candidate_wage_areas,
        candidate_frame_areas,
        eligible_donor_clusters,
        eligible_donor_units,
        eligible_donor_counties,
        eligible_donor_areas,
        oews_overlap_areas_excluded
      ),
      ~ replace_na(as.integer(.x), 0L)
    ),
    donor_unit_coverage = if_else(
      fixed_donor_units > 0,
      observed_donor_units / fixed_donor_units,
      NA_real_
    ),
    instrument_available = observed_donor_clusters ==
      DISSIMILARITY_IV_MIN_OBSERVED_DONOR_CLUSTERS &
      is.finite(z_dissimilarity_nominal_raw),
    z_dissimilarity_nominal = if_else(
      instrument_available,
      z_dissimilarity_nominal_raw,
      NA_real_
    ),
    policy_year = source_year + 1L
  ) %>%
  left_join(
    ppi %>%
      transmute(
        policy_year = as.integer(year),
        policy_year_ppi_2012 = as.numeric(ppi_2012)
      ),
    by = "policy_year",
    relationship = "many-to-one"
  ) %>%
  mutate(
    z_dissimilarity_real = if_else(
      instrument_available &
        is.finite(policy_year_ppi_2012) &
        policy_year_ppi_2012 > 0,
      z_dissimilarity_nominal / policy_year_ppi_2012,
      NA_real_
    ),
    aewr_iv_cluster_id = make_dissimilarity_cluster_id(
      aewr_region_id,
      target_cluster
    ),
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
    baseline_frame_employment,
    instrument_frame_employment,
    missing_wage_frame_share,
    eligible_weight_mass,
    observed_wage_weight_mass,
    missing_wage_weight_share,
    instrument_weight_ess,
    maximum_instrument_area_weight,
    weighted_oews_occupation_count,
    multi_county_oews_areas,
    multi_unit_oews_areas,
    multi_cluster_oews_areas,
    candidate_donor_clusters,
    candidate_donor_units,
    candidate_donor_counties,
    candidate_donor_areas,
    candidate_wage_areas,
    candidate_frame_areas,
    eligible_donor_clusters,
    eligible_donor_units,
    eligible_donor_counties,
    eligible_donor_areas,
    oews_overlap_areas_excluded,
    nearest_selected_donor_distance,
    farthest_selected_donor_distance,
    policy_year_ppi_2012
  ) %>%
  arrange(
    aewr_region_id,
    target_cluster,
    policy_year,
    desc(is_primary)
  )


write_parquet(
  instrument_grid,
  path_int("panel_iv_instrument_cluster_year.parquet")
)
