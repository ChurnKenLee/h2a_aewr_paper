# Purpose: Aggregate the shared county panel to panel-IV cluster-years.
# Output: data/processed/panel_iv_cluster_year.parquet.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "helpers.R"))
library(arrow)
library(dplyr)

sum_if_observed <- function(value) {
  observed <- value[is.finite(value)]
  if (length(observed) == 0L) {
    return(NA_real_)
  }
  sum(observed)
}

county_panel <- read_parquet(
  path_processed("county_year_panel.parquet")
) |>
  mutate(
    panel_iv_target_unit_id = make_panel_iv_target_unit_id(
      cz_id,
      aewr_region_id
    )
  )

cluster_assignments <- read_parquet(
  path_int("panel_iv_target_clusters.parquet")
) |>
  filter(iv_k == DISSIMILARITY_IV_PRIMARY_K) |>
  transmute(
    panel_iv_target_unit_id,
    aewr_region_id,
    target_cluster = iv_cluster
  ) |>
  distinct()

instrument_long <- read_parquet(
  path_int("panel_iv_instrument_cluster_year.parquet")
)

primary_instrument <- instrument_long |>
  filter(
    weight_spec == DISSIMILARITY_IV_PRIMARY_WEIGHT_SPEC,
    weight_specification == DISSIMILARITY_IV_PRIMARY_WEIGHT_SPECIFICATION,
    weight_component == DISSIMILARITY_IV_PRIMARY_WEIGHT_COMPONENT,
    is_primary,
    is.na(weight_draw_id)
  )

frame_benchmark <- instrument_long |>
  filter(
    weight_spec == DISSIMILARITY_IV_FRAME_WEIGHT_SPEC,
    weight_specification == "census_frame",
    !is_primary,
    is.na(weight_draw_id)
  ) |>
  transmute(
    aewr_region_id,
    target_cluster,
    policy_year,
    z_dissimilarity_census_frame_nominal = z_dissimilarity_nominal,
    z_dissimilarity_census_frame_real = z_dissimilarity_real,
    census_frame_instrument_available = instrument_available
  )

primary_instrument <- primary_instrument |>
  left_join(
    frame_benchmark,
    by = c(
      "aewr_region_id",
      "target_cluster",
      "policy_year"
    ),
    relationship = "one-to-one"
  )

county_iv_sample <- county_panel |>
  filter(
    year >= DISSIMILARITY_IV_POLICY_START_YEAR,
    year <= DISSIMILARITY_IV_POLICY_END_YEAR,
    any_cropland_2007
  ) |>
  inner_join(
    cluster_assignments,
    by = c("panel_iv_target_unit_id", "aewr_region_id"),
    relationship = "many-to-one"
  ) |>
  mutate(
    baseline_farm_employment_weight = if_else(
      is.finite(emp_farm_2011) & emp_farm_2011 > 0,
      emp_farm_2011,
      0
    )
  )

cluster_year_panel <- county_iv_sample |>
  group_by(aewr_region_id, target_cluster, year) |>
  summarise(
    counties = n_distinct(county_fips),
    target_units = n_distinct(panel_iv_target_unit_id),
    states = n_distinct(state_fips),
    baseline_farm_employment = sum(
      baseline_farm_employment_weight,
      na.rm = TRUE
    ),
    h2a_certified_workers_start_year = sum_if_observed(
      nbr_workers_certified_start_year
    ),
    h2a_cert_share_farm_workers_2011 = if_else(
      baseline_farm_employment > 0,
      h2a_certified_workers_start_year /
        baseline_farm_employment,
      NA_real_
    ),
    aewr_ppi = first(aewr_ppi[is.finite(aewr_ppi)]),
    ln_pop_census_l1 = positive_weighted_mean(
      ln_pop_census_l1,
      baseline_farm_employment_weight
    ),
    farm_emp_share_l1 = positive_weighted_mean(
      farm_emp_share_l1,
      baseline_farm_employment_weight
    ),
    emp_pop_ratio_l1 = positive_weighted_mean(
      emp_pop_ratio_l1,
      baseline_farm_employment_weight
    ),
    wage_p25_l1 = positive_weighted_mean(
      wage_p25_l1,
      baseline_farm_employment_weight
    ),
    no_border_cluster = all(
      !is.na(border_cz) & !border_cz
    ),
    .groups = "drop"
  ) |>
  mutate(
    policy_year = as.integer(year),
    aewr_iv_cluster_id = make_dissimilarity_cluster_id(
      aewr_region_id,
      target_cluster
    )
  ) |>
  left_join(
    primary_instrument,
    by = c(
      "aewr_region_id",
      "target_cluster",
      "policy_year"
    ),
    relationship = "one-to-one",
    suffix = c("", "_instrument")
  ) |>
  select(-aewr_iv_cluster_id_instrument)

if (
  nrow(cluster_year_panel) == 0L ||
    anyDuplicated(
      cluster_year_panel[
        c("aewr_region_id", "target_cluster", "policy_year")
      ]
    ) >
      0L
) {
  stop(
    "panel_iv_cluster_year must have unique region-cluster-year keys.",
    call. = FALSE
  )
}

write_parquet(
  cluster_year_panel,
  path_processed("panel_iv_cluster_year.parquet")
)
