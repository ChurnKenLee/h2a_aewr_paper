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
    instrument_spec_label %in% c(
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
    instrument_spec_label ==
      DISSIMILARITY_IV_WAGE_ONLY_INSTRUMENT_LABEL
  ) %>%
  transmute(
    across(all_of(instrument_keys)),
    z_wage_only_real = z_dissimilarity_real,
    wage_only_instrument_available = instrument_available
  )

wage_seasonal_instrument <- instrument_long %>%
  filter(
    instrument_spec_label ==
      DISSIMILARITY_IV_PRIMARY_INSTRUMENT_LABEL
  ) %>%
  transmute(
    across(all_of(instrument_keys)),
    z_wage_seasonal_real = z_dissimilarity_real,
    wage_seasonal_instrument_available = instrument_available
  )

instrument_pair <- wage_only_instrument %>%
  inner_join(
    wage_seasonal_instrument,
    by = instrument_keys,
    relationship = "one-to-one"
  )

if (
  nrow(instrument_pair) !=
    85L *
      (
        DISSIMILARITY_IV_POLICY_END_YEAR -
          DISSIMILARITY_IV_POLICY_START_YEAR +
          1L
      ) ||
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
