# Purpose: Attach full-gap wage instruments across cluster and donor-set counts.
# Inputs: county analysis panel, IV clusters, and long wage-target instruments.
# Output: data/processed/county_df_analysis_year_iv.parquet.
# Run after: 09_construct_donor_instruments.R.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("c00_shared", "iv_preferred_design.R"))
library(arrow)
library(tidyverse)

county_df <- read_parquet(
  path_processed("county_df_analysis_year.parquet")
)
iv_clusters <- read_parquet(path_int("iv_cz_aewr_clusters.parquet"))
iv_oews_long <- read_parquet(path_int("iv_oews_entropy_long.parquet"))
assert_geo_columns(
  county_df,
  c("county_fips", "state_fips", "aewr_region_id", "cz_id")
)
assert_geo_columns(iv_clusters, "aewr_region_id")
assert_geo_columns(iv_oews_long, "aewr_region_id")

make_instrument_spec_label <- function(
  iv_k,
  donor_cluster_count,
  gap_closure_label,
  weight_spec_label
) {
  base_label <- paste0(
    "k",
    iv_k,
    "_d",
    donor_cluster_count,
    "_",
    gap_closure_label
  )
  if_else(
    weight_spec_label == "wage_only_exact",
    base_label,
    paste0(base_label, "_", weight_spec_label)
  )
}

iv_design_specs <- iv_oews_long %>%
  distinct(
    iv_k,
    donor_cluster_count,
    gap_closure_label,
    prior_spec,
    moment_spec,
    calibration_mode,
    duration_analogue,
    soft_penalty,
    weight_spec_label
  ) %>%
  mutate(
    instrument_spec_label = make_instrument_spec_label(
      iv_k,
      donor_cluster_count,
      gap_closure_label,
      weight_spec_label
    )
  ) %>%
  arrange(weight_spec_label, iv_k, donor_cluster_count)

iv_cluster_assignments <- iv_clusters %>%
  transmute(
    cz_aewr_region_fe,
    aewr_region_id,
    iv_k_label = paste0("k", iv_k),
    iv_cluster
  ) %>%
  pivot_wider(
    names_from = iv_k_label,
    values_from = iv_cluster,
    names_glue = "iv_cluster_{iv_k_label}"
  )

iv_oews_grid <- iv_oews_long %>%
  mutate(
    instrument_spec_label = make_instrument_spec_label(
      iv_k,
      donor_cluster_count,
      gap_closure_label,
      weight_spec_label
    )
  ) %>%
  select(
    cz_aewr_region_fe,
    aewr_region_id,
    year,
    instrument_spec_label,
    z_oews_entropy_agwage_l1,
    oews_iv_donor_weight,
    oews_iv_donor_clusters,
    oews_iv_donor_areas,
    oews_iv_donor_units,
    oews_iv_candidate_donor_clusters,
    oews_iv_candidate_areas,
    oews_iv_candidate_units,
    oews_iv_overlap_areas_excluded,
    oews_iv_donor_cluster_distance,
    oews_iv_farthest_donor_cluster_distance
  ) %>%
  pivot_wider(
    names_from = instrument_spec_label,
    values_from = c(
      z_oews_entropy_agwage_l1,
      oews_iv_donor_weight,
      oews_iv_donor_clusters,
      oews_iv_donor_areas,
      oews_iv_donor_units,
      oews_iv_candidate_donor_clusters,
      oews_iv_candidate_areas,
      oews_iv_candidate_units,
      oews_iv_overlap_areas_excluded,
      oews_iv_donor_cluster_distance,
      oews_iv_farthest_donor_cluster_distance
    ),
    names_glue = "{.value}_{instrument_spec_label}"
  )

# Preserve the original column names for the k = 2, farthest-one benchmark.
iv_oews_benchmark <- iv_oews_long %>%
  filter(
    iv_k == min(iv_design_specs$iv_k),
    donor_cluster_count == 1,
    weight_spec_label == "wage_only_exact"
  ) %>%
  select(
    cz_aewr_region_fe,
    aewr_region_id,
    year,
    gap_closure_label,
    z_oews_entropy_agwage_l1,
    oews_iv_donor_weight,
    oews_iv_donor_clusters,
    oews_iv_donor_areas,
    oews_iv_donor_units,
    oews_iv_candidate_donor_clusters,
    oews_iv_candidate_areas,
    oews_iv_candidate_units,
    oews_iv_overlap_areas_excluded,
    oews_iv_donor_cluster_distance,
    oews_iv_farthest_donor_cluster_distance
  ) %>%
  pivot_wider(
    names_from = gap_closure_label,
    values_from = c(
      z_oews_entropy_agwage_l1,
      oews_iv_donor_weight,
      oews_iv_donor_clusters,
      oews_iv_donor_areas,
      oews_iv_donor_units,
      oews_iv_candidate_donor_clusters,
      oews_iv_candidate_areas,
      oews_iv_candidate_units,
      oews_iv_overlap_areas_excluded,
      oews_iv_donor_cluster_distance,
      oews_iv_farthest_donor_cluster_distance
    ),
    names_glue = "{.value}_{gap_closure_label}"
  )

county_df_iv <- county_df %>%
  mutate(
    dln_aewr = log(aewr) - log(aewr_l1),
    dln_aewr_ppi = log(aewr_ppi) - log(aewr_ppi_l1)
  ) %>%
  left_join(
    iv_cluster_assignments,
    by = c("cz_aewr_region_fe", "aewr_region_id")
  ) %>%
  left_join(
    iv_oews_grid,
    by = c("cz_aewr_region_fe", "aewr_region_id", "year")
  ) %>%
  left_join(
    iv_oews_benchmark,
    by = c("cz_aewr_region_fe", "aewr_region_id", "year")
  ) %>%
  mutate(
    aewr_iv_cluster_k5 = make_aewr_iv_cluster_id(
      aewr_region_id,
      iv_cluster_k5
    )
  )

assert_geo_columns(
  county_df_iv,
  c("state_fips", "county_fips", "cz_id", "aewr_region_id")
)
write_parquet(
  county_df_iv,
  path_processed("county_df_analysis_year_iv.parquet")
)

cat(
  "county_df_analysis_year_iv:",
  nrow(county_df_iv),
  "rows,",
  ncol(county_df_iv),
  "cols\n"
)

for (i in seq_len(nrow(iv_design_specs))) {
  iv_k <- iv_design_specs$iv_k[[i]]
  donor_cluster_count <- iv_design_specs$donor_cluster_count[[i]]
  weight_spec_label <- iv_design_specs$weight_spec_label[[i]]
  instrument_name <- paste0(
    "z_oews_entropy_agwage_l1_",
    iv_design_specs$instrument_spec_label[[i]]
  )
  cat(
    "Nonmissing",
    instrument_name,
    "rows:",
    sum(!is.na(county_df_iv[[instrument_name]])),
    "\n"
  )
}
