# Purpose: Attach full-gap wage instruments across cluster and donor-set counts.
# Inputs: county analysis panel, IV clusters, and long wage-target instruments.
# Output: data/processed/county_df_analysis_year_iv.parquet.
# Run after: 09_construct_donor_instruments.R.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
library(arrow)
library(tidyverse)

county_df <- read_parquet(
  path_processed("county_df_analysis_year.parquet")
)
iv_clusters <- read_parquet(path_int("iv_cz_aewr_clusters.parquet"))
iv_oews_long <- read_parquet(path_int("iv_oews_entropy_long.parquet"))
iv_design_specs <- iv_oews_long %>%
  distinct(iv_k, donor_cluster_count) %>%
  arrange(iv_k, donor_cluster_count)

iv_cluster_assignments <- iv_clusters %>%
  transmute(
    cz_aewr_region_fe,
    aewr_region_num,
    iv_k_label = paste0("k", iv_k),
    iv_cluster
  ) %>%
  pivot_wider(
    names_from = iv_k_label,
    values_from = iv_cluster,
    names_glue = "iv_cluster_{iv_k_label}"
  )

iv_oews_grid <- iv_oews_long %>%
  select(
    cz_aewr_region_fe,
    aewr_region_num,
    year,
    iv_k,
    donor_cluster_count,
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
    names_from = c(iv_k, donor_cluster_count, gap_closure_label),
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
    names_glue = paste0(
      "{.value}_k{iv_k}_d{donor_cluster_count}_",
      "{gap_closure_label}"
    )
  )

# Preserve the original column names for the k = 2, farthest-one benchmark.
iv_oews_benchmark <- iv_oews_long %>%
  filter(
    iv_k == min(iv_design_specs$iv_k),
    donor_cluster_count == 1
  ) %>%
  select(
    cz_aewr_region_fe,
    aewr_region_num,
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
    by = c("cz_aewr_region_fe", "aewr_region_num")
  ) %>%
  left_join(
    iv_oews_grid,
    by = c("cz_aewr_region_fe", "aewr_region_num", "year")
  ) %>%
  left_join(
    iv_oews_benchmark,
    by = c("cz_aewr_region_fe", "aewr_region_num", "year")
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
  instrument_name <- paste0(
    "z_oews_entropy_agwage_l1_",
    "k",
    iv_k,
    "_d",
    donor_cluster_count,
    "_g100"
  )
  cat(
    "Nonmissing",
    instrument_name,
    "rows:",
    sum(!is.na(county_df_iv[[instrument_name]])),
    "\n"
  )
}
