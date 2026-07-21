# Purpose: Construct lagged OEWS wage IVs from cumulative farthest-donor sets.
# Inputs: cluster assignments, donor pairs, county-area priors, and exact wage weights.
# Output: iv_oews_entropy_long.parquet.
# Run after: 05_wage_entropy_calibration.R and 08_cluster_cz_donor_units.R.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
library(arrow)
library(tidyverse)

iv_clusters <- read_parquet(path_int("iv_cz_aewr_clusters.parquet"))
iv_donor_clusters <- read_parquet(path_int("iv_donor_clusters.parquet"))
fls_county_oews_area_prior <- read_parquet(
  path_int("fls_county_oews_area_prior_weight.parquet")
) %>%
  mutate(year = as.integer(year))
fls_oews_area_calibrated <- read_parquet(
  path_int("fls_oews_area_weight_wage_calibrated.parquet")
) %>%
  mutate(year = as.integer(year)) %>%
  filter(near(gap_closure, 1))

county_oews_area_units <- fls_county_oews_area_prior %>%
  select(
    county_ansi = countyfips,
    year,
    aewr_region_num,
    cz_aewr_region_fe,
    oews_area_code,
    county_area_prior_weight
  ) %>%
  inner_join(
    iv_clusters,
    by = c("cz_aewr_region_fe", "aewr_region_num"),
    relationship = "many-to-many"
  )

target_cluster_oews_areas <- county_oews_area_units %>%
  transmute(
    aewr_region_num,
    year,
    iv_k,
    target_cluster = iv_cluster,
    oews_area_code
  ) %>%
  distinct()

oews_area_donor_candidates <- county_oews_area_units %>%
  inner_join(
    fls_oews_area_calibrated %>%
      filter(str_detect(calibration_status, "^calibrated")) %>%
      select(
        aewr_region_num,
        year,
        oews_area_code,
        gap_closure,
        oews_area_mean_hourly_wage,
        oews_area_prior_weight_all,
        oews_area_weight_wage_calibrated
      ),
    by = c("aewr_region_num", "year", "oews_area_code")
  ) %>%
  mutate(
    county_share_within_oews_area = county_area_prior_weight /
      oews_area_prior_weight_all,
    donor_weight = oews_area_weight_wage_calibrated *
      county_share_within_oews_area
  ) %>%
  filter(!is.na(donor_weight), donor_weight > 0) %>%
  rename(donor_cluster = iv_cluster) %>%
  # The donor map repeats ranks 1:d for each cumulative donor-set size d.
  inner_join(
    iv_donor_clusters,
    by = c("aewr_region_num", "iv_k", "donor_cluster"),
    relationship = "many-to-many"
  )

oews_donor_candidate_support <- oews_area_donor_candidates %>%
  group_by(
    aewr_region_num,
    year,
    iv_k,
    donor_cluster_count,
    target_cluster
  ) %>%
  summarise(
    oews_iv_candidate_donor_clusters = n_distinct(donor_cluster),
    oews_iv_candidate_areas = n_distinct(oews_area_code),
    oews_iv_candidate_units = n_distinct(cz_aewr_region_fe),
    .groups = "drop"
  )

# Exclude the entire OEWS area if any part of it touches the target cluster.
oews_area_donor_eligible <- oews_area_donor_candidates %>%
  anti_join(
    target_cluster_oews_areas,
    by = c(
      "aewr_region_num",
      "year",
      "iv_k",
      "target_cluster",
      "oews_area_code"
    )
  )

oews_donor_wages <- oews_area_donor_eligible %>%
  group_by(
    aewr_region_num,
    year,
    iv_k,
    donor_cluster_count,
    target_cluster
  ) %>%
  summarise(
    z_oews_entropy_agwage_l1 = sum(
      donor_weight * oews_area_mean_hourly_wage
    ) /
      sum(donor_weight),
    oews_iv_donor_weight = sum(donor_weight),
    oews_iv_donor_clusters = n_distinct(donor_cluster),
    oews_iv_donor_areas = n_distinct(oews_area_code),
    oews_iv_donor_units = n_distinct(cz_aewr_region_fe),
    oews_iv_donor_cluster_distance = min(donor_cluster_distance),
    oews_iv_farthest_donor_cluster_distance = max(
      donor_cluster_distance
    ),
    .groups = "drop"
  ) %>%
  left_join(
    oews_donor_candidate_support,
    by = c(
      "aewr_region_num",
      "year",
      "iv_k",
      "donor_cluster_count",
      "target_cluster"
    )
  ) %>%
  mutate(
    oews_iv_overlap_areas_excluded = oews_iv_candidate_areas -
      oews_iv_donor_areas
  )

iv_oews_long <- iv_clusters %>%
  transmute(
    cz_aewr_region_fe,
    aewr_region_num,
    iv_k,
    target_cluster = iv_cluster
  ) %>%
  inner_join(
    oews_donor_wages,
    by = c("aewr_region_num", "iv_k", "target_cluster"),
    relationship = "many-to-many"
  ) %>%
  # The outcome in year t uses the donor wage level from t - 1.
  mutate(
    year = year + 1L,
    gap_closure = 1,
    gap_closure_label = "g100",
    moment_spec = "wage_only"
  ) %>%
  arrange(cz_aewr_region_fe, year, iv_k, donor_cluster_count)

write_parquet(iv_oews_long, path_int("iv_oews_entropy_long.parquet"))
