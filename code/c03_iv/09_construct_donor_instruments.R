# Purpose: Construct lagged OEWS wage IVs from cumulative farthest-donor sets.
# Inputs: clusters, donor pairs, priors, wage-only weights, and FLS-moment weights.
# Output: iv_oews_entropy_long.parquet.
# Run after: 05, 06, and 08.

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
county_prior_by_spec_path <- path_int(
  "fls_county_oews_area_prior_weight_by_spec.parquet"
)
fls_county_oews_area_prior <- if (file.exists(
  county_prior_by_spec_path
)) {
  read_parquet(county_prior_by_spec_path)
} else {
  read_parquet(
    path_int("fls_county_oews_area_prior_weight.parquet")
  ) %>%
    mutate(prior_spec = "bea")
}
fls_county_oews_area_prior <- fls_county_oews_area_prior %>%
  mutate(year = as.integer(year))
wage_only_weights <- read_parquet(
  path_int("fls_oews_area_weight_wage_calibrated.parquet")
) %>%
  mutate(year = as.integer(year)) %>%
  filter(near(gap_closure, 1)) %>%
  transmute(
    aewr_region_num,
    year,
    oews_area_code,
    gap_closure,
    prior_spec = "bea",
    moment_spec = "wage_only_exact",
    weight_spec_label = "wage_only_exact",
    calibration_mode = "exact",
    duration_analogue = NA_character_,
    soft_penalty = NA_real_,
    oews_area_mean_hourly_wage,
    oews_area_prior_weight_all,
    entropy_weight = oews_area_weight_wage_calibrated,
    calibration_status
  )

fls_auxiliary_weights <- read_parquet(
  path_int("fls_oews_area_weight_soft_calibrated.parquet")
) %>%
  mutate(year = as.integer(year)) %>%
  filter(
    include_wage_target,
    near(gap_closure, 1)
  ) %>%
  transmute(
    aewr_region_num,
    year,
    oews_area_code,
    gap_closure,
    prior_spec,
    moment_spec,
    weight_spec_label,
    calibration_mode,
    duration_analogue,
    soft_penalty,
    oews_area_mean_hourly_wage,
    oews_area_prior_weight_all,
    entropy_weight = oews_area_weight_entropy_calibrated,
    calibration_status
  )

fls_oews_area_calibrated <- bind_rows(
  wage_only_weights,
  fls_auxiliary_weights
)

weight_spec_metadata <- fls_oews_area_calibrated %>%
  distinct(
    weight_spec_label,
    prior_spec,
    calibration_mode,
    duration_analogue
  )

county_oews_area_units <- fls_county_oews_area_prior %>%
  select(
    county_ansi = countyfips,
    year,
    aewr_region_num,
    cz_aewr_region_fe,
    prior_spec,
    oews_area_code,
    county_area_allocation,
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
        prior_spec,
        oews_area_code,
        gap_closure,
        moment_spec,
        soft_penalty,
        weight_spec_label,
        oews_area_mean_hourly_wage,
        oews_area_prior_weight_all,
        entropy_weight
      ),
    by = c(
      "aewr_region_num",
      "year",
      "prior_spec",
      "oews_area_code"
    ),
    relationship = "many-to-many"
  ) %>%
  mutate(
    county_share_within_oews_area = county_area_prior_weight /
      oews_area_prior_weight_all,
    donor_weight = entropy_weight *
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
    gap_closure,
    moment_spec,
    soft_penalty,
    weight_spec_label,
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

oews_unweighted_donor_county_wages <- oews_area_donor_eligible %>%
  group_by(
    aewr_region_num,
    year,
    gap_closure,
    moment_spec,
    soft_penalty,
    weight_spec_label,
    iv_k,
    donor_cluster_count,
    target_cluster,
    cz_aewr_region_fe,
    county_ansi
  ) %>%
  summarise(
    donor_county_oews_wage = weighted.mean(
      oews_area_mean_hourly_wage,
      county_area_allocation,
      na.rm = TRUE
    ),
    .groups = "drop"
  )

# Give every eligible donor county equal weight within its CZ-region unit,
# then give every donor CZ-region unit equal weight. This is the unweighted
# comparison to the entropy-weighted wage below; both use identical donor
# clusters and the same OEWS-area overlap exclusion.
oews_unweighted_donor_wages <- oews_unweighted_donor_county_wages %>%
  group_by(
    aewr_region_num,
    year,
    gap_closure,
    moment_spec,
    soft_penalty,
    weight_spec_label,
    iv_k,
    donor_cluster_count,
    target_cluster,
    cz_aewr_region_fe
  ) %>%
  summarise(
    donor_cz_oews_wage = mean(donor_county_oews_wage, na.rm = TRUE),
    oews_iv_unweighted_donor_counties = n_distinct(county_ansi),
    .groups = "drop"
  ) %>%
  group_by(
    aewr_region_num,
    year,
    gap_closure,
    moment_spec,
    soft_penalty,
    weight_spec_label,
    iv_k,
    donor_cluster_count,
    target_cluster
  ) %>%
  summarise(
    z_oews_unweighted_agwage_l1 = mean(
      donor_cz_oews_wage,
      na.rm = TRUE
    ),
    oews_iv_unweighted_donor_units = n_distinct(
      cz_aewr_region_fe
    ),
    oews_iv_unweighted_donor_counties = sum(
      oews_iv_unweighted_donor_counties
    ),
    .groups = "drop"
  )

oews_donor_wages <- oews_area_donor_eligible %>%
  group_by(
    aewr_region_num,
    year,
    gap_closure,
    moment_spec,
    soft_penalty,
    weight_spec_label,
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
      "gap_closure",
      "moment_spec",
      "soft_penalty",
      "weight_spec_label",
      "iv_k",
      "donor_cluster_count",
      "target_cluster"
    )
  ) %>%
  mutate(
    oews_iv_overlap_areas_excluded = oews_iv_candidate_areas -
      oews_iv_donor_areas
  ) %>%
  left_join(
    oews_unweighted_donor_wages,
    by = c(
      "aewr_region_num",
      "year",
      "gap_closure",
      "moment_spec",
      "soft_penalty",
      "weight_spec_label",
      "iv_k",
      "donor_cluster_count",
      "target_cluster"
    ),
    relationship = "one-to-one"
  ) %>%
  left_join(
    weight_spec_metadata,
    by = "weight_spec_label",
    relationship = "many-to-one"
  )

stopifnot(
  all(
    oews_donor_wages$oews_iv_unweighted_donor_units ==
      oews_donor_wages$oews_iv_donor_units
  ),
  all(is.finite(oews_donor_wages$z_oews_unweighted_agwage_l1))
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
    gap_closure_label = "g100"
  ) %>%
  arrange(
    cz_aewr_region_fe,
    year,
    weight_spec_label,
    iv_k,
    donor_cluster_count
  )

write_parquet(iv_oews_long, path_int("iv_oews_entropy_long.parquet"))
