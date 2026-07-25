# Purpose: Estimate exact, interval, and low-rho FLS entropy projections.
# Inputs: FLS targets, OEWS priors, and public auxiliary moments.
# Outputs: area/county calibrated weights and cell diagnostics.
# Run after: 03_oews_area_prior_weights.R and 04_auxiliary_moments.R.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
source(path_code("c00_shared", "entropy_calibration.R"))

minimum_auxiliary_observed_prior_mass <- 0.90
gap_closure_values <- 1
# In a soft specification rho multiplies one-half the squared standardized
# imbalance. These values are deliberately below or equal to one so the path
# stays close to the prior; rho is not a sampling fraction.
soft_penalty_values <- c(0.01, 0.03, 0.10, 0.30, 1)
soft_penalty_codes <- c("001", "003", "010", "030", "100")

# The publication specification is BEA-prior, exact wage plus January/April/
# July FLS shares. Duration, intervals, low-rho penalties, and alternative
# priors are explicitly labeled sensitivities rather than silent fallbacks.
primary_specs <- bind_rows(
  tribble(
    ~moment_spec                          , ~weight_spec_label       , ~calibration_mode ,
    ~duration_analogue                    , ~soft_penalty            ,
    "wage_seasonal_exact"                 , "wage_seasonal_exact"    , "exact"           ,
    NA_character_                         , NA_real_                 ,
    "wage_seasonal_qwi_duration_exact"    ,
    "wage_seasonal_qwi_duration_exact"    , "exact"                  ,
    "qwi"                                 , NA_real_                 ,
    "wage_seasonal_census_duration_exact" ,
    "wage_seasonal_census_duration_exact" , "exact"                  ,
    "census_bridge"                       , NA_real_                 ,
    "wage_seasonal_interval"              , "wage_seasonal_interval" , "interval"        ,
    NA_character_                         , NA_real_
  ),
  tibble(
    moment_spec = paste0(
      "wage_seasonal_soft_rho",
      soft_penalty_codes
    ),
    weight_spec_label = moment_spec,
    calibration_mode = "soft",
    duration_analogue = NA_character_,
    soft_penalty = soft_penalty_values
  )
) %>%
  mutate(prior_spec = "bea")

alternative_prior_specs <- tibble(
  prior_spec = c(
    "census_workers",
    "census_payroll",
    "qwi_employment"
  ),
  moment_spec = "wage_seasonal_exact",
  weight_spec_label = paste0(
    "wage_seasonal_exact_prior_",
    prior_spec
  ),
  calibration_mode = "exact",
  duration_analogue = NA_character_,
  soft_penalty = NA_real_
)

calibration_specs <- bind_rows(
  primary_specs,
  alternative_prior_specs
) %>%
  crossing(gap_closure = gap_closure_values) %>%
  mutate(include_wage_target = TRUE)

seasonal_features <- c(
  "qcew_qwi_employment_share_january",
  "qcew_qwi_employment_share_april",
  "qcew_qwi_employment_share_july"
)
seasonal_targets <- c(
  "fls_hired_worker_share_january",
  "fls_hired_worker_share_april",
  "fls_hired_worker_share_july"
)
seasonal_labels <- c(
  "seasonal_january",
  "seasonal_april",
  "seasonal_july"
)

feature_configuration <- function(moment_spec, duration_analogue) {
  features <- seasonal_features
  targets <- seasonal_targets
  labels <- seasonal_labels
  observed_share_names <- rep(
    "seasonal_feature_observed_share",
    length(seasonal_features)
  )
  if (identical(duration_analogue, "qwi")) {
    features <- c(features, "qwi_area_stable_employment_share")
    targets <- c(targets, "fls_hired_worker_150_plus_share")
    labels <- c(labels, "duration")
    observed_share_names <- c(
      observed_share_names,
      "qwi_duration_observed_share"
    )
  } else if (identical(duration_analogue, "census_bridge")) {
    features <- c(
      features,
      "census_hired_worker_150_plus_share_bridged"
    )
    targets <- c(targets, "fls_hired_worker_150_plus_share")
    labels <- c(labels, "duration")
    observed_share_names <- c(
      observed_share_names,
      "census_duration_observed_prior_share"
    )
  }
  list(
    features = features,
    targets = targets,
    labels = labels,
    observed_share_names = observed_share_names
  )
}

interval_band_path <- path_int("fls_auxiliary_interval_bands.parquet")
interval_bands <- if (file.exists(interval_band_path)) {
  read_parquet(interval_band_path) %>%
    filter(moment_label %in% seasonal_labels) %>%
    arrange(match(moment_label, seasonal_labels)) %>%
    select(moment_label, interval_half_width)
} else {
  tibble(
    moment_label = seasonal_labels,
    interval_half_width = NA_real_
  )
}
seasonal_interval_half_widths <- setNames(
  interval_bands$interval_half_width,
  interval_bands$moment_label
)[seasonal_labels]

fls_target <- read_parquet(path_int("fls_region.parquet"))
assert_geo_columns(fls_target, "aewr_region_id")
fls_target <- fls_target %>%
  transmute(
    aewr_region_id,
    year = preliminary_year,
    fls_target_wage = as.numeric(field_livestock_preliminary),
    fls_hired_worker_150_plus_share = as.numeric(
      fls_hired_worker_150_plus_share
    ),
    fls_hired_worker_share_january = as.numeric(
      fls_hired_worker_share_january
    ),
    fls_hired_worker_share_april = as.numeric(
      fls_hired_worker_share_april
    ),
    fls_hired_worker_share_july = as.numeric(
      fls_hired_worker_share_july
    )
  ) %>%
  filter(!is.na(aewr_region_id), !is.na(year)) %>%
  distinct(aewr_region_id, year, .keep_all = TRUE)

auxiliary_features <- read_parquet(path_int(
  "fls_oews_area_auxiliary_moments.parquet"
))
assert_geo_columns(
  auxiliary_features,
  c("aewr_region_id", "oews_area_code")
)

prior_by_spec_path <- path_int(
  "fls_oews_area_prior_weight_by_spec.parquet"
)
area_prior_by_spec <- if (file.exists(prior_by_spec_path)) {
  prior_source <- read_parquet(prior_by_spec_path)
  assert_geo_columns(
    prior_source,
    c("aewr_region_id", "oews_area_code")
  )
  prior_source
} else {
  auxiliary_features %>%
    transmute(
      aewr_region_id,
      year,
      prior_spec = "bea",
      oews_area_code,
      oews_area_name,
      oews_area_mean_hourly_wage,
      oews_area_tot_emp,
      oews_area_prior_weight_all,
      oews_area_county_count,
      oews_wage_observed,
      oews_observed_prior_mass,
      oews_area_prior_weight
    )
}

base_prior_columns <- c(
  "oews_area_name",
  "oews_area_mean_hourly_wage",
  "oews_area_tot_emp",
  "oews_area_prior_weight_all",
  "oews_area_county_count",
  "oews_wage_observed",
  "oews_observed_prior_mass",
  "oews_area_prior_weight"
)

fls_oews_area_prior <- area_prior_by_spec %>%
  left_join(
    auxiliary_features %>%
      select(
        -any_of(base_prior_columns)
      ),
    by = c("aewr_region_id", "year", "oews_area_code")
  ) %>%
  filter(
    oews_wage_observed,
    is.finite(oews_area_prior_weight),
    oews_area_prior_weight > 0
  ) %>%
  inner_join(fls_target, by = c("aewr_region_id", "year")) %>%
  group_by(aewr_region_id, year, prior_spec) %>%
  mutate(
    oews_prior_weighted_wage = sum(
      oews_area_prior_weight * oews_area_mean_hourly_wage
    )
  ) %>%
  ungroup()

fls_oews_area_calibration_grid <- fls_oews_area_prior %>%
  inner_join(calibration_specs, by = "prior_spec") %>%
  filter(!is.na(fls_target_wage), fls_target_wage > 0) %>%
  mutate(
    entropy_target_wage = oews_prior_weighted_wage +
      gap_closure * (fls_target_wage - oews_prior_weighted_wage)
  )

fls_oews_area_entropy_calibrated <- fls_oews_area_calibration_grid %>%
  group_by(
    aewr_region_id,
    year,
    prior_spec,
    include_wage_target,
    gap_closure,
    moment_spec,
    weight_spec_label,
    calibration_mode,
    duration_analogue,
    soft_penalty
  ) %>%
  group_modify(
    function(.x, .y) {
      configuration <- feature_configuration(
        moment_spec = .y$moment_spec[[1]],
        duration_analogue = .y$duration_analogue[[1]]
      )
      bands <- if (.y$calibration_mode[[1]] == "interval") {
        unname(seasonal_interval_half_widths[
          configuration$labels
        ])
      } else {
        numeric()
      }
      calibrate_entropy_cell(
        data = .x,
        entropy_target_wage = first(.x$entropy_target_wage),
        moment_spec = .y$moment_spec[[1]],
        calibration_mode = .y$calibration_mode[[1]],
        feature_names = configuration$features,
        target_names = configuration$targets,
        feature_labels = configuration$labels,
        feature_observed_share_names = configuration$observed_share_names,
        soft_penalty = .y$soft_penalty[[1]],
        interval_half_widths = bands,
        minimum_observed_prior_mass = minimum_auxiliary_observed_prior_mass
      )
    }
  ) %>%
  ungroup() %>%
  mutate(
    oews_area_weight_adjustment = oews_area_weight_entropy_calibrated /
      oews_area_prior_weight,
    entropy_kl_divergence_component = if_else(
      !is.na(oews_area_weight_entropy_calibrated) &
        oews_area_weight_entropy_calibrated > 0,
      oews_area_weight_entropy_calibrated *
        log(
          oews_area_weight_entropy_calibrated /
            oews_area_prior_weight
        ),
      0
    )
  )

fls_entropy_calibration_diagnostics <-
  fls_oews_area_entropy_calibrated %>%
  group_by(
    aewr_region_id,
    year,
    prior_spec,
    include_wage_target,
    gap_closure,
    moment_spec,
    weight_spec_label,
    calibration_mode,
    duration_analogue,
    soft_penalty
  ) %>%
  summarise(
    requested_moment_spec = first(requested_moment_spec),
    resolved_moment_spec = first(resolved_moment_spec),
    calibration_status = first(calibration_status),
    optimizer_convergence = first(optimizer_convergence),
    lp_feasibility_status = first(lp_feasibility_status),
    exact_max_abs_residual = first(exact_max_abs_residual),
    interval_max_violation = first(interval_max_violation),
    minimum_active_observed_prior_mass = first(
      minimum_active_observed_prior_mass
    ),
    maximum_active_imputed_prior_mass = if_else(
      is.na(minimum_active_observed_prior_mass),
      NA_real_,
      1 - minimum_active_observed_prior_mass
    ),
    fls_target_wage = first(fls_target_wage),
    entropy_target_wage = first(entropy_target_wage),
    oews_prior_weighted_wage = first(oews_prior_weighted_wage),
    oews_calibrated_weighted_wage = weighted_sum_if_observed(
      oews_area_weight_entropy_calibrated,
      oews_area_mean_hourly_wage
    ),
    wage_moment_error = oews_calibrated_weighted_wage -
      entropy_target_wage,
    fls_seasonal_january_target = first(
      fls_hired_worker_share_january
    ),
    calibrated_seasonal_january_moment = first(
      calibrated_seasonal_january_moment
    ),
    seasonal_january_standardized_imbalance = first(
      seasonal_january_standardized_imbalance
    ),
    seasonal_january_feature_observed_prior_mass = first(
      seasonal_january_feature_observed_prior_mass
    ),
    seasonal_january_interval_slack = first(
      seasonal_january_interval_slack
    ),
    fls_seasonal_april_target = first(
      fls_hired_worker_share_april
    ),
    calibrated_seasonal_april_moment = first(
      calibrated_seasonal_april_moment
    ),
    seasonal_april_standardized_imbalance = first(
      seasonal_april_standardized_imbalance
    ),
    seasonal_april_feature_observed_prior_mass = first(
      seasonal_april_feature_observed_prior_mass
    ),
    seasonal_april_interval_slack = first(
      seasonal_april_interval_slack
    ),
    fls_seasonal_july_target = first(fls_hired_worker_share_july),
    calibrated_seasonal_july_moment = first(
      calibrated_seasonal_july_moment
    ),
    seasonal_july_standardized_imbalance = first(
      seasonal_july_standardized_imbalance
    ),
    seasonal_july_feature_observed_prior_mass = first(
      seasonal_july_feature_observed_prior_mass
    ),
    seasonal_july_interval_slack = first(
      seasonal_july_interval_slack
    ),
    fls_duration_target = first(fls_hired_worker_150_plus_share),
    calibrated_duration_moment = first(calibrated_duration_moment),
    duration_standardized_imbalance = first(
      duration_standardized_imbalance
    ),
    duration_feature_observed_prior_mass = first(
      duration_feature_observed_prior_mass
    ),
    entropy_kl_divergence = if_else(
      all(is.na(oews_area_weight_entropy_calibrated)),
      NA_real_,
      sum(entropy_kl_divergence_component)
    ),
    prior_effective_area_count = 1 / sum(oews_area_prior_weight^2),
    calibrated_effective_area_count = if_else(
      all(is.na(oews_area_weight_entropy_calibrated)),
      NA_real_,
      1 / sum(oews_area_weight_entropy_calibrated^2)
    ),
    effective_area_count_ratio = calibrated_effective_area_count /
      prior_effective_area_count,
    maximum_weight_adjustment = max_if_observed(
      oews_area_weight_adjustment
    ),
    maximum_calibrated_area_weight = max_if_observed(
      oews_area_weight_entropy_calibrated
    ),
    .groups = "drop"
  )

county_prior_by_spec_path <- path_int(
  "fls_county_oews_area_prior_weight_by_spec.parquet"
)
fls_county_area_prior <- if (file.exists(county_prior_by_spec_path)) {
  read_parquet(county_prior_by_spec_path)
} else {
  read_parquet(path_int(
    "fls_county_oews_area_prior_weight.parquet"
  )) %>%
    mutate(prior_spec = "bea")
}
assert_geo_columns(
  fls_county_area_prior,
  c(
    "county_fips",
    "state_fips",
    "aewr_region_id",
    "cz_id",
    "oews_area_code"
  )
)

fls_county_weight_entropy_calibrated <- fls_county_area_prior %>%
  inner_join(
    fls_oews_area_entropy_calibrated %>%
      select(
        aewr_region_id,
        year,
        prior_spec,
        oews_area_code,
        include_wage_target,
        gap_closure,
        moment_spec,
        weight_spec_label,
        calibration_mode,
        duration_analogue,
        soft_penalty,
        oews_area_prior_weight_all,
        oews_area_weight_entropy_calibrated,
        calibration_status
      ),
    by = c(
      "aewr_region_id",
      "year",
      "prior_spec",
      "oews_area_code"
    ),
    relationship = "many-to-many"
  ) %>%
  mutate(
    county_share_within_oews_area = if_else(
      oews_area_prior_weight_all > 0,
      county_area_prior_weight / oews_area_prior_weight_all,
      NA_real_
    ),
    county_area_weight_entropy_calibrated = oews_area_weight_entropy_calibrated *
      county_share_within_oews_area
  ) %>%
  group_by(
    county_fips,
    year,
    state_fips,
    state_abbrev,
    aewr_region_id,
    cz_id,
    cz_aewr_region_fe,
    fls_county_weight_prior,
    prior_spec,
    include_wage_target,
    gap_closure,
    moment_spec,
    weight_spec_label,
    calibration_mode,
    duration_analogue,
    soft_penalty,
    calibration_status
  ) %>%
  summarise(
    fls_county_weight_entropy_calibrated = sum(
      county_area_weight_entropy_calibrated,
      na.rm = FALSE
    ),
    .groups = "drop"
  ) %>%
  mutate(
    # Historical alias for consumers that have not yet switched names.
    fls_county_weight_soft_calibrated = fls_county_weight_entropy_calibrated
  ) %>%
  arrange(county_fips, year) %>%
  group_by(county_fips, prior_spec, weight_spec_label, gap_closure) %>%
  mutate(
    fls_county_weight_entropy_calibrated_l1 = if_else(
      lag(year) == year - 1L,
      lag(fls_county_weight_entropy_calibrated),
      NA_real_
    ),
    fls_county_weight_soft_calibrated_l1 = fls_county_weight_entropy_calibrated_l1
  ) %>%
  ungroup()

assert_geo_columns(
  fls_oews_area_entropy_calibrated,
  c("aewr_region_id", "oews_area_code")
)
assert_geo_columns(
  fls_county_weight_entropy_calibrated,
  c("county_fips", "state_fips", "cz_id", "aewr_region_id")
)
write_parquet(
  fls_oews_area_entropy_calibrated,
  path_int("fls_oews_area_weight_soft_calibrated.parquet")
)

write_parquet(
  fls_county_weight_entropy_calibrated,
  path_int("fls_county_weight_soft_calibrated.parquet")
)

write_parquet(
  fls_entropy_calibration_diagnostics,
  path_int("fls_soft_entropy_diagnostics.parquet")
)
