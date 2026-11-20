# Purpose: Compare the instrument's weight/wage decomposition and first stages
# under the design OEWS wage and under an OEWS wage adjusted by annual county
# QWI factors, with weights recalibrated on the adjusted wage.
# Inputs: qwi_county_ag_quarterly_earnings.parquet, the stage 04 calibration
#   inputs and weights, panel_iv_county_donor_frame.parquet,
#   panel_iv_instrument_cluster_year.parquet, the stage 10 decomposition,
#   panel_iv_county_year.parquet, the shared county panel and PPI.
# Outputs (outputs/tables):
#   iv_qwi_wage_diagnostics_coverage.csv
#   iv_qwi_wage_diagnostics_summary.csv
#   iv_qwi_wage_diagnostics_first_stage.csv
#
# This is a diagnostic. The county-mapped OEWS-area hourly wage remains the
# only wage of the design, in the calibration and in the instrument. The stage
# swaps the wage in memory, rebuilds the instrument with stage 05's rules and
# repeats stage 10's decomposition. It writes no calibration or instrument
# artifact, estimates no outcome equation, and no result selects a wage source.
#
# Variants:
#   oews                     the design wage and weights; must reproduce
#                            stages 04, 05 and 10 exactly.
#   qwi_annual_recalibrated  the OEWS wage times an annual county factor, in
#                            the calibration's county wage feature and in the
#                            instrument, with weights recalibrated by stage
#                            04's moments and solver. The region's wage path
#                            stays OEWS; each county's position within the
#                            region follows QWI year by year.
# The annual factor is the county's QWI/OEWS ratio in that year divided by its
# AEWR region's mean ratio, weighted by mean prior weight times the OEWS wage.
# That keeps each region-year's prior-mean wage, and so the calibration's gap
# to the FLS wage target, close to the design's. A county-year without a ratio
# takes the county's nearest earlier year, else its nearest later year, and a
# county never observed keeps factor one.
# QWI earnings are average monthly earnings of stable NAICS 111/112 jobs with
# a published value, averaged over quarters and both industries with
# stable-employment weights and deflated with the source-year PPI. They mix
# pay with hours and are not an hourly wage. Dividing by 2080 / 12 hours only
# puts them on the OEWS scale; the regional normalization removes that
# constant from the factor.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "helpers.R"))
source(path_code("designs", "panel_iv", "io.R"))
source(path_code("designs", "panel_iv", "calibration.R"))
source(path_code("designs", "panel_iv", "specification.R"))
source(path_code("designs", "panel_iv", "decomposition.R"))
library(arrow)
library(dplyr)
library(fixest)
library(readr)
library(tibble)
library(tidyr)

source_years <- seq.int(
  DISSIMILARITY_IV_POLICY_START_YEAR - 1L,
  DISSIMILARITY_IV_POLICY_END_YEAR - 1L
)
factor_years <- sort(unique(c(IV_DECOMPOSITION_BASE_WAGE_YEARS, source_years)))
group_columns <- c("aewr_region_id", "target_cluster")
component_columns <- c("z_balanced", "z_weight", "z_wage", "z_interaction")
hours_per_month <- 2080 / 12

ppi <- read_parquet(path_int("ppi_2012.parquet")) %>%
  transmute(source_year = as.integer(year), source_year_ppi_2012 = as.numeric(ppi_2012))

# Annual real county QWI earnings ---------------------------------------------
qwi <- iv_read(path_int("qwi_county_ag_quarterly_earnings.parquet"))
iv_require_columns(
  qwi,
  c(
    "county_fips", "year", "qtr", "industry_code", "qwi_stable_employment",
    "qwi_stable_monthly_earnings", "qwi_stable_monthly_earnings_status"
  ),
  "QWI earnings"
)
assert_geo_columns(qwi, "county_fips", allow_na = character())
qwi_annual <- qwi %>%
  filter(
    industry_code %in% c("111", "112"),
    qwi_stable_monthly_earnings_status == 1L,
    is.finite(qwi_stable_monthly_earnings), qwi_stable_monthly_earnings > 0,
    is.finite(qwi_stable_employment), qwi_stable_employment > 0
  ) %>%
  group_by(county_fips, source_year = as.integer(year)) %>%
  summarise(
    qwi_monthly_earnings =
      sum(qwi_stable_monthly_earnings * qwi_stable_employment) /
        sum(qwi_stable_employment),
    qwi_quarters = n_distinct(qtr),
    .groups = "drop"
  ) %>%
  inner_join(ppi, by = "source_year", relationship = "many-to-one") %>%
  mutate(qwi_real_wage = qwi_monthly_earnings / hours_per_month / source_year_ppi_2012) %>%
  select(county_fips, source_year, qwi_real_wage, qwi_quarters)

# Annual county factor for the base and source years ----------------------------
county_wages <- read_parquet(
  path_processed("county_year_panel.parquet"),
  col_select = c(
    "county_fips", "year", "oews_big_six_mean_hourly_wage", "oews_wage_observed"
  )
) %>%
  as.data.frame()
oews_real <- iv_real_donor_wages(county_wages, ppi)
oews_base <- iv_decomposition_base_wages(county_wages, ppi) %>%
  rename(oews_base_wage = base_real_hourly_wage)

design_weights <- iv_read(path_int("panel_iv_fls_county_weight_summary.parquet")) %>%
  iv_require_calibration_version() %>%
  filter(specification == IV_PRIMARY_SPEC, source_year %in% source_years) %>%
  select(
    county_fips, aewr_region_id, source_year, frame_prior_weight,
    design_weight = calibrated_center_weight
  )
annual_factors <- design_weights %>%
  group_by(county_fips, aewr_region_id) %>%
  summarise(mean_prior_weight = mean(frame_prior_weight), .groups = "drop") %>%
  expand_grid(source_year = factor_years) %>%
  left_join(oews_real, by = c("county_fips", "source_year"), relationship = "one-to-one") %>%
  left_join(
    select(qwi_annual, county_fips, source_year, qwi_real_wage),
    by = c("county_fips", "source_year"), relationship = "one-to-one"
  ) %>%
  mutate(
    ratio = qwi_real_wage / real_hourly_wage,
    annual_factor_observed = is.finite(ratio),
    wage_mass = if_else(annual_factor_observed, mean_prior_weight * real_hourly_wage, 0)
  ) %>%
  group_by(aewr_region_id, source_year) %>%
  mutate(
    annual_factor = if_else(
      annual_factor_observed,
      ratio / (sum(wage_mass * ratio, na.rm = TRUE) / sum(wage_mass)),
      NA_real_
    )
  ) %>%
  group_by(county_fips) %>%
  arrange(source_year, .by_group = TRUE) %>%
  fill(annual_factor, .direction = "downup") %>%
  ungroup() %>%
  mutate(annual_factor = coalesce(annual_factor, 1))
iv_require_unique(annual_factors, c("county_fips", "source_year"), "Annual county factors")
adjusted_base <- annual_factors %>%
  filter(source_year %in% IV_DECOMPOSITION_BASE_WAGE_YEARS) %>%
  group_by(county_fips) %>%
  summarise(
    adjusted_base_wage = finite_mean(real_hourly_wage * annual_factor),
    .groups = "drop"
  )
annual_factors <- select(
  annual_factors,
  county_fips, aewr_region_id, source_year, annual_factor_observed, annual_factor
)

# Recalibrate with stage 04's inputs, moments and solver. The design wage must
# reproduce the stage 04 weights before the adjusted wage is used.
annual <- iv_annual_targets(iv_read(path_int("fls_region.parquet")), IV_SOURCE_YEARS)
quarterly <- iv_paired_targets(
  iv_read(path_int("fls_region_quarterly_workers.parquet")),
  iv_read(path_int("fls_region_quarterly_wages.parquet")),
  IV_SOURCE_YEARS
)
recalibrate <- function(wages) {
  cells <- iv_build_features(
    iv_read(path_int("panel_iv_fls_frame.parquet")),
    wages,
    iv_read(path_int("qcew_county_ag_quarterly_employment.parquet")),
    annual,
    quarterly,
    IV_SOURCE_YEARS,
    NULL
  )$cells
  iv_recover_cells(cells)
}
reproduced <- recalibrate(county_wages)$weights %>%
  inner_join(
    design_weights,
    by = c("county_fips", "aewr_region_id", "source_year"),
    relationship = "one-to-one"
  )
if (
  nrow(reproduced) != nrow(design_weights) ||
    max(abs(reproduced$calibrated_center_weight - reproduced$design_weight)) > 1e-12
) {
  stop("The design wage does not reproduce the stage 04 weights.", call. = FALSE)
}
recalibrated <- recalibrate(
  county_wages %>%
    mutate(source_year = as.integer(year)) %>%
    left_join(
      select(annual_factors, county_fips, source_year, annual_factor),
      by = c("county_fips", "source_year"), relationship = "one-to-one"
    ) %>%
    mutate(
      oews_big_six_mean_hourly_wage =
        oews_big_six_mean_hourly_wage * coalesce(annual_factor, 1)
    ) %>%
    select(-annual_factor, -source_year)
)
calibrations <- list(
  oews = iv_read(path_int("panel_iv_fls_county_calibration_diagnostics.parquet")) %>%
    filter(specification == IV_PRIMARY_SPEC, source_year %in% source_years),
  qwi_annual_recalibrated = recalibrated$calibration_diagnostics
)
weight_shift <- recalibrated$weights %>%
  inner_join(
    design_weights,
    by = c("county_fips", "aewr_region_id", "source_year"),
    relationship = "one-to-one"
  ) %>%
  group_by(aewr_region_id, source_year) %>%
  summarise(
    distance = sum(abs(calibrated_center_weight - design_weight)) / 2,
    .groups = "drop"
  )

donor_frame <- read_parquet(path_int("panel_iv_county_donor_frame.parquet")) %>%
  filter(
    instrument_spec_label == DISSIMILARITY_IV_PRIMARY_INSTRUMENT_LABEL,
    source_year %in% source_years
  ) %>%
  left_join(oews_base, by = "county_fips", relationship = "many-to-one") %>%
  left_join(adjusted_base, by = "county_fips", relationship = "many-to-one") %>%
  left_join(qwi_annual, by = c("county_fips", "source_year"), relationship = "many-to-one") %>%
  left_join(
    annual_factors,
    by = c("county_fips", "aewr_region_id", "source_year"), relationship = "many-to-one"
  ) %>%
  left_join(
    recalibrated$weights %>%
      select(
        county_fips, aewr_region_id, source_year,
        recalibrated_weight = calibrated_center_weight
      ),
    by = c("county_fips", "aewr_region_id", "source_year"),
    relationship = "many-to-one"
  )
if (anyNA(donor_frame$annual_factor) || anyNA(donor_frame$recalibrated_weight)) {
  stop("Donor counties are missing a factor or recalibrated weight.", call. = FALSE)
}

# Each variant supplies the donor wage, its base level and the calibrated
# weight.
variants <- list(
  oews = donor_frame %>% mutate(
    wage = donor_real_hourly_wage,
    base_real_hourly_wage = oews_base_wage,
    weight = calibrated_center_weight
  ),
  qwi_annual_recalibrated = donor_frame %>% mutate(
    wage = donor_real_hourly_wage * annual_factor,
    base_real_hourly_wage = adjusted_base_wage,
    weight = recalibrated_weight
  )
)

pipeline_instrument <- read_parquet(
  path_int("panel_iv_instrument_cluster_year.parquet")
) %>%
  filter(instrument_spec_label == DISSIMILARITY_IV_PRIMARY_INSTRUMENT_LABEL) %>%
  select(
    all_of(group_columns), source_year, policy_year,
    pipeline_z = z_dissimilarity_real, pipeline_available = instrument_available
  )
pipeline_components <- read_parquet(
  path_int("panel_iv_instrument_decomposition_cluster_year.parquet")
) %>%
  filter(decomposition_available) %>%
  select(all_of(c(group_columns, "source_year", component_columns)))

estimation_spec <- panel_iv_read_specification(
  path_code("designs", "panel_iv", "estimation_spec.json")
)
endogenous <- estimation_spec$endogenous
control_terms <- paste(estimation_spec$controls, collapse = " + ")
fixed_effects <- paste(estimation_spec$fixed_effects, collapse = " + ")
cluster_vcov <- stats::as.formula(paste("~", estimation_spec$primary_inference_cluster))
panel <- read_parquet(
  path_processed("panel_iv_county_year.parquet"),
  col_select = all_of(unique(c(
    "county_fips", "year", group_columns, "aewr_iv_cluster_id", endogenous,
    DISSIMILARITY_IV_CONTROL_COLUMNS
  )))
) %>%
  mutate(year = as.integer(year)) %>%
  filter(
    year >= estimation_spec$policy_years[[1]],
    year <= estimation_spec$policy_years[[2]],
    if_all(
      all_of(c(endogenous, DISSIMILARITY_IV_CONTROL_COLUMNS)),
      ~ !is.na(.x) & is.finite(.x)
    ),
    !is.na(aewr_iv_cluster_id)
  )

# Preferred-column first stage on one excluded instrument.
first_stage <- function(data, term, label) {
  model <- feols(
    as.formula(paste(endogenous, "~", term, "+", control_terms, "|", fixed_effects)),
    data = data, vcov = cluster_vcov, warn = FALSE, notes = FALSE
  )
  row <- coeftable(model)[term, ]
  tibble(
    first_stage = label,
    coefficient = unname(row["Estimate"]),
    standard_error = unname(row["Std. Error"]),
    f_statistic = unname(row["t value"])^2,
    observations = nobs(model),
    inference_clusters = n_distinct(data$aewr_iv_cluster_id[fixest::obs(model)])
  )
}

partial_out <- function(data, column) {
  as.numeric(resid(feols(
    as.formula(paste(column, "~", control_terms, "|", fixed_effects)),
    data = data, warn = FALSE, notes = FALSE
  )))
}

summary_rows <- first_stage_rows <- list()
for (variant in names(variants)) {
  message("variant = ", variant)
  # Stage 05's eligibility, renormalization and support rules.
  frame <- variants[[variant]] %>%
    mutate(
      weight_valid = is.finite(weight) & weight > 0,
      eligible = weight_valid & !target_area_overlap & donor_wage_available,
      donor_real_hourly_wage = wage
    ) %>%
    group_by(across(all_of(c(group_columns, "source_year")))) %>%
    mutate(
      eligible_mass = sum(if_else(eligible, weight, 0)),
      variant_instrument_weight = if_else(
        eligible & eligible_mass > 0, weight / eligible_mass, NA_real_
      )
    ) %>%
    ungroup()
  instrument <- frame %>%
    group_by(across(all_of(c(group_columns, "source_year")))) %>%
    summarise(
      z_raw = sum(variant_instrument_weight * donor_real_hourly_wage, na.rm = TRUE),
      weight_sum = sum(variant_instrument_weight, na.rm = TRUE),
      donor_clusters = n_distinct(donor_cluster[eligible]),
      .groups = "drop"
    ) %>%
    mutate(
      available = donor_clusters == DISSIMILARITY_IV_MIN_OBSERVED_DONOR_CLUSTERS &
        abs(weight_sum - 1) <= 1e-10 & is.finite(z_raw) & z_raw > 0,
      z_instrument = if_else(available, z_raw, NA_real_)
    )
  instrument <- pipeline_instrument %>%
    left_join(instrument, by = c(group_columns, "source_year"), relationship = "one-to-one") %>%
    mutate(available = coalesce(available, FALSE))

  balanced_donors <- iv_balanced_components(
    frame,
    group_columns,
    source_years,
    DISSIMILARITY_IV_MIN_OBSERVED_DONOR_CLUSTERS
  )
  components <- balanced_donors$components
  # Spread of base wage levels across each balanced donor set, and the share
  # of its mass that sits away from each county's mean weight.
  donor_sets <- balanced_donors$balanced %>%
    semi_join(components, by = group_columns) %>%
    group_by(across(all_of(c(group_columns, "county_fips")))) %>%
    mutate(reference_weight = mean(decomposition_weight)) %>%
    group_by(across(all_of(c(group_columns, "source_year")))) %>%
    summarise(
      mass_moved = sum(abs(decomposition_weight - reference_weight)) / 2,
      level_sd = sqrt(sum(
        reference_weight *
          (base_real_hourly_wage - sum(reference_weight * base_real_hourly_wage))^2
      )),
      donor_counties = n(),
      distinct_levels = n_distinct(round(base_real_hourly_wage, 6)),
      .groups = "drop"
    )

  if (variant == "oews") {
    stage10 <- inner_join(
      pipeline_components, components,
      by = c(group_columns, "source_year"), suffix = c("_pipeline", "")
    )
    component_gap <- max(vapply(
      component_columns,
      function(column) max(abs(stage10[[column]] - stage10[[paste0(column, "_pipeline")]])),
      numeric(1)
    ))
    if (
      !identical(instrument$available, instrument$pipeline_available) ||
        max(abs(instrument$z_instrument - instrument$pipeline_z), na.rm = TRUE) > 1e-10 ||
        nrow(stage10) != nrow(pipeline_components) ||
        component_gap > 1e-10
    ) {
      stop("The OEWS wage does not reproduce stages 05 and 10.", call. = FALSE)
    }
  }

  data <- panel %>%
    left_join(
      instrument %>%
        transmute(across(all_of(group_columns)), year = policy_year, z_instrument),
      by = c(group_columns, "year"), relationship = "many-to-one"
    ) %>%
    left_join(
      components %>%
        transmute(
          across(all_of(group_columns)), year = source_year + 1L,
          across(all_of(component_columns))
        ),
      by = c(group_columns, "year"), relationship = "many-to-one"
    )
  full_sample <- filter(data, is.finite(z_instrument))
  decomposition_sample <- filter(full_sample, is.finite(z_balanced))

  # Variance shares after fixed effects and controls, as in stage 10.
  residuals <- sapply(
    c("z_instrument", component_columns),
    function(column) partial_out(decomposition_sample, column)
  )
  covariance <- crossprod(residuals) / nrow(residuals)
  balanced_variance <- covariance["z_balanced", "z_balanced"]

  calibration <- calibrations[[variant]]
  summary_rows[[variant]] <- tibble(
    variant = variant,
    mean_moment_gap_closed = mean(
      1 - calibration$calibrated_standardized_residual_norm /
        calibration$input_standardized_residual_norm
    ),
    median_effective_counties = median(calibration$calibrated_effective_county_count),
    maximum_county_weight = max(calibration$maximum_calibrated_county_weight),
    mean_donor_mass_moved = mean(donor_sets$mass_moved),
    instrument_cluster_years = sum(instrument$available),
    decomposition_clusters = n_distinct(components[group_columns]),
    median_donor_counties = median(donor_sets$donor_counties),
    median_distinct_wage_levels = median(donor_sets$distinct_levels),
    mean_cross_donor_level_sd = mean(donor_sets$level_sd),
    weight_share = covariance["z_weight", "z_balanced"] / balanced_variance,
    wage_share = covariance["z_wage", "z_balanced"] / balanced_variance,
    interaction_share = covariance["z_interaction", "z_balanced"] / balanced_variance,
    weight_variance = covariance["z_weight", "z_weight"],
    wage_variance = covariance["z_wage", "z_wage"],
    balanced_variance = balanced_variance,
    instrument_variance = covariance["z_instrument", "z_instrument"],
    instrument_balanced_correlation = covariance["z_instrument", "z_balanced"] /
      sqrt(covariance["z_instrument", "z_instrument"] * balanced_variance),
    full_observations = nrow(full_sample),
    decomposition_observations = nrow(decomposition_sample)
  )
  first_stage_rows[[variant]] <- bind_cols(
    tibble(variant = variant),
    bind_rows(
      first_stage(full_sample, "z_instrument", "instrument, full sample"),
      first_stage(decomposition_sample, "z_instrument", "instrument, decomposition sample"),
      first_stage(decomposition_sample, "z_balanced", "balanced instrument"),
      first_stage(decomposition_sample, "z_weight", "weight component only"),
      first_stage(decomposition_sample, "z_wage", "wage component only")
    )
  )
}

# QWI coverage and the size of the adjustment -----------------------------------
donors <- donor_frame %>%
  filter(donor_eligible) %>%
  distinct(
    county_fips, source_year, qwi_real_wage, qwi_quarters, donor_real_hourly_wage,
    annual_factor_observed, annual_factor
  )
calibrated <- design_weights %>%
  inner_join(
    annual_factors,
    by = c("county_fips", "aewr_region_id", "source_year"), relationship = "one-to-one"
  ) %>%
  inner_join(
    county_wages %>%
      transmute(county_fips, source_year = as.integer(year),
        wage = oews_big_six_mean_hourly_wage),
    by = c("county_fips", "source_year"), relationship = "one-to-one"
  )
# Prior-mean calibration wage with and without the factor, by region-year.
prior_means <- calibrated %>%
  filter(is.finite(wage), wage > 0) %>%
  group_by(aewr_region_id, source_year) %>%
  summarise(
    relative_mean = sum(frame_prior_weight * wage * annual_factor) /
      sum(frame_prior_weight * wage),
    mass_with_factor = sum(frame_prior_weight * annual_factor_observed) /
      sum(frame_prior_weight),
    .groups = "drop"
  )
observed_factors <- donors$annual_factor[donors$annual_factor_observed]
coverage <- tibble(
  eligible_donor_county_years = nrow(donors),
  share_with_observed_factor = mean(donors$annual_factor_observed),
  share_with_four_qwi_quarters = mean(donors$qwi_quarters %in% 4L),
  share_with_filled_factor = mean(!donors$annual_factor_observed & donors$annual_factor != 1),
  calibrated_county_years = nrow(calibrated),
  calibrated_share_with_observed_factor = mean(calibrated$annual_factor_observed),
  mean_prior_mass_with_observed_factor = mean(prior_means$mass_with_factor),
  minimum_adjusted_to_oews_prior_mean_wage = min(prior_means$relative_mean),
  maximum_adjusted_to_oews_prior_mean_wage = max(prior_means$relative_mean),
  mean_distance_recalibrated_from_design_weights = mean(weight_shift$distance),
  qwi_oews_level_correlation = cor(
    donors$qwi_real_wage, donors$donor_real_hourly_wage,
    use = "complete.obs"
  ),
  factor_p05 = unname(quantile(observed_factors, 0.05)),
  factor_p25 = unname(quantile(observed_factors, 0.25)),
  factor_p75 = unname(quantile(observed_factors, 0.75)),
  factor_p95 = unname(quantile(observed_factors, 0.95))
)

summary_table <- bind_rows(summary_rows)
first_stage_table <- bind_rows(first_stage_rows)
dir.create(path_tables(), recursive = TRUE, showWarnings = FALSE)
write_csv(coverage, path_tables("iv_qwi_wage_diagnostics_coverage.csv"))
write_csv(summary_table, path_tables("iv_qwi_wage_diagnostics_summary.csv"))
write_csv(first_stage_table, path_tables("iv_qwi_wage_diagnostics_first_stage.csv"))
print(coverage, width = Inf)
print(summary_table, width = Inf)
print(first_stage_table, n = Inf, width = Inf)
