# Purpose: Compare fixed prepolicy labor denominators and calendar wage inputs.
# Outputs: data-input panel/denominator audits and separately labeled estimates/power.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "border_discontinuity", "redesign.R"))
source(path_code("designs", "border_discontinuity", "power.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(dplyr)
library(jsonlite)
library(readr)
fixest::setFixest_nthreads(1)

spec <- fromJSON(path_code("designs", "border_discontinuity", "data_input_design.json"))
design <- fromJSON(path_code("designs", "border_discontinuity", "design.json"))
stopifnot(identical(as.integer(spec$policy_years),
                   as.integer(c(design$policy_start_year, design$policy_end_year))))
panel <- as.data.frame(read_parquet(path_processed("border_discontinuity_panel.parquet")))
shared <- read_parquet(path_processed("county_year_panel.parquet")) %>%
  select(county_fips, year, nbr_workers_certified_start_year, emp_farm_2011)
count_check <- left_join(select(panel, county_fips, year, nbr_workers_certified_start_year),
  select(shared, county_fips, year, nbr_workers_certified_start_year),
  by = c("county_fips", "year"), suffix = c("_border", "_shared"), relationship = "one-to-one")
stopifnot(all(count_check$nbr_workers_certified_start_year_border ==
                count_check$nbr_workers_certified_start_year_shared))
employment <- read_parquet(path_int("bea_caemp25n_data_year.parquet"))
census <- read_parquet(path_int("census_ag_hired_worker_duration_county.parquet"))
exposure <- read_parquet(path_int("h2a_worksite_county_year_exposure.parquet"))
memberships <- as.data.frame(read_parquet(path_int("border_discontinuity_geography_memberships.parquet")))

denominators <- employment %>%
  filter(year %in% spec$baseline_years) %>%
  mutate(hired_farm_jobs = emp_farm - emp_farm_propr) %>%
  group_by(county_fips) %>%
  summarise(
    baseline_years_observed = n_distinct(year),
    baseline_farm_years_valid = sum(is.finite(emp_farm) & emp_farm > 0),
    baseline_hired_years_valid = sum(is.finite(hired_farm_jobs) & hired_farm_jobs > 0),
    farm_jobs_2008_2010 = if (baseline_years_observed == 3L && baseline_farm_years_valid == 3L)
      mean(emp_farm) else NA_real_,
    hired_farm_jobs_2008_2010 = if (baseline_years_observed == 3L && baseline_hired_years_valid == 3L)
      mean(hired_farm_jobs) else NA_real_, .groups = "drop") %>%
  full_join(employment %>% filter(year == 2011L) %>%
              transmute(county_fips, farm_jobs_2011 = if_else(emp_farm > 0, emp_farm, NA_real_)),
            by = "county_fips", relationship = "one-to-one") %>%
  full_join(census %>% filter(year == spec$census_baseline_year) %>%
              transmute(county_fips, census_hired_workers_2007 =
                          if_else(census_hired_workers_total > 0, census_hired_workers_total, NA_real_)),
            by = "county_fips", relationship = "one-to-one")
assert_geo_columns(denominators, "county_fips")
stopifnot(!anyDuplicated(denominators$county_fips), nrow(denominators) > 0)
denominators$all_denominators_available <- apply(
  as.matrix(denominators[, spec$denominators]), 1, function(z) all(is.finite(z) & z > 0))
write_parquet(denominators, path_int("border_data_input_denominators.parquet"))
panel <- left_join(panel, denominators, by = "county_fips", relationship = "many-to-one")
for (denominator in spec$denominators) {
  panel[[paste0("certifications_per_", denominator)]] <-
    panel$nbr_workers_certified_start_year / panel[[denominator]]
}
observed <- is.finite(panel$certifications_per_farm_jobs_2011) & is.finite(panel[[design$outcome]])
stopifnot(max(abs(panel$certifications_per_farm_jobs_2011[observed] - panel[[design$outcome]][observed])) < 1e-12)
panel <- left_join(panel, exposure, by = c("county_fips", "year"), relationship = "one-to-one")
panel$allocated_calendar_capacity_per_2011_farm_job <-
  panel$allocated_certification_position_days / 365.25 / panel$farm_jobs_2011
panel$fls_reference_coverage_share <-
  panel$position_days_with_fls_reference / panel$allocated_certification_position_days
panel$allocated_capacity_weighted_nominal_fls_reference <-
  panel$fls_reference_dollar_position_days / panel$position_days_with_fls_reference
panel$border_data_input_design_id <- spec$design_id
stopifnot(!anyDuplicated(panel[c("county_fips", "year")]))
write_parquet(panel, path_processed("border_data_input_county_year.parquet"))

fits <- audits <- simulations <- list()
for (sample in spec$sample_comparisons) {
  candidate <- panel
  if (sample == "common_all_denominators") {
    candidate <- candidate[candidate$all_denominators_available %in% TRUE, , drop = FALSE]
  }
  for (denominator in spec$denominators) {
    local_design <- design
    local_design$outcome <- paste0("certifications_per_", denominator)
    row <- data.frame(model_id = paste(sample, denominator, sep = "__"), spec_id = spec$geography,
                      weight_spec = "equal_county", control_spec = "none", role = "input_sensitivity")
    prepared <- border_prepare(candidate, memberships, row, local_design)
    fit <- border_fit(prepared)
    jackknife <- border_pair_jackknife(fit)
    fits[[length(fits) + 1L]] <- cbind(data.frame(
      model_id = row$model_id, sample = sample, denominator = denominator,
      effect_unit = paste0("certifications_per_", denominator, "_per_2012_dollar_hour")),
      fit$result, jackknife,
      approximate_pair_cv3j_mde_80 = (qt(.975, fit$result$n_pairs - 1) + qnorm(.8)) *
        jackknife$crv3_jackknife_standard_error)
    audits[[length(audits) + 1L]] <- prepared$audit
    if (sample == "common_all_denominators") {
      pre <- border_prepare(candidate, memberships, row, local_design, period = "placebo",
                            require_treatment = FALSE)
      simulations[[length(simulations) + 1L]] <- list(model_id = row$model_id, policy = prepared, pre = pre)
    }
  }
}
fits <- bind_rows(fits)
common <- fits %>% filter(sample == "common_all_denominators")
stopifnot(n_distinct(common$n_observations) == 1L, n_distinct(common$n_counties) == 1L,
          all(abs(fits$fwl_coefficient_error) < 1e-8))
write_csv(fits, path_tables("border_data_input_estimates.csv"))
write_csv(bind_rows(audits), path_tables("border_data_input_sample_audit.csv"))
power <- run_border_power(simulations, design)
names(power$power)[names(power$power) == "declared_meaningful_effect"] <- "reference_grid_effect"
for (name in c("power", "calibration", "mde", "simulation_diagnostics")) {
  write_csv(power[[name]], path_tables(paste0("border_data_input_", name, ".csv")))
}
stopifnot(all(power$calibration$n_preperiod_years == 3L),
          all(power$power$repetitions == design$redesign$power$repetitions))
paths <- c("code/designs/border_discontinuity/data_input_design.json",
  "code/designs/border_discontinuity/09_compare_data_inputs.R",
  "code/designs/border_discontinuity/redesign.R", "code/designs/border_discontinuity/power.R",
  "data/processed/border_discontinuity_panel.parquet", "data/processed/county_year_panel.parquet",
  "data/intermediate/bea_caemp25n_data_year.parquet",
  "data/intermediate/census_ag_hired_worker_duration_county.parquet",
  "data/intermediate/h2a_worksite_county_year_exposure.parquet")
write_json(list(design_id = spec$design_id,
  interpretation = spec$effect_units_warning,
  files = setNames(lapply(paths, function(p) digest::digest(file = here::here(p), algo = "sha256")), paths)),
  here::here("outputs", "reports", "border_data_input_provenance.json"), auto_unbox = TRUE, pretty = TRUE)
message("Validated denominator identity, common samples, FWL and bounded power metadata.")
