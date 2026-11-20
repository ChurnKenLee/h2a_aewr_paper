# Scratch diagnostic: weights-only first stage when the calibration does or
# does not see the FLS wage (= AEWR) target. Writes only to the output directory (argument 1).
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "helpers.R"))
source(path_code("designs", "panel_iv", "io.R"))
source(path_code("designs", "panel_iv", "calibration.R"))
source(path_code("designs", "panel_iv", "specification.R"))
source(path_code("designs", "panel_iv", "decomposition.R"))
suppressMessages({
  library(arrow)
  library(dplyr)
  library(fixest)
  library(readr)
})
out_dir <- commandArgs(trailingOnly = TRUE)[1]

specs <- tibble::tribble(
  ~spec, ~families, ~rho,
  "wage_only", list("annual_wage"), 0.10,
  "wage_only", list("annual_wage"), 1,
  "wage_only", list("annual_wage"), 10,
  "wage_seas_comp", list(c("annual_wage", "seasonal", "composition")), 0.10,
  "wage_seas_comp", list(c("annual_wage", "seasonal", "composition")), 1,
  "seas_comp", list(c("seasonal", "composition")), 0.10,
  "seas_comp", list(c("seasonal", "composition")), 1,
  "seas_comp", list(c("seasonal", "composition")), 10,
  "seas_only", list("seasonal"), 0.10,
  "seas_only", list("seasonal"), 1,
  "seas_only", list("seasonal"), 10
) %>%
  mutate(label = paste0(spec, "_rho", sub(".", "p", as.character(rho), fixed = TRUE)))

# Calibration -----------------------------------------------------------------
years <- IV_SOURCE_YEARS
annual <- iv_annual_targets(iv_read(path_int("fls_region.parquet")), years)
quarterly <- iv_paired_targets(
  iv_read(path_int("fls_region_quarterly_workers.parquet")),
  iv_read(path_int("fls_region_quarterly_wages.parquet")),
  years
)
features <- iv_build_features(
  iv_read(path_int("panel_iv_fls_frame.parquet")),
  iv_read(path_processed("county_year_panel.parquet")),
  iv_read(path_int("qcew_county_ag_quarterly_employment.parquet")),
  annual, quarterly, years, NULL
)
weights <- list()
calibration <- list()
for (cell in features$cells) {
  supported <- cell$prior > 0
  normalized <- cell$prior[supported] / sum(cell$prior[supported])
  for (s in seq_len(nrow(specs))) {
    selected <- iv_spec_design(cell, unlist(specs$families[[s]]))
    fit <- iv_solve_entropy(
      normalized, selected$design[supported, , drop = FALSE], selected$target,
      rho = specs$rho[[s]]
    )
    full <- numeric(length(cell$prior))
    full[supported] <- fit$weights[1, ]
    weights[[length(weights) + 1L]] <- data.frame(
      aewr_region_id = cell$region, source_year = as.integer(cell$year),
      county_fips = cell$county_codes, label = specs$label[[s]], weight = full
    )
    calibration[[length(calibration) + 1L]] <- data.frame(
      aewr_region_id = cell$region, source_year = as.integer(cell$year),
      label = specs$label[[s]], success = fit$success[1], status = fit$status[1],
      active = ncol(selected$design), kl = fit$kl_divergence[1],
      tv_shift = sum(abs(full - cell$prior / sum(cell$prior))) / 2,
      share_closed = 1 - fit$calibrated_imbalance_norm[1] /
        max(fit$input_imbalance_norm[1], 1e-12)
    )
  }
}
weights <- bind_rows(weights)
calibration <- bind_rows(calibration)
write_csv(calibration, file.path(out_dir, "nonwage_calibration.csv"))

# Weight components on the stage-10 balanced donor set -----------------------
group_columns <- c("aewr_region_id", "target_cluster", "label")
source_years <- seq.int(
  DISSIMILARITY_IV_POLICY_START_YEAR - 1L, DISSIMILARITY_IV_POLICY_END_YEAR - 1L
)
donor_frame <- read_parquet(path_int("panel_iv_county_donor_frame.parquet")) %>%
  filter(is_primary, source_year %in% source_years) %>%
  transmute(
    aewr_region_id, target_cluster, county_fips, source_year = as.integer(source_year),
    donor_cluster, donor_eligible, donor_real_hourly_wage
  )
ppi <- read_parquet(path_int("ppi_2012.parquet")) %>%
  transmute(source_year = as.integer(year), ppi = as.numeric(ppi_2012))
base <- read_parquet(
  path_processed("county_year_panel.parquet"),
  col_select = c("county_fips", "year", "oews_big_six_mean_hourly_wage", "oews_wage_observed")
) %>%
  mutate(source_year = as.integer(year)) %>%
  filter(source_year %in% IV_DECOMPOSITION_BASE_WAGE_YEARS)
base <- bind_cols(
  base,
  select_panel_iv_donor_wage(base$oews_big_six_mean_hourly_wage, base$oews_wage_observed)
) %>%
  left_join(ppi, by = "source_year") %>%
  mutate(real = if_else(is.finite(donor_nominal_hourly_wage) & ppi > 0,
    donor_nominal_hourly_wage / ppi, NA_real_)) %>%
  group_by(county_fips) %>%
  summarise(base_real_hourly_wage = finite_mean(real), .groups = "drop")
balanced <- donor_frame %>%
  left_join(base, by = "county_fips") %>%
  group_by(aewr_region_id, target_cluster, county_fips) %>%
  filter(
    n_distinct(source_year[donor_eligible]) == length(source_years),
    is.finite(first(base_real_hourly_wage))
  ) %>%
  ungroup()
supported <- balanced %>%
  group_by(aewr_region_id, target_cluster, source_year) %>%
  summarise(clusters = n_distinct(donor_cluster), .groups = "drop") %>%
  group_by(aewr_region_id, target_cluster) %>%
  filter(all(clusters == DISSIMILARITY_IV_MIN_OBSERVED_DONOR_CLUSTERS),
    n_distinct(source_year) == length(source_years)) %>%
  distinct(aewr_region_id, target_cluster)
components <- balanced %>%
  semi_join(supported, by = c("aewr_region_id", "target_cluster")) %>%
  inner_join(weights, by = c("aewr_region_id", "source_year", "county_fips"),
    relationship = "many-to-many") %>%
  group_by(aewr_region_id, target_cluster, label, source_year) %>%
  mutate(decomposition_weight = weight / sum(weight)) %>%
  ungroup() %>%
  select(all_of(c(group_columns, "county_fips", "source_year", "decomposition_weight",
    "donor_real_hourly_wage", "base_real_hourly_wage"))) %>%
  iv_kitagawa_components(group_columns)

# Reproduction check against retained stage-10 components ---------------------
retained <- read_parquet(path_int("panel_iv_instrument_decomposition_cluster_year.parquet")) %>%
  filter(decomposition_available) %>%
  transmute(aewr_region_id, target_cluster, source_year = as.integer(source_year),
    label = if_else(is_primary, "wage_seas_comp_rho0p1", "wage_only_rho0p1"),
    z_weight_retained = z_weight)
check <- inner_join(retained, components, by = c("aewr_region_id", "target_cluster",
  "source_year", "label"))
message("Reproduction: rows ", nrow(check), "; max |z_weight diff| ",
  signif(max(abs(check$z_weight - check$z_weight_retained)), 3))

# AEWR gap (the wage target minus the prior OEWS mean, standardized) ---------
gap <- read_parquet(path_int("panel_iv_fls_county_moment_diagnostics.parquet")) %>%
  filter(moment_family == "annual_wage", !is_primary) %>%
  transmute(aewr_region_id, year = as.integer(source_year) + 1L,
    aewr_gap = standardized_target - prior_standardized_moment)

# First stages -----------------------------------------------------------------
spec_json <- panel_iv_read_specification(path_code("designs", "panel_iv", "estimation_spec.json"))
endogenous <- spec_json$endogenous
fe <- paste(spec_json$fixed_effects, collapse = " + ")
controls <- paste(spec_json$controls, collapse = " + ")
cluster <- stats::as.formula(paste("~", spec_json$primary_inference_cluster))
panel <- panel_iv_prepare_outcomes(
  read_parquet(path_processed("panel_iv_county_year.parquet")), spec_json
) %>%
  left_join(gap, by = c("aewr_region_id", "year"))
wide <- components %>%
  transmute(aewr_region_id, target_cluster, year = as.integer(source_year) + 1L, label,
    z_weight, z_balanced) %>%
  tidyr::pivot_wider(names_from = label, values_from = c(z_weight, z_balanced))
data <- panel %>%
  left_join(wide, by = c("aewr_region_id", "target_cluster", "year"))
z_columns <- grep("^z_(weight|balanced)_", names(data), value = TRUE)
sample <- data %>%
  filter(year >= DISSIMILARITY_IV_POLICY_START_YEAR,
    year <= DISSIMILARITY_IV_BEA_CONTROL_END_YEAR,
    if_all(all_of(c(endogenous, DISSIMILARITY_IV_CONTROL_COLUMNS, z_columns, "aewr_gap")),
      ~ is.finite(.x)),
    !is.na(.data[[spec_json$primary_inference_cluster]]))
message("First-stage sample: ", nrow(sample), " county-years")

partial <- function(column) {
  as.numeric(resid(feols(as.formula(paste(column, "~", controls, "|", fe)),
    data = sample, warn = FALSE, notes = FALSE)))
}
gap_resid <- partial("aewr_gap")
rows <- list()
for (column in z_columns) {
  model <- feols(as.formula(paste(endogenous, "~", column, "+", controls, "|", fe)),
    data = sample, vcov = cluster, warn = FALSE, notes = FALSE)
  table <- coeftable(model)
  z_resid <- partial(column)
  rows[[column]] <- tibble::tibble(
    instrument = column,
    estimate = table[column, "Estimate"],
    standard_error = table[column, "Std. Error"],
    first_stage_f = table[column, "t value"]^2,
    within_r2 = as.numeric(r2(model, "wr2")),
    residual_sd = sd(z_resid),
    corr_with_aewr_gap = cor(z_resid, gap_resid),
    observations = nobs(model)
  )
}
results <- bind_rows(rows)
write_csv(results, file.path(out_dir, "nonwage_first_stage.csv"))
print(as.data.frame(results), digits = 3)
print(as.data.frame(calibration %>% group_by(label) %>% summarise(
  failures = sum(!success), median_active = median(active),
  median_tv_shift = median(tv_shift), p90_tv_shift = quantile(tv_shift, .9),
  median_kl = median(kl), median_share_closed = median(share_closed))), digits = 3)
