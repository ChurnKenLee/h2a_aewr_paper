# Purpose: Kitagawa decomposition of the Panel-IV instrument into
# calibrated-weight, donor-wage and interaction components, and the share of
# the instrument's identifying variance that each component carries.
# Inputs: panel_iv_county_donor_frame.parquet, panel_iv_instrument_cluster_year
#   .parquet, panel_iv_county_year.parquet, the shared county panel (2008--2010
#   base wages) and PPI.
# Outputs:
#   data/intermediate/panel_iv_instrument_decomposition_cluster_year.parquet
#   outputs/tables/iv_instrument_decomposition_support.csv
#   outputs/tables/iv_instrument_decomposition_variance.csv
#
# This is a diagnostic. It leaves the instrument and both estimated columns
# unchanged. Calibrated weights are used exactly as recovered, including their
# intended response to OEWS wage features.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "helpers.R"))
source(path_code("designs", "panel_iv", "io.R"))
source(path_code("designs", "panel_iv", "specification.R"))
source(path_code("designs", "panel_iv", "decomposition.R"))
library(arrow)
library(dplyr)
library(fixest)
library(readr)
library(tibble)

source_years <- seq.int(
  DISSIMILARITY_IV_POLICY_START_YEAR - 1L,
  DISSIMILARITY_IV_POLICY_END_YEAR - 1L
)
group_columns <- c("aewr_region_id", "target_cluster")

donor_frame <- read_parquet(path_int("panel_iv_county_donor_frame.parquet")) %>%
  filter(
    instrument_spec_label == DISSIMILARITY_IV_PRIMARY_INSTRUMENT_LABEL,
    source_year %in% source_years
  )
iv_require_columns(
  donor_frame,
  c(
    group_columns, "county_fips", "source_year", "donor_cluster",
    "donor_eligible", "instrument_weight", "calibrated_center_weight",
    "donor_real_hourly_wage"
  ),
  "County donor frame"
)

instruments <- read_parquet(
  path_int("panel_iv_instrument_cluster_year.parquet")
) %>%
  filter(instrument_spec_label == DISSIMILARITY_IV_PRIMARY_INSTRUMENT_LABEL) %>%
  transmute(
    aewr_region_id,
    target_cluster,
    aewr_iv_cluster_id,
    source_year,
    policy_year,
    z_dissimilarity_real,
    instrument_available
  )

# Predetermined donor base wage: mean 2008--2010 real OEWS hourly proxy with
# the same donor-wage selection and source-year PPI deflation as stage 05.
ppi <- read_parquet(path_int("ppi_2012.parquet")) %>%
  transmute(source_year = as.integer(year), source_year_ppi_2012 = as.numeric(ppi_2012))

base_wages <- iv_decomposition_base_wages(
  read_parquet(
    path_processed("county_year_panel.parquet"),
    col_select = c(
      "county_fips", "year", "oews_big_six_mean_hourly_wage", "oews_wage_observed"
    )
  ),
  ppi
)
assert_geo_columns(base_wages, "county_fips", allow_na = character())

# Balanced donor set: eligible in every source year and with a base wage.
# Weights are the calibrated weights renormalized over that set, and both
# selected donor clusters must contribute in every year.
balanced_donors <- donor_frame %>%
  left_join(base_wages, by = "county_fips", relationship = "many-to-one") %>%
  mutate(eligible = donor_eligible, weight = calibrated_center_weight) %>%
  iv_balanced_components(
    group_columns,
    source_years,
    DISSIMILARITY_IV_MIN_OBSERVED_DONOR_CLUSTERS
  )
supported_groups <- balanced_donors$supported_groups
components <- balanced_donors$components

support <- balanced_donors$balanced %>%
  group_by(across(all_of(c(group_columns, "source_year")))) %>%
  summarise(
    balanced_eligible_mass_share = sum(instrument_weight),
    balanced_donor_counties = first(balanced_donor_counties),
    .groups = "drop"
  )

decomposition <- instruments %>%
  left_join(supported_groups, by = group_columns, relationship = "many-to-one") %>%
  left_join(
    support,
    by = c(group_columns, "source_year"),
    relationship = "one-to-one"
  ) %>%
  left_join(
    components,
    by = c(group_columns, "source_year"),
    relationship = "one-to-one"
  ) %>%
  mutate(
    decomposition_available =
      coalesce(decomposition_supported, FALSE) & instrument_available,
    across(
      c(z_balanced, z_constant, z_weight, z_wage, z_interaction, identity_residual),
      ~ if_else(decomposition_available, .x, NA_real_)
    )
  ) %>%
  select(-decomposition_supported) %>%
  arrange(aewr_region_id, target_cluster, policy_year)

assert_geo_columns(decomposition, "aewr_region_id", allow_na = character())
if (
  nrow(decomposition) != nrow(instruments) ||
    anyDuplicated(decomposition[c(group_columns, "source_year")]) > 0L ||
    any(decomposition$source_year != decomposition$policy_year - 1L) ||
    !any(decomposition$decomposition_available)
) {
  stop("The decomposition grid violates its keys, timing or support.", call. = FALSE)
}

available <- filter(decomposition, decomposition_available)
within_cluster <- function(value) value - ave(value, available$aewr_iv_cluster_id)
support_summary <- tibble(
  cluster_years = nrow(decomposition),
  instrument_cluster_years = sum(decomposition$instrument_available),
  decomposition_cluster_years = nrow(available),
  decomposition_clusters = n_distinct(available$aewr_iv_cluster_id),
  mean_balanced_eligible_mass_share = mean(available$balanced_eligible_mass_share),
  minimum_balanced_eligible_mass_share = min(available$balanced_eligible_mass_share),
  mean_balanced_donor_counties = mean(available$balanced_donor_counties),
  within_cluster_correlation_balanced_instrument = stats::cor(
    within_cluster(available$z_balanced),
    within_cluster(available$z_dissimilarity_real)
  ),
  maximum_absolute_identity_residual = max(abs(available$identity_residual))
)

# Identifying variance ---------------------------------------------------------
# Residualize the instrument and its components on the preferred column's
# county and year fixed effects and controls, on the preferred first-stage
# sample where the decomposition exists. The three components then add up to
# the balanced instrument's residual, so their variances and covariances add up
# to its variance.

estimation_spec <- panel_iv_read_specification(
  path_code("designs", "panel_iv", "estimation_spec.json")
)
endogenous <- estimation_spec$endogenous
instrument <- estimation_spec$instrument
pieces <- c("z_weight", "z_wage", "z_interaction")

panel <- read_parquet(
  path_processed("panel_iv_county_year.parquet"),
  col_select = all_of(unique(c(
    "county_fips", "year", "aewr_region_id", "target_cluster",
    "aewr_iv_cluster_id", endogenous, instrument,
    DISSIMILARITY_IV_CONTROL_COLUMNS
  )))
)
sample <- panel %>%
  mutate(year = as.integer(year)) %>%
  left_join(
    decomposition %>%
      transmute(
        aewr_region_id, target_cluster, year = policy_year,
        z_dissimilarity_real, z_balanced, z_weight, z_wage, z_interaction
      ),
    by = c("aewr_region_id", "target_cluster", "year"),
    relationship = "many-to-one"
  ) %>%
  filter(
    year >= estimation_spec$policy_years[[1]],
    year <= estimation_spec$policy_years[[2]],
    if_all(
      all_of(c(
        endogenous, instrument, DISSIMILARITY_IV_CONTROL_COLUMNS,
        "z_balanced", pieces
      )),
      ~ !is.na(.x) & is.finite(.x)
    ),
    !is.na(aewr_iv_cluster_id)
  )
if (!isTRUE(all.equal(sample$z_dissimilarity_real, sample[[instrument]]))) {
  stop("The decomposition does not match the panel instrument.", call. = FALSE)
}

partial_out <- function(column) {
  model <- feols(
    as.formula(paste(
      column, "~", paste(estimation_spec$controls, collapse = " + "),
      "|", paste(estimation_spec$fixed_effects, collapse = " + ")
    )),
    data = sample, warn = FALSE, notes = FALSE
  )
  if (nobs(model) != nrow(sample)) {
    stop("Partialling dropped observations for ", column, call. = FALSE)
  }
  as.numeric(resid(model))
}
residuals <- sapply(c(instrument, "z_balanced", pieces), partial_out)
covariance <- crossprod(residuals) / nrow(residuals)
balanced_variance <- covariance["z_balanced", "z_balanced"]
correlation <- function(left, right) {
  covariance[left, right] / sqrt(covariance[left, left] * covariance[right, right])
}

pairs <- expand.grid(left = pieces, right = pieces, stringsAsFactors = FALSE)
pairs <- pairs[match(pairs$left, pieces) <= match(pairs$right, pieces), ]
variance_terms <- tibble(
  left = pairs$left,
  right = pairs$right,
  moment = if_else(left == right, "variance", "twice_covariance"),
  value = unname(mapply(function(l, r) covariance[l, r], left, right)) *
    if_else(left == right, 1, 2),
  correlation = unname(mapply(correlation, left, right))
)
if (abs(sum(variance_terms$value) - balanced_variance) > 1e-8 * max(1, balanced_variance)) {
  stop("Residualized components do not reproduce the balanced variance.", call. = FALSE)
}
# Each component's covariance with the balanced instrument; these sum to its
# variance, so the shares sum to one.
covariance_terms <- tibble(
  left = pieces,
  right = "z_balanced",
  moment = "covariance_with_balanced",
  value = unname(covariance[pieces, "z_balanced"]),
  correlation = unname(sapply(pieces, correlation, right = "z_balanced"))
)
instrument_term <- tibble(
  left = instrument,
  right = "z_balanced",
  moment = "instrument_variance",
  value = covariance[instrument, instrument],
  correlation = correlation(instrument, "z_balanced")
)
variance_table <- bind_rows(variance_terms, covariance_terms, instrument_term) %>%
  mutate(
    share_of_balanced_variance = value / balanced_variance,
    observations = nrow(sample),
    counties = n_distinct(sample$county_fips),
    inference_clusters = n_distinct(sample$aewr_iv_cluster_id)
  ) %>%
  relocate(share_of_balanced_variance, .after = value)

iv_atomic_parquet(
  decomposition,
  path_int("panel_iv_instrument_decomposition_cluster_year.parquet")
)
dir.create(path_tables(), recursive = TRUE, showWarnings = FALSE)
write_csv(support_summary, path_tables("iv_instrument_decomposition_support.csv"))
write_csv(variance_table, path_tables("iv_instrument_decomposition_variance.csv"))
print(support_summary, width = Inf)
print(variance_table, width = Inf)
