# Purpose: Kitagawa decomposition of both retained Panel-IV instruments into
# calibrated-weight, donor-wage and interaction components.
# Inputs: panel_iv_county_donor_frame.parquet, panel_iv_instrument_cluster_year
#   .parquet, the shared county panel (2008--2010 base wages) and PPI.
# Outputs:
#   data/intermediate/panel_iv_instrument_decomposition_cluster_year.parquet
#   outputs/tables/iv_instrument_decomposition_support.csv
#
# This is a diagnostic. It leaves the retained instruments and columns 1--4
# unchanged. Calibrated weights are used exactly as recovered, including their
# intended response to OEWS wage features.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "helpers.R"))
source(path_code("designs", "panel_iv", "io.R"))
source(path_code("designs", "panel_iv", "decomposition.R"))
library(arrow)
library(dplyr)
library(readr)

source_years <- seq.int(
  DISSIMILARITY_IV_POLICY_START_YEAR - 1L,
  DISSIMILARITY_IV_POLICY_END_YEAR - 1L
)
group_columns <- c("aewr_region_id", "target_cluster", "instrument_spec_label")

donor_frame <- read_parquet(path_int("panel_iv_county_donor_frame.parquet")) %>%
  filter(source_year %in% source_years)
iv_require_columns(
  donor_frame,
  c(
    group_columns, "county_fips", "source_year", "donor_cluster",
    "donor_eligible", "instrument_weight", "calibrated_center_weight",
    "donor_real_hourly_wage", "weight_specification", "is_primary"
  ),
  "County donor frame"
)

retained_instruments <- read_parquet(
  path_int("panel_iv_instrument_cluster_year.parquet")
) %>%
  transmute(
    aewr_region_id,
    target_cluster,
    aewr_iv_cluster_id,
    instrument_spec_label,
    weight_specification,
    is_primary,
    source_year,
    policy_year,
    z_dissimilarity_real,
    instrument_available
  )

# Predetermined donor base wage: mean 2008--2010 real OEWS hourly proxy with
# the same donor-wage selection and source-year PPI deflation as stage 05.
ppi <- read_parquet(path_int("ppi_2012.parquet")) %>%
  transmute(source_year = as.integer(year), source_year_ppi_2012 = as.numeric(ppi_2012))

base_wages <- read_parquet(
  path_processed("county_year_panel.parquet"),
  col_select = c(
    "county_fips", "year", "oews_big_six_mean_hourly_wage", "oews_wage_observed"
  )
) %>%
  mutate(source_year = as.integer(year)) %>%
  filter(source_year %in% IV_DECOMPOSITION_BASE_WAGE_YEARS)
base_wages <- bind_cols(
  base_wages,
  select_panel_iv_donor_wage(
    base_wages$oews_big_six_mean_hourly_wage,
    base_wages$oews_wage_observed
  )
) %>%
  left_join(ppi, by = "source_year", relationship = "many-to-one") %>%
  mutate(
    base_year_real_wage = if_else(
      is.finite(donor_nominal_hourly_wage) & is.finite(source_year_ppi_2012) &
        source_year_ppi_2012 > 0,
      donor_nominal_hourly_wage / source_year_ppi_2012,
      NA_real_
    )
  ) %>%
  group_by(county_fips) %>%
  summarise(
    base_real_hourly_wage = finite_mean(base_year_real_wage),
    base_wage_years = sum(is.finite(base_year_real_wage)),
    .groups = "drop"
  )
assert_geo_columns(base_wages, "county_fips", allow_na = character())

# Balanced donor set: eligible in every source year and with a base wage.
donor_status <- donor_frame %>%
  left_join(base_wages, by = "county_fips", relationship = "many-to-one") %>%
  group_by(across(all_of(c(group_columns, "county_fips")))) %>%
  mutate(
    balanced_donor =
      n_distinct(source_year[donor_eligible]) == length(source_years) &
        is.finite(first(base_real_hourly_wage))
  ) %>%
  ungroup()

balanced <- donor_status %>%
  filter(balanced_donor) %>%
  group_by(across(all_of(c(group_columns, "source_year")))) %>%
  mutate(
    balanced_eligible_mass_share = sum(instrument_weight),
    decomposition_weight = calibrated_center_weight / sum(calibrated_center_weight),
    balanced_donor_clusters = n_distinct(donor_cluster),
    balanced_donor_counties = n_distinct(county_fips)
  ) %>%
  ungroup()

# Both selected donor clusters must contribute in every year, mirroring the
# retained instrument's support rule.
supported_groups <- balanced %>%
  group_by(across(all_of(group_columns))) %>%
  summarise(
    decomposition_supported = all(
      balanced_donor_clusters == DISSIMILARITY_IV_MIN_OBSERVED_DONOR_CLUSTERS
    ) & n_distinct(source_year) == length(source_years),
    .groups = "drop"
  )

components <- balanced %>%
  semi_join(
    filter(supported_groups, decomposition_supported),
    by = group_columns
  ) %>%
  select(all_of(c(
    group_columns, "county_fips", "source_year", "decomposition_weight",
    "donor_real_hourly_wage", "base_real_hourly_wage"
  ))) %>%
  iv_kitagawa_components(group_columns)

support <- balanced %>%
  distinct(across(all_of(c(
    group_columns, "source_year", "balanced_eligible_mass_share",
    "balanced_donor_clusters", "balanced_donor_counties"
  ))))

decomposition <- retained_instruments %>%
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
    decomposition_supported = coalesce(decomposition_supported, FALSE),
    balanced_donor_clusters = coalesce(balanced_donor_clusters, 0L),
    balanced_donor_counties = coalesce(balanced_donor_counties, 0L),
    decomposition_available = decomposition_supported & instrument_available,
    across(
      c(starts_with("z_") & !any_of("z_dissimilarity_real"), identity_residual),
      ~ if_else(decomposition_available, .x, NA_real_)
    ),
    base_wage_years = paste(range(IV_DECOMPOSITION_BASE_WAGE_YEARS), collapse = "-"),
    reference_weight_rule = "source_year_mean_of_balanced_weights",
    decomposition_version = IV_DECOMPOSITION_VERSION
  ) %>%
  arrange(aewr_region_id, target_cluster, policy_year, desc(is_primary))

assert_geo_columns(decomposition, "aewr_region_id", allow_na = character())
if (
  nrow(decomposition) != nrow(retained_instruments) ||
    anyDuplicated(decomposition[c(group_columns, "source_year")]) > 0L ||
    any(decomposition$source_year != decomposition$policy_year - 1L) ||
    !any(decomposition$decomposition_available)
) {
  stop("The decomposition grid violates its keys, timing or support.", call. = FALSE)
}

support_summary <- decomposition %>%
  group_by(instrument_spec_label, weight_specification, is_primary) %>%
  summarise(
    cluster_years = n(),
    retained_instrument_cluster_years = sum(instrument_available),
    decomposition_cluster_years = sum(decomposition_available),
    decomposition_clusters = n_distinct(aewr_iv_cluster_id[decomposition_available]),
    mean_balanced_eligible_mass_share = finite_mean(
      balanced_eligible_mass_share[decomposition_available]
    ),
    minimum_balanced_eligible_mass_share = finite_min(
      balanced_eligible_mass_share[decomposition_available]
    ),
    mean_balanced_donor_counties = finite_mean(
      balanced_donor_counties[decomposition_available]
    ),
    within_cluster_correlation_balanced_retained = {
      keep <- decomposition_available
      stats::cor(
        z_balanced[keep] - ave(z_balanced[keep], aewr_iv_cluster_id[keep]),
        z_dissimilarity_real[keep] -
          ave(z_dissimilarity_real[keep], aewr_iv_cluster_id[keep])
      )
    },
    maximum_absolute_identity_residual = max(abs(identity_residual), na.rm = TRUE),
    .groups = "drop"
  )

iv_atomic_parquet(
  decomposition,
  path_int("panel_iv_instrument_decomposition_cluster_year.parquet")
)
dir.create(path_tables(), recursive = TRUE, showWarnings = FALSE)
write_csv(support_summary, path_tables("iv_instrument_decomposition_support.csv"))
print(support_summary, width = Inf)
