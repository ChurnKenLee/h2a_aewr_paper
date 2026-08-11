# Purpose: Construct a design-neutral OEWS Big-Six hourly-wage proxy by county-year.
# Inputs: oews_county_area_year_occupation.parquet and oews_area_definitions.parquet.
# Output: oews_county_year.parquet.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(dplyr)
library(stringr)

shared_panel_years <- 2008:2022
oews_big_six_occupation_codes <- c(
  "45-2041",
  "45-2091",
  "45-2092",
  "45-2093",
  "45-2099",
  "53-7064",
  "79011",
  "79021",
  "79856",
  "79858",
  "98902"
)

require_columns <- function(data, required, label) {
  missing_columns <- setdiff(required, names(data))
  if (length(missing_columns) > 0L) {
    stop(
      label,
      " is missing required columns: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }
}

oews_long <- read_parquet(
  path_int("oews_county_area_year_occupation.parquet")
)
require_columns(
  oews_long,
  c(
    "oews_area_code",
    "year",
    "occ_code",
    "oews_tot_emp",
    "oews_mean_hourly_wage",
    "oews_employment_published",
    "oews_hourly_wage_published"
  ),
  "County-mapped OEWS occupation data"
)

# OEWS measures are repeated for every county mapped to a reporting area.
# Recover each source area-year-occupation cell once before weighting.
oews_area_occupation <- oews_long %>%
  filter(year %in% shared_panel_years) %>%
  transmute(
    oews_area_code = oews_area_code(oews_area_code),
    year = as.integer(year),
    occ_code = str_trim(as.character(occ_code)),
    oews_tot_emp = as.numeric(oews_tot_emp),
    oews_mean_hourly_wage = as.numeric(oews_mean_hourly_wage),
    oews_employment_published = as.logical(oews_employment_published),
    oews_hourly_wage_published = as.logical(oews_hourly_wage_published)
  ) %>%
  filter(occ_code %in% oews_big_six_occupation_codes) %>%
  distinct()

assert_geo_columns(oews_area_occupation, "oews_area_code")
if (
  nrow(oews_area_occupation) == 0L ||
    anyNA(oews_area_occupation[c(
      "year",
      "occ_code",
      "oews_employment_published",
      "oews_hourly_wage_published"
    )]) ||
    anyDuplicated(
      oews_area_occupation[c("oews_area_code", "year", "occ_code")]
    ) > 0L
) {
  stop(
    "OEWS source cells must be nonempty and unique by area, year, and occupation.",
    call. = FALSE
  )
}

oews_area_wages <- oews_area_occupation %>%
  mutate(
    usable_wage = oews_employment_published &
      oews_hourly_wage_published &
      is.finite(oews_tot_emp) &
      oews_tot_emp > 0 &
      is.finite(oews_mean_hourly_wage) &
      oews_mean_hourly_wage > 0,
    wage_covered_employment = if_else(
      usable_wage,
      oews_tot_emp,
      0
    ),
    hourly_wage_bill = if_else(
      usable_wage,
      oews_tot_emp * oews_mean_hourly_wage,
      0
    )
  ) %>%
  group_by(oews_area_code, year) %>%
  summarise(
    oews_wage_covered_occupation_count = sum(usable_wage),
    wage_covered_employment = sum(wage_covered_employment),
    hourly_wage_bill = sum(hourly_wage_bill),
    .groups = "drop"
  ) %>%
  mutate(
    oews_area_big_six_mean_hourly_wage = if_else(
      wage_covered_employment > 0,
      hourly_wage_bill / wage_covered_employment,
      NA_real_
    )
  ) %>%
  select(
    oews_area_code,
    year,
    oews_area_big_six_mean_hourly_wage,
    oews_wage_covered_occupation_count
  )

oews_definitions <- read_parquet(
  path_int("oews_area_definitions.parquet")
)
require_columns(
  oews_definitions,
  c(
    "county_fips",
    "year",
    "oews_township_code",
    "oews_area_code"
  ),
  "OEWS area definitions"
)

oews_township_map <- oews_definitions %>%
  filter(year %in% shared_panel_years) %>%
  transmute(
    county_fips = harmonize_county_fips_2010(county_fips),
    year = as.integer(year),
    oews_township_code = str_trim(as.character(oews_township_code)),
    oews_area_code = oews_area_code(oews_area_code)
  ) %>%
  distinct()

assert_geo_columns(
  oews_township_map,
  c("county_fips", "oews_area_code")
)
if (
  nrow(oews_township_map) == 0L ||
    anyNA(oews_township_map[c("year", "oews_township_code")]) ||
    any(oews_township_map$oews_township_code == "")
) {
  stop(
    "OEWS township mappings must have nonmissing county, year, township, and area keys.",
    call. = FALSE
  )
}

township_area_conflicts <- oews_township_map %>%
  count(
    county_fips,
    year,
    oews_township_code,
    name = "mapped_area_count"
  ) %>%
  filter(mapped_area_count != 1L)
if (nrow(township_area_conflicts) > 0L) {
  stop(
    "An OEWS township maps to multiple reporting areas within a county-year.",
    call. = FALSE
  )
}

oews_county_area_shares <- oews_township_map %>%
  group_by(county_fips, year, oews_area_code) %>%
  summarise(
    mapped_township_count = n_distinct(oews_township_code),
    .groups = "drop"
  ) %>%
  group_by(county_fips, year) %>%
  mutate(
    county_mapped_township_count = sum(mapped_township_count),
    county_oews_area_share = mapped_township_count /
      county_mapped_township_count
  ) %>%
  ungroup()

share_contract <- oews_county_area_shares %>%
  group_by(county_fips, year) %>%
  summarise(
    share_sum = sum(county_oews_area_share),
    .groups = "drop"
  )
if (
  any(oews_county_area_shares$county_oews_area_share <= 0) ||
    any(abs(share_contract$share_sum - 1) > 1e-12)
) {
  stop(
    "OEWS county-area township shares must be positive and sum to one.",
    call. = FALSE
  )
}

oews_primary_areas <- oews_county_area_shares %>%
  arrange(
    county_fips,
    year,
    desc(county_oews_area_share),
    oews_area_code
  ) %>%
  group_by(county_fips, year) %>%
  slice_head(n = 1L) %>%
  ungroup() %>%
  transmute(
    county_fips,
    year,
    oews_area_code,
    oews_primary_area_share = county_oews_area_share
  )

oews_county_year <- oews_county_area_shares %>%
  left_join(
    oews_area_wages,
    by = c("oews_area_code", "year"),
    relationship = "many-to-one"
  ) %>%
  mutate(
    area_wage_observed = is.finite(oews_area_big_six_mean_hourly_wage) &
      oews_area_big_six_mean_hourly_wage > 0
  ) %>%
  group_by(county_fips, year) %>%
  summarise(
    oews_mapped_area_count = n_distinct(oews_area_code),
    oews_wage_observed_mapping_share = sum(if_else(
      area_wage_observed,
      county_oews_area_share,
      0
    )),
    township_weighted_wage = sum(if_else(
      area_wage_observed,
      county_oews_area_share * oews_area_big_six_mean_hourly_wage,
      0
    )),
    oews_wage_covered_occupation_count = sum(
      coalesce(oews_wage_covered_occupation_count, 0L)
    ),
    .groups = "drop"
  ) %>%
  mutate(
    oews_wage_observed = oews_wage_observed_mapping_share > 0,
    oews_big_six_mean_hourly_wage = if_else(
      oews_wage_observed,
      township_weighted_wage / oews_wage_observed_mapping_share,
      NA_real_
    )
  ) %>%
  select(-township_weighted_wage) %>%
  left_join(
    oews_primary_areas,
    by = c("county_fips", "year"),
    relationship = "one-to-one"
  ) %>%
  select(
    county_fips,
    year,
    oews_area_code,
    oews_big_six_mean_hourly_wage,
    oews_wage_observed,
    oews_wage_covered_occupation_count,
    oews_mapped_area_count,
    oews_primary_area_share,
    oews_wage_observed_mapping_share
  ) %>%
  arrange(county_fips, year)

oews_descriptions <- c(
  oews_area_code = "Primary OEWS reporting area for the county-year, selected by largest mapped-township share with an area-code tie-break",
  oews_big_six_mean_hourly_wage = "Nominal OEWS Big-Six mean hourly wage: employment-weighted within reporting area and mapped-township-share-weighted across county areas with observed wages",
  oews_wage_observed = "Whether at least one mapped OEWS reporting area has a usable Big-Six hourly wage",
  oews_wage_covered_occupation_count = "Number of mapped OEWS area-occupation cells with positive published employment and hourly wages",
  oews_mapped_area_count = "Number of OEWS reporting areas mapped to the county-year",
  oews_primary_area_share = "Share of mapped county townships assigned to the primary OEWS reporting area",
  oews_wage_observed_mapping_share = "Share of mapped county townships assigned to OEWS areas with usable Big-Six hourly wages"
)

for (column in names(oews_descriptions)) {
  attr(oews_county_year[[column]], "description") <-
    oews_descriptions[[column]]
  attr(oews_county_year[[column]], "level_of_aggregation") <- "county-year"
}

assert_geo_columns(
  oews_county_year,
  c("county_fips", "oews_area_code")
)
if (
  nrow(oews_county_year) == 0L ||
    anyNA(oews_county_year$year) ||
    !setequal(unique(oews_county_year$year), shared_panel_years) ||
    anyDuplicated(oews_county_year[c("county_fips", "year")]) > 0L
) {
  stop(
    "oews_county_year must have unique, nonmissing county-year keys.",
    call. = FALSE
  )
}

invalid_wage_contract <-
  oews_county_year$oews_wage_observed !=
    (
      is.finite(oews_county_year$oews_big_six_mean_hourly_wage) &
        oews_county_year$oews_big_six_mean_hourly_wage > 0
    )
invalid_support_contract <-
  oews_county_year$oews_mapped_area_count < 1L |
    oews_county_year$oews_wage_covered_occupation_count < 0L |
    oews_county_year$oews_wage_observed !=
      (oews_county_year$oews_wage_covered_occupation_count > 0L) |
    oews_county_year$oews_primary_area_share <= 0 |
    oews_county_year$oews_primary_area_share > 1 |
    oews_county_year$oews_wage_observed_mapping_share < 0 |
    oews_county_year$oews_wage_observed_mapping_share > 1 + 1e-12

if (any(invalid_wage_contract | invalid_support_contract)) {
  stop("OEWS county-year wage or support fields are invalid.", call. = FALSE)
}

write_parquet(
  oews_county_year,
  path_int("oews_county_year.parquet")
)
