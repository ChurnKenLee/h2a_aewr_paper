# Purpose: Aggregate QCEW employment and nominal wage bills to county-years.
# Inputs: data/intermediate/qcew.parquet and county_adjacency2010.parquet.
# Output: data/intermediate/qcew_county_year.parquet.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(dplyr)
library(stringr)
library(tidyr)

qcew_outcomes <- c("annual_avg_emplvl", "total_annual_wages")
qcew_detailed_ownership_codes <- c("1", "2", "3", "5")

# These source areas cannot be assigned to the project's 2010 county vintage
# without allocation. The obsolete Alaska and Virginia records precede or
# overlap canonical successor records. Connecticut's 2024 planning regions
# are not nested in the eight 2010 counties.
qcew_nonallocatable_area_fips <- c(
  "02201",
  "02232",
  "02280",
  sprintf("09%03d", seq(110L, 190L, by = 10L)),
  "51560"
)

canonical_counties <- read_parquet(
  path_int("county_adjacency2010.parquet")
) %>%
  distinct(county_fips)
assert_geo_columns(canonical_counties, "county_fips")
if (
  nrow(canonical_counties) == 0L ||
    anyDuplicated(canonical_counties$county_fips) > 0L
) {
  stop("The 2010 county universe must be nonempty and unique.", call. = FALSE)
}

qcew_cells <- open_dataset(path_int("qcew.parquet")) %>%
  select(
    area_fips,
    own_code,
    industry_code,
    agglvl_code,
    year,
    disclosure_code,
    all_of(qcew_outcomes)
  ) %>%
  filter(
    (agglvl_code == "70" &
      own_code == "0" &
      industry_code == "10") |
      (agglvl_code == "75" &
        own_code %in% qcew_detailed_ownership_codes &
        industry_code %in% c("111", "112"))
  ) %>%
  collect() %>%
  mutate(
    year = as.integer(year),
    qcew_series = case_when(
      industry_code == "10" ~ "all_sectors",
      industry_code == "111" ~ "crop_sector",
      industry_code == "112" ~ "animal_sector",
      TRUE ~ NA_character_
    ),
    county_fips = case_when(
      area_fips == "02158" ~ "02270",
      area_fips %in% c("02063", "02066") ~ "02261",
      TRUE ~ harmonize_county_fips_2010(area_fips)
    ),
    qcew_geography_priority = if_else(area_fips == county_fips, 0L, 1L),
    qcew_undefined_area = str_sub(area_fips, 3L, 5L) == "999"
  )

unexpected_area_fips <- qcew_cells %>%
  filter(
    !qcew_undefined_area,
    !area_fips %in% qcew_nonallocatable_area_fips
  ) %>%
  anti_join(canonical_counties, by = "county_fips") %>%
  distinct(area_fips, county_fips, year)

if (nrow(unexpected_area_fips) > 0L) {
  examples <- unexpected_area_fips %>%
    arrange(area_fips, year) %>%
    head(10L) %>%
    transmute(
      example = paste0(area_fips, "->", county_fips, " (", year, ")")
    ) %>%
    pull(example) %>%
    paste(collapse = ", ")
  stop(
    "QCEW contains unexpected areas outside the 2010 county universe: ",
    examples,
    call. = FALSE
  )
}

qcew_cells <- qcew_cells %>%
  filter(
    !qcew_undefined_area,
    !area_fips %in% qcew_nonallocatable_area_fips
  ) %>%
  semi_join(canonical_counties, by = "county_fips") %>%
  group_by(county_fips, year) %>%
  filter(qcew_geography_priority == min(qcew_geography_priority)) %>%
  ungroup()

qcew_county_year_long <- qcew_cells %>%
  group_by(county_fips, year, qcew_series) %>%
  summarise(
    disclosed = all(
      !is.na(disclosure_code) & disclosure_code == ""
    ),
    across(
      all_of(qcew_outcomes),
      \(value) {
        if (all(!is.na(disclosure_code) & disclosure_code == "")) {
          sum(as.numeric(value))
        } else {
          NA_real_
        }
      }
    ),
    .groups = "drop"
  )

county_year <- qcew_county_year_long %>%
  pivot_wider(
    id_cols = c("county_fips", "year"),
    names_from = qcew_series,
    names_glue = "qcew_{qcew_series}_{.value}",
    values_from = c(all_of(qcew_outcomes), "disclosed")
  ) %>%
  arrange(county_fips, year)

qcew_descriptions <- c(
  qcew_crop_sector_annual_avg_emplvl = "QCEW all-ownership NAICS 111 crop production average annual employment",
  qcew_crop_sector_total_annual_wages = "QCEW all-ownership NAICS 111 crop production nominal annual wage bill",
  qcew_animal_sector_annual_avg_emplvl = "QCEW all-ownership NAICS 112 animal production and aquaculture average annual employment",
  qcew_animal_sector_total_annual_wages = "QCEW all-ownership NAICS 112 animal production and aquaculture nominal annual wage bill",
  qcew_all_sectors_annual_avg_emplvl = "QCEW all-ownership, all-sector average annual employment",
  qcew_all_sectors_total_annual_wages = "QCEW all-ownership, all-sector nominal annual wage bill",
  qcew_crop_sector_disclosed = "Whether every reported QCEW crop ownership cell is disclosed",
  qcew_animal_sector_disclosed = "Whether every reported QCEW animal ownership cell is disclosed",
  qcew_all_sectors_disclosed = "Whether the QCEW all-ownership, all-sector cell is disclosed"
)

missing_output_columns <- setdiff(
  c("county_fips", "year", names(qcew_descriptions)),
  names(county_year)
)
if (length(missing_output_columns) > 0L) {
  stop(
    "QCEW output is missing required columns: ",
    paste(missing_output_columns, collapse = ", "),
    call. = FALSE
  )
}

for (column in names(qcew_descriptions)) {
  attr(county_year[[column]], "description") <- qcew_descriptions[[column]]
  attr(county_year[[column]], "level_of_aggregation") <- "county"
}

assert_geo_columns(county_year, "county_fips")

if (
  nrow(county_year) == 0L ||
    anyNA(county_year$year) ||
    anyDuplicated(county_year[c("county_fips", "year")]) > 0L
) {
  stop(
    "qcew_county_year must have unique, nonmissing county-year keys.",
    call. = FALSE
  )
}

qcew_measure_columns <- c(
  "qcew_crop_sector_annual_avg_emplvl",
  "qcew_crop_sector_total_annual_wages",
  "qcew_animal_sector_annual_avg_emplvl",
  "qcew_animal_sector_total_annual_wages",
  "qcew_all_sectors_annual_avg_emplvl",
  "qcew_all_sectors_total_annual_wages"
)

if (any(unlist(county_year[qcew_measure_columns]) < 0, na.rm = TRUE)) {
  stop("QCEW employment and wage measures must be nonnegative.", call. = FALSE)
}

for (series in c(
  "qcew_crop_sector",
  "qcew_animal_sector",
  "qcew_all_sectors"
)) {
  disclosed <- county_year[[paste0(series, "_disclosed")]]
  employment <- county_year[[paste0(series, "_annual_avg_emplvl")]]
  wages <- county_year[[paste0(series, "_total_annual_wages")]]
  invalid_disclosed <- !is.na(disclosed) &
    disclosed &
    (is.na(employment) | is.na(wages))
  invalid_suppressed <- !is.na(disclosed) &
    !disclosed &
    (!is.na(employment) | !is.na(wages))

  if (any(invalid_disclosed | invalid_suppressed)) {
    stop(
      "QCEW disclosure flags and published measures are inconsistent for ",
      series,
      ".",
      call. = FALSE
    )
  }
}

write_parquet(county_year, path_int("qcew_county_year.parquet"))
