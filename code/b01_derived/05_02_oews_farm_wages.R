# Purpose: Publish county-mapped OEWS agricultural occupation source measures.
# Inputs: oews.parquet and oews_area_definitions.parquet.
# Output: oews_county_area_year_occupation.parquet.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(dplyr)
library(stringr)

# The modern and historical OEWS occupation codes corresponding to the six
# occupations used by DOL to determine the AEWR. The historical releases do
# not publish a predecessor for every modern occupation.
oews_aewr_occupation_codes <- c(
  "45-2041", # Graders and Sorters, Agricultural Products
  "45-2091", # Agricultural Equipment Operators
  "45-2092", # Farmworkers: Crops, Nursery, and Greenhouse
  "45-2093", # Farmworkers: Farm, Ranch, and Aquacultural Animals
  "53-7064", # Packers and Packagers, Hand
  "45-2099", # Agricultural Workers, Other
  "79011", # Historical: graders and sorters
  "79021", # Historical: farm equipment operators
  "79856", # Historical: food and fiber crop farmworkers
  "79858", # Historical: farm and ranch animal workers
  "98902" # Historical: hand packers and packagers
)

parse_oews_numeric <- function(value) {
  value <- str_replace_all(str_trim(as.character(value)), ",", "")
  suppressWarnings(as.numeric(value))
}

oews_source <- read_parquet(path_int("oews.parquet")) %>%
  transmute(
    oews_area_code = oews_area_code(area),
    oews_area_name = str_squish(as.character(area_name)),
    occ_code = str_trim(as.character(occ_code)),
    occ_title = str_squish(as.character(occ_title)),
    year = as.integer(year),
    oews_tot_emp = parse_oews_numeric(tot_emp),
    oews_mean_hourly_wage = parse_oews_numeric(h_mean),
    oews_mean_annual_wage = parse_oews_numeric(a_mean)
  ) %>%
  filter(occ_code %in% oews_aewr_occupation_codes) %>%
  mutate(
    across(
      c(oews_tot_emp, oews_mean_hourly_wage, oews_mean_annual_wage),
      \(value) if_else(is.finite(value), value, NA_real_)
    ),
    oews_employment_published = !is.na(oews_tot_emp),
    oews_hourly_wage_published = !is.na(oews_mean_hourly_wage),
    oews_annual_wage_published = !is.na(oews_mean_annual_wage)
  )

assert_geo_columns(oews_source, "oews_area_code")
if (
  nrow(oews_source) == 0L ||
    anyNA(oews_source[c("year", "occ_code")]) ||
    anyDuplicated(
      oews_source[c("oews_area_code", "year", "occ_code")]
    ) >
      0L
) {
  stop(
    "OEWS source rows must be nonempty and unique by area, year, and occupation.",
    call. = FALSE
  )
}

oews_county_area_map <- read_parquet(
  path_int("oews_area_definitions.parquet")
) %>%
  transmute(
    county_fips = harmonize_county_fips_2010(county_fips),
    state_fips = state_from_county_fips(county_fips),
    county_code = county_code_from_county_fips(county_fips),
    year = as.integer(year),
    oews_area_code = oews_area_code(oews_area_code)
  ) %>%
  distinct()

assert_geo_columns(
  oews_county_area_map,
  c("state_fips", "county_code", "county_fips", "oews_area_code")
)
if (
  nrow(oews_county_area_map) == 0L ||
    anyNA(oews_county_area_map$year) ||
    anyDuplicated(
      oews_county_area_map[c("county_fips", "year", "oews_area_code")]
    ) >
      0L
) {
  stop(
    "OEWS county-area mappings must be nonempty and unique by county, year, and area.",
    call. = FALSE
  )
}

oews_county_area_year_occupation <- oews_county_area_map %>%
  inner_join(
    oews_source,
    by = c("oews_area_code", "year"),
    relationship = "many-to-many"
  ) %>%
  select(
    county_fips,
    state_fips,
    county_code,
    year,
    oews_area_code,
    oews_area_name,
    occ_code,
    occ_title,
    oews_tot_emp,
    oews_mean_hourly_wage,
    oews_mean_annual_wage,
    oews_employment_published,
    oews_hourly_wage_published,
    oews_annual_wage_published
  ) %>%
  arrange(county_fips, year, oews_area_code, occ_code)

oews_descriptions <- c(
  oews_area_code = "Source-defined OEWS reporting-area code; leading zeroes are preserved",
  oews_tot_emp = "OEWS source-published occupation employment in the reporting area",
  oews_mean_hourly_wage = "OEWS source-published mean hourly wage for the area and occupation",
  oews_mean_annual_wage = "OEWS source-published mean annual wage for the area and occupation",
  oews_employment_published = "Whether OEWS publishes numeric employment for the area and occupation",
  oews_hourly_wage_published = "Whether OEWS publishes a numeric hourly mean for the area and occupation",
  oews_annual_wage_published = "Whether OEWS publishes a numeric annual mean for the area and occupation"
)

missing_output_columns <- setdiff(
  c(
    "county_fips",
    "state_fips",
    "county_code",
    "year",
    "oews_area_name",
    "occ_code",
    "occ_title",
    names(oews_descriptions)
  ),
  names(oews_county_area_year_occupation)
)
if (length(missing_output_columns) > 0L) {
  stop(
    "OEWS output is missing required columns: ",
    paste(missing_output_columns, collapse = ", "),
    call. = FALSE
  )
}

for (column in names(oews_descriptions)) {
  attr(
    oews_county_area_year_occupation[[column]],
    "description"
  ) <- oews_descriptions[[column]]
  attr(
    oews_county_area_year_occupation[[column]],
    "level_of_aggregation"
  ) <- "OEWS reporting area"
}

assert_geo_columns(
  oews_county_area_year_occupation,
  c("state_fips", "county_code", "county_fips", "oews_area_code")
)

oews_output_key <- c(
  "county_fips",
  "year",
  "oews_area_code",
  "occ_code"
)
if (
  nrow(oews_county_area_year_occupation) == 0L ||
    anyNA(oews_county_area_year_occupation[oews_output_key]) ||
    anyDuplicated(oews_county_area_year_occupation[oews_output_key]) > 0L
) {
  stop(
    "OEWS output must be nonempty and unique by county, year, area, and occupation.",
    call. = FALSE
  )
}

oews_measure_columns <- c(
  "oews_tot_emp",
  "oews_mean_hourly_wage",
  "oews_mean_annual_wage"
)
if (
  any(
    unlist(oews_county_area_year_occupation[oews_measure_columns]) < 0,
    na.rm = TRUE
  )
) {
  stop("OEWS employment and wage measures must be nonnegative.", call. = FALSE)
}

oews_coverage_contract <- c(
  oews_tot_emp = "oews_employment_published",
  oews_mean_hourly_wage = "oews_hourly_wage_published",
  oews_mean_annual_wage = "oews_annual_wage_published"
)
for (measure in names(oews_coverage_contract)) {
  published <- oews_county_area_year_occupation[[
    oews_coverage_contract[[measure]]
  ]]
  value <- oews_county_area_year_occupation[[measure]]
  if (any(published != !is.na(value))) {
    stop(
      "OEWS publication flag is inconsistent with ",
      measure,
      ".",
      call. = FALSE
    )
  }
}

write_parquet(
  oews_county_area_year_occupation,
  path_int("oews_county_area_year_occupation.parquet")
)
