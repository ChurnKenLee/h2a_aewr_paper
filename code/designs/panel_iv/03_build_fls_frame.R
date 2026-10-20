# Build the original rolling county prior from existing local source artifacts.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("designs", "panel_iv", "io.R"))
source(path_code("designs", "panel_iv", "calibration.R"))
source(path_code("designs", "panel_iv", "frame.R"))

panel <- iv_read(path_processed("county_year_panel.parquet"))
counties <- unique(panel[c("county_fips", "state_fips", "state_abbrev", "aewr_region_id")])
# Read a Table so physical source-row order survives filtering. A Dataset scan
# may emit batches in parallel completion order, which is unsuitable here.
quickstats <- arrow::read_parquet(path_int("qs_census_economics.parquet"), as_data_frame = FALSE,
  col_select = c("year", "agg_level_desc", "freq_desc", "reference_period_desc", "commodity_desc",
    "domain_desc", "prodn_practice_desc", "short_desc", "state_fips", "county_code", "value")) |>
  dplyr::filter(year %in% IV_BENCHMARK_YEARS) |>
  dplyr::collect()
parse_value <- function(value) suppressWarnings(as.numeric(gsub(",", "", ifelse(value == "(Z)", "0", value))))
state <- quickstats |>
  dplyr::filter(agg_level_desc == "STATE", freq_desc == "ANNUAL", reference_period_desc == "YEAR",
    commodity_desc == "LABOR", domain_desc == "TOTAL", prodn_practice_desc == "ALL PRODUCTION PRACTICES",
    short_desc == "LABOR, HIRED - NUMBER OF WORKERS") |>
  dplyr::transmute(state_fips = normalize_fixed_width_code(as.character(state_fips), 2L, "state"),
    year = as.integer(year), state_census_hired_workers_reported = parse_value(value))
# The recorded selection preserves the former all-domain, unstable-sort floor.
# Validate it against the fingerprinted source rather than relying on R ties.
farms <- quickstats |>
  dplyr::filter(agg_level_desc == "COUNTY", short_desc == "FARM OPERATIONS - NUMBER OF OPERATIONS") |>
  dplyr::transmute(source_county_fips = paste0(normalize_fixed_width_code(as.character(state_fips), 2L, "state"),
    normalize_fixed_width_code(as.character(county_code), 3L, "county")),
    year = as.integer(year), census_eligible_farms = parse_value(value))
selection <- readr::read_csv(path_code("designs", "panel_iv", "census_farm_floor_selection.csv"),
  col_types = readr::cols(source_county_fips = readr::col_character(), year = readr::col_integer(),
    source_row = readr::col_integer(), census_eligible_farms = readr::col_double()))
metadata <- jsonlite::fromJSON(path_code("designs", "panel_iv", "census_farm_floor_source.json"))
farms <- iv_select_farm_floor(farms, selection, metadata, path_int("qs_census_economics.parquet")) |>
  # Preserve the original exact-key matching, including unmatched source codes.
  dplyr::filter(source_county_fips %in% counties$county_fips) |>
  dplyr::rename(county_fips = source_county_fips)
assert_geo_columns(farms, "county_fips")
frame <- iv_build_frame(counties, iv_read(path_int("census_ag_hired_worker_duration_county.parquet")),
  state, farms, panel, iv_read(path_int("qwi_county_ag_quarterly_employment.parquet")),
  iv_read(path_int("bea_caemp25n_data_year.parquet")))
assert_geo_columns(frame, c("county_fips", "state_fips", "aewr_region_id"))
diagnostics <- frame |>
  dplyr::filter((!census_benchmark_reported & !is.na(census_hired_workers_benchmark_filled)) |
    qwi_annual_fallback_used | bea_annual_fallback_used |
    unit_growth_fallback_used | extreme_annual_change | nonnegative_floor_applied)
iv_atomic_parquet(frame, path_int("panel_iv_fls_frame.parquet"))
iv_atomic_parquet(diagnostics, path_int("panel_iv_fls_frame_diagnostics.parquet"))
message("Wrote ", nrow(frame), " frame rows and ", nrow(diagnostics), " diagnostic rows")
