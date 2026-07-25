#!/usr/bin/env Rscript

here::i_am("code/paths.R")

source(here::here("code", "c00_shared", "geography.R"))

stopifnot(
  identical(state_fips(c(1, "06", 12)), c("01", "06", "12")),
  identical(county_code(c(1, "013", 101)), c("001", "013", "101")),
  identical(county_fips(c(1001, "06037")), c("01001", "06037")),
  identical(combine_county_fips(c(1, "06"), c(1, 37)), c("01001", "06037")),
  identical(state_from_county_fips(c("01001", 6037)), c("01", "06")),
  identical(county_code_from_county_fips(c("01001", 6037, "46102")),
    c("001", "037", "113")
  ),
  identical(harmonize_county_fips_2010(c("46102", "06037")),
    c("46113", "06037")
  ),
  identical(cz_id(c("00100", 200)), c("100", "200")),
  identical(aewr_region_id(c("01", 17)), c("1", "17")),
  identical(oews_area_code(c("0900001", 33100)), c("0900001", "33100"))
)

valid <- data.frame(
  state_fips = c("01", "06"),
  county_code = c("001", "037"),
  county_fips = c("01001", "06037"),
  cz_id = c("100", "200"),
  aewr_region_id = c("1", "17"),
  oews_area_code = c("0900001", "33100")
)
stopifnot(identical(assert_geo_columns(
  valid,
  names(valid)
), valid))

expect_error <- function(expression) {
  inherits(try(force(expression), silent = TRUE), "try-error")
}

stopifnot(
  expect_error(state_fips("123")),
  expect_error(county_fips("06A37")),
  expect_error(aewr_region_id("18")),
  expect_error(assert_geo_columns(
    transform(valid, county_fips = as.integer(county_fips)),
    "county_fips"
  )),
  expect_error(assert_geo_columns(
    transform(valid, county_fips = c(NA_character_, "06037")),
    "county_fips"
  )),
  expect_error(assert_geo_columns(
    transform(valid, county_fips = c("46102", "06037")),
    "county_fips"
  )),
  expect_error(assert_geo_columns(valid, "missing_geo_field"))
)

cat("geography tests passed\n")
