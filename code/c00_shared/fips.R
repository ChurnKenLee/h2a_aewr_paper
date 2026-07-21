# Shared FIPS normalization helpers.
#
# These functions are side-effect free. Scripts source this file explicitly
# when they need to normalize state or county identifiers.

clean_fips <- function(x) {
  x <- as.character(x)
  x <- stringr::str_trim(x)
  x <- stringr::str_replace_all(x, '"', "")
  x <- stringr::str_replace(x, "\\.0$", "")
  dplyr::na_if(x, "")
}

pad_fips <- function(x, width) {
  x <- clean_fips(x)
  dplyr::if_else(
    is.na(x),
    NA_character_,
    stringr::str_pad(x, width = width, side = "left", pad = "0")
  )
}

state_fips <- function(x) pad_fips(x, 2)

county_code <- function(x) pad_fips(x, 3)

county_fips <- function(x) pad_fips(x, 5)

combine_county_fips <- function(state, county) {
  stringr::str_c(state_fips(state), county_code(county))
}

state_from_county_fips <- function(county) {
  stringr::str_sub(county_fips(county), 1, 2)
}
