# Shared, explicit analysis transformations.

analysis_sample <- function(county_df) {
  county_df |>
    dplyr::filter(
      any_cropland_2007 == 1,
      county_simple_treatment_groups != "always takers"
    )
}

read_county_map <- function(county_zip) {
  shape_dir <- tempfile("h2a_counties_")
  dir.create(shape_dir)
  on.exit(unlink(shape_dir, recursive = TRUE), add = TRUE)
  utils::unzip(county_zip, exdir = shape_dir)

  county_shape <- sf::st_read(
    file.path(shape_dir, "tl_2020_us_county.shp"),
    quiet = TRUE
  )

  county_shape |>
    dplyr::mutate(
      statefip = state_fips(STATEFP),
      countyfips = combine_county_fips(STATEFP, COUNTYFP)
    ) |>
    dplyr::filter(
      as.integer(statefip) <= 56,
      !statefip %in% c("02", "15")
    ) |>
    sf::st_simplify(preserveTopology = FALSE, dTolerance = 1000)
}
