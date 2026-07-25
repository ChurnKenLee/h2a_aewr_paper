# Shared, explicit analysis transformations.

analysis_sample <- function(county_df) {
  county_df |>
    dplyr::filter(
      any_cropland_2007 == 1,
      county_simple_treatment_groups != "always takers"
    )
}

read_county_map <- function(
    county_zip,
    simplify = TRUE,
    simplify_tolerance = 1000) {
  shape_dir <- tempfile("h2a_counties_")
  dir.create(shape_dir)
  on.exit(unlink(shape_dir, recursive = TRUE), add = TRUE)
  utils::unzip(county_zip, exdir = shape_dir)

  county_shape <- sf::st_read(
    file.path(shape_dir, "tl_2020_us_county.shp"),
    quiet = TRUE
  )

  county_shape <- county_shape |>
    dplyr::mutate(
      state_fips = state_fips(STATEFP),
      county_fips = harmonize_county_fips_2010(
        combine_county_fips(STATEFP, COUNTYFP)
      )
    ) |>
    dplyr::filter(
      as.integer(state_fips) <= 56,
      !state_fips %in% c("02", "15")
    )

  if (simplify) {
    county_shape <- sf::st_simplify(
      county_shape,
      preserveTopology = FALSE,
      dTolerance = simplify_tolerance
    )
  }

  county_shape
}
