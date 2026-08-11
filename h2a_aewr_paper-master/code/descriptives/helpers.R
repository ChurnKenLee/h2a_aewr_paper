read_county_map <- function(county_zip, simplify_tolerance = 1000) {
  shape_dir <- tempfile("h2a-counties-")
  dir.create(shape_dir)
  on.exit(unlink(shape_dir, recursive = TRUE), add = TRUE)
  utils::unzip(county_zip, exdir = shape_dir)

  sf::st_read(
    file.path(shape_dir, "tl_2020_us_county.shp"),
    quiet = TRUE
  ) %>%
    dplyr::mutate(
      state_fips = state_fips(STATEFP),
      county_fips = harmonize_county_fips_2010(
        combine_county_fips(STATEFP, COUNTYFP)
      )
    ) %>%
    dplyr::filter(
      as.integer(state_fips) <= 56,
      !state_fips %in% c("02", "15")
    ) %>%
    sf::st_simplify(
      preserveTopology = FALSE,
      dTolerance = simplify_tolerance
    )
}
