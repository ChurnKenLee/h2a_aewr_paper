# Pure transformations for harmonizing BEA county identifiers.

prepare_bea_county_crosswalk <- function(crosswalk, full_county_set) {
  crosswalk <- crosswalk %>%
    mutate(
      realfips = county_fips(realfips),
      beafips = county_fips(beafips)
    )

  assert_geo_columns(full_county_set, "county_fips")
  county_list <- unique(select(full_county_set, county_fips, countyname)) %>%
    mutate(indata = 1)

  crosswalk <- merge(
    x = crosswalk,
    y = county_list,
    by.x = "realfips",
    by.y = "county_fips",
    all.x = TRUE,
    all.y = FALSE
  )

  crosswalk %>%
    filter(county == 1) %>%
    select(realfips, beafips)
}

apply_bea_county_crosswalk <- function(data, crosswalk) {
  if (!"county_fips" %in% names(data)) {
    stop("Missing required geographic column: county_fips", call. = FALSE)
  }
  if (!is.character(data$county_fips)) {
    stop("county_fips must be a character vector.", call. = FALSE)
  }
  if (anyNA(data$county_fips)) {
    stop("county_fips contains missing values.", call. = FALSE)
  }
  normalized_source_fips <- county_fips(data$county_fips)
  if (!identical(normalized_source_fips, data$county_fips)) {
    stop(
      "county_fips must contain canonical five-digit source codes.",
      call. = FALSE
    )
  }

  data <- merge(
    x = data,
    y = crosswalk,
    by.x = "county_fips",
    by.y = "beafips",
    all.x = TRUE,
    all.y = FALSE
  )

  data <- data %>%
    rename(source_county_fips = county_fips) %>%
    mutate(
      county_fips = harmonize_county_fips_2010(
        coalesce(realfips, source_county_fips)
      )
    ) %>%
    select(-source_county_fips, -realfips)

  assert_geo_columns(data, "county_fips")
  data
}
