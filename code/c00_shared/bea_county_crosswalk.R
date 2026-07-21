# Pure transformations for harmonizing BEA county identifiers.

prepare_bea_county_crosswalk <- function(crosswalk, full_county_set) {
  crosswalk <- crosswalk %>%
    mutate(
      realfips = county_fips(realfips),
      beafips = county_fips(beafips)
    )

  county_list <- unique(select(full_county_set, fipscounty, countyname)) %>%
    mutate(fipscounty = county_fips(fipscounty)) %>%
    mutate(indata = 1)

  crosswalk <- merge(
    x = crosswalk,
    y = county_list,
    by.x = "realfips",
    by.y = "fipscounty",
    all.x = TRUE,
    all.y = FALSE
  )

  crosswalk %>%
    filter(county == 1) %>%
    select(realfips, beafips)
}

apply_bea_county_crosswalk <- function(data, crosswalk) {
  data <- data %>%
    mutate(countyfips = county_fips(countyfips))

  data <- merge(
    x = data,
    y = crosswalk,
    by.x = "countyfips",
    by.y = "beafips",
    all.x = TRUE,
    all.y = FALSE
  )

  data %>%
    rename(oldfips = countyfips) %>%
    mutate(countyfips = coalesce(realfips, oldfips)) %>%
    select(-oldfips, -realfips)
}
