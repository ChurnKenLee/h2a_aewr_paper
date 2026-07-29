# Pure transformations for harmonizing BEA county identifiers.

prepare_bea_county_crosswalk <- function(crosswalk) {
  crosswalk %>%
    dplyr::mutate(
      realfips = county_fips(realfips),
      beafips = county_fips(beafips)
    ) %>%
    dplyr::filter(county == 1) %>%
    dplyr::select(realfips, beafips)
}

apply_bea_county_crosswalk <- function(data, crosswalk) {
  data %>%
    dplyr::left_join(
      crosswalk,
      by = c("county_fips" = "beafips")
    ) %>%
    dplyr::rename(source_county_fips = county_fips) %>%
    dplyr::mutate(
      county_fips = harmonize_county_fips_2010(
        dplyr::coalesce(realfips, source_county_fips)
      )
    ) %>%
    dplyr::select(-source_county_fips, -realfips)
}
