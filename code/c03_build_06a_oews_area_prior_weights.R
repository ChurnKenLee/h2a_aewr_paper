# Build OEWS-area prior weights from county FLS proxy weights
rm(list = ls())
if (!exists("path_code", mode = "function")) {
  source(file.path("code", "paths.R"))
}
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)

big_six_occ_codes <- c(
  "45-2041",
  "45-2091",
  "45-2092",
  "45-2093",
  "53-7064",
  "45-2099",
  "79011",
  "79021",
  "79856",
  "79858",
  "98902"
)

first_nonmissing <- function(x) {
  x <- x[!is.na(x) & x != ""]
  if (length(x) == 0) {
    return(NA_character_)
  }
  x[[1]]
}

fls_county_prior <- read_parquet(path_int(
  "fls_county_weight.parquet"
)) %>%
  transmute(
    countyfips,
    year,
    statefips,
    state_abbrev,
    aewr_region_num,
    cz_out10,
    cz_aewr_region_fe,
    fls_county_weight_prior = fls_county_weight
  )

oews_area_crosswalk <- read_parquet(path_int(
  "oews_area_definitions.parquet"
)) %>%
  transmute(
    countyfips = str_c(
      str_pad(as.character(oews_state_fips), 2, pad = "0"),
      str_pad(as.character(oews_county_fips), 3, pad = "0")
    ),
    year,
    oews_area_code,
    oews_area_name
  ) %>%
  filter(!is.na(oews_area_code), oews_area_code != "") %>%
  distinct(countyfips, year, oews_area_code, .keep_all = TRUE)

oews_area_wage <- read_parquet(path_int("oews.parquet")) %>%
  filter(occ_code %in% big_six_occ_codes) %>%
  transmute(
    oews_area_code = area,
    oews_area_name = area_name,
    year,
    oews_tot_emp = as.numeric(tot_emp),
    oews_mean_hourly_wage = as.numeric(h_mean)
  ) %>%
  mutate(
    oews_hourly_wage_bill = oews_tot_emp * oews_mean_hourly_wage,
    oews_wage_emp = if_else(
      !is.na(oews_mean_hourly_wage) &
        !is.na(oews_tot_emp) &
        oews_tot_emp > 0,
      oews_tot_emp,
      0
    )
  ) %>%
  group_by(oews_area_code, year) %>%
  summarise(
    oews_area_name_data = first_nonmissing(oews_area_name),
    oews_area_tot_emp = sum(oews_wage_emp, na.rm = TRUE),
    oews_area_hourly_wage_bill = sum(
      if_else(oews_wage_emp > 0, oews_hourly_wage_bill, 0),
      na.rm = TRUE
    ),
    .groups = "drop"
  ) %>%
  mutate(
    oews_area_mean_hourly_wage = if_else(
      oews_area_tot_emp > 0,
      oews_area_hourly_wage_bill / oews_area_tot_emp,
      NA_real_
    )
  )

fls_county_area_prior <- fls_county_prior %>%
  inner_join(
    oews_area_crosswalk,
    by = c("countyfips", "year")
  ) %>%
  left_join(
    oews_area_wage,
    by = c("oews_area_code", "year")
  ) %>%
  mutate(
    oews_area_name = coalesce(oews_area_name_data, oews_area_name),
    county_area_allocation_base = if_else(
      !is.na(oews_area_tot_emp) & oews_area_tot_emp > 0,
      oews_area_tot_emp,
      0
    )
  ) %>%
  # OEWS areas in New England use townships
  # multiple townships in each county = multiple counties for each OEWS area
  group_by(countyfips, year) %>%
  mutate(
    county_area_count = n(),
    county_area_allocation_total = sum(
      county_area_allocation_base,
      na.rm = TRUE
    ),
    county_area_allocation = if_else(
      county_area_allocation_total > 0,
      county_area_allocation_base / county_area_allocation_total,
      1 / county_area_count
    )
  ) %>%
  ungroup() %>%
  mutate(
    county_area_prior_weight = fls_county_weight_prior * county_area_allocation
  ) %>%
  select(
    countyfips,
    year,
    statefips,
    state_abbrev,
    aewr_region_num,
    cz_out10,
    cz_aewr_region_fe,
    oews_area_code,
    oews_area_name,
    oews_area_mean_hourly_wage,
    oews_area_tot_emp,
    fls_county_weight_prior,
    county_area_count,
    county_area_allocation,
    county_area_prior_weight
  )

fls_oews_area_prior <- fls_county_area_prior %>%
  group_by(
    aewr_region_num,
    year,
    oews_area_code
  ) %>%
  summarise(
    oews_area_name = first_nonmissing(oews_area_name),
    oews_area_mean_hourly_wage = first(oews_area_mean_hourly_wage),
    oews_area_tot_emp = first(oews_area_tot_emp),
    oews_area_prior_weight_all = sum(
      county_area_prior_weight,
      na.rm = TRUE
    ),
    oews_area_county_count = n_distinct(countyfips),
    .groups = "drop"
  ) %>%
  mutate(
    oews_wage_observed = !is.na(oews_area_mean_hourly_wage) &
      oews_area_mean_hourly_wage > 0
  ) %>%
  group_by(aewr_region_num, year) %>%
  mutate(
    oews_observed_prior_mass = sum(
      if_else(oews_wage_observed, oews_area_prior_weight_all, 0),
      na.rm = TRUE
    ),
    oews_area_prior_weight = if_else(
      oews_wage_observed & oews_observed_prior_mass > 0,
      oews_area_prior_weight_all / oews_observed_prior_mass,
      NA_real_
    )
  ) %>%
  ungroup()

write_parquet(
  fls_county_area_prior,
  path_int("fls_county_oews_area_prior_weight.parquet")
)

write_parquet(
  fls_oews_area_prior,
  path_int("fls_oews_area_prior_weight.parquet")
)
