# Purpose: Derive county crop-versus-animal employment shares from ACS and QCEW.
# Inputs: ACS five-year microdata, QCEW, and PUMA-county crosswalks.
# Outputs: data/intermediate/acs_qcew.parquet.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)

acs_df <- read_parquet(path_int(
  "acs_5year_for_immigrant_status_imputation.parquet"
)) %>%
  clean_names()
assert_geo_columns(
  distinct(acs_df, state_fips),
  "state_fips"
)

# We only care about NAICS 111 and 112
acs_df <- acs_df %>%
  filter(indnaics %in% c("111", "112")) %>%
  mutate(
    industry = case_when(
      indnaics == "111" ~ "acs_crop_emp",
      indnaics == "112" ~ "acs_animal_emp"
    )
  ) %>%
  select(-indnaics)

# Pad PUMA code for merging with GEOCORR
acs_df <- acs_df %>%
  mutate(puma = str_pad(puma, 5, side = c("left"), pad = "0"))


# Load GEOCORR crosswalk
# Have to skip the 2nd row
geocorr_2000_path <- path_raw(
  "geographic_crosswalks",
  "geocorr",
  "geocorr2014_puma2000_county2010.csv"
)

geocorr_2000_names <- names(read_csv(
  geocorr_2000_path,
  n_max = 0,
  show_col_types = FALSE
))

geocorr_2000_df <- read_csv(
  geocorr_2000_path,
  skip = 2,
  col_names = geocorr_2000_names,
  show_col_types = FALSE
) %>%
  mutate(state_fips = str_pad(state, 2, side = c("left"), pad = "0")) %>%
  mutate(puma = str_pad(puma2k, 5, side = c("left"), pad = "0")) %>%
  select(-c(state, puma2k))

geocorr_2012_path <- path_raw(
  "geographic_crosswalks",
  "geocorr",
  "geocorr2018_puma2010_county2010.csv"
)

geocorr_2012_names <- names(read_csv(
  geocorr_2012_path,
  n_max = 0,
  show_col_types = FALSE
))

geocorr_2012_df <- read_csv(
  geocorr_2012_path,
  skip = 2,
  col_names = geocorr_2012_names,
  show_col_types = FALSE
) %>%
  mutate(state_fips = str_pad(state, 2, side = "left", pad = "0")) %>%
  mutate(puma = str_pad(puma12, 5, side = "left", pad = "0")) %>%
  select(-c(state, puma12))

# Split by survey year, then merge with GEOCORR
acs_puma_2000_df <- acs_df %>%
  filter(multyear < 2012) %>%
  left_join(geocorr_2000_df, by = c("state_fips", "puma"))

acs_puma_2012_df <- acs_df %>%
  filter(multyear > 2011) %>%
  left_join(geocorr_2012_df, by = c("state_fips", "puma"))

acs_puma_allocated <- bind_rows(acs_puma_2000_df, acs_puma_2012_df)
unmatched_pumas <- acs_puma_allocated %>%
  filter(is.na(county) | is.na(afact)) %>%
  distinct(multyear, state_fips, puma)

# Louisiana PUMA 77777 is the special Hurricane Katrina displacement PUMA.
# It has no county allocation and therefore cannot enter a county artifact.
unexpected_unmatched_pumas <- unmatched_pumas %>%
  filter(
    !coalesce(
      multyear < 2012 & state_fips == "22" & puma == "77777",
      FALSE
    )
  )
if (nrow(unexpected_unmatched_pumas) > 0L) {
  stop(
    "Unexpected ACS PUMAs are missing from the PUMA-to-county crosswalk.",
    call. = FALSE
  )
}
if (nrow(unmatched_pumas) > 0L) {
  message(
    "Dropping unallocatable Louisiana Katrina-displacement PUMA 77777."
  )
}

# Combine back into one df and aggregate to county level
acs_agg <- acs_puma_allocated %>%
  filter(!is.na(county), !is.na(afact)) %>%
  mutate(county_perwt = perwt * afact) %>%
  group_by(year, county, industry) %>%
  summarise(n_emp = sum(county_perwt)) %>%
  ungroup()

acs_agg <- acs_agg %>%
  pivot_wider(names_from = industry, values_from = n_emp)

# Fill in missing with 0, these are PUMAs with no obs in NAICS 111 or 112
# acs_agg <- acs_agg %>%
#   mutate_at(c("acs_crop_emp", "acs_animal_emp"), ~replace_na(.,0))

# Read in QCEW data
qcew_df <- read_parquet(path_int("qcew.parquet"))

# Filter to private sector, county-level, NAICS 111 and 112, and not suppressed
qcew_df <- qcew_df %>%
  filter(own_code == "5") %>% # private
  filter(agglvl_code == "75") %>% # county 3-digit NAICS
  filter(industry_code == "111" | industry_code == "112") %>%
  filter(disclosure_code == "")

# Reshape to wide
qcew_df <- qcew_df %>%
  mutate(
    industry = if_else(
      industry_code == "111",
      "qcew_crop_emp",
      "qcew_animal_emp"
    )
  )

qcew_df <- qcew_df %>%
  pivot_wider(
    id_cols = c("area_fips", "year"),
    names_from = "industry",
    values_from = "annual_avg_emplvl"
  )

# Merge QCEW with ACS
# Pad county ACS FIPS code for merging with QCEW
acs_agg <- acs_agg %>%
  mutate(
    state_county_fips_code = str_pad(county, 5, side = c("left"), pad = "0")
  )

qcew_df <- qcew_df %>%
  filter(year %in% c(2007, 2012, 2017))

acs_qcew_df <- acs_agg %>%
  full_join(
    qcew_df,
    by = c("year" = "year", "state_county_fips_code" = "area_fips")
  ) %>%
  select(-county)

# Harmonize variables with other dataset
acs_qcew_df <- acs_qcew_df %>%
  mutate(
    county_fips = harmonize_county_fips_2010(state_county_fips_code),
    state_fips = state_from_county_fips(county_fips),
    county_code = county_code_from_county_fips(county_fips)
  ) %>%
  select(-state_county_fips_code)

# Export
assert_geo_columns(
  acs_qcew_df,
  c("state_fips", "county_code", "county_fips")
)
acs_qcew_df %>%
  write_parquet(path_int("acs_qcew.parquet"))

# # As a test, try comparing aggregates at the PUMA level
# all_content <- readLines(here("Data", "geocorr2000_county_puma.csv"))
# skip_second <- all_content[-2]
# geocorr_2000_df <- read_csv(textConnection(skip_second), header = TRUE, stringsAsFactors = FALSE)
# geocorr_2000_df <- geocorr_2000_df %>%
#   mutate(state_fips = str_pad(state, 2, side = c("left"), pad = "0")) %>%
#   mutate(countyfip = str_pad(county, 5, side = c("left"), pad = "0")) %>%
#   mutate(puma = str_pad(puma5, 5, side = c("left"), pad = "0")) %>%
#   select(-c(state, county, puma5))
#
#
# all_content <- readLines(here("Data", "geocorr2018_county_puma.csv"))
# skip_second <- all_content[-2]
# geocorr_2012_df <- read_csv(textConnection(skip_second), header = TRUE, stringsAsFactors = FALSE)
# geocorr_2012_df <- geocorr_2012_df %>%
#   mutate(state_fips = str_pad(state, 2, side = c("left"), pad = "0")) %>%
#   mutate(countyfip = str_pad(county, 5, side = c("left"), pad = "0")) %>%
#   mutate(puma = str_pad(puma12, 5, side = c("left"), pad = "0")) %>%
#   select(-c(state, county, puma12))
#
# # Read in QCEW data
# qcew_binaries_list <- list.files(here("binaries"), pattern = "qcew_.*\\.parquet$", full.names = TRUE)
# qcew_df <- map(qcew_binaries_list, read_parquet) %>%
#   reduce(bind_rows)
#
# # Filter to private sector, county-level, NAICS 111 and 112, and not suppressed
# qcew_df <- qcew_df %>%
#   filter(own_code == "5") %>% #private
#   filter(agglvl_code == "75") %>% # county 3-digit NAICS
#   filter(industry_code == "111" | industry_code == "112") %>%
#   filter(is.na(disclosure_code))
#
# # Add PUMA code
# qcew_puma_2000_df <- qcew_df %>%
#   filter(year < 2012) %>%
#   left_join(geocorr_2000_df, by = c("area_fips" = "countyfip"))
#
# qcew_puma_2012_df <- qcew_df %>%
#   filter(year > 2011) %>%
#   left_join(geocorr_2012_df, by = c("area_fips" = "countyfip"))
#
# qcew_df <- bind_rows(qcew_puma_2000_df, qcew_puma_2012_df)
#
# # Aggregate at the PUMA level to be consistent with ACS 5-year sample
# qcew_df <- qcew_df %>%
#   mutate(acs_year = if_else(year >= 2005 & year <= 2007, 2007, 0)) %>%
#   mutate(acs_year = if_else(year >= 2008 & year <= 2012, 2012, acs_year)) %>%
#   mutate(acs_year = if_else(year >= 2013 & year <= 2017, 2017, acs_year))
#
# qcew_df <- qcew_df %>%
#   mutate(puma_emp = annual_avg_emplvl*afact) %>%
#   group_by(state_fips, puma, acs_year, industry_code) %>%
#   summarise(n_emp = sum(puma_emp, na.rm = TRUE)) %>%
#   ungroup()
#
# # Reshape to wide
# qcew_df <- qcew_df %>%
#   mutate(industry = if_else(industry_code == "111", "crop_production", "")) %>%
#   mutate(industry = if_else(industry_code == "112", "animal_production", industry))
#
# qcew_df <- qcew_df %>%
#   pivot_wider(id_cols = c("state_fips", "puma", "acs_year"), names_from = "industry", values_from = "n_emp") %>%
#   mutate(qcew_ratio = crop_production/animal_production) %>%
#   filter(acs_year != 0)
#
# # Aggregate ACS to PUMA level
# acs_df <- read_parquet(here("binaries", "acs_5year_2007_2012_2017.parquet")) %>%
#   clean_names()
#
# acs_df <- acs_df %>%
#   filter(indnaics %in% c("111", "112")) %>%
#   mutate(industry = case_when(
#     indnaics == "111" ~ "crop_production",
#     indnaics == "112" ~ "animal_production"
#   )) %>%
#   select(-indnaics)
#
# acs_df <- acs_df %>%
#   mutate(state_fips = str_pad(state_fips, 2, side = c("left"), pad = "0")) %>%
#   mutate(puma = str_pad(puma, 5, side = c("left"), pad = "0"))
#
# acs_df <- acs_df %>%
#   group_by(state_fips, puma, year, industry) %>%
#   summarise(n_emp = sum(perwt))
#
# acs_df <- acs_df %>%
#   pivot_wider(names_from = industry, values_from = n_emp)
#
# acs_df <- acs_df %>%
#   mutate(acs_ratio = crop_production/animal_production)
#
# acs_qcew_df <- acs_df %>%
#   left_join(qcew_df, by = c("state_fips", "puma", "year" = "acs_year")) %>%
#   filter(!is.na(acs_ratio) & !is.na(qcew_ratio))
#
# # Scatterplot of ratios
# acs_qcew_df %>%
#   filter(acs_ratio < 10) %>%
#   filter(qcew_ratio < 10) %>%
#   filter(year == 2007) %>%
#   ggplot(aes(x = qcew_ratio, y = acs_ratio)) +
#   geom_point() +
#   geom_abline(intercept = 0, slope = 1) +
#   facet_wrap(~year) +
#   theme_bw()
#
# acs_qcew_df2 <- acs_qcew_df %>%
#   filter(!is.na(acs_ratio)) %>%
#   filter(!is.na(qcew_ratio)) %>%
#   filter(acs_ratio < 100) %>%
#   filter(qcew_ratio < 100)
#
# summary(lm(acs_qcew_df2$acs_ratio ~ acs_qcew_df2$qcew_ratio))
