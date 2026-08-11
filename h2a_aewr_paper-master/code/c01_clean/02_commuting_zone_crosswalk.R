# Purpose: Normalize the Penn State 2010 county-to-commuting-zone crosswalk.
# Input: data/raw/geographic_crosswalks/penn/counties10-zqvz0r.csv.
# Outputs: data/intermediate/cz_file_2010.parquet and cz_file_2010_small.parquet.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(dplyr)
library(readr)

cz_file <- read_csv(
  file = path_raw("geographic_crosswalks", "penn", "counties10-zqvz0r.csv"),
  show_col_types = FALSE
)

cz_file <- cz_file %>%
  rename(county_fips = FIPS, cz_id = OUT10)

cz_file <- cz_file %>%
  mutate(
    county_fips = county_fips(county_fips),
    cz_id = cz_id(cz_id),
    CBSA10 = as.character(CBSA10)
  )

write_parquet(cz_file, path_int("cz_file_2010.parquet"))


cz_file_small <- cz_file %>%
  select(county_fips, cz_id)

write_parquet(cz_file_small, path_int("cz_file_2010_small.parquet"))
