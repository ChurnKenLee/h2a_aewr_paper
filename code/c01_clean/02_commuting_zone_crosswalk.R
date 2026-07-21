# Purpose: Normalize the Penn State 2010 county-to-commuting-zone crosswalk.
# Input: data/raw/geographic_crosswalks/penn/counties10-zqvz0r.csv.
# Outputs: data/intermediate/cz_file_2010.parquet and cz_file_2010_small.parquet.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
source(path_code("c00_shared", "fips.R"))
library(arrow)
library(dplyr)
library(readr)

cz_file <- read_csv(
  file = path_raw("geographic_crosswalks", "penn", "counties10-zqvz0r.csv")
)

cz_file <- cz_file %>%
  rename(countyfips = FIPS, cz_out10 = OUT10)

cz_file <- cz_file %>%
  mutate(
    countyfips = county_fips(countyfips),
    cz_out10 = as.character(cz_out10),
    CBSA10 = as.character(CBSA10)
  )

write_parquet(cz_file, path_int("cz_file_2010.parquet"))


cz_file_small <- cz_file %>%
  select(countyfips, cz_out10)

write_parquet(cz_file_small, path_int("cz_file_2010_small.parquet"))
