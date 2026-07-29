# Purpose: Create the balanced 2008-2022 county-year backbone.
# Input: data/intermediate/county_adjacency2010.parquet.
# Output: data/intermediate/county_df_year.parquet.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
library(arrow)
library(dplyr)

full_county_set <- read_parquet(path_int("county_adjacency2010.parquet"))

county_df <- full_county_set %>%
  distinct(county_fips, countyname) %>%
  cross_join(data.frame(year = 2008:2022))

write_parquet(county_df, path_int("county_df_year.parquet"))
