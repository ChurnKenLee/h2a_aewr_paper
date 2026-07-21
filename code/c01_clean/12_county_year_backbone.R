# Purpose: Create the balanced 2008-2022 county-year backbone.
# Input: data/intermediate/county_adjacency2010.parquet.
# Output: data/intermediate/county_df_year.parquet.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
library(arrow)
library(dplyr)

full_county_set <- read_parquet(path_int("county_adjacency2010.parquet"))

county_list_unique <- unique(select(full_county_set, fipscounty, countyname))


years <- seq(2008, 2022)
county_df <- NULL

for (i in 1:length(years)) {
  temp <- county_list_unique %>%
    mutate(year = years[i])
  county_df <- bind_rows(county_df, temp)
}


county_df %>%
  group_by(year) %>%
  tally()

write_parquet(county_df, path_int("county_df_year.parquet"))
cat("county_df_year:", nrow(county_df), "rows,", ncol(county_df), "cols\n")
