# Purpose: Reshape and harmonize annual BEA county employment measures.
# Inputs: bea_CAEMP25N_trim.parquet, county_adjacency2010.parquet, and bea_fips_xwalk.csv.
# Output: data/intermediate/bea_caemp25n_data_year.parquet.
# Run after: code/a01_sources/08_bea_farm_nonfarm_emp.py.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
source(path_code("c00_shared", "fips.R"))
source(path_code("c00_shared", "bea_county_crosswalk.R"))
library(arrow)
library(dplyr)
library(readr)
library(tidyr)

full_county_set <- read_parquet(path_int("county_adjacency2010.parquet"))
bea_fips_xwalk <- read_csv(
  path_raw("geographic_crosswalks", "phil", "bea_fips_xwalk.csv"),
  show_col_types = FALSE
) %>%
  prepare_bea_county_crosswalk(full_county_set = full_county_set)

bea_caemp25n_data <- read_parquet(path_int("bea_CAEMP25N_trim.parquet"))
# save lines: 10 50 70 80 90

bea_caemp25n_data <- bea_caemp25n_data %>%
  filter(
    LineCode == 10 |
      LineCode == 50 |
      LineCode == 70 |
      LineCode == 80 |
      LineCode == 90
  )

bea_caemp25n_data <- bea_caemp25n_data %>%
  mutate(
    countyfips = county_fips(GeoFIPS), # remove quotes
    category = ifelse(
      LineCode == 10,
      "emp_tot",
      ifelse(
        LineCode == 50,
        "emp_farm_propr", # farm proprietors
        ifelse(
          LineCode == 70,
          "emp_farm",
          ifelse(
            LineCode == 80,
            "emp_nonfarm",
            ifelse(LineCode == 90, "emp_privatenonfarm", NA)
          )
        )
      )
    )
  )

bea_caemp25n_data <- bea_caemp25n_data %>%
  select(9:32)

bea_caemp25n_data <- bea_caemp25n_data %>% # wow, that was easy
  pivot_longer(
    cols = starts_with("y"),
    names_to = "year",
    names_prefix = "y",
    values_to = "temp",
    values_drop_na = F
  )

bea_caemp25n_data <- bea_caemp25n_data %>%
  mutate(emp = as.numeric(temp)) %>%
  select(-temp)

# put into year - county rows, so, pivot again

bea_caemp25n_data <- bea_caemp25n_data %>% # wow, that was easy
  pivot_wider(names_from = "category", values_from = "emp")

bea_caemp25n_data <- bea_caemp25n_data %>%
  filter(year > 2007 & year <= 2022)

bea_caemp25n_data <- apply_bea_county_crosswalk(
  bea_caemp25n_data,
  bea_fips_xwalk
)

# SD Oglala Lakota to Shannon
bea_caemp25n_data <- bea_caemp25n_data %>%
  mutate(
    countyfips = case_when(
      countyfips == "46102" ~ "46113",
      .default = countyfips
    )
  )


write_parquet(
  bea_caemp25n_data,
  path_int("bea_caemp25n_data_year.parquet")
)
cat(
  "bea_caemp25n_data_year:",
  nrow(bea_caemp25n_data),
  "rows,",
  ncol(bea_caemp25n_data),
  "cols\n"
)
