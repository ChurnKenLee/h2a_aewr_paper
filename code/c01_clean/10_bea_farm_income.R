# Purpose: Reshape, deflate, and harmonize annual BEA county farm-income measures.
# Inputs: bea_CAINC45_trim.parquet, ppi_2012.parquet, county adjacency, and BEA FIPS crosswalk.
# Output: data/intermediate/bea_cainc45_data_year.parquet.
# Run after: 03_producer_price_index.R and
# code/a01_sources/08_bea_farm_nonfarm_emp.py.

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
ppi_data <- read_parquet(path_int("ppi_2012.parquet"))
bea_fips_xwalk <- read_csv(
  path_raw("geographic_crosswalks", "phil", "bea_fips_xwalk.csv"),
  show_col_types = FALSE
) %>%
  prepare_bea_county_crosswalk(full_county_set = full_county_set)

bea_cainc45_data <- read_parquet(path_int("bea_CAINC45_trim.parquet"))
# save lines: 20 60 130 210 270 150

bea_cainc45_data <- bea_cainc45_data %>%
  filter(
    LineCode == 20 |
      LineCode == 60 |
      LineCode == 130 |
      LineCode == 210 |
      LineCode == 270 |
      LineCode == 150
  )

bea_cainc45_data <- bea_cainc45_data %>%
  mutate(
    countyfips = county_fips(GeoFIPS), # remove quotes
    category = ifelse(
      LineCode == 60,
      "farm_cashcrops", # Cash receipts: Crops
      ifelse(
        LineCode == 20,
        "farm_cashanimal", #  Cash receipts: Livestock and products
        ifelse(
          LineCode == 130,
          "farm_govpayments", # Government payments
          ifelse(
            LineCode == 210,
            "farm_laborexpense", # Hired farm labor expenses
            ifelse(
              LineCode == 270,
              "farm_cashandinc", # cash receipts and other income
              ifelse(LineCode == 150, "farm_prodexp", NA)
            )
          )
        )
      )
    )
  ) # Production expenses
bea_cainc45_data <- bea_cainc45_data %>%
  select(43:64) # be careful, this gets to be really a lot of data.


bea_cainc45_data <- bea_cainc45_data %>% # wow, that was easy
  pivot_longer(
    cols = starts_with("y"),
    names_to = "year",
    names_prefix = "y",
    values_to = "temp",
    values_drop_na = T
  )

bea_cainc45_data <- bea_cainc45_data %>%
  mutate(fin = as.numeric(temp)) %>%
  select(-temp)

# put into year - county rows, so, pivot again

bea_cainc45_data <- bea_cainc45_data %>% # wow, that was easy
  pivot_wider(names_from = "category", values_from = "fin")

# real

bea_cainc45_data <- merge(
  bea_cainc45_data,
  ppi_data,
  by = "year",
  all.x = T,
  all.y = F
)

bea_cainc45_data <- bea_cainc45_data %>%
  mutate(
    farm_cashanimal_ppi = farm_cashanimal / ppi_2012,
    farm_cashcrops_ppi = farm_cashcrops / ppi_2012,
    farm_govpayments_ppi = farm_govpayments / ppi_2012,
    farm_prodexp_ppi = farm_prodexp / ppi_2012,
    farm_laborexpense_ppi = farm_laborexpense / ppi_2012,
    farm_cashandinc_ppi = farm_cashandinc / ppi_2012
  )

bea_cainc45_data <- bea_cainc45_data %>%
  filter(year > 2007 & year <= 2022)

bea_cainc45_data <- apply_bea_county_crosswalk(bea_cainc45_data, bea_fips_xwalk)

write_parquet(
  bea_cainc45_data,
  path_int("bea_cainc45_data_year.parquet")
)
cat(
  "bea_cainc45_data_year:",
  nrow(bea_cainc45_data),
  "rows,",
  ncol(bea_cainc45_data),
  "cols\n"
)
