# Purpose: Build state-year and AEWR-region-year wage panels in real and nominal terms.
# Inputs: aewr.parquet, ppi_2012.parquet, fips_codes.csv, and aewr_regions.csv.
# Outputs: aewr_data_year.parquet and processed/aewr_data_full.parquet.
# Run after: 03_producer_price_index.R and the a-stage AEWR extraction.

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
library(tidyr)

fips_codes <- read_csv(
  path_raw("geographic_crosswalks", "phil", "fips_codes.csv"),
  show_col_types = FALSE
) %>%
  mutate(fips = state_fips(fips))
aewr_regions <- read_csv(
  path_raw("geographic_crosswalks", "phil", "aewr_regions.csv"),
  show_col_types = FALSE
)
ppi_data <- read_parquet(path_int("ppi_2012.parquet"))

aewr_data <- read_parquet(
  path_int("aewr.parquet"),
  stringsAsFactors = F
)

aewr_data <- merge(
  x = aewr_data,
  y = ppi_data,
  by = "year",
  all.x = T,
  all.y = F
)

aewr_data <- aewr_data %>%
  mutate(
    aewr = as.numeric(aewr),
    state_fips_code = state_fips(state_fips_code)
  ) %>%
  mutate(aewr_ppi = aewr / ppi_2012)

aewr_data <- aewr_data %>%
  arrange(state_fips_code, year) %>%
  group_by(state_fips_code) %>%
  mutate(
    aewr_ppi_l1 = lag(aewr_ppi, n = 1, order_by = state_fips_code),
    aewr_l1 = lag(aewr, n = 1, order_by = state_fips_code),
    aewr_ppi_l2 = lag(aewr_ppi, n = 2, order_by = state_fips_code),
    aewr_l2 = lag(aewr, n = 2, order_by = state_fips_code)
  )

aewr_data <- aewr_data %>%
  filter(year > 2007 & year <= 2022)

aewr_data <- merge(
  x = aewr_data,
  y = fips_codes,
  by.x = "state_fips_code",
  by.y = "fips",
  all.x = T,
  all.y = F
)

aewr_data <- aewr_data %>%
  select(
    aewr,
    aewr_ppi,
    aewr_l1,
    aewr_ppi_l1,
    aewr_l2,
    aewr_ppi_l2,
    year,
    state_abbrev
  )


write_parquet(aewr_data, path_int("aewr_data_year.parquet"))
cat("aewr_data_year:", nrow(aewr_data), "rows,", ncol(aewr_data), "cols\n")

# full for TS

aewr_data <- read_parquet(
  path_int("aewr.parquet"),
  stringsAsFactors = F
) %>%
  mutate(aewr = as.numeric(aewr)) %>%
  mutate(year = as.numeric(year)) %>%
  mutate(state_fips_code = state_fips(state_fips_code))

aewr_data <- aewr_data %>%
  arrange(state_fips_code, year)

aewr_data %>% group_by(year) %>% tally()

aewr_data <- merge(
  x = aewr_data,
  y = ppi_data,
  by = "year",
  all.x = T,
  all.y = F
)

aewr_data <- aewr_data %>%
  mutate(aewr_ppi = aewr / ppi_2012)

aewr_data <- merge(
  x = aewr_data,
  y = fips_codes,
  by.x = "state_fips_code",
  by.y = "fips",
  all.x = T,
  all.y = F
)


# collapse by regions

aewr_data <- merge(
  x = aewr_data,
  y = aewr_regions,
  by = "state_abbrev",
  all.x = T
)

aewr_data <- aewr_data %>%
  group_by(aewr_region_num, year) %>%
  summarise(aewr = mean(aewr, na.rm = T), aewr_ppi = mean(aewr_ppi, na.rm = T))


# make TS variables #
aewr_data <- aewr_data %>%
  arrange(aewr_region_num, year) %>%
  group_by(aewr_region_num) %>%
  mutate(
    aewr_ppi_l1 = lag(aewr_ppi, n = 1, order_by = aewr_region_num),
    aewr_l1 = lag(aewr, n = 1, order_by = aewr_region_num),
    aewr_ppi_l2 = lag(aewr_ppi, n = 2, order_by = aewr_region_num),
    aewr_l2 = lag(aewr, n = 2, order_by = aewr_region_num)
  )

aewr_data <- aewr_data %>%
  arrange(aewr_region_num, year) %>%
  mutate(
    ch_aewr = aewr - aewr_l1,
    ch_aewr_ppi = aewr_ppi - aewr_ppi_l1,
    pch_aewr = (aewr - aewr_l1) / aewr_l1,
    pch_aewr_ppi = (aewr_ppi - aewr_ppi_l1) / aewr_ppi_l1
  )


aewr_data <- aewr_data %>%
  filter(!is.na(aewr_region_num))

# leav one out averages

loo_averages <- NULL

for (i in 1:17) {
  temp <- aewr_data %>%
    filter(aewr_region_num != i) %>% # leave one out, so not the one we want
    group_by(year) %>%
    summarise(
      loo_aewr = mean(aewr, na.rm = T),
      loo_aewr_ppi = mean(aewr_ppi, na.rm = T),
      aewr_region_num = i
    )
  loo_averages <- bind_rows(loo_averages, temp)
  rm(temp)
}

aewr_data <- merge(
  x = aewr_data,
  y = loo_averages,
  by = c("year", "aewr_region_num"),
  all.x = T
)

aewr_data <- aewr_data %>%
  arrange(year, aewr_region_num) %>%
  mutate(
    aewr_diff = aewr - loo_aewr,
    aewr_ppi_diff = aewr_ppi - loo_aewr_ppi,
    ln_aewr = log(aewr),
    ln_aewr_l1 = log(aewr / aewr_l1)
  )


write_parquet(aewr_data, path_processed("aewr_data_full.parquet"))
cat("aewr_data_full:", nrow(aewr_data), "rows,", ncol(aewr_data), "cols\n")
