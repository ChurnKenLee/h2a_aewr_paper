# Purpose: Construct the annual farm-product PPI rebased to 2012.
# Input: data/raw/fred/WPU01.csv.
# Output: data/intermediate/ppi_2012.parquet.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
library(arrow)
library(dplyr)
library(readr)

ppi_data <- read_csv(
  file = path_raw("fred", "WPU01.csv")
)

# average by year

ppi_data <- ppi_data %>%
  mutate(year = as.numeric(substr(observation_date, 1, 4)))

ppi_data <- ppi_data %>%
  group_by(year) %>%
  summarise(wpu01 = mean(WPU01))

# change base to 2012

ppi_2012 <- ppi_data$wpu01[ppi_data$year == 2012]

ppi_data <- ppi_data %>%
  mutate(ppi_2012 = wpu01 / ppi_2012)

ppi_data <- ppi_data %>%
  select(year, ppi_2012)

write_parquet(ppi_data, path_int("ppi_2012.parquet"))
