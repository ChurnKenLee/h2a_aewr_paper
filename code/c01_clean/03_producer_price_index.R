# Purpose: Construct the annual farm-product PPI rebased to 2012.
# Input: data/raw/fred/WPU01.csv.
# Output: data/intermediate/ppi_2012.parquet.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
library(arrow)
library(dplyr)
library(readr)

ppi_data <- read_csv(
  file = path_raw("fred", "WPU01.csv"),
  show_col_types = FALSE
)

ppi_data <- ppi_data %>%
  mutate(
    observation_date = as.Date(observation_date),
    year = as.integer(format(observation_date, "%Y")),
    month = as.integer(format(observation_date, "%m"))
  )

if (anyNA(ppi_data$observation_date) ||
    anyDuplicated(ppi_data$observation_date) ||
    any(!is.finite(ppi_data$WPU01) | ppi_data$WPU01 <= 0)) {
  stop("Farm-product PPI needs unique dates and finite positive observations.",
       call. = FALSE)
}

ppi_data <- ppi_data %>%
  group_by(year) %>%
  summarise(
    wpu01 = mean(WPU01),
    complete_year = n() == 12L && setequal(month, 1:12),
    .groups = "drop"
  )

required_years <- union(2012L, SHARED_PANEL_START_YEAR:SHARED_PANEL_END_YEAR)
missing_years <- setdiff(required_years, ppi_data$year[ppi_data$complete_year])
if (length(missing_years)) {
  stop("Farm-product PPI lacks twelve months for required years: ",
       paste(missing_years, collapse = ", "), call. = FALSE)
}
# A partial future year is not an annual deflator.
ppi_data <- ppi_data %>% filter(complete_year)

# change base to 2012

ppi_2012 <- ppi_data$wpu01[ppi_data$year == 2012]

ppi_data <- ppi_data %>%
  mutate(ppi_2012 = wpu01 / ppi_2012)

ppi_data <- ppi_data %>%
  select(year, ppi_2012)

write_parquet(ppi_data, path_int("ppi_2012.parquet"))
