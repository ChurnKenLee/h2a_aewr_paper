library(here)
library(arrow)
library(dplyr)
library(tidylog, warn.conflicts = FALSE)
library(ipumsr)
library(haven)
library(readr)

rm(list = ls())

#### Submit extract request and download ####
# acs1_samples <- c()
# for (i in 2000:2024) {
#   acs1_samples <- c(acs1_samples, paste0("us", i, "a"))
# }
# acs5_samples <- c(
#   "us2007c",
#   "us2012e",
#   "us2017c",
#   "us2020c"
# )

# acs1_spec <- define_extract_from_json(
#   here("code", "json", "acs_wage_quantile_extract_spec.json")
# )

# acs5_spec <- define_extract_from_json(
#   here("code", "json", "acs_immigrant_status_imputation_extract_spec.json")
# )

# acs1_submitted_extract <- submit_extract(acs1_spec)
# acs5_submitted_extract <- submit_extract(acs5_spec)

# acs1_extract <- wait_for_extract(acs1_submitted_extract)
# acs1_ddi_path <- download_extract(
#   acs1_extract,
#   download_dir = here("data", "acs")
# )

# acs5_extract <- wait_for_extract(acs5_submitted_extract)
# acs5_ddi_path <- download_extract(
#   acs5_extract,
#   download_dir = here("data", "acs")
# )

#### Load extracts and save as parquets ####
# usa_00035 is ACS 1-year for wage quantiles
acs_data <- read_ipums_micro(
  here("data", "acs", "usa_00035.xml")
)

acs_data <- acs_data %>%
  zap_labels() %>%
  zap_label()

acs_data %>%
  write_parquet(
    here("binaries", "acs_1year_for_wage_quantiles.parquet")
  )

# usa_00034 is ACS 5-year for immigrant status imputation
acs_data <- read_ipums_micro(
  here("data", "acs", "usa_00034.xml")
)

acs_data <- acs_data %>%
  zap_labels() %>%
  zap_label()

acs_data %>%
  write_parquet(here(
    "binaries",
    "acs_5year_for_immigrant_status_imputation.parquet"
  ))
