rm(list = ls())
if (file.exists("paths.R")) {
  source("paths.R")
} else {
  source(file.path("code", "paths.R"))
}
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(readxl)
library(haven)
library(collapse)

acs_ds <- open_dataset(
  path_int("acs_1year_for_wage_quantiles.parquet")
)

#### Calculate hourly wage quantiles ####
acs_ds <- open_dataset(
  path_int("acs_1year_for_wage_quantiles.parquet")
)

acs_ds <- acs_ds %>%
  filter(
    CLASSWKR == 2,
    INCWAGE > 0 & INCWAGE < 999998,
    UHRSWORK > 0,
    YEAR == 2000 | YEAR >= 2005
  )

acs_ds <- acs_ds %>%
  select(
    "YEAR",
    "INCWAGE",
    "STATEFIP",
    "PERWT",
    "AGE",
    "OCCSOC",
    "WKSWORK2",
    "WKSWORK1",
    "UHRSWORK"
  )

# Calculate hourly wage
acs_df <- collect(acs_ds) %>%
  mutate(
    # Convert float STATEFIP to a proper 2-digit string (e.g., 6 -> "06")
    STATEFIP = sprintf("%02d", as.integer(STATEFIP))
  ) %>%
  mutate(
    old_weeks_worked = case_match(
      WKSWORK2,
      1 ~ 7,
      2 ~ 20,
      3 ~ 33,
      4 ~ 44,
      5 ~ 48.5,
      6 ~ 51
    )
  ) %>%
  mutate(
    weeks_worked = case_when(
      !is.na(old_weeks_worked) ~ old_weeks_worked,
      is.na(old_weeks_worked) ~ WKSWORK2,
      .default = NA
    )
  ) %>%
  mutate(hourly_wage = INCWAGE / (weeks_worked * UHRSWORK)) %>%
  select(YEAR, STATEFIP, PERWT, OCCSOC, hourly_wage)

occsoc <- acs_df %>% distinct(OCCSOC) %>% arrange()
