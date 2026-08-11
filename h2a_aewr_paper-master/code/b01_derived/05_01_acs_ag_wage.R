# Purpose: Estimate state-year agricultural wage proxies from ACS microdata.
# Inputs: data/intermediate/acs_1year_for_wages.parquet.
# Outputs: data/intermediate/acs_state_ag_wage.parquet.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(haven)
library(collapse)

aewr_core_occ2010 <- c(6040L, 6050L)
aewr_packer_occ2010 <- 9640L
aewr_packer_indnaics <- "^(111|112|1151|1152)"

required_acs_vars <- c(
  "YEAR",
  "INCWAGE",
  "state_fips",
  "PERWT",
  "AGE",
  "OCCSOC",
  "OCC2010",
  "INDNAICS",
  "WKSWORK2",
  "WKSWORK1",
  "UHRSWORK",
  "CLASSWKR"
)

acs_ds <- open_dataset(
  path_int("acs_1year_for_wages.parquet")
)

missing_acs_vars <- setdiff(required_acs_vars, names(acs_ds))
if (length(missing_acs_vars) > 0) {
  stop(
    "ACS extract is missing required variables: ",
    paste(missing_acs_vars, collapse = ", "),
    ". Add them to code/json/acs_1year_wages_extract_spec.json and rerun ",
    "code/b01_derived/03_01_acs_extract.R."
  )
}

acs_ds <- acs_ds %>%
  filter(
    CLASSWKR == 2,
    INCWAGE > 0 & INCWAGE < 999998,
    UHRSWORK > 0
  )

acs_ds <- acs_ds %>%
  select(all_of(required_acs_vars))

# Calculate hourly wage
acs_df <- collect(acs_ds) %>%
  mutate(
    weeks_worked_wkswork2 = case_when(
      WKSWORK2 == 1 ~ 7,
      WKSWORK2 == 2 ~ 20,
      WKSWORK2 == 3 ~ 33,
      WKSWORK2 == 4 ~ 44,
      WKSWORK2 == 5 ~ 48.5,
      WKSWORK2 == 6 ~ 51,
      .default = NA_real_
    )
  ) %>%
  mutate(
    weeks_worked = case_when(
      !is.na(WKSWORK1) & WKSWORK1 > 0 ~ WKSWORK1,
      !is.na(weeks_worked_wkswork2) ~ weeks_worked_wkswork2,
      .default = NA_real_
    )
  ) %>%
  mutate(
    hourly_wage = INCWAGE / (weeks_worked * UHRSWORK),
    INDNAICS = as.character(INDNAICS),
    aewr_occ_core = OCC2010 %in% aewr_core_occ2010,
    aewr_occ_packer_ag = OCC2010 == aewr_packer_occ2010 &
      str_detect(coalesce(INDNAICS, ""), aewr_packer_indnaics),
    aewr_occ = aewr_occ_core | aewr_occ_packer_ag,
    aewr_occ_component = case_when(
      aewr_occ_core ~ "core_farm_occupations",
      aewr_occ_packer_ag ~ "ag_industry_hand_packers",
      .default = "excluded"
    )
  )

summarise_acs_ag_wage <- function(df, group_vars) {
  df %>%
    filter(
      aewr_occ,
      PERWT > 0,
      !is.na(hourly_wage),
      is.finite(hourly_wage),
      hourly_wage > 0
    ) %>%
    group_by(across(all_of(group_vars))) %>%
    summarise(
      acs_ag_mean_hourly_wage = weighted.mean(
        hourly_wage,
        PERWT,
        na.rm = TRUE
      ),
      acs_ag_workers_perwt = sum(PERWT, na.rm = TRUE),
      .groups = "drop"
    )
}

acs_ag_df <- acs_df %>%
  transmute(
    year = YEAR,
    state_fips,
    PERWT,
    OCCSOC,
    OCC2010,
    INDNAICS,
    hourly_wage,
    aewr_occ,
    aewr_occ_core,
    aewr_occ_packer_ag,
    aewr_occ_component
  )

acs_ag_wage_state <- acs_ag_df %>%
  summarise_acs_ag_wage(c("year", "state_fips")) %>%
  arrange(year, state_fips)

assert_geo_columns(acs_ag_wage_state, "state_fips")
write_parquet(
  acs_ag_wage_state,
  path_int("acs_state_ag_wage.parquet")
)
