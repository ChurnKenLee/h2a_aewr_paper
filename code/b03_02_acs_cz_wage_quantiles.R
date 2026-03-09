library(here)
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(collapse)
library(haven)

rm(list = ls())

puma_2000_czone_xwalk <- read_dta(here(
  "data",
  "acs",
  "cw_puma2000_czone.dta"
)) %>%
  rename(czone2000 = czone, afactor2000 = afactor)

puma_2010_czone_xwalk <- read_dta(here(
  "data",
  "acs",
  "cw_puma2010_czone.dta"
)) %>%
  rename(czone2010 = czone, afactor2010 = afactor)

#### Calculate hourly wage quantiles ####
acs_ds <-
  open_dataset(
    here("binaries", "acs_1year_for_wage_quantiles.parquet")
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
    "PUMA",
    "PERWT",
    "AGE",
    "WKSWORK2",
    "WKSWORK1",
    "UHRSWORK"
  )

# Calculate hourly wage
acs_df <- collect(acs_ds) %>%
  mutate(
    # Convert float PUMA to a proper 5-digit string (e.g., 100 -> "00100")
    PUMA = sprintf("%05d", as.integer(PUMA)),
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
  select(YEAR, STATEFIP, PUMA, PERWT, hourly_wage)

# Add Census PUMA to CZ crosswalk
# Note switch in PUMA-CZ codes in 2012
acs_df <- acs_df %>%
  mutate(
    puma2000 = as.integer(paste0(STATEFIP, substr(PUMA, 2, nchar(PUMA)))),
    puma2010 = as.integer(paste0(STATEFIP, PUMA)),
  ) %>%
  mutate(
    puma2000 = case_when(
      YEAR < 2012 ~ puma2000,
      .default = 0
    ),
    puma2010 = case_when(
      YEAR >= 2012 ~ puma2010,
      .default = 0
    )
  )

acs_df <- acs_df %>%
  left_join(puma_2000_czone_xwalk, by = c("puma2000")) %>%
  left_join(puma_2010_czone_xwalk, by = c("puma2010")) %>%
  mutate(
    czone = coalesce(czone2000, czone2010),
    afactor = coalesce(afactor2000, afactor2010)
  ) %>%
  filter(!is.na(afactor)) %>% # There are PUMA 77777 in Louisiana with no afactor
  mutate(cz_perwt = PERWT * afactor)

wage_quantiles_czone <- acs_df %>%
  group_by(YEAR, czone) %>%
  summarize(
    wage_p10 = fquantile(
      hourly_wage,
      probs = 0.10,
      w = cz_perwt,
      na.rm = TRUE,
      names = FALSE
    ),
    wage_p25 = fquantile(
      hourly_wage,
      probs = 0.25,
      w = cz_perwt,
      na.rm = TRUE,
      names = FALSE
    ),
    wage_p50 = fquantile(
      hourly_wage,
      probs = 0.50,
      w = cz_perwt,
      na.rm = TRUE,
      names = FALSE
    ),
    wage_p75 = fquantile(
      hourly_wage,
      probs = 0.75,
      w = cz_perwt,
      na.rm = TRUE,
      names = FALSE
    ),
    wage_p90 = fquantile(
      hourly_wage,
      probs = 0.90,
      w = cz_perwt,
      na.rm = TRUE,
      names = FALSE
    )
  ) %>%
  ungroup()

wage_quantiles_state <- acs_df %>%
  group_by(YEAR, STATEFIP) %>%
  summarize(
    wage_p10 = fquantile(
      hourly_wage,
      probs = 0.10,
      w = PERWT,
      na.rm = TRUE,
      names = FALSE
    ),
    wage_p25 = fquantile(
      hourly_wage,
      probs = 0.25,
      w = PERWT,
      na.rm = TRUE,
      names = FALSE
    ),
    wage_p50 = fquantile(
      hourly_wage,
      probs = 0.50,
      w = PERWT,
      na.rm = TRUE,
      names = FALSE
    ),
    wage_p75 = fquantile(
      hourly_wage,
      probs = 0.75,
      w = PERWT,
      na.rm = TRUE,
      names = FALSE
    ),
    wage_p90 = fquantile(
      hourly_wage,
      probs = 0.90,
      w = PERWT,
      na.rm = TRUE,
      names = FALSE
    )
  ) %>%
  ungroup()

write_parquet(
  wage_quantiles_czone,
  here("binaries", "acs_czone_wage_quantiles.parquet")
)

write_parquet(
  wage_quantiles_state,
  here("binaries", "acs_state_wage_quantiles.parquet")
)
