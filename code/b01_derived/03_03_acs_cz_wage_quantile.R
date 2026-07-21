# Purpose: Estimate weighted ACS wage quantiles by commuting zone and county.
# Inputs: the ACS one-year wage extract and PUMA-county-CZ crosswalks.
# Outputs: data/intermediate/acs_czone_wage_quantiles.parquet.

if (!exists("path_code", mode = "function")) {
  source(
    if (file.exists(file.path("code", "bootstrap_paths.R"))) {
      file.path("code", "bootstrap_paths.R")
    } else {
      file.path("..", "bootstrap_paths.R")
    }
  )
}
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(collapse)

# Map each PUMA vintage to Penn State's 2010 commuting zones through counties.
# PUMAs and CZs do not nest, so keep the population allocation shares rather
# than assigning each PUMA to a single CZ.

read_geocorr_puma_county <- function(
  filename,
  puma_column,
  puma_vintage,
  county_column = "county"
) {
  geocorr_path <- path_raw(
    "geographic_crosswalks",
    "geocorr",
    filename
  )
  geocorr_names <- names(read_csv(
    geocorr_path,
    n_max = 0,
    show_col_types = FALSE
  ))

  read_csv(
    geocorr_path,
    skip = 2,
    col_names = geocorr_names,
    show_col_types = FALSE
  ) %>%
    transmute(
      puma_vintage = puma_vintage,
      statefip = str_pad(as.character(state), 2, side = "left", pad = "0"),
      puma = str_pad(
        as.character(.data[[puma_column]]),
        5,
        side = "left",
        pad = "0"
      ),
      # Convert post-2010 county definitions back to the county vintage used
      # by Penn's 2010 CZ file.
      countyfips_2010 = recode(
        str_pad(
          as.character(.data[[county_column]]),
          5,
          side = "left",
          pad = "0"
        ),
        "02063" = "02261", # Chugach -> 2010 Valdez-Cordova, Alaska
        "02066" = "02261", # Copper River -> 2010 Valdez-Cordova, Alaska
        "02158" = "02270", # Kusilvak (formerly Wade Hampton), Alaska
        "46102" = "46113" # Oglala Lakota (formerly Shannon), South Dakota
      ),
      puma_county_share = as.numeric(afact)
    )
}

penn_county_cz <- read_csv(
  path_raw("geographic_crosswalks", "penn", "counties10-zqvz0r.csv"),
  show_col_types = FALSE
) %>%
  transmute(
    countyfips_2010 = str_pad(
      as.character(FIPS),
      5,
      side = "left",
      pad = "0"
    ),
    cz_out10 = as.character(OUT10)
  ) %>%
  distinct()

stopifnot(!anyDuplicated(penn_county_cz$countyfips_2010))

puma_county_2000 <- read_geocorr_puma_county(
  "geocorr2014_puma2000_county2010.csv",
  "puma2k",
  "2000"
)
puma_county_2010 <- read_geocorr_puma_county(
  "geocorr2018_puma2010_county2010.csv",
  "puma12",
  "2010"
)
puma_county_2020 <- read_geocorr_puma_county(
  "geocorr2022_puma2020_county2020.csv",
  "puma22",
  "2020"
) %>%
  # Geocorr now reports Connecticut planning regions as counties. Replace
  # those rows below with the direct legacy-county relationship.
  filter(statefip != "09") %>%
  bind_rows(read_geocorr_puma_county(
    "geocorr2022_puma2020_ctcounty2010.csv",
    "puma22",
    "2020",
    county_column = "CTcounty"
  ))

collapse_puma_county_to_cz <- function(puma_county) {
  puma_cz <- puma_county %>%
    left_join(penn_county_cz, by = "countyfips_2010")

  if (any(is.na(puma_cz$cz_out10))) {
    stop("Some PUMA-to-county allocations do not match Penn's 2010 CZ file.")
  }

  puma_cz <- puma_cz %>%
    group_by(puma_vintage, statefip, puma, cz_out10) %>%
    summarise(
      puma_cz_share_raw = sum(puma_county_share, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    group_by(puma_vintage, statefip, puma) %>%
    mutate(
      puma_cz_share_total = sum(puma_cz_share_raw),
      puma_cz_share = puma_cz_share_raw / puma_cz_share_total
    ) %>%
    ungroup()

  # Geocorr allocation factors are rounded, so allow a small tolerance before
  # normalizing them to sum exactly to one within source PUMA.
  if (any(abs(puma_cz$puma_cz_share_total - 1) > 0.01)) {
    stop("PUMA-to-CZ allocation factors do not preserve PUMA population mass.")
  }

  puma_cz %>%
    select(puma_vintage, statefip, puma, cz_out10, puma_cz_share)
}

puma_cz_2000 <- collapse_puma_county_to_cz(puma_county_2000)
puma_cz_2010 <- collapse_puma_county_to_cz(puma_county_2010)
puma_cz_2020 <- collapse_puma_county_to_cz(puma_county_2020)

puma_cz_crosswalk <- bind_rows(
  puma_cz_2000,
  puma_cz_2010,
  puma_cz_2020
)

puma_cz_mass_check <- puma_cz_crosswalk %>%
  group_by(puma_vintage, statefip, puma) %>%
  summarise(puma_cz_share_total = sum(puma_cz_share), .groups = "drop")
stopifnot(all(abs(puma_cz_mass_check$puma_cz_share_total - 1) < 1e-8))

# ---- Calculate hourly wage quantiles ----
acs_ds <- open_dataset(
  path_int("acs_1year_for_wages.parquet")
)

acs_ds <- acs_ds %>%
  filter(
    CLASSWKR == 2,
    INCWAGE > 0 & INCWAGE < 999998,
    UHRSWORK > 0
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
    PUMA = if_else(
      is.na(PUMA),
      NA_character_,
      sprintf("%05d", as.integer(PUMA))
    ),
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
  mutate(
    puma_vintage = case_when(
      YEAR < 2012 ~ "2000",
      YEAR < 2022 ~ "2010",
      TRUE ~ "2020"
    ),
    statefip = str_pad(
      as.character(STATEFIP),
      2,
      side = "left",
      pad = "0"
    ),
    puma = str_pad(
      as.character(PUMA),
      5,
      side = "left",
      pad = "0"
    )
  ) %>%
  select(YEAR, statefip, puma, puma_vintage, PERWT, hourly_wage)

# Fractionally allocate every PUMS observation across Penn CZs. This preserves
# the PUMA population weight while allowing PUMAs that cross CZ boundaries to
# contribute to every intersecting CZ.
acs_df <- acs_df %>%
  # The 2002 ACS has no PUMA identifiers and cannot be assigned. The analysis
  # merge begins in 2005, so this does not remove an estimation-panel year.
  filter(!is.na(puma)) %>%
  left_join(
    puma_cz_crosswalk,
    by = c("puma_vintage", "statefip", "puma")
  )

unmatched_pumas <- acs_df %>%
  filter(is.na(cz_out10)) %>%
  distinct(puma_vintage, statefip, puma)

# PUMA 77777 in Louisiana is the special Hurricane Katrina displacement PUMA.
unexpected_unmatched_pumas <- unmatched_pumas %>%
  filter(!(puma_vintage == "2000" & statefip == "22" & puma == "77777"))
if (nrow(unexpected_unmatched_pumas) > 0) {
  stop("Unexpected ACS PUMAs are missing from the PUMA-to-Penn-CZ crosswalk.")
}

acs_df <- acs_df %>%
  filter(!is.na(cz_out10)) %>%
  mutate(cz_perwt = PERWT * puma_cz_share)

wage_quantiles_czone <- acs_df %>%
  filter(!is.na(cz_perwt)) %>%
  group_by(YEAR, cz_out10) %>%
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

# Attach county codes
wage_quantiles_county <- wage_quantiles_czone %>%
  left_join(
    penn_county_cz %>% rename(county_ansi = countyfips_2010),
    by = "cz_out10"
  ) %>%
  arrange(county_ansi)

stopifnot(!anyDuplicated(wage_quantiles_county[c("YEAR", "county_ansi")]))

wage_quantiles_county %>%
  write_parquet(path_int("acs_czone_wage_quantiles.parquet"))
