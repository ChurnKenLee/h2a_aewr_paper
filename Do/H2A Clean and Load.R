rm(list = ls())
if (!exists("path_code", mode = "function")) {
  source(file.path("code", "paths.R"))
}
ensure_project_dirs()
library(tidyverse)
library(arrow)
library(tidylog, warn.conflicts = FALSE)
## H2A: Load and clean datasets
## Phil Hoxie
## 1/11/24

## New Price Data (20260417) ----------------------

price_data <- read_parquet(path_int(
  "price_index_fisher_county_year.parquet"
)) %>%
  mutate(
    countyfips = as.numeric(fips),
    year = as.integer(year)
  ) %>%
  select(countyfips, year, fisher_index) %>%
  filter(year >= 2008 & year <= 2022)

write_parquet(
  price_data,
  path_int("nass_fisher_price_index.parquet")
)
head(price_data)
rm(price_data)

## requires running master file first ##

## Two datasets, one by census period and one by year (calendar year)

## CZ wage quantile ------------------------------------

cz_wage_quantiles <- read_parquet(path_int("acs_czone_wage_quantiles.parquet"))

## state min wages ------------------------------------

state_minwages <- read_parquet(path_int("fred_state_minwages.parquet"))

## Alt Min Wage from Ken -------------------------------

state_min_alt <- read_parquet(path_int("state_year_min_wage.parquet"))

## Fips Codes ------------------------------------------

fips_codes <- read.csv(
  file = path_raw("geographic_crosswalks", "phil", "fips_codes.csv"),
  stringsAsFactors = F
)

## AEWR Regions ----------------------------------------

aewr_regions <- read.csv(
  file = path_raw("geographic_crosswalks", "phil", "aewr_regions.csv"),
  stringsAsFactors = F
)

## Commuting Zones --------------------------------------

# source: https://sites.psu.edu/psucz/data/

cz_file <- read.csv(
  file = path_raw("geographic_crosswalks", "penn", "counties10-zqvz0r.csv"),
  stringsAsFactors = F
)

cz_file <- cz_file %>%
  rename(countyfips = FIPS, cz_out10 = OUT10)

write_parquet(cz_file, path_int("cz_file_2010.parquet"))


cz_file_small <- cz_file %>%
  select(countyfips, cz_out10)

write_parquet(cz_file_small, path_int("cz_file_2010_small.parquet"))


## PPI --------------------------------------------------
# Source: https://fred.stlouisfed.org/series/WPU01

ppi_data <- read.csv(
  file = path_raw("fred", "WPU01.csv"),
  stringsAsFactors = F
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

ggplot(data = ppi_data, aes(y = ppi_2012, x = year)) +
  geom_line()

ppi_data <- ppi_data %>%
  select(year, ppi_2012)

write_parquet(ppi_data, path_int("ppi_2012.parquet"))

## County Boundary ------------------------------------

# cleaned in the original stata file
state_border_pairs <- read.csv(
  file = path_raw("geographic_crosswalks", "state_border_pairs.csv"),
  stringsAsFactors = F
)
border_counties_allmatches <- read.csv(
  file = path_raw("geographic_crosswalks", "border_counties_allmatches.csv"),
  stringsAsFactors = F
)

full_county_set <- read_parquet(path_int("county_adjacency2010.parquet"))

# state minimum wages --------------------------------------

head(state_min_alt) # want the dummy from here
head(state_minwages)
# deflate by ppi

state_min_alt <- state_min_alt %>%
  mutate(fips = as.numeric(state_fips_code)) %>%
  filter(year == 2024) %>% # these are stable, so ignore them
  select(fips, agriculture_exemption)

state_minwage_ppi <- merge(
  x = state_minwages,
  y = ppi_data,
  by = "year",
  all.x = T,
  all.y = F
)

state_minwage_ppi <- merge(
  x = state_minwage_ppi,
  y = state_min_alt,
  by = "fips",
  all.x = T,
  all.y = F
)

state_minwage_ppi$agriculture_exemption[is.na(
  state_minwage_ppi$agriculture_exemption
)] <- TRUE

# make the prevailing ag min wage #

# fill in state min with federal if missing #

state_minwage_ppi <- state_minwage_ppi %>%
  transform(
    state_min_wage = ifelse(
      !is.na(state_min_wage),
      state_min_wage,
      federal_min_wage
    )
  )

state_minwage_ppi <- state_minwage_ppi %>%
  mutate(
    prevailing_ag_min_wage = ifelse(
      agriculture_exemption == T,
      federal_min_wage,
      pmax(state_min_wage, federal_min_wage, na.rm = TRUE)
    )
  )

state_minwage_ppi <- state_minwage_ppi %>%
  mutate(
    prevailing_min_wage_ppi = prevailing_min_wage / ppi_2012,
    prevailing_ag_min_wage_ppi = prevailing_ag_min_wage / ppi_2012,
    federal_min_wage_ppi = federal_min_wage / ppi_2012,
    state_min_wage_ppi = state_min_wage / ppi_2012
  ) %>%
  dplyr::select(-ppi_2012)

gc()

write_parquet(
  state_minwage_ppi,
  path_int("state_real_minwages.parquet")
)

## H2A Data -------------------------------------------
h2a_data <- read_parquet(
  file = path_int("h2a_aggregated.parquet")
)
h2a_predict <- read_parquet(
  file = path_int("h2a_prediction_using_elastic_net_continuous_basis.parquet")
) %>%
  mutate(
    countyfips = as.numeric(county_ansi)
  ) %>%
  select(-county_ansi) %>%
  write_parquet(path_int("h2a_predict.parquet"))

# census period
h2a_data <- h2a_data %>%
  mutate(
    census_period = ifelse(
      year >= 2008 & year < 2012,
      2012,
      ifelse(
        year >= 2012 & year < 2017,
        2017,
        ifelse(year >= 2017 & year < 2022, 2022, NA)
      )
    )
  )


# fix fips code

h2a_data <- h2a_data %>%
  mutate(
    cnty_fips_string = str_pad(
      as.character(county_fips_code),
      width = 3,
      side = "left",
      pad = "0"
    ),
    st_fips_string = str_pad(
      as.character(state_fips_code),
      width = 2,
      side = "left",
      pad = "0"
    )
  )

h2a_data <- h2a_data %>%
  mutate(countyfips = as.numeric(paste0(st_fips_string, cnty_fips_string)))


# clean by dropping old codes #

h2a_data <- h2a_data %>%
  filter(!is.na(census_period) & !is.na(county_fips_code)) %>%
  select(
    -cnty_fips_string,
    -st_fips_string,
    -state_fips_code,
    -county_fips_code
  )

# collapse by period, county (county and state fips)
h2a_prediction <- h2a_data %>%
  select(countyfips)

h2a_data <- h2a_data %>%
  group_by(census_period, countyfips) %>%
  summarise(
    nbr_workers_requested_all_years = sum(
      nbr_workers_requested_all_years,
      na.rm = T
    ),
    nbr_workers_certified_all_years = sum(
      nbr_workers_certified_all_years,
      na.rm = T
    ),
    man_hours_requested_all_years = sum(
      man_hours_requested_all_years,
      na.rm = T
    ),
    man_hours_certified_all_years = sum(
      man_hours_certified_all_years,
      na.rm = T
    ),
    nbr_applications_all_years = sum(nbr_applications_all_years, na.rm = T),
    nbr_workers_requested_start_year = sum(
      nbr_workers_requested_start_year,
      na.rm = T
    ),
    nbr_workers_certified_start_year = sum(
      nbr_workers_certified_start_year,
      na.rm = T
    ),
    man_hours_requested_start_year = sum(
      man_hours_requested_start_year,
      na.rm = T
    ),
    man_hours_certified_start_year = sum(
      man_hours_certified_start_year,
      na.rm = T
    ),
    nbr_applications_start_year = sum(nbr_applications_start_year, na.rm = T),
    nbr_workers_requested_fiscal_year = sum(
      nbr_workers_requested_fiscal_year,
      na.rm = T
    ),
    nbr_workers_certified_fiscal_year = sum(
      nbr_workers_certified_fiscal_year,
      na.rm = T
    ),
    man_hours_requested_fiscal_year = sum(
      man_hours_requested_fiscal_year,
      na.rm = T
    ),
    man_hours_certified_fiscal_year = sum(
      man_hours_certified_fiscal_year,
      na.rm = T
    ),
    nbr_applications_fiscal_year = sum(nbr_applications_fiscal_year, na.rm = T)
  )


write_parquet(h2a_data, path_processed("h2a_data.parquet"))

# yearly, for TS

h2a_data_ts <- read_parquet(
  path_int("h2a_aggregated.parquet"),
  stringsAsFactors = F
)

h2a_data_ts <- h2a_data_ts %>%
  group_by(year) %>%
  summarise(
    h2a_man_hours_certified = sum(man_hours_certified_start_year, na.rm = T),
    h2a_man_hours_requested = sum(man_hours_requested_start_year, na.rm = T),
    h2a_nbr_workers_certified = sum(
      nbr_workers_certified_start_year,
      na.rm = T
    ),
    h2a_nbr_workers_requested = sum(
      nbr_workers_requested_start_year,
      na.rm = T
    ),
    n_applications = sum(nbr_applications_start_year, na.rm = T)
  )


h2a_data_ts <- h2a_data_ts %>%
  rename(case_year = year)

write_parquet(h2a_data_ts, path_processed("h2a_data_ts.parquet"))

## Cropland Crocs Data Layer (CDL) -------------------------------------------

cdl_codes <- read_csv(
  file = path_raw("croplandcros_cdl", "croplandcros_cdl_crop_category.csv")
)

cdl_data <- read_parquet(path_int("croplandcros_county_crop_acres.parquet"))
head(cdl_data)
# need to put this in a wider format

head(cdl_codes)

cdl_data <- merge(
  x = cdl_data,
  y = cdl_codes,
  by = "crop_name",
  all.x = T,
  all.y = F
)

head(cdl_data)

cdl_data_collapse <- cdl_data %>%
  group_by(crop_type, year, fips) %>%
  summarise(
    acres = sum(acres, na.rm = T),
    crop_count = n()
  )

head(cdl_data_collapse)

cdl_data_collapse <- cdl_data_collapse %>%
  pivot_wider(
    id_cols = c("fips", "year"),
    names_from = "crop_type",
    values_from = c("acres", "crop_count")
  )

head(cdl_data_collapse)

cdl_data_collapse[is.na(cdl_data_collapse)] <- 0 # NAs are zeros

cdl_data_collapse <- cdl_data_collapse %>%
  ungroup() %>%
  mutate(countyfips = as.numeric(fips)) %>%
  dplyr::select(-fips)

write_parquet(cdl_data_collapse, path_int("cdl_cropshares.parquet"))

## NAWSPAD ---------------------------------------------
# state level data
nawspad_data <- read_parquet(
  path_int("nawspad.parquet"),
  stringsAsFactors = F
)

nawspad_data %>%
  group_by(crop_type) %>%
  tally()

## OEWS ------------------------------------------------
# wages
oews_agg_data <- read_parquet(
  file = path_int("oews_county_aggregated.parquet"),
  stringsAsFactors = F
)

## _____________________________________________________________________________
## Year by Year version --------------------------------------------------------
## _____________________________________________________________________________

## Fips Codes ------------------------------------------

fips_codes <- read.csv(
  file = path_raw("geographic_crosswalks", "phil", "fips_codes.csv"),
  stringsAsFactors = F
)

## AEWR Regions ----------------------------------------

aewr_regions <- read.csv(
  file = path_raw("geographic_crosswalks", "phil", "aewr_regions.csv"),
  stringsAsFactors = F
)

## PPI --------------------------------------------------
# Source: https://fred.stlouisfed.org/series/WPU01

ppi_data <- read_parquet(path_int("ppi_2012.parquet"))

## County Boundary ------------------------------------

# cleaned in the original stata file
state_border_pairs <- read.csv(
  file = path_raw("geographic_crosswalks", "state_border_pairs.csv"),
  stringsAsFactors = F
)
border_counties_allmatches <- read.csv(
  file = path_raw("geographic_crosswalks", "border_counties_allmatches.csv"),
  stringsAsFactors = F
)

full_county_set <- read_parquet(path_int("county_adjacency2010.parquet"))

dim(state_border_pairs)
dim(border_counties_allmatches)
dim(full_county_set)

## Census of Agriculture ------------------------------

options(arrow.skip_nul = TRUE)
census_of_agriculture <- read_parquet(path_int(
  "qs_census_selected_obs.parquet"
))

# general cleaning #

head(census_of_agriculture)

ag_census_data_items <- census_of_agriculture %>%
  group_by(commodity_desc) %>%
  tally()

# want: FARM OPERATIONS, AG LAND

census_of_agriculture_trim <- census_of_agriculture %>%
  filter(commodity_desc == "FARM OPERATIONS" | commodity_desc == "AG LAND")

census_of_agriculture_trim_items <- census_of_agriculture_trim %>%
  group_by(short_desc) %>%
  tally()

census_of_agriculture_trim <- census_of_agriculture_trim %>%
  filter(short_desc == "AG LAND, CROPLAND - ACRES")

# fix fips

census_of_agriculture_trim <- census_of_agriculture_trim %>%
  mutate(
    countyfips = as.numeric(str_c(
      as.character(state_fips_code),
      str_pad(as.character(county_code), width = 3, side = "left", pad = "0")
    ))
  )

census_of_agriculture_trim <- census_of_agriculture_trim %>%
  arrange(countyfips, year)

census_of_agriculture_trim <- census_of_agriculture_trim %>%
  mutate(label = "cropland_acr") %>%
  select(year, countyfips, value, label)

census_of_agriculture_trim <- census_of_agriculture_trim %>%
  filter(!is.na(countyfips))

census_of_agriculture_cropland <- census_of_agriculture_trim %>%
  pivot_wider(names_from = "label", values_from = "value")

head(census_of_agriculture_trim)

census_of_agriculture_cropland %>%
  write_parquet(path_int("census_ag_cropland_year.parquet"))

rm(census_of_agriculture_trim)

census_of_agriculture_cropland_base <- census_of_agriculture_cropland %>%
  filter(year == 2007) %>%
  rename(cropland_acr_2007 = cropland_acr) %>%
  select(-year)

census_of_agriculture_cropland_base %>%
  write_parquet(path_int("census_ag_cropland_2007_year.parquet"))

rm(census_of_agriculture_cropland_base)

## H2A Data -------------------------------------------

h2a_data <- read_parquet(
  path_int("h2a_aggregated.parquet"),
  stringsAsFactors = F
)

head(h2a_data)

# fix fips code

h2a_data <- h2a_data %>%
  mutate(
    cnty_fips_string = str_pad(
      as.character(county_fips_code),
      width = 3,
      side = "left",
      pad = "0"
    ),
    st_fips_string = str_pad(
      as.character(state_fips_code),
      width = 2,
      side = "left",
      pad = "0"
    )
  )

h2a_data <- h2a_data %>%
  mutate(countyfips = as.numeric(paste0(st_fips_string, cnty_fips_string)))


# clean by dropping old codes #

h2a_data <- h2a_data %>%
  filter(year <= 2022 & year > 2007 & !is.na(county_fips_code)) %>%
  select(
    -cnty_fips_string,
    -st_fips_string,
    -state_fips_code,
    -county_fips_code
  )

# collapse by period, county (county and state fips)

## Handle NAs?

head(h2a_data)

write_parquet(h2a_data, path_int("h2a_data_year.parquet"))
cat("h2a_data_year:", nrow(h2a_data), "rows,", ncol(h2a_data), "cols\n")

## AEWR -------------------------------------------------

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
    state_fips_code = as.numeric(state_fips_code)
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

head(aewr_data)

write_parquet(aewr_data, path_int("aewr_data_year.parquet"))
cat("aewr_data_year:", nrow(aewr_data), "rows,", ncol(aewr_data), "cols\n")

# full for TS

aewr_data <- read_parquet(
  path_int("aewr.parquet"),
  stringsAsFactors = F
) %>%
  mutate(aewr = as.numeric(aewr)) %>%
  mutate(year = as.numeric(year)) %>%
  mutate(state_fips_code = as.numeric(state_fips_code))

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

head(aewr_data)

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

head(aewr_data)

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

head(aewr_data)

aewr_data <- aewr_data %>%
  filter(!is.na(aewr_region_num))

write_parquet(aewr_data, path_processed("aewr_data_full.parquet"))
cat("aewr_data_full:", nrow(aewr_data), "rows,", ncol(aewr_data), "cols\n")

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

head(aewr_data)

## AEWR Region TS --------------------------------------------------------------

dir.create(path_figures("aewr_ts"), recursive = TRUE, showWarnings = FALSE)

for (i in 1:17) {
  plot <- ggplot(
    data = subset(aewr_data, aewr_region_num == i),
    aes(x = year, y = ln_aewr)
  ) +
    geom_line() +
    labs(title = paste0("AEWR Region Number: ", i)) +
    xlab("Log AEWR (nominal)")
  plot
  ggsave(
    filename = path_figures(
      "aewr_ts",
      paste0("ts_ln_aewr_nominal_", i, ".png")
    ),
    plot,
    device = "png"
  )
  rm(plot)

  plot <- ggplot(
    data = subset(aewr_data, aewr_region_num == i),
    aes(x = year, y = ln_aewr_l1)
  ) +
    geom_line() +
    labs(title = paste0("AEWR Region Number: ", i)) +
    xlab("Log Change AEWR (nominal)")
  plot
  ggsave(
    filename = path_figures(
      "aewr_ts",
      paste0("ts_ln_aewr_l1_nominal_", i, ".png")
    ),
    plot,
    device = "png"
  )
  rm(plot)
}

for (i in 1:17) {
  plot <- ggplot(
    data = subset(aewr_data, aewr_region_num == i),
    aes(x = year, y = aewr_diff)
  ) +
    geom_line() +
    labs(title = paste0("AEWR Region Number: ", i)) +
    xlab("AEWR (nominal) difference from LOO trend")
  plot
  ggsave(
    filename = path_figures(
      "aewr_ts",
      paste0("ts_aewr_diffloo_nominal_", i, ".png")
    ),
    plot,
    device = "png"
  )
  rm(plot)

  plot <- ggplot(
    data = subset(aewr_data, aewr_region_num == i),
    aes(x = year, y = aewr_ppi_diff)
  ) +
    geom_line() +
    labs(title = paste0("AEWR Region Number: ", i)) +
    xlab("AEWR (real) difference from LOO trend")
  plot
  ggsave(
    filename = path_figures(
      "aewr_ts",
      paste0("ts_aewr_diffloo_real_", i, ".png")
    ),
    plot,
    device = "png"
  )
  rm(plot)
}


for (i in 1:17) {
  plot <- ggplot(
    data = subset(aewr_data, aewr_region_num == i),
    aes(x = year, y = aewr)
  ) +
    geom_line() +
    labs(title = paste0("AEWR Region Number: ", i)) +
    xlab("AEWR (nominal)")
  plot
  ggsave(
    filename = path_figures("aewr_ts", paste0("ts_aewr_nominal_", i, ".png")),
    plot,
    device = "png"
  )
  rm(plot)

  plot <- ggplot(
    data = subset(aewr_data, aewr_region_num == i),
    aes(x = year, y = aewr_ppi)
  ) +
    geom_line() +
    labs(title = paste0("AEWR Region Number: ", i)) +
    xlab("AEWR (real)")
  plot
  ggsave(
    filename = path_figures("aewr_ts", paste0("ts_aewr_real_", i, ".png")),
    plot,
    device = "png"
  )
  rm(plot)

  plot <- ggplot(
    data = subset(aewr_data, aewr_region_num == i),
    aes(x = year, y = ch_aewr)
  ) +
    geom_line() +
    labs(title = paste0("AEWR Region Number: ", i)) +
    xlab("Change AEWR (nominal)")
  plot
  ggsave(
    filename = path_figures(
      "aewr_ts",
      paste0("ts_change_aewr_nominal_", i, ".png")
    ),
    plot,
    device = "png"
  )
  rm(plot)

  plot <- ggplot(
    data = subset(aewr_data, aewr_region_num == i),
    aes(x = year, y = ch_aewr_ppi)
  ) +
    geom_line() +
    labs(title = paste0("AEWR Region Number: ", i)) +
    xlab("Change AEWR (real)")
  plot

  ggsave(
    filename = path_figures(
      "aewr_ts",
      paste0("ts_change_aewr_real_", i, ".png")
    ),
    plot,
    device = "png"
  )
  rm(plot)

  plot <- ggplot(
    data = subset(aewr_data, aewr_region_num == i),
    aes(x = year, y = pch_aewr)
  ) +
    geom_line() +
    labs(title = paste0("AEWR Region Number: ", i)) +
    xlab("Percent Change AEWR (nominal)")
  plot
  ggsave(
    filename = path_figures(
      "aewr_ts",
      paste0("ts_percentchange_aewr_nominal_", i, ".png")
    ),
    plot,
    device = "png"
  )
  rm(plot)

  plot <- ggplot(
    data = subset(aewr_data, aewr_region_num == i),
    aes(x = year, y = pch_aewr_ppi)
  ) +
    geom_line() +
    labs(title = paste0("AEWR Region Number: ", i)) +
    xlab("Percent Change AEWR (real)")
  plot

  ggsave(
    filename = path_figures(
      "aewr_ts",
      paste0("ts_percentchange_aewr_real_", i, ".png")
    ),
    plot,
    device = "png"
  )
  rm(plot)
}

combo_plot <- ggplot(
  data = aewr_data,
  aes(
    x = year,
    y = pch_aewr,
    group = as.factor(aewr_region_num),
    color = as.factor(aewr_region_num)
  )
) +
  geom_line() +
  geom_vline(xintercept = 2007)
combo_plot


## BEA Data ---------------------------------------------

# job count data
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
    countyfips = as.numeric(str_trim(gsub("\"", "", GeoFIPS))), # remove qoutes
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

bea_caemp25n_data %>%
  group_by(year) %>%
  tally()

bea_caemp25n_data <- bea_caemp25n_data %>%
  filter(year > 2007 & year <= 2022)

# fix fips
# from: https://www.economy.com/support/blog/getfile.asp?did=869A03D1-5D74-4376-A606-00A8C64DDB0B&fid=a18d6d4873834f749d42ded633850a5e.xlsx

bea_fips_xwalk <- read.csv(
  file = path_raw("geographic_crosswalks", "phil", "bea_fips_xwalk.csv"),
  stringsAsFactors = F
)

county_list <- unique(select(full_county_set, fipscounty, countyname)) %>% # all county fips and names
  mutate(indata = 1)

bea_fips_xwalk <- merge(
  x = bea_fips_xwalk,
  y = county_list,
  by.x = "realfips",
  by.y = "fipscounty",
  all.x = T,
  all.y = F
)

# keep counties when conflict

bea_fips_xwalk <- bea_fips_xwalk %>%
  filter(county == 1) %>%
  select(realfips, beafips)

bea_caemp25n_data <- merge(
  x = bea_caemp25n_data,
  y = bea_fips_xwalk,
  by.x = "countyfips",
  by.y = "beafips",
  all.x = T,
  all.y = F
)

# fix fips

bea_caemp25n_data <- bea_caemp25n_data %>%
  rename(oldfips = countyfips) %>%
  mutate(countyfips = ifelse(!is.na(realfips), realfips, oldfips))

bea_caemp25n_data <- bea_caemp25n_data %>%
  select(-oldfips, -realfips)

# SD Oglala Lakota to Shannon
bea_caemp25n_data <- bea_caemp25n_data %>%
  mutate(
    countyfips = case_when(
      countyfips == 46102 ~ 46113,
      .default = countyfips
    )
  )

head(bea_caemp25n_data)

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

# farm finance
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
    countyfips = as.numeric(str_trim(gsub("\"", "", GeoFIPS))), # remove qoutes
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

bea_cainc45_data %>%
  group_by(category) %>%
  tally()

names(bea_cainc45_data)

bea_cainc45_data <- bea_cainc45_data %>%
  select(43:64) # be careful, this gets to be really a lot of data.

names(bea_cainc45_data)

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

bea_cainc45_data %>%
  group_by(year) %>%
  tally()

names(bea_cainc45_data)

bea_cainc45_data <- bea_cainc45_data %>%
  filter(year > 2007 & year <= 2022)

bea_cainc45_data <- merge(
  x = bea_cainc45_data,
  y = bea_fips_xwalk,
  by.x = "countyfips",
  by.y = "beafips",
  all.x = T,
  all.y = F
)

# fix fips

bea_cainc45_data <- bea_cainc45_data %>%
  rename(oldfips = countyfips) %>%
  mutate(countyfips = ifelse(!is.na(realfips), realfips, oldfips))

bea_cainc45_data <- bea_cainc45_data %>%
  select(-oldfips, -realfips)

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


## County Pop estimates ---------------------------------
# documentation: https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2000-2009/co-est2009-alldata.pdf

census_pop_2000 <- read.csv(
  path_raw("census_population_estimates", "co-est2009-alldata.csv"),
  stringsAsFactors = F
)
census_pop_2010 <- read.csv(
  path_raw("census_population_estimates", "co-est2019-alldata.csv"),
  stringsAsFactors = F
)
census_pop_2020 <- read.csv(
  path_raw("census_population_estimates", "co-est2022-alldata.csv"),
  stringsAsFactors = F
)

ct_xwalk <- read.csv(
  path_raw("geographic_crosswalks", "phil", "ct_region_xwalk.csv"),
  stringsAsFactors = F
)
ct_popgrth <- read.csv(
  path_raw("geographic_crosswalks", "phil", "ct_pop_grth.csv"),
  stringsAsFactors = F
)

census_pop_2000 <- census_pop_2000 %>%
  filter(SUMLEV == 50) %>%
  select(STATE, COUNTY, 10:19)
census_pop_2010 <- census_pop_2010 %>%
  filter(SUMLEV == 50) %>%
  select(STATE, COUNTY, 10:19)
census_pop_2020 <- census_pop_2020 %>%
  filter(SUMLEV == 50) %>%
  select(STATE, COUNTY, 9:11)

head(census_pop_2000)
head(census_pop_2010)
head(census_pop_2020)

# fix CT county to region nonsense.
# https://www.ctdata.org/blog/geographic-resources-for-connecticuts-new-county-equivalent-geography
# project using state growth rates

census_pop_ests <- merge(
  x = census_pop_2000,
  y = census_pop_2010,
  by = c("STATE", "COUNTY"),
  all = T
)
census_pop_ests <- merge(
  x = census_pop_ests,
  y = census_pop_2020,
  by = c("STATE", "COUNTY"),
  all = T
)

rm(census_pop_2000, census_pop_2010, census_pop_2020)

census_pop_ests <- census_pop_ests %>%
  mutate(
    countyfips = as.numeric(str_c(
      as.character(STATE),
      str_pad(as.character(COUNTY), width = 3, side = "left", pad = "0")
    ))
  )

census_pop_ests <- census_pop_ests %>%
  select(-STATE, -COUNTY)

census_pop_ests <- census_pop_ests %>%
  pivot_longer(
    cols = starts_with("POPESTIMATE"),
    names_to = "year",
    names_prefix = "POPESTIMATE",
    values_to = "pop_census",
    values_drop_na = F
  )

census_pop_ests %>%
  group_by(year) %>%
  tally()

census_pop_ests <- census_pop_ests %>%
  filter(year > 2007 & year <= 2022)

head(census_pop_ests)

# fix CT

census_pop_ests <- census_pop_ests %>%
  mutate(
    ct_region = ifelse(
      countyfips >= 9000 & countyfips <= 9999 & year >= 2020,
      1,
      0
    )
  ) # remove CT in 2020 on

ct_base <- census_pop_ests %>%
  filter(countyfips >= 9000 & countyfips <= 9999 & year == 2019)

census_pop_ests <- census_pop_ests %>%
  filter(ct_region == 0) %>%
  select(-ct_region)

census_pop_ests <- census_pop_ests %>%
  filter(countyfips < 9100 | countyfips > 9199) # get rid of the regions

ct_base <- ct_base %>%
  filter(countyfips < 9100) %>%
  select(countyfips, pop_census) %>%
  mutate(
    pop_census2020 = pop_census * ct_popgrth[2, 3],
    pop_census2021 = pop_census2020 * ct_popgrth[3, 3],
    pop_census2022 = pop_census2021 * ct_popgrth[4, 3]
  )

ct_base <- ct_base %>%
  select(-pop_census)

ct_fill <- ct_base %>%
  pivot_longer(
    cols = starts_with("pop_census"),
    names_to = "year",
    names_prefix = "pop_census",
    values_to = "pop_census",
    values_drop_na = F
  )


census_pop_ests <- rbind(census_pop_ests, ct_fill)

# fix SD county change

census_pop_ests$countyfips <- replace(
  census_pop_ests$countyfips,
  census_pop_ests$countyfips == 46102,
  46113
)

census_pop_ests %>% filter(countyfips == 46111) %>% tally() # double

census_pop_ests <- census_pop_ests %>%
  filter(!is.na(pop_census))

census_pop_ests %>% filter(countyfips == 46111) %>% tally() # fixed


write_parquet(
  census_pop_ests,
  path_int("census_pop_ests_year.parquet")
)
cat(
  "census_pop_ests_year:",
  nrow(census_pop_ests),
  "rows,",
  ncol(census_pop_ests),
  "cols\n"
)

## all county sample ----------------------------------------------------------

head(full_county_set)

county_list_unique <- unique(select(full_county_set, fipscounty, countyname))

dim(county_list_unique)

years <- seq(2008, 2022)
county_df <- NULL

for (i in 1:length(years)) {
  temp <- county_list_unique %>%
    mutate(year = years[i])
  county_df <- bind_rows(county_df, temp)
  rm(temp)
}

head(county_df)

county_df %>%
  group_by(year) %>%
  tally()

write_parquet(county_df, path_int("county_df_year.parquet"))
cat("county_df_year:", nrow(county_df), "rows,", ncol(county_df), "cols\n")

# remove files -------------------

str_detect(ls(), "folder_")

objects <- data.frame(name = ls(), keep = str_detect(ls(), "folder_")) %>%
  filter(keep == F)

rm(list = objects[, 1])
gc()
