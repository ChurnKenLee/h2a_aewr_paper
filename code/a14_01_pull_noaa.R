# =============================================================================
# NOAA nClimGrid-Daily: Annual GDD by County -- Field Crops + Specialty Crops
#                       + Annual Temperature Decile Bin Day Counts
#                       + Annual Precipitation Metrics + Precip Decile Bins
# =============================================================================
#
# SOURCE: NOAA nClimGrid-Daily county-level area averages
#   https://www.ncei.noaa.gov/pub/data/daily-grids/v1-0-0/averages/
#
# No special packages required beyond the tidyverse.
#
# =============================================================================
# UNITS: ALL temperatures throughout this script are in DEGREES CELSIUS.
#
# The NOAA nClimGrid-Daily files store tmin, tmax, and tavg in degrees Celsius
# (confirmed in the NOAA readme at the URL above: "temperature (degrees C)").
#
# All crop base/cap thresholds in the CROPS list below are in degrees Celsius.
# Where the agronomic literature cites Fahrenheit, the conversion applied is:
#   C = (F - 32) * 5/9
#
# Conversions used (F -> C, rounded to 1 decimal):
#   32 F = 0.0 C    34 F = 1.1 C    40 F = 4.4 C    41 F = 5.0 C
#   44 F = 6.7 C    45 F = 7.2 C    50 F = 10.0 C   55 F = 12.8 C
#   60 F = 15.6 C   86 F = 30.0 C   95 F = 35.0 C
#
# UNITS AUDIT NOTES:
#  - cotton base: 60 F = 15.56 C; coded as 15.6 C (rounded to 1 dp, delta 0.04 C)
#  - cotton cap : 86 F = 30.0 C exactly
#  - tobacco base: prior version incorrectly coded 13.0 C; the cited source
#    (VA/NC Cooperative Extension) specifies 55 F = 12.78 C -> corrected to 12.8 C
#  - All other thresholds verified against their cited sources.
# =============================================================================
#
# FIELD CROPS (10):
# -----------------------------------------------------------------------
#  Crop            Base (C)  Cap (C)  Season           Source
#  corn            10.0      30.0     Apr-Sep           Iowa State / NOAA MRCC
#  soybean         10.0      30.0     Apr-Oct           Same as corn
#  sorghum         10.0      30.0     Apr-Sep           Same as corn
#  rice            10.0      none     May-Sep           FAO / Wikipedia GDD
#  cotton          15.6      30.0     Apr-Oct           60 F / 86 F industry std
#  winter_wheat     0.0      35.0     Oct(Y-1)-Jun(Y)   Montana Climate Office
#  spring_wheat     0.0      35.0     Mar-Jul           Montana Climate Office
#  barley           0.0      35.0     Mar-Jul           Montana Climate Office
#  canola           5.0      none     Mar-Jun           Montana Climate Office
#  sunflower        6.7      none     May-Sep           NDSU (44 F base)
#
# SPECIALTY CROPS (9):
# -----------------------------------------------------------------------
#  Crop            Base (C)  Cap (C)  Season           Source
#  grape           10.0      none     Apr-Oct           Winkler Index, UC Davis
#  citrus          12.8      none     Jan-Dec           55 F; Ortolani et al. 1991
#  apple            6.1      none     Mar-Oct           WSU apple phenology
#  potato           4.4      30.0     Apr-Sep           40 F / 86 F USDA standard
#  sugar_beet       1.1      30.0     Apr-Oct           34 F / 86 F NDAWN/NDSU
#  tomato          10.0      30.0     May-Sep           50 F / 86 F extension std
#  alfalfa          5.0      none     Mar-Oct           Cool-season forage std
#  peanut          10.0      none     May-Oct           USDA/extension
#  tobacco         12.8      none     May-Sep           55 F; VA/NC Coop Extension
# -----------------------------------------------------------------------
#
# TEMPERATURE DECILE BINS:
#  Bin breakpoints are computed ONCE from the pooled distribution of all daily
#  tavg values across all county-years in the dataset (the 10th, 20th, ...,
#  90th percentiles). These fixed breaks are then applied uniformly to every
#  county-year, so days_D01 always means "days in the coldest decile of the
#  CONUS daily temperature distribution" -- interpretable and comparable across
#  space and time. Bins are based on tavg. The actual temperature ranges are
#  printed during the sanity check section.
#
# WINTER WHEAT NOTE:
#  The script downloads one extra year (START_YEAR - 1) to correctly handle the
#  October(Y-1)-June(Y) cross-year season. Output starts at START_YEAR.
#
# PRECIPITATION METRICS:
#  Precipitation is pulled from the same nClimGrid-Daily source as temperature
#  (variable "prcp") and is in MILLIMETERS. Why annual average alone is weak:
#  timing and variability matter as much as total volume for crop choice.
#  Variables computed (all annual, calendar-year):
#    prcp_ann      total annual precipitation (mm)
#    prcp_gs       total growing-season precipitation, Apr-Sep (mm)
#    prcp_spring   total spring precipitation, Mar-May (mm); pre-planting moisture
#    n_wet_days    days with prcp >= WET_DAY_MM (1 mm) annually
#    max_cdd_gs    maximum consecutive dry days (prcp < 1 mm) within Apr-Sep;
#                  a proxy for growing-season drought stress
#    days_P01..P10 precipitation decile bin counts: number of WET days (>= 1mm)
#                  in each decile of the pooled CONUS wet-day prcp distribution.
#                  Computed on wet days only -- otherwise the lower bins are
#                  uninformative zero-rain days. P01 = lightest rain, P10 = heaviest.
#                  Fixed breakpoints derived once from all county-years >= START_YEAR.
#
# OUTPUT: One row per county x year. All temperatures in deg C; precip in mm.
# =============================================================================

rm(list = ls())

# --- 0. Packages -------------------------------------------------------------
pkgs <- c(
  "dplyr",
  "tidyr",
  "readr",
  "purrr",
  "lubridate",
  "stringr",
  "here",
  "arrow"
)
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
}
library(dplyr)
library(tidyr)
library(readr)
library(purrr)
library(lubridate)
library(stringr)
library(arrow)
library(here)

# --- 1. Configuration --------------------------------------------------------

OUTPUT_FILE <- here(
  "binaries",
  "county_h2a_prediction_climate_gdd_annual.parquet"
)

START_YEAR <- 2000
END_YEAR <- as.integer(format(Sys.Date(), "%Y"))
BASE_URL <- "https://www.ncei.noaa.gov/pub/data/daily-grids/v1-0-0/averages"

MISSING_VAL <- -999.00 # NOAA sentinel for missing / non-existent days

# Number of temperature bins for the day-count distribution
N_BINS <- 10

# Precipitation settings
WET_DAY_MM <- 1.0 # minimum mm to count as a "wet day" (standard WMO threshold)
N_PRCP_BINS <- 10 # decile bins for wet-day precipitation intensity


# --- 2. Crop GDD parameters (ALL VALUES IN DEGREES CELSIUS) ------------------
# base_C : lower threshold; growth = 0 below this temperature
# cap_C  : upper threshold; NA = no cap applied
# months : calendar months included in the season window
# cross_year : TRUE = season spans Oct(Y-1) to Jun(Y); GDD assigned to year Y

CROPS <- list(
  # ---- Field crops ----------------------------------------------------------

  corn = list(
    # 50 F / 86 F
    base_C = 10.0,
    cap_C = 30.0,
    months = 4:9,
    cross_year = FALSE
  ),
  soybean = list(
    # 50 F / 86 F
    base_C = 10.0,
    cap_C = 30.0,
    months = 4:10,
    cross_year = FALSE
  ),
  sorghum = list(
    # 50 F / 86 F
    base_C = 10.0,
    cap_C = 30.0,
    months = 4:9,
    cross_year = FALSE
  ),
  rice = list(
    # 50 F / no cap
    base_C = 10.0,
    cap_C = NA,
    months = 5:9,
    cross_year = FALSE
  ),
  cotton = list(
    # 60 F = 15.56 C -> 15.6 C / 86 F = 30 C
    base_C = 15.6,
    cap_C = 30.0,
    months = 4:10,
    cross_year = FALSE
  ),
  winter_wheat = list(
    # 32 F = 0 C / 95 F = 35 C; cross-year season
    base_C = 0.0,
    cap_C = 35.0,
    months = c(10:12, 1:6),
    cross_year = TRUE
  ),
  spring_wheat = list(
    # 32 F = 0 C / 95 F = 35 C
    base_C = 0.0,
    cap_C = 35.0,
    months = 3:7,
    cross_year = FALSE
  ),
  barley = list(
    # 32 F = 0 C / 95 F = 35 C
    base_C = 0.0,
    cap_C = 35.0,
    months = 3:7,
    cross_year = FALSE
  ),
  canola = list(
    # 41 F = 5 C / no cap
    base_C = 5.0,
    cap_C = NA,
    months = 3:6,
    cross_year = FALSE
  ),
  sunflower = list(
    # 44 F = 6.67 C -> 6.7 C / no cap
    base_C = 6.7,
    cap_C = NA,
    months = 5:9,
    cross_year = FALSE
  ),

  # ---- Specialty crops ------------------------------------------------------

  grape = list(
    # 50 F = 10 C / no cap (Winkler Index convention)
    base_C = 10.0,
    cap_C = NA,
    months = 4:10,
    cross_year = FALSE
  ),
  citrus = list(
    # 55 F = 12.78 C -> 12.8 C / no cap; year-round
    base_C = 12.8,
    cap_C = NA,
    months = 1:12,
    cross_year = FALSE
  ),
  apple = list(
    # 43 F = 6.11 C -> 6.1 C / no cap
    base_C = 6.1,
    cap_C = NA,
    months = 3:10,
    cross_year = FALSE
  ),
  potato = list(
    # 40 F = 4.44 C -> 4.4 C / 86 F = 30 C
    base_C = 4.4,
    cap_C = 30.0,
    months = 4:9,
    cross_year = FALSE
  ),
  sugar_beet = list(
    # 34 F = 1.11 C -> 1.1 C / 86 F = 30 C
    base_C = 1.1,
    cap_C = 30.0,
    months = 4:10,
    cross_year = FALSE
  ),
  tomato = list(
    # 50 F = 10 C / 86 F = 30 C
    base_C = 10.0,
    cap_C = 30.0,
    months = 5:9,
    cross_year = FALSE
  ),
  alfalfa = list(
    # 41 F = 5 C / no cap
    base_C = 5.0,
    cap_C = NA,
    months = 3:10,
    cross_year = FALSE
  ),
  peanut = list(
    # 50 F = 10 C / no cap
    base_C = 10.0,
    cap_C = NA,
    months = 5:10,
    cross_year = FALSE
  ),
  tobacco = list(
    # 55 F = 12.78 C -> 12.8 C / no cap (CORRECTED from 13.0)
    base_C = 12.8,
    cap_C = NA,
    months = 5:9,
    cross_year = FALSE
  )
)


# --- 3. NCEI to FIPS state code crosswalk ------------------------------------
# NOAA uses its own alphabetical state numbering system, not Census FIPS.

ncei_to_fips <- tribble(
  ~ncei , ~fips_state , ~state_abbr , ~state_name      ,
  "01"  , "01"        , "AL"        , "Alabama"        ,
  "02"  , "04"        , "AZ"        , "Arizona"        ,
  "03"  , "05"        , "AR"        , "Arkansas"       ,
  "04"  , "06"        , "CA"        , "California"     ,
  "05"  , "08"        , "CO"        , "Colorado"       ,
  "06"  , "09"        , "CT"        , "Connecticut"    ,
  "07"  , "10"        , "DE"        , "Delaware"       ,
  "08"  , "12"        , "FL"        , "Florida"        ,
  "09"  , "13"        , "GA"        , "Georgia"        ,
  "10"  , "16"        , "ID"        , "Idaho"          ,
  "11"  , "17"        , "IL"        , "Illinois"       ,
  "12"  , "18"        , "IN"        , "Indiana"        ,
  "13"  , "19"        , "IA"        , "Iowa"           ,
  "14"  , "20"        , "KS"        , "Kansas"         ,
  "15"  , "21"        , "KY"        , "Kentucky"       ,
  "16"  , "22"        , "LA"        , "Louisiana"      ,
  "17"  , "23"        , "ME"        , "Maine"          ,
  "18"  , "24"        , "MD"        , "Maryland"       ,
  "19"  , "25"        , "MA"        , "Massachusetts"  ,
  "20"  , "26"        , "MI"        , "Michigan"       ,
  "21"  , "27"        , "MN"        , "Minnesota"      ,
  "22"  , "28"        , "MS"        , "Mississippi"    ,
  "23"  , "29"        , "MO"        , "Missouri"       ,
  "24"  , "30"        , "MT"        , "Montana"        ,
  "25"  , "31"        , "NE"        , "Nebraska"       ,
  "26"  , "32"        , "NV"        , "Nevada"         ,
  "27"  , "33"        , "NH"        , "New Hampshire"  ,
  "28"  , "34"        , "NJ"        , "New Jersey"     ,
  "29"  , "35"        , "NM"        , "New Mexico"     ,
  "30"  , "36"        , "NY"        , "New York"       ,
  "31"  , "37"        , "NC"        , "North Carolina" ,
  "32"  , "38"        , "ND"        , "North Dakota"   ,
  "33"  , "39"        , "OH"        , "Ohio"           ,
  "34"  , "40"        , "OK"        , "Oklahoma"       ,
  "35"  , "41"        , "OR"        , "Oregon"         ,
  "36"  , "42"        , "PA"        , "Pennsylvania"   ,
  "37"  , "44"        , "RI"        , "Rhode Island"   ,
  "38"  , "45"        , "SC"        , "South Carolina" ,
  "39"  , "46"        , "SD"        , "South Dakota"   ,
  "40"  , "47"        , "TN"        , "Tennessee"      ,
  "41"  , "48"        , "TX"        , "Texas"          ,
  "42"  , "49"        , "UT"        , "Utah"           ,
  "43"  , "50"        , "VT"        , "Vermont"        ,
  "44"  , "51"        , "VA"        , "Virginia"       ,
  "45"  , "53"        , "WA"        , "Washington"     ,
  "46"  , "54"        , "WV"        , "West Virginia"  ,
  "47"  , "55"        , "WI"        , "Wisconsin"      ,
  "48"  , "56"        , "WY"        , "Wyoming"
)


# --- 4. Download one monthly county CSV --------------------------------------
# Tries "scaled" (quality-controlled) files first; falls back to "prelim".

download_monthly <- function(variable, year, month) {
  ym <- sprintf("%04d%02d", year, month)
  yr_str <- sprintf("%04d", year)
  col_names <- c(
    "region_type",
    "ncei_code",
    "region_name",
    "year",
    "month",
    "variable",
    paste0("d", sprintf("%02d", 1:31))
  )
  col_types <- paste0("cccccc", strrep("d", 31))

  for (status in c("scaled", "prelim")) {
    url <- sprintf(
      "%s/%s/%s-%s-cty-%s.csv",
      BASE_URL,
      yr_str,
      variable,
      ym,
      status
    )
    result <- tryCatch(
      suppressMessages(read_csv(
        url,
        col_names = col_names,
        col_types = col_types,
        show_col_types = FALSE
      )),
      error = function(e) NULL,
      warning = function(w) NULL
    )
    if (!is.null(result) && nrow(result) > 0) return(result)
  }
  message(sprintf("    Could not download %s %04d-%02d", variable, year, month))
  NULL
}


# --- 5. Process monthly CSV: wide to long, replace missing sentinel ----------

process_monthly <- function(df, variable) {
  if (is.null(df)) {
    return(NULL)
  }
  df |>
    mutate(
      ncei_state = str_pad(
        as.integer(str_sub(ncei_code, 1, nchar(ncei_code) - 3)),
        2,
        pad = "0"
      ),
      ncei_county = str_sub(ncei_code, -3, -1)
    ) |>
    pivot_longer(
      starts_with("d"),
      names_to = "day_str",
      values_to = variable
    ) |>
    mutate(
      day = as.integer(str_remove(day_str, "d")),
      yr = as.integer(year),
      mo = as.integer(month),
      date = suppressWarnings(
        as.Date(sprintf("%04d-%02d-%02d", yr, mo, day))
      ),
      # Replace NOAA missing-value sentinel (-999.xx) with NA
      !!variable := if_else(
        .data[[variable]] < MISSING_VAL + 1,
        NA_real_,
        .data[[variable]]
      )
    ) |>
    filter(!is.na(date)) |>
    select(ncei_state, ncei_county, region_name, date, mo, all_of(variable))
}


# --- 6. Pull and merge tmin + tmax + tavg + prcp for one month ---------------

pull_month <- function(year, month) {
  tmin_df <- process_monthly(download_monthly("tmin", year, month), "tmin")
  tmax_df <- process_monthly(download_monthly("tmax", year, month), "tmax")
  tavg_df <- process_monthly(download_monthly("tavg", year, month), "tavg")
  prcp_df <- process_monthly(download_monthly("prcp", year, month), "prcp")
  if (is.null(tmin_df) || is.null(tmax_df) || is.null(tavg_df)) {
    return(NULL)
  }
  tmin_df |>
    left_join(
      select(tmax_df, ncei_state, ncei_county, date, tmax),
      by = c("ncei_state", "ncei_county", "date")
    ) |>
    left_join(
      select(tavg_df, ncei_state, ncei_county, date, tavg),
      by = c("ncei_state", "ncei_county", "date")
    ) |>
    left_join(
      if (!is.null(prcp_df)) {
        select(prcp_df, ncei_state, ncei_county, date, prcp)
      } else {
        tibble(
          ncei_state = character(),
          ncei_county = character(),
          date = as.Date(NA),
          prcp = numeric()
        )
      },
      by = c("ncei_state", "ncei_county", "date")
    )
}


# --- 7. Core GDD calculation (all temperatures in deg C) ---------------------
# Applies the standard modified GDD formula:
#   daily_gdd = max(0, mean(max(tmin, base), min(tmax, cap)) - base)
# When cap_C is NA, tmax is used without an upper constraint.

calc_gdd <- function(df_season, base_C, cap_C) {
  df_season |>
    mutate(
      tmin_adj = pmax(tmin, base_C),
      tmax_adj = if (!is.na(cap_C)) pmin(tmax, cap_C) else tmax,
      daily_gdd = pmax(0, (tmin_adj + tmax_adj) / 2 - base_C)
    ) |>
    group_by(ncei_state, ncei_county, region_name) |>
    summarise(
      gdd = sum(daily_gdd, na.rm = TRUE),
      n_days = sum(!is.na(daily_gdd)),
      .groups = "drop"
    )
}


# --- 8. Temperature decile bin day counts ------------------------------------
# bin_breaks: a numeric vector of N_BINS - 1 cut-points in deg C, computed
#   once from the pooled CONUS distribution and passed to every county-year.
#   This keeps the bin boundaries FIXED across all observations so that
#   days_D01 always refers to the same absolute temperature range.

compute_tavg_bins <- function(df_year, bin_breaks) {
  # Add -Inf and +Inf as outer fences so every observation is captured
  breaks_full <- c(-Inf, bin_breaks, Inf)
  bin_labels <- sprintf("days_D%02d", seq_len(N_BINS))

  df_year |>
    filter(!is.na(tavg)) |>
    mutate(
      bin = cut(
        tavg,
        breaks = breaks_full,
        labels = bin_labels,
        include.lowest = TRUE,
        right = FALSE
      )
    ) |>
    group_by(ncei_state, ncei_county, region_name, bin) |>
    summarise(n = n(), .groups = "drop") |>
    # Ensure all 10 bins are present for every county (fill 0 for missing bins)
    complete(
      tidyr::nesting(ncei_state, ncei_county, region_name),
      bin = factor(bin_labels, levels = bin_labels),
      fill = list(n = 0L)
    ) |>
    pivot_wider(names_from = bin, values_from = n)
}


# --- 8b. Precipitation metrics -----------------------------------------------
# All precipitation values are in millimeters (mm), as provided by NOAA.
# Seasons: prcp_ann = Jan-Dec; prcp_gs = Apr-Sep; prcp_spring = Mar-May.
# Dry day: prcp < WET_DAY_MM (1 mm). Wet day: prcp >= WET_DAY_MM.
# max_cdd_gs: maximum run of consecutive dry days within the Apr-Sep window,
#   a compact measure of within-season drought stress.

compute_prcp_metrics <- function(df_year) {
  # Helper: max consecutive dry days in a sorted daily vector
  max_consec_dry <- function(prcp_vec) {
    if (all(is.na(prcp_vec))) {
      return(NA_integer_)
    }
    dry <- ifelse(is.na(prcp_vec), FALSE, prcp_vec < WET_DAY_MM)
    runs <- rle(dry)
    if (!any(runs$values)) {
      return(0L)
    }
    max(runs$lengths[runs$values], na.rm = TRUE)
  }

  # Annual totals and wet-day count
  ann <- df_year |>
    group_by(ncei_state, ncei_county, region_name) |>
    summarise(
      prcp_ann = sum(prcp, na.rm = TRUE),
      n_wet_days = sum(!is.na(prcp) & prcp >= WET_DAY_MM),
      .groups = "drop"
    )

  # Growing-season total (Apr-Sep)
  gs <- df_year |>
    filter(mo %in% 4:9) |>
    group_by(ncei_state, ncei_county, region_name) |>
    summarise(
      prcp_gs = sum(prcp, na.rm = TRUE),
      max_cdd_gs = max_consec_dry(prcp),
      .groups = "drop"
    )

  # Spring total (Mar-May)
  sp <- df_year |>
    filter(mo %in% 3:5) |>
    group_by(ncei_state, ncei_county, region_name) |>
    summarise(
      prcp_spring = sum(prcp, na.rm = TRUE),
      .groups = "drop"
    )

  ann |>
    left_join(gs, by = c("ncei_state", "ncei_county", "region_name")) |>
    left_join(sp, by = c("ncei_state", "ncei_county", "region_name"))
}


# --- 8c. Precipitation decile bin day counts ---------------------------------
# Bins are derived from the pooled distribution of WET-DAY daily prcp across
# all CONUS county-years >= START_YEAR (computed once and passed in as
# prcp_bin_breaks). For each county-year the function counts how many wet days
# fall in each bin. Dry days are excluded -- they would dominate the lower
# bins and obscure precipitation intensity variation.
# Output columns: days_P01 (lightest rain) ... days_P10 (heaviest rain).

compute_prcp_bins <- function(df_year, prcp_bin_breaks) {
  breaks_full <- c(0, prcp_bin_breaks, Inf) # lower fence = WET_DAY_MM handled by filter
  prcp_labels <- sprintf("days_P%02d", seq_len(N_PRCP_BINS))

  df_year |>
    filter(!is.na(prcp), prcp >= WET_DAY_MM) |> # wet days only
    mutate(
      bin = cut(
        prcp,
        breaks = breaks_full,
        labels = prcp_labels,
        include.lowest = TRUE,
        right = FALSE
      )
    ) |>
    group_by(ncei_state, ncei_county, region_name, bin) |>
    summarise(n = n(), .groups = "drop") |>
    complete(
      tidyr::nesting(ncei_state, ncei_county, region_name),
      bin = factor(prcp_labels, levels = prcp_labels),
      fill = list(n = 0L)
    ) |>
    pivot_wider(names_from = bin, values_from = n)
}

# --- 9. Compute all annual metrics for one harvest year ----------------------

compute_annual_all_crops <- function(
  all_daily,
  harvest_year,
  bin_breaks,
  prcp_bin_breaks
) {
  df_year <- all_daily |> filter(year(date) == harvest_year)

  # Annual temperature summaries (full calendar year of harvest_year; deg C)
  ann_temp <- df_year |>
    group_by(ncei_state, ncei_county, region_name) |>
    summarise(
      tmin_ann = mean(tmin, na.rm = TRUE), # mean of daily min temps (deg C)
      tmax_ann = mean(tmax, na.rm = TRUE), # mean of daily max temps (deg C)
      tavg_ann = mean(tavg, na.rm = TRUE), # mean of daily avg temps (deg C)
      n_days = sum(!is.na(tavg)),
      .groups = "drop"
    )

  # GDD per crop
  gdd_list <- imap(CROPS, function(crop, crop_name) {
    if (isTRUE(crop$cross_year)) {
      df_season <- all_daily |>
        filter(
          (year(date) == harvest_year - 1 & mo %in% 10:12) |
            (year(date) == harvest_year & mo %in% 1:6)
        )
    } else {
      df_season <- df_year |> filter(mo %in% crop$months)
    }
    if (nrow(df_season) == 0) {
      return(NULL)
    }
    calc_gdd(df_season, crop$base_C, crop$cap_C) |>
      rename(
        !!paste0("GDD_", crop_name) := gdd,
        !!paste0("n_", crop_name) := n_days
      )
  })

  # Temperature decile bin day counts (full calendar year)
  temp_bin_counts <- compute_tavg_bins(df_year, bin_breaks)

  # Precipitation metrics and decile bins (full calendar year)
  prcp_metrics <- compute_prcp_metrics(df_year)
  prcp_bin_counts <- compute_prcp_bins(df_year, prcp_bin_breaks)

  # Combine everything
  result <- ann_temp
  for (gdd_df in gdd_list[!sapply(gdd_list, is.null)]) {
    result <- left_join(
      result,
      gdd_df,
      by = c("ncei_state", "ncei_county", "region_name")
    )
  }
  result <- left_join(
    result,
    temp_bin_counts,
    by = c("ncei_state", "ncei_county", "region_name")
  )
  result <- left_join(
    result,
    prcp_metrics,
    by = c("ncei_state", "ncei_county", "region_name")
  )
  result <- left_join(
    result,
    prcp_bin_counts,
    by = c("ncei_state", "ncei_county", "region_name")
  )

  result |> mutate(year = harvest_year)
}


# --- 10. Main loop -----------------------------------------------------------

DOWNLOAD_FROM <- START_YEAR - 1 # need prior year for winter wheat

message(sprintf(
  "Downloading nClimGrid-Daily county data %d-%d (extra year for winter wheat)",
  DOWNLOAD_FROM,
  END_YEAR
))
message("Each calendar year = 48 CSV downloads (4 variables x 12 months).")
message("Expect roughly 1-4 minutes per year.\n")

all_daily <- map(DOWNLOAD_FROM:END_YEAR, function(yr) {
  message(sprintf("  Downloading %d ...", yr))
  bind_rows(map(1:12, function(mo) pull_month(yr, mo)))
}) |>
  bind_rows()

message("\nDownload complete.")


# --- 11. Compute fixed decile breakpoints from the pooled distributions ------
# Temperature: use all daily tavg values (START_YEAR onward).
# Precipitation: use all WET-DAY prcp values (>= WET_DAY_MM, START_YEAR onward).
# Both sets of breakpoints are computed ONCE and applied to every county-year
# so that bin labels refer to fixed absolute ranges throughout the panel.

message(
  "\nComputing temperature decile breakpoints from pooled CONUS daily tavg ..."
)

tavg_for_breaks <- all_daily |>
  filter(year(date) >= START_YEAR, !is.na(tavg)) |>
  pull(tavg)

BIN_PROBS <- seq(1 / N_BINS, 1 - 1 / N_BINS, by = 1 / N_BINS) # 0.1, ..., 0.9
BIN_BREAKS <- quantile(tavg_for_breaks, probs = BIN_PROBS) # 9 cut-points in deg C

bin_labels <- sprintf("days_D%02d", seq_len(N_BINS))
bin_lo <- c(-Inf, BIN_BREAKS)
bin_hi <- c(BIN_BREAKS, Inf)

message(
  "\nFixed temperature decile bin ranges (deg C; applied to all county-years):"
)
bin_ref <- data.frame(
  bin = bin_labels,
  lo_C = round(bin_lo, 2),
  hi_C = round(bin_hi, 2),
  lo_F = round(bin_lo * 9 / 5 + 32, 1),
  hi_F = round(bin_hi * 9 / 5 + 32, 1)
)
print(bin_ref, row.names = FALSE)

message(
  "\nComputing precipitation decile breakpoints from pooled CONUS wet-day prcp ..."
)

prcp_for_breaks <- all_daily |>
  filter(year(date) >= START_YEAR, !is.na(prcp), prcp >= WET_DAY_MM) |>
  pull(prcp)

PRCP_BIN_PROBS <- seq(
  1 / N_PRCP_BINS,
  1 - 1 / N_PRCP_BINS,
  by = 1 / N_PRCP_BINS
)
PRCP_BIN_BREAKS <- quantile(prcp_for_breaks, probs = PRCP_BIN_PROBS) # mm

prcp_labels <- sprintf("days_P%02d", seq_len(N_PRCP_BINS))
prcp_lo <- c(WET_DAY_MM, PRCP_BIN_BREAKS)
prcp_hi <- c(PRCP_BIN_BREAKS, Inf)

message(
  "\nFixed precipitation decile bin ranges (mm; wet days only; all county-years):"
)
prcp_ref <- data.frame(
  bin = prcp_labels,
  lo_mm = round(prcp_lo, 2),
  hi_mm = round(prcp_hi, 2),
  lo_in = round(prcp_lo / 25.4, 3),
  hi_in = round(prcp_hi / 25.4, 3)
)
print(prcp_ref, row.names = FALSE)


# --- 12. Compute annual metrics year by year ---------------------------------

message("\nComputing annual GDD and bin counts ...\n")

results <- map(START_YEAR:END_YEAR, function(yr) {
  message(sprintf("  Computing %d ...", yr))
  compute_annual_all_crops(all_daily, yr, BIN_BREAKS, PRCP_BIN_BREAKS)
})


# --- 13. Combine, attach FIPS, and tidy output columns ----------------------

final_df <- bind_rows(results) |>
  left_join(ncei_to_fips, by = c("ncei_state" = "ncei")) |>
  mutate(
    county_fips = str_pad(ncei_county, 3, pad = "0"),
    fips = paste0(fips_state, county_fips),
    county_name = str_trim(str_split_fixed(region_name, ",", 2)[, 2])
  ) |>
  select(
    year,
    fips,
    state_fips = fips_state,
    county_fips,
    state_abbr,
    state_name,
    county_name,
    # Annual temperatures (deg C)
    tmin_ann,
    tmax_ann,
    tavg_ann,
    n_days,
    # Field crop GDD + observation counts
    GDD_corn,
    n_corn,
    GDD_soybean,
    n_soybean,
    GDD_sorghum,
    n_sorghum,
    GDD_rice,
    n_rice,
    GDD_cotton,
    n_cotton,
    GDD_winter_wheat,
    n_winter_wheat,
    GDD_spring_wheat,
    n_spring_wheat,
    GDD_barley,
    n_barley,
    GDD_canola,
    n_canola,
    GDD_sunflower,
    n_sunflower,
    # Specialty crop GDD + observation counts
    GDD_grape,
    n_grape,
    GDD_citrus,
    n_citrus,
    GDD_apple,
    n_apple,
    GDD_potato,
    n_potato,
    GDD_sugar_beet,
    n_sugar_beet,
    GDD_tomato,
    n_tomato,
    GDD_alfalfa,
    n_alfalfa,
    GDD_peanut,
    n_peanut,
    GDD_tobacco,
    n_tobacco,
    # Temperature decile bin day counts (D01 = coldest, D10 = warmest)
    all_of(bin_labels),
    # Precipitation metrics (mm)
    prcp_ann,
    prcp_gs,
    prcp_spring,
    n_wet_days,
    max_cdd_gs,
    # Precipitation decile bin day counts on wet days (P01 = lightest, P10 = heaviest)
    all_of(prcp_labels)
  ) |>
  arrange(fips, year)


# --- 14. Sanity checks -------------------------------------------------------

message("\n--- County coverage per year (expect ~3,100 for CONUS) ---")
coverage <- final_df |>
  group_by(year) |>
  summarise(n_counties = n_distinct(fips), .groups = "drop")
print(head(coverage, 10))

message("\n--- GDD medians across all county-years (deg C-days) ---")
gdd_cols <- names(final_df)[startsWith(names(final_df), "GDD_")]
print(
  final_df |>
    summarise(across(all_of(gdd_cols), ~ round(median(.x, na.rm = TRUE)))) |>
    pivot_longer(everything(), names_to = "crop", values_to = "median_GDD") |>
    arrange(median_GDD)
)

message("\n--- Mean days per temperature decile bin (should all be ~36.5) ---")
print(
  final_df |>
    summarise(across(all_of(bin_labels), ~ round(mean(.x, na.rm = TRUE), 1))) |>
    pivot_longer(everything(), names_to = "bin", values_to = "mean_days")
)

message("\n--- Temperature decile bin ranges (repeated for reference) ---")
print(bin_ref, row.names = FALSE)

message("\n--- Precipitation summary medians across all county-years ---")
print(
  final_df |>
    summarise(
      prcp_ann_mm = round(median(prcp_ann, na.rm = TRUE)),
      prcp_gs_mm = round(median(prcp_gs, na.rm = TRUE)),
      prcp_spring_mm = round(median(prcp_spring, na.rm = TRUE)),
      n_wet_days = round(median(n_wet_days, na.rm = TRUE)),
      max_cdd_gs = round(median(max_cdd_gs, na.rm = TRUE))
    )
)

message(
  "\n--- Mean wet days per precipitation decile bin (should be roughly equal) ---"
)
print(
  final_df |>
    summarise(across(
      all_of(prcp_labels),
      ~ round(mean(.x, na.rm = TRUE), 1)
    )) |>
    pivot_longer(everything(), names_to = "bin", values_to = "mean_wet_days")
)

message("\n--- Precipitation decile bin ranges (repeated for reference) ---")
print(prcp_ref, row.names = FALSE)


# --- 15. Save ----------------------------------------------------------------

write_parquet(final_df, OUTPUT_FILE)
message(sprintf(
  "\nDone! %d rows | %d unique counties | years %d-%d\nSaved to: %s",
  nrow(final_df),
  n_distinct(final_df$fips),
  min(final_df$year),
  max(final_df$year),
  OUTPUT_FILE
))

# =============================================================================
# COMPLETE GDD PARAMETERS REFERENCE (ALL TEMPERATURES IN DEGREES CELSIUS)
# =============================================================================
#
#  FIELD CROPS
#  Crop          Base (C)  Cap (C)  F equiv       Season           Source
#  corn          10.0      30.0     50/86 F        Apr-Sep          Iowa State
#  soybean       10.0      30.0     50/86 F        Apr-Oct          Iowa State
#  sorghum       10.0      30.0     50/86 F        Apr-Sep          Iowa State
#  rice          10.0      none     50 F           May-Sep          FAO
#  cotton        15.6      30.0     60/86 F        Apr-Oct          USDA
#  winter_wheat   0.0      35.0     32/95 F        Oct(Y-1)-Jun(Y)  Montana CO
#  spring_wheat   0.0      35.0     32/95 F        Mar-Jul          Montana CO
#  barley         0.0      35.0     32/95 F        Mar-Jul          Montana CO
#  canola         5.0      none     41 F           Mar-Jun          Montana CO
#  sunflower      6.7      none     44 F           May-Sep          NDSU
#
#  SPECIALTY CROPS
#  Crop          Base (C)  Cap (C)  F equiv       Season           Source
#  grape         10.0      none     50 F           Apr-Oct          UC Davis
#  citrus        12.8      none     55 F           Jan-Dec          Ortolani 1991
#  apple          6.1      none     43 F           Mar-Oct          WSU
#  potato         4.4      30.0     40/86 F        Apr-Sep          USDA
#  sugar_beet     1.1      30.0     34/86 F        Apr-Oct          NDAWN/NDSU
#  tomato        10.0      30.0     50/86 F        May-Sep          Extension
#  alfalfa        5.0      none     41 F           Mar-Oct          Extension
#  peanut        10.0      none     50 F           May-Oct          USDA
#  tobacco       12.8      none     55 F           May-Sep          VA/NC Ext.
#
#  Grape / Winkler Index regions (deg C-day units):
#    Region I  : < 850      (very cool: Champagne, Willamette Valley)
#    Region II : 850-1111   (cool: Burgundy, Napa Valley)
#    Region III: 1111-1389  (moderate: Bordeaux, Sonoma County)
#    Region IV : 1389-1667  (warm: Central Valley CA)
#    Region V  : > 1667     (hot: San Joaquin Valley)
#    Multiply by 9/5 to convert to the traditional Fahrenheit-based Winkler units.
#
# =============================================================================
# OUTPUT COLUMNS
# =============================================================================
#
#  Identifiers (7):
#    year, fips, state_fips, county_fips, state_abbr, state_name, county_name
#
#  Annual temperatures -- all in degrees Celsius (4):
#    tmin_ann  mean of daily minimum temperatures
#    tmax_ann  mean of daily maximum temperatures
#    tavg_ann  mean of daily average temperatures
#    n_days    non-missing days used in the above means
#
#  For each of 19 crops (38 columns):
#    GDD_<crop>   cumulative growing degree days over the defined season (deg C)
#    n_<crop>     non-missing days contributing to GDD_<crop>; use to flag
#                 incomplete seasons (max = length of season window)
#
#  Temperature decile bin day counts (10 columns):
#    days_D01 ... days_D10
#    Number of calendar-year days with tavg in each decile bin.
#    D01 = coldest decile, D10 = warmest decile.
#    Bin boundaries are printed at runtime and stored in BIN_BREAKS (deg C).
#    All bins sum to n_days.
#
#  Precipitation metrics (5 columns):
#    prcp_ann      total annual precipitation (mm); Jan-Dec
#    prcp_gs       total growing-season precipitation (mm); Apr-Sep
#    prcp_spring   total spring precipitation (mm); Mar-May
#    n_wet_days    annual count of days with prcp >= 1 mm
#    max_cdd_gs    maximum consecutive dry days (<1 mm) within Apr-Sep;
#                  a compact growing-season drought stress indicator
#
#  Precipitation decile bin day counts (10 columns):
#    days_P01 ... days_P10
#    Number of WET days (prcp >= 1 mm) in each decile of the pooled CONUS
#    wet-day precipitation intensity distribution. Dry days are excluded.
#    P01 = lightest rain events, P10 = heaviest rain events.
#    Bin boundaries (mm) are printed at runtime and stored in PRCP_BIN_BREAKS.
#    All bins sum to n_wet_days.
#
# =============================================================================
