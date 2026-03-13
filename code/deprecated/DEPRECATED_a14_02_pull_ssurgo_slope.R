# =============================================================================
# SSURGO County-Level Slope Metrics via Soil Data Access (SDA) API
# =============================================================================
#
# SOURCE: USDA Soil Data Access (SDA) — same snapshot used in
#         ssurgo_soil_shares.R.  No additional packages or credentials needed.
#
# Slope field: component.slope_r
#   The "representative value" of slope (percent) for each soil component.
#   Units are PERCENT (rise / run * 100), not degrees.
#   (Converting: degrees = atan(slope_pct / 100) * 180 / pi)
#
# SLOPE THRESHOLD FOR ROW CROPS
# -----------------------------------------------------------------------
# The threshold below which row crops are feasible with standard tillage
# is not a single bright line; it varies by tillage system and local
# rainfall. The agronomic literature converges around:
#
#   <= 2%   suitable for any tillage (spring plowing, moldboard)
#   <= 6%   practical limit for conventional no-till (Purdue extension)
#   <= 8%   practical limit for no-till + moderate management; NRCS
#           contour farming most effective below this level
#   <= 12%  no-till + cover crop can keep erosion below T-value (Purdue)
#   > 18%   Iowa State Extension: "avoid row crops regardless of tillage"
#
# STEEP_THRESH_PCT below is set to 8 (percent), representing the upper
# practical limit for mainstream no-till row crop production. Adjust to
# 6 for a stricter conventional-tillage reading, or 15 for a permissive
# modern-conservation reading. The mean slope is also output as a
# continuous variable so analysts can apply their own cutoffs.
#
# OUTPUTS (one row per county):
#   areasymbol       - 2-letter state + 3-digit county FIPS (e.g. "IA085")
#   areaname         - e.g. "Iowa County, Iowa"
#   total_acres      - total component-weighted acres in SSURGO for county
#   mean_slope_pct   - area-weighted mean slope (percent) across all major
#                      components; continuous variable, suitable as a
#                      regressor directly
#   steep_acres      - acres with slope_r > STEEP_THRESH_PCT
#   steep_share      - steep_acres / total_acres
#   steep_thresh_pct - the threshold used (recorded for reproducibility)
#
# NOTE: SSURGO is a static snapshot (refreshed Oct 1 each year). Treat
# these as fixed cross-sectional covariates, not annual panel variables.
# =============================================================================

rm(list = ls())

# --- 0. Packages -------------------------------------------------------------

if (!requireNamespace("soilDB", quietly = TRUE)) {
  install.packages("soilDB")
}
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}
if (!requireNamespace("readr", quietly = TRUE)) {
  install.packages("readr")
}
if (!requireNamespace("purrr", quietly = TRUE)) {
  install.packages("purrr")
}
if (!requireNamespace("here", quietly = TRUE)) {
  install.packages("here")
}
if (!requireNamespace("arrow", quietly = TRUE)) {
  install.packages("arrow")
}

library(soilDB)
library(dplyr)
library(readr)
library(purrr)
library(tidyverse)
library(here)
library(arrow)


# --- 1. Configuration --------------------------------------------------------

OUTPUT_FILE <- here(
  "binaries",
  "county_h2a_prediction_ssurgo_slopes.parquet"
)

# Slope threshold for "too steep for row crops" classification (percent).
# See header comment for the agronomic justification.
STEEP_THRESH_PCT <- 8.0

state_codes <- c(
  "AL",
  "AK",
  "AZ",
  "AR",
  "CA",
  "CO",
  "CT",
  "DE",
  "FL",
  "GA",
  "HI",
  "ID",
  "IL",
  "IN",
  "IA",
  "KS",
  "KY",
  "LA",
  "ME",
  "MD",
  "MA",
  "MI",
  "MN",
  "MS",
  "MO",
  "MT",
  "NE",
  "NV",
  "NH",
  "NJ",
  "NM",
  "NY",
  "NC",
  "ND",
  "OH",
  "OK",
  "OR",
  "PA",
  "RI",
  "SC",
  "SD",
  "TN",
  "TX",
  "UT",
  "VT",
  "VA",
  "WA",
  "WV",
  "WI",
  "WY"
)


# --- 2. Query function -------------------------------------------------------
# For each component we need:
#   muacres      - map unit area (acres); consistent with ssurgo_soil_shares.R
#   comppct_r    - representative % of map unit covered by this component
#   slope_r      - representative slope value (percent) for this component
#
# Component-level weighted acres = muacres * comppct_r / 100.
#
# We pull component-level rows (not pre-aggregated) so we can correctly
# weight slope by area when aggregating to the county level.
#
# NULL slope_r values are excluded from both the numerator and denominator
# of the weighted mean (NULLIF / CASE logic handled in R after pulling).

query_state_slopes <- function(state) {
  sql <- paste0(
    "
    SELECT
      legend.areasymbol,
      legend.areaname,
      CAST(mapunit.muacres    AS NUMERIC) AS muacres,
      CAST(component.comppct_r AS NUMERIC) AS comppct_r,
      CAST(component.slope_r   AS NUMERIC) AS slope_r
    FROM legend
      INNER JOIN mapunit   ON mapunit.lkey    = legend.lkey
      INNER JOIN component ON component.mukey = mapunit.mukey
    WHERE legend.areasymbol LIKE '",
    state,
    "%'
      AND legend.areasymbol != 'US'
      AND component.majcompflag = 'Yes'
  "
  )

  message("  Querying ", state, " ...")

  result <- tryCatch(
    SDA_query(sql),
    error = function(e) {
      message("  ERROR for ", state, ": ", conditionMessage(e))
      NULL
    },
    warning = function(w) {
      message("  WARNING for ", state, ": ", conditionMessage(w))
      NULL
    }
  )

  if (
    is.null(result) || inherits(result, "try-error") || !is.data.frame(result)
  ) {
    message("  No data returned for ", state)
    return(NULL)
  }
  if (nrow(result) == 0) {
    message("  Empty result for ", state)
    return(NULL)
  }

  message("  Got ", nrow(result), " component rows for ", state)
  Sys.sleep(1) # be polite to the API
  result
}


# --- 3. Pull data for all states ---------------------------------------------

message("Starting SDA slope pull for ", length(state_codes), " states...\n")

raw_list <- map(state_codes, query_state_slopes)


# --- 4. Aggregate to county level --------------------------------------------
# component weight (acres) = muacres * comppct_r / 100
# weighted mean slope       = sum(slope_r * weight) / sum(weight), over non-NA slope_r
# steep_acres               = sum(weight) where slope_r > STEEP_THRESH_PCT

raw_df <- bind_rows(raw_list) |>
  mutate(
    comp_acres = muacres * comppct_r / 100.0
  )

county_slopes <- raw_df |>
  group_by(areasymbol, areaname) |>
  summarise(
    total_acres = sum(comp_acres, na.rm = TRUE),

    # Weighted mean slope: only include components with non-missing slope_r.
    # Using explicit numerator / denominator to avoid biasing toward 0.
    mean_slope_pct = sum(slope_r * comp_acres, na.rm = TRUE) /
      sum(if_else(!is.na(slope_r), comp_acres, 0), na.rm = TRUE),

    # Steep acres: sum of component-weighted acres where slope_r > threshold.
    # Components with NA slope_r are treated as NOT steep (conservative).
    steep_acres = sum(
      if_else(!is.na(slope_r) & slope_r > STEEP_THRESH_PCT, comp_acres, 0),
      na.rm = TRUE
    ),

    .groups = "drop"
  ) |>
  mutate(
    steep_share = steep_acres / total_acres,
    steep_thresh_pct = STEEP_THRESH_PCT
  ) |>
  arrange(areasymbol)


# --- 5. Sanity checks --------------------------------------------------------

message(
  "\n--- Coverage: counties by state (spot-check against ~3,100 CONUS counties) ---"
)
state_coverage <- county_slopes |>
  mutate(state_abbr = substr(areasymbol, 1, 2)) |>
  group_by(state_abbr) |>
  summarise(n_counties = n(), .groups = "drop")
print(state_coverage, n = Inf)

message(sprintf("\nTotal counties: %d", nrow(county_slopes)))

message("\n--- Distribution of area-weighted mean slope (percent) ---")
print(summary(county_slopes$mean_slope_pct))

message(sprintf(
  "\n--- Steep land (slope > %.0f%%) ---",
  STEEP_THRESH_PCT
))
print(summary(county_slopes$steep_share))

message("\n--- 10 counties with highest steep_share ---")
print(
  county_slopes |>
    arrange(desc(steep_share)) |>
    select(areasymbol, areaname, mean_slope_pct, steep_share, steep_acres) |>
    head(10)
)

message("\n--- 10 counties with lowest steep_share (flattest) ---")
print(
  county_slopes |>
    arrange(steep_share) |>
    select(areasymbol, areaname, mean_slope_pct, steep_share, steep_acres) |>
    head(10)
)

# Confirm mean_slope and steep_share are directionally consistent
mean_slope_quintile <- ntile(county_slopes$mean_slope_pct, 5)
steep_share_quintile <- ntile(county_slopes$steep_share, 5)
message("\n--- Cross-tab: mean slope quintile vs steep share quintile ---")
message("(Should be roughly diagonal if the two measures agree)")
print(table(mean_slope_quintile, steep_share_quintile))


# --- 6. Save -----------------------------------------------------------------

write_parquet(county_slopes, OUTPUT_FILE)
message(sprintf(
  "\nDone! %d counties saved to: %s",
  nrow(county_slopes),
  OUTPUT_FILE
))

# =============================================================================
# OUTPUT COLUMNS
# =============================================================================
#
#  areasymbol       2-letter state postal code + 3-digit county FIPS
#                   e.g. "IA085" = Iowa County, Iowa
#                   Matches the areasymbol in ssurgo_soil_shares_by_county.csv
#
#  areaname         Human-readable county name
#
#  total_acres      Sum of component-weighted acres (muacres * comppct_r / 100)
#                   across all major components in the county's SSURGO survey
#
#  mean_slope_pct   Area-weighted mean slope in PERCENT across all major
#                   components with non-missing slope_r.
#                   Interpretation: 0-2% = nearly level; 2-6% = gently sloping;
#                   6-12% = moderately sloping; 12-20% = strongly sloping;
#                   >20% = steep / very steep.
#                   To convert to degrees: atan(mean_slope_pct / 100) * 180 / pi
#
#  steep_acres      Component-weighted acres where slope_r > STEEP_THRESH_PCT.
#                   Components with missing slope_r are treated as NOT steep.
#
#  steep_share      steep_acres / total_acres.
#                   A value of 0.40 means 40% of the county's surveyed soil
#                   area lies on slopes exceeding STEEP_THRESH_PCT percent.
#
#  steep_thresh_pct The threshold used (default 8.0 percent). Stored in the
#                   output for reproducibility. Change STEEP_THRESH_PCT at the
#                   top of the script to use a different cutoff.
#
# TO MERGE WITH THE PANEL DATASET:
#   county_slopes has one row per county (static). Join to the annual panel on
#   the areasymbol field (or the last 3 digits of areasymbol = county FIPS).
#
#   Example join with a panel that has a 5-digit FIPS column:
#
#     library(dplyr)
#     state_fips_xwalk <- tidycensus::fips_codes |>
#       distinct(state, state_code) |>
#       rename(state_abbr = state, state_fips = state_code)
#
#     slopes_with_fips <- county_slopes |>
#       mutate(
#         state_abbr  = substr(areasymbol, 1, 2),
#         county_fips = substr(areasymbol, 3, 5)
#       ) |>
#       left_join(state_fips_xwalk, by = "state_abbr") |>
#       mutate(fips = paste0(state_fips, county_fips))
#
#     panel_with_slopes <- annual_panel |>
#       left_join(
#         select(slopes_with_fips, fips, mean_slope_pct, steep_acres,
#                steep_share, steep_thresh_pct),
#         by = "fips"
#       )
#
# =============================================================================
