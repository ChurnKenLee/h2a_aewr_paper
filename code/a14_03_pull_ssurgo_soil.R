# =============================================================================
# SSURGO Soil Type Shares by County via Soil Data Access (SDA) API
# =============================================================================
# NOTE: SSURGO soil data is not truly annual — soils don't change year to year.
# The database is a current snapshot (refreshed each Oct 1). It is standard
# practice in research to treat it as a fixed cross-sectional covariate.
# There is no "2011 vintage" of SSURGO available via the API.
#
# This script pulls the current SSURGO snapshot and produces one row per
# (county x soil series), with acreage and share of county total.
#
# Output columns:
#   areasymbol      - e.g. "CA113" (state abbreviation + 3-digit county FIPS)
#   areaname        - e.g. "Yolo County, California"
#   compname        - soil series name, e.g. "Rincon"
#   taxorder        - soil taxonomy order, e.g. "Mollisols"
#   taxsuborder     - e.g. "Xerolls"
#   taxgrtgroup     - e.g. "Haploxerolls"
#   comp_acres      - weighted acres of this component in the county
#   total_acres     - total surveyed acres in the county
#   soil_share      - comp_acres / total_acres
# =============================================================================

rm(list = ls())
# --- 0. Install & load packages ----------------------------------------------

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

library(soilDB) # SDA_query() — wraps the REST/POST API cleanly
library(dplyr)
library(readr)
library(purrr)
library(tidyverse)
library(here)
library(arrow)

output_path <- here(
  "binaries",
  "county_h2a_prediction_soil_shares.parquet"
)

# --- 1. State codes ----------------------------------------------------------
# Standard 2-letter postal codes; SDA area symbols are like "CA113", "IL031"

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
# Pulls all major soil components for all counties in one state.
#
# Key fields:
#   muacres      = area of the map unit polygon (acres)
#   comppct_r    = representative % of that map unit covered by this component
#   comp_acres   = muareaacres * comppct_r / 100  (weighted acres per component)
#
# majcompflag = 'Yes' keeps only major components (drops minor inclusions).
# areasymbol != 'US' excludes STATSGO (the coarser national-scale survey),
# which is stored in the same database with areasymbol = 'US'.

query_state <- function(state) {
  sql <- paste0(
    "
    SELECT
      legend.areasymbol,
      legend.areaname,
      component.compname,
      component.taxorder,
      component.taxsuborder,
      component.taxgrtgroup,
      SUM(CAST(mapunit.muacres AS NUMERIC) * CAST(component.comppct_r AS NUMERIC) / 100.0) AS comp_acres,
      SUM(CAST(mapunit.muacres AS NUMERIC)) AS mapunit_acres_sum
    FROM legend
      INNER JOIN mapunit  ON mapunit.lkey   = legend.lkey
      INNER JOIN component ON component.mukey = mapunit.mukey
    WHERE legend.areasymbol LIKE '",
    state,
    "%'
      AND legend.areasymbol != 'US'
      AND component.majcompflag = 'Yes'
    GROUP BY
      legend.areasymbol,
      legend.areaname,
      component.compname,
      component.taxorder,
      component.taxsuborder,
      component.taxgrtgroup
    ORDER BY legend.areasymbol, comp_acres DESC
  "
  )

  message("  Querying ", state, "...")

  # SDA_query() can return: a data.frame, NULL, or a try-error object.
  # We wrap in tryCatch to catch hard errors, then explicitly check for
  # try-error class before calling nrow() — which is what caused the bug.
  result <- tryCatch(
    SDA_query(sql),
    error = function(e) {
      message("  ERROR for ", state, ": ", conditionMessage(e))
      return(NULL)
    },
    warning = function(w) {
      message("  WARNING for ", state, ": ", conditionMessage(w))
      return(NULL)
    }
  )

  # Guard against try-error objects, NULL, non-data-frame results
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

  message("  Got ", nrow(result), " rows for ", state)
  Sys.sleep(1) # be polite to the API
  result
}

# --- 3. Pull data for all states ---------------------------------------------

message("Starting SDA pull for ", length(state_codes), " states...")

results_list <- map(state_codes, query_state)

# --- 4. Combine and compute county-level shares ------------------------------

raw_df <- bind_rows(results_list)

# comp_acres is already summed at the county x compname level from the SQL.
# Now compute total surveyed acres per county and derive share.
county_totals <- raw_df |>
  group_by(areasymbol, areaname) |>
  summarise(total_acres = sum(comp_acres, na.rm = TRUE), .groups = "drop")

final_df <- raw_df |>
  left_join(county_totals, by = c("areasymbol", "areaname")) |>
  mutate(soil_share = comp_acres / total_acres) |>
  arrange(areasymbol, desc(soil_share))

# --- 5. Quick sanity check ---------------------------------------------------
# Shares within each county should sum to ~1.0 (may be slightly off due to
# minor components being excluded and rounding in muareaacres).

share_check <- final_df |>
  group_by(areasymbol) |>
  summarise(total_share = sum(soil_share, na.rm = TRUE)) |>
  filter(abs(total_share - 1) > 0.05)

if (nrow(share_check) > 0) {
  message(
    "Warning: ",
    nrow(share_check),
    " counties have shares that don't sum close to 1.0."
  )
  message("This is expected for counties with large 'Not Completed' areas.")
}

# --- 6. Save output ----------------------------------------------------------

write_parquet(final_df, output_path)

message("\nDone!")
message("Rows: ", nrow(final_df))
message("Counties: ", n_distinct(final_df$areasymbol))
message("Saved to: ", output_path)

# --- 7. Preview --------------------------------------------------------------

message("\nSample output:")
print(head(final_df, 10))

# =============================================================================
# OUTPUT NOTES
# =============================================================================
# The CSV will have one row per county x soil series combination.
#
# To get a WIDE format (one column per soil order, values = share), use:
#
#   library(tidyr)
#   wide_df <- final_df |>
#     group_by(areasymbol, areaname, taxorder) |>
#     summarise(order_share = sum(soil_share), .groups = "drop") |>
#     pivot_wider(names_from = taxorder, values_from = order_share,
#                 values_fill = 0)
#
# To merge with a county FIPS crosswalk:
#   The first 2 characters of areasymbol = state abbreviation
#   The last 3 characters = county FIPS within state (e.g. "113" for Yolo Co.)
#   Full 5-digit FIPS needs the state FIPS number, which you can join from
#   a standard state FIPS lookup table.
# =============================================================================
