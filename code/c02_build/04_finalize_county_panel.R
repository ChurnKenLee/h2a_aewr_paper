# Purpose: Report panel integrity diagnostics and publish the analysis county panel.
# Input: data/intermediate/county_df_classified_year.parquet.
# Output: data/processed/county_df_analysis_year.parquet.
# Run after: 03_classify_treatment_exposure.R.
# Diagnostics are reports, not substantive data repairs.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
library(arrow)
library(dplyr)

# --- Diagnostic: county_df_analysis_year ---
county_df <- read_parquet(path_int("county_df_classified_year.parquet"))
cat("county_df rows:", nrow(county_df), " | cols:", ncol(county_df), "\n")
stopifnot(nrow(county_df) > 10000)
stopifnot(all(
  c(
    "aewr_state_ag_ppi_l1",
    "high_h2a_share_75",
    "high_h2a_share_75_inverse",
    "h2a_cert_share_farm_workers_2011_start_year",
    "county_fe",
    "year_fe",
    "statefips",
    "ln_pop_census",
    "emp_pop_ratio"
  ) %in%
    names(county_df)
))
cat("Diagnostic passed: county_df_analysis_year\n")


write_parquet(county_df, path_processed("county_df_analysis_year.parquet"))
