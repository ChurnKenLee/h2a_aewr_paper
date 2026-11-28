# Purpose: Retired duplicate FRED minimum-wage fetcher.
# Superseded by: code/a01_sources/07_state_minimum_wages.py
#   (writes data/intermediate/state_year_min_wage.parquet, consumed by
#    code/c01_clean/04_state_minimum_wages.R).

stop(
  paste(
    "code/utilities/pull_fred_minimum_wages.R is retired.",
    "Use code/a01_sources/07_state_minimum_wages.py (FRED_API_KEY) to refresh",
    "data/intermediate/state_year_min_wage.parquet instead."
  ),
  call. = FALSE
)
