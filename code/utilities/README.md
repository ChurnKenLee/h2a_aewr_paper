# Standalone utilities

Utilities are intentionally outside the numbered C workflow.

- `pull_fred_minimum_wages.R` is **retired**. State and federal minimum wages
  from FRED release 387 are now owned solely by
  `code/a01_sources/07_state_minimum_wages.py`, which writes
  `data/intermediate/state_year_min_wage.parquet` for
  `code/c01_clean/04_state_minimum_wages.R`.
