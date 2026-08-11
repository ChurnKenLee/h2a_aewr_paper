# Standalone utilities

Utilities are intentionally outside the numbered C workflow. Run them only to
refresh an external input.

- `pull_fred_minimum_wages.R` downloads state and federal minimum wages and
  rewrites `data/intermediate/fred_state_minwages.parquet`. It requires
  `FRED_API_KEY_PHIL` and network access.
