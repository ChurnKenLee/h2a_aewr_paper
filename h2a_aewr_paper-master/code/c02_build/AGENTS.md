# C02 shared-panel instructions

- `01_build_county_panel.R` owns `data/processed/county_year_panel.parquet` and nothing design-specific.
- Reusable outcomes, controls, consecutive-year lags, baseline exposures, wage gaps, cropland eligibility, border-CZ status, and stable identifiers may live here.
- Treatment classification, post indicators, event time, fixed-effect factors, year dummies, target clusters, excluded instruments, design weights, and design-specific sample filters must be created downstream.
- The selected H-2A propensity is a static one-row-per-county score joined by the explicit global cutoff and repeated across years. Do not turn it into a rolling annual score.
- Any schema change requires review of descriptives and all three supported design branches.
