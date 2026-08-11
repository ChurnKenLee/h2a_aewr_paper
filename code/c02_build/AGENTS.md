# Shared-panel instructions

- `01_build_county_panel.R` owns
  `data/processed/county_year_panel.parquet` and remains design-neutral.
- Reusable outcomes, controls, consecutive-year lags, baseline exposures, wage
  gaps, cropland eligibility, border-CZ status, and stable identifiers belong
  here.
- Treatments, post indicators, event time, fixed-effect factors, target
  clusters, excluded instruments, design weights, and design samples belong
  downstream.
- The selected H-2A propensity is a static one-row-per-county score joined by
  the explicit global cutoff and repeated across years. Do not make it rolling.
- Schema changes require review of descriptives and every supported design.
