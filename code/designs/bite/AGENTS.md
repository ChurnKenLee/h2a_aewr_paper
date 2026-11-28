# Full-panel FE-PPML bite design instructions

- Read `README.md` and `content/designs/bite.md` before changing the estimator,
  exposures, inference, or sample conventions.
- This is a standalone branch (`scripts/run_bite.sh`) that remains outside
  `scripts/run_all.sh`. Do not modify `code/c01_clean/`, `code/c02_build/`, or
  `code/designs/did/` from here.
- Preserve the balanced 3,108-county by 18-year (`2008--2025`) full-panel frame,
  string geographic identifiers (`GEO-001`), and canonical `2009 aewr` NULL
  (`AEWR-001`). Never drop `"always takers"` or high-H-2A counties.
- Keep FE-PPML (`fixest::fepois`) for count and hours outcomes; do not replace
  it with `log(Y + 1)` or `asinh` transformations.
- Freeze all baseline exposures (`E_p25`, `E_oews`, all-real `E_sf_log`) over
  `2008/2010` only and keep predetermined `2008--2010` covariates (`X_pre x Year`)
  so `2023--2025` observations are never truncated by discontinued BEA controls.
- Never relabel `E_c x ln(AEWR)` relative differential coefficients as aggregate
  AEWR elasticities, and never translate the standalone `ln(prevailing_ag_min_wage)`
  domestic-floor margin into an AEWR elasticity.
- Always report exact delete-one-cluster jackknife (**CV3J**) SEs, MDE
  (`2.8 x CV3J`), LOO min/max ranges, `1a-PPML` variance retention, Fisher-info
  `G_eff`, Addendum-B extensive-margin thresholds, and the `2011--2019` vs
  `2020--2025` balanced-employer split, even when unfavorable.
- Run `03_validate_bite.R` after any edit and keep `outputs/reports/bite_provenance.json`
  synchronized with the unified `county_year_panel.parquet` SHA-256.
