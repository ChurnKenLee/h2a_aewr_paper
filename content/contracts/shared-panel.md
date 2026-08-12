+++
title = "Shared county-year panel"
description = "Design-neutral ownership, prediction semantics, and artifact invariants."

[extra]
scopes = ["code/c01_clean", "code/c02_build", "code/descriptives", "code/designs"]
+++

{{ grounding(path="code/c02_build/01_build_county_panel.R", anchor="shared-panel-contract", sha256="ffb30f884e3403e1fd5c2b88d13d20c21eace1d1e1df0957fe7ac8c24b55aada") }}

For the complete C01 merge, C02 transformation, validation, and rebuild path,
see [Generating the shared county-year panel](@/architecture/shared-panel-generation.md).
For the cross-design outcome inventory and required estimation fields, see
[Causal outcomes and model inputs](@/contracts/causal-outcomes-and-model-inputs.md).

`01_build_county_panel.R` consumes
`data/intermediate/county_year_merged.parquet` and writes:

```text
data/processed/county_year_panel.parquet
```

It owns reusable outcomes, controls, consecutive-year control lags, 2011 farm
employment, AEWR-p25 gaps, cropland eligibility, and border-CZ status. It does
not create any design treatment, post period, fixed-effect factor, year dummy,
target cluster, or instrument.

The panel's static predicted H-2A propensity comes from the one global cutoff
selected by `H2A_PREDICTION_CUTOFF_YEAR` in `code/paths.R`. The corresponding
one-row-per-county score is joined by county and repeated unchanged over panel
years. Predicted counts use fixed 2011 BEA farm employment, and predicted shares
divide by that same stored exposure.

The shared H-2A employer outcome uses the raw balanced-linkage employer count.
Conservative and high-recall counts remain in the panel for
linkage-sensitivity analyses.

The shared panel carries BEA CAEMP25N employment totals, including all-industry
wage-and-salary jobs, and CAINC45 farm wages and wage supplements in nominal
thousands of dollars and PPI-deflated 2012-dollar equivalents. It does not
construct hired farm jobs, average annual farm wages, or average annual farm
compensation. Designs that use those statistics own their denominator and
sample rules.

The shared panel also carries nominal OEWS Big-Six mean hourly- and annual-wage
proxies. The C01 producer employment-weights each wage concept within reporting
areas and uses mapped-township shares when a county-year spans multiple areas.
It retains the primary `oews_area_code`, mapped-area count, observed mapping
shares, and wage-covered area-occupation counts. OEWS source years are not
shifted to AEWR policy years, missing wages are not replaced with zero, and
reporting-area employment is not represented as county employment.

Separate all-ownership QCEW NAICS 111, NAICS 112, and all-sector annual
employment and nominal wage bills also pass through with disclosure flags.
Suppressed totals stay null. The shared layer does not divide wage bills by
employment or combine crop and animal agriculture; downstream designs own any
denominator, disclosure, timing, and sample rules.

Manual command:

```sh
Rscript --vanilla code/c02_build/01_build_county_panel.R
```
