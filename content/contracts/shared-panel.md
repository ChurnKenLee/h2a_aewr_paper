+++
title = "Shared county-year panel"
description = "Design-neutral ownership, prediction semantics, and artifact invariants."

[extra]
scopes = ["code/c01_clean", "code/c02_build", "code/descriptives", "code/designs"]
+++

{{ <grounding path="code/c02_build/01_build_county_panel.R" anchor="shared-panel-contract" sha256="b8fd6cf733f2ea700c01ab53c1b5d582073d588ee0707dff590b17387ffd2677" /> }}

For the complete C01 merge, C02 transformation, validation, and rebuild path,
see [Generating the shared county-year panel](@/architecture/shared-panel-generation.md).
For the cross-design outcome inventory and required estimation fields, see
[Causal outcomes and model inputs](@/contracts/causal-outcomes-and-model-inputs.md).
Employer identity and count semantics are defined by the
[H-2A worksite-business linkage contract](@/contracts/employer-linkage.md).

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

The shared H-2A business outcome uses the raw balanced v2 count of
positive-certified linked worksite businesses. Conservative and high-recall
counts remain in the panel for linkage-sensitivity analyses. C02 requires the
reviewed FY2008--2024 linkage metadata and fills its unit, version, vintage,
review status, and review-report hash across H-2A zero-use county-years.

The shared panel carries BEA CAEMP25N employment totals, including all-industry
wage-and-salary jobs, and CAINC45 farm wages and wage supplements in nominal
thousands of dollars and PPI-deflated 2012-dollar equivalents. It does not
construct hired farm jobs, average annual farm wages, or average annual farm
compensation. Designs that use those statistics own their denominator and
sample rules.

The canonical 2009 `aewr` is missing because the cached 2010 final rule's
Table 1 reports application wage means, not statutory OES determinations.
`aewr_nonstatutory_application_wage_2009` and its PPI-deflated counterpart
retain that source under explicit semantics. Source status and document ID
pass through C01. The shared panel retains observed 2009 outcomes; designs
must select any wage-complete sample themselves. The source-correct annual
reference history now spans 1990--2025, independently of the USDA validation
workbook's support. See the [AEWR input contract](@/contracts/aewr-inputs.md)
for effective dates, occupational lookups, gaps and competing floors.

Border stage04 explicitly reconstructs its historical contaminated comparator
from the named nonstatutory field; stage07 uses canonical wages and retains
2008--2010 outcomes for FE-only calibration without requiring a wage value.
The 2011--2022 policy treatment is unchanged. Panel IV uses moving OEWS wages
from source year t-1; its former frozen-2009 loading exercise is retired.

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
