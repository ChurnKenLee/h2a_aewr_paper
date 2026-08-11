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

Manual command:

```sh
Rscript --vanilla code/c02_build/01_build_county_panel.R
```
