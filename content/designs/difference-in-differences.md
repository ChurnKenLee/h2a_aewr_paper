+++
title = "Difference-in-differences"
description = "Treatment construction, estimation samples, fixed effects, and event-study specification."

[extra]
scopes = ["code/designs/did", "scripts/run_did.sh"]
+++

{{ grounding(path="code/designs/did/helpers.R", anchor="did-estimation-contract", sha256="abc47e8490b0857dc03f19e055b214c3a94a4ec866568d0741614007118fdebd") }}

The cross-design distinction between causal outcomes and required model inputs
is maintained in [Causal outcomes and model inputs](@/contracts/causal-outcomes-and-model-inputs.md).

The DiD branch consumes `data/processed/county_year_panel.parquet`. Script `01`
adds the baseline H-2A treatment classification and post-2011 indicator and
writes `data/processed/did_county_year_panel.parquet`.

Scripts `02` through `06` retain exactly four columns for every regression
family: full/no-border samples, each without/with controls. County and year
fixed effects use `county_fips` and `year` directly; inference clusters by the
CZ-by-AEWR-region interaction. The event study uses
`i(year, aewr_cz_p25_l1, ref = 2011)` plus the 2011 slope, so coefficients keep
their original 2011-reference interpretation.

Retained outputs:

- `table_1_main_results.tex`
- `table_2_event_study.tex`
- `table_sumstats_dd_variables.tex`
- `table_fisher_price_dd.tex`
- `table_laborshare_dd.tex`
- `coefplot_dd_controls.png`

Run:

```sh
./scripts/run_did.sh
```
