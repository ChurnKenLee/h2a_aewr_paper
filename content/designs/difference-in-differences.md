+++
title = "Difference-in-differences"
description = "Treatment construction, estimation samples, fixed effects, and event-study specification."

[extra]
scopes = ["code/designs/did", "scripts/run_did.sh"]
+++

{{ <grounding path="code/designs/did/helpers.R" anchor="did-estimation-contract" sha256="4822f93257431fac0a7ffe9399d475d9aa14324d6f76580d666775e9b3b5ab9c" /> }}

The cross-design distinction between causal outcomes and required model inputs
is maintained in [Causal outcomes and model inputs](@/contracts/causal-outcomes-and-model-inputs.md).

The DiD branch consumes `data/processed/county_year_panel.parquet`. Script `01`
adds the baseline H-2A treatment classification and post-2011 indicator and
writes `data/processed/did_county_year_panel.parquet`.

The [AEWR source correction](@/contracts/aewr-inputs.md) makes canonical 2009
wages missing and preserves the corresponding outcome rows. The 2010
lagged-wage regressor is therefore missing, and the unchanged regression
complete-case rule excludes those observations. This changes empirical
support relative to the former nonstatutory-wage benchmark; it does not change
the treatment thresholds, 2011 reference or inference convention.

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

The common input calendar is now 2008--2025. Each regression continues to use
its existing complete-case rule, so missing source controls can shorten its
estimation window. Following the C01/13 lag-construction fix restoring
`2024 -> 2025` `wage_p25_l1` (and `aewr_cz_p25_l1` in C02), uncontrolled
Columns 1 and 3 of `table_1_main_results.tex` (`N = 38,998` and `N = 31,790`)
and `table_2_event_study.tex` explicitly include provisional 2025 H-2A
start-year observations (subject to FY2027 revision) by user instruction.
Controlled Columns 2 and 4 remain capped at 2022 (`N = 32,116` and `N = 26,180`)
because `emp_pop_ratio` uses BEA CAEMP25N (discontinued after 2022);
`table_fisher_price_dd.tex` ends in 2024 (`fisher_index_ppi`),
`table_laborshare_dd.tex` ends in 2022 (BEA CAINC45),
`table_sumstats_dd_variables.tex` reports contemporaneous `aewr_cz_p25` (missing
in 2025 because 2025 ACS quantiles are unavailable), and
`coefplot_dd_controls.png` plots the controlled Column 2 event study through
2022. Event-study code applies that same complete-case rule before expanding
year interactions: this avoids a fixest dimension error when an entire year
has missing regressors. It does not fill controls or change the 2011 reference.
