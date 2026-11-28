# Difference-in-differences branch

This branch consumes `data/processed/county_year_panel.parquet`, constructs the
DiD treatment panel, and produces the retained DiD tables and coefficient plot.

## Run

From the repository root:

```sh
./scripts/run_did.sh
```

## Stages

| Script | Responsibility |
| --- | --- |
| `01_build_did_panel.R` | Add the baseline treatment classification and post-2011 indicator |
| `02_main_results.R` | Estimate the main four-column results |
| `03_event_study.R` | Estimate and plot the event-study specification |
| `04_summary_statistics.R` | Produce the DiD summary table |
| `05_fisher_price.R` | Estimate the Fisher-price outcome |
| `06_labor_share.R` | Estimate the labor-share outcome |

The branch writes `data/processed/did_county_year_panel.parquet` and retained
products under `outputs/tables` and `outputs/figures`.

The corrected canonical 2009 AEWR is missing. Its 2010 lag is consequently
missing, so the existing regression complete-case rule excludes those rows.
The shared/DiD panels retain the outcome rows. Rebuild the branch after this
source correction; no treatment thresholds, reference year or clustering rule
changes. The [AEWR input contract](../../../content/contracts/aewr-inputs.md)
documents the unresolved statutory source history.

## Design documentation

The treatment definition, samples, fixed effects, clustering, and event-study
interpretation are documented in the grounded
[DiD design](../../../content/designs/difference-in-differences.md).

The common input calendar is now 2008--2025. Each regression continues to use
its existing complete-case rule, so missing source controls can shorten its
estimation window. After C01/13 restored the `2024 -> 2025` lag for
`wage_p25_l1` (and thus `aewr_cz_p25_l1` in C02), uncontrolled Columns 1 and 3
of `table_1_main_results.tex` and `table_2_event_study.tex` explicitly include
provisional `2025` H-2A start-year observations (subject to FY2027 decision
revision). Controlled Columns 2 and 4 remain capped at `2022` because
`emp_pop_ratio` relies on BEA `CAEMP25N` (discontinued after `2022`),
`table_fisher_price_dd.tex` ends in `2024` (`fisher_index_ppi`),
`table_laborshare_dd.tex` ends in `2022` (BEA `CAINC45`),
`table_sumstats_dd_variables.tex` reports contemporaneous `aewr_cz_p25` (missing
in `2025` because 2025 ACS quantiles are unavailable), and
`coefplot_dd_controls.png` plots the controlled Column 2 event study (through
`2022`). Event-study code applies that same complete-case rule before expanding
year interactions: this avoids a fixest dimension error when an entire year
has missing regressors. It does not fill controls or change the 2011 reference.
