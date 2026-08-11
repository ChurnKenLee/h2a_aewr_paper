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

## Design documentation

The treatment definition, samples, fixed effects, clustering, and event-study
interpretation are documented in the grounded
[DiD design](../../../content/designs/difference-in-differences.md).
