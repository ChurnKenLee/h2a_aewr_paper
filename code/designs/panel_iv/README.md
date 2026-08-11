# Panel IV

This branch consumes `data/processed/county_year_panel.parquet`, constructs
the panel-IV instruments and analysis panel, and produces the retained
estimates and diagnostics.

## Run

From the repository root:

```sh
./scripts/run_panel_iv.sh
```

## Order and artifacts

| Script | Principal output |
| --- | --- |
| `01_build_county_features.R` | `panel_iv_county_features.parquet` |
| `02_cluster_target_units.R` | Fixed target/donor subregions and cluster map |
| `03_build_fls_frame.py` | `panel_iv_fls_frame.parquet` |
| `04_recover_fls_geography.py` | Entropy weights and diagnostics |
| `05_construct_instruments.R` | Area frame and cluster-year instruments |
| `06_build_county_year_panel.R` | `data/processed/panel_iv_county_year.parquet` |
| `07_estimate_panel_iv.R` | First stages, 2SLS tables, diagnostics, and summary statistics |
| `08_generate_figures.R` | Diagnostic figures and plotting-data CSVs |

Retained products are written under `outputs/tables` and `outputs/figures`.

## Design documentation

Instrument construction, controls, samples, inference, preferred
specifications, outcomes, and the complete retained-output inventory are
documented in the grounded
[panel-IV design](../../../content/designs/panel-iv.md).
