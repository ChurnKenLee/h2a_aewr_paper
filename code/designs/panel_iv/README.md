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
| `04_recover_fls_geography.py` | County-keyed features, soft-entropy weights, moment diagnostics, and deterministic draws |
| `05_construct_instruments.R` | County donor frame and cluster-year instruments |
| `06_build_county_year_panel.R` | `data/processed/panel_iv_county_year.parquet` |
| `07_estimate_panel_iv.R` | First stages, 2SLS tables, diagnostics, summary statistics, and email-ready PDF/CSVs |
| `08_generate_figures.R` | QCEW-feature/FLS/OEWS-proxy diagnostic figures and plotting-data CSVs |
| `09_validate_artifacts.py` | Source, calibration, wage-proxy, timing, sample, and retained-table contracts |

Retained products are written under `outputs/tables` and `outputs/figures`.
`panel_iv_email_results.pdf` contains one landscape page per specification,
with outcomes in columns. Four matching
`panel_iv_email_results_spec*.csv` files provide editable transposed tables.

## Instrument contract

The Census hired-worker benchmark supplies a county prior. Annual county paths
use disclosed all-ownership QCEW 111+112 employment first, QWI next, and BEA
hired-farm jobs last, followed by state raking. One normalized county-weight
distribution is estimated per AEWR region and source year for each
specification; the four survey weeks never receive separate distributions and
FLS worker totals never set regional employment levels.

The wage-only specification softly targets annual FLS combined
field/livestock wages using county-mapped OEWS hourly wages. The preferred
specification adds three independent FLS-worker/QCEW-employment seasonal
contrasts and one undivided field/livestock composition residual per quarter:

```text
(FLS field wage - FLS combined wage) * QCEW 111 employment
+ (FLS livestock wage - FLS combined wage) * QCEW 112 employment
```

Every active moment is standardized under the county prior; zero-variation or
unavailable published moments remain explicit inactive diagnostics. USDA did
not conduct the April 2011 FLS survey, so the corresponding seasonal contrast
and composition moment are inactive rather than imputed.

Donor wage levels use only the shared county-mapped OEWS-area Big-Six mean
hourly-wage proxy. This is a reporting-area local-labor-market measure assigned
through the shared county mapping, not a direct county farm-wage observation.
QCEW remains primary for the county employment path, quarterly seasonality,
and field/livestock composition; QCEW wage bills and BEA wages are not donor
wage fallbacks. The source-year hourly proxy is PPI-deflated, donor county
weights are restricted and renormalized after the existing target-area overlap
exclusion, and policy year always equals source year plus one.

The retained instrument fields are `z_wage_only_real` and
`z_wage_seasonal_composition_real`. Columns 1--4 remain wage-only,
seasonal/composition, wage-only with controls, and seasonal/composition with
controls; column 4 remains preferred.

## Design documentation

Instrument construction, controls, samples, inference, preferred
specifications, outcomes, and the complete retained-output inventory are
documented in the grounded
[panel-IV design](../../../content/designs/panel-iv.md).
