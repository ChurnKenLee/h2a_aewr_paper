# Panel-IV branch

This branch consumes `data/processed/county_year_panel.parquet` directly. It
does not read the DiD panel or any DiD treatment classification.

The design fixes five crop/climate/soil clusters within each AEWR region and
uses the two most dissimilar donor clusters. The primary donor wage applies a
soft-entropy center jointly targeting published FLS quarter-by-duration worker
composition and the annual FLS combined field-and-livestock wage used to set
the following year's AEWR. OEWS big-six agricultural wages supply the
area-level wage moment. The Census hired-worker frame remains an explicit
benchmark; outcomes and first-stage estimates do not enter the fit.

## Order and artifacts

| Script | Principal output |
| --- | --- |
| `01_build_county_features.R` | `panel_iv_county_features.parquet` |
| `02_cluster_target_units.R` | `panel_iv_target_clusters.parquet`, `panel_iv_donor_clusters.parquet`, cluster map |
| `03_build_fls_frame.py` | `panel_iv_fls_frame.parquet` |
| `04_recover_fls_geography.py` | Composition features, wage features, entropy weights, and diagnostics under the `panel_iv_fls_geography_*` prefix |
| `05_construct_instruments.R` | `panel_iv_area_frame.parquet`, `panel_iv_instrument_cluster_year.parquet` |
| `06_build_cluster_year_panel.R` | `data/processed/panel_iv_cluster_year.parquet` |
| `07_estimate_panel_iv.R` | Estimate/AR CSVs, four-column IV table, first-stage figure |
| `08_generate_figures.R` | Five diagnostic figures and their reproducible plotting-data CSVs |

The four retained estimates are primary levels, levels with lagged controls,
no-border levels, and the Census-frame benchmark. All use target-cluster and
year fixed effects, AEWR-region clustered inference, Webb six-point
wild-cluster tests, and Anderson--Rubin sets.

Run:

```sh
./scripts/run_panel_iv.sh
```

Retained final outputs are
`iv_dissimilarity_model_estimates.csv`,
`iv_dissimilarity_ar_intervals.csv`,
`table_iv_dissimilarity_panel.tex`,
`fig_iv_dissimilarity_first_stage.png`, and these six design figures:

- `fig_iv_aewr_region_real_wage_series.png`
- `fig_iv_fls_oews_cz_scatter.png`
- `fig_iv_cz_entropy_weight_changes_pp.png`
- `fig_iv_dissimilarity_clusters_k5.png`
- `fig_iv_california_target_and_donors.png`
- `fig_iv_target_donor_similarity_slopes.png`

The plotting data for the five figures produced by `08_generate_figures.R`
are retained under `outputs/tables` with the corresponding `iv_*.csv` names.
The wage figures use observed OEWS area wages and 2012-dollar PPI deflation.
The map continues to be produced with the fixed `k = 5` assignments in
`02_cluster_target_units.R`, now at 300 DPI.
