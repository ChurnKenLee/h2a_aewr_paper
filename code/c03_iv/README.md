# C03: Instrumental-variable construction

The active instrument fixes the entropy weights at the exact, 100% regional
wage-gap target and varies the number of within-AEWR donor clusters over
`k = 2, 3, 4, 5`. Within each partition it uses the furthest one, furthest two,
and so forth through all `k - 1` non-target clusters. Construct it by running
`02`, `03`, `05`, and `07` through `10` in that order after the C02 panel
exists. The proxy comparison in `01` is optional.

| Script | Responsibility | Primary output |
| --- | --- | --- |
| `01_proxy_wage_diagnostics.R` | Compare FLS changes with public wage proxies | Diagnostic figures |
| `02_county_prior_weights.R` | Construct county FLS contribution priors | `fls_county_weight.parquet` |
| `03_oews_area_prior_weights.R` | Allocate county priors to OEWS areas | Area-prior parquets |
| `05_wage_entropy_calibration.R` | Calibrate to regional wage targets | Wage-calibrated weights |
| `07_build_cz_features.R` | Build fixed crop, climate, and soil features | `iv_county_features.parquet` |
| `08_cluster_cz_donor_units.R` | Cluster CZ-region units for `k = 2, 3, 4, 5` and rank donor clusters | Cluster, donor, diagnostic, and map artifacts |
| `09_construct_donor_instruments.R` | Form full-gap wage-only instruments by `k` and donor-set size | `iv_oews_entropy_long.parquet` |
| `10_attach_instruments_to_panel.R` | Attach instruments to the analysis panel | `processed/county_df_analysis_year_iv.parquet` |

`04_auxiliary_moments.R` and `06_soft_entropy_calibration.R` remain available
for later auxiliary-moment sensitivity work but are not inputs to the active
instrument. `07` requires the upstream H-2A prediction climate and soil inputs.
The attached panel retains the original `k = 2`, farthest-one column names as
aliases and adds explicit `_k{k}_d{d}_` instrument columns for every valid
cluster-count (`k`) and cumulative donor-count (`d`) pair.
