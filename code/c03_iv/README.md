# C03: Instrumental-variable construction

The calibration weights are interpreted as an entropy-projection proxy for
realized FLS sampling composition. They are not claimed to recover the
unobserved original survey weights. The publication specification minimizes
KL divergence from the BEA farm-employment prior subject to exact regional FLS
wage and January/April/July worker-share constraints.

Every exact cell first receives a linear-program feasibility check. An
infeasible cell is labeled `exact_infeasible` and receives no weights; it is
never silently softened. The sensitivity set retains separately named QWI and
publicly bridged Census duration constraints, 90-percent source-discrepancy
intervals, low penalties `rho = 0.01, 0.03, 0.10, 0.30, 1`, and Census/QWI
alternative priors. H-2A administrative records do not enter calibration.

Only the soft sensitivity specifications use `rho`. After standardizing each
auxiliary moment, they minimize
`KL(weights || prior) + rho / 2 * sum(moment imbalance^2)`, while retaining
the wage moment as an exact constraint. Thus `rho` is the strength of the
auxiliary-balance penalty—not a correlation, sampling fraction, or weight.
As `rho` approaches zero, the solution approaches the wage-only projection;
larger values pursue the auxiliary targets more aggressively and can sharply
reduce effective area count. The preferred exact specification has no `rho`:
all four published FLS targets are constraints.

Run `02` through `10` in order after the C02 panel and A-stage QCEW/Census
extracts exist. Run the QWI extractor first to enable suppression fills,
duration sensitivities, and interval bands.

| Script | Responsibility | Primary output |
| --- | --- | --- |
| `01_proxy_wage_diagnostics.R` | Compare FLS changes with public wage proxies | Diagnostic figures |
| `02_county_prior_weights.R` | Construct the BEA primary and separate Census/QWI sensitivity priors | `fls_county_weight.parquet` |
| `03_oews_area_prior_weights.R` | Allocate each county prior to OEWS areas | Area-prior parquets |
| `04_auxiliary_moments.R` | Combine QCEW with QWI suppression fills, build QWI persistence, bridge Census duration using public data only, and estimate fixed interval bands | Auxiliary-moment, bridge, band, and diagnostic parquets |
| `05_wage_entropy_calibration.R` | Construct the exact wage-only benchmark | Wage-calibrated weights |
| `06_soft_entropy_calibration.R` | Construct exact, interval, and low-rho entropy projections | Calibrated weights and diagnostics |
| `07_build_cz_features.R` | Build fixed crop, climate, and soil features | `iv_county_features.parquet` |
| `08_cluster_cz_donor_units.R` | Cluster CZ-region units for `k = 2, 3, 4, 5` and rank donor clusters | Feature, cluster, donor, diagnostic, and map artifacts |
| `09_construct_donor_instruments.R` | Form entropy-weighted instruments and equal-weight donor comparisons by calibration, `k`, and donor-set size | `iv_oews_entropy_long.parquet` |
| `10_attach_instruments_to_panel.R` | Attach all instruments to the analysis panel | `processed/county_df_analysis_year_iv.parquet` |

Fixed calibration labels are:

- `wage_only_exact`
- `wage_seasonal_exact` (publication specification)
- `wage_seasonal_qwi_duration_exact`
- `wage_seasonal_census_duration_exact`
- `wage_seasonal_interval`
- `wage_seasonal_soft_rho{001,003,010,030,100}`

Alternative-prior exact instruments append
`_prior_{census_workers,census_payroll,qwi_employment}`. Explicit wage-only
instrument columns end in `_k{k}_d{d}_g100`; other columns append their fixed
calibration label. The original `k = 2`, farthest-one wage-only column names
remain as aliases.

For the unweighted comparison, script `09` first forms an OEWS wage for each
eligible donor county, averages counties equally within each donor CZ-region
unit, and then averages the donor units equally. Weighted and unweighted
series use identical donor clusters and OEWS-area overlap exclusions.

The preferred publication design uses `k = 5`, the two furthest donor clusters,
the BEA prior, and `wage_seasonal_exact`; its names are centralized in
`code/c00_shared/iv_preferred_design.R`. Preferred tables cluster standard
errors by commuting zone (`cz_out10`).
