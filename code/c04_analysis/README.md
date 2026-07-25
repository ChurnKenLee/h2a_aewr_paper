# C04: Analysis and exhibits

These scripts are organized by exhibit family and may be run individually.

| Script | Responsibility |
| --- | --- |
| `01_aewr_descriptives.R` | AEWR distributions, trends, maps, and regional diagnostics |
| `02_h2a_descriptives.R` | H-2A national trends, maps, and predicted-use figures |
| `03_exposure_descriptives.R` | Treatment-exposure maps and DD visualizations |
| `04_main_dd_results.R` | Main DD estimates and robustness checks |
| `05_event_study.R` | Flexible DD event studies and coefficient plots |
| `06_summary_statistics.R` | Main-sample summary statistics |
| `07_price_outcomes.R` | Fisher price-index descriptives and DD estimates |
| `08_labor_share_outcome.R` | Farm labor-share DD estimates |
| `09_stacked_did_matching.R` | Matching and stacked staggered-DiD analysis |
| `10_iv_calibration_diagnostics.R` | Wage/FLS-moment balance and IV support across weight and donor designs |
| `11_iv_first_stages.R` | Real AEWR and p10-bite TWFE first stages across weight and donor designs |
| `12_iv_preferred_descriptives.R` | Preferred-design wage series, scatters, weight changes, maps, and cluster-similarity diagnostics |
| `13_iv_preferred_results.R` | Preferred four-column first stage, outcome-by-outcome 2SLS tables, and regression-variable summary statistics |
| `14_soc_decision_intensity_crosswalk.R` | Validated three-digit SOC decision-intensity crosswalk from the Deming appendix |

`03` reads the regional-trend summary written by `01`. Scripts `10` and `11`
require IV artifacts from C03; the remaining scripts consume the finalized C02
panel and their explicitly listed raw or processed inputs.

Scripts `12` and `13` implement the publication design defined in
`code/c00_shared/iv_preferred_design.R`: five clusters within each AEWR region,
two donor clusters, the BEA-prior exact wage-plus-seasonal instrument, all
valid counties from 2011 onward, county and year fixed effects, and standard
errors clustered by commuting zone (`cz_id`). The AEWR-region-by-`k = 5`
assignment identifier remains available for diagnostics and a possible later
AEWR-region-level AAIW/CCW correction. Script `11` remains the broader
sensitivity grid rather than the preferred table.

The national wage series in script `12` compares wage-only, exact-seasonal,
interval-seasonal, and `rho = 1` soft-seasonal weights alongside the
equal-weight donor comparison and AEWR. The CZ-year scatterplots use the
preferred exact-seasonal weights.
