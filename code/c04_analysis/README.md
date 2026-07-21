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
| `10_iv_calibration_diagnostics.R` | Full-gap wage-only IV support by cluster and donor-set counts |
| `11_iv_first_stages.R` | Real-wage TWFE first stages by cluster and donor-set counts |

`03` reads the regional-trend summary written by `01`. Scripts `10` and `11`
require IV artifacts from C03; the remaining scripts consume the finalized C02
panel and their explicitly listed raw or processed inputs.
