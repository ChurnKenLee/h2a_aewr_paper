# C00: Shared helpers

These files contain reusable transformations only. They do not load project
data, write artifacts, or attach packages globally.

| File | Responsibility |
| --- | --- |
| `geography.R` | Normalize and validate canonical geographic identifiers |
| `bea_county_crosswalk.R` | Harmonize BEA county codes to the project vintage |
| `analysis_helpers.R` | Define the main analysis sample and read the county map |
| `entropy_calibration.R` | Exact, interval, and soft KL-projection routines plus LP feasibility checks |
| `auxiliary_moment_helpers.R` | Public-data duration bridge and interval-band helpers |
| `iv_preferred_design.R` | Define the preferred IV instruments, controls, period, and CZ inference cluster |

Scripts source only the helper files they use. `iv_preferred_design.R` is the
single source of truth for the publication IV:
five clusters per AEWR region, the two furthest donor clusters, full wage-gap
closure, exact January/April/July FLS moment constraints, and the BEA farm
employment prior. The resulting weights are an entropy-projection proxy for
the realized FLS sampling composition, not identified original survey weights.
