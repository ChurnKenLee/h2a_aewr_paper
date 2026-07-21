# C00: Shared helpers

These files contain reusable transformations only. They do not load project
data, write artifacts, or attach packages globally.

| File | Responsibility |
| --- | --- |
| `fips.R` | Normalize state and county FIPS identifiers |
| `bea_county_crosswalk.R` | Harmonize BEA county codes to the project vintage |
| `analysis_helpers.R` | Define the main analysis sample and read the county map |
| `entropy_calibration.R` | Numerical routines shared by IV calibration scripts |

Scripts source only the helper files they use.
