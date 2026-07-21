# C01: Clean source panels

Run scripts from the repository root. The numbering is the recommended reading
order; independent source panels may be rebuilt separately.

| Script | Primary output |
| --- | --- |
| `01_county_price_index.R` | `nass_fisher_price_index.parquet` |
| `02_commuting_zone_crosswalk.R` | Full and compact 2010 CZ crosswalks |
| `03_producer_price_index.R` | `ppi_2012.parquet` |
| `04_state_minimum_wages.R` | `state_real_minwages.parquet` |
| `05_h2a_county_panels.R` | Census-period, annual, and national H-2A panels |
| `06_cdl_county_crop_acres.R` | County-year crop-type acreage |
| `07_census_agriculture.R` | Annual and 2007-baseline cropland panels |
| `08_aewr_panel.R` | State-year and region-year AEWR panels |
| `09_bea_employment.R` | Annual county BEA employment panel |
| `10_bea_farm_income.R` | Annual real and nominal BEA farm-income panel |
| `11_census_population.R` | Annual county population panel |
| `12_county_year_backbone.R` | Balanced 2008–2022 county-year backbone |

`04`, `08`, and `10` require the PPI from `03`. The minimum-wage script consumes
the existing `fred_state_minwages.parquet`; refresh it separately with
`code/utilities/pull_fred_minimum_wages.R` only when needed.
