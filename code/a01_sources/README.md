# A01: Source data

These scripts acquire or normalize one external source at a time. Run them
individually from the repository root; the numbering groups related scripts
but is not a command that runs the whole folder.

## Manual execution

Most files are Marimo applications:

```sh
uv run marimo edit code/a01_sources/01_aewr_extract_tables.py
```

The two ordinary Python programs can be run directly:

```sh
uv run python code/a01_sources/03_03_nass_census_hired_worker_duration.py
uv run python code/a01_sources/04_02_qcew_quarterly_employment.py
```

Old root-level A paths map here by dropping the leading `a` from the filename.
The sole numbering normalization is `a04_qcew_create_binaries.py`, now
`04_01_qcew_create_binaries.py`.

## Script contracts

| Script | Responsibility | Primary output |
| --- | --- | --- |
| `00_crosswalk_harmonization.py` | Normalize the Census county adjacency source | `county_adjacency2010.parquet` |
| `01_aewr_extract_tables.py` | Assemble historical AEWR tables | `aewr.parquet` |
| `02_01_h2a_match_locations.py` | Standardize worksites and assign county FIPS | H-2A and Addendum B files with FIPS |
| `02_02_h2a_clean_unmatched_locations_using_gemini.py` | Suggest corrections for unmatched locations | Unmatched-location suggestion CSVs |
| `02_03_h2a_use_places_api_to_get_county.py` | Resolve suggested locations through Places | Place-ID and address-component caches |
| `03_01_nass_extract_quickstats.py` | Partition raw QuickStats census and survey extracts | `qs_census_*.parquet`, `qs_survey_*.parquet` |
| `03_02_nass_select_quickstats_obs.py` | Select and harmonize downstream NASS observations | Selected census and survey Parquet files |
| `03_03_nass_census_hired_worker_duration.py` | Derive county hired-worker duration shares | `census_ag_hired_worker_duration_county.parquet` |
| `04_01_qcew_create_binaries.py` | Combine annual QCEW archives | `qcew.parquet` |
| `04_02_qcew_quarterly_employment.py` | Extract quarterly calibration moments | `qcew_county_ag_quarterly_employment.parquet` |
| `05_01_oews_geographic_crosswalk.py` | Harmonize OEWS reporting areas to counties | `oews_area_definitions.parquet` |
| `05_02_oews_binaries.py` | Combine historical OEWS releases | `oews.parquet` |
| `06_01_croplandcros_cdl_aggregate_using_exactextract.py` | Aggregate CDL rasters to county crop counts | County crop-pixel counts |
| `06_02_croplandcros_cdl_extract_crop_name.py` | Convert county crop counts to labeled acres | `croplandcros_county_crop_acres.parquet` |
| `06_03_croplandcros_cdl_nass_quickstats_crop_crosswalk.py` | Generate candidate NASS-to-CDL mappings | `nass_to_cdl_mappings.jsonl` |
| `06_04_croplandcros_cdl_calculate_synthetic_price_and_yield.py` | Review mappings and construct synthetic crop series | NASS-CDL crosswalk and synthetic price/yield files |
| `06_05_croplandcros_cdl_crop_type.py` | Classify CDL crops for analysis | `cdl_crop_type.parquet` |
| `07_state_minimum_wages.py` | Assemble state-year minimum wages | `state_year_min_wage.parquet` |
| `08_bea_farm_nonfarm_emp.py` | Extract county BEA farm employment and income sources | Trimmed BEA files and farm employment |
| `09_01_h2a_prediction_pull_noaa.py` | Build annual county climate predictors | Climate basis Parquet |
| `09_02_h2a_prediction_pull_gnatsgo.py` | Build county soil predictors | gNATSGO soil-cell Parquet |
| `10_mymarketnews_get_reports.py` | Download USDA market reports | Manifest, headers, and report partitions |
| `11_risk_management_agency_summary_of_business_binaries.py` | Convert RMA Summary of Business archives | Coverage and type-practice-unit Parquet files |
| `12_risk_management_agency_actuarial_data_master_binaries.py` | Inspect ADM price records and layouts | No persistent artifact yet |
| `13_farm_labor_survey.py` | Parse FLS wage and worker tables | Regional, state, quarterly, and auxiliary FLS files |

## Ordering and credentials

- The H-2A geocoding loop is `02_01`, `02_02`, `02_03`, then `02_01` again
  to incorporate the completed caches into the final FIPS outputs.
- Within the NASS family, run `03_01` before `03_02`, and `03_02` before
  `03_03`.
- Within the CDL family, run `06_01` through `06_05` in numeric order.
- Run `00` before the OEWS crosswalk and county-vintage harmonization steps.
- `GOOGLE_GEMINI_API_KEY` is used by `02_02`, `03_02`, `06_03`, and `06_04`;
  `GOOGLE_PLACES_API_KEY` by `02_03`; `FRED_API_KEY` by `07`; and
  `USDA_MYMARKETNEWS_API_KEY` by `10`.
