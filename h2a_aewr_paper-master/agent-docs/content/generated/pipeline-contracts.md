+++
title = "Pipeline contracts"
description = "Runner-to-step projection generated from the supported shell entry points."
weight = 2
+++

> [!NOTE]
> Generated file. Change repository sources or `scripts/agent_grounding.py`, then regenerate.

## Runner summary

| Runner | Direct steps | Child runners |
|---|---:|---:|
| `scripts/run_all.sh` | 0 | 7 |
| `scripts/run_derived.sh` | 12 | 0 |
| `scripts/run_descriptives.sh` | 2 | 0 |
| `scripts/run_did.sh` | 6 | 0 |
| `scripts/run_h2a_prediction_cutoffs.sh` | 1 | 0 |
| `scripts/run_mundlak_chamberlain.sh` | 8 | 0 |
| `scripts/run_optional_sources.sh` | 3 | 0 |
| `scripts/run_panel_iv.sh` | 8 | 0 |
| `scripts/run_shared_panel.sh` | 14 | 0 |
| `scripts/run_sources.sh` | 24 | 0 |

## `scripts/run_all.sh`

Calls, in source order:

1. `scripts/run_sources.sh`
1. `scripts/run_derived.sh`
1. `scripts/run_shared_panel.sh`
1. `scripts/run_descriptives.sh`
1. `scripts/run_did.sh`
1. `scripts/run_panel_iv.sh`
1. `scripts/run_mundlak_chamberlain.sh`

## `scripts/run_derived.sh`

Direct `run_step` targets, in execution order:

1. `code/b01_derived/01_h2a_aggregation_nodupes.R`
1. `code/b01_derived/02_price_index_nass_synthetic_cdl.py`
1. `code/b01_derived/03_01_acs_extract.R`
1. `code/b01_derived/03_02_acs_immigrant_imputation.R`
1. `code/b01_derived/03_03_acs_cz_wage_quantile.R`
1. `code/b01_derived/04_acs_qcew_crop_animal_employment_ratio.R`
1. `code/b01_derived/05_01_acs_ag_wage.R`
1. `code/b01_derived/05_02_oews_farm_wages.R`
1. `code/b01_derived/05_03_qcew_ag_wages.R`
1. `code/b01_derived/06_nawspad_work_hours.R`
1. `code/b01_derived/07_h2a_prediction_elastic_net.py`
1. `code/b01_derived/08_h2a_prediction_from_estimated_weights.py`

## `scripts/run_descriptives.sh`

Direct `run_step` targets, in execution order:

1. `code/descriptives/01_h2a_workers.R`
1. `code/descriptives/02_aewr_p25_map.R`

## `scripts/run_did.sh`

Direct `run_step` targets, in execution order:

1. `code/designs/did/01_build_did_panel.R`
1. `code/designs/did/02_main_results.R`
1. `code/designs/did/03_event_study.R`
1. `code/designs/did/04_summary_statistics.R`
1. `code/designs/did/05_fisher_price.R`
1. `code/designs/did/06_labor_share.R`

## `scripts/run_h2a_prediction_cutoffs.sh`

Direct `run_step` targets, in execution order:

1. `code/b01_derived/07_h2a_prediction_elastic_net.py`

## `scripts/run_mundlak_chamberlain.sh`

Direct `run_step` targets, in execution order:

1. `code/designs/mundlak_chamberlain/01_build_panel.R`
1. `code/designs/mundlak_chamberlain/01_01_build_specification_registry.R`
1. `code/designs/mundlak_chamberlain/02_estimate_models.R`
1. `code/designs/mundlak_chamberlain/02_01_estimate_specification_program.R`
1. `code/designs/mundlak_chamberlain/03_01_report_specification_program.R`
1. `code/designs/mundlak_chamberlain/04_01_diagnostics.R`
1. `code/designs/mundlak_chamberlain/05_generate_tables.py`
1. `code/designs/mundlak_chamberlain/06_01_validate_specification_program.R`

## `scripts/run_optional_sources.sh`

Direct `run_step` targets, in execution order:

1. `code/a01_sources/10_mymarketnews_get_reports.py`
1. `code/a01_sources/11_risk_management_agency_summary_of_business_binaries.py`
1. `code/a01_sources/12_risk_management_agency_actuarial_data_master_binaries.py`

## `scripts/run_panel_iv.sh`

Direct `run_step` targets, in execution order:

1. `code/designs/panel_iv/01_build_county_features.R`
1. `code/designs/panel_iv/02_cluster_target_units.R`
1. `code/designs/panel_iv/03_build_fls_frame.py`
1. `code/designs/panel_iv/04_recover_fls_geography.py`
1. `code/designs/panel_iv/05_construct_instruments.R`
1. `code/designs/panel_iv/06_build_county_year_panel.R`
1. `code/designs/panel_iv/07_estimate_panel_iv.R`
1. `code/designs/panel_iv/08_generate_figures.R`

## `scripts/run_shared_panel.sh`

Direct `run_step` targets, in execution order:

1. `code/c01_clean/01_county_price_index.R`
1. `code/c01_clean/02_commuting_zone_crosswalk.R`
1. `code/c01_clean/03_producer_price_index.R`
1. `code/c01_clean/04_state_minimum_wages.R`
1. `code/c01_clean/05_h2a_county_panels.R`
1. `code/c01_clean/06_cdl_county_crop_acres.R`
1. `code/c01_clean/07_census_agriculture.R`
1. `code/c01_clean/08_aewr_panel.R`
1. `code/c01_clean/09_bea_employment.R`
1. `code/c01_clean/10_bea_farm_income.R`
1. `code/c01_clean/11_census_population.R`
1. `code/c01_clean/12_county_year_backbone.R`
1. `code/c01_clean/13_merge_county_panel.R`
1. `code/c02_build/01_build_county_panel.R`

## `scripts/run_sources.sh`

Direct `run_step` targets, in execution order:

1. `code/a01_sources/00_crosswalk_harmonization.py`
1. `code/a01_sources/01_aewr_extract_tables.py`
1. `code/a01_sources/02_01_h2a_match_locations.py`
1. `code/a01_sources/02_02_h2a_clean_unmatched_locations_using_gemini.py`
1. `code/a01_sources/02_03_h2a_use_places_api_to_get_county.py`
1. `code/a01_sources/02_04_h2a_match_employers.py`
1. `code/a01_sources/03_01_nass_extract_quickstats.py`
1. `code/a01_sources/03_02_nass_select_quickstats_obs.py`
1. `code/a01_sources/03_03_nass_census_worker_duration.py`
1. `code/a01_sources/04_01_qcew_create_binaries.py`
1. `code/a01_sources/04_02_qcew_quarterly_employment.py`
1. `code/a01_sources/04_03_qwi_quarterly_employment.py`
1. `code/a01_sources/05_01_oews_geographic_crosswalk.py`
1. `code/a01_sources/05_02_oews_binaries.py`
1. `code/a01_sources/06_01_croplandcros_cdl_aggregate_using_exactextract.py`
1. `code/a01_sources/06_02_croplandcros_cdl_extract_crop_name.py`
1. `code/a01_sources/06_03_croplandcros_cdl_nass_quickstats_crop_crosswalk.py`
1. `code/a01_sources/06_04_croplandcros_cdl_calculate_synthetic_price_and_yield.py`
1. `code/a01_sources/06_05_croplandcros_cdl_crop_type.py`
1. `code/a01_sources/07_state_minimum_wages.py`
1. `code/a01_sources/08_bea_farm_nonfarm_emp.py`
1. `code/a01_sources/09_01_h2a_prediction_pull_noaa.py`
1. `code/a01_sources/09_02_h2a_prediction_pull_gnatsgo.py`
1. `code/a01_sources/13_farm_labor_survey.py`
