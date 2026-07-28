#!/usr/bin/env bash

source "$(dirname "$0")/pipeline_helpers.sh"

run_step code/b01_derived/01_h2a_aggregation_nodupes.R
run_step code/b01_derived/02_price_index_nass_synthetic_cdl.py
run_step code/b01_derived/03_01_acs_extract.R
run_step code/b01_derived/03_02_acs_immigrant_imputation.R
run_step code/b01_derived/03_03_acs_cz_wage_quantile.R
run_step code/b01_derived/04_acs_qcew_crop_animal_employment_ratio.R
run_step code/b01_derived/05_01_acs_ag_wage.R
run_step code/b01_derived/05_02_oews_farm_wages.R
run_step code/b01_derived/05_03_qcew_ag_wages.R
run_step code/b01_derived/06_nawspad_work_hours.R
run_step code/b01_derived/07_h2a_prediction_elastic_net.py
