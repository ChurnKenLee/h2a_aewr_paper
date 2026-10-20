#!/usr/bin/env bash

source "$(dirname "$0")/pipeline_helpers.sh"

# Cached source corrections and their affected downstream consumers. Public
# point-geocoding acquisition requires H2A_RUN_PUBLIC_GEOCODER=1 explicitly.
run_step code/a01_sources/01_aewr_extract_tables.py
run_step code/a01_sources/03_03_nass_census_worker_duration.py
run_step code/b01_derived/01_h2a_aggregation_nodupes.R
run_step code/b01_derived/01b_h2a_worksite_exposure.py
run_step code/a01_sources/02_05_h2a_geocode_worksites.py

./scripts/run_shared_panel.sh
./scripts/run_descriptives.sh
./scripts/run_did.sh
./scripts/run_panel_iv.sh
./scripts/run_mundlak_chamberlain.sh
./scripts/run_border_discontinuity.sh

run_step code/designs/border_discontinuity/09_compare_data_inputs.R
run_step code/designs/border_discontinuity/10_report_data_inputs.py
run_step code/designs/border_discontinuity/11_validate_data_inputs.py
