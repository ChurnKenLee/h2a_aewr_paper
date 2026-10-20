#!/usr/bin/env bash

source "$(dirname "$0")/pipeline_helpers.sh"

run_step code/designs/border_discontinuity/01_build_county_features.py
run_step code/designs/border_discontinuity/02_construct_geographies.py
run_step code/designs/border_discontinuity/03_build_panel.py

if [[ ${H2A_BORDER_INPUTS_ONLY:-0} == 1 ]]; then
  run_step code/designs/border_discontinuity/06_validate_artifacts.py
  exit 0
fi

run_step code/designs/border_discontinuity/04_estimate_models.R
run_step code/designs/border_discontinuity/05_generate_diagnostics.R
run_step code/designs/border_discontinuity/07_estimate_redesign.R
run_step code/designs/border_discontinuity/08_power_and_report.R
run_step code/designs/border_discontinuity/06_validate_artifacts.py
run_step code/designs/border_discontinuity/12_construct_statewide_geography.py
run_step code/designs/border_discontinuity/13_statewide_power.R
