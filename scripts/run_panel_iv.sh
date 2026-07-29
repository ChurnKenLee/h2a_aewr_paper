#!/usr/bin/env bash

source "$(dirname "$0")/pipeline_helpers.sh"

run_step code/designs/panel_iv/01_build_county_features.R
run_step code/designs/panel_iv/02_cluster_target_units.R
run_step code/designs/panel_iv/03_build_fls_frame.py
run_step code/designs/panel_iv/04_recover_fls_geography.py
run_step code/designs/panel_iv/05_construct_instruments.R
run_step code/designs/panel_iv/06_build_county_year_panel.R
run_step code/designs/panel_iv/07_estimate_panel_iv.R
run_step code/designs/panel_iv/08_generate_figures.R
