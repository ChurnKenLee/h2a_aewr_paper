#!/usr/bin/env bash

source "$(dirname "$0")/pipeline_helpers.sh"

run_step code/designs/did/01_build_did_panel.R
run_step code/designs/did/02_main_results.R
run_step code/designs/did/03_event_study.R
run_step code/designs/did/04_summary_statistics.R
run_step code/designs/did/05_fisher_price.R
run_step code/designs/did/06_labor_share.R
