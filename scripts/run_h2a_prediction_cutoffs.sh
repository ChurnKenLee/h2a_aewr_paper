#!/usr/bin/env bash

source "$(dirname "$0")/pipeline_helpers.sh"

model_script=code/b01_derived/07_h2a_prediction_elastic_net.py

for cutoff_year in {2008..2025}; do
  printf '\nH2A_CUTOFF_YEAR=%s\n' "$cutoff_year"
  H2A_CUTOFF_YEAR="$cutoff_year" run_step "$model_script"
done
