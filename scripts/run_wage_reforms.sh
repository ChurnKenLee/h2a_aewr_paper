#!/usr/bin/env bash

source "$(dirname "$0")/pipeline_helpers.sh"

export MPLCONFIGDIR=${MPLCONFIGDIR:-"$PIPELINE_TMP/matplotlib"}

# docs-ground:start wage-reforms-runner
run_step code/designs/wage_reforms/01_build.py
run_step code/designs/wage_reforms/02_estimate.py
run_step code/designs/wage_reforms/03_report.py
run_step code/designs/wage_reforms/04_validate.py
# docs-ground:end wage-reforms-runner
