#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/pipeline_helpers.sh"

run_step code/designs/mundlak_chamberlain/01_build_panel.R
run_step code/designs/mundlak_chamberlain/02_estimate_models.R
run_step code/designs/mundlak_chamberlain/03_postestimation.R
run_step code/designs/mundlak_chamberlain/04_diagnostics.R
run_step code/designs/mundlak_chamberlain/05_generate_tables.py
run_step code/designs/mundlak_chamberlain/05_validate.R
