#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/pipeline_helpers.sh"

run_step code/designs/mundlak_chamberlain/01_build_panel.R
run_step code/designs/mundlak_chamberlain/01_01_build_specification_registry.R
MC_BENCHMARK_ONLY=1 \
  run_step code/designs/mundlak_chamberlain/02_estimate_models.R
run_step code/designs/mundlak_chamberlain/02_01_estimate_specification_program.R
run_step code/designs/mundlak_chamberlain/03_01_report_specification_program.R
run_step code/designs/mundlak_chamberlain/04_01_diagnostics.R
run_step code/designs/mundlak_chamberlain/05_generate_tables.py
run_step code/designs/mundlak_chamberlain/06_01_validate_specification_program.R
