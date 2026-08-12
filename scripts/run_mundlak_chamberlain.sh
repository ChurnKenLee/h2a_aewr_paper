#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/pipeline_helpers.sh"

run_step code/designs/mundlak_chamberlain/01_build_panel.py
run_step code/designs/mundlak_chamberlain/02_build_registry.py
run_step code/designs/mundlak_chamberlain/03_estimate.py
run_step code/designs/mundlak_chamberlain/04_report.py
run_step code/designs/mundlak_chamberlain/05_validate.py
