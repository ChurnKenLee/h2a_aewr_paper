#!/usr/bin/env bash

source "$(dirname "$0")/pipeline_helpers.sh"

# docs-ground:start shared-panel-runner
for script in code/c01_clean/*.R; do
  run_step "$script"
done

for script in code/c02_build/*.R; do
  run_step "$script"
done
run_step code/c02_build/02_validate_source_coverage.py
# docs-ground:end shared-panel-runner
