#!/usr/bin/env bash

source "$(dirname "$0")/pipeline_helpers.sh"

for script in code/c01_clean/*.R; do
  run_step "$script"
done

for script in code/c02_build/*.R; do
  run_step "$script"
done

for script in code/c03_iv/*.R; do
  run_step "$script"
done

for script in code/c04_analysis/*.R; do
  run_step "$script"
done
