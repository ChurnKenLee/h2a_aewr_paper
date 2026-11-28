#!/usr/bin/env bash

source "$(dirname "$0")/pipeline_helpers.sh"

# docs-ground:start bite-runner
run_step code/designs/bite/01_build_bite_panel.R
run_step code/designs/bite/02_estimate_bite.R
run_step code/designs/bite/03_validate_bite.R
# docs-ground:end bite-runner
