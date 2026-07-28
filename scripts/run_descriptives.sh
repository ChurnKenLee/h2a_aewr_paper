#!/usr/bin/env bash

source "$(dirname "$0")/pipeline_helpers.sh"

run_step code/descriptives/01_h2a_workers.R
run_step code/descriptives/02_aewr_p25_map.R
