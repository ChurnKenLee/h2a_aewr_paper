#!/usr/bin/env bash

source "$(dirname "$0")/pipeline_helpers.sh"
export PATH="$PROJECT_ROOT/.devenv/profile/bin:$PATH"
export POLICY2025_WORK="$PIPELINE_TMP/work"
export POLICY2025_EXTRACTOR="$PIPELINE_TMP/05_extract_earnings.flat.py"
export CD_EARNINGS_MODE=pre

# docs-ground:start policy-2025-runner
run_step code/designs/policy_2025/00_prepare.py
run_step code/designs/policy_2025/01_build_wages.R "$POLICY2025_WORK" "$POLICY2025_WORK/filer"
run_step code/designs/policy_2025/02_estimate_wages.R
run_step code/designs/policy_2025/05_extract_earnings.py
run_step code/designs/policy_2025/06_estimate_earnings.R "$POLICY2025_WORK" pre
run_step code/designs/policy_2025/09_extract_interim.py
run_step code/designs/policy_2025/10_estimate_interim.R
run_step code/designs/policy_2025/tests/check_interim.R
run_step code/designs/policy_2025/07_validate.R
run_step code/designs/policy_2025/08_report.R
run_step code/designs/policy_2025/12_publish.R
# docs-ground:end policy-2025-runner
