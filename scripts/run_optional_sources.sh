#!/usr/bin/env bash

source "$(dirname "$0")/pipeline_helpers.sh"

run_step code/a01_sources/10_mymarketnews_get_reports.py
run_step code/a01_sources/11_risk_management_agency_summary_of_business_binaries.py
run_step code/a01_sources/12_risk_management_agency_actuarial_data_master_binaries.py
