#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/pipeline_helpers.sh"

run_step code/welfare/01_narrow_welfare.py
