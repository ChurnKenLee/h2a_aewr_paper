#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

"$SCRIPT_DIR/run_sources.sh"
"$SCRIPT_DIR/run_derived.sh"
"$SCRIPT_DIR/run_analysis.sh"
