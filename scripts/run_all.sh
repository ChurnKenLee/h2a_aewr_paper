#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# docs-ground:start pipeline-order
"$SCRIPT_DIR/run_sources.sh"
"$SCRIPT_DIR/run_derived.sh"
"$SCRIPT_DIR/run_shared_panel.sh"
"$SCRIPT_DIR/run_descriptives.sh"
"$SCRIPT_DIR/run_did.sh"
"$SCRIPT_DIR/run_panel_iv.sh"
"$SCRIPT_DIR/run_mundlak_chamberlain.sh"
# docs-ground:end pipeline-order
