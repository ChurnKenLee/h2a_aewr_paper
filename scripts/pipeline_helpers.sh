#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C.utf8
export TZ=${TZ:-UTC}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/h2a-uv-cache}

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

shopt -s nullglob
project_r_libraries=(renv/library/*/R-*/*)
shopt -u nullglob
if [[ ${#project_r_libraries[@]} == 1 ]]; then
  export R_LIBS_USER="$PROJECT_ROOT/${project_r_libraries[0]}"
  export RENV_CONFIG_AUTOLOADER_ENABLED=FALSE
fi

PIPELINE_TMP=$(mktemp -d /tmp/h2a-pipeline.XXXXXX)
trap 'rm -rf -- "$PIPELINE_TMP"' EXIT

run_step() {
  local script=$1
  shift

  printf '\n==> %s\n' "$script"
  if [[ ${DRY_RUN:-0} == 1 ]]; then
    return
  fi

  case "$script" in
    *.R)
      Rscript --vanilla "$script" "$@"
      ;;
    *.py)
      if grep -q 'marimo\.App' "$script"; then
        local flat_script="$PIPELINE_TMP/$(basename "$script" .py).flat.py"
        uv run --no-sync marimo export script "$script" --output "$flat_script" --force
        uv run --no-sync python "$flat_script"
      else
        uv run --no-sync python "$script" "$@"
      fi
      ;;
    *)
      printf 'Unsupported pipeline file: %s\n' "$script" >&2
      return 2
      ;;
  esac
}
