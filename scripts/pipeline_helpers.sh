#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

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
      Rscript "$script" "$@"
      ;;
    *.py)
      if grep -q 'marimo\.App' "$script"; then
        local flat_script="$PIPELINE_TMP/$(basename "$script" .py).flat.py"
        uv run marimo export script "$script" --output "$flat_script" --force
        uv run python "$flat_script"
      else
        uv run python "$script" "$@"
      fi
      ;;
    *)
      printf 'Unsupported pipeline file: %s\n' "$script" >&2
      return 2
      ;;
  esac
}
