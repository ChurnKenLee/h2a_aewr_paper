#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C.utf8
export TZ=${TZ:-UTC}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/h2a-uv-cache}

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

r_library_configured=0

configure_r_library() {
  if [[ $r_library_configured == 1 ]]; then
    return
  fi

  local r_series
  local r_platform
  IFS=$'\t' read -r r_series r_platform < <(
    Rscript --vanilla -e '
      minor <- strsplit(R.version$minor, ".", fixed = TRUE)[[1L]][1L]
      series <- paste(R.version$major, minor, sep = ".")
      cat(paste(series, R.version$platform, sep = "\t"), "\n", sep = "")
    '
  )

  local project_r_libraries=()
  shopt -s nullglob
  project_r_libraries=(renv/library/*/"R-$r_series"/"$r_platform")
  shopt -u nullglob

  if [[ ${#project_r_libraries[@]} != 1 ]]; then
    printf 'Expected one project renv library for R %s (%s), found %d.\n' \
      "$r_series" "$r_platform" "${#project_r_libraries[@]}" >&2
    if [[ ${#project_r_libraries[@]} -gt 0 ]]; then
      printf '  %s\n' "${project_r_libraries[@]}" >&2
    else
      printf 'Run r-renv-restore in the Devenv shell.\n' >&2
    fi
    return 1
  fi

  export R_LIBS_USER="$PROJECT_ROOT/${project_r_libraries[0]}"
  export RENV_CONFIG_AUTOLOADER_ENABLED=FALSE
  r_library_configured=1
}

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
      configure_r_library
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
