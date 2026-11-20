#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: bash explorations/chapter_directions/run_interim.sh /scratch/outside/repository" >&2
  exit 2
fi
export CHAPTER_SCRATCH="$1" UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/h2a-uv-cache}"
export PATH="$PWD/.devenv/profile/bin:$PATH"
if [[ -z "${R_LIBS_USER:-}" ]]; then
  cd_libraries=("$PWD"/renv/library/*/R-*/*)
  export R_LIBS_USER="${cd_libraries[0]}"
fi
sha256sum --check explorations/chapter_directions/earnings_amendment.sha256
sha256sum --check explorations/chapter_directions/interim_amendment.sha256
uv run --no-sync marimo export script explorations/chapter_directions/05_extract_earnings.py -o "$CHAPTER_SCRATCH/05_extract_earnings.flat.py" --force
uv run --no-sync marimo export script explorations/chapter_directions/09_extract_interim.py -o "$CHAPTER_SCRATCH/09_extract_interim.flat.py" --force
if [[ ! -f "$CHAPTER_SCRATCH/earnings_interim.parquet" ]]; then
  uv run --no-sync python "$CHAPTER_SCRATCH/09_extract_interim.flat.py"
fi
Rscript --vanilla explorations/chapter_directions/10_estimate_interim.R "$CHAPTER_SCRATCH"
Rscript --vanilla explorations/chapter_directions/tests/check_interim.R "$CHAPTER_SCRATCH"
Rscript --vanilla explorations/chapter_directions/11_report_interim.R "$CHAPTER_SCRATCH"
