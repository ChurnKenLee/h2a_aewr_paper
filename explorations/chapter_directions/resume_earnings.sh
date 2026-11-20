#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: bash explorations/chapter_directions/resume_earnings.sh /scratch /path/to/verified_release_metadata.json" >&2
  exit 2
fi
export CHAPTER_SCRATCH="$1" CD_RELEASE_METADATA="$2" CD_EARNINGS_MODE=post CD_ACQUIRE=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/h2a-uv-cache}"
export PATH="$PWD/.devenv/profile/bin:$PATH"
if [[ -z "${R_LIBS_USER:-}" ]]; then
  cd_libraries=("$PWD"/renv/library/*/R-*/*)
  export R_LIBS_USER="${cd_libraries[0]}"
fi
if [[ -f "$CHAPTER_SCRATCH/earnings_post_completed.json" ]]; then
  echo "Full window already estimated; do not acquire or inspect a second vintage without a dated amendment" >&2
  exit 1
fi
uv run --no-sync marimo export script explorations/chapter_directions/05_extract_earnings.py -o "$CHAPTER_SCRATCH/05_extract_earnings.flat.py" --force
uv run --no-sync python "$CHAPTER_SCRATCH/05_extract_earnings.flat.py"
Rscript --vanilla explorations/chapter_directions/06_estimate_earnings.R "$CHAPTER_SCRATCH" post
