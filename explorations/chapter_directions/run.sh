#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: bash explorations/chapter_directions/run.sh /scratch/outside/repository" >&2
  exit 2
fi
export CHAPTER_SCRATCH="$1"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/h2a-uv-cache}"
export PATH="$PWD/.devenv/profile/bin:$PATH"
if [[ -z "${R_LIBS_USER:-}" ]]; then
  cd_libraries=("$PWD"/renv/library/*/R-*/*)
  export R_LIBS_USER="${cd_libraries[0]}"
fi
run() {
  if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
run mkdir -p "$CHAPTER_SCRATCH/logs"
run uv run --no-sync marimo export script explorations/chapter_directions/00_prepare.py -o "$CHAPTER_SCRATCH/00_prepare.flat.py" --force
run uv run --no-sync python "$CHAPTER_SCRATCH/00_prepare.flat.py"
for tau in 2025 2026; do
  run Rscript --vanilla explorations/identification/10_track1_employer_decomposition.R "$CHAPTER_SCRATCH/benchmark" "$tau" "$CHAPTER_SCRATCH/benchmark"
done
run Rscript --vanilla explorations/episode_2025/08_domestic_wage_estimate.R "$CHAPTER_SCRATCH/benchmark" "$CHAPTER_SCRATCH/benchmark/qcew_crop_county_quarter_2014_2024.parquet" 2024
run Rscript --vanilla explorations/chapter_directions/01_build_wages.R "$CHAPTER_SCRATCH" "$CHAPTER_SCRATCH/filer"
run Rscript --vanilla explorations/chapter_directions/02_estimate_wages.R "$CHAPTER_SCRATCH"
run Rscript --vanilla explorations/chapter_directions/03_build_administration.R "$CHAPTER_SCRATCH" "$CHAPTER_SCRATCH/filer"
run Rscript --vanilla explorations/chapter_directions/04_estimate_administration.R "$CHAPTER_SCRATCH"
run uv run --no-sync marimo export script explorations/chapter_directions/05_extract_earnings.py -o "$CHAPTER_SCRATCH/05_extract_earnings.flat.py" --force
if [[ ! -f "$CHAPTER_SCRATCH/earnings_pre.parquet" ]]; then
  run env CD_EARNINGS_MODE=pre uv run --no-sync python "$CHAPTER_SCRATCH/05_extract_earnings.flat.py"
fi
run Rscript --vanilla explorations/chapter_directions/06_estimate_earnings.R "$CHAPTER_SCRATCH" pre
run bash explorations/chapter_directions/validate.sh
run Rscript --vanilla explorations/chapter_directions/08_report.R "$CHAPTER_SCRATCH"
