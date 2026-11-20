#!/usr/bin/env bash
set -euo pipefail
: "${CHAPTER_SCRATCH:?Set scratch outside the repository}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/h2a-uv-cache}"
export PATH="$PWD/.devenv/profile/bin:$PATH"
if [[ -z "${R_LIBS_USER:-}" ]]; then
  cd_libraries=("$PWD"/renv/library/*/R-*/*)
  export R_LIBS_USER="${cd_libraries[0]}"
fi
mkdir -p "$CHAPTER_SCRATCH/logs"
printf 'check,status,log\n' > "$CHAPTER_SCRATCH/execution_checks.csv"
check() {
  local name="$1"
  shift
  if "$@" > "$CHAPTER_SCRATCH/logs/check_${name}.log" 2>&1; then
    printf '%s,passed,%s\n' "$name" "logs/check_${name}.log" >> "$CHAPTER_SCRATCH/execution_checks.csv"
  else
    printf '%s,failed,%s\n' "$name" "logs/check_${name}.log" >> "$CHAPTER_SCRATCH/execution_checks.csv"
    cat "$CHAPTER_SCRATCH/logs/check_${name}.log" >&2
    return 1
  fi
}
check shell_syntax bash -c 'for script in scripts/*.sh explorations/chapter_directions/*.sh; do bash -n "$script" || exit; done'
check R_syntax Rscript --vanilla -e 'for (f in c(list.files("code", pattern="\\.R$", recursive=TRUE, full.names=TRUE), list.files("explorations/chapter_directions", pattern="\\.R$", recursive=TRUE, full.names=TRUE))) parse(file=f)'
check python_ruff uv run --no-sync ruff check explorations/chapter_directions/00_prepare.py explorations/chapter_directions/05_extract_earnings.py explorations/chapter_directions/tests/test_earnings.py
check marimo_prepare_export uv run --no-sync marimo export script explorations/chapter_directions/00_prepare.py -o "$CHAPTER_SCRATCH/00_prepare.flat.py" --force
check marimo_earnings_export uv run --no-sync marimo export script explorations/chapter_directions/05_extract_earnings.py -o "$CHAPTER_SCRATCH/05_extract_earnings.flat.py" --force
export CD_EXPORTED_EARNINGS="$CHAPTER_SCRATCH/05_extract_earnings.flat.py"
check earnings_edge_tests uv run --no-sync python explorations/chapter_directions/tests/test_earnings.py
check artifact_and_R_edge_checks Rscript --vanilla explorations/chapter_directions/07_validate.R "$CHAPTER_SCRATCH"
check project_pipeline_dry_run env DRY_RUN=1 ./scripts/run_all.sh
check exploration_dry_run env DRY_RUN=1 bash explorations/chapter_directions/run.sh "$CHAPTER_SCRATCH"
check grounding_verify uv run --no-sync python scripts/agent_docs.py verify
check grounding_tests uv run --no-sync python scripts/test_agent_docs.py
check zola_check zola check --skip-external-links
check zola_build zola build
Rscript --vanilla -e 'sessionInfo()' > "$CHAPTER_SCRATCH/logs/R_session.txt"
uv run --no-sync python --version > "$CHAPTER_SCRATCH/logs/Python_version.txt"
cat "$CHAPTER_SCRATCH/execution_checks.csv"
