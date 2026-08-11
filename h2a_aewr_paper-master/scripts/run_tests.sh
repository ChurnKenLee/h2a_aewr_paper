#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

# Sandboxed agents may have a read-only home directory. Keep uv's disposable
# cache outside the repository unless the caller explicitly selected a cache.
UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/h2a-aewr-uv-cache}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${TMPDIR:-/tmp}/h2a-aewr-uv-python}"
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR

STRICT_TOOLING=${STRICT_TOOLING:-0}

skip_or_fail() {
  local tool=$1
  if [[ $STRICT_TOOLING == 1 ]]; then
    printf 'Required tool is unavailable in strict mode: %s\n' "$tool" >&2
    exit 1
  fi
  printf 'SKIP: %s is unavailable (enter devenv shell for the strict gate)\n' "$tool"
}

run_if_available() {
  local tool=$1
  shift
  if command -v "$tool" >/dev/null 2>&1; then
    "$@"
  else
    skip_or_fail "$tool"
  fi
}

if command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  skip_or_fail python
  exit 0
fi

printf '==> Agent grounding and freshness\n'
"$PYTHON" scripts/agent_grounding.py verify

printf '==> Bash syntax\n'
for script in scripts/*.sh; do
  bash -n "$script"
done

printf '==> Python syntax\n'
"$PYTHON" - <<'PY'
import ast
from pathlib import Path

roots = [Path("code"), Path("src"), Path("scripts")]
files = sorted(path for root in roots for path in root.rglob("*.py"))
for path in files:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print(f"Parsed {len(files)} Python files")
PY

printf '==> Agent-grounding regression tests\n'
"$PYTHON" scripts/test_agent_grounding.py

printf '==> uv lock consistency\n'
run_if_available uv uv lock --check

printf '==> R syntax\n'
if command -v Rscript >/dev/null 2>&1; then
  LOCKED_R_VERSION=$("$PYTHON" -c 'import json; print(json.load(open("renv.lock"))["R"]["Version"])')
  RUNNING_R_VERSION=$(Rscript --vanilla -e "cat(paste(R.version\$major, R.version\$minor, sep = '.'))")
  if [[ ${RUNNING_R_VERSION%.*} != "${LOCKED_R_VERSION%.*}" ]]; then
    printf 'R major/minor mismatch: running %s, renv.lock %s\n' \
      "$RUNNING_R_VERSION" "$LOCKED_R_VERSION" >&2
    exit 1
  fi
  printf 'R runtime %s is patch-compatible with renv.lock %s\n' \
    "$RUNNING_R_VERSION" "$LOCKED_R_VERSION"
  Rscript --vanilla - <<'RS'
files <- sort(list.files("code", pattern = "[.]R$", recursive = TRUE, full.names = TRUE))
for (file in files) {
  parse(file = file)
}
cat(sprintf("Parsed %d R files\n", length(files)))
RS
else
  skip_or_fail Rscript
fi

printf '==> Structured source catalog\n'
if command -v yq >/dev/null 2>&1; then
  yq eval '.' documentation/raw_data_sources.yaml >/dev/null
else
  skip_or_fail yq
fi

printf '==> Shell lint\n'
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck -x -P SCRIPTDIR scripts/*.sh
else
  skip_or_fail shellcheck
fi

printf '==> GitHub Actions lint\n'
if command -v actionlint >/dev/null 2>&1; then
  mapfile -t workflow_files < <(
    find .github/workflows -maxdepth 1 -type f \( -name '*.yml' -o -name '*.yaml' \) -print | sort
  )
  if (( ${#workflow_files[@]} == 0 )); then
    printf 'No GitHub Actions workflows found\n' >&2
    exit 1
  fi
  actionlint "${workflow_files[@]}"
else
  skip_or_fail actionlint
fi

printf '==> Zola build and internal links\n'
if command -v zola >/dev/null 2>&1; then
  zola --root agent-docs check --skip-external-links
else
  skip_or_fail zola
fi

printf '==> Pipeline ordering dry run\n'
DRY_RUN=1 ./scripts/run_all.sh
DRY_RUN=1 ./scripts/run_optional_sources.sh
DRY_RUN=1 ./scripts/run_h2a_prediction_cutoffs.sh

printf 'All available fast checks passed.\n'
