+++
title = "Source-linked code grounding"
description = "Authoritative excerpts re-extracted from repository sources and rejected on unreviewed drift."
weight = 5
+++

> [!NOTE]
> Generated file. Change repository sources or `scripts/agent_grounding.py`, then regenerate.

These are not copied examples. Every fence is deterministically extracted from the named source, syntax-checked according to its registry rule, and compared with a reviewed excerpt SHA-256. Ordinary generation refuses changed excerpts; acceptance requires the explicit `accept-snippet-drift` review command. The complete machine-readable projection is `agent-docs/static/grounding-snippets.json`.

Registered source-linked snippets: **47** across **5** groups.

## Execution, paths, and validation

How supported entry points locate the repository, execute mixed R/Python stages, and enforce the fast gate.

### R project paths and prediction metadata

Grounds every supported R path and the canonical static H-2A prediction contract.

- Snippet ID: `r-path-contract`
- Source: `code/paths.R`
- Stable selector: `whole file`
- Source-file SHA-256: `3886719e5c6d6066a105c53550e7e4c3f933ab8771f5fd9621c4afb9124f9020`
- Extracted-text SHA-256: `3886719e5c6d6066a105c53550e7e4c3f933ab8771f5fd9621c4afb9124f9020`
- Validation: `r-parse`

<!-- grounding-snippet:r-path-contract excerpt-sha256=3886719e5c6d6066a105c53550e7e4c3f933ab8771f5fd9621c4afb9124f9020 -->
```r
# Shared project paths. The `.here` marker at the repository root is the
# single source of truth for every supported R entry point.

project_path <- function(...) here::here(...)

path_root <- project_path
path_do <- function(...) here::here("Do", ...)
path_code <- function(...) here::here("code", ...)
path_json <- function(...) here::here("code", "json", ...)

path_data <- function(...) here::here("data", ...)
path_raw <- function(...) here::here("data", "raw", ...)
path_int <- function(...) here::here("data", "intermediate", ...)
path_processed <- function(...) here::here("data", "processed", ...)
path_cache <- function(...) here::here("data", "intermediate", "cache", ...)

path_outputs <- function(...) here::here("outputs", ...)
path_figures <- function(...) here::here("outputs", "figures", ...)
path_tables <- function(...) here::here("outputs", "tables", ...)
path_logs <- function(...) here::here("outputs", "logs", ...)

# Canonical H-2A prediction used by every shared and design-specific panel.
# Changing the training window is a source-controlled specification change and
# requires rebuilding the prediction artifact and all downstream panels.
H2A_PREDICTION_CUTOFF_YEAR <- 2011L
H2A_PREDICTION_MODEL_SPEC <- "climate_norm_static_v1"

env_file <- here::here(".env")
if (file.exists(env_file)) {
  dotenv::load_dot_env(file = env_file)
}
rm(env_file)
```

### Python project-root discovery

Shows how Python and flattened Marimo programs escape temporary working directories.

- Snippet ID: `python-project-root`
- Source: `src/h2a/paths.py`
- Stable selector: `python symbol 'find_project_root'`
- Source-file SHA-256: `92b67554170dfec18a4936e6bad1c71024c53a60329b7b2b494c01a7f352f2a9`
- Extracted-text SHA-256: `ff8ea18a6f7a48d0c33f1779393d546fff184a700f9e457a890518b323e88053`
- Validation: `python-parse`

<!-- grounding-snippet:python-project-root excerpt-sha256=ff8ea18a6f7a48d0c33f1779393d546fff184a700f9e457a890518b323e88053 -->
```python
def find_project_root() -> Path:
    # pyprojroot finds the repo root from the calling context / project markers.
    root = Path(pyprojroot.find_root(criterion="pyproject.toml")).resolve()

    # Load .env from the project root, independent of Marimo's temp cwd.
    load_dotenv(root / ".env")

    # Optional override, useful only for unusual machine setups.
    env_root = os.getenv("H2A_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    return root
```

### Python path ownership

Defines the one-to-one path constants used by Python stages and retained outputs.

- Snippet ID: `python-path-layout`
- Source: `src/h2a/paths.py`
- Stable selector: `from exact line 'ROOT = find_project_root()' to EOF`
- Source-file SHA-256: `92b67554170dfec18a4936e6bad1c71024c53a60329b7b2b494c01a7f352f2a9`
- Extracted-text SHA-256: `7fd0010f45d5132ee49be4e236db89b32701fe552f5dde46f5e2ef8e90c7244e`
- Validation: `python-parse`

<!-- grounding-snippet:python-path-layout excerpt-sha256=7fd0010f45d5132ee49be4e236db89b32701fe552f5dde46f5e2ef8e90c7244e -->
```python
ROOT = find_project_root()

CODE = ROOT / "code"
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERMEDIATE = DATA / "intermediate"
PROCESSED = DATA / "processed"
CACHE = DATA / "intermediate" / "cache"

OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
LOGS = OUTPUTS / "logs"
```

### Mixed-language pipeline dispatch

Grounds dry-run behavior, R invocation, Marimo flattening, and ordinary Python execution.

- Snippet ID: `pipeline-run-step`
- Source: `scripts/pipeline_helpers.sh`
- Stable selector: `shell function 'run_step'`
- Source-file SHA-256: `74c3a07834f554f033a756cc714cee24eeb822a2fcd851aa4657af222601435c`
- Extracted-text SHA-256: `a697855e27db1d20fe73d6bdf0eeddc75e1a06af38d1d0163e00499fb20adeee`
- Validation: `bash-parse`

<!-- grounding-snippet:pipeline-run-step excerpt-sha256=a697855e27db1d20fe73d6bdf0eeddc75e1a06af38d1d0163e00499fb20adeee -->
```bash
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
        local flat_script
        flat_script="$PIPELINE_TMP/$(basename "$script" .py).flat.py"
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
```

### Top-level pipeline order

The supported source-to-design orchestration order, without inferred directory sorting.

- Snippet ID: `pipeline-runner-order`
- Source: `scripts/run_all.sh`
- Stable selector: `whole file`
- Source-file SHA-256: `4fc70f94db9f9d4609c724683efd46b0efb70f8a2cfe60a17c3a7e9a39342b23`
- Extracted-text SHA-256: `4fc70f94db9f9d4609c724683efd46b0efb70f8a2cfe60a17c3a7e9a39342b23`
- Validation: `bash-parse`

<!-- grounding-snippet:pipeline-runner-order excerpt-sha256=4fc70f94db9f9d4609c724683efd46b0efb70f8a2cfe60a17c3a7e9a39342b23 -->
```bash
#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

"$SCRIPT_DIR/run_sources.sh"
"$SCRIPT_DIR/run_derived.sh"
"$SCRIPT_DIR/run_shared_panel.sh"
"$SCRIPT_DIR/run_descriptives.sh"
"$SCRIPT_DIR/run_did.sh"
"$SCRIPT_DIR/run_panel_iv.sh"
"$SCRIPT_DIR/run_mundlak_chamberlain.sh"
```

### Fast repository validation gate

The exact static, parser, linter, documentation, and dry-run checks used before handoff.

- Snippet ID: `fast-validation-gate`
- Source: `scripts/run_tests.sh`
- Stable selector: `whole file`
- Source-file SHA-256: `127129f1231ec0e03e6ecbf8b127187e12ca4a45bd6c8316131bae9ae908ccc9`
- Extracted-text SHA-256: `127129f1231ec0e03e6ecbf8b127187e12ca4a45bd6c8316131bae9ae908ccc9`
- Validation: `bash-parse`

<!-- grounding-snippet:fast-validation-gate excerpt-sha256=127129f1231ec0e03e6ecbf8b127187e12ca4a45bd6c8316131bae9ae908ccc9 -->
```bash
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
```

### Pinned devenv test contract

Shows module composition and why CI makes optional local tooling strict.

- Snippet ID: `devenv-test-contract`
- Source: `devenv.nix`
- Stable selector: `whole file`
- Source-file SHA-256: `46408d8de966d10c3ce71d5fe110594c3a63a911e412bf29a61083b484482797`
- Extracted-text SHA-256: `46408d8de966d10c3ce71d5fe110594c3a63a911e412bf29a61083b484482797`
- Validation: `nix-parse`

<!-- grounding-snippet:devenv-test-contract excerpt-sha256=46408d8de966d10c3ce71d5fe110594c3a63a911e412bf29a61083b484482797 -->
```nix
{ lib, ... }:

let
  project = import ./nix/project.nix;
in
{
  # Secrets stay in direnv's process environment; do not enable devenv.dotenv,
  # which copies the .env file into the Nix store.
  dotenv.disableHint = true;

  imports = [
    ./nix/modules/base.nix
  ]
  ++ lib.optionals project.python [ ./nix/modules/python-uv.nix ]
  ++ lib.optionals project.gpu [ ./nix/modules/gpu-runtime.nix ]
  ++ lib.optionals project.cudaDev [ ./nix/modules/cuda-dev.nix ]
  ++ lib.optionals (project.r == "renv") [ ./nix/modules/r-positron-renv.nix ]
  ++ lib.optionals (project.r == "nix") [ ./nix/modules/r-positron-nix.nix ];

  enterTest = ''
    STRICT_TOOLING=1 ./scripts/run_tests.sh
  '';
}
```

### Documentation and validation tools

Grounds the packages and commands that generate, verify, check, and serve agent documentation.

- Snippet ID: `devenv-base-tools`
- Source: `nix/modules/base.nix`
- Stable selector: `whole file`
- Source-file SHA-256: `a5a98240036cbea3650517672a9ca4f648a876b0a1483c08ecadb9f31012ee79`
- Extracted-text SHA-256: `a5a98240036cbea3650517672a9ca4f648a876b0a1483c08ecadb9f31012ee79`
- Validation: `nix-parse`

<!-- grounding-snippet:devenv-base-tools excerpt-sha256=a5a98240036cbea3650517672a9ca4f648a876b0a1483c08ecadb9f31012ee79 -->
```nix
{ pkgs, ... }:

{
  packages = with pkgs; [
    coreutils
    curl
    git
    gnugrep
    gnumake
    gnused
    jq
    actionlint
    shellcheck
    yq-go
    zola
  ];

  scripts.agent-grounding = {
    description = "Inspect, verify, generate, or query coding-agent grounding";
    exec = ''
      exec python scripts/agent_grounding.py "$@"
    '';
  };

  scripts.agent-docs-generate = {
    description = "Regenerate machine-derived Zola agent documentation";
    exec = ''
      exec python scripts/agent_grounding.py generate
    '';
  };

  scripts.agent-docs-check = {
    description = "Verify grounding freshness and build the Zola site";
    exec = ''
      python scripts/agent_grounding.py verify
      exec zola --root agent-docs check --skip-external-links
    '';
  };

  scripts.agent-docs-serve = {
    description = "Serve the local coding-agent knowledge base";
    exec = ''
      python scripts/agent_grounding.py verify
      exec zola --root agent-docs serve "$@"
    '';
  };
}
```

### R and renv isolation

Grounds the plain-R launcher, native libraries, renv bootstrap, and restore commands.

- Snippet ID: `devenv-r-renv`
- Source: `nix/modules/r-positron-renv.nix`
- Stable selector: `whole file`
- Source-file SHA-256: `5247febb97b266a8d6400898aa7f96eb127ed1b6bd18816002033fb545d4c33f`
- Extracted-text SHA-256: `5247febb97b266a8d6400898aa7f96eb127ed1b6bd18816002033fb545d4c33f`
- Validation: `nix-parse`

<!-- grounding-snippet:devenv-r-renv excerpt-sha256=5247febb97b266a8d6400898aa7f96eb127ed1b6bd18816002033fb545d4c33f -->
```nix
{ lib, pkgs, ... }:

let
  renvBootstrap = pkgs.rPackages.renv;
  # System libraries commonly required when CRAN packages are compiled from
  # source (for example, tidyverse, ragg, and xml2). These are not R packages.
  rNativeDeps = with pkgs; [
    cmake
    curl
    fontconfig
    freetype
    fribidi
    harfbuzz
    libjpeg
    libpng
    libtiff
    libuv
    libwebp
    libxml2
    openssl
    gdal
    geos
    proj
    sqlite
    udunits
    zlib
    zstd
  ];
in
{
  # Positron needs the plain R launcher on PATH. All R packages, including
  # renv itself after initialization, belong to the project-local renv library.
  # Mixing rPackages into R_LIBS_SITE defeats renv's isolation and can load a
  # package compiled for a different R ABI.
  packages = [
    pkgs.R
    pkgs.gfortran
  ]
  ++ rNativeDeps;

  # pkg-config finds headers at build time; the loader needs the same libraries
  # at install and runtime. Retain the NVIDIA driver path for GPU projects.
  env = {
    PKG_CONFIG_PATH = lib.makeSearchPathOutput "dev" "lib/pkgconfig" rNativeDeps;
    LD_LIBRARY_PATH = "${lib.makeLibraryPath rNativeDeps}:/run/opengl-driver/lib";
    # Nix's udunits package does not ship a pkg-config file. The CRAN `units`
    # configure script explicitly supports these two variables.
    UDUNITS2_INCLUDE = "${pkgs.udunits}/include";
    UDUNITS2_LIBS = "-L${pkgs.udunits}/lib -ludunits2";
    # Polars writes Parquet with zstd by default. Ensure source installs of the
    # R arrow package can read those files.
    ARROW_WITH_ZSTD = "ON";
  };

  scripts.r-renv-init = {
    description = "Bootstrap an isolated renv project in the current directory";
    exec = ''
      R --vanilla --quiet -e 'library(renv, lib.loc = "${renvBootstrap}/library"); renv::init(bare = TRUE)'
    '';
  };

  scripts.r-renv-restore = {
    description = "Restore R packages from renv.lock";
    exec = ''
      R --quiet -e 'renv::restore(prompt = FALSE)'
    '';
  };
}
```

## Geography and shared-data contracts

Canonical identifiers, fail-closed joins, price fallbacks, prediction metadata, and panel keys.

### Python source-value cleanup

Makes the limited pre-normalization cleanup visible instead of implying broad coercion.

- Snippet ID: `python-clean-geo`
- Source: `src/h2a/geography.py`
- Stable selector: `python symbol 'clean_geo_code'`
- Source-file SHA-256: `57ba6ab7a018e9ed122483d04f9f5287d391e6bea1775fa89f7bcb914d4f7479`
- Extracted-text SHA-256: `e29e47e26bf98d01dd962c2792e45b080757b5a169e38c58edcaba7f3c9fc60f`
- Validation: `python-parse`

<!-- grounding-snippet:python-clean-geo excerpt-sha256=e29e47e26bf98d01dd962c2792e45b080757b5a169e38c58edcaba7f3c9fc60f -->
```python
def clean_geo_code(value: Any) -> str | None:
    """Clean one source value without changing its substantive digits."""
    if value is None:
        return None
    text = str(value).strip().replace('"', "")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text or None
```

### Python canonical identifier normalization

Grounds fixed-width, unpadded, variable-width, and AEWR-region validation.

- Snippet ID: `python-normalize-geo`
- Source: `src/h2a/geography.py`
- Stable selector: `python symbol 'normalize_geo_code'`
- Source-file SHA-256: `57ba6ab7a018e9ed122483d04f9f5287d391e6bea1775fa89f7bcb914d4f7479`
- Extracted-text SHA-256: `be35552f34ca5f9f6e1ed0c1ef8bcfe3efa962a603c62c0368e198eddc742525`
- Validation: `python-parse`

<!-- grounding-snippet:python-normalize-geo excerpt-sha256=be35552f34ca5f9f6e1ed0c1ef8bcfe3efa962a603c62c0368e198eddc742525 -->
```python
def normalize_geo_code(value: Any, name: str) -> str | None:
    """Return one canonical identifier or raise for a malformed value."""
    text = clean_geo_code(value)
    if text is None:
        return None
    if not text.isdigit():
        raise ValueError(f"{name} must contain digits only: {value!r}")

    if name in _FIXED_WIDTH:
        width = _FIXED_WIDTH[name]
        if len(text) > width:
            raise ValueError(f"{name} must contain at most {width} digits: {value!r}")
        text = text.zfill(width)
    elif name in _UNPADDED:
        text = text.lstrip("0") or "0"
    elif name in _VARIABLE_WIDTH:
        pass
    else:
        raise ValueError(f"Unknown geographic identifier: {name}")

    if name == "aewr_region_id" and text not in {str(i) for i in range(1, 18)}:
        raise ValueError(f"aewr_region_id must be between 1 and 17: {value!r}")
    return text
```

### Python 2010-vintage county harmonization

Shows the explicit 46102 to 46113 remapping after canonical FIPS normalization.

- Snippet ID: `python-harmonize-county`
- Source: `src/h2a/geography.py`
- Stable selector: `python symbol 'harmonize_county_fips_2010'`
- Source-file SHA-256: `57ba6ab7a018e9ed122483d04f9f5287d391e6bea1775fa89f7bcb914d4f7479`
- Extracted-text SHA-256: `aac548c97955bb0d82092c3c17dd02b2cc7d0f202cb467df5380f83e9edc3100`
- Validation: `python-parse`

<!-- grounding-snippet:python-harmonize-county excerpt-sha256=aac548c97955bb0d82092c3c17dd02b2cc7d0f202cb467df5380f83e9edc3100 -->
```python
def harmonize_county_fips_2010(value: Any) -> str | None:
    """Normalize a county identifier to the project's 2010 county vintage."""
    county = normalize_geo_code(value, "county_fips")
    return "46113" if county == "46102" else county
```

### Python artifact geography assertion

Grounds missing-column, type, null, and noncanonical-value failures.

- Snippet ID: `python-assert-geo`
- Source: `src/h2a/geography.py`
- Stable selector: `python symbol 'assert_geo_columns'`
- Source-file SHA-256: `57ba6ab7a018e9ed122483d04f9f5287d391e6bea1775fa89f7bcb914d4f7479`
- Extracted-text SHA-256: `398b5dc1c862f600f591f945bee17b9de8181a0d336ac76ef26e21e6479e8851`
- Validation: `python-parse`

<!-- grounding-snippet:python-assert-geo excerpt-sha256=398b5dc1c862f600f591f945bee17b9de8181a0d336ac76ef26e21e6479e8851 -->
```python
def assert_geo_columns(
    frame: pl.DataFrame,
    required: Iterable[str],
    *,
    allow_null: Iterable[str] = (),
) -> None:
    """Fail when an artifact does not satisfy the canonical geo contract."""
    required = tuple(required)
    allow_null = set(allow_null)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required geographic columns: {', '.join(missing)}")

    for name in required:
        if frame.schema[name] != pl.String:
            raise TypeError(f"{name} must use Polars String, got {frame.schema[name]}")
        values = frame.get_column(name)
        if name not in allow_null and values.null_count() > 0:
            raise ValueError(f"{name} contains missing values")
        for value in values.drop_nulls().unique().to_list():
            normalized = (
                harmonize_county_fips_2010(value)
                if name in {"county_fips", "neighbor_county_fips"}
                else normalize_geo_code(value, name)
            )
            if normalized != value:
                raise ValueError(f"{name} contains noncanonical value {value!r}")
```

### R source-value cleanup

The R-side cleanup contract used before fixed-width and unpadded validation.

- Snippet ID: `r-clean-geo`
- Source: `code/c00_shared/geography.R`
- Stable selector: `r function 'clean_geo_code'`
- Source-file SHA-256: `672377d1b457422205d4086ebde5e9740a37103ff559c12ddebf4dfab1db6fc1`
- Extracted-text SHA-256: `d9b38c61e46835a3d8a6672eb67be83ad87c5cc8a9e379d8516c7665c4c5b57c`
- Validation: `r-parse`

<!-- grounding-snippet:r-clean-geo excerpt-sha256=d9b38c61e46835a3d8a6672eb67be83ad87c5cc8a9e379d8516c7665c4c5b57c -->
```r
clean_geo_code <- function(x) {
  x <- trimws(as.character(x))
  x <- gsub('"', "", x, fixed = TRUE)
  x <- sub("\\.0+$", "", x)
  x[x == ""] <- NA_character_
  x
}
```

### R fixed-width normalization

Grounds digits-only validation, width rejection, and leading-zero preservation.

- Snippet ID: `r-normalize-fixed-geo`
- Source: `code/c00_shared/geography.R`
- Stable selector: `r function 'normalize_fixed_width_code'`
- Source-file SHA-256: `672377d1b457422205d4086ebde5e9740a37103ff559c12ddebf4dfab1db6fc1`
- Extracted-text SHA-256: `eeda7f7dd29890ca9fa6602c5e4b315df386756522a8f8f19ebcd8a169b9cca8`
- Validation: `r-parse`

<!-- grounding-snippet:r-normalize-fixed-geo excerpt-sha256=eeda7f7dd29890ca9fa6602c5e4b315df386756522a8f8f19ebcd8a169b9cca8 -->
```r
normalize_fixed_width_code <- function(x, width, label) {
  x <- clean_geo_code(x)
  invalid <- !is.na(x) & (!grepl("^[0-9]+$", x) | nchar(x) > width)

  if (any(invalid)) {
    examples <- paste(utils::head(unique(x[invalid]), 5L), collapse = ", ")
    stop(
      label,
      " must contain at most ",
      width,
      " digits; invalid values: ",
      examples,
      call. = FALSE
    )
  }

  padding <- pmax(width - nchar(x), 0L)
  padded <- paste0(
    vapply(padding, function(n) strrep("0", n), character(1)),
    x
  )
  padded[is.na(x)] <- NA_character_
  padded
}
```

### R 2010-vintage county harmonization

The R-side 46102 to 46113 remapping that must agree with Python.

- Snippet ID: `r-harmonize-county`
- Source: `code/c00_shared/geography.R`
- Stable selector: `r function 'harmonize_county_fips_2010'`
- Source-file SHA-256: `672377d1b457422205d4086ebde5e9740a37103ff559c12ddebf4dfab1db6fc1`
- Extracted-text SHA-256: `189703f74224cbb86575544dd4e96d5583b99c35329558c1642b1488e3bffcb9`
- Validation: `r-parse`

<!-- grounding-snippet:r-harmonize-county excerpt-sha256=189703f74224cbb86575544dd4e96d5583b99c35329558c1642b1488e3bffcb9 -->
```r
harmonize_county_fips_2010 <- function(x) {
  x <- county_fips(x)
  x[x == "46102"] <- "46113"
  x
}
```

### R artifact geography assertion

Grounds required columns, character storage, canonical formatting, and missingness rules.

- Snippet ID: `r-assert-geo`
- Source: `code/c00_shared/geography.R`
- Stable selector: `r function 'assert_geo_columns'`
- Source-file SHA-256: `672377d1b457422205d4086ebde5e9740a37103ff559c12ddebf4dfab1db6fc1`
- Extracted-text SHA-256: `3e7fae72279b05adae81bb343b5b3ee31e1b4d9d21f2d16e25f3164a2e9eb1c1`
- Validation: `r-parse`

<!-- grounding-snippet:r-assert-geo excerpt-sha256=3e7fae72279b05adae81bb343b5b3ee31e1b4d9d21f2d16e25f3164a2e9eb1c1 -->
```r
assert_geo_columns <- function(data, required, allow_na = character()) {
  missing_columns <- setdiff(required, names(data))
  if (length(missing_columns) > 0L) {
    stop(
      "Missing required geographic columns: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }

  for (name in required) {
    value <- data[[name]]

    if (!is.character(value)) {
      stop(name, " must be a character vector.", call. = FALSE)
    }

    invalid <- !geo_code_is_valid(value, name)
    if (any(invalid)) {
      examples <- paste(
        utils::head(unique(value[invalid]), 5L),
        collapse = ", "
      )
      stop(name, " contains malformed values: ", examples, call. = FALSE)
    }

    if (!name %in% allow_na && anyNA(value)) {
      stop(name, " contains missing values.", call. = FALSE)
    }
  }

  invisible(data)
}
```

### Synthetic-price lookup uniqueness

Shows the precondition that makes state and national joins many-to-one.

- Snippet ID: `price-lookup-uniqueness`
- Source: `src/h2a/price_index.py`
- Stable selector: `python symbol '_require_unique'`
- Source-file SHA-256: `4ed72c2748f4444127e7e63f99280f42466d9136b7b993cf90fa97e25ce2d581`
- Extracted-text SHA-256: `5b9844b6d770a8dfb3a825873c5a6f63ac4d73f23748326123525ce465a9f2e2`
- Validation: `python-parse`

<!-- grounding-snippet:price-lookup-uniqueness excerpt-sha256=5b9844b6d770a8dfb3a825873c5a6f63ac4d73f23748326123525ce465a9f2e2 -->
```python
def _require_unique(frame: pl.DataFrame, keys: list[str], label: str) -> None:
    """Fail with a useful message when a price lookup is not many-to-one."""
    missing = [key for key in keys if key not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing key columns: {', '.join(missing)}")
    if frame.select(keys).is_duplicated().any():
        raise ValueError(f"{label} contains duplicate keys: {', '.join(keys)}")
```

### State price/yield with national fallback

Grounds independent price/yield fallback, diagnostic source labels, and row-count preservation.

- Snippet ID: `price-national-fallback`
- Source: `src/h2a/price_index.py`
- Stable selector: `python symbol 'attach_synthetic_price_yield'`
- Source-file SHA-256: `4ed72c2748f4444127e7e63f99280f42466d9136b7b993cf90fa97e25ce2d581`
- Extracted-text SHA-256: `3323335ea79db22e36e1b109e38cae159b1441f012c6fa28c8a98aee7d7074a6`
- Validation: `python-parse`

<!-- grounding-snippet:price-national-fallback excerpt-sha256=3323335ea79db22e36e1b109e38cae159b1441f012c6fa28c8a98aee7d7074a6 -->
```python
def attach_synthetic_price_yield(
    cdl_acres: pl.DataFrame,
    state_synthetic_cdl: pl.DataFrame,
    national_synthetic_cdl: pl.DataFrame,
) -> pl.DataFrame:
    """Attach state values with an independent national fallback per field.

    The state and national tables must each be unique on their lookup keys.
    Source columns and source labels are retained for diagnostics; callers may
    select only ``cdl_syn_price`` and ``cdl_syn_yield`` for downstream use.
    """
    _require_unique(state_synthetic_cdl, _STATE_KEYS, "state synthetic CDL")
    _require_unique(
        national_synthetic_cdl,
        _NATIONAL_KEYS,
        "national synthetic CDL",
    )

    state_lookup = state_synthetic_cdl.select(
        *_STATE_KEYS,
        "p_syn_state",
        "y_syn_state",
    )
    national_lookup = national_synthetic_cdl.select(
        *_NATIONAL_KEYS,
        "p_syn_nat",
        "y_syn_nat",
    )

    result = (
        cdl_acres.join(
            state_lookup,
            on=_STATE_KEYS,
            how="left",
            validate="m:1",
        )
        .join(
            national_lookup,
            on=_NATIONAL_KEYS,
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.coalesce("p_syn_state", "p_syn_nat").alias("cdl_syn_price"),
            pl.coalesce("y_syn_state", "y_syn_nat").alias("cdl_syn_yield"),
            pl.when(pl.col("p_syn_state").is_not_null())
            .then(pl.lit("state"))
            .when(pl.col("p_syn_nat").is_not_null())
            .then(pl.lit("national"))
            .otherwise(pl.lit("missing"))
            .alias("price_source"),
            pl.when(pl.col("y_syn_state").is_not_null())
            .then(pl.lit("state"))
            .when(pl.col("y_syn_nat").is_not_null())
            .then(pl.lit("national"))
            .otherwise(pl.lit("missing"))
            .alias("yield_source"),
        )
    )

    unresolved_fallback = result.filter(
        (
            pl.col("p_syn_state").is_null()
            & pl.col("p_syn_nat").is_not_null()
            & pl.col("cdl_syn_price").is_null()
        )
        | (
            pl.col("y_syn_state").is_null()
            & pl.col("y_syn_nat").is_not_null()
            & pl.col("cdl_syn_yield").is_null()
        )
    )
    if unresolved_fallback.height:
        raise AssertionError(
            "Usable national synthetic values were not applied as fallback"
        )
    if result.height != cdl_acres.height:
        raise AssertionError("Synthetic price joins changed the acreage row count")

    return result
```

### Shared-panel prediction and key checks

Grounds canonical prediction metadata, denominator equivalence, nonnegative shares, and unique county-year output.

- Snippet ID: `shared-panel-output-contract`
- Source: `code/c02_build/01_build_county_panel.R`
- Stable selector: `from exact line 'prediction_contract <- county_panel %>%' to EOF`
- Source-file SHA-256: `2edf8bde30c3d2a7a5a1e149905e9be76ebcbd67500de56c1ac865f5622be5c1`
- Extracted-text SHA-256: `548fb81f1e322b796bcd49c8fc0af2755a5629abc6ca2d98b71fd864134946ed`
- Validation: `r-parse`

<!-- grounding-snippet:shared-panel-output-contract excerpt-sha256=548fb81f1e322b796bcd49c8fc0af2755a5629abc6ca2d98b71fd864134946ed -->
```r
prediction_contract <- county_panel %>%
  filter(!is.na(h2a_prediction_cutoff_year)) %>%
  select(
    county_fips,
    h2a_prediction_cutoff_year,
    h2a_prediction_model_spec,
    predicted_h2a_count,
    bea_farm_emp_2011,
    h2a_predicted_share_2011,
    emp_farm_2011
  ) %>%
  distinct()

if (
  nrow(prediction_contract) == 0L ||
    anyDuplicated(prediction_contract$county_fips) > 0L ||
    any(
      prediction_contract$h2a_prediction_cutoff_year !=
        H2A_PREDICTION_CUTOFF_YEAR
    ) ||
    !identical(
      unique(prediction_contract$h2a_prediction_model_spec),
      H2A_PREDICTION_MODEL_SPEC
    ) ||
    any(
      !is.finite(prediction_contract$predicted_h2a_count) |
        prediction_contract$predicted_h2a_count < 0
    ) ||
    any(
      !is.finite(prediction_contract$emp_farm_2011) |
        prediction_contract$emp_farm_2011 <= 0
    ) ||
    any(
      !is.finite(prediction_contract$bea_farm_emp_2011) |
        prediction_contract$bea_farm_emp_2011 <= 0
    ) ||
    any(
      abs(
        prediction_contract$bea_farm_emp_2011 -
          prediction_contract$emp_farm_2011
      ) > 1e-8,
      na.rm = TRUE
    ) ||
    any(
      !is.finite(prediction_contract$h2a_predicted_share_2011) |
        prediction_contract$h2a_predicted_share_2011 < 0
    ) ||
    !isTRUE(all.equal(
      prediction_contract$h2a_predicted_share_2011,
      prediction_contract$predicted_h2a_count /
        prediction_contract$bea_farm_emp_2011,
      tolerance = 2e-6,
      check.attributes = FALSE
    ))
) {
  stop(
    "The shared panel must use one valid canonical static prediction per county.",
    call. = FALSE
  )
}

if (
  nrow(county_panel) == 0L ||
    anyDuplicated(county_panel[c("county_fips", "year")]) > 0L
) {
  stop("county_year_panel must have unique county-year keys.", call. = FALSE)
}

write_parquet(
  county_panel,
  path_processed("county_year_panel.parquet")
)
```

## Difference-in-differences design

Treatment classification, sample selection, formulas, fixed effects, and clustered inference.

### DiD treatment classification and post period

Grounds the 2008 classification slice, propensity/observed-share cells, post timing, and output checks.

- Snippet ID: `did-treatment-panel`
- Source: `code/designs/did/01_build_did_panel.R`
- Stable selector: `whole file`
- Source-file SHA-256: `be1f6632cc75c0ff71b08c3b4c32823c54f02f6155a689bfb954d33bde30d282`
- Extracted-text SHA-256: `be1f6632cc75c0ff71b08c3b4c32823c54f02f6155a689bfb954d33bde30d282`
- Validation: `r-parse`

<!-- grounding-snippet:did-treatment-panel excerpt-sha256=be1f6632cc75c0ff71b08c3b4c32823c54f02f6155a689bfb954d33bde30d282 -->
```r
# Purpose: Add the treatment classification and post period used by the DiD.
# Output: data/processed/did_county_year_panel.parquet.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
library(arrow)
library(dplyr)

true_share_cutoff <- 0.01
predicted_share_cutoff <- 0.01

county_panel <- read_parquet(
  path_processed("county_year_panel.parquet")
)

treatment_groups <- county_panel %>%
  # The PPML propensity is static; 2008 selects the observed H-2A baseline
  # used with it to define time-invariant treatment groups.
  filter(year == 2008L) %>%
  transmute(
    county_fips,
    county_treatment_group_classification = case_when(
      h2a_predicted_share_2011 > predicted_share_cutoff &
        h2a_cert_share_farm_workers_2011_start_year >
          true_share_cutoff ~ "always takers",
      h2a_predicted_share_2011 > predicted_share_cutoff &
        h2a_cert_share_farm_workers_2011_start_year <
          true_share_cutoff ~ "adopters",
      h2a_predicted_share_2011 < predicted_share_cutoff &
        h2a_cert_share_farm_workers_2011_start_year >
          true_share_cutoff ~ "defiers",
      h2a_predicted_share_2011 < predicted_share_cutoff &
        h2a_cert_share_farm_workers_2011_start_year <
          true_share_cutoff ~ "never takers"
    ),
    county_simple_treatment_groups = if_else(
      county_treatment_group_classification == "always takers",
      "always takers",
      "exposed adopters",
      missing = NA_character_
    )
  )

did_panel <- county_panel %>%
  left_join(
    treatment_groups,
    by = "county_fips",
    relationship = "many-to-one"
  ) %>%
  mutate(post = year > 2011L)

prediction_contract <- did_panel %>%
  filter(!is.na(h2a_predicted_share_2011)) %>%
  distinct(county_fips, h2a_predicted_share_2011)

if (
  nrow(did_panel) == 0L ||
    anyDuplicated(did_panel[c("county_fips", "year")]) > 0L ||
    anyDuplicated(treatment_groups$county_fips) > 0L ||
    anyDuplicated(prediction_contract$county_fips) > 0L ||
    any(
      !is.na(did_panel$h2a_prediction_cutoff_year) &
        did_panel$h2a_prediction_cutoff_year !=
          H2A_PREDICTION_CUTOFF_YEAR
    ) ||
    !identical(
      unique(did_panel$h2a_prediction_model_spec[
        !is.na(did_panel$h2a_prediction_model_spec)
      ]),
      H2A_PREDICTION_MODEL_SPEC
    )
) {
  stop(
    "did_county_year_panel must have unique county-year keys.",
    call. = FALSE
  )
}

write_parquet(
  did_panel,
  path_processed("did_county_year_panel.parquet")
)
```

### DiD estimation sample

Shows the cropland and treatment-group restrictions applied before estimation.

- Snippet ID: `did-sample`
- Source: `code/designs/did/helpers.R`
- Stable selector: `r function 'did_sample'`
- Source-file SHA-256: `97efc588b26e85f85ca00351774c198f3898b7033834e8105bdb9c73cdd8ffa7`
- Extracted-text SHA-256: `005b18e9c301e8bf2d389725b79987892faabe554c82f2c2caeb41bd93c3d04a`
- Validation: `r-parse`

<!-- grounding-snippet:did-sample excerpt-sha256=005b18e9c301e8bf2d389725b79987892faabe554c82f2c2caeb41bd93c3d04a -->
```r
did_sample <- function(panel) {
  panel %>%
    dplyr::filter(
      any_cropland_2007,
      county_simple_treatment_groups != "always takers"
    )
}
```

### DiD main formula and inference

Grounds treatment interaction, optional controls, county/year effects, and the shared cluster formula.

- Snippet ID: `did-model`
- Source: `code/designs/did/helpers.R`
- Stable selector: `r function 'did_model'`
- Source-file SHA-256: `97efc588b26e85f85ca00351774c198f3898b7033834e8105bdb9c73cdd8ffa7`
- Extracted-text SHA-256: `efc8f3fe070a7f9115d425f883d7a9acf7d4f05558b9d3bfa8368c17d5a77125`
- Validation: `r-parse`

<!-- grounding-snippet:did-model excerpt-sha256=efc8f3fe070a7f9115d425f883d7a9acf7d4f05558b9d3bfa8368c17d5a77125 -->
```r
did_model <- function(data, outcome, controls = FALSE) {
  control_terms <- if (controls) {
    " + ln_pop_census + emp_pop_ratio"
  } else {
    ""
  }
  formula <- stats::as.formula(paste0(
    outcome,
    " ~ aewr_cz_p25_l1 * post",
    control_terms,
    " | county_fips + year"
  ))
  fixest::feols(
    formula,
    data = data,
    vcov = did_cluster_formula,
    notes = FALSE
  )
}
```

### DiD event-study formula

Grounds the reference year, event interaction, controls, fixed effects, and clustered covariance.

- Snippet ID: `did-event-model`
- Source: `code/designs/did/helpers.R`
- Stable selector: `r function 'did_event_model'`
- Source-file SHA-256: `97efc588b26e85f85ca00351774c198f3898b7033834e8105bdb9c73cdd8ffa7`
- Extracted-text SHA-256: `d6fd8b48ab89732b401593f685fa48ad787550dd76e3f1448534b4ce6e8311d5`
- Validation: `r-parse`

<!-- grounding-snippet:did-event-model excerpt-sha256=d6fd8b48ab89732b401593f685fa48ad787550dd76e3f1448534b4ce6e8311d5 -->
```r
did_event_model <- function(data, controls = FALSE) {
  control_terms <- if (controls) {
    " + ln_pop_census + emp_pop_ratio"
  } else {
    ""
  }
  formula <- stats::as.formula(paste0(
    "h2a_cert_share_farm_workers_2011_start_year ~ ",
    "aewr_cz_p25_l1 + ",
    "i(year, aewr_cz_p25_l1, ref = 2011)",
    control_terms,
    " | county_fips + year"
  ))
  fixest::feols(
    formula,
    data = data,
    vcov = did_cluster_formula,
    notes = FALSE
  )
}
```

## Panel-IV design

Frozen instrument constants, controls, identifiers, weights, sample restrictions, and coefficient extraction.

### Panel-IV frozen design constants

Grounds the policy window, donor count, target moments, entropy tuning, labels, allocation, and inference cluster.

- Snippet ID: `panel-iv-frozen-design`
- Source: `code/designs/panel_iv/design.R`
- Stable selector: `exact lines 'DISSIMILARITY_IV_K_VALUES <- 5L' … 'DISSIMILARITY_IV_BIG_SIX_OCC_CODES <- c('`
- Source-file SHA-256: `f4eab8956d2f00929aef7a752b8cdf4212b4e2f8f0fcaf2314ba6c9a7cfb9321`
- Extracted-text SHA-256: `e1aa12290453890f02ef1c9aaaaead9ac75a34f45ad0fd3c68a07b1f5ab008ac`
- Validation: `r-parse`

<!-- grounding-snippet:panel-iv-frozen-design excerpt-sha256=e1aa12290453890f02ef1c9aaaaead9ac75a34f45ad0fd3c68a07b1f5ab008ac -->
```r
DISSIMILARITY_IV_K_VALUES <- 5L
DISSIMILARITY_IV_PRIMARY_K <- 5L
DISSIMILARITY_IV_PRIMARY_DONOR_COUNT <- 2L

DISSIMILARITY_IV_FEATURE_START_YEAR <- 2008L
DISSIMILARITY_IV_FEATURE_END_YEAR <- 2011L
DISSIMILARITY_IV_POLICY_START_YEAR <- 2011L
DISSIMILARITY_IV_POLICY_END_YEAR <- 2022L

DISSIMILARITY_IV_INSTRUMENT_FAMILY <- "dissimilarity_cluster"
DISSIMILARITY_IV_AGGREGATION_SPEC <- "unique_oews_area"
DISSIMILARITY_IV_FRAME_WEIGHT_SPEC <-
  "census_hired_workers_qcew_updated"
DISSIMILARITY_IV_PRIMARY_WEIGHT_SPEC <-
  "fls_realized_geography_dirichlet_entropy"
DISSIMILARITY_IV_WAGE_ONLY_WEIGHT_SPECIFICATION <-
  "fls_geo_wage_only_soft_rho010"
DISSIMILARITY_IV_PRIMARY_WEIGHT_SPECIFICATION <-
  "fls_geo_wage_seasonal_soft_rho010"
DISSIMILARITY_IV_PRIMARY_WEIGHT_COMPONENT <-
  "calibrated_center"
DISSIMILARITY_IV_WAGE_ONLY_MOMENT_SPEC <-
  "fls_field_livestock_wage_only"
DISSIMILARITY_IV_PRIMARY_MOMENT_SPEC <-
  "fls_field_livestock_wage_plus_quarterly_worker_shares"
DISSIMILARITY_IV_PRIMARY_RHO <- 0.10
DISSIMILARITY_IV_PRIMARY_KAPPA_MULTIPLIER <- 10
DISSIMILARITY_IV_WAGE_ONLY_INSTRUMENT_LABEL <- paste0(
  "k",
  DISSIMILARITY_IV_PRIMARY_K,
  "_d",
  DISSIMILARITY_IV_PRIMARY_DONOR_COUNT,
  "_wage_only_soft_rho010_center"
)
DISSIMILARITY_IV_PRIMARY_INSTRUMENT_LABEL <- paste0(
  "k",
  DISSIMILARITY_IV_PRIMARY_K,
  "_d",
  DISSIMILARITY_IV_PRIMARY_DONOR_COUNT,
  "_wage_seasonal_soft_rho010_center"
)
DISSIMILARITY_IV_BASELINE_FRAME_PROXY <-
  "census_ag_direct_hired_workers"
DISSIMILARITY_IV_ANNUAL_UPDATE_SPEC <-
  "qcew_qwi_bea_two_sided_state_raked"
DISSIMILARITY_IV_GEOGRAPHIC_ALLOCATION_SPEC <-
  "oews_township_share_within_county"
DISSIMILARITY_IV_CLUSTER_SIZE_RULE <- "none"
DISSIMILARITY_IV_INFERENCE_CLUSTER <- "aewr_iv_cluster_id"

# All selected donor clusters must contribute a wage in a given year. Donor
# unit counts are diagnostics only; no minimum cluster-size rule is imposed.
DISSIMILARITY_IV_MIN_OBSERVED_DONOR_CLUSTERS <-
  DISSIMILARITY_IV_PRIMARY_DONOR_COUNT
```

### Panel-IV occupation frame

Shows the complete frozen occupation-code set feeding the recovered wage frame.

- Snippet ID: `panel-iv-occupation-frame`
- Source: `code/designs/panel_iv/design.R`
- Stable selector: `exact lines 'DISSIMILARITY_IV_BIG_SIX_OCC_CODES <- c(' … 'DISSIMILARITY_IV_BASELINE_CONTROL_COLUMNS <- c('`
- Source-file SHA-256: `f4eab8956d2f00929aef7a752b8cdf4212b4e2f8f0fcaf2314ba6c9a7cfb9321`
- Extracted-text SHA-256: `ac7d2f4bc36cb731db9724d486587f07ffd46e7ff32f629c4f47a381e5683b73`
- Validation: `r-parse`

<!-- grounding-snippet:panel-iv-occupation-frame excerpt-sha256=ac7d2f4bc36cb731db9724d486587f07ffd46e7ff32f629c4f47a381e5683b73 -->
```r
DISSIMILARITY_IV_BIG_SIX_OCC_CODES <- c(
  "45-2041",
  "45-2091",
  "45-2092",
  "45-2093",
  "53-7064",
  "45-2099",
  "79011",
  "79021",
  "79856",
  "79858",
  "98902"
)
```

### Panel-IV controls and propensity trend

Grounds baseline covariates and makes explicit that the static propensity enters as a differential trend, not an instrument.

- Snippet ID: `panel-iv-control-contract`
- Source: `code/designs/panel_iv/design.R`
- Stable selector: `exact lines 'DISSIMILARITY_IV_BASELINE_CONTROL_COLUMNS <- c(' … 'make_dissimilarity_cluster_id <- function(aewr_region_id, target_cluster) {'`
- Source-file SHA-256: `f4eab8956d2f00929aef7a752b8cdf4212b4e2f8f0fcaf2314ba6c9a7cfb9321`
- Extracted-text SHA-256: `d4249c2d9fb5ffd262d070ed2b1798a4fe8c25a29f6ea516a87b4db0e501d7d4`
- Validation: `r-parse`

<!-- grounding-snippet:panel-iv-control-contract excerpt-sha256=d4249c2d9fb5ffd262d070ed2b1798a4fe8c25a29f6ea516a87b4db0e501d7d4 -->
```r
DISSIMILARITY_IV_BASELINE_CONTROL_COLUMNS <- c(
  "ln_pop_census_l1",
  "farm_emp_share_l1",
  "emp_pop_ratio_l1",
  "wage_p10_l1"
)

# The time-invariant PPML level is absorbed by county fixed effects. The
# controlled publication specifications instead allow counties with different
# baseline propensities to follow different linear trends.
DISSIMILARITY_IV_PROPENSITY_COLUMN <- "h2a_ppml_static_propensity_z"
DISSIMILARITY_IV_PROPENSITY_TREND_TERM <- paste0(
  DISSIMILARITY_IV_PROPENSITY_COLUMN,
  ":year_centered"
)

DISSIMILARITY_IV_CONTROL_COLUMNS <- c(
  DISSIMILARITY_IV_BASELINE_CONTROL_COLUMNS,
  DISSIMILARITY_IV_PROPENSITY_COLUMN,
  "year_centered"
)

DISSIMILARITY_IV_CONTROL_TERMS <- c(
  DISSIMILARITY_IV_BASELINE_CONTROL_COLUMNS,
  DISSIMILARITY_IV_PROPENSITY_TREND_TERM
)
```

### Panel-IV inference-cluster identifier

Grounds the stable AEWR-region by target-cluster string representation.

- Snippet ID: `panel-iv-cluster-id`
- Source: `code/designs/panel_iv/design.R`
- Stable selector: `r function 'make_dissimilarity_cluster_id'`
- Source-file SHA-256: `f4eab8956d2f00929aef7a752b8cdf4212b4e2f8f0fcaf2314ba6c9a7cfb9321`
- Extracted-text SHA-256: `665d790502c7ba3693b90de80ce3bf2dce57861674fbccdd969fb361cc6a7f3f`
- Validation: `r-parse`

<!-- grounding-snippet:panel-iv-cluster-id excerpt-sha256=665d790502c7ba3693b90de80ce3bf2dce57861674fbccdd969fb361cc6a7f3f -->
```r
make_dissimilarity_cluster_id <- function(aewr_region_id, target_cluster) {
  ifelse(
    is.na(aewr_region_id) | is.na(target_cluster),
    NA_character_,
    paste0(
      sprintf("%02d", as.integer(aewr_region_id)),
      "_",
      sprintf("%02d", as.integer(target_cluster))
    )
  )
}
```

### Panel-IV target-unit identifier

Grounds the CZ by AEWR-region target unit used by donor recovery and joins.

- Snippet ID: `panel-iv-target-unit-id`
- Source: `code/designs/panel_iv/design.R`
- Stable selector: `r function 'make_panel_iv_target_unit_id'`
- Source-file SHA-256: `f4eab8956d2f00929aef7a752b8cdf4212b4e2f8f0fcaf2314ba6c9a7cfb9321`
- Extracted-text SHA-256: `907470b3b61da0407a87b54fca29c170fb55060ea1fc07367091f98e77da88db`
- Validation: `r-parse`

<!-- grounding-snippet:panel-iv-target-unit-id excerpt-sha256=907470b3b61da0407a87b54fca29c170fb55060ea1fc07367091f98e77da88db -->
```r
make_panel_iv_target_unit_id <- function(cz_id, aewr_region_id) {
  ifelse(
    is.na(cz_id) | is.na(aewr_region_id),
    NA_character_,
    paste0(cz_id, "_", aewr_region_id)
  )
}
```

### Panel-IV area-weight construction

Grounds eligible mass, missing-wage diagnostics, normalized area weights, and normalized instrument weights.

- Snippet ID: `panel-iv-area-weight`
- Source: `code/designs/panel_iv/05_construct_instruments.R`
- Stable selector: `r function 'make_area_weight_spec'`
- Source-file SHA-256: `9147773b7803388bceac7ba6644e060e52fe061302636c6058d237b6f12a4034`
- Extracted-text SHA-256: `ab9fc157f8d92b2a195b1f7e33b3af0ce45e3db731976a77bc9b91a147104a3f`
- Validation: `r-parse`

<!-- grounding-snippet:panel-iv-area-weight excerpt-sha256=ab9fc157f8d92b2a195b1f7e33b3af0ce45e3db731976a77bc9b91a147104a3f -->
```r
make_area_weight_spec <- function(
  data,
  selected_weight_column,
  weight_spec,
  baseline_weight_spec,
  weight_component,
  weight_specification,
  moment_spec,
  wage_target_used,
  rho,
  kappa_multiplier,
  is_primary,
  instrument_spec_label
) {
  data %>%
    mutate(
      selected_weight_mass = .data[[selected_weight_column]],
      weight_mass_observed = is.finite(selected_weight_mass) &
        selected_weight_mass > 0,
      weight_wage_observed = weight_mass_observed & oews_wage_observed
    ) %>%
    group_by(aewr_region_id, target_cluster, source_year) %>%
    mutate(
      eligible_weight_mass = sum(
        if_else(weight_mass_observed, selected_weight_mass, 0),
        na.rm = TRUE
      ),
      observed_wage_weight_mass = sum(
        if_else(weight_wage_observed, selected_weight_mass, 0),
        na.rm = TRUE
      ),
      missing_wage_weight_share = if_else(
        eligible_weight_mass > 0,
        1 - observed_wage_weight_mass / eligible_weight_mass,
        NA_real_
      ),
      area_weight = if_else(
        weight_mass_observed & eligible_weight_mass > 0,
        selected_weight_mass / eligible_weight_mass,
        NA_real_
      ),
      instrument_weight = if_else(
        weight_wage_observed & observed_wage_weight_mass > 0,
        selected_weight_mass / observed_wage_weight_mass,
        NA_real_
      )
    ) %>%
    ungroup() %>%
    mutate(
      weight_spec = .env$weight_spec,
      baseline_weight_spec = .env$baseline_weight_spec,
      weight_component = .env$weight_component,
      weight_specification = .env$weight_specification,
      moment_spec = .env$moment_spec,
      wage_target_used = .env$wage_target_used,
      rho = .env$rho,
      kappa_multiplier = .env$kappa_multiplier,
      is_primary = .env$is_primary,
      weight_draw_id = NA_integer_,
      instrument_spec_label = .env$instrument_spec_label
    )
}
```

### Panel-IV endogenous variable, fixed effects, and clustering

Grounds the publication estimation skeleton before the outcome registry.

- Snippet ID: `panel-iv-estimation-contract`
- Source: `code/designs/panel_iv/07_estimate_panel_iv.R`
- Stable selector: `exact lines 'endogenous <- "aewr_ppi"' … 'outcomes <- tribble('`
- Source-file SHA-256: `f9c8dfc6f6eec72225b6aa84462104db862eb061f9ed1baef9f96204a1f9f128`
- Extracted-text SHA-256: `dbba31672eb3f9f858f926d980e9e79c63c70a1931c4cea5b0e50850cc5dd0b4`
- Validation: `r-parse`

<!-- grounding-snippet:panel-iv-estimation-contract excerpt-sha256=dbba31672eb3f9f858f926d980e9e79c63c70a1931c4cea5b0e50850cc5dd0b4 -->
```r
endogenous <- "aewr_ppi"
fixed_effects <- "county_fips + year"
cluster_vcov <- ~aewr_iv_cluster_id
control_terms <- paste(
  DISSIMILARITY_IV_CONTROL_TERMS,
  collapse = " + "
)
```

### Panel-IV complete-case policy sample

Grounds policy-year limits and finite numeric/nonmissing identifier requirements.

- Snippet ID: `panel-iv-finite-sample`
- Source: `code/designs/panel_iv/07_estimate_panel_iv.R`
- Stable selector: `r function 'finite_complete'`
- Source-file SHA-256: `f9c8dfc6f6eec72225b6aa84462104db862eb061f9ed1baef9f96204a1f9f128`
- Extracted-text SHA-256: `c468cfbdb3f31d2bffc6b3daeb03f816dd9d7f6fccc0a92fb91e9360eb3b1862`
- Validation: `r-parse`

<!-- grounding-snippet:panel-iv-finite-sample excerpt-sha256=c468cfbdb3f31d2bffc6b3daeb03f816dd9d7f6fccc0a92fb91e9360eb3b1862 -->
```r
finite_complete <- function(data, numeric_columns, id_columns) {
  data %>%
    filter(
      year >= DISSIMILARITY_IV_POLICY_START_YEAR,
      year <= DISSIMILARITY_IV_POLICY_END_YEAR,
      if_all(
        all_of(numeric_columns),
        ~ !is.na(.x) & is.finite(.x)
      ),
      if_all(all_of(id_columns), ~ !is.na(.x))
    )
}
```

### Panel-IV coefficient extraction

Fails closed unless exactly one coefficient matches the requested publication pattern.

- Snippet ID: `panel-iv-coefficient-row`
- Source: `code/designs/panel_iv/07_estimate_panel_iv.R`
- Stable selector: `r function 'coefficient_row'`
- Source-file SHA-256: `f9c8dfc6f6eec72225b6aa84462104db862eb061f9ed1baef9f96204a1f9f128`
- Extracted-text SHA-256: `e859e73d3d3010a2ee24cbd41918695698b49bc3901efd5a6f0e0a7f14115d50`
- Validation: `r-parse`

<!-- grounding-snippet:panel-iv-coefficient-row excerpt-sha256=e859e73d3d3010a2ee24cbd41918695698b49bc3901efd5a6f0e0a7f14115d50 -->
```r
coefficient_row <- function(model, pattern) {
  table <- coeftable(model)
  selected <- grep(pattern, rownames(table), value = TRUE)
  if (length(selected) != 1L) {
    stop(
      "Expected one coefficient matching ",
      pattern,
      "; found: ",
      paste(selected, collapse = ", "),
      call. = FALSE
    )
  }
  row <- table[selected, , drop = FALSE]
  tibble(
    coefficient_name = selected,
    estimate = unname(row[1, "Estimate"]),
    standard_error = unname(row[1, "Std. Error"]),
    t_statistic = unname(row[1, "t value"]),
    p_value = unname(row[1, "Pr(>|t|)"])
  )
}
```

## Mundlak-Chamberlain specification program

Calendars, resource/rank budgets, registries, finite-design reference states, and CCV covariance.

### MC calendar and design version

Grounds baseline, treatment-history, analysis, and reference years together with the supported design version.

- Snippet ID: `mc-calendar-version`
- Source: `code/designs/mundlak_chamberlain/design.R`
- Stable selector: `exact lines 'MC_BASELINE_YEARS <- 2008:2010' … 'MC_SPEC_PROGRAM_VERSION <- "3.0.0"'`
- Source-file SHA-256: `8c30073569b87e1c677cead4dd37c12aab2be5aae91f3a0f03bf57ccb92a4fef`
- Extracted-text SHA-256: `bd3c81e85d002baa0c8d7e3e775e54de4cb5e9bc87879012603099773d18fbe3`
- Validation: `r-parse`

<!-- grounding-snippet:mc-calendar-version excerpt-sha256=bd3c81e85d002baa0c8d7e3e775e54de4cb5e9bc87879012603099773d18fbe3 -->
```r
MC_BASELINE_YEARS <- 2008:2010
MC_TREATMENT_HISTORY_YEARS <- 2011:2022
MC_ANALYSIS_YEARS <- 2013:2022
MC_REFERENCE_YEAR <- 2013L
MC_LEGACY_DESIGN_VERSION <- "2.3.0"
MC_DESIGN_VERSION <- "3.0.0"

# Version 3 is a specification program.  The constants above remain the
# calendar of the frozen version-2.3 compatibility record and the target
# calendar for the version-3 primary model.  All grid models read their
# calendar and causal dictionary from an explicit specification record.
```

### MC specification and resource bounds

Grounds the program grid, region/rank reserve, compact default, worker/thread limits, and memory guardrails.

- Snippet ID: `mc-program-resource-contract`
- Source: `code/designs/mundlak_chamberlain/design.R`
- Stable selector: `exact lines 'MC_SPEC_PROGRAM_VERSION <- "3.0.0"' … 'MC_SPEC_RICHNESS_LABELS <- c('`
- Source-file SHA-256: `8c30073569b87e1c677cead4dd37c12aab2be5aae91f3a0f03bf57ccb92a4fef`
- Extracted-text SHA-256: `dab355e4f5be6c5c95d8900c7378e3f2ff57680ca790111b3a21a1eb442c7768`
- Validation: `r-parse`

<!-- grounding-snippet:mc-program-resource-contract excerpt-sha256=dab355e4f5be6c5c95d8900c7378e3f2ff57680ca790111b3a21a1eb442c7768 -->
```r
MC_SPEC_PROGRAM_VERSION <- "3.0.0"
MC_SPEC_HISTORY_YEARS <- 2011:2022
MC_SPEC_ANALYSIS_END <- 2022L
MC_SPEC_PREPERIOD_LENGTHS <- 2:4
MC_SPEC_ANALYSIS_START_DELAYS <- 0:2
MC_SPEC_POLYNOMIAL_DEGREES <- 1:3
MC_SPEC_RICHNESS_TIERS <- 0:3
MC_SPEC_REGION_BUDGET <- 16L
# The per-year ledger is not the whole rank calculation once region and year
# effects are included jointly.  Six additional region-time coordinates are
# held out before the all-state basis audit; this reproduces the slack in the
# identified version-2.3 block without privileging a reference year.
MC_SPEC_GLOBAL_REGION_RESERVE <- 6L
MC_SPEC_FULL_SAMPLE_PARAMETER_ROW_MAX <- 0.25
MC_SPEC_RESTRICTED_SAMPLE_PARAMETER_ROW_MAX <- 0.15
MC_SPEC_DF_ADJUSTMENT <- "N_over_N_minus_K"
# Resource defaults are intentionally conservative.  One outcome process may
# still use several fixest threads, so two forked workers can otherwise occupy
# every logical CPU while duplicating multi-gigabyte design matrices.
MC_SPEC_DEFAULT_STAGE <- "compact"
MC_SPEC_DEFAULT_WORKERS <- 1L
MC_SPEC_DEFAULT_FIXEST_THREADS <- 4L
MC_SPEC_MAX_DENSE_MATRIX_GIB <- 1.25
MC_SPEC_MAX_ESTIMATED_PEAK_GIB <- 6
MC_SPEC_DENSE_PEAK_COPIES <- 4
MC_SPEC_GRAM_PEAK_COPIES <- 3
MC_SPEC_GRADIENT_TOLERANCE <- 1e-9
```

### MC finite-design reference law

Grounds 17 assignment states, 16 reference degrees of freedom, covariance method, and reference design.

- Snippet ID: `mc-ccv-reference-contract`
- Source: `code/designs/mundlak_chamberlain/design.R`
- Stable selector: `exact lines 'MC_CCV_REFERENCE_STATES <- 17L' … 'MC_COUNTERFACTUAL_DOSES <- c(1, 5, 10)'`
- Source-file SHA-256: `8c30073569b87e1c677cead4dd37c12aab2be5aae91f3a0f03bf57ccb92a4fef`
- Extracted-text SHA-256: `1b8951fd5b4409e62ae98fcfc0483d028bd2e49e351b024eade323409ebaaa42`
- Validation: `r-parse`

<!-- grounding-snippet:mc-ccv-reference-contract excerpt-sha256=1b8951fd5b4409e62ae98fcfc0483d028bd2e49e351b024eade323409ebaaa42 -->
```r
MC_CCV_REFERENCE_STATES <- 17L
MC_CCV_DF <- MC_CCV_REFERENCE_STATES - 1L
MC_CLUSTER_DF <- MC_CCV_DF
MC_CCV_METHOD <- "finite_design_covariance_ccv"
MC_CCV_REFERENCE_DESIGN <- "balanced_cyclic_aewr_path_assignment"
MC_PRIMARY_MODEL_ID <- "chamberlain_rich"

# A change is measured in log percentage points: 100 * Delta log(AEWR).
```

### MC treatment horizons and basis

Grounds the binding-margin moderator, current/lag horizons, powers, and cross-horizon products available in the design.

- Snippet ID: `mc-treatment-basis`
- Source: `code/designs/mundlak_chamberlain/design.R`
- Stable selector: `exact lines 'MC_Z_VARIABLE <- "aewr_bite"' … 'MC_HIERARCHY_LEVELS <- c("county", "market", "state", "region")'`
- Source-file SHA-256: `8c30073569b87e1c677cead4dd37c12aab2be5aae91f3a0f03bf57ccb92a4fef`
- Extracted-text SHA-256: `21df4b86e63586b032a130db980f6a519bc85584c45834183cd6df2c21e24a4c`
- Validation: `r-parse`

<!-- grounding-snippet:mc-treatment-basis excerpt-sha256=21df4b86e63586b032a130db980f6a519bc85584c45834183cd6df2c21e24a4c -->
```r
MC_Z_VARIABLE <- "aewr_bite"
MC_Z_COLUMN <- "mc_z"
MC_Z_LABEL <- "Baseline AEWR bite (standard deviations)"
MC_DYNAMIC_HORIZONS <- c(
  contemporaneous = "mc_dose_current",
  one_year = "mc_dose_lag1",
  two_year = "mc_dose_lag2"
)

MC_TREATMENT_BASIS_TERMS <- c(
  unname(MC_DYNAMIC_HORIZONS),
  "mc_dose_current_sq",
  "mc_dose_current_cu",
  "mc_dose_lag1_sq",
  "mc_dose_lag1_cu",
  "mc_dose_lag2_sq",
  "mc_dose_lag2_cu",
  "mc_dose_current_x_lag1",
  "mc_dose_current_x_lag2",
  "mc_dose_lag1_x_lag2",
  "mc_dose_current_x_lag1_x_lag2"
)

# The state x CZ x AEWR-region cell creates a strict hierarchy:
# AEWR region > state > local market cell > county.
```

### MC local-market identifier

Grounds the state by CZ by AEWR-region hierarchy used for multilevel components.

- Snippet ID: `mc-market-id`
- Source: `code/designs/mundlak_chamberlain/design.R`
- Stable selector: `r function 'mc_make_market_id'`
- Source-file SHA-256: `8c30073569b87e1c677cead4dd37c12aab2be5aae91f3a0f03bf57ccb92a4fef`
- Extracted-text SHA-256: `49dd482e1a31692186ca1be480e1d679fe487244c38358dce3204dbbdd216cbe`
- Validation: `r-parse`

<!-- grounding-snippet:mc-market-id excerpt-sha256=49dd482e1a31692186ca1be480e1d679fe487244c38358dce3204dbbdd216cbe -->
```r
mc_make_market_id <- function(aewr_region_id, state_fips, cz_id) {
  ifelse(
    is.na(aewr_region_id) | is.na(state_fips) | is.na(cz_id),
    NA_character_,
    paste(
      sprintf("%02d", as.integer(aewr_region_id)),
      sprintf("%02d", as.integer(state_fips)),
      as.character(cz_id),
      sep = "_"
    )
  )
}
```

### MC calendar registry construction

Grounds how preperiod lengths, start delays, and treatment horizons expand into valid calendar rows.

- Snippet ID: `mc-calendar-registry`
- Source: `code/designs/mundlak_chamberlain/specification_program.R`
- Stable selector: `r function 'mc_sp_calendar_registry'`
- Source-file SHA-256: `babb47d165a7057ac377d57df3951dc8efc9ab3e7560f92c64856db0a9293c48`
- Extracted-text SHA-256: `ccc5c467f43ccdaf054d1d57b5f01dda26849b248294fc3fef0dc6a79b1415bd`
- Validation: `r-parse`

<!-- grounding-snippet:mc-calendar-registry excerpt-sha256=ccc5c467f43ccdaf054d1d57b5f01dda26849b248294fc3fef0dc6a79b1415bd -->
```r
mc_sp_calendar_registry <- function() {
  rows <- list()
  for (horizon_count in 1:3) {
    maximum_lag <- horizon_count - 1L
    earliest_analysis <- min(MC_SPEC_HISTORY_YEARS) + maximum_lag
    analysis_starts <- earliest_analysis +
      MC_SPEC_ANALYSIS_START_DELAYS

    for (analysis_start in analysis_starts) {
      for (preperiod_length in MC_SPEC_PREPERIOD_LENGTHS) {
        for (preperiod_start in 2008:2012) {
          preperiod_end <- preperiod_start + preperiod_length - 1L
          admissible <- (
            preperiod_end <= 2012L &&
              preperiod_end < analysis_start - maximum_lag
          )
          if (!admissible) {
            next
          }
          rows[[length(rows) + 1L]] <- data.frame(
            horizon_count = as.integer(horizon_count),
            maximum_lag = as.integer(maximum_lag),
            preperiod_start = as.integer(preperiod_start),
            preperiod_end = as.integer(preperiod_end),
            preperiod_length = as.integer(preperiod_length),
            analysis_start = as.integer(analysis_start),
            analysis_end = MC_SPEC_ANALYSIS_END,
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }

  registry <- unique(do.call(rbind, rows))
  registry <- registry[
    order(
      registry$horizon_count,
      registry$analysis_start,
      registry$preperiod_start,
      registry$preperiod_end
    ),
    ,
    drop = FALSE
  ]
  rownames(registry) <- NULL
  registry$calendar_id <- sprintf(
    "h%d_a%d_p%d_%d",
    registry$horizon_count,
    registry$analysis_start,
    registry$preperiod_start,
    registry$preperiod_end
  )
  registry <- registry[
    c(
      "calendar_id",
      "horizon_count",
      "maximum_lag",
      "preperiod_start",
      "preperiod_end",
      "preperiod_length",
      "analysis_start",
      "analysis_end"
    )
  ]

  counts <- table(registry$horizon_count)
  if (
    nrow(registry) != 54L ||
      !identical(unname(as.integer(counts)), rep(18L, 3L))
  ) {
    stop(
      "The calendar compiler must emit 54 records, 18 per horizon count.",
      call. = FALSE
    )
  }
  registry
}
```

### MC specification registry

Grounds the cross-product of calendars, polynomial degree, richness, heredity, covariance, and primary/default flags.

- Snippet ID: `mc-specification-registry`
- Source: `code/designs/mundlak_chamberlain/specification_program.R`
- Stable selector: `r function 'mc_specification_registry'`
- Source-file SHA-256: `babb47d165a7057ac377d57df3951dc8efc9ab3e7560f92c64856db0a9293c48`
- Extracted-text SHA-256: `0e9696aba12062d937c01510fbbe9c92e071474433afa30915fb2a2fe6d5ab80`
- Validation: `r-parse`

<!-- grounding-snippet:mc-specification-registry excerpt-sha256=0e9696aba12062d937c01510fbbe9c92e071474433afa30915fb2a2fe6d5ab80 -->
```r
mc_specification_registry <- function() {
  calendars <- mc_sp_calendar_registry()
  rows <- list()
  for (calendar_index in seq_len(nrow(calendars))) {
    calendar <- calendars[calendar_index, , drop = FALSE]
    for (degree in MC_SPEC_POLYNOMIAL_DEGREES) {
      for (richness_tier in MC_SPEC_RICHNESS_TIERS) {
        rows[[length(rows) + 1L]] <- data.frame(
          calendar,
          polynomial_degree = as.integer(degree),
          moderated_polynomial_degree = as.integer(degree),
          richness_tier = as.integer(richness_tier),
          richness_label =
            MC_SPEC_RICHNESS_LABELS[[richness_tier + 1L]],
          heredity = "strong",
          history_rule = "maximal_common_basis",
          covariance_method = MC_CCV_METHOD,
          df_adjustment = MC_SPEC_DF_ADJUSTMENT,
          stringsAsFactors = FALSE
        )
      }
    }
  }
  registry <- do.call(rbind, rows)
  rownames(registry) <- NULL
  registry$spec_id <- sprintf(
    "%s_d%d_r%d",
    registry$calendar_id,
    registry$polynomial_degree,
    registry$richness_tier
  )
  registry$primary_target <- with(
    registry,
    horizon_count == 3L &
      preperiod_start == min(MC_BASELINE_YEARS) &
      preperiod_end == max(MC_BASELINE_YEARS) &
      analysis_start == min(MC_ANALYSIS_YEARS) &
      analysis_end == max(MC_ANALYSIS_YEARS) &
      polynomial_degree == 2L &
      richness_tier == max(MC_SPEC_RICHNESS_TIERS)
  )
  registry$primary_family <- with(
    registry,
    horizon_count == 3L &
      preperiod_start == min(MC_BASELINE_YEARS) &
      preperiod_end == max(MC_BASELINE_YEARS) &
      analysis_start == min(MC_ANALYSIS_YEARS) &
      analysis_end == max(MC_ANALYSIS_YEARS) &
      polynomial_degree == 2L
  )
  registry$compact_lag_basis <- with(
    registry,
    preperiod_start == min(MC_BASELINE_YEARS) &
      preperiod_end == max(MC_BASELINE_YEARS) &
      analysis_start == min(MC_ANALYSIS_YEARS) &
      analysis_end == max(MC_ANALYSIS_YEARS) &
      richness_tier == 1L
  )
  registry$compact_calendar <- with(
    registry,
    horizon_count == 3L &
      polynomial_degree == 2L &
      richness_tier == 1L &
      (
        (
          analysis_start == min(MC_ANALYSIS_YEARS) &
            paste(preperiod_start, preperiod_end) %in%
              c("2008 2009", "2009 2010")
        ) |
          (
            preperiod_start == min(MC_BASELINE_YEARS) &
              preperiod_end == max(MC_BASELINE_YEARS) &
              analysis_start %in% c(2014L, 2015L)
          )
      )
  )
  registry$default_execution <- with(
    registry,
    primary_family | compact_lag_basis | compact_calendar
  )
  registry$candidate_status <- "candidate"

  if (
    nrow(registry) != 648L ||
      anyDuplicated(registry$spec_id) > 0L ||
      sum(registry$primary_target) != 1L
  ) {
    stop(
      "The specification compiler must emit 648 unique candidates and one target.",
      call. = FALSE
    )
  }
  registry
}
```

### MC execution stages and queue sizes

Grounds primary, compact, and exhaustive selection and their expected 4, 16, and 648 specification counts.

- Snippet ID: `mc-execution-registry`
- Source: `code/designs/mundlak_chamberlain/specification_program.R`
- Stable selector: `r function 'mc_sp_execution_registry'`
- Source-file SHA-256: `babb47d165a7057ac377d57df3951dc8efc9ab3e7560f92c64856db0a9293c48`
- Extracted-text SHA-256: `c84627e351e9fd1120984c28b2b6138cafb99dcfeaf6d30f652df61bc3c9aa23`
- Validation: `r-parse`

<!-- grounding-snippet:mc-execution-registry excerpt-sha256=c84627e351e9fd1120984c28b2b6138cafb99dcfeaf6d30f652df61bc3c9aa23 -->
```r
mc_sp_execution_registry <- function(
  registry,
  stage = MC_SPEC_DEFAULT_STAGE
) {
  supported <- c("primary", "compact", "exhaustive")
  if (!stage %in% supported) {
    stop(
      "MC_SPEC_STAGE must be one of: ",
      paste(supported, collapse = ", "),
      call. = FALSE
    )
  }
  selected <- switch(
    stage,
    primary = registry$primary_family,
    compact = registry$default_execution,
    exhaustive = rep(TRUE, nrow(registry))
  )
  result <- registry[selected, , drop = FALSE]
  result$execution_stage <- stage
  result$execution_reason <- ifelse(
    result$primary_family,
    "primary_family",
    ifelse(
      result$compact_lag_basis,
      "lag_basis_sensitivity",
      ifelse(
        result$compact_calendar,
        "calendar_sensitivity",
        "exhaustive_grid"
      )
    )
  )
  result$execution_priority <- match(
    result$execution_reason,
    c(
      "primary_family",
      "lag_basis_sensitivity",
      "calendar_sensitivity",
      "exhaustive_grid"
    )
  )
  result <- result[
    order(
      result$execution_priority,
      -result$richness_tier,
      result$horizon_count,
      result$polynomial_degree,
      result$calendar_id
    ),
    ,
    drop = FALSE
  ]
  rownames(result) <- NULL

  expected <- switch(
    stage,
    primary = 4L,
    compact = 16L,
    exhaustive = 648L
  )
  if (nrow(result) != expected || anyDuplicated(result$spec_id) > 0L) {
    stop(
      "Execution-stage compiler emitted ",
      nrow(result),
      " specifications; expected ",
      expected,
      ".",
      call. = FALSE
    )
  }
  result
}
```

### MC executable specification object

Shows the exact calendar, horizons, lag/polynomial degrees, history, region budget, and primary flag passed downstream.

- Snippet ID: `mc-specification-object`
- Source: `code/designs/mundlak_chamberlain/specification_program.R`
- Stable selector: `r function 'mc_sp_specification'`
- Source-file SHA-256: `babb47d165a7057ac377d57df3951dc8efc9ab3e7560f92c64856db0a9293c48`
- Extracted-text SHA-256: `8182c6fcff64b5f7f0a4374ca22fdcc9e54aa9e4aeba119a650e575ad32a9ae2`
- Validation: `r-parse`

<!-- grounding-snippet:mc-specification-object excerpt-sha256=8182c6fcff64b5f7f0a4374ca22fdcc9e54aa9e4aeba119a650e575ad32a9ae2 -->
```r
mc_sp_specification <- function(registry_row) {
  if (!is.data.frame(registry_row) || nrow(registry_row) != 1L) {
    stop("A specification must be one registry row.", call. = FALSE)
  }
  list(
    spec_id = registry_row$spec_id[[1]],
    calendar_id = registry_row$calendar_id[[1]],
    preperiod_years = seq.int(
      registry_row$preperiod_start[[1]],
      registry_row$preperiod_end[[1]]
    ),
    analysis_years = seq.int(
      registry_row$analysis_start[[1]],
      registry_row$analysis_end[[1]]
    ),
    horizon_count = registry_row$horizon_count[[1]],
    treatment_columns = unname(MC_DYNAMIC_HORIZONS)[
      seq_len(registry_row$horizon_count[[1]])
    ],
    lag_orders = 0:registry_row$maximum_lag[[1]],
    polynomial_degrees =
      seq_len(registry_row$polynomial_degree[[1]]),
    moderated_polynomial_degrees =
      seq_len(registry_row$moderated_polynomial_degree[[1]]),
    richness_tier = registry_row$richness_tier[[1]],
    richness_label = registry_row$richness_label[[1]],
    heredity = registry_row$heredity[[1]],
    history_years = MC_SPEC_HISTORY_YEARS,
    region_budget = MC_SPEC_REGION_BUDGET,
    primary_target = registry_row$primary_target[[1]]
  )
}
```

### MC per-year region-coordinate audit

Grounds the explicit causal/history coordinate ledger and fail status at the region budget.

- Snippet ID: `mc-region-budget-audit`
- Source: `code/designs/mundlak_chamberlain/specification_program.R`
- Stable selector: `r function 'mc_sp_region_budget_audit'`
- Source-file SHA-256: `babb47d165a7057ac377d57df3951dc8efc9ab3e7560f92c64856db0a9293c48`
- Extracted-text SHA-256: `6d8c61142ba83c6188ee2736763792c98a0ba1513a985c872a4b4d30da31f8eb`
- Validation: `r-parse`

<!-- grounding-snippet:mc-region-budget-audit excerpt-sha256=6d8c61142ba83c6188ee2736763792c98a0ba1513a985c872a4b4d30da31f8eb -->
```r
mc_sp_region_budget_audit <- function(registry) {
  rows <- list()
  for (index in seq_len(nrow(registry))) {
    specification <- mc_sp_specification(
      registry[index, , drop = FALSE]
    )
    history <- mc_sp_history_selection(specification)
    for (outcome_year in specification$analysis_years) {
      history_count <- sum(
        history$outcome_year == outcome_year & history$kept
      )
      forced_count <- 1L +
        length(mc_sp_treatment_columns(specification)) *
          length(specification$polynomial_degrees)
      total <- forced_count + history_count
      rows[[length(rows) + 1L]] <- data.frame(
        spec_id = specification$spec_id,
        outcome_year = as.integer(outcome_year),
        year_coordinate = 1L,
        causal_region_coordinates =
          length(mc_sp_treatment_columns(specification)) *
            length(specification$polynomial_degrees),
        candidate_history_coordinates = sum(
          history$outcome_year == outcome_year
        ),
        retained_history_coordinates = history_count,
        region_coordinates = total,
        region_budget = specification$region_budget,
        status = ifelse(
          total <= specification$region_budget,
          "budget_admissible",
          "region_budget_exceeded"
        ),
        stringsAsFactors = FALSE
      )
    }
  }
  audit <- do.call(rbind, rows)
  if (any(audit$region_coordinates > audit$region_budget)) {
    stop("The history compiler emitted an over-budget record.", call. = FALSE)
  }
  audit
}
```

### MC runtime memory budget

Grounds dense/Gram matrix memory estimates and the stop condition before an unsafe fit.

- Snippet ID: `mc-resource-budget`
- Source: `code/designs/mundlak_chamberlain/specification_program.R`
- Stable selector: `r function 'mc_sp_resource_budget'`
- Source-file SHA-256: `babb47d165a7057ac377d57df3951dc8efc9ab3e7560f92c64856db0a9293c48`
- Extracted-text SHA-256: `84800e74f7554a04e540ab68588373ae4c2687d56df30cd65294303dd356a21f`
- Validation: `r-parse`

<!-- grounding-snippet:mc-resource-budget excerpt-sha256=84800e74f7554a04e540ab68588373ae4c2687d56df30cd65294303dd356a21f -->
```r
mc_sp_resource_budget <- function(observations, parameters) {
  dense_gib <- observations * parameters * 8 / 1024^3
  gram_gib <- parameters^2 * 8 / 1024^3
  estimated_peak_gib <-
    MC_SPEC_DENSE_PEAK_COPIES * dense_gib +
      MC_SPEC_GRAM_PEAK_COPIES * gram_gib
  list(
    dense_matrix_gib = dense_gib,
    gram_matrix_gib = gram_gib,
    estimated_peak_gib = estimated_peak_gib,
    dense_matrix_guard_gib = mc_sp_numeric_environment(
      "MC_SPEC_MAX_DENSE_GIB",
      MC_SPEC_MAX_DENSE_MATRIX_GIB
    ),
    estimated_peak_guard_gib = mc_sp_numeric_environment(
      "MC_SPEC_MAX_PEAK_GIB",
      MC_SPEC_MAX_ESTIMATED_PEAK_GIB
    )
  )
}
```

### MC assignment-column discovery

Grounds how region-level treatment-history coordinates are resolved before finite-design reassignment.

- Snippet ID: `mc-ccv-assignment-columns`
- Source: `code/designs/mundlak_chamberlain/helpers.R`
- Stable selector: `r function 'mc_ccv_assignment_columns'`
- Source-file SHA-256: `7fc8f2aa18e58ba62371b38980e5b6d6503b5597b24b743c5934dee27e129cb8`
- Extracted-text SHA-256: `47577d22471fd463d3fd12e842caf6f883a3683d63a3f71b14524e71b1d1663b`
- Validation: `r-parse`

<!-- grounding-snippet:mc-ccv-assignment-columns excerpt-sha256=47577d22471fd463d3fd12e842caf6f883a3683d63a3f71b14524e71b1d1663b -->
```r
mc_ccv_assignment_columns <- function(data, metadata) {
  columns <- unique(c(
    unname(MC_DYNAMIC_HORIZONS),
    "mc_dose_lead1",
    metadata$region_treatment_history_map$constructed_column
  ))
  absent <- setdiff(columns, names(data))
  if (length(absent) > 0L) {
    stop(
      "CCV assignment columns are absent: ",
      paste(absent, collapse = ", "),
      call. = FALSE
    )
  }
  columns
}
```

### MC balanced cyclic reference state

Grounds construction of one finite-design assignment state and its balance checks.

- Snippet ID: `mc-ccv-reference-state`
- Source: `code/designs/mundlak_chamberlain/helpers.R`
- Stable selector: `r function 'mc_ccv_reference_state'`
- Source-file SHA-256: `7fc8f2aa18e58ba62371b38980e5b6d6503b5597b24b743c5934dee27e129cb8`
- Extracted-text SHA-256: `4ffd28c3fc36a6ee01626edba82a79aa448a453c2cd4825c67386cb0bd3955b5`
- Validation: `r-parse`

<!-- grounding-snippet:mc-ccv-reference-state excerpt-sha256=4ffd28c3fc36a6ee01626edba82a79aa448a453c2cd4825c67386cb0bd3955b5 -->
```r
mc_ccv_reference_state <- function(
  data,
  metadata,
  state_index
) {
  regions <- sort(unique(data$aewr_region_id))
  region_count <- length(regions)
  if (
    region_count != MC_CCV_REFERENCE_STATES ||
      !state_index %in% 0:(region_count - 1L)
  ) {
    stop(
      "CCV requires one cyclic state for each of the 17 AEWR regions.",
      call. = FALSE
    )
  }

  assignment_columns <- mc_ccv_assignment_columns(data, metadata)
  lookup_columns <- c(
    "aewr_region_id",
    "year",
    assignment_columns
  )
  lookup <- unique(data[, lookup_columns, drop = FALSE])
  lookup_key <- paste(
    lookup$aewr_region_id,
    lookup$year,
    sep = ":"
  )
  if (anyDuplicated(lookup_key) > 0L) {
    stop(
      "AEWR assignment variables are not unique within region-year cells.",
      call. = FALSE
    )
  }

  recipient_position <- match(data$aewr_region_id, regions)
  donor_position <- (
    recipient_position - 1L + as.integer(state_index)
  ) %% region_count + 1L
  donor_region <- regions[donor_position]
  donor_key <- paste(donor_region, data$year, sep = ":")
  donor_row <- match(donor_key, lookup_key)
  if (anyNA(donor_row)) {
    stop(
      "A CCV reference state lacks a donor region-year treatment cell.",
      call. = FALSE
    )
  }

  state_data <- data
  state_data[, assignment_columns] <-
    lookup[donor_row, assignment_columns, drop = FALSE]

  # Linear, quadratic, cubic, and cross-horizon columns must all describe the
  # reassigned path.  Recomputing them here prevents a hybrid state in which a
  # linear dose comes from the donor but a polynomial term comes from the
  # observed recipient path.
  mc_refresh_treatment_basis(state_data)
}
```

### MC finite-design CCV covariance

Grounds the observed fit, cyclic reference refits, score covariance, rank checks, degrees of freedom, and returned inference metadata.

- Snippet ID: `mc-design-covariance-ccv`
- Source: `code/designs/mundlak_chamberlain/helpers.R`
- Stable selector: `r function 'mc_design_covariance_ccv'`
- Source-file SHA-256: `7fc8f2aa18e58ba62371b38980e5b6d6503b5597b24b743c5934dee27e129cb8`
- Extracted-text SHA-256: `45795be480dd6ebd04b163a06bcf61ad55365fcf4ce3483e5895ddcfefa93c0d`
- Validation: `r-parse`

<!-- grounding-snippet:mc-design-covariance-ccv excerpt-sha256=45795be480dd6ebd04b163a06bcf61ad55365fcf4ce3483e5895ddcfefa93c0d -->
```r
mc_design_covariance_ccv <- function(
  model,
  data,
  formula,
  metadata
) {
  if (!inherits(model, "fixest") || !identical(model$method, "feols")) {
    stop(
      paste(
        "The current CCV implementation is the linear OLS design",
        "formalized in the Lean file; nonlinear estimators require a",
        "separate score/Hessian argument."
      ),
      call. = FALSE
    )
  }

  coefficient_names <- names(stats::coef(model))
  residual <- stats::residuals(model)
  if (
    length(residual) != nrow(data) ||
      any(!is.finite(residual))
  ) {
    stop("CCV requires one finite fitted residual per estimation row.", call. = FALSE)
  }

  # `uhat` is held fixed across assignment states exactly as in the finite
  # design theorem.  For each reassigned design matrix X_s we solve
  #
  #   b_error(s) = (X_s' X_s)^(-1) X_s' uhat.
  #
  # Re-solving, instead of holding the observed bread fixed, incorporates the
  # random denominator/Gram matrix.  It is the vector-OLS counterpart of using
  # Dtilde/(Dtilde'Dtilde) in the scalar Lean theorem.
  ccv_formula <- mc_ccv_residual_formula(formula)
  state_count <- metadata$ccv_reference_states
  state_errors <- matrix(
    NA_real_,
    nrow = state_count,
    ncol = length(coefficient_names),
    dimnames = list(
      paste0("state_", 0:(state_count - 1L)),
      coefficient_names
    )
  )

  for (state_index in 0:(state_count - 1L)) {
    if (state_index == 0L) {
      # State zero is the observed design.  The fitted OLS residual satisfies
      # X_0' uhat = 0 by the normal equations, hence its coefficient-error
      # vector is exactly zero.  Setting that identity directly avoids feeding
      # roundoff from a second solve of this very ill-conditioned rich design
      # into the finite-state covariance.
      state_errors[1L, ] <- 0
      next
    }
    state_data <- mc_ccv_reference_state(
      data = data,
      metadata = metadata,
      state_index = state_index
    )
    state_data$.mc_ccv_residual <- residual
    state_coefficient <- fixest::feols(
      fml = ccv_formula,
      data = state_data,
      warn = FALSE,
      notes = FALSE,
      only.coef = TRUE
    )
    missing_coefficient <- setdiff(
      coefficient_names,
      names(state_coefficient)
    )
    if (
      length(missing_coefficient) > 0L ||
        any(!is.finite(state_coefficient[coefficient_names]))
    ) {
      stop(
        "CCV state ",
        state_index,
        " does not retain the observed model's coefficient basis.",
        call. = FALSE
      )
    }
    state_errors[state_index + 1L, ] <-
      state_coefficient[coefficient_names]
  }

  # The finite design has p_s = 1/17.  Thus this is a probability-weighted
  # covariance (division by 17), not the sample covariance division by 16:
  #
  #   V_dcCCV = sum_s p_s (b_error(s)-E_p[b_error])
  #                         (b_error(s)-E_p[b_error])'.
  #
  # Writing it as crossprod(centered_errors)/17 makes positive
  # semidefiniteness immediate, matching `DesignCovariance.dcCCV_nonneg`.
  centered_errors <- scale(
    state_errors,
    center = TRUE,
    scale = FALSE
  )
  covariance <- crossprod(centered_errors) / state_count
  covariance <- (covariance + t(covariance)) / 2

  if (
    any(!is.finite(covariance)) ||
      any(diag(covariance) < -1e-10)
  ) {
    stop("The constructed CCV covariance is not finite and PSD.", call. = FALSE)
  }

  # The nonzero eigenvalues of A'A equal those of AA'.  Inspecting the small
  # 17 x 17 matrix avoids an unnecessary eigendecomposition of a roughly
  # 900 x 900 coefficient covariance.
  small_kernel <- tcrossprod(centered_errors) / state_count
  kernel_eigenvalues <- eigen(
    (small_kernel + t(small_kernel)) / 2,
    symmetric = TRUE,
    only.values = TRUE
  )$values
  positive_tolerance <- max(abs(kernel_eigenvalues), 1) * 1e-10

  list(
    covariance = covariance,
    diagnostics = list(
      method = metadata$ccv_method,
      reference_design = metadata$ccv_reference_design,
      reference_states = state_count,
      design_df = state_count - 1L,
      covariance_rank = sum(kernel_eigenvalues > positive_tolerance),
      minimum_kernel_eigenvalue = min(kernel_eigenvalues),
      minimum_variance = min(diag(covariance)),
      maximum_observed_state_error = max(abs(state_errors[1L, ])),
      mean_state_error_norm = sqrt(sum(colMeans(state_errors)^2))
    )
  )
}
```

