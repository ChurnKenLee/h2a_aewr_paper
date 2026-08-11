# H-2A Paper repository instructions

## Project scope

This repository is a reproducible empirical-economics paper combining R,
Python/Marimo, shell pipeline runners, and LaTeX.

- Run commands from the repository root.
- Before editing, inspect `git status --short` and preserve existing work.
- Prefer focused changes; do not perform unrelated cleanup or formatting.
- Read the root `README.md`, the nearest branch `README.md`, and any design
  document linked from it before changing pipeline or estimation behavior.

## Pipeline architecture

Respect the documented dependency graph:

`a01_sources` -> `b01_derived` -> `c01_clean` -> `c02_build` -> design branches

- Each persistent artifact has one owning producer. Fix schemas and
  transformations at that producer rather than adding downstream repairs.
- Treat `data/raw` and hand-maintained crosswalks as inputs. Modify them only
  when the task explicitly concerns source data or a crosswalk.
- Write exchange artifacts to `data/intermediate`, analysis-ready panels to
  `data/processed`, and retained manuscript products to `outputs`.
- Keep `c01_clean` and `c02_build` design-neutral. Treatment definitions, post
  indicators, fixed-effect factors, target clusters, and instruments belong
  under the relevant `code/designs/` branch.
- Keep reusable R helpers in `code/c00_shared` and reusable Python helpers in
  `src/h2a`. Keep design-specific constants and numerical methods with their
  design.
- Use `code/paths.R` and `src/h2a/paths.py`; do not introduce hard-coded
  absolute paths.

## Data and research integrity

- Do not silently change an outcome, treatment, sample restriction, timing
  convention, weight, fixed effect, clustering level, instrument, normalization,
  seed, or resource guard.
- When the requested task changes an empirical specification, state the change
  explicitly and update the nearest README or design document.
- Changing `H2A_PREDICTION_CUTOFF_YEAR` or
  `H2A_PREDICTION_MODEL_SPEC` requires rebuilding the prediction artifact and
  all downstream panels.
- Preserve nonempty and unique artifact keys. County-year panels must be unique
  by `county_fips` and `year`.
- Geographic identifiers are strings governed by
  `documentation/geographic_code_contract.md`. Use
  `code/c00_shared/geography.R` or `h2a.geography`; never coerce geographic
  identifiers to numeric storage.
- Do not hand-edit generated tables, figures, or PDFs to change results. Modify
  the producing code and regenerate the artifact.
- Do not add manuscript claims, numerical results, or citations that cannot be
  traced to generated outputs or an identified source.

## Environment and dependencies

- Use the existing direnv/devenv environment.
- Run Python through `uv run --no-sync`. When needed in a restricted
  environment, set `UV_CACHE_DIR=/tmp/h2a-uv-cache`.
- Run R entry points with `Rscript --vanilla`.
- Do not install packages globally.
- Update `pyproject.toml` and `uv.lock` together for requested Python dependency
  changes. Update `renv.lock` for requested R dependency changes.
- Secrets belong only in `.env`. Never print, commit, or copy credential values.
- Edit Marimo source `.py` files, not `__marimo__` session/cache files or
  generated flat exports.

## Execution safety

- Start with the narrowest relevant script or branch runner.
- Use `DRY_RUN=1` to inspect pipeline order before a broad run.
- Do not run a real `scripts/run_all.sh`, source acquisition runner, optional
  source refresh, or credentialed API step unless the task explicitly requires
  it.
- Do not set `MC_SPEC_STAGE=exhaustive` unless explicitly requested; the normal
  Mundlak–Chamberlain pipeline uses its bounded declared queue.
- Preserve existing intermediate artifacts unless regeneration is required by
  the task.

## Validation

Match validation to the changed files and report exactly what ran.

- Pipeline orchestration:
  `DRY_RUN=1 ./scripts/run_all.sh`
- Shell changes:
  `bash -n scripts/*.sh`
- R syntax:
  `Rscript --vanilla -e 'for (f in list.files("code", pattern = "\\.R$", recursive = TRUE, full.names = TRUE)) parse(file = f)'`
- Ordinary Python modules:
  `UV_CACHE_DIR=/tmp/h2a-uv-cache uv run --no-sync ruff check <changed-files>`
- Marimo applications: export the changed application to a temporary file
  under `/tmp` before executing it.
- Analysis changes: run the smallest affected branch runner when its required
  local inputs exist.
- Manuscript changes: rebuild from the source under `draft/` when a suitable
  TeX tool is available and check for missing citations, references, tables,
  and figures.

The repository has pre-existing repo-wide Ruff findings, especially in Marimo
applications. Do not apply repo-wide automatic fixes; distinguish new failures
from existing ones.

## Code Review Rules

- Flag geographic identifiers stored as numbers or bypassing the shared
  normalizers.
- Flag downstream schema repairs that should occur in the owning producer.
- Flag design-specific treatment or instrument construction added to the shared
  panel.
- Flag new sample definitions based on post-treatment outcomes unless the
  design documentation explicitly justifies them.
- Flag silent changes to inference, specification, units, or preferred-column
  definitions.
- Flag manual edits to generated empirical results or exposure of credentials.
