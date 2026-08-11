# H-2A Paper repository instructions

## Project scope

This repository is a reproducible empirical-economics paper combining R,
Python/Marimo, shell pipeline runners, and LaTeX.

- Run commands from the repository root.
- Before editing, inspect `git status --short` and preserve existing work.
- Prefer focused changes; do not perform unrelated cleanup or formatting.
- Read the root `README.md`, the nearest branch `README.md`, and any design
  document linked from it before changing pipeline or estimation behavior.

## Agent grounding

- Before planning a repository change, run
  `python scripts/agent_docs.py snapshot --scope <target-path>`. Read every
  `AGENTS.md`, README, canonical page, and high-risk assumption named by the
  snapshot.
- Run `python scripts/agent_docs.py verify` before relying on
  `static/grounding-manifest.json`. Generated context is authoritative only
  when verification passes.
- A nested `AGENTS.md` narrows these repository-wide rules for its directory;
  it never relaxes data safety, research integrity, or validation requirements.
- When sources disagree, use this authority order: executable code, runners,
  locks, and machine checks; verified generated context; canonical pages under
  `content/` and the nearest README; the root README; then historical notes and
  retained outputs.

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
  `content/contracts/geographic-codes.md`. Use
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

## Documentation

- Keep repository and branch READMEs focused on commands, execution order,
  inputs, outputs, and local ownership. Put cross-cutting contracts and
  empirical-design explanations under `content/`.
- Zola pages use `grounding` shortcodes to depend on named
  `docs-ground:start`/`docs-ground:end` source regions. When an anchored region
  changes, review the linked page and update its recorded SHA-256 digest.
- Review drift with `python scripts/agent_docs.py accept-drift --document
  <content-page> --anchor <anchor>`. The `--write` form refuses to accept a new
  digest unless that page's prose or context also changed.
- Do not refresh grounding digests automatically or edit
  `static/grounding-manifest.json` and `static/llms.txt` directly. Run
  `python scripts/agent_docs.py generate` after reviewed documentation changes.
- Update `agent/assumptions.toml` when a high-risk invariant, owner, source,
  affected scope, or review trigger changes. Its text checks are guardrails,
  not proof of empirical validity.

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
- Documentation changes: `zola check --skip-external-links` and `zola build`
- Agent context and grounding: `python scripts/agent_docs.py verify` and
  `python scripts/test_agent_docs.py`
- GitHub workflow changes: `actionlint`

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
