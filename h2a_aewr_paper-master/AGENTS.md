# H-2A AEWR paper: repository instructions

## Start every task by grounding

1. Work from the repository root. Run `python scripts/agent_grounding.py snapshot --scope <target-path>` before planning a change.
2. Run `python scripts/agent_grounding.py verify` before trusting generated agent documentation. If it reports drift, inspect the named source and regenerate with `python scripts/agent_grounding.py generate`; never paper over a failed check.
   Source-linked code-excerpt drift is different: inspect the displayed diff and use `accept-snippet-drift --id <id> --write` only after deciding the new excerpt still states the intended contract. Ordinary generation is intentionally unable to accept that drift.
3. Read the nearest nested `AGENTS.md` plus the pages named by the snapshot. A nested file narrows these rules; it does not relax research-integrity, data-safety, or validation requirements.
4. Establish whether required data, credentials, R packages, Python packages, GPU support, and network access actually exist. This repository intentionally does not contain `data/`; never equate a static or dry-run pass with a completed empirical pipeline.

## Authority and freshness

When sources disagree, use this order and report the conflict:

1. Executable code, pipeline runners, lockfiles, and machine-checked contracts.
2. Generated pages under `agent-docs/content/generated/`, but only after `agent_grounding.py verify` passes.
3. Curated pages under `agent-docs/content/` and the nearest code-directory README.
4. The root README and retained output artifacts.
5. Draft prose, `markdowns/`, papers, PDFs, and Word files.

The two Word files in `documentation/` are historical design notes, not current implementation contracts. In particular, the 2023 border-enforcement shift-share idea and the 2024 Nielsen/FAF gravity-price proposal are not the current supported DiD, panel-IV, Mundlak–Chamberlain, or NASS/CDL Fisher implementations.

Do not preserve an assertion merely because it appears in prose. Find its producing code, runner, lockfile, or explicit assumption check. If no executable source supports it, label it proposed, historical, or unverified.

## Repository architecture

- `code/a01_sources/`: external acquisition and source normalization; may use APIs, network access, credentials, and costly geospatial work.
- `code/b01_derived/`: reusable transformations and static H-2A propensity models.
- `code/c00_shared/`: design-neutral R helpers only.
- `code/c01_clean/`: source-family normalization and the merged county-year exchange artifact.
- `code/c02_build/`: design-neutral shared county-year panel.
- `code/descriptives/`: retained shared figures.
- `code/designs/did/`, `panel_iv/`, and `mundlak_chamberlain/`: mutually distinct empirical designs. Do not leak treatment definitions, instruments, samples, or inference rules across branches.
- `scripts/`: the supported execution order. Directory numbering is informative; runners are authoritative.
- `outputs/`: retained manuscript products, not general scratch space.
- `agent-docs/`: Zola source for agent-oriented documentation. Generated pages are projections of repository state.
- `agent-docs/snippets.toml`: reviewed source selectors and excerpt digests for authoritative grounding fences. `content/generated/code-grounding.md` and `static/grounding-snippets.json` are derived from it.

## Environment and commands

- Enter the pinned environment with `devenv shell`. Python is uv-managed and project-local; R packages are renv-managed.
- Python target: the `.python-version` and `pyproject.toml` contract. Use `uv run --no-sync` only after the environment has been restored; do not silently resolve a new environment.
- R target: the R version and packages in `renv.lock`. Run pipeline entry points with `Rscript --vanilla` as the runners do.
- Full pipeline: `./scripts/run_all.sh`. It is network-, credential-, data-, CPU-, memory-, and sometimes GPU-intensive; do not run it unless the task requires it and prerequisites are established.
- Execution-order check: `DRY_RUN=1 ./scripts/run_all.sh`.
- Fast repository checks: `./scripts/run_tests.sh`.
- Agent docs: `agent-docs-generate`, `agent-docs-check`, and `agent-docs-serve` inside `devenv shell`, or the equivalent commands documented in `agent-docs/README.md`.

Run the narrowest meaningful checks while iterating and `./scripts/run_tests.sh` before handing off changes to code, runners, contracts, AGENTS files, or agent documentation. State every check that was skipped and why.

## Data, credentials, and side effects

- Treat `data/raw` and hand-maintained crosswalks as irreplaceable inputs. Never delete, rewrite, refresh, or bulk-download them unless explicitly requested.
- `data/intermediate` artifacts have one producer. Fix the producer instead of repairing a downstream copy.
- `data/processed` is analysis-ready output. Preserve documented keys and schemas.
- Keep secrets in process environment variables loaded from an ignored `.env`. Never print, commit, move into Nix, or place them in documentation.
- Source stages can call IPUMS, Census, Gemini, Google Places, FRED, USDA, BLS, NOAA, and other services. Confirm authorization, expected cost, rate limits, caching, and target paths before running them.
- Do not modify retained tables or figures by hand. Regenerate them from the owning script, or explicitly identify a prose-only edit.

## Research-integrity invariants

- Preserve the geographic contract in `documentation/geographic_code_contract.md`: identifiers are strings; county geography uses the 2010 vintage; `46102` maps to `46113` where specified.
- The shared panel is design-neutral. Treatment, post indicators, fixed-effect encodings, target clusters, instruments, and design samples belong in their design branch.
- Never change an estimand, treatment timing, normalization, sample restriction, fixed effect, cluster, instrument, weighting rule, or inference method merely to make a test pass. Surface the substantive change and update its assumption record.
- Do not interpret software validation as causal validation. Report weak support, placebo failures, extrapolation, rank limitations, or unstable specifications rather than suppressing them.
- Preserve canonical H-2A prediction metadata and reject mixed or stale model specifications as the current code does.
- Joins and persistent outputs must keep canonical geographic strings and enforce their documented uniqueness/non-emptiness contracts.

## Editing conventions

- Use `code/paths.R` for supported R paths, `code/c00_shared/geography.R` for R geographic normalization, and `h2a.geography` for Python geographic normalization.
- Keep new Python compatible with the project target. Do not add `from __future__ import annotations` to new files.
- Prefer explicit schemas, named constants, deterministic ordering, atomic writes, and fail-closed validation.
- Preserve resumability and caches in expensive source/model stages.
- Do not add a dependency without explaining why the locked R, Python, or Nix environment cannot already provide the capability; update the relevant lockfile in the same change.
- Update the nearest README, curated agent page, assumption registry, and generated grounding outputs when behavior or a contract changes.
- Never paste repository code into an agent-grounding page. Register a stable symbol or exact-boundary selector in `agent-docs/snippets.toml`; classify non-authoritative literal fences as illustrative, pseudocode, or expected output.

## Completion evidence

In the handoff, distinguish:

- static checks and syntax checks;
- dry-run ordering checks;
- tests using fixtures or existing artifacts;
- stages actually executed on empirical data;
- outputs actually regenerated and inspected.

Never claim reproducibility, numerical equivalence, or end-to-end success beyond the evidence obtained in the current environment.

## Code review rules

- Flag undocumented changes to samples, years, treatment definitions, lags, normalizations, outcome denominators, fixed effects, clusters, instruments, weights, covariance estimators, support rules, and output ownership.
- Flag downstream schema repair that should occur in the owning producer.
- Flag direct edits to generated agent pages or retained analytical outputs.
- Flag prose that presents a proposed or historical design as current.
- Flag network/API stages that are made non-resumable, uncached, nondeterministic without justification, or liable to expose credentials.
- Flag documentation claims not connected to an executable check or named source of truth.
- Flag literal grounding fences, missing fence languages, unstable line-number selectors, or acceptance of snippet drift without reviewing the source/excerpt diff.
