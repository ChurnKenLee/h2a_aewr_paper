+++
title = "Drift report"
description = "Current static warnings and the checks used to reject stale grounding."
weight = 6
+++

> [!NOTE]
> Generated file. Change repository sources or `scripts/agent_grounding.py`, then regenerate.

## Verification result

Assumption records checked: **19**. Source/catalog IDs indexed: **27**. Runner contracts inspected: **10**. Source-linked code snippets verified: **47**.

All enforced repository and assumption checks passed at generation time.

## Explicit warnings and boundaries

- nix-instantiate unavailable; skipped nix-parse for devenv-test-contract
- nix-instantiate unavailable; skipped nix-parse for devenv-base-tools
- nix-instantiate unavailable; skipped nix-parse for devenv-r-renv
- The repository snapshot has no data/ directory; empirical stages were not validated.
- main.py is still a scaffold and is not a supported pipeline entry point.
- code/.codex is an empty regular file; it has no instruction/configuration effect.
- snowflake.log is retained repository noise, not a pipeline log contract.
- ccv_symlink.lean is a broken developer-local symlink; it cannot ground the CCV implementation.
- The two documentation/*.docx files are historical proposals, not active implementation contracts.

## What makes verification fail

- Any watched file hash changes without regeneration.
- A runner or README references a missing script.
- A curated assumption no longer matches its named source.
- A registered source excerpt differs from its reviewed SHA-256.
- A grounding fence is literal, unclassified, or missing a language.
- A Python, R, Bash, Nix, TOML, or JSON snippet fails its configured parser.
- The source catalog loses required structure or unique IDs.
- An AGENTS chain exceeds the default discovery budget.
- A checked local Markdown link breaks.
- A generated page or machine-readable manifest differs from its deterministic projection.
