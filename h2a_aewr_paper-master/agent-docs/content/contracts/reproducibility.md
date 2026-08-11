+++
title = "Reproducibility tiers"
description = "What is pinned, what remains external, and how to avoid overstating a successful check."
weight = 4
+++

Use the execution/runtime excerpts in [source-linked code grounding](@/generated/code-grounding.md)
to inspect the current R/Python dispatcher, strict validation gate, runner
order, and Nix/devenv environment without relying on prose copies.

## Pinned layers

- Nix/devenv inputs in `devenv.lock`.
- Python target and dependencies in `.python-version`, `pyproject.toml`, and `uv.lock`.
- R target and packages in `renv.lock`.
- Source-controlled design constants and runners.
- Grounding input hashes in `agent-docs/static/grounding-manifest.json`.

Current lock facts are generated on [runtime locks](@/generated/runtime-locks.md).

## External state

API responses, revised government files, locally held raw data, credentials,
GPU/driver compatibility, and cached geocoding/model artifacts are not made
reproducible merely by pinning code. Preserve manifests, access dates where
available, caches, model metadata, and artifact schemas.

## Validation ladder

Use the evidence labels in the [operating protocol](@/operating-protocol.md).
`./scripts/run_tests.sh` is intentionally fast and data-free. An end-to-end
claim requires actual empirical execution with the input snapshot recorded.
