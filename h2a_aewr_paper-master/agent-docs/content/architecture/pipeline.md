+++
title = "Pipeline ownership"
description = "The supported source-to-panel-to-design flow and its artifact boundaries."
weight = 2
+++

## Stages

| Stage | Owner | Boundary |
|---|---|---|
| A01 | `code/a01_sources/` | Acquire/normalize one external source; network and credentials may be required |
| B01 | `code/b01_derived/` | Produce reusable aggregates, proxies, and static propensity inputs |
| C01 | `code/c01_clean/` | Normalize source families and merge `county_year_merged.parquet` |
| C02 | `code/c02_build/` | Produce design-neutral `county_year_panel.parquet` |
| Descriptives | `code/descriptives/` | Shared retained figures |
| DiD | `code/designs/did/` | DiD panel, estimates, event study, and retained products |
| Panel IV | `code/designs/panel_iv/` | Target/donor recovery, instruments, 2SLS, and diagnostics |
| Mundlak–Chamberlain | `code/designs/mundlak_chamberlain/` | Version-3 registry, estimation, finite-design CCV, reports, and validation |

The runner projection is generated on the [pipeline contracts](@/generated/pipeline-contracts.md) page. Use it instead of transcribing the current step list into prose.
Read the execution/runtime section of [source-linked code grounding](@/generated/code-grounding.md) for the actual dispatcher, top-level runner, validation gate, and pinned environment.

## Ownership rule

An artifact has one producer. A downstream consumer may validate but may not
silently rename, backfill, or reinterpret a stale schema. Shared stages do not
own treatment definitions, instruments, design samples, fixed-effect encodings,
or inference choices.
