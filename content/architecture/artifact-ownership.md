+++
title = "Artifact ownership"
description = "Which pipeline stage owns each persistent artifact."

[extra]
scopes = ["code", "scripts"]
+++

Each persistent artifact has one owning producer. Schema corrections and
transformations belong in that producer rather than in downstream repair code.

| Stage | Responsibility | Principal artifact class |
| --- | --- | --- |
| `code/a01_sources` | Acquire and normalize external sources | Source Parquets |
| `code/b01_derived` | Construct reusable measures and prediction inputs | Derived Parquets |
| `code/c01_clean` | Normalize source families and merge the county-year backbone | `county_year_merged.parquet` |
| `code/c02_build` | Construct the design-neutral shared panel | `county_year_panel.parquet` |
| `code/descriptives` | Produce design-neutral figures | Retained figures |
| `code/designs/*` | Construct treatments, instruments, samples, and estimators | Design panels and results |

{{ grounding(path="scripts/run_all.sh", anchor="pipeline-order", sha256="6ce943803a89bf2ce24018b67f7d970cdafdaa846a5ecf3eed14c0c64d8ecccf") }}

Storage locations are part of the contract:

- `data/raw` contains source files and hand-maintained crosswalks.
- `data/intermediate` contains exchange artifacts owned by one producer.
- `data/processed` contains analysis-ready panels.
- `outputs/figures` and `outputs/tables` contain retained manuscript products.

The shared stages remain design-neutral. Treatment definitions, post
indicators, fixed-effect factors, target clusters, and instruments are owned by
their design branches.
