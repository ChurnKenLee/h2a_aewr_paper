+++
title = "Pipeline architecture"
description = "Supported execution order and branch structure."

[extra]
scopes = ["code", "scripts"]
+++

The supported workflow separates source preparation, reusable derived data, a
design-neutral county-year panel, and empirical-design branches.

```text
code/a01_sources -> code/b01_derived -> code/c01_clean -> code/c02_build
                                                               |
                          +----------------+---------------------+-------------+
                          |                |                     |             |
                    descriptives         DiD                panel IV     MC dose response
```

The complete runner executes sources, derived data, the shared panel,
descriptives, DiD, panel IV, and Mundlak–Chamberlain in that order.

{{ grounding(path="scripts/run_all.sh", anchor="pipeline-order", sha256="6ce943803a89bf2ce24018b67f7d970cdafdaa846a5ecf3eed14c0c64d8ecccf") }}

Run the complete chain from the repository root:

```sh
./scripts/run_all.sh
```

Set `DRY_RUN=1` to print the declared order without executing analysis code.
Branch-specific commands remain documented in the repository root and branch
READMEs.
