# H-2A Paper

This repository uses three manually run workflow layers. A scripts acquire and
normalize source data, B scripts construct derived data, and C scripts build
the analysis panels and results. There is intentionally no master runner: run
only the scripts needed for the artifact you are updating.

## Data workflow

| Layer | Directory | Responsibility |
| --- | --- | --- |
| A | [`code/a01_sources`](code/a01_sources/) | External-source acquisition and source-level preparation |
| B | [`code/b01_derived`](code/b01_derived/) | Aggregates, derived measures, predictors, and county harmonization |
| C | [`code/c01_clean`](code/c01_clean/) through [`code/c04_analysis`](code/c04_analysis/) | Analysis-panel construction, IV construction, and results |

The A and B directories document each script's inputs, outputs, dependencies,
credentials, and manual command. Their numeric prefixes group related source
families; they are not an instruction to rebuild unrelated sources.

## C workflow

| Stage | Directory | Responsibility |
| --- | --- | --- |
| Shared | [`code/c00_shared`](code/c00_shared/) | Side-effect-free helpers sourced explicitly by individual scripts |
| Clean | [`code/c01_clean`](code/c01_clean/) | Normalize upstream source data into project panels |
| Build | [`code/c02_build`](code/c02_build/) | Assemble and classify the county-year analysis panel |
| IV | [`code/c03_iv`](code/c03_iv/) | Construct FLS/OEWS calibration weights and donor-wage instruments |
| Analysis | [`code/c04_analysis`](code/c04_analysis/) | Produce thematic figures, tables, and estimates |

Each stage directory contains its manual execution order and artifact
contracts. Every R script is standalone: it loads its own packages, reads its
inputs from disk, and writes its own outputs. Scripts should be launched from
the repository root, for example:

```sh
Rscript code/c02_build/01_merge_county_panel.R
```

Opening a stage script directly in RStudio is also supported. The script loads
`code/bootstrap_paths.R`, which walks upward to the `.here` marker, activates
the repository's `renv` environment when needed, and then loads the shared path
helpers.

The A and B scripts remain the owners of upstream extraction and aggregation
artifacts. `code/utilities/pull_fred_minimum_wages.R` is an optional
external-data refresh utility and is not part of the C sequence. Legacy
scripts under `Do/` are not part of the supported workflow.

## Conventions

- `data/raw` contains source files and hand-maintained crosswalks.
- `data/intermediate` contains exchange artifacts between scripts.
- `data/processed` contains analysis-ready panels.
- `outputs/figures` and `outputs/tables` contain analysis products.
- `code/bootstrap_paths.R` locates the repository and initializes R pathing;
  `code/paths.R` is the shared path interface used after initialization.
- Scripts do not depend on objects left in an interactive R workspace.

The reorganization preserves the existing analytical definitions. Known data
issues discovered by diagnostics, including duplicate county-year keys, should
be addressed as separate substantive changes.
