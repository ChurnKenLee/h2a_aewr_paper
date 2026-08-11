# Multilevel Mundlak–Chamberlain AEWR dose-response design

This branch consumes `data/processed/county_year_panel.parquet` and runs the
version-3 dose-response specification program together with the retained
version-2.3 benchmark.

## Run

From the repository root:

```sh
./scripts/run_mundlak_chamberlain.sh
```

The default specification-program stage is the bounded `compact` queue.
`MC_SPEC_STAGE=exhaustive` is an explicit opt-in and is not used by the
pipeline default.

## Program stages

| Script | Responsibility |
| --- | --- |
| `01_build_panel.R` | Construct the analysis panel and design metadata |
| `01_01_build_specification_registry.R` | Compile the declared specification registry |
| `02_estimate_models.R` | Run the retained benchmark when `MC_BENCHMARK_ONLY=1` |
| `02_01_estimate_specification_program.R` | Estimate restartable specification-outcome checkpoints |
| `03_01_report_specification_program.R` | Select admissible primaries and report effects |
| `04_01_diagnostics.R` | Audit support, alignment, influence, and placebos |
| `05_generate_tables.py` | Produce retained tables and figures |
| `06_01_validate_specification_program.R` | Enforce registry, rank, resource, and gradient contracts |

The principal controls are `MC_SPEC_STAGE`, `MC_SPEC_IDS`,
`MC_OUTCOME_IDS`, `MC_SPEC_MAX`, `MC_SPEC_WORKERS`,
`MC_FIXEST_THREADS`, `MC_SPEC_MAX_DENSE_GIB`,
`MC_SPEC_MAX_PEAK_GIB`, and `MC_SPEC_FORCE`.

The resource-safe defaults use one outcome worker, four `fixest` threads, a
1.25-GiB ceiling for one dense matrix, and a 6-GiB estimated per-worker
working set. Completed compatible checkpoints are reused.

## Design documentation

Identification, treatment histories, moderators, the model hierarchy,
specification budgets, estimands, finite-design CCV inference, diagnostics,
and retained outputs are documented in the grounded
[Mundlak–Chamberlain design](../../../content/designs/mundlak-chamberlain.md).

Supporting research notes remain under `markdowns/` and `papers/metrics/`.
