# Pipeline runners

From the repository root, run:

```sh
./scripts/run_all.sh
```

Individual stages are:

```sh
./scripts/run_sources.sh
./scripts/run_derived.sh
./scripts/run_analysis.sh
```

Use a dry run to print the execution order without running anything:

```sh
DRY_RUN=1 ./scripts/run_all.sh
```

The scripts stop at the first error. Source steps require their documented raw
files, API credentials in `.env`, `uv`, and the restored R environment.
Marimo applications are exported to temporary flat scripts for noninteractive
execution.

A10–A12 are not consumed by the analysis pipeline. Run them separately with
`./scripts/run_optional_sources.sh`.
