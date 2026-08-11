# C01 cleaning and merge instructions

- Scripts `01` through `12` normalize one source family. Script `13` owns the merge into `data/intermediate/county_year_merged.parquet`.
- Normalize schema at the owning source-family script. Do not add downstream aliases or repairs to make a stale local Parquet pass.
- The merged artifact must be nonempty and unique by `county_fips, year`. Diagnose duplication at the contributing producer and document intentional coverage differences.
- The glob order in `run_shared_panel.sh` is filename order. Renaming or inserting a script is an execution-order change and requires runner, documentation, and generated-contract updates.
