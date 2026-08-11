# Cleaning and merge instructions

- Scripts `01` through `12` normalize individual source families. Script `13`
  owns `data/intermediate/county_year_merged.parquet`.
- Normalize a schema in its owning source-family producer. Do not add aliases
  or repairs downstream merely to accommodate stale local Parquets.
- The merged artifact must be nonempty and unique by `county_fips, year`.
  Diagnose duplication at the contributing producer.
- `run_shared_panel.sh` uses filename order. Renaming or inserting a script is
  an execution-order change requiring runner and documentation review.
