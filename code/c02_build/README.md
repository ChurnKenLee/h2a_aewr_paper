# C02: Build the shared county-year panel

`01_build_county_panel.R` consumes
`data/intermediate/county_year_merged.parquet` and writes
`data/processed/county_year_panel.parquet`.

## Run

From the repository root:

```sh
Rscript --vanilla code/c02_build/01_build_county_panel.R
```

The output must be nonempty and unique by `county_fips` and `year`.
The QCEW crop, animal, and all-sector source totals and disclosure flags pass
through unchanged. The producer validates disclosed/non-disclosed semantics
but does not construct design-specific wages, instruments, or timing shifts.

## Panel contract

The shared panel's ownership boundaries, reusable variables, static prediction
semantics, and validation invariants are documented in the grounded
[shared-panel contract](../../content/contracts/shared-panel.md).
The complete C01 merge, C02 transformation, and rebuild boundary are documented
in [Generating the shared county-year panel](../../content/architecture/shared-panel-generation.md).
