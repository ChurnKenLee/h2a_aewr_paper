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
The common baseline is the complete 2008--2025 county-year calendar declared
in `code/paths.R`. Run `./scripts/run_shared_panel.sh` to rebuild C01, C02,
and `02_validate_source_coverage.py`. The validator writes
`outputs/reports/shared_panel_validation.json`,
`outputs/tables/shared_panel_source_coverage.csv`, and
`outputs/tables/shared_panel_h2a_source_cohorts.csv`. These report numeric
coverage and recent start-year/fiscal-year/occupation cohorts without imposing
a design sample. They reconcile certifications to the owning B01 aggregate.

Employer counts are positive-certified linked worksite-business counts under
the balanced and sensitivity v2 partitions. The producer validates the one
FY2008--2026 Q4 linkage contract (review `complete` or `pending`) and fills its provenance across
zero-use county-years.
The QCEW crop, animal, and all-sector source totals and disclosure flags pass
through unchanged. The producer validates disclosed/non-disclosed semantics
but does not construct design-specific wages, instruments, or timing shifts.

## Panel contract

The shared panel's ownership boundaries, reusable variables, static prediction
semantics, and validation invariants are documented in the grounded
[shared-panel contract](../../content/contracts/shared-panel.md).
The complete C01 merge, C02 transformation, and rebuild boundary are documented
in [Generating the shared county-year panel](../../content/architecture/shared-panel-generation.md).

Missing statutory AEWR coverage no longer deletes observed outcome rows.
The 2009 wage stays null with source status and a separately named historical
application-wage comparator. Wage-complete samples belong to each design.

Requested positions and requested hours retain B01's missing values. C02 fills
these measures with zero only for county-years absent from the H-2A aggregate;
it uses the aggregate's pre-fill provenance to distinguish absence from an
unknown total. Rebuild B01 `01` and C01 `05`/`13` first when these inputs change.
