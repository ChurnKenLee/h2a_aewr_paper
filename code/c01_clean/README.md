# C01: Normalize and merge the county panel

Scripts `01` through `12` normalize individual source families. Script `13`
joins their county-, state-, CZ-, and year-level outputs onto the balanced
county-year backbone.

The branch artifact is:

```text
data/intermediate/county_year_merged.parquet
```

Run the complete shared branch with:

```sh
./scripts/run_shared_panel.sh
```

Run only the final merge after its normalized inputs exist:

```sh
Rscript code/c01_clean/13_merge_county_panel.R
```

The artifact must be nonempty and unique by `county_fips` and `year`.

The downstream ownership boundary and supported invariants are documented in
the grounded [shared-panel contract](../../content/contracts/shared-panel.md).
The complete producer chain and join map are documented in
[Generating the shared county-year panel](../../content/architecture/shared-panel-generation.md).
