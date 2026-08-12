# C01: Normalize and merge the county panel

Scripts before `13` normalize individual source families. In particular,
`09_bea_employment.R` and `10_bea_farm_income.R` publish unique county-year
BEA employment, farm-financial, farm-wage, and farm-wage-supplement totals.
`12_oews_county_year.R` writes `oews_county_year.parquet`: unique county-year
OEWS Big-Six hourly- and annual-wage proxies with geographic and publication
support fields. Script `13` joins that artifact and the separate QCEW NAICS
111, NAICS 112, and all-sector employment, wage-bill, and disclosure fields
onto the balanced county-year backbone using validated one-to-one county-year
joins.

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
Rscript --vanilla code/c01_clean/13_merge_county_panel.R
```

Rebuild only the normalized BEA inputs with:

```sh
Rscript --vanilla code/c01_clean/09_bea_employment.R
Rscript --vanilla code/c01_clean/10_bea_farm_income.R
```

The BEA artifacts retain source totals and column metadata. Hired-farm-job,
average-wage, and average-compensation statistics are delegated to downstream
design code.

Rebuild only the normalized OEWS county-year input with:

```sh
Rscript --vanilla code/c01_clean/12_oews_county_year.R
```

The OEWS and QCEW inputs and the merged artifact must be nonempty and unique by
`county_fips` and `year`. Suppressed QCEW employment and wage bills remain
null; the shared merge does not construct a QCEW average wage.

The downstream ownership boundary and supported invariants are documented in
the grounded [shared-panel contract](../../content/contracts/shared-panel.md).
The complete producer chain and join map are documented in
[Generating the shared county-year panel](../../content/architecture/shared-panel-generation.md).
