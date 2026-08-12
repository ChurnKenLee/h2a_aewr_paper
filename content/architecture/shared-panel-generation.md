+++
title = "Generating the shared county-year panel"
description = "Exact producer chain, joins, derived fields, validation, and rebuild boundary for county_year_panel.parquet."

[extra]
scopes = ["scripts/run_shared_panel.sh", "code/c01_clean", "code/c02_build"]
+++

The supported shared-panel runner produces the analysis-ready file:

```text
data/processed/county_year_panel.parquet
```

It does so in two ordered stages. All C01 scripts run before all C02 scripts;
with the current filenames, C01 scripts `01` through `12` prepare normalized
artifacts, C01 script `13` merges the subset listed below, and C02 script `01`
constructs the final design-neutral panel.

{{ grounding(path="scripts/run_shared_panel.sh", anchor="shared-panel-runner", sha256="0b5a5366cd9efbf9742c0b33b6735817f552506eba9c914fca37f7858689f560") }}

Run the complete branch from the repository root:

```sh
./scripts/run_shared_panel.sh
```

Use `DRY_RUN=1` to inspect the file order without reading data or starting R.

## Stage 1: construct the merge backbone

`12_county_year_backbone.R` takes the distinct 2010-vintage counties in
`county_adjacency2010.parquet` and crosses them with every year from 2008
through 2022. This creates `county_df_year.parquet`, the balanced starting
grid used by the final C01 merge.

`13_merge_county_panel.R` derives `state_fips` from `county_fips`, attaches the
state and AEWR-region dimensions, and then joins the following normalized
artifacts:

| Artifact | Join key | Information added |
| --- | --- | --- |
| `aewr_data_year.parquet` | state, state abbreviation, year | Nominal and 2012-PPI-deflated AEWR, with one- and two-year lags |
| `cz_file_2010_small.parquet` | county | 2010 commuting zone |
| `bea_caemp25n_data_year.parquet` | county, year | Total, wage-and-salary, farm, proprietor, and nonfarm jobs |
| `bea_cainc45_data_year.parquet` | county, year | Farm receipts, government payments, production costs, hired-labor expenses, wages, and wage supplements in nominal and real terms |
| `h2a_data_year.parquet` | county, year | Aggregated H-2A applications, positions, hours, wages, and employer counts |
| `census_pop_ests_year.parquet` | county, year | Harmonized annual population |
| `census_ag_cropland_year.parquet` | county, year | Census of Agriculture cropland |
| `nass_fisher_price_index.parquet` | county, year | Chained Fisher price and quantity indexes |
| `h2a_predict.parquet` | county | One canonical static H-2A propensity record |
| `census_ag_cropland_2007_year.parquet` | county | Fixed 2007 cropland baseline |
| `state_real_minwages.parquet` | state, year | Nominal and real minimum-wage measures, plus lags created before the join |
| `oews_county_year.parquet` | county, year | Nominal OEWS Big-Six mean hourly- and annual-wage proxies, primary reporting area, and coverage support |
| `qcew_county_year.parquet` | county, year | Separate all-ownership NAICS 111, NAICS 112, and all-sector annual employment, wage bills, and disclosure flags |
| `acs_czone_wage_quantiles.parquet` | county, year, CZ | Real local wage quantiles and their lags |

The merge reads `ppi_2012.parquet` to deflate the ACS wage quantiles. The same
PPI field, retained on the normalized BEA farm-income rows, is used to deflate
the Fisher price index. Raw state/AEWR crosswalks attach stable state names and
AEWR regions.

The two BEA normalizers retain source totals rather than constructing
per-job statistics. CAEMP25N wage-and-salary employment is an all-industry job
count. CAINC45 farm wages and salaries and farm wage supplements are nominal
thousands of dollars, with parallel 2012-dollar fields. Hired farm jobs,
average annual farm wages, and average annual farm compensation remain
downstream design constructions.

`12_oews_county_year.R` first recovers each source reporting-area-year-
occupation cell once because the upstream OEWS artifact repeats source measures
for mapped counties. It uses source employment to weight usable Big-Six
occupation wages within reporting areas. When a county-year maps to multiple
reporting areas, it averages observed area wages using each area's share of
mapped county townships and renormalizes the shares over areas with observed
wages. Hourly and annual wage proxies are pooled separately under the same
mapping rules. The primary `oews_area_code` is the area with the largest
mapped-township share, with reporting-area code as the deterministic
tie-break. Missing wages remain missing, and reporting-area employment is not
published as county employment.

`qcew_county_year.parquet` retains source totals rather than average wages.
NAICS 111 and 112 each use all reported ownership components and carry
independent disclosure flags; suppressed employment and wage bills remain
null. The all-sector totals remain separate and are not used to repair either
agricultural industry.

Except for the AEWR-region dimension, which is an inner join by state
abbreviation, source panels are left-joined onto the county-year backbone.
Missing source coverage therefore remains missing rather than deleting the
backbone row. Each join declares its expected cardinality. The BEA, OEWS, and
QCEW joins are one-to-one on `county_fips, year`; repeated state-, CZ-, or
other county-level values use many-to-one joins. Duplicate source keys
therefore fail during the join. The completed intermediate artifact must also
be nonempty and unique by `county_fips, year`.

{{ grounding(path="code/c01_clean/13_merge_county_panel.R", anchor="county-year-merge", sha256="faad687f82c863b634a4a2c847db984773bf48b1f0fe052ec349ecbc3469970a") }}

The result of this stage is:

```text
data/intermediate/county_year_merged.parquet
```

## Stage 2: construct analysis variables

`01_build_county_panel.R` reads the merged intermediate and applies the
following transformations:

1. It replaces missing H-2A counts and hours with zero for the explicitly
   enumerated H-2A measures. Emergency-application counts remain unavailable
   before 2021; from 2021 onward, a county-year with zero applications receives
   zero emergency applications.
2. It constructs certified-hour shares, real AEWR-to-local-p25 wage gaps,
   employment-to-population and farm-employment shares, logged AEWR and
   population, farm expense/receipt shares, and a 2007-cropland indicator.
3. It retains only rows with both an AEWR value and an AEWR-region identifier.
   Consequently, the final panel is derived from a balanced backbone but is
   not promised to remain balanced after this required-coverage filter.
4. Within county, it creates population, farm-employment-share, and
   employment-to-population lags only when the previous observation is the
   immediately preceding calendar year.
5. It copies 2011 BEA farm employment onto every year for each county and marks
   commuting zones that span more than one AEWR region.
6. It constructs normalized H-2A position, hour, and application measures plus
   application-status rates using current farm employment, fixed-2011 farm
   employment, application counts, or certified positions as the appropriate
   positive denominator. The three employer-linkage counts remain separate
   source measures.

The nominal OEWS hourly- and annual-wage proxies and their geographic and
publication-support fields pass through this stage without policy-year
realignment, zero filling, or further aggregation. The QCEW crop, animal, and
all-sector source totals and disclosure flags likewise pass through without
an upstream average-wage construction.

The BEA wage-and-salary job count, farm wage bill, and farm wage supplements
also pass through without constructing hired-job counts, per-job wages, or
compensation ratios.

{{ grounding(path="code/c02_build/01_build_county_panel.R", anchor="shared-panel-construction", sha256="2b7bc73b507e5b8c67b4964eb693412066519ae003469531e4b6ef2972d9e2df") }}

The shared stage does not construct treatments, post indicators, event time,
fixed-effect encodings, target clusters, excluded instruments, design weights,
or design-specific samples. Those remain owned by downstream design branches.

## Validation and write

Before writing, the producer requires exactly one valid static prediction per
county represented in the prediction contract. Its cutoff and model
specification must equal the global values in `code/paths.R`; predicted counts,
2011 employment, and predicted shares must be finite and nonnegative; the BEA
and panel copies of 2011 farm employment must agree; and the share must equal
predicted count divided by 2011 farm employment within tolerance.

This validation is applied to the distinct rows with nonmissing prediction
metadata. The current producer does not separately require prediction coverage
for every county-year retained in the panel.

The final data must be nonempty and unique by `county_fips, year`. The producer
also verifies that each observed QCEW disclosure flag agrees with its source
values: disclosed employment and wages are finite and nonnegative, while
suppressed values remain null.

{{ grounding(path="code/c02_build/01_build_county_panel.R", anchor="shared-panel-contract", sha256="ffb30f884e3403e1fd5c2b88d13d20c21eace1d1e1df0957fe7ac8c24b55aada") }}

Only after those checks pass does the producer write
`data/processed/county_year_panel.parquet`.

{{ grounding(path="code/c02_build/01_build_county_panel.R", anchor="shared-panel-output", sha256="3529d63a7ff3a82e529475253c3bcf46166d3b9161f6e470269a74ef7131cdaf") }}

## Narrow rebuilds

Run only the final C02 producer when
`data/intermediate/county_year_merged.parquet` is known to be current:

```sh
Rscript --vanilla code/c02_build/01_build_county_panel.R
```

If a normalized source, join key, geographic mapping, prediction artifact, or
C01 schema changed, rerun the complete shared branch instead. A QCEW schema
change requires rebuilding `05_03_qcew_ag_wages.R`, the final C01 merge, and
C02 in that order. Changes to the final shared-panel schema require downstream
review of descriptives, DiD, panel IV, and Mundlak–Chamberlain consumers.
