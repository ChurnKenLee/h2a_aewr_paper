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

{{ <grounding path="scripts/run_shared_panel.sh" anchor="shared-panel-runner" sha256="0754bdd5871b5761e3383d8fcbca213582ed5cf0f4c7630881ac77b411bd2b41" /> }}

Run the complete branch from the repository root:

```sh
./scripts/run_shared_panel.sh
```

Use `DRY_RUN=1` to inspect the file order without reading data or starting R.

## Stage 1: construct the merge backbone

`12_county_year_backbone.R` takes the distinct 2010-vintage counties in
`county_adjacency2010.parquet` and crosses them with every year from
`SHARED_PANEL_START_YEAR` through `SHARED_PANEL_END_YEAR` in `code/paths.R`,
currently 2008--2025. This creates `county_df_year.parquet`, the balanced
starting grid used by the final C01 merge. The H-2A, AEWR, population, OEWS
and price-index normalizers use the same calendar.

The 2025 extension is the user-approved common baseline. It uses complete
FY2008--FY2025 disclosures plus the cached partial FY2026 Q3 release, which
contains later decisions on contracts starting in 2025. Later releases may
still revise those starts. No design sample is imposed at this layer.
`02_validate_source_coverage.py` runs after C02 and writes the shared-panel
validation JSON, numeric coverage CSV, and recent H-2A source-cohort CSV under
`outputs/`. It checks the full calendar and reconciles start-year certifications
to B01. Coverage reports retain missing later-year controls and other outcomes.

The PPI normalizer requires twelve distinct months for every shared year and
the 2012 base year. The 2025 observations were refreshed from the public FRED
WPU01 table on 2026-10-26; the raw snapshot and refresh manifest retain the
source URL, revision details and hashes. Earlier monthly observations were
preserved. An incomplete future year is excluded, never averaged as an annual
deflator.

BEA discontinued the county CAEMP25N employment and CAINC45 farm-income tables
after their 2022 vintage, and removed the employment lines from CAINC4. Their
fields, the ratios built from them, and outcomes that use current farm
employment are therefore missing in 2023--2025; nothing is extrapolated. The
continuing CAINC5N table supplies county farm earnings and earnings by place
of work through 2024. Population uses the Census Vintage 2025 county file for
2020 onward; Connecticut's 2010-vintage counties are projected from 2019 with
statewide growth from that vintage. The fixed 2011 farm-employment
denominators are unaffected.

`13_merge_county_panel.R` derives `state_fips` from `county_fips`, attaches the
state and AEWR-region dimensions, and then joins the following normalized
artifacts:

| Artifact | Join key | Information added |
| --- | --- | --- |
| `aewr_data_year.parquet` | state, state abbreviation, year | Nominal and 2012-PPI-deflated AEWR, with one- and two-year lags |
| `cz_file_2010_small.parquet` | county | 2010 commuting zone |
| `bea_caemp25n_data_year.parquet` | county, year | Total, wage-and-salary, farm, proprietor, and nonfarm jobs |
| `bea_cainc45_data_year.parquet` | county, year | Farm receipts, government payments, production costs, hired-labor expenses, wages, and wage supplements in nominal and real terms |
| `bea_cainc5n_data_year.parquet` | county, year | Nominal farm earnings and total earnings by place of work |
| `h2a_data_year.parquet` | county, year | Aggregated H-2A applications, positions, hours, wages, positive-certified linked worksite-business counts, and v2 linkage provenance |
| `census_pop_ests_year.parquet` | county, year | Harmonized annual population |
| `census_ag_cropland_year.parquet` | county, year | Census of Agriculture cropland |
| `nass_fisher_price_index.parquet` | county, year | Chained Fisher price and quantity indexes |
| `h2a_predict.parquet` | county | One canonical static H-2A propensity record |
| `census_ag_cropland_2007_year.parquet` | county | Fixed 2007 cropland baseline |
| `state_real_minwages.parquet` | state, year | Nominal and real minimum-wage measures, the applicable agricultural floor from the dated coverage table `code/a01_sources/state_ag_minwage_coverage.csv` (spell in effect on 1 July; `ag_floor_rule`, `ag_specific_min_wage`, `ag_coverage_backcast` before 2008), plus lags created before the join |
| `oews_county_year.parquet` | county, year | Nominal OEWS Big-Six mean hourly- and annual-wage proxies, primary reporting area, and coverage support |
| `qcew_county_year.parquet` | county, year | Separate all-ownership NAICS 111, NAICS 112, and all-sector annual employment, wage bills, and disclosure flags |
| `acs_czone_wage_quantiles.parquet` | county, year, CZ | Real local wage quantiles and their lags |

The merge reads `ppi_2012.parquet` to deflate the ACS wage quantiles and joins
the same annual PPI by year to deflate the Fisher price index. The deflator no
longer comes from the normalized BEA farm-income rows, so it is defined in
2023--2025 and for the few counties without BEA farm-income rows. Raw state/AEWR crosswalks attach stable state names and
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

{{ <grounding path="code/c01_clean/13_merge_county_panel.R" anchor="county-year-merge" sha256="42d6578ce958f298f14bbfca2d9b26e2f43c1735f2e922f1b194dd025e833885" /> }}

The result of this stage is:

```text
data/intermediate/county_year_merged.parquet
```

## Stage 2: construct analysis variables

`01_build_county_panel.R` reads the merged intermediate and applies the
following transformations:

1. It replaces missing H-2A counts and hours with zero for the explicitly
   enumerated measures, except requested positions and requested hours. B01
   retains those totals as missing when any contributing allocation is unknown
   and supplies zeros for calendars with no allocated records. C02 preserves
   those values and fills zero only when the county-year has no matching H-2A
   aggregate, identified before filling its provenance fields. This prevents
   absent requested-count fields from becoming measured zero demand.
   Emergency-application counts remain unavailable
   before 2021; from 2021 onward, a county-year with zero applications receives
   zero emergency applications.
2. It constructs certified-hour shares, real AEWR-to-local-p25 wage gaps,
   employment-to-population and farm-employment shares, their continuing-source
   analogues (QCEW all-sector employment per resident and the CAINC5N
   farm-earnings share of earnings by place of work), logged AEWR and
   population, farm expense/receipt shares, and a 2007-cropland indicator.
3. It requires an AEWR-region identifier and retains outcome rows when the
   statutory AEWR is unavailable. The corrected 2009 AEWR is missing, with
   source status and a separately named nonstatutory application wage.
   Designs own their wage-complete estimation samples.
4. Within county, it creates population, farm-employment-share,
   employment-to-population, QCEW employment-to-population, and farm-earnings
   share lags only when the previous observation is the immediately preceding
   calendar year.
5. It copies 2011 BEA farm employment onto every year for each county and marks
   commuting zones that span more than one AEWR region.
6. It constructs normalized H-2A position, hour, and application measures plus
   application-status rates using current farm employment, fixed-2011 farm
   employment, application counts, or certified positions as the appropriate
   positive denominator. The three v2 worksite-business linkage counts remain
   separate source measures. It validates one FY2008--2026 Q3 linkage contract
   (review status `complete` or `pending`) and fills its unit, version, vintage, review status, and
   review-report hash across zero-use county-years.

The nominal OEWS hourly- and annual-wage proxies and their geographic and
publication-support fields pass through this stage without policy-year
realignment, zero filling, or further aggregation. The QCEW crop, animal, and
all-sector source totals and disclosure flags likewise pass through without
an upstream average-wage construction.

The BEA wage-and-salary job count, farm wage bill, and farm wage supplements
also pass through without constructing hired-job counts, per-job wages, or
compensation ratios.

{{ <grounding path="code/c02_build/01_build_county_panel.R" anchor="shared-panel-construction" sha256="99a5725bc861f4b3eb1e7c2be8cb1e1ac339cc0d22c8e259a62e99d261428eed" /> }}

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

{{ <grounding path="code/c02_build/01_build_county_panel.R" anchor="shared-panel-contract" sha256="7998a0df92b2bf71599e1db0ef7bdb9c723b17ee29efc544e3e0bbb897664c29" /> }}

Only after those checks pass does the producer write
`data/processed/county_year_panel.parquet`.

{{ <grounding path="code/c02_build/01_build_county_panel.R" anchor="shared-panel-output" sha256="3529d63a7ff3a82e529475253c3bcf46166d3b9161f6e470269a74ef7131cdaf" /> }}

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

The CAINC5N and OEWS normalizers validate the years published in their sources within the
shared calendar, rather than requiring every shared year to be available.
Both current sources end in 2024, leaving their 2025 values missing.
