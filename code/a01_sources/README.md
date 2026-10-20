# A01: Source data

These scripts acquire or normalize one external source at a time. Run them
individually from the repository root; the numbering groups related scripts
but is not a command that runs the whole folder.

## Manual execution

Most files are Marimo applications:

```sh
uv run --no-sync marimo edit code/a01_sources/01_aewr_extract_tables.py
```

The ordinary Python source programs can be run directly:

```sh
uv run --no-sync python code/a01_sources/04_02_qcew_quarterly_employment.py
```

Old root-level A paths map here by dropping the leading `a` from the filename.
The sole numbering normalization is `a04_qcew_create_binaries.py`, now
`04_01_qcew_create_binaries.py`.

## Script contracts

| Script | Responsibility | Primary output |
| --- | --- | --- |
| `00_crosswalk_harmonization.py` | Normalize the Census county adjacency source | `county_adjacency2010.parquet` |
| `01_aewr_extract_tables.py` | Assemble historical AEWR tables | `aewr.parquet` |
| `02_01_h2a_match_locations.py` | Parse FY2008--2025 disclosure and Addendum B files and match worksite locations locally. City fields listing several places are split into pieces, attachment notes are stripped and placeholders (VARIOUS, SEE ADDENDUM) are blanked. Matching order per piece is ZIP, county name, exact normalized Census place name, whole-string ratio fuzzy matching that must beat any other county by 5 points (queries under 4 characters and ambiguous names stay unmatched), then exact unique-county names in Census 2010 county subdivisions and USGS GNIS populated places. Reference lists are sorted so results are reproducible. FY2025 Addendum B adds a reported worksite county used as a match key | `h2a_disclosure_records.parquet`, `h2a_addendum_b_records.parquet`, `h2a_location_pieces.parquet` |
| `02_02_h2a_resolve_locations_with_apis.py` | Queue unmatched pieces that carry a city or county name; Gemini suggests a cleaned name and Places looks up both names; a county is accepted only when both share it. Cache-only by default | `h2a_location_api_counties.parquet`; appends to Gemini/Places caches only when networking is enabled |
| `02_03_h2a_assign_worksite_counties.py` | Union piece counties per location (local match, else API answer), attach them to records, then apply record-level fallbacks labeled in `county_fips_fallback_method`: Addendum B coordinates in the address (2010 county polygons, same state only), "within X County" address text, or the case's main-file county in the same state; main-file rows use the case's located Addendum B counties in the same state | `h2a_with_fips.parquet`, `h2a_addendum_b_with_fips.parquet` |
| `02_04_h2a_match_employers.py` | Build conservative, balanced, and high-recall FY2008--2024 worksite-business identifiers and linkage audits | `h2a_employer_crosswalk.parquet` plus membership, edge, cluster, vintage, and review artifacts |
| `02_05_h2a_geocode_worksites.py` | Prepare actual worksite address batches and parse public Census geocodes with quality/provenance flags | `h2a_worksite_geocodes.parquet` and request/record/address artifacts |
| `03_01_nass_extract_quickstats.py` | Partition raw QuickStats census and survey extracts | `qs_census_*.parquet`, `qs_survey_*.parquet` |
| `03_02_nass_select_quickstats_obs.py` | Select and harmonize downstream NASS observations | Selected census and survey Parquet files |
| `03_03_nass_census_worker_duration.py` | Derive county hired-worker duration shares and hired-labor payroll | `census_ag_hired_worker_duration_county.parquet` |
| `04_01_qcew_create_binaries.py` | Combine annual QCEW archives | `qcew.parquet` |
| `04_02_qcew_quarterly_employment.py` | Aggregate all-ownership NAICS 111/112 reference-month employment by county, year, quarter, and industry | `qcew_county_ag_quarterly_employment.parquet` |
| `04_03_qwi_quarterly_employment.py` | Download private NAICS 111/112 beginning-quarter, stable, and any-quarter QWI employment | `qwi_county_ag_quarterly_employment.parquet` |
| `05_01_oews_geographic_crosswalk.py` | Harmonize OEWS reporting areas to counties | `oews_area_definitions.parquet` |
| `05_02_oews_binaries.py` | Combine historical OEWS releases | `oews.parquet` |
| `06_01_croplandcros_cdl_aggregate_using_exactextract.py` | Aggregate CDL rasters to county crop counts | County crop-pixel counts |
| `06_02_croplandcros_cdl_extract_crop_name.py` | Convert county crop counts to labeled acres | `croplandcros_county_crop_acres.parquet` |
| `06_03_croplandcros_cdl_nass_quickstats_crop_crosswalk.py` | Generate candidate NASS-to-CDL mappings | `nass_to_cdl_mappings.jsonl` |
| `06_04_croplandcros_cdl_calculate_synthetic_price_and_yield.py` | Review mappings and construct synthetic crop series | NASS-CDL crosswalk and synthetic price/yield files |
| `06_05_croplandcros_cdl_crop_type.py` | Classify CDL crops for analysis | `cdl_crop_type.parquet` |
| `07_state_minimum_wages.py` | Assemble state-year minimum wages | `state_year_min_wage.parquet` |
| `08_bea_farm_nonfarm_emp.py` | Extract archived county employment and farm-income tables and map BEA combined areas to canonical counties | Trimmed BEA files and farm employment |
| `09_01_h2a_prediction_pull_noaa.py` | Build annual county climate predictors | Climate basis Parquet |
| `09_02_h2a_prediction_pull_gnatsgo.py` | Build county soil predictors | gNATSGO soil-cell Parquet |
| `10_mymarketnews_get_reports.py` | Download USDA market reports | Manifest, headers, and report partitions |
| `11_risk_management_agency_summary_of_business_binaries.py` | Convert RMA Summary of Business archives | Coverage and type-practice-unit Parquet files |
| `12_risk_management_agency_actuarial_data_master_binaries.py` | Inspect ADM price records and layouts | No persistent artifact yet |
| `13_farm_labor_survey.py` | Parse annual FLS wages and pair quarterly regional worker/wage tables from the same release | `fls_region.parquet`, `fls_region_quarterly_workers.parquet`, `fls_region_quarterly_wages.parquet`, and auxiliary files |

## Ordering and credentials

- H-2A location matching runs `02_01`, `02_02`, then `02_03`, once each.
  `02_02` makes no network calls unless `H2A_RUN_LOCATION_APIS=1`; without it,
  every answer comes from the caches in `data/intermediate/cache` and the
  script prints how many Gemini requests (20 locations each, one batch job per
  source), Places text searches (at most 600 per minute) and Places details
  requests (at most 9 per second) a live run needs. There is no automatic
  retry; caches are appended or rewritten in place every 50 requests, and a
  pending Gemini batch job is resumed from its job-ID file on the next run
  (`H2A_GEMINI_MAX_WAIT_SECONDS`, default 3600, bounds the wait). Run `02_04`
  after `02_03`; it reads both with-FIPS files without rewriting them,
  classifies source roles without outcomes, and retains raw identity and role
  fields as membership keys. It always writes the seeded 200-pair review
  sample, then stops without promoting v2 until
  `data/raw/h2a/employer_linkage_review_labels.csv` contains one valid label
  for every pair. See the grounded
  [worksite-business linkage contract](../../content/contracts/employer-linkage.md).
- Within the NASS family, run `03_01` before `03_02`, and `03_02` before
  `03_03`.
- Quarterly QCEW uses ownership components 1, 2, 3, and 5 for both NAICS 111
  and 112. A county-industry-quarter is disclosed only when every reported
  ownership component is disclosed; suppressed totals remain null. The key is
  `county_fips, year, qtr, industry_code`, with disclosure component counts and
  `qcew_reference_month_emplvl` retained as source diagnostics.
- FLS quarterly worker and wage rows are selected as a release pair. The
  selected annual-report release is preferred; otherwise the latest paired
  release no later than it is used. The supported 2010--2021 keys are complete.
  USDA did not conduct the April 2011 survey, so those 17 region rows carry an
  explicit `survey_not_conducted` value status rather than fabricated values.
  The wage artifact retains `fls_field_hourly_wage`,
  `fls_livestock_hourly_wage`, `fls_field_livestock_hourly_wage`, and
  `fls_all_hired_hourly_wage` plus paired release and source-table metadata.
- Within the CDL family, run `06_01` through `06_05` in numeric order.
- Rebuild each county-bearing producer after changing the geographic contract;
  downstream scripts do not rename or repair stale artifact schemas.
- `GOOGLE_GEMINI_API_KEY` is used by `02_02`, `03_02`, `06_03`, and `06_04`;
  `GOOGLE_PLACES_API_KEY` by `02_02` (both only with `H2A_RUN_LOCATION_APIS=1`); `FRED_API_KEY` by `07`; and
  `USDA_MYMARKETNEWS_API_KEY` by `10`. `04_03` uses `CENSUS_API_KEY` and
  caches each state-industry API response for resumable downloads.

## Audited wage and historical labor inputs

`01_aewr_extract_tables.py` uses only cached files. Its tracked
`aewr_rate_schedule_manifest.csv` supplies reviewed source/effective dates;
`aewr_policy_events.csv` supplies policy events and confounders. It owns
`aewr.parquet` (1990--2025 annual references), `aewr_rate_schedule.parquet`,
`aewr_occupation_skill_rates.parquet` (cached 2025--2027 lookup vintages),
`aewr_policy_events.parquet`, `aewr_reference_rate_source_audit.parquet`,
`aewr_2009_nonstatutory_wages.parquet`, and `aewr_source_manifest.json`.
The canonical 2009 rate is null. See the
[AEWR input contract](../../content/contracts/aewr-inputs.md).

The Census hired-labor normalizer retains available 1997 and 2002 history
alongside 2007--2022, with missing/suppressed totals left missing.
It is a Marimo application. To export and execute either changed source app,
use the existing helper, for example:

```sh
bash -c 'source scripts/pipeline_helpers.sh; run_step code/a01_sources/03_03_nass_census_worker_duration.py'
```

H-2A location matching reads two manually placed public gazetteers, retrieved
2026-10-19 and never refreshed silently:

| File | Provider and URL | SHA-256 |
| --- | --- | --- |
| `data/raw/geographic_crosswalks/census/national_cousub.txt` | Census Bureau 2010 ANSI county subdivision codes, <https://www2.census.gov/geo/docs/reference/codes/files/national_cousub.txt> | `cb22b6b41fdf2fd5c10d0cbe55d92cb730e822162b1bd3ad58f18fb4359cec79` |
| `data/raw/geographic_crosswalks/usgs_gnis/DomesticNames_National_Text.zip` | USGS GNIS Domestic Names, national text file (public domain; provider revision of 2026-09-29), <https://prd-tnm.s3.amazonaws.com/StagedProducts/GeographicNames/DomesticNames/DomesticNames_National_Text.zip> | `180651b9fba01e65fc65182fb61de5be73bf6f42082a1b9b82cca2f74d3c16e9` |

Only GNIS `Populated Place` features without `(historical)` are used. GNIS
county codes are current: Connecticut planning-region rows are dropped (its
towns are county subdivisions), Alaska 02158/02063/02066 map to 2010 codes
02270/02261/02261, and codes outside the 2010 county list are dropped. A name
found in more than one county within a state is ignored. No credentials are
needed; replacing either file is a declared source revision.

`02_05_h2a_geocode_worksites.py` owns point-geocoding records, addresses,
request batches, geocodes and manifest under the `h2a_worksite_geocode*` names.
It reads actual Addendum B worksite streets and the separate main disclosure
`WORKSITE_ADDRESS` fields for FY2020--2024. Main records retain an unresolved
case/worksite linkage flag; employer mailing addresses are never substituted.
Default execution prepares requests and reads caches without networking.
With `H2A_RUN_PUBLIC_GEOCODER=1`, the public Census geography batch endpoint
receives at most 10,000 addresses per request and 50 batches per run. Timeout
is 10 seconds to connect/120 seconds to read, with no automatic retry; the
first failure stops further requests. Successful complete responses are cached
immutably by request hash in `data/raw/h2a/census_geocoder`. Subsequent runs
resume missing batches. The current benchmark/vintage, request/response hashes
and retrieval time are retained; the moving provider vintage is not represented
as 2010 geometry. No API key is needed. Address-range matches still require
coordinate-quality, boundary-vintage and outcome-linkage review before spatial RD.
