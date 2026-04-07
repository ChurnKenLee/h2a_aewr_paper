# County Geographic Vintage Audit

**Date:** 2026-04-03  
**Purpose:** Identify the county boundary vintage for every dataset in the pipeline, whether a crosswalk was applied to reconcile vintages, and any hard-coded FIPS patches in the R scripts.  
**Target vintage:** 2010 Census county definitions.

---

## Summary Table

| Dataset / Component | Source Script(s) | County Vintage | Crosswalk Applied? | Crosswalk File | Hard-coded FIPS Patches | Notes |
|---|---|---|---|---|---|---|
| **County shapefile (analysis maps)** | `Do/H2A Analysis Figs and Tables.R` | 2020 TIGER | No | — | None | `tl_2020_us_county.shp`; used for map exhibits only, not for data merges |
| **County shapefile (CDL aggregation)** | `code/a09_01_aggregate_croplandcros_cdl_county_using_exactextract.py` | **2023 TIGER** | No | — | None | `tl_2023_us_county.shp`; reprojected to EPSG:5070; **vintage mismatch vs. 2010 target** |
| **Border county pairs** | `Do/H2A Build Dataset.R` | **2010 Census** | Yes | `Data Int/county_adjacency2010.csv` | None | Border pairs defined from 2010 adjacency file; aligns with target vintage |
| **Commuting zones (CZ)** | `Do/H2A Clean and Load.R` (lines 48–62) | **2010 Census** | Yes | `Data Int/counties10-zqvz0r.csv` (PSU CZ crosswalk) | None | 2010 CZ definitions; `cz_file_2010.parquet` output; aligns with target vintage |
| **H-2A location matching** | `code/a02_01_match_h2a_locations.py` | **2020 Census** | Yes | `national_county2020.txt`, `national_place_by_county2020.txt`, `national_cousub2020.txt`, `tab20_zcta520_county20_natl.txt` | Multi-county fuzzy matching with equal worker split | FIPS assigned via string match to 2020 Census geography files; **vintage mismatch vs. 2010 target** |
| **H-2A aggregation** | `code/b01_01_h2a_aggregation_nodupes.R` | Inherited from a02 (2020 Census) | No additional crosswalk | — | None | FIPS inherited from location matching; multi-county records split equally across counties |
| **ACS wage quantiles (CZ-level)** | `code/b03_02_acs_cz_wage_quantiles.R` | Mixed — CZ locked to **1990 CZ definitions**; county codes via Eckert 2010 xwalk | Yes | `cz_crosswalk.csv` (Fabian Eckert), `cw_puma2010_czone.dta` (Autor), `PUMA2010_PUMA2020_crosswalk.xls` | None (but see open Bug 1 in `cz_wage_quantiles_investigation.md`) | ACS 2022+ PUMAs forward-mapped to 2010 PUMA → 2010 CZ for consistency; county ANSI from Eckert xwalk (1990 CZ → 2010 county) |
| **ACS immigrant imputation** | `code/b03_01_acs_immigrant_imputation.R` | Implicit Census vintage from IPUMS ACS samples | No | — | None | STATEFIP + county codes taken as-is from IPUMS extract `usa_00034.xml`; vintage not explicitly declared |
| **Census of Agriculture (NASS)** | `Do/H2A Clean and Load.R` (lines 183–255) | Implicit — assumed **2010 Census** (standard NASS reporting) | No | — | None | FIPS constructed as `state_fips_code \|\| LPAD(county_code, 3)`; no explicit vintage; NASS uses current Census FIPS for census years 2007–2022 |
| **CDL crop shares** | `Do/H2A Clean and Load.R` | **2023 TIGER** (inherited from a09 aggregation) | No | — | None | Acreages aggregated by a09 using 2023 shapefile; **vintage mismatch vs. 2010 target** |
| **BEA employment (CAEMP25N)** | `code/b05_bea_farm_nonfarm_employment.R` | BEA-proprietary GeoFIPS (generally tracks Census but lags) | Yes | `Data Int/bea_fips_xwalk.csv` | **Virginia independent city consolidations** (55 remappings; e.g., Bedford City 51515 → Bedford County group 51019/51909) | BEA forces city–county merges for employment reporting; crosswalk reconciles BEA GeoFIPS back toward Census FIPS |
| **BEA farm income (CAINC45)** | `code/b05_bea_farm_nonfarm_employment.R` | Same as CAEMP25N | Yes | `Data Int/bea_fips_xwalk.csv` | Same Virginia consolidations | Same BEA GeoFIPS logic as employment data |
| **QCEW (agricultural employment)** | `code/a05_create_qcew_binaries.py` | BLS-reported Census FIPS as of submission year (no explicit vintage) | No | — | None | `area_fips` field used directly; BLS tracks current Census FIPS; coverage 2005–2017 |
| **OEWS (occupational wages)** | `code/a04_create_oews_binaries.py`, `code/a06_create_oews_location_crosswalk.py` | OEWS MSA/township definitions (1997–present) | Yes | Internal OEWS area crosswalk (built by a06) | None | OEWS areas often span multiple counties; county-level aggregation is approximate |
| **Census population estimates** | `Do/H2A Clean and Load.R` | Census Bureau vintage — **2010-based** for 2010–2019, **2020-based** for 2020+ | No additional crosswalk | — | **Connecticut filter** (see below); **South Dakota** county code change (Shannon County 46113 → Oglala Lakota County 46102, renamed 2015) | Connecticut 2020+ rows filtered out entirely (see separate row); SD fix may be implicit in Census file |
| **Connecticut (2020+)** | `Do/H2A Clean and Load.R` | 2022 planning regions replace historic counties | Crosswalk file exists but not used in pipeline | `Data Int/ct_region_xwalk.csv` | **Hard-coded filter:** all CT FIPS (9000–9999) dropped for year ≥ 2020 | Connecticut abolished counties as administrative units in 2022; pipeline excludes CT 2020+ rather than remapping to new planning regions |
| **NOAA climate / GDD** | `code/a14_01_pull_noaa.R` (lines 293–343) | NOAA NCEI vintage; county FIPS assumed current Census | Partial — state codes only | Hard-coded NCEI→Census state FIPS lookup (48 states) | None at county level | NOAA provides county-level data; county FIPS taken as-is from NOAA source; no county crosswalk applied |
| **RMA Summary of Business** | `code/a11_create_rma_sob_binaries.py` | RMA-reported county codes (typically tracks BEA/Census; vintage unclear) | No | — | None documented | RMA county codes used directly; reconciliation with Census FIPS not documented in pipeline scripts |

---

## Hard-Coded FIPS Patches in `Do/` R Scripts

| Script | Location | Issue Handled | Code Pattern |
|---|---|---|---|
| `Do/H2A Clean and Load.R` | Census population estimates block | **Connecticut 2022 boundary abolishment** — CT counties replaced by planning regions | `filter(!(countyfips >= 9000 & countyfips <= 9999 & year >= 2020))` |
| `Do/H2A Clean and Load.R` | Census population estimates block | **South Dakota** — Shannon County (46113) renamed Oglala Lakota County (46102) effective 2015 | Likely handled by Census file; check if explicit `case_when` exists |
| `code/b05_bea_farm_nonfarm_employment.R` | BEA FIPS crosswalk join (lines 9–56) | **Virginia independent cities** — 55 independent cities consolidated with county equivalents for BEA | Join on `bea_fips_xwalk.csv`; replaces BEA GeoFIPS with Census county FIPS |

---

## Key Vintage Mismatches vs. 2010 Target

| Mismatch | Affected Datasets | Risk | Recommended Fix |
|---|---|---|---|
| **CDL uses 2023 TIGER shapefile** | `cdl_cropshares.parquet`, any CDL-derived variables | Counties created or redefined after 2010 will not match 2010 FIPS in main panel | Re-run `a09_01` with `tl_2010_us_county.shp` (or 2020 with a crosswalk to 2010) |
| **H-2A locations geocoded to 2020 Census** | `h2a_aggregated.parquet` | Any county that changed boundaries 2010→2020 will have a FIPS mismatch when merged into 2010-vintage panel | Apply a 2020→2010 county crosswalk (Census relationship file) to `h2a_aggregated.parquet` before merging |
| **ACS immigrant data vintage undeclared** | `acs_immigrant_imputed.parquet` | Unknown; likely minor if IPUMS follows Census vintage conventions | Confirm IPUMS vintage in `usa_00034.xml` metadata |
| **Connecticut 2020+ excluded entirely** | Population estimates, any county-year panel for 2020–2022 | CT observations missing for last 3 analysis years | Acceptable as-is if documenting exclusion; alternative is to apply `ct_region_xwalk.csv` to map 2022 planning regions back to 2010 counties |
| **BEA GeoFIPS ≠ Census FIPS (Virginia)** | `bea_farm_nonfarm_emp.parquet` | Virginia analysis confounded by city–county consolidations unless crosswalk is applied consistently | Crosswalk is applied via `bea_fips_xwalk.csv`; verify all 55 Virginia entries are covered and that downstream merges use the remapped FIPS |

---

## Crosswalk Files Inventory

| File | Location | Maps From | Maps To | Used By |
|---|---|---|---|---|
| `county_adjacency2010.csv` | `Data Int/` | 2010 county FIPS | Adjacent county FIPS | `Do/H2A Build Dataset.R` |
| `counties10-zqvz0r.csv` | `Data Int/` | 2010 county FIPS | Commuting zone ID (2010) | `Do/H2A Clean and Load.R` |
| `cz_crosswalk.csv` | `Data Int/` (or `code/`) | 1990 CZ ID (Eckert) | 2010 county ANSI code | `code/b03_02_acs_cz_wage_quantiles.R` |
| `cw_puma2010_czone.dta` | `code/` | 2010 PUMA | 1990 CZ ID (Autor) | `code/b03_02_acs_cz_wage_quantiles.R` |
| `cw_puma2000_czone.dta` | `code/` | 2000 PUMA | 1990 CZ ID (Autor) | `code/b03_02_acs_cz_wage_quantiles.R` (pre-2012 ACS) |
| `PUMA2010_PUMA2020_crosswalk.xls` | `code/` | 2020 PUMA | 2010 PUMA | `code/b03_02_acs_cz_wage_quantiles.R` (2022+ ACS) |
| `bea_fips_xwalk.csv` | `Data Int/` | BEA GeoFIPS (incl. Virginia consolidated cities) | Census county FIPS | `code/b05_bea_farm_nonfarm_employment.R` |
| `ct_region_xwalk.csv` | `Data Int/` | CT 2022 planning region FIPS | CT historic county FIPS | **Not currently used** in pipeline (CT filtered out for 2020+) |
| `national_county2020.txt` | `code/` (Census file) | Place/cousub names | 2020 Census county FIPS | `code/a02_01_match_h2a_locations.py` |
| `fips_codes.csv` | `Data Int/` | State abbreviations | State/county FIPS | Multiple scripts (general lookup) |
