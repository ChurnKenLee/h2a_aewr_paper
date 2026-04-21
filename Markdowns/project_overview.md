# H-2A Paper: Project Overview and Script Documentation

**Author:** Phil Hoxie (primary R/analysis), Ken (coauthor, Python data pipeline)
**Last updated:** 2026-03-30
**Working directory root:** `C:/Users/pghox/Dropbox/H-2A Paper/`

---

## Research Question

This paper estimates the impact of the **Adverse Effect Wage Rate (AEWR)** — a federally mandated minimum wage for H-2A agricultural guest workers — on H-2A program utilization by US agricultural employers. The 2011 AEWR policy change is the primary source of identification. A secondary goal is to understand downstream effects on food prices.

---

## Key Concepts

- **H-2A program:** US guest worker visa program for temporary agricultural workers
- **AEWR:** Adverse Effect Wage Rate — the minimum hourly wage employers must pay H-2A workers, set annually by region (roughly by state, but with USDA Farm Labor Survey regions)
- **Policy change:** 2011 methodology change to AEWR setting, creating variation across regions
- **Identification strategy:** Triple Difference (DDD)
  - _Dimension 1:_ High vs. low H-2A pre-adoption counties (exposure to program)
  - _Dimension 2:_ High vs. low AEWR growth regions
  - _Dimension 3:_ Pre-2011 vs. post-2011 period

---

## Folder Structure

```
H-2A Paper/
├── Do/                         # Main R analysis scripts (Phil's scripts)
│   ├── H2A Master.R            # Orchestrates full pipeline
│   ├── H2A Pull Min Wages.R    # Fetches min wages from FRED API
│   ├── H2A Clean and Load.R    # Cleans and saves intermediate data
│   ├── H2A Build Dataset.R     # Constructs analysis-ready datasets
│   ├── H2A Analysis Figs and Tables.R  # All figures, tables, regressions
│   ├── Read Parquet.R          # Validates parquet files from Ken
│   ├── TSCB.R                  # Monte Carlo simulation for inference
│   └── Pre 2026/               # Archived older versions of all scripts
│
├── Data Int/                   # Intermediate data (parquet + shapefiles)
│   ├── tl_2020_us_county.shp   # County shapefile (US Census TIGER)
│   ├── fips_codes.csv          # FIPS crosswalk
│   ├── aewr.parquet            # Raw AEWR data
│   ├── aewr_data.parquet       # Cleaned AEWR (merged with PPI)
│   ├── aewr_data_full.parquet  # AEWR with all years
│   ├── aewr_regions.parquet    # AEWR region assignments by state
│   ├── border_counties_allmatches.parquet  # All cross-state border county pairs
│   ├── state_border_pairs.parquet
│   ├── h2a_aggregated.parquet  # H-2A records (from Ken)
│   ├── h2a_prediction.parquet  # ML-predicted H-2A usage (elastic net)
│   ├── cz_wage_quantiles.parquet    # Commuting zone wage quantiles (ACS)
│   ├── acs_immigrant_imputed.parquet
│   ├── acs_qcew_data.parquet
│   ├── commuting_zones.parquet
│   ├── CAEMP25N/               # BEA county employment data (parquet partitions)
│   ├── CAINC45/                # BEA county farm income data (parquet partitions)
│   ├── WPU01.csv               # Producer Price Index for farm products (FRED)
│   ├── fred_state_minwages.parquet      # Output of H2A Pull Min Wages.R
│   ├── state_real_minwages.parquet      # Output of H2A Clean and Load.R
│   ├── census_ag_cropland.parquet
│   ├── census_ag_cropland_2007.parquet
│   ├── h2a_data.parquet                 # Census-period H-2A data
│   ├── h2a_data_ts.parquet              # Annual H-2A time series
│   ├── cdl_cropshares.parquet           # CDL crop shares by county-year
│   ├── census_pop_ests.parquet
│   ├── border_df_analysis.parquet       # Output of H2A Build Dataset.R
│   ├── border_df_analysis_year.parquet
│   └── county_df_analysis_year.parquet  # Main analysis dataset
│
├── Output/                     # Figures (PNG) and tables (TEX)
│   ├── ts_national_aewr_real.png
│   ├── ts_national_aewr_nominal.png
│   ├── fig_line_ts_h2a_workers_certified.png
│   ├── fig_line_ts_h2a_indexes.png
│   ├── [map figures for H-2A use, AEWR change, sample classification]
│   ├── table_1_main_results.tex
│   ├── table_2_event_study.tex
│   └── coefplot_*.png
│
├── code/                       # Python data pipeline (Ken's scripts)
│   ├── a01–a13 series          # Data extraction from raw sources
│   ├── a02_0x series           # H-2A location geocoding (Google Places + Gemini)
│   ├── a09_0x series           # CDL/cropland raster aggregation
│   ├── b03–b16 series          # Data transformation and aggregation (R + Python)
│   ├── bootstrap/              # Bootstrap simulation (TSCB)
│   ├── deprecated/             # Old code no longer in use
│   ├── json/                   # Geocoding request/response files
│   └── nathaniel_ai_code/      # AI-generated code versions for review
│
├── Markdowns/                  # Documentation (this folder)
├── literature/                 # Academic references
├── papers/                     # Paper drafts and submissions
├── binaries/                   # Large binary files (not tracked by git)
└── .claude/                    # Claude Code workspace memory
```

---

## Data Flow

```
Raw Sources (H-2A DOL records, FRED, USDA NASS, BEA, Census, CDL rasters)
        │
        ▼
Python pipeline (code/a01–a13): Extract, geocode, aggregate
        │
        ▼
Parquet files in Data Int/ (shared with Phil via Dropbox)
        │
        ▼
Do/H2A Pull Min Wages.R → fred_state_minwages.parquet
        │
        ▼
Do/H2A Clean and Load.R → cleaned parquet files (h2a_data, cdl_cropshares, etc.)
        │
        ▼
Do/H2A Build Dataset.R → analysis datasets (county_df_analysis_year, border_df_analysis)
        │
        ▼
Do/H2A Analysis Figs and Tables.R → Output/ (PNGs, TEX tables)
```

---

## Key External Data Sources

| Source                 | Description                                             | Access                          |
| ---------------------- | ------------------------------------------------------- | ------------------------------- |
| DOL H-2A records       | Visa applications, worker counts, job orders            | Ken's Python pipeline           |
| FRED API               | State/federal minimum wages (STTMINWG\*), PPI (WPU01)   | Direct API in Pull Min Wages.R  |
| BEA CAEMP25N           | County-level farm/nonfarm employment                    | Parquet partitions in Data Int/ |
| BEA CAINC45            | County-level farm income and expenses                   | Parquet partitions in Data Int/ |
| USDA NASS Census of Ag | Cropland acres, farm labor expenses                     | Ken's pipeline                  |
| USDA CDL               | Cropland Data Layer (raster, crop type by county)       | Ken's exactextract pipeline     |
| Census TIGER           | County shapefiles (tl_2020_us_county.shp)               | Data Int/                       |
| ACS                    | Immigrant populations, wage quantiles by commuting zone | Ken's pipeline                  |
| QCEW                   | Agricultural employment by county                       | Ken's pipeline                  |

---

## Script Summaries: `Do/` Folder

### `H2A Master.R`

**Role:** Top-level orchestration script. Run this to execute the entire pipeline.

**What it does:**

1. Clears the workspace and loads core packages (`tidyverse`, `arrow`, `tidylog`)
2. Sets folder paths using `here()` for portability across machines
3. Calls the four sub-scripts in sequence via `source(..., echo = TRUE)`

**Path variables set here (inherited by all sub-scripts):**

- `folder_dir` — project root
- `folder_do` — `Do/`
- `folder_data` — `Data Int/`
- `folder_output` — `Output/`

**Note:** `Read Parquet.R` is commented out (only needed when receiving new data from Ken).

---

### `H2A Pull Min Wages.R`

**Role:** Fetches annual state and federal minimum wage data from the FRED API and saves a merged file.

**Inputs:**

- FRED API key (hardcoded in script)
- `Data Int/fips_codes.csv` — for state FIPS-to-abbreviation crosswalk
- FRED series IDs: federal `STTMINWGFG`, state `STTMINWG{abbrev}`

**Process:**

1. Loops over all 50 states + DC, pulling annual minimum wages from 1968–2025
2. Uses `tryCatch` to handle states with no FRED series (fills with `NA`)
3. Merges state and federal rates
4. Computes `prevailing_min_wage = max(state, federal)` for each state-year

**Output:**

- `Data Int/fred_state_minwages.parquet`
  - Variables: `fips`, `year`, `state_min_wage`, `federal_min_wage`, `prevailing_min_wage`

---

### `H2A Clean and Load.R`

**Role:** The main data loading and cleaning script. Reads raw parquet files from both Ken's pipeline and external sources, processes them, and writes cleaned intermediate parquets.

**Inputs (parquet files from Ken or external):**

- ACS: `cz_wage_quantiles`, `acs_immigrant_imputed`, `acs_qcew_data`
- Geographic: `fips_codes`, `aewr_regions`, `commuting_zones`, `border_counties_allmatches`, `state_border_pairs`
- H-2A: `h2a_aggregated`, `h2a_prediction`
- BEA: `CAEMP25N/` (employment), `CAINC45/` (farm income)
- NASS: Census of Agriculture cropland data
- Population: Census Bureau annual population estimates
- Prices: `WPU01.csv` (PPI for farm products)
- Min wages: `fred_state_minwages.parquet` (from Pull Min Wages.R)

**Key processing steps:**

1. **Producer Price Index (PPI):** Loads WPU01 from FRED CSV, computes annual averages, rebases to 2012=100. Used to convert nominal wages/prices to real.

2. **Minimum wages:** Merges state min wages with PPI; creates agriculture-exemption flag; computes real prevailing ag min wage.

3. **AEWR:** Merges AEWR data with PPI to create `aewr_ppi` (real AEWR). Also creates lagged AEWR (`aewr_state_ag_ppi_l1`).

4. **Census of Agriculture cropland:** Filters for cropland acres; constructs FIPS; restricts to census years (2007, 2012, 2017, 2022); creates 2007 baseline reference (`census_ag_cropland_2007`).

5. **H-2A records:**
   - Assigns each year to a census period: 2008–2011 → 2012; 2012–2016 → 2017; 2017–2021 → 2022
   - Collapses to county × census_period (for cross-section) and county × year (for time series)
   - Creates `nbr_workers_certified_start_year` as the census-period average

6. **Cropland Data Layer (CDL):** Loads crop-specific acreage; classifies crops (fruit, vegetable, field, etc.); pivots to county-year wide format.

7. **ACS & QCEW employment:** Combines two sources of ag employment; handles inconsistencies with cleaning flags.

8. **BEA employment/income:** Loads partitioned parquets for CAEMP25N and CAINC45; creates farm employment, income, and expense variables.

9. **Population estimates:** Loads Census 2008–2022 estimates; handles Connecticut regional boundary changes (2020) and South Dakota county code changes.

**Outputs (all to `Data Int/`):**

- `state_real_minwages.parquet`
- `census_ag_cropland.parquet`
- `census_ag_cropland_2007.parquet`
- `h2a_data.parquet` — census-period panel
- `h2a_data_ts.parquet` — annual national time series
- `cdl_cropshares.parquet`
- `census_pop_ests.parquet`
- `aewr_data.parquet`, `aewr_data_full.parquet`

---

### `H2A Build Dataset.R`

**Role:** Constructs the two final analysis datasets: (1) a border county-pair dataset and (2) a county-year panel. These are the inputs to the regression analysis.

**Inputs:** All cleaned parquets from `H2A Clean and Load.R` plus geographic crosswalks.

**Border County-Pair Dataset (`border_df_analysis.parquet`):**

1. Starts from `border_counties_allmatches.parquet` — pre-identified pairs of counties on opposite sides of state borders
2. Merges AEWR for both the "state side" and "neighbor side" of each pair
3. Creates difference variables: `aewr_diff` (nominal), `aewr_ppi_diff` (real), `pct_aewr_diff`
4. Creates pair-level fixed effects: pair ID, pair × time, AEWR region–border combinations
5. Flags pairs where either county has no cropland in 2007 (for sample restriction)
6. Also produces `border_df_analysis_year.parquet` (same structure, by calendar year)

**County-Year Panel (`county_df_analysis_year.parquet`):**

1. Merges county-level data: H-2A, AEWR, population, employment (BEA + QCEW), farm income/expenses (BEA), cropland (NASS + CDL), min wages
2. Creates **H-2A exposure classification** based on pre-2011 usage:
   - `high_h2a_share_75`: county above 75th percentile of H-2A workers / farm employment
   - `high_h2a_share_66`, `high_h2a_share_50`: alternative thresholds
   - `high_h2a_count_75/66/50`: same but using absolute worker counts
   - `high_h2a_share_75_inverse`: 1 minus the above (used in DDD spec)
   - `county_treatment_group_classification`: "always takers", "adopters", "never takers"
3. Creates **year dummies** (`yeardummy_2008`–`yeardummy_2022`), base year = 2011
4. Creates `postdummy` = 1 if year ≥ 2012
5. Creates **fixed effect variables:** `county_fe`, `year_fe`, `statefips` (for clustering)
6. Creates **outcome variable:** `h2a_cert_share_farm_workers_2011_start_year` — H-2A workers certified normalized by 2011 farm employment baseline
7. Flags counties spanning multiple AEWR regions (commuting zone analysis)
8. Restricts to counties with `any_cropland_2007 == 1` for main sample
9. Runs missing-data diagnostics

**Outputs:**

- `border_df_analysis.parquet`
- `border_df_analysis_year.parquet`
- `county_df_analysis_year.parquet` ← main analysis dataset

---

### `H2A Analysis Figs and Tables.R`

**Role:** All econometric estimation and output production. Generates every figure and table in the paper.

**Inputs:**

- `county_df_analysis_year.parquet` — main panel
- `border_df_analysis.parquet` — border pairs
- `border_df_analysis_year.parquet`
- `aewr_data.parquet`, `aewr_data_full.parquet`
- `h2a_data.parquet`, `h2a_data_ts.parquet`
- `tl_2020_us_county.shp` — county shapefile for maps

**Libraries:** `sf`, `tidyverse`, `ggspatial`, `scales`, `cowplot`, `ggthemes`, `fixest`, `ggfixest`

#### Exhibits Produced

| Exhibit | Description                                                   | Output file                                                            |
| ------- | ------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 1       | Real AEWR time series (national avg, PPI-adjusted)            | `ts_national_aewr_real.png`                                            |
| 2       | Nominal AEWR time series                                      | `ts_national_aewr_nominal.png`                                         |
| 3       | H-2A program use over time (levels + indexed to 2011=100)     | `fig_line_ts_h2a_workers_certified.png`, `fig_line_ts_h2a_indexes.png` |
| 4       | County maps of H-2A use (log scale) for 2012, 2017, 2022      | map PNG files                                                          |
| 5       | Map of change in H-2A workers 2012→2022; new adopter counties | map PNG files                                                          |
| 6       | Map of AEWR growth by region relative to national trend       | diverging color map PNG                                                |
| 7       | County classification map (high/low H-2A exposure)            | sample classification map PNG                                          |
| 8       | AEWR region time series (AEWR levels by region)               | ts PNG                                                                 |
| 9A      | DDD parallel trends graph (commented-out version)             | —                                                                      |
| 9B      | DDD parallel trends using predicted usage groups              | `fig_ts_aewr_growth_exposure_using_predicted_DDD.png`                  |
| 10      | Main regression table (DD + DDD)                              | `table_1_main_results.tex`                                             |
| 11–12   | Event study coefficients (pre/post 2011, base year = 2011)    | `table_2_event_study.tex`, `coefplot_*.png`                            |

#### Main Regression Models

**Standard Difference-in-Differences (DD):**

```
H2A_share ~ AEWR_l1 × Post + ln_pop + emp_pop_ratio | county_fe + year_fe
```

- SE clustered by state
- Sample: counties with any cropland in 2007

**Triple Difference (DDD) — Main Specification:**

```
H2A_share ~ AEWR_l1 × HighH2A_75th_inverse × Post + controls | county_fe + year_fe
```

- `high_h2a_share_75_inverse` = 1 for low pre-treatment H-2A counties (those most affected by AEWR increases)
- SE clustered by state

**Event Study (DDD, year-by-year):**

```
H2A_share ~ AEWR_l1 × yeardummy_t × HighH2A_75th_inverse + controls | county_fe + year_fe
  for t ∈ {2008,...,2010, 2012,...,2022}, base year = 2011
```

- Pre-trend years (2008–2010) should have near-zero coefficients to validate parallel trends
- Coefficients extracted manually from `coeftable` rows 33–46 and plotted

**Outcome variable:** `h2a_cert_share_farm_workers_2011_start_year`
**Key regressor:** `aewr_state_ag_ppi_l1` (real lagged AEWR)
**Controls:** `ln_pop_census`, `emp_pop_ratio`
**Fixed effects:** County + year, via `fixest::feols()`
**Table export:** `fixest::etable()` to LaTeX

---

### `Read Parquet.R`

**Role:** Utility script for validating parquet files received from Ken. Only run this when new data arrives; it is commented out in `H2A Master.R`.

**What it does:** Iterates over a specified list of parquet files in `files_for_phil/`, reads each with `read_parquet()` to verify the file is valid and inspect structure. No outputs are written.

**Files it checks:**

- `state_year_min_wage.parquet`
- `oews_county_aggregated.parquet`
- `h2a_aggregated.parquet`
- `croplandcros_county_crop_acres.parquet`
- `aewr.parquet`
- `nass_census_selected_obs.parquet`

---

### `TSCB.R`

**Role:** Monte Carlo simulation to validate the cluster-robust inference procedure used in the main analysis. Based on the "Two-way Split-sample Cluster Bootstrap" (TSCB) paper published in the _Quarterly Journal of Economics_ (Abadie et al., 2023).

**Source:** https://academic.oup.com/qje/article/138/1/1/6750017

**Why it exists:** The border county-pair design has a complex clustering structure (32 AEWR regions, ~750 county pairs). This simulation checks whether state-clustered SEs provide correct coverage.

**Data Generating Process:**

- Population: 1,502 units paired into 751 border pairs
- Clusters: 32 (matching the number of AEWR region borders in the real data)
- Time periods: 3 years
- True treatment effect: β = 2
- Model: Y = α + β×Treatment + unit FE + pair×time FE + ε

**Simulation loop (1,000 iterations):**

1. Generate synthetic paired panel data with the DGP above
2. Estimate DDD model with pair and pair×time fixed effects via `fixest::feols()`
3. Compute cluster-robust (state) vs. bootstrap standard errors
4. Record coverage rates for 95% CIs

**Output:**

- `tscb_sim_results_{date}.csv` — simulation results for comparison of inferential approaches

**Note:** This script uses a hardcoded network path (`//acsnfs4.ucsd.edu/...`) for the output folder that is specific to Phil's institutional server. Run it standalone, not through Master.R.

---

## Notes on Coauthor Workflow

- **Phil** (this machine): runs all R analysis scripts in `Do/`
- **Ken** (coauthor): runs the Python data pipeline in `code/` and produces the parquet files in `Data Int/` that Phil consumes
- Data sharing is via Dropbox
- `Read Parquet.R` is Phil's tool for validating files received from Ken
- The `code/nathaniel_ai_code/` and `code/deprecated/` folders contain older/alternative versions that are no longer used in the main pipeline
- `Do/Pre 2026/` contains archived versions of all main R scripts from before 2026

---

## Coding Conventions

- All scripts use `here()` (via `H2A Master.R`) for portable paths
- Intermediate data is always stored as Apache Parquet (fast, typed, compressed)
- All regression output uses the `fixest` package for speed and `etable()` for LaTeX export
- Figures use `ggplot2` with `theme_clean()` (from `ggthemes`) as the default theme
- Real (inflation-adjusted) values use PPI WPU01 rebased to 2012=100
- Geographic unit: US counties (FIPS codes), sometimes aggregated to commuting zones or AEWR regions
- Sample years: 2008–2022 (census periods: 2007, 2012, 2017, 2022)
- Base/policy year: 2011
