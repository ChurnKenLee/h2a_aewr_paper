# H-2A Pipeline Refactor Execution Spec

This execution spec is authoritative. Any legacy boundary-audit material below is superseded unless it is explicitly restated in this spec.

## 0. Purpose

This document is the execution spec for refactoring the current H-2A R workflow from large monolithic scripts into smaller, explicitly ordered pipeline modules.

The goal is not to change the empirical design, variable definitions, datasets, figures, or tables. The goal is to make the existing workflow reproducible, inspectable, and easier to run in stages.

Hard constraint from Ken: the `code/` directory must remain flat. Do not create `code/01_ETL/`, `code/02_Assembly/`, `code/03_Analysis/`, or any other subdirectory for this refactor. All new refactor scripts go directly in `code/`.

## 1. Current Authoritative Entry Points

The current workflow is controlled by `Do/H2A Master.R`.

Current run order:

1. `Do/H2A Pull Min Wages.R`
2. `Do/H2A Clean and Load.R`
3. `Do/H2A Build Dataset.R`
4. `Do/H2A Analysis Figs and Tables.R`

Current pathing is already centralized through `code/paths.R`. That file discovers the project root, defines `path_*()` helpers, loads `.env`, and creates output/data directories.

The refactor must preserve that path layer. Do not reintroduce standalone hard-coded project roots such as `C:/Users/.../Dropbox/H-2A Paper/` inside new pipeline modules.

## 2. Scope and Non-Goals

In scope:

- Split the current R workflow into flat `code/p*.R` modules.
- Update `Do/H2A Master.R` so it can run either legacy scripts or refactored modules through stage switches.
- Migrate reads and writes away from legacy `Data Int` path assumptions and through the canonical `path_int()` helper.
- Add validation checks based on file contracts, schemas, keys, year ranges, and model smoke tests. Legacy-output comparison is optional because the current legacy run is already broken during the path migration.
- Perform code extraction and path edits through parse-data parquet transformations using `code/parse_script_to_parquet.R`, Python Polars, and `code/reconstruct_r_code_from_parse_data.py`.
- Move package loading, seeds, dates, and common assertions into one shared runtime file.
- Remove hidden reliance on ad hoc standalone path blocks in new modules.

Out of scope for the first pass:

- Changing model specifications.
- Renaming empirical variables.
- Changing sample restrictions.
- Performing broader storage reorganization beyond the active intermediate/output path migration.
- Deleting the legacy `Do/*.R` scripts before validation passes.
- Converting every block to functions. Function extraction is allowed only where it reduces immediate risk.
- Manually copy-pasting code blocks from the legacy scripts into new modules. The first-pass edits should be generated from parse-data tables.

## 3. Flat File Naming Convention

All refactor modules live directly under `code/`.

Use this naming pattern:

```text
code/p00_runtime.R
code/p09_etl_pull_min_wages.R
code/p10_etl_price_inputs.R
code/p11_etl_reference_crosswalks.R
code/p12_etl_minimum_wages.R
code/p13_etl_h2a.R
code/p14_etl_cropland_and_census_ag.R
code/p15_etl_aewr.R
code/p16_etl_bea.R
code/p17_etl_population.R
code/p18_etl_panel_spine.R
code/p20_assembly_load_inputs.R
code/p21_assembly_merge_panel.R
code/p22_assembly_variables.R
code/p23_assembly_fixed_effects_sample.R
code/p24_assembly_h2a_lags_groups.R
code/p25_assembly_export_panel.R
code/p40_analysis_load_inputs.R
code/p41_analysis_descriptive_trends.R
code/p42_analysis_spatial_maps.R
code/p43_analysis_dd_event_tables.R
code/p44_analysis_fisher_price.R
code/p45_analysis_stacked_matched_did.R
code/p90_validate_refactor.R
```

Use these flat helper scripts for parse-data editing:

```text
code/parse_script_to_parquet.R
code/reconstruct_r_code_from_parse_data.py
code/refactor_plan_polars.py
code/refactor_generate_modules_polars.py
code/refactor_validate_generated_code_polars.py
```

The first two files already exist. The `refactor_*_polars.py` scripts are implementation drivers to be added during the refactor; they must use Polars for reading, filtering, annotating, and writing parse-data parquet files.

Reserve number ranges:

- `p00`: runtime, package loading, helpers, assertions.
- `p09` to `p18`: ETL and intermediate data construction.
- `p20` to `p25`: analysis panel assembly.
- `p40` to `p45`: figures, maps, regressions, and tables.
- `p90`: validation.

## 4. Module Header Contract

Every new `code/p*.R` module must start with a contract comment:

```r
# Module: pXX_name.R
# Stage: etl | assembly | analysis | validation
# Source lines: legacy file and approximate line range
# Inputs: files or in-memory objects required before this module runs
# Outputs: files or in-memory objects produced by this module
# Side effects: figures, tables, logs, global objects, or none
# Validation: checks that must pass after this module runs
```

The contract is part of the implementation. A module is not complete until its contract is accurate.

## 5. Boundary Rules

Use these rules when extracting code:

1. A module boundary is valid only if inputs and outputs can be named explicitly.
2. RStudio section comments are useful hints, not sufficient boundaries.
3. ETL modules must read declared files and write declared parquet outputs. They should not depend on objects created by a previous ETL module except through declared files.
4. Assembly modules may pass `county_df` through memory from `p20` to `p25` during the first refactor. This preserves the current workflow and reduces risk.
5. Analysis modules must load their data explicitly in `p40_analysis_load_inputs.R`, then share `county_df`, `county_map`, `h2a_data_ts`, `h2a_data`, and `aewr_data_full` through the analysis run.
6. New modules must use `path_*()` helpers from `code/paths.R` or compatibility aliases set by `Do/H2A Master.R`.
7. Do not clear the whole workspace inside modules. If cleanup is needed, remove only local temporary objects owned by that module.
8. Set `set.seed(12345)` directly before stochastic steps, especially matching.
9. External credentials, including the FRED API key, must come from `.env` or environment variables. Do not hard-code API keys in refactored modules.
10. Treat the path migration as part of the refactor. New modules should write to `path_int()` and read from `path_int()` unless a specific raw input belongs under `path_raw()`.
11. If the intended migration target is `data/intermediates`, reconcile that spelling with `code/paths.R` once, then keep module code path-helper based. Do not spell `data/intermediates` directly throughout modules.
12. Code extraction edits must be made by transforming parse-data parquet with Polars, then reconstructing R code. Do not use manual text slicing or ad hoc source regex replacement as the primary edit mechanism.

## 5A. Parse-Data Editing Workflow

All first-pass code extraction and path rewrites should be generated from parse-data parquet files.

### Required Tools

- `code/parse_script_to_parquet.R`: parses an R script with `utils::getParseData()` and writes token data to parquet.
- `code/reconstruct_r_code_from_parse_data.py`: reads parse-data parquet with Polars and reconstructs R source from terminal tokens.
- Python Polars: performs all table joins, filters, line-range selections, token text updates, module annotations, and generated parquet writes.

### Parse Artifacts

Write parse artifacts under:

```text
data/intermediate/parse_data/
```

or the canonical `path_int("parse_data")` equivalent after `code/paths.R` is settled.

Required parse files:

```text
h2a_master_parse.parquet
h2a_pull_min_wages_parse.parquet
h2a_clean_and_load_parse.parquet
h2a_build_dataset_parse.parquet
h2a_analysis_figs_tables_parse.parquet
```

### Edit Manifest

Create a Polars-produced manifest at:

```text
outputs/logs/refactor_parse_manifest.parquet
```

The manifest must contain at least:

```text
source_file
source_parse_file
module_file
stage
source_line_start
source_line_end
selection_reason
required_inputs
declared_outputs
path_rewrite_policy
validation_checks
```

The manifest is the source of truth for which token rows become each generated module.

### Generation Steps

1. Parse each legacy R script with `Rscript code/parse_script_to_parquet.R`.
2. Load all parse parquet files with Polars.
3. Use Polars to annotate tokens with source file, module target, stage, and line-range membership.
4. Generate module parse tables by filtering terminal and non-terminal rows according to the manifest.
5. Rebase `line1`, `line2`, `col1`, and `col2` coordinates for each generated module so reconstruction starts at line 1 and preserves intra-module spacing.
6. Apply path edits in Polars by changing token text for path literals and path helper calls. New modules should call `path_int()`, `path_raw()`, `path_outputs()`, `path_figures()`, `path_tables()`, or compatibility aliases from `p00_runtime.R`.
7. For inserted headers or boilerplate, parse the boilerplate snippet into parquet first, then combine the parsed boilerplate tokens with the selected legacy tokens using Polars. Do not prepend raw strings directly to reconstructed files.
8. Write one generated parse parquet per target module under `path_int("parse_data/generated_modules")`.
9. Reconstruct each generated module with `code/reconstruct_r_code_from_parse_data.py`.
10. Re-parse each reconstructed module with `code/parse_script_to_parquet.R` and compare the second parse to the generated parse table for structural sanity.

### Polars Edit Rules

- Use Polars expressions for token filtering, line rebasing, file assignment, and path literal replacement.
- Keep an audit table of every token whose `text` value changes, including old text, new text, source file, source line, and reason.
- Preserve comments from parse data where available.
- Do not use Python string concatenation to assemble full R modules, except inside the existing reconstruction utility.
- Do not use shell `sed`, `awk`, `perl`, or manual editor operations for source-code transformations.
- If a required edit cannot be represented safely as a parse-data token transformation, record it in `outputs/logs/refactor_manual_review_required.parquet` and stop before writing that module.

## 6. Master Runner Target

`Do/H2A Master.R` remains the user-facing entrypoint.

Target behavior:

```r
source(file.path(project_root, "code", "paths.R"))
ensure_project_dirs()

source(path_code("p00_runtime.R"))

run_legacy <- FALSE
run_min_wages <- TRUE
run_etl <- TRUE
run_assembly <- TRUE
run_analysis <- TRUE
run_validation <- TRUE

if (run_legacy) {
  source(paste0(folder_do, "H2A Pull Min Wages.R"), echo = TRUE)
  source(paste0(folder_do, "H2A Clean and Load.R"), echo = TRUE)
  source(paste0(folder_do, "H2A Build Dataset.R"), echo = TRUE)
  source(paste0(folder_do, "H2A Analysis Figs and Tables.R"), echo = TRUE)
} else {
  if (run_min_wages) source(path_code("p09_etl_pull_min_wages.R"))

  if (run_etl) {
    source(path_code("p10_etl_price_inputs.R"))
    source(path_code("p11_etl_reference_crosswalks.R"))
    source(path_code("p12_etl_minimum_wages.R"))
    source(path_code("p13_etl_h2a.R"))
    source(path_code("p14_etl_cropland_and_census_ag.R"))
    source(path_code("p15_etl_aewr.R"))
    source(path_code("p16_etl_bea.R"))
    source(path_code("p17_etl_population.R"))
    source(path_code("p18_etl_panel_spine.R"))
  }

  if (run_assembly) {
    source(path_code("p20_assembly_load_inputs.R"))
    source(path_code("p21_assembly_merge_panel.R"))
    source(path_code("p22_assembly_variables.R"))
    source(path_code("p23_assembly_fixed_effects_sample.R"))
    source(path_code("p24_assembly_h2a_lags_groups.R"))
    source(path_code("p25_assembly_export_panel.R"))
  }

  if (run_analysis) {
    source(path_code("p40_analysis_load_inputs.R"))
    source(path_code("p41_analysis_descriptive_trends.R"))
    source(path_code("p42_analysis_spatial_maps.R"))
    source(path_code("p43_analysis_dd_event_tables.R"))
    source(path_code("p44_analysis_fisher_price.R"))
    source(path_code("p45_analysis_stacked_matched_did.R"))
  }

  if (run_validation) source(path_code("p90_validate_refactor.R"))
}
```

The implementation can be staged, but the final runner must support independent stage switches.

## 7. ETL Module Contracts

| Module | Legacy source | Reads | Writes | Required validation |
|---|---|---|---|---|
| `p09_etl_pull_min_wages.R` | `Do/H2A Pull Min Wages.R`, lines 1-121 | FRED API, `fips_codes.csv`, `FRED_API_KEY` from `.env` | `fred_state_minwages.parquet` | All state FIPS <= 56 represented where FRED has data; federal wage non-missing by year; no hard-coded API key |
| `p10_etl_price_inputs.R` | `Do/H2A Clean and Load.R`, lines 16-26 and 90-120 | `price_index_fisher_county_year_nass_price_yield_cdl_acres.parquet`, `WPU01.csv` | `nass_fisher_price_index.parquet`, `ppi_2012.parquet` | `ppi_2012 == 1` for 2012 within tolerance; NASS price index limited to 2008-2022 |
| `p11_etl_reference_crosswalks.R` | `Do/H2A Clean and Load.R`, lines 34-87 and reference reads reused later | `fips_codes.csv`, `aewr_regions.csv`, `counties10-zqvz0r.csv`, county boundary CSVs | `cz_file_2010.parquet`, `cz_file_2010_small.parquet` | `countyfips` non-missing; `cz_file_2010_small` has exactly `countyfips` and `cz_out10` |
| `p12_etl_minimum_wages.R` | `Do/H2A Clean and Load.R`, lines 139-203 | `fred_state_minwages.parquet`, `state_year_min_wage.parquet`, `ppi_2012.parquet` | `state_real_minwages.parquet` | Real wage columns exist; agriculture exemptions filled; no missing `prevailing_ag_min_wage_ppi` for covered state-years |
| `p13_etl_h2a.R` | `Do/H2A Clean and Load.R`, lines 208-359 and 543-591 | `h2a_aggregated.parquet`, `h2a_prediction_using_elastic_net.parquet` | `h2a_predict.parquet`, `h2a_data.parquet`, `h2a_data_ts.parquet`, `h2a_data_year.parquet` | `countyfips` constructed as numeric 5-digit FIPS; annual panel restricted to 2008-2022 |
| `p14_etl_cropland_and_census_ag.R` | `Do/H2A Clean and Load.R`, lines 361-411 and 475-539 | `croplandcros_cdl_crop_category.csv`, `croplandcros_county_crop_acres.parquet`, `qs_census_selected_obs.parquet` | `cdl_cropshares.parquet`, `census_ag_cropland_year.parquet`, `census_ag_cropland_2007_year.parquet` | No missing crop share cells after zero fill; 2007 cropland base has one row per county |
| `p15_etl_aewr.R` | `Do/H2A Clean and Load.R`, lines 594-803 | `aewr.parquet`, `aewr_crs_1990_2008.csv`, `ppi_2012.parquet`, `fips_codes.csv`, `aewr_regions.csv` | `aewr_data_year.parquet`, `aewr_data_full.parquet` | Annual AEWR data covers 2008-2022; region time series has non-missing `aewr_region_num` |
| `p16_etl_bea.R` | `Do/H2A Clean and Load.R`, lines 1008-1259 | BEA CAEMP25N and CAINC45 CSVs, `bea_fips_xwalk.csv`, `county_adjacency2010.csv`, `ppi_2012.parquet` | `bea_caemp25n_data_year.parquet`, `bea_cainc45_data_year.parquet` | Years restricted to 2008-2022; Oglala Lakota/Shannon handling preserved; PPI-deflated finance columns exist |
| `p17_etl_population.R` | `Do/H2A Clean and Load.R`, lines 1263-1414 | Census population CSVs, `ct_region_xwalk.csv`, `ct_pop_grth.csv` | `census_pop_ests_year.parquet` | Years restricted to 2008-2022; CT 2020-2022 county equivalent handling preserved; SD FIPS change preserved |
| `p18_etl_panel_spine.R` | `Do/H2A Clean and Load.R`, lines 1417-1441 | `county_adjacency2010.csv` | `county_df_year.parquet` | One county-year row for each base county and year 2008-2022 |

Notes:

- `NAWSPAD` and `OEWS` reads in `Do/H2A Clean and Load.R`, lines 413-430, currently appear diagnostic or unused by downstream modules. Preserve them only if a downstream dependency is confirmed. Otherwise list them as deferred cleanup in the implementation notes.
- AEWR diagnostic plots from `Do/H2A Clean and Load.R`, lines 803-1006, should move to analysis or an optional diagnostics switch, not remain inside core ETL.

## 8. Assembly Module Contracts

| Module | Legacy source | Inputs | Outputs | Required validation |
|---|---|---|---|---|
| `p20_assembly_load_inputs.R` | `Do/H2A Build Dataset.R`, lines 23-101 | All ETL parquet files plus `aewr_regions.csv`, `fips_codes.csv` | In-memory inputs for assembly | All required files exist before merge begins |
| `p21_assembly_merge_panel.R` | `Do/H2A Build Dataset.R`, lines 103-276 | Objects loaded by `p20` | In-memory `county_df` with merged AEWR, BEA, H-2A, cropland, population, min wage, wage quantile, CZ, and price data | Key uniqueness preserved for `countyfips, year`; no unintended row multiplication |
| `p22_assembly_variables.R` | `Do/H2A Build Dataset.R`, lines 276-477 | `county_df`, `cz_file_small` | `county_df` with wage gaps, log variables, budget shares, H-2A normalized outcomes | Required variables exist: `aewr_cz_p25_l1`, `ln_pop_census`, `emp_pop_ratio`, `h2a_cert_share_farm_workers_2011_start_year` |
| `p23_assembly_fixed_effects_sample.R` | `Do/H2A Build Dataset.R`, lines 478-580 | `county_df` | `county_df` with fixed effects, sample flags, NA diagnostics | FE variables exist; `any_cropland_2007` created; AK, HI, and DC handling unchanged through missing AEWR filter |
| `p24_assembly_h2a_lags_groups.R` | `Do/H2A Build Dataset.R`, lines 581-938 | `county_df` | `county_df` with H-2A lags, treatment groups, H-2A cut variables, year dummies, border CZ flags, post dummy | Year dummies cover 2008-2022; treatment group labels match legacy set; border CZ counts match legacy |
| `p25_assembly_export_panel.R` | `Do/H2A Build Dataset.R`, lines 939-953 | Final `county_df` | `county_df_analysis_year.parquet`; validation log | Row count > 10000; required columns present; saved parquet reloads successfully |

Assembly extraction rule:

- During the first refactor, `county_df` may remain an in-memory object passed through `p20` to `p25`. Do not force a functional rewrite until after equivalence is established.

## 9. Analysis Module Contracts

| Module | Legacy source | Inputs | Outputs | Required validation |
|---|---|---|---|---|
| `p40_analysis_load_inputs.R` | `Do/H2A Analysis Figs and Tables.R`, lines 1-52 | `tl_2020_us_county.shp`, `h2a_data_ts.parquet`, `h2a_data.parquet`, `aewr_data_full.parquet`, `county_df_analysis_year.parquet` | In-memory `county_map`, `h2a_data_ts`, `h2a_data`, `aewr_data_full`, `county_df` | County shapefile loads; simplified map has numeric `countyfips`; analysis panel has required columns |
| `p41_analysis_descriptive_trends.R` | `Do/H2A Analysis Figs and Tables.R`, lines 53-287, 903-1030, 1599-1683 | Objects from `p40` | Descriptive histograms, AEWR/H-2A time series, event coefficient plots, summary statistics table | Expected figure/table files exist and are non-empty |
| `p42_analysis_spatial_maps.R` | `Do/H2A Analysis Figs and Tables.R`, lines 288-902 and 1835-1897 | Objects from `p40` | H-2A maps, AEWR maps, exposure maps, Fisher price county map | Expected map files exist and are non-empty |
| `p43_analysis_dd_event_tables.R` | `Do/H2A Analysis Figs and Tables.R`, lines 1031-1683 | `county_df` | Main DD models, event study models, `table_1_main_results.tex`, `table_2_event_study.tex` | `samp_base` row count > 1000; DD/event coefficients match legacy within tolerance |
| `p44_analysis_fisher_price.R` | `Do/H2A Analysis Figs and Tables.R`, lines 1737-1975 | `county_df`, `county_map` | Fisher price time series, Fisher price DD table, labor share investigation | Fisher model coefficients match legacy within tolerance; figure/table files exist |
| `p45_analysis_stacked_matched_did.R` | `Do/H2A Analysis Figs and Tables.R`, lines 1976-2546 | `county_df` | Treatment classification, PSM outputs, stacked event studies, coefficient plots | Matching seed set locally; matched sample counts match legacy; stacked model coefficients match legacy within tolerance |

Analysis extraction rule:

- Keep estimation objects local to their module unless a later module explicitly needs them.
- If a plot depends on data generated in an earlier plot block, promote that data construction into `p40_analysis_load_inputs.R` or the earliest module that requires it.

## 10. Validation Spec

Create `code/p90_validate_refactor.R`.

Validation has three levels. The current legacy scripts are not assumed to be runnable because the project is already mid-migration from `Data Int` to the canonical intermediate data path. Therefore, validation must not require a fresh legacy baseline.

If old output snapshots already exist and are known to be valid, they may be used as reference material. They are not a blocker for the refactor.

### Level 1: File and Schema Validation

For each refactored parquet output:

- File exists.
- File size is greater than zero.
- Expected key columns exist.
- Expected key is unique where applicable.
- Expected year range is preserved.
- Column names satisfy the module contract.

Required parquet outputs:

```text
fred_state_minwages.parquet
nass_fisher_price_index.parquet
ppi_2012.parquet
cz_file_2010.parquet
cz_file_2010_small.parquet
state_real_minwages.parquet
h2a_predict.parquet
h2a_data.parquet
h2a_data_ts.parquet
h2a_data_year.parquet
cdl_cropshares.parquet
census_ag_cropland_year.parquet
census_ag_cropland_2007_year.parquet
aewr_data_year.parquet
aewr_data_full.parquet
bea_caemp25n_data_year.parquet
bea_cainc45_data_year.parquet
census_pop_ests_year.parquet
county_df_year.parquet
county_df_analysis_year.parquet
```

### Level 2: Dataset Contract and Plausibility Validation

For each dataset, write a validation summary to `path_logs("refactor_validation_current/")`.

For each dataset, save:

- `nrow`
- `ncol`
- sorted column names
- key counts
- year counts
- numeric summaries for variables used in regressions
- totals for H-2A workers, applications, man-hours, farm employment, and cropland

Validate each summary against explicit expectations:

- Required columns exist.
- Required key columns are non-missing.
- Keys are unique where the dataset claims one row per key.
- Years are in the expected range for the dataset.
- H-2A count variables are non-negative.
- Share variables that are intended to be shares are finite or explicitly `NA`.
- Monetary variables that should be real 2012 dollars are generated only after joining `ppi_2012`.
- Join steps do not multiply rows unexpectedly.

Optional comparison:

- If a trusted old summary exists, compare numeric summaries with `all.equal(..., tolerance = 1e-8)`.
- If no trusted old summary exists, record the current summary as the first refactored reference snapshot.

### Level 3: Model Smoke and Output Validation

Run these model families and record observation counts, coefficient names, coefficient values, standard errors, fixed effects, and clustering variables:

- Main DD models: `dd_1`, `dd_2`, `dd_3`, `dd_4`.
- Event study models: `es_1`, `es_2`, `es_3`, `es_4`.
- Fisher price DD models.
- Labor share DD models.
- Stacked matched DiD event study models.

For each model:

- Model runs without error.
- Required coefficient names are present.
- Coefficients and standard errors are finite unless a term is intentionally dropped for collinearity.
- Number of observations is recorded and is positive.
- Fixed effect structure is unchanged.
- Cluster variable is unchanged.
- Matching and stacked DiD steps set their seed locally.

Optional comparison:

- If trusted legacy estimates already exist, compare coefficients, standard errors, and observation counts within tolerance.
- If no trusted legacy estimates exist, save the first successful refactored estimates as the reference snapshot for future changes.

For figures and tables:

- Required filenames exist.
- File sizes are greater than zero.
- LaTeX table files contain the expected model titles.
- Figure checksum equality is optional because graphics devices can produce harmless metadata differences.

## 11. Migration Sequence

Execute the refactor in this order.

### Step 1: Stabilize Paths and Inventory Available Data

Do not require the current legacy scripts to run before refactoring. They are already broken by the active path migration.

First, settle the intermediate data path contract:

- Confirm whether the canonical intermediate directory is `data/intermediate` or `data/intermediates`.
- Update `code/paths.R` once if the intended target is `data/intermediates`.
- Require all new modules to use `path_int()` rather than literal intermediate paths.

Then inventory currently available files:

- List all files currently available under the canonical intermediate directory.
- Identify each required input as available, missing, or produced by an upstream refactor module.
- Save this inventory under `path_logs("refactor_input_inventory.csv")`.

Do not block on missing outputs that are supposed to be produced by upstream modules.

### Step 2: Parse Legacy Scripts and Build the Polars Edit Manifest

Parse the current legacy scripts to parquet:

```bash
Rscript code/parse_script_to_parquet.R "Do/H2A Master.R" data/intermediate/parse_data/h2a_master_parse.parquet
Rscript code/parse_script_to_parquet.R "Do/H2A Pull Min Wages.R" data/intermediate/parse_data/h2a_pull_min_wages_parse.parquet
Rscript code/parse_script_to_parquet.R "Do/H2A Clean and Load.R" data/intermediate/parse_data/h2a_clean_and_load_parse.parquet
Rscript code/parse_script_to_parquet.R "Do/H2A Build Dataset.R" data/intermediate/parse_data/h2a_build_dataset_parse.parquet
Rscript code/parse_script_to_parquet.R "Do/H2A Analysis Figs and Tables.R" data/intermediate/parse_data/h2a_analysis_figs_tables_parse.parquet
```

If the canonical directory is changed to `data/intermediates`, update these paths through `path_int()`-based driver code rather than scattering the literal path.

Then implement `code/refactor_plan_polars.py` to:

- Read the parse parquet files with Polars.
- Create the module manifest from the contracts in this spec.
- Verify that every target source line range exists.
- Flag overlapping or unassigned extraction ranges.
- Write `outputs/logs/refactor_parse_manifest.parquet`.

### Step 3: Generate Modules from Parse Data

Implement `code/refactor_generate_modules_polars.py`.

This script must:

- Read `outputs/logs/refactor_parse_manifest.parquet`.
- Read source parse parquet files with Polars.
- Select token rows for each module.
- Rebase token coordinates for each module.
- Apply path-helper token edits in Polars.
- Combine parsed boilerplate snippets with extracted module tokens when headers or runtime scaffolding are needed.
- Write generated module parse parquet files.
- Reconstruct module `.R` files with `code/reconstruct_r_code_from_parse_data.py`.

No module should be created by manual copy-paste.

### Step 4: Add Runtime and Runner Switches

Create `code/p00_runtime.R`.

It should:

- Load required packages.
- Define `RUN_DATE`.
- Set common seed value.
- Define assertion helpers such as `assert_file_exists()`, `assert_has_cols()`, `assert_unique_key()`, and `assert_year_range()`.
- Define compatibility aliases only if needed, using `path_*()` helpers from `code/paths.R`.

Update `Do/H2A Master.R` to support `run_legacy`, `run_etl`, `run_assembly`, `run_analysis`, and `run_validation`.

Default during migration:

```r
run_legacy <- FALSE
```

Run only the refactored stages whose modules already exist. Keep the legacy branch available for reference, but do not require it to execute.

`p00_runtime.R` and the updated runner should also be generated through the parse-data workflow. For new boilerplate that has no legacy source range, create a temporary R snippet, parse it to parquet, transform it with Polars if needed, and reconstruct the final file.

### Step 5: Extract ETL Modules

Extract modules `p09` through `p18` one at a time.

After each module:

1. Run only the module and its prerequisites.
2. Run validation for the files it writes.
3. Update the current validation summary.
4. Commit or checkpoint only after validation passes.

### Step 6: Extract Assembly Modules

Extract `p20` through `p25`.

After `p25`, validate `county_df_analysis_year.parquet`. Do not proceed to analysis extraction until the panel passes contract and plausibility checks.

Minimum panel checks:

- Row count is positive and large enough for the intended county-year panel.
- Required column names exist.
- `countyfips, year` is unique.
- `any_cropland_2007` distribution by year is recorded and has both expected sample categories when applicable.
- `border_cz` is generated and has non-missing values for regression rows.
- H-2A totals by year are non-negative and recorded.
- Main regression variables have finite numeric summaries for the estimation sample.

### Step 7: Extract Analysis Modules

Extract `p40` through `p45`.

After each module:

1. Check expected outputs exist.
2. Check model estimates where applicable.
3. Keep output filenames unchanged unless a rename is explicitly approved.

### Step 8: Validate Generated Source Round Trip

Implement `code/refactor_validate_generated_code_polars.py`.

For each generated module:

1. Parse the reconstructed `.R` file back to parquet.
2. Compare token counts, terminal token order, and edited-token audit rows with the generated parse parquet.
3. Confirm no forbidden path literals or hard-coded credentials are present in terminal token text.
4. Record results under `path_logs("refactor_generated_code_validation.parquet")`.

### Step 9: Switch Default Runner

When ETL, assembly, analysis, and validation pass:

```r
run_legacy <- FALSE
```

Keep the legacy `Do/*.R` scripts for at least one validation cycle after switching the default.

## 12. Acceptance Criteria

The refactor is accepted only when all criteria pass:

- `code/` remains flat.
- No new code subdirectories are created.
- Legacy scripts were parsed with `code/parse_script_to_parquet.R`.
- Module extraction and path rewrites were generated from parse-data parquet using Polars.
- `outputs/logs/refactor_parse_manifest.parquet` exists and maps each generated module to source line ranges or parsed boilerplate snippets.
- An edited-token audit exists and records every Polars token text change.
- Each generated module has a corresponding generated parse parquet under `path_int("parse_data/generated_modules")`.
- Each generated module can be parsed again by `code/parse_script_to_parquet.R`.
- `Do/H2A Master.R` can run the full refactored pipeline.
- `Do/H2A Master.R` can run ETL, assembly, and analysis independently through switches.
- New `code/p*.R` files do not contain hard-coded machine-specific project roots.
- New `code/p*.R` files do not contain hard-coded API keys.
- Refactored parquet schemas satisfy the module contracts.
- `county_df_analysis_year.parquet` passes key, schema, year-range, and estimation-sample checks.
- Main regression and event-study modules run and write model summaries under `path_logs()`.
- Trusted legacy comparisons are run only when trusted old snapshots already exist.
- Required figures and LaTeX tables are produced with existing filenames.

Suggested path hygiene check:

```bash
rg -n "C:/Users|Dropbox/H-2A Paper|Data Int|Output/" code/p*.R "Do/H2A Master.R"
```

Suggested parse-token path hygiene check:

```bash
python code/refactor_validate_generated_code_polars.py
```

Suggested secret hygiene check:

```bash
rg -n "fredr_set_key\\(\"|[A-Za-z0-9]{32}" code/p*.R
```

Both checks should return no actionable hits.

## 13. Open Decisions

These decisions must be made before or during implementation:

1. Canonical data directory: confirm whether the intended path is `data/intermediate` or `data/intermediates`, then encode that choice once in `code/paths.R`.
2. AEWR diagnostic plots: move them into `p41_analysis_descriptive_trends.R` or create a separate optional flat module such as `p46_analysis_aewr_diagnostics.R`.
3. NAWSPAD and OEWS reads: confirm whether they are dead diagnostics or needed for future analysis.
4. Functional rewrite depth: keep first refactor source-based and side-effect compatible, then convert stable modules into functions later.
5. Baseline tolerance: use `1e-8` for optional numeric comparisons only when trusted old snapshots already exist.

## 14. Implementation Principle

This is a behavior-preserving refactor. If a cleaner design would change outputs, defer it. First make the current workflow modular and testable; then make methodological or design improvements in separate, reviewable changes.

<!--
## Legacy Boundary Audit (Superseded)

The monolithic code block is composed of three distinct functional files originally authored as separate scripts. This combination masks an econometric research pipeline on the H-2A agricultural worker program, spanning from raw data extraction and transform (ETL) to dataset assembly and regression modeling.

Below is the project-level boundary map showing where each script begins, where it ends, and its core functional scope:

```
[Auxiliary Raw Data Sources] ──>  FILE 3: ETL & Cleaning  ──> [Data Int/ Intermediate Parquet Files]
                                                                        │
[Analytical Panel Dataset]     <──  FILE 2: Data Assembly ◄─────────────┘
          │
          ▼
FILE 1: Estimations & Plots   ──> [Output/ Figures, Map Files, LaTeX Tables]
```

### High-Level File Boundary Directory

| Original File ID & Name                        | Context / Start Boundary                                 | Context / End Boundary                                                         | Functional Scope                                                                                                                                                                      |
| :--------------------------------------------- | :------------------------------------------------------- | :----------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **File 1**: `H2A: Analysis Figures and Tables` | `## H2A: Analysis Figures and Tables` <br>(Line 1)       | `ggsave(paste0(folder_output, "coefplot_stacked_dd_price_decrease.png"), ...)` | Computes difference-in-differences (DiD), event studies, stacked matched staggered DiDs, generates GIS maps (`sf`), renders charts, exports LaTeX regression tables (`etable`).       |
| **File 2**: `H2A Build Dataset`                | `﻿## H2A Build Dataset` <br>(Line 1445)                  | `rm(objects)` <br>`gc()` <br>`rm(objects)`                                     | Merges commuting zones, BEA employment, NASS price index, AEWR, and ACS wage quantiles; scales, deflates, computes lags, constructs fixed effects, and writes the analytical dataset. |
| **File 3**: `H2A: Load and clean datasets`     | `﻿## Run standalone or via H2A Master.R` <br>(Line 2235) | `rm(list = objects[, 1])` <br>`gc()` <br>`rm(objects)`                         | ETL engine. Normalizes Census of Ag, BEA finance, Commuting Zone spatial joins, state/federal agricultural minimum wage exceptions, price indices, and BLS OEWS records.              |

---

## 2. RStudio Delineation Syntax Standards

To split these monolithic files automatically or systematically, we must exploit RStudio's native document structural parsing rules. RStudio reads comments as structural outlines if they match one of the following patterns:

1. **RStudio Native Section Break (Class A)**: A comment line starting with at least one `#` (frequently `##`, `###`, or `####`) and ending with four or more trailing hyphens (`----`), equal signs (`====`), or pound signs (`####`). RStudio parses these as anchors to populate the **Document Outline Pane** (Ctrl+Shift+O).
2. **Double-Hash Structural Metadata Header (Class B)**: Comments starting with `## ` at the absolute beginning of a line followed by capitalized titles, names, dates, or run-time instructions.
3. **Execution Gate and Clean-up Checkpoints (Class C)**: Statements that modify or reset the global environment state (e.g., `rm()`, `gc()`, `set.seed()`, `Sys.sleep()`, or conditional path initializations `if (!exists("folder_dir"))`).
4. **Data Input/Output Contract Transitions (Class D)**: Code lines that write data out to the disk (`write_parquet`, `ggsave`, `write_xlsx`, `etable`) or read files from the disk (`read_parquet`, `read.csv`, `st_read`). These mark the end of one process and the start of another.

---

## 3. Exhaustive Directory of Potential Delineation Indicators

Below is an exhaustive line-by-line audit of every delineation indicator across the three scripts.

### File 1: Analysis Figures and Tables (Phil Hoxie, 1/31/24)

This script is structured around rendering descriptive figures, spatial maps, regression outputs, and staggered/stacked models.

| Relative Line # | Raw Indicator Text                                                                     | Syntactic Class                      | Purpose in Decoupling / Decoupled Destination File                                         |
| :-------------- | :------------------------------------------------------------------------------------- | :----------------------------------- | :----------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **1**           | `## H2A: Analysis Figures and Tables`                                                  | Class B: Script Header               | Title header for the descriptive and analytical output script.                             |
| **2**           | `## Phil Hoxie`                                                                        | Class B: Author Metadata             | Author metadata checkpoint.                                                                |
| **3**           | `## 1/31/24`                                                                           | Class B: Date Metadata               | Version control temporal anchor.                                                           |
| **5**           | `## Run standalone or via H2A Master.R`                                                | Class B: Execution Metadata          | Indicates parent runner setup; key for setting up master execution.                        |
| **6-11**        | `if (!exists("folder_dir")) { ... }`                                                   | Class C: Execution Gate              | Path setup contract; must be factored into a global config file.                           |
| **13**          | `gc()`                                                                                 | Class C: Clean-up Block              | Invocations of garbage collection; perfect point for environment transition.               |
| **15-24**       | `library(sf) ... library(writexl)`                                                     | Class B: Library Block               | Package dependency block; belongs in a dedicated setup/loading file.                       |
| **28**          | `set.seed(12345)`                                                                      | Class C: Seed Anchor                 | Standardizes stochastic steps (e.g., matching). Must be placed at the start of estimators. |
| **30-34**       | `date <- paste0(substr(Sys.Date(), 1, 4) ... )`                                        | Class C: Variable Generator          | Creates R-run timestamps; can be unified in a global properties file.                      |
| **36**          | `#### Load and Clean County Shape Files -------------------------------------------`   | Class A: RStudio Section Divider     | Structural break. Separates environmental setup from spatial data ingestion.               |
| **44**          | `#### County DDD Design -----------------------------------------------------------`   | Class A: RStudio Section Divider     | Structural break. Marks the loading of the primary processed county panel file.            |
| **53**          | `#### Exhibit 0: Distribution of AEWR Bite Variables (real, non-lagged) ---------`     | Class A: RStudio Section Divider     | **Sub-file Break**: Start of `01_descriptive_plots.R`.                                     |
| **89**          | `# map`                                                                                | Class B: Segment Label               | Segregates general plots from spatial maps.                                                |
| **97**          | `# simplify the map`                                                                   | Class B: Segment Label               | Explains the spatial compression process (`st_simplify`) to save rendering memory.         |
| **113**         | `#### Exhibit 1: AEWR TS Real ---------------------------------------------------`     | Class A: RStudio Section Divider     | Plot divider. Segment for national real AEWR time series.                                  |
| **133**         | `#### Exhibit 2: AEWR TS Nominal --------------------------------------------------`   | Class A: RStudio Section Divider     | Plot divider. Segment for nominal AEWR national time series.                               |
| **148**         | `#### Exhibit 1C: Real AEWR vs p25 CZ Wage Bite TS ## ---------------------------`     | Class A: RStudio Section Divider     | Plot divider. Segment for spatial wage gap calculations.                                   |
| **175**         | `#### Exhibit 3: H2A Use TS ---------------------------------------------------------` | Class A: RStudio Section Divider     | Plot divider. Segment for plotting aggregate program trends.                               |
| **177**         | `# levels`                                                                             | Class B: Segment Label               | Subsection indicator for raw levels plotting.                                              |
| **197**         | `# indexed`                                                                            | Class B: Segment Label               | Subsection indicator for base year (=2011) index plotting.                                 |
| **235**         | `# zeros should not be graphed #`                                                      | Class B: Segment Label               | Explains outlier trimming rules prior to visualization.                                    |
| **288**         | `#### Exhibit 4: H2A Map: 2012 use ----------------------------------------------`     | Class A: RStudio Section Divider     | **Sub-file Break**: Start of `02_spatial_mapping.R` (GIS visualizer).                      |
| **346**         | `# 2012`                                                                               | Class B: Segment Label               | Segregates logarithmic coordinate scale processing from level maps.                        |
| **383**         | `# no logs`                                                                            | Class B: Segment Label               | Subsection indicator for map generation using absolute worker counts.                      |
| **424**         | `## Predicted H2A Share Histogram -----------------------------------------------`     | Class A: RStudio Section Divider     | Plot divider. Distribution chart of pre-treatment propensity.                              |
| **462**         | `## Predicted H2A map -----------------------------------------------------------`     | Class A: RStudio Section Divider     | Plot divider. Geographical breakdown of the first-stage predicted counties.                |
| **505**         | `#### Exhibit 5: H2A Map: Change from 2012 use ----------------------------------`     | Class A: RStudio Section Divider     | Plot divider. Visualizes decadal spatial shifts in H-2A adoption.                          |
| **533**         | `# change from 2012`                                                                   | Class B: Segment Label               | Map configuration helper block.                                                            |
| **571**         | `# New from 2012`                                                                      | Class B: Segment Label               | Segment mapping program entry points in historically unexposed counties.                   |
| **614**         | `#### Exhibit 6: AEWR Map: AEWR Difference from National Trend ------------------`     | Class A: RStudio Section Divider     | Map divider. Detrended wage progression across DOL regions.                                |
| **625**         | `# need to make high growth and low growth regions`                                    | Class B: Explanatory Comment         | Mathematical explanation of detrending logic; marks logic block start.                     |
| **681**         | `# need to make an AEWR region to stat xwalk`                                          | Class B: Explanatory Comment         | Highlights crosswalk generation step; important join checkpoint.                           |
| **746**         | `#### Exhibit 6B: AEWR p25 Bite Change 2008–2022 (County-Level Map) ## ----------`     | Class A: RStudio Section Divider     | Map divider. Localized wage compression visualizer.                                        |
| **748**         | `# County-level change in AEWR p25 bite from 2008 to 2022 (in 2012 $ levels)`          | Class B: Explanatory Comment         | Metadata comment defining visual scale ranges.                                             |
| **819**         | `#### Exhibit 7: Exposure Map: Counties by Sample Classification ----------------`     | Class A: RStudio Section Divider     | Map divider. Categorization map for triple-difference validation.                          |
| **820**         | `## DDD Sample Map`                                                                    | Class B: Segment Label               | GIS rendering for DDD controls.                                                            |
| **880**         | `#### Exhibit 8: AEWR TS Difference from Trend by Region ------------------------`     | Class A: RStudio Section Divider     | Plot divider. Renders detrended wage vectors.                                              |
| **882**         | `# input data is made with the map above`                                              | Class B: Scope Dependency            | Highlights critical sequential dependency across code segments.                            |
| **920**         | `#### Exhibit 9: Distribution of AEWR - p10 wage --------------------------------`     | Class A: RStudio Section Divider     | Plot divider. Density distributions of Local Wage biting dynamics.                         |
| **921**         | `# Calculate diffs`                                                                    | Class B: Segment Label               | Data processing segment prior to density calculations.                                     |
| **942**         | `# CZ 10th percentile wage`                                                            | Class B: Segment Label               | Density generation for local 10th percentile dynamics.                                     |
| **956**         | `# CZ 25th percentile wage`                                                            | Class B: Segment Label               | Density generation for local 25th percentile dynamics.                                     |
| **970**         | `# CZ 50th percentile wage`                                                            | Class B: Segment Label               | Density generation for local median dynamics.                                              |
| **984**         | `## % CH`                                                                              | Class B: Segment Label               | Transition indicator to percentage-growth estimators.                                      |
| **986**         | `# CZ 10th percentile wage`                                                            | Class B: Segment Label               | Percentage density generation (10th).                                                      |
| **1003**        | `# CZ 25th percentile wage`                                                            | Class B: Segment Label               | Percentage density generation (25th).                                                      |
| **1020**        | `# CZ 50th percentile wage`                                                            | Class B: Segment Label               | Percentage density generation (median).                                                    |
| **1037**        | `#### Exhibit 10: DDD by Graph with predicted usage -----------------------`           | Class A: RStudio Section Divider     | **Sub-file Break**: Start of `03_diff_in_diff_regressions.R`.                              |
| **1085**        | `# index to 2012`                                                                      | Class B: Segment Label               | Visual scaling base manipulation.                                                          |
| **1137**        | `#### Exhibit 11: CZ x AEWR Region Deviation from Trend (Deciles) ## -------`          | Class A: RStudio Section Divider     | Analysis divider. Segment for decile bin construction and visualization.                   |
| **1214**        | `#### Exhibit 12: DD Visual - Indexed H-2A Use, Above vs Below Trend ## ---------`     | Class A: RStudio Section Divider     | Analysis divider. Evaluates empirical trajectories of treatment vs control.                |
| **1264**        | `## Exhibit 13: DD Main Results -------------------------------------------------`     | Class A: RStudio Section Divider     | Analysis divider. Segment running standard panel regressions via `fixest`.                 |
| **1278**        | `# DD model 1: no controls, all CZs`                                                   | Class B: Segment Label               | Estimation segment for Model 1 (uncontrolled base panel).                                  |
| **1322**        | `# DD model 2: with controls, all CZs`                                                 | Class B: Segment Label               | Estimation segment for Model 2 (controlled base panel).                                    |
| **1334**        | `# DD model 3: no controls, no border CZs`                                             | Class B: Segment Label               | Estimation segment for Model 3 (robustness: drops border units).                           |
| **1343**        | `# DD model 4: with controls, no border CZs`                                           | Class B: Segment Label               | Estimation segment for Model 4 (robustness: controlled clean borders).                     |
| **1355**        | `# DD model 5: robustness check by excluding potentially influential CZs ...`          | Class B: Segment Label               | Estimation segment for Model 5 (share distribution analysis).                              |
| **1356**        | `# How many total ag workers are there within each AEWR region?`                       | Class B: Explanatory Comment         | Explains computation of agricultural region scale weights.                                 |
| **1382**        | `# Pick arbitrary cutoff of 0.1`                                                       | Class B: Explanatory Comment         | Explains the trimming rules applied to dominant commuting zones.                           |
| **1397**        | `# What if we just dropped the largest CZ within each AEWR region?`                    | Class B: Segment Label               | Estimation segment for Model 6 (influence testing).                                        |
| **1411**        | `# Tables`                                                                             | Class B: Segment Label               | Generates formatted LaTeX output tables via `etable()`.                                    |
| **1445**        | `## Exhibit 14: Event Study (Flexible DD, Base Year = 2011) --------------------`      | Class A: RStudio Section Divider     | Analysis divider. Runs event study regression equations.                                   |
| **1474**        | `# Wald test on post-2011 interactions (joint significance)`                           | Class B: Segment Label               | Joint statistical significance execution block.                                            |
| **1523**        | `## Exhibit 15: Event Study Coefficient Plots -----------------------------------`     | Class A: RStudio Section Divider     | Plot divider. Extracts betas and plots event study dynamics.                               |
| **1605**        | `## Exhibit 16: Summary Statistics Table -----------------------------------------`    | Class A: RStudio Section Divider     | Analysis divider. Tabulates mean and standard deviation matrices.                          |
| **1649**        | `#### Exhibit 17: Fisher Price Index Time Series -----------------------------------`  | Class A: RStudio Section Divider     | Plot divider. Segment for tracing temporal shifts in output prices.                        |
| **1679**        | `#### Exhibit 18: DD Regressions — Fisher Price Index as Outcome ------------------`   | Class A: RStudio Section Divider     | Analysis divider. Runs prices as the endogenous outcome.                                   |
| **1723**        | `#### Exhibit 19: County Map — Fisher Price Index Ratio 2022 / 2008 ---------------`   | Class A: RStudio Section Divider     | Map divider. Geographic pricing ratio rendering.                                           |
| **1771**        | `#### Exhibit 20: Fisher Price Index Investigation — Labor Share DD --------------`    | Class A: RStudio Section Divider     | **Sub-file Break**: Start of `04_stacked_did_pipeline.R` (Advanced DiD).                   |
| **1772-1782**   | `# Fisher price index (fisher_index_ppi) construction: ...`                            | Class B: Explanatory Comment         | In-depth commentary explaining variable construction and limitations.                      |
| **1829**        | `#### Exhibit 21: Stacked Matched Staggered DiD — Treatment Classification ------`     | Class A: RStudio Section Divider     | Analysis divider. Treatment timeline classification block.                                 |
| **1831**        | `## Step 1: Sample and % change variable ----------------------------------------`     | Class A: RStudio Section Divider     | Processing segment. Calculates relative percent changes in Local Wage gaps.                |
| **1841**        | `## Step 2: Quartile thresholds from finite observations ------------------------`     | Class A: RStudio Section Divider     | Processing segment. Defines treatment threshold cutoffs (q25, q75).                        |
| **1851**        | `## Step 3: Tag large changes by county-year ------------------------------------`     | Class A: RStudio Section Divider     | Processing segment. Flags localized shock occurrences in the panel.                        |
| **1861**        | `## Step 4: County-level event classification -----------------------------------`     | Class A: RStudio Section Divider     | Processing segment. Determines the timing of first exposure.                               |
| **1908**        | `## Step 5: Subsequent change variables ----------------------------------------`      | Class A: RStudio Section Divider     | Processing segment. Handles multi-exposure tracking.                                       |
| **1957**        | `#### Exhibit 21b: Treatment status by year figure ------------------------------`     | Class A: RStudio Section Divider     | Analysis divider. Tracks distribution shifts across treatment cohorts.                     |
| **1958**        | `# Motivates the "not-yet-treated" control group strategy.`                            | Class B: Explanatory Comment         | Econometric defense for Stacked DiD design assumptions.                                    |
| **2040**        | `#### Exhibit 22: Correlation Matrix → Excel ------------------------------------`     | Class A: RStudio Section Divider     | Output divider. Generates pairwise correlations for covariates.                            |
| **2069**        | `#### Exhibit 23: Propensity Score Matching -------------------------------------`     | Class A: RStudio Section Divider     | Analysis divider. Conducts cross-sectional propensity matches.                             |
| **2070-2072**   | `# Review corr_matrix_matching_controls.xlsx before running...`                        | Class B: Explanatory Comment         | Alerts researchers to check multicollinearity before scaling matching steps.               |
| **2089**        | `# --- Increase group vs never-treated ---`                                            | Class B: Segment Label               | Segment running matching algorithms for positive shock cohorts.                            |
| **2106**        | `# --- Decrease group vs never-treated ---`                                            | Class B: Segment Label               | Segment running matching algorithms for wage-drop cohorts.                                 |
| **2125**        | `#### Exhibits 24-27: Stacked DiD Event Studies ---------------------------------`     | Class A: RStudio Section Divider     | Analysis divider. Estimates cohort-stacked models on matched panels.                       |
| **2126-2129**   | `# Specification: i(rel_year, treated, ref = -1)                                       | unit_cohort_id + year^cohort_id ...` | Class B: Explanatory Comment                                                               | Specifies exact high-dimensional fixed effects syntax. |
| **2191**        | `# Event study regressions — H-2A utilization`                                         | Class B: Segment Label               | Code block estimating program adoption outcomes.                                           |
| **2205**        | `# Event study regressions — Fisher price index`                                       | Class B: Segment Label               | Code block estimating price outcomes under stacked cohort conditions.                      |
| **2219**        | `# Coefficient plots`                                                                  | Class B: Segment Label               | Decoupled visualization generator step.                                                    |

---

### File 2: H2A Build Dataset (Phil Hoxie, 1/31/24)

This script structures data merges, constructs lagged inputs, generates scaled H-2A metrics, and applies deflation transforms using the Producer Price Index (WPU01).

| Relative Line # | Raw Indicator Text                                                                              | Syntactic Class                  | Purpose in Decoupling / Decoupled Destination File                                      |
| :-------------- | :---------------------------------------------------------------------------------------------- | :------------------------------- | :-------------------------------------------------------------------------------------- |
| **1**           | `## H2A Build Dataset`                                                                          | Class B: Script Header           | Title header for the primary analytical panel compiler.                                 |
| **2**           | `## Phil Hoxie`                                                                                 | Class B: Author Metadata         | Author metadata checkpoint.                                                             |
| **3**           | `## 1/31/24`                                                                                    | Class B: Date Metadata           | Version control temporal anchor.                                                        |
| **5**           | `## Run Master File First (or set paths/packages here if running standalone) ##`                | Class B: Execution Metadata      | Reinforces directory structure requirements.                                            |
| **7-14**        | `if (!exists("folder_data")) { ... }`                                                           | Class C: Execution Gate          | Configures local folder pointers.                                                       |
| **16**          | `## Yearly Full County Dataset ------------------------------------------------`                | Class A: RStudio Section Divider | **Sub-file Break**: Start of `01_load_inputs.R`. Loading intermediate parquet files.    |
| **18**          | `## ------- Full County ----------------------------------------------------------------------` | Class A: RStudio Section Divider | Segment separator for full county structure setup.                                      |
| **19**          | `## Yearly Dataset ## -----------------------------------------------------------`              | Class A: RStudio Section Divider | Segment separator.                                                                      |
| **20**          | `## -------- Full County -----------------------------------------------------------`           | Class A: RStudio Section Divider | Segment separator.                                                                      |
| **22**          | `## Load Data -------------------------------------------------------------------`              | Class A: RStudio Section Divider | Outlines the directory of intermediate target reads.                                    |
| **25**          | `# yearly versions #`                                                                           | Class B: Segment Label           | Marks the ingestion of cleaned annual indices.                                          |
| **53**          | `# base for full county dataset`                                                                | Class B: Segment Label           | Identifies primary panel spine (`county_df_year.parquet`).                              |
| **57**          | `# CZ`                                                                                          | Class B: Segment Label           | Marks the ingestion of Commuting Zone crosswalks.                                       |
| **61**          | `# merge in for each side:`                                                                     | Class B: Segment Label           | Notes the beginning of dataset merging operations.                                      |
| **63**          | `# by state and year`                                                                           | Class B: Segment Label           | Tracks state-year cross-sectional joins.                                                |
| **66**          | `# by state`                                                                                    | Class B: Segment Label           | Tracks time-invariant state parameters.                                                 |
| **69**          | `# merge for only one side`                                                                     | Class B: Segment Label           | Notes simple asymmetric joins.                                                          |
| **71**          | `# by`                                                                                          | Class B: Segment Label           | Section pointer.                                                                        |
| **74**          | `# loop ?`                                                                                      | Class B: Segment Label           | Marks obsolete code or placeholders in structural transitions.                          |
| **76**          | `# by county and census_period`                                                                 | Class B: Segment Label           | Marks multi-level period-specific aggregation joins.                                    |
| **80**          | `## a tad of prep ---------------------------------------------------------------`              | Class A: RStudio Section Divider | **Sub-file Break**: Start of `02_merge_and_deflate.R`. Prepares joining keys.           |
| **82**          | `# make state fips`                                                                             | Class B: Segment Label           | Key engineering (calculates state codes from FIPS).                                     |
| **86**          | `# it worked!`                                                                                  | Class B: Inline Comment          | Informal execution success note.                                                        |
| **88**          | `## both sides merge ----------------------------------------------------------`                | Class A: RStudio Section Divider | Primary dataset compilation block.                                                      |
| **90**          | `# fips first`                                                                                  | Class B: Segment Label           | Begins geography merges.                                                                |
| **103**         | `# only states`                                                                                 | Class B: Segment Label           | Drops non-state jurisdictions (e.g., US territories).                                   |
| **107**         | `# first side`                                                                                  | Class B: Segment Label           | Ingests basic wage data.                                                                |
| **118**         | `# DC missing`                                                                                  | Class B: Segment Label           | Tracks handling of missing data for Washington DC.                                      |
| **121**         | `# no need to rename these`                                                                     | Class B: Segment Label           | Data dictionary alignment check.                                                        |
| **142**         | `# county only #`                                                                               | Class B: Segment Label           | Begins merging BEA structural measures.                                                 |
| **171**         | `# add in minimum wages`                                                                        | Class B: Segment Label           | **Sub-file Break**: Start of `03_variable_engineering.R`. Incorporates state min wages. |
| **177**         | `# make a few lags`                                                                             | Class B: Segment Label           | Builds lagged variables for regulatory controls.                                        |
| **188**         | `# add in wage quantiles`                                                                       | Class B: Segment Label           | Integrates CZ wage distribution markers from ACS.                                       |
| **189**         | `# annual data starts 2005`                                                                     | Class B: Segment Label           | Clarifies time series range constraints.                                                |
| **196**         | `# Deflate wage percentiles to real 2012 terms using PPI WPU01 (rebased 2012=100)`              | Class B: Segment Label           | Crucial methodological shift: real wage conversions.                                    |
| **197**         | `# aewr_ppi is real; wage_p* must also be real before computing bite variables`                 | Class B: Segment Label           | Explicit check to ensure units are consistent before calculating indicators.            |
| **204**         | `# Add lags as in state minimum wages`                                                          | Class B: Segment Label           | Ensures temporal equivalence across wage variables.                                     |
| **218**         | `# deflate fisher price index to real 2012 terms`                                               | Class B: Segment Label           | Converted to match base wage structures.                                                |
| **219**         | `# ppi_2012 is already present in county_df via bea_cainc45_data_year merge`                    | Class B: Explanatory Comment     | Tracks variables loaded through prior steps.                                            |
| **223**         | `## Variable cleaning ## -----------------`                                                     | Class A: RStudio Section Divider | Begins dataset normalization and scaling.                                               |
| **225**         | `# AEWR vs ag min diff #`                                                                       | Class B: Segment Label           | Computes real gaps relative to minimum wage baselines.                                  |
| **228**         | `# new CZ-distribution-based bites (non-lagged):`                                               | Class B: Segment Label           | Calculates wage bites relative to CZ percentile thresholds.                             |
| **231**         | `# lagged CZ bites:`                                                                            | Class B: Segment Label           | Calculates historical bites.                                                            |
| **252**         | `# AEWR vs state min diff #`                                                                    | Class B: Segment Label           | Identifies the start of state-level min wage comparisons.                               |
| **256**         | `# H2A NAs to zero`                                                                             | Class B: Segment Label           | Imputes zero values for counties not using H-2A.                                        |
| **295**         | `# cropland zeros`                                                                              | Class B: Segment Label           | Sets null cropland metrics to zero.                                                     |
| **300**         | `# emp pop ratio`                                                                               | Class B: Segment Label           | Constructs labor supply control proxies.                                                |
| **304**         | `# logs of some vars`                                                                           | Class B: Segment Label           | Applies log transformations for elasticities.                                           |
| **338**         | `# budget shares`                                                                               | Class B: Segment Label           | Computes factor expense metrics.                                                        |
| **348**         | `# H2A outcome variables`                                                                       | Class B: Segment Label           | Normalizes H-2A usage metrics by total farm jobs.                                       |
| **361**         | `# add h2a apps per farm later`                                                                 | Class B: Roadmap Comment         | Legacy comment/roadmap planning marker.                                                 |
| **382**         | `# add in CZs`                                                                                  | Class B: Segment Label           | Incorporates structural geographic cluster markers.                                     |
| **392**         | `# fixed effects`                                                                               | Class B: Segment Label           | **Sub-file Break**: Start of `04_fixed_effects.R`. Creates regression fixed effects.    |
| **394**         | `# period`                                                                                      | Class B: Segment Label           | Formats calendar year indicators.                                                       |
| **394**         | `# treats counties in separate clusters as separate`                                            | Class B: Segment Label           | Explains variance groupings. Repeated for several levels of FE.                         |
| **398**         | `# county`                                                                                      | Class B: Segment Label           | Unique entity FE structure.                                                             |
| **402**         | `# state`                                                                                       | Class B: Segment Label           | State-level fixed effects.                                                              |
| **406**         | `# AEWR Region`                                                                                 | Class B: Segment Label           | Regional wage zone fixed effects.                                                       |
| **410**         | `# cZ`                                                                                          | Class B: Segment Label           | Commuting Zone fixed effects.                                                           |
| **414**         | `# CZ x time`                                                                                   | Class B: Segment Label           | Computes high-dimensional localized trend metrics.                                      |
| **422**         | `# first period is 0`                                                                           | Class B: Explanatory Comment     | Defines baseline periods for model identification.                                      |
| **424**         | `# aewr region  x time`                                                                         | Class B: Segment Label           | Sets regional trend metrics over time.                                                  |
| **436**         | `# CZ x AEWR region FE — each (CZ, AEWR region) pair is a distinct FE level.`                   | Class B: Segment Label           | Defines cross-border treatment definitions.                                             |
| **437-439**     | `# CZs that span AEWR region borders are split...`                                              | Class B: Explanatory Comment     | Explains how cross-border units are handled to prevent contamination.                   |
| **446**         | `# sample restriction to counties with cropland ------------------------------------`           | Class A: RStudio Section Divider | **Sub-file Break**: Start of `05_sample_filters.R`. Restricts sample boundaries.        |
| **448**         | `# we want to drop counties with no cropland in 2007`                                           | Class B: Explanatory Comment     | Baseline definition filter.                                                             |
| **456**         | `# remove HI, AK, and DC`                                                                       | Class B: Segment Label           | Excludes areas without AEWR coverage.                                                   |
| **461**         | `# check for NAs in key vars ------------------------------------`                              | Class A: RStudio Section Divider | Diagnostic block mapping missing data profiles.                                         |
| **463**         | `# use sum(is.na(x))`                                                                           | Class B: Inline Comment          | Notes the diagnostic method used.                                                       |
| **477**         | `# easiest / most important to fix`                                                             | Class B: Segment Label           | Identifies key missing data corrections.                                                |
| **481**         | `# BEA issue for VA counties. We need to drop bristol city. It will be ok.`                     | Class B: Segment Label           | Documents a manual geographical trim for Virginia independent cities.                   |
| **484**         | `## lags of h2a variables`                                                                      | Class A: RStudio Section Divider | Implements dynamic panel outcomes.                                                      |
| **601**         | `# new variables`                                                                               | Class B: Segment Label           | Additional transformation steps.                                                        |
| **605**         | `# Cut using pred X actual use rates in 2008`                                                   | Class B: Explanatory Comment     | Sets up treatment groups (Always Takers, Adopters, Defiers, Never Takers).              |
| **627**         | `# cuts by 2008 h2a usage`                                                                      | Class B: Segment Label           | Sets up descriptive usage splits.                                                       |
| **638**         | `# cut by count`                                                                                | Class B: Segment Label           | Sets up quantile splits by worker count.                                                |
| **645**         | `# cut by share`                                                                                | Class B: Segment Label           | Sets up quantile splits by employment share.                                            |
| **695**         | `# year dummys`                                                                                 | Class B: Segment Label           | Converts temporal periods to explicit flags.                                            |
| **715**         | `# ID border CZs`                                                                               | Class B: Segment Label           | Flags cross-border regional units to support policy spillover tests.                    |
| **737**         | `# pre post dummy`                                                                              | Class B: Segment Label           | Renders the primary timing switch (post-2011).                                          |
| **741**         | `# low use dummy`                                                                               | Class B: Segment Label           | Sets up inverse indicators for control cohorts.                                         |
| **754**         | `# --- Diagnostic: county_df_analysis_year ---`                                                 | Class B: Execution Gate          | Script assert checkpoint (e.g., verifying `nrow(county_df) > 10000`).                   |
| **768**         | `## Remove files ## -------------------------------------------------------------`              | Class C: Clean-up Block          | Empties memory to ensure subsequent runs start clean.                                   |

---

### File 3: H2A: Load and clean datasets (Phil Hoxie, 1/11/24)

This script contains the primary data loading, parsing, and cleaning (ETL) routines. It ingests raw files and standardizes county, state, and regional indicators.

| Relative Line # | Raw Indicator Text                                                                 | Syntactic Class                  | Purpose in Decoupling / Decoupled Destination File                                            |
| :-------------- | :--------------------------------------------------------------------------------- | :------------------------------- | :-------------------------------------------------------------------------------------------- |
| **1**           | `## Run standalone or via H2A Master.R`                                            | Class B: Script Header           | Tracks execution dependencies.                                                                |
| **9**           | `## H2A: Load and clean datasets`                                                  | Class B: Script Header           | Sub-title indicating raw ETL module scope.                                                    |
| **10**          | `## Phil Hoxie`                                                                    | Class B: Author Metadata         | Author metadata checkpoint.                                                                   |
| **11**          | `## 1/11/24`                                                                       | Class B: Date Metadata           | Version control temporal anchor.                                                              |
| **13**          | `## New Price Data (20260417) ----------------------`                              | Class A: RStudio Section Divider | **Sub-file Break**: Start of `etl_price_index.R`. _Crucial timestamp reference (April 2026)._ |
| **23**          | `## requires running master file first ##`                                         | Class B: Execution Metadata      | Reinforces directory structure requirements.                                                  |
| **25**          | `## Two datasets, one by census period and one by year (calendar year)`            | Class B: Explanatory Comment     | Methodological split in the ETL pipeline: 5-year Census vs. Annual Panels.                    |
| **27**          | `## CZ wage quantile ------------------------------------`                         | Class A: RStudio Section Divider | **Sub-file Break**: Start of `etl_commuting_zones.R`.                                         |
| **33**          | `## state min wages ------------------------------------`                          | Class A: RStudio Section Divider | **Sub-file Break**: Start of `etl_minimum_wages.R`. Ingests FRED min wage files.              |
| **39**          | `## Alt Min Wage from Ken -------------------------------`                         | Class A: RStudio Section Divider | Integrates agricultural state-level exemption indices.                                        |
| **45**          | `## Fips Codes ------------------------------------------`                         | Class A: RStudio Section Divider | Ingests structural FIPS crosswalk files.                                                      |
| **51**          | `## AEWR Regions ----------------------------------------`                         | Class A: RStudio Section Divider | **Sub-file Break**: Start of `etl_aewr.R`. Maps state-to-region configurations.               |
| **57**          | `## Commuting Zones --------------------------------------`                        | Class A: RStudio Section Divider | Integrates Penn State commuting zone files.                                                   |
| **60**          | `# source: https://sites.psu.edu/psucz/data/`                                      | Class B: Explanatory Comment     | Tracks primary data source URL.                                                               |
| **71**          | `## PPI --------------------------------------------------`                        | Class A: RStudio Section Divider | **Sub-file Break**: Start of `etl_ppi.R`. Ingests FRED BLS WPU01 PPI series.                  |
| **72**          | `# Source: https://fred.stlouisfed.org/series/WPU01`                               | Class B: Explanatory Comment     | Tracks primary data source URL.                                                               |
| **78**          | `# average by year`                                                                | Class B: Segment Label           | Aggregates monthly PPI measures to annual averages.                                           |
| **83**          | `# change base to 2012`                                                            | Class B: Segment Label           | Methodological adjustment: shifts PPI reference year.                                         |
| **95**          | `## County Boundary ------------------------------------`                          | Class A: RStudio Section Divider | Ingests adjacency databases.                                                                  |
| **98**          | `# cleaned in the original stata file`                                             | Class B: Explanatory Comment     | Identifies dependencies on legacy Stata files.                                                |
| **106**         | `# state minimum wages --------------------------------------`                     | Class B: Segment Label           | Cleans state minimum wage variables.                                                          |
| **109**         | `# deflate by ppi`                                                                 | Class B: Segment Label           | Converts minimum wage variables to real 2012 values.                                          |
| **111**         | `# want the dummy from here`                                                       | Class B: Segment Label           | Extracts agriculture exemption status variables.                                              |
| **112**         | `# these are stable, so ignore them`                                               | Class B: Explanatory Comment     | Cleans up time-invariant parameters.                                                          |
| **124**         | `# make the prevailing ag min wage #`                                              | Class B: Segment Label           | Builds effective state agricultural minimums.                                                 |
| **126**         | `# fill in state min with federal if missing #`                                    | Class B: Explanatory Comment     | Imputes standard federal minimum values.                                                      |
| **152**         | `## H2A Data -------------------------------------------`                          | Class A: RStudio Section Divider | **Sub-file Break**: Start of `etl_h2a.R`. Processes DOL H-2A intermediate data.               |
| **157**         | `# census period`                                                                  | Class B: Segment Label           | Aggregates annual records into 5-year Census intervals.                                       |
| **171**         | `# fix fips code`                                                                  | Class B: Segment Label           | Normalizes string lengths for county FIPS joins.                                              |
| **190**         | `# clean by dropping old codes #`                                                  | Class B: Explanatory Comment     | Excludes legacy structural geocodes.                                                          |
| **199**         | `# collapse by period, county (county and state fips)`                             | Class B: Explanatory Comment     | Performs database consolidation operations.                                                   |
| **243**         | `# yearly, for TS`                                                                 | Class B: Segment Label           | Formats the annual time series dataset.                                                       |
| **266**         | `## Cropland Crocs Data Layer (CDL) -------------------------------------------`   | Class A: RStudio Section Divider | **Sub-file Break**: Start of `etl_census_ag.R`. Normalizes CDL files.                         |
| **274**         | `# need to put this in a wider format`                                             | Class B: Explanatory Comment     | Converts long acreage records to wide tables.                                                 |
| **314**         | `## NAWSPAD ---------------------------------------------`                         | Class A: RStudio Section Divider | Ingests intermediate agricultural surveys.                                                    |
| **315**         | `# state level data`                                                               | Class B: Explanatory Comment     | Documents the geographic resolution of the files.                                             |
| **321**         | `## OEWS ------------------------------------------------`                         | Class A: RStudio Section Divider | Ingests intermediate BLS wage distributions.                                                  |
| **322**         | `# wages`                                                                          | Class B: Segment Label           | Marks wage parameter definitions.                                                             |
| **327**         | `## _____________________________________________________________________________` | Class A: Structural Divider      | Marks transition from census-period files to annual panels.                                   |
| **328**         | `## Year by Year version --------------------------------------------------------` | Class A: RStudio Section Divider | Outlines the start of annual panel ETL files.                                                 |
| **329**         | `## _____________________________________________________________________________` | Class A: Structural Divider      | Visual separator.                                                                             |
| **331**         | `## Fips Codes ------------------------------------------`                         | Class A: RStudio Section Divider | Loads annual FIPS keys.                                                                       |
| **337**         | `## AEWR Regions ----------------------------------------`                         | Class A: RStudio Section Divider | Loads annual wage boundaries.                                                                 |
| **343**         | `## PPI --------------------------------------------------`                        | Class A: RStudio Section Divider | Loads annual Producer Price Index variables.                                                  |
| **345**         | `## County Boundary ------------------------------------`                          | Class A: RStudio Section Divider | Loads annual geographical boundary files.                                                     |
| **359**         | `## Census of Agriculture ------------------------------`                          | Class A: RStudio Section Divider | **Sub-file Break**: Start of `etl_census_ag.R` (Annual component).                            |
| **365**         | `# general cleaning #`                                                             | Class B: Segment Label           | Normalizes Census data variables.                                                             |
| **369**         | `# want: FARM OPERATIONS, AG LAND`                                                 | Class B: Segment Label           | Filters records down to targeted descriptors.                                                 |
| **378**         | `# fix fips`                                                                       | Class B: Segment Label           | Normalizes spatial join identifiers.                                                          |
| **402**         | `## H2A Data -------------------------------------------`                          | Class A: RStudio Section Divider | Integrates annual program metrics.                                                            |
| **404**         | `# fix fips code`                                                                  | Class B: Segment Label           | Formats FIPS characters.                                                                      |
| **423**         | `# clean by dropping old codes #`                                                  | Class B: Segment Label           | Drops deprecated county codes.                                                                |
| **432**         | `# collapse by period, county (county and state fips)`                             | Class B: Segment Label           | Consolidates yearly program metrics.                                                          |
| **436**         | `## AEWR -------------------------------------------------`                        | Class A: RStudio Section Divider | Normalizes state and regional AEWR schedules.                                                 |
| **453**         | `# pre 1995 source (CRS): https://www.everycrsreport.com/files/...`                | Class B: Explanatory Comment     | Explains where historical AEWR figures were sourced.                                          |
| **475**         | `# append dataseries`                                                              | Class B: Segment Label           | Concatenates historical and modern wage panels.                                               |
| **496**         | `# collapse by regions`                                                            | Class B: Segment Label           | Aggregates wage metrics by AEWR region boundaries.                                            |
| **502**         | `# make TS variables #`                                                            | Class B: Segment Label           | Generates wage variables for detrending operations.                                           |
| **520**         | `## AEWR Region TS --------------------------------------------------------------` | Class A: RStudio Section Divider | Diagnostic block generating regional check plots.                                             |
| **671**         | `## BEA Data ---------------------------------------------`                        | Class A: RStudio Section Divider | **Sub-file Break**: Start of `etl_bea_employment.R` / `etl_bea_finance.R`.                    |
| **673**         | `# job count data`                                                                 | Class B: Segment Label           | Loads CAEMP25N sectoral employment datasets.                                                  |
| **680**         | `# save lines: 10 50 70 80 90`                                                     | Class B: Segment Label           | Documents the raw row-filtering rules applied.                                                |
| **699**         | `# put into year - county rows, so, pivot again`                                   | Class B: Segment Label           | Pivots long survey metrics to analytical rows.                                                |
| **710**         | `# fix fips`                                                                       | Class B: Segment Label           | Normalizes independent spatial codes.                                                         |
| **713**         | `# keep counties when conflict`                                                    | Class B: Segment Label           | Implements resolution rules for conflicting geographies.                                      |
| **737**         | `# SD Oglala Lakota to Shannon`                                                    | Class B: Explanatory Comment     | Handles a specific South Dakota county code update (FIPS 46102 -> 46113).                     |
| **752**         | `# farm finance`                                                                   | Class B: Segment Label           | Loads agricultural output valuations (CAINC45).                                               |
| **753**         | `# save lines: 20 60 130 210 270 150`                                              | Class B: Segment Label           | Documents raw row-filtering rules for financial indicators.                                   |
| **788**         | `# real`                                                                           | Class B: Segment Label           | Applies output price transformations.                                                         |
| **815**         | `## County Pop estimates ---------------------------------`                        | Class A: RStudio Section Divider | **Sub-file Break**: Start of `etl_population.R`. Ingests population figures.                  |
| **816**         | `# documentation: https://www2.census.gov/programs-surveys/popest/...`             | Class B: Explanatory Comment     | Tracks source layout documentation.                                                           |
| **836**         | `# fix CT county to region nonsense.`                                              | Class B: Segment Label           | Handles the transition of Connecticut's planning regions to county equivalents.               |
| **837**         | `# project using state growth rates`                                               | Class B: Explanatory Comment     | Explains the projection method used for CT population gaps.                                   |
| **853**         | `# fix CT`                                                                         | Class B: Segment Label           | Implements projection overrides for Connecticut.                                              |
| **871**         | `# fix SD county change`                                                           | Class B: Segment Label           | Updates Shannon County, SD population pointers.                                               |
| **883**         | `## all county sample ----------------------------------------------------------`  | Class A: RStudio Section Divider | Compiles the time series panel master matrix (`county_df_year.parquet`).                      |
| **907**         | `# remove files -------------------`                                               | Class C: Clean-up Block          | Runs environment cleanup routines.                                                            |

---

## 4. Directed Acyclic Graph (DAG) Pipeline & Dependency Map

To separate these combined scripts successfully, we must document how the output of one script becomes the input for the next. This ensures no scripts are run out of order, preventing execution failures.

```
                  Auxiliary Data Extraction & Cleaning (File 3)
     ┌──────────────────────┬───────────────────────┼──────────────────────┐
     ▼                      ▼                       ▼                      ▼
etl_ppi.R             etl_aewr.R             etl_commuting_zones.R   etl_bea_finance.R
  │                         │                       │                      │
  ▼                         ▼                       ▼                      ▼
 [ppi_2012.parquet]  [aewr_data_year.parquet] [cz_file_2010.parquet] [bea_cainc45_data_year]
     │                      │                       │                      │
     └──────────────────────┼───────────┬───────────┴──────────────────────┘
                            ▼           ▼
                         Dataset Assembly & Normalization (File 2)
                                        │
                                        ▼
                         [county_df_analysis_year.parquet]
                                        │
                                        ▼
                         Econometric Output Generator (File 1)
                                        │
                      ┌─────────────────┴─────────────────┐
                      ▼                                   ▼
          [LaTeX Tables (.tex)]                 [Visualizations (.png)]
```

### Data Pipeline Contracts

| Outputting Source File / Module | Intermediate Produced File Path            | Primary Consuming File / Module                             |
| :------------------------------ | :----------------------------------------- | :---------------------------------------------------------- |
| **File 3 ETL**                  | `Data Int/nass_fisher_price_index.parquet` | **File 2 Assembly** (loads crop index trackers)             |
| **File 3 ETL**                  | `Data Int/ppi_2012.parquet`                | **File 2 Assembly** (deflator for monetary values)          |
| **File 3 ETL**                  | `Data Int/state_real_minwages.parquet`     | **File 2 Assembly** (baseline agricultural wage floor)      |
| **File 3 ETL**                  | `Data Int/h2a_data.parquet`                | **File 2 Assembly** (H-2A counts by census period)          |
| **File 3 ETL**                  | `Data Int/h2a_data_ts.parquet`             | **File 1 Visualization** (time series lines)                |
| **File 3 ETL**                  | `Data Int/cz_file_2010_small.parquet`      | **File 2 Assembly** (commuting zone keys)                   |
| **File 3 ETL**                  | `Data Int/census_ag_cropland_year.parquet` | **File 2 Assembly** (annual cropland acreage checks)        |
| **File 3 ETL**                  | `Data Int/census_pop_ests_year.parquet`    | **File 2 Assembly** (demographic controls)                  |
| **File 3 ETL**                  | `Data Int/bea_caemp25n_data_year.parquet`  | **File 2 Assembly** (sectoral job counts)                   |
| **File 3 ETL**                  | `Data Int/bea_cainc45_data_year.parquet`   | **File 2 Assembly** (cash receipts and production expenses) |
| **File 3 ETL**                  | `Data Int/aewr_data_year.parquet`          | **File 2 Assembly** (annual regional AEWR schedules)        |
| **File 3 ETL**                  | `Data Int/county_df_year.parquet`          | **File 2 Assembly** (time series panel spine)               |
| **File 2 Assembly**             | `Data Int/county_df_analysis_year.parquet` | **File 1 Visualization** (regression panel engine)          |

---

## 5. Blueprint for Modular Project Reconstruction

To decouple these scripts cleanly into a professional analytical workflow, we should adopt a modular structure. Rather than maintaining one monolithic file, we will separate the code into specialized ETL, assembly, and analysis sub-scripts, coordinated by a master runner file.

### Proposed Folder Structure

```
H-2A-Paper-Workspace/
│
├── H2A_Master.R                           # Global project controller & environment setup
├── config/
│   └── global_parameters.R                # Path declarations and shared helper variables
│
├── code/
│   ├── 01_ETL/
│   │   ├── etl_ppi.R                      # Cleans Producer Price Index series
│   │   ├── etl_aewr.R                     # Prepares DOL AEWR regions & historic panel
│   │   ├── etl_commuting_zones.R          # Cleans Penn State Commuting Zones maps
│   │   ├── etl_minimum_wages.R            # Processes FRED & Ken state wage structures
│   │   ├── etl_h2a.R                      # Aggregates DOL H-2A case records
│   │   ├── etl_census_ag.R                # Parses USDA NASS cropland & census data
│   │   ├── etl_bea_employment.R           # Restructures BEA CAEMP25N employment data
│   │   ├── etl_bea_finance.R              # Restructures BEA CAINC45 financial indicators
│   │   └── etl_population.R               # Resolves historical county population models
│   │
│   ├── 02_Assembly/
│   │   ├── load_intermediates.R           # Ingests intermediate ETL parquet files
│   │   ├── merge_and_deflate.R            # Merges panels and applies price index deflations
│   │   ├── variable_engineering.R         # Calculates wage gaps, lags, and budget shares
│   │   ├── fixed_effects_setup.R          # Computes multi-level interacted fixed effects
│   │   └── export_analysis_panel.R        # Applies filters and exports the analysis dataset
│   │
│   └── 03_Analysis/
│       ├── run_descriptive_plots.R        # Generates basic trend lines & distribution density plots
│       ├── run_spatial_mapping.R          # Generates shapefile simplify & GIS mapping plots
│       ├── run_panel_regressions.R        # Standard DiD regressions & summary statistics
│       └── run_stacked_matched_did.R      # Matched cohort-stacked event study models
│
├── Data Raw/                              # Raw input files (FRED CSVs, BEA CSVs, shapefiles)
├── Data Int/                              # Intermediate processed parquet databases
└── Output/                                # Generated LaTeX Tables (.tex) and Plots (.png)
```

---

### Step-by-Step Decoupling Execution Guide

#### Step 1: Establish the Global project controller & environment setup (`H2A_Master.R`)

Create a master controller script at the project root to manage environment paths and library loads. This keeps package calls and working directories consistent across all sub-scripts.

```R
# ==============================================================================
# H2A_Master.R
# Core master controller to run the entire H-2A paper data & modeling pipeline
# Author: Phil Hoxie (Refactored to Production DAG Model)
# ==============================================================================

# Clear workspace
rm(list = ls(all.names = TRUE))
gc()

# Define project-wide directories
folder_dir    <- "C:/Users/Phil/Dropbox/H-2A Paper/"
folder_code   <- paste0(folder_dir, "code/")
folder_data   <- paste0(folder_dir, "Data Int/")
folder_output <- paste0(folder_dir, "Output/")

# Shared execution parameters
set.seed(12345)
global_run_date <- format(Sys.Date(), "%Y%m%d")

# Unified package initialization
required_packages <- c("sf", "tidyverse", "ggspatial", "scales", "cowplot",
                       "ggthemes", "fixest", "ggfixest", "MatchIt", "writexl",
                       "arrow", "tidylog")

new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

lapply(required_packages, library, character.only = TRUE)

# Pipeline execution switches
run_etl      <- TRUE
run_assembly <- TRUE
run_analysis <- TRUE

# ------------------------------------------------------------------------------
# 1. Pipeline Execution Gate: ETL Processes (File 3)
# ------------------------------------------------------------------------------
if (run_etl) {
  message("=== Executing ETL Pipeline Tasks ===")
  source(paste0(folder_code, "01_ETL/etl_ppi.R"))
  source(paste0(folder_code, "01_ETL/etl_aewr.R"))
  source(paste0(folder_code, "01_ETL/etl_commuting_zones.R"))
  source(paste0(folder_code, "01_ETL/etl_minimum_wages.R"))
  source(paste0(folder_code, "01_ETL/etl_h2a.R"))
  source(paste0(folder_code, "01_ETL/etl_census_ag.R"))
  source(paste0(folder_code, "01_ETL/etl_bea_employment.R"))
  source(paste0(folder_code, "01_ETL/etl_bea_finance.R"))
  source(paste0(folder_code, "01_ETL/etl_population.R"))
}

# ------------------------------------------------------------------------------
# 2. Pipeline Execution Gate: Assembly & Normalization (File 2)
# ------------------------------------------------------------------------------
if (run_assembly) {
  message("=== Executing Panel Dataset Assembly Tasks ===")
  source(paste0(folder_code, "02_Assembly/load_intermediates.R"))
  source(paste0(folder_code, "02_Assembly/merge_and_deflate.R"))
  source(paste0(folder_code, "02_Assembly/variable_engineering.R"))
  source(paste0(folder_code, "02_Assembly/fixed_effects_setup.R"))
  source(paste0(folder_code, "02_Assembly/export_analysis_panel.R"))
}

# ------------------------------------------------------------------------------
# 3. Pipeline Execution Gate: Analysis & Visualizations (File 1)
# ------------------------------------------------------------------------------
if (run_analysis) {
  message("=== Generating Econometric Results & Output Figures ===")
  source(paste0(folder_code, "03_Analysis/run_descriptive_plots.R"))
  source(paste0(folder_code, "03_Analysis/run_spatial_mapping.R"))
  source(paste0(folder_code, "03_Analysis/run_panel_regressions.R"))
  source(paste0(folder_code, "03_Analysis/run_stacked_matched_did.R"))
}

message("=== Pipeline successfully executed ===")
```

#### Step 2: Extract individual ETL files (`code/01_ETL/`)

Each dataset load from File 3 should be split into its own clean file inside the `01_ETL/` folder. This isolate failures, meaning if the USDA NASS server structure or a local BEA file name changes, only that single loader file breaks, keeping the rest of the workspace functional.

Example decoupling for **Producer Price Index Ingestion** (`etl_ppi.R`):

```R
# ==============================================================================
# code/01_ETL/etl_ppi.R
# Parses raw BLS WPU01 PPI series and constructs a 2012-based deflator
# ==============================================================================

message("--> Processing Producer Price Index (WPU01)")

ppi_raw <- read.csv(
  file = paste0(folder_dir, "Data Raw/WPU01.csv"),
  stringsAsFactors = FALSE
)

# Extract year and compute annual averages
ppi_annual <- ppi_raw %>%
  mutate(year = as.numeric(substr(observation_date, 1, 4))) %>%
  group_by(year) %>%
  summarise(wpu01_mean = mean(WPU01, na.rm = TRUE))

# Change base year to 2012
ppi_2012_val <- ppi_annual$wpu01_mean[ppi_annual$year == 2012]

ppi_processed <- ppi_annual %>%
  mutate(ppi_2012 = wpu01_mean / ppi_2012_val) %>%
  select(year, ppi_2012)

# Save to the intermediate data folder
write_parquet(
  ppi_processed,
  paste0(folder_data, "ppi_2012.parquet")
)

# Clean up variables to prevent environment contamination
rm(ppi_raw, ppi_annual, ppi_processed, ppi_2012_val)
```

#### Step 3: Extract the Assembly Pipeline (`code/02_Assembly/`)

Break down the complex joins, deflations, and lag creations from File 2 into sequentially sourced R files:

- **`load_intermediates.R`**: Reads the parquet outputs written by your ETL step.
- **`merge_and_deflate.R`**: Merges files sequentially using clear geographic keys (`countyfips`, `statefips`, `year`) and applies real wage deflators.
- **`variable_engineering.R`**: Runs calculations for wage gaps, lags, and employment shares.
- **`fixed_effects_setup.R`**: Prepares interacted terms (e.g., `cztime_fe` and `cz_aewr_region_fe`).
- **`export_analysis_panel.R`**: Applies final sample restrictions (e.g., dropping Alaska, Hawaii, and DC, and filtering for counties with active cropland in 2007), checks for null values, and saves the final regression-ready panel:
  ```R
  write_parquet(county_df, paste0(folder_data, "county_df_analysis_year.parquet"))
  ```

#### Step 4: Extract Analysis & Visualizations (`code/03_Analysis/`)

Isolate analytical logic from data transformations. This allows you to quickly tweak plot labels, test alternative colors, or add covariates to regressions without having to rebuild the dataset on every run.

- **`run_descriptive_plots.R`**: Focuses on generating wage distributions (Exhibits 0, 9) and time-series trends (Exhibits 1, 2, 1C, 3).
- **`run_spatial_mapping.R`**: Loads shapefiles (`sf`), runs `st_simplify` to keep rendering fast, and maps trends and exposures geographically (Exhibits 4, 5, 6, 6B, 7, 19).
- **`run_panel_regressions.R`**: Estimates DiD models using `fixest::feols`, runs summary statistics, and saves LaTeX-formatted tables to your outputs (Exhibits 13, 14, 15, 16, 18, 20).
- **`run_stacked_matched_did.R`**: Isolates stacked-DiD logic. It handles treatment tagging (Steps 1–5), propensity score matches, cohort-stacking joins, and plots event study results (Exhibits 21, 21b, 22, 23, 24–27).

---

## 6. Project Interoperability & System Architecture Overview

When separating monolithic scripts into a modular workflow, the design must protect pipeline integrity and ensure consistency. The structural specifications below detail how to coordinate system inputs, execute transitions cleanly, and handle pathing.

```
       [Raw External CSV & Shapefiles]
                      │
                      ▼ (Step 1: Modular Ingestion)
          code/01_ETL/etl_*.R Scripts
                      │
                      ▼ (Step 2: Save Standardized parquets)
          Data Int/ Intermediate Datasets
                      │
                      ▼ (Step 3: Sequential Assembly)
     code/02_Assembly/merge_and_deflate.R, etc.
                      │
                      ▼ (Step 4: Save Final Panel)
     Data Int/county_df_analysis_year.parquet
                      │
                      ▼ (Step 5: Run Results & Visuals)
        code/03_Analysis/run_*.R Scripts
                      │
                      ▼ (Step 6: Write Output Artifacts)
              Output/ Plots and Tables
```

### Path & Directory Separation

Instead of hardcoding absolute folders in multiple files, use a unified design. All paths should be set in a single, centralized controller (like `H2A_Master.R` or a `config/global_parameters.R` file).

This allows you to migrate the entire project between a local computer, Dropbox, or a remote server by updating just one file:

- **`folder_dir`**: The root project directory.
- **`folder_data`**: Puts intermediate standardized files in a structured data directory (`Data Int/`).
- **`folder_output`**: Routes figures, maps, and LaTeX tables to a clean output folder (`Output/`).

### Global State & Execution Isolation

To prevent variables from leaking between scripts and causing unintended side effects, apply strict isolation policies:

- Run R's garbage collector (`gc()`) and clear the environment (`rm()`) at the end of each script to remove temporary dataframes from memory.
- Use distinct, standardized prefixes for intermediate variables in each sub-script.
- Define explicit function scopes. For helper tasks like generating coefficient plots, wrap the logic in a function (like `make_coefplot <- function(model) { ... }`) to keep plot-specific variables out of the global environment.
- Standardize stochastic steps, such as propensity score matching, by setting random seeds (`set.seed(12345)`) directly before the estimation steps.
-->
