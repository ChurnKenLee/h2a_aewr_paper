# Process Cleaning: Removing Unused Code Blocks

**Date:** 2026-05-05
**Scope:** Remove code blocks in `Do/H2A Build Dataset.R` and `Do/H2A Clean and Load.R` that produce datasets not consumed by `Do/H2A Analysis Figs and Tables.R`.

---

## Background Findings

### Analysis loads objects it never uses

| Variable | File | Verdict |
|----------|------|---------|
| `aewr_df` | `aewr.parquet` | Loaded at line 34, never referenced again |
| `border_df` | `border_df_analysis.parquet` | Loaded at line 36, never referenced again |
| `border_df_yearly` | `border_df_analysis_year.parquet` | Loaded at lines 41–43, never referenced again |
| `fips_codes` | `fips_codes.csv` | Loaded at line 32, never referenced again |

### Build Dataset.R has three structurally independent sections

| Lines | Purpose | Output | Used in Analysis? |
|-------|---------|--------|------------------|
| 1–626 | Census-period border pairs | `border_df_analysis.parquet` | **No** |
| 640–1412 | Annual border pairs | `border_df_analysis_year.parquet` | **No** |
| 1425–2365 | Annual county panel | `county_df_analysis_year.parquet` | **Yes** |

Each section clears its own workspace before the next begins. Section 3 re-loads everything it needs independently. Sections 1 and 2 can be deleted without affecting section 3.

Section 3 also contains dead reads: it loads `border_df_year.parquet`, `state_border_pairs.csv`, and `border_counties_allmatches.csv` (lines 1447–1465), but only calls `head()` on them — none of these feed any computation in `county_df`.

### Clean and Load.R has two parallel tracks

- **Census-period track** (lines 1–1215): feeds into Build Dataset sections 1 & 2
- **Year-by-year track** (lines 1216–2308): feeds into Build Dataset section 3

Once Build Dataset sections 1 & 2 are removed, several census-period outputs in Clean and Load lose all downstream consumers.

---

## Step-by-Step Plan

### Step 1 — Clean Analysis Figs and Tables.R (4 dead reads)

Remove the four `read_parquet` / `read_csv` lines at the top of the script:

```r
fips_codes <- read_csv(...)           # line 32
aewr_df    <- read_parquet(...)       # line 34
border_df  <- read_parquet(...)       # line 36
border_df_yearly <- read_parquet(...) # lines 41–43
```

**Risk:** None. These objects are assigned but never referenced downstream.

---

### Step 2 — Remove Build Dataset.R sections 1 and 2 (~1,412 lines)

Delete lines 1–1412 in their entirety:
- Lines 1–626: Census-period border pair construction → `border_df_analysis.parquet`
- Lines 628–639: Workspace cleanup between sections
- Lines 640–1412: Annual border pair construction → `border_df_analysis_year.parquet`

The script after deletion begins at the existing line 1425 header:
```
## Yearly Full County Dataset
```

Renumber accordingly (or simply let the file start there).

**Risk:** Low. Section 3 is fully self-contained and re-declares all its own inputs.

---

### Step 3 — Clean section 3's dead reads in Build Dataset.R (~8 lines)

In section 3's load block, remove these reads and their associated `head()` diagnostic calls:

```r
border_df              <- read_parquet(paste0(folder_data, "border_df_year.parquet"))  # line 1447
state_border_pairs     <- read.csv(paste0(folder_data, "state_border_pairs.csv"), ...)  # lines 1460–1462
border_counties_allmatches <- read.csv(paste0(folder_data, "border_counties_allmatches.csv"), ...) # lines 1464–1465

head(border_df)                  # line 1503
head(state_border_pairs)         # line 1504
head(border_counties_allmatches) # line 1505
```

**Risk:** None. None of these are used in any subsequent computation in section 3.

---

### Step 4 — Audit and remove dead blocks in Clean and Load.R

After steps 2–3, the following census-period outputs lose all downstream consumers and their producing code can be removed:

| Block | Lines (approx.) | Output file | Why removable |
|-------|-----------------|-------------|---------------|
| Census of Agriculture (census period) | ~197–270 | `census_ag_cropland.parquet`, `census_ag_cropland_2007.parquet` | Only read by Build Dataset section 1 |
| ACS & QCEW (census period) | ~476–548 | `acs_qcew_data.parquet` | Only read by Build Dataset section 1 — **verify first** |
| BEA Employment (census period) | ~624–800 | `bea_caemp25n_data.parquet` | Only read by Build Dataset section 1 |
| BEA Farm Income (census period) | ~800–930 | `bea_cainc45_data.parquet` | Only read by Build Dataset section 1 |
| ACS Immigration imputation (census period) | ~933–1042 | `acs_immigrant_cleaned.parquet` | Only read by Build Dataset section 1 |
| Census pop estimates (census period) | ~1043–1134 | `census_pop_ests.parquet` | Only read by Build Dataset section 1 |
| Border pair data (census period) | ~1135–1215 | `border_df.parquet` | Only read by Build Dataset section 1 |

**One pre-check required:** Confirm `acs_qcew_data.parquet` is not read by Build Dataset section 3 before removing it from Clean and Load.

#### Census-period blocks that must be kept

These are still consumed by Analysis directly or by Build Dataset section 3:

| Block | Output | Consumed by |
|-------|--------|-------------|
| Price data (top of script) | `nass_fisher_price_index.parquet` | Build Dataset section 3 |
| CZ file | `cz_file_2010.parquet`, `cz_file_2010_small.parquet` | Build Dataset section 3 |
| PPI | `ppi_2012.parquet` | Build Dataset section 3 |
| Min wages | `state_real_minwages.parquet` | Build Dataset section 3 |
| H-2A prediction | `h2a_predict.parquet` | Build Dataset section 3 |
| H-2A data | `h2a_data.parquet`, `h2a_data_ts.parquet` | Analysis directly |
| AEWR | `aewr_data.parquet` | Analysis directly |

---

### Step 5 — Verify and run

After each step, use the R skill to smoke-test:

1. Run Build Dataset section 3 standalone and confirm `county_df_analysis_year.parquet` writes with the expected column count.
2. Run Analysis through Exhibit 13 (the first regression) to confirm `samp_base` is non-empty.
3. Spot-check key columns: `aewr_cz_p25_l1`, `county_simple_treatment_groups`, `cz_aewr_region_fe`, `border_cz`.

---

## Summary of Expected Line Reduction

| Script | Estimated lines removed |
|--------|------------------------|
| `H2A Analysis Figs and Tables.R` | ~5 lines |
| `H2A Build Dataset.R` | ~1,415 lines (sections 1 & 2 + dead reads) |
| `H2A Clean and Load.R` | ~600–700 lines (census-period blocks no longer feeding downstream) |

Total estimated reduction: **~2,100 lines** across the three scripts.
