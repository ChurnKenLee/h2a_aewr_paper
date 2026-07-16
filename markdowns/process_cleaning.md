# Process Cleaning: Removing Unused Code Blocks

**Date:** 2026-05-05
**Completed:** 2026-05-07
**Status: COMPLETE**
**Scope:** Remove code blocks in `Do/H2A Build Dataset.R` and `Do/H2A Clean and Load.R` that produce datasets not consumed by `Do/H2A Analysis Figs and Tables.R`.

---

## Background Findings

### Analysis loads objects it never uses

| Variable | File | Verdict |
|----------|------|---------|
| `aewr_df` | `aewr.parquet` | Loaded at line 34, never referenced again |
| `aewr_data` | `aewr_data.parquet` | Loaded at line 37, never referenced again (`aewr_data_full` is distinct and IS used) |
| `border_df` | `border_df_analysis.parquet` | Loaded at line 36, never referenced again |
| `border_df_yearly` | `border_df_analysis_year.parquet` | Loaded at lines 41–43, never referenced again |
| `fips_codes` | `fips_codes.csv` | Loaded at line 32, never referenced again |

### Build Dataset.R has three structurally independent sections

| Lines (original) | Purpose | Output | Used in Analysis? |
|------------------|---------|--------|------------------|
| 1–626 | Census-period border pairs | `border_df_analysis.parquet` | **No** |
| 640–1412 | Annual border pairs | `border_df_analysis_year.parquet` | **No** |
| 1425–2365 | Annual county panel | `county_df_analysis_year.parquet` | **Yes** |

Each section clears its own workspace before the next begins. Section 3 re-loads everything it needs independently. Sections 1 and 2 can be deleted without affecting section 3.

Section 3 also contains dead reads: it loads `border_df_year.parquet`, `state_border_pairs.csv`, and `border_counties_allmatches.csv`, but only calls `head()` on them — none feed any computation in `county_df`.

**Correction discovered during execution:** `fips_codes` in Build Dataset section 3 was initially classified as a dead read (only `head()` visible at load site), but is actively used in a `merge()` at line 118 of the cleaned file. It was removed in error and then restored. The other four dead reads (`border_df`, `ppi_data`/WPU01.csv, `state_border_pairs`, `border_counties_allmatches`) were confirmed genuinely unused via a full merge trace of the cleaned script.

### Clean and Load.R has two parallel tracks

- **Census-period track** (lines 1–1215): feeds into Build Dataset sections 1 & 2
- **Year-by-year track** (lines 1216–2308): feeds into Build Dataset section 3

Once Build Dataset sections 1 & 2 are removed, several census-period outputs in Clean and Load lose all downstream consumers.

---

## What Was Cut

### Step 1 — H2A Analysis Figs and Tables.R

Removed 5 dead reads (1 more than originally planned — `aewr_data` was also unused):

```r
fips_codes    <- read_csv(...)            # removed
aewr_df       <- read_parquet(...)        # removed
border_df     <- read_parquet(...)        # removed
aewr_data     <- read_parquet(...)        # removed (distinct from aewr_data_full, which is kept)
border_df_yearly <- read_parquet(...)     # removed
```

Added standalone path header (`if (!exists("folder_dir")) { ... }`) at top of file.

Added diagnostic before Exhibit 13 regression:
```r
cat("samp_base rows:", nrow(samp_base), "\n")
stopifnot(nrow(samp_base) > 1000)
```

---

### Step 2 — H2A Build Dataset.R: sections 1 and 2 removed

Deleted original lines 16–1423 in their entirety (keeping the existing standalone header at lines 1–15):
- Lines 16–626: Census-period border pair construction → `border_df_analysis.parquet`
- Lines 628–638: Workspace cleanup between sections
- Lines 640–1423: Annual border pair construction → `border_df_analysis_year.parquet`

The file now opens directly at the county-year panel section (original line 1425 header `## Yearly Full County Dataset`). **Net removal: ~1,408 lines.**

---

### Step 3 — H2A Build Dataset.R: dead reads in section 3

Removed from the section 3 load block:

```r
border_df              <- read_parquet("border_df_year.parquet")      # removed
fips_codes             <- read.csv("fips_codes.csv")                  # removed, then RESTORED (see note above)
ppi_data               <- read.csv("WPU01.csv")                       # removed
state_border_pairs     <- read.csv("state_border_pairs.csv")          # removed
border_counties_allmatches <- read.csv("border_counties_allmatches.csv") # removed

head(border_df)                  # removed
head(state_border_pairs)         # removed
head(border_counties_allmatches) # removed
```

`fips_codes` was restored after the re-run exposed a merge failure at line 118. All other removals confirmed safe via full merge trace.

Added diagnostic block after `write_parquet(county_df, ...)`:
```r
cat("county_df rows:", nrow(county_df), " | cols:", ncol(county_df), "\n")
stopifnot(nrow(county_df) > 10000)
stopifnot(all(c("aewr_state_ag_ppi_l1", "high_h2a_share_75", ...) %in% names(county_df)))
cat("Diagnostic passed: county_df_analysis_year\n")
```

---

### Step 4 — H2A Clean and Load.R: census-period dead blocks removed

Removed the following census-period output blocks (original line ranges):

| Block | Original lines | Output file removed |
|-------|---------------|---------------------|
| Census of Agriculture | 197–270 | `census_ag_cropland.parquet`, `census_ag_cropland_2007.parquet` |
| ACS & QCEW | 476–548 | `acs_qcew_data.parquet` |
| AEWR census-period | 567–621 | `aewr_data.parquet` |
| BEA CAEMP25N + CAINC45 | 622–932 | `bea_caemp25n_data.parquet`, `bea_cainc45_data.parquet` |
| ACS Immigration imputation | 933–1042 | `acs_immigrant_cleaned.parquet` |
| Census pop estimates | 1043–1134 | `census_pop_ests.parquet` |
| Border DF census-period + rm() cleanup | 1135–1214 | `border_df.parquet` |

Also removed in year-by-year track:
- Redundant CZ re-writes (original lines 1219–1238): `cz_file_2010.parquet` and `cz_file_2010_small.parquet` were already written earlier in the script; these were duplicate writes.
- Border DF year production (original lines 2215–2273): `border_df_year.parquet` — consumed only by Build Dataset section 2 (now deleted) and confirmed dead read in section 3.

Added standalone header (`if (!exists("folder_dir")) { library(...); ... }`) at top — this file had no `library()` calls at all previously.

Added `cat()` diagnostics after each major year-track `write_parquet()` call:
- `h2a_data_year`, `aewr_data_year`, `aewr_data_full`, `bea_caemp25n_data_year`, `bea_cainc45_data_year`, `census_pop_ests_year`, `county_df_year`

**Net removal: ~856 lines.**

---

### Step 5 — H2A Pull Min Wages.R

No dead code removed. Added standalone path header at top of file.

---

## Actual Line Counts

| Script | Before | After | Lines removed |
|--------|--------|-------|---------------|
| `H2A Analysis Figs and Tables.R` | ~2,546 | ~2,546 | ~0 net (removed 5 reads, added header + diagnostics) |
| `H2A Build Dataset.R` | 2,376 | 959 | **1,417** |
| `H2A Clean and Load.R` | 2,308 | 1,452 | **856** |
| `H2A Pull Min Wages.R` | ~113 | ~121 | +8 (header added) |

**Total net reduction: ~2,273 lines.**

---

## What Was Kept (not cut)

| Block in Clean and Load | Output | Reason kept |
|-------------------------|--------|-------------|
| New price data (lines 7–17) | `nass_fisher_price_index.parquet` | Used by Build Dataset section 3 |
| CZ file (lines 62–76) | `cz_file_2010.parquet`, `cz_file_2010_small.parquet` | Used by Build Dataset section 3 |
| PPI (lines 82–109) | `ppi_2012.parquet` | Used by Build Dataset section 3 |
| State min wages (lines 135–196) | `state_real_minwages.parquet` | Used by Build Dataset section 3 |
| H-2A predict (lines 275–282) | `h2a_predict.parquet` | Used by Build Dataset section 3 |
| H-2A data (lines 272–422) | `h2a_data.parquet`, `h2a_data_ts.parquet` | Used directly by Analysis |
| CDL cropshares (lines 426–474) | `cdl_cropshares.parquet` | Downstream use uncertain; left in |
| NAWSPAD + OEWS reads | (no parquet output) | Variable construction; not cut per policy |
| All year-by-year track blocks (except border DF) | `*_year.parquet` files | Used by Build Dataset section 3 |
| `aewr_data_full.parquet` production | `aewr_data_full.parquet` | Used directly by Analysis |
