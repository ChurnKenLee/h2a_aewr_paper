# CZ Wage Quantiles: Status and Next Steps

**Date:** 2026-03-30
**Status:** Blocked — waiting on Ken to fix and regenerate `acs_czone_wage_quantiles.parquet`

---

## Background

Ken added CZ-level hourly wage percentiles to `county_df_analysis_year.parquet` by inserting a block into `Do/H2A Build Dataset.R` (lines 1648–1672). The new columns are `wage_p10`, `wage_p25`, `wage_p50`, `wage_p75`, `wage_p90`, and their lags (`*_l1`), sourced from `Data Int/acs_czone_wage_quantiles.parquet`.

The purpose is to create a **new, model-inspired definition of the AEWR "bite"**: instead of comparing the AEWR to state minimum wages, we compare it to percentiles of the local (CZ-level) wage distribution. This requires two changes to the analysis:

1. **New bite variables** in `Do/H2A Build Dataset.R` around lines 1678–1688, alongside the existing `aewr_state_ag_ppi_l1` definition, e.g.:
   ```r
   aewr_cz_p10_l1 = aewr_ppi_l1 - wage_p10_l1,
   aewr_cz_p25_l1 = aewr_ppi_l1 - wage_p25_l1,
   ...
   ```
2. **CZ-level clustering** in `Do/H2A Analysis Figs and Tables.R` — change all five `vcov = ~statefips` calls (lines 947, 961, 989, 1010, 1155) to `vcov = ~cz_1990`. The `cz_1990` column is already present in `county_df` via the wage quantile merge.

---

## Problem Discovered

Phil added a diagnostic check at line 1691 of `H2A Build Dataset.R` that revealed widespread NAs in the wage percentile columns for states that should have data (CO, AR, and likely others).

---

## Root Causes (Both in `code/b03_02_acs_cz_wage_quantiles.R`)

Both bugs are upstream in Ken's script. `H2A Build Dataset.R` itself is fine.

### Bug 1 — NHGIS trailing zero on county code (primary, likely universal)

**Location:** `b03_02_acs_cz_wage_quantiles.R`, line 33

Fabian Eckert's `cz_crosswalk.csv` uses NHGIS codes, which append a trailing zero to both state and county FIPS:
- NHGISST: `"010"` = state FIPS `"01"` + trailing `"0"`
- NHGISCTY: `"0010"` = county FIPS `"001"` + trailing `"0"`

The existing code correctly strips the trailing zero from NHGISST on line 32:
```r
state_ansi = substr(NHGISST, 1, nchar(NHGISST) - 1)  # "010" → "01" ✓
```

But line 33 pastes NHGISCTY without stripping its trailing zero:
```r
county_ansi = paste0(state_ansi, NHGISCTY)
# "01" + "0010" = "010010"  ← 6-char code, not standard 5-char FIPS
```

When `H2A Build Dataset.R` converts this to numeric (`as.numeric(county_ansi)`), it gets `10010` instead of `1001` — a systematic mismatch against `county_df$countyfips` for every county in every state. The merge returns NA universally.

**Fix:** Strip the trailing zero from NHGISCTY on line 33, parallel to the treatment of NHGISST on line 32.

---

### Bug 2 — Potential CZ ID vintage mismatch (secondary, residual NAs after Bug 1 fixed)

**Location:** `b03_02_acs_cz_wage_quantiles.R`, lines 212–213

Wage quantiles are computed at the CZ level using **Autor's** PUMA-to-CZ crosswalk (`cw_puma2000_czone.dta` / `cw_puma2010_czone.dta`). County codes are then attached via a left-join to **Eckert's** county-to-CZ crosswalk (`cz_crosswalk.csv`, filtered to Year == 2010):

```r
wage_quantiles_county <- wage_quantiles_czone %>%
  left_join(czone_1990_county_2010_xwalk, by = join_by(czone == cz))
```

Both crosswalks claim to use 1990 CZ definitions, but their CZ integer IDs may not be identical. Any CZ present in Autor's crosswalk but absent from Eckert's will produce `county_ansi = NA` after the left_join, and those counties will always be missing in the final merge — even after Bug 1 is fixed.

**Fix:** After fixing Bug 1, Ken should check how many CZs from `wage_quantiles_czone` fail to match in `czone_1990_county_2010_xwalk`. If there are mismatches, a reconciliation of the two CZ ID systems is needed before the left_join.

---

## Action Items for Ken

- [ ] Fix line 33 in `b03_02_acs_cz_wage_quantiles.R`: strip the trailing zero from NHGISCTY when constructing `county_ansi`
- [ ] After that fix, check whether any CZs in `wage_quantiles_czone` fail to join to `czone_1990_county_2010_xwalk` (Bug 2); if so, reconcile the CZ ID systems
- [ ] Regenerate `acs_czone_wage_quantiles.parquet` and push to Dropbox

## Action Items for Phil (after Ken's parquet is updated)

- [ ] Re-run `Do/H2A Build Dataset.R` and confirm NAs are gone using the line 1691 check
- [ ] Add the new CZ-bite variables to the `mutate` block at lines 1678–1688 of `H2A Build Dataset.R`
- [ ] Change all five `vcov = ~statefips` calls in `H2A Analysis Figs and Tables.R` (lines 947, 961, 989, 1010, 1155) to `vcov = ~cz_1990`
- [ ] Run analysis and compare results under CZ-bite vs. state-min-wage-bite specifications
