# B01: Derived data

These scripts turn A-stage source artifacts into aggregates, proxy measures,
and prediction inputs. Geographic identifiers are normalized by their owning
producer; run only the artifact family you need from the repository root.

## Manual execution

Run R scripts with `Rscript`:

```sh
Rscript code/b01_derived/01_h2a_aggregation_nodupes.R
```

The three Python files are Marimo applications:

```sh
uv run marimo edit code/b01_derived/02_price_index_nass_synthetic_cdl.py
uv run marimo edit code/b01_derived/07_h2a_prediction_elastic_net.py
uv run marimo edit code/b01_derived/08_h2a_prediction_from_estimated_weights.py
```

Old root-level B paths map here by dropping the leading `b` from the filename;
the remaining family and substep numbers are unchanged.

## Script contracts

| Script | Responsibility | Primary output |
| --- | --- | --- |
| `01_h2a_aggregation_nodupes.R` | Aggregate worksite records without duplicating case totals | `h2a_aggregated.parquet` |
| `02_price_index_nass_synthetic_cdl.py` | Construct county-year chained Fisher crop price and quantity indexes | `price_index_fisher_county_year.parquet` |
| `03_01_acs_extract.R` | Request, cache, and convert ACS extracts | One-year wage and five-year imputation extracts |
| `03_02_acs_immigrant_imputation.R` | Impute and aggregate immigrant agricultural labor | `acs_immigrant_imputed.parquet` |
| `03_03_acs_cz_wage_quantile.R` | Estimate weighted commuting-zone wage quantiles | `acs_czone_wage_quantiles.parquet` |
| `04_acs_qcew_crop_animal_employment_ratio.R` | Combine ACS and QCEW crop/animal employment shares | `acs_qcew.parquet` |
| `05_01_acs_ag_wage.R` | Estimate ACS agricultural wage proxies | `acs_state_ag_wage.parquet` |
| `05_02_oews_farm_wages.R` | Map source-published OEWS agricultural occupation employment and wages to counties while retaining reporting areas | `oews_county_area_year_occupation.parquet` |
| `05_03_qcew_ag_wages.R` | Aggregate separate all-ownership NAICS 111, NAICS 112, and all-sector employment and nominal wage bills to 2010-vintage county-years | `qcew_county_year.parquet` |
| `06_nawspad_work_hours.R` | Derive regional work hours and seasonality | `nawspad.parquet` |
| `07_h2a_prediction_elastic_net.py` | Fit cutoff-specific H-2A PPML models from climate normals, static soil features, and fixed 2011 employment | Stamped model Parquet and diagnostics |
| `08_h2a_prediction_from_estimated_weights.py` | Score every completed compatible PPML model once per county | `h2a_prediction_using_elastic_net_by_cutoff.parquet` keyed by cutoff and county |

## Dependencies

- `01` consumes the final A02 H-2A location outputs.
- `02` consumes A06 CDL acres and synthetic price/yield outputs.
- Run `03_01` before `03_02`, `03_03`, `04`, and `05_01`.
- `04` also requires A04 annual QCEW. `05_02` requires both A05 OEWS artifacts;
  it repeats each reporting area's source measures for its mapped counties,
  retains separate county-area-occupation rows, and leaves occupation pooling,
  wage-bill construction, and other derived statistics to downstream consumers.
- `05_03` requires A04 annual QCEW and the A00 2010 county-adjacency artifact.
  It retains separate crop-sector, animal-sector, and all-sector source totals
  plus disclosure flags; it constructs no average wage. QCEW disclosure-coded
  cells remain missing, and nonallocatable source geographies are excluded
  rather than repaired downstream. The NAICS 112 fields are
  `qcew_animal_sector_annual_avg_emplvl`,
  `qcew_animal_sector_total_annual_wages`, and
  `qcew_animal_sector_disclosed`.
- `07` requires `01` plus the A08 BEA and A09 climate/soil artifacts.
- `08` consumes the cutoff-specific model Parquets from `07` plus the same BEA,
  climate, and soil artifacts.
- The cutoff, model-specification, scoring, and downstream panel invariants are
  documented in the grounded
  [H-2A prediction contract](../../content/contracts/prediction-model.md).
- `03_01` requires `IPUMS_API_KEY`. Other B scripts use local artifacts only.
