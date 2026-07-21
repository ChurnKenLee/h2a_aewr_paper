# B01: Derived data

These scripts turn A-stage source artifacts into aggregates, proxy measures,
prediction inputs, and harmonized county files. Run only the artifact family
you need from the repository root; there is intentionally no master runner.

## Manual execution

Run R scripts with `Rscript`:

```sh
Rscript code/b01_derived/01_h2a_aggregation_nodupes.R
```

The two Python files are Marimo applications:

```sh
uv run marimo edit code/b01_derived/02_price_index_nass_synthetic_cdl.py
uv run marimo edit code/b01_derived/07_h2a_prediction_elastic_net.py
```

Old root-level B paths map here by dropping the leading `b` from the filename;
the remaining family and substep numbers are unchanged.

## Script contracts

| Script | Responsibility | Primary output |
| --- | --- | --- |
| `01_h2a_aggregation_nodupes.R` | Aggregate worksite records without duplicating case totals | `h2a_aggregated.parquet` |
| `02_price_index_nass_synthetic_cdl.py` | Construct a county-year chained Fisher crop price index | `price_index_fisher_county_year.parquet` |
| `03_01_acs_extract.R` | Request, cache, and convert ACS extracts | One-year wage and five-year imputation extracts |
| `03_02_acs_immigrant_imputation.R` | Impute and aggregate immigrant agricultural labor | `acs_immigrant_imputed.parquet` |
| `03_03_acs_cz_wage_quantile.R` | Estimate weighted commuting-zone wage quantiles | `acs_czone_wage_quantiles.parquet` |
| `04_acs_qcew_crop_animal_employment_ratio.R` | Combine ACS and QCEW crop/animal employment shares | `acs_qcew.parquet` |
| `05_01_acs_ag_wage.R` | Estimate ACS agricultural wage proxies | `acs_state_ag_wage.parquet` |
| `05_02_oews_farm_wages.R` | Aggregate OEWS agricultural wages to counties and states | County and state OEWS Parquet files |
| `05_03_qcew_ag_wages.R` | Estimate QCEW agricultural wage proxies | State wage and industry diagnostic files |
| `06_nawspad_work_hours.R` | Derive regional work hours and seasonality | `nawspad.parquet` |
| `07_h2a_prediction_elastic_net.py` | Fit county H-2A exposure predictions | Elastic-net prediction Parquet and diagnostics |
| `08_harmonize_county_ansi_vintage.R` | Convert county-bearing artifacts to the 2010 ANSI vintage | Rewrites listed intermediate artifacts in place |

## Dependencies

- `01` consumes the final A02 H-2A location outputs.
- `02` consumes A06 CDL acres and synthetic price/yield outputs.
- Run `03_01` before `03_02`, `03_03`, `04`, and `05_01`.
- `04` also requires A04 annual QCEW; `05_02` requires A05 OEWS; `05_03`
  requires A04 annual QCEW.
- `07` requires `01` plus the A08 BEA and A09 climate/soil artifacts.
- Run `08` only after rebuilding the county-bearing inputs it should
  harmonize. It intentionally overwrites those intermediate artifacts rather
  than creating a second set of files.
- `03_01` requires `IPUMS_API_KEY`. Other B scripts use local artifacts only.
