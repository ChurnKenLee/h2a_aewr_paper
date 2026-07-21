# C02: Build the analysis panel

Run these scripts in order:

```sh
Rscript code/c02_build/01_merge_county_panel.R
Rscript code/c02_build/02_derive_analysis_variables.R
Rscript code/c02_build/03_classify_treatment_exposure.R
Rscript code/c02_build/04_finalize_county_panel.R
```

| Script | Responsibility | Output |
| --- | --- | --- |
| `01_merge_county_panel.R` | Merge cleaned panels onto the county-year backbone | `county_df_build_merge.parquet` |
| `02_derive_analysis_variables.R` | Construct outcomes, real measures, fixed effects, and lags | `county_df_variable_cleaned_year.parquet` |
| `03_classify_treatment_exposure.R` | Create baseline exposure, treatment, border, and period classifications | `county_df_classified_year.parquet` |
| `04_finalize_county_panel.R` | Report integrity diagnostics and publish the panel | `processed/county_df_analysis_year.parquet` |

Finalization reports existing integrity problems but does not silently alter
the analytical sample.
