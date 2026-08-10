# Panel IV

This branch consumes `data/processed/county_year_panel.parquet` directly and
estimates the requested county-year design for 2011--2022. It does not use the
DiD panel, DiD treatment classifications, cropland restrictions, or
observation weights.

The design fixes five crop/climate/soil subregions within each of the 17 AEWR
regions and uses the two most dissimilar donor subregions. Both instruments
use source information from year \(t-1\), a common Census hired-worker frame
prior, and soft entropy calibration with \(\rho=0.10\). The basic instrument
targets the annual FLS field-and-livestock wage only. The alternative-moment
instrument also targets the published quarterly FLS worker shares; this is
the preferred instrument. Worker-duration targets and a separate Census-frame
instrument are not estimation specifications.

All models include county and year fixed effects and cluster standard errors
by AEWR-region-by-subregion (`aewr_iv_cluster_id`). The four first-stage
columns use a common complete-case sample:

1. wage-only instrument;
2. wage-plus-seasonal instrument;
3. wage-only instrument with lagged controls; and
4. wage-plus-seasonal instrument with lagged controls (preferred).

The controls are lagged log county population, lagged farm-employment share,
lagged employment-to-population ratio, the lagged real county 10th-percentile
wage, and a differential trend formed by interacting the standardized static
H-2A PPML propensity with `year - 2011`. The propensity comes from the single
cutoff selected globally with `H2A_PREDICTION_CUTOFF_YEAR`; its predicted count
uses fixed 2011 farm employment and is constant across panel years. The score
level is omitted because county fixed effects absorb it, and the interaction is
a control in both IV stages, never an excluded instrument. Each
outcome has four 2SLS columns: wage-only and wage-plus-seasonal instruments,
each without and with controls. All four columns use a common outcome-specific
sample, and the controlled wage-plus-seasonal specification in column 4 is
preferred. The twelve outcomes
are normalized H-2A certifications, certified contract hours, applications,
the balanced-linkage employer count per 2011 farm employee, positions per application, hours per
position, real crop prices, farm employment, farm-production expense share,
real farm income per current-year farm worker, farm-labor share, and output
quantities.

`07_estimate_panel_iv.R` also writes an identical-sample diagnostic comparing
the committed four-control preferred specification with those same controls
plus the static propensity differential trend. For all twelve outcomes it
reports the AEWR coefficient, clustered standard error, within-$R^2$,
excluded-instrument F, observations, counties, and changes from the four-control
specification without imposing a precision-improvement sign.

## Order and artifacts

| Script | Principal output |
| --- | --- |
| `01_build_county_features.R` | `panel_iv_county_features.parquet` |
| `02_cluster_target_units.R` | Fixed target/donor subregions and cluster map |
| `03_build_fls_frame.py` | `panel_iv_fls_frame.parquet` |
| `04_recover_fls_geography.py` | Wage-only and wage-plus-seasonal entropy weights and diagnostics |
| `05_construct_instruments.R` | Area frame and the two cluster-year instruments |
| `06_build_county_year_panel.R` | `data/processed/panel_iv_county_year.parquet` |
| `07_estimate_panel_iv.R` | First stages, twelve four-column 2SLS tables, static-trend diagnostic, H-2A margin table, and summary statistics |
| `08_generate_figures.R` | Six diagnostic figures and reproducible plotting-data CSVs |

Run:

```sh
./scripts/run_panel_iv.sh
```

The retained estimation products under `outputs/tables` are:

- `table_iv_preferred_first_stage.tex` and
  `iv_preferred_first_stage_estimates.csv`;
- `table_iv_h2a_normalized.tex`, `table_iv_h2a_certified_hours.tex`,
  `table_iv_h2a_applications.tex`, `table_iv_h2a_employers.tex`,
  `table_iv_h2a_positions_per_application.tex`,
  `table_iv_h2a_hours_per_position.tex`, `table_iv_prices.tex`,
  `table_iv_farm_employment.tex`, `table_iv_production_expense_share.tex`,
  `table_iv_farm_income.tex`, `table_iv_farm_labor_share.tex`, and
  `table_iv_output_quantities.tex`;
- `table_iv_h2a_adjustment_margins.tex` and
  `iv_preferred_h2a_adjustment_margin_estimates.csv`;
- `iv_preferred_second_stage_estimates.csv` and
  `iv_preferred_second_stage_samples.csv`; and
- `iv_static_propensity_trend_diagnostic.csv` and
  `table_iv_static_propensity_trend_diagnostic.tex`; and
- `table_iv_preferred_summary_statistics.tex` and
  `iv_preferred_summary_statistics.csv`.

The diagnostic figures are
`fig_iv_dissimilarity_clusters_k5.png`,
`fig_iv_aewr_region_real_wage_series.png`,
`fig_iv_national_real_wage_series.png`,
`fig_iv_fls_oews_cz_scatter.png`,
`fig_iv_cz_entropy_weight_changes_pp.png`,
`fig_iv_california_target_and_donors.png`, and
`fig_iv_target_donor_similarity_slopes.png`. The cluster map is produced by
script 02; the remaining six figures are produced by script 08.
