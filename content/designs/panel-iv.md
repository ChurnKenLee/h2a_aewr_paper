+++
title = "Panel IV"
description = "QCEW-primary county calibration, OEWS-area hourly donor wages, preferred specifications, inference, and retained outputs."

[extra]
scopes = ["code/designs/panel_iv", "scripts/run_panel_iv.sh"]
+++

{{ grounding(path="code/designs/panel_iv/design.R", anchor="panel-iv-design-contract", sha256="e810543e472e5b67b5b0b2a9a8cd051a1718842e13f3932f4d69fb8ed0ed9960") }}

The cross-design distinction between causal outcomes and required model inputs
is maintained in [Causal outcomes and model inputs](@/contracts/causal-outcomes-and-model-inputs.md).

This branch consumes `data/processed/county_year_panel.parquet` directly and
estimates the county-year design for 2011--2022. It does not use the DiD panel,
DiD treatment classifications, cropland restrictions, or observation weights.
The treatment, fixed effects, outcomes, controls, samples, target/donor
clusters, and inference rule are unchanged by the county-weight and donor-wage
measurement updates.

## Fixed target and donor clusters

The design fixes five crop/climate/soil subregions within each of the 17 AEWR
regions and uses the two most dissimilar donor subregions. Cluster membership
is constructed once from 2008--2011 features. All models cluster standard
errors by the resulting AEWR-region-by-subregion identifier
`aewr_iv_cluster_id`.

Both instruments use source information from year (t-1), a common Census
hired-worker county prior, and soft entropy calibration with `rho = 0.10`.
They differ only in the active target moments described below.

## County prior and annual path

The frame begins with Census of Agriculture county hired-worker benchmarks.
Its annual path is updated on the shared panel with disclosed all-ownership
QCEW NAICS 111+112 employment first, QWI agricultural employment when QCEW is
incomplete, and positive BEA hired-farm jobs as the final employment fallback.
State raking and the existing prior-period timing are preserved. Each artifact
records the selected annual update source and fallback flags.

Calibration estimates normalized county weights, not OEWS-area weights. For
each specification there is exactly one county distribution per AEWR region
and source year, shared by all four survey weeks. Published FLS worker totals
set seasonal shares only; they never constrain a regional employment level.

## Paired FLS targets

USDA FLS publishes regional worker counts and separate field, livestock,
combined field/livestock, and all-hired hourly wage rates for its survey
reference weeks. Worker and wage tables are paired within the same release.
The selected annual-report release is preferred; otherwise the latest paired
release no later than that report is selected. Release date, source ZIP, source
CSV, table title, annual source, and selection method travel with both
quarterly artifacts. See the [USDA NASS Farm Labor survey guide](https://www.nass.usda.gov/Surveys/Guide_to_NASS_Surveys/Farm_Labor/).

The supported 2010--2021 grid contains all 816 region-year-quarter keys. USDA
explicitly did not conduct the April 2011 survey. Those 17 paired table rows
carry `survey_not_conducted`; no worker count or wage rate is fabricated. For
2011 the available three survey weeks yield two independent seasonal
contrasts, while the third seasonal contrast and April composition moment are
retained as explicit inactive diagnostics.

## Calibration moments

The wage-only specification uses the annual FLS combined
field-and-livestock/OEWS-hourly moment. County-mapped OEWS-area Big-Six hourly
wages supply the wage feature for that benchmark; QCEW continues to supply the
county employment distribution used by the prior and the preferred moments.

The preferred specification adds QCEW-primary seasonal and composition
information:

1. Three independent Helmert contrasts of the four FLS quarterly worker
   shares against county QCEW 111+112 quarterly employment shares.
2. One field/livestock composition moment per quarter. For county $c$ and
   quarter $q$, its feature is the undivided soft residual

   \[
   g_{cq}=(w^F_q-w^C_q)E^{111}_{cq}
          +(w^L_q-w^C_q)E^{112}_{cq},
   \]

   with target zero. Every wage value remains on the FLS target side. The
   construction never divides by a rounded field/livestock wage difference,
   infers a share, or clips an implied share.

Every active moment is centered and scaled under the county prior. A
zero-variation or unavailable-published moment is marked inactive with its
status and coverage; it is not silently discarded. Worker-duration fields and
gross hours remain diagnostic-only. Soft entropy recovery produces the
deterministic center plus seeded Dirichlet prior-draw diagnostics. Optimizer
status, residual norms, effective county counts, maximum weights, active
moment counts, and observation coverage are retained.

## Donor OEWS-area hourly wage proxy and instrument

For each eligible donor county and source year, the donor wage level is the
shared county-mapped OEWS-area Big-Six mean hourly-wage proxy. The mapping
assigns an OEWS reporting-area local-labor-market measure to counties and can
employment-weight multiple mapped areas; it does not turn OEWS into a directly
observed county farm-wage series. OEWS estimates combine six semiannual panels
over three years, so the proxy is deliberately described as an area-level,
model-based wage measure rather than a county administrative observation. See
the [BLS OEWS overview](https://www.bls.gov/opub/hom/oews/home.htm) and
[calculation methodology](https://www.bls.gov/opub/hom/oews/calculation.htm).

QCEW remains primary for the county employment path, quarterly seasonality,
and field/livestock composition. QCEW wage bills and BEA farm wages are not
used as donor wage fallbacks, and no QCEW hourly conversion or hours-per-week
assumption is introduced. The OEWS hourly proxy is deflated with the
source-year PPI. Recovered county weights are restricted and renormalized over
counties with an observed proxy. The existing target-area overlap exclusion
remains in force using the county's primary OEWS reporting area. Both selected
donor clusters must contribute support.

The cluster-year artifacts record the declared OEWS proxy and geography,
OEWS wage-proxy calibrated-mass coverage, QCEW annual-employment coverage,
eligible calibrated mass, effective donor county count, maximum county weight,
donor-cluster support, active moments, and calibration identifiers. Policy
year must equal source year plus one.

## Estimation contract

The retained instrument fields are:

- `z_wage_only_real`; and
- `z_wage_seasonal_composition_real`, the preferred field.

The four first-stage columns use a common complete-case sample:

1. wage-only instrument;
2. wage-plus-seasonal/composition instrument;
3. wage-only instrument with controls; and
4. wage-plus-seasonal/composition instrument with controls (preferred).

The controls are lagged log county population, lagged farm-employment share,
lagged employment-to-population ratio, the lagged real county 10th-percentile
wage, and a differential trend formed by interacting the standardized static
H-2A PPML propensity with `year - 2011`. The propensity comes from the one
global `H2A_PREDICTION_CUTOFF_YEAR`; its predicted count uses fixed 2011 farm
employment and is constant across panel years. The score level is omitted
because county fixed effects absorb it, and the interaction is a control in
both IV stages, never an excluded instrument.

Each outcome has the same four 2SLS columns and an outcome-specific common
sample. Column 4 remains preferred. The twelve outcomes are normalized H-2A
certifications, certified contract hours, applications, the balanced-linkage
employer count per 2011 farm employee, positions per application, hours per
position, real crop prices, farm employment, farm-production expense share,
real farm income per current-year farm worker, farm-labor share, and output
quantities.

`07_estimate_panel_iv.R` also writes an identical-sample diagnostic comparing
the committed four-control preferred specification with those same controls
plus the static propensity differential trend. It reports the AEWR
coefficient, clustered standard error, within-(R^2), excluded-instrument F,
observations, counties, and changes without imposing a precision-improvement
sign.

## Order and artifacts

| Script | Principal output |
| --- | --- |
| `01_build_county_features.R` | `panel_iv_county_features.parquet` |
| `02_cluster_target_units.R` | Fixed target/donor subregions and cluster map |
| `03_build_fls_frame.py` | `panel_iv_fls_frame.parquet` and annual-update diagnostics |
| `04_recover_fls_geography.py` | County-keyed features, weights, moment/calibration diagnostics, and draw partitions |
| `05_construct_instruments.R` | `panel_iv_county_donor_frame.parquet` and `panel_iv_instrument_cluster_year.parquet` |
| `06_build_county_year_panel.R` | `data/processed/panel_iv_county_year.parquet` |
| `07_estimate_panel_iv.R` | First stages, twelve four-column 2SLS tables, diagnostics, H-2A margins, and summary statistics |
| `08_generate_figures.R` | Six county-calibration diagnostic figures and plotting-data CSVs |
| `09_validate_artifacts.py` | Source, calibration, wage-proxy, timing, sample, and retained-table checks |

The newly named recovery artifacts are:

- `panel_iv_fls_county_features.parquet`;
- `panel_iv_fls_county_weight_summary.parquet`;
- `panel_iv_fls_county_calibration_diagnostics.parquet`;
- `panel_iv_fls_county_moment_diagnostics.parquet`; and
- partitioned diagnostic draws under `panel_iv_fls_county_draws/`.

These identifiers and paths prevent stale OEWS-area weight partitions from
being reused.

Run:

```sh
./scripts/run_panel_iv.sh
```

The retained estimation filenames under `outputs/tables` are unchanged:

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
  `iv_preferred_second_stage_samples.csv`;
- `iv_static_propensity_trend_diagnostic.csv` and
  `table_iv_static_propensity_trend_diagnostic.tex`; and
- `table_iv_preferred_summary_statistics.tex` and
  `iv_preferred_summary_statistics.csv`.

For email circulation, the same estimates are also transposed into
`panel_iv_email_results.pdf`, which contains one landscape table per model
specification and one outcome per column. Editable copies are written as
`panel_iv_email_results_spec1_wage_only.csv`,
`panel_iv_email_results_spec2_seasonal_composition.csv`,
`panel_iv_email_results_spec3_wage_only_controls.csv`, and
`panel_iv_email_results_spec4_preferred.csv`. These exports change only table
orientation and formatting; they do not re-estimate or rescale any model.

The diagnostic figures are
`fig_iv_dissimilarity_clusters_k5.png`,
`fig_iv_aewr_region_wage_calibration.png`,
`fig_iv_national_wage_calibration.png`,
`fig_iv_qcew_fls_moment_residuals.png`,
`fig_iv_county_entropy_weight_changes_pp.png`,
`fig_iv_oews_wage_proxy_coverage.png`, and
`fig_iv_target_donor_support.png`. Script 02 produces the fixed cluster map;
script 08 produces the remaining six figures and corresponding CSVs.

Synthetic regression checks live under `code/designs/panel_iv/tests/`. The
final validator checks geographic strings, declared keys, QCEW disclosure
semantics, paired FLS releases and the documented 2011 gap, weight sums,
optimizer success, OEWS hourly proxy semantics and coverage, (t-1) timing,
four-column ordering, common samples, cluster counts, and first-stage
diagnostics.
