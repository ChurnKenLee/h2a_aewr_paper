+++
title = "Panel IV"
description = "Original moving-wage county panel IV with deterministic R calibration."

[extra]
scopes = ["code/designs/panel_iv", "scripts/run_panel_iv.sh"]
+++

{{ <grounding path="code/designs/panel_iv/design.R" anchor="panel-iv-design-contract" sha256="bddefbfc8e4431bd23752b2a4e738ae63b3d05b5321dca38f16140a007e684e7" /> }}

This branch consumes `data/processed/county_year_panel.parquet` and estimates
the original moving-wage design for 2011--2022. Its two instruments average
current source-year OEWS hourly wage proxies with recovered county weights
among the selected donors, with policy year equal to source year plus one.
Both wages and weights may move. Recovered county distributions are
pseudo-weights calibrated to published aggregates, not observed survey
inclusion probabilities or identified sampling errors. FLS wage targets can
induce mechanical relevance; the construction does not establish exclusion.

The user explicitly retired the fixed-wage revision and all later separation
and innovation exercises, including their results. This restores the original
four-column estimator as the sole supported design; it is not a selection
based on favorable first-stage statistics. All retained implementation,
reporting and validation is R. The outcome registry and reviewed business
linkage remain current rather than reverting to an older repository commit.
The user later re-authorized one diagnostic, the Kitagawa instrument
decomposition below. It leaves every retained instrument, column and table
unchanged.

`estimation_spec.json` and its R reader declare the twelve outcomes, real AEWR
treatment, controls, county/year fixed effects and four columns. All county-year
observations receive equal weight. Column 4 remains the preferred controlled
seasonal/composition specification. There are no DiD classifications, cropland
restrictions or new sample definitions in this branch. See
[Causal outcomes and model inputs](@/contracts/causal-outcomes-and-model-inputs.md).

Calibration now retains deterministic centers only, with rho=.10, gradient
tolerance 1e-10 and the 100-iteration guard. No simulated weights or associated
uncertainty envelopes are produced. This removes diagnostic simulation, not
the weights used in the original instruments. The deterministic schema version
is `panel_iv_deterministic_v1`; incompatible partial caches require a full
rebuild. Source and input corrections remain owned upstream.

The same-region donors and shared FLS targets can transmit regional economic
shocks. Original inference clusters by target subregion and is conditional on
the constructed inputs; cleanup does not resolve dependence, generated-input
uncertainty or the exclusion restriction. Numerical equivalence to the original
estimator is a migration check, not empirical validation.

## Fixed target and donor clusters

The design fixes five crop/climate/soil subregions within each of the 17 AEWR
regions and uses the two most dissimilar donor subregions. Cluster membership
is constructed once from 2008--2011 features. The starting models cluster standard
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
The original implementation uses forward growth from the latest preceding
benchmark, sets annual state-raking factors to one and two-sided drift to zero,
despite historical metadata naming those operations. Missing benchmark fills
can consult future Census/growth evidence; the farm floor is an order-dependent
selection without a TOTAL-domain filter. The R implementation records those
original source-row choices in `census_farm_floor_selection.csv`, checks their
values and the QuickStats SHA-256 fingerprint in `census_farm_floor_source.json`,
and fails on source drift. This preserves the old Polars tie ordering without
a Python runtime. Source county codes remain explicit provenance; the original
exact-key match to the analysis geography is preserved, including zero farm
floors for unmatched source codes. These limitations are disclosed and
retained in the design. Each artifact
records the selected annual update source and fallback flags.

The farm-floor arithmetic adds the selected number of farm operations to
projected hired workers with coefficient one, before normalizing within the
region. This is a legacy prior adjustment, not a published FLS farm-count
constraint, and no sampling-frame justification for that addition is supplied
here. It can create positive prior mass where measured hired employment is
zero. Removing it or replacing it with a different prior would change the
weights and requires a separately declared empirical revision.

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

   {% <equation> %}
   g_{cq}=(w^F_q-w^C_q)E^{111}_{cq}
          +(w^L_q-w^C_q)E^{112}_{cq},
   {% </equation> %}

   with target zero. Regional FLS wages enter the coefficients of this county
   QCEW residual; they are not county wage observations. The construction
   never divides by a rounded field/livestock wage difference,
   infers a share, or clips an implied share.

For each region-year, let $q_c$ be the normalized county prior, $x_c$ the
standardized county features, and $t$ their standardized regional targets.
The deterministic recovery chooses $p_c$ to minimize

{% <equation> %}
\sum_c p_c\log(p_c/q_c)
 + \frac{\rho}{2}\left\|\sum_c p_c x_c-t\right\|^2,
\qquad p_c\geq 0,\quad \sum_c p_c=1,\quad \rho=0.10.
{% </equation> %}

Zero-prior counties remain outside the support. This trades off departure from
the prior against moment mismatch; it does not impose exact matching or
identify actual survey inclusion probabilities. The wage-only distribution
uses one moment, while the preferred distribution adds the three seasonal
contrasts and four composition residuals. One distribution is shared across
the year's survey weeks. Missing county features use their observed-county
mean weighted by the prior within that region-year, with coverage recorded.

Every active moment is centered and scaled under the county prior. A
zero-variation or unavailable-published moment is marked inactive with its
status and coverage; it is not silently discarded. Worker-duration fields and
gross hours remain diagnostic-only. Soft entropy recovery produces only the deterministic center. The R migration
removes simulated draws and their envelopes, parameters, and success-rate
guards; deterministic convergence and moment-fit guards remain. Optimizer
status, residual norms, effective county counts, maximum weights, active
moment counts, and observation coverage are retained.

## Donor OEWS-area hourly wage proxy and instrument

For each eligible donor county and source year, the donor wage level is the
shared county-mapped OEWS-area Big-Six mean hourly-wage proxy. The mapping
assigns an OEWS reporting-area local-labor-market measure to counties. Available
occupation hourly means are employment-weighted within each area, then mapped
area means are combined using township shares, renormalized over observed areas.
This does not turn OEWS into a directly observed county farm-wage series. OEWS
combines six semiannual panels over three years; overlapping samples complicate
annual comparisons, and the MB3 estimation method begins only in 2021. The
historical source principally excludes agricultural producers except logging
and agricultural support industries. See the
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
certifications, certified contract hours, applications, the balanced v2
linked-worksite-business count per 2011 farm employee, positions per application, hours per
position, real crop prices, farm employment, farm-production expense share,
real farm income per current-year farm worker, farm-labor share, and output
quantities.

The panel builder and final validator require the reviewed FY2008--2024
`worksite_business` v2 provenance before the balanced business count can enter
this registry. The review metrics do not select the IV specification or tier.

`07_estimate_panel_iv.R` also writes an identical-sample diagnostic comparing
the committed four-control preferred specification with those same controls
plus the static propensity differential trend. It reports the AEWR
coefficient, clustered standard error, within-(R^2), excluded-instrument F,
observations, counties, and changes without imposing a precision-improvement
sign.

## Kitagawa instrument decomposition (diagnostic)

Stages 10--11 ask how much of each retained instrument's identifying variation
comes from reallocating calibrated mass across donors rather than from donor
wage movements. The recovered weights are used exactly as produced. Their
response to OEWS wage features through the calibration is part of the design,
so the weight component below includes that response. There is no
recalibration with frozen wages.

For target cluster $g$ and source year $t$, the decomposition restricts each
instrument to a balanced donor set: counties eligible in every source year
2010--2021 that have a 2008--2010 base wage. It renormalizes their calibrated
weights to $p_{ct}$, and requires both selected donor clusters in every year.
Clusters without such a set are excluded from the diagnostic, and the retained
eligible mass is reported. With $\bar w_c$ the 2008--2010 mean real OEWS
hourly proxy (same selection and PPI deflation as stage 05) and $\bar p_c$
the source-year mean of $p_{ct}$, the identity

{% <equation> %}
Z^{b}_{gt}=\sum_c\bar p_c\bar w_c
 +\underbrace{\sum_c(p_{ct}-\bar p_c)\bar w_c}_{\text{weight}}
 +\underbrace{\sum_c\bar p_c(w_{ct}-\bar w_c)}_{\text{wage}}
 +\underbrace{\sum_c(p_{ct}-\bar p_c)(w_{ct}-\bar w_c)}_{\text{interaction}}
{% </equation> %}

holds exactly. County fixed effects absorb the constant, and the weight
component depends on $\bar p_c$ only through that constant. The Shapley
components add half the interaction to each of the weight and wage terms.

Stage 11 uses the controlled column-3 (wage-only) and column-4 (preferred)
specifications: the same controls, county/year FE, equal weights and
target-subregion CRV1. It runs on the subsample where the decomposition exists
and reports:

- first stages on the retained, balanced, and each component instrument, plus
  joint models with a clustered test of equal component coefficients;
- variance and covariance shares of the components after partialling out FE and
  controls;
- just-identified 2SLS for every outcome and instrument, with clustered
  Anderson--Rubin confidence sets; and
- 2SLS with the two Shapley components as separate instruments, with a
  cluster-robust two-step-GMM Hansen $J$ test of equal implied effects.

As with the stage 07 diagnostic, each outcome's sample is the one fixest
retains for the retained-instrument 2SLS model. These diagnostics do not
establish exclusion for either channel, and no result selects a specification.

## Retained artifacts

The runner executes stages 01--11 in R. Each retained artifact has one owner:

| Script | Principal output |
| --- | --- |
| `01_build_county_features.R` | `panel_iv_county_features.parquet` |
| `02_cluster_target_units.R` | Fixed target/donor subregions and cluster map |
| `03_build_fls_frame.R` | `panel_iv_fls_frame.parquet` and annual-update diagnostics |
| `04_recover_fls_geography.R` | County-keyed features, deterministic weights, and moment/calibration diagnostics |
| `05_construct_instruments.R` | `panel_iv_county_donor_frame.parquet` and `panel_iv_instrument_cluster_year.parquet` |
| `06_build_county_year_panel.R` | `data/processed/panel_iv_county_year.parquet` |
| `07_estimate_panel_iv.R` | First stages, twelve four-column 2SLS tables, diagnostics, H-2A margins, and summary statistics |
| `08_generate_figures.R` | Six county-calibration diagnostic figures and plotting-data CSVs |
| `09_validate_artifacts.R` | Source, calibration, wage-proxy, timing, sample, and retained-table checks |
| `10_decompose_instruments.R` | `panel_iv_instrument_decomposition_cluster_year.parquet` and decomposition support summary |
| `11_estimate_instrument_decomposition.R` | Decomposition first stages, variance shares, component 2SLS with AR sets, and Hansen $J$ tables |

The newly named recovery artifacts are:

- `panel_iv_fls_county_features.parquet`;
- `panel_iv_fls_county_weight_summary.parquet`;
- `panel_iv_fls_county_calibration_diagnostics.parquet`;
- `panel_iv_fls_county_moment_diagnostics.parquet`.

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
  `table_iv_h2a_applications.tex`,
  `table_iv_h2a_linked_worksite_businesses.tex`,
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
