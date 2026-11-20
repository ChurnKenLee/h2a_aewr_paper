+++
title = "Panel IV"
description = "Original moving-wage county panel IV with deterministic R calibration."

[extra]
scopes = ["code/designs/panel_iv", "scripts/run_panel_iv.sh"]
+++

{{ <grounding path="code/designs/panel_iv/design.R" anchor="panel-iv-design-contract" sha256="4a662b10cda761571109a03086866464b2d10a58f019a02fe917dbf0825d7f51" /> }}

This branch consumes `data/processed/county_year_panel.parquet` and estimates
the original moving-wage design for policy years 2011--2023. Its instrument
averages
current source-year OEWS hourly wage proxies with recovered county weights
among the selected donors, with policy year equal to source year plus one.
Both wages and weights may move. Recovered county distributions are
pseudo-weights calibrated to published aggregates, not observed survey
inclusion probabilities or identified sampling errors. FLS wage targets can
induce mechanical relevance; the construction does not establish exclusion.

The user explicitly retired the fixed-wage revision and all later separation
and innovation exercises, including their results. The original moving-wage
estimator is the sole supported design; it is not a selection
based on favorable first-stage statistics. All retained implementation,
reporting and validation is R. The outcome registry and reviewed business
linkage remain current rather than reverting to an older repository commit.

On 2026-11-09 the user reduced the specifications. The wage-only instrument
(its calibration, its `z_wage_only_real` field and its estimation columns)
and the `continuing` control set with its 2024 samples were removed together
with their outputs. The two retained columns reproduce former columns 2 and 4
exactly, and the retained calibration, instrument and panel values are
unchanged; the removal was requested as a simplification and was not chosen
from the retired results. The Kitagawa instrument decomposition was first
removed and then restored the same day in the reduced form described below:
the components and their variance shares, without the former component
2SLS, Anderson--Rubin sets or Hansen $J$ test. Component first stages
returned with the stage 11 QWI wage-granularity diagnostic.

`estimation_spec.json` and its R reader declare the twelve outcomes, real AEWR
treatment, the single estimated instrument, controls, county/year fixed effects
and two columns. All county-year
observations receive equal weight. Column 2 is the preferred controlled
specification. There are no DiD classifications, cropland
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

The instrument uses source information from year (t-1), a Census
hired-worker county prior, and soft entropy calibration with `rho = 0.10`
on the target moments described below.

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
the retained specification there is exactly one county distribution per AEWR region
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

The supported 2010--2023 source-year grid contains all 952 region-year-quarter
keys. USDA
explicitly did not conduct the April 2011 survey. Those 17 paired table rows
carry `survey_not_conducted`; no worker count or wage rate is fabricated. For
2011 the available three survey weeks yield two independent seasonal
contrasts, while the third seasonal contrast and April composition moment are
retained as explicit inactive diagnostics.

## Calibration moments

The calibration targets the annual FLS combined
field-and-livestock/OEWS-hourly moment. County-mapped OEWS-area Big-Six hourly
wages supply the wage feature for that benchmark; QCEW supplies the
county employment distribution used by the prior and the remaining moments.
A separate distribution calibrated to this wage moment alone was retired on
2026-11-09 and is no longer recovered.

The calibration also targets QCEW-primary seasonal and composition
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
identify actual survey inclusion probabilities. The distribution uses the
annual wage moment, the three seasonal contrasts and the four composition
residuals. One distribution is shared across
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

The panel carries one instrument field,
`z_wage_seasonal_composition_real`. The validator rejects a panel that still
carries the retired `z_wage_only_real`.

The two first-stage columns use a common complete-case sample:

1. wage-plus-seasonal/composition instrument without controls; and
2. wage-plus-seasonal/composition instrument with controls (preferred).

The controls are lagged log county population, lagged farm-employment share,
lagged employment-to-population ratio, the lagged real county 10th-percentile
wage, and a differential trend formed by interacting the standardized static
H-2A PPML propensity with `year - 2011`. The propensity comes from the one
global `H2A_PREDICTION_CUTOFF_YEAR`; its predicted count uses fixed 2011 farm
employment and is constant across panel years. The score level is omitted
because county fixed effects absorb it, and the interaction is a control in
both IV stages, never an excluded instrument.

## Estimation window

BEA discontinued the county CAEMP25N employment table after 2022, so the
lagged farm-employment share and employment-to-population controls end in
policy year 2023. Both columns use the controlled specification's complete
cases, so every estimation sample runs from policy year 2011 through 2023
(source years 2010--2022). `DISSIMILARITY_IV_ESTIMATION_END_YEAR` and
`policy_years` in `estimation_spec.json` declare that window.

Stages 05--06 still construct the instrument and the county-year panel through
policy year 2024 (`DISSIMILARITY_IV_POLICY_END_YEAR`); the 2024 rows are not
estimated. The former `continuing` control set, which replaced the two BEA
employment ratios with the CAINC5N farm-earnings share and QCEW all-sector
employment per resident in order to reach 2024, was removed at the user's
request. Restoring it or otherwise extending the window is a separate
specification change. The common shared baseline reaches 2025 and includes
FY2026 Q3 disclosures. The 2023 occupation-specific rates and late-2025
job-order cohort are source applicability diagnostics, not a wholesale
replacement of annual regional AEWR. Outcomes built from current BEA farm
employment or CAINC45 farm income end in 2022.

Each outcome has the same two 2SLS columns and an outcome-specific common
sample. Column 2 is preferred. The twelve outcomes are normalized H-2A
certifications, certified contract hours, applications, the balanced v2
linked-worksite-business count per 2011 farm employee, positions per application, hours per
position, real crop prices, farm employment, farm-production expense share,
real farm income per current-year farm worker, farm-labor share, and output
quantities.

The panel builder and final validator require the FY2008--2026 Q3
`worksite_business` v2 provenance with the review status pinned in `design.R`
(currently `pending`) before the balanced business count can enter
this registry. The review metrics do not select the IV specification or tier.

`07_estimate_panel_iv.R` also writes an identical-sample diagnostic comparing
the four lagged controls alone with the preferred specification, which adds
the static propensity differential trend. It reports the AEWR
coefficient, clustered standard error, within-(R^2), excluded-instrument F,
observations, counties, and changes without imposing a precision-improvement
sign.

## Kitagawa instrument decomposition (diagnostic)

Stage 10 asks how much of the instrument's identifying variation comes from
reallocating calibrated mass across donor counties rather than from donor wage
movements. The recovered weights are used exactly as produced. Their response
to OEWS wage features through the calibration is part of the design, so the
weight component below includes that response. There is no recalibration with
frozen wages.

For target cluster $g$ and source year $t$, the decomposition restricts the
instrument to a balanced donor set: counties eligible in every source year
2010--2023 that have a 2008--2010 base wage. It renormalizes their calibrated
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
component depends on $\bar p_c$ only through that constant.

The stage then residualizes the instrument, the balanced instrument $Z^b$ and
the three components on the preferred column's county and year fixed effects
and controls, on the preferred first-stage sample where the decomposition
exists. It reports each component's variance, the pairwise covariances, and
each component's covariance with $Z^b$ as a share of the variance of $Z^b$;
the last three shares sum to one. It also reports the residual variance of the
actual instrument relative to $Z^b$ and their correlation. Entry and exit of
donor counties is excluded by the balanced-donor rule, so that comparison
shows how much of the instrument's variation the decomposition does not
cover. The shares describe the instrument only. They do not establish
exclusion for either channel, and no result selects a specification.

## QWI wage-granularity diagnostic

The weight component is small partly because about five donor counties share
each OEWS reporting area and therefore one wage level, so reallocating
calibrated mass among them changes nothing. Stage 11, added at the user's
request on 2026-11-09, shows what a county-level wage would do. It uses QWI
average monthly earnings of stable NAICS 111/112 jobs
(`qwi_county_ag_quarterly_earnings.parquet`, published values only), averaged
over quarters and both industries with stable-employment weights and deflated
with the source-year PPI. Two variants are compared:

- `oews`, the design wage and weights, which must reproduce the stage 04
  weights, the stage 05 instrument and the stage 10 components; and
- `qwi_annual_recalibrated`, the OEWS wage times an annual county factor, in
  the calibration's county wage feature and in the instrument, with weights
  recalibrated in memory by stage 04's moments, solver and `rho = 0.10`. The
  region's wage path stays OEWS while each county's position within the
  region follows QWI year by year.

The annual factor is the county's QWI/OEWS ratio in that year divided by its
AEWR region's mean ratio, weighted by mean prior weight times the OEWS wage,
from 2008 on. That normalization keeps each region-year's prior-mean wage, and
therefore the calibration's gap to the FLS wage target, close to the design's,
so recalibration responds to the added county dispersion rather than to a
level shift. A county-year without a ratio takes the county's nearest earlier
year, else its nearest later year, so some early factors use later data; a
county never observed keeps factor one. Earlier versions of the stage also
reported a fixed 2008--2010 factor, the design weights with adjusted wages,
and QWI earnings outright; the user dropped them on 2026-11-09.

For each the stage rebuilds the instrument with stage 05's eligibility,
renormalization and support rules, repeats the stage 10 decomposition and
reports QWI coverage, the spread of wage levels across donors, the weight,
wage and interaction shares of the balanced instrument's variance after fixed
effects and controls, and preferred-column first stages for the instrument,
the balanced instrument, and the weight and wage components alone.

The county-mapped OEWS-area wage remains the only wage of the design, in the
calibration and in the instrument. The stage writes no calibration or
instrument artifact, estimates no outcome equation, and
its results do not select a wage source. QWI earnings mix pay with hours and
are not an hourly wage; dividing them by 2080/12 hours only places them on the
OEWS scale, and the regional normalization removes that constant from the
factor. A weight-only first stage is
the first stage of a fixed-wage instrument on the balanced donors, so this
stage also reports on that retired revision without restoring it.

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
| `07_estimate_panel_iv.R` | First stage, twelve two-column 2SLS tables, diagnostics, H-2A margins, and summary statistics |
| `08_generate_figures.R` | Six county-calibration diagnostic figures and plotting-data CSVs |
| `09_validate_artifacts.R` | Source, calibration, wage-proxy, timing, sample, and retained-table checks |
| `10_decompose_instrument.R` | `panel_iv_instrument_decomposition_cluster_year.parquet`, decomposition support summary and variance shares |
| `11_qwi_wage_diagnostics.R` | `iv_qwi_wage_diagnostics_coverage.csv`, `iv_qwi_wage_diagnostics_summary.csv` and `iv_qwi_wage_diagnostics_first_stage.csv` |

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
`panel_iv_email_results_spec1_no_controls.csv` and
`panel_iv_email_results_spec2_preferred.csv`. These exports change only table
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
two-column ordering, common samples, cluster counts, and first-stage
diagnostics.
