+++
title = "Causal outcomes and model inputs"
description = "Canonical cross-design outcome inventory and the treatments, controls, moderators, denominators, samples, and inference fields required to estimate them."

[extra]
scopes = ["code/c02_build", "code/designs", "scripts/run_did.sh", "scripts/run_panel_iv.sh", "scripts/run_mundlak_chamberlain.sh", "scripts/run_border_discontinuity.sh"]
+++

This page distinguishes variables that are causal outcomes from variables used
to define treatment, scale an outcome, select a sample, adjust a model, absorb
heterogeneity, construct an instrument, or calculate inference. A field can be
scientifically important without being an outcome.

The canonical union contains thirteen causal outcomes. Individual designs use
different subsets and sometimes different units, so similar labels are not
automatically interchangeable across designs.

## H-2A utilization and adjustment outcomes

| Causal outcome | Executable column or construction | Supported designs |
| --- | --- | --- |
| Certified positions/workers relative to a declared farm-employment baseline | Shared: `h2a_cert_share_farm_workers_2011_start_year`; MC fits raw `mc_y_certified_positions`, then reports a post-fit effect per 1,000 mean 2008–2010 farm workers | DiD primary and event study; panel IV; AEWR-border ring experiment; MC |
| Certified hours relative to a declared farm-employment baseline | Shared: `h2a_cert_hours_per_farm_worker_2011_start_year`; MC fits raw `mc_y_certified_hours`, then reports a post-fit effect per mean 2008–2010 farm worker | Panel IV; MC |
| Applications relative to a declared farm-employment baseline | Shared: `h2a_applications_per_farm_worker_2011_start_year`; MC fits raw `mc_y_applications`, then reports a post-fit effect per 1,000 mean 2008–2010 farm workers | Panel IV; MC |
| H-2A linked worksite businesses using balanced v2 linkage | Source: `nbr_employers_balanced_start_year`; panel IV divides by `emp_farm_2011`; MC models the raw count as `mc_y_linked_worksite_businesses_balanced` | Panel IV; MC |
| Certified positions per application | Shared: `h2a_cert_positions_per_application_start_year`; MC derives the ratio-of-aggregate effect after jointly fitting raw certified positions and applications | Panel IV; MC |
| Certified hours per certified position | Shared: `h2a_cert_hours_per_position_start_year`; MC derives the ratio-of-aggregate effect after jointly fitting raw certified hours and positions | Panel IV; MC |
| Any H-2A application | MC constructs `mc_y_any_application = 1[nbr_applications_start_year > 0]` and estimates it as a Gaussian linear-probability outcome | MC |

The shared panel preserves observed outcomes when statutory wage coverage is missing, including 2009. Each design owns its required wage-complete sample.

The shared-panel producer supplies the fixed-2011 denominators and the common
ratios used by DiD and panel IV. It retains conservative, balanced, and
high-recall worksite-business counts separately; the balanced v2 count is the
supported causal business outcome. These are positive-certified county-year
counts constructed under the FY2008--2026 Q3 (review pending)
[employer-linkage contract](@/contracts/employer-linkage.md).

H-2A start-year positions and hours use the source-owned employment-period
dates; applications and their status counts use one intended-start cohort.
A usable start with a missing end contributes positions but no interval
hours. Thus positions per application compares these annual cohorts, not
necessarily the same set of cases. The era mapping and fallback flags are
specified in the [date contract](@/contracts/h2a-worksite-exposure.md).

Requested positions are unavailable throughout the FY2008--2010 and
FY2013--2014 main disclosures. B01 preserves unknown requested-position and
requested-hour totals whenever any contributing allocated record is missing;
it does not report a partial observed sum as a complete total. A calendar with
no allocated records still has zero. C02 preserves this distinction, including
after joining to the county-year backbone. Calendar-year totals can be
incomplete when they combine disclosures from adjacent fiscal years.
Requested-outcome models must handle this coverage explicitly; certified-count
samples must not be restricted to requested-count availability. MC version 4.1.0 therefore excludes requested positions and their derived
rate from the baseline, as authorized by the user. Its five remaining outcomes
retain the full declared sample; requested positions are no longer a required
MC input. Shared source values and missingness are preserved.

{{ <grounding path="code/c02_build/01_build_county_panel.R" anchor="shared-panel-construction" sha256="99a5725bc861f4b3eb1e7c2be8cb1e1ac339cc0d22c8e259a62e99d261428eed" /> }}

## Farm-economic outcomes

| Causal outcome | Executable column or construction | Supported designs |
| --- | --- | --- |
| Real Fisher crop price index | `fisher_index_ppi` | DiD; panel IV |
| Farm employment | `emp_farm` | Panel IV |
| Farm production expenses relative to cash receipts and other income | `share_farm_prodexp_cashandinc` | Panel IV |
| Real farm cash receipts and other income per current farm worker | Panel IV constructs `1000 * farm_cashandinc_ppi / emp_farm` as `farm_cashandinc_ppi_per_farm_worker` | Panel IV |
| Hired-labor share of farm production expenses | `share_farm_laborexp_prodexp` | DiD; panel IV |
| Fisher crop output-quantity index | `fisher_quantity_index` | Panel IV |

{{ <grounding path="code/c01_clean/13_merge_county_panel.R" anchor="county-year-merge" sha256="42d6578ce958f298f14bbfca2d9b26e2f43c1735f2e922f1b194dd025e833885" /> }}

`fisher_index_ppi` divides the nominal index by the annual `ppi_2012` joined
by year in the merge, so the real price index is defined through 2024 even
though BEA farm-income rows end in 2022.

These are downstream responses to the policy exposure, not controls. Moving
one of them to the right-hand side or using it for sample selection would
change the estimand and requires an explicit design revision.

## Difference-in-differences mapping

The retained DiD outcomes are:

1. normalized H-2A certified positions/workers, which is the primary outcome
   and the event-study outcome;
2. the real Fisher crop price index; and
3. the hired-labor share of farm production expenses.

{{ <grounding path="code/designs/did/02_main_results.R" anchor="did-primary-outcome" sha256="96bbc956b09e9017a67f9fdae375579f17371c9f01d88598e30e9c7a6a846ab5" /> }}

{{ <grounding path="code/designs/did/05_fisher_price.R" anchor="did-fisher-price-outcome" sha256="7187abfdf20e081257ccef4ce429c815c1c981d692a28ede90188631efb87c80" /> }}

{{ <grounding path="code/designs/did/06_labor_share.R" anchor="did-labor-share-outcome" sha256="b2467ca3ab9f44066b4e3727e7444cc94c8b84056d3cc822ad14b5bbfea56650" /> }}

The DiD model additionally needs:

- the lagged real AEWR-minus-local-p25 wage gap `aewr_cz_p25_l1`, the
  post-2011 indicator, and their interaction;
- the static predicted H-2A share and 2008 observed H-2A share used to create
  time-invariant treatment groups;
- `any_cropland_2007` and the treatment group for the retained sample, plus
  `border_cz` for the no-border columns;
- `ln_pop_census` and `emp_pop_ratio` in controlled columns;
- `county_fips` and `year` fixed effects; and
- `cz_id` and `aewr_region_id` for clustered inference.

{{ <grounding path="code/designs/did/01_build_did_panel.R" anchor="did-treatment-inputs" sha256="107031bcdcda7d02bfff630c95b423659522a4ecdee0637374ecb585dce55bb7" /> }}

{{ <grounding path="code/designs/did/helpers.R" anchor="did-estimation-contract" sha256="4822f93257431fac0a7ffe9399d475d9aa14324d6f76580d666775e9b3b5ab9c" /> }}

## Panel-IV mapping

Panel IV retains twelve outcomes; raw requested positions and the binary
any-application margin are outside its registry. The shared estimation
specification declares those outcomes, the endogenous AEWR, the single
estimated instrument, controls, fixed effects, the clustering field, the
2011--2023 estimation window and the two ordered columns (without and with
controls).
Its reader constructs the normalized linked-worksite-business and farm-income
outcomes locally and rejects an unrecognized timing, fixed-effect, model, or
derived-outcome contract.

{{ <grounding path="code/designs/panel_iv/specification.R" anchor="panel-iv-outcome-and-input-registry" sha256="fcf3f7ceab0c7e1bf9cd236bab3f3783df7fa48e3b061fb58b095aff6fd98020" /> }}

The model treats real AEWR `aewr_ppi` as endogenous and uses the single
`z_wage_seasonal_composition_real` instrument; the wage-only calibration and
its `z_wage_only_real` field are no longer constructed. The instrument uses
source-year t-1 wages and deterministic recovered county weights on the original
observed-wage donor support. QCEW 111/112 supplies employment, seasonality and
composition; OEWS supplies a mapped reporting-area hourly proxy, not a direct
county farm wage. No QCEW or BEA wage fallback is used.

The fixed-wage revision and its later separation/innovation exercises were
retired by user instruction, as were the wage-only instrument and the
`continuing` control set. The sole
pipeline is stages 01--11 in R, including frame construction, deterministic
recovery, validation, a diagnostic Kitagawa decomposition of the
instrument's variance into weight, wage and interaction components, and a
diagnostic of that decomposition's sensitivity to county-level QWI donor
wages, which never replace the OEWS proxy. Simulated weight
draws and their fields/guards are removed. The existing prior-construction
limitations remain documented in the [panel-IV design](@/designs/panel-iv.md).

County/year FE, equal county-year weights, lagged log population, lagged
farm-employment share, lagged employment/population, lagged real p10 and static
H-2A propensity × `year - 2011` are the only control set. Its
BEA employment ratios end in 2022, so the lagged controls and every common
sample end in policy year 2023. The instrument and the branch panel are still
built for every policy year through 2024; the 2024 rows are not estimated.
Column 2 is the preferred controlled model. Target-subregion CRV1
inference is conditional on the generated inputs. Recovered distributions are
not observed sampling weights, and FLS calibration does not establish exclusion.

{{ <grounding path="code/designs/panel_iv/design.R" anchor="panel-iv-design-contract" sha256="4a662b10cda761571109a03086866464b2d10a58f019a02fe917dbf0825d7f51" /> }}

## AEWR-border mapping

The standalone border branch estimates certifications per fixed 2011 farm
worker (`h2a_cert_share_farm_workers_2011_start_year`) against current real
AEWR (`aewr_ppi`) over the user-approved 2011--2025 baseline, with county and border-component-year FE.
The coefficient is a direct continuous-wage slope; it is not an IV or pointwise
spatial RD estimand. County adjacency does not supply worksite distances or
independent policy assignments. The branch adds no causal outcome to the union.

The original four geographies/seven models and their entire v5 audit remain
benchmarks. The old endpoint screen requires 7.5% 2007 cropland, 30 mean
2008--2010 farm jobs per 100 square miles and 90% measurement coverage. Original
one-ring, core, two-ring, stacked, side-weight and baseline-trend comparisons,
annual coefficients, pair deletion, balance and exact FWL decompositions remain
available. Stage04's primary references now mean its v5 diagnostic benchmark.
Its old 2008--2010 wage placebo is retained as a contaminated historical audit.

{{ <grounding path="code/designs/border_discontinuity/04_estimate_models.R" anchor="border-discontinuity-estimation-contract" sha256="9edb28bc44a6abd7f44ded792bcfbf2d1ad3f60d9b6908847a14d6332c59ad4a" /> }}

The bounded v6 structural candidate applies the SAME frozen agricultural
thresholds to every member and requires final locality within one own-region
adjacency step of a surviving qualifying edge, plus the declared noncore
similarity caliper. This changes the local population without selecting on
policy-period participation. Sensitivities use fixed preemployment WLS weights,
standardized baseline outcome mean × linear time, and geography-only
region-disjoint matching. The 2011 denominator stays unchanged; preemployment
weights do not turn the outcome into a ratio of aggregate certifications to
preemployment. The history adjustment imposes an additional trend restriction.

{{ <grounding path="code/designs/border_discontinuity/07_estimate_redesign.R" anchor="border-redesign-estimation-contract" sha256="26cafd82f053db6e9061ad96da75a6f6c8f381717fa58cb6e0ca59398a161683" /> }}

Pair CRV1, WCR11-FNW with 9,999 Webb draws/seed201103, CV3J and raw-row
pair-by-own-region CRV1 remain comparisons. New dyadic covariance allows either
shared endpoint; its small-node correction/reference law is a sensitivity,
not a finite-sample guarantee. Incident-region deletion is influence, not an
ordinary node jackknife. Region matching removes endpoint reuse but not all
spatial dependence. Signed size/power/MDE simulations retain negative-variance
failures and binomial uncertainty; frozen DGPs and limited prehistory cannot
establish policy-gap exogeneity. More counties or bootstrap draws add no
independent regional wage assignments.

The recorded 2009 shared wage is an average certified-application wage, not a
statutory AEWR schedule. New wage diagnostics use 2008/2010 only; primary power
calibration uses 2008--2010 outcome residuals after FE only, with no wage
regressor. See the [shared-panel source limitation](@/contracts/shared-panel.md).
The current inputs remain preserved; correction belongs to the source producer
and a coordinated downstream rebuild. AEWR already existed before 2011, so even
a corrected early slope is not guaranteed to be zero.

The retained `outputs/reports/border_redesign.md` distinguishes implementation
from causal adequacy and informative precision. It does not supply a manuscript
claim or welfare elasticity automatically. The full border contracts are in the
[border design](@/designs/border-discontinuity.md).

The separate `statewide_design.json` registry (stages 12--13) declares two
outcomes for the 2025-10-02 statewide-AEWR state-border design, which is
power-only so far. The quantity outcome is fiscal-year certified positions
(`nbr_workers_certified_fiscal_year`) per fixed 2011 farm employment. The wage
first stage is the hours-weighted hourly offered wage by contract start year,
`nominal_offered_wage_bill_certified_start_year /
cert_hours_with_hourly_wage_start_year`, on counties with hourly offers in every
calibration year. Treatment is the nominal rate in effect on April 1 of each
year. Both b01 wage definitions keep offers in the (0, 100] dollar range; the
offered wage additionally requires an hourly pay unit. The upstream
`mean_hourly_wage_fiscal_year` is worker-weighted over certified workers with a
valid wage and is not used.

## Mundlak-Chamberlain mapping

MC version 4.1.0 jointly estimates five primitive H-2A outcomes on one common
design and sample: applications, balanced v2 linked worksite businesses,
certified positions, certified hours, and any application. It then
constructs per-worker rates, positions per application, hours per position,
raw-log-AEWR elasticities, and percent-of-observed-mean effects per declared
treatment unit after fitting. Only untransformed `aewr_log_level` and
`aewr_log_change` coordinates receive elasticity labels because their 2010-level
or preceding-year comparison enters additively and drops out of the own-coordinate
log-AEWR derivative. A ratio of fitted
aggregates is not confused with an average conditional unit ratio, and the
supported fit never selects a sample using a positive post-treatment outcome.

The treatment registry contains separate linear 2011-through-outcome-year
coordinates for 2012–2022 outcome rows and a current-plus-one-lag benchmark for regional AEWR levels,
frozen county bite, and predetermined county exposure times the regional path.
It contains no polynomial dose or lag terms, cross-dose products, imposed
trends, nonlinear outcome models, randomization inference, or bootstrap.

The model also needs:

- county, state, commuting-zone, AEWR-region, strict-market, and declared
  sensitivity-cluster identifiers, all stored as strings;
- positive mean 2008–2010 farm employment for eligibility and post-fit
  constructed per-worker quantities, rather than a fixed-2011 filter;
- 2008–2010 baseline means and their nested county/market/state/region Mundlak
  components, with categorical calendar interactions but no imposed trend;
- complete 2010–2022 real and nominal AEWR paths, applicable agricultural
  minimum wages, and frozen 2008–09/2008–10 wage quantiles used by the declared
  bite approximation;
- the common causal-first full-rank basis and exact full-model leverage; and
- analytic HC3 and cluster covariances plus an explicitly experimental scalar
  continuous CCV-HC3 comparator.

The executable Python contract is the authority for outcome columns,
treatment families, history rules, moderators, fixed effects, clusters,
rejected methods, and inference status.

{{ <grounding path="code/designs/mundlak_chamberlain/mcw/design.py" anchor="mundlak-design-contract" sha256="cab1980b1699b44ddd46f93e438301aa3fe9d505224b22bb429a3cf6efef509c" /> }}

## Variables that are not causal outcomes

The following are required inputs or diagnostics, not outcomes in the current
supported estimators:

- AEWR levels, AEWR wage gaps, AEWR growth doses, and excluded instruments;
- prediction scores, treatment groups, post indicators, and moderators;
- population, employment shares, local wages, baseline histories, and trend
  interactions used as controls or correlated-effects projections;
- BEA all-industry wage-and-salary jobs, farm wages and salaries, and farm wage
  supplements; these are shared source totals, not average-wage outcomes unless
  a design defines a positive hired-job denominator and promotes the result;
- the shared nominal OEWS Big-Six hourly- and annual-wage proxies and their
  geographic and publication-support fields, joined one-to-one by county-year
  but not treated as outcomes unless an executable design registry explicitly
  promotes them;
- separate QCEW NAICS 111/112 employment and wage bills, all-sector QCEW
  totals, quarterly QCEW employment, FLS workers, and FLS wage rates used as
  disclosed instrument features or targets rather than outcomes;
- county/year fixed-effect identifiers, clusters, reassignment states, and
  design weights;
- fixed-2011 farm employment and positive application/position counts used as
  denominators or sample rules in other supported designs; and
- conservative/high-recall linked-worksite-business counts, log outcomes,
  support measures, first-stage statistics, placebos, and other diagnostics not
  present in an executable outcome registry.

Do not promote an auxiliary field to a causal outcome, change a denominator,
or condition the sample on a downstream response without updating the owning
design, this inventory, its design page, and the retained output contract.

## 2025 policy-shift estimands

The [separate 2025 design](@/designs/policy-2025.md) retains the inspected
exploratory wage and earnings endpoints below. Organizational promotion to
its own branch does not establish causal validity or change the thirteen
outcomes in the established causal-outcome union.

| Endpoint | Construction and comparison |
| --- | --- |
| Nominal hourly offered wage | Continuing filer–state–occupation mean-offer change, FY2025–FY2026 decisions through June 30; U.S. skill-I dollar dose, former-region effects, baseline-hours primary weights |
| Covered crop weekly earnings | Mean YoY log change in private NAICS 111 average weekly earnings for 2025Q4/2026Q1; frozen intensity × dollar cut, intensity control, state effects, equal county weights |

Wage retention/reduction, reference-premium distributions, composition,
selection and applicability measures are diagnostics. Quarterly earnings
require all current/prior observations, positive disclosed earnings and at
least ten employees in every quarter; state samples require two counties.
Animal, all-private and QWI stable-monthly earnings retain separate labels
and diagnostic roles. No weekly-to-hourly conversion is made. UI splits
do not establish a domestic-worker-only population. The original complete
three-quarter target remains distinct from the authorized interim window.
Both components use state deletion jackknife t(G−1) inference, retain
unidentified fits/deletions and preserve the entire declared sensitivity menu.

## Historical wage-reform diagnostics

The [historical wage-reform branch](@/designs/wage-reforms.md) has a separate
`diagnostic_outcomes` registry. It does not add a supported causal outcome or
produce a causal wage elasticity. Its unit is a source case-year or a fixed
baseline worksite business, rather than the county-year causal-outcome union.

`certified_participation` is any positive-certified use by a fixed FY2008 or
FY2011 baseline business in a disclosure fiscal year, including subsequent jobs
outside its baseline type. The denominator is the fixed incumbent cohort;
absence is zero certification, not proof of closure. `same_type_participation`
audits changes in reported job/pay type. `log_wage` is the log median positive
nominal offer over distinct business-case-wage observations in the baseline
type, and stays missing on nonparticipation. Comparing two observed wages
conditions on survival and is explicitly diagnostic. It cannot be combined
with the zero-inclusive participation coefficient into a Wald elasticity.

Case-level `certified_positions` uses one positive singleton/master total per
fiscal-year/case. All-case weekly filing counts include nonpositive disclosures
with a unique valid receipt date. Neither counts unique workers or visas, and
neither assigns full joint-case totals to every member business. Early requested
positions and weekly hours remain missing. Baseline group, geography and years,
the local-window menu, FE, equal weights, clusters, conditional MDE formula,
identity-reachability audits and promotion gates are explicit in the branch
registry and design page.

The common input calendar is now 2008--2025. Each regression continues to use
its existing complete-case rule, so missing source controls can shorten its
estimation window. Event-study code applies that same rule before expanding
year interactions: this avoids a fixest dimension error when an entire year
has missing regressors. It does not fill controls or change the 2011 reference.
