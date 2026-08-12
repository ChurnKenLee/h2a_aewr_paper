+++
title = "Causal outcomes and model inputs"
description = "Canonical cross-design outcome inventory and the treatments, controls, moderators, denominators, samples, and inference fields required to estimate them."

[extra]
scopes = ["code/c02_build", "code/designs", "scripts/run_did.sh", "scripts/run_panel_iv.sh", "scripts/run_mundlak_chamberlain.sh"]
+++

This page distinguishes variables that are causal outcomes from variables used
to define treatment, scale an outcome, select a sample, adjust a model, absorb
heterogeneity, construct an instrument, or calculate inference. A field can be
scientifically important without being an outcome.

The canonical union contains fourteen causal outcomes. Individual designs use
different subsets and sometimes different units, so similar labels are not
automatically interchangeable across designs.

## H-2A utilization and adjustment outcomes

| Causal outcome | Executable column or construction | Supported designs |
| --- | --- | --- |
| Certified positions/workers relative to a declared farm-employment baseline | Shared: `h2a_cert_share_farm_workers_2011_start_year`; MC fits raw `mc_y_certified_positions`, then reports a post-fit effect per 1,000 mean 2008–2010 farm workers | DiD primary and event study; panel IV; MC |
| Certified hours relative to a declared farm-employment baseline | Shared: `h2a_cert_hours_per_farm_worker_2011_start_year`; MC fits raw `mc_y_certified_hours`, then reports a post-fit effect per mean 2008–2010 farm worker | Panel IV; MC |
| Applications relative to a declared farm-employment baseline | Shared: `h2a_applications_per_farm_worker_2011_start_year`; MC fits raw `mc_y_applications`, then reports a post-fit effect per 1,000 mean 2008–2010 farm workers | Panel IV; MC |
| Requested H-2A positions/workers | MC fits raw `mc_y_requested_positions` jointly with the other primitive outcomes, then constructs any per-worker rate after fitting with the declared 2008–2010 mean farm-employment denominator | MC |
| H-2A employers using balanced linkage | Source: `nbr_employers_balanced_start_year`; panel IV divides by `emp_farm_2011`; MC models the raw count as `mc_y_employers_balanced` | Panel IV; MC |
| Certified positions per application | Shared: `h2a_cert_positions_per_application_start_year`; MC derives the ratio-of-aggregate effect after jointly fitting raw certified positions and applications | Panel IV; MC |
| Certified hours per certified position | Shared: `h2a_cert_hours_per_position_start_year`; MC derives the ratio-of-aggregate effect after jointly fitting raw certified hours and positions | Panel IV; MC |
| Any H-2A application | MC constructs `mc_y_any_application = 1[nbr_applications_start_year > 0]` and estimates it as a Gaussian linear-probability outcome | MC |

The shared-panel producer supplies the fixed-2011 denominators and the common
ratios used by DiD and panel IV. It retains conservative, balanced, and
high-recall employer counts separately; the balanced count is the supported
causal employer outcome.

{{ grounding(path="code/c02_build/01_build_county_panel.R", anchor="shared-panel-construction", sha256="2b7bc73b507e5b8c67b4964eb693412066519ae003469531e4b6ef2972d9e2df") }}

## Farm-economic outcomes

| Causal outcome | Executable column or construction | Supported designs |
| --- | --- | --- |
| Real Fisher crop price index | `fisher_index_ppi` | DiD; panel IV |
| Farm employment | `emp_farm` | Panel IV |
| Farm production expenses relative to cash receipts and other income | `share_farm_prodexp_cashandinc` | Panel IV |
| Real farm cash receipts and other income per current farm worker | Panel IV constructs `1000 * farm_cashandinc_ppi / emp_farm` as `farm_cashandinc_ppi_per_farm_worker` | Panel IV |
| Hired-labor share of farm production expenses | `share_farm_laborexp_prodexp` | DiD; panel IV |
| Fisher crop output-quantity index | `fisher_quantity_index` | Panel IV |

{{ grounding(path="code/c01_clean/13_merge_county_panel.R", anchor="county-year-merge", sha256="faad687f82c863b634a4a2c847db984773bf48b1f0fe052ec349ecbc3469970a") }}

These are downstream responses to the policy exposure, not controls. Moving
one of them to the right-hand side or using it for sample selection would
change the estimand and requires an explicit design revision.

## Difference-in-differences mapping

The retained DiD outcomes are:

1. normalized H-2A certified positions/workers, which is the primary outcome
   and the event-study outcome;
2. the real Fisher crop price index; and
3. the hired-labor share of farm production expenses.

{{ grounding(path="code/designs/did/02_main_results.R", anchor="did-primary-outcome", sha256="96bbc956b09e9017a67f9fdae375579f17371c9f01d88598e30e9c7a6a846ab5") }}

{{ grounding(path="code/designs/did/05_fisher_price.R", anchor="did-fisher-price-outcome", sha256="7187abfdf20e081257ccef4ce429c815c1c981d692a28ede90188631efb87c80") }}

{{ grounding(path="code/designs/did/06_labor_share.R", anchor="did-labor-share-outcome", sha256="b2467ca3ab9f44066b4e3727e7444cc94c8b84056d3cc822ad14b5bbfea56650") }}

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

{{ grounding(path="code/designs/did/01_build_did_panel.R", anchor="did-treatment-inputs", sha256="107031bcdcda7d02bfff630c95b423659522a4ecdee0637374ecb585dce55bb7") }}

{{ grounding(path="code/designs/did/helpers.R", anchor="did-estimation-contract", sha256="abc47e8490b0857dc03f19e055b214c3a94a4ec866568d0741614007118fdebd") }}

## Panel-IV mapping

Panel IV estimates all twelve outcomes except the binary any-application
margin. Its registry constructs the normalized employer and farm-income
outcomes locally, then requires every outcome together with the treatment,
instruments, controls, identifiers, and prediction metadata.

{{ grounding(path="code/designs/panel_iv/07_estimate_panel_iv.R", anchor="panel-iv-outcome-and-input-registry", sha256="9122a648fbd9208e60f06f4bcfaf8f3ee398de7566687497368e9fac31d1f61d") }}

The model treats real AEWR `aewr_ppi` as endogenous. It requires
`z_wage_only_real` and the preferred
`z_wage_seasonal_composition_real`. Both use a prior-period county-weight
distribution and a county-mapped OEWS-area Big-Six hourly-wage proxy for the
donor wage level. QCEW 111/112 supplies the annual county employment path and
the preferred calibration's three independent FLS-worker/QCEW-employment
seasonal contrasts and quarterly undivided FLS field/livestock-composition
residuals. QCEW and BEA wage totals are not donor wage fallbacks. These source
features construct excluded instruments; none is a causal outcome. The
controlled specifications add lagged log population, lagged farm-employment
share, lagged
employment-to-population ratio, lagged real p10 wage, and the standardized
static H-2A propensity interacted with `year - 2011`.

All specifications include county and year fixed effects and cluster by the
AEWR-region-by-target-subregion identifier `aewr_iv_cluster_id`. Each outcome
uses its own complete-case sample, but all four instrument/control columns for
that outcome must use identical observations.

{{ grounding(path="code/designs/panel_iv/design.R", anchor="panel-iv-design-contract", sha256="e810543e472e5b67b5b0b2a9a8cd051a1718842e13f3932f4d69fb8ed0ed9960") }}

## Mundlak-Chamberlain mapping

MC version 4 jointly estimates six primitive H-2A outcomes on one common
design and sample: applications, balanced-linkage employers, requested
positions, certified positions, certified hours, and any application. It then
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

{{ grounding(path="code/designs/mundlak_chamberlain/mcw/design.py", anchor="mundlak-design-contract", sha256="6278d1708c14e260861bf0fc8639e79310743d6795f2bfb4aab3bc609dccb252") }}

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
- conservative/high-recall employer counts, log outcomes,
  support measures, first-stage statistics, placebos, and other diagnostics not
  present in an executable outcome registry.

Do not promote an auxiliary field to a causal outcome, change a denominator,
or condition the sample on a downstream response without updating the owning
design, this inventory, its design page, and the retained output contract.
