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

The canonical union contains thirteen causal outcomes. Individual designs use
different subsets and sometimes different units, so similar labels are not
automatically interchangeable across designs.

## H-2A utilization and adjustment outcomes

| Causal outcome | Executable column or construction | Supported designs |
| --- | --- | --- |
| Certified positions/workers relative to fixed 2011 farm employment | Shared: `h2a_cert_share_farm_workers_2011_start_year`; MC: `mc_y_certified_positions_per_1000`, which multiplies the same source count by 1,000 before dividing by `emp_farm_2011` | DiD primary and event study; panel IV; MC |
| Certified hours relative to fixed 2011 farm employment | `h2a_cert_hours_per_farm_worker_2011_start_year`; MC equivalent: `mc_y_certified_hours_per_worker` | Panel IV; MC |
| Applications relative to fixed 2011 farm employment | Shared: `h2a_applications_per_farm_worker_2011_start_year`; MC: `mc_y_applications_per_1000`, which reports per 1,000 baseline workers | Panel IV; MC |
| H-2A employers using balanced linkage | Source: `nbr_employers_balanced_start_year`; panel IV divides by `emp_farm_2011`; MC models the raw count as `mc_y_employers` | Panel IV; MC |
| Certified positions per application | `h2a_cert_positions_per_application_start_year`; MC equivalent: `mc_y_positions_per_application` | Panel IV; MC |
| Certified hours per certified position | `h2a_cert_hours_per_position_start_year`; MC equivalent: `mc_y_hours_per_position` | Panel IV; MC |
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

{{ grounding(path="code/c01_clean/13_merge_county_panel.R", anchor="county-year-merge", sha256="972f599deffedb5dfd467f7d965bd0a0fcecb713dd3383a545c8690c467f247a") }}

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

{{ grounding(path="code/designs/panel_iv/07_estimate_panel_iv.R", anchor="panel-iv-outcome-and-input-registry", sha256="08334d9cb7d1517baef359f577c13a84deb7a3e6824926eb453242ca288f5cc3") }}

The model treats real AEWR `aewr_ppi` as endogenous. It requires the wage-only
and wage-plus-seasonal excluded instruments, with the latter preferred. The
controlled specifications add lagged log population, lagged farm-employment
share, lagged employment-to-population ratio, lagged real p10 wage, and the
standardized static H-2A propensity interacted with `year - 2011`.

All specifications include county and year fixed effects and cluster by the
AEWR-region-by-target-subregion identifier `aewr_iv_cluster_id`. Each outcome
uses its own complete-case sample, but all four instrument/control columns for
that outcome must use identical observations.

{{ grounding(path="code/designs/panel_iv/design.R", anchor="panel-iv-design-contract", sha256="c4b4dc58eb8f69d52e674403bbca8668e5b0720f6618d8664af4a7d14a88c650") }}

## Mundlak-Chamberlain mapping

MC estimates the seven H-2A outcomes and does not currently estimate the six
farm-economic outcomes. Four volume outcomes—applications, balanced-linkage
employers, certified positions, and certified hours—are flagged as primary
totals. Any application and the two conditional adjustment ratios are
additional margins. Positions per application require positive applications;
hours per position require positive certified positions.

Its causal treatment is AEWR growth in log percentage points, with current,
one-year-lag, and two-year-lag coordinates plus the declared polynomial and
interaction basis. The main predetermined moderator is standardized mean
2008–2010 AEWR bite.

The model also needs:

- county, state, commuting-zone, AEWR-region, and strict market identifiers;
- positive fixed-2011 farm employment for eligibility and scaling;
- the 2008–2010 means and trends of baseline H-2A intensity, AEWR bite,
  population, employment structure, farm-income composition, low wages,
  cropland, and predicted H-2A intensity;
- separate 2008, 2009, and 2010 Chamberlain histories for the declared subset
  of selection variables, plus multilevel county/market/state/region components
  constructed from baseline summaries and histories;
- the 2011–2022 AEWR treatment history and region-level reassignment states;
  and
- model-specific year/region structure, resource guards, and finite-design CCV
  reference law.

The executable `MC_OUTCOMES` registry is the authority for outcome columns,
labels, sample rules, effect units, and primary-total status. The same design
contract declares the treatment basis, baseline variables, moderator,
hierarchy, model ladder, and inference reference.

{{ grounding(path="code/designs/mundlak_chamberlain/design.R", anchor="mundlak-design-contract", sha256="c70937a1d9f59124bb62ddbc5b6c80e313545b905310cff064791c1851db6e93") }}

## Variables that are not causal outcomes

The following are required inputs or diagnostics, not outcomes in the current
supported estimators:

- AEWR levels, AEWR wage gaps, AEWR growth doses, and excluded instruments;
- prediction scores, treatment groups, post indicators, and moderators;
- population, employment shares, local wages, baseline histories, and trend
  interactions used as controls or correlated-effects projections;
- county/year fixed-effect identifiers, clusters, reassignment states, and
  design weights;
- fixed-2011 farm employment and positive application/position counts used as
  denominators or sample rules; and
- requested positions, conservative/high-recall employer counts, log outcomes,
  support measures, first-stage statistics, placebos, and other diagnostics not
  present in an executable outcome registry.

Do not promote an auxiliary field to a causal outcome, change a denominator,
or condition the sample on a downstream response without updating the owning
design, this inventory, its design page, and the retained output contract.
