+++
title = "AEWR-border redesign"
description = "Bounded local border redesign, preserved benchmarks, dependence and power evidence, and identification limitations."

[extra]
scopes = ["code/designs/border_discontinuity", "scripts/run_border_discontinuity.sh"]
+++

The v6 registry investigates a stricter agricultural border population while
preserving the complete v5 ring experiment. The structural candidate is
`ring_h1_screened_disjoint`; its designation follows locality and agricultural
support, not the estimated coefficient or p-value. Empirical adequacy is a
separate gate. The generated `outputs/reports/border_redesign.md` reports the
bounded investigation and its explicit feasibility finding. This branch remains
outside `run_all.sh`, panel IV, the manuscript and welfare.
The subsequent [data-input comparisons](@/designs/border-data-inputs.md) correct
upstream wage semantics, preserve site-date allocation evidence and compare
fixed labor denominators in a separate registry. Their runner rebuilds this
branch after shared-source changes; external acquisition remains explicit.

## V6 changes and roles

The original four geography variants and seven models remain in stage04 with
the original annual/placebo/balance/influence/FWL audit. Its “primary” references
now mean the **v5 benchmark**, not an approved causal specification.
Stage07 implements four new policy comparisons:

| Model | Population, weight or adjustment |
| --- | --- |
| `screened_h1__base` | Every member passes the old endpoint agricultural screen; final retained locality is at most one own-region adjacency step from a surviving eligible edge |
| `screened_h1__preemployment_weight` | Same geography, fixed mean 2008–2010 farm-employment WLS weights |
| `screened_h1__baseline_history` | Same geography, standardized 2008–2010 outcome mean interacted with linear time |
| `screened_h1__region_disjoint` | Maximum-cardinality region-disjoint matching of the screened geography; ties maximize qualifying core edges then minimize lexical pair IDs |

The all-member screen is applied before disjoint assignment. After assignment,
locality is pruned until every retained county has a path within its retained
own-region counties to a surviving qualifying border endpoint. Noncore members
must also satisfy the frozen feature-distance caliper relative to a surviving
opposite core. Surviving eligible core endpoints keep the caliper exemption.
Connected components must span both regions and include an eligible direct
cross-region edge. All exclusions and unmatched candidates remain in the
geography audit. The old geographies follow their original construction exactly;
the new final-locality condition is not retrospectively imposed on them.

The treatment, 2011–2022 period, county/component-year FE, certifications outcome
and fixed-2011 denominator remain the benchmark convention. Fixed preemployment
weights change the population-weighted slope; they do not create a ratio of total
certifications to preemployment. The history term is an explicit additional
counterfactual trend restriction. With balanced support and time-invariant
weights, the unadjusted model exactly equals a regression of side-outcome gaps
on wage gaps, with component intercepts and harmonic side-total weights.
The numerical check is retained. A new label for this equivalent regression
does not create new assignments.

The redesign was informed by old outcome concentration, so it is not an untouched
preregistration. It reuses the frozen screen thresholds for a substantive
agricultural-population restriction; it does not drop a county based on its
outcome. The full bounded menu, roles, assumptions and choices are recorded in
`design.json` before new policy-outcome comparisons. Fixes triggered by source
validation are documented separately from model selection.

{{ <grounding path="code/designs/border_discontinuity/07_estimate_redesign.R" anchor="border-redesign-estimation-contract" sha256="977d64f3bc6431c7f3ae48da30ed2a71e6c5a3217ae24e8dc4c6266bf80f3d5b" /> }}

## V6 dependence and precision

Every pair shares both endpoints' policy paths with other incident pairs.
The branch preserves pair CRV1, 9,999-draw WCR11-FNW Webb inference and CV3J,
and the existing raw-row pair-by-own-region two-way CRV1 sensitivity. It also
reports partial-treatment dyadic score covariance, allowing any two pairs with
either endpoint in common. Negative raw dyadic variance is unavailable, not
silently repaired. The R/(R−1) correction and t_(R−1) reference are explicit
small-node sensitivities; the dyadic asymptotic result is not a finite-sample
proof for a small geographic graph. Region deletion removes every incident pair
and is labeled influence rather than a jackknife reference law. Region-disjoint
matching removes direct endpoint sharing but not all spatial dependence.

The declared meaningful effects are 5, 10 and 20 certifications per 1,000
fixed-2011 farm workers per $1/hour of real AEWR, with 10 the central assumption.
They are working thresholds, not user-supplied welfare values. Power targets
0.80 at two-sided alpha0.05. The simulation fixes observed treatment/support and
injects signed effects independently of the observed coefficient. It uses
999 simulations and 999 calibration draws, seed201103, one thread, AR(1)rho0.5,
and four frozen county/region/signed-pair/spatial noise processes. A signed MDE
grid runs from0.001 through0.20. A procedure needs no invalid covariance draws
and an upper95%Wilson null-rejection bound at most0.075 before its MDE is labeled
size-valid. This gate does not establish exogeneity.

A scalar WCR supplement uses the same first199 simulated draws for the old h1,
new base and region-disjoint base; each test retains9,999 Webb draws. Its score
algorithm must match the locked package numerically. The fitted-history model
is not assigned a scalar bootstrap shortcut. Raw two-way simulation variances
remain invalid if negative, whereas the observed benchmark also retains the
package's PSD-adjusted comparison; these are explicitly distinguished.

Primary noise calibration projects only 2008–2010 outcomes on unit and
component-year FE, without a preperiod wage regressor. Full-policy-residual
calibration is an explicitly retrospective stress sensitivity. These assumed
noise processes and short historical data do not establish causal identification
or transportability of the variance. All model-specific null rejection rates,
binomial uncertainty, invalid variances, signed power curves and MDE failures
are retained. The generated report attributes precision differences to support,
noise, weights, populations and inference, with common-county-year comparisons.

## Historical source and identification limitations

The source correction makes canonical 2009 AEWR missing and retains application
means only in a named nonstatutory field. Stage04 reconstructs its historical
contaminated wage comparator explicitly, including the old baseline wage
standardization. The persistent panel and stage07 retain canonical wages.
New wage diagnostics still use 2008/2010; three-year outcome noise calibration
now explicitly permits missing wage values because its projection uses FE only.
Policy-period treatment and the original v5 results remain numerically testable.
See the [AEWR input contract](@/contracts/aewr-inputs.md).

AEWR existed before2011, and its wage methodology changed during2009–2010.
Consequently, even corrected early data are not a no-policy untreated period.
A nonsignificant early coefficient is not proof of parallel counterfactual
trends. The direct coefficient requires cross-boundary wage-gap changes to be
orthogonal to untreated outcome-gap changes conditional on the declared FE and
adjustment. FLS-based wage changes may reflect the same regional labor-demand
conditions that change H-2A utilization. State policies, sorting, anticipation,
differential trends and cross-border interference remain threats. Region-year
FE absorb the entire policy treatment and cannot preserve the direct slope.

County adjacency and aggregate outcomes support a local border panel, not
pointwise spatial RD. The implemented county comparison uses public labor
denominators and requires no farm-level nonparticipant records. Such records
would be needed for a farm-participation estimand, which is not pursued through
unavailable non-public NASS data or custom tabulations. A separately declared
public-data spatial extension could study certification capacity per
predetermined agricultural area, retaining zero-certification areas in its
geographic frame. It would require true worksite coordinates, audited allocation,
boundaries/coverage, exact wage-floor dates and competing policies, plus
informative policy innovations. More rows do not create new AEWR assignments.
The sourced methodological comparison and precise next-data requirement are in
the generated report. Success is credible identification plus informative
precision, including a precise null; significance is not the objective.

## Preserved v5 estimand and timing

For policy years 2011--2022, the outcome is
`h2a_cert_share_farm_workers_2011_start_year`: certified H-2A positions divided
by the county's fixed 2011 farm-employment baseline. The treatment is current
real AEWR, `aewr_ppi`. The coefficient is the change in certified positions per
2011 farm worker associated with a one-2012-dollar increase in the applicable
hourly AEWR, conditional on county and border-component-by-year fixed effects.

In a disjoint sample the estimating equation is

$$
Y_{it} = \beta W_{r(i)t} + \alpha_i + \lambda_{s(i)t} + \varepsilon_{it}.
$$

Identification comes from counties in different AEWR regions that share a
retained border-pair component in a given year. This is a direct wage design,
not an IV design. It requires cross-border wage changes to be conditionally as
good as random within stratum and year. Region-specific shocks, policy
endogeneity, anticipation, sorting, spillovers, and differential trends remain
threats.

## Predetermined agricultural and similarity screens

The design first screens each directly adjacent cross-AEWR county pair, not an
entire AEWR border. Both endpoint counties must satisfy all of the following:

- 2007 cropland is at least 7.5 percent of 2010 county land area;
- mean 2008--2010 farm employment is at least 30 jobs per 100 square miles of
  2010 county land area; and
- measurement coverage is at least 90 percent for both inputs. With one
  cropland observation and three annual employment observations, this requires
  observed cropland and all three employment years.

The share and density rules normalize support by the size of each endpoint
county. There is no raw county-count or total-employment threshold. A county
pair passes only if its weaker endpoint passes every rule. This screen is
applied before ring expansion, the feature-distance caliper, and disjoint
assignment. The complete set of passing and failing cross-border county pairs
is retained as an audit artifact.

This restriction changes the target population from all AEWR borders to
agriculturally relevant county-pair segments of those borders. It does not
establish that a county's agriculture lies immediately beside its shared
boundary; county aggregates cannot make that finer claim. No post-2010 H-2A
outcome, wage treatment, or employment response enters the screen.

County similarity uses only 2008--2010 crop composition, climate normals, and
static soil measures. Crop shares are formed within year and then averaged;
compositional shares receive square-root transforms. Missing values are
globally median-imputed, columns are standardized, and each source block is
scaled by the square root of its number of columns.

The border core consists of 2010-vintage counties joined by a direct adjacency
edge whose endpoints have different AEWR-region identifiers and pass the
agricultural county-pair screen. Region pairs are the initial strata. Ring
depth is shortest-path distance from the eligible county-pair endpoints within
the county's own AEWR region.

The global distance caliper is the 90th percentile of standardized feature
distance across eligible direct cross-border county pairs. A non-core ring
county is eligible only when its distance to at least one eligible core county
on the opposite side is no larger than that frozen caliper. Eligible core
counties are always retained.

When a county is eligible for multiple border pairs, a disjoint sample chooses,
in order, the smallest ring depth, smallest feature distance, and
lexicographically smallest region-pair identifier. Each induced pair graph is
then split into connected components. A component is retained only if it spans
both AEWR regions and contains an internal direct cross-AEWR county pair that
passed the agricultural screen.

The declared geographies are:

- `ring_h1_disjoint`: one-ring primary window;
- `ring_h0_disjoint`: adjacent border counties only, as a benchmark;
- `ring_h2_disjoint`: two-ring locality sensitivity; and
- `ring_h1_stacked`: overlap sensitivity retaining every supported one-ring
  pair assignment. County weights sum to one across duplicated memberships,
  and membership-by-county replaces county as the unit fixed effect.

## Registry, weights, and inference

All four geographies receive a base policy-period regression with equal county
weights. The primary one-ring geography also receives:

- a weighting sensitivity giving equal total weight to each
  stratum-by-AEWR-region side;
- a differential-trend sensitivity interacting calendar time with standardized
  2008--2010 means of log population, farm-employment share,
  employment-to-population ratio, and real local p10 wage; and
- a 2008--2010 preperiod placebo using the same outcome and real AEWR.

Rows with missing required variables are dropped model by model, after which
singleton fixed-effect groups are removed iteratively. The sample audit records
each loss.

The three primary inference procedures cluster by `region_pair_id`, the
two-region AEWR border pair underlying the local comparison. In the primary
sample, disconnected qualifying segments of the same AEWR border remain in
the same inference cluster because they share the same region-year treatment
contrast.

The standard cluster-robust covariance is the usual finite-sample-adjusted
cluster sandwich, conventionally called CRV1. Preferred inference uses a null-imposed wild
cluster bootstrap, conventionally called WCR11, computed by the Fast and Wild
(`fnw11`) algorithm with Webb weights, 9,999 draws, seed 201103, and
test-inverted 95 percent confidence intervals. The bootstrap reports a test
p-value and inverted interval, not a conventional standard error. Before
bootstrapping, the code absorbs unit and stratum-year effects with `fixest`,
applies the square-root weight transformation, and verifies that the resulting
Frisch--Waugh--Lovell regression reproduces every reported coefficient.

WCR11 is not assumed to dominate the standard cluster-robust calculation. The
two procedures can give materially different p-values. Both are retained, and
a conclusion that depends on their disagreement is treated as an inference
warning. A delete-one-border-pair cluster jackknife, conventionally called
CV3J, is retained as a third comparison. It refits the identical model after
removing each border pair and iteratively removing singleton fixed-effect
groups, then forms its standard error from the dispersion of the $G$
leave-one-pair-out coefficients. Its tests and 95 percent intervals use a
$t_{G-1}$ reference distribution. This is a cluster jackknife, not another
wild bootstrap. Whichever method gives the smallest p-value is not selected
after seeing the results. AAIW/TSCB is not reported because that
binary-treatment procedure does not apply to this continuous real-wage
fixed-effect estimand.

Because the same AEWR region can appear in several border pairs, the model
registry also reports a two-way CRV1 sensitivity clustered simultaneously by
`region_pair_id` and `aewr_region_id`. This inclusion-exclusion covariance
allows dependence within either dimension and uses the smaller cluster count
for the finite-sample reference distribution. It is a standard two-way
cluster-robust calculation, not a two-way wild bootstrap or two-way CV3J
procedure, and it does not replace the three border-pair inference procedures.
The output records both cluster counts, pair-by-region intersection counts,
and whether positive-semidefinite covariance repair was required.

Stage04 selects `design$diagnostic_audit$primary_spec_id` explicitly, preserving
the old h1 benchmark even as new geography roles are introduced. Its primary
registry does not follow the v6 structural candidate's role label.

{{ <grounding path="code/designs/border_discontinuity/04_estimate_models.R" anchor="border-discontinuity-estimation-contract" sha256="9edb28bc44a6abd7f44ded792bcfbf2d1ad3f60d9b6908847a14d6332c59ad4a" /> }}

## Bounded credibility audit

The primary specification has a prespecified diagnostic audit rather than a
search across new bandwidths, weights, or outcomes. It retains:

- annual 2008--2022 treatment coefficients with standard cluster-robust, wild
  cluster bootstrap, and delete-one-pair jackknife inference, so the preperiod
  and policy path can be inspected directly;
- the explicit 2008--2010 placebo;
- a model-by-model comparison of the three border-pair procedures plus the
  two-way CRV1 sensitivity;
- leave-one-border-pair-out estimates and the border-pair cluster jackknife;
- absorbed-regressor treatment leverage, cluster score contributions, and an
  effective inference-cluster count;
- every primary stratum-year's cross-region treatment support;
- exact Frisch--Waugh--Lovell coefficient, score, and leverage decompositions
  by county-year, county, border-stratum-region-year, AEWR region, and border
  pair; and
- observed requested positions, certified positions, applications, the
  balanced v2 linked-worksite-business count, and their annual changes alongside those
  decompositions.

The business margin is the supported balanced v2 county-year count in the
shared panel. Business identities are not carried into this analysis panel, so
the branch does not claim to identify which business changed behavior. Sums
across counties are labeled as summed linked worksite-business counts because
one linked business operating in multiple counties can appear more than once.

These outputs diagnose whether the result is concentrated in particular AEWR
regions or border pairs, or depends on the finite-cluster procedure. They do not cure weak
identification, establish parallel counterfactual trends, or authorize
relabeling a sensitivity as primary.

## Statewide-AEWR state-border design (separate registry)

The interim rule effective 2025-10-02 replaced regional FLS field-and-livestock
AEWRs with statewide OEWS-based rates by skill level. Adjacent states in the
same former FLS region had identical AEWRs before FY2026 and different ones
afterwards, so their borders are new treatment contrasts and states, not FLS
regions, become the policy nodes. `statewide_design.json` declares this design
before any post-policy outcome is parsed:

- units are counties adjacent to a contiguous state border, assigned disjointly
  to the border with the most qualifying cross-border neighbors. The primary
  geography keeps borders inside a former FLS region and applies the frozen
  agricultural screen to both endpoints; unscreened and all-border geographies
  are sensitivities;
- the treatment is the rate in effect on April 1 of each federal fiscal year:
  the regional reference before FY2026, then the statewide U.S.-worker
  skill-level-I rate (H-2A skill-I and U.S. skill-II rates are sensitivities),
  in nominal dollars per hour;
- the quantity outcome is fiscal-year certified positions per fixed 2011 farm
  worker, with county and border-pair-by-year fixed effects and equal weights;
- the wage first stage is the hours-weighted hourly offered wage by contract
  start year, on counties with hourly offers in every calibration year, with a
  declared 0.25 pass-through threshold (0.5 and 1.0 as sensitivities).

Stage 13 computes design-stage size and power with the unchanged v6 error
processes, seed, effect grid and size gate. The scale comes only from
pre-period FE-only residuals (FY2020--FY2022 primary, FY2015--FY2022
sensitivity), and the actual FY2026 and FY2027 statewide rates are used. FY2026
contains cases decided under the prior rates, so the slope is
intent-to-treat. The rule's national changes, its housing adjustment and other
2025 policy changes are not represented in the simulation, and simulated power
does not establish exogeneity. Estimation needs the upstream producers to parse
the FY2025 and later disclosure files. The generated finding is
`outputs/reports/border_statewide_power.md`.

Run the branch from the repository root with
`./scripts/run_border_discontinuity.sh`. Operational inputs, outputs, and stage
ownership are listed in the branch README.
