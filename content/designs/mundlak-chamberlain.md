+++
title = "Wooldridge–Mundlak–Chamberlain continuous-dose design"
description = "Version-4 treatment registry, linear-history OLS, estimands, inference, and diagnostics."

[extra]
scopes = ["code/designs/mundlak_chamberlain", "scripts/run_mundlak_chamberlain.sh"]
+++

{{ grounding(path="code/designs/mundlak_chamberlain/mcw/design.py", anchor="mundlak-design-contract", sha256="6278d1708c14e260861bf0fc8639e79310743d6795f2bfb4aab3bc609dccb252") }}

The cross-design distinction between causal outcomes and required model inputs
is maintained in [Causal outcomes and model inputs](@/contracts/causal-outcomes-and-model-inputs.md).

## What version 4 replaces

The supported branch no longer estimates the version-3 program of current,
one-year, and two-year AEWR-growth doses with quadratic/cubic grids, cross-dose
products, separate 2008/2009/2010 histories, imposed trends, or cyclic
17-path reassignment covariance. Those R files remain in the repository as a
historical compatibility record, but `scripts/run_mundlak_chamberlain.sh`
calls only the version-4 Python stages.

Version 4 implements the conflict-resolved late decisions in `ckl_12Aug`:

- pooled identity-link OLS only;
- a full-history family in which every 2011-through-outcome-year dose is a
  separate linear coordinate in the identified history space;
- a current-plus-one-lag benchmark;
- interactions with predetermined moderators, without polynomial dose or lag
  restrictions, cross-dose products, or imposed parametric trends;
- DuckDB for schema-aware source scans, Polars for relational construction,
  frames, and artifacts, and bounded NumPy/SciPy linear algebra after FWL
  compression;
- six primitive outcomes on one common design and sample, with ratios, rates,
  raw-log-AEWR elasticities, and other treatment-unit-normalized effects formed
  after estimation; and
- analytic HC3/cluster covariance plus a transparently experimental continuous
  scalar CCV-HC3 comparator, with no randomization inference or bootstrap.

The exhaustive treatment, FE, moderator, transform, and cluster menu is an
opt-in diagnostic exercise, not a license to select whichever specification
is significant. The bounded compact registry spans the main unresolved
choices without silently naming one as the empirical primary.

The later archive provisionally loosened unit absorption while exploring a
rich explicit Wooldridge–Mundlak–Chamberlain projection, but it did not settle
one final equation. The compact queue therefore starts with transparently
labeled pooled/no-county candidates for log and dollar AEWR levels. Their
untreated-mean dictionary contains an intercept, unrestricted calendar and
AEWR-region contrasts, all pre-period hierarchical components, component by
calendar interactions, and non-region component by AEWR-region interactions.
The causal block uses all thirteen predetermined baseline moderators as deviations
from their AEWR-region means; this centering is not imposed on the nuisance
projection dictionary. Causal-first rank selection keeps every named history
term and only removes redundant nuisance columns. County plus
year/state-year/region-year models remain named sensitivities, particularly
for lower-geography bite definitions; they are not silently presented as the
sole WMC architecture.

The compact queue also makes a finite-support constraint explicit. With the
2008–09 fraction affected, the full exposure history contains an exact causal
dependence after region-by-year or state-by-year absorption, concentrated in
the last outcome-year history block. It is therefore not estimated from
numerical noise. The compact queue fits that window's full history with
county-plus-year effects, retains its one-lag region-by-year benchmark, and
includes both full and one-lag region-by-year models for the 2008–10 window,
whose causal blocks pass the rank guard. The opt-in menu remains subject to
the same guard.

## Data and calendar

The county is the outcome unit. The shared design-neutral panel supplies 2008
through 2022 data and string-valued county, state, commuting-zone, and AEWR
region identifiers. The frozen baseline is 2008–2010. Treatment history begins
in 2011, the first clean regional coordinate after that boundary; outcome rows
cover 2012–2022, as explicitly approved in the archive. This does not make
2011 a special outcome denominator, eligibility year, or assumed policy onset.
County fixed effects force one transparent rank normalization in the
full-history model: the common levels of its 2011 and 2012 history paths are
time invariant over the outcome window, so the 2012 outcome cells are their
references. The one-lag model retains all of its 2012 cells because those paths
become inactive in later years and are not absorbed.

Eligibility for the supported common sample requires positive mean farm
employment over 2008–2010 and a balanced outcome panel. The same pre-period
mean is available for constructed per-worker quantities. The branch does not
use `emp_farm_2011` as a magic denominator or sample filter.

Rhode Island's five counties lack 2009 support. The panel builder records the
actual observation count in each frozen window. The 2008–09 definition is not
silently described as a two-year average for those counties; the 2008–10
definition remains a declared sensitivity.

## Treatment registry

The archive does not select one unique primary treatment unit. Version 4
therefore retains the following named families:

1. cumulative real-AEWR log level relative to the end of the pre-period,
   (100[\log A_{rh}-\log A_{r,2010}]);
2. cumulative real-dollar AEWR level, (A_{rh}-A_{r,2010});
3. annual real-AEWR log change as a sensitivity;
4. direct frozen-distribution county bite for 2008–09 and 2008–10 windows; and
5. predetermined county fraction affected times the regional cumulative log
   AEWR path.

The first three vary at the AEWR-region-by-year assignment level. The latter
two create predetermined continuous variation below the AEWR region and can
coexist with state-by-year or region-by-year fixed effects.

For county (c) in AEWR region (r), the candidate frozen bite is

\[
B^0_{ct}=\int [A_{rt}-\max\{w,M_{ct}\}]_+\,dF_c^0(w),
\]

where (M_{ct}) is the applicable agricultural minimum wage and (F_c^0) is
frozen before the analysis period. The shared panel does not contain the wage
microdistribution. The executable builder therefore uses a declared
five-quantile point-mass approximation from p10, p25, p50, p75, and p90 and
persists that method label. It must not be described as the exact integral.
The fraction-affected exposure is the mean, over the named 2008–09 or
2008–10 window, of the same five point masses lying below that year's
applicable AEWR after the agricultural wage-floor maximum. The separate
baseline bite moderator is the 2008–2010 mean real AEWR-minus-county-p25 gap.
Thus 2011 supplies a treatment-history coordinate but is not a bite moderator
or frozen-exposure normalization year.
Contemporaneous local-wage gaps and Kaitz measures are excluded from the
supported registry because they can mechanically respond to the policy and
local outcomes.

Continuous raw treatment is the default transform. Within-region standardized
continuous treatment, binary median/upper-quartile indicators, and ranks are
exhaustive-menu sensitivities only. Binarization is not required to make CCV
applicable and is not privileged.

## Conditional-mean model

For outcome (j), year (t), history coordinate (h\le t), and a declared
treatment family (D), the full-history causal block is

\[
\sum_{t=2012}^{2022}\sum_{h=2011}^{t}
  1[T=t]D_{ih}\left(\beta^j_{th}+\widetilde Z_{ir}'\theta^j_{th}\right),
\qquad \widetilde Z_{ir}=Z_i-\overline Z_r.
\]

The pooled candidate retains all 77 full-history cells as named causal coordinates;
causal-first selection chooses their maximal nuisance complement. In the
county-FE sensitivities, every (h>2012) coordinate is retained in level, but
for both (h=2011) and (h=2012), the sum over the eleven outcome-year columns is
a time-invariant county path and is exactly absorbed. That basis therefore
omits each path's 2012 outcome cell and names the remaining coefficients as
differences from that cell. Its 75 dimensions are the complete identified
subspace of the raw 77-column history, not a lag-profile compression; an
absolute 2012 current effect is not identified with county FE. An invertible QR
or contrast map may be used internally only within the applicable identified
space. The one-lag benchmark restricts the inner sum to
(h\in\{t-1,t\}) and retains all 22 identified cells. Disagreement between the
two model families is a diagnostic, not an automatic failure.

Pre-period variables enter as means, not separate annual values or linear
slopes. Their Mundlak components telescope across county-within-market,
market-within-state, state-within-region, and region-within-national levels.
The untreated mean interacts those components with categorical year contrasts;
these are unrestricted calendar categories, not parametric trends. The
causal-first rank rule keeps every named dose coordinate and selects only a
maximal identified nuisance complement.

The pooled candidate uses the explicit region/calendar/projection dictionary just
described and no county fixed effects. The sensitivity registry contains
county plus year, county plus AEWR-region by year, and county plus state by
year. Region-specific linear trends are not imposed: the archive explicitly
rejected forced trends, and later assistant experimentation did not override
that user decision. A residualized-treatment variance-share guard flags nearly
absorbed specifications before their numerical noise can be read as a large
coefficient.

## Outcomes and constructed estimands

The common OLS fit contains six primitive H-2A outcomes:

1. application count;
2. balanced-linkage employer count;
3. requested-position count;
4. certified-position count;
5. certified-hour count; and
6. (1[\text{applications}>0]), estimated as a linear probability outcome.

For comparability, the branch reports both models' average current-coordinate
effects over the common identified target 2013–2022. The one-lag model's
identified 2012 current coefficient remains in the coefficient artifact but is
not silently added to that cross-model average. The branch then constructs
applications, requested positions,
and certified positions per baseline farm worker; hours per baseline farm worker;
positions per application; hours per position; raw-log-AEWR elasticities; and
percentage-of-observed-mean effects per declared treatment unit. A ratio of
aggregate fitted totals and an average of unit-level ratios are different
estimands and must retain different names. No ratio outcome is estimated only
among units selected by a positive post-treatment outcome.

Each result row persists its named target, observation count, weight sum, and
weighting rule. Joint cross-outcome HC3, CR0, and CR1 delta-method covariance is
available for every constructed gradient. The experimental scalar CCV
comparator is additionally formed when every nonzero outcome loading is
proportional to one common coefficient contrast. This includes the declared
ratio-of-aggregates gradients because both outcomes share the same current-dose
row gradient. It is not attached to arbitrary multivariate gradients: a scalar
second-moment lambda does not supply a general cross-outcome assignment kernel.

For raw `aewr_log_level` and `aewr_log_change` coordinates,
(D=100\log(AEWR)) up to, respectively, an additive 2010-level or
preceding-year comparison term. That comparison term drops out of the
own-coordinate derivative, so their
conditional-design elasticity is therefore

\[
\varepsilon
=\frac{\partial Y/\partial\log(AEWR)}{\bar Y}
=100\frac{\partial Y/\partial D}{\bar Y}.
\]

The observed mean is fixed and must be computed on exactly the same target
population and weights as the derivative. County-year, region-year,
predetermined-employment weighted, and aggregate-total targets are not
interchangeable.

For all other treatment families and for every non-raw treatment transform,
the retained normalization is

\[
100\frac{\partial Y/\partial D}{\bar Y},
\]

reported as `percent_of_observed_mean_per_treatment_unit`. It is not called an
elasticity because the registry does not supply a valid unit-specific chain
rule from those coordinates to a proportional AEWR change.

The central causal interpretation is a supported own-dose response under the
rich conditional-mean restriction. A cross-unit derivative comparing units
that receive different doses can also reflect selection on gains; this
parameterization does not by itself identify a global average causal response
curve. Any welfare calculation must consequently remain a local
sufficient-statistic exercise rather than a global extrapolation.

## Analytic inference

Let the full design be (X=[Z,D]), with all fixed effects and nuisance
regressors included in its column space, and let
(B=(X'X)^{-1}). The supported robust comparator is full-model HC3:

\[
\widehat V_{HC3}
=B\sum_i x_ix_i'
  \frac{\hat u_i^2}{(1-H_{ii})^2}B,
\qquad H_{ii}=x_i'Bx_i.
\]

The implementation computes the exact fixed-effect leverage contribution for
the balanced nested panels and adds the selected within-design contribution.
The archive prototype's partial-regressor leverage is not used.

CR0 and a declared conventional CR1 correction are reported at each registry
partition. Dense full-model CR2 is a guarded small-design oracle. CV3 requires
literal leave-one-cluster-out refits that rebuild the fixed-effect projection;
the FWL-conditional Woodbury shortcut is not called exact.

For a scalar reported contrast, the experimental continuous CCV comparator
uses

\[
\widehat\Omega_g=n_g^{-1}\sum_{i\in g}\widetilde D_i^2,
\qquad
\widehat\lambda=1-
\frac{[G^{-1}\sum_g\widehat\Omega_g]^2}
     {G^{-1}\sum_g\widehat\Omega_g^2},
\]

with (q=1), and

\[
\widehat V_{CCV-HC3}^{exp}
=\widehat\lambda\widehat V_{CR0}
+(1-\widehat\lambda)\widehat V_{HC3}.
\]

This scalar mixture is **experimental**. The Lean development establishes
arbitrary-continuous covariance-kernel identities, binary reduction, lambda
bounds, and exactness when the assignment covariance kernel itself is correct.
It does not derive the residualized-second-moment scalar substitution as the
correct kernel for arbitrary continuous assignments. Lambda, omega dispersion,
and kurtosis dispersion are therefore reported as diagnostics, not proof or a
binary validity gate. A scalar lambda also does not automatically license a
multivariate history covariance; each reported scalar contrast declares its
direction.

The cluster sensitivity registry contains county, CZ, CZ-by-region, state,
AEWR region, region-year, state-year, market, year, agro-2/3/5, exposure
decile, and exposure-decile-by-region. AEWR region is the natural candidate
for a regional AEWR shifter, but the archive did not resolve one primary
dependence partition for all lower-geography treatments. Partitions are
reported as design sensitivities, never searched for significance.

## Diagnostics and status

The retained artifacts include source/panel/code/environment/sample and output
hashes; county-year and geographic contracts; frozen-window support columns;
residualized treatment variation; causal/nuisance rank; condition number;
normal-equation residual; full leverage; cluster counts; omega, lambda, and
kappa; LPM fitted-support warnings; and like-for-like full-history/one-lag
differences where both are in the selected queue. Future-dose/lead checks,
exact-vs-FWL CV3 comparisons, and targeted Monte Carlo exercises remain named
diagnostic extensions rather than silently executed gates.

Warnings remain visible even when unfavorable. A future-dose coefficient,
one-lag disagreement, an out-of-range LPM fitted value, high leverage, or a
sensitivity failure is not converted into an invented binary empirical
"release gate."

Run from the repository root:

```sh
./scripts/run_mundlak_chamberlain.sh
```

The default compact queue is bounded. `MC_SPEC_STAGE=exhaustive` is an explicit
opt-in and is not called by `scripts/run_all.sh`.
