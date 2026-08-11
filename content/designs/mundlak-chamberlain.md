+++
title = "Mundlak–Chamberlain dose response"
description = "Identification, specification program, estimands, inference, and diagnostics."

[extra]
scopes = ["code/designs/mundlak_chamberlain", "scripts/run_mundlak_chamberlain.sh"]
+++

{{ grounding(path="code/designs/mundlak_chamberlain/design.R", anchor="mundlak-design-contract", sha256="c70937a1d9f59124bb62ddbc5b6c80e313545b905310cff064791c1851db6e93") }}

The cross-design distinction between causal outcomes and required model inputs
is maintained in [Causal outcomes and model inputs](@/contracts/causal-outcomes-and-model-inputs.md).

For a brace-annotated walkthrough of the specification, treatment contrasts,
and delta-method ATE/AME calculation, see
[`specification_and_delta_method.md`](https://github.com/ChurnKenLee/h2a_aewr_paper/blob/master/markdowns/specification_and_delta_method.md).

For the implemented version-3 architecture—which turns lag order, dose basis,
moderator richness, and calendar windows into an audited specification
program—see
[saturation_budgets_and_specification_program.md](https://github.com/ChurnKenLee/h2a_aewr_paper/blob/master/markdowns/saturation_budgets_and_specification_program.md).
The frozen version-2.3 estimator is retained as a compatibility record and
benchmark.

Version 3 is run by `scripts/run_mundlak_chamberlain.sh`. Its program-specific
stages are:

1. `01_01_build_specification_registry.R` compiles 648 declared records over
   54 admissible calendars and writes the arithmetic region-rank ledger. The
   declaration registry is not the default execution queue.
2. `02_01_estimate_specification_program.R` builds calendar caches and writes
   restartable specification-outcome checkpoints. By default it executes a
   compact, predeclared set of 16 specifications: four primary-family records,
   eight additional lag/basis records, and four calendar records. It supports
   `MC_SPEC_STAGE`, `MC_SPEC_IDS`, `MC_OUTCOME_IDS`, `MC_SPEC_MAX`,
   `MC_SPEC_WORKERS`, `MC_FIXEST_THREADS`, `MC_SPEC_MAX_DENSE_GIB`,
   `MC_SPEC_MAX_PEAK_GIB`, and `MC_SPEC_FORCE`.
3. `03_01_report_specification_program.R` promotes the richest admissible
   predeclared primary per outcome and computes all effects through the exact
   delta method.
4. `04_01_diagnostics.R` audits support, moderator alignment, reassignment
   influence, and the selected-primary one-year-lead placebo.
5. `06_01_validate_specification_program.R` enforces the registry, rank,
   row-guard, common-basis, variance-adjustment, gradient, and employer-scale
   contracts.

The resource-safe defaults are one outcome worker, four `fixest` threads, a
1.25-GiB ceiling for one dense \(N\times K\) matrix, and a 6-GiB estimated
per-worker matrix working set. Models that exceed either memory ceiling are
guard-rejected before fitting. Delta gradients are aggregated directly from
the named causal dictionary and do not construct full factual and
counterfactual model matrices.

`MC_SPEC_STAGE=primary` runs only the four primary-family records (32
specification-outcome checkpoints). `MC_SPEC_STAGE=exhaustive` explicitly
opts into all 648 records; it is not used by the pipeline default. Exact
`MC_SPEC_IDS` override the stage. Completed checkpoints satisfying the current
resource limits are reused.

This is a standalone design for estimating how annual changes in the Adverse
Effect Wage Rate (AEWR) affect county H-2A program use. It does not modify or
depend on the repository's existing DiD or panel-IV designs.

The design implements the researcher's unified Chamberlain equation as richly
as the observed treatment support permits. Its primary model is a pooled OLS
identity-link conditional-mean model. Every reported causal quantity is
constructed after estimation as a named combination of the estimated
parameters and is assigned a delta-method standard error using the complete
continuous-treatment design-covariance CCV matrix described below.

The relevant Wooldridge background is summarized in
[`deep-research-report-wooldridge.md`](https://github.com/ChurnKenLee/h2a_aewr_paper/blob/master/papers/metrics/wooldridge/deep-research-report-wooldridge.md).
The two most directly relevant local sources are the treatment-heterogeneity
discussion in
[`wooldridge_stata-conf_2025_nonbinary_treatments.md`](https://github.com/ChurnKenLee/h2a_aewr_paper/blob/master/papers/metrics/wooldridge/wooldridge_stata-conf_2025_nonbinary_treatments.md)
and the pooled-regression/Mundlak equivalence discussion in
[`wooldridge_empiricalecon_2025_twfe_twmundlak_did.md`](https://github.com/ChurnKenLee/h2a_aewr_paper/blob/master/papers/metrics/wooldridge/wooldridge_empiricalecon_2025_twfe_twmundlak_did.md).

## Central identification statement

AEWR changes do **not** have to be randomly assigned, independent of baseline
outcomes, or independent of persistent county, market, state, or regional
heterogeneity.

The model explicitly allows

\[
\operatorname{Cov}(\mathbf W_r,c_i)\ne 0,
\]

selection on pre-policy outcome and covariate histories, different untreated
trends by predetermined characteristics, and dose-response heterogeneity by
the baseline binding margin \(Z_i\). This is the same important sense in which
a DiD design does not require treatment assignment itself to be exogenous.

That statement does not mean that no identifying restriction is required.
After conditioning on the full specified history and trend projection, the
remaining AEWR dose coordinate must be unrelated to the remaining innovation
in the relevant untreated potential-outcome trend. The design therefore rests
on:

1. conditional no anticipation for the focal dose;
2. conditional parallel trends, stated at the level of the richly
   parameterized untreated conditional mean;
3. correct separation of the causal dose terms from the correlated-effects
   projection;
4. support for the dose perturbation being reported;
5. no unmodeled spillovers across the treatment-assignment units; and
6. a correctly specified finite-dimensional dose-response and history
   projection.

Thus, “AEWR changes need not be exogenous” means that unconditional
independence and random policy assignment are unnecessary. It does not mean
that a contemporaneous AEWR change can respond arbitrarily to an unobserved
county outcome innovation that remains in the error after conditioning.

## Observational structure

| Object | Definition |
| --- | --- |
| Outcome unit \(i\) | County |
| Local market \(m(i)\) | State × commuting zone × AEWR-region cell |
| State \(q(i)\) | State FIPS |
| Assignment region \(r(i)\) | One of 17 AEWR regions |
| Baseline years | 2008–2010 |
| AEWR history used by Chamberlain projection | 2011–2022 |
| Outcome years | 2013–2022 |
| Dynamic horizons | Current, one-year lag, two-year lag |
| Eligible counties | Positive 2011 farm employment and complete hierarchy |
| Estimation panel | 30,410 observations; 3,041 counties |
| Hierarchy | 17 regions > 48 states > 745 market cells > 3,041 counties |

The market identifier includes the state because commuting zones can cross
state boundaries. Defining a market as state × CZ × AEWR region makes the
nesting strict:

\[
i\subset m(i)\subset q(i)\subset r(i).
\]

The 2013 start date is substantive, not arbitrary. It permits the model to
include the current AEWR change and two complete lags while avoiding the
state-specific 2009 OES exception. From 2010 onward, AEWR is unique within
each AEWR region-year.

## Treatment

For AEWR region \(r\) and year \(t\), define

\[
W_{rt}
  =100\left[\log(\operatorname{AEWR}_{rt})
            -\log(\operatorname{AEWR}_{r,t-1})\right].
\]

One unit of \(W_{rt}\) is one log percentage point of annual AEWR growth. The
three focal treatment coordinates in county-year observation \((i,t)\) are

\[
W^{(0)}_{it}=W_{r(i),t},\qquad
W^{(1)}_{it}=W_{r(i),t-1},\qquad
W^{(2)}_{it}=W_{r(i),t-2}.
\]

The complete policy-history vector is

\[
\mathbf W_{r(i)}
  =(W_{r(i),2011},\ldots,W_{r(i),2022})'.
\]

Although outcomes are observed for thousands of counties, there are only
\(17\times10=170\) observed region-year treatment cells. County replication
increases precision for conditional outcome relationships; it does not create
additional independent AEWR paths.

## Predetermined moderator \(Z_i\)

The continuous moderator is the standardized 2008–2010 mean local AEWR bite:

\[
Z_i=
\frac{\overline{\operatorname{AEWRbite}}_{i,2008:2010}
      -\overline{\operatorname{AEWRbite}}}
     {\operatorname{sd}
       (\overline{\operatorname{AEWRbite}}_{i,2008:2010})}.
\]

The source variable is `aewr_cz_p25`. A higher \(Z_i\) means that the AEWR is
more binding relative to the local low-wage margin. It is measured before the
2011–2022 treatment-history window. The implementation stores it as `mc_z`.

## Nested Mundlak components

For each predetermined baseline summary \(x_i\), the design constructs four
telescoping components:

\[
\begin{aligned}
x_i^{C} &=
  x_i-\bar x_{m(i)},\\
x_i^{M} &=
  \bar x_{m(i)}-\bar x_{q(i)},\\
x_i^{S} &=
  \bar x_{q(i)}-\bar x_{r(i)},\\
x_i^{R} &=
  \bar x_{r(i)}-\bar x.
\end{aligned}
\]

Consequently,

\[
x_i-\bar x
  =x_i^{C}+x_i^{M}+x_i^{S}+x_i^{R}.
\]

Each component is standardized after construction. This parameterizes the
correlation between the latent intercept and covariate histories separately
at the county-within-market, market-within-state, state-within-region, and
region-within-national levels.

Unrestricted AEWR-region indicators absorb the region intercept projection.
County, market, and state components enter the high-dimensional baseline
trend engine. Region-level baseline-by-year terms are not saturated because
they would compete directly with the 17 region-level treatment cells in every
year.

## Baseline variables and transformations

All baseline inputs use only 2008–2010 information.

| Design variable | Shared-panel source | Transform |
| --- | --- | --- |
| H-2A certification intensity | `h2a_cert_share_farm_workers_2011_start_year` | inverse hyperbolic sine |
| H-2A application intensity | `h2a_applications_per_farm_worker_2011_start_year` | inverse hyperbolic sine |
| AEWR bite | `aewr_cz_p25` | level |
| Population | `ln_pop_census` | log already supplied |
| Farm-employment share | `farm_emp_share` | level |
| Employment/population | `emp_pop_ratio` | level |
| Crop-income share | `share_farm_crop_cashandinc` | level |
| Animal-income share | `share_farm_animal_cashandinc` | level |
| Hired-labor cost share | `share_farm_laborexp_prodexp` | level |
| Low wage | `wage_p25` | log |
| Cropland intensity | `census_cropland_2007` | \(\log(1+\text{2007 acres}/\max(\text{farm employment},1))\) |
| Predicted H-2A intensity | `h2a_predicted_share_2011` | inverse hyperbolic sine |

Predicted H-2A intensity is the selected cutoff's static county propensity,
based on fixed 2011 farm-employment exposure. It is repeated unchanged over
panel years; repeated pre-period entries therefore carry no annual-score
interpretation.

Finite missing baseline values are imputed first by AEWR-region median and
then, if necessary, by the national median. Missingness indicators are
retained in the constructed panel. Scaling constants and the complete term
inventory are written to the intermediate-data directory.

For the nine variables most closely related to H-2A selection, farm
structure, local binding, and untreated outcome paths, the Chamberlain
intercept projection retains the separate 2008, 2009, and 2010 values. The
remaining variables enter through their three-year mean and linear trend.

## Implemented master conditional-mean model

Let \(d_{ts}=1[t=s]\), let \(s_0=2013\) be the reference year, and define the
scaled polynomial basis

\[
b_1(w)=w,\qquad b_2(w)=w^2/25.
\]

For outcome \(j\), the primary identity-link conditional mean is

\[
\begin{aligned}
E[Y_{it}^{j}\mid\mathcal H_i]
&=
\theta_t+\alpha_{r(i)}
+H_i'\lambda
+\sum_{s\ne s_0}\psi_s Z_i d_{ts}\\
&\quad+
\sum_{p=0}^{2}
\sum_{m=1}^{2}
\sum_{s=2013}^{2022}
\left(
  \beta_{pms}^{(0)}
  +\beta_{pms}^{(1)}Z_i
\right)
b_m(W_{r(i),t-p})d_{ts}\\
&\quad+
\sum_{s\ne s_0}
\sum_{h\in\mathcal A_s}
\left(
  \rho_{sh}^{(0)}
  +\rho_{sh}^{(1)}Z_i
\right)
\widetilde W_{r(i),h}d_{ts}\\
&\quad+
\sum_{\ell\in\{C,M,S\}}
\sum_{k=1}^{K}
\sum_{s\ne s_0}
\left(
  \delta_{k\ell s}^{(0)}
  +\delta_{k\ell s}^{(1)}Z_i
\right)
X_{ik}^{\ell}d_{ts}.
\end{aligned}
\tag{1}
\]

Here:

- \(H_i\) is the full Chamberlain intercept projection from separate
  pre-period histories and mean/trend summaries;
- \(\widetilde W_{r,h}\) is a standardized element of the complete
  2011–2022 AEWR-change history;
- \(X_{ik}^{\ell}\) is baseline covariate \(k\)'s component at hierarchy
  level \(\ell\); and
- the allowed history set is

\[
\mathcal A_s
=\{2011,\ldots,2022\}\setminus\{s,s-1,s-2\}.
\tag{2}
\]

Equation (1) contains:

- 120 separately parameterized causal columns:
  \(3\) horizons × \(2\) powers × \(10\) years × baseline/\(Z\) slope;
- the complete non-focal treatment vector in every nonreference year,
  both alone and in a triple interaction with \(Z_i\);
- 33 county/market/state baseline mean components, each interacted with
  all nine nonreference years and again with \(Z_i\), for 594 baseline
  trend columns;
- nine unrestricted \(Z_i\times\)year trends;
- 99 candidate Chamberlain intercept terms;
- unrestricted year indicators; and
- unrestricted AEWR-region indicators.

The principal full-sample models retain 922 coefficients after 89 purely
algebraic redundancies are removed. The code fails if any current or lagged
causal basis column is removed.

## Why the literal full equation cannot be estimated

The researcher's unrestricted equation places both

\[
W_{i,t-p}d_{ts}
\]

in the causal block and

\[
W_{ir}d_{ts}
\]

in the Chamberlain history block for every \(r\). When \(r=s-p\), these are
the same regressor. If the history term is standardized, it remains an affine
transformation of the same regressor; the year indicator absorbs the
constant. Likewise,

\[
W_{i,t-p}Z_id_{ts}
\]

and the corresponding \(r=s-p\) triple interaction are collinear after
\(Z_i\times d_{ts}\) is included.

Therefore \(\beta_{p,1,s}\) and \(\rho_{s,s-p}\) cannot be separately
estimated. No software, sample size, or delta-method routine can recover both
from the same column. The implementation removes the focal current and lag
coordinates from the nuisance history set in (2). It retains every non-focal
history coordinate.

There is a second assignment-support constraint. In a given year, the literal
non-\(Z\) equation requests:

\[
1+3\times3+12=22
\]

region-level columns from only 17 AEWR regions. Merely removing the three
exact focal duplicates leaves

\[
1+3\times3+9=19>17.
\]

The richest cellwise polynomial compatible with the full non-focal history
projection is quadratic:

\[
1+3\times2+9=16<17.
\]

Accordingly, the primary master model uses a quadratic in every
lag-by-calendar-year cell. Cubic treatment columns remain constructed and
documented for sensitivity work, but a cellwise cubic is not labeled as
identified. The exact count and duplicate audit is retained in
`mc_identification_rank_audit.csv`.

## Outcomes

The master equation is estimated separately for:

| ID | Outcome | Sample |
| --- | --- | --- |
| `applications` | Applications per 1,000 workers in 2011 farm employment | All eligible county-years |
| `employers` | Raw balanced-linkage H-2A employer count | All |
| `certified_positions` | Certified positions per 1,000 baseline farm workers | All |
| `certified_hours` | Certified hours per baseline farm worker | All |
| `any_application` | Indicator for any application | All; linear probability model |
| `positions_per_application` | Certified positions per application | Positive applications |
| `hours_per_position` | Certified hours per certified position | Positive certified positions |

The identity-link specification exactly matches equation (1) and keeps the
polynomial marginal effects transparent. The shared helper code also supports
Poisson QMLE and logit links, but those are nonlinear-outcome extensions, not
silently substituted for the supplied pooled-OLS master equation.

## Model ladder

Every outcome is estimated under four declared specifications:

1. `twfe_benchmark`: current and two lagged AEWR changes with county and year
   fixed effects;
2. `mundlak_multilevel`: the nested mean/trend projection, heterogeneous
   predetermined trends, and treatment interactions;
3. `chamberlain_rich`: equation (1), the primary model; and
4. `chamberlain_lead_test`: equation (1) plus year-specific linear and
   quadratic one-year-ahead dose terms and their \(Z_i\) interactions.

In the lead model, \(W_{r,s+1}\) is removed from the nuisance history block
before being reintroduced as a placebo causal coordinate. Otherwise the lead
test would reproduce the same exact-collinearity problem as the focal dose.
Because the 2022 lead is unavailable, the lead sample ends in 2021.

## Potential-outcome interpretation

For a focal horizon \(p\), define the dose-response contrast while holding the
observed type and all nuisance-history coordinates fixed:

\[
\Delta_{p,it}(\delta)
=
Y_{it}\!\left(
  W_{r,t-p}+\delta;
  \mathbf W_{r,-(t-p)}
\right)
-
Y_{it}\!\left(
  W_{r,t-p};
  \mathbf W_{r,-(t-p)}
\right).
\tag{3}
\]

Holding the Chamberlain projection fixed is deliberate. It treats the
projection as a description of the unit's latent type and selection history,
not as a structural causal channel that should be redrawn when a focal dose
is perturbed.

The primary standardized estimand is

\[
\operatorname{ASF}_{p}(\delta)
=
\frac{\sum_{it}\omega_{it}
E[\Delta_{p,it}(\delta)\mid\mathcal H_i]}
{\sum_{it}\omega_{it}}.
\tag{4}
\]

The retained standardizations are:

- equal county-year weight;
- equal region-year weight, implemented by weighting each county by the
  inverse number of estimation-sample counties in its region-year;
- baseline-farm-employment weight for the four farm-employment-normalized
  program-volume outcomes; and
- application or certified-position exposure weights for the conditional
  ratio outcomes.

Sample-period total effects are also reported for the five volume outcomes.
Employer-count effects are summed directly; the other fitted rates are
converted back to application, position, or hour totals.

## Average marginal effects and finite changes

Under the quadratic identity-link model, the unit-level marginal effect at
horizon \(p\), year \(s\), dose \(w\), and moderator \(z\) is

\[
\operatorname{ME}_{p,s}(w,z)
=
\beta_{p,1,s}^{(0)}
+\beta_{p,1,s}^{(1)}z
+\frac{2w}{25}
\left(
  \beta_{p,2,s}^{(0)}
  +\beta_{p,2,s}^{(1)}z
\right).
\tag{5}
\]

The exact finite change for \(\delta\) log percentage points is

\[
\begin{aligned}
\Delta_{p,s}(w,z;\delta)
&=
\delta
\left(
  \beta_{p,1,s}^{(0)}
  +\beta_{p,1,s}^{(1)}z
\right)\\
&\quad+
\frac{2w\delta+\delta^2}{25}
\left(
  \beta_{p,2,s}^{(0)}
  +\beta_{p,2,s}^{(1)}z
\right).
\end{aligned}
\tag{6}
\]

Let \(\widehat{\boldsymbol\gamma}\) contain every estimated coefficient,
including all \(\beta,\rho,\delta,\theta,\psi,\lambda\) parameters. For any
reported scalar \(q=h(\widehat{\boldsymbol\gamma})\), the code constructs the
named gradient

\[
\widehat{\mathbf g}
=
\frac{\partial h(\widehat{\boldsymbol\gamma})}
     {\partial\widehat{\boldsymbol\gamma}'}
\]

and reports

\[
\widehat{\operatorname{Var}}(\widehat q)
=
\widehat{\mathbf g}'
\widehat{\mathbf V}_{\mathrm{dcCCV}}
\widehat{\mathbf g}.
\tag{7}
\]

In the identity-link master model, holding the nuisance-history projection
fixed makes the direct entries of the gradient zero for the \(\rho\) and
\(\delta\) blocks. Their estimation still affects the relevant \(\beta\)
covariance submatrix. In a nonlinear-link extension, nuisance parameters also
enter the response-scale gradient through the baseline prediction.

The code validates (7) by comparing its analytic named-parameter gradient
with the gradient from the full counterfactual formula matrices. The maximum
observed discrepancy is approximately \(5.6\times10^{-17}\).

## Continuous-treatment CCV inference

The reporting covariance implements the finite-design, random-denominator
logic in `ccv_symlink.lean`. It does not relabel an ordinary clustered
covariance and it does not use the unsupported scalar continuous-dose
convex-combination proposal.

Let \(s=0,\ldots,16\) index a balanced cyclic assignment of the 17 complete
observed AEWR policy paths to the 17 AEWR-region labels. State zero is the
observed assignment. In the other states, every current, lagged, lead,
polynomial, cross-horizon, and Chamberlain AEWR-history column is regenerated
from the assigned donor path. Outcomes, county characteristics, geographic
labels, and the fitted residual vector \(\widehat{\mathbf u}\) remain fixed.
Every path appears in every recipient region exactly once across the 17
equally likely states.

For the complete identified coefficient basis, the code re-solves OLS in every
state:

\[
\widehat{\mathbf b}_{e,s}
=
(\mathbf X_s'\mathbf X_s)^{-1}
\mathbf X_s'\widehat{\mathbf u}.
\]

Re-solving the Gram matrix is important. It is the vector-OLS counterpart to
the Lean result that a random scalar OLS denominator requires the covariance
kernel of \(\widetilde D/(\widetilde D'\widetilde D)\), not the covariance of
raw \(\widetilde D\) multiplied by the observed denominator. The feasible
coefficient covariance is

\[
\widehat{\mathbf V}_{\mathrm{dcCCV}}
=
\frac{1}{17}
\sum_{s=0}^{16}
(\widehat{\mathbf b}_{e,s}-\overline{\mathbf b}_e)
(\widehat{\mathbf b}_{e,s}-\overline{\mathbf b}_e)'.
\tag{8}
\]

The divisor is 17 because (8) is the probability covariance under a specified
finite reference law, not an ordinary sample covariance. The implementation
forms (8) as a centered matrix cross-product divided by 17, making it positive
semidefinite by construction. State zero is set exactly to zero using the OLS
normal equations. All reported post-estimation standard errors then use (7).
The conventional 17-region clustered covariance is retained only in the model
bundle and coefficient-comparison table.

This reference law is a transparent conditional assignment model; it is not a
claim that the Department of Labor historically randomized AEWR paths. The
Lean file proves the scalar finite-design covariance identity and supplies the
vector OLS/matrix-sandwich algebra. The nonlinear dose-response contrasts in
this branch additionally rely on a first-order delta-method approximation;
they are not themselves an exact nonlinear finite-design theorem from the Lean
file.

The rich design needs a common full-rank coefficient basis in every reference
state. One nonfocal 2013 history coordinate is therefore omitted from the
terminal estimable year; retaining it makes all 16 non-observed assignment
states singular. In the simpler Mundlak benchmark, region-constant slope
moderators are omitted because treatment-path reassignment otherwise changes
which coefficient is identified. These are explicit rank safeguards, not
silent collinearity repairs.

Only 17 independent policy paths are available, so the CCV covariance has rank
at most 16 even when the model contains more than 900 coefficients. Reported
95-percent intervals use \(t_{16}\) critical values. County-level sample size
must not be presented as solving the assignment-level inference problem.

## Support and current diagnostic status

Observed current AEWR changes range from about \(-4.81\) to \(20.56\) log
percentage points. A positive five-point perturbation lies outside the
same-year observed range for roughly 54 percent of the 170 treatment cells,
although it lies outside the pooled range for less than one percent. A
ten-point perturbation leaves same-year support for roughly 86–88 percent of
cells.

For that reason:

- average marginal effects evaluated at observed doses are the primary
  retained table and figure;
- one-, five-, and ten-point finite changes are retained as explicitly
  support-audited counterfactuals; and
- no large finite change should be interpreted without consulting
  `mc_counterfactual_support.csv`.

The present rich model also produces linear-probability effects outside the
logical \([-100,100]\) percentage-point range for a five-point change, and
multiple five-point future-dose placebo effects reject zero at five percent.
These are evidence of weak support, severe overparameterization,
functional-form instability, anticipation/feedback, or some combination.
They are not suppressed. A successful software validation means that the
requested design was implemented correctly; it does **not** certify that its
causal identifying assumptions fit these data.

## No bad controls

The primary model does not condition on contemporaneous or lagged
post-treatment economic outcomes. Its covariate histories end in 2010, before
the treatment-history window begins in 2011. Full AEWR histories are included
as assignment/projection variables, and the focal coordinates are excluded
from the nuisance history block as shown in (2).

## Files and execution order

| Script | Responsibility |
| --- | --- |
| `design.R` | Frozen years, treatment basis, \(Z_i\), outcomes, and model IDs |
| `helpers.R` | Nested decomposition, formula generator, finite-design CCV, compact model methods, and delta-method routines |
| `01_build_panel.R` | Baseline histories, nested components, treatment vector, outcomes, metadata |
| `02_estimate_models.R` | Four-model ladder for eight outcomes; CCV and comparison clustered covariances |
| `03_postestimation.R` | Finite effects, sample AMEs, year effects, heterogeneity, and \(w\times z\times s\) grid |
| `04_diagnostics.R` | Support, hierarchy, literal-rank audit, and lead placebos |
| `05_generate_tables.py` | Great Tables (the Python gt implementation) HTML and LaTeX tables |
| `05_validate.R` | Artifact, nesting, rank, finiteness, and analytic-gradient contracts |

Run the branch with:

```sh
./scripts/run_mundlak_chamberlain.sh
```

## Principal artifacts

### Analysis and model artifacts

- `data/processed/mundlak_chamberlain_county_year.parquet`
- `data/intermediate/mundlak_chamberlain_metadata.rds`
- `data/intermediate/mundlak_chamberlain_scaling.csv`
- `data/intermediate/mundlak_chamberlain_variable_inventory.csv`
- `data/intermediate/mundlak_chamberlain_models.rds`

The compact model bundle retains every coefficient, the complete CCV
covariance, the comparison clustered covariance, CCV diagnostics, formula,
collinearity list, outcome, model ID, observation count, and exact sample row
IDs. It deliberately omits large \(N\times K\) score structures and the 17
state-specific coefficient-error matrices after their covariance is formed.

### Full-granularity CSV outputs

- `mc_model_diagnostics.csv`
- `mc_collinear_terms.csv`
- `mc_model_warnings.csv`
- `mc_parameter_estimates.csv`
- `mc_ccv_diagnostics.csv`
- `mc_finite_dose_effects.csv`
- `mc_average_marginal_effects.csv`
- `mc_year_effects.csv`
- `mc_heterogeneity_effects.csv`
- `mc_ame_grid.csv`
- `mc_treatment_support.csv`
- `mc_counterfactual_support.csv`
- `mc_hierarchy_counts.csv`
- `mc_identification_rank_audit.csv`
- `mc_lead_placebo_effects.csv`
- `mc_validation_summary.csv`

### Retained gt tables and ggplot2 figures

- `table_mc_dynamic_effects.tex`
- `table_mc_heterogeneity.tex`
- `table_mc_support.tex`
- `table_mc_lead_placebos.tex`
- `table_mc_ccv_coefficients.tex`
- HTML counterparts for all five tables
- `fig_mc_ccv_coefficients.png`
- `fig_mc_dynamic_effects.png`
- `fig_mc_year_effects.png`
- `fig_mc_heterogeneity.png`
- `fig_mc_treatment_support.png`

## Interpretation boundary

This branch answers a sharply defined question: conditional on the complete
implemented type, policy-history, baseline-trend, calendar-time, and
hierarchical projection, how does the fitted H-2A outcome change when one
coordinate of annual AEWR growth changes while the other assignment-history
coordinates are held fixed?

It does not claim that arbitrary time-varying confounding can be absorbed by
adding enough terms. If untreated innovations can still determine AEWR growth
after this conditioning set, the causal interpretation fails. That boundary
is the continuous-dose analogue of the conditional parallel-trends boundary
in DiD.
