# Saturation, rank budgets, and a specification program for Mundlak–Chamberlain

## Status and purpose

This is the design and implementation note for the version-3
Mundlak–Chamberlain specification program. The program is implemented in
`specification_program.R` and the numbered `*_01_*.R` drivers in this
directory. The frozen version-2.3 design remains available as a compatibility
benchmark; the selected version-3 primary replaces its `chamberlain_rich`
result in the reporting layer.

The current version-2.3 design is one admissible point in the implemented
specification space:

\[
\underbrace{2008\text{--}2010}_{\text{baseline window}},
\qquad
\underbrace{p\in\{0,1,2\}}_{\text{three dose horizons}},
\qquad
\underbrace{d\in\{1,2\}}_{\text{linear and quadratic basis}},
\qquad
\underbrace{Z_i=\text{baseline AEWR bite}}_{\text{causal moderator}}.
\]

Version 3 turns those fixed choices into a program whose input is a
specification record and whose output is an average treatment effect, its
delta-method uncertainty, and a complete audit trail. The organizing
principle is

\[
\overbrace{\text{add flexibility in county and market space}}^{
  \text{large rank budgets}
}
\quad\text{while}\quad
\underbrace{\text{rationing region-by-year coordinates}}_{
  \text{only 17 policy paths}
}.
\]

This architecture applies to every outcome, including the raw balanced
employer count. Employer count remains measured in employers; it is not divided
by farm employment. Its average effects are nevertheless calculated by the
same delta-method contract as every other outcome.

## 1. The estimand does not change when the dictionary grows

Let \(W_{r,t}\) be annual AEWR growth in log percentage points for region \(r\)
and year \(t\). County \(i\) belongs to region \(r(i)\). For a dose change
\(h\), the sample average finite-change effect under specification \(s\) is

\[
\widehat{\tau}_s(h)
=
\underbrace{
  \frac{1}{\sum_i\omega_i}
  \sum_i\omega_i
  \left[
    \widehat{\mu}_s\!\left\{W_{r(i),t-p}+h,X_i\right\}
    -
    \widehat{\mu}_s\!\left\{W_{r(i),t-p},X_i\right\}
  \right]
}_{\text{average fitted treatment contrast}}
=
\overbrace{\mathbf g_s(h)^\top\widehat{\boldsymbol\theta}_s}^{
  \text{reported causal quantity}
}.
\]

For an average marginal effect,

\[
\widehat{\operatorname{AME}}_s
=
\underbrace{
  \frac{1}{\sum_i\omega_i}
  \sum_i\omega_i
  \frac{\partial\widehat{\mu}_s(W,X_i)}{\partial W}
}_{\text{average slope at observed doses}}
=
\overbrace{\mathbf g_s^\top\widehat{\boldsymbol\theta}_s}^{
  \text{another named aggregation, not a raw coefficient}
}.
\]

Thus the coefficient vector may contain hundreds or thousands of entries
without changing the object reported to the reader. The dictionary expands
the conditional mean; the estimand remains a weighted average of factual and
counterfactual predictions.

This distinction is especially important for employer count:

\[
\underbrace{Y_{it}^{\mathrm{emp}}}_{\text{raw number of linked employers}}
\neq
\overbrace{
  Y_{it}^{\mathrm{emp}}/\text{farm employment}_i
}^{\text{not an outcome in this design}}.
\]

## 2. The geometry: four different rank budgets

Within an outcome year, the coarsest unit among the factors in an interaction
determines the space in which the resulting column can vary. In the current
balanced panel, the relevant ceilings are

\[
\underbrace{3{,}041}_{\text{county cells}}
\;>\;
\underbrace{745}_{\text{market cells}}
\;>\;
\underbrace{48}_{\text{state cells}}
\;>\;
\underbrace{17}_{\text{AEWR-region cells}}.
\]

For a year indicator \(D_t\), a dose basis \(b_d(W_{r(i),t-p})\), and a
predetermined moderator \(Z_{ik}^{(\ell)}\) measured at hierarchy level
\(\ell\),

\[
\underbrace{
  D_t\,b_d(W_{r(i),t-p})Z_{ik}^{(c)}
}_{\text{county-level causal heterogeneity}}
\in\mathbb R^{3{,}041},
\]

\[
\underbrace{
  D_t\,b_d(W_{r(i),t-p})Z_{ik}^{(m)}
}_{\text{market-level causal heterogeneity}}
\in\mathbb R^{745},
\]

\[
\underbrace{
  D_t\,b_d(W_{r(i),t-p})Z_{ik}^{(q)}
}_{\text{state-level causal heterogeneity}}
\in\mathbb R^{48},
\]

\[
\underbrace{
  D_t\,b_d(W_{r(i),t-p})Z_{ik}^{(r)}
}_{\text{region-level causal heterogeneity}}
\in\mathbb R^{17}.
\]

“Cheap” below means cheap in assignment-cell rank. It does not mean free in
memory, run time, residual degrees of freedom, or numerical conditioning.
Every accepted specification must therefore pass both

\[
\underbrace{\text{blockwise rank guards}}_{\text{identification}}
\quad\text{and}\quad
\overbrace{\text{parameter-to-row and conditioning guards}}^{
  \text{estimation stability}
}.
\]

## 3. The region-by-year budget is the binding constraint

Let a specification have

- \(H_s\) dose horizons;
- \(D_s^{(0)}\) unmoderated dose-basis degrees;
- \(J_{st}\) admitted leave-focal-out treatment-history coordinates in year
  \(t\);
- \(U_{st}^{(r)}\) year-specific region-level untreated-mean coordinates;
- \(M_{st}^{(r)}\) region-level dose-moderator coordinates; and
- \(C_{st}^{(r)}\) cross-horizon or other region-level causal coordinates.

The raw region-cell demand in year \(t\) is

\[
\underbrace{B_{st}^{(r)}}_{\text{region-cell demand}}
=
\underbrace{1}_{\text{year cell}}
+
\underbrace{H_sD_s^{(0)}}_{\text{unmoderated dose bases}}
+
\underbrace{J_{st}}_{\text{history projection}}
+
\underbrace{U_{st}^{(r)}}_{\text{region-level untreated mean}}
+
\underbrace{M_{st}^{(r)}}_{\text{region moderators}}
+
\underbrace{C_{st}^{(r)}}_{\text{cross-horizon terms}}.
\]

The current specification reaches

\[
\underbrace{1}_{\text{year cell}}
+
\underbrace{3\times2}_{\substack{\text{current, lag 1, lag 2}\\
                                  \text{linear and quadratic}}}
+
\underbrace{(12-3)}_{\text{leave-focal-out history}}
=
\overbrace{16}^{\text{coordinates used}}
<
\underbrace{17}_{\text{region cells}}.
\]

The terminal year drops one additional nuisance-history coordinate so that
all cyclic reference-design states retain a common identified coefficient
basis. The version-3 compiler preserves that safety margin:

\[
\boxed{\max_t B_{st}^{(r)}\le 16}
\qquad
\underbrace{\text{followed by a global rank test in every one of 17 states}}_{
  \text{the arithmetic guard is necessary, not sufficient}
}.
\]

This inequality makes the expensive choices transparent. A third
unmoderated polynomial degree costs \(H_s\) region coordinates per year.
Another lag has a gross cost of \(D_s^{(0)}\); when leave-focal-out also
removes that lag's corresponding linear history coordinate, its net cost is
\(D_s^{(0)}-1\). A cross-horizon dose product costs another coordinate. A
year-specific region trend costs one. A region-level moderator crossed with
every basis may cost \(H_sD_s^{(0)}\). These choices must trade against the
history projection, not enter by accident.

The state budget needs its own assertion. A raw column count is not enough
because the region span is nested inside the state span. The relevant
quantity is the incremental rank after projecting out previously admitted
columns:

\[
\underbrace{B_{st}^{(q)}}_{\text{incremental state demand}}
=
\operatorname{rank}
\left[
  \underbrace{
    M_{\mathcal A_{st}}\mathbf X_{st}^{(q)}
  }_{\substack{\text{state block residualized on year,}\\
                \text{region, history, and earlier blocks}}}
\right]
\le
\underbrace{
  48-\operatorname{rank}(\mathcal A_{st}\text{ within state space})
}_{\text{remaining state capacity}}.
\]

A declaration such as “six dose bases by eight state moderators” therefore
must fail before estimation unless the residualized QR audit shows available
capacity. County and market blocks receive the same audit even though their
ceilings are much larger.

## 4. The target causal response surface

For specification \(s\), write the causal part of the conditional mean as

\[
\begin{aligned}
\mu_{it,s}^{\mathrm{causal}}
=
\sum_{p\in\mathcal P_s}
\sum_{d\in\mathcal D_s}
\underbrace{
  D_t\,b_d(W_{r(i),t-p})
}_{\text{year-specific dose basis}}
\Bigg[
&
\underbrace{\beta_{pdt,s}}_{\text{response at centered moderators}}
\\
&+
\underbrace{
  \sum_{k\in\mathcal K_s}
  \gamma_{kpdt,s}Z_{ik}
}_{\text{predetermined county/market/state heterogeneity}}
\\
&+
\underbrace{
  \sum_{k\in\mathcal K_s^{B}}
  \xi_{kpdt,s}B_iZ_{ik}
}_{\text{bite gradient that itself varies with }Z_k}
\Bigg],
\end{aligned}
\]

where \(B_i\) is the baseline AEWR bite and all moderators are centered before
interactions are constructed.

This equation supports a ladder rather than a single jump:

| Tier | Causal dictionary | Purpose |
| --- | --- | --- |
| 0 | Bite only | Reproduces the current primary causal block |
| 1 | County and market predetermined means | Rich level heterogeneity |
| 2 | Tier 1 plus separate pre-period values or an equivalent full-rank trajectory transform | Conditions response on path shape |
| 3 | Tier 2 plus dose × bite × county/market moderators | Heterogeneity of the bite gradient |
| 4 | Tier 3 plus selected state moderators that pass the state-rank audit | Adds higher-level heterogeneity cautiously |

Region-level moderators are excluded from the causal dictionary by default.
They may remain in the untreated-mean or correlated-effects projection when
identified, but crossing them with dose spends the scarce 17-cell budget.

### Pre-period trajectory rather than pre-period mean alone

For a baseline variable \(X\), a mean-only moderator treats these histories as
equivalent:

\[
\underbrace{(-1,0,1)}_{\text{rising path}}
\quad\text{and}\quad
\overbrace{(1,0,-1)}^{\text{falling path}},
\qquad
\underbrace{\bar X=0}_{\text{same mean}}.
\]

The trajectory tier instead admits

\[
\underbrace{
  \left(X_{i,t_0},X_{i,t_0+1},\ldots,X_{i,t_1}\right)
}_{\text{separate predetermined positions}}
\quad\text{or an equivalent full-rank transform}\quad
\overbrace{
  \left(\bar X_i,\operatorname{slope}_i,\operatorname{curvature}_i,\ldots\right)
}^{\text{trajectory coordinates}}.
\]

An orthogonal transform is acceptable for conditioning and numerical
stability, but the transformation matrix must be stored so that factual and
counterfactual design matrices use exactly the same coordinates.

### Curvature and heredity

County-level cubic heterogeneity can be rank-cheap even when an unmoderated
cubic is unaffordable:

\[
\underbrace{
  D_t\,W_{r(i),t-p}^{3}Z_{ik}^{(c)}
}_{\text{county-varying column}}
\quad\text{does not occupy only region space, whereas}\quad
\overbrace{
  D_t\,W_{r(i),t-p}^{3}
}^{\text{region-only column}}
\quad\text{does}.
\]

That is coherent under **weak heredity**, but it makes the response at
\(Z=0\) lower-order than the response away from \(Z=0\). The specification
record must state one of

\[
\underbrace{
  D_s^{(\mathrm{moderated})}\le D_s^{(0)}
}_{\text{strong heredity}}
\qquad\text{or}\qquad
\overbrace{
  D_s^{(\mathrm{moderated})}>D_s^{(0)}
}^{\text{weak-heredity sensitivity}}.
\]

Strong heredity is the safer primary convention. Weak heredity belongs in the
grid because it uses the geometry efficiently, but it should never arise
silently from a formula builder.

## 5. Preventing an invisible region interaction

For an unstandardized hierarchical decomposition,

\[
\underbrace{X_i-\bar X_{m(i)}}_{\text{county component}}
+
\underbrace{\bar X_{m(i)}-\bar X_{q(i)}}_{\text{market component}}
+
\underbrace{\bar X_{q(i)}-\bar X_{r(i)}}_{\text{state component}}
+
\underbrace{\bar X_{r(i)}}_{\text{region component}}
=
\overbrace{X_i}^{\text{level}}.
\]

After componentwise standardization, the same identity holds with known scale
coefficients. Consequently, admitting the standardized level and every
sub-regional component to a dose interaction can implicitly span the omitted
region interaction:

\[
\underbrace{W X_i}_{\text{level interaction}}
-
\overbrace{
  \left(a_cWZ_i^{(c)}+a_mWZ_i^{(m)}+a_qWZ_i^{(q)}\right)
}^{\text{sub-regional interactions}}
=
\underbrace{a_rWZ_i^{(r)}}_{\text{region interaction recovered implicitly}}.
\]

This is not merely redundant notation; it spends region rank invisibly.

The dictionary therefore needs a representation rule for every source
variable:

1. use the level alone;
2. use an explicitly declared subset of hierarchical components; or
3. use a full decomposition and charge the implied region coordinate.

For AEWR bite, use the level as the interpretive channel and exclude its
hierarchical components from the **causal dose dictionary**. The components
may still be used in the untreated-mean projection if the formula and rank
audits keep the roles separate.

## 6. Saturating the untreated mean

Let \(\mathbf H_i\) collect predetermined histories and hierarchical
components. A rich untreated mean can be written

\[
\mu_{it,s}^{0}
=
\underbrace{\alpha_t+a_{r(i)}}_{\text{calendar and persistent region terms}}
+
\underbrace{
  \mathbf H_i^\top\boldsymbol\lambda_{t,s}
}_{\text{unrestricted year-specific first-order projection}}
+
\overbrace{
  \sum_{j<k}H_{ij}H_{ik}\lambda_{jk,t,s}
}^{\text{selected second-order county/market projection}}
+
\underbrace{
  \mathbf L_{r(i)}^\top\boldsymbol\rho_{t,s}
}_{\text{leave-focal-out treatment history}}.
\]

No dose factor appears in the pairwise \(\mathbf H_i\) terms, so county- and
market-level pairs do not spend the region-only causal budget. They can still
create severe computational and residual-degrees-of-freedom costs. The
program should therefore add second-order terms in declared blocks, not emit
every possible pair in one opaque formula.

Recommended nuisance ladder:

| Tier | Untreated-mean projection |
| --- | --- |
| U0 | Current first-order hierarchy and history controls |
| U1 | All predetermined county/market first-order terms by year |
| U2 | U1 plus separate pre-period trajectory coordinates by year |
| U3 | U2 plus within-source and substantively declared cross-source pairs |
| U4 | U3 plus admissible state trends after the state-rank audit |

The causal and nuisance ladders should be separate specification fields. This
allows the grid to distinguish a change in conditional parallel trends from a
change in treatment-effect heterogeneity.

## 7. Calendar conventions become specification fields

### Symmetric coding

The current implementation uses a reference-year parameterization. The
proposed symmetric form is

\[
\underbrace{
  \sum_{t\in\mathcal T_s}D_t\alpha_t
}_{\text{one intercept for every calendar year}}
+
\overbrace{a_{r(i)}}^{\text{absorbed region effect}},
\qquad
\underbrace{\text{no global intercept}}_{\text{no omitted calendar year}}.
\]

This should span the same space as the current intercept-plus-omitted-year
coding. Before it becomes the default, an equivalence test must show, for the
current specification, that

\[
\underbrace{
  \max_i|\widehat\mu_i^{\mathrm{reference}}
          -\widehat\mu_i^{\mathrm{symmetric}}|
}_{\text{fitted-value discrepancy}}
\le\varepsilon,
\qquad
\overbrace{
  |\widehat\tau^{\mathrm{reference}}
   -\widehat\tau^{\mathrm{symmetric}}|
}^{\text{estimand discrepancy}}
\le\varepsilon.
\]

The change is then a coding simplification, not a robustness specification.

### Pre-period and analysis windows

The calendar record should contain

\[
\underbrace{t_{\mathrm{pre,start}}}_{\text{where histories begin}},
\qquad
\underbrace{t_{\mathrm{pre,end}}}_{\text{where they end}},
\qquad
\underbrace{t_{\mathrm{analysis,start}}}_{\text{first outcome year}},
\qquad
\underbrace{t_{\mathrm{analysis,end}}}_{\text{last outcome year}}.
\]

For a maximum lag \(P_s\), a clean predetermined-history rule is

\[
\underbrace{t_{\mathrm{pre,end}}}_{\text{last moderator year}}
<
\overbrace{
  t_{\mathrm{analysis,start}}-P_s
}^{\text{earliest focal dose year}}.
\]

The treatment-history window must cover every focal dose year and every
non-focal history coordinate admitted by the specification. Candidate
calendar records should be generated from observed support and rejected if
they violate either rule; years should not be inserted merely to complete a
rectangular grid.

## 8. One specification record, no global design constants

Conceptually, one record should look like:

~~~r
mc_spec <- list(
  spec_id = "pre_2008_2010__h3_d2__cm_traj__bite3",
  calendar = list(
    pre_years = 2008:2010,
    history_years = 2011:2022,
    analysis_years = 2013:2022,
    coding = "symmetric"
  ),
  dynamics = list(
    lags = 0:2,
    unmoderated_degrees = 1:2,
    moderated_degrees = 1:2,
    cross_horizon = "none",
    heredity = "strong"
  ),
  moderators = list(
    tier = "county_market_trajectory",
    bite_interactions = TRUE,
    region_dose_moderators = character()
  ),
  untreated_mean = list(
    tier = "trajectory_first_order",
    pair_dictionary = character()
  ),
  history = list(rule = "leave_focal_out"),
  guards = list(
    reserve_region_coordinates = 1L,
    max_parameter_row_ratio = "predeclared",
    require_common_ccv_basis = TRUE
  ),
  covariance = list(
    method = "finite_design_covariance_ccv",
    residual_df_adjustment = "N_over_N_minus_K"
  )
)
~~~

The literal values are illustrative; the important point is ownership.
Formula construction, counterfactual construction, gradients, rank guards,
CCV state construction, file names, and table annotations must all consume
this record. None may infer the design from separate global constants.

Each moderator-dictionary row should include at least:

| Field | Meaning |
| --- | --- |
| moderator_id | Stable semantic name |
| source_column | Predetermined panel variable |
| summary | Mean, separate year, slope, or hierarchy component |
| hierarchy_level | County, market, state, or region |
| causal_roles | Main heterogeneity, bite interaction, or neither |
| untreated_roles | First-order trend or pair blocks |
| allowed_horizons | Dose horizons it may modify |
| allowed_degrees | Polynomial degrees it may modify |
| representation_group | Prevents level/component duplication |
| budget_class | County, market, state, or region rank ledger |

## 9. The delta-method contract must be generated from the same dictionary

The present named gradient knows the base term and the AEWR-bite interaction.
That is safe for the current model but cannot be extended by appending formula
terms alone. Under the program,

\[
\underbrace{
  \mathcal C_s^{\mathrm{formula}}
}_{\text{causal columns emitted by formula builder}}
=
\overbrace{
  \mathcal C_s^{\mathrm{counterfactual}}
}^{\text{columns changed by dose perturbation}}
=
\underbrace{
  \mathcal C_s^{\mathrm{gradient}}
}_{\text{columns resolved by effect builder}}.
\]

The equality is an executable assertion, not documentation.

For the identity-link models currently used, the safest general gradient is
constructed directly from model matrices:

\[
\underbrace{\mathbf g_s(h)}_{\text{finite-change gradient}}
=
\frac{1}{\sum_i\omega_i}
\sum_i\omega_i
\overbrace{
  \left[
    \mathbf x_{i,s}(W+h)-\mathbf x_{i,s}(W)
  \right]
}^{\text{column-wise factual/counterfactual difference}}.
\]

Then

\[
\underbrace{
  \widehat{\tau}_s(h)
}_{\text{average treatment effect}}
=
\overbrace{
  \mathbf g_s(h)^\top\widehat{\boldsymbol\theta}_s
}^{\text{delta-method estimand}},
\qquad
\underbrace{
  \widehat{\operatorname{se}}\{\widehat\tau_s(h)\}
}_{\text{delta-method standard error}}
=
\overbrace{
  \sqrt{\mathbf g_s(h)^\top
  \widehat{\mathbf V}_s
  \mathbf g_s(h)}
}^{\text{full-covariance propagation}}.
\]

An analytical gradient may be retained for speed, but every specification
must validate it against the model-matrix gradient:

\[
\boxed{
\left\|
\mathbf g_s^{\mathrm{analytic}}(h)
-
\mathbf g_s^{\mathrm{matrix}}(h)
\right\|_\infty
\le 10^{-9}
}.
\]

For an AME, the corresponding matrix gradient is

\[
\underbrace{
  \mathbf g_s^{\mathrm{AME}}
}_{\text{average derivative gradient}}
=
\frac{1}{\sum_i\omega_i}
\sum_i\omega_i
\overbrace{
  \frac{\partial\mathbf x_{i,s}(W)}{\partial W}
}^{\text{column-wise derivative at the observed dose}}.
\]

It uses the same full-covariance quadratic form. Its analytical derivative
should be checked against a centered finite difference of the complete model
matrix at progressively smaller step sizes.

Because the fitted identity-link model is linear in its columns, this is an
exact algebraic comparison up to floating-point error. A newly generated
causal term that receives a silent zero in the analytic gradient must stop the
run.

## 10. Covariance, residual degrees of freedom, and the 17-state limit

For reference state \(a\in\{1,\ldots,17\}\), let

\[
\underbrace{
  \widehat{\mathbf e}_{s,a}
}_{\text{state-specific coefficient-error vector}}
=
\overbrace{
  (\mathbf X_{s,a}^{\top}\mathbf X_{s,a})^{-1}
  \mathbf X_{s,a}^{\top}\widehat{\mathbf u}_s
}^{\text{fixed fitted residual, reassigned complete policy paths}}.
\]

The finite-design covariance is

\[
\underbrace{
  \widehat{\mathbf V}_{s}^{\mathrm{dcCCV}}
}_{\text{probability covariance}}
=
\frac{1}{17}
\sum_{a=1}^{17}
\overbrace{
  (\widehat{\mathbf e}_{s,a}-\bar{\mathbf e}_s)
  (\widehat{\mathbf e}_{s,a}-\bar{\mathbf e}_s)^\top
}^{\text{state contribution}},
\qquad
\operatorname{rank}
\left(\widehat{\mathbf V}_{s}^{\mathrm{dcCCV}}\right)
\le
\underbrace{16}_{17-1}.
\]

No amount of county-level saturation increases that rank.

To address mechanical residual shrinkage as \(K_s\) grows, the program should
predeclare an HC1-style sensitivity adjustment

\[
\underbrace{
  \widehat{\mathbf V}_{s}^{\mathrm{df}}
}_{\text{reported adjusted covariance}}
=
\overbrace{
  \frac{N_s}{N_s-K_s^{\mathrm{eff}}}
}^{\text{residual-df multiplier}}
\underbrace{
  \widehat{\mathbf V}_{s}^{\mathrm{dcCCV}}
}_{\text{finite-design covariance}},
\]

where \(K_s^{\mathrm{eff}}\) is the rank of the complete fitted design,
including absorbed degrees of freedom. This correction addresses residual
leverage; it does **not** create more policy paths. The critical value remains
based on \(17-1=16\) design degrees of freedom.

Because this scalar correction is not part of the finite-design covariance
identity itself, both adjusted and unadjusted results should be stored. The
predeclared headline convention must be the same across the specification
curve. Changing the convention after seeing which specifications narrow is
not allowed.

For each outcome-specific estimation sample, require

\[
\underbrace{
  \frac{K_s^{\mathrm{eff}}}{N_s}
}_{\text{parameter-to-row ratio}}
\le
\overbrace{\kappa_{\max}}^{\text{predeclared by outcome class}},
\qquad
\underbrace{
  N_s-K_s^{\mathrm{eff}}
}_{\text{residual degrees of freedom}}
>0.
\]

The ratio matters especially for positions-per-application and
hours-per-position, which use restricted samples. A specification that fails
may use a reduced dictionary declared in advance; it may not drop terms based
on the realized coefficient.

## 11. The specification grid

The grid should cross four substantive dimensions:

\[
\underbrace{
  H_s\in\{1,2,3\}
}_{\text{dynamic horizons}}
\times
\overbrace{
  D_s^{(0)}\in\{1,2,3\}
}^{\text{unmoderated dose degree}}
\times
\underbrace{
  \text{moderator tier}\in\{0,1,2,3\}
}_{\text{response-surface richness}}
\times
\overbrace{
  \text{admissible pre-period records}
}^{\text{calendar choice}}.
\]

Not every Cartesian product is estimable. The budget compiler should first
emit a candidate registry and then classify each row:

\[
\underbrace{\text{candidate}}_{\text{record exists}}
\longrightarrow
\underbrace{\text{budget admissible}}_{\text{rank ledgers pass}}
\longrightarrow
\underbrace{\text{common basis}}_{\text{all 17 states pass}}
\longrightarrow
\underbrace{\text{estimated}}_{\text{numerical guards pass}}
\longrightarrow
\overbrace{\text{reported}}^{\text{delta audit passes}}.
\]

The complete declaration registry retains rejected rows and a human-readable reason, such as
region budget exceeded, state incremental rank exceeded, parameter-row guard,
or CCV state 07 lost basis. This prevents the reported curve from looking like
a hand-selected subset.

Declaration is intentionally separated from execution. The default execution
registry contains 16 predeclared records rather than evaluating all 648
Cartesian cells: the four H3-D2 primary-family richness tiers, an R1 lag/basis
ladder on the target calendar, and four R1 calendar perturbations. The full
declaration grid remains available only through an explicit `exhaustive`
stage. Before a fit begins, the compiler also requires

\[
\underbrace{8N_sK_s/2^{30}}_{\text{one dense design matrix in GiB}}
\le 1.25
\quad\text{and}\quad
\overbrace{
  4(8N_sK_s)+3(8K_s^2)
}^{\text{conservative matrix working-set proxy}}
/2^{30}
\le 6.
\]

These are computational admissibility conditions, distinct from the
statistical parameter-to-row guard.

Cross-horizon products should be a separate, expensive grid field. They
should not be bundled into “cubic” because

\[
\underbrace{W_{t}^{3}}_{\text{within-horizon curvature}}
\quad\text{and}\quad
\overbrace{W_tW_{t-1}}^{\text{cross-horizon complementarity}}
\]

encode different response surfaces and spend different ledger entries.

## 12. Diagnostics that make saturation interpretable

### Within-year treatment alignment

For a candidate moderator block \(\mathbf M_{st}\), let
\(\mathbf A_{st}\) be the already admitted design and
\(\mathbf D_{st}\) the focal dose-basis block. Residualize both:

\[
\underbrace{\widetilde{\mathbf D}_{st}}_{\text{remaining dose coordinates}}
=
M_{\mathbf A_{st}}\mathbf D_{st},
\qquad
\overbrace{\widetilde{\mathbf M}_{st}}^{\text{new moderator block}}
=
M_{\mathbf A_{st}}\mathbf M_{st}.
\]

Report the largest canonical correlation

\[
\underbrace{\rho_{st}^{\max}}_{\text{block alignment}}
=
\sigma_{\max}
\left[
  (\widetilde{\mathbf D}_{st}^{\top}\widetilde{\mathbf D}_{st})^{-1/2}
  \widetilde{\mathbf D}_{st}^{\top}\widetilde{\mathbf M}_{st}
  (\widetilde{\mathbf M}_{st}^{\top}\widetilde{\mathbf M}_{st})^{-1/2}
\right].
\]

A nearly orthogonal block may add conditional-mean flexibility while doing
little to distinguish dose responses and potentially worsening conditioning.
This diagnostic should be reported, not used to delete a block after observing
the outcome estimate unless a deletion rule was preregistered.

Also store each block's incremental rank, smallest singular value, and
condition number. A formula can be technically full rank and still be
numerically unusable.

### Leave-one-reference-state-out influence

For each reference state \(a\), form

\[
\underbrace{
  \widehat{\mathbf V}_{s,-a}
}_{\text{covariance without state }a}
=
\frac{1}{16}
\sum_{b\ne a}
\overbrace{
  (\widehat{\mathbf e}_{s,b}-\bar{\mathbf e}_{s,-a})
  (\widehat{\mathbf e}_{s,b}-\bar{\mathbf e}_{s,-a})^\top
}^{\text{remaining-state contribution}}.
\]

For the same gradient, report

\[
\underbrace{
  \operatorname{se}_{s,-a}
}_{\text{leave-one-state-out uncertainty}}
=
\overbrace{
  \sqrt{\mathbf g_s^\top
  \widehat{\mathbf V}_{s,-a}
  \mathbf g_s}
}^{\text{projection onto the estimand}}.
\]

The point estimate need not be redefined for this diagnostic. The useful
influence summaries are the largest proportional change in standard error,
the state producing it, and whether the interval conclusion changes when the
diagnostic uses \(t_{15}\). This reveals whether one cyclic reassignment is
carrying the uncertainty calculation.

## 13. Required program outputs

Every run should write:

| Output | Contents |
| --- | --- |
| mc_specification_registry.csv | Every declared candidate and its design fields |
| mc_specification_model_diagnostics.csv | Outcome-specific estimate/rejection status, row guard, and rank fields |
| mc_rank_budget_audit.csv | Region-coordinate ledger by specification and year |
| mc_common_basis_audit.csv | Identified coefficient basis in all 17 states |
| mc_delta_gradient_audit.csv | Formula, counterfactual, and named-gradient equality plus matrix error |
| mc_moderator_alignment.csv | Within-year marginal and partial multiple correlations with each dose basis |
| mc_reference_state_influence.csv | Leave-one-state-out uncertainty changes |
| mc_specification_effects.csv | Effect, standard error, interval, and all specification fields |

The headline figure is a specification curve:

\[
\underbrace{
  \left\{\widehat{\tau}_s:s\in\mathcal S_{\mathrm{admissible}}\right\}
}_{\text{effects ordered by magnitude}}
\quad+\quad
\overbrace{
  \left\{
  H_s,D_s,\text{moderator tier},\text{pre-period},\text{history rule}
  \right\}
}^{\text{annotations explaining each point}}.
\]

Raw employer-count effects should be labeled “employers per county-year” for
an average effect and “employers over the sample period” only when the
post-estimation weights are intentionally summed rather than normalized.

## 14. Implemented sequence

### Phase A: architecture and compatibility record

1. Use the version-2.3 bite channel as the bite-only richness tier R0 while
   retaining version 2.3 itself as a separate frozen compatibility path.
2. Make the version-3 formula builder read only a specification record.
3. Generate causal terms and counterfactual gradients from one moderator
   dictionary.
4. Retain the version-2.3 estimator and its benchmarks as a frozen
   compatibility path.
5. Add region/state rank ledgers and all-17-state common-basis assertions.

This phase is complete only if

\[
\underbrace{
  \widehat{\tau}_{\mathrm{record}}
  -\widehat{\tau}_{\mathrm{current}}
}_{\text{architecture-only change}}
\approx 0
\quad\text{and}\quad
\overbrace{
  \widehat{\operatorname{se}}_{\mathrm{record}}
  -\widehat{\operatorname{se}}_{\mathrm{current}}
}^{\text{same covariance contract}}
\approx 0.
\]

### Phase B: cheap-rank saturation — implemented

Add county and market first-order moderators, trajectory coordinates, and
declared untreated-mean pairs. Introduce blocks one tier at a time so failures
can be attributed to a dictionary choice rather than to one enormous formula.

### Phase C: expensive-coordinate sweep — implemented

Sweep lag count, unmoderated polynomial degree, cross-horizon products, and
admissible state terms. Compile the rank budget before any model fit and run
the common-basis test before computing a CCV.

### Phase D: calendar sweep — implemented

Generate admissible pre-period and analysis-window records from actual data
support. Keep the causal and history rules fixed within a comparison unless
the changed rule is itself the labeled specification dimension.

### Phase E: specification-curve reporting — implemented

Estimate all outcomes under every admissible record, apply the same
delta-method and covariance conventions, and render the effect distribution
with design annotations and state-influence diagnostics.

## 15. A defensible primary-specification rule

The primary specification is named before outcome estimates are inspected.
The implemented rule is:

> Choose the richest strong-heredity specification with three dynamic
> horizons, a quadratic unmoderated dose basis, the full admissible
> leave-focal-out history projection, county/market trajectory moderators,
> no region-level dose moderators, and all predeclared rank, common-basis,
> conditioning, and parameter-to-row guards satisfied.

This rule spends abundant county/market rank before trading away scarce
region-level history coordinates. If the richest declared tier fails a row,
rank, or common-basis guard for an outcome, the selector promotes the next
richest admissible tier. The complete admissible grid remains the substance
of the result:

\[
\underbrace{\text{one preregistered primary estimate}}_{\text{headline}}
\quad\text{inside}\quad
\overbrace{\text{the full specification distribution}}^{
  \text{robustness argument}
}.
\]

## 16. What this program can and cannot accomplish

Richer conditioning can make the conditional parallel-trends restriction
more credible and the response surface less brittle. It cannot manufacture
new independent AEWR paths:

\[
\underbrace{
  \text{more flexible point-estimate model}
}_{\text{bias and credibility}}
\;\not\Rightarrow\;
\overbrace{
  \text{more than 16 covariance directions}
}^{\text{inferential resolution}}.
\]

The program succeeds if it makes every modeling convention visible, every
rank trade explicit, and every reported effect an audited delta-method
aggregation. It fails if “saturation” becomes a long formula assembled in one
place while the counterfactual, gradient, covariance, or reporting code still
assumes the old bite-only design.
