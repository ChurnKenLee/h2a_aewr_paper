# Relationships Among Recent

# Difference-in-Differences Estimators and How to

# Compute Them in Stata

## 30th UK Stata Conference

12-13 September 2024 London School of Economics

Jeff Wooldridge Department of Economics Michigan State University

1

1. Introduction
2. Staggered Interventions and Assumptions
3. Estimators without Controls
4. Adding Time-Constant Controls
5. Nonlinear Models
6. Extensions and Stata Wish List

2

## 1. Introduction

- Exploiting different timing of interventions can be powerful for determining causality.
- What is often (loosely) called “two-way fixed effects” (TWFE) imposes a constant effect across treatment cohort and calendar time.
- Constant effect model can be too restrictive.
    - Resulting estimates can consistently estimate uninteresting weighted averages of “treatment” effects.
    - Borusyak, Jaravel, Spiess (2017, 2024, REStud); Goodman-Bacon (2021, J of E);, de Chaisemartin and D’Haultfoeuille (2020, AER).

3

- The “event study” (ES) or “leads and lags” estimator estimates effects for different exposure times.
    - This still can impose unwanted restrictions.
    - Sun and Abraham (2021, J of E).

4

- At least two reactions to the limitations of constant effect (or constant by exposure time) TWFE.

1. Try to characterize the nature of the TWFE estimates.
2. Use more flexible models/estimation methods that allow more heterogeneity.
    - Possible with or without controls.
    - Callaway and Sant’Anna (2021); BJS (2024); Sun and Abraham (2021).

5

- For much analysis, do not need special commands for DiD.
    - Can use existing commands in Stata, especially `regress` , `xtreg` , `teffects` .
    - `glm` , `logit` , `fracreg` , `poisson` are useful for nonlinear models.
- With many time periods, treatment cohorts, and controls, the commands become long and messy.
- Output is very busy, but you can see everything:
    - Average treatment effects; moderating effects; selection into treatment cohort; trends as a function of controls.

6

- Community-contributed commands: `csdid` , `jwdid` (Fernando Rios-Avila).
- Stata 17 command `xtdidregress` .
    - Assumes homogeneous effects (TWFE); want to relax this.
- Stata 18: `xthdidregress` .
    - Staggered interventions and heterogeneous TEs.

7

## 2. Staggered Interventions: Notation and Assumptions

- *T* time periods with no units treated in *t* = 1.
- First unit is treated at *t* = *q* &lt; *T* .
- Initially, no reversibility: once a unit is subjected to the intervention, it stays in place.
- Treated units are added up through *t* = *T* .
- Is there a never treated group?
    - Determines whether certain ATTs are identified; assume so here.

8

- For $g \in { q,\ldots,T } $, $Y\_t(g)$ is the outcome if the unit is first subjected to the intervention at time g.
    - In $Y\_t(g)$, the number of treated periods decreases with g.
    - $Y\_t(T)$ is the outcome treated in only the final period.
    - Adopt Athey and Imbens (2021, Journal of Econometrics): $Y\_t(\infty)$ is the outcome if a unit is never treated in $ { q,\ldots,T } $.
    - $Y\_t(0)$ is common but more confusing in this context.

9

- Treatment effects of primary focus:
$$TE\_{gt}=Y\_t(g)-Y\_t(\infty), \quad g=q,\ldots,T;\ t=g,\ldots,T$$
▸ For any $t$, $Y\_t(\infty)$ is the outcome in the control state.
- Exhaustive and mutually exclusive dummy variables:$$D\_g=1 \text{ if unit is first subjected to intervention at } g \in { q,\ldots,T } $$
$$D\_\infty=1-(D\_q+D\_{q+1}+\cdots+D\_T)$$
▸ $D\_{i\infty}=1$ means unit $i$ is never treated (up through $T$).
▸ $D\_{i\infty}=0$ for all $i$ means that all units are treated by time $T$.

10

- Goal is to estimate[ \tau\_{gt}=E[Y\_t(g)-Y\_t(\infty)\mid D\_g=1],\ t=g,g+1,\ldots,T ]▸ Sometimes the focus is on the instantaneous effects, ( \tau\_{gg} ) .▸ ( \tau\_{gt}, t&gt;g ) allows us to estimate persistence.

11

**Assumption NA (No Anticipation):** All pre-intervention treatment effects are zero:

[ E[Y\_t(g)-Y\_t(\infty)\mid D\_g,\ldots,D\_T]=0,\ t\in { 1,2,\ldots,g-1 } ,\ g\in { q,\ldots,T } . \square ]

▸ Implies ( \tau\_{gt}=0,\ t&lt;g ) .

**Assumption PT (Parallel Trends):** For ( t=2,\ldots,T ) ,

[ E[Y\_t(\infty)-Y\_1(\infty)\mid D\_g,\ldots,D\_T]=E[Y\_t(\infty)-Y\_1(\infty)]. \square ]

▸ Allows the ( D\_g ) to be correlated with ( Y\_1(\infty) ) ; selection into treatment.

12

- We observe $ { D\_{i0}, D\_{iq}, D\_{i,q+1}, \ldots, D\_{iT} } $ and the outcome$$ Y\_{it}=D\_{i0}\cdot Y\_i(\infty)+D\_{iq}\cdot Y\_{it}(q)+\cdots+D\_{iT}\cdot Y\_{it}(T) $$
    - If $D\_{ig}=0$ for all $i$, simply drop that dummy: no units in cohort $g$.
- Often start with $W\_{it}$, the time-varying treatment indicator.
    - $W\_{i,t-1}=1 \Rightarrow W\_{it}=1$.
- Define post-treatment time dummies by cohort:$$ p\_{gt}=f\_{gt}+\cdots+f\_{Tt};\quad p\_{gt}=1 \text{ if } t\ge g $$
- Then$$ W\_{it}=D\_{iq}\cdot p\_{qt}+D\_{i,q+1}\cdot p(q+1)t+\cdots+D\_{iT}\cdot pTt $$

13

## 3. Estimators without Controls

- Simplest pooled OLS regression estimates a single coefficient:
    - $D\_{iq}, \ldots, D\_{iT}, f2\_t, \ldots, fT\_t$ act as controls.
$Y\_{it}$ on $W\_{it}, 1, D\_{iq}, \ldots, D\_{iT}, f2\_t, \ldots, fT\_t$
- Algebraically the same as replacing $1, D\_{iq}, \ldots, D\_{iT}$ with unit fixed effects:
    - Equivalence very useful to simply computation; extends to nonlinear models.
$Y\_{it}$ on $W\_{it}, C\_{i1}, \ldots, C\_{iN}, f2\_t, \ldots, fT\_t$
POLS = TWFE

14

- What does $\hat{\beta}\_W$ estimate?
    - Weighted average of many 2 × 2 DiDs.
    - Some “bad comparisons” or “forbidden contrasts.”
- `xdidregress` imposes a constant effect.
- Reproduces `xtreg` .

15

- The event study estimator – leads and lags – estimates an effect for different exposure times.
    - Chooses a base period for comparison – usually ( g - 1 ) for each treated cohort ( g ) .

[ Y\_{it}\text{ on }EXP\_{i,t,1-\tau}, EXP\_{i,t,2-\tau}, ..., EXP\_{i,t,-2}, ] [ EXP\_{i,t,0}, EXP\_{i,t,1}, ..., EXP\_{i,t,T-g}, ] [ 1, D\_{ig}, ..., D\_{iT}, f\_{2t}, ..., f\_{Tt} ]

- Still the same as two-way FE.
- Pre-treatment indicators used to detect pre-trends (failure of parallel trends).
- `eventdd` in Stata.

16

- How can we make the constant effect regression more flexible?
- Under PT, the pre-treatment indicators are redundant.$$ Y\_{it}\ \text{on}\ D\_{iq}\cdot f\_{qt}, \ldots, D\_{iq}\cdot f\_{Tt}, \ldots, $$$$ D\_{i,q+1}\cdot f\_{(q+1)t}, \ldots, D\_{i,q+1}\cdot f\_{Tt}, \ldots, D\_{iT}\cdot f\_{Tt}, $$$$ 1, D\_{iq}, \ldots, D\_{iT}, f\_{2t}, \ldots, f\_{Tt} $$
- Gives estimates by cohort-time pairs:$$ \hat{\tau}\_{gt},\ t = g, \ldots, T;\ g = q, \ldots, T $$

17

- Can aggregate these, typically weighted by cohort share:
    - Exposure time:$\hat{\tau}\_0, \hat{\tau} *1, \ldots, \hat{\tau}* {T-q}$
    - Or, a single, weighted effect.
- Avoids the “bad comparisons.”
- Still the same as TWFE with treatment indicators
$D\_{iq}\cdot f\_{0t}, \ldots, D\_{iq}\cdot f\_{Tt}, \ldots,$
$D\_{i,q+1}\cdot f\_{(q+1)t}, \ldots, D\_{i,q+1}\cdot f\_{Tt}, \ldots, D\_{iT}\cdot f\_{Tt}$
- “Extended” TWFE [Wooldridge (2021)].

18

- Same as a two-step imputation estimator based on cohort and time dummies: Use $W\_{it}=0$ observations to impute $Y\_{it}(\infty)$.

$$ \widehat{TE} *{it}=Y* {it}-\widehat{Y}\_{it}(\infty) $$

▸ Average by $(g,t)$.

▸ Recovers the POLS = ETWFE estimates [Wooldridge (2021, 2023)].

▸ Also the same as BJS imputation using unit FEs.

- POLS/TWFE makes inference easy.

19

- Estimated in Stata 18:

```
xthdidregress twfe (y) (w), group(id)
estat aggregation, dynamic graph
estat aggregation
```

- For computing proper standard errors, extending to nonlinear models, useful to introduce $W\_{it}$ explicitly.
    - Useful for emphasizing the difference with the constant coefficient estimation.

20

- Useful trick for obtaining standard errors that account for sampling error in weights:( Y\_{it} ) on ( W\_{it} \cdot D\_{iq} \cdot f\_{qt}, ..., W\_{it} \cdot D\_{iq} \cdot fT\_t, ..., ) ( W\_{it} \cdot D\_{i,q+1} \cdot f(q+1) *t, ..., W* {it} \cdot D\_{i,q+1} \cdot fT\_t, ..., D\_{iT} \cdot fT\_t, ) ( 1, D\_{iq}, ..., D\_{iT}, f2\_t, ..., fT\_t )
- By exposure time:`margins, dydx(w) subpop(if expj == 1) vce` `(uncond)`
- Single effect:`margins, dydx(w) subpop(if w == 1) vce` `(uncond)`

21

- Can add the pre-treatment indicators for a fully saturated regression:

$$ D\_{iq}\cdot f1\_t,\ldots,D\_{iq}\cdot f(q-2)\_t, $$

$$ D\_{i,q+1}\cdot f1\_t,\ldots,D\_{i,q+1}\cdot f(q-1)\_t, $$

$$ \ldots $$

$$ D\_{iT}\cdot f1\_t,\ldots,D\_{iq}\cdot f(T-2)\_t, $$

▸ Leads and lags estimator. ▸ Equivalent to TWFE: Sun and Abraham (2021) `(eventstudyinteract)` .

22

- Also the same as Callaway and Sant’Anna (2021) regression adjustment:

```
xthdidregress ra (y) (w), group(id)
estate aggregate, dynamic graph
estate aggregate
```

- Equivalent to 2 × 2 DiDs using the NT group as the controls.

Treated cohort: g

Pre-treatment period: g − 1

23

- Technically, the ES (leads and lags) only requires that PT holds starting in period ( g - 1 ) .
- Extended TWFE (lags only) effectively averages the pre-treatment periods.
- *Might* be a tradeoff between efficiency and robustness.
    - Under the PT and the “ideal” second moment assumptions, ETWFE is more efficient.
        - ES adds redundant, collinear regressors.
    - If PT holds after ( g - 1 ) but fails before, ES is consistent; ETWFE is inconsistent.

24

- However:
    - Under strong, positive serial correlation, ES can be more efficient (FD versus FE).
    - If PT is violated once the treatment begins, ES can have more bias than ETWFE.

25

- Using any other set of pre-treatment dummies, such as
$D\_{iq} \cdot f2, \ldots, D\_{iq} \cdot f(q - 1), \ldots, D\_{i\tau} \cdot f2, \ldots, D\_{i\tau} \cdot f(T - 1),$results in the same test. ▸ Estimates on the treatment dummies will differ; they will be relative to the first time period (coefficients normalized to be zero).

26

## 4 Adding Time-Constant Controls

• Assume ( X\_i ) not affected by the intervention (or analyze mediating effects).

• Adding

[ X\_i, D\_{ig}\cdot X\_i, ..., D\_{iT}\cdot X\_i ]

does not change estimated effects.

• Also should add

[ f\_{2t}\cdot X\_i, ..., f\_{Tt}\cdot X\_i ] (observed heterogenous trends)

[ D\_{ig}\cdot f\_{st}\cdot \dot{X}\_{ig} ] (moderating effects)

[ \dot{X}\_{ig}=X\_i-\bar{X}\_g ]

27

- POLS:
$Y\_{it}$ on $D\_{iq}\cdot f\_{qt}, \ldots, D\_{iq}\cdot f\_{Tt}, \ldots,$
$D\_{i,q+1}\cdot f\_{(q+1)t}, \ldots, D\_{i,q+1}\cdot f\_{Tt}, \ldots, D\_{iT}\cdot f\_{Tt},$$D\_{iq}\cdot f\_{qt}\cdot \dot{X} *{iq}, \ldots, D* {iq}\cdot f\_{Tt}\cdot \dot{X}\_{iT},$$D\_{i,q+1}\cdot f\_{(q+1)t}\cdot \dot{X} *{iq}, \ldots, D* {i,q+1}\cdot f\_{Tt}\cdot \dot{X} *{iT}, \ldots, D* {iT}\cdot f\_{Tt}\cdot \dot{X}\_{iT}$
$1, D\_{iq}, \ldots, D\_{iT}, X\_i, D\_{iq}\cdot X\_i, \ldots, D\_{iT}\cdot X\_i, f\_{2t}, \ldots, f\_{Tt}, f\_{2t}\cdot X\_i, \ldots, f\_{Tt}\cdot X\_i$
- Same as TWFE and random effects: Wooldridge (2021).
- Same as cohort imputation [Wooldridge (2021)] and unit-specific imputation [BJS (2024)].

28

```
xthdidregress twfe (y x1 ... xK) (w),
group(id)
estat aggregation, dynamic graph
estat aggregation
```

29

- How should one include the pre-treatment indicators?
    - As a test, probably just
    $D\_{iq}\cdot f\_1, \ldots, D\_{iq}\cdot f(q-2), \ldots, D\_{iT}\cdot f\_1, \ldots, D\_{iT}\cdot f(T-2),$
- For symmetry, include the interactions with $\dot{X}\_{ig}$.
    - Fully saturated regression with $g-1$ as the base period for treatment cohort $g$.

30

$$ \begin{aligned} Y\_{it}\ \text{on }&amp; D\_{iq}\cdot f\_{1t}, \ldots, D\_{iq}\cdot f\_{(q-2)t}, D\_{iq}\cdot f\_{qt}, \ldots, D\_{iq}\cdot f\_{Tt}, \ldots, \ &amp; D\_{i,q+1}\cdot f\_{1t}, \ldots, D\_{i,q+1}\cdot f\_{(q-1)t}, D\_{i,q+1}\cdot f\_{(q+1)t}, \ldots, D\_{i,q+1}\cdot f\_{Tt}, \ &amp; \ldots, D\_{iT}\cdot f\_{1t}, \ldots, D\_{iT}\cdot f\_{(T-2)t}, D\_{iT}\cdot f\_{Tt}, \ &amp; D\_{iq}\cdot f\_{1t}\cdot \dot{X} *{iq}, \ldots, D* {iq}\cdot f\_{(q-2)t}\cdot \dot{X} *{iq}, D* {iq}\cdot f\_{qt}\cdot \dot{X} *{iq}, \ldots, D* {iq}\cdot f\_{Tt}\cdot \dot{X} *{iT},* *\* *&amp; \ldots* *\* *&amp; D* {iT}\cdot f\_{1t}\cdot \dot{X} *{iT}, \ldots, D* {iT}\cdot f\_{(T-2)t}\dot{X} *{iT}, D* {iT}\cdot f\_{Tt}, \ldots, D\_{iT}\cdot f\_{Tt}\cdot \dot{X} *{iT},* *\* *&amp; 1, D* {ig}, \ldots, D\_{iT}, X\_i, D\_{ig}\cdot X\_i, \ldots, D\_{iT}\cdot X\_i, f\_{2t}, \ldots, f\_{Tt}, f\_{2t}\cdot X\_i, \ldots, f\_{Tt}\cdot X\_i \end{aligned} $$

- Still the same as TWFE on flexible equation.

31

- Under violation of conditional PT, not necessarily to add the extra terms.
- Under CPT and ideal second moment assumptions, adding the extra terms is inefficient.
    - But with serial correlation, can be better to add the controls!
- Fully flexible Sun and Abraham (2021) or ES version of Wooldridge (2021).
- Gives “pre-treatment” effects and estimated ATTs:
    - ( \hat{\theta}\_{g,g-1} = 0 ) is the normalization.[ \hat{\theta}\_{gs},\ g \in { q,\ldots,T } ,\ s \in { 1,\ldots,g-2 } ][ \hat{\tau}\_{gs},\ g \in { q,\ldots,T } ,\ s \in { g,\ldots,T } ]

32

- Typically the pre-trends test focuses on $D\_{ig} \cdot f\_S$, $s = 1, \ldots, g - 2$ and not these interacted with $X\_{ig}$.
    - Under CPT, these terms have zero population coefficients.
- For each cohort, could create an ES plot.
    - Can be noisy unless we have many units in each treated cohort.
- Can weight the $\hat{\theta} *{gs}$, $\hat{\tau}* {gs}$ by the cohort shares to create a single ES plot.

33

- Equivalently, define
$$NW\_{it}=1-W\_{it}$$
- Interact $W\_{it}$ with $D\_{ig}\cdot f\_{st}$, $s\in { g,\ldots,T } $ (treatment).
- Interact $NW\_{it}$ with $D\_{ig}\cdot f\_{st}$, $s\in { 1,\ldots,g-2 } $ (pre-treatment).
- Use the `subpop(expj == 1) vce(uncond)` options to account for sampling error in weights and sample averages $\bar{X}\_g$.

34

- There are other useful characterizations of this extended extended TWFE.
- For both ( \hat{\tau} *{gs}* *)* *and* *(* *\hat{\theta}* {gs} ) , same estimates as using time periods ( (g - 1, s) ) and the flexible ( 2 \times 2 ) DiD:

[ Y\_{it} = \alpha + \beta D\_{ig} + X\_i \gamma + (D\_{ig} \cdot X\_i)\delta + \gamma\_s f\_{st} + (f\_{st} \cdot X\_i)\pi\_s ]

[

- \tau\_{gs}(D\_{ig} \cdot f\_{st}) + (D\_{ig} \cdot f\_{st} \cdot X\_i)\rho\_{gs} + U\_{it}, \ t \in (g - 1, s) ]

▸ Use only the subset ( D\_{ig} = 1 ) or ( D\_{i\infty} = 1 ) .

35

- Also the same as the regression adjustment version of Callaway and Sant’Anna (2021).
    - Run separate RA on “long” differences:( Y\_{is} - Y\_{i,g-1} ) on ( 1, D\_{ig}, \mathbf{X} *{ig}, D* {ig} \cdot \mathbf{X} *{ig}* *)* *using* *(* *D* {ig} = 1 ) or ( D\_{i\infty} = 1 )
    - CS (2021) only tests the pre-treatment dummies; not the interactions with ( \mathbf{X}\_i ) .
    - The coefficient on ( D\_{ig} ) is ( \hat{\zeta} *{gs}* *)* *(* *(s \ge g)* *)* *or* *(* *\hat{\theta}* {gs} ) ( (s \le g - 2) ) for each cohort ( g \in { q, \ldots, T } ) .

36

- Computed by `xthdidregress ra` , but does the original CS (2021).
    - Does not use g − 1 as base period.
- Computed by `csdid, method(reg) long2` .
    - Does `xthdidregress` have the “long2” option?
- See `did_staggered_6_es.do` .

37

```
. qui xthdidregress twfe (y x) (w), group(id)
. estat aggregation, dynamic graph

Duration of exposure ATET                                      Number of obs = 3,000

                                             (Std. err. adjusted for 500 clusters in id)
------------------------------------------------------------------------------
             |               Robust
    Exposure |     ATET   std. err.        t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
           0 |  3.109089   .2158719    14.40   0.000     2.684959    3.533219
           1 |  4.018795   .2491253    16.13   0.000     3.529331    4.508258
           2 |  4.209541   .341013     12.34   0.000     3.539543    4.879539
------------------------------------------------------------------------------
Note: Exposure is the number of periods since the first treatment time.

. estat aggregation

Overall ATET                                                  Number of obs = 3,000

                                             (Std. err. adjusted for 500 clusters in id)
------------------------------------------------------------------------------
             |               Robust
           y |     ATET   std. err.        t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
    (1 vs 0) |  3.672084   .1752452    20.95   0.000     3.327775    4.016394
------------------------------------------------------------------------------
```

38

```
. qui jwdid y x, ivar(id) tvar(year) gvar(first_treat)
. estat event
. estat plot
```

<!-- image -->

Plot with y-axis labeled "ATT" and x-axis labeled "Periods to Treatment". Legend: Pre-treatment, Post-treatment.

39

. qui csdid y x, ivar(id) time(year) gvar(first\_treat) method(reg) long2

. estat event

## 

ATT by Periods Before and After treatment

Event Study:Dynamic effects

```
| Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
```

## 

-------------+----------------------------------------------------------------

Pre\_avg |   .1255849   .2166593    0.58   0.562    -.2990596    .5502294

Post\_avg |   3.764128   .2144565   17.55   0.000     3.343801    4.184455

Tm5 |   .0428767   .5030756    0.09   0.932    -.9431333    1.028887

Tm4 |   .2440144   .288585     0.85   0.398    -.3216019    .8096306

Tm3 |   .0139107   .1969577    0.07   0.944    -.3721194    .3999408

Tm2 |  -.2015377   .2220663   -0.91   0.360    -.6361763    .2332518

Tp0 |   3.129432   .2422789   12.92   0.000     2.654574    3.60429

Tp1 |   4.129554   .2796401   14.77   0.000     3.58147     4.677639

Tp2 |   4.033398   .3559803   11.33   0.000     3.33569     4.731107

## 

. estat simple

Average Treatment Effect on Treated

```
| Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
```

## 

-------------+----------------------------------------------------------------

ATT |   3.683272   .2053372   17.94   0.000     3.280819    4.085726

40

<!-- image -->

Chart: ATT vs. Periods to Treatment. Y-axis label: ATT. Y-axis ticks: -2, 0, 2, 4, 6. X-axis label: Periods to Treatment. X-axis ticks: -6, -4, -2, 0, 2. Legend: Pre-treatment (blue), Post-treatment (red).

41

- Can reproduce CS regression with the very long regression and
- properly adjust standard errors:
- Pre-treatment exposure times:

```
gen expm1 = d4f03 + d5f04 + d6f05
gen expm2 = d4f02 + d5f03 + d6f04
gen expm3 = d4f01 + d5f02 + d6f03
gen expm4 = d5f01 + d6f02
gen expm5 = d6f01
```

- Treatment exposure times:

```
gen exp0 = d4f04 + d5f05 + d6f06
gen exp1 = d4f05 + d5f06
gen exp2 = d4f06
```

42

. qui reg y c.mw#c.d4E01 c.mw#c.d4E02 c.mw#c.d4E04 c.mw#c.d4E05 c.mw#c.d4E06 ///

```
c.mw#c.d5E01 c.mw#c.d5E02 c.mw#c.d5E03 c.mw#c.d5E05 c.mw#c.d5E06 ///
c.mw#c.d6E01 c.mw#c.d6E02 c.mw#c.d6E03 c.mw#c.d6E04 c.mw#c.d6E06 ///
c.mw#c.d4E01#c.x_dm4 c.mw#c.d4E02#c.x_dm4 c.mw#c.d4E04#c.x_dm4 c.mw#c.d4E05#c.x_dm4
c.mw#c.d5E01#c.x_dm5 c.mw#c.d5E02#c.x_dm5 c.mw#c.d5E03#c.x_dm5 c.mw#c.d5E05#c.x_
c.mw#c.d6E01#c.x_dm6 c.mw#c.d6E02#c.x_dm6 c.mw#c.d6E03#c.x_dm6 c.mw#c.d6E04#c.x_
c.d4 c.d5 c.d6 x c.d4#c.x c.d5#c.x c.d6#c.x ///
i.year i.year#c.x, vce(cluster id)
```

. margins, dydx(mw) subpop(if expm5 == 1) vce(uncond)

```
(Std. err. adjusted for 500 clusters in id)
```

```
|            Unconditional
         |      dy/dx   std. err.      t    P>|t|     [95% conf. interval]
```

## 

-------------+----------------------------------------------------------------

mw |   .0428767    .5075724     0.08   0.933     -.9543658    1.040119

. margins, dydx(mw) subpop(if expm4 == 1) vce(uncond)

```
(Std. err. adjusted for 500 clusters in id)
```

```
|            Unconditional
         |      dy/dx   std. err.      t    P>|t|     [95% conf. interval]
```

## 

-------------+----------------------------------------------------------------

mw |   .2440144    .2911646     0.84   0.402     -.3280453    .816074

43

. margins, dydx(mw) subpop(if expm3 == 1) vce(uncond)

```
(Std. err. adjusted for 500 clusters in id)
```

```
|            Unconditional
         |      dy/dx   std. err.      t    P>|t|     [95% conf. interval]
```

## 

-------------+----------------------------------------------------------------

mw |   .0139107    .1987183    0.07   0.944     -.376517    .4043384

. margins, dydx(mw) subpop(if expm2 == 1) vce(uncond)

```
(Std. err. adjusted for 500 clusters in id)
```

```
|            Unconditional
         |      dy/dx   std. err.      t    P>|t|     [95% conf. interval]
```

## 

-------------+----------------------------------------------------------------

mw |   .2015377    .2222352    0.91   0.365     -.2350943   .6381698

44

. margins, dydx(w) subpop(if exp0 == 1) vce(uncond)

```
(Std. err. adjusted for 500 clusters in id)
```

```
|            Unconditional
         |      dy/dx   std. err.      t    P>|t|     [95% conf. interval]
```

## 

-------------+----------------------------------------------------------------

w |   3.129432    .2444446    12.80   0.000      2.649165    3.609699

. margins, dydx(w) subpop(if exp1 == 1) vce(uncond)

```
(Std. err. adjusted for 500 clusters in id)
```

```
|            Unconditional
         |      dy/dx   std. err.      t    P>|t|     [95% conf. interval]
```

## 

-------------+----------------------------------------------------------------

w |   4.129554    .2821397    14.64   0.000      3.575226    4.683882

45

```
. margins, dydx(w) subpop(if exp2 == 1) vce(uncond)

                          (Std. err. adjusted for 500 clusters in id)

------------------------------------------------------------------------------
             |               Unconditional
             |      dy/dx   std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
           w |   4.033398    .3591623    11.23   0.000     3.327742    4.739055
------------------------------------------------------------------------------

* Single weighted effect:

. margins, dydx(w) subpop(if w == 1) vce(uncond)

                          (Std. err. adjusted for 500 clusters in id)

------------------------------------------------------------------------------
             |               Unconditional
             |      dy/dx   std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
           w |   3.683272    .2071727    17.78   0.000     3.276234    4.09031
------------------------------------------------------------------------------
```

46

- See did\_staggered\_6\_es.do for other characterizations of the estimators.
    - As 2 × 2 DiDs.
    - As imputation estimators.

48

**Other Treatment Effect Estimators**

- The default CS (2021) is not the linear RA estimator.
    - Doubly robust estimator based on linear RA and IPW.
- Can apply *any* treatment effect estimator to the cross sections

[ { (Y\_{it} - Y\_{i,g-1}, D\_{ig}, X\_i) } ]

49

- Lee and Wooldridge (2023): Replace long differences $Y\_{it} - Y\_{i,g-1}$
with$$ \dot{Y} *{itg} = Y* {it} - \frac{1}{(g - 1)} \sum\_{r=1}^{g-1} Y\_{ir} $$
- Apply any TE estimator to$$ { (\dot{Y} *{itg}, D* {ig}, X\_i) } $$
- Approaches have different sensitivities to violations of CPT.
- LW (2023): Replace $\dot{Y}\_{itg}$ with unit-specific detrended outcomes.

50

## 5 Nonlinear Models

- Only rarely does adding many unit FEs not result in the incidental parameters problem.
- In linear case, equivalent to controlling for a (small) number of cohort dummies.
- Can include the cohort dummies in a variety of nonlinear models.

51

- Wooldridge (2023, Econometrics Journal): Use an index version of conditional PT.

# 

[

E[Y\_t(\infty)|D\_q,\ldots,D\_T,\mathbf{X}]

G\left( \alpha + \sum\_{g=q}^{T}\beta\_g D\_g + \mathbf{X}\boldsymbol{\kappa} + \sum\_{g=q}^{T}(D\_g\cdot\mathbf{X})\boldsymbol{\eta}\_g + \gamma\_t + \mathbf{X}\boldsymbol{\pi}\_t \right) ]

52

- Linear case [Wooldridge (2021)]:

[ E[Y\_t(\infty)|D\_q,\ldots,D\_T,\mathbf{X}] = \alpha + \sum\_{g=q}^{T}\beta\_g D\_g + \mathbf{X}\boldsymbol{\kappa} ]

[

- \sum\_{g=q}^{T}(D\_g \cdot \mathbf{X})\boldsymbol{\eta}\_g + \gamma\_t + \mathbf{X}\boldsymbol{\pi}\_t ]

## 

[

E[Y\_t(\infty)|D\_q,\ldots,D\_T,\mathbf{X}]

# E[Y\_1(\infty)|D\_q,\ldots,D\_T,\mathbf{X}]

\gamma\_t + \mathbf{X}\boldsymbol{\pi}\_t ]

53

- The “lags only” estimator – extension of pooled OLS in linear case – simply replaces `regress` with `logit` , `fraclogit` , `poisson` .
- Advantage in using canonical link pairs.
    - Pooled estimation gives same ATT estimates as (theoretically justified) imputation.
- See `did_common_6_logit_es.do` and `did_staggered_6_poisson_es.do` .

54

- Exponential, pooled Poisson:

<!-- image -->

Weighted Effects for Different Exposures. Y-axis: Weighted Treatment Effects. X-axis: Exposure Length. X-axis ticks: -5, -4, -3, -2, -1, 0, 1, 2. Y-axis ticks: 0, 5, 10.

55

- Linear model, pooled OLS:

**Weighted Effects for Different Exposures**

**Y-axis:** Weighted Treatment Effects **Y-axis ticks:** 0, 5, 10

**X-axis:** Exposure Length **X-axis ticks:** -5, -4, -3, -2, -1, 0, 1, 2

56

## 6. Extensions and Stata Wish List

- Regression methods can easily allow staggered exit from treatment.
- Generally, index cohorts by entry and exit time.
    - The potential outcomes are now $Y\_t(g,h)$.
    - First treated period is $g$; exit occurs in $h$.
- $D\_{g,h}$ for $g &lt; h \le T$ are the new cohort indicators.
- With a never treated group, $$\tau\_{gh,t} = E[Y\_t(g,h) - Y\_t(\infty) \mid D\_{g,h} = 1],\ t = g, g + 1, \ldots, T$$
    - $Y\_t(\infty)$ is the PO in the never treated state.

57

- ATTs are defined even when $t \ge h$ – that is, after the intervention has been removed.
    - Can estimate persistence even after program is eliminated.
    - When $t \ge h$, can see whether an effect dissipates after the intervention disappears.
- Estimation: In place of the interactions $D\_g \cdot f\_{st},\ s = g, \ldots, T$, include

$$ D\_{gh} \cdot f\_{st},\ g &lt; h,\ s = g, \ldots, T $$

- See did\_exit\_6.do.

58

**Extensions to xthdidregress?**

```
xthdidregress twfe (y x), (w) group(id)
xthdidregress ra (y x), (w) group(id)
xthdidregress logit (y x), (w) group(id)
xthdidregress fraclogit (y x), (w) group(id)
xthdidregress poisson (y x), (w) group(id)
xthdidregress logit (y x), (w) group(id)
   event
xthdidregress twfe, (y x) (w) group(id)
   hetrend
```

- Easy to automatically detect exit.

59

```
. gen exp0 = d4inf4 + d464 + d454 + d5inf5 + d565 + d6inf6

. gen exp1 = d4inf5 + d465 + d5inf6

. gen exp2 = d4inf6

. qui reg y c.w#c.d4inf4 c.w#c.d4inf5 c.w#c.d4inf6 c.w#c.d464 c.w#c.d465 c.d466 ///
>        c.w#c.d454 c.d455 c.d456 c.w#c.d5inf5 c.w#c.d5inf6 c.w#c.d565 c.d566 c.w#c.d6inf6
>        c.w#c.d4inf4#c.x_dm4_inf c.w#c.d4inf5#c.x_dm4_inf c.w#c.d4inf6#c.x_dm4_inf c.w#
>        c.w#c.d465#c.x_dm4_6 c.d466#c.x_dm4_6 c.w#c.d454#c.x_dm4_5 c.d455#c.x_dm4_5 c.d456
>        c.w#c.d5inf5#c.x_dm5_inf c.w#c.d5inf6#c.x_dm5_inf c.w#c.d565#c.x_dm5_6 c.d566#c
>        i.year i.year#c.x ///
>        c.d4_inf c.d4_5 c.d4_6 c.d5_inf c.d5_6 c.d6_inf ///
>        c.d4_inf#c.x c.d4_5#c.x c.d4_6#c.x c.d5_inf#c.x c.d5_6#c.x c.d6_inf#c.x, vce(cluster

. margins, dydx(w) subpop(if exp0 == 1) vce(uncond)

                          (Std. err. adjusted for 1,000 clusters in id)

------------------------------------------------------------------------------
             |               Unconditional
             |      dy/dx     std. err.       t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
           w |   3.579169      .1347143    26.57   0.000      3.314814    3.843525
------------------------------------------------------------------------------
```

60

```
. margins, dydx(w) subpop(if exp1 == 1) vce(uncond)

                 (Std. err. adjusted for 1,000 clusters in id)
------------------------------------------------------------------------------
             |            Unconditional
             |      dy/dx   std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
           w |   4.635792   .1635907    28.34   0.000     4.314771    4.956813
------------------------------------------------------------------------------

. margins, dydx(w) subpop(if exp2 == 1) vce(uncond)

                 (Std. err. adjusted for 1,000 clusters in id)
------------------------------------------------------------------------------
             |            Unconditional
             |      dy/dx   std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
           w |    5.88682   .2415575    24.37   0.000     5.412802    6.360838
------------------------------------------------------------------------------

. margins, dydx(w) subpop(if w == 1) vce(uncond)

                 (Std. err. adjusted for 1,000 clusters in id)
------------------------------------------------------------------------------
             |            Unconditional
             |      dy/dx   std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
           w |   4.355817   .1216767    35.80   0.000     4.117046    4.594588
------------------------------------------------------------------------------
```

61

## Exit in Event Study Estimation

- Now include $D\_{gh} \cdot f\_{st}$ for all
    - Again, period $g - 1$ is the comparison (base) period.
$s \ne g - 1$
- Apply weights to obtain estimates by exposure time.
- Can also use weights to estimate effects of time since last exposure.
    - A bit easier to use `margins` with `subpop()` .
- See `did_exit_6_es.do` .

62