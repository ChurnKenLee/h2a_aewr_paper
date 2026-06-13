# Difference-in-Differences for Nonbinary Treatments in Stata

**2025 Stata Conference** Nashville, TN 31 July - 1 August, 2025

Jeff Wooldridge Department of Economics Michigan State University

1

1. Review of the Binary Case
2. Nonbinary Treatments with $T = 2$
3. A Heterogeneous Slopes Model with Staggered Assignment
4. Application to Walmart Store Sitings
5. Summary

2

# 1 Review of the Binary Case

- For staggered interventions with a binary treatment, many estimators now exist.
- The Stata 18+ command `xtdidregress` supports:
    - `twfe` : Extended TWFE, a “lags only” estimator: Wooldridge (2025, forthcoming Empirical Economics).
    - `ra` : The “leads and lags” (event study) regression adjustment estimator: Callaway and Sant’Anna (2021, Journal of Econometrics).
        - Same as ETWFE with all leads and lags and full flexibility in controls [Wooldridge (2025)].
        - Same as Sun and Abraham (2021, J of E) without controls.

3

- `ipw` : CS (2021) inverse probability weighting; Abadie (2005, REStat), Sant’Anna and Zhou (2020, J of E) as special cases.
- `aipw` : CS (2021) augmented IPW – doubly robust.
- Estimates can be aggregated and graphed by exposure time.

4

• `jwdid` [Rios-Avila, Nagengast, Yotov] produces ETWFE with moderating effects and heterogeneous trends. • `jwdid` with the `never` option is the same as `xthdidregress ra` with a balanced panel, time-constant controls. • `csdid` [Rios-Avila, Callaway, Sant’Anna] reproduces `ra` , `ipw` , `aipw` . • Get more information doing estimation “by hand”: `reg` , `xtreg` , `teffects` (after transforming the data). • `reg` , `xtreg` allow inclusion of cohort-specific trends.

5

- Lee and Wooldridge (2023, Working Paper) show how one can remove pre-treatment averages or pre-treatment (unit-specific) trends to obtain lags only estimators.
    - One can apply any TE estimator: IPWRA, matching, machine learning on top of the others.
    - Can use “long differencing” instead, as in CS (2021).
    - In both cases, `teffects` can be applied after simple transformation.
    - Can apply causal machine learning methods, too.

6

- What can we do with non-binary treatments?
    - Minimum wages deviate from a national minimum wage.
    - New tax rates are imposed at the county level.
    - Distance to a natural disaster or new garbage incinerator.
    - Different levels of participation in a training program.
- Here I focus on time-invariant controls, balanced panel data.
    - Unbalanced panels easy to handle.
- Spoiler: I suggest a flexible equation estimated by TWFE.

7

## Setup and Results for Binary Treatment

- Summary of Wooldridge (2021, 2025).
- Treatment cohorts $g \in { q, q + 1, \ldots, T } $.
- Potential outcomes $Y\_t(g)$, with $Y\_t(\infty)$ in the never treated state.
- Cohort dummy variables (time constant): $D\_g$, $g \in { q, q + 1, \ldots, T, \infty } $.
    - For unit $i$, $D\_{ig}$.
- Parameters of interest are ATTs (or ATETs) for each cohort g:

$$ \tau\_{gt} = E[Y\_t(g) - Y\_t(\infty) \mid D\_g = 1], \ t = g, \ldots, T. $$

8

**Assumption NBC (No Bad Controls):** For time-constant covariates **X** ( *g* ),

**X** = **X** ( *g* ) = **X** (∞). □

**Assumption CNA (Conditional No Anticipation):** For each treatment cohort *g* ∈ { *q* , ..., *T* },

*E* [ *Y* *t* ( *g* )| *D* *q* , ..., *D* *T* , **X** ] = *E* [ *Y* *t* (∞)| *D* *q* , ..., *D* *T* , **X** ], *t* &lt; *g* . □

▸ No conditional pre-treatment effects.

9

**Assumption CPT (Conditional Parallel Trends):** For $t = 2,\ldots,T$ and time-constant controls $\mathbf{X}$,

$$ E[Y\_t(\infty) - Y\_1(\infty)|D\_q,\ldots,D\_T,\mathbf{X}] = E[Y\_t(\infty) - Y\_1(\infty)|\mathbf{X}]. \square $$

- Let $\mathbf{D} \equiv (D\_q,\ldots,D\_T)$ be the vector of cohort (“treatment”) indicators.
- Assumption CPT implies that, conditional on $\mathbf{X}$, $\mathbf{D}$ is unconfounded with respect to the trends

$$ Y\_t(\infty) - Y\_1(\infty),\ t = 2,\ldots,T $$

- $\mathbf{D}$ is allowed to be confounded with $Y\_1(\infty)$ even conditional on $\mathbf{X}$.

10

**Assumption LIN (Linearity):** For treatment cohort indicators $D\_g$ and control variables $\mathbf{X}$,

$$ E[Y\_1(\infty)|\mathbf{D},\mathbf{X}] = \alpha + \sum\_{g=q}^{T} \beta\_g D\_g + \mathbf{X}\kappa + \sum\_{g=q}^{T}(D\_g \cdot \mathbf{X})\xi\_g $$

$$ E[Y\_t(\infty)|\mathbf{D},\mathbf{X}] - E[Y\_1(\infty)|\mathbf{D},\mathbf{X}] = \sum\_{s=2}^{T}\gamma\_s f\_{st} $$

$$

- \sum\_{s=2}^{T}(f\_{st} \cdot \mathbf{X})\pi\_s,\ t = 2,\ldots,T. \quad \square $$

▸ The second part implies CPT.

11

- For a random draw $i$, with time-varying treatment indicator $W\_{it} \in { 0,1 } $:

$$ \begin{aligned} E(Y\_{it}\mid D\_{iq}, \ldots, D\_{iT}, \mathbf{X} *i)* *&amp;= \sum* {g=q}^{T}\sum\_{s=g}^{T}\tau\_{gs}(W\_{it}\cdot D\_{ig}\cdot f\_{st}) \ &amp;\quad + \sum\_{g=q}^{T}\sum\_{s=g}^{T}(W\_{it}\cdot D\_{ig}\cdot f\_{st}\cdot \dot{\mathbf{X}} *{ig})\rho* {gs} \ &amp;\quad + \sum\_{s=2}^{T}\gamma\_s f\_{st} + \sum\_{s=2}^{T}(f\_{st}\cdot \mathbf{X} *i)\pi\_s* *\* *&amp;\quad + \alpha + \sum* {g=q}^{T}\beta\_g D\_{ig} + \mathbf{X} *i\kappa + \sum* {g=q}^{T}(D\_{ig}\cdot \dot{\mathbf{X}}\_{ig})\xi\_g \end{aligned} $$

12

- Obvious estimator is pooled OLS controlling for time dummies, cohort dummies, and lots of interactions.
- Many equivalent estimators:
    - Imputation methods use an initial regression with $W\_{it}=0$ observations.
$$\text{POLS} = \text{RE} = \text{FE} = \text{Cohort Imputation} = \text{BJS Imputation}$$
- No “bad comparisons” unless we impose restrictions on the $\tau\_{gt}$.
    - For example, $\tau\_{gt}=\tau$, all $g$ and $t$.

13

- POLS (and RE) give the most information: moderating effects, heterogeneous trends, selection effects.
    - `xthdidregress twfe` gives ATTs.
    - `jwdid` gives ATTs, moderating effects, heterogeneous trends.
- Can include cohort-specific (linear) trends; same as imputation with trends, same as unit-specific trends.

14

## 2 Non-Binary Treatments with $T = 2$

- Assume initially that the treatment has quantitative meaning (“continuous”).
    - $D = 0$ is control; $D &gt; 0$ is treatment.
    - Treated unit $i$ goes from zero to a positive amount:
    $W\_{i1} = 0$ (no treatment in first period)
    $W\_{i2} = D\_i$ (some treated units)

15

- Define the dose-response function in ( t = 2 ) :[ TE(d) = Y\_2(d) - Y\_2(0), d &gt; 0 ]
- Callaway, Goodman-Bacon, Sant’Anna (2024, WP) define the average D-R function:
    - Does not reduce to the usual ATT with binary treatment.[ ATE(d) = E[Y\_2(d) - Y\_2(0)] ]
- Instead, define the dose-response function for the treated:[ ATT(d) = E[Y\_2(d) - Y\_2(0) \mid D &gt; 0] ]

16

- Consider a linear model with random slope, $B\_2$:$$ Y\_2(d) = A + \gamma\_2 + B\_2d + U\_2,\ E(U\_2) = 0 $$$$ ATT(d) = E(B\_2|D &gt; 0) \cdot d = \delta\_2 \cdot d $$$$ \delta\_2 = E(B\_2|D &gt; 0) $$
- $\delta\_2$ is identified under an assumption that limits selection:$$ E(B\_2|D) = E(B\_2|D &gt; 0) $$

17

- Can apply TWFE to more flexible models:
    - \delta\_{23}(f2\_t \cdot D\_i^3) + \gamma 2 f2\_t + C\_i + U\_{it} $$
▸ Same as$$ \Delta Y\_i \text{ on } 1, D\_i, D\_i^2, D\_i^3,\ i = 1,\ldots,N $$$$ Y\_{it} = \delta\_{21}(f2\_t \cdot D\_i) + \delta\_{22}(f2\_t \cdot D\_i^2) $$ $$
- Currently, the fancier methods are nonparametric versions of simple TWFE based on first differencing.

19

## 3. A Heterogeneous Slopes Model with Staggered Assignment

- Borrows from Wooldridge (2005, REStat).
- Let $W\_{it}$ be a row vector of treatment variables; could be continuous or discrete.
    - Could include functions of underlying treatment variables.
    - Can vary over time.
- Return to notation where $D\_{ig}$ denotes cohort indicators.
    - Assumes a control periods and a well-defined first period of treatment.

20

- Think of a potential outcomes (dose-response) function:$Y\_{it}(\mathbf{w})=\mathbf{w}\mathbf{B} *{it}+\gamma\_t+C\_i+U* {it}(\mathbf{w})$
▸ $\mathbf{B}\_{it}$ a vector of heterogenous (random) coefficients; vary by $i$ and $t$.
- Assume strictly exogenous treatment with respect to shocks:$E[U\_{it}(\mathbf{w})|W\_{i1},\ldots,W\_{iT},C\_i,\mathbf{B} *{i1},\ldots,\mathbf{B}* {iT}]=0$
▸ No feedback from shocks to $Y\_{it}$ to future treatment.

21

- Define the cohort/time dose response function:$$ E[Y\_{it}(w)|D\_{iq}, \ldots, D\_{iT}] = wE(B\_i|D\_i) + \gamma\_t + E(C\_i|D\_i) $$▸ Can only identify this function for $(g,t)$ pairs with $t \in { g, \ldots, T } $.
- The following representation imposes no assumption when treatments are zero in $t \in { 1, \ldots, g - 1 } $ for cohort $g$:$$ E(B\_i|D\_{iq}, \ldots, D\_{iT}) = \sum\_{g=q}^{T} \sum\_{s=g}^{T} (D\_{ig} \cdot f\_{st})\delta\_{gs} $$

22

- In terms of observable $Y\_{it}$:$$Y\_{it} = Y\_{it}(W\_{it}) = \mathbf{W} *{it}\mathbf{B}* {it} + \gamma\_t + C\_i + U\_{it}$$
- Treatment assignment can be arbitrarily correlated with $C\_i$.
- As usual, allow additive time effects.
    - Unrestricted trend in the mean of the untreated state.
    - Additivity in $\gamma\_t$ and $A\_i$ imposes a PT assumption.

23

- Key restriction for identification:[ E(B\_{it}\mid W\_{i1},\ldots,W\_{iT})=E(B\_{it}\mid D\_{iq},\ldots,D\_{iT}) ][ =\sum\_{g=q}^{T}\sum\_{s=g}^{T}(D\_{ig}\cdot f\_{st})\delta\_{gs}+\sum\_{s=1}^{T}(D\_{i\infty}\cdot f\_{st})\delta\_{\infty s} ]

▸ The *timing* of treatment can be related to the per-unit gain, ( B\_{it} ) .

▸ The *level* of treatment cannot be related to the per-unit gain.

▸ Imposes no extra assumptions in the binary case.

24

- If $W\_{it} \cdot D\_{ig} \cdot f\_{st} = 0$ – treatment is zero for the NT cohort – we can write

$$ Y\_{it} = \sum\_{g=q}^{T} \sum\_{s=g}^{T} (W\_{it} \cdot D\_{ig} \cdot f\_{st})\delta\_{gs} + \sum\_{s=q}^{T} \gamma\_s f\_{st} + A\_i + U\_{it} $$

$$ E(U\_i|\mathbf{D} *i,\mathbf{W}* {i1},\ldots,\mathbf{W}\_{iT},A\_i)=0 $$

- Consistently estimate the $\delta\_{gs}$ by TWFE.
- Contains the binary case when

$$ W\_{it} = D\_{iq} \cdot p\_{qt} + \cdots + D\_{iT} \cdot p\_{Tt} $$

$$ p\_{gt} = f\_{gt} + \cdots + f\_{Tt} $$

25

• Might aggregate the coefficients by exposure time, or obtain a single weighted δ:

$$ \delta = \sum\_{g=q}^{T} \sum\_{t=g}^{T} \omega\_{gt}\delta\_{gt} $$

**Examples:**

1. $W\_{it}$ a scalar: continuous, count, corner (zero before intervention).
2. $W\_{it} = (R\_{it}, R\_{it}^2, R\_{it}^3)$ for some quantitative treatment $R\_{it}$.
3. $W\_{it} = (W\_{i1}, W\_{i2}, \ldots, W\_{iJ})$ (different treatment levels).
4. $W\_{it} = (W\_{i1}, W\_{i2}, \ldots, W\_{iJ})$ (different treatments).

26

- Add controls, $\dot{\mathbf{X}}\_{ig} = \mathbf{X}\_i - \bar{\mathbf{X}}\_g$.

$$ Y\_{it} = \sum\_{g=q}^{T} \sum\_{s=g}^{T} (W\_{it} \cdot D\_{ig} \cdot f\_{st})\delta\_{gs} $$

$$

- \sum\_{g=q}^{T} \sum\_{s=g}^{T} (W\_{it} \otimes D\_{ig} \cdot f\_{st} \cdot \dot{\mathbf{X}} *{ig})\beta* {gs} $$

$$ \sum\_{s=2}^{T} \gamma\_s f\_{st} + \sum\_{s=2}^{T} (f\_{st} \cdot \mathbf{X} *i)\pi\_s + C\_i + U* {it} $$

- Allows heterogeneity in the “dose-response” function to vary by cohort and time.
- In the binary $W\_{it}$ case, reduces to Wooldridge (2021, 2025).

27

- Allows for the heterogeneous trends, $f\_{St}\cdot X\_i$ – relaxes PT.
- Conditioning on $X\_i$ may also help with selection on the return to treatment:
$E(B\_{it}\mid W\_{i1},\ldots,W\_{iT},X\_i)=E(B\_{it}\mid D\_{iq},\ldots,D\_{iT},X\_i)$
- The additive effects, $C\_i$, may be arbitrarily correlated with the $W\_{it}$. ▸ The usual additive FE selection is allowed.

28

- **$W\_{it}$** can be a vector of treatment indicators of units whose treatment may spill over into the outcome for unit $i$.
- If unit $i$’s neighbors do not change over time, TWFE estimation of the flexible equation eliminates not just $C\_i$, but $C\_{j(i)}$ for all neighbors $j(i)$ of unit $i$.
- To be completely general, should allow $W\_{j(i)t}$ to interact with $D\_{j(i),g(j(i))} \cdot f\_s$, where $g(j(i))$ is the treatment cohort for unit $j(i)$.

29

**Checking and Correcting for Violation of PT**

- Add pre-treatment indicators $D\_{ig} \cdot f\_{st}$ for $s \leq g - 2$ to the flexible equation.
    - Event study estimation with binary $W\_{it}$.
    - Check for significance of these “lead” indicators.
- Can add cohort-specific linear trends, $D\_{ig} \cdot t$.
    - Wooldridge (2025) in binary case.
    - Allows additional selection into treatment.
- Test for violation of strict exogeneity by including $W\_{i,t+1}$ in TWFE.

30

**Non-Zero Treatment Level for Controls**

- Suppose the “treatment” is never zero for any unit.
    - For example, distance from a house to a new garbage incinerator.
- Define $W\_{it} = 0$ in the periods before the intervention.
- $W\_{it}$ is the treatment variable post-intervention; may never be zero.
    - The stronger assumption on selection is needed because all units are “treated” at some level.

31

## 4. Application to Number of Walmart Stores on Retail Employment

- The treatment is the number of Walmarts open at a particular point in time.
- Can allow heterogeneous effects by exposure time.
- Can allow unrestricted heterogeneity by cohort/calendar time.
- Can also include heterogeneous trends.
- Equations only change from binary case by interacting the treatment dummies with the number of stores.
- See walmart\_lw\_retail\_n\_open\_slides.do.

32

. use walmart\_1w, clear

. bysort year: tab n\_open

-&gt; year = 1986

```
n_open |      Freq.     Percent        Cum.
```

------------------+----------------------------------- 0 |      1,219       94.64       94.64 1 |         62        4.81       99.46 2 |          6        0.47       99.92 4 |          1        0.08      100.00 ------------------+----------------------------------- Total |      1,288      100.00

-&gt; year = 1987

```
n_open |      Freq.     Percent        Cum.
```

------------------+----------------------------------- 0 |      1,145       88.90       88.90 1 |        125        9.70       98.60 2 |         17        1.32       99.92 3 |          1        0.08      100.00 ------------------+----------------------------------- Total |      1,288      100.00

33

...

-&gt; year = 1999

```
n_open |      Freq.     Percent        Cum.
```

------------+----------------------------------- 0 |        395       30.67       30.67 1 |        664       51.55       82.22 2 |        137       10.64       92.86 3 |         47        3.65       96.51 4 |         24        1.86       98.37 5 |          9        0.70       99.07 6 |          1        0.08       99.15 7 |          3        0.23       99.38 8 |          2        0.16       99.53 9 |          1        0.08       99.61 10 |          1        0.08       99.69 11 |          2        0.16       99.84 12 |          1        0.08       99.92 13 |          1        0.08      100.00 ------------+----------------------------------- Total |      1,288      100.00

34

```
. gen w = n_open

. * Each new store is estimated to increase retail employment by about 3.4%.

. xtreg log_retail_emp w i.year, fe vce(cluster fips)

Fixed-effects (within) regression                 Number of obs     =     29,624
Group variable: fips                              Number of groups  =      1,288

R-squared:                                        Obs per group:
     Within  = 0.5100                                  min =         23
     Between = 0.2818                                  avg =       23.0
     Overall = 0.0255                                  max =         23

                                                 F(23, 1287)       =     236.15
corr(u_i, Xb) = 0.0573                           Prob > F          =     0.0000

                               (Std. err. adjusted for 1,288 clusters in fips)
------------------------------------------------------------------------------
             |               Robust
log_retail~p | Coefficient  std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
           w |    .034412   .0056512     6.09   0.000      .0233255    .0454986
```

35

```
year |
   1978    .0730682    .0019088     38.28   0.000     .0693234     .076813
   1979    .1090979    .0025137     43.40   0.000     .1041666     .1140292
   1980    .0852645    .0028303     30.13   0.000     .0797121     .0908169
   1981    .0788761    .0023318     33.82   0.000     .0743568     .0833853
   1982    .0733567    .0039143     18.74   0.000     .0656776     .0810359
   1983    .0476426    .0041387     11.51   0.000     .0395233     .055762
   1984    .0904116    .0045211     20.00   0.000     .0815642     .0992811
   1985    .1263614    .0049682     25.43   0.000     .1166148     .136108
   1986    .1502841    .005596      26.86   0.000     .1393057     .1612625
   1987    .1953998    .0061582     31.73   0.000     .1833186     .2074809
   1988    .2301552    .0066879     34.41   0.000     .2170348     .2432757
   1989    .2576538    .0070179     36.71   0.000     .2438861     .2714215
   1990    .2850406    .0073995     38.52   0.000     .2705243     .2995569
   1991    .2735001    .0076954     35.54   0.000     .2584032     .288597
   1992    .2674434    .0079728     33.54   0.000     .2518022     .2830846
   1993    .2722996    .0085955     31.68   0.000     .255437      .2891623
   1994    .3012396    .0089872     33.52   0.000     .2836086     .3188707
   1995    .331336     .0094764     34.96   0.000     .3127452     .3499269
   1996    .3449819    .0099649     34.62   0.000     .3254325     .3645312
   1997    .3711286    .0103622     35.82   0.000     .3507999     .3914573
   1998    .3710202    .0106592     34.81   0.000     .3501089     .3919315
   1999    .3878193    .010936      35.46   0.000     .3663648     .4092737
         |
  _cons    7.547977    .0051162   1475.32   0.000     7.53794      7.558014
```

36

```
. xtreg log_retail_emp c.exp0#c.w c.exp1#c.w c.exp2#c.w c.exp3#c.w c.exp4#c.w ///
>     c.exp5#c.w c.exp6#c.w c.exp7#c.w c.exp8#c.w c.exp9#c.w c.exp10#c.w ///
>     c.exp11#c.w c.exp12#c.w c.exp13#c.w i.year, fe vce(cluster fips)

--------------------------------------------------------------------------------
 log_retail~p | Coefficient  std. err.      t    P>|t|     [95% conf. interval]
--------------+-----------------------------------------------------------------
  c.exp0#c.w  |    .0260239    .0043407     6.00   0.000      .0175082    .0345395
  c.exp1#c.w  |    .0405417    .0044418     9.13   0.000      .0318278    .0492557
  c.exp2#c.w  |    .0327549    .0045172     7.25   0.000       .023893    .0416168
  c.exp3#c.w  |    .0281223    .0047973     5.86   0.000      .0187109    .0375337
  c.exp4#c.w  |     .026921     .005103     5.28   0.000      .0169097    .0369322
  c.exp5#c.w  |    .0294158    .0054844     5.36   0.000      .0186564    .0401751
  c.exp6#c.w  |    .0316174    .0060162     5.26   0.000      .0198147    .0434201
  c.exp7#c.w  |    .0371473    .0070739     5.25   0.000      .0232697     .051025
  c.exp8#c.w  |    .0406383    .0083609     4.86   0.000      .0242358    .0570407
```

37

```
c.exp9h0.w  |   .0418278    .0094813     4.41   0.000     .0232273    .0604283
c.exp10h0.w |   .0593063    .0098245     6.04   0.000     .0400324    .0785801
c.exp11h0.w |   .0658462    .0105913     6.22   0.000      .045068    .0866243
c.exp12h0.w |   .0734672    .0144296     5.09   0.000     .0451591    .1017752
c.exp13h0.w |    .087957    .0160304     5.49   0.000     .0565085    .1194056
```

38

```
year |
   1978 |    .0730682    .0019093     38.27    0.000     .0693226     .0768138
   1979 |    .1090979    .0025142     43.39    0.000     .1041655     .1140303
   1980 |    .0852645    .0028309     30.12    0.000     .0797108     .0908181
   1981 |    .0708761    .0033187     21.36    0.000     .0643654     .0773868
   1982 |    .0732567    .0039152     18.71    0.000     .0655799     .0809336
   1983 |    .0476426    .0041396     11.51    0.000     .0395215     .0557638
   1984 |    .0904116    .0045221     19.99    0.000     .0815401     .0992831
   1985 |    .1263614    .0049693     25.43    0.000     .1166126     .1361102
   1986 |    .1507921    .0056007     26.92    0.000     .1398045     .1617797
   1987 |    .1955097    .0061657     31.71    0.000     .1834139     .2076056
   1988 |    .2302807     .006669     34.42    0.000     .2171563     .2434052
   1989 |    .2584009    .0069906     36.96    0.000     .2446867     .2721151
   1990 |    .2865656    .0073255     39.12    0.000     .2721945     .3009368
   1991 |    .2750423    .0075585     36.39    0.000      .260214     .2898706
   1992 |     .269186    .0078107     34.46    0.000     .2538628     .2845091
   1993 |    .2743704    .0084198     32.59    0.000     .2578524     .2908883
   1994 |    .3027812    .0088957     34.04    0.000     .2853295     .3202329
   1995 |    .3328265    .0094779     35.12    0.000     .3143297     .3513204
   1996 |    .3442326    .0100762     34.16    0.000      .324465     .3640002
   1997 |    .3675389    .0105959     34.69    0.000     .3467517      .388326
   1998 |    .3635317    .0111372     32.64    0.000     .3416827     .3853807
   1999 |    .3750971     .011751     31.92    0.000     .3520439     .3981502
        |
  _cons |    7.547977    .0051117   1476.62    0.000     7.537949     7.558005
```

-------------+---------------------------------------------------------------

```
39
```

```
. * Estimate separate effects by cohort/year, and then weight the coefficients.
. * Similar to imposing constant effects by exposure time.

. qui xtreg log_retail_emp c.wic_d1986#(c.f1986 c.f1987 c.f1988 c.f1989 c.f1990 c.f1991 ///
>     c.f1992 c.f1993 c.f1994 c.f1995 c.f1996 c.f1997 c.f1998 c.f1999) ///
>     c.wic_d1987#(c.f1987 c.f1988 c.f1989 c.f1990 c.f1991 ///
>     c.f1992 c.f1993 c.f1994 c.f1995 c.f1996 c.f1997 c.f1998 c.f1999) ///
>     c.wic_d1988#(c.f1988 c.f1989 c.f1990 c.f1991 ///
>     c.f1992 c.f1993 c.f1994 c.f1995 c.f1996 c.f1997 c.f1998 c.f1999) ///
>     c.wic_d1989#(c.f1989 c.f1990 c.f1991 ///
>     c.f1992 c.f1993 c.f1994 c.f1995 c.f1996 c.f1997 c.f1998 c.f1999) ///
>     c.wic_d1990#(c.f1990 c.f1991 c.f1992 c.f1993 c.f1994 c.f1995 c.f1996 ///
>     c.f1997 c.f1998 c.f1999) ///
>     c.wic_d1991#(c.f1991 c.f1992 c.f1993 c.f1994 c.f1995 c.f1996 c.f1997 c.f1998 c.f1999) ///
>     c.wic_d1992#(c.f1992 c.f1993 c.f1994 c.f1995 c.f1996 c.f1997 c.f1998 c.f1999) ///
>     c.wic_d1993#(c.f1993 c.f1994 c.f1995 c.f1996 c.f1997 c.f1998 c.f1999) ///
>     c.wic_d1994#(c.f1994 c.f1995 c.f1996 c.f1997 c.f1998 c.f1999) ///
>     c.wic_d1995#(c.f1995 c.f1996 c.f1997 c.f1998 c.f1999) ///
>     c.wic_d1996#(c.f1996 c.f1997 c.f1998 c.f1999) ///
>     c.wic_d1997#(c.f1997 c.f1998 c.f1999) ///
>     c.wic_d1998#(c.f1998 c.f1999) ///
>     c.wic_d1999#c.f1999 i.year, fe vce(cluster fips)
```

40

. lincom omega1986\_0 *c.wH#c.di1986#c.fi1986 + omega1987\_0* c.wH#c.di1987#c.fi1987 ///

```
+ omega1988_0*c.wH#c.di1988#c.fi1988 + omega1989_0*c.wH#c.di1989#c.fi1989 ///
   + omega1990_0*c.wH#c.di1990#c.fi1990 + omega1991_0*c.wH#c.di1991#c.fi1991 ///
   + omega1992_0*c.wH#c.di1992#c.fi1992 + omega1993_0*c.wH#c.di1993#c.fi1993 ///
   + omega1994_0*c.wH#c.di1994#c.fi1994 + omega1995_0*c.wH#c.di1995#c.fi1995 ///
   + omega1996_0*c.wH#c.di1996#c.fi1996 + omega1997_0*c.wH#c.di1997#c.fi1997 ///
   + omega1998_0*c.wH#c.di1998#c.fi1998 + omega1999_0*c.wH#c.di1999#c.fi1999
```

## 

log\_retail~p | Coefficient  Std. err.      t    P&gt;|t|     [95% conf. interval]

-------------+----------------------------------------------------------------

(1) |    .0290702   .0040876     7.11   0.000      .0210512    .0370893

. lincom omega1986\_1 *c.wH#c.di1986#c.fi1987 + omega1987\_1* c.wH#c.di1987#c.fi1988 ///

```
+ omega1988_1*c.wH#c.di1988#c.fi1989 + omega1989_1*c.wH#c.di1989#c.fi1990 ///
   + omega1990_1*c.wH#c.di1990#c.fi1991 + omega1991_1*c.wH#c.di1991#c.fi1992 ///
   + omega1992_1*c.wH#c.di1992#c.fi1993 + omega1993_1*c.wH#c.di1993#c.fi1994 ///
   + omega1994_1*c.wH#c.di1994#c.fi1995 + omega1995_1*c.wH#c.di1995#c.fi1996 ///
   + omega1996_1*c.wH#c.di1996#c.fi1997 + omega1997_1*c.wH#c.di1997#c.fi1998 ///
   + omega1998_1*c.wH#c.di1998#c.fi1999
```

## 

log\_retail~p | Coefficient  Std. err.      t    P&gt;|t|     [95% conf. interval]

-------------+----------------------------------------------------------------

(1) |    .0438954   .0042264    10.39   0.000      .0356041    .0521867

41

```
. lincom omega1986_11*c.wic.d1986#c.f1997 + omega1987_11*c.wic.d1987#c.f1998 ///
>        + omega1988_11*c.wic.d1988#c.f1999

------------------------------------------------------------------------------
log_retail~p | Coefficient  Std. err.        t     P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
         (1) |   .0670685    .0120642     5.56    0.000      .0434008    .0907361
------------------------------------------------------------------------------

. lincom omega1986_12*c.wic.d1986#c.f1998 + omega1987_12*c.wic.d1987#c.f1999

------------------------------------------------------------------------------
log_retail~p | Coefficient  Std. err.        t     P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
         (1) |   .0705872    .0160293     4.40    0.000      .0391408    .1020336
------------------------------------------------------------------------------

. * One coefficient for 13 years exposure:

. lincom c.wic.d1986#c.f1999

------------------------------------------------------------------------------
log_retail~p | Coefficient  Std. err.        t     P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
         (1) |   .0894949    .0209551     4.27    0.000      .0483851    .1306048
------------------------------------------------------------------------------
```

42

```
. * Heterogeneous trends by cohort. Effects are uniformly smaller:

. xtreg log_retail_emp c.exp0#c.w c.exp1#c.w c.exp2#c.w c.exp3#c.w c.exp4#c.w ///
>     c.exp5#c.w c.exp6#c.w c.exp7#c.w c.exp8#c.w c.exp9#c.w c.exp10#c.w ///
>     c.exp11#c.w c.exp12#c.w c.exp13#c.w ///
>     c.year#c.d1996 c.d1987 c.d1988 c.d1989 c.d1990 c.d1991 c.d1992 c.d1993 ///
>     c.d1994 c.d1995 c.d1996 c.d1997 c.d1998 c.d1999 i.year, fe vce(cluster fips)

                                                Robust
log_retail_emp | Coefficient  std. err.      t    P>|t|     [95% conf. interval]
---------------+----------------------------------------------------------------
     c.exp0#c.w |  -.0034236    .0036042    -0.95   0.342    -.0104942    .0036471
     c.exp1#c.w |   .0184111    .0037281     4.94   0.000     .0110973    .0257248
     c.exp2#c.w |   .0112132    .0037578     2.98   0.003     .0038411    .0185853
     c.exp3#c.w |   .0069585     .003944     1.76   0.078    -.0007788    .0146959
     c.exp4#c.w |   .0059511    .0042638     1.40   0.163    -.0024137    .0143158
     c.exp5#c.w |   .0065552    .0046092     1.42   0.155    -.0024873    .0155976
     c.exp6#c.w |   .0084336     .005066     1.66   0.096    -.0015049     .018372
     c.exp7#c.w |   .0129591    .0058303     2.22   0.026     .0015212     .024397
```

43

```
c.exp8#c.w    .0155207    .0068423    2.27    0.023    .0020975    .0289439
c.exp9#c.w    .0166873    .0079335    2.10    0.036    .0011233    .0322513
c.exp10#c.w   .0250665    .0080798    3.10    0.002    .0092155    .0409175
c.exp11#c.w   .0280936    .0095083    2.95    0.003    .0094402    .0467471
c.exp12#c.w   .0404849    .0120386    3.36    0.001    .0168675    .0641023
c.exp13#c.w   .0527214    .0137155    3.84    0.000    .0258142    .0796286
```

44

```
c.year#c.d1986      .008707    .0026515     3.28   0.001     .0035052    .0139088
c.year#c.d1987      .007829    .0023175     3.38   0.001     .0032825    .0123755
c.year#c.d1988     .0121625    .0020285     6.01   0.000     .0081899    .0161351
c.year#c.d1989     .0096362    .0017829     5.40   0.000     .0061384     .013134
c.year#c.d1990     .0085395    .0014877     5.74   0.000     .0056208    .0114581
c.year#c.d1991     .0097681    .0016158     6.05   0.000     .0065983    .0129379
c.year#c.d1992     .0095503    .0017016     5.61   0.000      .006212    .0128885
c.year#c.d1993     .0087822    .0016119     5.45   0.000       .00562    .0119444
c.year#c.d1994     .0103002    .0018614     5.53   0.000     .0066486    .0139518
c.year#c.d1995     .0063531    .0017904     3.55   0.000     .0028406    .0098656
c.year#c.d1996     .0135357    .0033423     4.05   0.000     .0069789    .0200926
c.year#c.d1997     .0133329    .0030466     4.38   0.000     .0073561    .0193097
c.year#c.d1998     .0096277    .0031339     3.07   0.002     .0034795    .0157759
c.year#c.d1999     .0096849    .0036959     2.62   0.009     .0024343    .0169356
...
```

45

- See `walmart_lw_retail_n_open_slides.do` for full heterogeneity in slopes with trend.
- For the immediate effect:

```
. lincom omega1986_0*c.whe.d1986#c.f1986 + omega1987_0*c.whe.d1987#c.f1987 ///
>       + omega1988_0*c.whe.d1988#c.f1988 + omega1989_0*c.whe.d1989#c.f1989 ///
>       + omega1990_0*c.whe.d1990#c.f1990 + omega1991_0*c.whe.d1991#c.f1991 ///
>       + omega1992_0*c.whe.d1992#c.f1992 + omega1993_0*c.whe.d1993#c.f1993 ///
>       + omega1994_0*c.whe.d1994#c.f1994 + omega1995_0*c.whe.d1995#c.f1995 ///
>       + omega1996_0*c.whe.d1996#c.f1996 + omega1997_0*c.whe.d1997#c.f1997 ///
>       + omega1998_0*c.whe.d1998#c.f1998 + omega1999_0*c.whe.d1999#c.f1999

------------------------------------------------------------------------------
log_retail~p | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
        (1)  |   .0049229    .0033239     1.48   0.139     -.001598     .0114437
------------------------------------------------------------------------------
```

46

. lincom omega1986\_1+c.w#c.d1986#c.f1987 + omega1987\_1+c.w#c.d1987#c.f1988 ///

```
+ omega1988_1+c.w#c.d1988#c.f1989 + omega1989_1+c.w#c.d1989#c.f1990 ///
  + omega1990_1+c.w#c.d1990#c.f1991 + omega1991_1+c.w#c.d1991#c.f1992 ///
  + omega1992_1+c.w#c.d1992#c.f1993 + omega1993_1+c.w#c.d1993#c.f1994 ///
  + omega1994_1+c.w#c.d1994#c.f1995 + omega1995_1+c.w#c.d1995#c.f1996 ///
  + omega1996_1+c.w#c.d1996#c.f1997 + omega1997_1+c.w#c.d1997#c.f1998 ///
  + omega1998_1+c.w#c.d1998#c.f1999
```

## 

log\_retail~p | Coefficient  Std. err.      t    P&gt;|t|     [95% conf. interval]

-------------+----------------------------------------------------------------

(1) |    .0195912   .0033805     5.80   0.000      .0129594    .026223

47

```
. lincom omega1986_11*c.wic.d1986#c.f1997 + omega1987_11*c.wic.d1987#c.f1998 ///
>        + omega1988_11*c.wic.d1988#c.f1999

------------------------------------------------------------------------------
log_retail~p | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
         (1) |   .0422307    .0146472    2.88   0.004      .0134956    .0709657
------------------------------------------------------------------------------

. lincom omega1986_12*c.wic.d1986#c.f1998 + omega1987_12*c.wic.d1987#c.f1999

------------------------------------------------------------------------------
log_retail~p | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
         (1) |   .0587202    .0200277    2.93   0.003      .0194296    .0980108
------------------------------------------------------------------------------

. lincom c.wic.d1986#c.f1999

------------------------------------------------------------------------------
log_retail~p | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
         (1) |   .0925373    .0301741    3.07   0.002      .0333414    .1517332
------------------------------------------------------------------------------
```

48

- Collapsing to a binary treatment:

**Weighted Effects by Exposure Length**

Y-axis: **Weighted Treatment Effects**

X-axis: **Exposure Length**

Y-axis tick labels: .4, .2, 0, -.2, -.4

X-axis tick labels: -21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12

49

- Cohort-specific linear trends:

**Weighted Effects by Exposure Length**

**Y-axis:** Weighted Treatment Effects

**X-axis:** Exposure Length

**Y-axis ticks:** .15, .1, .05, 0, -.05, -.1

**X-axis ticks:** -21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12

50

## 5 Summary

- TWFE has been applied to flexible equations to estimate ATTs by cohort/time.
    - Can obtain “lags only” or “leads and lags.”
    - Aggregation is straightforward, with valid standard errors.
- Can extend to non-binary treatments in a heterogeneous coefficients setting.
    - Dose-response function for the treated.
    - Impose a restriction on selection into treatment.

51

- Can apply TWFE to a flexible equation, and aggregate the coefficients.
    - Use leads or heterogeneous trends to test for violation of PT.
- Would be easy to modify `xthdidregress` to allow non-binary treatments.
    - Can reduce treatment to binary indicator to check `xtreg` commands.
- `wooldid` (Thomas Hegland) allows a single “continuous” treatment, but lots of moving parts.
- Could even allow for exit: keep estimating effects after the policy goes away.

52