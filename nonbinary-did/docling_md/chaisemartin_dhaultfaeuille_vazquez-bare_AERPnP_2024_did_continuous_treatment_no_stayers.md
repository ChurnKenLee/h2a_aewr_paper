*AEA Papers and Proceedings 2021, 111: 630–633* https://doi.org/10.1257/pandp.20211090

# Difference-in-Difference Estimators with Continuous Treatments and No Stayers†

*By Clément de Chaisemartin, Xavier D’Haultfoeuille, and Gonzalo Vazquez-Bare*

***

Many treatments or policy interventions are continuous in nature. Examples include prices, taxes, and temperatures. Empirical researchers have usually relied on two-way fixed effect regressions to estimate treatment effects in such cases (see, e.g., Deschênes and Greenstone 2012). However, such estimators are not robust to heterogeneous treatment effects in general (de Chaisemartin and D’Haultfoeuille 2020); they also rely on the linearity of treatment effects. We propose estimators for continuous treatments that do not impose those restrictions and that can be used when there are no stayers: the treatment of all units changes from one period to the next. This is, for instance, the case when the treatment is precipitation or temperatures: for example, temperatures of all US counties change, if ever so slightly, between two consecutive years. We start by extending the nonparametric results of de Chaisemartin et al. (2022) to cases without stayers. We also present a parametric estimator and use it to revisit Deschênes and Greenstone (2012).

## I. Setup, Assumptions, and Parameter of Interest

A representative unit is drawn from an infinite superpopulation and observed at two time periods. All expectations below are taken with respect to the distribution of variables in the superpopulation. We are interested in the effect of a continuous and scalar treatment variable on that unit’s outcome. Let ( D\_t ) denote the unit’s treatment at period ( t \in { 1,2 } ) , and let ( \mathcal{D}\_t ) denote its support; let also ( \mathcal{D} ) denote the support of ( (D\_1,D\_2) ) . For any ( (d\_1,d\_2) \in \mathcal{D} ) , let ( Y\_t(d\_1,d\_2) ) denote the unit’s potential outcome at ( t ) with treatment ( d\_1 ) and ( d\_2 ) ; let ( Y\_t ) denote their observed outcomes: ( Y\_t = Y\_t(D\_1,D\_2) ) . Finally, for any two real numbers ( (y\_1,y\_2) ) , let ( \Delta y = y\_2 - y\_1 ) . We impose the following assumptions:

**ASSUMPTION 1 (Static Model):** For all ( t \in { 1,2 } ) and ( (d\_1,d\_2) \in \mathcal{D} ) , ( Y\_t(d\_1,d\_2) ) only depends on ( d\_t ) ; we denote it by ( Y\_t(d\_t) ) .

# 

**ASSUMPTION 2 (Parallel Trends):**

[

\forall d \in \mathcal{D}\_1,\quad

E[\Delta Y(d)\mid D\_1=d,D\_2]

E[\Delta Y(d)\mid D\_1=d]. ]

**ASSUMPTION 3 (Bounded Treatment-Induced Lipschitz Potential Outcomes):**

(i) ( \mathcal{D}\_1 ) and ( \mathcal{D}\_2 ) are bounded subsets of ( \mathbb{R} ) .

(ii) ( \exists \bar{\gamma}&gt;0: )

[ \sup\_{d,d'\in \mathcal{D}\_1\cup\mathcal{D}\_2,\ d\ne d'} \left| \frac{Y\_2(d')-Y\_2(d)}{d'-d} \right| \le \bar{\gamma}. ]

Assumptions 2 and 3 are also imposed by de Chaisemartin et al. (2023) and are discussed therein.

**ASSUMPTION 4 (No Stayers but Quasistayers):** [ P(\Delta D=0)=0,\quad P(|\Delta D|\le \eta)&gt;0,\quad \forall \eta&gt;0. ]

First, Assumption 4 states that there are no “stayers,” namely units for which ( D\_1 = D\_2 ) . This is in contrast with de Chaisemartin et al. (2023), who assume throughout that there are stayers. Second, Assumption 4 states that there are quasi-stayers, namely units whose treatment change may be infinitesimally small. This assumption is realistic when the treatment is, say, temperatures: some counties may have

* de Chaisemartin: Sciences Po Paris (email: clement.dechaisemartin@sciencespo.fr); D’Haultfoeuille: CREST-ENSAE (email: xavier.dhaultfoeuille@ensae.fr); Vazquez-Bare: University of California, Santa Barbara (email: gvazquezbare@ucsb.edu). Clément de Chaisemartin was funded by the European Union (ERC, REALLYCREDIBLE, Grant agreement number 101043899).

† Go to https://doi.org/10.1257/pandp.20211090 to visit the article page for additional materials and author disclosure statement(s).

630

VOL. 114 DIFFERENCE-IN-DIFFERENCE ESTIMATORS WITH CONTINUOUS TREATMENTS AND NO STAYERS 611

very similar temperatures from one year to the next, though no county has exactly the same temperatures.

Hereafter, we focus on the following effect:

[ \begin{aligned} (1)\quad \theta\_s &amp;= \frac{ E\left[ s(\Delta D)\times \frac{Y\_1(D\_1)-Y\_1(D\_0)}{D\_1-D\_0} \right] }{ E[s(\Delta D)] } \ &amp;\quad - \frac{ E\left[ s(\Delta D)\times \frac{Y\_0(D\_1)-Y\_0(D\_0)}{D\_1-D\_0} \right] }{ E[s(\Delta D)] }. \end{aligned} ]

( \theta\_s ) is a weighted average of the slopes of units’ potential-outcome functions, from their period-one to their period-two treatment, the so-called WAOSS in de Chaisemartin et al. (2023). It follows from the mean-value theorem that it may be seen as a weighted average marginal effect.

## II. Nonparametric Identification and Estimation

**THEOREM 1:** *If Assumptions 1–4 hold,*

[ \theta\_0 = \frac{E[S\Delta Y]-c\_0}{E[|\Delta D|]}, ]

*with*

*(*

*S=\operatorname{sgn}(\Delta D)*

*)*

*and*

[ c\_0= E\left[ S\lim\_{\epsilon\downarrow 0} E[\Delta Y\mid D\_1-D\_0\in[-\epsilon,\epsilon]] \right]. ]

Theorem 1 shows that without stayers, ( \theta\_0 ) is identified by the limit (as ( \epsilon\downarrow 0 ) ) of a difference-in-differences comparing the ( \Delta Y ) of all units and of quasi-stayers.

We now discuss estimation of ( \theta\_0 ) . Only the estimation of ( c\_0 ) raises difficulties. We show in the proof of Theorem 1 that under our assumptions, ( s(d,c)=E[\Delta Y\mid D\_1-D\_0=d,D\_0=c] ) is well defined and continuous at ( (d,c) ) for any ( c ) . Then, ( c\_0 ) satisfies ( c\_0=E[S]\int s(0,c)f\_{D\_0}(c)dc ) . This formulation links our problem to the estimation of nonparametric additive models. To see this, suppose that the variables ( (Y,X)\in \mathbb{R}\times\mathbb{R}^d ) satisfy ( E(Y\mid X)=E(Y\mid X)=\sum\_{j=1}^d h\_j(X\_j) ) for some unknown functions ( (h\_j)\_{j=1}^d ) . Then under the normalization ( E[h\_j(X\_j)]=0 ) for ( j&lt;d ) , we can identify and estimate ( h\_d ) by remarking that

[ (2)\quad h\_d(x\_d)=E[h(X\_1,\ldots,X\_{d-1},x\_d)]. ]

We can then estimate ( h\_d(x\_d) ) by first estimating ( h ) with any usual nonparametric estimator and then plugging it in the sample counterpart of the expectation in (2). As Linton and Nielsen (1995) and Kong, Linton, and Xia (2010) show, the corresponding estimator is, under regularity conditions, asymptotically normal and converges at the standard univariate nonparametric rate (namely, ( n^{-2/5} ) , with ( n ) the sample size). This rate is also the optimal convergence rate for this problem (Stone 1985). Up to minor changes (in C9 ( p ) plays the role of ( d ) in (2) and ( c\_0 ) also includes ( S ) ), our parameter ( c\_0 ) can be obtained in the same way as ( h\_d(x\_d) ) , so we can also obtain an asymptotically normal estimator converging at the ( n^{-2/5} ) rate.

This contrasts with the standard ( (n/h)^{1/2} ) rate obtained for the estimators of the WAOSS in the presence of stayers, as shown by de Chaisemartin et al. (2023). To understand the difference, note that with stayers, the proportion of units used as controls to reconstruct switchers’ counterfactual outcome evolution remains positive as ( n\to\infty ) . On the other hand, it tends to zero here since we need to consider quasi-stayers, with ( \gamma\_n\to 0 ) as ( n\to\infty ) to avoid any bias. This results in a lower rate of convergence.

Finally, in applications with no stayers, it is more difficult to propose placebo estimators of the parallel trends assumption. When a three-period of data, period zero, is available, a placebo mimics the actual estimator, replacing ( \Delta Y ) by units’ period-zero-to-one outcome evolution. However, as small treatments may have changed from period zero to one, one would need to restrict the sample to period-zero-to-one quasi-stayers to avoid the placebo differing from zero due to the treatment’s effect. Thus, the placebo would compare the period-zero-to-one outcome evolution of period-one-to-two switchers and quasi-stayers, restricting the sample to period-zero-to-one quasi-stayers. Then we conjecture that the number of units used as controls by the placebo may tend to zero faster than the number of units used as controls by the actual estimator (for instance if being a period-zero-to-one and a period-one-to-two quasi-stayer are independent events). Then the placebo may converge at an even slower rate than the actual estimator.

## III. A Parametric Approach

We now consider a parametric root- ( n ) consistent estimator that avoids issues related to nonparametric estimation and inference while still allowing for heterogeneous and

612 IEEE PAPERS AND PROCEEDINGS MAY 2024

nonlinear effects. Specifically, we impose that ( \delta\_1(d\_0,\delta)=g\_1(d\_0,\delta) ) , where the family ( (g\_t)\_{t\geq0} ) is known (but ( \lambda\_0 ) is not). By definition of ( \delta ) and Assumption 2,

# 

[

\delta\_1(d\_0,\delta)

E[Y\_i(d\_1)-Y\_i(d\_0)\mid D\_i=d\_1]+\delta ]

[ \times E\left[ \frac{Y\_i(d\_1+\delta)-Y\_i(d\_1)}{\delta} \mid D\_i=d\_1,\Delta D=\delta \right]. ]

Thus, the parametric assumption amounts to imposing restrictions on both ( d\_1\mapsto E[Y\_i(d\_1)-Y\_i(d\_0)\mid D\_i=d\_1] ) and on the average slope ( (Y\_i(d\_1+\delta)-Y\_i(d\_1))/\delta ) at ( d\_1 ) if ( \Delta D=\delta ) . For instance, if ( \delta\_1(d\_0,\delta) ) is linear, we assume that the former functions is linear and the latter is constant. Similarly, ( \delta\_1 ) is a polynomial if both functions are polynomial. Note that we can test the null ( E[\Delta Y\_i\mid D\_i=d\_i,\Delta D=\delta]=\delta\_1(d\_i,\delta) ) for some ( d\_i ) by a parametric specification test (see, e.g., Bierens 1982 or Hong and White 1995).

We consider a simple two-step GMM estimator, based on this parametric restriction and an i.i.d. sample ( (D\_i,\Delta D\_i,\Delta Y\_i)\_{i=1,\ldots,n} ) . In the first step, we estimate ( \lambda ) by (linear or nonlinear) least squares or, more generally, a generalized method of moments (GMM) estimator ( \hat\lambda ) . In the second step, we estimate ( \theta ) by

[ \hat\theta= \frac{\sum\_{i=1}^n \Delta Y\_i-\hat\delta\_1(D\_i,0)} {\sum\_{i=1}^n \Delta D\_i}. ]

Since ( \hat\theta ) may be seen as a two-step GMM estimator, we obtain, under Assumptions 1–4 and standard regularity conditions on ( \lambda\mapsto \delta\_1(d\_0,\delta) ) ,

[ \sqrt{n}(\hat\theta-\theta\_0)\xrightarrow{d}\mathcal{N}(0,V(\theta)), ]

where the influence function ( \ell ) satisfies

# 

[

\ell

E[\Delta D\_i]^{-1} \left { S\_i(\Delta Y\_i-\delta\_\lambda(D\_i,0)) \right. ]

## 

[

\left.

E[\dot\delta\_\lambda(D\_i,0) *{\lambda=\lambda\_0}]* *\times* *[-G* \lambda^{-1}A\_\lambda(D\_i)] \right } . ]

with ( G ) the influence function of ( \hat\lambda ) . We can thus simply estimate ( V(\theta) ) by a plug-in estimator, using an initial estimator of ( \ell ) .

## IV. Application

We use the data from Deschênes and Greenstone (2012) to compute our parametric estimator. The authors use a balanced panel of 2,342 US counties in years 1987, 1992, 1997, and 2002 and consider two-way fixed effect (TWFE) regressions weighted by counties’ farmland acres, of annual agricultural profits in county ( c ) and year ( t ) on four treatment variables: growing season degree days, growing season degree days squared, precipitation, and precipitation squared. To fit in the three-periods-one-treatment case we consider, we restrict the data to years 1997 and 2002, and we focus on the growing season degree days treatment. The coefficient of that treatment in a TWFE regression estimated on years 1997 and 2002 and weighted by counties’ farmland acres is equal to ( -0.024 ) (s.e. clustered at the county level: 0.087), which is close to the corresponding TWFE coefficient keeping the four years and all treatments ( ( -0.015 ) , standard error clustered at the county level: 0.038). Assuming that

# 

[

E[Y\_i(d\_1)-Y\_i(d\_0)\mid D\_i=d\_1]

\lambda\_{01}+\lambda\_{02}d\_1 ]

and

# 

[

E\left[

\frac{Y\_i(d\_1+\delta)-Y\_i(d\_1)}{\delta}

\mid

D\_i=d\_1,\Delta D=\delta

\right]

\lambda\_{03}+\lambda\_{04}d\_1+\lambda\_{05}\delta, ]

we find that ( \theta ) , weighted by counties’ farmland acres as well, is equal to ( -0.018 ) (standard error: 0.011). Thus, the conclusion from the TWFE regression seems robust to allowing for some effect heterogeneity, even though the estimated effect is less significant. While arguably restrictive, our model for the conditional expectation function of slopes allows for some nonlinearity and heterogeneity in the effects of temperature on agricultural output.

## Appendix

### Proof of Theorem 1:

It suffices to show that almost surely,

[ \lim\_{|\delta|\to0}E[\Delta Y\_i\mid D\_i,\Delta D\_i=\delta] ]

# [

E[Y\_i(D\_i)-Y\_i(D\_i)\mid D\_i,D\_i]. ]

VOL. 114 DIFFERENCE-IN-DIFFERENCE ESTIMATORS WITH CONTINUOUS TREATMENTS AND NO STAYERS 617

Fix ( \eta &gt; 0 ) . By Assumption 4, ( P[|\Delta D\_i| \le \eta \mid \boldsymbol{D}\_i] &gt; 0 ) . Thus, ( E[|\Delta Y\_i| \mid \boldsymbol{D}\_i, |\Delta D\_i| \le \eta] ) is well defined. Moreover,

[ \begin{aligned} (4)\quad &amp;E[|\Delta Y\_i| \mid \boldsymbol{D} *i, |\Delta D\_i| \le \eta]* *\* *&amp;\le E[|Y* {i2}(D\_{i1}) - Y\_{i1}(D\_{i1})| \mid \boldsymbol{D} *i, |\Delta D\_i| \le \eta]* *\* *&amp;\quad + E[|Y* {i2}(D\_{i2}) - Y\_{i2}(D\_{i1})| \mid \boldsymbol{D}\_i, |\Delta D\_i| \le \eta]. \end{aligned} ]

Now, by Jensen’s inequality and (ii) of Assumption 3,

[ \begin{aligned} (5)\quad &amp;|E[Y\_{i2}(D\_{i2}) - Y\_{i2}(D\_{i1}) \mid \boldsymbol{D} *i, |\Delta D\_i| \le \eta]|* *\* *&amp;\le E[|Y* {i2}(D\_{i2}) - Y\_{i2}(D\_{i1})| \mid \boldsymbol{D} *i, |\Delta D\_i| \le \eta]* *\* *&amp;\le E[|D* {i2} - D\_{i1}| \mid \boldsymbol{D} *i, |\Delta D\_i| \le \eta]* *\* *&amp;\quad \times E\left[\sup* {d\_{i2},d\_{i1}:d\_{i2}\ne d\_{i1}} \left|\frac{Y\_{i2}(d\_{i2}) - Y\_{i2}(d\_{i1})}{d\_{i2} - d\_{i1}}\right|\right] \ &amp;\le \eta K \end{aligned} ]

for some ( K &lt; \infty ) . Next, by Assumption 2,

[ \begin{aligned} &amp;E[Y\_{i2}(D\_{i1}) - Y\_{i1}(D\_{i1}) \mid \boldsymbol{D} *i, |\Delta D\_i| \le \eta]* *\* *&amp;\quad = E[Y* {i2}(D\_{i1}) - Y\_{i1}(D\_{i1}) \mid \boldsymbol{D} *i]* *\* *&amp;\quad = E[Y* {i2}(D\_{i1}) - Y\_{i1}(D\_{i1}) \mid D\_{i1}]. \end{aligned} ]

Combined with (4)–(5), this yields (3). ( \blacksquare )

## REFERENCES

Bierens, Herman J. 1982. “Consistent Model Specification Tests.” *Journal of Econometrics* 20 (1): 105–34.

de Chaisemartin, Clément, and Xavier D’Haultfœuille. 2020. “Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects.” *American Economic Review* 110 (9): 2964–96.

de Chaisemartin, Clément, Xavier D’Haultfœuille, Félix Pasquier, and Gonzalo Vazquez-Bare. 2023. “Difference-in-Differences Estimators for Treatments Continuously Distributed at Every Period.” arXiv:2301.06898v3.

Deschênes, Olivier, and Michael Greenstone. 2012. “The Economic Impacts of Climate Change: Evidence from Agricultural Output and Random Fluctuations in Weather: Reply.” *American Economic Review* 102 (7): 3761–73.

Hong, Yongmiao, and Halbert White. 1995. “Consistent Specification Testing via Nonparametric Series Regression.” *Econometrica* 63 (5): 1133–59.

Kang, Haina, Oliver Linton, and Myung Hwan Seo. 2020. “Uniform Bahadur Representation for Local Polynomial Estimates of M-Regression and Its Application to the Additive Model.” *Econometric Theory* 36 (5): 1529–64.

Linton, Oliver, and Jens Perch Nielsen. 1995. “A Kernel Method of Estimating Structured Nonparametric Regression Based on Marginal Integration.” *Biometrika* 82 (1): 93–100.

Stone, Charles J. 1985. “Additive Regression and Other Nonparametric Models.” *Annals of Statistics* 13 (2): 689–705.