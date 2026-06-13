# Difference-in-Differences with a Continuous Treatment*

Brantly Callaway^†^ Andrew Goodman-Bacon^‡^ Pedro H. C. Sant’Anna^§^

First draft on arXiv: July 6, 2021. This draft: December 31, 2025

## Abstract

This paper analyzes difference-in-differences designs with a continuous treatment. We show that treatment-on-the-treated-type parameters are identified under a parallel trends assumption analogous to the binary treatment case. However, comparing these parameters across treatments is challenging because parallel trends does not rule out selection bias. We discuss alternative, typically stronger, assumptions that eliminate selection bias. We further show that popular two-way fixed effects estimands admit multiple interpretations, depending on the underlying causal building block, all having important limitations as meaningful summaries of treatment effects. Finally, we introduce alternative estimation procedures that avoid these drawbacks and demonstrate them in an empirical application.

**JEL Codes:** C14, C21, C23

**Keywords:** Difference-in-Differences, Continuous Treatment, Multi-Valued Discrete Treatment, Parallel Trends, Two-way Fixed Effects, Multiple Periods, Variation in Treatment Timing, Treatment Effect Heterogeneity

* We thank the participants of many seminars, workshops, and conferences for their comments. We are grateful to Xiaohong Chen for numerous discussions about implementing the distribution sieve estimator used in this paper, Amy Finkelstein for sharing her data with us, Cristi Cristia, Greg Caetano, Stefan Hoderlein, Jo Mullins, Jon Roth, and Abbie Wozniak for their comments, and Honey Batra for valuable research assistance. Code implementing the methods proposed in the paper is available in the R package contdid, which is available on CRAN. The views expressed here are those of the authors and do not necessarily represent those of the Federal Reserve Bank of Minneapolis or the Federal Reserve System. The Supplementary Appendix is available here. ^†^ University of Georgia. Email: brantly.callaway@uga.edu ^‡^ Federal Reserve Bank of Minneapolis and NBER. Email: andrew@jgoodman-bacon.com ^§^ Emory University. Email: pedro.santanna@emory.edu

# 1 Introduction

The canonical difference-in-differences (DID) research design compares outcomes before and after treatment started (difference one), between treated and untreated groups (difference two). However, in many DID applications, the treatment does not simply “turn on”; it has a “dose” or operates with varying intensity. Pollution diffuses across space, affecting locations near its source more severely than faraway locations. Localities spend different amounts on public goods and services, or set different minimum wages. Students choose how long to stay in school.

Continuous treatments can offer advantages over binary ones.[^1] Variation in intensity makes it possible to evaluate treatments that all units receive. A clear “dose-response” relationship between outcomes and treatment intensity can bolster the case for a causal interpretation or test a theoretical prediction. Finally, we may care more about the effect of changes in treatment intensity, such as increased funding, pollution abatement, or expanded eligibility, than about the effect of the existence of a treatment that already exists.

Despite these conceptually useful and practically common continuous DID designs, currently available econometric results provide little guidance on applying and interpreting them, except in some specific cases. In this paper, we introduce a set of tools that are suitable for DID setups with variation in treatment dosage. In particular, we (a) discuss how one can identify a variety of treatment effect parameters by exploiting parallel-trends-type assumptions, (b) demonstrate that a simple linear two-way fixed effects (TWFE) estimand accommodates multiple decompositions that are difficult to justify as meaningful summaries of treatment effects, and (c) propose “second-generation” estimators that directly target well-defined causal objects, allowing for more transparent interpretation and robust inference in applications with treatment effect heterogeneity.

To foster intuition and simplify exposition, we start by discussing causal parameters in a two-period DID design in which units move from no treatment to a non-zero dose. We call the difference between a unit’s potential outcome under dose *d* and its untreated potential outcome a *level treatment effect* . We call the difference in a unit’s potential outcome with a marginal increase in the dose a *causal response* (Angrist and Imbens, 1995). When treatment is binary, these two notions of treatment effects coincide, but they do not under a continuous treatment. Importantly, level treatment effects and causal responses can have meaningfully different interpretations, and we establish that they generally require different identifying assumptions as well.

Comparisons between treated and untreated units identify average (level) treatment effect parameters under a parallel trends assumption on untreated potential outcomes, similar to binary DID designs. Comparisons between adjacent dose groups, however, do not identify average causal response parameters under the “standard” parallel trends assumption. They include causal responses but are contaminated by an additional term involving possibly different treatment effects of the same dose for different dose groups — we refer to this additional term as selection bias.[^2] We discuss an alterna-

[^1]: We generally use “continuous” treatments also to mean multivalued ordered discrete treatments, but make the distinction explicit for certain results.

[^2]: In applications where units choose their amount of the treatment, it is natural to refer to this term as selection bias. In other applications where the dose measures a unit’s amount of exposure to some treatment, a different term,

1

tive, typically stronger assumption, which we call strong parallel trends, that says that the average evolution of outcomes for the entire treated population under dose $d$ is equal to the path of outcomes that dose group $d$ actually experienced. Thus, strong parallel trends justifies comparing dose groups by restricting treatment effect heterogeneity. Strong parallel trends may not be plausible in many applications. Currently, in empirical work, it is common for papers to write as if they have assumed standard parallel trends and interpret their results as if they have assumed strong parallel trends. Our results clarify what causal questions can be answered under standard parallel trends and what causal questions require stronger assumptions.

The ideas discussed above are in the spirit of what Mogstad and Torgovitsky (2024) call *forward engineering* , where the researcher clearly specifies target parameters and assumptions up front and builds estimators to implement the identification strategy. Our second main contribution is to reverse engineer the most common way that practitioners estimate continuous DiD designs, which is to run a TWFE regression that includes time fixed effects ($\theta\_t$), unit fixed effects ($\eta\_i$), and the interaction of a dummy for the post-treatment period ($Post\_t$) with a variable that measures unit $i$’s dose or treatment intensity, $D\_i$:

$$ Y\_{it} = \theta\_t + \eta\_i + \beta^{fe} D\_i \cdot Post\_t + \epsilon\_{it}. \tag{1.1} $$

This TWFE specification is clearly motivated by DiD setups with two periods and two treatment groups, though many prominent textbooks suggest using it in more general setups (e.g., Cameron and Trivedi, 2005, Angrist and Pischke, 2008, and Wooldridge, 2010). There are several ways to interpret $\beta^{fe}$, each corresponding to a different type of causal parameter. We decompose it in terms of level effects, scaled level effects, causal responses, and scaled high-versus-low ($2 \times 2$) effects. Each decomposition is a weighted integral of dose-specific causal parameters, and some provides a clear causal and policy-relevant interpretation of $\beta^{fe}$, at least not when treatment effects are allowed to vary across doses and/or groups.[^3]

For instance, we show that $\beta^{fe}$ can be expressed as a weighted integral of average level treatment effect parameters, but where the weights integrate to zero, indicating that $\beta^{fe}$ should not be interpreted as an average (level) treatment effect. Interestingly, however, TWFE puts negative weights on the below-average dose units and positive weights on above-average dose units, and, thus, after re-scaling by a weighted average of the difference between doses for high- and low-dose units, is equivalent to a weighted binary DiD using higher-dose units as the “treated” group and lower-dose units as the “comparison” group, with weights proportional to a unit’s absolute distance from the mean dose. Our next decomposition, based on average level treatment effect parameters scaled by their dose, also displays negative weights, though their weights integrate up to one and not zero.

In contrast, a TWFE decomposition in terms of average causal response parameters has weights that integrate up to one and are non-negative, but also includes a selection bias term stemming from

[^3]: As discussed below, in some cases other labels for the selection bias term, such as “heterogeneity bias,” could be more appropriate. For simplicity, throughout the paper, we simply refer to this term as selection bias.

[^2]: The decompositions that we provide are specific to the particular TWFE regression specification in Equation (1.1), which we focus on due to its ubiquity in empirical work with a continuous treatment. Some of the drawbacks we discuss below, particularly regarding weighting schemes inherited from the TWFE regression, could be addressed by considering a more flexible specification. See, e.g., Wooldridge (2025) for a discussion involving binary treatments.

2

effect heterogeneity across doses. The strong parallel trends assumption eliminates this selection bias. The weights on causal responses at different doses, however, differ from the distribution of the dose among the treated, which creates a further challenge to interpreting $\beta^{fe} *{ML}$, even if strong parallel trends holds. The weights are also undesirably sensitive to the size of the untreated group. In our application, if we drop the untreated group, which changes the weights but does not change the underlying average causal response, our estimate of $\beta^{fe}* {ML}$ shrinks by 78%. Our decomposition of $\beta^{fe} *{ML}$ based on scaled 2 × 2 average effects as building blocks also highlights limitations of using $\beta^{fe}* {ML}$ as a causal summary parameter.

We demonstrate that these drawbacks are easily avoidable and discuss different DiD estimators that build upon our identification results and recover interpretable causal parameters. When the treatment is discrete, this is as simple as running a linear regression with multiple treatment indicators, which is similar to staggered DiD setups (Callaway and Sant’Anna, 2021). When the treatment is continuous, there are several options, including adopting a parametric, semiparametric, or nonparametric regression model. In particular, we discuss how to adapt the data-driven sieve-based nonparametric regression proposed by Chen, Christensen, and Kankanala (2025) to our context, although we note that other semi/nonparametric procedures are also possible. We also show how to construct causal summary measures of our average treatment effect functions that bypass the TWFE weighting problem by using the dose density as weights. Our results suggest that one can easily summarize average level treatment effects among treated units by comparing the average change in outcomes for all treated units to the average change in outcomes for untreated units. This can be estimated by running a binary DiD with a “treatment dummy” equal to one for any units with positive doses. Summarizing average causal responses using dose density weights involves estimating an average derivative, which is simple to compute using “flexible” linear regressions. We also discuss how to construct event-study results using these summary measures, which can then be used to assess the plausibility of the parallel trends assumptions.

To contrast our proposed estimators with TWFE in practical settings, we revisit Acemoglu and Finkelstein’s 2008 study of a 1983 Medicare reform that eliminated labor subsidies for hospitals. The original paper uses a TWFE estimator to compare the change in capital-labor ratios between hospitals whose input prices were more or less affected by the end of the subsidy. It concludes that price regulations favoring capital significantly increase capital use. The distinction between level treatment effect parameters and causal responses is important in this example: a positive level treatment effect shows that the policy as a whole increased the use of capital, while causal responses describe which subsidy levels generated the largest responses. We find that the reform raised capital-labor ratios by about 18 percent (on average), which is 50 percent larger than the comparable TWFE estimate because of the weighting issues highlighted by our decompositions. We also estimate variable average causal response (ACRT) parameters that are quite large at low subsidy levels—implying elasticities of substitution greater than 2—yet slightly negative for most positive doses. These negative ACRT estimates cast doubt about the plausibility of the strong parallel trends assumption, the simple two-factor model of hospital production, or both. Our results support Acemoglu and Finkelstein’s 2008 conclusion that the 1983 Medicare reform led hospitals to favor capital over labor, but suggest

3

caution in a policy interpretation about which subsidy levels have the largest effects or an economic interpretation in terms of production function parameters.

**Related Literature:** Our paper contributes to the literature on modern DiD methods; see, e.g., Baker et al. (2025) for an overview. We contribute to this literature by highlighting challenges associated with using TWFE with continuous treatments, discussing the role of different parallel trends assumptions to learn about different causal parameters, and providing easy-to-use estimation procedures that can highlight treatment effect heterogeneity with continuous treatments.

The closest paper to ours is Fricke (2017), which focuses on DiD setups with two time periods and three treatment dosages: H, L, and 0. Fricke (2017) shows that under standard parallel trends, one can identify dose-specific average treatment effect among dose groups in two-period DiD designs. He also considers stronger assumptions that permit causal interpretation of the H vs. L contrast when an untreated group is not available. We generalize his identification results to DiD settings with richer treatment distributions, including continuous cases, multiple time periods, and staggered treatment adoption. This allows us to (i) discuss a broader set of parameters of interest that are suitable for incremental changes in treatment dose, (ii) discuss event-study and other types of treatment aggregation, (iii) derive decomposition results that question the causal meaning of TWFE estimates under treatment effect heterogeneity, and (iv) offer identification-based estimation templates for researchers to avoid the pitfalls of simple TWFE specifications with continuous/multivalued treatments.

Our paper is also related to a series of papers on more complicated non-binary DiD setups. de Chaisemartin and D’Haultfœuille (2018) focuses on fuzzy designs, where a researcher is interested in individual-level effects of a binary treatment that has been aggregated across units into a continuous “treatment rate.” In contrast, we study “sharp” designs in which the treatment exposure is itself continuous or multi-valued discrete at the unit-level. The approach proposed in de Chaisemartin and D’Haultfœuille (2025) can also accommodate continuous treatments, although they focus on aggregated target parameters rather than dose-specific estimands of treatment effect heterogeneity, as we do. They also do not discuss identification, nor estimation, related to average causal response parameters, an important focus of our analysis. de Chaisemartin et al. (2025) builds on de Chaisemartin and D’Haultfœuille (2025) and considers a DiD setup with continuous treatments with potentially non-staggered (but static) treatments. Our proposal differs in its target parameters and DiD designs. Similarly to de Chaisemartin and D’Haultfœuille (2025), de Chaisemartin et al. (2025) average effects of discrete rather than marginal changes of treatments. On the other hand, de Chaisemartin et al. (2025) allows for units to already be exposed to the treatment in the first period and considers instrumental variable extensions, which we do not.

Our decomposition also relates to the literature on TWFE bias in heterogeneous treatment effect settings with a binary treatment (e.g., Borusyak, Jaravel, and Spiess, 2024; de Chaisemartin and D’Haultfœuille, 2020; Goodman-Bacon, 2021; Sun and Abraham, 2021) but we extend this logic to continuous treatments and highlight that the same TWFE regression coefficient can have multiple interpretations depending on the “building blocks,” and that new “bias” terms may appear, depending on the type of parallel trends assumption being used. A perhaps more practically relevant message

4

from our decompositions is that, even when all weights are non-negative, TWFE can still provide an unappealing causal summary parameter. Interestingly, if one replaces our DiD setting with a cross-sectional design with a randomly assigned dose, all four of our decomposition results would still hold, highlighting that linear specifications may not be desirable with continuous treatments, even when the dose is fully randomized.

Finally, we note that some of our causal response decomposition builds on Yitzhaki (1996, Proposition 2), which expresses the slope coefficient in a regression of an outcome on a continuous variable as a weighted average of underlying local slopes. Besides differences related to causal interpretations and panel data, we mildly extend those results to accommodate a mass of untreated units.

## 2 Motivating Continuous DiD from an Empirical Perspective

To fix ideas and provide intuition for our results, we revisit Acemoglu and Finkelstein’s 2008 (AF) study of how price regulations affect firms’ input choices. When Medicare began in 1965, hospitals received reimbursements from the federal government for a share of their labor and capital expenditures proportional to the fraction of total patient days accounted for by Medicare recipients ( (m\_i) ) . Hospital ( i ) thus faced input prices equal to ( (1-\tau\_l m\_i)w\_i ) for labor and ( (1-\tau\_k m\_i)r\_i ) for capital, where ( \tau\_l ) and ( \tau\_k ) are the labor and capital subsidy rates and ( w\_i ) and ( r\_i ) are market wages and rental rates. In 1983, Medicare moved to the Prospective Payment System (PPS), which replaced the labor subsidy with a small payment per episode/diagnosis. This set ( \tau\_l=0 ) but left the capital subsidy unchanged. Therefore, the price of labor for a given hospital rose from ( (1-\tau\_l m\_i)w\_i ) to ( w\_i ) , skewing relative factor prices.

The statutory relationship between a hospital’s Medicare volume, ( m\_i ) , and the change in its price of labor, ( m\_i\tau\_l ) , motivates AF’s use of a continuous DiD design comparing changes in capital/labor ratios before and after 1983 between hospitals with different pre-PPS Medicare inpatient shares. AF’s description, estimation, and interpretation of this empirical strategy touch on some of the most common ways of justifying and implementing continuous DiD designs.

One motivation for this design is practical: variation in a dose (or exposure) permits the evaluation of treatments for which binary DiD is either infeasible or undesirable. In AF’s case, about 15 percent of hospitals were “untreated” by the change in Medicare’s subsidy policy because they served non-Medicare-eligible populations, like children or psychiatric patients, so one may be concerned about whether these constitute a valid comparison group. AF therefore describe ( m\_i ) , which is the hospital’s Medicare volume in 1983, as an “attractive source of variation” in the price of labor both because it varies substantially—the mean of ( m\_i ) among treated hospitals is 0.45, and the standard deviation is 0.15—and because hospitals with ( m\_i&gt;0 ) may be more comparable to each other than treated hospitals are to untreated hospitals.

Another common justification for continuous DiD designs is that a “dose-response” relationship between exposure and outcomes can support a causal interpretation or test a theoretical prediction. Meyer (1995, p. 158), for example, argues that “differences in the intensity of the treatment across different groups allow one to examine if the changes in outcomes differ across treatment levels in the

5

expected direction.” AF lay out a simple theoretical framework in which the move to PPS should (i) raise capital/labor ratios and (ii) do so more strongly for hospitals with higher pre-PPS values of $m\_j$. They view their continuous DiD design as a way to estimate a causal effect of PPS as a whole and test the theoretical predictions of their model.

Finally, researchers often advocate for continuous DiD designs because they can be used to estimate average causal effects of small changes in the dose. In many economic models, price and income elasticities determine optimal policies like tax rates, tax bases, subsidies, and regulations (Hendren, 2016), but these are continuous concepts that can be estimated accurately only with continuous variation. AF’s theoretical framework implies that, under some assumptions, DiD estimators provide information about hospitals’ elasticity of substitution between capital and labor, although AF do not argue for this kind of “marginal” interpretation.

Figure 1: Two-Way Fixed Effects Event-Study Estimates of the Effect of Medicare’s Reimbursement Reform on Hospital Input Mix

<!-- image -->

Two-Way Fixed Effects Event-Study Estimates of the Effect of Medicare’s Reimbursement Reform on Hospital Input Mix

*Notes:* The figure plots TWFE event-study coefficients and their 95% confidence intervals, from regressions with hospital fixed effects, year fixed effects, and the 1983 Medicare inpatient share ($m\_j$) interacted with either a dummy for years after 1983 (static TWFE specification) or the year dummies (event study). The outcome variable is the depreciation share of total operating expenses, a measure of hospitals’ capital/labor ratio. The data cover the years 1980-1986 and come from the American Hospital Association’s annual survey (American Hospital Association, 1986). The results are not necessarily identical to AF’s because we drop all hospitals that do not report capital/labor data in each of 1981-1985.

In terms of estimation, AF use the standard tool for continuous DiD designs: a TWFE regression with hospital and year fixed effects. They follow textbook advice. Wooldridge (2010, p. 132) observes that a two-period DiD regression estimator “can be easily modified to allow for continuous, or at least non-binary, ‘treatments.’” Angrist and Pischke (2009, p. 234) emphasize “a second advantage of regression DD is that it facilitates the study of policies other than those that can be described by a dummy.” They also follow common practice and describe their identifying assumption as an extension of the parallel trends assumption from binary designs: “Without the introduction of PPS, hospitals with different $m\_j$’s would not have experienced differential changes in their outcomes in the post-PPS period” (emphasis added).

Figure 1 reproduces AF’s DiD event-study coefficients for each calendar year, relative to 1983, and the estimate of $\psi^S$ from an equation like (1). AF interpret these results as indicative that, after 1983, capital/labor ratios rose more strongly for hospitals with higher values of $m\_j$, without a substantial differential change in input mix before PPS. Our impression is that event-study results

6

like those in Figure 1 would usually be interpreted as strong causal evidence because there are (relatively) small pre-trend estimates, large estimates in post-treatment periods, and tight confidence intervals. What is missing from most continuous DiD analyses, however, is a specific statement about what causal parameters researchers would like to estimate, the assumptions under which they are identified, and a formal justification for a particular estimator. Our goal is to shed light on these three issues.

## 3 Baseline Case: A New Treatment with Two Periods

We illustrate our main points in a setup with two periods of panel data, $t = 1$ and $t = 2$. In the first period, no unit is treated. In the second period, some units receive a treatment “dose,” denoted by $D\_i$, and others remain untreated. Extensions to multiple periods and staggered setups are discussed in Section 5. We denote the support of $D$ by $\mathcal{D}$, $D\_i$ can be (absolutely) continuous or can be multi-valued discrete, but to simplify the exposition, we refer to it as “continuous.” We define potential outcomes for unit $i$ in period $t$ as $Y\_{it}(d, \delta)$, where potential outcomes are indexed by the treatment sequence (Robins, 1986). As we focus on the setup where all units have $d = 0$ in period $t = 1$, we simplify the potential outcome notation and henceforth write $Y\_{it}(d)$, where $d$ is the dosage in period $t = 2$. This is the outcome that unit $i$ would experience in period $t$ under (period-two) dose $d$. In each time period $t$, the observed outcome for unit $i$ is $Y\_{it} = Y\_{it}(D\_i)$. We assume that all expectations are finite and well-defined. Henceforth, we omit the unit index $i$ to make the notation less cluttered and define $\Delta Y = Y\_2 - Y\_1$.

### 3.1 Parameters of Interest with a Continuous Treatment

The potential outcomes notation $Y\_t(d)$ reflects that treatment can take many values, and so each unit can experience many types of causal effects. The *level treatment effect* of dose $d$ in time period $t$ for a given unit is defined as its potential outcome when $D = d$ minus its untreated potential outcome $Y\_t(d) - Y\_t(0)$. Level treatment effects measure the treatment effect at time $t$ from switching treatment dosage from 0 to $d$. This is a straightforward extension of a binary “treatment effect” to a continuous “treatment effect function” or “dose-response function.”

But zero-treatment is not the only relevant counterfactual. We define a unit’s *causal response* at $d$ as $Y\_t'(d)$, the derivative of the potential outcome with respect to dose $d$ (when the treatment is continuous),[^4] or as the difference in potential outcomes between adjacent doses scaled by the difference in the doses, $(Y\_t(d\_j) - Y\_t(d\_{j-1}))/(d\_j - d\_{j-1})$ (when the treatment is discrete). Causal responses measure the treatment effect at time $t$ of a marginal increment of dose $d$. These two types of treatment effects—the level of $Y\_t(d) - Y\_t(0)$ or its slope, $Y\_t'(d)$—define unit-level causal parameters in continuous designs, and connect to results in the instrumental variables (IV) literature on multi-valued discrete or continuous endogenous variables (Angrist and Imbens, 1995, Angrist, Graddy, and Imbens, 2000).

[^4]: This is a slight abuse of notation as we do not require $Y\_t(d)$ to be differentiable (or even continuous), but rather we mean here the causal effect of a marginal increase in the dose on a unit’s outcome: $\lim\_{h \to 0} (Y\_t(d + h) - Y\_t(d))/h$.

7

We focus on "building block" parameters that are averages of these two kinds of causal effects in the post-treatment period, $t = 2$. Average level treatment effects (which we refer to as average treatment effects) extend definitions from the binary case:

$$ ATT(d|d) = E[Y\_{i2}(d) - Y\_{i2}(0)|D = d] \quad \text{and} \quad ATT(d) = E[Y\_{i2}(d) - Y\_{i2}(0)|D &gt; 0], $$

where $ATT(d|d)$ is the average effect of dose $d$ compared to zero dosage in the post treatment period $t = 2$ on units that actually experienced dose $d$. When $d' = d$, this is the $ATT$ local to units that received dose $d$. $ATT(d)$ is the average difference between potential outcomes under dose $d$ relative to untreated potential outcomes across all treated units, not just those that experienced dose $d$, in time period $t = 2$.

Average causal response parameters for absolutely continuous treatments are defined as

$$ ACRT(d|d) = \left.\frac{\partial E[Y\_{i2}(l)|D = d]}{\partial l}\right| *{l=d}* *= \left.\frac{\partial E[Y* {i2}(l) - Y\_{i2}(d)|D = d]}{\partial l}\right|\_{l=d} $$

and

$$ ACRT(d) = \left.\frac{\partial E[Y\_{i2}(l)|D &gt; 0]}{\partial l}\right| *{l=d}* *= \left.\frac{\partial E[Y* {i2}(l) - Y\_{i2}(d)|D &gt; 0]}{\partial l}\right|\_{l=d}. $$

$ACRT(d|d)$ is the average effect of a marginal dose increase from $d$ for dose group $d$. It equals the derivative of $ATT(l|d)$ with respect to $l$, evaluated at $l = d$, which is equivalent to the derivative of the $t = 2$ average potential outcome function with respect to dose $d$ among dose group $d$. $ACRT(d)$ is the average causal response of dose $d$ across all treated units. For discrete treatments, average causal responses are defined in a similar way but with slightly different notation:

$$ ACRT(d\_j|d\_j) = \frac{E[Y\_{i2}(d\_j) - Y\_{i2}(d\_{j-1})|D = d\_j]}{d\_j - d\_{j-1}} \quad \text{and} \quad ACRT(d\_j) = \frac{E[Y\_{i2}(d\_j) - Y\_{i2}(d\_{j-1})|D &gt; 0]}{d\_j - d\_{j-1}}, $$

where $d\_j|d\_j$ equals the difference in mean potential outcomes between dose level $d\_j$ and the next lowest dose $d\_{j-1}$ in period $t = 2$ for dose group $d\_j$, scaled by the difference between the two doses. Similarly, $ACRT(d\_j)$ gives the average causal response of dose $d\_j$ relative to dose $d\_{j-1}$, but it is for the entire treated group. We note that our definition of $ACRT(d\_j|d\_j)$ and $ACRT(d\_j)$ differs from the definitions in Angrist and Imbens (1995), as it scales the changes in expected potential outcomes by the change in dosage.

Figure 2 illustrates these parameters graphically. The concave line plots an average treatment effect function against the dose for dose group $d$, $ATT(l|d)$. If we consider dose levels $d$ and $d'$, there are two possible $ATT$ parameters. The first, $ATT(d|d)$, the level of dose group $d$'s average treatment effect function at $d$, is an average treatment effect that is "local" to units that experienced dose $d$. The second, $ATT(d'|d)$, is also "local" to dose group $d$, but refers to the effect they would experience at dose $d'$ even though they did not actually receive that dose. The continuous-dose $ACRT$ parameters are the slopes of tangent lines to the $ATT(l|d)$ function, and the discrete-dose $ACRT$ parameters are the slopes of lines connecting two points on the $ATT(l|d)$ function. As with $ATT$'s, our definitions encompass causal responses to doses other than the one a group actually receives (i.e., $ACRT(d'|d)$).

A proper interpretation of continuous DiD results hinges on which type of parameter one wants, can identify, and can estimate. For instance, even if all $ATT(d|d)$ parameters are large and positive, some $ACRT(d|d)$ parameters could be zero or negative. A researcher misinterpreting a large $ATT$ estimate as an $ACRT$, in this case, would mistakenly conclude that a policy to raise every unit's dose would have large effects. A researcher confusing a small $ACRT$ for an $ATT$ would mistakenly conclude that a policy was ineffective, even though it actually just has small effects at the margin.

8

Figure 2: Causal Parameters in a Continuous Difference-in-Differences Design

- ( ATT(d \mid d) )
- ( ATT(d' \mid d') )
- ( ACRT^{C}(d \mid d) ) (continuous)
- ( ACRT^{C}(d' \mid d') ) (continuous)
- ( ACRT^{D}(d' \mid d) ) (discrete)
- ( d )
- ( d' )

*Note:* The figure plots ( ATT(d \mid d) ) (the average effect of experiencing each dose for dose group ( d ) ). We highlight causal parameters for two doses, ( d ) and ( d' ) . ( ATT(d \mid d) ) and ( ATT(d' \mid d') ) are average treatment effects on the treated parameters and refer to the height of the curve. ( ACRT^{C}(d \mid d) ) and ( ACRT^{C}(d' \mid d') ) are average causal response parameters and refer to the slope of the curve. We show three dose-response functions, where ( ACRT^{C}(d \mid d) ) is the slope of a tangent line, and for a discrete dose share ( ACRT^{D} ) is the slope of a line connecting two discrete points on ( ATT(d \mid d) ) .

The above-mentioned causal parameters are functional parameters because they are allowed to vary arbitrarily across dose groups ( d ) and across (counterfactual) doses ( t ) . This contrasts with ( \beta^{fe} ) from (1.1), which is a single number. In many applications, it may be desirable to aggregate these functional parameters into lower-dimensional objects that are easier to report and may be more precisely estimated. We focus on aggregations that average the functional parameters discussed above using the distribution of the dose among all treated units,

[ ATT^{loc} := \mathbb{E}[ATT(D \mid D) \mid D &gt; 0] \quad \text{and} \quad ATT^{glb} := \mathbb{E}[ATT(D) \mid D &gt; 0] ]

[ ACRT^{loc} := \mathbb{E}[ACRT(D \mid D) \mid D &gt; 0] \quad \text{and} \quad ACRT^{glb} := \mathbb{E}[ACRT(D) \mid D &gt; 0]. ]

These provide natural ways to summarize the underlying parameters. We use the loc superscript to denote that ( ATT^{loc} ) and ( ACRT^{loc} ) summarize treatment effects that are local effects of particular doses, while we use the superscript glb to denote that ( ATT^{glb} ) and ( ACRT^{glb} ) summarize treatment effects of particular doses globally (i.e., across all treated units). All four of these parameters provide “best” approximations in the sense of minimizing the mean squared distance between the summary parameter and the functional parameters. Also, note that ( ACRT^{glb} ) and ( ACRT^{loc} ) are average derivative-type parameters, and average derivatives have been widely studied in econometrics.

## 3.2 Identification with a Continuous Treatment

This section discusses the identification of average treatment effect and average causal response parameters. Toward this end, we make the following assumptions.

**Assumption 1 (Random Sampling).** The observed data consist of ( { Y\_{i,t-1}, Y\_{it}, D\_i } \_{i=1}^{n} ) , which is independent and identically distributed.

**Assumption 2 (Treatment).** In period ( t = 1 ) , no unit is treated, while in period ( t = 2 ) , the treatment dosage ( D ) has support ( \mathcal{D} = { 0 } \cup \mathcal{D} *{+}* *)* *, where* *(* *\mathcal{D}* {+} \subset (0, \infty) ) . In addition, ( P(D = 0) &gt; 0 ) .

9

**Assumption 3 (No-Anticipation and Observed Outcomes).** *For all units, and all* *(* *d \in \mathcal{D}* *)* *,*

[ Y\_{i,t-1}=Y\_{i,t-1}(0)=Y\_{i,t-1}(d) \quad \text{and} \quad Y\_{i,t}=Y\_{i,t}(D\_i). ]

Assumption 1 says that we observe two periods of iid panel data. Assumption 2 formalizes that a mass of units do not participate in the treatment in either period (we discuss the case with no untreated units in more detail at the end of this section), and the rest receive a positive amount of the treatment that can vary in amount across units. Assumption 3 says that units do not anticipate future treatments, so we observe untreated potential outcomes for all units in the first period. In the second period, we observe the potential outcome corresponding to the actual dose that unit ( i ) experienced.

### 3.2.1 Identification under parallel trends

Identification of average level treatment effects follows closely from the DiD setup with binary treatments. In particular, our results rely on an extension of the binary parallel trends assumption.

**Assumption PT (Parallel Trends).** *For all* *(* *d \in \mathcal{D}\_+* *)* *,*

## 

[

\mathbb{E}[Y\_{i,t}(0)-Y\_{i,t-1}(0)\mid D=d]

\mathbb{E}[Y\_{i,t}(0)-Y\_{i,t-1}(0)\mid D=0]=0. ]

Assumption PT says that the average evolution of outcomes that units with any dose ( d ) would have experienced without treatment is the same as the evolution of outcomes that units in the untreated group actually experienced. Binary DiD designs also rely on assumptions like this. To simplify the exposition below, we often simply refer to Assumption PT as parallel trends (PT). The following result shows that under Assumption PT, ( ATT(d\mid d) ) is identified; all proofs are in Appendix A.

**Theorem 3.1** *Under Assumptions 1, 2, 3, and PT,* *(* *ATT(d\mid d)* *)* *is identified for all* *(* *d \in \mathcal{D}\_+* *)* *, and it is given by*

[ ATT(d\mid d)=\mathbb{E}[\Delta Y\mid D=d]-\mathbb{E}[\Delta Y\mid D=0]. ]

*Furthermore,*

*(*

*ATT^o=\mathbb{E}[\Delta Y\mid D&gt;0]-\mathbb{E}[\Delta Y\mid D=0]*

*)*

*.*

Theorem 3.1 states that ( ATT(d\mid d) ) equals the difference between the change in outcomes for dose group ( d ) and the untreated group. It generalizes Fricke (2017)’s result for two doses to richer treatment patterns. As a direct consequence, by averaging all the ( ATT(d\mid d) ) ’s over the distribution of non-zero dosages, we have that the summary parameter ( ATT^o ) is identified by simply comparing units with a positive dose to untreated units. On the other hand, parallel trends, as defined in Assumption PT, is not strong enough to guarantee the identification of ( ATT(d) ) .

The identification of average causal response parameters differs from the identification of ( ATT ) parameters because it requires comparisons between dose groups.

**Assumption 4 (Continuous or Multi-Valued Discrete Treatment).** *The treatment is either continuous or multi-valued discrete. More precisely, one of the following is true:*

(a) ( \mathcal{D} *+ = \mathcal{D}* +^c ) , where ( \mathcal{D} *+^c = (d\_L,d\_U)* *)* *with* *(* *f* {D\mid D&gt;0} ) a Lebesgue density which satisfies ( f\_{D\mid D&gt;0}(d)&gt;0 ) for all ( d \in \mathcal{D} *+^c* *)* *, and* *(* *\mathbb{E}[\Delta Y\mid D=d]* *)* *is differentiable on* *(* *\mathcal{D}* +^c ) .

10

(b) ( \mathcal{D}\_t = \mathcal{D} *t^{sub}* *)* *where* *(* *\mathcal{D}* *t^{sub} \subset \mathbb{N}* *+* *)* *, with* *(* *\mathbb{N}* + = { 1,2,3,\ldots } ) denotes the strictly positive natural numbers. Let ( d\_j ) denote the ( j ) th element of ( \mathcal{D}\_t^{sub} ) . In addition, ( P(D=d)&gt;0 ) for all ( d \in \mathcal{D}\_t^{sub} ) .

Assumption 4 distinguishes between cases with a continuous 4(a) or discrete 4(b) treatment. Assumption 4(a) allows for the smallest value of the treatment to be arbitrarily close to zero or strictly larger than zero, both of which are common in applications.

Our central identification result is that causal response parameters are not identified under Assumption PT, because comparisons between different dose groups are biased when treatment effects (of the same dose) vary across dose groups, even when the average evolution of untreated potential outcomes is the same.

**Theorem 3.2.** *Under Assumptions 1, 2, 3, and PT, comparisons of paths of outcomes among different dose groups recover a mix of causal effect parameters and selection bias terms. Specifically,*

(a) *For* *(* *(h,l) \in \mathcal{D}\_t \times \mathcal{D}\_t* *)* *,*

[ E[\Delta Y \mid D=d\_h] - E[\Delta Y \mid D=d\_l] = ATT(d\_h \mid d\_h) - ATT(d\_l \mid d\_l) ]

[ = E[Y\_t(d\_h)-Y\_t(d\_l)\mid D=d\_h]

- \underbrace{(ATT(d\_l \mid d\_h)-ATT(d\_l \mid d\_l))}\_{\text{selection bias}}. ]

(b) *If Assumption 4(a) also holds, then, for* *(* *d \in \mathcal{D}\_t^C* *)* *,*

# 

[

\frac{\partial E[\Delta Y \mid D=d]}{\partial d}

# \frac{\partial ATT(d \mid d)}{\partial d}

ACRT(d \mid d) + \underbrace{\left.\frac{\partial ATT(d \mid l)}{\partial l}\right| *{l=d}}* {\text{selection bias}}. ]

(c) *Alternatively, if Assumption 4(b) also holds,*

# 

[

\frac{E[\Delta Y \mid D=d\_j]-E[\Delta Y \mid D=d\_{j-1}]}{d\_j-d\_{j-1}}

ACRT(d\_j,d\_{j-1}\mid d\_j) + \underbrace{\frac{ATT(d\_{j-1}\mid d\_j)-ATT(d\_{j-1}\mid d\_{j-1})}{d\_j-d\_{j-1}}}\_{\text{selection bias}}. ]

Theorem 3.2 says that under parallel trends, comparisons of outcome paths between higher- and lower-dose groups mix together (i) causal responses and (ii) a “selection bias” type of term that comes from differences in average treatment effects of the same dose for different dose groups. Intuitively, even if untreated potential outcomes evolve in the same way, observed paths of outcomes differ between dose groups for two reasons. One is the causal response itself, which comes from differences in doses ( ( h ) versus ( l ) ) causing differences in outcomes. The other is a selection bias type of contamination, which comes from differences across dose groups in the average level effect of the particular dose ( l ) —parallel trends does not rule out that different dose groups could experience different treatment effects of the same dose.

Figure 3 illustrates this result for an example with two dose groups and two doses: ( d ) and ( d' = d+1 ) . The slope of the line that connects the points ( (d, ATT(d\mid d)) ) and ( (d', ATT(d'\mid d')) ) is steeper than the average causal response of interest, ( ACRT(d'\mid d) ) , because it jumps from one ATT function to the other. This is captured by the selection bias term, a version of selection-on-gains that equals the difference in treatment effects at the lower dose: ( ATT(d\mid d') - ATT(d\mid d) ) . It breaks the causal interpretation because observed outcomes for lower-dose units are not a valid counterfactual for what higher-dose units would have experienced at the lower dose. The selection bias is not identified

11

Figure 3: Non-Identification of Average Causal Response with Treatment Effect Heterogeneity, Two Discrete Doses

```
ATT(d'|d')                                      ATT(d')
    |
ATT(d|d')        ACRT(d,d'|d')
    |                              ACRT(d,d') + ATT(d|d') - ATT(d|d)
ATT(d|d)
    |                              ATT(d)
    |
    +------------------------------ d
                  d        d'
```

*Note:* The figure shows that comparing adjacent ( ATT(d) ) 's equals an ( ACRT ) parameter (the slope of the higher-dose group's ( ATT ) function) and selection bias (the difference between the two groups' ( ATT ) functions at the lower dose).

as we do not observe ( Y\_{i,t}(d') ) for units that experienced dose ( d ) . Such a result precludes a causal interpretation of ( ATT(d) ) differences across doses under Assumption PT.

### 3.2.2 Identification under strong parallel trends

This section discusses an alternative, typically stronger assumption that allows for the identification of ( ACRT(d) ) and ( ATT(d) ) parameters, which we refer to as *strong parallel trends* (SPT).

**Assumption SPT (Strong Parallel Trends).** For all ( d \in \mathcal{D} ) ,

# 

[

E[Y\_{i,t+1}(d)-Y\_{i,t}(0)\mid D\_i(0)&gt;0]

E[Y\_{i,t+1}(d)-Y\_{i,t}(0)\mid D\_i(0)=d]. ]

Under Assumption 3, the right-hand side of the equation in Assumption SPT is the (observed) average evolution of outcomes for dose group ( d ) . Assumption SPT says that the average evolution of outcomes for the entire treated population if all experienced dose ( d ) (the left-hand side of the previous equation) is equal to the path of outcomes that dose group ( d ) actually experienced. In applications where the treatment is binary, Assumption SPT, like Assumption PT, reduces to the usual parallel trends assumption. Like the case with a binary treatment, it allows for treated units to select into being treated. Among treated units, though, it rules out selection into a particular amount of the treatment. With more complicated treatments, Assumption SPT notably differs from Assumption PT because it involves potential outcomes under different doses, ( Y\_i(d) ) , rather than only untreated potential outcomes, ( Y\_i(0) ) . While Assumption SPT is not strictly stronger than Assumption PT (e.g., notice that it does not require parallel trends in untreated potential outcomes for all dose groups), we refer to it as *strong parallel trends* to indicate that in many applications it would be a stronger, perhaps much stronger, assumption.

An alternative way to think about Assumption SPT is as an assumption that restricts treatment effect heterogeneity.[^5] In particular, if one maintains Assumption PT, Assumption SPT is equivalent

[^5]: There are some instances of versions of strong parallel trends implicitly being discussed in empirical work.

12

to assuming that $ATT(d|d)=ATT(d)$ for all doses. This condition can also be viewed as a structural assumption in the sense that it effectively allows one to extrapolate treatment effects of dose $d$ among dose group $d$ to treatment effects of dose $d$ for the entire treated population.

In the remainder of this section, we show that Assumption SPT is useful for recovering “global” average causal effect parameters, which are straightforward to compare to each other, and, hence, sidestep the selection bias issues discussed above. Before doing that, it is worth mentioning that we are not proposing Assumption SPT as an assumption that empirical researchers should readily adopt; in fact, in many applications, Assumption SPT may be a strong or implausible assumption. Rather, our aim is to clarify that many natural target parameters in DiD applications with a continuous treatment require stronger assumptions than the parallel trends as defined in Assumption PT.

**Theorem 3.3.** *Under Assumptions 1, 2, 3, and SPT,*

*(a) For $d \in \mathcal{D}\_{+}$, it follows that*

$$ ATT(d)=\mathbb{E}[\Delta Y|D=d]-\mathbb{E}[\Delta Y|D=0]. $$

*(b) For $(d\_1,d\_2)\in \mathcal{D}*

*{+}\times\mathcal{D}*

*{+}$,*

$$ \mathbb{E}[Y\_t(d\_1)-Y\_t(d\_2)|D&gt;0]=ATT(d\_1)-ATT(d\_2)=\mathbb{E}[\Delta Y|D=d\_1]-\mathbb{E}[\Delta Y|D=d\_2] $$

*(c) When Assumption (e) holds (i.e., treatment is continuous), it follows that, for $d\in\mathcal{D}\_{+}$,*

$$ ACRT(d)=\frac{\partial \mathbb{E}[\Delta Y|D=d]}{\partial d}. $$

*(d) When Assumption (f) holds (i.e., treatment is discrete), it follows that*

$$ ACRT(d\_j)=\frac{\mathbb{E}[\Delta Y|D=d\_j]-\mathbb{E}[\Delta Y|D=d\_{j-1}]}{d\_j-d\_{j-1}}. $$

For part (a) of Theorem 3.3, recall that $ATT(d|d)$ and $ATT(d)$ differ when there is selection into dose group $d$ on the basis of treatment effects. Strong parallel trends rules out that kind of selection, which means that comparing average outcome changes of dose group $d$ to the untreated group identifies $ATT(d)$. Part (b) says that comparisons of the average change in outcomes over time for different dose groups have a causal interpretation under Assumption SPT. For parts (c) and (d), strong parallel trends ensures that each dose group $d$ serves as a valid counterfactual for the entire treated population under that specific dose $d$, and, hence, that causal response parameters are identified under Assumption SPT.

Strong parallel trends only changes the interpretation of the estimand, not its form. One important implication is that conventional pre-tests for differential changes across groups before treatment cannot distinguish between Assumption PT and Assumption SPT.[^7] That is, because only untreated

[^7]: Chodorow-Reich, Nenov, and Simsek (2021, p. 1365)’s cross-region study of marginal propensities to consume (MPC) notes the possibility of finding a zero cross-sectional coefficient when the MPCs are positive in all areas: “If low wealth areas have high MPCs and high wealth areas have low MPCs, an increase in the stock market could induce the same change in spending in both low and high wealth areas.” Similarly, Saez, Slemrod, and Giertz (2012, p. 5) discuss a more restrictive version of strong parallel trends in the context of estimating the elasticity of taxable income for two groups facing different positive tax changes: “If the control group faces a tax change, difference-in-differences estimates will be consistent only if the elasticities are the same for the two groups.”

13

potential outcomes are observed before treatment under Assumption 3, these periods cannot test the additional content of an assumption like SPT that necessarily involves treated potential outcomes.

Finally, the identification results in Theorem 3.3 immediately imply that averages of the $ATT(d)$ and $ACRT(d)$ building blocks are identified as well. The following corollary states these results.

**Corollary 3.1.** *Under Assumptions 1, 2, 3, and SPT,*

*(a) It follows that*

$$ ATT^{post} = E[\Delta Y \mid D &gt; 0] - E[\Delta Y \mid D = 0]. $$

*(b) When Assumption 4(a) holds (i.e., treatment is continuous), it follows that*

# 

$$

ACRT^{post}

# \frac{\partial E[\Delta Y \mid D = d]}{\partial d} \Big/ \frac{\partial d}{\partial d}, \ D &gt; 0

\int\_{\mathcal{D} *{+}}* *\frac{\partial E[\Delta Y \mid D = d]}{\partial d}* *f* {D \mid D&gt;0}(x) dx. $$

*(c) When Assumption 4(b) holds (i.e., treatment is multi-valued), it follows that*

# 

$$

ACRT^{post}

\sum\_{j=1}^{\bar{d}} \left( E[\Delta Y \mid D = d\_j] - E[\Delta Y \mid D = d\_{j-1}] \right) P(D = d\_j \mid D &gt; 0). $$

These results highlight how identification in continuous DiD designs is fundamentally a question about dose-specific building block parameters and the underlying parallel trends assumption, not the aggregation choices that lead to particular summary parameters.

**Remark 3.1 (No untreated units).** *Researchers often use continuous designs when all units in their sample receive some amount of the treatment, having in mind comparing units that are “more treated” to units that are “less treated.” Without untreated units, it is impossible to compare dose group $d$ to an untreated group, and, hence, it is infeasible to directly recover $ATT(d)$ or $ATT(d)$. However, a natural alternative is to compare dose group $d$ to dose group $d\_0$ (the lowest possible amount of the treatment). In Appendix SD.1 in the Supplementary Appendix, we show that, under parallel trends, when there are no untreated units,*

$$ E[\Delta Y \mid D = d] - E[\Delta Y \mid D = d\_0] = ATT(d) - ATT(d\_0). $$

*This shows that this comparison is related to underlying causal effect parameters under parallel trends; however, recall from Theorem 3.2 that the expression on the right-hand side mixes together the average causal response of moving from $d\_0$ to $d$ with selection bias. Under strong parallel trends, we have instead that*

# 

$$

E[\Delta Y \mid D = d] - E[\Delta Y \mid D = d\_0]

# ATT(d) - ATT(d\_0)

E[Y\_t(d) - Y\_t(d\_0) \mid D &gt; 0], $$

*which does not include selection bias terms. This discussion highlights that (unlike a setting with a binary treatment) continuous variation in the dose can be used to learn about causal effects even if there is no untreated comparison group, but interpreting these results as causal effects of the treatment requires strengthening Assumption PT. See also Fricke (2017) for a related discussion.*

### 3.3 What Parameter Does TWFE Estimate?

Empirical researchers using continuous DiD designs typically estimate a single summary parameter using a linear TWFE regression like Equation (1.1). This section links the TWFE estimand to

14

our identification results for dose-specific parameters, describes the assumptions necessary to give TWFE some causal interpretation, and discusses what that interpretation is. We focus on continuous treatments and defer the discussion of multi-valued discrete treatments to Appendix SD.3 in the Supplementary Appendix.

Our impression is that empirical researchers typically interpret ( \beta^{twfe} ) in three main (and related) ways, implicitly relying on different building blocks. First, ( \beta^{twfe} ) is often directly interpreted as a causal response parameter; that is, how much the outcome causally increases on average when the treatment increases by one unit. This is the causal version of how regression coefficients are often taught to be interpreted in introductory econometrics classes. Second, it is common to pick a representative value for ( d ) , to report ( d \times \beta^{twfe} ) , and interpret this quantity as ( ATT(d) ) . This is the main interpretation provided in Acemoglu and Finkelstein (2008): “Given that the average hospital has a 38 percent Medicare share prior to PPS, this estimate [i.e., of ( \beta^{twfe} ) , here equal to 1.129] suggests that in its first 3 years, the introduction of PPS was associated with an increase in the depreciation share of about 0.42 ( (= 1.129 \times 0.38) ) for the average hospital.” Rearranging this expression shows that under this interpretation ( \beta^{twfe} = ATT(d)/d ) , which relates ( \beta^{twfe} ) to a scaled level effect. Third, it is common to take two different representative values of the dose, ( d\_1 ) and ( d\_2 ) —a common choice is the 25th percentile and 75th percentiles of the dose—and interpret ( \beta^{twfe} ) as the average causal response of moving from dose ( d\_1 ) to dose ( d\_2 ) scaled by the distance between ( d\_1 ) and ( d\_2 ) ; this is a scaled ( 2 \times 2 ) effect. We aim to assess whether such types of interpretations are justified and under which conditions.

Table 1: TWFE Decomposition Weights

| Decomposition      | (D > 0)Weights                                                                                | (D = 0)Weights                                                          |
|--------------------|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Causal response    | (w^{cr}(g,t)=\dfrac{\ddot D_{gt}D_{gt}}{\sum_{(g,t):D_{gt}>0}\ddot D_{gt}D_{gt}})             | (w^{cr}0(g,t)=\dfrac{\ddot D{gt}}{\sum_{(g,t):D_{gt}=0}\ddot D_{gt}})   |
| Levels             | (w^\ell(g,t)=\dfrac{\ddot D_{gt}}{\sum_{(g,t):D_{gt}>0}\ddot D_{gt}})                         | (w^\ell_0(g,t)=\dfrac{\ddot D_{gt}}{\sum_{(g,t):D_{gt}=0}\ddot D_{gt}}) |
| Scaled levels      | (w^{sl}(g,t)=\dfrac{\ddot D_{gt}D_{gt}}{\sum_{(g,t):D_{gt}>0}\ddot D_{gt}D_{gt}})             | (w^{sl}0(g,t)=0)                                                        |
| Scaled(2 \times 2) | (w^{2\times2}(g,t,d)=\dfrac{\ddot D{gt}1{D_{gt}=d}}{\sum_{(g,t):D_{gt}>0}\ddot D_{gt}D_{gt}}) | (w^{2\times2}_0(g,t)=0)                                                 |

*Note:* The table provides the formulas for the weights used in the decompositions of ( \beta^{twfe} ) provided in this section.

The next proposition presents our decompositions of ( \beta^{twfe} ) under parallel trends (Assumption PT) and under strong parallel trends (Assumption SPT). The decompositions differ on the basis of the underlying building block parameters: causal response parameters ( ( ACR^T(d) ) and ( ACRT(d) ) ), level treatment effect parameters ( ( ATT(d) ) and ( ATT(d) ) ), scaled level effects ( ( ATT(d)/d ) and ( ATT(d)/d ) ), or scaled ( 2 \times 2 ) effects ( ( RTT(d\mid d') = (Y\_t(d)-Y\_t(d'))/(d-d') ) and ( RTT(d\mid 0) ) ). These building blocks are connected with the dose-parameters discussed in Section 3.2 and how empirical researchers interpret ( \beta^{twfe} ) ; see Appendix SD.2 in the Supplementary Appendix for additional decompositions based on different building blocks. The weights attached to each of these decompositions are presented in Table 1.

15

**Theorem 3.4.** *Under Assumptions 1, 2, 3, 4(a), and PT,* *(* *\psi^{fe}* *)* *can be decomposed in the following ways:*

**(a) Overall Response Decomposition**

# 

[

\psi^{fe}

\int\_{d\_L}^{d\_U} \omega^{att}(d) \left( ACRT(d) + \underbrace{ \frac{\partial ATT(\ell)\big| *{\ell=d}}{\partial \ell}* *}* {\text{selection bias}} \right) , dd + d\_L \times \omega^{att} \frac{ATT(d\_L|d\_L)}{d\_L}, ]

where the weights are always positive and integrate to 1.

**(b) Levels Decomposition:**

# 

[

\psi^{fe}

\int\_{d\_L}^{d\_U} \omega^{fe}(d) ATT(d|d) , dd. ]

where ( \omega^{fe}(d) \le 0 ) for ( d \le E[D] ) , and ( \int\_{d\_L}^{d\_U} \omega^{fe}(d) , dd + \omega^{fe} = 0 ) .

**(c) Scaled Levels Decomposition:**

# 

[

\psi^{fe}

\int\_{d\_L}^{d\_U} \omega^{fe,S}(d) \frac{ATT(d|d)}{d} , dd ]

where ( \omega^{fe,S}(d) \le 0 ) for ( d \le E[D] ) , and ( \int\_{d\_L}^{d\_U} \omega^{fe,S}(d) , dd = 1 ) .

**(d) Scaled**

**(**

**2 \times 2**

**)**

**Decomposition**

# 

[

\psi^{fe}

\int\_{d\_L}^{d\_U} \int\_{d\_L}^{d} \int\_{\ell}^{d\_U} \omega\_{2 \times 2}^{fe,S}(d,\ell,h) \left( \underbrace{ \frac{E[Y\_{it}(h)-Y\_{it}(\ell)\mid D\_i=d]}{h-\ell} } *{\text{causal response}}* *+* *\underbrace{* *\frac{ATT(\ell|d)-ATT(\ell|\ell)}{h-\ell}* *}* {\text{selection bias}} \right) , dh , d\ell ]

[ + \int\_{d\_L}^{d\_U} \omega\_{2 \times 2}^{fe,S}(d) \frac{ATT(d|d)}{d} , dd ]

where the weights ( \omega\_{2 \times 2}^{fe,S} ) and ( \omega\_{2 \times 2}^{fe,S} ) are always positive and integrate to 1.

If one imposes Assumption SPT instead of Assumption PT, then the selection bias terms from Part (a) and Part (d) become zero, and the remainder of the decompositions remain true, except one needs to replace ( ACRT(d) ) with ( ACRT^O(d) ) in Part (a), ( ATT(\ell) ) with ( ATT^O(\ell) ) in Parts (b), (c) and (d), and ( E[Y\_{it}(h)-Y\_{it}(\ell)\mid D\_i=d] ) with ( E[Y\_{it}(h)-Y\_{it}(\ell)\mid D\_i&gt;0] ) in Part (d).

Theorem 3.4 shows that the same TWFE estimand yields very different decomposition results, depending on the type of parallel trends used and the particular causal parameter employed as a building block for the analysis. Despite these multiple possible decompositions, one important feature that arises from Theorem 3.4 is that the weighting associated with any of the decompositions of ( \psi^{fe} ) has some undesirable properties, making ( \psi^{fe} ) an unappealing causal summary parameter in DiD setups with continuous treatments. Yet, each of these different decompositions highlights distinct concerns, as we discuss below.

Theorem 3.4(a) shows that when causal responses are taken as the building blocks of the analysis, under Assumption PT, ( \psi^{fe} ) is equal to a convex weighted average of ( ACRT(d) ) and the same selection bias derived in Theorem 3.2. ( ^6 ) The sign of this selection bias depends on how treatment

( ^6 ) Part (a) is mechanically related to the results in Vazquez (1999) on interpreting linear projection coefficients with a continuous regressor where the conditional expectation may be nonlinear. Part (a) also includes a term that shows how TWFE handles a discrete jump from 0 to the minimum treated dose, ( d\_L ) . Paths of outcomes are not observed for doses below ( d\_L ) , but the scaled ATT for dose group ( d\_L ) , ( ATT(d\_L|d\_L)/d\_L ) , is averaged into ( \psi^{fe} ) .

16

effects vary across dose groups at a given dose. If units in higher dose groups would have had larger positive treatment effects at every dose, for example, then ( d^{twfe} ) will be larger than the weighted average of the ( ACRT(d) ) ’s that appear in Theorem 3.4(a). Figure 3 illustrates this case for two groups. Strong parallel trends eliminates the selection bias term, but does not affect the weights.

Even under strong parallel trends, the particular interpretation of ( \beta^{twfe} ) in terms of ( ACRT(d) ) ’s hinges on the aggregation embodied in the weights ( w^{twfe}(d) ) . Because ( w^{twfe}(d) ) is positive and integrates to 1, under Assumption SPT ( \beta^{twfe} ) is usually causal (Blackwell, Bonney, Moyahed, and Torlovskyi, 2025). However, it does not estimate a natural target parameter like ( ACRT^{pop} ) because the TWFE weights do not generally equal the dose distribution among treated, ( f\_{D \mid D&gt;0}(d) ) . Interestingly, the weights ( w^{twfe}(d) ) underlying ( \beta^{twfe} ) depend on the entire distribution of the dose, making it sensitive to the size of the untreated group. This property is rather unappealing. For example, in our application, if we drop the untreated group (dropping the untreated group does not alter the underlying average causal responses), our estimate of ( \beta^{twfe} ) shrinks by 78%. Instead of letting the estimation method implicitly summarize the ( ACRT ) ’s, we recommend that researchers choose then aggregation scheme explicitly. In our view, a natural and econometrically-guided way to aggregate the ( ACRT ) ’s into a summary parameter is given by ( ACRT^{sum} ) , which is identified (as indicated in Corollary 3.1) and can also be easily estimated.

Part (b) expresses ( \beta^{twfe} ) as a weighted integral of ( ATT(d \mid d) ) under parallel trends, with weights that integrate to zero rather than one. Therefore, some weights are negative, and, hence, ( \beta^{twfe} ) is not weakly causal when ( ATT(d \mid d) ) is taken as the building block. More significantly, ( \beta^{twfe} ) puts the same amount of negative weight on ( ATT(d \mid d) ) ’s for doses below ( \mathbb{E}[D] ) as it does positive weight on ( ATT(d \mid d) ) ’s for doses above ( \mathbb{E}[D] ) . One way to view this result is that TWFE uses above-average dose units as an “effective treated group” and below-average dose units as an “effective comparison group” that potentially includes some treated units. While the cumulative positive weights and negative weights are equal to each other, they do not generally integrate to one within these groups, which means that ( \beta^{twfe} ) does not equal the difference between a weighted average of outcome paths for the effective treated group relative to the effective comparison group. In Appendix SD.2 in the Supplementary Appendix, however, we bridge this gap and derive a corollary of the result in Part (b) that makes the scaling issue related to this interpretation explicit and allows us to express ( \beta^{twfe} ) as the following weighted Wald-estimand:[^7]

# 

[

\beta^{twfe}

## 

\frac{

\mathbb{E}\left[\nu^{twfe}(D)\Delta Y \mid D &gt; \mathbb{E}[D]\right]

## 

\mathbb{E}\left[\nu^{twfe}(D)\Delta Y \mid D &lt; \mathbb{E}[D]\right]

}{

\mathbb{E}\left[\nu^{twfe}(D)D \mid D &gt; \mathbb{E}[D]\right]

\mathbb{E}\left[\nu^{twfe}(D)D \mid D &lt; \mathbb{E}[D]\right] } \tag{3.1} ]

The numerator of Equation (3.1) shows that ( \beta^{twfe} ) compares weighted average outcome changes above and below ( \mathbb{E}[D] ) with weights proportional to how far a unit’s dose is from ( \mathbb{E}[D] ) . The denominator scales this comparison by the same weighted difference in ( D ) . This representation highlights some challenges of using ( \beta^{twfe} ) to summarize the average level-effect of a continuous treatment. First, while

[^7]: The exact expressions for the weights are ( w^{twfe}(d) = \frac{(d-\mathbb{E}[D])(1-F\_D(d))}{\int\_0^\infty (\ell-\mathbb{E}[D])(1-F\_D(\ell))d\ell} ) and ( \nu^{twfe}(d) = \frac{|d-\mathbb{E}[D]|}{\mathbb{E}[|D-\mathbb{E}[D]| \mid D \gtrless \mathbb{E}[D]]} ) . See Appendix SD.2 in the Supplementary Appendix for more details.

17

the numerator is (roughly) a weighted level-effect, the denominator shows that ( \beta\_{fe} ) additionally depends on a measure of the average distance between the effective treated and comparison group. Second, the effective comparison group can include treated units. Third, ( \beta\_{fe} ) uses “distance” weights ( w\_{j\ell}^{fe} ) to aggregate across dosages. In contrast, ( ATT(d) ) does not suffer from any of these issues. In applications where the researcher is targeting level-effect parameters, we recommend favoring ( ATT(d) ) over TWFE regressions vis-a-vis ( \beta\_{fe} ) .

Parts (c) and (d) of Theorem 3.4 provide interpretations of ( \beta\_{fe} ) taking scaled paths of outcomes as building blocks. For part (c), ( ATT(d \mid d) ) (under parallel trends) and ( ATT(d' \mid d) ) (under strong parallel trends) are “per-dosage” causal parameters. This part shows that the TWFE estimand includes negative weights under the same conditions as in part (b), though the weights integrate to one. We note that, in the case of a discrete dose, this result in part (c) corresponds to the one in Theorem S3 of the Supplementary Appendix of de Chaisemartin and D’Haultfoeuille (2020). Therefore, using “average slopes” as the underlying parameter of interest eliminates neither TWFE’s potential for negative weights nor its non-intuitive weighting scheme. For part (d), when ( \beta\_{fe} ) is interpreted in terms of all possible ( 2 \times 2 ) comparisons of changes of outcomes for higher dose groups relative to lower dose groups, the weights are all positive and integrate to 1, but, under parallel trends, these comparisons all mix causal effects of the higher treatment with selection bias terms. Although strong parallel trends removes the selection bias, the weights attached to the causal parameters are still hard to interpret.

**Remark 3.2 (Decomposition with no untreated units).** *It is straightforward to extend the TWFE decompositions discussed above to settings with no untreated units. For the causal response decomposition (part (a)), the exact same result applies, with the exception that the second term involving* *(* *w\_{jt}^{cont}* *)* *is equal to 0. Similarly, for the scaled* *(* *2 \times 2* *)* *decomposition (part (d)), nothing changes except that the second term involving* *(* *w\_{jt}^{2 \times 2}* *)* *is equal to 0. For the levels decomposition and the scaled levels decomposition (parts (b) and (c)), with no untreated units,* *(* *ATT(d \mid 0)* *)* *(or* *(* *ATT(d)* *)* *) is not identified; instead, along the lines mentioned in Remark 3.1, instead of using the untreated comparison group, we can instead compare to the path of outcomes of the “least treated.” Thus, the same decompositions continue to apply except that* *(* *ATT^* (d) ) should be replaced by ( ATT(d) - ATT(d\_L) ) . This immediately means that these decomposition (in addition to negative weights) become complicated by issues related to selection bias.*

## 4 DiD estimators that can highlight or summarize heterogeneity

In this section, we discuss how one can bypass the limitations of the TWFE regression specification in Equation (1.1) by proposing data-driven estimation procedures that target well-defined causal parameters. For simplicity, in this section, we rely on Assumption SPT so we can get all causal parameters under the same identification assumptions. If one is interested in ( ATT(d \mid d) ) or their functionals, one can rely on Assumption PT and use the same estimation procedures for ( ATT(d) ) that we discuss below. In this case, though, we stress that one should not interpret derivatives of estimates of ( ATT(d) ) as estimates of ( ACRT(d \mid d) ) .

18

#### 4.1 Estimating average causal functions among the treated

We start with the estimation of the dose-specific functions, *ATT(d)* and *ACRT(d)* under Assumption SPT. First, recall that, from Theorem 3.3, we have that, for a positive dose *d* ,

[ ATT(d)=\mathbb{E}[\Delta Y\mid D=d]-\mathbb{E}[\Delta Y\mid D=0], ]

as well as

[ ACRT(d)=\partial \mathbb{E}[\Delta Y\mid D=d]/\partial d ]

when the treatment is continuous, and

[ ACRT(d\_j)=\frac{\mathbb{E}[\Delta Y\mid D=d\_j]-\mathbb{E}[\Delta Y\mid D=d\_{j-1}]}{d\_j-d\_{j-1}} ]

when the treatment is multi-valued discrete. As ( \mathbb{E}[\Delta Y\mid D=0] ) can be estimated using its sample analog,

[ \widehat{\mathbb{E}}[\Delta Y\mid D=0]=n\_{D=0}^{-1}\sum\_{i=1}^n \Delta Y\_i \mathbf{1} { D\_i=0 } , ]

the main challenge in estimating all these functions resides in estimating ( \mathbb{E}[\Delta Y\mid D=d] ) among treated units ( (d&gt;0) ) and its derivative.

Note that this is a standard regression setup, and, as such, researchers have different options for how to approach it. Examples include adopting a parametric model for ( \mathbb{E}[\Delta Y\mid D=d] ) (e.g., assuming a quadratic model in dose among the treated), or pursuing nonparametric estimators using kernels or sieves/series. We discuss these considerations below.

For simplicity, we start with setups where the treatment is multi-valued discrete, and takes on a relatively few values. In this case, one can estimate *ATT* ( (d\_j) ) and *ACRT* ( (d\_j) ) for any positive treatment dose ( d\_j ) in the dose support using a simple saturated regression[^5]

[ \Delta Y\_i=\beta\_0+\sum\_{j=1}^{J}\mathbf{1} { D\_i=d\_j } \beta\_j+\varepsilon\_i, \tag{4.1} ]

where we use the zero treatment dosage as the omitted category. It will then follow that ( \hat{\beta} *j* *)* *and* *(* *(\hat{\beta}* *j-\hat{\beta}* *{j-1})/(d\_j-d* {j-1}) ) are consistent estimators for the *ATT* ( (d\_j) ) and *ACRT* ( (d\_j) ) , respectively, and inference procedures are standard. Note that, in this setup, all that our regression (4.1) is doing is to automate the appropriate comparison of means justified under our identification assumptions.

When the dose (among treated units) is continuous, (4.1) becomes infeasible. One straightforward estimation approach is to impose a parametric functional form restriction on how ( \Delta Y ) varies with ( D ) among treated. For instance, one can consider a model in which ( \mathbb{E}[\Delta Y\_i\mid D=d] ) is quadratic in ( D ) among treated units, and run the following regression for observations with ( D\_i&gt;0 ) [^6]

[ \Delta Y\_i=\beta\_0+\beta\_1D\_i+\beta\_2D\_i^2+\varepsilon\_i. \tag{4.2} ]

When this regression specification is correctly specified, ( \widehat{ATT}^{par}(d)=\hat{\beta}\_0+\hat{\beta}\_1d+\hat{\beta}\_2d^2 ) and ( \widehat{ACRT}^{par}(d)=\hat{\beta}\_1+2\hat{\beta}\_2d ) are consistent estimators for *ATT(d)* and *ACRT(d)* . Pointwise and uniform-in- *d* inference procedures are standard. Of course, other parametric functional forms can also be adopted.

A limitation of parametric models, such as (4.2), is their reliance on potentially incorrect functional form restrictions. In fact, Theorem 3.4 exactly highlights the consequences of misspecification in the linear case. Provided that the sample size is large, however, researchers can use nonparametric procedures to avoid functional form restrictions. This entails considering a nonparametric regression

[^5]: One can also use the more flexible TWFE regression specification ( Y\_{it}=\sum\_{j=1}^{J}Post\_t\mathbf{1} { D\_i=d\_j } \beta\_j+\alpha\_i+\theta\_t+\varepsilon\_{it} ) , ( t=1,2 ) . We also note that we implicitly take ( d\_0=0 ) .

[^6]: One could also consider the regression ( \Delta Y\_i=\alpha+\mathbf{1} { D\_i&gt;0 } (\beta\_0+\beta\_1D\_i+\beta\_2D\_i^2)+\varepsilon\_i ) for all observations.

19

model of ( \Delta \widetilde{Y}\_i ) on ( D\_i ) among treated units,

[ \Delta \widetilde{Y}\_i = ATT(D\_i) + \epsilon\_i \tag{4.3} ]

One can estimate (4.3) in any number of ways. In our application, we have adopted the data-driven nonparametric estimators proposed by Chen, Christensen, and Kankanala (2025). An appealing feature of this procedure is that it resembles (4.2) in the sense that, upon computing the optimal sieve-dimension ( \widehat{K} ) , one runs a linear regression of ( \Delta \widetilde{Y}\_i ) on flexible ( \widehat{K} ) -dimensional transformations of ( D ) (cubic B-splines), ( q^{\widehat{K}}(D) ) , in the subsample of units with ( D\_i &gt; 0 ) ,

[ \Delta \widetilde{Y}\_i = (q^{\widehat{K}}(D\_i))' \beta + \epsilon\_i, \tag{4.4} ]

and then forming the nonparametric estimators for ( ATT(d) ) and ( ACRT(d) ) as

[ \widehat{ATT} *{cck}(d) = (q^{\widehat{K}}(d))' \widehat{\beta}* {cck}, \quad \widehat{ACRT} *{cck}(d) = (\partial q^{\widehat{K}}(d))' \widehat{\beta}* {cck}, \tag{4.5} ]

where ( \widehat{\beta} *{cck} = \left( \sum* {i:D\_i&gt;0} q^{\widehat{K}}(D\_i)q^{\widehat{K}}(D\_i)' \right)^{-1} \sum\_{i:D\_i&gt;0} q^{\widehat{K}}(D\_i)\Delta \widetilde{Y} *i* *)* *, and* *(* *\widehat{\beta}* {cck} ) is the ( \widehat{K} ) -dimension vector of OLS estimators for ( \beta ) .[^20] Chen, Christensen, and Kankanala (2025)’s results imply that the nonparametric estimators for ( ATT(d) ) and ( ACRT(d) ) curves converge at the fastest possible (i.e., minimax) rate in sup-norm, and lead to uniform confidence bands that are asymptotically narrower (more precise) than those based on undersmoothing, and contract at, or within a loglogn factor of, the minimax rate. See Appendix B for more details on how to compute ( \widehat{K} ) , as well as how to construct uniform confidence bands based on ( \widehat{ATT} *{cck}(d)* *)* *and* *(* *\widehat{ACRT}* {cck}(d) ) . Of course, one can adopt other nonparametric estimation and inference procedures and select tuning parameters using alternative criteria, although these may lead to different statistical guarantees.

## 4.2 Estimating summary measures of treatment effects

Researchers frequently want to report summary estimates to enhance interpretability and/or statistical precision, or because a lower-dimensional parameter is an input into some model or post-estimation calculation. As we showed in Section 3, however, a linear TWFE regression generally fails to deliver an interpretable summary parameter. In this section, we discuss estimation of ( ATT^s ) and ( ACRT^s ) , which are summary causal effect parameters that have a clear interpretation.

When there are untreated units, part (a) of Corollary 3.1 suggests an extremely simple and familiar estimator of ( ATT^s ) ; the difference between the average change in outcomes among treated units minus the average outcome change for untreated units. This “linearized” DID estimator can be obtained from the following simple linear regression specification:

[ \Delta Y\_i = \phi^{lin} + D\_i^0 \beta^{lin} + \epsilon\_i, \tag{4.6} ]

where ( D\_i^0 = 1 { D\_i &gt; 0 } ) . It is straightforward to show that under the identification assumptions in Corollary 3.1, ( \beta^{lin} = ATT^s ) . Note that this estimator applies equally to continuous and multivalued discrete treatments.

[^20]: As these nonparametric procedures have slower-than- ( \sqrt{n} ) rates of convergence, there is no estimation effect from estimating ( E[\Delta Y\_i \mid D\_i = 0] ) .

20

Aggregated average causal response parameters can be constructed easily by weighting the estimated average causal functions across doses using the dose distribution itself. For discrete treatments, it is straightforward to aggregate these ( ACRT(d) ) 's based on the coefficients from (4.1) to form a plug-in estimator for the ( ACRT^{Plug} ) , using the identification formula in Corollary 3.1(e), i.e.,

# 

[

\widehat{ACRT}^{Plug}

\sum\_{d=1}^{\bar D} \hat{\beta}\_{jd} \frac{\hat{p} *d-\hat{p}* {d-1}}{\hat{p}\_d} \hat{P}(D=d\mid D&gt;0), \tag{4.7} ]

where ( \hat{P}(D=d\mid D&gt;0)=\sum\_{i=1}^{n} 1(D\_i=d)/\sum\_{i=1}^{n} 1(D\_i&gt;0) ) . Inference procedures follow from the Delta Method. One can follow a similar strategy when using the scaled ( ACRT(d) ) as the “building block” of the aggregation. A similar approach applies to estimating ( ACRT^{Plug} ) with a continuous dose. Our proposed estimator is simple to compute as it is based on the plug-in principle, i.e.,

# 

[

\widehat{ACRT}^{Plug}

# \mathbb{E}\_n[\widehat{ACRT}(D\_i)\mid D\_i&gt;0]

\frac{1}{n\_{D\_i&gt;0}}\sum\_{D\_i&gt;0}\widehat{ACRT}(D\_i), ]

with ( n\_{D\_i&gt;0}=\sum\_i 1(D\_i&gt;0) ) denoting the sample size with a positive dose, and ( \widehat{ACRT}(D) ) being a parametric or nonparametric estimator. Under some regularity conditions, one can show that ( \widehat{ACRT}^{Plug} ) is ( \sqrt{n\_{D&gt;0}} ) consistent and asymptotically normal; see, e.g., Section 4.1 of Ai and Chen (2007).

We close this section by noticing that it is also possible to consider alternative estimators for ( ACRT^{Plug} ) using a so-called Neyman-Orthogonal moment representation. More precisely, by exploring the efficient influence function for ( ACRT^{Plug} ) implied by Theorem 3.1 of Newey and Stoker (1993), it is straightforward to show that

# 

[

ACRT^{Plug}

## 

\mathbb{E}

\left[

ACRT(D)

(\Delta Y-\mathbb{E}[\Delta Y\mid D,D&gt;0]) \frac{f' *{D\mid D&gt;0}(D)}{f* {D\mid D&gt;0}(D)} \mid D&gt;0 \right]. \tag{4.8} ]

Based on this representation, one can then use flexible nonparametric or machine-learning-based estimators for the nuisance functions and still conduct asymptotically valid inference procedures. This opens the door for leveraging double machine learning procedures to estimate ( ACRT^{Plug} ) in DiD contexts. We leave this topic for future research.

# 5 Extensions

In this section, we briefly summarize several extensions of our main results that are further discussed in the Appendix and Supplementary Appendix.

## 5.1 Relaxing Strong Parallel Trends

Under traditional DiD assumptions, Assumption PT led to the identification of local ( ATT(d\mid d) ) parameters that are difficult to compare across dosages. On the other hand, the strong parallel trends assumption led to ( ATT(d) ) parameters. These can be seen as extreme cases, and it is possible to trade off the strength of assumptions with the type of parameters that can be identified in different ways. The number of these intermediate possibilities is large, however. Here, we sketch what we consider

21

to be three main ideas to relax strong parallel trends. Appendix SE of the Supplementary Appendix provides substantially more detail.

First, in many cases, researchers may be willing to assume that they know the direction of the selection bias. For example, suppose that a researcher is willing to assume that, for all ( d ) and any dose groups ( \ell &lt; h ) , ( ATT^{(d)}(\ell) \leq ATT^{(d)}(h) ) , i.e., that higher dose groups would experience larger treatment effects at any value of the dose. In the Supplementary Appendix, we show that this type of assumption leads to bounds on causal effect parameters without requiring strong parallel trends. For example, it implies that, for all ( d )

[ ACRT(d \mid d) \leq \frac{E[\Delta Y \mid D=d]}{d}, ]

which provides a bound on ( ACRT(d \mid d) ) . See Proposition S7 in the Supplementary Appendix for more details.

A second possibility for relaxing strong parallel trends is to define a sub-region ( \mathcal{D}\_s \subseteq \mathcal{D} ) , for which strong parallel trends holds, i.e., to assume that

[ E[Y\_{t=2}(d)-Y\_{t=1}(0)\mid D=d] = E[Y\_{t=2}(d)-Y\_{t=1}(0)\mid D=d'] \tag{5.1} ]

holds for all ( d,d' \in \mathcal{D}\_s ) . Under this assumption, we show in Proposition S8 in the Supplementary Appendix that, for ( d,\ell \in \mathcal{D}\_s ) ,

[ E[\Delta Y \mid D=d] - E[\Delta Y \mid D=\ell] = E[Y\_{t=2}(d)-Y\_{t=2}(\ell)\mid D \in \mathcal{D}\_s]. ]

In other words, comparing the trends in outcomes over time for dose group ( d ) to dose group ( \ell ) delivers the average causal effect of dose ( d ) relative to dose ( \ell ) among those dose groups in ( \mathcal{D}\_s ) . Under PT, the same comparison would include selection bias terms.

While the assumption in Equation (5.1) is weaker than SPT, the tradeoff is that now only comparisons within the set ( \mathcal{D}\_s ) have a causal interpretation. In some applications, this assumption could be notably weaker than Assumption SPT—in fact, this assumption should, at least arguably, no longer be called “strong parallel trends” because it is non-trivially non-nested with Assumption PT. For example, suppose that ( \mathcal{D}\_s ) contains large doses. The assumption in (5.1) says that we can learn about the trend in outcomes for a higher-dose group at a counterfactual lower dose by looking at the trend in outcomes for that lower-dose group, but only for doses in ( \mathcal{D}\_s ) . This could be much more plausible than Assumption PT, which assumes that even very high dose groups would have experienced the same trend in untreated potential outcomes as the untreated group, even though these units might be very different from each other. This local version of the SPT assumption is appealing in applications where there is substantial variation in the dose and the researcher is willing to assume that there is no selection bias among units that select similar doses, but the researcher is unwilling to assume that there is no selection bias among units that select substantially different doses.

Finally, in some applications, strong parallel trends may be more plausible after conditioning on some observed covariates ( X ) . Under a version of strong parallel trends conditional on covariates, one can show that the conditional average treatment effect, ( ATT\_x(d)=E[Y\_{t=2}(d)-Y\_{t=2}(0)\mid X=x,D&gt;0] ) , is identified. Since this parameter is not local to dose group ( d ) , conditional on ( X=x ) , one can compare ( ATT\_x(d) ) across different values of the dose without inducing selection bias terms. This

22

is an intermediate case, however, in that these are more local parameters than $ATT(g)$ because they are local to the particular value of the covariates $x$. See the discussion in Appendix SE in the Supplementary Appendix for more details.

### 5.2 Multiple time periods and variation in treatment timing

Although our results so far focus on two-period cases, we can extend them to setups with multiple time periods and variation in treatment timing across units by combining the ideas discussed in Section 3.2 with those in Callaway and Sant’Anna (2021). We consider this setting in detail in Appendix C.

In a setting with staggered treatment adoption (i.e., where once a unit becomes treated with dose $d$, that unit remains treated with dose $d$ in subsequent periods), knowing the time period that a unit becomes treated with a positive dose (which we denote by $G\_i$ and refer to as a unit’s *timing group* ) and dose $D\_i$ (i.e., *dose group* ) fully characterizes a unit’s sequence of treatment across all periods. In this context, we need to augment our potential outcomes terminology and write $Y\_{it}(g,d)$ as the potential outcome of unit $i$ at time $t$ if it were first treated in period $g$ with dose $d$; we write $Y\_{it}(0,0)$ to denote a unit’s untreated potential outcome—the potential outcome in time period $t$ if that unit did not participate in the treatment in any available period. With this notation at hand, we can define a multi-period analog of $ATT(d|d)$ as

$$ ATT(g,t,d|d) = E[Y\_t(g,d) - Y\_t(0,0) | G=g, D=d] $$

and

$$ ACRT(g,t,d,d'|d) = \left. \frac{\partial ATT(g,t,l|d)}{\partial l} \right|\_{l=d'} $$

which are the average treatment effect and average causal response in period $t$ of (i) becoming treated in period $g$ and (ii) experiencing dose $d$ among those in timing group $g$ and dose group $d$.

Under no anticipation and a multiple-period version of the parallel trends assumption, we show in Appendix C that, in post-treatment periods (i.e., periods where $t \geq g$),

## 

$$

ATT(g,t,d|d) =

E[Y\_t - Y\_{g-1} | G=g, D=d]

E[Y\_t - Y\_{g-1} | G=\infty, D=0]. \tag{5.2} $$

The argument is similar to the two-period case discussed earlier. The main difference is that the expression above invokes the “long difference” in changes in outcomes over time, i.e., from period $g-1$ to $t$. The reason for this difference is that $g-1$ is the most recent period for which units in group $g$ were untreated. The expression above uses the never-treated group ($G=\infty$) as the comparison group, but, like the case with a binary treatment, one can use alternative comparison groups such as the not-yet-treated. Under a multiple-period version of the strong parallel trends assumption, one can take the derivative of the right-hand side of Equation (5.2) with respect to $d$ to identify $ACRT(g,t,d,d|d)$.

One complication that arises in the staggered case is that $ATT(g,t,d|d)$ and $ACRT(g,t,d,d|d)$ are often relatively high-dimensional objects that can be hard to report (and perhaps hard to estimate precisely). In Appendix C, we discuss two main strategies for aggregating these parameters into lower-dimensional objects. First, we average across timing groups and time periods to target causal effect parameters that are a function of only the dose: $ATT^{dos}(d|d)$, and $ACRT^{dos}(d|d)$—these parameters highlight heterogeneous effects across different doses and are analogous to $ATT(d|d)$ and $ACRT(d|d)$ in the two-period case that we have emphasized above. They can be averaged across the

23

dose to deliver scalar summary parameters. Second, we consider event-study parameters: ( ATT^{es}\_j(e) ) , and ( ACR^{es}\_j(e) ) that average across the dose and highlight how treatment effects and/or causal responses vary with the length of exposure to the treatment—these parameters are the event study analogs of ( ATT^s\_j ) and ( ACR^s\_j ) in the two period case above. See Callaway, Goodman-Bacon, and Sant’Anna (2024) for alternative, intermediate aggregations. The discussion here focuses on causal effect parameters that are local to a specific dose group and timing group, but, like the two-period case discussed above, it is also possible to recover causal effect parameters across all treated units under strong parallel trends; see Appendix SB in the Supplementary Appendix for more details.

## 5.3 Interpreting TWFE Regressions with Multiple Periods/Groups

In Appendix SB.2 of the Supplementary Appendix, we also extend our TWFE decomposition results from Theorem 3.4 to cover setups beyond the two-period case, including setups with staggered treatment adoption with continuous or multi-valued discrete treatments. These results generalize the decompositions in de Chaisemartin and D’Haultfœuille (2020) and Goodman-Bacon (2021) to the case of a continuous treatment. Those results demonstrate that TWFE regressions with multiple periods and variation in treatment timing (i) continue to suffer from the weighting and selection bias issues that we highlighted in Theorem 3.4, (ii) inherit weighting issues (including possible negative weights) that are prevalent in TWFE regressions with binary, staggered treatment adoption, and (iii) are affected by violations of parallel trends in pre-treatment periods.

## 5.4 Event-Study and Pre-Treatment Differences

When there are multiple periods of data available, DiD applications typically assess the plausibility of the parallel trends assumption by checking whether or not parallel trends holds in pre-treatment periods. In a setting with a continuous treatment, one can check whether ( E[\Delta Y\_{jt}\mid D=d]-E[\Delta Y\_{jt}\mid D=0] ) is approximately equal to zero for all pre-treatment time periods ( t&lt;g ) and all ( d ) , with ( g ) being the time of treatment adoption (where we simplify and consider a single treatment date setup). Implementing these tests, however, can be complicated because it involves multiple dose-response nonparametric estimates. A convenient alternative is to report aggregated event study parameters such as ( ATT^{es}\_j(e) ) or ( ACR^{es}\_j(e) ) in pre-treatment periods (i.e., ( e&lt;0 ) ). Plotting estimates of ( ATT^{es}\_j(e) ) and ( ACR^{es}\_j(e) ) for pre-treatment periods ( ( e&lt;0 ) ) can be used to assess the plausibility of parallel trends. We repeat these for our empirical application in Figures 8 and 10. Having said that, we note that one possible drawback of this test is that it may overlook violations of the parallel trends assumption that these event-study versions of the test would not detect.

An interesting (though subtle) caveat is that in cases where an aggregate level effect such as ( ATT^{es}\_j ) or its event study version ( ATT^{es}\_j(e) ) is the target parameter of the analysis, it is possible to recover it under “weaker” parallel trends assumptions that allow for violations of parallel trends where the average violation of parallel trends across dose groups is equal to zero (rather than the violation of parallel trends being equal to zero for all dose groups)—we refer to the corresponding

24

averaged version of parallel trends as aggregate parallel trends. If one maintains aggregate parallel trends, then only $ATT^{c}\_{fe}(g)$ (and not, e.g., $ACT^{c}(g)$) is relevant for assessing its plausibility using pre-treatment periods. That being said, it is debatable whether or not the violations of parallel trends that can be allowed for under aggregate parallel trends should be counted as evidence against the design.

## 6 Continuous DiD in Practice: Causal Effects of Medicare PPS

We have so far shown that the causal question of interest shapes identification in a continuous DiD design and argued that it should guide the estimation approach, too. We now apply our preferred average level treatment effect and average causal response estimators to Acemoglu and Finkelstein (2008)’s study of Medicare PPS. To map their setting to our theoretical analysis, we consider the balanced panel data component of Acemoglu and Finkelstein (2008), which comprises 585 hospitals, and also average all pre-treatment outcomes and post-treatment outcomes over time. Thus, we use $t = 1$ to denote the average of pre-treatment periods (1980-1983), and $t = 2$ to denote the average of post-treatment periods (1984-1986). We also denote treatment dose here by $M$ instead of $D$, as $M$ is a short-hand notation for the 1983 Medicare inpatient share that determines treatment exposure in the AF application.

To begin, consider the profit maximization problem for a hospital with Medicare inpatient share $M$. We follow AF and assume a production function, $F(L, K)$, that is homothetic in labor $(L)$ and capital $(K)$. Market wages and rental rates are normalized by the output price, and Medicare subsidies mean that net input prices are $(1 - k\_L M)w$ and $(1 - k\_K M)r$. Firms consider the following profit maximization problem:

$$ \max\_{L,K} F(L, K) - (1 - k\_L M)wL - (1 - k\_K M)rK. $$

The solution to this problem generates factor demands and a capital-labor ratio that is only a function of the input price ratio $\frac{(1-k\_LM)w}{(1-k\_KM)r}$. We write the subsidy ratio $\frac{(1-k\_LM)}{(1-k\_KM)}$ as $1 + S\_u(M) = 1 + \frac{(k\_K-k\_L)M}{1-k\_KM}$. This reflects the fact that hospitals with no Medicare patients $(M = 0)$, and all hospitals before PPS (when $k\_{K,t=1}=k\_{L,t=1}=0$) face no relative price distortion. PPS set $k\_{L,t}=0$ in 1983, making $S\_{u,t}(M)=\frac{k\_{K,t}M}{1-k\_{K,t}M}$.

This structure allows us to define the capital-labor ratio potential outcomes in terms of Medicare inpatient share $M$:

$$ Y\_{t=1}=Y\_{t=1}(0)=\mathcal{K} *{t=1}\left(\frac{w}{r}\right)* *\quad \text{and} \quad* *Y* {t=2}=Y\_{t=2}(M)=\mathcal{K} *{t=2}\left((1+S* {u,t=2}(M))\frac{w}{r}\right). $$

Three details of the theoretical setup are worth noting. First, homotheticity allows us to connect potential outcomes as a function of $M$ to a firm’s optimal capital-labor ratio as a function of relative prices (as a function of $M$). Without this assumption, a hospital’s scale affects its input mix, and capital-labor ratios are a function of net labor and capital prices separately, complicating the theoretical interpretation of causal parameters. Second, we define our parameters of interest in terms of causal effects of $M$ on $Y$. A structural interpretation of those parameters in terms of $\mathcal{K}$ necessar-

25

ity involves the non-linear way in which $M$ changes the subsidy ratio, $S(M)$ (as well as a kind of exclusion restriction that rules out direct effects of $M$ on outcomes). Third, we use time subscripts to match the fact that PPS changed over time, but this is not a dynamic model. The assumed lack of forward-looking behavior implies the no anticipation assumption (Assumption 3) and allows us to write $Y\_m = Y\_m(0)$. All these details are in line with AP’s theoretical model.

### 6.1 Causal Questions Around Medicare PPS

AP are primarily interested in the question: did PPS raise capital-labor ratios? PPS sought to help hospitals invest in new medical technologies with the aim of improving patient outcomes (Office of Technology Assessment, 1984). But regulators also worried about the “incentive for hospitals to adopt expensive capital equipment that reduces operating costs but raises total costs per case” (Office of Technology Assessment, 1984, p. 14). Thus, Medicare’s role in technology investments has important policy implications. Moreover, the theoretical model predicts that PPS would raise capital-labor ratios for all treated hospitals, so the sign of its effects is a test of a simple neoclassical production theory. The building block parameters that answer these questions are the average treatment effect of PPS on hospitals with $M = m$:

$$ ATT(m) = E[Y\_m(m) - Y\_m(0) \mid M = m] = E\left[\frac{K\_m}{L\_m}(1 + S\_m(z\_m))^{\frac{\sigma}{1-\sigma}} - \frac{K\_m(0)}{L\_m(0)} \mid M = m\right]. $$

Estimating and plotting the entire $ATT(m)$ function shows which hospitals responded most to PPS and tests the prediction that all treated hospitals increase their capital intensity. Under parallel trends alone, one cannot compare across $ATT(m)$’s, as it is not possible to discern whether variation from $ATT(m)$ comes directly from subsidy differences or from treatment effect heterogeneity. Averaging this function across treated hospitals yields $ATT^{TOT} = E[ATT(M) \mid M &gt; 0]$, a summary parameter that directly answers the question “did PPS raise capital-labor ratios on average?”

One may also be interested in which subsidy levels have larger causal effects. For example, if technologies are “lumpy”, then hospitals may not respond to subsidies too small to cover the minimum investment costs. Improving the design of input subsidies thus requires causal estimates of the responsiveness to different subsidy levels. The causal effects of marginal changes in the subsidy ratio also represent another test of the theoretical model because they are proportional to a hospital’s elasticity of substitution, $\sigma\_m(m) = \frac{\partial Y\_m(m)}{\partial m} \times \frac{(1 + S\_m(m))}{Y\_m}$, which, with two inputs, must be positive. The building block parameters that answer these questions are the average causal responses of PPS:

$$ \begin{aligned} ACR(m) &amp;= E[Y'\_m(m) \mid M = m] \ &amp;= E\left[\frac{K\_m(z\_m)}{L\_m(z\_m)} (1 + S\_m(z\_m))^{\frac{\sigma}{1-\sigma}} \frac{S'\_m(z\_m)}{1 - \sigma\_m(z\_m)} \mid M = m\right] \ &amp;= E\left[\sigma\_m(z\_m) K\_m(z\_m)\left((1 + S\_m(z\_m))^{\frac{1}{1-\sigma\_m}}\right) \frac{s'\_m}{1 + m} \mid M = m\right] &gt; 0 \end{aligned} \tag{6.1} $$

Again, reporting estimates of the entire $ACR(m)$ function highlights heterogeneity in how hospitals respond to subsidies, and the summary parameter $ACR^{TOT}$ provides a single measure of how much hospitals respond on average to small subsidy differences.

Before turning to our formal estimates, Figure 4 presents a binned scatter plot of the change in mean capital-labor ratios before (1980-1983) and after (1984-1986) PPS against the Medicare share

26

of inpatient days in 1983, ( m ) . Following AF, we measure the capital-labor ratio using the depreciation share of total costs.

Figure 4: Changes in Capital-Labor Ratios before and after 1983 versus the Medicare Inpatient Share

<!-- image -->

Figure 4: Changes in Capital-Labor Ratios before and after 1983 versus the Medicare Inpatient Share

*Notes:* The figure presents a binned scatter plot of the change in the average depreciation share (capital-labor ratio) between the periods 1984-1985 and 1980-1982 for hospitals in 2 percentage-point bins of the 1983 Medicare inpatient share, ( M ) . In the lowest bin, hospitals with ( M = 0 ) are plotted separately from hospitals with ( M \in (0, 0.02) ) . We also consider a single bin for all hospitals with ( M &gt; 0.84 ) .

The horizontal line equals the mean change in capital-labor ratio for untreated hospitals (0.37). Each circle is the mean outcome change for a given bin of the Medicare inpatient share, with its size proportional to the number of hospitals in that bin. Almost all groups of treated hospitals had stronger growth in capital intensity than untreated hospitals, consistent with the theoretical prediction. The relationship is nonlinear, however, which indicates heterogeneity in average treatment effects, at least, and perhaps heterogeneity in the sign of average causal responses.

## 6.2 Average Treatment Effects of PPS

Figure 5 presents our proposed data-adaptive nonparametric estimates of ( ATT(m|m) ) based on (4.5). For inference, we cluster at the hospital level. Our data-driven procedure to optimally choose the sieve dimension selected ( K = 4 ) . These estimates formalize what the scatter plot suggests: that ( ATT(m|m) ) is positive. We plot pointwise 95% confidence intervals in the dark-shaded region and the wider (honest) 95% uniform confidence bands in the light-shaded region. We do not detect an effect for values of ( m ) below 5 percent, but we reject zero for doses between 0.05 and 0.78, which contains 96 percent of treated hospitals. Significant values of ( ATT(m|m) ) range from about 0.44 percentage points at ( m = 0.1 ) to 0.88 percentage points at ( m = 0.41 ) . The average across all doses, ( ATT^{O} ) , is 0.80 (s.e. = 0.05), or about 18 percent of the 1983 mean outcome (measured by the depreciation share) of 4.5. This evidence suggests that PPS substantially raised capital-labor ratios.

For comparison, we report in Figure 6 parametric estimates for ( ATT(m|m) ) that use the quadratic regression specification in Equation 4.2. Different from Figure 5, the interpretability of ( \widehat{ATT}^{Q}(m|m) ) in Figure 6 depends on the quadratic specification being correctly specified. When we know that to be the case, it is clear from Figure 6 that this results in substantially more precise estimates, as these now fully leverage the functional form. Importantly, these gains in precision are more substantial

27

Figure 5: Nonparametric Estimates of ( ATT(m|x) ) for Medicare PPS

*Notes:* The figure plots nonparametric estimate of ( ATT(m|x) ) that adapts the Chen, Christensen, and Kankanala (2025) data-driven estimator to our context, as discussed in Section 4.1 and Appendix E. The dark-shaded region is the 95-percent point-wise confidence interval, and the lighter-shaded region is the 95-percent lowest and sup-norm rate-adaptive uniform confidence band. We display the histogram of the treatment dose among the treated in yellow.

in the regions where data for particular treatment doses are scarce, e.g., for treatment doses above 0.75. Overall, we have 4987 observations with a positive treatment dose. Among these, only 57 have a treatment dose above 0.75, 20 above 0.80, and 3 above 0.90. The rationale for this is very simple: parametric models are good at extrapolating, whereas nonparametric procedures are more cautious about it. The reliability of the extrapolation, once again, crucially depends on the parametric model for ( ATT(m|x) ) being correctly specified.

Figure 6: Parametric Estimates of ( ATT(m|x) ) for Medicare PPS using quadratic specification

*Notes:* The figure plots parametric estimate of ( ATT(m|x) ) that use the quadratic regression specification in Equation 4.2. The dark-shaded region is the 95-percent point-wise confidence interval, and the lighter-shaded region is the 95-percent uniform-in-treatment-dose confidence band. We display the histogram of the treatment dose among the treated in yellow. We use the same y-scale as in Figure 5.

Although gains in precision are desirable, we caution against using nonparametric results to pick a parametric specification. This, to some extent, resembles a pre-testing problem, and inference based on the parametric model could be misleading. In fact, the appeal of the uniform confidence bands from Chen, Christensen, and Kankanala (2025) that we report in light-shaded blue in Figure 5

28

is that they account for this type of pre-testing issue and are honest, i.e., they are guaranteed to have asymptotically correct coverage over a large (and generic) class of data-generating processes. The uniform confidence bands in Figure 6 are uniform only in treatment dosage, highlighting that it reflects a narrower type of uncertainty than those in Figure 5. Henceforth, as we find it challenging to fixate on motivating a parametric functional form for ( ATT(m \mid w) ) using arguments grounded in economic theory, we focus our attention on our nonparametric estimators.

In Section 3.3, we argued that ( \mu^{sel} *{TWFE}* *)* *should not be relied upon to summarize level effects. However, the TWFE coefficient is 1.14—roughly similar to our estimate of* *(* *ATT^{O}* *)* *. What accounts for their similarity? One explanation comes from Equation (3.1). The numerator compares weighted averages of the paths of outcomes for the “effective” treated group (those with above-average doses) to the “effective” comparison group (those with below-average doses). However, in our example, slightly more than half of the weight on paths of outcomes in the effective comparison group falls on hospitals with a positive dose. This biases* *(* *\mu^{sel}* *)* *downward relative to* *(* *ATT^{O}* *)* *—our estimate of the numerator in Equation (3.1) is 0.60. In contrast, the “weighted distance” between the effective treated and comparison groups in the denominator of Equation (3.1) is estimated to be 0.53, and dividing by 0.53 results in* *(* *\mu^{sel}* {TWFE} ) being upward biased relative to ( ATT^{O} ) . That these two biases work in opposite directions and have similar magnitudes in our particular application happens to result in ( \mu^{sel} *{TWFE}* *)* *being fairly close to* *(* *ATT^{O}* *)* *. Interestingly, though, if we instead were to code a hospital’s dose on a scale of 0 to 100, our estimate of* *(* *\mu^{sel}* {TWFE} ) shrinks to ( 0.0114 = 1.14/100 ) while our estimate of ( ATT^{O} ) remains unchanged.

Figure 7: Weighting Schemes for TWFE and Dose Distribution Among Treated

*Note:* The dashed lines are the weights that TWFE puts on ( ATT(w \mid w) ) and ( ACRT(w) ) parameters, as in Theorem 3.4. The solid line is a smoothed estimate of the density of the Medicare inpatient share, ( W ) .

Figure 7 abstracts from dynamics since it is based on average outcomes in the pre- and post-treatment periods. As an alternative, Figure 8 plots estimates of event-study summary parameters, ( ATT^{ES}(e) = E[Y\_{i,t+e}(D\_i) - Y\_{i,t+e}(0) \mid D\_i &gt; 0] / E[D\_i \mid D\_i &gt; 0] ) , using 1983 as the baseline year. The patterns are similar to the TWFE event-study in Figure 1, but their magnitudes reflect proper

29

averages of year-specific ( ATT(m \mid m) ) parameters.[^11]

**Figure 8: Event-Study Estimates of**

**(**

**ATT**

**)**

*Y-axis:* ( ATT(g,t) ) Estimates

*X-axis:* Time Relative to Treatment

*Annotation:* Average of post-treatment ( ATT(g,t) ) 's: 0.7418 (0.2045)

*Note:* The figure plots the event-study estimates of ( ATT^{es}(e) ) , with their 95% pointwise confidence intervals reported in black, and the 95% uniform confidence bands reported in red.

### 6.3 Average Causal Responses to PPS

Figure 9 plots our proposed data-adaptive nonparametric estimate of the slope of the function estimated in Figure 5. Under Assumption SPT, the function in Figure 5 is the ( ATT(m) ) and its slope in Figure 9 equals the ( ACRT(m) ) . The bump shape in Figure 5 is reflected in an ( ACRT(m) ) function that starts positive, and declines through most of its support. We estimate negative ( ACRT(m) ) parameters for doses above ( m = 0.41 ) , a range that includes 71 percent of treated hospitals. The 95% uniform confidence band covers zero everywhere, although we are able to detect positive ( ACRT(m) ) values for doses below the mean as well as negative ( ACRT(m) ) values for doses between about 0.5 and 0.7 using pointwise confidence intervals.

PPS’ average causal response parameter weighted by the actual dose distribution of treated hospitals is ( ACRT^O = -0.08 ) (se = 0.18) and is not significantly different from zero.[^12] This differs substantially from the TWFE coefficient, ( \hat{\beta}^{twfe} = 1.14 ) . From Theorem 3.4(a), the difference between these estimates is fully driven by differences in the weighting scheme. Our estimate of ( ACRT^O ) comes from mapping the estimates of ( ACRT(m) ) in Figure 9 to the dose distribution weights, ( f\_{M \mid T=1}(m) ) , in Figure 7; our estimate of ( \hat{\beta}^{twfe} ) comes from mapping the estimates of ( ACRT(m) ) to the TWFE causal response weights, ( \omega^{twfe}(m) ) , in Figure 7. As discussed in Theorem 3.4(a), the TWFE causal response weights are positive for all values of the dose and integrate to one, providing a reason to hope that estimates of ( ACRT^{twfe} ) and ( \hat{\beta}^{twfe} ) would be similar. However, the TWFE weighting scheme turns out to be much different from the dose distribution weighting

[^11]: The negative pre-PPS coefficient may reflect the fact that PPS was passed in April 1983 and partially took effect in that calendar year, and also that hospitals report labor and capital costs for different fiscal years. Therefore, some 1983 outcomes may include post-treatment months. The results also show that the ( ATT^{es}(e) ) grows each year following PPS, which matches the fact that PPS’ subsidy reforms actually phased in over three years. We also note, however, that these can represent other types of violations of parallel trends.

[^12]: We treat the dose dimension used to compute ( ACRT^O ) as a non-random sequence, which is in line with the theoretical justification in Ai and Chen (2003). A formal theoretical treatment that accounts for the stochastic nature of our logistic-type selection is interesting but left for future research.

30

scheme. Combining these differences with the high degree of heterogeneity in ( ACRT(m) ) across ( m ) is what leads to the sharp differences in the estimates. Another reason to emphasize the large difference between these estimates is that the literature has often viewed negative weights as a dividing line between an “unreasonable” or “reasonable” weighting scheme (see, e.g., Angrist (1998), de Chaisemartin and D’Haultfoeuille (2020), and Blandhol, Bonney, Mogstad, and Torgovitsky (2025) for related discussions of this point in different contexts). The results here suggest that, at least in our context, articulating a well-defined causal effect parameter and targeting that parameter directly is likely to be more important than checking that weights are all positive and integrate to one.

**Figure 9: Nonparametric Estimates of**

**(**

**ACRT(m)**

**)**

**for Medicare PPS**

<!-- image -->

Nonparametric estimates of ACRT(m) for Medicare PPS. The horizontal axis is Average Inpatient Share in 1983. The plot shows a nonparametric estimate with confidence bands and a histogram of the treatment dose among the treated.

*Notes:* The figure plots nonparametric estimates of ( ACRT(m) ) that adapts the Chen, Christensen, and Kankanala (2025) data-driven estimator to our context, as discussed in Section 5.1 and Appendix B. The dark-shaded region is the 95-percent point-wise confidence interval, and the lighter-shaded region is the 95-percent honest and sup-norm rate-adaptive uniform confidence band. We display the histogram of the treatment dose among the treated in yellow.

Under Assumption SPT, one policy implication of these estimates is that Medicare could have achieved similar, if not greater, capital investments while providing lower capital subsidies. Figure 9 shows that marginal increases in the subsidy ratio increase capital intensity only for those with low subsidy levels. Hospitals that received large capital subsidies under PPS responded with smaller increases in capital intensity than hospitals with slightly smaller subsidies, a fact easily seen in the binned scatter plot in Figure 4. The strong parallel trends assumption means that these estimated responses are “externally valid” for all treated hospitals, which means that only low subsidies matter for hospitals’ input choices. Because higher subsidy ratios do not create further investments in capital, capping capital subsidies may not affect input choices very much.

An important economic implication, however, is that negative ( ACRT(m) ) estimates contradict AF’s two-factor economic model. ( ACRT(m) ) is proportional to the average derivative of the optimal capital-labor ratio for hospitals with Medicare share equal to ( m ) , and Equation (6.1) shows specifically how: it relates to the elasticity of substitution, ( \sigma\_y(m) ) . To approximate ( E[\sigma\_y(m)\mid M&gt;0] ) , we separate out the two terms in (6.1) and construct

[ \frac{\widehat{\int ACRT(m) , dF\_{M\mid M&gt;0}(m)}}{\widehat{\int \left(\frac{1}{1-m}+\frac{(1-m)\widehat{R}(m)}{m}\right)dF\_{M\mid M&gt;0}(m)}}=-0.75. ]

With only two inputs, a rise in the relative price of one input lead to a reduction in its relative use: the elasticity of substitution must be positive. The point estimates of ( E[\sigma\_y(m)\mid M&gt;0] ) do not fit that

31

prediction, although our uniform confidence bands do not reject an average elasticity of substitution of zero. Alternative models, such as a three-factor production function (which AF consider in their working paper) or non-homothetic production, could potentially rationalize this finding.

Finally, both the policy and structural interpretations of Figure 9 depend on the strong parallel trends assumption. Without SPT, the slope of $ATT(m|m)$ may be negative for higher-Medicare-share hospitals simply because their treatment effect functions are systematically lower. Medicare might not have been able to achieve similar capital increases with lower subsidy rates if high-subsidy hospitals just responded differently to low subsidy levels than low-subsidy hospitals did. A negative slope also does not necessarily reject a two-factor production model; just a constant-coefficient model with homogeneous firms, as considered by AF.

Figure 10: Event-Study Estimates of $ACRT$

Average of post-treatment ACRTs, among all possible dosages $=0.240 , (0.79)$

Note: The figure plots the event-study estimates of $ACRT\_e^{pt}(m|m)$, with their 95% pointwise confidence intervals reported in black, and the 95% uniform confidence bands reported in red.

Another indirect way to assess the plausibility of SPT that justifies a causal interpretation of $ACRT^{pt}(m)$ is to compute $ACRT\_e^{pt}(m)$, the event-study version of $ACRT^{pt}(m)$. These parameters can be estimated using the same procedure discussed in Section 4, and we plot these in Figure 10. The no-anticipation assumption means that prior to treatment, when all observed outcomes are untreated potential outcomes, both Assumptions PT and SPT have the same implication: that the average relationship between outcome changes for adjacent dose groups should be zero. Our estimates of these pre-trends reject this in 1981, which is a pre-treatment period. Figure 10 corroborates our conclusions about the implausibility of SPT based on implausibly high implied elasticities of substitution.

In summary, our empirical results align with AF’s conclusion that the 1983 Medicare reform led hospitals to favor capital over labor. We find evidence against parallel trends in pre-treatment periods, though the magnitudes of these violations are small relative to estimated effects in post-treatment periods. Finally, our negative estimates of $ACRT(m)$ at high values of $m$ cut against the theoretical predictions of the model discussed above; this provides a piece of evidence that casts doubt on the plausibility of strong parallel trends in this application, indicating that one should be cautious when interpreting $ACRT$ parameters.

32

# References

Acemoglu, Daron and Amy Finkelstein (2008). “Input and technology choices in regulated industries: Evidence from the health care sector”. *Journal of Political Economy* 116.5, pp. 837–880.

Ai, Chunrong and Xiaohong Chen (2003). “Estimation of possibly misspecified semiparametric conditional moment restriction models with different conditioning variables”. *Journal of Econometrics* 141, pp. 5–43.

American Hospital Association (1986). *AHA Annual Survey Database* . Tech. rep. Health Forum, LLC.

Angrist, Joshua D (1998). “Estimating the labor market impact of voluntary military service using Social Security data on military applicants”. *Econometrica* 66.2, pp. 249–288.

Angrist, Joshua D, Kathryn Graddy, and Guido W Imbens (2000). “The interpretation of instrumental variables estimators in simultaneous equations models with an application to the demand for fish”. *The Review of Economic Studies* 67.3, pp. 499–527.

Angrist, Joshua D and Guido W Imbens (1995). “Two-stage least squares estimation of average causal effects in models with variable treatment intensity”. *Journal of the American Statistical Association* 90.430, pp. 431–442.

Angrist, Joshua D and Jorn-Steffen Pischke (2008). *Mostly Harmless Econometrics: An Empiricist’s Companion* . Princeton University Press.

Baker, Andrew, Brantly Callaway, Scott Cunningham, Andrew Goodman-Bacon, and Pedro H.C. Sant’Anna (2025). “Difference-in-differences designs: A practitioner’s guide”. *Journal of Economic Literature* Forthcoming.

Blandhol, Christian, John Bonney, Magne Mogstad, and Alexander Torgovitsky (2025). “When Is TSLS actually LATE?” *Review of Economic Studies* Forthcoming.

Borusyak, Kirill, Xavier Jaravel, and Jann Spiess (2024). “Revisiting event-study designs: Robust and efficient estimation”. *Review of Economic Studies* 91.6, pp. 3253–3285.

Callaway, Brantly, Andrew Goodman-Bacon, and Pedro H. C. Sant’Anna (2024). “Event studies with a continuous treatment”. *AEA Papers and Proceedings* 114, pp. 601–605.

Callaway, Brantly and Pedro H.C Sant’Anna (2021). “Difference-in-differences with multiple time periods”. *Journal of Econometrics* 225.2, pp. 200–230.

Cameron, A. Colin and Pravin K. Trivedi (2005). *Microeconometrics: Methods and Applications* . Cambridge University Press.

Chen, Xiaohong, Timothy Christensen, and Sid Kankanala (2025). “Adaptive estimation and uniform confidence bands for nonparametric structural functions and elasticities”. *Review of Economic Studies* 92.1, pp. 162–196.

Chodorow-Reich, Gabriel, Plamen T. Nenov, and Alp Simsek (2021). “Stock market wealth and the real economy: A local labor market approach”. *American Economic Review* 111.5, pp. 1613–57.

de Chaisemartin, Clement and Xavier D’Haultfoeuille (2018). “Fuzzy differences-in-differences”. *The Review of Economic Studies* 85.2, pp. 999–1028.

— (2020). “Two-way fixed effects estimators with heterogeneous treatment effects”. *American Economic Review* 110.9, pp. 2964–2996.

de Chaisemartin, Clément and Xavier D’Haultfoeuille (2023). “Difference-in-differences estimators of intertemporal treatment effects”. *Review of Economics and Statistics* , pp. 1–45.

de Chaisemartin, Clement, Xavier D’Haultfoeuille, Félix Pasquier, Doulo Sow, and Gonzalo Vazquez-Bare (2025). “Difference-in-differences for continuous treatments and instruments with stayers”. Working Paper.

Fricke, Hans (2017). “Identification based on difference-in-differences approaches with multiple treatments”. *Oxford Bulletin of Economics and Statistics* 79.3, pp. 426–433.

Goodman-Bacon, Andrew (2021). “Difference-in-differences with variation in treatment timing”. *Journal of Econometrics* 225.2, pp. 254–277.

Hendren, Nathaniel (2016). “The policy elasticity”. *Tax Policy and the Economy* 30.1, pp. 51–89.

33

Meyer, Bruce D. (1995). “Natural and quasi-experiments in economics”. *Journal of Business &amp; Economic Statistics* 13.2, pp. 151–161.

Miguel, Edward, Shanker Satyanath and Ernest Sergenti (2004). “Economic shocks and civil conflict: an instrumental variables approach”. *Journal of Political Economy* 112.4, pp. 725–753.

Newey, Whitney K. and Thomas M. Stoker (1993). “Efficiency of weighted average derivative estimators and index models”. *Econometrica* 61.5, pp. 1199–1223.

Office of Technology Assessment (1984). “Medical Technology and Costs of the Medicare Program”. OTA-H-227.

Robins, James (1986). “A new approach to causal inference in mortality studies with a sustained exposure period—application to control of the healthy worker survivor effect”. *Mathematical Modelling* 7.9, pp. 1393–1512.

Saez, Emmanuel, Joel Slemrod, and Seth H. Giertz (2012). “The elasticity of taxable income with respect to marginal tax rates: A critical review”. *Journal of Economic Literature* 50.1, pp. 3–50.

Sun, Liyang and Sarah Abraham (2021). “Estimating dynamic treatment effects in event studies with heterogeneous treatment effects”. *Journal of Econometrics* 225.2, pp. 175–199.

Wooldridge, Jeffrey M (2010). *Econometric Analysis of Cross Section and Panel Data* . MIT press.

— (2025). “Two-way fixed effects, the two-way Mundlak regression, and difference-in-differences estimators”. *Empirical Economics* . Forthcoming.

Yitzhaki, Shlomo (1996). “On using linear regressions in welfare economics”. *Journal of Business &amp; Economic Statistics* 14.4, pp. 478–486.

# A Proofs of Main Results

## A.1 Proofs of Results in Section 3.2

This section contains the proofs of the results in Section 3.2 on identifying causal effect parameters such as $ATT(d|d)$ and $ATT(d)$ under parallel trends assumptions and with a continuous treatment or multi-valued discrete treatment.

### Proof of Theorem 3.1

*Proof.* To show the result, notice that

$$ \begin{aligned} ATT(d|d) &amp;= E[Y\_t(d) - Y\_t(0)|D=d] \ &amp;= E[Y\_t(d) - Y\_{t-1}(0)|D=d] - E[Y\_t(0) - Y\_{t-1}(0)|D=d] \ &amp;= E[Y\_t - Y\_{t-1}|D=d] - \delta \ &amp;= E[Y\_t|D=d] - E[Y\_{t-1}|D=d] - \delta \end{aligned} \tag{A.1} $$

where the second equality holds by adding and subtracting $E[Y\_{t-1}(0)|D=d]$, the third equality holds by Assumption PT, and the last equality holds because $Y\_t(d)$ and $Y\_{t-1}(0)$ are observed potential outcomes when $D=d$ and $Y\_t(0)$ and $Y\_{t-1}(0)$ are observed potential outcomes when $D=0$. That $ATT^o$ is identified holds immediately given its definition and that $ATT(d|d)$ is identified. To derive the particular expression for $ATT^o$, notice that

$$ ATT^o = E[ATT(D|D)|D&gt;0] $$

34

[ = \mathbb{E}\left[\left(\mathbb{E}[\Delta Y \mid D]-\mathbb{E}[\Delta Y \mid D=0]\right)\mid D&gt;0\right] ]

[ = \mathbb{E}[\Delta Y \mid D&gt;0]-\mathbb{E}[\Delta Y \mid D=0] ]

where the first equality is the definition of ( ATT^{disc} ) , the second equality holds from Equation (A.1), the first part of the third equality holds by an implication of the law of iterated expectations, and the second part of the third equality holds because ( \mathbb{E}[\Delta Y \mid D=0] ) is non-random. ( \square )

**Proof of Theorem 3.2**

*Proof.* To prove part (a), notice that

[ \begin{aligned} \mathbb{E}[\Delta Y \mid D=h]-\mathbb{E}[\Delta Y \mid D=l] &amp;= \left(\mathbb{E}[\Delta Y \mid D=h]-\mathbb{E}[\Delta Y \mid D=0]\right) - \left(\mathbb{E}[\Delta Y \mid D=l]-\mathbb{E}[\Delta Y \mid D=0]\right) \ &amp;= ATT(h \mid h)-ATT(l \mid l) \end{aligned} \tag{A.2} ]

where the first equality holds by adding and subtracting ( \mathbb{E}[\Delta Y \mid D=0] ) , and the second equality holds by Theorem 3.1. Next,

[ \begin{aligned} ATT(h \mid h)-ATT(l \mid l) &amp;= \mathbb{E}[Y\_2(h)-Y\_2(0)\mid D=h]-\mathbb{E}[Y\_2(l)-Y\_2(0)\mid D=l] \ &amp;= \mathbb{E}[Y\_2(h)-Y\_2(l)\mid D=h] \ &amp;\quad + \mathbb{E}[Y\_2(l)-Y\_2(0)\mid D=h]-\mathbb{E}[Y\_2(l)-Y\_2(0)\mid D=l] \ &amp;= \mathbb{E}[Y\_2(h)-Y\_2(l)\mid D=h] + \left(ATT(l \mid h)-ATT(l \mid l)\right) \end{aligned} \tag{A.3} ]

where the first equality holds by the definition of ( ATT(d \mid d) ) , the second equality holds by adding and subtracting ( \mathbb{E}[Y\_2(l)\mid D=h] ) , and the third equality holds by the definition of ( ATT(l \mid h) ) and ( ATT(l \mid l) ) . Notice that ( \mathbb{E}[Y\_2(h)-Y\_2(l)\mid D=h] ) is a causal response of going from dose ( l ) to dose ( h ) for dose group ( h ) . An alternative expression for this term is

[ \mathbb{E}[Y\_2(h)-Y\_2(l)\mid D=h] = ATT(h \mid h)-ATT(l \mid h) \tag{A.4} ]

Next, we prove part (b). Using a similar argument as above, notice that, for ( d \in \mathcal{D}^C ) and ( (d+h) \in \mathcal{D}^C ) ,

[ \begin{aligned} \frac{\mathbb{E}[\Delta Y \mid D=d]-\mathbb{E}[\Delta Y \mid D=d+h]}{h} &amp;= \frac{ATT(d \mid d)-ATT(d+h \mid d+h)}{h} \ &amp;= \frac{ATT(d \mid d)-ATT(d+h \mid d)+ATT(d+h \mid d)-ATT(d+h \mid d+h)}{h} \ &amp;= \frac{ATT(d \mid d)-ATT(d+h \mid d)}{h} + \frac{ATT(d+h \mid d)-ATT(d+h \mid d+h)}{h} \end{aligned} ]

where the first equality holds using the same argument as for Equation (A.2), and the second equality holds by using the arguments in Equations (A.3) and (A.4). The result holds by taking the limit as ( h \to 0 ) and the definition of ( ACRT(d \mid d) ) .

Finally, the second result in part (c) involving a discrete treatment holds by taking ( h=d\_h ) and ( l=d\_l ) in Equations (A.2) and (A.3) and by the definition of ( ACRT(d\_h \mid d\_l) ) . ( \square )

**Proof of Theorem 3.3**

*Proof.* For part (a), notice that

[ ATT(d)=\mathbb{E}[Y\_2(d)-Y\_2(0)\mid D&gt;0] ]

35

[ = \mathbb{E}[Y\_{i2}(d)-Y\_{i1}(0)\mid D&gt;0]-\mathbb{E}[Y\_{i2}(0)-Y\_{i1}(0)\mid D=0] ]

[ = \mathbb{E}[Y\_{i2}(d)-Y\_{i2}(0)\mid D&gt;0]+\mathbb{E}[Y\_{i2}(0)-Y\_{i1}(0)\mid D&gt;0] ]

[ -\mathbb{E}[Y\_{i2}(0)-Y\_{i1}(0)\mid D=0] ]

[ = \mathbb{E}[\Delta Y\mid D=d]-\mathbb{E}[\Delta Y\mid D=0] ]

where the second equality holds by adding and subtracting ( \mathbb{E}[Y\_{i2}(0)\mid D&gt;0] ) , the third equality holds by Assumption SPT, and the fourth equality holds because ( Y\_{i2}(d) ) and ( Y\_{i1}(0) ) are observed outcomes when ( D=d ) .

Next, we prove the first part of part (b). First, notice that

[ ATT(d)-ATT(0)=\mathbb{E}[Y\_{i2}(d)-Y\_{i2}(0)\mid D&gt;0]-\mathbb{E}[Y\_{i2}(0)-Y\_{i2}(0)\mid D&gt;0] ]

[ = \mathbb{E}[Y\_{i2}(d)-Y\_{i2}(0)\mid D&gt;0] ]

where the first equality holds by the definition of ( ATT(d) ) , and the second equality holds by cancelling the terms involving ( Y\_{i2}(0) ) . For the second part, notice that, from part (a), we have that

[ ATT(d)-ATT(0)=\left(\mathbb{E}[\Delta Y\mid D=d]-\mathbb{E}[\Delta Y\mid D=0]\right)-\left(\mathbb{E}[\Delta Y\mid D=0]-\mathbb{E}[\Delta Y\mid D=0]\right) ]

[ = \mathbb{E}[\Delta Y\mid D=d]-\mathbb{E}[\Delta Y\mid D=0]. ]

Now, for part (c), notice that for ( d\in \mathcal{D}\_c ) and ( (d+h)\in \mathcal{D}\_c ) ,

[ \frac{ATT(d)-ATT(d+h)}{h}=\frac{\mathbb{E}[\Delta Y\mid D=d]-\mathbb{E}[\Delta Y\mid D=d+h]}{h} ]

which follows from part (b). The result holds by taking the limit as ( h\to 0 ) and from the definition of ( ACRT(d) ) . Finally, the result in part (d) involving a discrete treatment holds from part (b) by taking ( h=d\_h ) and ( l=d\_l ) and by the definition of ( ACRT(d\_h) ) .

**Proof of Corollary 3.1**

*Proof.* The result holds immediately by averaging the results in Theorem 3.3 over the distribution of the dose among dose groups that experienced any positive amount of the treatment. ( \square )

## B Adapting CCK to DiD Contexts

In this Appendix, we provide more details on how to adapt the Chen, Christensen, and Kankanala (2025) (henceforth, CCK) data-driven nonparametric estimation and inference procedures in our DiD context. As discussed in Section 4.1, the CCK estimator for ( ATT(d) ) and ( ACRT(d) ) are given by

[ \widehat{ATT}\_{cck}(d)=\left(s^K(d)\right)'\widehat{\beta} *K,\qquad \widehat{ACRT}* {cck}(d)=\left(\partial s^K(d)\right)'\widehat{\beta}\_K, ]

where ( s^K(d) ) is a ( K ) -dimensional vector of cubic B-splines basis functions, ( \partial s^K(d)=(\partial s\_1^K(d)/\partial d,\ldots,\partial s\_K^K(d)/\partial d)' ) , ( \widehat{\beta} *K* *)* *is the* *(* *K* *)* *-dimensional vector of OLS estimators for* *(* *\beta\_K* *)* *, and* *(* *K* *)* *is the CCK data-driven estimator for the optimal sieve dimension. Henceforth, let* *(* *p* {&gt;0}=\sum\_{i=1}^n \mathbf{1} { D\_i&gt;0 } ) be the sample size with positive treatment dose.

To discuss the optimal choice of the sieve dimension ( K ) derived in CCK, we need to add more

36

notation. Let ( \mathcal{K}= { (k^2+3): k\in\mathbb{N}\cup { 0 } } ) be the set of possible sieve dimensions for the cubic B-splines. Let ( K^*=\min { k\in\mathcal{K}:k&gt;K } ) be the smallest sieve dimension in ( \mathcal{K} ) exceeding ( K ) , and ( v\_n=\max\_t { 1,(0.1\log n\_t)^4 } ) . Let ( { \omega\_i } *{i=1}^N* *)* *be iid standard normal draws independent of the data* *(* *{* *W\_i* *}* {i=1}^N= { Y\_i,D\_i,X\_i } \_{i=1}^N ) . In addition, let

[ \widehat{\phi} *K(W\_i,d)=\widehat v\_K(d)'\left(E\_n\left[1(D&gt;0)\cdot \widehat v\_K(D)\widehat v\_K(D)'\right]\right)^{-1}1(D\_i&gt;0)\widehat v\_K(D\_i)\widehat u* {i,K}, ]

with

[ \widehat u\_{i,K}=\Delta Y\_i-\widehat E\_K[\Delta Y\_i\mid D=0]-(s\_K(D\_i))'\widehat\beta\_K. ]

Finally, for a given ( K ) and ( K\_2 ) , let

[ \widehat\sigma\_{K,K\_2}^2(d)=\frac{1}{n}\sum\_{i=1}^n\left(\widehat\phi\_K(W\_i,d)-\widehat\phi\_{K\_2}(W\_i,d)\right)^2 ]

be an estimator of the (asymptotic) variance of the contrast ( \sqrt n(\widehat{ATT} *K(d)-\widehat{ATT}* {K\_2}(d)) ) , and consider the bootstrap process

[ Z\_{n,K,K\_2}^*(d)=\frac{1}{\sqrt n , \widehat\sigma\_{K,K\_2}(d)} \sum\_{i=1}^n(\omega\_i-\bar\omega)\left(\widehat\phi\_K(W\_i,d)-\widehat\phi\_{K\_2}(W\_i,d)\right). ]

For a given sieve dimension ( K\in\mathcal K ) , let

[ \widehat{ATT}\_K(d)=(\widehat v^K(d))'\widehat\beta\_K,\qquad \widehat{ACRT}\_K(d)=(\partial\_d\widehat v^K(d))'\widehat\beta\_K, \tag{B.1} ]

where ( \partial\_d v^K(s)=(\partial/\partial s)v^K(s) ) , and

[ \widehat\beta\_K=\arg\min\_{\beta\in\mathbb R^K}E\_n\left[\left(\Delta Y-\widehat E\_K[\Delta Y\mid D=0]-e^K(D)'\beta\right)^2\mid D&gt;0\right] ]

[ =E\_n\left[1(D&gt;0)e^K(D)e^K(D)'\right]^{-1} E\_n\left[1(D&gt;0)e^K(D)(\Delta Y-\widehat E\_K[\Delta Y\mid D=0])\right], \tag{B.2} ]

and ( A^{-} ) denote the Moore-Penrose inverse of a generic matrix ( A ) , and for a generic variable ( B ) ,

[ E\_n[B\mid D&gt;0]=\frac{\sum\_{i=1}^n 1(D\_i&gt;0)B\_i}{\sum\_{i=1}^n 1(D\_i&gt;0)}. ]

The next algorithm adapts Procedure 1 of CCK to our DiD context and provides the Lepski-type data-driven selection ( K ) of the sieve dimension ( K ) .

**Algorithm 1** Computation of data-driven choice of sieve dimension ( K ) based on CCKJ.

1. Compute the data-driven index set of sieve dimensions

[ \widehat{\mathcal K}=\left { K\in\mathcal K:0.1(\log R\_{\max})^2\le K\le R\_{\max}\right } \tag{B.3} ]

where

[ R\_{\max}:=\min\left { K\in\mathcal K:\sqrt K\log K\_n\le 10\sqrt n , K^{-s}\sqrt{\log K\_n}\right } . \tag{B.4} ]

1. Let ( \widehat{\mathcal X} *n* *)* *contain* *(* *0.5\sqrt{\log R* {\max}/n} ) . For each independent draw of ( { \omega\_i } \_{i=1}^n ) , compute

[ \sup\_{(d,K\_1,K\_2)\in\widehat{\mathcal L}}\left|Z\_n^*(d,K\_1,K\_2)\right|, \tag{B.5} ]

where

[ \widehat{\mathcal L}:=\widehat{\mathcal X}\_n\times\left { (K\_1,K\_2):K\_1,K\_2\in\widehat{\mathcal K},K\_1&gt;K\_2\right } . ]

Let ( \widehat c\_{1-\alpha}^* ) denote the ( (1-\alpha) ) quantile of the sup-t statistic (B.5) across a large number of independent draws of ( { \omega\_i } \_{i=1}^n ) , say, 1,000.

37

1. The data-driven choice of the size dimension is

[ \widehat K=\inf\left { K\in\mathcal K:\frac{\displaystyle\sup\_{K'\in\mathcal K:K'\ge K}\left|\widehat{ATT} *{K}(d)-\widehat{ATT}* {K'}(d)\right|}{\widehat\sigma\_{K,K'}(d)}\le 1.1t\_{1-\alpha}\right } . \tag{B.6} ]

Next, we show how one can form data-driven uniform confidence bands (UCBs) for both ( ATT(d) ) and ( ACRR(d) ) by adapting Procedure 2 of CCK to our DID context. Toward this end, let ( A=\log K ) and set ( \ell\_K=(\ell\_{K,1},\ldots,\ell\_{K,K}) ) . Define the bootstrap processes

[ Z\_K(d,\mathcal X)=\frac{1}{\sqrt n}\sum\_{i=1}^n \xi\_i\frac{\widehat\psi\_K(W\_i,d)}{\widehat\sigma^K(d)},\qquad Z\_K^{acrr}(d,\mathcal X)=\frac{1}{\sqrt n}\sum\_{i=1}^n \xi\_i\frac{\widehat\psi'\_K(W\_i,d)}{\widehat\sigma\_K^{acrr}(d)}, ]

where

[ q^K(W\_i,d)=\left(q\_K^K(d)\right)'\widehat Q\_K^{-1}q\_K(W\_i), ]

and

[ \widehat\sigma\_K^2(d)=\frac{1}{n}\sum\_{i=1}^n\left(\widehat\psi\_K(W\_i,d)\right)^2,\qquad \widehat\sigma\_K^{acrr}(d)=\frac{1}{n}\sum\_{i=1}^n\left(\widehat\psi'\_K(W\_i,d)\right)^2. ]

**Algorithm 2** Computation of UCBs for ( ATT(\cdot) ) and ( ACRR(\cdot) ) based on CCK.

1. For each independent draw of ( { \omega\_i } \_{i=1}^n ) , compute

[ t^ *=\sup\_{(d,K)\in\mathcal D\times\mathcal K}\left|Z\_K(d,\mathcal X)\right|,* *\qquad* *t^{acrr,* }=\sup\_{(d,K)\in\mathcal D\times\mathcal K}\left|Z\_K^{acrr}(d,\mathcal X)\right|. \tag{B.7} ]

Let ( c\_{1-\alpha} ) and ( c\_{1-\alpha}^{acrr} ) denote the ( (1-\alpha) ) quantile of the sup- ( t ) statistic ( t^ *)* *and* *(* *t^{acrr,* } ) , respectively, across a large number of independent draws of ( { \omega\_i } \_{i=1}^n ) .

1. The data-driven ( 100(1-\alpha) % ) UCB for ( ATT(d) ) and ( ACRR(d) ) , ( d\in\mathcal D ) , are respectively given by

[ C\_{cb}(d)= \left[ \widehat{ATT} *{\widehat K}(d)-\left(c* {1-\alpha}+\widehat A\cdot \widehat t\_{1-\alpha}\right)\frac{\widehat\sigma^{\widehat K}(d)}{\sqrt n}, \widehat{ATT} *{\widehat K}(d)+\left(c* {1-\alpha}+\widehat A\cdot \widehat t\_{1-\alpha}\right)\frac{\widehat\sigma^{\widehat K}(d)}{\sqrt n} \right] \tag{B.8} ]

[ C\_{acrr}(d)= \left[ \widehat{ACRR} *{\widehat K}(d)-\left(c* {1-\alpha}^{acrr}+\widehat A\cdot \widehat t\_{1-\alpha}^{acrr}\right)\frac{\widehat\sigma\_{\widehat K}^{ACRR}(d)}{\sqrt n}, \widehat{ACRR} *{\widehat K}(d)+\left(c* {1-\alpha}^{acrr}+\widehat A\cdot \widehat t\_{1-\alpha}^{acrr}\right)\frac{\widehat\sigma\_{\widehat K}^{ACRR}(d)}{\sqrt n} \right]. \tag{B.9} ]

## C Multiple Periods and Variation in Treatment Timing and Dose

DID applications often use more than two time periods, wherein treatments, whether binary or not, can turn on at different times for different units. This section extends the results from the main text to allow for multiple time periods ( ( t=1,\ldots,T ) ) with variation in the time when units become treated. We refer to the time period when a unit becomes treated as a unit’s *timing group* , which we denote by ( G\_i ) , which takes values in the set ( \mathcal G ) . By convention, we set ( G=\infty ) for units that remain untreated across all time periods, and we exclude units that are treated in the first period so that ( \mathcal G\subseteq { 2,\ldots,T,\infty } ) ; we also set ( \mathcal G=\mathcal G\setminus { \infty } ) to be the set of all timing groups that ever participate in the treatment. Treated units receive dose ( D=d\in\mathcal D\_g ) . As in the two-period case, the dose actually experienced, ( D ) , also defines a unit’s *dose group* .

We extend the potential outcomes notation from the previous section to allow for variation in treatment timing. Therefore, potential outcomes ( Y\_{i,t}(g,d) ) denote the outcome for unit ( i ) at time period ( t ) when such a unit is first treated in period ( g ) with dosage ( d ) . Note that treated potential outcomes at time ( t ) depend on when a unit first becomes treated—i.e., ( Y\_{i,t}(g,d) ) may not equal ( Y\_{i,t}(g',d) ) for

38

( g \ne g' ) — which allows for general treatment effect dynamics. ( Y\_{it}(\infty,0) ) is the outcome that unit ( i ) would experience if it did not participate in the treatment in any period. We write ( Y\_{it}(0)=Y\_{it}(\infty,0) ) and refer to this as a unit’s untreated potential outcome. We also define the variable ( W\_{it}=D\_i(1 { t \ge G\_i } ) ) , which is the amount of dose that unit ( i ) experiences in time period ( t ) . ( W\_{it}=0 ) for all units that are not yet treated by time period ( t ) .

Throughout this section, we make the following assumptions.

**Assumption 1-MP (Random Sampling).** The observed data consists of ( { Y\_{i1},\ldots,Y\_{iT},D\_i,G\_i } \_{i=1}^n ) which is independent and identically distributed.

**Assumption 2-MP (Support).** (a) The support of ( D ) , ( \mathcal{D}= { 0 } \cup\mathcal{D} *+* *)* *, where* *(* *\mathcal{D}* + \subseteq (0,\infty) ) . In addition, ( \mathbb{P}(D=0)&gt;0 ) and ( dF\_{G|D}(g|d)&gt;0 ) for all ( (g,d)\in \mathcal{G}\times\mathcal{D} *+* *)* *.* *(b)* *(* *\mathcal{D}* +=(d\_1,\ldots,d\_J) ) . In addition, for all ( g\in\mathcal{G} ) and ( t=2,\ldots,T ) , ( E[\Delta Y\_t|G=g,D=d] ) is continuously differentiable in ( d ) on ( \mathcal{D}\_+ ) .

**Assumption 3-MP (No Anticipation / Staggered Adoption).** (a) For all ( g\in\mathcal{G} ) and ( t=1,\ldots,T ) with ( t&lt;g ) (i.e., in pre-treatment periods), ( Y\_{it}(g,d)=Y\_{it}(0) ) . (b) ( W\_{i1}=0 ) almost surely, and, for ( t=2,\ldots,T ) and ( d\in\mathcal{D} *+* *)* *,* *(* *W* {i,t-1}=d ) implies that ( W\_{it}=d ) .

We next introduce versions of Assumption PT and SPT that are suitable for the setting with multiple periods and variation in treatment timing.^13

**Assumption PT-MP (Parallel Trends with Multiple Periods and Variation in Treatment Timing).** For all ( g,g'\in\mathcal{G} ) , ( t=2,\ldots,T ) , ( d\in\mathcal{D}\_+ ) , ( E[\Delta Y\_t(0)|G=g,D=d]=E[\Delta Y\_t(0)|G=\infty,D=0] ) .

**Assumption SPT-MP (Strong Parallel Trends with Multiple Periods and Variation in Treatment Timing).** For all ( g\in\mathcal{G} ) , ( t=2,\ldots,T ) , and ( d,d'\in\mathcal{D} ) , ( E[Y\_t(g,d)-Y\_{t-1}(g,d)|G=g,D=d]=E[Y\_t(g,d)-Y\_{t-1}(g,d)|G=g,D=d'] ) .

For each unit, we observe their outcome in period ( t ) , ( Y\_{it} ) , which is given by

[ Y\_{it}=Y\_{it}(0)1 { t&lt;G\_i } +Y\_{it}(G\_i,D\_i)1 { t\ge G\_i } . ]

## C.1 Identification with a Staggered Continuous Treatment

The causal parameters of interest are the same as in our baseline case, except that they are separately defined for each timing group and in each post-treatment time period:

[ ATT(g,t,d|d):=E[Y\_t(g,d)-Y\_t(0)|G=g,D=d], ]

and

[ ATT(g,t,d):=E[Y\_t(g,d)-Y\_t(0)|G=g,D&gt;0]. ]

^13 Besides differences related to multiple periods and variation in treatment timing, the version of strong parallel trends made here is slightly different from Assumption SPT in the main text. Part of the difference comes from there being no untreated units in group ( g\ne g' ) , which is why there is a separate part of the assumption for untreated potential outcomes. The other difference is that both parts of the assumption hold for all dose groups rather than on average (i.e., we condition on dose group ( d ) in the first part and on dose group ( d ) in the second part for untreated potential outcomes). The version here is stronger and is made for clarity and because we target ( ACRT(g,t,d,d') ) rather than ( ACRT(g,t,d) ) in this part of the paper.

39

Causal response parameters are similarly defined as the effect of a marginal change in the dose on the outcomes of timing group ( g ) in period ( t ) . For continuous treatments, these are defined as

# 

[

ACRT(g,t,d\_g,\boldsymbol{d}) =

\frac{\partial ATT(g,t,d\_g,\boldsymbol{d})}{\partial d\_g}

\left. \frac{\partial E[Y\_t(g,\delta)-Y\_t(0)\mid G\_g=1,\boldsymbol{D}=\boldsymbol{d}]}{\partial \delta} \right|\_{\delta=d\_g}, ]

[ ACRT(g,t,\boldsymbol{d}) = \frac{\partial ATT(g,t,\boldsymbol{d})}{\partial d} ]

For discrete treatments, these are defined as

[ ACRT(g,t,d\_g,\boldsymbol{d}) = E[Y\_t(g,d\_g)-Y\_t(g,d\_g-1)\mid \boldsymbol{D}=\boldsymbol{d},G\_g=1], ]

[ ACRT(g,t,d)=E[Y\_t(g,d)-Y\_t(g,d-1)\mid D\ge d,G\_g=1], ]

For brevity, henceforth we focus on the “local” causal effect parameters ( ATT(g,t,d\_g,\boldsymbol{d}) ) and ( ACRT(g,t,d\_g,\boldsymbol{d}) ) , which are analogous to the local causal effect parameters ( ATT^l(d) ) and ( ACRT^l(d) ) in the two-period case that we emphasized in the main text.

**Theorem C.1.** *Under Assumptions 1-MP, 2-MP(g), 3-MP, and PT-MP, and for all* *(* *g \in \mathcal{G}* *)* *,* *(* *t=2,\ldots,T* *)* *such that* *(* *t \ge g* *)* *, and for all* *(* *\boldsymbol{d}\in \mathcal{D}\_g* *)* *,*

## 

[

ATT(g,t,d\_g,\boldsymbol{d}) =

E[Y\_t-Y\_{g-1}\mid G\_g=1,\boldsymbol{D}=\boldsymbol{d}]

E[Y\_t-Y\_{g-1}\mid \boldsymbol{D}\_t=\boldsymbol{0}]. ]

*If, in addition, Assumptions 2-MP(b) and SPT-MP hold, then, for all*

*(*

*\boldsymbol{d}\in \mathcal{D}\_g*

*)*

*,*

[ ACRT(g,t,d\_g,\boldsymbol{d}) = \frac{\partial E[Y\_t-Y\_{g-1}\mid G\_g=1,\boldsymbol{D}=\boldsymbol{d}]}{\partial d\_g}. ]

The proof of Theorem C.1 is provided in Appendix SC in the Supplementary Appendix. The result is broadly similar to the one in the case with two periods. The first part says that, under Assumption PT-MP, ( ATT(g,t,d\_g,\boldsymbol{d}) ) can be recovered by a DID comparison between the path of outcomes from period ( g-1 ) to period ( t ) for units in group ( g ) treated with dose ( d ) and the path of outcomes among units that have not participated in the treatment yet (the setup in this section also rationalizes using the never-treated group, ( G=\infty ) , as the comparison group as was mentioned in Section 5). Relative to the case with two time periods, the main difference is that the “base period” is ( g-1 ) . The reason for using the base period ( g-1 ) is that it is the most recent time period where the researcher observes untreated potential outcomes for units in group ( g ) . Thus, the result is very much like the case with two time periods: take the most recent untreated potential outcomes for units in a particular group, impute the path of outcomes that they would have experienced in the absence of participating in the treatment from the group of not-yet-treated units (these steps yield mean untreated potential outcomes that units in group ( g ) would have experienced in time period ( t ) ) and compare this to the outcomes that are actually observed for units in group ( g ) that experienced dose ( d ) . The second part says that, under Assumption SPT-MP, ( ACRT(g,t,d\_g,\boldsymbol{d}) ) can be recovered by taking the derivative of the average path of outcomes from period ( g-1 ) to period ( t ) among timing group ( g ) that experienced dose ( d ) . Similarly to the arguments in the main text, if Assumption PT-MP held rather than SPT-MP, then the same derivative term would additionally include selection bias terms.

40

Given the results in Theorem C.1, it follows that causal summary parameters that are aggregations of these dose-and-timing-group-specific parameters are also identified. Of course, one can consider many different types of aggregation, as discussed in Callaway and Sant’Anna (2021) and Callaway, Goodman-Bacon, and Sant’Anna (2024), for example. Here, given that the treatment is continuous, we discuss some aggregations that can remain dose-specific and help highlight heterogeneity in treatment dosages. We provide these estimands as being explicit about them is useful for estimation and inference, and provide more transparency when comparing across procedures (Baker et al., 2025).

We start discussing natural aggregated parameters, including the average treatment effect of dose ( d ) across post-treatment periods for dose group ( d ) ,

[ ATT^{dose}(d)=E[TE\_i(d)\mid D=d,G\le \mathcal{T}] =\sum\_{g=2}^{\mathcal{T}}\sum\_{t=g}^{\mathcal{T}}\omega^{dose}(g,t,d)ATT(g,t,d\mid d), ]

where

[ \omega^{dose}(g,t,d)=\frac{1 { g\le t } P(G=g\mid D=d,G\le \mathcal{T})}{\mathcal{T}-g+1}, ]

and

[ TE\_i(d)=\frac{1}{\mathcal{T}-G\_i+1}\sum\_{t=G\_i}^{\mathcal{T}}\left(Y\_{it}(G\_i,d)-Y\_{it}(0)\right). ]

We can likewise define a causal response parameter,

[ ACR^{dose}(d)=\frac{\partial ATT^{dose}([d])}{\partial d} =\sum\_{g=2}^{\mathcal{T}}\sum\_{t=g}^{\mathcal{T}}\omega^{dose}(g,t,d)ACRT(g,t,d\mid d), ]

and even further aggregate these parameters into scalar summary parameters:

[ ATT^c=E[ATT^{dose}(D)\mid D\in \mathcal{D}] \quad \text{and} \quad ACRT^c=E[ACRT^{dose}(D)\mid D\in \mathcal{D}]. ]

Empirical researchers are also often interested in analyzing how average treatment effects vary with elapsed treatment timing and consider event-study-type parameters. In our context, with continuous and staggered treatments, one can consider the following dose-specific event study parameters,

[ \overline{ATT}^{es-dose}(d,\ell)=E[TE\_i(d)\mid D=d,G+\ell\le \mathcal{T},\ell\le \mathcal{T}] =\sum\_{g=2}^{\mathcal{T}}\sum\_{t=g}^{\mathcal{T}}\omega^{es}(g,t,d;\ell)ATT(g,t,d\mid d), ]

[ \overline{ACR}^{es-dose}(d,\ell)=\frac{\partial \overline{ATT}^{es-dose}([d],\ell)}{\partial d}\bigg| *{d=d}* *=\sum* {g=2}^{\mathcal{T}}\sum\_{t=g}^{\mathcal{T}}\omega^{es}(g,t,d;\ell)ACRT(g,t,d\mid d), ]

where ( \overline{ATT}^{es-dose}(d,\ell) ) and ( \overline{ACR}^{es-dose}(d,\ell) ) are the average treatment effect of dose ( d ) and average causal response of dose ( d ) among those in dose group ( d ) for those that have been exposed to the treatment for ( \ell ) periods, ( TE\_i(\ell)=Y\_{i,G\_i+\ell}(G\_i,d)-Y\_{i,G\_i+\ell}(0) ) , ( \omega^{es}(g,t,d;\ell)=1 { g+\ell=t } 1 { g+\ell\le \mathcal{T} } P(G\_i=g\mid D=d,G+\ell\le \mathcal{T},\ell\le \mathcal{T}) ) , and ( \omega^{es}(g,t,d)=1 { g+\ell=t } ) .

When one also wants to aggregate over treatment dosages to get an easier-to-estimate causal parameter of interest,

[ \overline{ATT}^{es}(\ell)=E[\overline{ATT}^{es-dose}(D,\ell)\mid G+\ell\in[2,\mathcal{T}],G\le \mathcal{T}] ]

[ \overline{ACR}^{es}(\ell)=E[\overline{ACR}^{es-dose}(D,\ell)\mid G+\ell\in[2,\mathcal{T}],G\le \mathcal{T}] ]

41

which provide event study versions of average treatment effects and average causal responses across different lengths of exposure to the treatment. For values of $e \geq 0$, $ATT^{es}(e)$ and $ACR^{es}(e)$ are related to treatment effect dynamics. It is also interesting to consider cases where $e &lt; 0$, which can be interpreted as a pre-test of the parallel trends assumption. See also Callaway, Goodman-Bacon, and Sant’Anna (2024) for a discussion.

**Remark C.1.** *We do not provide formal estimation results for the setting with multiple periods and variation in treatment timing. However, we note that, if one bases estimation on the sample analog of the results in Theorem C.4, then the results in the main text for the case with two periods apply directly to the disaggregated parameters $ATT(g,t,\delta,g)$ and $ACR^T(g,t,\delta,g)$. Then, one can estimate any of the aggregated parameters discussed above as the appropriate weighted average of $ATT(g,t,\delta,g)$ or $ACR^T(g,t,\delta,g)$. Interestingly, given the results in Corollary 3.1 and Callaway and Sant’Anna (2021), when $ATT^{es}(e)$ is the target parameter, one can binarize the treatment (i.e., classify units as being treated if they experience any positive amount of the treatment) and simply rely on the event-study procedures proposed by Callaway and Sant’Anna (2021).*

42