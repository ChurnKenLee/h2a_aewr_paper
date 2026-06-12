arXiv:2503.13323v3 [econ.EM] 17 Jun 2025

# Difference-in-Differences Designs: A Practitioner’s Guide

Andrew Baker *      Brantly Callaway †      Scott Cunningham ‡ Andrew Goodman-Bacon §      Pedro H. C. Sant’Anna ¶

June 18, 2025

## Abstract

Difference-in-differences (DiD) is arguably the most popular quasi-experimental research design. Its canonical form, with two groups and two periods, is well-understood. However, empirical practices can be ad hoc when researchers go beyond that simple case. This article provides an organizing framework for discussing different types of DiD designs and their associated DiD estimators. It discusses covariates, weights, handling multiple periods, and staggered treatments. The organizational framework, however, applies to other extensions of DiD methods as well.

# 1 Introduction

Dating to the 1840s, difference-in-differences (DiD) is now the most common research design for estimating causal effects in the social sciences. 1 A basic DiD design requires two time periods, one before and one after some treatment begins, and two groups, one that receives a treatment and one that does not. The DiD estimate equals the change in outcomes for the treated group minus the change in outcomes for the untreated group: the difference of two differences. If the average change in the outcomes would have been the same in the two groups had the treatment not occurred, which is referred to as a “parallel trends” assumption, this comparison estimates the average treatment effect among treated units.

* University of California, Berkeley † University of Georgia ‡ Baylor University § Opportunity and Inclusive Growth Institute, Federal Reserve Bank of Minneapolis ¶ Emory University

1 Currie, Kleven and Zwiers (2020) find that almost 25% of all NBER empirical working papers and 17% of empirical articles in five leading general-interest economics journals in 2018 mention DiD. The earliest DiD applications we are aware of are from Ignaz Semmelweis from the 1840s (Semmelweis, 1983) and Snow (1855). For a brief overview of the long history of DiD in economics, see Section 2 of Lechner (2011).

1

In practice, however, researchers apply DiD methods to situations that are more complicated than the classic two-period and two-group (2 × 2) setup. Most datasets cover multiple periods, and units may enter (or exit) treatment at different times. Treatment might also vary in its amount or intensity. Other variables are often used to make treated and untreated units more comparable. Today’s typical DiD study includes at least one of these deviations from the canonical 2 × 2 setup.

For many years, the common practice in applied research was to estimate complex DiD designs using linear regressions with unit and time fixed effects (two-way fixed effects, henceforth TWFE). Their identifying assumptions and interpretation were informally traced to the fact that, in the 2 × 2 case, a TWFE estimator gives the same estimate as a DiD estimator calculated directly from sample means, and thus inherits a clear causal interpretation under a specific parallel trends identification assumption. This appeared to justify the use of a single technique for any type of design or specification. Recent research, however, has shown that simple regressions can fail to estimate meaningful causal parameters when DiD designs are complex and treatment effects vary, producing estimates that are not only misleading in their magnitudes but potentially of the wrong sign. The significance of these findings is substantial; given the prevalence of DiD analysis in modern applied econometrics work, common empirical practices have almost certainly yielded misleading results in several concrete cases (Baker, Larcker and Wang, 2022).

So, what should applied researchers do instead? This paper proposes a unified framework for discussing and conducting DiD studies that is rooted in the principles of causal inference in the presence of treatment effect heterogeneity. The central conclusion of recent methodological research is that even complex DiD studies can be understood as aggregations of 2 × 2 comparisons between one set of units for which treatment changes and another set for which it does not. This fact links a wide variety of DiD designs used in practice and guides methodological choices about estimating them. Viewing DiD studies through the lens of 2 × 2 “building blocks” aids in interpretability by clarifying that they yield causal quantities that aggregate the treatment effects identified by each 2 × 2 component. It also means that identification comes from the simple parallel trends assumptions required for each 2 × 2 building block. Practically, this framework suggests first estimating each 2 × 2 building block and then aggregating them. As long as the effective sample size is large, this approach allows for asymptotically valid inference using standard techniques.

This framework is a “forward-engineering” approach to DiD that embraces treatment effect heterogeneity and constructs estimators that recover well-motivated causal parameters under explicitly stated assumptions. By fixing the goals of the study (the target parameters) and deriving analytical techniques, forward engineering provides clear benefits over “reverse-engineering” approaches that begin with a familiar regression specification and derive the assumptions under which it has some causal interpretation. The methods we describe in this paper combine familiar techniques with some newer ones, but expressly avoid the difficulties of interpretation inherent in common regression estimators (Goodman-Bacon, 2021; de Chaisemartin and D’Haultfoeuille, 2020; Sun and Abraham, 2021; Borusyak, Jaravel and Spiess, 2024). Moreover, the interpretation

2

of common regression estimators changes across specifications, which makes it hard to understand the difference between non-robustness and a shifting target parameter. In contrast, our proposed framework naturally leads to estimation procedures that target the same parameter under different transparent identification assumptions. Thus, two estimates can be distinguished easily by their identifying assumptions. Finally, the principles of the forward-engineering approach provide guidance to good econometric practices even in settings without well-established methodological findings.

This paper is not designed to be a comprehensive literature review; its goal is to provide guidelines for practitioners who want to better understand DiD and its various forms. Because of the tremendous variations in design, data, and specification that practitioners encounter, we opt to focus on three of the most common aspects of modern DiD studies: the use of weights, covariates, and staggered treatment timing. Table A1 includes a list of the acronyms that we, and the econometrics literature on DiD, use to distinguish different methods. We apply techniques to address these issues to a specific example: the causal effect of recent public health insurance expansions in the US on county-level mortality. Our replication materials include data as well as R and Stata code that can serve as a template for any DiD study using these methods. In an appendix, we briefly discuss related DiD designs with different treatment variables (ones that turn on and off or take many values), additional comparisons (i.e., triple-difference designs), distributional target parameters, or different data structures (repeated cross-sections or unbalanced panels). Several recent reviews follow the logic laid out here and cover additional DiD-related topics and technical details: Roth, Sant’Anna, Bilinski and Poe (2023); de Chaisemartin and D’Haultfoeuille (2023b); Callaway (2023).

The rest of the paper is structured as follows. Section 2 introduces the Medicaid example. Section 3 discusses the canonical 2×2 DiD setups with and without weights, and Section 4 discusses threats to the identification assumptions, how to assess them, and how to incorporate covariates. Section 5 extends the 2×2 setup to multiple periods with potentially staggered treatment adoption. Section 6 concludes and briefly discusses some extensions that involve more complex DiD designs.

## 2 Medicaid and mortality: The running example

To make our methodological discussion concrete, we revisit a timely and important causal question: How did the expansion of public health insurance (Medicaid) under the Affordable Care Act (ACA) affect mortality?

Medicaid expansion is a great example of a staggered treatment adoption. The ACA originally mandated that in 2014 all states expand Medicaid eligibility to adults with incomes up to 138% of the federal poverty threshold. In upholding the law’s constitutionality in a 2012 decision, however, the Supreme Court made Medicaid expansion optional. As a result, many states expanded Medicaid after 2014, but several have not done so as of 2024.

3

Columns 1 and 2 of Table 1 illustrate the variation in Medicaid expansion dates.

Table 1: Medicaid Expansion under the Affordable Care Act

| Expansion Year   | States                                                                                 |   Share of States |   Share of Counties |   Share of Adults (2013) |
|------------------|----------------------------------------------------------------------------------------|-------------------|---------------------|--------------------------|
| Pre-2014         | DE, MA, NY, VT                                                                         |              0.08 |                0.03 |                     0.09 |
| 2014             | AR, AZ, CA, CO, CT, IL, IA, HI, KY, MD, MI, MN, ND, NH, NJ, NM, NV, OH, OR, RI, WA, WV |              0.44 |                0.36 |                     0.45 |
| 2015             | AK, IN, PA                                                                             |              0.06 |                0.06 |                     0.06 |
| 2016             | LA, MT                                                                                 |              0.04 |                0.04 |                     0.02 |
| 2019             | ME, VA                                                                                 |              0.04 |                0.05 |                     0.03 |
| 2020             | ID, NE, UT                                                                             |              0.06 |                0.04 |                     0.02 |
| 2021             | MO, OK                                                                                 |              0.04 |                0.05 |                     0.03 |
| 2023             | NC, SD                                                                                 |              0.04 |                0.05 |                     0.03 |
| Non-Expansion    | AL, FL, GA, KS, MS, SC, TN, TX, WI, WY                                                 |              0.20 |                0.31 |                     0.26 |

The table shows which states adopted the ACA’s Medicaid expansion in each year, as well as the share of all states, counties, and adults in each expansion year.

States expanded Medicaid largely because of economic and political considerations (Sommers and Epstein, 2013), which created observable differences between expansion and non-expansion states. For instance, just four out of the 22 states that expanded Medicaid in 2014 are in the southern Census region; conversely, seven out of 10 non-expansion states are in the South. This suggests a potential role for covariates when analyzing Medicaid expansion.

Finally, mortality is measured in jurisdictions like states and counties, which are of very different sizes. Choices about (population) weights determine not only how different estimation approaches average the units within a given expansion group but also how a given estimation technique averages estimated effects across those groups. California, for example, represented 4.5% of the states that expanded Medicaid in 2014, 0.4% of the counties, but 27.7% of the adults ages 20-64; its contribution to “the” average outcome for the 2014 expansion group is very different with weights than without. The final three columns of Table 1 show that, in our data the entire 2014 expansion group contains 44% of the states, 36% of the counties, but 45% of all adults. Weighting will therefore change how important the estimated treatment effects are for the 2014 group.

Several recent papers study the effect of ACA Medicaid expansion on mortality rates for lower-income adults, who are most likely to gain insurance through Medicaid. Miller, Johnson and Wherry (2021) and Wyse and Meyer (2024) use simple DiD methods to provide evidence that Medicaid reduced adult mortality rates for targeted subpopulations. Unfortunately, their analyses require restricted links between income and mortality data, which are important for overcoming the low statistical power in studies using aggregate mortality data (Black, Hollingsworth, Nunes and Simon, 2022). Our goal is to pursue a replicable and shareable example based on a related analysis by Borgschulte and Vogler (2020). They use a sophisticated strategy to select and use covariates in a weighted TWFE regression using restricted access data, and find that Medicaid

4

expansion reduced aggregate county-level mortality rates. We use publicly available data, which we include in a fully-reproducible replication package, and consider only a handful of intuitive demographic and economic covariates sufficient to illustrate several practical challenges that can arise with DiD. This empirical exercise is meant solely to illustrate how to tackle several common features of DiD designs. The results are pedagogical in spirit and do not represent the best possible estimates of Medicaid’s effect on adult mortality.

Our outcome variable is the crude adult mortality rate, ( Y\_{it} ) , for people ages 20-64 (measured per 100,000) by county ( ( i ) ) from 2009 to 2019 released by the Centers for Disease Control and Prevention (2024).[^2] We denote county ( i ) ’s adult population in 2013 by ( W\_i ) and its socioeconomic covariates in year ( t ) (discussed below) by ( X\_{it} ) . The information in Table 1 defines the treatment group variable ( G\_i ) that equals the year in which county ( i ) ’s state expanded Medicaid: ( G\_i = \infty ) for the non-expansion states. Our final sample contains 2,604 counties in states with complete data on mortality rates from 2009 to 2019 and covariates for 2013 and 2014.

Faced with a setup such as this, researchers need to make a range of tightly related choices. Which treatment groups in Table 1 should be compared with each other and over what time horizons? What must be true for those comparisons to identify causal effects, and how should one empirically evaluate their plausibility? How can other information, such as covariates or pre-period outcomes, be used to improve the credibility of the design? How do these methodological choices affect the causal interpretation of a given analysis? The aim of this review is to demonstrate to practitioners using DiD in realistic scenarios why and how to use state-of-the-art econometric tools to answer these questions.

## 3 2×2 DiD designs

We begin our discussion by focusing on the canonical 2×2 DiD setup, which has two time periods, one before and one after treatment—and two groups—one that remains untreated in both periods and one that becomes treated in the second period. In our Medicaid example, we focus on comparisons in 2014 and 2013 between the 2014 expansion group (978 counties) and the group that had not expanded by 2019 (1,222 counties). When we consider more complex designs, this kind of comparison will still play a role: it will be one 2 × 2 “building block” among many.

Using these basic ingredients, we can now define a 2×2 DiD design, composed of a causal target parameter, a treatment variable, an assumption under which it is identified, and an estimation approach, which will be the classic difference of two differences. This may be familiar territory in the simple case, but it is a crucial framework for building up appropriate techniques in more

[^2]: It is common to adjust mortality rates by the county age distribution. Unfortunately, the CDC measurement of age-specific deaths are restricted for counties with fewer than 10 annual deaths. We aim to use publicly available and shareable data for pedagogical purposes; we follow Borgholt and Vogler (2020) and use the crude mortality rate.