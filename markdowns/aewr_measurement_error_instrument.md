# Instrumental Variable Design: FLS Preliminary-to-Final Revision as Survey Measurement Error

---

## 1. Motivation

The Adverse Effect Wage Rate (AEWR) for H-2A agricultural workers is set annually by the U.S. Department of Labor (DOL) using wage data from the USDA National Agricultural Statistics Service (NASS) Farm Labor Survey (FLS). Under the 2010 Final Rule — the regulatory regime governing our study period of 2008–2023 — DOL sets each region's AEWR equal to the annual weighted average gross hourly wage for field and livestock workers combined, as published in the **November FLS report** for the prior year. That is:

$$\text{AEWR}_{r,t} = \hat{W}^{\text{Nov}}_{r,t-1}$$

where $\hat{W}^{\text{Nov}}_{r,t-1}$ is the preliminary annual average wage for FLS region $r$ published in the November report of year $t-1$.

A key institutional feature of the FLS is that these November estimates are **preliminary**. NASS formally revises them in two subsequent rounds: a first revision published in the May report six months later, and a final revision published in the following November report twelve months later. Revised values replace the preliminary estimates in NASS's current data systems; the preliminary figures survive only in the original archived PDF reports as originally published.

This creates a cleanly identified source of exogenous variation: DOL acts on an estimate of the true regional wage that contains measurement error. The subsequent revisions reveal the direction and magnitude of that error. Because the measurement error in a given November's preliminary estimate is driven by late-arriving survey responses, nonresponse weighting adjustments, and stratum-level data quality issues — none of which are related to labor market conditions in any specific commuting zone — the preliminary estimate is a valid instrument for the AEWR bite in the IV framework described below.

---

## 2. The FLS Revision Process

NASS conducts the FLS semi-annually, collecting data in April (covering January and April reference weeks) and October (covering July and October reference weeks). The November FLS report, released in mid-to-late November each year, contains two types of estimates:

- **Quarterly estimates** for the July and October reference weeks of that year, published for the first time
- **Annual average estimates** for the full calendar year, computed as an employment-weighted average across all four quarterly reference weeks

The annual averages in the November report are designated **preliminary** because:

1. The October reference week data — collected only weeks before the November release — has had limited time for editing, quality review, and nonresponse follow-up
2. Late-responding farm operators whose data arrives after the November release deadline are excluded from the preliminary estimate and incorporated in subsequent revisions
3. The weighting and calibration procedures that adjust for differential nonresponse across farm size strata are applied iteratively; only the first-pass adjustment is used in the preliminary release

NASS's published revision policy states that preliminary estimates are subject to a first revision six months after initial publication, and a final revision twelve months after initial publication. Revised values are published in the subsequent Farm Labor reports and replace the preliminary figures in NASS Quick Stats. The preliminary figures are preserved only in the original archived PDF reports held by the USDA National Agricultural Library's Economics, Statistics, and Market Information System (ESMIS), accessible at `esmis.nal.usda.gov`.

Because DOL uses the preliminary November figure — not the subsequently revised figure — to set the following year's AEWR, the AEWR embeds whatever measurement error was present in the preliminary estimate. Formally, let $W^*_{r,t}$ be the true annual average agricultural wage in region $r$ in year $t$, and let $W^{\text{final}}_{r,t}$ be NASS's best estimate of this true value after all revisions are complete. Then:

$$\hat{W}^{\text{Nov}}_{r,t} = W^{\text{final}}_{r,t} + \nu_{r,t}$$

where $\nu_{r,t}$ is the revision — the difference between the preliminary and final estimate — which represents the component of the preliminary figure that was subsequently corrected. The AEWR DOL sets for year $t+1$ therefore contains $\nu_{r,t}$ as a component that is unrelated to the true wage level.

---

## 3. Instrument Design

### 3.1 The Endogenous Variable

Our endogenous variable is the **AEWR bite** in commuting zone $i$, FLS region $r(i)$, in year $t$:

$$\text{Bite}_{it} = \text{AEWR}_{r(i),t} - w^{10}_{it}$$

where $w^{10}_{it}$ is the 10th percentile hourly wage of agricultural workers in CZ $i$ estimated from the American Community Survey (ACS). The bite is higher in CZs where the regional wage floor substantially exceeds the local low end of the agricultural wage distribution — that is, where the floor is most likely to bind.

The endogeneity of the bite arises because local labor market conditions affect $w^{10}_{it}$ directly. Additionally, if local agricultural wages contribute to the FLS regional average — as they do, mechanically — then local conditions also affect $\text{AEWR}_{r(i),t}$ through the aggregation of the FLS. An instrument is needed that generates variation in the AEWR independent of local labor market conditions.

### 3.2 The Instrument

Our instrument exploits the fact that DOL sets the AEWR using the **preliminary** November FLS estimate rather than the subsequently revised figure. Define:

$$\nu_{r,t-1} \equiv \hat{W}^{\text{Nov}}_{r,t-1} - W^{\text{final}}_{r,t-1}$$

as the revision to region $r$'s annual average wage in year $t-1$ — positive if the preliminary estimate overstated the true wage, negative if it understated it.

Because $\text{AEWR}_{r,t} = \hat{W}^{\text{Nov}}_{r,t-1} = W^{\text{final}}_{r,t-1} + \nu_{r,t-1}$, the revision directly enters the AEWR as a component that is orthogonal to local labor market fundamentals. The instrument for the bite in CZ $i$ in year $t$ is:

$$Z_{it} = \nu_{r(i),t-1} = \hat{W}^{\text{Nov}}_{r(i),t-1} - W^{\text{final}}_{r(i),t-1}$$

Equivalently, since $\hat{W}^{\text{Nov}}_{r,t-1}$ is the actual AEWR set by DOL (a known quantity), and $W^{\text{final}}_{r,t-1}$ is the final revised estimate (recoverable from current NASS Quick Stats), the revision is directly computable from public data once the preliminary estimates are recovered from the archived PDF reports.

The first-stage regression is:

$$\text{Bite}_{it} = \alpha + \beta \nu_{r(i),t-1} + \mathbf{X}_{it}'\gamma + \delta_i + \lambda_t + \epsilon_{it}$$

where $\delta_i$ are CZ fixed effects, $\lambda_t$ are year fixed effects, and $\mathbf{X}_{it}$ is a vector of controls. The coefficient $\beta$ should be positive: a region where the preliminary estimate overstated the true wage (positive $\nu$) will have a higher AEWR than warranted, increasing the bite in all CZs within that region.

### 3.3 Relevance

The instrument is relevant because the revision $\nu_{r,t-1}$ directly enters the AEWR through DOL's use of the preliminary estimate. Every dollar of upward revision error in the November FLS translates mechanically into a one dollar higher AEWR the following year for all CZs in the region. The first-stage coefficient should therefore be close to one, and the F-statistic should be strong in regions and years where revisions are large.

First-stage power varies across regions and years with the magnitude of $|\nu_{r,t}|$. Regions with lower survey response rates, smaller sample sizes, or more volatile agricultural wage distributions will tend to have larger revisions and thus more first-stage variation. We document the distribution of $|\nu_{r,t}|$ across regions and years as part of the data description, and report first-stage F-statistics separately for high- and low-revision region-year cells.

### 3.4 Exclusion Restriction

The exclusion restriction requires that $\nu_{r,t-1}$ — the revision to the FLS preliminary estimate — affects labor market outcomes in CZ $i$ only through the AEWR channel, and not through any direct channel.

This assumption is supported by the institutional nature of the revision process. Revisions arise from:

- **Late survey returns**: farm operators who submitted their FLS questionnaires after the November editorial deadline are excluded from the preliminary estimate and incorporated in subsequent revisions. Whether a particular farm operator mails their survey in November versus January is plausibly unrelated to labor demand in any specific CZ.
- **Nonresponse weighting adjustments**: when response rates are lower than expected in a given stratum, NASS upweights the responding farms. If the initial nonresponse weights are imprecise and later corrected, the revision reflects this reweighting rather than any change in underlying wages. The imprecision of initial nonresponse weights is driven by administrative factors within NASS, not by local labor market conditions.
- **Data editing corrections**: NASS edits individual farm responses for consistency and plausibility. Errors discovered after the initial editorial pass are corrected in subsequent revisions. These corrections reflect data quality issues in individual survey responses, not local economic conditions.

None of these revision sources are plausibly correlated with labor market conditions in individual CZs, particularly after conditioning on CZ and year fixed effects that absorb time-invariant CZ characteristics and common year-specific shocks.

One potential concern is that the revision $\nu_{r,t-1}$ is correlated with the true wage change $\Delta W^{\text{final}}_{r,t-1}$ — for instance, if NASS systematically underestimates wages in years when they are rising rapidly (because late responders tend to have higher wages). This would create a correlation between the instrument and the error in the structural equation if rising regional wages also affect local CZ outcomes through non-AEWR channels. We address this by:

1. Including region-specific time trends to control for systematic patterns in regional wage growth
2. Documenting the correlation between $\nu_{r,t-1}$ and $\Delta W^{\text{final}}_{r,t-1}$ directly as an empirical test of this concern
3. Running the falsification test described in Section 5 below

---

## 4. Inference: Clustering Standard Errors

The measurement error instrument generates a specific and non-trivial inference problem. The instrument $Z_{it} = \nu_{r(i),t-1}$ is **constant within FLS region and year**: every CZ in region $r$ in year $t$ receives an identical instrument value. The endogenous variable (bite) and the outcome vary at the CZ-year level. This mismatch between the level of variation in the instrument and the level of observation creates problems for standard error estimation that the current econometrics literature addresses directly and with specific guidance.

### 4.1 The Moulton Problem

The foundational issue is classical: Moulton (1986, 1990) showed that when a regressor varies only at an aggregate level (here, region-year) but the regression is estimated at a finer level (CZ-year), OLS standard errors are downward-biased because within-region-year residuals are mechanically correlated through the shared regressor. The bias factor depends on the intraclass correlation of the outcome and the number of observations per cluster; with large FLS regions containing many CZs, this bias can be severe.

Naively computing heteroskedasticity-robust standard errors — or even CZ-level clustered standard errors — will not resolve this problem. CZ-level clustering accounts for serial correlation within a CZ over time, but it does not account for the cross-CZ within-region correlation induced by the region-year-level instrument. The correct response is to cluster at the level at which the instrument varies, which is the **FLS region**.

### 4.2 The Design-Based Rationale for Region-Level Clustering

Abadie, Athey, Imbens, and Wooldridge (2023, *QJE*) reframe the clustering decision as a design problem rather than a model problem. From their perspective, clustering is warranted when treatment (or instrument) assignment is correlated within clusters — an experimental design issue. Here, the instrument is assigned at the region-year level: all CZs in a region receive the same $\nu_{r,t-1}$. This is precisely the design-based scenario Abadie et al. describe, and it provides a clean justification for clustering at the FLS region level that does not depend on assumptions about unobserved common shocks.

Importantly, Abadie et al. also note that when the sample includes all (or nearly all) clusters in the relevant population — as is the case here, since we observe all 17 CONUS FLS regions — conventional clustered standard errors may actually be *upward* biased relative to the true finite-population variance. This is a useful counterweight to the Moulton concern: with all 17 regions observed, there is less sampling uncertainty at the region level than the CRVE formula assumes. In practice, this finite-population correction is rarely implemented, but it is worth acknowledging in the paper that our inference may be conservative in this respect.

### 4.3 The Nesting Structure and Multi-Way Clustering

The geographic hierarchy of this paper is: CZ $\subset$ state $\subset$ FLS region. Because CZs are split at FLS region boundaries (which coincide with state lines), each CZ-portion belongs to exactly one state, and each state belongs to exactly one FLS region. The clustering levels are strictly nested.

This nesting has an important implication for multi-way clustering. The Cameron, Gelbach, and Miller (2011) two-way clustering variance estimator is defined as:

$$\hat{V}_{\text{two-way}} = \hat{V}_{\text{CZ}} + \hat{V}_{\text{region}} - \hat{V}_{\text{CZ} \cap \text{region}}$$

Because CZ is nested within region, the intersection CZ $\cap$ region = CZ, so:

$$\hat{V}_{\text{two-way}} = \hat{V}_{\text{CZ}} + \hat{V}_{\text{region}} - \hat{V}_{\text{CZ}} = \hat{V}_{\text{region}}$$

Two-way clustering on CZ and FLS region therefore **collapses to region-level clustering** when CZs are nested within regions. There is no additional information gained from simultaneously clustering at the CZ level; region-level clustering already absorbs all within-region cross-CZ correlation, including whatever serial correlation exists within individual CZs. The practical upshot is that region-level clustering is both necessary and sufficient, and two-way clustering provides no correction beyond it.

### 4.4 The Small-Cluster Problem: 17 Regions

The most serious inferential challenge is that clustering at the FLS region level yields only **17 clusters**. The conventional cluster-robust variance estimator (CRVE, sometimes called CV1) is known to be unreliable with few clusters. Cameron and Miller (2015) survey simulation evidence suggesting that 50 clusters are generally needed for the asymptotic approximation underlying the CRVE $t$-distribution to be accurate; with fewer clusters, the CRVE can produce severely undersized standard errors and over-rejection of true null hypotheses.

MacKinnon, Nielsen, and Webb (2023, *Journal of Econometrics*, "Cluster-Robust Inference: A Guide to Empirical Practice") provide the authoritative current treatment of this problem. Their key recommendations for the small-cluster setting are:

1. **Do not rely on the CRVE with the $t$-distribution.** With 17 clusters, the asymptotic approximation is poor and conventional CRVE $t$-tests will over-reject.

2. **Use bias-corrected CRVEs.** The CV3 estimator (a jackknife-based bias correction, discussed in MacKinnon, Nielsen, and Webb, 2023, *JAE*) performs substantially better than CV1 in small-cluster settings. The improvement is particularly notable when cluster sizes are heterogeneous, as they will be here — FLS regions vary substantially in the number of CZs they contain.

3. **Use the wild cluster bootstrap for hypothesis testing.** The wild cluster bootstrap (WCB), introduced by Cameron, Gelbach, and Miller (2008), provides asymptotic refinements over the CRVE $t$-test by generating a bootstrap distribution under the null. The **restricted wild cluster bootstrap (WCR)**, which imposes the null hypothesis when generating bootstrap data, is preferred for hypothesis testing because it more accurately controls size. MacKinnon, Nielsen, and Webb (2023, *JAE*) refer to this as the "31" bootstrap (CV3-studentized, restricted), which their simulations show performs best across a range of small-cluster settings.

4. **Use Webb weights rather than Rademacher weights when the number of clusters is very small.** Webb (2023, *Canadian Journal of Economics*) shows that with fewer than roughly 12 clusters, Rademacher weights ($\pm 1$ with equal probability) can produce bootstrap distributions with poor properties because there are too few possible weight assignments. Webb weights (a six-point distribution) provide better coverage. With 17 clusters, Rademacher weights are likely adequate but Webb weights are a conservative choice worth reporting.

### 4.5 The IV-Specific Complication

The small-cluster problem is compounded in the IV setting. Wang and Zhang (2022, working paper, "Wild Bootstrap for Instrumental Variables Regressions with Weak and Few Clusters") study the wild cluster bootstrap specifically for IV models with few clusters. Their key result is that the wild bootstrap Wald test for the IV estimand controls size asymptotically — up to a small error — as long as the instrument is strongly identified in at least one cluster. The Anderson-Rubin (AR) test, which is robust to weak identification, also controls size under the wild cluster bootstrap even when identification is weak across all clusters.

Given that first-stage strength will vary across FLS regions (depending on the magnitude of revisions in each region), we recommend reporting Anderson-Rubin confidence sets alongside the Wald-based IV results. The AR test is simultaneously robust to weak instruments and valid under wild cluster bootstrap inference with few clusters, making it a natural complement to the main 2SLS results.

Young (2022) provides additional motivation for robustness: in a survey of published IV papers, dropping a single cluster or observation frequently renders reported results insignificant. With 17 clusters, leave-one-region-out sensitivity analysis — re-estimating the IV specification dropping each FLS region in turn — is a straightforward and credible robustness exercise that should be reported.

### 4.6 A Practical Alternative: State-Level Clustering

A practical alternative that increases the number of clusters is to cluster at the **state level**, which yields approximately 48 clusters for the CONUS sample. Since states are nested within FLS regions, state-level clustering does not fully account for the across-state within-region correlation induced by the instrument — it is therefore somewhat anticonservative relative to region-level clustering for the Moulton problem. However, with 48 clusters the asymptotic approximation underlying the CRVE is substantially more reliable, and the standard CRVE with the $t$-distribution (or the wild bootstrap) performs much better.

We recommend reporting state-level clustered standard errors alongside region-level bootstrapped confidence intervals as a robustness comparison. If results are qualitatively similar at both levels of clustering, this is reassuring. If state-level clustering yields substantially smaller standard errors than region-level bootstrapping, the discrepancy should be discussed explicitly as evidence that within-region cross-state correlation is empirically relevant for inference.

### 4.7 Summary of Recommended Inference Procedure

In order of priority:

| Method | Rationale | Implementation |
|---|---|---|
| WCR bootstrap ("31"), cluster at FLS region | Preferred for hypothesis testing with 17 clusters; controls size in small-cluster IV (Wang and Zhang 2022) | Stata: `boottest`; R: `fwildclusterboot` |
| CV3 CRVE, cluster at FLS region | Bias-corrected point estimate of the CRVE; preferred over CV1 | Stata: `vce(hc3 region)`; R: `clubSandwich` |
| Anderson-Rubin confidence sets | Robust to weak identification; valid under WCB with few clusters | Stata: `weakiv`; R: `ivmodel` |
| State-level CRVE (CV1 or CV3) | Robustness check; more clusters but anticonservative for Moulton problem | Standard clustering commands |
| Leave-one-region-out sensitivity | Detects leverage of individual regions on IV estimates (Young 2022) | Re-estimate dropping each of 17 regions |

---

## 5. Falsification Test

Test whether $\nu_{r,t-1}$ predicts labor market outcomes for non-agricultural workers in CZ $i$. Specifically, regress the 10th percentile wage of non-agricultural low-wage workers in CZ $i$ on $\nu_{r,t-1}$, controlling for CZ and year fixed effects. A near-zero coefficient is evidence that survey measurement noise in the FLS is not proxying for regional economic conditions that independently affect the CZ's labor market.

Additionally, test whether $\nu_{r,t-1}$ predicts any pre-period (pre-2010) outcome trends in CZs, which would indicate the revision is correlated with pre-existing local conditions rather than being purely administrative noise.

All falsification tests should be conducted using the same inference procedure described in Section 4 — in particular, the wild cluster bootstrap at the region level — to maintain consistency with the main results.

---

## 6. Limitations and Caveats

**The 2020 gap.** The FLS suspension in September 2020 means there is no November 2020 report and therefore no preliminary estimate for the 2020 annual average that can be compared to a final revised figure. The February 2021 delayed release set the 2022 AEWR, but its status as a preliminary-vs-revised document is ambiguous. We recommend treating the 2020–2021 revision cycle as missing and including a year indicator for 2022 (the affected AEWR year) as a robustness check.

**The December 2013 and November 2006 "revision" releases.** The ESMIS archive labels these reports with `_revision` in their filenames, indicating the archived document is itself a revised release that superseded an earlier preliminary release. For these years, the instrument requires recovering the *original* preliminary figure rather than the labeled revision. If the original preliminary PDFs are not separately archived, these years may need to be excluded from the measurement error instrument or treated as having a zero revision (a conservative assumption that biases toward zero first-stage variation in those years).

**Magnitude of revisions.** This instrument is only powerful in region-years where $|\nu_{r,t}|$ is economically meaningful. If revisions are systematically small across the study period — say, consistently below $0.10 per hour — the instrument will have limited first-stage power. The descriptive analysis of the revision series is therefore a prerequisite for deciding whether to proceed with this instrument.

**Power with 17 clusters.** The wild cluster bootstrap controls size but does not guarantee power. With only 17 clusters, the bootstrap distribution may have limited support, particularly when using Rademacher weights (which allow only $2^{17} = 131,072$ distinct bootstrap draws — sufficient but not unlimited). Webb weights, which produce a continuous draw, may provide modestly better power. Regardless, the paper should acknowledge that inference with 17 clusters is inherently less precise than a design with many clusters, and that results should be interpreted accordingly.

---

## 7. Data Requirements and Construction

### Step 1: Recover Preliminary FLS Annual Average Wages from Archived PDF Reports

**Data source:** USDA NASS FLS November reports as originally published, archived at the USDA National Agricultural Library ESMIS system (`esmis.nal.usda.gov/publication/farm-labor`)

**Required reports and direct PDF links:**

| FLS Year | Release Date | AEWR Year Set | PDF Link |
|---|---|---|---|
| 2007 | Nov 16, 2007 | 2008 | [FarmLabo-11-16-2007.pdf](https://esmis.nal.usda.gov/sites/default/release-files/x920fw89s/b8515p953/0v838198x/FarmLabo-11-16-2007.pdf) |
| 2008 | Nov 21, 2008 | 2009 | [FarmLabo-11-21-2008.pdf](https://esmis.nal.usda.gov/sites/default/release-files/x920fw89s/bc386k99q/1j92g916h/FarmLabo-11-21-2008.pdf) |
| 2009 | Nov 20, 2009 | 2010 | [FarmLabo-11-20-2009.pdf](https://esmis.nal.usda.gov/sites/default/release-files/x920fw89s/bk128c597/r494vm600/FarmLabo-11-20-2009.pdf) |
| 2010 | Nov 18, 2010 | 2011 | [FarmLabo-11-18-2010.pdf](https://esmis.nal.usda.gov/sites/default/release-files/x920fw89s/j6731567p/w37638501/FarmLabo-11-18-2010.pdf) |
| 2011 | Nov 17, 2011 | 2012 | [FarmLabo-11-17-2011.pdf](https://esmis.nal.usda.gov/sites/default/release-files/x920fw89s/vm40xt37s/6395w854g/FarmLabo-11-17-2011.pdf) |
| 2012 | Nov 19, 2012 | 2013 | [FarmLabo-11-19-2012.pdf](https://esmis.nal.usda.gov/sites/default/release-files/x920fw89s/02870x617/jw827d471/FarmLabo-11-19-2012.pdf) |
| 2013 | Dec 5, 2013 | 2014 | [FarmLabo-12-05-2013_revision.pdf](https://esmis.nal.usda.gov/sites/default/release-files/x920fw89s/7s75df000/xd07gv37z/FarmLabo-12-05-2013_revision.pdf) |
| 2014 | Nov 20, 2014 | 2015 | [FarmLabo-11-20-2014.pdf](https://esmis.nal.usda.gov/sites/default/release-files/x920fw89s/gm80hx185/nc580p20n/FarmLabo-11-20-2014.pdf) |
| 2015 | Nov 19, 2015 | 2016 | [FarmLabo-11-19-2015.pdf](https://esmis.nal.usda.gov/sites/default/release-files/x920fw89s/pg15bg444/9w0324494/FarmLabo-11-19-2015.pdf) |
| 2016 | Nov 17, 2016 | 2017 | [FarmLabo-11-17-2016.pdf](https://esmis.nal.usda.gov/sites/default/release-files/x920fw89s/gh93h108v/c534fq60m/FarmLabo-11-17-2016.pdf) |
| 2017 | Nov 16, 2017 | 2018 | [FarmLabo-11-16-2017.pdf](https://esmis.nal.usda.gov/sites/default/release-files/x920fw89s/m613n0170/db78td76w/FarmLabo-11-16-2017.pdf) |
| 2018 | Nov 15, 2018 | 2019 | [fmla1118.pdf](https://esmis.nal.usda.gov/sites/default/release-files/x920fw89s/9g54xm59d/j96024106/fmla1118.pdf) |
| 2019 | Nov 21, 2019 | 2020 | [fmla1119.pdf](https://esmis.nal.usda.gov/sites/default/release-files/x920fw89s/c821h164m/fq9788943/fmla1119.pdf) |
| 2021 | Feb 11, 2021* | 2022 | [fmla0221.pdf](https://esmis.nal.usda.gov/sites/default/release-files/x920fw89s/f7624565c/9k420769j/fmla0221.pdf) |
| 2022 | Nov 23, 2022 | 2023 | [fmla1122.pdf](https://esmis.nal.usda.gov/sites/default/release-files/x920fw89s/pv63h9083/gq67m157z/fmla1122.pdf) |
| 2023 | Nov 22, 2023 | 2024 | [fmla1123.pdf](https://esmis.nal.usda.gov/sites/default/release-files/x920fw89s/v405tw18s/dn39zk84n/fmla1123.pdf) |

*The FLS was suspended in September 2020 under the first Trump administration. No November 2020 report was published. The delayed report covering 2020 annual averages was released in February 2021. The 2020–2021 revision cycle requires special handling; see Section 6.

**Extraction procedure:** From each November report, extract the table titled "Annual Average Gross Wage Rates by Type of Worker — Regions and United States" (or equivalent). The relevant row is "Field and livestock combined" and the relevant columns are the 17 CONUS FLS regions. The `.txt` versions of each report (available at the same ESMIS URLs with `.txt` substituted for `.pdf`) contain identical data in fixed-width format and are substantially easier to parse programmatically than the PDFs.

---

### Step 2: Obtain Final Revised FLS Annual Average Wages

**Data source:** NASS Quick Stats database (`quickstats.nass.usda.gov`) or API

**Query parameters:**
- Program: Survey
- Sector: Labor
- Commodity: Workers, Field & Livestock
- Data Item: `WORKERS, FIELD & LIVESTOCK - WAGE RATE, MEASURED IN $ / HOUR`
- Domain: Total
- Geographic Level: Region (multi-state) and State (for California and Florida)
- Period: YEAR (annual average)
- Years: 2007–2023

Quick Stats returns the most recently revised estimate for each region-year cell. Since all revisions within the study period have now been completed, these represent the final figures $W^{\text{final}}_{r,t}$ for each region and year.

---

### Step 3: Construct the Revision Series

For each FLS region $r$ and year $t$, compute:

$$\nu_{r,t} = \hat{W}^{\text{Nov}}_{r,t} - W^{\text{final}}_{r,t}$$

This produces a panel of revisions across 17 CONUS regions and years 2007–2023 (with the 2020 gap noted above).

**Descriptive analysis of the revision series:** Document the distribution of $|\nu_{r,t}|$; test whether revisions are systematically correlated with regional wage growth $\Delta W^{\text{final}}_{r,t}$; identify any outlier region-year revision cells. This analysis informs both the plausibility of the exclusion restriction and the expected first-stage power.

---

### Step 4: Construct the Instrument and First Stage

For each CZ $i$ in region $r(i)$ and year $t$, the instrument is $Z_{it} = \nu_{r(i),t-1}$.

The endogenous variable is $\text{Bite}_{it} = \hat{W}^{\text{Nov}}_{r(i),t-1} - w^{10}_{it}$, where the AEWR is directly equal to the preliminary November estimate and can be verified against DOL Federal Register notices (available at `federalregister.gov`).

Estimate the first stage by OLS and report the first-stage F-statistic using wild cluster bootstrap inference at the FLS region level, as described in Section 4.

---

## 8. Summary of Data Sources

| Data Source | Provider | Access | Purpose |
|---|---|---|---|
| FLS November reports (archived), 2007–2023 | USDA NASS via ESMIS/NAL | Public; text extraction | Preliminary annual average wages $\hat{W}^{\text{Nov}}_{r,t}$ |
| NASS Quick Stats, farm labor annual average, 2007–2023 | USDA NASS | Public; direct download | Final revised annual average wages $W^{\text{final}}_{r,t}$ |
| Federal Register AEWR notices, 2008–2024 | federalregister.gov | Public | Verification of AEWR values and effective dates |
| ACS 1-year PUMS, 2008–2023 | IPUMS USA | Public | 10th percentile agricultural wage $w^{10}_{it}$; falsification outcome |

---

## 9. Key References

Abadie, A., Athey, S., Imbens, G. W., and Wooldridge, J. M. (2023). When should you adjust standard errors for clustering? *Quarterly Journal of Economics*, 138(1), 1–35.

Cameron, A. C., Gelbach, J. B., and Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics*, 90(3), 414–427.

Cameron, A. C., Gelbach, J. B., and Miller, D. L. (2011). Robust inference with multiway clustering. *Journal of Business and Economic Statistics*, 29(2), 238–249.

Cameron, A. C., and Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317–372.

MacKinnon, J. G., Nielsen, M. Ø., and Webb, M. D. (2023a). Cluster-robust inference: A guide to empirical practice. *Journal of Econometrics*, 232(2), 272–299.

MacKinnon, J. G., Nielsen, M. Ø., and Webb, M. D. (2023b). Fast and reliable jackknife and bootstrap methods for cluster-robust inference. *Journal of Applied Econometrics*, 38(5), 671–694.

MacKinnon, J. G., and Webb, M. D. (2017). Wild bootstrap inference for wildly different cluster sizes. *Journal of Applied Econometrics*, 32(2), 233–254.

MacKinnon, J. G., and Webb, M. D. (2018). The wild bootstrap for few (treated) clusters. *Econometrics Journal*, 21(2), 114–135.

Moulton, B. R. (1986). Random group effects and the precision of regression estimates. *Journal of Econometrics*, 32(3), 385–397.

Moulton, B. R. (1990). An illustration of a pitfall in estimating the effects of aggregate variables on micro units. *Review of Economics and Statistics*, 72(2), 334–338.

Roodman, D., MacKinnon, J. G., Nielsen, M. Ø., and Webb, M. D. (2019). Fast and wild: Bootstrap inference in Stata using boottest. *Stata Journal*, 19(1), 4–60.

Wang, W., and Zhang, Y. (2022). Wild bootstrap for instrumental variables regressions with weak and few clusters. Working paper.

Webb, M. D. (2023). Reworking wild bootstrap-based inference for clustered errors. *Canadian Journal of Economics*, 56(3), 839–858.

Young, A. (2022). Consistency without inference: Instrumental variables in practical application. *European Economic Review*, 147, 104–155.
