# Instrumental Variable Design: Agro-Dissimilarity Leave-Cluster-Out AEWR Bite

---

## 1. Motivation and Setting

The Adverse Effect Wage Rate (AEWR) is the federally mandated minimum wage for H-2A agricultural guestworkers, set annually by the U.S. Department of Labor using Farm Labor Survey (FLS) data collected by USDA's National Agricultural Statistics Service (NASS). Under the 2010 Final Rule — the regulatory regime governing our study period of 2008–2023 — the AEWR is set at the level of one of 17 contiguous U.S. FLS regions, each comprising multiple states or, in the cases of California and Florida, a single state. All commuting zones (CZs) within a given FLS region face the identical AEWR regardless of local agricultural conditions.

Our outcome of interest is the **AEWR bite** in commuting zone *i*, defined as:

$$\text{Bite}_{it} = \text{AEWR}_{r(i),t} - w^{10}_{it}$$

where $r(i)$ denotes the FLS region containing CZ *i*, $\text{AEWR}_{r,t}$ is the AEWR for region *r* in year *t*, and $w^{10}_{it}$ is the 10th percentile hourly wage of agricultural workers in CZ *i* in year *t*, estimated from the American Community Survey (ACS). The bite measures how far the regional wage floor exceeds the local low-wage agricultural labor market. It is higher in CZs where local agricultural wages are low relative to the regional AEWR — precisely the CZs where the floor is most likely to bind.

The bite is endogenous. Local labor market conditions — unobserved demand shocks, industry restructuring, enforcement intensity — affect both the local agricultural wage $w^{10}_{it}$ and, through the FLS aggregation mechanism, the AEWR itself, since local wages contribute to the regional average. A credible instrument must generate variation in the AEWR bite in CZ *i* that originates outside CZ *i*'s own labor market, while remaining relevant — that is, strongly predictive of the actual bite CZ *i* faces.

---

## 2. Instrument Design

### 2.1 Agricultural Similarity Clustering

Within each FLS region *r*, we partition the constituent CZs into agro-ecological subregions using a pre-determined dissimilarity measure constructed from two sets of primitives, both fixed prior to the study period:

1. **Soil and climate characteristics** — [to be specified by author]
2. **Baseline crop composition** — the share of cropland devoted to each major crop type in each CZ, taken from the USDA Cropland Data Layer (CDL) for 2008, the first year of the study period and prior to any AEWR treatment variation of interest.

Using these primitives, we apply a clustering algorithm within each FLS region, requiring that every region contains at least two distinct agro-ecological subregions, denoted $a \in \{1, \ldots, A_r\}$ with $A_r \geq 2$ for all *r*. Cluster assignment is fixed for the entire study period. The use of predetermined physical and agronomic characteristics ensures that cluster membership is exogenous to subsequent labor market developments.

The intuition for the dissimilarity requirement is substantive: CZs assigned to different agro-ecological subregions within the same FLS region grow different crops in different climates. Their agricultural wages are therefore driven by different commodity price cycles, different weather realizations, and different seasonal labor demand patterns. Despite this, they share the same regulatory wage floor — because the AEWR is set at the FLS region level, aggregating across all subregions. Wage shocks originating in agro-dissimilar subregions thus transmit to CZ *i*'s AEWR without sharing CZ *i*'s local agricultural labor market conditions.

### 2.2 Instrument Construction

Let $a(i)$ denote the agro-ecological subregion containing CZ *i*, within FLS region $r(i)$. The instrument for CZ *i* in year *t* is the employment-weighted average ACS agricultural wage in year $t-1$ across all CZs *j* that satisfy two conditions: (1) *j* is in the same FLS region as *i*, and (2) *j* is in a different agro-ecological subregion from *i*:

$$Z_{it} = \frac{1}{\sum_{j \in \mathcal{J}_{-a(i),r(i)}} n_{jt-1}} \sum_{j \in \mathcal{J}_{-a(i),r(i)}} n_{jt-1} \cdot \bar{w}^{\text{ag}}_{jt-1}$$

where:

- $\mathcal{J}_{-a(i),r(i)}$ is the set of all CZs in FLS region $r(i)$ that are **not** in subregion $a(i)$
- $\bar{w}^{\text{ag}}_{jt-1}$ is the ACS-estimated mean agricultural wage in CZ *j* in year $t-1$
- $n_{jt-1}$ is the ACS-estimated count of agricultural workers in CZ *j* in year $t-1$, used as employment weights

The one-year lag reflects the AEWR-setting mechanism: the FLS annual average from year $t-1$ (published in the November FLS report) becomes the AEWR effective in year *t*. The instrument therefore captures, for each CZ *i*, the wage signal from agro-dissimilar parts of the same FLS region that fed into the determination of the AEWR CZ *i* faces in year *t*, while excluding the wage signal from CZ *i*'s own agro-ecological neighborhood.

The first stage of the IV is:

$$\text{Bite}_{it} = \alpha + \beta Z_{it} + \mathbf{X}_{it}'\gamma + \delta_i + \lambda_t + \epsilon_{it}$$

where $\delta_i$ are CZ fixed effects, $\lambda_t$ are year fixed effects, and $\mathbf{X}_{it}$ is a vector of controls. The instrument is relevant because $Z_{it}$ tracks the actual AEWR through the FLS aggregation mechanism. The exclusion restriction — discussed in Section 3 — requires that agricultural wage shocks in agro-dissimilar subregions affect CZ *i*'s labor market outcomes only through the AEWR channel.

---

## 3. Identification Assumptions and Empirical Validation

### 3.1 Relevance

The instrument is relevant by construction: the AEWR in region *r* is a weighted average of agricultural wages across all CZs in the region, including those in $\mathcal{J}_{-a(i),r(i)}$. Wage movements in agro-dissimilar subregions mechanically shift the AEWR that CZ *i* faces, generating first-stage variation. We report first-stage F-statistics by FLS region and flag regions where the instrument is weak due to a small number of contributing CZs.

### 3.2 Exclusion Restriction

The exclusion restriction requires that $Z_{it}$ affects labor market outcomes in CZ *i* only through the AEWR bite, and not through any direct channel. We argue for this on three grounds.

**Agricultural dissimilarity limits commodity-market co-movement.** By design, the contributing CZs in $\mathcal{J}_{-a(i),r(i)}$ grow different crops under different soil and climate conditions. Their wage dynamics are therefore driven by different commodity price cycles. A weather shock, input cost movement, or commodity price change that raises wages in one agro-ecological subregion is unlikely to affect labor demand in a CZ growing fundamentally different crops.

**Agro-dissimilarity limits labor supply spillovers.** Agricultural labor mobility within an FLS region is most likely to operate within agro-similar areas, where workers follow the same crops through the season. By excluding agro-similar CZs, our instrument minimizes the cross-CZ labor supply channel through which wages in other parts of the region could affect CZ *i* directly. We validate this empirically in Section 3.3.

**FLS region boundaries are administratively fixed.** The FLS regions were drawn for statistical convenience and have not changed in response to local economic conditions. Membership in a given FLS region is therefore exogenous to local labor market outcomes, and the instrument inherits this exogeneity.

### 3.3 Empirical Validation of the Exclusion Restriction

We conduct two empirical exercises to discipline the exclusion restriction.

**Exercise 1: Cross-cluster agricultural labor mobility test (H-2A disclosure data).** Using DOL OFLC case disclosure data, we construct, for each pair of agro-ecological subregions within each FLS region, the share of H-2A job certifications involving the same farm labor contractor or employer association operating in both subregions. Low overlap is direct evidence that the formal agricultural labor supply channel between agro-dissimilar subregions is weak. We additionally test whether the same H-2A employers post certifications in both the target CZ's subregion and the instrument subregions, which would indicate shared labor pools inconsistent with the exclusion restriction.

**Exercise 2: ACS-based migration flow test.** Using the ACS one-year migration question, we construct bilateral agricultural worker flows between CZ pairs within each FLS region. We estimate whether flows between agro-dissimilar subregion pairs are significantly lower than flows between agro-similar subregion pairs, conditional on geographic distance. If our clustering variable predicts the actual pattern of agricultural labor mobility — with workers moving predominantly within agro-similar clusters — this provides empirical support for the assumption that agro-dissimilar subregions constitute distinct agricultural labor markets.

**Exercise 3: Falsification test.** We test whether the instrument predicts labor market outcomes in non-agricultural sectors — specifically, the 10th percentile wage among non-agricultural low-wage workers in CZ *i*. Under the exclusion restriction, the instrument should affect non-agricultural wages, if at all, only through general equilibrium spillovers from the agricultural sector. A zero or economically small falsification coefficient is evidence against spurious correlation between $Z_{it}$ and unobserved regional economic conditions.

---

## 4. Data Requirements and Construction Steps

### Step 1: Define and Split Commuting Zones at FLS Region Boundaries

**Data:** 
- CZ boundary shapefiles (David Dorn's website, 1990 or 2000 vintage county-based CZ definitions)
- FLS region state membership (NASS Farm Labor Survey documentation)
- County-to-CZ crosswalk

**Actions:**
- Assign each county to its FLS region using state membership
- Identify CZs that straddle an FLS region boundary (i.e., contain counties in two different FLS regions)
- Split straddling CZs at the FLS boundary, assigning each portion to its respective region; use county-level agricultural employment from the Census of Agriculture to allocate the straddling CZ's characteristics proportionally
- Produce a final crosswalk: CZ (or CZ-portion) → FLS region, covering all 17 CONUS regions

---

### Step 2: Construct Agro-Ecological Primitives at the CZ Level

**Data:**
- Soil characteristics — [to be specified by author]
- Climate characteristics — [to be specified by author]
- USDA Cropland Data Layer (CDL), 2008 vintage: 30-meter resolution raster of crop types for the contiguous U.S., available from USDA NASS CropScape

**Actions:**
- Aggregate soil and climate variables to the CZ level (area-weighted or cropland-area-weighted means)
- From the 2008 CDL, compute for each CZ the share of total cropland in each major crop category (e.g., corn, soybeans, wheat, cotton, vegetables, fruits and nuts, hay, pasture)
- Combine soil, climate, and crop share variables into a single CZ-level primitives matrix, standardized to zero mean and unit variance
- All primitives fixed at 2008 values for the entire study period

---

### Step 3: Cluster CZs into Agro-Ecological Subregions Within Each FLS Region

**Data:** Primitives matrix from Step 2

**Actions:**
- Apply a clustering algorithm (e.g., k-medoids or hierarchical clustering with a chosen dissimilarity metric) separately within each of the 17 FLS regions
- Require a minimum of 2 clusters per region ($A_r \geq 2$ for all *r*); choose the number of clusters and dissimilarity threshold to maximize within-region agro-dissimilarity across clusters while maintaining adequate CZ counts in each cluster for first-stage power
- Assign each CZ a subregion label $a(i)$, fixed for the study period
- Document cluster composition and produce robustness checks showing results are stable across alternative clustering specifications (algorithm choice, number of clusters, dissimilarity metric)

---

### Step 4: Construct CZ-Level Agricultural Wages and Employment from the ACS

**Data:**
- ACS 1-year PUMS microdata, 2008–2023 (person and household files), from IPUMS USA
- ACS PUMS geographic crosswalk to PUMAs; PUMA-to-CZ crosswalk (available from IPUMS or constructable from Census TIGER shapefiles)

**Actions:**
- Restrict sample to wage and salary workers with positive hours and earnings in the reference year
- Define agricultural workers to mirror the FLS "field and livestock workers combined" category:
  - Industry: NAICS 111 (crop production), 112 (animal production), 1151 (crop support activities), 1152 (animal support activities)
  - Occupation: SOC 45-20xx field and livestock worker group, excluding 45-1011 (first-line supervisors) to match FLS exclusion of supervisory workers
- Compute for each CZ *i* and year *t*:
  - $\bar{w}^{\text{ag}}_{it}$: ACS survey-weighted mean hourly wage for agricultural workers (usual weekly earnings / usual weekly hours)
  - $w^{10}_{it}$: ACS survey-weighted 10th percentile hourly wage for agricultural workers
  - $n_{it}$: ACS survey-weighted count of agricultural workers
- Flag CZ-year cells with fewer than 30 unweighted observations; report results with and without these thin-cell CZs, and consider using 5-year ACS pooled estimates as an alternative for robustness

---

### Step 5: Construct the FLS Regional AEWR from NASS Quick Stats

**Data:**
- NASS Quick Stats database (quickstats.nass.usda.gov) or API

**Actions:**
- Query: Program = Survey; Sector = Labor; Commodity = Workers, Field & Livestock; Data Item = `WORKERS, FIELD & LIVESTOCK - WAGE RATE, MEASURED IN $ / HOUR`; Geographic Level = Region (multi-state) and State (for California and Florida); Period = YEAR (annual average); Years = 2008–2023
- This returns the annual weighted average combined field and livestock worker gross wage rate for each of the 17 CONUS FLS regions in each year — the figure the DOL uses to set the following year's AEWR
- Apply the one-year lag: $\text{AEWR}_{r,t} = \text{FLS annual average}_{r,t-1}$
- Validate against published AEWR Federal Register notices (available at federalregister.gov, search "Adverse Effect Wage Rate") to confirm the lag structure and exact values

---

### Step 6: Construct the Instrument

**Data:** Outputs from Steps 1–5

**Actions:**
- For each CZ *i* in FLS region $r(i)$ and agro-ecological subregion $a(i)$, identify the set $\mathcal{J}_{-a(i),r(i)}$: all CZs in the same FLS region that are **not** in subregion $a(i)$
- Compute the employment-weighted mean agricultural wage across this set in year $t-1$:

$$Z_{it} = \frac{\sum_{j \in \mathcal{J}_{-a(i),r(i)}} n_{jt-1} \cdot \bar{w}^{\text{ag}}_{jt-1}}{\sum_{j \in \mathcal{J}_{-a(i),r(i)}} n_{jt-1}}$$

- Construct the bite: $\text{Bite}_{it} = \text{AEWR}_{r(i),t} - w^{10}_{it}$
- Document the first-stage relationship between $Z_{it}$ and $\text{Bite}_{it}$ by FLS region; flag regions where the instrument is weak

---

### Step 7: Empirical Validation — H-2A Cross-Cluster Labor Mobility Test

**Data:**
- DOL OFLC H-2A Case Disclosure Data, 2008–2023 (available at dol.gov/agencies/eta/foreign-labor/performance)

**Actions:**
- Geocode employer and worksite addresses to CZs
- Assign each H-2A certification to an FLS region and agro-ecological subregion using the crosswalk from Steps 1 and 3
- For each FLS region and year, identify farm labor contractors and employer associations that post certifications in more than one agro-ecological subregion
- Compute the share of certifications involving contractors with cross-subregion operations, separately for within-subregion pairs and cross-subregion pairs
- Test whether the cross-subregion contractor overlap rate is significantly lower than the within-subregion overlap rate; a low cross-subregion rate is evidence against the labor supply spillover concern

---

### Step 8: Empirical Validation — ACS Agricultural Migration Flow Test

**Data:**
- ACS 1-year PUMS microdata, using migration variables (MIGPUMA, MIGSP: state and PUMA of residence one year ago)

**Actions:**
- Restrict to agricultural workers (as defined in Step 4) who report a different residence one year prior
- Construct bilateral flow counts between CZ pairs within each FLS region, using the PUMA-to-CZ crosswalk
- Estimate a gravity-style regression of bilateral flows on: an indicator for whether the pair is within the same agro-ecological subregion, log geographic distance between CZ centroids, and FLS-region fixed effects
- A significantly negative coefficient on the cross-subregion indicator (conditional on distance) supports the argument that agricultural workers move predominantly within agro-similar clusters

---

### Step 9: Falsification Test

**Data:** ACS PUMS (full sample, not restricted to agricultural workers)

**Actions:**
- Compute the 10th percentile hourly wage for non-agricultural low-wage workers in each CZ and year (e.g., retail, food service, janitorial, and other service occupations below the 25th percentile of the national wage distribution)
- Regress this non-agricultural low-wage outcome on the instrument $Z_{it}$, controlling for CZ and year fixed effects
- A near-zero coefficient is evidence that the instrument is not proxying for general regional economic conditions; a non-trivial coefficient signals a potential exclusion restriction violation requiring further investigation

---

## 5. Summary of Data Sources

| Data Source | Provider | Access | Purpose |
|---|---|---|---|
| CZ boundary shapefiles and county crosswalk | David Dorn (dorn.page) | Public | Step 1 |
| FLS region state membership | USDA NASS | Public | Step 1 |
| Soil characteristics | [TBD] | [TBD] | Step 2 |
| Climate characteristics | [TBD] | [TBD] | Step 2 |
| Cropland Data Layer 2008 | USDA NASS CropScape | Public | Step 2 |
| ACS 1-year PUMS microdata 2008–2023 | IPUMS USA | Public | Steps 4, 8, 9 |
| PUMA-to-CZ crosswalk | IPUMS / Census TIGER | Public | Steps 4, 8, 9 |
| NASS Quick Stats farm labor wage data | USDA NASS | Public | Step 5 |
| Federal Register AEWR notices | federalregister.gov | Public | Step 5 |
| H-2A OFLC Case Disclosure Data 2008–2023 | DOL OFLC | Public | Step 7 |
