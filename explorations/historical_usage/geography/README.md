# Where did H-2A use expand?

Between the 2008–10 and 2023–25 three-year averages, the shared county panel gained about 303,220 allocated certified start-year positions per year. Three quarters of the gain occurred in the 1,624 counties with any baseline positions. The remaining 1,484 counties gained 75,733 positions. County coverage expanded from 1,435 of 3,108 in 2008 to 2,896 in 2025, so the number of places reached and their contributions to volume tell different parts of the growth story. Multiple employers can enter an already active county.

| Comparison | Share of net growth in counties active at baseline |
| --- | ---: |
| All start-year positions, 2008–10 vs 2023–25 | 75.0% |
| Certified start-year contract hours, same position-based county groups | 76.5% |
| Fiscal-year positions, baseline county groups recomputed in fiscal time | 74.8% |
| All start-year positions, 2009–10 vs 2023–25 | 68.6% |
| Single-county cases, groups recomputed on single-county positions in 2008–10 | 44.3% |

Each cell uses net change in the specified population and unit. Dropping 2008 shortens the window in which a county can qualify as established and reduces the baseline-active count to 1,446. This start-year sensitivity does not reconcile the fiscal-year 2008 source total. Contract hours represent certifications and include producer-imputed schedules.

The single-county check requires a reconciled case and a single unambiguous panel county across all its positive allocations. These cases account for 82.6% of allocated positions in 2008, 56.7% in 2019, 26.7% in 2020 and 24.4% in 2025. There is also a large 2014–15 break. The restriction selects a changing part of the program. Of full-panel net growth, 10.4% comes from those single-county cases and 9.9% comes from reconciled cases spanning multiple panel counties with every positive row unambiguously located. The remaining 79.8% is allocated to panel counties from cases that do not pass either complete-geography test. A single unlocated or ambiguous positive site excludes the whole case from the two fully resolved groups. This group can contain real multi-county employers. Its size rules out reading the entire complement as verified multi-county expansion. A claim about real changes in worksite organization requires separating differences in reporting, geocoding and allocation from changes in behavior.

## Can neighboring counties forecast a first recorded expansion?

Every county starts its observed history in 2008. A county remains at risk through the first year in which it crosses the indicated threshold. The geography model adds the share of adjacent counties crossing the same threshold in the previous year to a logistic forecast using only county farm employment in 2011, cropland in 2007, the static climate-based H-2A score trained through 2011, and other counties’ lagged threshold prevalence in its state. County-years carry equal weight. Both models use identical complete cases. Fit years are 2012–18. Coefficients stay fixed for one-year-ahead predictions in 2019–25, while observed histories advance yearly.

| Event in the 2019–25 evaluation period | Events / county-years | Baseline AUC | With neighbors | Baseline Brier loss | With neighbors |
| --- | ---: | ---: | ---: | ---: | ---: |
| First positive start-year allocation | 488 / 2,089 | 0.654 | 0.663 | 0.18364 | 0.18016 |
| First year with at least 10 positions | 579 / 6,384 | 0.664 | 0.694 | 0.08007 | 0.07908 |
| First year with at least 10 single-county-case positions | 304 / 10,183 | 0.671 | 0.704 | 0.02870 | 0.02873 |

AUC measures how often a higher forecast goes to a county-year that crosses. Brier loss is the mean squared error of the predicted probability. For the full ten-position target, the observed crossing rate rose from 6.1% where at most one-third of neighbors were above threshold to 18.0% where more than two-thirds were. Adding neighbors reduced Brier loss by 0.00099. The 95th percentile improvement across 40 placebo refits that assigned whole neighbor histories to other counties within state was 0.00005. Leaving each state out of the evaluation in turn yielded improvements of 0.00073 to 0.00121. These comparisons describe score stability. They provide no design-based significance test.

The adjacent model for first positive certification predicted 12.4% on average while 23.4% of held-out county-years crossed. For the full ten-position target, performance deteriorated in 2019 and 2025 on Brier loss. With single-county cases, neighbor information improved ranking while overall Brier loss rose slightly. Neighboring areas share crop suitability, contractors, weather and historic filing patterns. A mechanism paper could compare the first arrival of a filer already serving a neighbor with the first arrival of a filer new to the area, conditional on reconciling source-era changes in case geography and filer identity. These forecasts alone cannot identify information spillovers or an AEWR elasticity.

## Reproduction

From the repository root, using the project R environment:

```sh
source scripts/pipeline_helpers.sh
configure_r_library
Rscript --vanilla explorations/historical_usage/geography/01_geography.R
```

Inputs are the shared county-year panel, the 2010 county adjacency graph, allocated worksite rows, and the case-level allocation audit. The script prints full-year support, thresholds, aggregate accounting identities, coverage, scores, and placebo results. It checks unique balanced county-year keys, nonnegative positions, valid string county codes, symmetric links, exact growth reconciliation, and that single-county positions never exceed the panel total. In the graph, self-links and endpoints outside the 3,108 panel counties are removed.

The current release includes FY2008–FY2025 and partial FY2026 Q3 evidence. More decisions can revise calendar-2025 starts. The 2008 left endpoint gives first **recorded** threshold crossings, including returning users with activity before 2008. All fits are exploratory. Later-year scores use the current assembled vintage. They are retrospective evaluations, and the project has previously examined these years. Public certifications measure approved positions and contract schedules. County splits and imputed site staffing can move geographic exposure across counties. See the [shared panel validation](../../../outputs/reports/shared_panel_validation.json) and [worksite allocation contract](../../../content/contracts/h2a-worksite-exposure.md).
