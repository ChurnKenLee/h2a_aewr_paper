# Frozen AEWR bite across panel OLS, panel IV, and border comparisons

The county interaction is useful as supporting evidence about differential responses. It does not rescue identification of the average AEWR effect. Adding exposure-specific year effects removes most of the apparent precision. The border interaction fails the lead-placebo and precision checks. No retained specification should be replaced on this evidence.

This executes `markdowns/bite_designs_agent_plan.md`, the detailed plan accompanying the script catalog in `explorations/bite/README.md`. All estimates are exploratory and use cached local inputs. No acquisition, retained-artifact edits, commits, or specification promotion occurred. Numerical artifacts and figures are in the external scratch directory linked below; the report itself is the requested repository deliverable.

## Findings that change the interpretation

- The retained preferred first stage reproduces: coefficient **0.1701**, target-subregion F **44.71**, partial R² **0.1055**. The older OLS −0.040 benchmark uses the smaller 2012–2023 sample where the naive OEWS comparator is available; `partial_r2.R` reproduces −0.04007. The full BEA-control sample here gives −0.04427. This is a sample distinction, not a failed reproduction.
- Primary county 1(b) certified-position interactions are small, positive, and imprecisely different from zero, with region-cluster MDEs below 4% of the outcome mean. Region-year IV changes them little. With exposure-specific year effects, MDEs rise to roughly half the mean.
- The screened border interaction is positive, but its future-AEWR interaction also predicts the outcome. Its pair-CRV1 interaction MDE is about 68% of the mean. Statistical significance does not pass the declared design gate.
- **MC reconciliation found a unit mismatch.** Current MC code compares nominal AEWR/minimum wages with PPI-deflated ACS quantiles. Its stored fraction affected differs from the all-real calculation in **2356 / 3108 counties**; correlation is **0.2925**. A literal reconstruction of the current code matches the stored fraction exactly. This prevents reading its retained bite coefficient as the same treatment as the real-gap interaction here.

## 1. One predetermined exposure table

`cross_design_exposure.R` produces `exposures.parquet` and its CSV counterpart, unique by string county FIPS. The primary raw gap is the mean of real AEWR minus max(real CZ p25, real agricultural minimum wage) in 2008 and 2010, requiring both years. Every exposure is centered within AEWR region and divided by its pooled within-region standard deviation; rank uses average ranks with ties. No current or lagged local wage enters these exposure assignments.

| exposure | counties | mean | sd | within_sd | within_variance_share | min | p10 | median | p90 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gap |  3108 | 1.139 | 1.307 | 1.031 | 0.6221 | -3.984 | -0.4816 | 1.308 | 2.677 | 3.838 |
| frac |  3108 | 0.3384 | 0.07106 | 0.06261 | 0.7764 | 0.175 | 0.175 | 0.375 | 0.375 | 0.375 |
| frac_equal |  3108 | 0.3634 | 0.07106 | 0.06261 | 0.7764 |   0.2 |   0.2 |   0.4 |   0.4 |   0.4 |
| early_gap |  3026 | 0.9036 | 1.595 | 1.319 | 0.6841 | -5.711 | -1.17 | 1.157 | 2.687 |   4.3 |
| ag_gap |  2941 | -0.66 | 1.355 | 1.257 |  0.86 | -7.215 | -2.481 | -0.5544 | 0.8491 | 2.577 |
| rank_gap |  3108 |   0.5 | 0.2885 | 0.2885 |     1 | 0.001852 | 0.1012 |   0.5 | 0.8966 | 0.9981 |

The five variants are: `E_gap` (primary); `E_frac` (five-quantile fraction affected with MC weights 0.175/0.20/0.25/0.20/0.175, consistently real units); `E_gap_early` (2005–2007, all three years required); `E_gap_ag` (2008/2010 mapped OEWS Big-Six hourly mean, both years required); and `E_rank` (within-region rank of the primary gap). The early source also has 2000; the requested 2005–2007 window is used. Early construction deflates the intermediate ACS quantiles, which are nominal; shared-panel quantiles are already real.

The agricultural proxy is an occupation/reporting-area mean, not a farm-worker quantile or a directly observed county farm wage. The existing minimum-wage producer applies its 2024 agricultural-exemption classification historically; this inherited limitation also affects early exposure. Early ACS exposure therefore removes the 2009–2010 wage-method window but does not establish historically exact agricultural-floor coverage.

The early-exposure panel has 16 observed AEWR regions, versus 17 for the other variants. Its CRV1 reference degrees of freedom and Webb multipliers use that actual count. It is a changed support sensitivity, not an identical-population timing comparison.

| exposure | E_gap | E_frac | E_gap_early | E_gap_ag | E_rank |
| --- | --- | --- | --- | --- | --- |
| E_gap |     1 | 0.7465 | 0.9314 | 0.1618 | 0.9254 |
| E_frac | 0.7465 |     1 | 0.7228 | 0.07902 | 0.6467 |
| E_gap_early | 0.9314 | 0.7228 |     1 | 0.1904 | 0.8418 |
| E_gap_ag | 0.1618 | 0.07902 | 0.1904 |     1 | 0.1664 |
| E_rank | 0.9254 | 0.6467 | 0.8418 | 0.1664 |     1 |
| E_frac_equal | 0.7465 |     1 | 0.7228 | 0.07902 | 0.6467 |

The equal-weight fraction in the older scratch script differs in level, but its standardized exposure is exactly the same in this sample. The corrected real-unit fraction is **weight-consistent**, not numerically identical to the current MC artifact. The OEWS variant correlates weakly with the primary gap, so agreement on it is a demanding measurement check.

| year | czs | distinct_p25 |
| --- | --- | --- |
|  2008 |   608 |   347 |
|  2010 |   608 |   355 |
|  2015 |   608 |   324 |

2011 within-region level associations (state CRV1; descriptive, not a validation of causal exposure):

| exposure | outcome | n | estimate | se | p |
| --- | --- | --- | --- | --- | --- |
| E_gap | any_use |  3108 | -0.03499 | 0.0146 | 0.02061 |
| E_gap | cert |  3041 | 0.00156 | 0.003945 | 0.6943 |
| E_gap | applications |  3041 | -0.0003378 | 0.0002204 | 0.1321 |
| E_frac | any_use |  3108 | -0.02927 | 0.009645 | 0.003914 |
| E_frac | cert |  3041 | 0.002127 | 0.003538 | 0.5506 |
| E_frac | applications |  3041 | -0.0003592 | 0.0001804 | 0.05232 |
| E_gap_early | any_use |  3026 | -0.02696 | 0.01574 | 0.09364 |
| E_gap_early | cert |  2962 | 0.003477 | 0.004708 | 0.4639 |
| E_gap_early | applications |  2962 | -0.0002422 | 0.0001993 | 0.2307 |
| E_gap_ag | any_use |  2941 | 0.007542 | 0.01807 | 0.6784 |
| E_gap_ag | cert |  2881 | 0.0004539 | 0.004095 | 0.9122 |
| E_gap_ag | applications |  2881 | -0.0001105 | 0.0002939 | 0.7086 |
| E_rank | any_use |  3108 | -0.04141 | 0.01336 | 0.003269 |
| E_rank | cert |  3041 | -0.0004257 | 0.004632 | 0.9272 |
| E_rank | applications |  3041 | -0.0004018 | 0.00029 | 0.1724 |

Full exposure table: [exposures.csv](/tmp/h2a-bite-cross-design/exposures.csv); diagnostic source: `cross_design_exposure.R`.

## 2. Plain panel regression

All outcomes use the retained panel-IV registry: certifications, certified hours, applications, and balanced linked worksite businesses, each per fixed 2011 farm employment. BEA controls use 2011–2023; continuing controls use 2011–2024. Baseline and heterogeneity OLS include the relevant control set and county/year FE. The 1(b) regression uses A×E with county and region-year FE; 1(a) additionally absorbs E×year. These bite-only regressions omit controls, as in plan A3, but use the corresponding complete IV/control sample so IV–OLS comparisons hold observations fixed. Every sample requires both retained instruments, the outcome and all controls, without selecting on positive outcomes.

Baseline A and interaction X are different estimands. A in the heterogeneity model is the slope at E=0; the sample-average slope additionally depends on mean(E). X is the change in the AEWR slope per one standardized unit of frozen exposure. Standard errors and confidence intervals below use AEWR-region CRV1; all state and target-subregion alternatives remain in the full estimate CSV. MDE is the requested approximate 2.8×SE, not size-validated power.

| control_set | model | term | estimate | se | low | high | wild_p | mde | mde_pct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bea | baseline_ols | A | -0.04427 | 0.03871 | -0.1263 | 0.0378 | 0.3624 | 0.1084 | 119.3 |
| bea | heterogeneity | A | -0.04449 | 0.03878 | -0.1267 | 0.03772 | 0.3599 | 0.1086 | 119.5 |
| bea | heterogeneity | X | 0.002352 | 0.001947 | -0.001775 | 0.006479 | 0.2455 | 0.005451 |     6 |
| bea | bite_1b | X | 0.001512 | 0.001179 | -0.0009863 | 0.004011 | 0.2379 | 0.0033 | 3.632 |
| bea | bite_1a | X | -0.01166 | 0.01569 | -0.04491 | 0.0216 | 0.759 | 0.04393 | 48.35 |
| continuing | baseline_ols | A | -0.0418 | 0.03866 | -0.1238 | 0.04015 | 0.4287 | 0.1082 | 112.6 |
| continuing | heterogeneity | A | -0.042 | 0.03874 | -0.1241 | 0.04012 | 0.4274 | 0.1085 | 112.8 |
| continuing | heterogeneity | X | 0.002342 | 0.002104 | -0.002118 | 0.006803 | 0.3007 | 0.005892 | 6.129 |
| continuing | bite_1b | X | 0.001698 | 0.001327 | -0.001114 | 0.00451 | 0.2301 | 0.003714 | 3.864 |
| continuing | bite_1a | X | -0.01193 | 0.01572 | -0.04525 | 0.02139 | 0.7322 | 0.04401 | 45.78 |

![County interaction estimates with confidence intervals](/tmp/h2a-bite-cross-design/panel_comparison.png)

Other outcomes, primary 1(b) and 1(a):

| control_set | outcome | model | estimate | se | wild_p | mde | mde_pct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bea | hours | bite_1b | 1.346 | 1.115 | 0.2891 | 3.122 |  3.67 |
| bea | hours | bite_1a | -7.562 | 11.23 | 0.7963 | 31.46 | 36.98 |
| bea | applications | bite_1b | -0.00002464 | 0.00003166 | 0.4532 | 0.00008864 | 1.646 |
| bea | applications | bite_1a | -0.00001692 | 0.0001874 | 0.9405 | 0.0005246 | 9.742 |
| bea | businesses | bite_1b | 0.000007007 | 0.00009855 | 0.945 | 0.0002759 | 2.406 |
| bea | businesses | bite_1a | -0.0009992 | 0.000786 | 0.2327 | 0.002201 | 19.19 |
| continuing | hours | bite_1b | 1.411 | 1.163 | 0.2783 | 3.256 | 3.601 |
| continuing | hours | bite_1a | -7.443 | 10.67 | 0.7513 | 29.87 | 33.03 |
| continuing | applications | bite_1b | -0.00001566 | 0.00003465 | 0.6616 | 0.00009703 | 1.672 |
| continuing | applications | bite_1a | -0.0000451 | 0.0002072 | 0.8446 | 0.0005801 | 9.997 |
| continuing | businesses | bite_1b | 0.00005757 | 0.0001209 | 0.6534 | 0.0003386 | 2.585 |
| continuing | businesses | bite_1a | -0.0008375 | 0.000893 | 0.3731 | 0.0025 | 19.09 |

Exposure robustness for certified positions:

| control_set | exposure | model | estimate | se | wild_p | mde | mde_pct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bea | E_gap | bite_1b | 0.001512 | 0.001179 | 0.2379 | 0.0033 | 3.632 |
| bea | E_gap | bite_1a | -0.01166 | 0.01569 | 0.759 | 0.04393 | 48.35 |
| bea | E_frac | bite_1b | 0.001606 | 0.001091 | 0.2151 | 0.003055 | 3.362 |
| bea | E_frac | bite_1a | -0.01073 | 0.01239 | 0.7184 | 0.03469 | 38.19 |
| bea | E_gap_early | bite_1b | 0.001877 | 0.001366 | 0.2063 | 0.003824 |  4.41 |
| bea | E_gap_early | bite_1a | -0.02024 | 0.02061 | 0.4512 | 0.05771 | 66.55 |
| bea | E_gap_ag | bite_1b | 0.0008948 | 0.001064 | 0.5531 | 0.002979 | 3.281 |
| bea | E_gap_ag | bite_1a | -0.007692 | 0.009966 | 0.6235 | 0.0279 | 30.74 |
| bea | E_rank | bite_1b | 0.001616 | 0.001447 | 0.2966 | 0.00405 | 4.458 |
| bea | E_rank | bite_1a | -0.01165 | 0.01538 | 0.6675 | 0.04306 | 47.39 |
| continuing | E_gap | bite_1b | 0.001698 | 0.001327 | 0.2301 | 0.003714 | 3.864 |
| continuing | E_gap | bite_1a | -0.01193 | 0.01572 | 0.7322 | 0.04401 | 45.78 |
| continuing | E_frac | bite_1b | 0.001811 | 0.001213 | 0.2044 | 0.003395 | 3.532 |
| continuing | E_frac | bite_1a | -0.01204 | 0.01283 | 0.6023 | 0.03592 | 37.36 |
| continuing | E_gap_early | bite_1b | 0.002117 | 0.001557 | 0.2031 | 0.004361 | 4.744 |
| continuing | E_gap_early | bite_1a | -0.02089 | 0.02045 | 0.3855 | 0.05727 |  62.3 |
| continuing | E_gap_ag | bite_1b | 0.001017 | 0.001166 | 0.5065 | 0.003264 | 3.396 |
| continuing | E_gap_ag | bite_1a | -0.006845 | 0.009948 | 0.6542 | 0.02785 | 28.98 |
| continuing | E_rank | bite_1b | 0.001828 | 0.001641 | 0.2952 | 0.004596 | 4.781 |
| continuing | E_rank | bite_1a | -0.01148 | 0.01534 | 0.6696 | 0.04294 | 44.67 |

### Pre-years and leads

The event study is E×year, reference 2011, with county and region-year FE over 2008–2024. It uses the full shared outcome sample, including observed 2009 outcomes, without requiring a 2009 statutory wage or IV controls. It is a descriptive exposure-slope path, not a treatment-onset event study: AEWR existed before 2011. Both 2009 and 2010 are reversal-affected. This short, contaminated pre-period cannot certify parallel counterfactual trends.

![Frozen-gap event-study paths](/tmp/h2a-bite-cross-design/event_study.png)

| exposure | outcome | cluster | F | p | df1 | df2 |
| --- | --- | --- | --- | --- | --- | --- |
| E_gap | cert | aewr_region_id | 1.267 | 0.319 |     3 |    16 |
| E_gap | hours | aewr_region_id |  3.66 | 0.03507 |     3 |    16 |
| E_gap | applications | aewr_region_id | 1.087 | 0.3831 |     3 |    16 |
| E_gap | businesses | aewr_region_id | 0.9564 | 0.4371 |     3 |    16 |

Certified hours reject the joint pre-year null with region inference even though the primary certifications diagnostic does not. Do not describe all outcomes as passing pre-trends. Lead models include both current and next-year AEWR interactions. Next-year AEWR comes from the cached shared history through 2025, so the continuing-controls sample can test the 2024 lead without extrapolation.

| control_set | outcome | model | estimate | se | p | wild_p | mde | mde_pct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bea | cert | lead_1b | 0.0009524 | 0.001095 | 0.3971 | 0.5808 | 0.003065 | 3.373 |
| bea | cert | lead_1a | -0.006876 | 0.009905 | 0.4975 | 0.7192 | 0.02773 | 30.53 |
| bea | hours | lead_1b | 0.5958 | 0.7635 | 0.4466 | 0.6456 | 2.138 | 2.513 |
| bea | hours | lead_1a | -5.64 | 7.537 | 0.4651 | 0.6646 |  21.1 | 24.81 |
| bea | applications | lead_1b | -0.000003373 | 0.00002345 | 0.8874 | 0.8898 | 0.00006565 | 1.219 |
| bea | applications | lead_1a | -0.00004794 | 0.0001695 | 0.7809 | 0.787 | 0.0004745 | 8.813 |
| bea | businesses | lead_1b | 0.00003667 | 0.0001092 | 0.7414 | 0.7552 | 0.0003058 | 2.666 |
| bea | businesses | lead_1a | 0.00008474 | 0.0004706 | 0.8593 | 0.8597 | 0.001318 | 11.49 |
| continuing | cert | lead_1b | 0.00104 | 0.001291 | 0.4324 | 0.7178 | 0.003616 | 3.762 |
| continuing | cert | lead_1a | -0.003166 | 0.005383 | 0.5646 | 0.5877 | 0.01507 | 15.68 |
| continuing | hours | lead_1b | 0.5543 | 0.8417 | 0.5195 | 0.795 | 2.357 | 2.606 |
| continuing | hours | lead_1a | -5.237 | 5.558 | 0.3601 | 0.4165 | 15.56 | 17.21 |
| continuing | applications | lead_1b | -0.00001255 | 0.00002377 | 0.6049 | 0.6083 | 0.00006656 | 1.147 |
| continuing | applications | lead_1a | -0.0001286 | 0.000163 | 0.4417 | 0.4506 | 0.0004564 | 7.865 |
| continuing | businesses | lead_1b | -0.000009857 | 0.0001348 | 0.9426 | 0.9502 | 0.0003775 | 2.882 |
| continuing | businesses | lead_1a | -0.0003113 | 0.0005303 | 0.5654 | 0.5718 | 0.001485 | 11.34 |

## 3. Panel IV

The baseline is column 4. The interacted model instruments A and A×E with Z and Z×E, keeping controls and county/year FE. The within-region-year model instruments A×E with Z×E and matches the no-control 1(b) regression on exactly the same rows. AEWR itself is absorbed in the latter. Instruments retain source-year t−1 timing and the original donor construction.

| control_set | model | term | estimate | se | low | high | wild_p | mde | mde_pct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bea | baseline_iv | fit_A | -0.08931 | 0.06276 | -0.2224 | 0.04374 | 0.2552 | 0.1757 | 193.4 |
| bea | interacted_iv | fit_A | -0.0912 | 0.06298 | -0.2247 | 0.0423 | 0.2492 | 0.1763 | 194.1 |
| bea | interacted_iv | fit_X | 0.003434 | 0.002365 | -0.00158 | 0.008447 | 0.1802 | 0.006622 | 7.289 |
| bea | within_iv | fit_X | 0.00183 | 0.001438 | -0.001217 | 0.004878 | 0.2489 | 0.004025 |  4.43 |
| continuing | baseline_iv | fit_A | -0.09324 | 0.07002 | -0.2417 | 0.0552 | 0.2803 | 0.1961 |   204 |
| continuing | interacted_iv | fit_A | -0.09476 | 0.07032 | -0.2438 | 0.05431 | 0.277 | 0.1969 | 204.8 |
| continuing | interacted_iv | fit_X | 0.002974 | 0.002239 | -0.001772 | 0.007721 | 0.2318 | 0.00627 | 6.522 |
| continuing | within_iv | fit_X | 0.001828 | 0.001443 | -0.00123 | 0.004886 | 0.253 | 0.004039 | 4.202 |

The within-region-year IV and OLS point estimates are close relative to their uncertainty; this is not a formal Hausman equivalence test. IV does not materially improve precision. The dissimilar-donor exclusion restriction must hold for exposure-weighted target shocks as well as the original outcome error; regional calibration and correlated labor-demand shocks remain potential violations.

### First-stage and weak-IV diagnostics

| control_set | model | cluster | endogenous | term | estimate | joint_F | kp_F | partial_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bea | baseline_iv | aewr_region_id | A | Z | 0.1701 | 33.14 | unavailable | 0.1055 |
| bea | baseline_iv | aewr_iv_cluster_id | A | Z | 0.1701 | 44.71 | unavailable | 0.1055 |
| bea | interacted_iv | aewr_region_id | A | Z | 0.1693 | 16.89 | unavailable | 0.106 |
| bea | interacted_iv | aewr_region_id | A | ZE | 0.004122 | 16.89 | unavailable | 0.106 |
| bea | interacted_iv | aewr_region_id | X | Z | -0.04373 | 549.8 | unavailable | 0.4517 |
| bea | interacted_iv | aewr_region_id | X | ZE | 0.6674 | 549.8 | unavailable | 0.4517 |
| bea | interacted_iv | aewr_iv_cluster_id | A | Z | 0.1693 | 22.63 | unavailable | 0.106 |
| bea | interacted_iv | aewr_iv_cluster_id | A | ZE | 0.004122 | 22.63 | unavailable | 0.106 |
| bea | interacted_iv | aewr_iv_cluster_id | X | Z | -0.04373 |  1079 | unavailable | 0.4517 |
| bea | interacted_iv | aewr_iv_cluster_id | X | ZE | 0.6674 |  1079 | unavailable | 0.4517 |
| bea | interacted_iv | aewr_region_id | A | SW_conditional | unavailable | 32.96 | 16.56 | unavailable |
| bea | interacted_iv | aewr_region_id | X | SW_conditional | unavailable |  1141 | 16.56 | unavailable |
| bea | interacted_iv | aewr_iv_cluster_id | A | SW_conditional | unavailable | 43.85 | 22.01 | unavailable |
| bea | interacted_iv | aewr_iv_cluster_id | X | SW_conditional | unavailable |  1552 | 22.01 | unavailable |
| bea | within_iv | aewr_region_id | X | ZE | 0.6797 |  1115 | unavailable | 0.4738 |
| bea | within_iv | aewr_iv_cluster_id | X | ZE | 0.6797 |  2255 | unavailable | 0.4738 |
| continuing | baseline_iv | aewr_region_id | A | Z | 0.1821 | 38.68 | unavailable | 0.1139 |
| continuing | baseline_iv | aewr_iv_cluster_id | A | Z | 0.1821 | 50.86 | unavailable | 0.1139 |
| continuing | interacted_iv | aewr_region_id | A | Z | 0.1813 | 19.67 | unavailable | 0.1143 |
| continuing | interacted_iv | aewr_region_id | A | ZE | 0.004016 | 19.67 | unavailable | 0.1143 |
| continuing | interacted_iv | aewr_region_id | X | Z | -0.03492 | 526.9 | unavailable | 0.4521 |
| continuing | interacted_iv | aewr_region_id | X | ZE | 0.6796 | 526.9 | unavailable | 0.4521 |
| continuing | interacted_iv | aewr_iv_cluster_id | A | Z | 0.1813 | 26.13 | unavailable | 0.1143 |
| continuing | interacted_iv | aewr_iv_cluster_id | A | ZE | 0.004016 | 26.13 | unavailable | 0.1143 |
| continuing | interacted_iv | aewr_iv_cluster_id | X | Z | -0.03492 |  1041 | unavailable | 0.4521 |
| continuing | interacted_iv | aewr_iv_cluster_id | X | ZE | 0.6796 |  1041 | unavailable | 0.4521 |
| continuing | interacted_iv | aewr_region_id | A | SW_conditional | unavailable | 38.62 |  19.4 | unavailable |
| continuing | interacted_iv | aewr_region_id | X | SW_conditional | unavailable |  1100 |  19.4 | unavailable |
| continuing | interacted_iv | aewr_iv_cluster_id | A | SW_conditional | unavailable | 50.34 | 25.32 | unavailable |
| continuing | interacted_iv | aewr_iv_cluster_id | X | SW_conditional | unavailable |  1638 | 25.32 | unavailable |
| continuing | within_iv | aewr_region_id | X | ZE | 0.6928 |  1093 | unavailable | 0.4735 |
| continuing | within_iv | aewr_iv_cluster_id | X | ZE | 0.6928 |  2272 | unavailable | 0.4735 |

`joint_F` in ordinary first-stage rows tests all excluded instruments jointly. `SW_conditional` rows give Sanderson–Windmeijer conditional F statistics; `kp_F` gives the Kleibergen–Paap rk Wald F for the two-endogenous model. These use clustered reduced-form scores after the exact nuisance projection, with G/(G−1) CRV1. They are not Stock–Yogo critical-value tests with 17 clusters. The installed fixest KP helper fails with a score/model dimension mismatch when extra controls are present; `cross_design_helpers.R` implements the same SVD rank construction with a directly computed cluster meat. Synthetic tests check its scalar-Wald identity and scale/order invariance.

Region-cluster Anderson–Rubin sets for the primary exposure (exact quadratic inversion for one endogenous regressor, conditional on constructed inputs):

| control_set | outcome | model | ar_type | ar_low | ar_high |
| --- | --- | --- | --- | --- | --- |
| bea | cert | baseline_iv | bounded | -0.2236 | 0.06313 |
| bea | cert | within_iv | bounded | -0.001282 | 0.004826 |
| bea | hours | baseline_iv | bounded | -149.9 | 30.45 |
| bea | hours | within_iv | bounded | -1.542 | 4.347 |
| bea | applications | baseline_iv | bounded | -0.004512 | 0.00004083 |
| bea | applications | within_iv | bounded | -0.0001111 | 0.00005329 |
| bea | businesses | baseline_iv | bounded | -0.005891 | 0.007389 |
| bea | businesses | within_iv | bounded | -0.0003584 | 0.0006028 |
| continuing | cert | baseline_iv | bounded | -0.2416 | 0.07474 |
| continuing | cert | within_iv | bounded | -0.001298 | 0.004832 |
| continuing | hours | baseline_iv | bounded | -159.1 | 35.59 |
| continuing | hours | within_iv | bounded | -1.553 | 4.339 |
| continuing | applications | baseline_iv | bounded | -0.004751 | -0.0005721 |
| continuing | applications | within_iv | bounded | -0.0001094 | 0.00006197 |
| continuing | businesses | baseline_iv | bounded | -0.008613 | 0.006642 |
| continuing | businesses | within_iv | bounded | -0.000372 | 0.0006532 |

The two-endogenous model uses joint AR inversion over a coarse tail grid plus a dense local grid. Below are coordinate projections of accepted **grid points**, not a rectangular joint set or exact continuous endpoints. The full grid preserves holes and nonconvexity. A boundary hit means an unresolved tail; absence of a hit does not prove boundedness beyond the searched domain. Inference uses F(2,G−1), an asymptotic cluster approximation, not finite-sample exact coverage.

| control_set | outcome | grid_points | accepted_points | accepted_boundary | min_A | max_A | min_X | max_X |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bea | applications | 104006 |  2538 | FALSE | -0.005115 | 0.0008225 | -0.0002118 | 0.00007349 |
| bea | businesses | 104329 |  3361 | FALSE | -0.007023 | 0.01068 | -0.0008743 | 0.0006077 |
| bea | cert | 104329 |  2084 | FALSE | -0.2625 | 0.1229 | -0.003898 | 0.009346 |
| bea | hours | 104329 |  2342 | FALSE |  -175 | 67.95 | -3.25 | 7.369 |
| continuing | applications | 104329 |  2630 | FALSE | -0.005381 | 0.00006206 | -0.0002532 | 0.00008055 |
| continuing | businesses | 104329 |  3412 | FALSE | -0.01035 |  0.01 | -0.0009873 | 0.0006553 |
| continuing | cert | 104329 |  2068 | FALSE | -0.2846 | 0.1375 | -0.003967 | 0.008572 |
| continuing | hours | 104329 |  2349 | FALSE | -185.5 |    75 | -3.53 | 6.855 |

Joint accepted sets and all rejected grid points: [iv_joint_ar_grid.csv](/tmp/h2a-bite-cross-design/iv_joint_ar_grid.csv).

### Donor co-movement by exposure

Target subregions are classified high/low by their mean frozen E relative to the within-region median (ties high). County/year residualization and target/donor definitions reproduce `common_shocks.R`; the available panel covers 2011–2023 for this comparison. Confidence bounds resample whole AEWR regions 999 times, seed 20261102. Donor outcomes net of AEWR co-move less than nondonor outcomes in both exposure groups, but this is evidence about observed correlation, not proof of exclusion.

| series | exposure_group | donor | donor_low | donor_high | nondonor | difference | difference_low | difference_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oews_log_wage | high | 0.1115 | 0.004721 | 0.2275 | 0.2379 | -0.1264 | -0.2231 | -0.01086 |
| oews_log_wage | low | 0.1446 | 0.01572 | 0.2762 | 0.341 | -0.1964 | -0.5319 | 0.1304 |
| cert_share | high | 0.06718 | -0.1945 | 0.312 | 0.2675 | -0.2003 | -0.8294 | -0.03315 |
| cert_share | low | 0.0975 | -0.1019 | 0.1845 | 0.2272 | -0.1297 | -0.4754 | -0.05431 |
| cert_net_aewr | high | 0.07312 | -0.04386 | 0.2835 | 0.2673 | -0.1942 | -0.6422 | -0.04956 |
| cert_net_aewr | low | 0.08825 | -0.01031 | 0.1758 | 0.2004 | -0.1122 | -0.4385 | -0.04754 |
| applications_net_aewr | high | -0.05924 | -0.2159 | 0.1131 | 0.2438 | -0.303 | -0.4877 | -0.1384 |
| applications_net_aewr | low | 0.02469 | -0.1363 | 0.1353 |  0.37 | -0.3453 | -0.5275 | -0.1965 |

## 4. Border discontinuity

Both the original one-ring disjoint benchmark and the screened one-ring candidate use their existing memberships, county/component-year FE and 2011–2025 calendar, with equal county-year weights. All four outcomes are estimated. E_pair is the frozen mean of member counties' standardized exposure within each region pair, with each county weighted equally. The heterogeneity regression includes A and A×E_pair. This is heterogeneity across borders, not new independent policy assignments.

| geography | model | term | estimate | se | p | wcb_p_value | wild_p | mde | mde_pct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ring_h1_disjoint | baseline | main_a | -0.008954 | 0.007655 | 0.2536 | 0.2758 | 0.1853 | 0.02143 | 31.78 |
| ring_h1_disjoint | heterogeneity | main_a | -0.009515 | 0.007638 | 0.2249 | 0.2364 | 0.152 | 0.02139 | 31.71 |
| ring_h1_disjoint | heterogeneity | X | 0.01384 | 0.01326 | 0.3069 | 0.2963 | 0.3139 | 0.03711 | 55.03 |
| ring_h1_disjoint | frozen_floor_gap | G | -0.008954 | 0.007655 | 0.2536 | 0.2758 | 0.1853 | 0.02143 | 31.78 |
| ring_h1_disjoint | lead | X | 0.003058 | 0.0131 | 0.8174 | 0.8148 | 0.8416 | 0.03669 | 57.12 |
| ring_h1_disjoint | lead | L | 0.02089 | 0.00929 | 0.03399 | 0.0265 | 0.0074 | 0.02601 |  40.5 |
| ring_h1_screened_disjoint | baseline | main_a | 0.0003855 | 0.004872 | 0.9376 | 0.9442 | 0.9538 | 0.01364 | 23.33 |
| ring_h1_screened_disjoint | heterogeneity | main_a | -0.0001689 | 0.004644 | 0.9713 | 0.973 | 0.9715 | 0.013 | 22.24 |
| ring_h1_screened_disjoint | heterogeneity | X | 0.0374 | 0.0141 | 0.01392 | 0.0118 | 0.0211 | 0.03947 |  67.5 |
| ring_h1_screened_disjoint | frozen_floor_gap | G | 0.0003855 | 0.004872 | 0.9376 | 0.9442 | 0.9538 | 0.01364 | 23.33 |
| ring_h1_screened_disjoint | lead | X | 0.02648 | 0.01424 | 0.07538 | 0.06131 | 0.0752 | 0.03988 | 72.58 |
| ring_h1_screened_disjoint | lead | L | 0.03087 | 0.01246 | 0.02069 | 0.0179 | 0.0204 | 0.0349 | 63.51 |

Pair WCR11-FNW uses the branch's existing helper, 9,999 Webb draws and seed 201103. The additional region bootstrap is reported separately; neither removes shared-endpoint dependence. Failed confidence inversion is retained in the bootstrap CSV. For the screened interaction, the positive lead and poor MDE rule out promotion even when the interaction's own pair test rejects zero.

### Exposure support and the frozen-floor identity

| geography | pairs | mean_abs_E_difference | sd_E_pair | mean_abs_floor_difference |
| --- | --- | --- | --- | --- |
| ring_h1_disjoint |    25 | 0.4211 | 0.5057 | 0.3139 |
| ring_h1_screened_disjoint |    25 | 0.4054 | 0.4838 | 0.3629 |

| geography | adjacent_edges | same_CZ | same_floor | same_CZ_different_floor |
| --- | --- | --- | --- | --- |
| ring_h1_disjoint |   464 |   175 |   166 |     9 |
| ring_h1_screened_disjoint |   382 |   143 |   134 |     9 |

Even counties sharing a CZ can have different agricultural state floors. A shared ACS quantile alone therefore does not ensure identical local floors. Full side and direct-adjacency diagnostics are retained.

For G_ct=A_rt−floor_c, county FE absorb the frozen floor exactly. Across all four outcomes and both geographies, the largest absolute difference between residualized G and residualized A is **0.00000000000005773**. Gap-as-treatment reproduces the AEWR coefficient and standard error; it creates no new policy variation.

| geography | raw_within_component_A_var | raw_within_component_floor_var | raw_within_component_cov | raw_within_component_gap_var | policy_FWL_variance |
| --- | --- | --- | --- | --- | --- |
| ring_h1_disjoint | 0.4699 | 0.7273 | 0.03236 | 1.133 | 0.05732 |
| ring_h1_screened_disjoint | 0.4678 | 0.6338 | 0.0275 | 1.047 | 0.0561 |

These component-year-only variances obey Var(G)=Var(A)+Var(floor)−2Cov(A,floor). With county FE as well, all remaining G variation equals A variation. Treating the raw floor variation as policy would be erroneous.

### Border pre-period and signed power

An E_pair×year event path is fully absorbed by component-year FE, so it is not an estimable border pre-trend test. The report instead retains the 2008/2010 A and A×E_pair slope diagnostic; 2009 has no canonical wage and is not filled. The diagnostic is already-exposed and reversal-contaminated, not a clean untreated placebo.

| geography | estimate | se | p | mde | mean_y | years |
| --- | --- | --- | --- | --- | --- | --- |
| ring_h1_disjoint | 0.001361 | 0.02466 | 0.9564 | 0.06904 | 0.02572 | 2008;2010 |
| ring_h1_screened_disjoint | 0.009178 | 0.02292 | 0.6923 | 0.06416 | 0.01877 | 2008;2010 |

Signed power reuses `power.R`: four declared county/region/pair/spatial DGPs, AR(1)=0.5, 999 calibration and simulation draws, seed 201103, one thread, the original signed grid through 0.20 and the upper-Wilson size gate of 0.075. The effect being injected is now the interaction slope while A remains a nuisance regressor. Preperiod scale uses outcome FE residuals only; policy-residual scale is retained as sensitivity. The scalar WCR simulation shortcut does not support the extra slope, so it is not applied. Observed multiregressor WCR inference still uses the branch helper.

| model_id | effect_sign | grid_mde | credible_grid_mde | size_gate_pass | null_rejection_rate |
| --- | --- | --- | --- | --- | --- |
| ring_h1_disjoint__E_gap |    -1 | 0.122 | 0.122 | TRUE | 0.05205 |
| ring_h1_disjoint__E_gap |     1 | 0.125 | 0.125 | TRUE | 0.05205 |
| ring_h1_screened_disjoint__E_gap |    -1 | 0.092 | 0.092 | TRUE | 0.05105 |
| ring_h1_screened_disjoint__E_gap |     1 | 0.092 | 0.092 | TRUE | 0.05105 |

The early exposure loses all observed counties on one side of the Southeast–Florida edge (10_11). The inherited power engine correctly refuses this incomplete border. Those power results remain unavailable; the full original membership is not silently pruned to obtain a favorable simulation. Estimates using the remaining exposure-complete rows are labeled sensitivities.

| model_id | status |
| --- | --- |
| ring_h1_disjoint__E_gap | eligible |
| ring_h1_disjoint__E_frac | eligible |
| ring_h1_disjoint__E_gap_early | unavailable_exposure_removes_entire_border_side |
| ring_h1_disjoint__E_gap_ag | eligible |
| ring_h1_disjoint__E_rank | eligible |
| ring_h1_screened_disjoint__E_gap | eligible |
| ring_h1_screened_disjoint__E_frac | eligible |
| ring_h1_screened_disjoint__E_gap_early | unavailable_exposure_removes_entire_border_side |
| ring_h1_screened_disjoint__E_gap_ag | eligible |
| ring_h1_screened_disjoint__E_rank | eligible |

## 5. Permutations and decision gates

Each primary certified-position comparison uses 500 county-level exposure permutations within AEWR region, keeping a county's permuted exposure fixed over time. OLS permutations are within its estimation counties; IV does the same. Border permutations use the common exposure universe before recomputing pair means. These are placebo reference distributions, **not randomization p-values**: no exchangeable treatment assignment is established.

| design | control_set | model | draws | p025 | p975 | observed |
| --- | --- | --- | --- | --- | --- | --- |
| OLS | bea | bite_1a |   500 | -0.01696 | 0.02296 | -0.01166 |
| OLS | bea | bite_1b |   500 | -0.001238 | 0.001343 | 0.001512 |
| OLS | continuing | bite_1a |   500 | -0.01725 | 0.0188 | -0.01193 |
| OLS | continuing | bite_1b |   500 | -0.001567 | 0.001321 | 0.001698 |
| panel_iv | bea | interacted_iv |   500 | -0.001599 | 0.004126 | 0.003434 |
| panel_iv | bea | within_iv |   500 | -0.002496 | 0.003269 | 0.00183 |
| panel_iv | continuing | interacted_iv |   500 | -0.001942 | 0.004166 | 0.002974 |
| panel_iv | continuing | within_iv |   500 | -0.00268 | 0.003428 | 0.001828 |

| geography | model | draws | p025 | p975 |
| --- | --- | --- | --- | --- |
| ring_h1_disjoint | heterogeneity |   500 | -0.1263 | 0.1408 |
| ring_h1_screened_disjoint | heterogeneity |   500 | -0.08956 | 0.07706 |

Passing a nonsignificant pre-test is not evidence of absence of pre-trends. A precise null can be useful without rejecting zero in a wild bootstrap; no selection on significance is used. All variants, outcomes, controls, inference partitions and failures remain in the linked files. The 25%-of-mean threshold is applied to the specified differential-effect unit, not to the unidentified average effect.

## 6. What an interaction would imply for an average effect

Impose the structural restriction β_c=β̄·gap_c/μ, where μ is the target-sample mean frozen raw gap. Write gap_c=gap̄_r+sE_c, where s is the pooled within-region gap SD used in normalization. Then β_c A_rt=(β̄/μ)gap̄_r A_rt+(β̄s/μ)E_cA_rt. Region-year FE absorb the first term, so θ_1b=β̄s/μ and **β̄=θ_1b·μ/s**. Confidence endpoints scale by the same positive factor; the table uses the region-CRV1 interval and treats estimated exposures as fixed.

| control_set | outcome | gap_mean | multiplier | average_effect | average_low | average_high | average_mde |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bea | cert | 1.237 |   1.2 | 0.001815 | -0.001184 | 0.004813 | 0.00396 |
| bea | hours | 1.237 |   1.2 | 1.616 | -1.221 | 4.452 | 3.746 |
| bea | applications | 1.237 |   1.2 | -0.00002957 | -0.0001101 | 0.00005097 | 0.0001064 |
| bea | businesses | 1.237 |   1.2 | 0.000008409 | -0.0002423 | 0.0002591 | 0.0003312 |
| continuing | cert | 1.237 |   1.2 | 0.002037 | -0.001337 | 0.005411 | 0.004456 |
| continuing | hours | 1.237 |   1.2 | 1.692 | -1.266 |  4.65 | 3.907 |
| continuing | applications | 1.237 |   1.2 | -0.00001879 | -0.0001069 | 0.00006935 | 0.0001164 |
| continuing | businesses | 1.237 |   1.2 | 0.00006906 | -0.0002385 | 0.0003766 | 0.0004062 |

This restriction sets the AEWR effect to zero at zero gap, reverses its sign for negative gaps, and imposes a common proportional response. Negative gaps occur in 550 of 3108 exposure counties. It additionally requires valid differential counterfactual trends, no confounding correlated with exposure, correct local-floor measurement, and a stable mapping over the analysis window. A nonzero common response unrelated to gap is absorbed by region-year FE and **cannot be bounded by θ**. Consequently, these are conditional model-based intervals, not assumption-free bounds on the average AEWR effect or a welfare elasticity.

## 7. Reconciliation with Mundlak–Chamberlain

| counties | max_fraction_error_literal | max_fraction_error_real | fraction_counties_different | fraction_correlation | max_baseline_gap_error |
| --- | --- | --- | --- | --- | --- |
|  3108 | 0 |   0.2 |  2356 | 0.2925 | 0 |

The MC baseline gap moderator itself matches real AEWR minus real p25 (without the agricultural-floor maximum). Its fraction-affected construction and direct five-quantile bite coordinates instead use nominal policy/floor columns with real quantiles. This is visible in `_baseline_fraction_affected` and `_bite_expression` in `mcw/build.py`, and the source quantile deflation is owned by C01/13. The unit audit reproduces the stored fraction exactly under that literal mixed-unit expression; the discrepancy is not an inference choice or numerical rounding issue.

Retained MC any-application current-coordinate summaries (unaltered, reported for reconciliation rather than endorsed as an equivalent real-gap effect):

| specification_id | estimate | standard_error | target_population | target_observations |
| --- | --- | --- | --- | --- |
| bite_f0809__full__county_state_year__bite__aewr_region__continuous_raw | -0.2404 | 0.0988 | eligible_county_years_2013_2022_with_identified_current_effects | 30410 |
| bite_f0810__full__county_state_year__bite__aewr_region__continuous_raw | -0.2141 | 0.1136 | eligible_county_years_2013_2022_with_identified_current_effects | 30410 |

Even after a unit correction, the estimands differ: MC uses five-point positive-part bite and separate treatment-history coordinates, unrestricted history/moderator projections, 2012–2022 outcome rows with current-effect averages over 2013–2022, and raw counts or any-application followed by baseline-employment normalization. Here the moderator is a signed frozen p25 gap, treatment is contemporaneous real AEWR×E, the main outcome is a county ratio with a 2011 denominator, and the sample/calendar differs. Static same-MC-sample bridge regressions show the scale/FE distinction without claiming to reproduce its dynamic estimand:

| outcome | treatment | FE | estimate | se | p | n |
| --- | --- | --- | --- | --- | --- | --- |
| mc_y_any_application | X | county_fips + aewr_region_id^year | -0.0001739 | 0.001466 | 0.9071 | 34188 |
| mc_y_any_application | X | county_fips + state_fips^year | 0.0002665 | 0.001205 | 0.8277 | 34188 |
| mc_y_any_application | bite_current | county_fips + aewr_region_id^year | 0.02586 | 0.03223 | 0.434 | 34188 |
| mc_y_any_application | bite_current | county_fips + state_fips^year | 0.01465 | 0.05337 | 0.7872 | 34188 |
| h2a_cert_share_farm_workers_2011_start_year | X | county_fips + aewr_region_id^year | 0.001917 | 0.001737 | 0.286 | 33451 |
| h2a_cert_share_farm_workers_2011_start_year | X | county_fips + state_fips^year | 0.002306 | 0.002275 | 0.3259 | 33451 |
| h2a_cert_share_farm_workers_2011_start_year | bite_current | county_fips + aewr_region_id^year | 0.05465 | 0.04575 | 0.2496 | 33451 |
| h2a_cert_share_farm_workers_2011_start_year | bite_current | county_fips + state_fips^year | 0.144 | 0.1367 | 0.3079 | 33451 |
| mc_y_certified_positions | X | county_fips + aewr_region_id^year |  1.15 | 1.419 | 0.4297 | 34188 |
| mc_y_certified_positions | X | county_fips + state_fips^year | 1.283 | 2.037 | 0.5378 | 34188 |
| mc_y_certified_positions | bite_current | county_fips + aewr_region_id^year | 45.91 | 53.72 | 0.4053 | 34188 |
| mc_y_certified_positions | bite_current | county_fips + state_fips^year | 82.54 | 84.59 | 0.3437 | 34188 |

The DiD branch's contemporaneous/lagged gap is a time-varying regressor; it is not used as predetermined exposure here. Its 2010 lag is unavailable because canonical 2009 AEWR is missing. No DiD coefficient is silently relabeled as the present interaction.

## 8. Recommendations and proposed integration

- **Plain panel: supporting evidence.** Retain the frozen-gap 1(b) differential null with the 1(a) loss of precision, early/rank/fraction/OEWS alternatives, lead tests and the adverse hours pre-test. Do not promote it to an unconditional average causal effect. A supplemental integration would update the nearest exploratory README and a new design note before adding any producer or retained figure.
- **Panel IV: supporting robustness, not a new preferred column.** The region-year interaction IV adds little to the corresponding OLS comparison. Keep the original preferred four-column registry unchanged. Any future integration needs an explicit proposal updating `code/designs/panel_iv/README.md`, `content/designs/panel-iv.md`, and the outcome/specification contract, preserving both control sets and weak-IV evidence.
- **Border: drop as a route to a credible, well-powered bite estimate with these inputs.** Preserve the positive lead, high interaction MDE, shared-region dependence, incomplete early-exposure border and size failures. A frozen-floor treatment is algebraically redundant. Do not change `design.json` or nominate the significant screened interaction as primary.
- **MC: review units before substantive cross-design claims.** Propose a separate owning-producer change in `mcw/build.py`, with unit tests, README/design-page/grounding updates and a rebuild of its panel, compact queue and report. Decide whether frozen wages represent real purchasing-power values or nominal baseline wages before changing the code. This report does not authorize or perform that empirical revision.

## 9. Reproduction, complete results, and limitations

Run from repository root with the existing R library. Execute `cross_design_exposure.R`, `cross_design_panel.R`, `cross_design_border.R`, `cross_design_diagnostics.R`, `cross_design_placebos.R`, and `cross_design_report.R`, each with the same external output directory as its first argument. `test_cross_design.R` is the focused numerical test; `validate_cross_design.R` checks the completed artifacts. The README gives exact commands. The report is generated in scratch and copied byte-for-byte to `markdowns/bite_designs_cross_design.md`.

Validation passed for 296 panel specifications and 96 border specifications, input hashes, unique string keys, exposure arithmetic, retained baseline reproduction, MDE and frozen-floor identities, 500-draw placebo coverage, AR-grid summaries, the MC unit reconciliation, and inherited power resource guards. Synthetic numerical tests passed. All repository R sources and the new exploration scripts parsed. Agent grounding verification and its seven tests passed. Documentation check and build passed using `env -u LD_LIBRARY_PATH zola check --skip-external-links` and `env -u LD_LIBRARY_PATH zola build`; clearing that environment variable resolved the local shared-library conflict.

The baseline reproduction logs are generated by the pre-existing `county_bite_acs.R` and `partial_r2.R`. New numerical code owns only scratch artifacts. Input SHA-256 values are retained, and validation checks that they remain unchanged. A complete-case/sample loss is recorded through n, county count and actual first/last years in each estimate row. State, region and target-subregion CRV1 are separate rows; border adds region-pair CRV1. Bootstrap seeds/counts/statuses are explicit.

The restricted region Webb test uses 9,999 draws/seed 20261102 and projects each cluster's null residual through every nuisance term before generating draws. The IV Wald version conditions on X/Z and is not weak-IV robust; AR supplies the separate weak-IV diagnostic. Synthetic literal raw-row OLS and IV refits independently reproduce the implemented bootstrap p-values, including global year FE. Exposure and instrument generation uncertainty is not included. Border WCR inversion failures remain explicit rather than receiving repaired endpoints.

County counts may be fractional because multi-county cases are allocated. Any-use is not a farm adoption probability. Linked-business counts have a 2020 disclosure break; later full-vintage linkage can revise earlier counts. Certified hours inherit pooled schedule imputation and are not actual paid hours. From 2023, occupational wage applicability complicates a single regional reference. Neither the zero post-2023 at-AEWR diagnostic nor near-universal H-2A-offer bindingness validates the domestic outside-option proxy. The 2025 border year uses the cached partial FY2026-Q3 vintage and retains its revision risk.

Every specification run, including unsuccessful confidence inversion and unsupported power, is available below. Secondary outcomes and exposure variants are not filtered on statistical significance. Figures also reside in scratch to respect the plan's output boundary; their absolute local links require this scratch directory to remain available.

- [exposures.csv](/tmp/h2a-bite-cross-design/exposures.csv)
- [exposure_diagnostics.csv](/tmp/h2a-bite-cross-design/exposure_diagnostics.csv)
- [exposure_correlations.csv](/tmp/h2a-bite-cross-design/exposure_correlations.csv)
- [acs_bunching.csv](/tmp/h2a-bite-cross-design/acs_bunching.csv)
- [exposure_levels_2011.csv](/tmp/h2a-bite-cross-design/exposure_levels_2011.csv)
- [panel_estimates.csv](/tmp/h2a-bite-cross-design/panel_estimates.csv)
- [panel_wild.csv](/tmp/h2a-bite-cross-design/panel_wild.csv)
- [panel_failures.csv](/tmp/h2a-bite-cross-design/panel_failures.csv)
- [event_study.csv](/tmp/h2a-bite-cross-design/event_study.csv)
- [pretrend_tests.csv](/tmp/h2a-bite-cross-design/pretrend_tests.csv)
- [iv_first_stages.csv](/tmp/h2a-bite-cross-design/iv_first_stages.csv)
- [iv_ar_sets.csv](/tmp/h2a-bite-cross-design/iv_ar_sets.csv)
- [iv_joint_ar_grid.csv](/tmp/h2a-bite-cross-design/iv_joint_ar_grid.csv)
- [comovement_by_exposure.csv](/tmp/h2a-bite-cross-design/comovement_by_exposure.csv)
- [border_estimates.csv](/tmp/h2a-bite-cross-design/border_estimates.csv)
- [border_bootstrap.csv](/tmp/h2a-bite-cross-design/border_bootstrap.csv)
- [border_pair_exposure.csv](/tmp/h2a-bite-cross-design/border_pair_exposure.csv)
- [border_gap_identity.csv](/tmp/h2a-bite-cross-design/border_gap_identity.csv)
- [border_preperiod.csv](/tmp/h2a-bite-cross-design/border_preperiod.csv)
- [border_power_support.csv](/tmp/h2a-bite-cross-design/border_power_support.csv)
- [border_bite_power.csv](/tmp/h2a-bite-cross-design/border_bite_power.csv)
- [border_bite_mde.csv](/tmp/h2a-bite-cross-design/border_bite_mde.csv)
- [border_bite_calibration.csv](/tmp/h2a-bite-cross-design/border_bite_calibration.csv)
- [border_bite_simulation_diagnostics.csv](/tmp/h2a-bite-cross-design/border_bite_simulation_diagnostics.csv)
- [permutation_placebos.csv](/tmp/h2a-bite-cross-design/permutation_placebos.csv)
- [iv_permutation_placebos.csv](/tmp/h2a-bite-cross-design/iv_permutation_placebos.csv)
- [border_permutation_placebos.csv](/tmp/h2a-bite-cross-design/border_permutation_placebos.csv)
- [average_effect_conversion.csv](/tmp/h2a-bite-cross-design/average_effect_conversion.csv)
- [mc_unit_audit.csv](/tmp/h2a-bite-cross-design/mc_unit_audit.csv)
- [mc_retained_comparison.csv](/tmp/h2a-bite-cross-design/mc_retained_comparison.csv)
- [mc_static_bridge.csv](/tmp/h2a-bite-cross-design/mc_static_bridge.csv)
- [input_hashes.csv](/tmp/h2a-bite-cross-design/input_hashes.csv)
- [numerical_tests.log](/tmp/h2a-bite-cross-design/numerical_tests.log)
- [prior_bite.log](/tmp/h2a-bite-cross-design/prior_bite.log)
- [prior_panel.log](/tmp/h2a-bite-cross-design/prior_panel.log)
- [validation.csv](/tmp/h2a-bite-cross-design/validation.csv)
- [validation.log](/tmp/h2a-bite-cross-design/validation.log)
- [zola_check.log](/tmp/h2a-bite-cross-design/zola_check.log)
- [zola_build.log](/tmp/h2a-bite-cross-design/zola_build.log)

Implementation checks establish reproducibility and arithmetic within this exploration; they do not establish exclusion, parallel counterfactual trends or causal adequacy.
