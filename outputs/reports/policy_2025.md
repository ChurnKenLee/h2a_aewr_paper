# The 2025 H-2A policy shift

Generated 2026-11-17 02:22:19.020376. This standalone design preserves the inspected wage-offer and interim earnings specifications. The November 16, 2026 migration changes ownership and output locations, not outcomes, samples, treatment, weights or inference. It is not a new preregistration. The 2013 shutdown is a separate historical exploration.

## Offered wages

| specification | weights | n | states | estimate | low | high |
| --- | --- | --- | --- | --- | --- | --- |
| occupation_dw | h0 |  12386 |     46 | 0.64036 | 0.44689 | 0.83384 |
| occupation_dw | equal |  12386 |     46 | 0.60135 | 0.24785 | 0.95485 |
| filer_state_fixed_composition | h0 |  11691 |     46 | 0.63931 | 0.44372 | 0.8349 |
| filer_state_fixed_composition | equal |  11691 |     46 | 0.62181 | 0.28737 | 0.95624 |
| within_filer_across_state | h0 |   1903 |     46 | 0.40783 | 0.13533 | 0.68033 |
| within_filer_across_state | equal |   1903 |     46 | 0.18321 | 0.076783 | 0.28965 |

The continuing filer–state–occupation estimate is 0.640 dollars per statutory dollar, with a 95% interval [0.447, 0.834]. A larger statutory cut is associated with a larger offer reduction.

Offers are nominal hourly dollars. The main contrast is FY2025–FY2026, using decisions through June 30 in each fiscal year and baseline certified-hours weights. Former AEWR-region effects absorb regional differences; state deletion jackknife intervals use t(G−1). R7 identifies filers, not worksite businesses or individual farms. Composition checks condition on occupations surviving in both periods.

Effective job-level legal floors remain unresolved. The reference-schedule envelope cannot establish worker category, skill, job-order submission timing, prevailing wage or a market wage. The result does not by itself identify realized pay, employment, monopsony or welfare.

### Linkage and pre-policy comparisons

| tier | tau | status | n | states | estimate | low | high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R7 |   2024 | estimated |   9492 |     46 | 0.017212 | -0.032281 | 0.066704 |
| R7 |   2025 | estimated |  11430 |     46 | 0.013896 | -0.036452 | 0.064245 |
| R7 |   2026 | estimated |  12386 |     46 | 0.64036 | 0.44689 | 0.83384 |
| R5 |   2024 | estimated |   9398 |     46 | 0.018655 | -0.032002 | 0.069312 |
| R5 |   2025 | estimated |  11324 |     46 | 0.013963 | -0.036795 | 0.06472 |
| R5 |   2026 | estimated |  12287 |     46 |  0.639 | 0.44591 | 0.83209 |
| R8 |   2024 | estimated |   9589 |     46 | 0.01769 | -0.033222 | 0.068602 |
| R8 |   2025 | estimated |  11580 |     46 | 0.013581 | -0.037162 | 0.064325 |
| R8 |   2026 | estimated |  12485 |     46 | 0.63837 | 0.44503 | 0.83172 |
| FEIN |   2024 | insufficient_support | 0 | 0 | — | — | — |
| FEIN |   2025 | estimated |  11671 |     46 | 0.013203 | -0.035299 | 0.061705 |
| FEIN |   2026 | estimated |  12507 |     46 | 0.63957 | 0.4456 | 0.83354 |

### Composition, selection and applicability

| tier | tau | continuing_cells | unrestricted_filer_states | common_occupation_filer_states | baseline_hours_coverage | current_hours_coverage | mean_unrestricted | mean_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R7 |   2024 |   9492 |  10647 |   9152 | 0.92202 | 0.91299 | 1.1532 | 1.0652 |
| R7 |   2025 |  11430 |  11477 |  10775 | 0.95369 | 0.9502 | 0.66102 | 0.70995 |
| R7 |   2026 |  12386 |  12313 |  11691 | 0.95732 | 0.95914 | -2.3748 | -2.3517 |
| R5 |   2024 |   9398 |  10552 |   9068 | 0.92144 | 0.91246 | 1.1527 | 1.0643 |
| R5 |   2025 |  11324 |  11379 |  10679 | 0.95368 | 0.94976 | 0.65896 | 0.70882 |
| R5 |   2026 |  12287 |  12227 |  11606 | 0.95738 | 0.95923 | -2.3763 | -2.3546 |
| R8 |   2024 |   9589 |  10752 |   9242 | 0.91975 | 0.90815 | 1.1561 | 1.0583 |
| R8 |   2025 |  11580 |  11607 |  10902 | 0.95412 | 0.95164 | 0.67101 | 0.71468 |
| R8 |   2026 |  12485 |  12406 |  11781 | 0.95766 | 0.95893 | -2.3778 | -2.3534 |
| FEIN |   2024 | 0 | 0 | 0 | — | — | — | — |
| FEIN |   2025 |  11671 |  11703 |  10997 | 0.95402 | 0.95219 | 0.68002 | 0.71693 |
| FEIN |   2026 |  12507 |  12446 |  11813 | 0.95784 | 0.95788 | -2.3802 | -2.3552 |

| fiscal_year | hours | case_SOC_known_share | row_SOC_missing_share | candidate_coverage | near_any_candidate | baseline_minimum_exceeds_AEWR |
| --- | --- | --- | --- | --- | --- | --- |
|   2023 | 256517516 |      1 | 0.56343 | 0.87463 | 0.69048 | 0 |
|   2024 | 266526316 |      1 | 0.55688 | 0.78874 | 0.69353 | 0 |
|   2025 | 276199095 |      1 | 0.56789 | 0.79288 | 0.73725 | 0 |
|   2026 | 322238712 |      1 | 0.58127 | 0.67669 | 0.85193 | 0.23673 |

| specification | weights | term | status | estimate | low | high | p_holm | low_simultaneous | high_simultaneous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selection_stay | h0 | dose | estimated | -0.0036107 | -0.014684 | 0.0074629 | — | — | — |
| selection_any_cert | h0 | dose | estimated | -0.0040229 | -0.015523 | 0.0074773 | — | — | — |
| selection_stay | equal | dose | estimated | -0.0037002 | -0.013354 | 0.005953 | — | — | — |
| selection_any_cert | equal | dose | estimated | -0.0045592 | -0.016445 | 0.0073267 | — | — | — |
| selection_baseline_premium | equal | stay | estimated | -0.40424 | -1.0058 | 0.19728 | — | — | — |
| selection_baseline_premium | equal | dose:stay | estimated | -0.024256 | -0.16531 | 0.1168 | — | — | — |
| selection_prechange | equal | stay | estimated | 0.42687 | -0.032846 | 0.88658 | — | — | — |
| selection_prechange | equal | dose:stay | estimated | 0.086095 | -0.01018 | 0.18237 | — | — | — |
| heterogeneity_highpremium | h0 | dose:highpremium | estimated | -0.20456 | -1.2274 | 0.8183 |      1 | -1.4675 | 1.0584 |
| heterogeneity_contractor | h0 | dose:contractor | estimated | 0.27043 | -0.057195 | 0.59805 | 0.3101 | -0.13408 | 0.67494 |
| heterogeneity_large | h0 | dose:large | estimated | 0.032383 | -0.19287 | 0.25764 |      1 | -0.24573 | 0.3105 |

## Covered crop earnings: 2025Q4–2026Q1

The primary interim outcome averages two year-over-year log changes in private crop-production (NAICS 111) average weekly earnings. Every county needs both current/prior pairs, positive disclosed earnings and employment of at least ten in all four observations. The frozen 2022–2024 H-2A intensity interacts with the statutory cut, with intensity and state effects included, equal county weights, and state deletion jackknife inference. States require at least two eligible counties.

| window | n | states | estimate | low | high | percent_at_S03_cut_minus4 | percent_low | percent_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| interim_2025Q4_2026Q1 |   1457 |     47 | 0.016736 | 0.0020924 | 0.031379 | -1.9882 | -3.6954 | -0.25078 |
| quarter_2025Q4 |   1543 |     47 | 0.031195 | -0.0025335 | 0.064923 | -3.6742 | -7.495 | 0.30448 |
| quarter_2026Q1 |   1562 |     47 | -0.0025112 | -0.035111 | 0.030088 | 0.30179 | -3.5462 | 4.3033 |
| quarter_2025Q4_common_sample |   1457 |     47 | 0.031304 | -0.00041815 | 0.063026 | -3.6868 | -7.2842 | 0.050191 |
| quarter_2026Q1_common_sample |   1457 |     47 | 0.0021672 | -0.024023 | 0.028358 | -0.25972 | -3.3457 | 2.9248 |
| prior_window_lead |   1421 |     47 | -0.015019 | -0.037489 | 0.0074518 | 1.8186 | -0.89023 | 4.6015 |

At exposure 0.30 and a −$4 cut, the primary estimate implies -1.99% relative earnings change [-3.70%, -0.25%]. This is a model-implied relative difference, not the overall change in earnings.

### Sensitivities and comparison industries

| specification | weights | status | n | states | estimate | low | high | percent_at_S03_cut_minus4 | percent_low | percent_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crop_primary | equal | estimated |   1457 |     47 | 0.016736 | 0.0020924 | 0.031379 | -1.9882 | -3.6954 | -0.25078 |
| crop_employment_weighted | emp_l | estimated |   1457 |     47 | 0.0049824 | -0.014692 | 0.024657 | -0.5961 | -2.9155 | 1.7787 |
| crop_cash_dose | equal | estimated |   1457 |     47 | 0.014554 | -0.0017982 | 0.030907 | -1.7314 | -3.6409 | 0.21602 |
| crop_commuting_zone | equal | estimated |   1457 |     47 | 0.016888 | -0.0047174 | 0.038493 | -2.0061 | -4.5141 | 0.5677 |
| crop_no_intensity_control | equal | estimated |   1457 |     47 | 0.0012405 | -0.0029295 | 0.0054105 | -0.14875 | -0.64715 | 0.35216 |
| crop_UI_covered | equal | insufficient_support |     78 |      2 | — | — | — | — | — | — |
| crop_UI_excluded | equal | estimated |    142 |      3 | -0.0086976 | -4.6583 | 4.6409 | 1.0492 | -99.619 |  26674 |
| crop_UI_ambiguous | equal | estimated |   1237 |     42 | 0.015631 | 0.00099937 | 0.030263 | -1.8582 | -3.5664 | -0.11985 |
| animal_primary | equal | estimated |   1216 |     45 | -0.0024889 | -0.041027 | 0.036049 | 0.29911 | -4.2336 | 5.0464 |
| all_private_primary | equal | estimated |   3011 |     47 | 0.0032012 | -0.0019202 | 0.0083226 | -0.3834 | -0.99374 | 0.2307 |

The employment-weighted, cash-dose and commuting-zone intervals include zero in this frozen vintage. Removing the intensity control brings the point estimate near zero. Quarter-specific estimates suggest concentration in Q4 but do not establish a distinct timing pattern. These findings are suggestive and specification-sensitive; quiet placebos do not establish exogeneity.

Weekly earnings combine pay rates, hours and workforce composition. Incomplete UI classification and sparse covered/excluded groups do not identify a general domestic-worker spillover. Animal production and all-private outcomes are diagnostics, not guaranteed unaffected controls. Differential state shocks and contemporary policy changes remain identification concerns.

### Matched pre-policy windows

| tau | n | states | estimate | low | high |
| --- | --- | --- | --- | --- | --- |
|   2016 |   1346 |     47 | 0.0068481 | -0.016224 | 0.02992 |
|   2017 |   1376 |     47 | -0.0005876 | -0.026219 | 0.025044 |
|   2018 |   1380 |     48 | 0.012238 | -0.0071846 | 0.031661 |
|   2019 |   1401 |     48 | -0.0099671 | -0.04212 | 0.022186 |
|   2020 |   1428 |     48 | 0.0060872 | -0.01747 | 0.029645 |
|   2021 |   1466 |     48 | -0.0079512 | -0.026278 | 0.010376 |
|   2022 |   1491 |     48 | -0.0064743 | -0.037908 | 0.024959 |
|   2023 |   1485 |     48 | 0.0089402 | -0.01708 | 0.03496 |
|   2024 |   1504 |     47 | -0.006245 | -0.025919 | 0.013428 |

### Coverage and outstanding series

| series | window | tau | counties | states | mean_change | mean_exposure |
| --- | --- | --- | --- | --- | --- | --- |
| crop | interim_2025Q4_2026Q1 |   2026 |   1457 |     47 | 0.03401 | 0.1896 |
| animal | interim_2025Q4_2026Q1 |   2026 |   1216 |     45 | 0.038288 | 0.11314 |
| all_private | interim_2025Q4_2026Q1 |   2026 |   3011 |     47 | 0.039388 | 0.13435 |
| crop | quarter_2025Q4 |   2026 |   1543 |     47 | 0.041767 | 0.18452 |
| animal | quarter_2025Q4 |   2026 |   1312 |     46 | 0.048408 | 0.11545 |
| all_private | quarter_2025Q4 |   2026 |   3012 |     47 | 0.04502 | 0.13432 |
| crop | quarter_2026Q1 |   2026 |   1562 |     47 | 0.026504 | 0.18482 |
| animal | quarter_2026Q1 |   2026 |   1318 |     46 | 0.030232 | 0.11664 |
| all_private | quarter_2026Q1 |   2026 |   3013 |     47 | 0.033885 | 0.13428 |
| crop | quarter_2025Q4_common_sample |   2026 |   1457 |     47 | 0.041457 | 0.1896 |
| crop | quarter_2026Q1_common_sample |   2026 |   1457 |     47 | 0.026563 | 0.1896 |
| crop | prior_window_lead |   2025 |   1421 |     47 | 0.027073 | 0.19218 |
| crop | matched_two_quarter_placebo |   2016 |   1346 |     47 | 0.033749 | 0.20157 |
| crop | matched_two_quarter_placebo |   2017 |   1376 |     47 | 0.028027 | 0.19473 |
| crop | matched_two_quarter_placebo |   2018 |   1380 |     48 | 0.031173 | 0.19458 |
| crop | matched_two_quarter_placebo |   2019 |   1401 |     48 | 0.016943 | 0.18919 |
| crop | matched_two_quarter_placebo |   2020 |   1428 |     48 | 0.037668 | 0.19014 |
| crop | matched_two_quarter_placebo |   2021 |   1466 |     48 | 0.055491 | 0.18848 |
| crop | matched_two_quarter_placebo |   2022 |   1491 |     48 | 0.058911 | 0.1892 |
| crop | matched_two_quarter_placebo |   2023 |   1485 |     48 | 0.061741 | 0.18864 |
| crop | matched_two_quarter_placebo |   2024 |   1504 |     47 | 0.033625 | 0.18755 |

| specification | status | detail |
| --- | --- | --- |
| QWI_secondary | missing_declared_interim_quarters | 2025Q4,2026Q1,2025Q1 |

The original three-quarter target still requires 2026Q2 and verified publication. No missing quarter is imputed. The authorized interim analysis already inspected outcomes; later windows are updates, not independent confirmations. The historical full-window command and its release guard remain preserved in the exploration archive, outside this runner.

## Reproduction and evidence

```sh
./scripts/run_policy_2025.sh
```

The runner reads cached inputs only. Its source-vintage guard stops on drift. [Design and commands](../../code/designs/policy_2025/README.md), [complete coefficient ledger](policy_2025_specifications.md), [provenance](policy_2025/provenance.json), and [artifact validation](policy_2025/validation.json) accompany this report. State deletions, coverage, floor and allocation audits are retained under `outputs/tables/policy_2025/`. Earlier exploration reports remain historical records.

All 347 wage, 135 pre-policy earnings and 46 interim coefficient rows were compared with the inspected reference ledgers, including sample sizes, statuses and inference, at absolute tolerance 1e−10. Independent wage WLS and earnings OLS and the 12 interim checks also passed. Passing validation establishes computational consistency, not causal identification.

