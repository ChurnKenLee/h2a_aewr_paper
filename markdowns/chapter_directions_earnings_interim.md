# Interim agricultural earnings: authorized partial-window analysis

Generated 2026-11-14 23:13:11.503771. This dated addendum supersedes the earlier report's deferred-earnings status for the available quarters. The user authorized partial-window estimates before these outcomes were opened. The original three-quarter target remains a later update; it is no longer wholly uninspected.

## Files and available window

No upload or new download was needed. The existing `data/raw/qcew_episode_2025_2026/2025_qtrly_singlefile.zip` supplies 2025Q1/Q2 for the prior-window lead and 2025Q4 for the first post-policy quarter. The adjacent `2026_qtrly_singlefile.zip` supplies 2026Q1. Prior-year comparisons use the already available 2023/2024 history. Both new archive hashes match their acquisition manifest; retrieved 2026-11-14 UTC, source Last-Modified 2026-08-21.

The primary interim outcome is the mean of YoY log average-weekly-earnings changes in **2025Q4 and 2026Q1**, requiring all four current/prior observations for every county. It retains private crop production (NAICS 111), positive disclosed earnings, employment at least ten, the frozen H-2A exposure and statutory dose, intensity control, state effects, equal county weights and state deletion jackknife t intervals. This changes the time window and potentially the sample from the original three-quarter target.

## Main results

A positive coefficient means earnings fell more where H-2A exposure and the statutory cut were greater. The percent translation uses exposure S=0.30 and a −$4 statutory cut: 100 × [exp(−1.2β)−1]. All intervals are 95% state jackknife intervals. These are covered-job earnings, which combine hourly pay, hours and workforce composition.

| window | n | states | estimate | low | high | percent_at_S03_cut_minus4 | percent_low | percent_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| interim_2025Q4_2026Q1 |  1457 |    47 | 0.01674 | 0.002092 | 0.03138 | -1.988 | -3.695 | -0.2508 |
| quarter_2025Q4 |  1543 |    47 | 0.03119 | -0.002534 | 0.06492 | -3.674 | -7.495 | 0.3045 |
| quarter_2026Q1 |  1562 |    47 | -0.002511 | -0.03511 | 0.03009 | 0.3018 | -3.546 | 4.303 |
| quarter_2025Q4_common_sample |  1457 |    47 | 0.0313 | -0.0004182 | 0.06303 | -3.687 | -7.284 | 0.05019 |
| quarter_2026Q1_common_sample |  1457 |    47 | 0.002167 | -0.02402 | 0.02836 | -0.2597 | -3.346 | 2.925 |
| prior_window_lead |  1421 |    47 | -0.01502 | -0.03749 | 0.007452 | 1.819 | -0.8902 | 4.601 |

At the stated exposure and cut, the two-quarter estimate is -1.99%, with a 95% interval of [-3.70%, -0.25%]. The interval excludes zero in this specification; the identifying assumptions still require scrutiny.

Quarter-specific rows use each quarter's eligible counties. The rows ending `common_sample` use exactly the two-quarter counties, so their difference is not driven by changes in county eligibility. The prior-window lead measures 2024Q4–2025Q2, before the reform; it is a falsification diagnostic, not a reform effect.

## Sensitivities and comparison industries

| specification | weights | status | n | states | estimate | low | high | percent_at_S03_cut_minus4 | percent_low | percent_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crop_primary | equal | estimated |  1457 |    47 | 0.01674 | 0.002092 | 0.03138 | -1.988 | -3.695 | -0.2508 |
| crop_employment_weighted | emp_l | estimated |  1457 |    47 | 0.004982 | -0.01469 | 0.02466 | -0.5961 | -2.916 | 1.779 |
| crop_cash_dose | equal | estimated |  1457 |    47 | 0.01455 | -0.001798 | 0.03091 | -1.731 | -3.641 | 0.216 |
| crop_commuting_zone | equal | estimated |  1457 |    47 | 0.01689 | -0.004717 | 0.03849 | -2.006 | -4.514 | 0.5677 |
| crop_no_intensity_control | equal | estimated |  1457 |    47 | 0.00124 | -0.002929 | 0.00541 | -0.1487 | -0.6472 | 0.3522 |
| crop_UI_covered | equal | insufficient_support |    78 |     2 | — | — | — | — | — | — |
| crop_UI_excluded | equal | estimated |   142 |     3 | -0.008698 | -4.658 | 4.641 | 1.049 | -99.62 | 26674 |
| crop_UI_ambiguous | equal | estimated |  1237 |    42 | 0.01563 | 0.0009994 | 0.03026 | -1.858 | -3.566 | -0.1199 |
| animal_primary | equal | estimated |  1216 |    45 | -0.002489 | -0.04103 | 0.03605 | 0.2991 | -4.234 | 5.046 |
| all_private_primary | equal | estimated |  3011 |    47 | 0.003201 | -0.00192 | 0.008323 | -0.3834 | -0.9937 | 0.2307 |

The employment-weighted estimate is -0.60% [-2.92%, 1.78%]. The equal-county primary therefore does not establish a precise employment-weighted response. The separate-quarter point estimates concentrate the decline in Q4; their broad intervals do not establish a statistically distinct timing pattern. Cash-dose and commuting-zone intervals also include zero. This is suggestive interim evidence, sensitive to specification and weighting, with continuing state-level confounding concerns.

The UI classification remains frozen and incomplete; sparse covered/excluded groups cannot establish a general non-H-2A earnings effect. Animal production and all-private earnings are diagnostics, not guaranteed unaffected controls. Employment weights and commuting-zone exposure change the comparison and are reported alongside the equal-county primary.

## Matched pre-policy windows

| tau | n | states | estimate | low | high |
| --- | --- | --- | --- | --- | --- |
|  2016 |  1346 |    47 | 0.006848 | -0.01622 | 0.02992 |
|  2017 |  1376 |    47 | -0.0005876 | -0.02622 | 0.02504 |
|  2018 |  1380 |    48 | 0.01224 | -0.007185 | 0.03166 |
|  2019 |  1401 |    48 | -0.009967 | -0.04212 | 0.02219 |
|  2020 |  1428 |    48 | 0.006087 | -0.01747 | 0.02964 |
|  2021 |  1466 |    48 | -0.007951 | -0.02628 | 0.01038 |
|  2022 |  1491 |    48 | -0.006474 | -0.03791 | 0.02496 |
|  2023 |  1485 |    48 | 0.00894 | -0.01708 | 0.03496 |
|  2024 |  1504 |    47 | -0.006245 | -0.02592 | 0.01343 |

These placebo windows use the same Q4/Q1 timing, thresholds and primary model. Quiet placebos do not establish exogeneity. The post-policy and quarter-specific estimates are multiple looks, and subsequent full-window results will be updates rather than independent confirmations.

## Complete declared interim result ledger

| window | tau | specification | weights | status | n | states | estimate | se_jackknife | low | high | p | se_crv1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| interim_2025Q4_2026Q1 |  2026 | crop_primary | equal | estimated |  1457 |    47 | 0.01674 | 0.007275 | 0.002092 | 0.03138 | 0.026 | 0.006877 |
| interim_2025Q4_2026Q1 |  2026 | crop_employment_weighted | emp_l | estimated |  1457 |    47 | 0.004982 | 0.009774 | -0.01469 | 0.02466 | 0.6127 | 0.006754 |
| interim_2025Q4_2026Q1 |  2026 | crop_cash_dose | equal | estimated |  1457 |    47 | 0.01455 | 0.008124 | -0.001798 | 0.03091 | 0.07979 | 0.007428 |
| interim_2025Q4_2026Q1 |  2026 | crop_commuting_zone | equal | estimated |  1457 |    47 | 0.01689 | 0.01073 | -0.004717 | 0.03849 | 0.1225 | 0.009425 |
| interim_2025Q4_2026Q1 |  2026 | crop_no_intensity_control | equal | estimated |  1457 |    47 | 0.00124 | 0.002072 | -0.002929 | 0.00541 | 0.5522 | 0.001961 |
| interim_2025Q4_2026Q1 |  2026 | crop_UI_covered | equal | insufficient_support |    78 |     2 | — | — | — | — | — | — |
| interim_2025Q4_2026Q1 |  2026 | crop_UI_excluded | equal | estimated |   142 |     3 | -0.008698 | 1.081 | -4.658 | 4.641 | 0.9943 | 0.01268 |
| interim_2025Q4_2026Q1 |  2026 | crop_UI_ambiguous | equal | estimated |  1237 |    42 | 0.01563 | 0.007245 | 0.0009994 | 0.03026 | 0.03688 | 0.006779 |
| interim_2025Q4_2026Q1 |  2026 | animal_primary | equal | estimated |  1216 |    45 | -0.002489 | 0.01912 | -0.04103 | 0.03605 | 0.897 | 0.01765 |
| interim_2025Q4_2026Q1 |  2026 | all_private_primary | equal | estimated |  3011 |    47 | 0.003201 | 0.002544 | -0.00192 | 0.008323 | 0.2147 | 0.001969 |
| quarter_2025Q4 |  2026 | crop_primary | equal | estimated |  1543 |    47 | 0.03119 | 0.01676 | -0.002534 | 0.06492 | 0.06904 | 0.0142 |
| quarter_2025Q4 |  2026 | crop_employment_weighted | emp_l | estimated |  1543 |    47 | 0.00619 | 0.01137 | -0.01669 | 0.02907 | 0.5888 | 0.01008 |
| quarter_2025Q4 |  2026 | crop_cash_dose | equal | estimated |  1543 |    47 | 0.02556 | 0.0198 | -0.0143 | 0.06543 | 0.2033 | 0.01711 |
| quarter_2025Q4 |  2026 | crop_commuting_zone | equal | estimated |  1543 |    47 | 0.02827 | 0.02174 | -0.01549 | 0.07202 | 0.1999 | 0.01789 |
| quarter_2025Q4 |  2026 | crop_no_intensity_control | equal | estimated |  1543 |    47 | 0.003643 | 0.003728 | -0.00386 | 0.01115 | 0.3335 | 0.003462 |
| quarter_2025Q4 |  2026 | crop_UI_covered | equal | insufficient_support |    82 |     2 | — | — | — | — | — | — |
| quarter_2025Q4 |  2026 | crop_UI_excluded | equal | estimated |   152 |     3 | 0.04412 | 1.842 | -7.879 | 7.968 | 0.9831 | 0.0236 |
| quarter_2025Q4 |  2026 | crop_UI_ambiguous | equal | estimated |  1309 |    42 | 0.02965 | 0.01673 | -0.004138 | 0.06344 | 0.08379 | 0.01402 |
| quarter_2025Q4 |  2026 | animal_primary | equal | estimated |  1312 |    46 | 0.02341 | 0.02907 | -0.03515 | 0.08196 | 0.425 | 0.02737 |
| quarter_2025Q4 |  2026 | all_private_primary | equal | estimated |  3012 |    47 | 0.006493 | 0.003338 | -0.0002251 | 0.01321 | 0.05785 | 0.002746 |
| quarter_2026Q1 |  2026 | crop_primary | equal | estimated |  1562 |    47 | -0.002511 | 0.0162 | -0.03511 | 0.03009 | 0.8775 | 0.01264 |
| quarter_2026Q1 |  2026 | crop_employment_weighted | emp_l | estimated |  1562 |    47 | 0.00354 | 0.01779 | -0.03226 | 0.03934 | 0.8431 | 0.009354 |
| quarter_2026Q1 |  2026 | crop_cash_dose | equal | estimated |  1562 |    47 | 0.001122 | 0.01629 | -0.03166 | 0.03391 | 0.9454 | 0.01373 |
| quarter_2026Q1 |  2026 | crop_commuting_zone | equal | estimated |  1562 |    47 | 0.0007993 | 0.01429 | -0.02797 | 0.02957 | 0.9557 | 0.01274 |
| quarter_2026Q1 |  2026 | crop_no_intensity_control | equal | estimated |  1562 |    47 | -0.002523 | 0.002683 | -0.007924 | 0.002877 | 0.3518 | 0.002525 |
| quarter_2026Q1 |  2026 | crop_UI_covered | equal | insufficient_support |    79 |     2 | — | — | — | — | — | — |
| quarter_2026Q1 |  2026 | crop_UI_excluded | equal | estimated |   147 |     3 | -0.09125 | 0.8452 | -3.728 | 3.545 | 0.9239 | 0.01003 |
| quarter_2026Q1 |  2026 | crop_UI_ambiguous | equal | estimated |  1336 |    42 | -0.002413 | 0.01664 | -0.03601 | 0.03119 | 0.8854 | 0.0127 |
| quarter_2026Q1 |  2026 | animal_primary | equal | estimated |  1318 |    46 | -0.01841 | 0.02088 | -0.06046 | 0.02363 | 0.3824 | 0.01908 |
| quarter_2026Q1 |  2026 | all_private_primary | equal | estimated |  3013 |    47 | 0.0007224 | 0.002982 | -0.005281 | 0.006726 | 0.8097 | 0.00224 |
| quarter_2025Q4_common_sample |  2026 | crop_primary | equal | estimated |  1457 |    47 | 0.0313 | 0.01576 | -0.0004182 | 0.06303 | 0.05297 | 0.01354 |
| quarter_2026Q1_common_sample |  2026 | crop_primary | equal | estimated |  1457 |    47 | 0.002167 | 0.01301 | -0.02402 | 0.02836 | 0.8684 | 0.01087 |
| prior_window_lead |  2025 | crop_primary | equal | estimated |  1421 |    47 | -0.01502 | 0.01116 | -0.03749 | 0.007452 | 0.1851 | 0.009972 |
| prior_window_lead |  2025 | crop_employment_weighted | emp_l | estimated |  1421 |    47 | 0.002779 | 0.01782 | -0.03309 | 0.03865 | 0.8768 | 0.009957 |
| prior_window_lead |  2025 | crop_cash_dose | equal | estimated |  1421 |    47 | -0.01398 | 0.01259 | -0.03932 | 0.01136 | 0.2725 | 0.01099 |
| prior_window_lead |  2025 | crop_commuting_zone | equal | estimated |  1421 |    47 | -0.01591 | 0.0146 | -0.04531 | 0.01348 | 0.2816 | 0.01232 |
| prior_window_lead |  2025 | crop_no_intensity_control | equal | estimated |  1421 |    47 | -0.001862 | 0.002219 | -0.006329 | 0.002605 | 0.4058 | 0.002089 |
| matched_two_quarter_placebo |  2016 | crop_primary | equal | estimated |  1346 |    47 | 0.006848 | 0.01146 | -0.01622 | 0.02992 | 0.5531 | 0.009944 |
| matched_two_quarter_placebo |  2017 | crop_primary | equal | estimated |  1376 |    47 | -0.0005876 | 0.01273 | -0.02622 | 0.02504 | 0.9634 | 0.01131 |
| matched_two_quarter_placebo |  2018 | crop_primary | equal | estimated |  1380 |    48 | 0.01224 | 0.009655 | -0.007185 | 0.03166 | 0.2112 | 0.008591 |
| matched_two_quarter_placebo |  2019 | crop_primary | equal | estimated |  1401 |    48 | -0.009967 | 0.01598 | -0.04212 | 0.02219 | 0.5359 | 0.01242 |
| matched_two_quarter_placebo |  2020 | crop_primary | equal | estimated |  1428 |    48 | 0.006087 | 0.01171 | -0.01747 | 0.02964 | 0.6056 | 0.009998 |
| matched_two_quarter_placebo |  2021 | crop_primary | equal | estimated |  1466 |    48 | -0.007951 | 0.00911 | -0.02628 | 0.01038 | 0.3872 | 0.007672 |
| matched_two_quarter_placebo |  2022 | crop_primary | equal | estimated |  1491 |    48 | -0.006474 | 0.01562 | -0.03791 | 0.02496 | 0.6805 | 0.01229 |
| matched_two_quarter_placebo |  2023 | crop_primary | equal | estimated |  1485 |    48 | 0.00894 | 0.01293 | -0.01708 | 0.03496 | 0.4928 | 0.01126 |
| matched_two_quarter_placebo |  2024 | crop_primary | equal | estimated |  1504 |    47 | -0.006245 | 0.009774 | -0.02592 | 0.01343 | 0.526 | 0.008496 |

## Data coverage and outstanding series

| series | window | tau | counties | states | mean_change | mean_exposure |
| --- | --- | --- | --- | --- | --- | --- |
| crop | interim_2025Q4_2026Q1 |  2026 |  1457 |    47 | 0.03401 | 0.1896 |
| animal | interim_2025Q4_2026Q1 |  2026 |  1216 |    45 | 0.03829 | 0.1131 |
| all_private | interim_2025Q4_2026Q1 |  2026 |  3011 |    47 | 0.03939 | 0.1344 |
| crop | quarter_2025Q4 |  2026 |  1543 |    47 | 0.04177 | 0.1845 |
| animal | quarter_2025Q4 |  2026 |  1312 |    46 | 0.04841 | 0.1155 |
| all_private | quarter_2025Q4 |  2026 |  3012 |    47 | 0.04502 | 0.1343 |
| crop | quarter_2026Q1 |  2026 |  1562 |    47 | 0.0265 | 0.1848 |
| animal | quarter_2026Q1 |  2026 |  1318 |    46 | 0.03023 | 0.1166 |
| all_private | quarter_2026Q1 |  2026 |  3013 |    47 | 0.03388 | 0.1343 |
| crop | quarter_2025Q4_common_sample |  2026 |  1457 |    47 | 0.04146 | 0.1896 |
| crop | quarter_2026Q1_common_sample |  2026 |  1457 |    47 | 0.02656 | 0.1896 |
| crop | prior_window_lead |  2025 |  1421 |    47 | 0.02707 | 0.1922 |
| crop | matched_two_quarter_placebo |  2016 |  1346 |    47 | 0.03375 | 0.2016 |
| crop | matched_two_quarter_placebo |  2017 |  1376 |    47 | 0.02803 | 0.1947 |
| crop | matched_two_quarter_placebo |  2018 |  1380 |    48 | 0.03117 | 0.1946 |
| crop | matched_two_quarter_placebo |  2019 |  1401 |    48 | 0.01694 | 0.1892 |
| crop | matched_two_quarter_placebo |  2020 |  1428 |    48 | 0.03767 | 0.1901 |
| crop | matched_two_quarter_placebo |  2021 |  1466 |    48 | 0.05549 | 0.1885 |
| crop | matched_two_quarter_placebo |  2022 |  1491 |    48 | 0.05891 | 0.1892 |
| crop | matched_two_quarter_placebo |  2023 |  1485 |    48 | 0.06174 | 0.1886 |
| crop | matched_two_quarter_placebo |  2024 |  1504 |    47 | 0.03363 | 0.1876 |

| specification | status | detail |
| --- | --- | --- |
| QWI_secondary | missing_declared_interim_quarters | 2025Q4,2026Q1,2025Q1 |

The full 2025Q4–2026Q2 target still needs 2026Q2. No missing quarter is imputed. The complete pre-policy lead has now been estimated from the cached annual archive, removing the earlier separate-slice retrieval obstacle.

## Reproduction

```sh
bash explorations/chapter_directions/run_interim.sh /tmp/h2a-chapter-directions
```

See the [dated amendment](../explorations/chapter_directions/interim_amendment_2026_11_14.json) and [README](../explorations/chapter_directions/README.md). Scratch contains `earnings_interim_estimates.csv`, every state deletion, coverage, input hashes and quarterly inventory. Original frozen registries and their code are preserved. No acquisition pipeline or automatic schedule was run.

| check | status |
| --- | --- |
| two_quarter_mean_uses_both_year_pairs | passed |
| missing_declared_quarter_stops | passed |
| suppressed_quarter_excludes_county | passed |
| employment_threshold_applies_to_each_quarter | passed |
| unique_string_county_keys | passed |
| suppression_propagates | passed |
| full_prior_lead_matches_frozen_helper | passed |
| complete_declared_specification_grid | passed |
| independent_base_R_primary_reproduction | passed |
| common_quarter_samples_and_mean_coefficient | passed |
| intervals_and_unavailable_fits_retained | passed |
| original_pre_policy_results_preserved | passed |

