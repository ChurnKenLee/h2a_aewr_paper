# The 2025 H-2A policy shift

Generated 2026-11-27 05:34:41.043564. This standalone design uses the complete FY2026 H-2A disclosure (Q4 release) as its baseline since the November 21, 2026 update, with the November 25, 2026 unified minimum-wage vintage update (`history/vintage_update_2026_11_25.json`) restoring federal-floor agricultural minimum wages for GA/WY (2024-2026) and AL/LA/MS/SC/TN (2026) in the 48 reference-schedule envelope rows while leaving all other wage-offer and earnings specifications unchanged. Outcomes had already been inspected; this is not a new preregistration. The 2013 shutdown is a separate historical exploration.

## Offered wages

| specification | weights | n | states | estimate | low | high |
| --- | --- | --- | --- | --- | --- | --- |
| occupation_dw | h0 |  13463 |     46 | 0.62289 | 0.42222 | 0.82356 |
| occupation_dw | equal |  13463 |     46 | 0.59961 | 0.26623 | 0.93299 |
| filer_state_fixed_composition | h0 |  12540 |     46 | 0.62014 | 0.41801 | 0.82228 |
| filer_state_fixed_composition | equal |  12540 |     46 | 0.62474 | 0.29961 | 0.94986 |
| within_filer_across_state | h0 |   2024 |     46 | 0.44169 | 0.20486 | 0.67852 |
| within_filer_across_state | equal |   2024 |     46 | 0.18296 | 0.092449 | 0.27347 |

The continuing filer–state–occupation estimate is 0.623 dollars per statutory dollar, with a 95% interval [0.422, 0.824]. A larger statutory cut is associated with a larger offer reduction.

Offers are nominal hourly dollars. The main contrast is FY2025–FY2026, using decisions over the full fiscal year (October 1 through September 30) and baseline certified-hours weights. FY2026 cases received on or after August 3, 2026 fall under the second statewide schedule; the dose and reference rates remain those of October 2, 2025. Former AEWR-region effects absorb regional differences; state deletion jackknife intervals use t(G−1). R7 identifies filers, not worksite businesses or individual farms. Composition checks condition on occupations surviving in both periods.

Effective job-level legal floors remain unresolved. The reference-schedule envelope cannot establish worker category, skill, job-order submission timing, prevailing wage or a market wage. The result does not by itself identify realized pay, employment, monopsony or welfare.

### Linkage and pre-policy comparisons

| tier | tau | status | n | states | estimate | low | high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R7 |   2024 | estimated |  10411 |     46 | 0.02632 | -0.016617 | 0.069257 |
| R7 |   2025 | estimated |  12442 |     46 | 0.078721 | -0.12316 | 0.2806 |
| R7 |   2026 | estimated |  13463 |     46 | 0.62289 | 0.42222 | 0.82356 |
| R5 |   2024 | estimated |  10310 |     46 | 0.027965 | -0.015868 | 0.071798 |
| R5 |   2025 | estimated |  12333 |     46 | 0.079794 | -0.12369 | 0.28328 |
| R5 |   2026 | estimated |  13374 |     46 | 0.62274 | 0.41843 | 0.82704 |
| R8 |   2024 | estimated |  10519 |     46 | 0.027793 | -0.018315 | 0.0739 |
| R8 |   2025 | estimated |  12594 |     46 | 0.077841 | -0.1247 | 0.28038 |
| R8 |   2026 | estimated |  13563 |     46 | 0.6205 | 0.42051 | 0.8205 |
| FEIN |   2024 | insufficient_support | 0 | 0 | — | — | — |
| FEIN |   2025 | estimated |  12691 |     46 | 0.076812 | -0.12127 | 0.2749 |
| FEIN |   2026 | estimated |  13601 |     46 | 0.62433 | 0.42525 | 0.82341 |

### Composition, selection and applicability

| tier | tau | continuing_cells | unrestricted_filer_states | common_occupation_filer_states | baseline_hours_coverage | current_hours_coverage | mean_unrestricted | mean_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R7 |   2024 |  10411 |  11414 |   9894 | 0.92545 | 0.91514 |  1.129 | 1.0448 |
| R7 |   2025 |  12442 |  12302 |  11577 |  0.956 | 0.95091 | 0.63237 | 0.68666 |
| R7 |   2026 |  13463 |  13182 |  12540 | 0.95922 | 0.96082 | -2.3949 | -2.3803 |
| R5 |   2024 |  10310 |  11314 |   9806 | 0.92528 | 0.91488 | 1.1284 | 1.0438 |
| R5 |   2025 |  12333 |  12204 |  11480 | 0.95594 | 0.95053 | 0.63053 | 0.68533 |
| R5 |   2026 |  13374 |  13105 |  12466 | 0.95909 | 0.96111 | -2.3959 | -2.3824 |
| R8 |   2024 |  10519 |  11523 |   9992 | 0.92356 | 0.91113 | 1.1327 |   1.04 |
| R8 |   2025 |  12594 |  12428 |  11702 | 0.95631 | 0.9524 | 0.63931 | 0.68891 |
| R8 |   2026 |  13563 |  13273 |  12628 | 0.95969 | 0.96095 | -2.3993 | -2.3836 |
| FEIN |   2024 | 0 | 0 | 0 | — | — | — | — |
| FEIN |   2025 |  12691 |  12523 |  11799 | 0.95616 | 0.95288 | 0.64755 | 0.69063 |
| FEIN |   2026 |  13601 |  13323 |  12672 | 0.95979 | 0.95914 | -2.3985 | -2.3891 |

| fiscal_year | hours | case_SOC_known_share | row_SOC_missing_share | candidate_coverage | near_any_candidate | baseline_minimum_exceeds_AEWR |
| --- | --- | --- | --- | --- | --- | --- |
|   2023 | 291571260 |      1 | 0.56689 | 0.86872 | 0.72373 | 0 |
|   2024 | 301450995 |      1 | 0.56303 | 0.8563 | 0.73211 | 0 |
|   2025 | 315226304 |      1 | 0.5752 | 0.85458 | 0.77183 | 0 |
|   2026 | 363341244 |      1 | 0.58316 | 0.85832 | 0.8313 | 0.1761 |

| specification | weights | term | status | estimate | low | high | p_holm | low_simultaneous | high_simultaneous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selection_stay | h0 | dose | estimated | -0.0018129 | -0.0090848 | 0.0054591 | — | — | — |
| selection_any_cert | h0 | dose | estimated | -0.0022012 | -0.009548 | 0.0051455 | — | — | — |
| selection_stay | equal | dose | estimated | -0.005179 | -0.019223 | 0.0088655 | — | — | — |
| selection_any_cert | equal | dose | estimated | -0.0062876 | -0.021839 | 0.0092641 | — | — | — |
| selection_baseline_premium | equal | stay | estimated | -0.50946 | -1.1415 | 0.12257 | — | — | — |
| selection_baseline_premium | equal | dose:stay | estimated | -0.050413 | -0.19132 | 0.090492 | — | — | — |
| selection_prechange | equal | stay | estimated | 0.36644 | -0.08255 | 0.81544 | — | — | — |
| selection_prechange | equal | dose:stay | estimated | 0.06222 | -0.034382 | 0.15882 | — | — | — |
| heterogeneity_highpremium | h0 | dose:highpremium | estimated | -0.23221 | -1.0705 | 0.60604 |      1 | -1.2672 | 0.80277 |
| heterogeneity_contractor | h0 | dose:contractor | estimated | 0.25822 | -0.0035085 | 0.51996 | 0.15906 | -0.064933 | 0.58138 |
| heterogeneity_large | h0 | dose:large | estimated | 0.019968 | -0.20346 | 0.2434 |      1 | -0.2559 | 0.29583 |

## County H-2A quantity response (Track 1, FY2025–FY2026)

Added on November 26, 2026 (`history/quantity_amendment_2026_11_26.json`) after prior FY2026 Q1–Q3 and Q4 outcomes had been inspected. Estimates cover all 2,921 positive-2011-farm-employment counties in the 15 multi-state former AEWR regions (46 states) with County and Former-Region × FY fixed effects and exact delete-one-state jackknife (CV3J, t(45)) inference. The national level shift is absorbed; only within-former-region dispersion in the October 2, 2025 statewide rate identifies the slope.

| specification | outcome | estimator | states | estimate | se_jackknife | p_jackknife | elasticity | elasticity_low_jk | elasticity_high_jk | mde_elasticity_jk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B_primary | hours | PPML |     46 | -0.0050697 | 0.016835 | 0.7647 | -0.077841 | -0.59847 | 0.44279 | 0.72378 |
| B_primary | cert | PPML |     46 | 0.0041397 | 0.012157 | 0.73504 | 0.063563 | -0.31238 | 0.43951 | 0.52264 |
| B_primary | y_hours | OLS |     46 | 2.9283 | 5.4708 | 0.59511 | 0.26128 | -0.72189 | 1.2445 | 1.3668 |
| B_primary | y_cert | OLS |     46 | 0.0059559 | 0.0068758 | 0.39097 | 0.54371 | -0.72052 |  1.808 | 1.7575 |
| B_primary_log_rate | hours | PPML |     46 | -0.062019 | 0.19364 | 0.75023 | -0.062019 | -0.45202 | 0.32799 | 0.54218 |
| B_primary_log_rate | cert | PPML |     46 | 0.05665 | 0.1351 | 0.67697 | 0.05665 | -0.21545 | 0.32875 | 0.37828 |
| B_primary_first_stage | y_wage | OLS |     46 | 0.67138 | 0.098798 | 0.000000020497 | — | — | — | — |

On the complete FY2026 Q4 disclosure (full fiscal year October 1–September 30), the size-weighted PPML certified-hours statutory elasticity is -0.078 with a 95% state-jackknife interval [-0.598, 0.443] (MDE 0.72). This provides a credible one-season lower bound on elastic demand rather than a tightly pinned point estimate.

### Cutoffs, Q1–Q3 benchmark, and pre-policy placebos (PPML certified hours)

| window | contrast | specification | states | estimate | se_jackknife | elasticity | elasticity_low_jk | elasticity_high_jk | mde_elasticity_jk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_fy | fy2025_2026 | B_primary |     46 | -0.0050697 | 0.016835 | -0.077841 | -0.59847 | 0.44279 | 0.72378 |
| full_fy | placebo_fy2024_2025 | placebo_fy2024_fy2025 |     46 | -0.0021655 | 0.0086276 | -0.032505 | -0.29334 | 0.22833 | 0.36261 |
| q1_q3 | fy2025_2026 | B_primary |     46 | -0.012873 | 0.026082 | -0.19768 | -1.0044 | 0.60902 | 1.1215 |
| q1_q3 | placebo_fy2024_2025 | placebo_fy2024_fy2025 |     46 | -0.0052837 | 0.017402 | -0.079304 | -0.60537 | 0.44676 | 0.73134 |
| pre_aug03_decision | fy2025_2026 | B_primary |     46 | -0.010904 | 0.021126 | -0.16741 | -0.82071 | 0.48589 | 0.90822 |
| pre_aug03_receipt | fy2025_2026 | B_primary |     46 | -0.011727 | 0.020877 | -0.18006 | -0.82572 | 0.46559 | 0.89759 |
| pre_aug26_decision | fy2025_2026 | B_primary |     46 | -0.011588 | 0.019753 | -0.17793 | -0.78877 | 0.43292 | 0.84919 |
| pre_aug26_receipt | fy2025_2026 | B_primary |     46 | -0.0073713 |  0.014 | -0.11318 | -0.54615 | 0.31978 | 0.60191 |

### FY2024–FY2026 window and legal-cutoff coverage audit

| fiscal_year | cases_full_fy | positions_all_fy | hours_all_fy | positions_full_fy | hours_full_fy | positions_q1_q3 | hours_q1_q3 | q4_positions_share | q4_hours_share | pre_2025_10_02_hours_share | ge_aug03_decision_hours_share | ge_aug03_receipt_hours_share | ge_aug26_decision_hours_share | ge_aug26_receipt_hours_share | ge_sep02_decision_hours_share | ge_sep02_receipt_hours_share | missing_weekly_hours_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|   2024 |  20069 | 297002 | 309018129 | 297002 | 309018129 | 246497 | 273221019 | 0.17005 | 0.11584 | — | 0.055538 | 0.030197 | 0.033827 | 0.012076 | 0.029297 | 0.0044579 | 0.008593 |
|   2025 |  21241 | 303579 | 321566454 | 303497 | 321443050 | 250730 | 281128274 | 0.17386 | 0.12542 | — | 0.066748 | 0.043933 | 0.040742 | 0.025432 | 0.03775 | 0.019381 | 0.0089418 |
|   2026 |  21546 | 330670 | 370224904 | 330670 | 370224904 | 280515 | 327631588 | 0.15168 | 0.11505 | 0.016197 | 0.070135 | 0.043077 | 0.044946 | 0.019265 | 0.036737 | 0.013623 | 0.0075998 |

### Post-hoc cost-elasticity conversion (central ratio 1.2401)

| window | outcome | pass_through_source | pi_pass_through | w_bar_offered | eps_statutory | eps_cost | eps_cost_low_jk | eps_cost_high_jk | mde_eps_cost_jk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_fy | hours | policy_2025_R7_occupation_dw_h0_fixed_pi | 0.62289 | 16.051 | -0.077841 | -0.162 | -1.2455 | 0.92151 | 1.5063 |
| full_fy | hours | track1_county_first_stage_fixed_pi | 0.67138 | 16.051 | -0.077841 | -0.1503 | -1.1556 | 0.85495 | 1.3975 |
| full_fy | hours | track1_county_first_stage_paired_state_jk | 0.67138 | 16.051 | -0.077841 | -0.1503 | -1.1485 | 0.84792 | 1.3877 |
| full_fy | cert | policy_2025_R7_occupation_dw_h0_fixed_pi | 0.62289 | 16.051 | 0.063563 | 0.13228 | -0.65012 | 0.91469 | 1.0877 |
| full_fy | cert | track1_county_first_stage_fixed_pi | 0.67138 | 16.051 | 0.063563 | 0.12273 | -0.60316 | 0.84862 | 1.0091 |
| full_fy | cert | track1_county_first_stage_paired_state_jk | 0.67138 | 16.051 | 0.063563 | 0.12273 | -0.59961 | 0.84507 | 1.0042 |
| q1_q3 | hours | policy_2025_R7_occupation_dw_h0_fixed_pi | 0.62289 | 16.065 | -0.19768 | -0.41172 | -2.0919 | 1.2684 | 2.3357 |
| q1_q3 | hours | track1_county_first_stage_fixed_pi | 0.66339 | 16.065 | -0.19768 | -0.38658 | -1.9641 |  1.191 | 2.1931 |
| q1_q3 | hours | track1_county_first_stage_paired_state_jk | 0.66339 | 16.065 | -0.19768 | -0.38658 | -1.9392 |  1.166 | 2.1584 |

## Mechanism heterogeneity (FY2025–FY2026)

Added on November 26, 2026 (`history/mechanisms_amendment_2026_11_26.json`) as a post-inspection mechanism audit on cached local inputs (not a preregistration). Estimates decompose the Track 1 county quantity response across five mechanism panels on the same 2,921-county, 46-state, 15-former-region frame with County and Former-Region × FY fixed effects, exact delete-one-state jackknife (CV3J) inference, paired 46-state jackknife subgroup contrasts, and Holm step-down multiplicity correction across the 12 declared primary FY2025–FY2026 hypotheses.

### Primary 12-hypothesis mechanism family (Holm-adjusted)

| panel | specification | estimator | states | estimate | se_jackknife | elasticity | elasticity_low_jk | elasticity_high_jk | mde_elasticity_jk | p_jackknife | p_holm_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1_pipeline | pipeline_requested_hours | PPML |     46 | 0.0037954 | 0.021526 | 0.05827 | -0.60736 | 0.7239 | 0.92536 | 0.86084 |      1 |
| P1_pipeline | pipeline_requested_positions | PPML |     46 | 0.010549 | 0.015846 | 0.16196 | -0.32805 | 0.65196 | 0.6812 |  0.509 |      1 |
| P1_pipeline | pipeline_approval_rate | OLS |     46 | -0.0062402 | 0.0082385 | — | — | — | — | 0.45273 |      1 |
| P2_incumbent_entry | wb_1yr_incumbent_hours | PPML |     46 | -0.005626 | 0.025219 | -0.0864 | -0.86644 | 0.69364 | 1.0844 | 0.82448 |      1 |
| P2_incumbent_entry | wb_1yr_entrant_hours | PPML |     46 | -0.013852 | 0.13112 | -0.21308 | -4.2752 | 3.8491 | 5.6472 | 0.91633 |      1 |
| P2_incumbent_entry | wb_1yr_diff_entrant_minus_incumbent_hours | PPML_DIFF_JK |     46 | -0.0082262 | 0.1541 | -0.12668 | -4.8998 | 4.6465 | 6.6356 | 0.95761 |      1 |
| P3_h2alc_direct | contractor_h2alc_hours | PPML |     46 | -0.006683 | 0.07424 | -0.10338 | -2.4165 | 2.2098 | 3.2157 | 0.92867 |      1 |
| P3_h2alc_direct | contractor_direct_hours | PPML |     46 | -0.0043163 | 0.012088 | -0.066273 | -0.44009 | 0.30754 | 0.51968 | 0.72271 |      1 |
| P3_h2alc_direct | contractor_diff_h2alc_minus_direct_hours | PPML_DIFF_JK |     46 | -0.0023667 | 0.069158 | -0.037111 | -2.1929 | 2.1187 |  2.997 | 0.97249 |      1 |
| P4_leadtime_crops | leadtime_late_start_hours | PPML |     46 | -0.043892 | 0.024358 | -0.67556 | -1.4307 | 0.079532 | 1.0497 | 0.078252 | 0.93903 |
| P4_leadtime_crops | leadtime_diff_late_minus_early_start_hours | PPML_DIFF_JK |     46 | -0.04747 | 0.05338 | -0.7305 | -2.3828 | 0.92184 | 2.2971 | 0.37797 |      1 |
| P4_leadtime_crops | cdl_perennial_continuous_interaction | PPML |     46 | 0.00075407 | 0.026033 | 0.011578 | -0.79348 | 0.81664 | 1.1192 | 0.97702 |      1 |

### Mechanism subgroup shares and CDL thresholds (FY2024–FY2026)

| year | hours_full_fy | requested_hours | pipeline_approval_ratio | worksite_bus_hours_share | incumbent_1yr_hours_share | h2alc_hours_share | late_start_hours_share | perennial_high_hours_share | perennial_p75_cutoff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|   2024 | 309018129 | 314249111 | 0.98248 | 0.84615 | 0.79555 | 0.23994 | 0.43722 | — | 0.004003 |
|   2025 | 321443050 | 335124865 | 0.96412 | 0.84405 | 0.83703 | 0.25085 | 0.42549 | — | 0.004003 |
|   2026 | 370224904 | 379524059 | 0.97506 | 0.84977 | 0.86188 | 0.25319 | 0.36852 | — | 0.004003 |

## Covered crop earnings: 2025Q4–2026Q1

The primary interim outcome averages two year-over-year log changes in private crop-production (NAICS 111) average weekly earnings. Every county needs both current/prior pairs, positive disclosed earnings and employment of at least ten in all four observations. The 2022–2024 H-2A intensity interacts with the statutory cut, with intensity and state effects included, equal county weights, and state deletion jackknife inference. States require at least two eligible counties.

| window | n | states | estimate | low | high | percent_at_S03_cut_minus4 | percent_low | percent_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| interim_2025Q4_2026Q1 |   1457 |     47 | 0.016549 | 0.0025366 | 0.030562 | -1.9663 | -3.601 | -0.30393 |
| quarter_2025Q4 |   1543 |     47 | 0.031007 | -0.0013378 | 0.063351 | -3.6524 | -7.3203 | 0.16066 |
| quarter_2026Q1 |   1562 |     47 | -0.0025555 | -0.033892 | 0.028781 | 0.30713 | -3.3948 | 4.1509 |
| quarter_2025Q4_common_sample |   1457 |     47 | 0.031093 | 0.00069786 | 0.061489 | -3.6625 | -7.113 | -0.083708 |
| quarter_2026Q1_common_sample |   1457 |     47 | 0.0020052 | -0.023081 | 0.027091 | -0.24034 | -3.1987 | 2.8084 |
| prior_window_lead |   1421 |     47 | -0.015024 | -0.037519 | 0.0074702 | 1.8193 | -0.89242 | 4.6052 |

At exposure 0.30 and a −$4 cut, the primary estimate implies -1.97% relative earnings change [-3.60%, -0.30%]. This is a model-implied relative difference, not the overall change in earnings.

### Sensitivities and comparison industries

| specification | weights | status | n | states | estimate | low | high | percent_at_S03_cut_minus4 | percent_low | percent_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crop_primary | equal | estimated |   1457 |     47 | 0.016549 | 0.0025366 | 0.030562 | -1.9663 | -3.601 | -0.30393 |
| crop_employment_weighted | emp_l | estimated |   1457 |     47 | 0.0050272 | -0.014226 | 0.02428 | -0.60145 | -2.8716 | 1.7217 |
| crop_cash_dose | equal | estimated |   1457 |     47 | 0.0146 | -0.0012692 | 0.03047 | -1.7368 | -3.5904 | 0.15242 |
| crop_commuting_zone | equal | estimated |   1457 |     47 | 0.016899 | -0.0047057 | 0.038503 | -2.0074 | -4.5152 | 0.56628 |
| crop_no_intensity_control | equal | estimated |   1457 |     47 | 0.0012195 | -0.0029369 | 0.0053758 | -0.14623 | -0.64302 | 0.35305 |
| crop_UI_covered | equal | insufficient_support |     78 |      2 | — | — | — | — | — | — |
| crop_UI_excluded | equal | estimated |    142 |      3 | -0.0086976 | -4.6583 | 4.6409 | 1.0492 | -99.619 |  26674 |
| crop_UI_ambiguous | equal | estimated |   1237 |     42 | 0.015556 | 0.0015956 | 0.029516 | -1.8494 | -3.4799 | -0.19128 |
| animal_primary | equal | estimated |   1216 |     45 | -0.0025012 | -0.041042 | 0.03604 | 0.30059 | -4.2326 | 5.0484 |
| all_private_primary | equal | estimated |   3011 |     47 | 0.0031814 | -0.0018425 | 0.0082053 | -0.38104 | -0.9798 | 0.22134 |

The employment-weighted, cash-dose and commuting-zone intervals include zero in this vintage. Removing the intensity control brings the point estimate near zero. Quarter-specific estimates suggest concentration in Q4 but do not establish a distinct timing pattern. These findings are suggestive and specification-sensitive; quiet placebos do not establish exogeneity.

Weekly earnings combine pay rates, hours and workforce composition. Incomplete UI classification and sparse covered/excluded groups do not identify a general domestic-worker spillover. Animal production and all-private outcomes are diagnostics, not guaranteed unaffected controls. Differential state shocks and contemporary policy changes remain identification concerns.

### Matched pre-policy windows

| tau | n | states | estimate | low | high |
| --- | --- | --- | --- | --- | --- |
|   2016 |   1346 |     47 | 0.0058418 | -0.016469 | 0.028153 |
|   2017 |   1376 |     47 | -0.0024629 | -0.028553 | 0.023627 |
|   2018 |   1380 |     48 | 0.012888 | -0.0062116 | 0.031988 |
|   2019 |   1401 |     48 | -0.0089704 | -0.039684 | 0.021743 |
|   2020 |   1428 |     48 | 0.0043534 | -0.018486 | 0.027192 |
|   2021 |   1466 |     48 | -0.0073337 | -0.024672 | 0.010005 |
|   2022 |   1491 |     48 | -0.0063953 | -0.036105 | 0.023315 |
|   2023 |   1485 |     48 | 0.0093033 | -0.016835 | 0.035442 |
|   2024 |   1504 |     47 | -0.0065108 | -0.026123 | 0.013102 |

### Coverage and outstanding series

| series | window | tau | counties | states | mean_change | mean_exposure |
| --- | --- | --- | --- | --- | --- | --- |
| crop | interim_2025Q4_2026Q1 |   2026 |   1457 |     47 | 0.03401 | 0.18993 |
| animal | interim_2025Q4_2026Q1 |   2026 |   1216 |     45 | 0.038288 | 0.11315 |
| all_private | interim_2025Q4_2026Q1 |   2026 |   3011 |     47 | 0.039388 | 0.13453 |
| crop | quarter_2025Q4 |   2026 |   1543 |     47 | 0.041767 | 0.18484 |
| animal | quarter_2025Q4 |   2026 |   1312 |     46 | 0.048408 | 0.11545 |
| all_private | quarter_2025Q4 |   2026 |   3012 |     47 | 0.04502 | 0.13449 |
| crop | quarter_2026Q1 |   2026 |   1562 |     47 | 0.026504 | 0.18513 |
| animal | quarter_2026Q1 |   2026 |   1318 |     46 | 0.030232 | 0.11666 |
| all_private | quarter_2026Q1 |   2026 |   3013 |     47 | 0.033885 | 0.13445 |
| crop | quarter_2025Q4_common_sample |   2026 |   1457 |     47 | 0.041457 | 0.18993 |
| crop | quarter_2026Q1_common_sample |   2026 |   1457 |     47 | 0.026563 | 0.18993 |
| crop | prior_window_lead |   2025 |   1421 |     47 | 0.027073 | 0.19213 |
| crop | matched_two_quarter_placebo |   2016 |   1346 |     47 | 0.033749 | 0.20192 |
| crop | matched_two_quarter_placebo |   2017 |   1376 |     47 | 0.028027 | 0.19508 |
| crop | matched_two_quarter_placebo |   2018 |   1380 |     48 | 0.031173 | 0.19493 |
| crop | matched_two_quarter_placebo |   2019 |   1401 |     48 | 0.016943 | 0.18953 |
| crop | matched_two_quarter_placebo |   2020 |   1428 |     48 | 0.037668 | 0.19048 |
| crop | matched_two_quarter_placebo |   2021 |   1466 |     48 | 0.055491 | 0.1888 |
| crop | matched_two_quarter_placebo |   2022 |   1491 |     48 | 0.058911 | 0.18951 |
| crop | matched_two_quarter_placebo |   2023 |   1485 |     48 | 0.061741 | 0.18859 |
| crop | matched_two_quarter_placebo |   2024 |   1504 |     47 | 0.033625 | 0.1875 |

| specification | status | detail |
| --- | --- | --- |
| QWI_secondary | missing_declared_interim_quarters | 2025Q4,2026Q1,2025Q1 |

The original three-quarter target still requires 2026Q2 and verified publication. No missing quarter is imputed. The authorized interim analysis already inspected outcomes; later windows are updates, not independent confirmations. The historical full-window command and its release guard remain preserved in the exploration archive, outside this runner.

## Reproduction and evidence

```sh
./scripts/run_policy_2025.sh
```

The runner reads cached inputs only. Its source-vintage guard stops on drift. [Design and commands](../../code/designs/policy_2025/README.md), [complete coefficient ledger](policy_2025_specifications.md), [provenance](policy_2025/provenance.json), and [artifact validation](policy_2025/validation.json) accompany this report. State deletions, coverage, floor and allocation audits are retained under `outputs/tables/policy_2025/`. Earlier exploration reports remain historical records.

All 347 wage, 79 Track 1 quantity, 96 mechanism-heterogeneity, 135 pre-policy earnings and 46 interim coefficient rows (703 total) were compared with the reference ledgers recorded for this vintage, including sample sizes, statuses and inference, at absolute tolerance 1e−10. Independent wage WLS, quantity PPML, mechanism PPML, earnings OLS and the 12 interim checks also passed. Passing validation establishes computational consistency, not causal identification.

