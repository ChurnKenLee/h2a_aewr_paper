# Three chapter directions: current-data assessment

**Historical assessment:** The user subsequently authorized interim earnings estimates. See the [interim earnings addendum](chapter_directions_earnings_interim.md) for the now-completed prior-window lead and 2025Q4/2026Q1 estimates. The assessment below records the earlier state of evidence, before that outcome look; its deferred-earnings status is superseded for those quarters.

As of November 14, 2026 (Pacific). Generated from the isolated exploration; no post-policy earnings have been inspected. All intervals below are 95% state deletion jackknife intervals using t(G−1), unless explicitly labeled. A dollar of statutory reduction corresponds to a negative dose; a positive wage slope therefore implies lower offers after a cut.

| Track | Recommendation | Assessment |
| --- | --- | --- |
| 2025 wage offers | Supporting finding | Continuing-cell response 0.640 [0.447, 0.834] dollars per statutory dollar; exact legal floors unresolved. |
| 2013 certification interruption | Not identified | Timely certification -2.4 [-23.0, 18.3] percentage points per full exposure unit; persistence dates unavailable. |
| Local agricultural earnings | Awaiting data | Preserved full-window primary; 2016–2024 diagnostics complete. Separate prior-window lead is implemented but its public quarter slices could not be retrieved. |

The wage track is the strongest development path, but the current evidence supports a substantial within-filer offer response rather than a demonstrated causal mechanism beyond floor compliance. The administrative result does not establish either no disruption or a causal processing-delay effect. The earnings primary must wait for full publication. Every declared coefficient, unsuccessful fit and diagnostic appears in the [specification appendix](chapter_directions_specifications.md); nothing is retained according to significance.

## Questions, estimands and identifying comparisons

| Track | Question | Estimand | Necessary_assumption |
| --- | --- | --- | --- |
| Wage setting | How do offers adjust when statutory floors fall? | Continuing filer-state-occupation mean-offer change per statutory dollar within former regions | No correlated differential state shocks; comparable offer measurement and continuation |
| Program access | Does predicted exposure to suspended processing reduce timely certification? | Exposure-by-year difference relative to 2012 within cohort and state-season-year effects | Baseline projected intervals predict interruption exposure independently of differential seasonal shocks |
| Covered crop earnings | Do earnings change more where predetermined H-2A intensity is higher? | Mean of three YoY county log-earnings changes on intensity × statutory cut, controlling for intensity and state | Parallel exposure-related earnings changes within states; stable covered-job measurement; no correlated shocks |

Filer IDs represent filing entities, not individual farms, workers or the worksite-business entities in the original decomposition. Fixed-composition estimates condition on occupations observed in both periods. None of the three designs directly measures realized hours, visa arrival, worker-level wage rates or welfare.

## 1. Offered wages after the 2025 reform

### Reproduction and unit of observation

| FY | Component | Estimate | Jackknife_SE | Counties | States |
| --- | --- | --- | --- | --- | --- |
| 2025 | total | 0.0030322 | 0.0573955 | 2655 | 46 |
| 2025 | within | 0.0187479 | 0.0272328 | 2655 | 46 |
| 2025 | between | -0.0035256 | 0.0055684 | 2655 | 46 |
| 2025 | entry | 0.0102341 | 0.0503214 | 2655 | 46 |
| 2025 | exit | -0.0283043 | 0.0308247 | 2655 | 46 |
| 2025 | no_continuer | 0.0058801 | 0.0152945 | 2655 | 46 |
| 2026 | total | 0.6623321 | 0.0945617 | 2686 | 46 |
| 2026 | within | 0.6158394 | 0.0944880 | 2686 | 46 |
| 2026 | between | -0.0079389 | 0.0155172 | 2686 | 46 |
| 2026 | entry | 0.0207771 | 0.0330233 | 2686 | 46 |
| 2026 | exit | 0.0266960 | 0.0330721 | 2686 | 46 |
| 2026 | no_continuer | 0.0069585 | 0.0129084 | 2686 | 46 |

The original county response, original weights/sample and matched-window placebo reproduce to numerical tolerance. Components reconcile both at the county and coefficient levels. The balanced worksite-business within component accounts for 93.0% of the county slope (jackknife SE 3.7 percentage points); the high-recall identity yields 91.7%. These are shares of a regression slope, not shares of the raw wage decline. Reproduction checks are retained separately from the new filer results.

The extension takes hourly offers in (0,100], positive allocated certified hours, and decisions during October–June of each fiscal year. It preserves the original county and former-region sample and pooled weekly-hours imputation. County allocation conserves positions and hours. Case SOC is joined to worksite allocations: many Addendum B rows do not repeat it. An initially incorrect row-SOC implementation was corrected after inspection; the dated amendment and the initial CSVs in scratch `implementation_audit/row_soc/` preserve that history. This is exploratory evidence developed after related FY2026 results were known.

| Specification | Weights | Cells | States | Estimate_CI |
| --- | --- | --- | --- | --- |
| occupation_dw | h0 | 12386 | 46 | 0.640 [0.447, 0.834] |
| occupation_dw | equal | 12386 | 46 | 0.601 [0.248, 0.955] |
| filer_state_unrestricted | h0 | 12313 | 46 | 0.650 [0.470, 0.830] |
| filer_state_fixed_composition | h0 | 11691 | 46 | 0.639 [0.444, 0.835] |
| filer_state_unrestricted_common_sample | h0 | 11691 | 46 | 0.645 [0.454, 0.836] |
| filer_state_unrestricted | equal | 12313 | 46 | 0.615 [0.288, 0.943] |
| filer_state_fixed_composition | equal | 11691 | 46 | 0.622 [0.287, 0.956] |
| filer_state_unrestricted_common_sample | equal | 11691 | 46 | 0.623 [0.289, 0.956] |
| within_filer_across_state | h0 | 1903 | 46 | 0.408 [0.135, 0.680] |
| within_filer_across_state | equal | 1903 | 46 | 0.183 [0.077, 0.290] |

Common occupations cover 95.7% of baseline and 95.9% of current hours among unrestricted continuing filer-state cells. The fixed-composition comparison uses 11691 of 12313 continuing filer-state cells. It supports the conclusion that changing occupation shares explain little of the mean response within this selected sample. It does not recover offers of exiting filers or disappearing occupations. The across-state contrast compares states within the same continuing multi-state filer; it absorbs filer-wide changes but retains state-specific confounding.

### Retention, reference premiums and applicability

In continuing R7 occupation cells, baseline-hours-weighted offers average $17.28 before and $14.92 after. 82.4% of baseline hours belong to cells reducing their mean offer by more than five cents; 3.9% retain their mean within five cents. These are cell-mean classifications, not worker-level transition probabilities.

| Outcome | Cells | Slope_CI |
| --- | --- | --- |
| occupation_reduce | 12386 | -0.010 [-0.069, 0.049] |
| occupation_retain | 12386 | -0.012 [-0.044, 0.020] |
| occupation_dp | 12386 | -0.360 [-0.553, -0.166] |
| occupation_dnear | 12386 | 0.026 [-0.040, 0.092] |
| occupation_dcdf0 | 12386 | 0.019 [-0.030, 0.067] |
| occupation_dcdf25 | 12386 | 0.021 [-0.023, 0.064] |
| occupation_dcdf100 | 12386 | 0.047 [-0.019, 0.112] |
| schedule_envelope_dnear_candidate | 8272 | 0.037 [-0.086, 0.159] |
| schedule_envelope_dpremium_lower | 8272 | -0.305 [-0.845, 0.235] |

Probability/CDF slopes are in probability units per statutory dollar. Premiums subtract a reference schedule that itself changes; the reference-premium slope partly reflects that arithmetic. Distributional comparisons against that changing reference cannot isolate bargaining or a market wage.

| Reference_premium_threshold | Baseline_CDF | Current_CDF |
| --- | --- | --- |
| 0.00 | 0.94195 | 0.44127 |
| 0.25 | 0.95893 | 0.52648 |
| 1.00 | 0.97419 | 0.66347 |
| 2.00 | 0.97655 | 0.71027 |
| 5.00 | 0.98848 | 0.91885 |

This distribution table thresholds each cell's mean reference premium. The CDF regression outcomes instead use hours-weighted shares of allocation-row premiums within continuing cells; their different observation units are intentional and should not be conflated. April schedule references also need not be the rate legally applicable to an earlier contract in the October–June decision window.

Final report review also found and corrected a grouped-summary ordering defect in the three row-CDF outcomes. Their indicators are now formed before grouping, with explicit length and completeness checks. The initial outputs remain in scratch `implementation_audit/cdf_aggregation/`; all non-CDF estimates were checked for unchanged results. This correction, made after inspection, is recorded in the dated amendment.

| FY | Case_SOC_known | Allocation_row_SOC_missing | Reference_envelope_coverage | Near_any_candidate | Ag_minimum_above_reference |
| --- | --- | --- | --- | --- | --- |
| 2023 | 1 | 0.56343 | 0.87463 | 0.69048 | 0.00000 |
| 2024 | 1 | 0.55688 | 0.78874 | 0.69353 | 0.00000 |
| 2025 | 1 | 0.56789 | 0.79288 | 0.73725 | 0.00000 |
| 2026 | 1 | 0.58127 | 0.67669 | 0.85193 | 0.23673 |

Exact effective floors remain unknown for every row. The October 2 rule applies by job-order submission, whereas these disclosures supply case receipt dates. Worker category, skill, prevailing/collective rates and itinerary-specific requirements cannot be established. The separate reference envelope uses the four US/H-2A skill-I/II schedules and the observed annual agricultural minimum, only for the five common SOCs, single-state cases and known receipt dates. Even this envelope is a measurement diagnostic; it is not a corrected estimate of the legal floor. [October 2025 rule](https://www.govinfo.gov/content/pkg/FR-2025-10-02/html/2025-19365.htm); [DOL wage obligations](https://www.dol.gov/agencies/whd/fact-sheets/26-H2A).

The mean slope establishes incomplete pass-through relative to the unchanged statutory dose under the comparison assumptions. It is consistent with several mechanisms, including higher competing floors, a mix of contract regimes and partial adjustment. Retention and schedule-distance regressions do not resolve those mechanisms. Above-reference offers must not be labeled market wages.

### Placebos, linkage, continuation and heterogeneity

| Tier | FY | Cells | Status | Estimate_CI |
| --- | --- | --- | --- | --- |
| R7 | 2024 | 9492 | estimated | 0.017 [-0.032, 0.067] |
| R7 | 2025 | 11430 | estimated | 0.014 [-0.036, 0.064] |
| R7 | 2026 | 12386 | estimated | 0.640 [0.447, 0.834] |
| R5 | 2024 | 9398 | estimated | 0.019 [-0.032, 0.069] |
| R5 | 2025 | 11324 | estimated | 0.014 [-0.037, 0.065] |
| R5 | 2026 | 12287 | estimated | 0.639 [0.446, 0.832] |
| R8 | 2024 | 9589 | estimated | 0.018 [-0.033, 0.069] |
| R8 | 2025 | 11580 | estimated | 0.014 [-0.037, 0.064] |
| R8 | 2026 | 12485 | estimated | 0.638 [0.445, 0.832] |
| FEIN | 2024 | 0 | insufficient_support | unavailable [unavailable, unavailable] |
| FEIN | 2025 | 11671 | estimated | 0.013 [-0.035, 0.062] |
| FEIN | 2026 | 12507 | estimated | 0.640 [0.446, 0.834] |

The FEIN 2023–2024 contrast has no continuing matched cells and remains unavailable; it was not replaced with another ID. R5/R7/R8 and the feasible FEIN results are close. Quiet matched-window placebos do not prove exogeneity.

| Specification | Weights | Term | Estimate_CI |
| --- | --- | --- | --- |
| selection_stay | h0 | dose | -0.004 [-0.015, 0.007] |
| selection_any_cert | h0 | dose | -0.004 [-0.016, 0.007] |
| selection_stay | equal | dose | -0.004 [-0.013, 0.006] |
| selection_any_cert | equal | dose | -0.005 [-0.016, 0.007] |
| selection_baseline_premium | equal | stay | -0.404 [-1.006, 0.197] |
| selection_baseline_premium | equal | dose:stay | -0.024 [-0.165, 0.117] |
| selection_prechange | equal | stay | 0.427 [-0.033, 0.887] |
| selection_prechange | equal | dose:stay | 0.086 [-0.010, 0.182] |

Continuation denotes a subsequent observed hourly offer or positive certification in the study window, not verified operation or employment. The pre-change comparison further selects filers with earlier observed wages. Small or insignificant selection coefficients cannot establish random survival.

| Interaction | Estimate_CI | Unadjusted_p | Holm_p | Simultaneous_low | Simultaneous_high |
| --- | --- | --- | --- | --- | --- |
| dose:highpremium | -0.205 [-1.227, 0.818] | 0.68901 | 1.0000 | -1.46747 | 1.05835 |
| dose:contractor | 0.270 [-0.057, 0.598] | 0.10337 | 0.3101 | -0.13408 | 0.67494 |
| dose:large | 0.032 [-0.193, 0.258] | 0.77349 | 1.0000 | -0.24573 | 0.31050 |

The secondary family is fixed at three baseline interactions: premium above $0.25, contractor status, and above-median filer hours. Holm p-values and Bonferroni simultaneous intervals accompany ordinary intervals. None distinguishes a reliable heterogeneous mechanism at the family level.

Contemporaneous shocks weaken a causal floor interpretation. DOL suspended enforcement of the 2024 farmworker rule on June 20, 2025; pre-existing requirements remained enforceable. OFLC also stopped processing October 1–November 2, 2025, overlapping the reform and potentially shifting the decision-window sample. These are distinct changes with possible heterogeneous state incidence. [DOL enforcement notice](https://www.dol.gov/newsroom/releases/whd/whd20250620); [OFLC recovery notice, state-hosted agency copy](https://mn.gov/deed/assets/ofcl-2025-11-05_tcm1045-724262.pdf).

Worksite immigration enforcement also changed locally: DOJ documents a June 10, 2025 operation at an Omaha food-processing business. This establishes a contemporaneous enforcement event, not a measured agricultural labor-supply shock or its correlation with the dose. Annual minimum-wage floors are audited, but no comprehensive enforcement, crop-demand or weather-shock panel is available here. Former-region effects and within-filer comparisons do not absorb those state-specific changes. [DOJ account](https://www.justice.gov/usao-ne/pr/take-back-america-task-force-operation-south-omaha).

**Recommendation: supporting finding.** The substantial within-filer adjustment and small composition contribution are informative, but unidentified effective floors prevent a firm claim beyond floor-linked offer setting. Do not infer realized earnings, monopsony, welfare gains or zero employment effects.

## 2. The 2013 administrative interruption

### Institutional timing and construction

Primary agency notices establish suspended OFLC processing during October 1–16, reopening October 17, and administrative treatment of October 1–18 receipts as October 18. Recovery included emailed certification copies and expedited originals. These procedures can shift recorded dates independently of actual filing behavior. The baseline projection was fixed before event-period regressions; receipt recoding is flagged, never inverted into an invented submission date. [Archived DOL notices](https://www.aila.org/aila-files/EFC46464-B9FD-4F44-BC5E-3D6F3C38D0D2/13122348.pdf?1697590280=); [House shutdown chronology](https://history.house.gov/Institution/Shutdown/Government-Shutdowns/).

The R7 cohort frame contains 2462 recurring filer-state-requested-start-month cohorts; 136 have positive projected exposure across 23 states. Each 2011/2012 receipt-to-decision interval is shifted by calendar year to 2013, with leap day mapped to February 28. Exposure is the baseline case share overlapping the interruption. Cohort effects and state-by-broad-season-by-year effects define the comparison; the effective estimation sample contains 46 states after fixed-effect support restrictions.

| tier | baseline_cohorts | recurring | complete_recurring | unknown_baseline_positions |
| --- | --- | --- | --- | --- |
| R7 | 7408 | 2462 | 2462 | 247 |
| R5 | 7484 | 2395 | 2395 | 253 |
| R8 | 7369 | 2488 | 2488 | 247 |

Master/sub-employer labels are audited; sub-employer rows are removed where labeled. Exact duplicates collapse, conflicting dates/status/positions become unknown, and repeated case numbers across fiscal extracts are excluded. Labels and administrative records cannot prove that every remaining filing entity is an independent farm. The exposure-input case ledger is unique and contains only 2011/2012 dates.

| fiscal_year | date_basis | cases | unknown_status | invalid_dates | duplicate_source_cases |
| --- | --- | --- | --- | --- | --- |
| 2010 | dates_of_need | 4427 | 0 | 0 | 261 |
| 2010 | missing | 4 | 0 | 4 | 0 |
| 2011 | dates_of_need | 4937 | 0 | 1 | 160 |
| 2011 | missing | 4 | 0 | 4 | 0 |
| 2012 | dates_of_need | 5457 | 0 | 1 | 231 |
| 2012 | missing | 1 | 0 | 1 | 0 |
| 2012 | — | 1 | 0 | 0 | 1 |
| 2013 | dates_of_need | 5972 | 0 | 0 | 114 |
| 2013 | missing | 1 | 0 | 1 | 0 |
| 2014 | dates_of_need | 6725 | 0 | 0 | 110 |
| 2014 | missing | 1 | 0 | 1 | 0 |
| 2015 | certification_dates | 7544 | 0 | 5 | 127 |
| 2015 | certification_dates_start_only | 9 | 0 | 0 | 0 |
| 2015 | missing | 2 | 0 | 2 | 0 |
| 2015 | — | 12 | 0 | 2 | 12 |
| 2016 | dates_of_need | 8682 | 0 | 1 | 0 |
| 2016 | missing | 2 | 0 | 2 | 0 |

FY2015 has certification-date-based starts rather than requested starts. Calendar 2014 and 2015 outcomes are consequently unknown for every retained cohort: missing assignments must not become false nonfiling. The intended five-year design and its 2014/2015 temporal placebos are unavailable. The feasible result is explicitly a shorter 2011–2013 comparison.

### Main and sensitivity results

| Outcome | Weights | Observations | Status | Estimate_CI |
| --- | --- | --- | --- | --- |
| primary_frame_timely | equal | 7320 | estimated | -0.024 [-0.230, 0.183] |
| primary_frame_observed_filing | equal | 7320 | estimated | 0.005 [-0.285, 0.295] |
| primary_frame_certified_positions | equal | 7158 | estimated | 5.011 [-6.098, 16.120] |
| primary_frame_timing | equal | 6567 | estimated | 3.153 [-2.828, 9.134] |
| primary_frame_observed_denial | equal | 7320 | estimated | 0.004 [-0.177, 0.184] |
| primary_frame_unknown_dates_or_status | equal | 12310 | fit_failed | unavailable [unavailable, unavailable] |
| primary_frame_timely | baseline_positions | 7098 | estimated | 0.136 [-0.497, 0.770] |
| primary_frame_observed_filing | baseline_positions | 7098 | estimated | 0.073 [-0.479, 0.625] |
| primary_frame_certified_positions | baseline_positions | 7098 | estimated | 23.195 [-22.475, 68.865] |
| primary_frame_timing | baseline_positions | 6399 | estimated | -5.379 [-24.157, 13.400] |
| primary_frame_observed_denial | baseline_positions | 7098 | estimated | -0.042 [-0.230, 0.147] |
| primary_frame_unknown_dates_or_status | baseline_positions | 11940 | fit_failed | unavailable [unavailable, unavailable] |

Timely certification and filing are probability outcomes; positions are certified positions, and timing is days from the observed requested start among certified applications with usable dates. A missing observed case yields zero observed filing/timely certification only in years with usable coverage. It is not evidence that an employer did not apply, hire or operate. Denial and unknown-record diagnostics remain separate. The unknown-date/status diagnostic describes cases assignable to a cohort; its constant zero cannot establish that unassignable cases are complete. The separate calendar-coverage flag controls the missing 2014/2015 outcomes. Constant outcomes and unidentified terms are recorded explicitly.

| Specification | Estimate_CI | States |
| --- | --- | --- |
| month_comparison_timely | -0.162 [-0.286, -0.038] | 45 |
| aug_dec_timely | -0.024 [-0.231, 0.184] | 33 |
| exclude_largest_timely | -0.024 [-0.230, 0.183] | 46 |
| month_comparison_timing | 4.824 [-2.974, 12.621] | 45 |
| catchup7 | -0.024 [-0.239, 0.192] | 46 |
| catchup14 | -0.024 [-0.239, 0.192] | 46 |
| catchup30 | -0.039 [-0.234, 0.156] | 46 |
| catchup60 | -0.039 [-0.234, 0.156] | 46 |

The month-based comparison is materially more negative for timely certification than the broad-season primary. This sensitivity is retained, not promoted because its interval excludes zero. Catch-up outcomes do not identify a distinct recovery effect; they remain exposed-versus-less-exposed comparisons with uncertain selection and predicted start dates.

| Lead | Estimate_CI |
| --- | --- |
| primary_frame_timely | -0.011 [-0.100, 0.078] |
| primary_frame_timing | 1.243 [-2.472, 4.958] |

| Comparison | Term | Status | Estimate_CI |
| --- | --- | --- | --- |
| calendar_placebo_2012_timely | year::2012:exposure | estimated | 0.011 [-0.078, 0.100] |
| calendar_placebo_2014_timely | year::2014:exposure | fit_failed | unavailable [unavailable, unavailable] |
| calendar_placebo_2015_timely | year::2015:exposure | insufficient_support | unavailable [unavailable, unavailable] |
| calendar_placebo_2012_observed_filing | year::2012:exposure | fit_failed | unavailable [unavailable, unavailable] |
| calendar_placebo_2014_observed_filing | year::2014:exposure | fit_failed | unavailable [unavailable, unavailable] |
| calendar_placebo_2015_observed_filing | year::2015:exposure | insufficient_support | unavailable [unavailable, unavailable] |
| calendar_placebo_2012_certified_positions | year::2012:exposure | estimated | 1.281 [-4.071, 6.634] |
| calendar_placebo_2014_certified_positions | year::2014:exposure | fit_failed | unavailable [unavailable, unavailable] |
| calendar_placebo_2015_certified_positions | year::2015:exposure | insufficient_support | unavailable [unavailable, unavailable] |
| calendar_placebo_2012_timing | year::2012:exposure | estimated | -1.243 [-4.958, 2.472] |
| calendar_placebo_2014_timing | year::2014:exposure | fit_failed | unavailable [unavailable, unavailable] |
| calendar_placebo_2015_timing | year::2015:exposure | insufficient_support | unavailable [unavailable, unavailable] |

The 2012 temporal placebo is a pre-period contrast using the same projected exposure; 2014/2015 tests cannot be estimated from the missing requested-start coverage. Observed filing in both 2011 and 2012 is a selection criterion, so its pre-period contrast is mechanically constant. State jackknife uncertainty and temporal-placebo evidence answer different questions; neither supplies many independent shutdowns.

| tier | exposed | cohorts | states | mean_positions | mean_processing | mean_lead |
| --- | --- | --- | --- | --- | --- | --- |
| R7 | FALSE | 2326 | 49 | 13.839 | 24.569 | 53.676 |
| R7 | TRUE | 136 | 23 | 47.664 | 23.143 | 54.054 |

| state_fips | exposure_mass | positive_cohorts | share |
| --- | --- | --- | --- |
| 22 | 30.0 | 38 | 0.266667 |
| 12 | 29.0 | 32 | 0.257778 |
| 48 | 6.0 | 7 | 0.053333 |
| 49 | 5.0 | 5 | 0.044444 |
| 56 | 5.0 | 6 | 0.044444 |
| 04 | 4.5 | 5 | 0.040000 |
| 06 | 3.5 | 4 | 0.031111 |
| 13 | 3.5 | 5 | 0.031111 |
| 47 | 3.5 | 4 | 0.031111 |
| 08 | 3.0 | 4 | 0.026667 |

Louisiana and Florida account for 52.4% of baseline exposure mass. Exposed cohorts have much larger baseline certified-position counts, making weighting consequential. Support exists in some state-season strata but is highly concentrated; all strata and filer concentration are retained in scratch. The largest-baseline-filer exclusion and every state deletion are reported in the appendix and CSVs.

| exposed | cases | receipt_before_shutdown | september_receipts | recoded_receipts | certifications |
| --- | --- | --- | --- | --- | --- |
| FALSE | 1834 | 1805 | 21 | 0 | 1803 |
| TRUE | 120 | 59 | 56 | 5 | 114 |

This timing audit shows observed pre-shutdown and September receipts among recurring cohorts and the October 18 recoding mass. It cannot distinguish anticipatory decisions from seasonal scheduling. The historical predicted start is not the 2013 hiring requirement. Other shutdown channels, endogenous processing durations in the baseline and differential agricultural seasons remain alternative explanations.

**Recommendation: not identified.** The shortened panel, concentrated exposure, sensitivity to the seasonal comparison and broad uncertainty prevent a defensible causal program-access chapter from these records. The descriptive administrative audit is reusable. Do not instrument processing delay with this shutdown to claim a pure delay effect.

## 3. Local agricultural earnings: preparation and deferred estimate

The original registry, estimator and frozen exposure hashes pass. The primary remains private crop-production county average weekly earnings, equal county weights, frozen 2022–2024 intensity, state effects, an intensity control and state jackknife inference. The target is the average YoY log change for 2025Q4–2026Q2. The full QCEW 2026Q2 release is scheduled for December 2, 2026 at 10 a.m. New York time; a schedule is not proof of publication. [BLS calendar](https://www.bls.gov/schedule/news_release/cew.htm).

| rows | joint_disclosed | changed_disclosed_wages | max_joint_wage_difference | excluded_old_rows |
| --- | --- | --- | --- | --- |
| 123328 | 78595 | 0 | 0.000000000001819 | 28 |

The pre-policy audit corrects an aggregation-order defect and adds the existing canonical-geography priority rules in a separate extractor. Among jointly disclosed crop records, no reported weekly wage changes beyond numerical tolerance. Removed old records are nonallocatable geography, not an outcome-selected sample. All five original tau2024 specification coefficients and jackknife standard errors reproduce. The amendment precedes any post-policy earnings access.

| Specification | Counties | States | Status | Coefficient_CI | Percent_at_reference | Percent_low | Percent_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crop_primary | 1483 | 47 | estimated | -0.008 [-0.027, 0.011] | 0.979292 | -1.25460 | 3.26372 |
| crop_employment_weighted | 1483 | 47 | estimated | -0.002 [-0.022, 0.018] | 0.259592 | -2.12970 | 2.70722 |
| crop_cash_dose | 1483 | 47 | estimated | -0.004 [-0.028, 0.020] | 0.501266 | -2.35776 | 3.44400 |
| crop_commuting_zone | 1483 | 47 | estimated | -0.016 [-0.050, 0.018] | 1.974765 | -2.09195 | 6.21040 |
| crop_no_intensity_control | 1483 | 47 | estimated | -0.003 [-0.009, 0.003] | 0.375012 | -0.34832 | 1.10360 |
| crop_UI_covered | 85 | 2 | insufficient_support | unavailable [unavailable, unavailable] | — | — | — |
| crop_UI_excluded | 145 | 3 | estimated | -0.014 [-1.464, 1.436] | 1.713389 | -82.14487 | 479.41978 |
| crop_UI_ambiguous | 1253 | 42 | estimated | -0.006 [-0.025, 0.013] | 0.693067 | -1.59999 | 3.03956 |
| animal_primary | 1219 | 46 | estimated | -0.011 [-0.062, 0.041] | 1.274275 | -4.77138 | 7.70374 |
| all_private_primary | 3017 | 47 | estimated | -0.002 [-0.009, 0.005] | 0.236519 | -0.61731 | 1.09768 |
| QWI_stable_monthly_primary | 1855 | 46 | estimated | -0.014 [-0.029, 0.001] | 1.671706 | -0.17789 | 3.55558 |
| QWI_stable_monthly_employment_weighted | 1855 | 46 | estimated | 0.001 [-0.012, 0.014] | -0.089015 | -1.66102 | 1.50812 |
| QWI_stable_monthly_cash_dose | 1855 | 46 | estimated | -0.008 [-0.033, 0.016] | 1.000421 | -1.94185 | 4.03098 |
| QWI_stable_monthly_commuting_zone | 1855 | 46 | estimated | -0.027 [-0.072, 0.018] | 3.303615 | -2.14870 | 9.05973 |
| QWI_stable_monthly_no_intensity_control | 1855 | 46 | estimated | -0.002 [-0.007, 0.002] | 0.290208 | -0.23359 | 0.81675 |

These are pre-policy diagnostics, not reform effects. Percent translations use intensity S=0.30 and a −$4 statutory cut: 100×[exp(−1.2β)−1], with interval endpoints reversed under the negative dose. The appendix includes every 2016–2024 crop, animal, all-private and QWI result, both weighting choices, cash-dose and commuting-zone sensitivities, and the no-intensity-control result.

| direct_agriculture | contractor_agriculture | States |
| --- | --- | --- |
| ambiguous | ambiguous | 43 |
| covered | covered | 2 |
| excluded | excluded | 3 |

The policy-period UI review classifies California and Washington as covered based on agency/statutory agricultural-employment provisions, and Florida, Minnesota and New York as excluding qualifying H-2A agricultural service. Direct and contractor service are considered separately; threshold, localization and voluntary-coverage caveats remain. Oregon and Texas remain unresolved, as do states without sufficient state-specific evidence. The complete sourced review is [retained here](../explorations/chapter_directions/sources/ui_classification_review.json). National FUTA treatment alone cannot determine a state's UI reporting rules. Coverage of jobs and eligibility to receive benefits are different questions.

The two-state covered subgroup cannot support the declared jackknife, and the three-state excluded subgroup is extremely imprecise. The fixed policy-period classification used in historical diagnostics is not a reconstructed historical legal panel. No subgroup is labeled a complete non-H-2A earnings series. The interpretation remains earnings in covered crop-production jobs; wage rates, hours, worker mix and establishment composition are not separately observed.

| specification | status | reason |
| --- | --- | --- |
| primary | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |
| employment_weighted | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |
| cash_dose | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |
| commuting_zone | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |
| no_intensity_control | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |
| all_private_diagnostic | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |
| animal_diagnostic | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |
| prior_window_lead | unavailable_separate_pre_policy_slices | Needs public 2025Q1 and Q2 crop slices; local DNS unavailable, web CSV unavailable, Cloud tool disallowed by session approval policy, browser connection timed out. Annual archives containing post-policy earnings remain closed. |
| UI_splits | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |
| QWI_secondary | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |

The prior-window lead implementation needs 2025Q1/Q2 crop slices plus the pre-policy history. Those slices are public but were inaccessible in this session: local DNS failed, the web reader could not retrieve CSV, the Cloud tool required approval unavailable under this session's policy, and the browser connection timed out. The annual archive containing post-policy earnings stayed closed. The README supplies a separate pre-policy-only command; this retrieval gap is explicitly outstanding. All-private and animal earnings are diagnostic comparisons, not guaranteed unaffected controls.

After actual publication, verify official release metadata, confirm complete quarters, and run once:

```sh
bash explorations/chapter_directions/resume_earnings.sh /tmp/h2a-chapter-directions /tmp/h2a-chapter-directions/release_metadata.json
```

The [README](../explorations/chapter_directions/README.md) gives the exact metadata schema and optional QWI prerequisites. The command checks frozen hashes, calendar and publication evidence, acquires a consistent current vintage into scratch, and verifies all six quarters for each series. A completion marker prevents a second full-window run. QWI requires its own complete extract, publication evidence and hash; otherwise it stays deferred. No partial-window estimate and no automatic scheduling.

**Recommendation: awaiting data.** The pre-policy diagnostics do not establish the post-policy effect or its precision. The unresolved prior-window lead and limited UI classification should remain visible when the full-window results arrive.

## Precision against economic reference magnitudes

| Outcome | Reference | Low | High | Positive_reference_inside | Negative_reference_inside |
| --- | --- | --- | --- | --- | --- |
| Offer response ($/$) | 0.10 | 0.44689 | 0.83384 | FALSE | FALSE |
| Offer response ($/$) | 0.25 | 0.44689 | 0.83384 | FALSE | FALSE |
| Offer response ($/$) | 0.50 | 0.44689 | 0.83384 | TRUE | FALSE |
| Timely certification (pp, full exposure) | 2.00 | -22.99473 | 18.28154 | TRUE | TRUE |
| Timely certification (pp, full exposure) | 5.00 | -22.99473 | 18.28154 | TRUE | TRUE |
| Timely certification (pp, full exposure) | 10.00 | -22.99473 | 18.28154 | TRUE | TRUE |
| Conditional certification timing (days) | 3.00 | -2.82826 | 9.13388 | TRUE | FALSE |
| Conditional certification timing (days) | 7.00 | -2.82826 | 9.13388 | TRUE | FALSE |
| Conditional certification timing (days) | 14.00 | -2.82826 | 9.13388 | FALSE | FALSE |
| Pre-policy crop earnings (%, S=.30, cut=-4) | 1.00 | -1.25460 | 3.26372 | TRUE | TRUE |
| Pre-policy crop earnings (%, S=.30, cut=-4) | 3.00 | -1.25460 | 3.26372 | TRUE | FALSE |
| Pre-policy crop earnings (%, S=.30, cut=-4) | 5.00 | -1.25460 | 3.26372 | FALSE | FALSE |

These comparisons contextualize intervals; they do not select a preferred specification or constitute prospective power calculations. Wage intervals rule out very small positive responses but include economically substantial adjustment. Timely-certification intervals admit both 2/5/10-point gains and losses. The timing primary includes delays of 3 and 7 days; a narrower interval than 14 days does not repair identification. The earnings row describes only a pre-policy calculation and cannot substitute for the deferred outcome.

## Contribution and proposed outline for the strongest direction

The distinctive opportunity is employers' downward adjustment of offers after a wage-floor reduction. Bracero-exclusion research studies a labor-supply restriction and agricultural adjustment; the H-2B lottery literature studies randomized firm access to labor. Neither establishes the present statutory-dose comparison. Payroll-based nominal rigidity research studies actual paid wages, whereas these data describe offers and changing contract composition. These are focused comparators, not an exhaustive claim of novelty. [Clemens, Lewis and Postel](https://www.nber.org/papers/w23125); [Clemens and Lewis](https://www.nber.org/papers/w30589); [payroll wage rigidity study](https://www.nber.org/papers/w25470).

Certification-delay concerns already appear in the [2012 GAO review](https://www.gao.gov/products/gao-12-706). A chapter contribution would require a credible disruption comparison, not simply documenting delays again. The current administrative panel does not supply that comparison reliably. The earnings track could connect an offer response to local covered-job earnings, but such a result would still combine rates, hours and composition.

A conditional wage-setting chapter outline is:

1. **Institution and measurement:** old/new schedules, applicability rules, the difference between a filer and a worksite business, and the unresolved legal-floor audit.
2. **Offer adjustment:** reproduced county response, exact within/composition decomposition, then filer-state-occupation results and baseline weights.
3. **Distribution and continuation:** common-occupation coverage, retention/reduction, reference-premium distributions, fixed heterogeneity family and all selection checks.
4. **Credibility:** matched-window placebos, identity sensitivities, state deletions, within-filer state comparisons, concurrent processing/enforcement shocks, and the limits of those tests.
5. **Earnings extension after publication:** the frozen complete-window test and diagnostic comparisons, clearly separated from offered wages.
6. **Interpretation:** evidence supports substantial within-filer offer adjustment; stronger claims require verified applicable floors and a credible comparison addressing state-specific shocks.

This outline is a development proposal, not promotion to a causal chapter candidate. The two concrete weaknesses to resolve are job-order/skill/worker-category applicability and confounding at the state-by-episode level. No confidential NASS microdata, institutional access or new production-linkage rebuild is required by this exploration.

## Reproducibility and validation

The isolated [runner and ownership README](../explorations/chapter_directions/README.md), [registry](../explorations/chapter_directions/registry.json), [amendment](../explorations/chapter_directions/measurement_amendment_2026_11_14.json) and frozen amendment hashes define this run. All large panels, raw-input hashes, source vintages, model CSVs, state deletions and failed fits live at `/tmp/h2a-chapter-directions`. The initial provenance manifest was recorded after dependency reconstruction and the first estimates; it is a reproducibility record, not a claim that its timestamp establishes preregistration.

| check | reference | reproduced | absolute_difference |
| --- | --- | --- | --- |
| county_response_2025 | 0.0030322 | 0.0030322 | 0.0000000000000001101549 |
| county_response_2026 | 0.6623321 | 0.6623321 | 0.0000000000000015543122 |
| crop_primary | -0.0081211 | -0.0081211 | 0.0000000000000000294903 |
| crop_employment_weighted | -0.0021605 | -0.0021605 | 0.0000000000000000099747 |
| crop_cash_dose | -0.0041668 | -0.0041668 | 0.0000000000000000598480 |
| crop_commuting_zone | -0.0162960 | -0.0162960 | 0.0000000000000000728584 |
| crop_no_intensity_control | -0.0031193 | -0.0031193 | 0.0000000000000000000000 |

Artifact validation passed 54 checks, including nonempty unique keys, string geography, conserved allocations, baseline-only exposure reconstruction, retained nonreturners, unknown unavailable years, suppression, original-estimate reproduction and complete specification grids.

| check | status | log |
| --- | --- | --- |
| shell_syntax | passed | logs/check_shell_syntax.log |
| R_syntax | passed | logs/check_R_syntax.log |
| python_ruff | passed | logs/check_python_ruff.log |
| marimo_prepare_export | passed | logs/check_marimo_prepare_export.log |
| marimo_earnings_export | passed | logs/check_marimo_earnings_export.log |
| earnings_edge_tests | passed | logs/check_earnings_edge_tests.log |
| artifact_and_R_edge_checks | passed | logs/check_artifact_and_R_edge_checks.log |
| project_pipeline_dry_run | passed | logs/check_project_pipeline_dry_run.log |
| exploration_dry_run | passed | logs/check_exploration_dry_run.log |
| grounding_verify | passed | logs/check_grounding_verify.log |
| grounding_tests | passed | logs/check_grounding_tests.log |
| zola_check | passed | logs/check_zola_check.log |
| zola_build | passed | logs/check_zola_build.log |

Executable edge checks cover leap-day projection, receipt recoding, interval boundaries, conflicting dates, six-quarter completeness, suppression, employment denominators, duplicate source keys, geographic priority, hash failures and the pre-publication access gate. Build assertions cover ambiguous floors and multi-state itineraries. Tests do not prove the identifying assumptions. No manuscript, source-acquisition runner, production-linkage rebuild or real whole-project pipeline was run.

The appendix contains 347 wage, 220 administrative and 135 pre-policy earnings coefficient rows, including every declared unavailable term. It also retains original decomposition estimates and state-deletion ranges for each fit. The complete state-deletion draws and formulae remain in the linked scratch CSVs. Future results must be produced by the guarded scripts and reviewed in a new report; this current-data report is not overwritten by an automatic future job.

