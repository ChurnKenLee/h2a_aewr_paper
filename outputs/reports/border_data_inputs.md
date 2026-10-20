# Border study: data input improvements

The cached-data corrections, worksite calendar, available historical extensions and denominator comparisons are implemented. The current county comparisons use public data and do not require nonparticipant farm microdata or point geocodes. Non-public NASS data/custom tabulations are unavailable by user instruction; the former access proposal is withdrawn. Exact 2009 statutory determinations and worksite coordinates remain gaps for historical and finer spatial extensions. No causal design is certified by this input work.

## Wage sources and dated references

The source produces 1,764 state-year references for 1990--2025. All 49 canonical 2009 rates are missing; the application means are separately named and retained. The old USDA validation join had dropped 1990--1999. The corrected source retains that history, independent of workbook validation coverage. The cached occupation/skill workbooks supply 297,768 state/SOC/skill/worker-category lookup rows across the 2025--2027 releases.

The annual reference is not every job's wage obligation. Dated FLS intervals and source hashes are retained separately. The 2009 OES regime needs area, occupation, skill and cohort determinations; 2023 onward requires occupational applicability; December 2024 has state/entity injunction timing; prevailing, collective-bargaining, federal/state minimum and promised wages can also bind. The three uncached historical OEWS lookup downloads failed under restricted networking. See the [AEWR contract](../../content/contracts/aewr-inputs.md) and [source audit](../tables/border_data_input_source_coverage.csv).

The shared panel preserves 2009 outcomes when the statutory wage is missing. Its old certification and fixed-2011-denominator values are retained on common rows. Source normalization propagates the wage gap into its lags. Border stage04 explicitly reconstructs the historical nonstatutory comparator; stage07 uses canonical wages and does not require a wage observation for FE-only outcome calibration.

The existing DiD complete-case rule now excludes 2010 because its lagged statutory wage is missing. The MC available-value baseline rule uses 2008 alone for the 2008--2009 fraction affected and 2008/2010 for the longer wage-based means. Separate wage-support counts distinguish these from complete quantile histories. These source corrections require downstream rebuilds; the design registries, prediction cutoff and policy-period treatment definitions are unchanged.

## Worksite allocation and timing

The new producer writes 8,666,143 worksite–county–month rows. It uses inclusive requested site intervals where valid, labels case-date fallback, retains missing weekly hours and equally splits only explicitly ambiguous county lists. It neither replaces site intervals with case-modal dates nor claims actual visa entries or employment. 2 cases fail case-total reconciliation and are quarantined from the new calendar; the [case audit](../tables/border_data_input_unreconciled_cases.csv) retains them.

| fiscal_year | allocated_rows | rows_with_valid_site_dates | site_intervals_different_from_case | rows_without_valid_interval |
| --- | --- | --- | --- | --- |
| 2020 | 94855 | 82783 | 22414 | 0 |
| 2021 | 133221 | 119410 | 20197 | 0 |
| 2022 | 169241 | 152216 | 24250 | 16 |
| 2023 | 197997 | 177641 | 22587 | 0 |
| 2024 | 235834 | 213523 | 23853 | 1 |

The outcome is allocated certification position-days. Requested staffing weights are not observed certified staffing by site, and a crew can rotate among sequential worksites. Hours use reported or case-inherited weekly schedules without mean imputation. County-year artifacts distinguish observed site dates, fallback dates, ambiguous counties, imputed staffing and weekly-hours coverage. The historical case-date certification benchmark remains separate. FLS references are matched by interval overlap, with uncovered days retained. The capacity-weighted nominal reference is a timing diagnostic using outcome-dependent weights, not a predetermined treatment or instrument; none of the denominator regressions uses it. It does not establish the binding job wage.

## Predetermined denominators and precision

Four fixed denominators use the same screened geography, 2011--2022 annual real AEWR, equal county-year weights and county/component-year FE. BEA farm jobs include proprietors; hired farm jobs are the design-owned difference between farm and farm-proprietor jobs. The 2008--2010 means require all three positive observed years. The 2007 Census denominator counts hired-worker reports across farms, not a deduplicated population of people. Suppressed/missing values are not interpolated or repaired with later outcomes.

| sample | denominator | n_counties | n_pairs | estimate_per_1000_denominator_units | pair_CV3J_SE_per_1000 | dyadic_SE_per_1000 |
| --- | --- | --- | --- | --- | --- | --- |
| native | farm_jobs_2011 | 807 | 25 | 2.11226 | 5.56664 | 6.91199 |
| native | farm_jobs_2008_2010 | 807 | 25 | 1.98747 | 5.68024 | 6.99715 |
| native | hired_farm_jobs_2008_2010 | 807 | 25 | 11.7375 | 22.8556 | 27.4097 |
| native | census_hired_workers_2007 | 804 | 25 | 1.84176 | 4.80749 | 5.87481 |
| common_all_denominators | farm_jobs_2011 | 804 | 25 | 2.10448 | 5.58149 | 6.9303 |
| common_all_denominators | farm_jobs_2008_2010 | 804 | 25 | 1.97536 | 5.69472 | 7.01094 |
| common_all_denominators | hired_farm_jobs_2008_2010 | 804 | 25 | 11.6269 | 22.9109 | 27.4038 |
| common_all_denominators | census_hired_workers_2007 | 804 | 25 | 1.84176 | 4.80749 | 5.87481 |

The common-sample rows hold counties and observations fixed across denominators. Changing the denominator changes units and can change the implicit weighting of heterogeneous effects; smaller numerical standard errors are not proof of more information. None of these ratios is a share of uniquely identified people.

Common-sample power uses the existing four declared serial/shared-region/spatial DGPs, 999 simulation and calibration draws, seed 201103 and one thread. Reference effects and signed MDEs are expressed in each denominator's own units; they are not economically identical thresholds. The [power](../tables/border_data_input_power.csv), [calibration](../tables/border_data_input_calibration.csv), [MDE](../tables/border_data_input_mde.csv) and [invalid-variance diagnostics](../tables/border_data_input_simulation_diagnostics.csv) retain failures. Approximate CV3J MDEs in the estimates file are separate from simulation-based size-valid MDEs.

| model_id | inference_method | maximum_null_rejection | maximum_null_Wilson_upper | all_four_DGP_size_gates |
| --- | --- | --- | --- | --- |
| common_all_denominators__census_hired_workers_2007 | pair_CV3J | 0.177177 | 0.202077 | False |
| common_all_denominators__census_hired_workers_2007 | shared_endpoint_dyadic | 0.211211 | 0.237604 | False |
| common_all_denominators__farm_jobs_2008_2010 | pair_CV3J | 0.181181 | 0.206272 | False |
| common_all_denominators__farm_jobs_2008_2010 | shared_endpoint_dyadic | 0.23023 | 0.257339 | False |
| common_all_denominators__farm_jobs_2011 | pair_CV3J | 0.154154 | 0.177867 | False |
| common_all_denominators__farm_jobs_2011 | shared_endpoint_dyadic | 0.203203 | 0.22927 | False |
| common_all_denominators__hired_farm_jobs_2008_2010 | pair_CV3J | 0.17017 | 0.194726 | False |
| common_all_denominators__hired_farm_jobs_2008_2010 | shared_endpoint_dyadic | 0.218218 | 0.244884 | False |

## Historical and policy coverage

| source | first_year | last_year | source_cells | observed_numeric_cells |
| --- | --- | --- | --- | --- |
| AEWR annual reference | 1990 | 2025 | 1764 | 1715 |
| BEA total farm jobs | 2001 | 2022 | 68442 | 68400 |
| Census hired-worker reports | 1997 | 2022 | 18426 | 15175 |
| H2A allocated fiscal worksite/county records | 2008 | 2024 | 1312030 | 1312030 |
| QCEW 111/112 quarterly covered jobs | 2000 | 2024 | 530984 | 299704 |

The source/year/AEWR-region coverage CSV retains reporting cells and observed numeric cells. QCEW measures covered jobs and excludes proprietors and some agricultural workers; suppression is not zero. The wider BEA/Census histories do not extend the shared 2008--2022 estimation backbone or fabricate pre-2008 H-2A outcomes. Later disclosure years likewise do not automatically extend the annual wage design.

The sourced policy ledger contains 11 dated entries, including the enjoined 2009 suspension, 2010 return to FLS, 2021 timing, occupational changes and injunction exceptions. All carry an explicit absence of an exogeneity assumption. Routine FLS/OEWS wage updates reflect economic conditions; methodology changes also alter other program rules; occupation and injunction eligibility need job/entity information. The ledger does not identify a clean differential wage innovation by itself. Annual state minimum-wage references cannot resolve every agricultural exemption or local/contractual floor.

## Spatial inputs and external dependencies

The geocoding producer extracted 852,375 public worksite records and 237,448 distinct normalized addresses. 211,342 are requestable in 22 bounded Census batches. Actual matched coordinates: 0. This run prepared requests and read caches without live acquisition. Pending coordinates stay null. Main disclosure worksite streets are extracted separately from employer mailing addresses, with unresolved case/worksite linkage flagged. Addendum B record IDs align with the B01 allocation IDs when those rows are retained.

Nonparticipant farm records would be needed to estimate a farm-level probability of applying or adopting H-2A: application files cannot enumerate farms that never apply. That is a different estimand from certifications per county farm job. The current county denominators already come from public BEA and Census totals, so this is not a missing input for the implemented analysis. Restricted NASS access and custom tabulations are not pursued.

A possible public-data spatial extension could measure certifications or position-days per predetermined agricultural acre, using a complete geographic frame that includes zero-certification areas. It would measure utilization intensity rather than farm adoption, and would require actual worksite coordinates, boundary/coverage audits and an explicitly declared new outcome. Missing geocodes must not become zero activity. This extension is not implemented or promoted by the current registry.

The [public-data plan and DOL archival draft](border_data_access_requests.md) replaces the former NASS request. The [variable request CSV](../tables/border_data_access_variable_request.csv) now lists only DOL historical concepts and purposes. No institutional-access details are requested for the county analysis. Public downloads failing in the execution environment are a separate technical limitation.

## Reproduction

Run `./scripts/run_border_data_inputs.sh` to reproduce cached source corrections, allocation/calendar and geocoding preparation, shared panel, descriptives, DiD, panel IV, bounded MC, ordinary border, and stages09--11. Inspect its order with `DRY_RUN=1` first. Public acquisition is a separate explicit environment flag. See the branch README for requirements. All generated artifacts have one producer; raw inputs, manifests and separate candidate estimates remain traceable. The input validator checks source keys, missingness, calendar conservation, common samples, simulation settings and provenance; external acquisition is a separate incomplete requirement.

Public definitions: [QCEW coverage](https://www.bls.gov/cew/overview.htm), [Census geocoder](https://geocoding.geo.census.gov/geocoder/Geocoding_Services_API.html), [FLAG wage archives](https://flag.dol.gov/wage-data/adverse-effect-wage-rates).
