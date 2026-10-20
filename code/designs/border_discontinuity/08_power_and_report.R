# Purpose: Bounded conditional size/power and generated border redesign finding.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
library(jsonlite)
library(readr)
library(ggplot2)
source(path_code("designs", "border_discontinuity", "power.R"))
source(path_code("designs", "border_discontinuity", "bootstrap_power.R"))
design <- fromJSON(path_code("designs", "border_discontinuity", "design.json"))
inputs <- readRDS(path_int("border_discontinuity_redesign_simulation_inputs.rds"))
result <- run_border_power(inputs, design)
write_csv(result$power, path_tables("border_discontinuity_redesign_power.csv"))
write_csv(result$calibration, path_tables("border_discontinuity_redesign_power_calibration.csv"))
write_csv(result$mde, path_tables("border_discontinuity_redesign_mde.csv"))
write_csv(result$simulation_diagnostics, path_tables("border_discontinuity_redesign_simulation_diagnostics.csv"))
bootstrap_power <- run_border_bootstrap_power(result$bootstrap_inputs, design)
write_csv(bootstrap_power, path_tables("border_discontinuity_redesign_bootstrap_power.csv"))

est <- read_csv(path_tables("border_discontinuity_redesign_estimates.csv"),show_col_types=FALSE)
noise <- read_csv(path_tables("border_discontinuity_redesign_noise_diagnosis.csv"),col_types=cols(county_fips=col_character()))
common <- read_csv(path_tables("border_discontinuity_redesign_comparable_samples.csv"),show_col_types=FALSE)
placebo <- read_csv(path_tables("border_discontinuity_redesign_placebo.csv"),show_col_types=FALSE)
balance <- read_csv(path_tables("border_discontinuity_redesign_balance.csv"),show_col_types=FALSE)
deletes <- read_csv(path_tables("border_discontinuity_redesign_region_deletion.csv"),col_types=cols(excluded_region_id=col_character()))
support <- read_csv(path_tables("border_discontinuity_redesign_support.csv"),show_col_types=FALSE)
fwl <- read_csv(path_tables("border_discontinuity_redesign_fwl.csv"),show_col_types=FALSE)
assignment <- read_csv(path_tables("border_discontinuity_redesign_assignment_audit.csv"),col_types=cols(aewr_region_id=col_character()))
legacy <- read_csv(path_tables("border_discontinuity_estimates.csv"),show_col_types=FALSE)
old <- est[est$model_id=="ring_h1_disjoint__base",]
main <- est[est$model_id=="screened_h1__base",]
matched <- est[est$model_id=="screened_h1__region_disjoint",]
weighted <- est[est$model_id=="screened_h1__preemployment_weight",]
fmt <- function(x,d=3) ifelse(is.na(x),"unavailable",formatC(x,format="f",digits=d))
line <- function(...) paste0(...)
md_table <- function(frame) {
  frame[] <- lapply(frame,function(x) if(is.numeric(x)) fmt(x) else as.character(x))
  c(paste0("| ",paste(names(frame),collapse=" | ")," |"),
    paste0("| ",paste(rep("---",ncol(frame)),collapse=" | ")," |"),
    apply(frame,1,function(x) paste0("| ",paste(x,collapse=" | ")," |")))
}
comparison_table <- data.frame(Model=est$model_id,Counties=est$n_counties,Pairs=est$n_pairs,
  Estimate_per_1000=1000*est$estimate,Pair_CV3J_SE_per_1000=1000*est$crv3_jackknife_standard_error,
  Dyadic_SE_per_1000=1000*est$dyadic_standard_error,
  Pair_WCR_low_per_1000=1000*est$wcb_confidence_low,Pair_WCR_high_per_1000=1000*est$wcb_confidence_high)
pre_table <- data.frame(Model=placebo$model_id,Prior_slope_per_1000=1000*placebo$estimate,
  Pair_SE_per_1000=1000*placebo$crv1_standard_error,
  Approximate_MDE_per_1000=1000*placebo$approximate_pair_mde)
nulls <- result$power[result$power$effect==0 & result$power$scale_information=="preperiod_only",]
null_table <- aggregate(cbind(rejection_rate,mc_confidence_high,invalid_repetitions)~inference_method+dgp_id,nulls,max)
mde <- result$mde[result$mde$dgp_id=="shared_region_and_pair_serial" &
  result$mde$scale_information=="policy_residual_sensitivity" &
  result$mde$inference_method=="pair_CV3J",]
mde_table <- data.frame(Model=mde$model_id,Sign=mde$effect_sign,
  Raw_MDE_per_1000=1000*mde$grid_mde,Size_valid_MDE_per_1000=1000*mde$credible_grid_mde,
  Size_gate=mde$size_gate_pass,Null_rejection=mde$null_rejection_rate)
wb <- bootstrap_power[bootstrap_power$scale_information=="preperiod_only",]
wb_rows <- list()
for (id in design$redesign$power$bootstrap_model_ids) {
  for (dgp in design$redesign$power$dgp_ids) {
    z <- wb[wb$model_id==id & wb$dgp_id==dgp,]
    null <- z[z$effect==0,]
    neg <- z[abs(z$effect+.01)<1e-12,]
    pos <- z[abs(z$effect-.01)<1e-12,]
    wb_rows[[length(wb_rows)+1]] <- data.frame(Model=id,DGP=dgp,
      Null_rejection=null$rejection_rate,Null_Wilson_upper=null$mc_confidence_high,
      Power_minus10=neg$rejection_rate,Power_plus10=pos$rejection_rate,Size_gate=null$size_gate_pass)
  }
}
wb_table <- do.call(rbind,wb_rows)
main_deletion <- deletes[deletes$model_id==main$model_id & deletes$status=="ok",]
main_leverage <- support[support$model_id==main$model_id,]
central <- design$redesign$precision_assumptions$primary_threshold_beta
interval_half <- pmax((est$crv3_jackknife_confidence_high-est$crv3_jackknife_confidence_low)/2,
  (est$dyadic_confidence_high-est$dyadic_confidence_low)/2,na.rm=TRUE)
new_rows <- est$model_id %in% design$redesign$models$model_id
# Local data do not establish the causal identifying restriction. Sampling power
# cannot change this evidence status. The report keeps conditional precision separate.
decisions <- list()
shared_dgps <- c("shared_region_serial","shared_region_and_pair_serial","spatial_region_serial")
for (id in est$model_id[new_rows]) {
  e <- est[est$model_id==id,]
  for (method in c("pair_CV3J","shared_endpoint_dyadic")) {
    z <- result$power[result$power$model_id==id & result$power$inference_method==method &
      result$power$scale_information=="preperiod_only" & result$power$dgp_id %in% shared_dgps &
      abs(abs(result$power$effect)-central)<1e-12,]
    mm <- result$mde[result$mde$model_id==id & result$mde$inference_method==method &
      result$mde$scale_information=="preperiod_only" & result$mde$dgp_id %in% shared_dgps,]
    half <- if(method=="pair_CV3J") (e$crv3_jackknife_confidence_high-e$crv3_jackknife_confidence_low)/2 else
      (e$dyadic_confidence_high-e$dyadic_confidence_low)/2
    size_ok <- nrow(z)==6 && all(z$size_gate_pass)
    min_power <- if(nrow(z)) min(z$rejection_rate) else NA_real_
    max_mde <- if(nrow(mm)==6 && all(is.finite(mm$credible_grid_mde))) max(mm$credible_grid_mde) else NA_real_
    precision_ok <- size_ok && is.finite(half) && half<=central && is.finite(min_power) && min_power>=.8 &&
      is.finite(max_mde) && max_mde<=central
    decisions[[length(decisions)+1]] <- data.frame(model_id=id,inference_method=method,
      central_effect=central,minimum_signed_central_power=min_power,
      all_shared_dgp_size_gates=size_ok,maximum_size_valid_mde=max_mde,
      confidence_halfwidth=half,interval_precision_pass=is.finite(half)&&half<=central,
      conditional_precision_pass=precision_ok,identification_gate_pass=FALSE,overall_requirement_met=FALSE)
  }
}
decisions <- do.call(rbind,decisions)
write_csv(decisions,path_tables("border_discontinuity_redesign_decision.csv"))
decision_table <- data.frame(Model=decisions$model_id,Method=decisions$inference_method,
  Min_power_at_both_signs=decisions$minimum_signed_central_power,
  All_shared_DGP_size_gates=decisions$all_shared_dgp_size_gates,
  Maximum_valid_MDE_per_1000=1000*decisions$maximum_size_valid_mde,
  CI_halfwidth_per_1000=1000*decisions$confidence_halfwidth,
  Conditional_precision_pass=decisions$conditional_precision_pass)

identification_status <- "not_established_with_current_local_data"
empirical_status <- "infeasible_to_claim_credible_and_informative_causal_redesign_from_current_inputs"

plot_frame <- est
plot_frame$model_id <- factor(plot_frame$model_id,levels=rev(est$model_id))
figure <- ggplot(plot_frame,aes(x=1000*estimate,y=model_id,xmin=1000*dyadic_confidence_low,xmax=1000*dyadic_confidence_high))+
  geom_vline(xintercept=c(-10,10),linetype="dashed",color="grey65")+
  geom_vline(xintercept=0,color="grey70")+geom_errorbar(orientation="y",width=.15)+geom_point(color="#0072B2")+
  labs(x="Certifications per 1,000 fixed-2011 farm workers per $1/hour (2012 dollars)",y=NULL,
    title="Border redesign: conditional precision and shared-region dependence",
    subtitle="Dyadic t intervals are a small-node sensitivity; dashed lines mark assumed ±10 meaningful effects")+
  theme_minimal(base_size=10)
ggsave(path_figures("border_discontinuity_redesign_precision.png"),figure,width=11,height=5,dpi=220)

report <- c(
  "# AEWR-border redesign: bounded investigation and implementation",
  "",
  line("Generated by `08_power_and_report.R`; registry `",design$design_id,"`; seed 201103. All effects below are per $1/hour of real AEWR in2012 dollars. These are exploratory comparisons informed by the v5 results, not a preregistration or held-out confirmation."),
  "",
  "## Finding",
  "",
  "The stricter local agricultural comparison reduces the original design's outcome noise substantially. It does **not** establish a credible, adequately precise causal design with the current local inputs. The bounded redesign and inference investigation are implemented; the empirical causal objective remains unresolved. This is an explicit infeasibility finding for the requested causal claim using the available data, not a finding of no effect and not a claim that more rows or a significant coefficient would solve identification.",
  "",
  line("The original one-ring pair-CRV1 SE was ",fmt(1000*old$crv1_standard_error)," certifications per1,000 baseline farm workers. The all-member-screened candidate reduces it to ",fmt(1000*main$crv1_standard_error)," (a factor of ",fmt(old$crv1_standard_error/main$crv1_standard_error,2),"). Its dyadic SE is ",fmt(1000*main$dyadic_standard_error),", and its pair-WCR95% interval is [",fmt(1000*main$wcb_confidence_low),", ",fmt(1000*main$wcb_confidence_high),"]. The region-disjoint sample has only ",matched$n_pairs," pairs and its CV3J SE is ",fmt(1000*matched$crv3_jackknife_standard_error),"."),
  "",
  "The working precision standard, declared before new outcome comparisons, is two-sided alpha 0.05, power 0.80, and an effect of10 certifications per1,000 fixed-2011 farm workers per $1;5 and20 are sensitivity magnitudes. These are substantive assumptions rather than user-supplied welfare thresholds. Conditional confidence width, null rejection size and signed MDE are reported separately. No coefficient sign, p-value or placebo nonrejection selects a model.",
  "",
  "## Why the starting design was imprecise",
  "",
  line("The reproduced original has ",old$n_counties," counties and ",old$n_observations," observations, but ",old$n_pairs," border edges share only ",old$n_regions," regions. Its signed region-incidence rank is ",old$region_graph_incidence_rank,"; this is an algebraic contrast ceiling, not an independent-cluster count. Treatment's within SD is ",fmt(old$treatment_within_sd)," versus raw SD ",fmt(old$treatment_raw_sd),". Effective pair leverage count is ",fmt(old$effective_pair_leverage_count,2),"."),
  "",
  line("Counties added beyond the old core contribute ",fmt(100*sum(noise$treatment_ss_share[!noise$in_old_core]),2),"% of old one-ring residual treatment sum of squares and ",fmt(100*sum(noise$rss_share[!noise$in_old_core]),2),"% of its residual outcome sum of squares. The largest county contribution is FIPS",noise$county_fips[1]," at ",fmt(100*noise$rss_share[1],2),"% of residual outcome SS. The old endpoint screen did not apply to ring additions. This observed concentration motivated evaluating the SAME frozen thresholds for every member; no county was removed because of its outcome or residual."),
  "",
  "The noise diagnosis, complete-case/singleton losses, original annual coefficients, leave-one-pair estimates, exact county/stratum/region/pair FWL decompositions and all old geography variants remain retained. Denominator noise, contractor/application allocation and episodic certifications can dominate a county rate; a certification is neither a unique worker nor a realized employment spell. No downstream source correction or outcome winsorization was used.",
  "",
  "## What changed and the causal parameter",
  "",
  "The structural candidate applies the unchanged2007 cropland-share7.5%,2008–2010 employment-density30jobs/100sqmi and90%coverage requirements to ALL members before disjoint assignment. It then prunes counties whose path within their retained own-region counties is farther than one adjacency step from a surviving qualifying border endpoint, or whose distance to a surviving opposite core exceeds the frozen feature caliper. Core endpoints keep the original caliper exemption. Supported components stay connected and cross an actual AEWR boundary. This narrows the target population; it does not put farms physically at the county edge.",
  "",
  "The direct treatment, outcome C_it/F_i,2011,2011–2022 years, and county plus component-year FE remain unchanged. The fixed denominator is the retained2011 convention and is not strictly pre2011. The weighted sensitivity uses mean2008–2010 farm employment as a fixed WLS weight: it averages F_pre*C/F_2011, and is NOT the ratio of total certifications to F_pre. The history sensitivity adds standardized2008–2010 mean outcome × linear time. Neither uses policy-period participation as eligibility. The region-disjoint maximum matching maximizes pair count, then surviving eligible core edges, then lexical IDs using geography alone.",
  "",
  "The base equation is Y_it=beta W_r(i)t+countyFE+componentYearFE+error. With constant weights, balanced support and no additional controls, it equals weighted side-outcome gaps regressed on wage gaps with component intercepts; the side-gap weight is W_left*W_right/(W_left+W_right). Numerical FWL and gap checks establish this equivalence for those models. The history-adjusted model retains county-level covariate variation and is verified by FWL, not by this simple side-gap collapse. Renaming the base design difference-in-discontinuities does not add policy variation. Beta is a direct wage slope/projection in certifications per fixed baseline worker, not an elasticity or an IV coefficient; heterogeneous effects need additional restrictions for an average causal interpretation.",
  "",
  "The counterfactual restriction is that residual untreated outcome-gap changes are orthogonal to residual wage-gap changes, conditional on the stated FE/adjustment. Local agriculture and covariate similarity do not establish this. AEWR moves with regional farm wages; labor shortages and crop-demand shocks can change both. State wage floors/enforcement, differential trends, anticipation, sorting and cross-border contractor/labor-market spillovers remain potential confounders. Region-year FE absorb the entire policy treatment, so adding them cannot retain an identified direct slope. The local border-panel approach and state/border dependence have precedent in [Dube, Lester and Reich(2010)](https://irle.berkeley.edu/wp-content/uploads/2010/11/Minimum-Wage-Effects-Across-State-Borders.pdf), but its assumptions must be assessed in this setting.",
  "",
  "## Estimates and comparable precision",
  "",
  "All entries are per1,000 baseline workers; all old and new candidates are retained. Pair procedures assume independent pairs, an especially strong restriction for pairs sharing regions. Dyadic intervals additionally remain a finite-node sensitivity, not an exact reference law.",
  "",md_table(comparison_table),"",
  line("There are ",nrow(common)," comparable-sample model rows in `border_discontinuity_redesign_comparable_samples.csv`, including v5h1 versus screenedh1, old core versus screenedh1, weighting/history changes and region matching. They report each model's estimate, SE, residual SS and treatment SS on common county-years while retaining its own FE/weights. Thus changed locality/population, residual noise, treatment support and inferential assumptions can be distinguished. The screened candidate's effective pair leverage count is ",fmt(main$effective_pair_leverage_count,2)," and maximum pair share is ",fmt(100*main$maximum_pair_leverage,2),"%."),
  "",
  line("On the same screened county-years, the old geography's CRV1 SE is ",
    fmt(1000*common$crv1_standard_error[common$comparison_id=="ring_h1_disjoint__base__versus__screened_h1__base" & common$model_id=="ring_h1_disjoint__base"]),
    " per1,000, versus ",fmt(1000*main$crv1_standard_error)," under the new geography. Most of the improvement therefore comes from the changed agricultural population and noise, not additional independent policy variation. Old-to-screened residual SS changes from ",fmt(old$residual_outcome_ss,2)," to ",fmt(main$residual_outcome_ss,2),", while partial treatment SS changes from ",fmt(old$treatment_partial_ss,2)," to ",fmt(main$treatment_partial_ss,2),". The matched sample's effective pair leverage count is ",fmt(matched$effective_pair_leverage_count,2),", and its largest pair carries ",fmt(100*matched$maximum_pair_leverage,1),"% of partial-treatment leverage."),
  "",
  "## Dependence, power and size",
  "",
  "Pair CRV1,9,999-draw null-imposed WCR11-FNW Webb intervals, and exact delete-pair CV3J preserve the v5 procedures. Raw-row pair+own-region two-way CRV1 remains a comparison under its stated dependence structure. The new signed partial-treatment pair scores also yield dyadic meat=sum_r(incident_pair_scores_r)^2−sum_pair(pair_score)^2, counting every shared endpoint. The observed dyadic correction is R/(R−1), with t_(R−1) sensitivity. Negative raw variances are unavailable and retained, not clipped to fabricate precision. [Aronow, Samii and Assenova(2015)](https://arxiv.org/abs/1312.3398) provides a dyadic sandwich framework; its growing-node assumptions do not make a sparse16-region graph large-sample evidence.",
  "",
  "Region deletion removes ALL comparisons incident to each region and refits. Overlapping region deletions are influence diagnostics, not a valid ordinary cluster-jackknife variance. More disconnected county components do not create new AEWR shocks. Region-disjoint pairs remove shared endpoints but still require limited dependence across neighboring regions. No justified policy-assignment permutation mechanism is available, so no randomization p-value is reported.",
  "",
  line("For the structural candidate, deleting a region moves the estimate across [",fmt(1000*min(main_deletion$estimate)),", ",fmt(1000*max(main_deletion$estimate)),"] per1,000. Full deletions and invalid fits are in the retained region-deletion table."),
  "",
  "The conditional simulation fixes the observed geography/treatment, uses999 draws and seed 201103, and injects fixed signed effect magnitudes independently of beta-hat. AR(1)rho = .5 shocks have frozen variance fractions: county-only;50%region/50%county;25%region/25%signed pair/50%county; and spatial50%region/25%pair/25%county. Spatial innovations use the observed region graph with coefficient.4 and normalized marginal variance. County shocks are shared across duplicate stacked memberships; signed pair shocks survive component-year FE. Each draw repeats the exact FE/weight/control projection. Fractions and persistence are assumptions because three historical years cannot estimate them reliably.",
  "",
  "Primary scale calibration uses2008–2010 OUTCOME RSS after unit/component-year FE only, deliberately excluding the contaminated2009 wage regressor. Simulated unit-scale RSS is projected identically using999 calibration draws, with Monte Carlo error retained. A separate full-policy-residual scale is explicitly retrospective and never selects a candidate. Both are assumed-DGP calibrations, not proof of exogeneity; the short prehistory and institutional changes limit transportability. Size uses effect zero; signed MDEs invert a.001 grid through.20. Size-valid MDE requires no invalid covariance draws and the upper95%Wilson bound for null rejection≤.075. Unavailable/above-grid MDEs and sign asymmetries remain. High raw power from an oversized procedure does not pass.",
  "",
  "Worst observed null rejection and Wilson upper bound across the declared candidate models, by method/DGP (full model-specific curves and binomial uncertainty are retained):",
  "",md_table(null_table),"",
  "Policy-residual-scale, shared-region-plus-pair CV3J MDE sensitivity (a retrospective noise stress, not the primary information set):",
  "",md_table(mde_table),"",
  "The scalar WCR simulation uses the SAME first199 Monte Carlo draws for three declared base models, four DGPs, two calibration scales and seven effects (0,±.005,±.01,±.02), with9,999 Webb draws per null-imposed test. `tests/test_bootstrap_power.R` compares three signed artificial outcomes against the locked fwildclusterboot0.14.3 implementation: maximum t discrepancy2.66e-15 across29,997 draws, identical p-values. The retained CSV records the package version and strict-tail convention. No scalar shortcut is used for fitted controls. A bootstrap creates no new assignments, and DGP size evidence cannot establish policy exogeneity.",
  "",
  "Primary historical-outcome-scale WCR results at the central signed threshold are below. A failed upper-Wilson size gate can mean detected overrejection OR insufficient Monte Carlo precision to certify the threshold; it is not by itself proof that true size exceeds7.5%. The effect-power shortfall and the identification limitation are separate findings.",
  "",md_table(wb_table),"",
  "",
  "## Does the declared precision standard pass?",
  "",
  "The central threshold is10 per1,000 workers. The following primary-preperiod-scale audit requires BOTH effect signs across ALL three shared-region/spatial DGPs to achieve80%power, every size gate to pass, a maximum size-valid MDE≤10, and the observed95%interval halfwidth≤10. The unshared county-only DGP cannot pass this check by itself. 'Unavailable' includes invalid covariance/size evidence, not a zero MDE. Pair-CV3J rows on shared-region data are conditional simulation checks, not an assertion of independent actual pairs. The source/regional-exogeneity limitation separately fails the identification gate for every candidate.",
  "",md_table(decision_table),"",
  line("Conditional precision passes for ",sum(decisions$conditional_precision_pass)," of ",nrow(decisions)," candidate-method combinations. Overall credible-and-informative causal requirements pass for0: the current data do not establish policy-gap exogeneity. `border_discontinuity_redesign_decision.csv` retains each explicit gate. Bootstrap central-threshold rejection rates and size uncertainty are separately retained, and the retrospective noise stress above is not used to choose a primary model."),
  "",
  "## Historical diagnostics and source limitation",
  "",
  "AEWR existed before 2011. The former upstream 2009 extraction selected average certified H-2A application wages from Table 1 of the [2010 final rule](https://www.govinfo.gov/content/pkg/FR-2010-02-12/pdf/2010-2731.pdf), rather than a statutory AEWR schedule. The source correction makes canonical 2009 AEWR missing and retains the application means only in an explicitly nonstatutory field. Stage04 reconstructs that historical comparator; the persistent panel and assignment audit retain the canonical missing wage. The original three-year annual/balance/placebo audit is retained as a contaminated benchmark, not valid evidence for a policy placebo. New earlier-wage diagnostics and wage-balance associations use only 2008/2010. Power calibration uses all three outcome years with FE only. Outcome-history adjustment and employment weights do not use that wage field.",
  "",
  "The2008 rule operated under a different wage methodology during2009 until the2010 rule restored FLS-based treatment. The [court's historical chronology](https://www.govinfo.gov/content/pkg/USCOURTS-ncmd-1_09-cv-00411/pdf/USCOURTS-ncmd-1_09-cv-00411-3.pdf) documents the transition. The two remaining wage observations cannot establish a clean counterfactual trend. Below, approximate pair MDEs describe how weak the earlier-period diagnostic can be; a nonsignificant slope is not proof of parallel trends. [Roth(2022)](https://www.jonathandroth.com/assets/files/roth_pretrends_testing.pdf) explains the limitations of pretrend testing.",
  "",md_table(pre_table),"",
  line("The new baseline-balance table retains ",nrow(balance)," weighted covariate/outcome associations using the valid2008/2010 wage mean. These are observed comparability diagnostics, not randomized balance tests or a selection gate. Annual v5 coefficients remain exploratory; no held-out confirmation is claimed."),
  "",
  "AEWR is one component of the legally applicable wage floor, not every farm's paid wage. FLS regional wages and applicable minimum/prevailing wage rules matter. The [2022 notice](https://www.govinfo.gov/content/pkg/FR-2021-12-15/pdf/2021-27119.pdf) ties rates to FLS and makes them effective December29,2021; annual start-year exposure approximates actual contract timing. A finer design needs effective dates, actual work locations and competing wage floors.",
  "",
  "## Bounded alternatives and concrete requirement",
  "",
  "All old rings, the stricter local screen, frozen employment weights, baseline outcome adjustment and deterministic independent-region matching were implemented. A side-gap representation was verified numerically; it is algebraically the same temporal comparison. Flexible outcome-history fitting is poorly supported by three years and was not substituted for a transparent linear interaction. Predetermined bite would change the estimand to exposure-specific responsiveness and needs an admissible farm-worker wage distribution and differential-trend restriction. Nonfarm p10/p25 or a prediction trained through2011 is not a clean pre2011 farm-wage bite. This candidate is assessed but not selected.",
  "",
  "County aggregates/adjacency have no unit-level distance running variable and cannot identify a pointwise spatial RD. [Cattaneo, Idrobo and Titiunik's RD extensions](https://arxiv.org/abs/2301.08958) requires the relevant assignment geometry and assumptions; county centroids do not supply missing farm outcomes at the boundary. The current2008–2022 panel adds no independent policy units through relabeling, stacking or bandwidth expansion. Extending beyond2022 requires aligned outcomes and treatment under the changed occupation-specific regime.",
  "",
  "Historical extensions need the still-unavailable 2009 OES determinations by area/occupation/skill/cohort and longer consistently measured outcomes and competing-policy histories. Credible identification also needs differential wage-floor changes whose local gap does not track the same regional labor-demand shocks. The current county comparison already uses public labor denominators and does not require farm-level nonparticipant records or point geocodes. Those records would be necessary for a different farm-participation estimand, because applications alone omit farms that never apply. Non-public NASS data/custom tabulations are unavailable by user instruction and are not an access requirement for this project. A separately declared public-data spatial extension could instead study allocated certifications per predetermined agricultural area, requiring worksite geocodes, a complete geographic frame including zero-certification areas, and audited boundaries/coverage; it would measure utilization intensity, not farm adoption. More accurate outcomes could reduce noise, but finer units alone do not increase independent AEWR assignments. Additional data must satisfy BOTH identification and the declared size/MDE target.",
  "",
  "## Reproduction and evidence boundaries",
  "",
  "Run `./scripts/run_border_discontinuity.sh` from the repository root. Stages01–03 own local features/geography/panel,04–05 retain the benchmark audit,07 owns fits/FWL/support/deletions/assignment evidence,08 owns power/report/precision figure, and06 validates all artifacts. Python entry points are Marimo sources exported to a temporary directory; R runs with `Rscript --vanilla` in the existing project library. The coordinated input upgrade corrects upstream AEWR semantics and rebuilds affected shared-panel consumers. Its source history, allocation calendar, denominator comparisons and unresolved external requirements are documented in [border_data_inputs.md](border_data_inputs.md). Reproduce that dependency chain with `./scripts/run_border_data_inputs.sh`.",
  "",
  line("Machine checks include weighted full-dummy/FWL equality, exact side-gap equivalence for models without additional controls, full delete-pair equivalence, dyadic score/rank/scaling identities, geographic threshold/path/matching invariants, fixed baseline inputs, covariance/package checks and bounded simulation. Current largest retained FWL error is ",format(max(abs(fwl$fwl_coefficient_error)),scientific=TRUE),". Stage06 runs after this report; report generation alone does not assert that its later validation passed. The separate input validator records its checks in `outputs/reports/border_data_inputs_validation.json`. Neither validator establishes causal identification."),
  "",
  "The baseline was reproduced before redesign, and the original four memberships were independently reconstructed from preserved source. The later input correction retains canonical 2009 outcome rows while removing the nonstatutory wage from current source fields. Historical benchmark reconstruction is explicit and local to stage04. Provenance hashes below and in the JSON identify the inputs used for this generated report; they are not an assertion that all inputs equal the earlier baseline.",
  ""
)
provenance_paths <- c("code/designs/border_discontinuity/design.json","code/designs/border_discontinuity/02_construct_geographies.py",
 "code/designs/border_discontinuity/geography.py","code/designs/border_discontinuity/03_build_panel.py",
 "code/designs/border_discontinuity/04_estimate_models.R","code/designs/border_discontinuity/07_estimate_redesign.R",
 "code/designs/border_discontinuity/redesign.R","code/designs/border_discontinuity/power.R",
 "code/designs/border_discontinuity/bootstrap_power.R","code/designs/border_discontinuity/08_power_and_report.R",
 "data/processed/county_year_panel.parquet","data/processed/border_discontinuity_panel.parquet",
 "data/intermediate/border_discontinuity_county_features.parquet","data/intermediate/border_discontinuity_geography_memberships.parquet",
 "renv.lock","uv.lock","devenv.lock")
provenance <- list(design_id=design$design_id, empirical_status=empirical_status,
 identification_status=identification_status,generated_utc=format(Sys.time(),tz="UTC",usetz=TRUE),
 R_version=R.version.string,packages=sapply(c("fixest","fwildclusterboot","dqrng"),function(x) as.character(packageVersion(x))),
 files=setNames(lapply(provenance_paths,function(p) digest::digest(file=here::here(p),algo="sha256")),provenance_paths))
report_dir <- here::here("outputs","reports")
dir.create(report_dir,recursive=TRUE,showWarnings=FALSE)
write_json(provenance,file.path(report_dir,"border_discontinuity_redesign_provenance.json"),auto_unbox=TRUE,pretty=TRUE)
writeLines(report,file.path(report_dir,"border_redesign.md"))
message("Generated bounded power evidence and explicit empirical infeasibility finding.")
