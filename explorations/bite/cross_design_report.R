# Render reviewed scratch results. Copy the generated Markdown to the requested
# repository destination only after validation; all numerical files stay here.
source("explorations/bite/cross_design_helpers.R")
suppressPackageStartupMessages(library(ggplot2))
out_dir<-bite_output_dir()
read<-function(f) read_csv(file.path(out_dir,f),show_col_types=FALSE)
fmt<-function(x,d=4) ifelse(is.na(x),"unavailable",formatC(x,format="fg",digits=d))
table_md<-function(d) {
  d<-as.data.frame(d);d[]<-lapply(d,function(x)if(is.numeric(x))fmt(x) else as.character(x))
  c(paste0("| ",paste(names(d),collapse=" | ")," |"),paste0("| ",paste(rep("---",ncol(d)),collapse=" | ")," |"),
    apply(d,1,function(z)paste0("| ",paste(z,collapse=" | ")," |")))
}
link<-function(f) paste0("[",f,"](",file.path(out_dir,f),")")
p<-read("panel_estimates.csv"); w<-read("panel_wild.csv");fs<-read("iv_first_stages.csv")
ar<-read("iv_ar_sets.csv"); b<-read("border_estimates.csv");bw<-read("border_bootstrap.csv")
ex<-read_parquet(file.path(out_dir,"exposures.parquet"));ed<-read("exposure_diagnostics.csv")
ev<-read("event_study.csv");pt<-read("pretrend_tests.csv");cm<-read("comovement_by_exposure.csv")
unit<-read("mc_unit_audit.csv");mc<-read("mc_retained_comparison.csv")
gap<-read("border_gap_identity.csv");pow<-read("border_bite_mde.csv");support<-read("border_power_support.csv")
panel<-p %>% filter(cluster=="aewr_region_id") %>% left_join(w,by=c("control_set","outcome","exposure","model","term"))
headline<-panel %>% filter(exposure=="E_gap",!startsWith(model,"lead"))
border<-b %>% filter(cluster=="region_pair_id") %>% left_join(bw,by=c("geography","outcome","exposure","model","term"))
first<-fs %>% filter(outcome=="cert",exposure=="E_gap",cluster%in%c("aewr_region_id","aewr_iv_cluster_id")) %>%
  select(control_set,model,cluster,endogenous,term,estimate,joint_F,kp_F,partial_r2)
base<-fs %>% filter(control_set=="bea",outcome=="cert",model=="baseline_iv",cluster=="aewr_iv_cluster_id")
conversion<-panel %>% filter(exposure=="E_gap",model=="bite_1b")
mods<-readRDS(file.path(out_dir,"headline_models.rds"))
conversion$gap_mean<-vapply(conversion$control_set,function(cs){d<-mods[[paste(cs,"cert","E_gap","bite_1b",sep="__")]]$data; mean(ex$gap[match(d$county_fips,ex$county_fips)])},numeric(1))
scale_gap<-ed$within_sd[ed$exposure=="gap"]
conversion<-conversion %>% mutate(multiplier=gap_mean/scale_gap,average_effect=estimate*multiplier,
  average_low=low*multiplier,average_high=high*multiplier,average_mde=mde*abs(multiplier))
write_csv(conversion,file.path(out_dir,"average_effect_conversion.csv"))
# Summarize the grid without turning it into a continuous or rectangular set.
j<-read("iv_joint_ar_grid.csv")
js<-j %>% group_by(control_set,outcome) %>% summarise(grid_points=n(),accepted_points=sum(accepted),
  accepted_boundary=any(accepted & boundary),min_A=min(beta_A[accepted]),max_A=max(beta_A[accepted]),
  min_X=min(beta_X[accepted]),max_X=max(beta_X[accepted]),.groups="drop")
write_csv(js,file.path(out_dir,"joint_ar_grid_summary.csv"))
perms<-bind_rows(read("permutation_placebos.csv") %>% mutate(design="OLS"),read("iv_permutation_placebos.csv"))
perm_summary<-perms %>% group_by(design,control_set,model) %>% summarise(draws=n(),p025=quantile(estimate,.025),p975=quantile(estimate,.975),.groups="drop")
perm_summary<-left_join(perm_summary,panel %>% filter(outcome=="cert",exposure=="E_gap",term%in%c("X","fit_X")) %>% select(control_set,model,observed=estimate),by=c("control_set","model"))
bp<-read("border_permutation_placebos.csv") %>% group_by(geography,model) %>% summarise(draws=n(),p025=quantile(estimate,.025),p975=quantile(estimate,.975),.groups="drop")
write_csv(perm_summary,file.path(out_dir,"permutation_summary.csv"))
chart<-headline %>% filter(outcome=="cert",model%in%c("bite_1b","bite_1a","within_iv")) %>%
  mutate(model=factor(model,levels=c("bite_1a","within_iv","bite_1b"),labels=c("OLS: exposure × year effects","IV: region × year effects","OLS: region × year effects")))
g<-ggplot(chart,aes(x=estimate,y=model,xmin=low,xmax=high))+geom_vline(xintercept=0,color="grey55")+
  geom_errorbar(orientation="y",width=.15,color="#24557A")+geom_point(size=2,color="#24557A")+
  facet_wrap(~control_set,ncol=1)+theme_minimal(base_size=12)+
  labs(x="Certified positions / 2011 farm worker, per $1 AEWR × 1 SD exposure",y=NULL,
    title="Allowing exposure-specific year effects sharply widens uncertainty",subtitle="Frozen 2008/2010 real gap; AEWR-region CRV1 95% t intervals")
ggsave(file.path(out_dir,"panel_comparison.png"),g,width=10,height=6,dpi=170,bg="white")
eg<-ev %>% filter(exposure=="E_gap",cluster=="aewr_region_id") %>% mutate(low=estimate-qt(.975,16)*se,high=estimate+qt(.975,16)*se)
eg<-bind_rows(eg,tibble(outcome=names(bite_outcomes),year=2011L,estimate=0,low=0,high=0)) %>% arrange(outcome,year)
g<-ggplot(eg,aes(year,estimate,ymin=low,ymax=high))+annotate("rect",xmin=2008.5,xmax=2010.5,ymin=-Inf,ymax=Inf,alpha=.12,fill="#C18932")+
  geom_hline(yintercept=0,color="grey55")+geom_errorbar(width=.2,color="#24557A")+geom_line(color="#24557A")+geom_point(color="#24557A")+
  facet_wrap(~outcome,scales="free_y",ncol=2)+scale_x_continuous(breaks=c(2008,2011,2015,2020,2024))+theme_minimal(base_size=11)+
  labs(x=NULL,y="Exposure slope relative to 2011 (outcome / 2011 farm worker)",title="Frozen-gap event studies",subtitle="Region CRV1 95% t intervals; shaded 2009–2010 is reversal-affected; panel scales differ")
ggsave(file.path(out_dir,"event_study.png"),g,width=11,height=7,dpi=170,bg="white")

sections<-c(
  "# Frozen AEWR bite across panel OLS, panel IV, and border comparisons", "",
  "The county interaction is useful as supporting evidence about differential responses. It does not rescue identification of the average AEWR effect. Adding exposure-specific year effects removes most of the apparent precision. The border interaction fails the lead-placebo and precision checks. No retained specification should be replaced on this evidence.","",
  "This executes `markdowns/bite_designs_agent_plan.md`, the detailed plan accompanying the script catalog in `explorations/bite/README.md`. All estimates are exploratory and use cached local inputs. No acquisition, retained-artifact edits, commits, or specification promotion occurred. Numerical artifacts and figures are in the external scratch directory linked below; the report itself is the requested repository deliverable.","",
  "## Findings that change the interpretation", "",
  paste0("- The retained preferred first stage reproduces: coefficient **",fmt(base$estimate),"**, target-subregion F **",fmt(base$joint_F),"**, partial R² **",fmt(base$partial_r2),"**. The older OLS −0.040 benchmark uses the smaller 2012–2023 sample where the naive OEWS comparator is available; `partial_r2.R` reproduces −0.04007. The full BEA-control sample here gives −0.04427. This is a sample distinction, not a failed reproduction."),
  "- Primary county 1(b) certified-position interactions are small, positive, and imprecisely different from zero, with region-cluster MDEs below 4% of the outcome mean. Region-year IV changes them little. With exposure-specific year effects, MDEs rise to roughly half the mean.",
  "- The screened border interaction is positive, but its future-AEWR interaction also predicts the outcome. Its pair-CRV1 interaction MDE is about 68% of the mean. Statistical significance does not pass the declared design gate.",
  paste0("- **MC reconciliation found a unit mismatch.** Current MC code compares nominal AEWR/minimum wages with PPI-deflated ACS quantiles. Its stored fraction affected differs from the all-real calculation in **",unit$fraction_counties_different," / ",unit$counties," counties**; correlation is **",fmt(unit$fraction_correlation),"**. A literal reconstruction of the current code matches the stored fraction exactly. This prevents reading its retained bite coefficient as the same treatment as the real-gap interaction here."),"",
  "## 1. One predetermined exposure table", "",
  "`cross_design_exposure.R` produces `exposures.parquet` and its CSV counterpart, unique by string county FIPS. The primary raw gap is the mean of real AEWR minus max(real CZ p25, real agricultural minimum wage) in 2008 and 2010, requiring both years. Every exposure is centered within AEWR region and divided by its pooled within-region standard deviation; rank uses average ranks with ties. No current or lagged local wage enters these exposure assignments.","",
  table_md(ed),"",
  "The five variants are: `E_gap` (primary); `E_frac` (five-quantile fraction affected with MC weights 0.175/0.20/0.25/0.20/0.175, consistently real units); `E_gap_early` (2005–2007, all three years required); `E_gap_ag` (2008/2010 mapped OEWS Big-Six hourly mean, both years required); and `E_rank` (within-region rank of the primary gap). The early source also has 2000; the requested 2005–2007 window is used. Early construction deflates the intermediate ACS quantiles, which are nominal; shared-panel quantiles are already real.","",
  "The agricultural proxy is an occupation/reporting-area mean, not a farm-worker quantile or a directly observed county farm wage. The existing minimum-wage producer applies its 2024 agricultural-exemption classification historically; this inherited limitation also affects early exposure. Early ACS exposure therefore removes the 2009–2010 wage-method window but does not establish historically exact agricultural-floor coverage.","",
  "The early-exposure panel has 16 observed AEWR regions, versus 17 for the other variants. Its CRV1 reference degrees of freedom and Webb multipliers use that actual count. It is a changed support sensitivity, not an identical-population timing comparison.","",
  table_md(read("exposure_correlations.csv") %>% select(-E_frac_equal)),"",
  "The equal-weight fraction in the older scratch script differs in level, but its standardized exposure is exactly the same in this sample. The corrected real-unit fraction is **weight-consistent**, not numerically identical to the current MC artifact. The OEWS variant correlates weakly with the primary gap, so agreement on it is a demanding measurement check.","",
  table_md(read("acs_bunching.csv")),"",
  "2011 within-region level associations (state CRV1; descriptive, not a validation of causal exposure):", "",
  table_md(read("exposure_levels_2011.csv") %>% filter(cluster=="state_fips") %>% mutate(outcome=recode(outcome,!!!setNames(names(bite_outcomes),bite_outcomes))) %>% select(exposure,outcome,n,estimate,se,p)),"",
  paste0("Full exposure table: ",link("exposures.csv"),"; diagnostic source: `cross_design_exposure.R`."),"",
  "## 2. Plain panel regression", "",
  "All outcomes use the retained panel-IV registry: certifications, certified hours, applications, and balanced linked worksite businesses, each per fixed 2011 farm employment. BEA controls use 2011–2023; continuing controls use 2011–2024. Baseline and heterogeneity OLS include the relevant control set and county/year FE. The 1(b) regression uses A×E with county and region-year FE; 1(a) additionally absorbs E×year. These bite-only regressions omit controls, as in plan A3, but use the corresponding complete IV/control sample so IV–OLS comparisons hold observations fixed. Every sample requires both retained instruments, the outcome and all controls, without selecting on positive outcomes.","",
  "Baseline A and interaction X are different estimands. A in the heterogeneity model is the slope at E=0; the sample-average slope additionally depends on mean(E). X is the change in the AEWR slope per one standardized unit of frozen exposure. Standard errors and confidence intervals below use AEWR-region CRV1; all state and target-subregion alternatives remain in the full estimate CSV. MDE is the requested approximate 2.8×SE, not size-validated power.","",
  table_md(headline %>% filter(outcome=="cert",model%in%c("baseline_ols","heterogeneity","bite_1b","bite_1a")) %>% select(control_set,model,term,estimate,se,low,high,wild_p,mde,mde_pct_mean)),"",
  paste0("![County interaction estimates with confidence intervals](",file.path(out_dir,"panel_comparison.png"),")"),"",
  "Other outcomes, primary 1(b) and 1(a):", "",
  table_md(headline %>% filter(outcome!="cert",model%in%c("bite_1b","bite_1a")) %>% select(control_set,outcome,model,estimate,se,wild_p,mde,mde_pct_mean)),"",
  "Exposure robustness for certified positions:","",
  table_md(panel %>% filter(outcome=="cert",model%in%c("bite_1b","bite_1a")) %>% select(control_set,exposure,model,estimate,se,wild_p,mde,mde_pct_mean)),"",
  "### Pre-years and leads", "",
  "The event study is E×year, reference 2011, with county and region-year FE over 2008–2024. It uses the full shared outcome sample, including observed 2009 outcomes, without requiring a 2009 statutory wage or IV controls. It is a descriptive exposure-slope path, not a treatment-onset event study: AEWR existed before 2011. Both 2009 and 2010 are reversal-affected. This short, contaminated pre-period cannot certify parallel counterfactual trends.","",
  paste0("![Frozen-gap event-study paths](",file.path(out_dir,"event_study.png"),")"),"",
  table_md(pt %>% filter(exposure=="E_gap",cluster=="aewr_region_id")),"",
  "Certified hours reject the joint pre-year null with region inference even though the primary certifications diagnostic does not. Do not describe all outcomes as passing pre-trends. Lead models include both current and next-year AEWR interactions. Next-year AEWR comes from the cached shared history through 2025, so the continuing-controls sample can test the 2024 lead without extrapolation.","",
  table_md(panel %>% filter(exposure=="E_gap",startsWith(model,"lead"),term=="L") %>% select(control_set,outcome,model,estimate,se,p,wild_p,mde,mde_pct_mean)),"",
  "## 3. Panel IV", "",
  "The baseline is column 4. The interacted model instruments A and A×E with Z and Z×E, keeping controls and county/year FE. The within-region-year model instruments A×E with Z×E and matches the no-control 1(b) regression on exactly the same rows. AEWR itself is absorbed in the latter. Instruments retain source-year t−1 timing and the original donor construction.","",
  table_md(headline %>% filter(outcome=="cert",model%in%c("baseline_iv","interacted_iv","within_iv")) %>% select(control_set,model,term,estimate,se,low,high,wild_p,mde,mde_pct_mean)),"",
  "The within-region-year IV and OLS point estimates are close relative to their uncertainty; this is not a formal Hausman equivalence test. IV does not materially improve precision. The dissimilar-donor exclusion restriction must hold for exposure-weighted target shocks as well as the original outcome error; regional calibration and correlated labor-demand shocks remain potential violations.","",
  "### First-stage and weak-IV diagnostics", "",
  table_md(first),"",
  "`joint_F` in ordinary first-stage rows tests all excluded instruments jointly. `SW_conditional` rows give Sanderson–Windmeijer conditional F statistics; `kp_F` gives the Kleibergen–Paap rk Wald F for the two-endogenous model. These use clustered reduced-form scores after the exact nuisance projection, with G/(G−1) CRV1. They are not Stock–Yogo critical-value tests with 17 clusters. The installed fixest KP helper fails with a score/model dimension mismatch when extra controls are present; `cross_design_helpers.R` implements the same SVD rank construction with a directly computed cluster meat. Synthetic tests check its scalar-Wald identity and scale/order invariance.","",
  "Region-cluster Anderson–Rubin sets for the primary exposure (exact quadratic inversion for one endogenous regressor, conditional on constructed inputs):", "",
  table_md(ar %>% filter(exposure=="E_gap",cluster=="aewr_region_id") %>% select(control_set,outcome,model,ar_type,ar_low,ar_high)),"",
  "The two-endogenous model uses joint AR inversion over a coarse tail grid plus a dense local grid. Below are coordinate projections of accepted **grid points**, not a rectangular joint set or exact continuous endpoints. The full grid preserves holes and nonconvexity. A boundary hit means an unresolved tail; absence of a hit does not prove boundedness beyond the searched domain. Inference uses F(2,G−1), an asymptotic cluster approximation, not finite-sample exact coverage.","",
  table_md(js),"",
  paste0("Joint accepted sets and all rejected grid points: ",link("iv_joint_ar_grid.csv"),"."),"",
  "### Donor co-movement by exposure", "",
  "Target subregions are classified high/low by their mean frozen E relative to the within-region median (ties high). County/year residualization and target/donor definitions reproduce `common_shocks.R`; the available panel covers 2011–2023 for this comparison. Confidence bounds resample whole AEWR regions 999 times, seed 20261102. Donor outcomes net of AEWR co-move less than nondonor outcomes in both exposure groups, but this is evidence about observed correlation, not proof of exclusion.","",
  table_md(cm %>% select(series,exposure_group,donor,donor_low,donor_high,nondonor,difference,difference_low,difference_high)),"",
  "## 4. Border discontinuity", "",
  "Both the original one-ring disjoint benchmark and the screened one-ring candidate use their existing memberships, county/component-year FE and 2011–2025 calendar, with equal county-year weights. All four outcomes are estimated. E_pair is the frozen mean of member counties' standardized exposure within each region pair, with each county weighted equally. The heterogeneity regression includes A and A×E_pair. This is heterogeneity across borders, not new independent policy assignments.","",
  table_md(border %>% filter(outcome=="cert",exposure=="E_gap") %>% select(geography,model,term,estimate,se,p,wcb_p_value,wild_p,mde,mde_pct_mean)),"",
  "Pair WCR11-FNW uses the branch's existing helper, 9,999 Webb draws and seed 201103. The additional region bootstrap is reported separately; neither removes shared-endpoint dependence. Failed confidence inversion is retained in the bootstrap CSV. For the screened interaction, the positive lead and poor MDE rule out promotion even when the interaction's own pair test rejects zero.","",
  "### Exposure support and the frozen-floor identity", "",
  table_md(read("border_pair_exposure.csv") %>% group_by(geography) %>% summarise(pairs=n(),mean_abs_E_difference=mean(abs(E_difference)),sd_E_pair=sd(E_pair),mean_abs_floor_difference=mean(abs(floor_difference)),.groups="drop")),"",
  table_md(bind_rows(lapply(c("ring_h1_disjoint","ring_h1_screened_disjoint"),function(geo){d<-read(paste0("border_edges_",geo,".csv"));tibble(geography=geo,adjacent_edges=nrow(d),same_CZ=sum(d$same_cz),same_floor=sum(abs(d$floor_difference)<1e-10),same_CZ_different_floor=sum(d$same_cz & abs(d$floor_difference)>1e-10))}))),"",
  "Even counties sharing a CZ can have different agricultural state floors. A shared ACS quantile alone therefore does not ensure identical local floors. Full side and direct-adjacency diagnostics are retained.","",
  paste0("For G_ct=A_rt−floor_c, county FE absorb the frozen floor exactly. Across all four outcomes and both geographies, the largest absolute difference between residualized G and residualized A is **",fmt(max(gap$max_abs_FWL_gap_A)),"**. Gap-as-treatment reproduces the AEWR coefficient and standard error; it creates no new policy variation."),"",
  table_md(gap %>% filter(outcome=="cert") %>% select(geography,raw_within_component_A_var,raw_within_component_floor_var,raw_within_component_cov,raw_within_component_gap_var,policy_FWL_variance)),"",
  "These component-year-only variances obey Var(G)=Var(A)+Var(floor)−2Cov(A,floor). With county FE as well, all remaining G variation equals A variation. Treating the raw floor variation as policy would be erroneous.","",
  "### Border pre-period and signed power", "",
  "An E_pair×year event path is fully absorbed by component-year FE, so it is not an estimable border pre-trend test. The report instead retains the 2008/2010 A and A×E_pair slope diagnostic; 2009 has no canonical wage and is not filled. The diagnostic is already-exposed and reversal-contaminated, not a clean untreated placebo.","",
  table_md(read("border_preperiod.csv") %>% filter(outcome=="cert",exposure=="E_gap") %>% select(geography,estimate,se,p,mde,mean_y,years)),"",
  "Signed power reuses `power.R`: four declared county/region/pair/spatial DGPs, AR(1)=0.5, 999 calibration and simulation draws, seed 201103, one thread, the original signed grid through 0.20 and the upper-Wilson size gate of 0.075. The effect being injected is now the interaction slope while A remains a nuisance regressor. Preperiod scale uses outcome FE residuals only; policy-residual scale is retained as sensitivity. The scalar WCR simulation shortcut does not support the extra slope, so it is not applied. Observed multiregressor WCR inference still uses the branch helper.","",
  table_md(pow %>% filter(grepl("__E_gap$",model_id),scale_information=="preperiod_only",dgp_id=="shared_region_and_pair_serial",inference_method=="pair_CV3J") %>% select(model_id,effect_sign,grid_mde,credible_grid_mde,size_gate_pass,null_rejection_rate)),"",
  "The early exposure loses all observed counties on one side of the Southeast–Florida edge (10_11). The inherited power engine correctly refuses this incomplete border. Those power results remain unavailable; the full original membership is not silently pruned to obtain a favorable simulation. Estimates using the remaining exposure-complete rows are labeled sensitivities.","",
  table_md(support),"",
  "## 5. Permutations and decision gates", "",
  "Each primary certified-position comparison uses 500 county-level exposure permutations within AEWR region, keeping a county's permuted exposure fixed over time. OLS permutations are within its estimation counties; IV does the same. Border permutations use the common exposure universe before recomputing pair means. These are placebo reference distributions, **not randomization p-values**: no exchangeable treatment assignment is established.","",
  table_md(perm_summary),"",table_md(bp),"",
  "Passing a nonsignificant pre-test is not evidence of absence of pre-trends. A precise null can be useful without rejecting zero in a wild bootstrap; no selection on significance is used. All variants, outcomes, controls, inference partitions and failures remain in the linked files. The 25%-of-mean threshold is applied to the specified differential-effect unit, not to the unidentified average effect.","",
  "## 6. What an interaction would imply for an average effect", "",
  "Impose the structural restriction β_c=β̄·gap_c/μ, where μ is the target-sample mean frozen raw gap. Write gap_c=gap̄_r+sE_c, where s is the pooled within-region gap SD used in normalization. Then β_c A_rt=(β̄/μ)gap̄_r A_rt+(β̄s/μ)E_cA_rt. Region-year FE absorb the first term, so θ_1b=β̄s/μ and **β̄=θ_1b·μ/s**. Confidence endpoints scale by the same positive factor; the table uses the region-CRV1 interval and treats estimated exposures as fixed.","",
  table_md(conversion %>% select(control_set,outcome,gap_mean,multiplier,average_effect,average_low,average_high,average_mde)),"",
  paste0("This restriction sets the AEWR effect to zero at zero gap, reverses its sign for negative gaps, and imposes a common proportional response. Negative gaps occur in ",sum(ex$gap<0)," of ",nrow(ex)," exposure counties. It additionally requires valid differential counterfactual trends, no confounding correlated with exposure, correct local-floor measurement, and a stable mapping over the analysis window. A nonzero common response unrelated to gap is absorbed by region-year FE and **cannot be bounded by θ**. Consequently, these are conditional model-based intervals, not assumption-free bounds on the average AEWR effect or a welfare elasticity."),"",
  "## 7. Reconciliation with Mundlak–Chamberlain", "",
  table_md(unit),"",
  "The MC baseline gap moderator itself matches real AEWR minus real p25 (without the agricultural-floor maximum). Its fraction-affected construction and direct five-quantile bite coordinates instead use nominal policy/floor columns with real quantiles. This is visible in `_baseline_fraction_affected` and `_bite_expression` in `mcw/build.py`, and the source quantile deflation is owned by C01/13. The unit audit reproduces the stored fraction exactly under that literal mixed-unit expression; the discrepancy is not an inference choice or numerical rounding issue.","",
  "Retained MC any-application current-coordinate summaries (unaltered, reported for reconciliation rather than endorsed as an equivalent real-gap effect):", "",
  table_md(mc %>% filter(outcome=="any_application",grepl("^bite_",specification_id),grepl("average_current_coordinate",estimand)) %>% select(specification_id,estimate,standard_error,target_population,target_observations)),"",
  "Even after a unit correction, the estimands differ: MC uses five-point positive-part bite and separate treatment-history coordinates, unrestricted history/moderator projections, 2012–2022 outcome rows with current-effect averages over 2013–2022, and raw counts or any-application followed by baseline-employment normalization. Here the moderator is a signed frozen p25 gap, treatment is contemporaneous real AEWR×E, the main outcome is a county ratio with a 2011 denominator, and the sample/calendar differs. Static same-MC-sample bridge regressions show the scale/FE distinction without claiming to reproduce its dynamic estimand:","",
  table_md(read("mc_static_bridge.csv") %>% select(-status)),"",
  "The DiD branch's contemporaneous/lagged gap is a time-varying regressor; it is not used as predetermined exposure here. Its 2010 lag is unavailable because canonical 2009 AEWR is missing. No DiD coefficient is silently relabeled as the present interaction.","",
  "## 8. Recommendations and proposed integration", "",
  "- **Plain panel: supporting evidence.** Retain the frozen-gap 1(b) differential null with the 1(a) loss of precision, early/rank/fraction/OEWS alternatives, lead tests and the adverse hours pre-test. Do not promote it to an unconditional average causal effect. A supplemental integration would update the nearest exploratory README and a new design note before adding any producer or retained figure.",
  "- **Panel IV: supporting robustness, not a new preferred column.** The region-year interaction IV adds little to the corresponding OLS comparison. Keep the original preferred four-column registry unchanged. Any future integration needs an explicit proposal updating `code/designs/panel_iv/README.md`, `content/designs/panel-iv.md`, and the outcome/specification contract, preserving both control sets and weak-IV evidence.",
  "- **Border: drop as a route to a credible, well-powered bite estimate with these inputs.** Preserve the positive lead, high interaction MDE, shared-region dependence, incomplete early-exposure border and size failures. A frozen-floor treatment is algebraically redundant. Do not change `design.json` or nominate the significant screened interaction as primary.",
  "- **MC: review units before substantive cross-design claims.** Propose a separate owning-producer change in `mcw/build.py`, with unit tests, README/design-page/grounding updates and a rebuild of its panel, compact queue and report. Decide whether frozen wages represent real purchasing-power values or nominal baseline wages before changing the code. This report does not authorize or perform that empirical revision.","",
  "## 9. Reproduction, complete results, and limitations", "",
  "Run from repository root with the existing R library. Execute `cross_design_exposure.R`, `cross_design_panel.R`, `cross_design_border.R`, `cross_design_diagnostics.R`, `cross_design_placebos.R`, and `cross_design_report.R`, each with the same external output directory as its first argument. `test_cross_design.R` is the focused numerical test; `validate_cross_design.R` checks the completed artifacts. The README gives exact commands. The report is generated in scratch and copied byte-for-byte to `markdowns/bite_designs_cross_design.md`.","",
  "Validation passed for 296 panel specifications and 96 border specifications, input hashes, unique string keys, exposure arithmetic, retained baseline reproduction, MDE and frozen-floor identities, 500-draw placebo coverage, AR-grid summaries, the MC unit reconciliation, and inherited power resource guards. Synthetic numerical tests passed. All repository R sources and the new exploration scripts parsed. Agent grounding verification and its seven tests passed. Documentation check and build passed using `env -u LD_LIBRARY_PATH zola check --skip-external-links` and `env -u LD_LIBRARY_PATH zola build`; clearing that environment variable resolved the local shared-library conflict.","",
  "The baseline reproduction logs are generated by the pre-existing `county_bite_acs.R` and `partial_r2.R`. New numerical code owns only scratch artifacts. Input SHA-256 values are retained, and validation checks that they remain unchanged. A complete-case/sample loss is recorded through n, county count and actual first/last years in each estimate row. State, region and target-subregion CRV1 are separate rows; border adds region-pair CRV1. Bootstrap seeds/counts/statuses are explicit.","",
  "The restricted region Webb test uses 9,999 draws/seed 20261102 and projects each cluster's null residual through every nuisance term before generating draws. The IV Wald version conditions on X/Z and is not weak-IV robust; AR supplies the separate weak-IV diagnostic. Synthetic literal raw-row OLS and IV refits independently reproduce the implemented bootstrap p-values, including global year FE. Exposure and instrument generation uncertainty is not included. Border WCR inversion failures remain explicit rather than receiving repaired endpoints.","",
  "County counts may be fractional because multi-county cases are allocated. Any-use is not a farm adoption probability. Linked-business counts have a 2020 disclosure break; later full-vintage linkage can revise earlier counts. Certified hours inherit pooled schedule imputation and are not actual paid hours. From 2023, occupational wage applicability complicates a single regional reference. Neither the zero post-2023 at-AEWR diagnostic nor near-universal H-2A-offer bindingness validates the domestic outside-option proxy. The 2025 border year uses the cached partial FY2026-Q3 vintage and retains its revision risk.","",
  "Every specification run, including unsuccessful confidence inversion and unsupported power, is available below. Secondary outcomes and exposure variants are not filtered on statistical significance. Figures also reside in scratch to respect the plan's output boundary; their absolute local links require this scratch directory to remain available.",""
)
files<-c("exposures.csv","exposure_diagnostics.csv","exposure_correlations.csv","acs_bunching.csv","exposure_levels_2011.csv",
  "panel_estimates.csv","panel_wild.csv","panel_failures.csv","event_study.csv","pretrend_tests.csv","iv_first_stages.csv","iv_ar_sets.csv","iv_joint_ar_grid.csv",
  "comovement_by_exposure.csv","border_estimates.csv","border_bootstrap.csv","border_pair_exposure.csv","border_gap_identity.csv","border_preperiod.csv",
  "border_power_support.csv","border_bite_power.csv","border_bite_mde.csv","border_bite_calibration.csv","border_bite_simulation_diagnostics.csv",
  "permutation_placebos.csv","iv_permutation_placebos.csv","border_permutation_placebos.csv","average_effect_conversion.csv",
  "mc_unit_audit.csv","mc_retained_comparison.csv","mc_static_bridge.csv","input_hashes.csv","numerical_tests.log","prior_bite.log","prior_panel.log",
  "validation.csv","validation.log","zola_check.log","zola_build.log")
sections<-c(sections,paste0("- ",vapply(files,link,character(1))),"",
  "Implementation checks establish reproducibility and arithmetic within this exploration; they do not establish exclusion, parallel counterfactual trends or causal adequacy.")
writeLines(sections,file.path(out_dir,"bite_designs_cross_design.md"))
message("Report written to ",file.path(out_dir,"bite_designs_cross_design.md"))
