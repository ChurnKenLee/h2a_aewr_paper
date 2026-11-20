# Independent arithmetic and artifact checks; no empirical promotion gate.
source("explorations/bite/cross_design_helpers.R")
out_dir<-bite_output_dir()
read<-function(f)read_csv(file.path(out_dir,f),show_col_types=FALSE)
ex<-read_parquet(file.path(out_dir,"exposures.parquet"));bite_check_keys(ex,"county_fips")
stopifnot(all(vapply(ex[c("county_fips","state_fips","cz_id","aewr_region_id")],is.character,logical(1))))
s<-read_parquet(path_processed("county_year_panel.parquet"));bite_check_keys(s)
baseline<-s[s$year%in%c(2008,2010),]
raw<-baseline$aewr_ppi-pmax(baseline$wage_p25,baseline$prevailing_ag_min_wage_ppi)
means<-tapply(raw,baseline$county_fips,mean)
stopifnot(max(abs(ex$gap-means[ex$county_fips]))<1e-10,abs(sd(ex$E_gap)-1)<1e-10,
  max(abs(tapply(ex$E_gap,ex$aewr_region_id,mean)))<1e-10,all(is.na(s$aewr_ppi[s$year==2009])))
hashes<-read("input_hashes.csv")
stopifnot(all(vapply(hashes$path,function(f)digest::digest(file=f,algo="sha256"),character(1))==hashes$sha256))
p<-read("panel_estimates.csv"); b<-read("border_estimates.csv");w<-read("panel_wild.csv");bw<-read("border_bootstrap.csv")
stopifnot(file.info(file.path(out_dir,"panel_failures.csv"))$size==0,
  nrow(distinct(p,control_set,outcome,exposure,model))==296,
  nrow(distinct(b,geography,outcome,exposure,model))==96,
  !anyDuplicated(p[c("control_set","outcome","exposure","model","term","cluster")]),
  !anyDuplicated(b[c("geography","outcome","exposure","model","term","cluster")]),
  all(is.finite(p$estimate)),all(is.finite(b$estimate)),all(is.finite(w$wild_p)),
  all(w$wild_B==9999),all(w$wild_regions==ifelse(w$exposure=="E_gap_early",16,17)),all(w$wild_p>=0 & w$wild_p<=1),
  all(bw$wcb_repetitions==9999),all(bw$wcb_seed==201103),
  max(abs(p$mde-2.8*p$se))<1e-10,max(abs(b$mde-2.8*b$se))<1e-10)
for(cs in c("bea","continuing")) for(v in bite_variants) for(y in names(bite_outcomes)) {
  d<-p %>% filter(control_set==cs,exposure==v,outcome==y,model%in%c("bite_1b","within_iv"),cluster=="aewr_region_id")
  stopifnot(nrow(d)==2,n_distinct(d$n)==1,n_distinct(d$mean_y)==1,all(d$last_year==if(cs=="bea")2023 else 2024))
}
fs<-read("iv_first_stages.csv")
f<-fs %>% filter(control_set=="bea",outcome=="cert",exposure=="E_gap",model=="baseline_iv",cluster=="aewr_iv_cluster_id")
retained<-read_csv(path_tables("iv_preferred_first_stage_estimates.csv"),show_col_types=FALSE)
stopifnot(abs(f$estimate-.17013555496276947)<1e-9,abs(f$joint_F-44.706664667487736)<1e-7,
  all(is.finite(fs$joint_F)),all(fs$joint_F>0))
gap<-read("border_gap_identity.csv")
stopifnot(max(gap$max_abs_FWL_gap_A)<1e-10,
  max(abs(gap$raw_within_component_gap_var-gap$raw_within_component_A_var-gap$raw_within_component_floor_var+2*gap$raw_within_component_cov))<1e-10)
for(geo in unique(b$geography)) for(y in names(bite_outcomes)) {
  d<-b %>% filter(geography==geo,outcome==y,model%in%c("baseline","frozen_floor_gap"),cluster=="region_pair_id")
  stopifnot(nrow(d)==2,diff(range(d$estimate))<1e-10,diff(range(d$se))<1e-10)
}
old<-read_csv(path_tables("border_discontinuity_redesign_estimates.csv"),show_col_types=FALSE)
for(geo in unique(b$geography)) {
  id<-if(geo=="ring_h1_disjoint")"ring_h1_disjoint__base" else "screened_h1__base"
  d<-b %>% filter(geography==geo,outcome=="cert",model=="baseline",cluster=="region_pair_id")
  r<-old[old$model_id==id,]
  stopifnot(nrow(r)==1,abs(d$estimate-r$estimate)<1e-8,abs(d$se-r$crv1_standard_error)<1e-8,d$n==r$n_observations)
}
for(file in c("permutation_placebos.csv","iv_permutation_placebos.csv","border_permutation_placebos.csv")) {
  d<-read(file);groups<-intersect(c("control_set","geography","model"),names(d))
  sizes<-d %>% group_by(across(all_of(groups))) %>% summarise(n=n(),distinct_draws=n_distinct(draw),.groups="drop")
  stopifnot(all(sizes$n==500),all(sizes$distinct_draws==500))
}
j<-read("iv_joint_ar_grid.csv"); js<-read("joint_ar_grid_summary.csv")
for(i in seq_len(nrow(js))) {
  z<-j[j$control_set==js$control_set[i] & j$outcome==js$outcome[i],]
  accept<-z[z$accepted,]
  stopifnot(nrow(accept)>0,js$accepted_points[i]==nrow(accept),
    abs(js$min_A[i]-min(accept$beta_A))<1e-10,abs(js$max_X[i]-max(accept$beta_X))<1e-10,
    js$accepted_boundary[i]==any(accept$boundary))
}
unit<-read("mc_unit_audit.csv");stopifnot(unit$max_fraction_error_literal<1e-12,unit$fraction_counties_different>0)
power<-read("border_bite_power.csv")
stopifnot(all(power$repetitions==999),all(power$seed==201103),
  !any(power$size_gate_pass & power$invalid_repetitions>0))
report<-readLines(file.path(out_dir,"bite_designs_cross_design.md"))
stopifnot(length(report)>100,all(file.exists(file.path(out_dir,c("event_study.png","panel_comparison.png")))))
summary<-tibble(check=c("input_hashes_unchanged","string_keys_and_exposure_arithmetic","panel_menu_296_models","border_menu_96_models",
  "baseline_reproductions","MDE_and_FWL_gap_identity","three_500_permutation_menus","AR_grid_summary","MC_unit_reconciliation","power_guards"),status="pass")
write_csv(summary,file.path(out_dir,"validation.csv"))
print(summary)
