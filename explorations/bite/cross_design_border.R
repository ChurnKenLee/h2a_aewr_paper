source("explorations/bite/cross_design_helpers.R")
source(path_code("designs","border_discontinuity","redesign.R"))
source(path_code("designs","border_discontinuity","power.R"))
out_dir<-bite_output_dir()
design<-jsonlite::fromJSON(path_code("designs","border_discontinuity","design.json"))
ex<-read_parquet(file.path(out_dir,"exposures.parquet"))
shared<-panel_iv_prepare_outcomes(read_parquet(path_processed("county_year_panel.parquet")),bite_spec)
bp<-as.data.frame(read_parquet(path_processed("border_discontinuity_panel.parquet")))
members<-as.data.frame(read_parquet(path_int("border_discontinuity_geography_memberships.parquet")))
clusters<-read_parquet(path_processed("panel_iv_county_year.parquet")) %>% distinct(county_fips,aewr_iv_cluster_id)
bp<-left_join(bp,shared %>% select(county_fips,year,all_of(setdiff(bite_outcomes,names(bp)))),by=c("county_fips","year"))
bp<-left_join(bp,clusters,by="county_fips",relationship="many-to-one")
bp<-inner_join(bp,ex %>% select(county_fips,floor,gap,baseline_aewr,cz_id,all_of(bite_variants)),by="county_fips",relationship="many-to-one")
leads<-shared %>% distinct(aewr_region_id,year,aewr_ppi) %>% transmute(aewr_region_id,year=year-1L,A_lead=aewr_ppi)
bp<-left_join(bp,leads,by=c("aewr_region_id","year"),relationship="many-to-one")
rows<-boots<-checks<-pairdiag<-pretests<-perms<-list(); sims<-list()
for(geo in c("ring_h1_disjoint","ring_h1_screened_disjoint")) {
  member<-members[members$spec_id==geo,]
  ee<-inner_join(member,ex %>% select(-aewr_region_id),by="county_fips")
  # Pair means average all member counties equally, frozen before policy years.
  pairmean<-ee %>% group_by(region_pair_id) %>% summarise(across(all_of(bite_variants),mean,na.rm=TRUE),.groups="drop")
  names(pairmean)[-1]<-paste0(names(pairmean)[-1],"_pair")
  pp<-left_join(bp,member %>% select(county_fips,region_pair_id),by="county_fips",relationship="many-to-one") %>%
    left_join(pairmean,by="region_pair_id",relationship="many-to-one") %>% select(-region_pair_id)
  sides<-ee %>% group_by(region_pair_id,aewr_region_id) %>% summarise(n=n(),gap=mean(gap),floor=mean(floor),A=mean(baseline_aewr),E=mean(E_gap),.groups="drop")
  pairdiag[[geo]]<-sides %>% group_by(region_pair_id) %>% summarise(geography=geo,counties=sum(n),gap_difference=diff(gap),
    floor_difference=diff(floor),aewr_difference=diff(A),E_difference=diff(E),E_pair=weighted.mean(E,n),.groups="drop")
  row<-data.frame(model_id=geo,spec_id=geo,weight_spec="equal_county",control_spec="none")
  for(nm in names(bite_outcomes)) for(v in bite_variants) {
    cfg<-design; cfg$outcome<-bite_outcomes[[nm]]
    local<-pp[is.finite(pp[[paste0(v,"_pair")]]) & is.finite(pp[[v]]),]
    prepared<-border_prepare(local,members,row,cfg)
    d<-prepared$data; d$main_a<-d$border_x; d$E<-d[[paste0(v,"_pair")]]; d$X<-d$main_a*d$E
    d$L<-d$A_lead*d$E; d$G<-d$main_a-d$floor
    for(id in c("baseline","heterogeneity","frozen_floor_gap","lead")) {
      if(v!="E_gap" && id%in%c("baseline","frozen_floor_gap")) next
      dd<-if(id=="lead") bite_complete(d,c("L","A_lead")) else d
      rhs<-switch(id,baseline="main_a",heterogeneity="main_a + X",frozen_floor_gap="G",lead="main_a + X + A_lead + L")
      terms<-switch(id,baseline="main_a",heterogeneity=c("main_a","X"),frozen_floor_gap="G",lead=c("X","L"))
      m<-bite_fit("border_y",rhs,"unit_fe_id + stratum_year_id",dd)
      meta<-tibble(geography=geo,outcome=nm,exposure=v,model=id)
      rr<-bite_rows(m,dd,terms,meta)
      sm<-summary(m,vcov=~region_pair_id); ct<-coeftable(sm); ci<-confint(sm,terms)
      pairrows<-bind_cols(meta[rep(1,length(terms)),],tibble(term=terms,cluster="region_pair_id",clusters=n_distinct(dd$region_pair_id),
        estimate=ct[terms,1],se=ct[terms,2],p=ct[terms,4],low=ci[,1],high=ci[,2],n=nobs(m),counties=n_distinct(dd$county_fips),
        first_year=min(dd$year),last_year=max(dd$year),mean_y=mean(dd$border_y),mde=2.8*ct[terms,2],mde_pct_mean=280*ct[terms,2]/abs(mean(dd$border_y))))
      rows[[length(rows)+1]]<-bind_rows(rr,pairrows)
      if(nm=="cert" || v=="E_gap") for(term in terms) {
        # Reuse the branch's WCR11-FNW helper, with target as first regressor.
        xx<-model.matrix(as.formula(paste("~ 0 +",rhs)),dd)
        xx<-xx[,c(term,setdiff(colnames(xx),term)),drop=FALSE]
        dd$border_x<-xx[,1]; colnames(xx)[1]<-"border_x"
        for(cn in colnames(xx)) dd[[cn]]<-xx[,cn]
        ff<-border_fit(list(data=dd,x=xx,y=dd$border_y))
        wb<-border_wild(ff,design$bootstrap)
        reg<-tryCatch(bite_wild(m,dd,term),error=function(e)tibble(wild_p=NA_real_,wild_status=conditionMessage(e)))
        boots[[length(boots)+1]]<-bind_cols(meta,tibble(term=term),wb,reg %>% select(-any_of("term")))
        if(nm=="cert" && id=="heterogeneity" && term=="X") {
          pre<-border_prepare(local,members,row,cfg,period="placebo",require_treatment=FALSE)
          keep<-intersect(unique(pre$data$county_fips),unique(dd$county_fips))
          stopifnot(setequal(keep,unique(dd$county_fips)),setequal(keep,unique(pre$data$county_fips)))
          sims[[length(sims)+1]]<-list(model_id=paste(geo,v,sep="__"),policy=list(data=dd,x=xx,y=dd$border_y),pre=pre)
        }
      }
      if(id=="frozen_floor_gap") {
        a<-resid(bite_fit("main_a","1","unit_fe_id + stratum_year_id",dd))
        g<-resid(bite_fit("G","1","unit_fe_id + stratum_year_id",dd))
        cy_a<-resid(bite_fit("main_a","1","stratum_year_id",dd))
        cy_f<-resid(bite_fit("floor","1","stratum_year_id",dd))
        checks[[length(checks)+1]]<-bind_cols(meta,tibble(max_abs_FWL_gap_A=max(abs(a-g)),raw_within_component_A_var=var(cy_a),
          raw_within_component_floor_var=var(cy_f),raw_within_component_cov= cov(cy_a,cy_f),
          raw_within_component_gap_var=var(cy_a-cy_f),policy_FWL_variance=var(a)))
      }
    }
    pre<-border_prepare(local,members,row,cfg,period="placebo",years=c(2008,2010))
    dd<-pre$data; dd$X<-dd$border_x*dd[[paste0(v,"_pair")]]
    m<-bite_fit("border_y","border_x + X","unit_fe_id + stratum_year_id",dd)
    ct<-coeftable(summary(m,vcov=~region_pair_id))
    pretests[[length(pretests)+1]]<-tibble(geography=geo,outcome=nm,exposure=v,estimate=ct["X",1],se=ct["X",2],p=ct["X",4],
      mde=2.8*ct["X",2],mean_y=mean(dd$border_y),years="2008;2010",status="already_exposed_reversal_contaminated_not_clean_pretrend")
  }
  # County-adjacency audit restricts to pairs assigned to this geography.
  adj<-read_parquet(path_int("county_adjacency2010.parquet")) %>% filter(county_fips<neighbor_county_fips)
  edge<-inner_join(adj,ee %>% select(county_fips,region_pair_id,aewr_region_id,cz_id,floor,gap,baseline_aewr),by="county_fips") %>%
    inner_join(ee %>% select(county_fips,region_pair_id,aewr_region_id,cz_id,floor,gap,baseline_aewr),
      by=c("neighbor_county_fips"="county_fips","region_pair_id"),suffix=c("_left","_right")) %>%
    filter(aewr_region_id_left!=aewr_region_id_right) %>% mutate(same_cz=cz_id_left==cz_id_right,
      floor_difference=floor_left-floor_right,gap_difference=gap_left-gap_right,aewr_difference=baseline_aewr_left-baseline_aewr_right)
  write_csv(edge,file.path(out_dir,paste0("border_edges_",geo,".csv")))
}
write_csv(bind_rows(rows),file.path(out_dir,"border_estimates.csv"))
write_csv(bind_rows(boots),file.path(out_dir,"border_bootstrap.csv"))
write_csv(bind_rows(checks),file.path(out_dir,"border_gap_identity.csv"))
write_csv(bind_rows(pairdiag),file.path(out_dir,"border_pair_exposure.csv"))
write_csv(bind_rows(pretests),file.path(out_dir,"border_preperiod.csv"))
saveRDS(sims,file.path(out_dir,"border_bite_simulation_inputs.rds"))
# Same declared DGPs, seeds, 999 calibration/simulation draws, signed grids,
# size gates and one-thread guard. The scalar WCR simulation cannot handle the
# extra AEWR nuisance slope; the multiregressor CRV1/CV3J/dyadic engine can.
design$redesign$power$bootstrap_model_ids<-character()
valid<-vapply(sims,function(m) all(tapply(m$policy$data$aewr_region_id,m$policy$data$region_pair_id,function(z)n_distinct(z))==2),logical(1))
write_csv(tibble(model_id=vapply(sims,`[[`,character(1),"model_id"),status=ifelse(valid,"eligible","unavailable_exposure_removes_entire_border_side")),file.path(out_dir,"border_power_support.csv"))
power<-run_border_power(sims[valid],design)
for(nm in c("power","calibration","mde","simulation_diagnostics")) write_csv(power[[nm]],file.path(out_dir,paste0("border_bite_",nm,".csv")))
