source("explorations/bite/cross_design_helpers.R")
out_dir<-bite_output_dir()
ex<-read_parquet(file.path(out_dir,"exposures.parquet"))
p<-panel_iv_prepare_outcomes(read_parquet(path_processed("panel_iv_county_year.parquet")),bite_spec)
groups<-p %>% distinct(county_fips,aewr_region_id,target_cluster) %>%
  inner_join(ex %>% select(county_fips,E_gap),by="county_fips") %>%
  group_by(aewr_region_id,target_cluster) %>% summarise(E=mean(E_gap),.groups="drop") %>%
  group_by(aewr_region_id) %>% mutate(exposure_group=if_else(E>=median(E),"high","low")) %>% ungroup()
write_csv(groups,file.path(out_dir,"target_exposure_groups.csv"))
donors<-read_parquet(path_int("panel_iv_county_donor_frame.parquet")) %>% filter(is_primary) %>%
  distinct(aewr_region_id,target=target_cluster,other=donor_cluster) %>% mutate(is_donor=TRUE)
subs<-p %>% distinct(aewr_region_id,sub=target_cluster) %>% filter(!is.na(sub))
pairs<-inner_join(subs %>% rename(target=sub),subs %>% rename(other=sub),by="aewr_region_id",relationship="many-to-many") %>%
  filter(target!=other) %>% left_join(donors,by=c("aewr_region_id","target","other")) %>% mutate(is_donor=coalesce(is_donor,FALSE)) %>%
  left_join(groups %>% rename(target=target_cluster),by=c("aewr_region_id","target"))
series<-list()
for(nm in c("oews_log_wage","cert_share","cert_net_aewr","applications_net_aewr")) {
  y<-switch(nm,oews_log_wage="lw",cert_share=bite_outcomes[["cert"]],cert_net_aewr=bite_outcomes[["cert"]],applications_net_aewr=bite_outcomes[["applications"]])
  d<-p %>% filter(year<=2023,!is.na(target_cluster)) %>% mutate(lw=log(oews_big_six_mean_hourly_wage/ppi_2012))
  d<-bite_complete(d,c(y,if(grepl("net_aewr",nm)) "aewr_ppi"))
  m<-bite_fit(y,if(grepl("net_aewr",nm)) "aewr_ppi" else "1","county_fips + year",d)
  d<-d[obs(m),]; d$r<-resid(m)
  series[[nm]]<-d %>% group_by(aewr_region_id,sub=target_cluster,year) %>% summarise(v=mean(r),.groups="drop")
}
cr<-list(); set.seed(20261102)
for(nm in names(series)) for(group in c("high","low")) {
  s<-series[[nm]]
  x<-pairs %>% filter(exposure_group==group) %>%
    inner_join(s %>% rename(target=sub,vt=v),by=c("aewr_region_id","target"),relationship="many-to-many") %>%
    inner_join(s %>% rename(other=sub,vo=v),by=c("aewr_region_id","other","year"))
  splitx<-split(x,x$aewr_region_id); regions<-names(splitx)
  getcor<-function(d) c(donor=cor(d$vt[d$is_donor],d$vo[d$is_donor]),nondonor=cor(d$vt[!d$is_donor],d$vo[!d$is_donor]))
  point<-getcor(x)
  draws<-replicate(999,getcor(bind_rows(splitx[sample(regions,replace=TRUE)])))
  cr[[length(cr)+1]]<-tibble(series=nm,exposure_group=group,donor=point[1],nondonor=point[2],difference=point[1]-point[2],
    donor_low=quantile(draws[1,],.025),donor_high=quantile(draws[1,],.975),nondonor_low=quantile(draws[2,],.025),nondonor_high=quantile(draws[2,],.975),
    difference_low=quantile(draws[1,]-draws[2,],.025),difference_high=quantile(draws[1,]-draws[2,],.975),pairs_years=nrow(x))
}
write_csv(bind_rows(cr),file.path(out_dir,"comovement_by_exposure.csv"))

# Unit and estimand reconciliation, without modifying MC artifacts.
mc<-read_parquet(path_processed("mundlak_chamberlain_county_year_v4.parquet"))
mcbase<-mc %>% distinct(county_fips,mc_fraction_affected_f0810,mc_baseline_bite)
compare<-inner_join(ex,mcbase,by="county_fips")
shared<-read_parquet(path_processed("county_year_panel.parquet"))
base<-shared %>% filter(year %in% c(2008,2010))
qnames<-paste0("wage_p",c(10,25,50,75,90)); weights<-c(.175,.2,.25,.2,.175)
literal<-vapply(qnames,function(v)as.numeric(pmax(base[[v]],base$prevailing_ag_min_wage)<base$aewr),numeric(nrow(base)))
base$literal_frac<-drop(literal%*%weights)
literal<-base %>% group_by(county_fips) %>% summarise(literal_frac=mean(literal_frac),real_gap_no_floor=mean(aewr_ppi-wage_p25),.groups="drop")
compare<-left_join(compare,literal,by="county_fips")
audit<-tibble(counties=nrow(compare),max_fraction_error_literal=max(abs(compare$literal_frac-compare$mc_fraction_affected_f0810)),
  max_fraction_error_real=max(abs(compare$frac-compare$mc_fraction_affected_f0810)),
  fraction_counties_different=sum(abs(compare$frac-compare$mc_fraction_affected_f0810)>1e-10),
  fraction_correlation=cor(compare$frac,compare$mc_fraction_affected_f0810),
  max_baseline_gap_error=max(abs(compare$real_gap_no_floor-compare$mc_baseline_bite)))
write_csv(audit,file.path(out_dir,"mc_unit_audit.csv"))
retained<-read_csv(path_tables("mc_v4_constructed_estimands.csv"),show_col_types=FALSE) %>%
  filter(grepl("^bite_|^exposure_log_",specification_id),inference_method=="cr1_cluster_sandwich")
write_csv(retained,file.path(out_dir,"mc_retained_comparison.csv"))
mc<-inner_join(mc,ex %>% select(county_fips,E_gap,E_frac),by="county_fips")
mc$bite_current<-vapply(seq_len(nrow(mc)),function(i)mc[[paste0("mc_bite_f0810_",mc$year[i])]][i],numeric(1))
mc$X<-mc$aewr_ppi*mc$E_gap
bridge<-list()
for(y in c("mc_y_any_application",bite_outcomes[["cert"]],"mc_y_certified_positions")) for(t in c("X","bite_current")) for(fe in c("county_fips + aewr_region_id^year","county_fips + state_fips^year")) {
  m<-bite_fit(y,t,fe,mc); ct<-coeftable(m)
  bridge[[length(bridge)+1]]<-tibble(outcome=y,treatment=t,FE=fe,estimate=ct[1,1],se=ct[1,2],p=ct[1,4],n=nobs(m),
    status="static_bridge_on_MC_sample_not_replication_of_full_history_estimand")
}
write_csv(bind_rows(bridge),file.path(out_dir,"mc_static_bridge.csv"))
print(audit)
