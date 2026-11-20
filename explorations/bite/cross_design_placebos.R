source("explorations/bite/cross_design_helpers.R")
out_dir<-bite_output_dir()
ex<-read_parquet(file.path(out_dir,"exposures.parquet"))
p<-panel_iv_prepare_outcomes(read_parquet(path_processed("panel_iv_county_year.parquet")),bite_spec) %>%
  inner_join(ex %>% select(county_fips,E_gap),by="county_fips")
rows<-list(); set.seed(20261102)
for(cs in c("bea","continuing")) {
  ctl<-panel_iv_control_set(bite_spec,cs); controls<-paste(ctl$control_terms,collapse=" + ")
  d<-p %>% filter(year<=ctl$policy_end_year) %>% mutate(A=aewr_ppi,Z=z_wage_seasonal_composition_real,E=E_gap,X=A*E,ZE=Z*E)
  d<-bite_complete(d,c(bite_outcomes[["cert"]],"A","Z","z_wage_only_real",ctl$control_columns,"aewr_iv_cluster_id"))
  county<-distinct(d,county_fips,aewr_region_id,E)
  for(b in 1:500) {
    ep<-county %>% group_by(aewr_region_id) %>% mutate(E=sample(E)) %>% ungroup()
    d$E<-ep$E[match(d$county_fips,ep$county_fips)]; d$X<-d$A*d$E;d$ZE<-d$Z*d$E
    for(id in c("interacted_iv","within_iv")) {
      m<-if(id=="interacted_iv") bite_fit(bite_outcomes[["cert"]],controls,"county_fips + year",d,"A + X ~ Z + ZE") else
        bite_fit(bite_outcomes[["cert"]],"1","county_fips + aewr_region_id^year",d,"X ~ ZE")
      rows[[length(rows)+1]]<-tibble(design="panel_iv",control_set=cs,model=id,draw=b,estimate=coef(m)[["fit_X"]])
    }
  }
}
write_csv(bind_rows(rows),file.path(out_dir,"iv_permutation_placebos.csv"))
members<-read_parquet(path_int("border_discontinuity_geography_memberships.parquet"))
shared<-read_parquet(path_processed("county_year_panel.parquet"))
rows<-list();set.seed(20261102)
for(geo in c("ring_h1_disjoint","ring_h1_screened_disjoint")) {
  mem<-members %>% filter(spec_id==geo) %>% select(county_fips,region_pair_id,stratum_id)
  county<-inner_join(mem,ex %>% select(county_fips,aewr_region_id,E_gap),by="county_fips")
  d<-inner_join(shared,mem,by="county_fips") %>% filter(year>=2011,year<=2025) %>%
    bite_complete(c(bite_outcomes[["cert"]],"aewr_ppi"))
  for(b in 1:500) {
    ep<-ex %>% group_by(aewr_region_id) %>% mutate(E=sample(E_gap)) %>% ungroup() %>% select(county_fips,E)
    ep<-inner_join(mem,ep,by="county_fips") %>% group_by(region_pair_id) %>% summarise(E=mean(E),.groups="drop")
    d$E<-ep$E[match(d$region_pair_id,ep$region_pair_id)];d$X<-d$aewr_ppi*d$E
    m<-bite_fit(bite_outcomes[["cert"]],"aewr_ppi + X","county_fips + stratum_id^year",d)
    rows[[length(rows)+1]]<-tibble(geography=geo,model="heterogeneity",draw=b,estimate=coef(m)[["X"]])
  }
}
write_csv(bind_rows(rows),file.path(out_dir,"border_permutation_placebos.csv"))
