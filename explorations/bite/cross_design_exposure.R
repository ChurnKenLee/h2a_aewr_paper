source("explorations/bite/cross_design_helpers.R")
out_dir <- bite_output_dir()
p <- read_parquet(path_processed("county_year_panel.parquet"))
bite_check_keys(p)
qnames <- paste0("wage_p", c(10,25,50,75,90))
weights <- c(.175,.2,.25,.2,.175)
b <- p %>% filter(year %in% c(2008L,2010L)) %>%
  mutate(floor = pmax(wage_p25, prevailing_ag_min_wage_ppi), gap = aewr_ppi-floor,
    ag_floor = pmax(oews_big_six_mean_hourly_wage/ppi_2012, prevailing_ag_min_wage_ppi),
    ag_gap = aewr_ppi-ag_floor)
affected <- vapply(qnames, function(v) as.numeric(pmax(b[[v]], b$prevailing_ag_min_wage_ppi)<b$aewr_ppi), numeric(nrow(b)))
b$frac <- drop(affected %*% weights)
b$frac_equal <- rowMeans(affected)
ex <- b %>% group_by(county_fips, state_fips, aewr_region_id, cz_id) %>%
  summarise(gap_years = sum(is.finite(gap)), ag_years = sum(is.finite(ag_gap)),
    gap = mean(gap), floor = mean(floor), baseline_aewr = mean(aewr_ppi),
    frac = mean(frac), frac_equal = mean(frac_equal), ag_gap = mean(ag_gap),
    .groups="drop")
# Strict two-year baseline preserves the earlier scratch primary. No later-year
# substitution when a baseline proxy is absent.
ex <- ex %>% filter(gap_years==2, is.finite(gap))
q <- read_parquet(path_int("acs_czone_wage_quantiles.parquet")) %>%
  mutate(county_fips=harmonize_county_fips_2010(county_fips), year=as.integer(YEAR))
bite_check_keys(q)
ppi <- read_parquet(path_int("ppi_2012.parquet"))
aw <- read_parquet(path_int("aewr.parquet")) %>% select(state_fips,year,aewr)
mw <- read_parquet(path_int("state_real_minwages.parquet")) %>% select(state_fips,year,prevailing_ag_min_wage_ppi)
early <- q %>% filter(year %in% 2005:2007) %>%
  mutate(state_fips=state_from_county_fips(county_fips)) %>%
  left_join(ppi,by="year",relationship="many-to-one") %>%
  left_join(aw,by=c("state_fips","year"),relationship="many-to-one") %>%
  left_join(mw,by=c("state_fips","year"),relationship="many-to-one") %>%
  mutate(early_gap=aewr/ppi_2012-pmax(wage_p25/ppi_2012,prevailing_ag_min_wage_ppi)) %>%
  group_by(county_fips) %>% summarise(early_years=sum(is.finite(early_gap)),early_gap=mean(early_gap),.groups="drop")
ex <- left_join(ex,early,by="county_fips",relationship="one-to-one") %>%
  mutate(E_gap=bite_center(gap,aewr_region_id), E_frac=bite_center(frac,aewr_region_id),
    E_frac_equal=bite_center(frac_equal,aewr_region_id),
    E_gap_early=bite_center(ifelse(early_years==3,early_gap,NA_real_),aewr_region_id),
    E_gap_ag=bite_center(ifelse(ag_years==2,ag_gap,NA_real_),aewr_region_id)) %>%
  group_by(aewr_region_id) %>% mutate(rank_gap=(rank(gap,ties.method="average")-.5)/n()) %>% ungroup() %>%
  mutate(E_rank=bite_center(rank_gap,aewr_region_id))
bite_check_keys(ex,"county_fips")
stopifnot(all(is.na(p$aewr_ppi[p$year==2009])), all(abs(tapply(ex$E_gap,ex$aewr_region_id,mean))<1e-10))
write_parquet(ex,file.path(out_dir,"exposures.parquet"))
write_csv(ex,file.path(out_dir,"exposures.csv"))
diag <- bind_rows(lapply(c("gap","frac","frac_equal","early_gap","ag_gap","rank_gap"), function(v) {
  x<-ex[[v]]; w<-x-ave(x,ex$aewr_region_id,FUN=function(z)mean(z,na.rm=TRUE))
  tibble(exposure=v,counties=sum(is.finite(x)),mean=mean(x,na.rm=TRUE),sd=sd(x,na.rm=TRUE),
    within_sd=sd(w,na.rm=TRUE),within_variance_share=var(w,na.rm=TRUE)/var(x,na.rm=TRUE),
    min=min(x,na.rm=TRUE),p10=quantile(x,.1,na.rm=TRUE),median=median(x,na.rm=TRUE),p90=quantile(x,.9,na.rm=TRUE),max=max(x,na.rm=TRUE))
}))
write_csv(diag,file.path(out_dir,"exposure_diagnostics.csv"))
write_csv(as.data.frame(cor(ex[c(bite_variants,"E_frac_equal")],use="pairwise.complete.obs")) %>% tibble::rownames_to_column("exposure"),file.path(out_dir,"exposure_correlations.csv"))
bunch <- p %>% filter(year %in% c(2008,2010,2015)) %>% distinct(year,cz_id,wage_p25) %>% group_by(year) %>%
  summarise(czs=n_distinct(cz_id),distinct_p25=n_distinct(wage_p25),.groups="drop")
write_csv(bunch,file.path(out_dir,"acs_bunching.csv"))
lv <- panel_iv_prepare_outcomes(p,bite_spec) %>% filter(year==2011) %>%
  mutate(any_use=as.numeric(nbr_workers_certified_start_year>0)) %>%
  inner_join(ex %>% select(county_fips,all_of(bite_variants)),by="county_fips")
rr<-list()
for(v in bite_variants) for(y in c("any_use",bite_outcomes[c("cert","applications")])) {
  m<-bite_fit(y,v,"aewr_region_id",lv)
  for(cl in c("state_fips","aewr_region_id")) {
    ct<-coeftable(summary(m,vcov=as.formula(paste0("~",cl))))
    rr[[length(rr)+1]]<-tibble(exposure=v,outcome=y,cluster=cl,n=nobs(m),estimate=ct[1,1],se=ct[1,2],p=ct[1,4])
  }
}
write_csv(bind_rows(rr),file.path(out_dir,"exposure_levels_2011.csv"))
inputs<-c("data/processed/county_year_panel.parquet","data/processed/panel_iv_county_year.parquet",
  "data/intermediate/acs_czone_wage_quantiles.parquet","data/intermediate/aewr.parquet",
  "data/intermediate/ppi_2012.parquet","data/intermediate/state_real_minwages.parquet")
write_csv(tibble(path=inputs,sha256=vapply(inputs,function(f)digest::digest(file=f,algo="sha256"),character(1))),file.path(out_dir,"input_hashes.csv"))
print(diag)
