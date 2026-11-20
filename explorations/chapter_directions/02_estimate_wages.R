source("explorations/chapter_directions/helpers.R")
out<-id_output_dir();cd_verify()
p<-read_parquet(file.path(out,"wage_filer_state_occupation.parquet"))
cert<-read_parquet(file.path(out,"wage_certified_participation.parquet"))
st<-read_csv(file.path(out,"wage_state_dose.csv"),col_types=cols(state_fips="c",aewr_region_id="c",.default=col_guess())) %>% select(state_fips,dose=dose1_us)
results<-list();deletions<-list();coverage<-list(); distributions<-list()
add<-function(d,y,spec,w,tau,tier,fe="aewr_region_id",rhs="dose",terms="dose") {
  fit<-cd_fit(d,paste(y,"~",rhs,"|",fe),terms,spec,w,tau=tau,tier=tier)
  results[[length(results)+1]]<<-fit$rows;deletions[[length(deletions)+1]]<<-fit$loo
}
pair<-function(d,tau,keys) {
  b<-d %>% filter(fiscal_year==tau-1L) %>% select(-fiscal_year)
  a<-d %>% filter(fiscal_year==tau) %>% select(-fiscal_year)
  inner_join(b,a,by=keys,suffix=c("0","1"),relationship="one-to-one") %>% left_join(st,by="state_fips") %>% mutate(dw=wage1-wage0)
}
fs<-p %>% group_by(tier,filer,state_fips,aewr_region_id,fiscal_year) %>% summarise(wage=cd_mean(wage,h),premium=cd_mean(premium,h),contractor=cd_mean(contractor,h),h=sum(h),.groups="drop")
for(tier in c("R7","R5","R8","FEIN")) for(tau in 2024:2026) {
  cells<-pair(filter(p,.data$tier==.env$tier,soc!="unknown"),tau,c("tier","filer","state_fips","aewr_region_id","soc")) %>%
    mutate(reduce=as.numeric(dw < -0.05),retain=as.numeric(abs(dw)<=.05),dp=premium1-premium0,dnear=near1-near0,
      dcdf0=cdf01-cdf00,dcdf25=cdf251-cdf250,dcdf100=cdf1001-cdf1000,
      dnear_candidate=near_candidate1-near_candidate0,dpremium_lower=premium_lower1-premium_lower0)
  for(w in list("h0",NULL)) for(y in c("dw","reduce","retain","dp","dnear","dcdf0","dcdf25","dcdf100"))
    add(cells,y,paste0("occupation_",y),w,tau,tier)
  for(w in list("h0",NULL)) for(y in c("dnear_candidate","dpremium_lower")) add(cells,y,paste0("schedule_envelope_",y),w,tau,tier)
  unrestricted<-pair(filter(fs,.data$tier==.env$tier),tau,c("tier","filer","state_fips","aewr_region_id"))
  fixed<-cells %>% group_by(tier,filer,state_fips,aewr_region_id,dose) %>% summarise(dw=cd_mean(dw,h0),h_common0=sum(h0),h_common1=sum(h1),.groups="drop") %>%
    left_join(unrestricted %>% select(filer,state_fips,h0,h1,unrestricted_dw=dw),by=c("filer","state_fips"))
  coverage[[length(coverage)+1]]<-tibble(tier=tier,tau=tau,continuing_cells=nrow(cells),unrestricted_filer_states=nrow(unrestricted),common_occupation_filer_states=nrow(fixed),
    baseline_hours_coverage=sum(fixed$h_common0)/sum(unrestricted$h0),current_hours_coverage=sum(fixed$h_common1)/sum(unrestricted$h1),
    mean_unrestricted=cd_mean(unrestricted$dw,unrestricted$h0),mean_fixed=cd_mean(fixed$dw,fixed$h0))
  for(w in list("h0",NULL)) {
    add(unrestricted,"dw","filer_state_unrestricted",w,tau,tier)
    add(fixed,"dw","filer_state_fixed_composition",w,tau,tier)
    add(fixed,"unrestricted_dw","filer_state_unrestricted_common_sample",w,tau,tier)
  }
  if(tier=="R7") {
    mult<-unrestricted %>% group_by(filer) %>% filter(n_distinct(state_fips)>1) %>% ungroup()
    for(w in list("h0",NULL)) add(mult,"dw","within_filer_across_state",w,tau,tier,fe="filer + aewr_region_id")
    base<-fs %>% filter(.data$tier==.env$tier,fiscal_year==tau-1L) %>% select(-fiscal_year) %>% rename(h0=h,wage0=wage,premium0=premium) %>%
      left_join(unrestricted %>% transmute(filer,state_fips,stay=1),by=c("filer","state_fips")) %>% mutate(stay=replace_na(stay,0)) %>%
      left_join(cert %>% filter(.data$tier==.env$tier,fiscal_year==tau) %>% transmute(filer,state_fips,any_cert=as.numeric(positions>0)),by=c("filer","state_fips")) %>%
      mutate(any_cert=replace_na(any_cert,0)) %>% left_join(st,by="state_fips")
    for(w in list("h0",NULL)) for(y in c("stay","any_cert")) add(base,y,paste0("selection_",y),w,tau,tier)
    add(base,"premium0","selection_baseline_premium",NULL,tau,tier,rhs="dose * stay",terms=c("stay","dose:stay"))
    pre<-pair(filter(fs,.data$tier==.env$tier),tau-1L,c("tier","filer","state_fips","aewr_region_id")) %>% select(filer,state_fips,pre_dw=dw)
    add(left_join(base,pre,by=c("filer","state_fips")),"pre_dw","selection_prechange",NULL,tau,tier,rhs="dose * stay",terms=c("stay","dose:stay"))
    if(tau==2026) {
      size<-fs %>% filter(.data$tier==.env$tier,fiscal_year==2025) %>% group_by(filer) %>% summarise(size=sum(h),.groups="drop") %>% mutate(large=as.numeric(size>median(size)))
      cells<-cells %>% left_join(size,by="filer") %>% mutate(highpremium=as.numeric(premium0>.25),contractor=ifelse(is.na(contractor0),NA_real_,as.numeric(contractor0>.5)))
      for(v in c("highpremium","contractor","large")) add(cells,"dw",paste0("heterogeneity_",v),"h0",tau,tier,rhs=paste("dose *",v),terms=paste0("dose:",v))
      # Reference-schedule diagnostic: applicability is unresolved, even here.
      for(w in list("h0",NULL)) add(filter(cells,common_soc0,common_soc1,single_state0,single_state1,receipt_new1),"dw","receipt_aligned_single_state_common_SOC",w,tau,tier)
      distributions[[1]]<-cells %>% summarise(n=n(),mean_baseline=cd_mean(wage0,h0),mean_current=cd_mean(wage1,h0),
        reduction_share=cd_mean(reduce,h0),retention_share=cd_mean(retain,h0),near_reference0=cd_mean(near0,h0),near_reference1=cd_mean(near1,h0))
      for(t in c(0,.25,1,2,5)) distributions[[length(distributions)+1]]<-tibble(n=nrow(cells),threshold=t,baseline_cdf=cd_mean(as.numeric(cells$premium0<=t),cells$h0),current_cdf=cd_mean(as.numeric(cells$premium1<=t),cells$h0))
    }
  }
}
res<-bind_rows(results);res$p_holm<-NA_real_;res$low_simultaneous<-NA_real_;res$high_simultaneous<-NA_real_
k<-grepl("^heterogeneity_",res$specification);res$p_holm[k]<-p.adjust(res$p[k],method="holm",n=3)
res$low_simultaneous[k]<-res$estimate[k]-qt(1-.05/6,res$states[k]-1)*res$se_jackknife[k]
res$high_simultaneous[k]<-res$estimate[k]+qt(1-.05/6,res$states[k]-1)*res$se_jackknife[k]
cd_write(res,out,"wage_estimates.csv");cd_write(bind_rows(deletions),out,"wage_state_deletions.csv");cd_write(bind_rows(coverage),out,"wage_composition_coverage.csv");cd_write(bind_rows(distributions),out,"wage_distributions.csv")
print(res %>% filter(tier=="R7",tau==2026,specification %in% c("occupation_dw","filer_state_fixed_composition","within_filer_across_state")))
