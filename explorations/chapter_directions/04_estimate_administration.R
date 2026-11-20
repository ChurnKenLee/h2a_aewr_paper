source("explorations/chapter_directions/helpers.R")
out<-id_output_dir();cd_verify()
p<-read_parquet(file.path(out,"admin_cohort_panel.parquet"));b<-read_parquet(file.path(out,"admin_baseline_cohorts.parquet"))
res<-list();los<-list()
add<-function(d,y,spec,w,tier,fe="cohort + state_season_year",ref=2012L) {
  terms<-paste0("year::",setdiff(sort(unique(d$year)),ref),":exposure")
  m<-cd_fit(d,paste0(y," ~ i(year, exposure, ref = ",ref,") | ",fe),terms,spec,w,track="administration",tau=2013,tier=tier)
  res[[length(res)+1]]<<-m$rows;los[[length(los)+1]]<<-m$loo
}
for(tier in c("R7","R5","R8")) {
  d<-filter(p,.data$tier==.env$tier)
  for(w in list(NULL,"baseline_positions")) for(y in c("timely","observed_filing","certified_positions","timing","observed_denial","unknown_dates_or_status")) add(d,y,paste0("primary_frame_",y),w,tier)
  if(tier=="R7") {
    for(y in c("timely","observed_filing","certified_positions","timing")) {
      add(d,y,paste0("month_comparison_",y),NULL,tier,fe="cohort + state_month_year")
      add(filter(d,month>=8),y,paste0("aug_dec_",y),NULL,tier)
      largest<-filter(b,.data$tier==.env$tier) %>% group_by(filer) %>% summarise(v=sum(baseline_positions),.groups="drop") %>% arrange(desc(v)) %>% slice_head(n=1) %>% pull(filer)
      add(filter(d,!filer %in% largest),y,paste0("exclude_largest_",y),NULL,tier)
      for(yr in c(2012L,2014L,2015L)) add(filter(d,year %in% c(yr-1L,yr)),y,paste0("calendar_placebo_",yr,"_",y),NULL,tier,ref=yr-1L)
    }
    for(y in c("catchup7","catchup14","catchup30","catchup60")) add(d,y,y,NULL,tier)
  }
}
support<-b %>% group_by(tier,state_fips,season) %>% summarise(cohorts=n(),positive=sum(exposure>0),zero=sum(exposure==0),sd_exposure=sd(exposure),exposure_mass=sum(exposure),.groups="drop")
cd_write(support,out,"admin_support.csv")
cd_write(b %>% group_by(tier,exposed=exposure>0) %>% summarise(cohorts=n(),states=n_distinct(state_fips),mean_positions=mean(baseline_positions,na.rm=TRUE),mean_processing=mean(baseline_processing),mean_lead=mean(baseline_lead),.groups="drop"),out,"admin_baseline_balance.csv")
cd_write(b %>% group_by(tier,filer) %>% summarise(exposure_mass=sum(exposure),baseline_positions=sum(baseline_positions),.groups="drop") %>% group_by(tier) %>% mutate(share=exposure_mass/sum(exposure_mass)) %>% arrange(tier,desc(share)),out,"admin_exposure_concentration.csv")
cd_write(p %>% group_by(tier,year,exposed=exposure>0) %>% summarise(cohorts=n(),records=sum(!no_record),timely=mean(timely,na.rm=TRUE),observed_filing=mean(observed_filing),positions=mean(certified_positions,na.rm=TRUE),timing=mean(timing,na.rm=TRUE),.groups="drop"),out,"admin_year_means.csv")
cd_write(bind_rows(res),out,"admin_estimates.csv");cd_write(bind_rows(los),out,"admin_state_deletions.csv")
print(bind_rows(res) %>% filter(tier=="R7",specification=="primary_frame_timely"))
