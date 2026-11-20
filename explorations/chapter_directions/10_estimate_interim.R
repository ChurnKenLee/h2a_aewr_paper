source("explorations/chapter_directions/interim_helpers.R")
out<-id_output_dir();cd_verify();cd_verify(file.path(CD,"earnings_amendment.sha256"));cd_verify(file.path(CD,"interim_amendment.sha256"))
q<-read_parquet(file.path(out,"earnings_interim.parquet"))
expo<-read_csv("explorations/episode_2025/domestic_wage_exposure_frozen.csv",col_types=cols(county_fips="c",state_fips="c",cz_id="c",.default="d"))
ui<-read_csv(file.path(out,"ui_classification.csv"),col_types=cols(.default="c"))
cd_keys(expo,"county_fips");cd_keys(ui,"state_fips")
results<-list();deletions<-list();coverage<-list();queue<-list()
windows<-list(interim_2025Q4_2026Q1=tibble(year=c(2025L,2026L),qtr=c(4L,1L)),quarter_2025Q4=tibble(year=2025L,qtr=4L),quarter_2026Q1=tibble(year=2026L,qtr=1L))
prepare<-function(qq,target,common=NULL) {
  d<-interim_window(qq,target) %>% inner_join(expo,by="county_fips")
  if(!is.null(common))d<-d %>% filter(county_fips %in% common)
  d %>% group_by(state_fips) %>% filter(n()>=2) %>% ungroup() %>% mutate(D=S_winsorized*cut_us,D_h2a=S_winsorized*cut_h2a,D_cz=S_cz_winsorized*cut_us) %>% left_join(ui,by="state_fips")
}
fit<-function(d,series,window,tau=2026L,all_specs=TRUE,splits=TRUE) {
  coverage[[length(coverage)+1]]<<-tibble(series=series,window=window,tau=tau,counties=nrow(d),states=n_distinct(d$state_fips),mean_change=mean(d$change),mean_exposure=mean(d$S_winsorized))
  configs<-list(list("primary","change ~ D + S_winsorized | state_fips","D",NULL),list("employment_weighted","change ~ D + S_winsorized | state_fips","D","emp_l"),list("cash_dose","change ~ D_h2a + S_winsorized | state_fips","D_h2a",NULL),list("commuting_zone","change ~ D_cz + S_cz_winsorized | state_fips","D_cz",NULL),list("no_intensity_control","change ~ D | state_fips","D",NULL))
  if(!all_specs)configs<-configs[1]
  append_fit<-function(dd,cfg,label) {
    z<-cd_fit(dd,cfg[[2]],cfg[[3]],paste(series,label,sep="_"),cfg[[4]],track="earnings_interim",tau=tau,tier="county")
    results[[length(results)+1]]<<-mutate(z$rows,window=window)
    if(nrow(z$loo))deletions[[length(deletions)+1]]<<-mutate(z$loo,window=window)
  }
  for(cfg in configs)append_fit(d,cfg,cfg[[1]])
  if(splits)for(g in c("covered","excluded","ambiguous"))append_fit(filter(d,direct_agriculture==g),configs[[1]],paste0("UI_",g))
}
for(window in names(windows))for(industry in c("111","112","10")) {
  series<-c("111"="crop","112"="animal","10"="all_private")[[industry]]
  fit(prepare(filter(q,industry_code==industry),windows[[window]]),series,window,all_specs=industry=="111",splits=industry=="111")
}
common<-prepare(filter(q,industry_code=="111"),windows[[1]])$county_fips
for(window in names(windows)[-1])fit(prepare(filter(q,industry_code=="111"),windows[[window]],common),"crop",paste0(window,"_common_sample"),all_specs=FALSE,splits=FALSE)
lead<-tibble(year=c(2024L,2025L,2025L),qtr=c(4L,1L,2L))
fit(prepare(filter(q,industry_code=="111"),lead),"crop","prior_window_lead",tau=2025L,splits=FALSE)
for(tau in 2016:2024)fit(prepare(filter(q,industry_code=="111"),tibble(year=c(tau-1L,tau),qtr=c(4L,1L))),"crop","matched_two_quarter_placebo",tau=tau,all_specs=FALSE,splits=FALSE)
qw_source<-open_dataset(path_int("qwi_county_ag_quarterly_earnings.parquet"))
qw_periods<-qw_source %>% filter(industry_code=="111") %>% select(year,qtr) %>% distinct() %>% collect()
need<-bind_rows(windows[[1]],windows[[1]] %>% mutate(year=year-1L))
if(nrow(anti_join(need,qw_periods,by=c("year","qtr")))==0) {
  qw<-qw_source %>% filter(industry_code=="111",year>=2024L,year<=2026L) %>% select(county_fips,year,qtr,qwi_stable_monthly_earnings,qwi_stable_employment,qwi_stable_monthly_earnings_status) %>% collect() %>% transmute(county_fips,year,qtr,avg_wkly_wage=qwi_stable_monthly_earnings,emp=qwi_stable_employment,disclosed=qwi_stable_monthly_earnings_status==1L)
  fit(prepare(qw,windows[[1]]),"QWI_stable_monthly","interim_2025Q4_2026Q1",splits=FALSE)
} else queue[[1]]<-tibble(specification="QWI_secondary",status="missing_declared_interim_quarters",detail=paste(apply(anti_join(need,qw_periods,by=c("year","qtr")),1,paste,collapse="Q"),collapse=","))
r<-bind_rows(results) %>% mutate(percent_at_S03_cut_minus4=100*expm1(-1.2*estimate),percent_low=100*expm1(-1.2*high),percent_high=100*expm1(-1.2*low))
cd_keys(r,c("window","tau","specification","weights","term"))
cd_write(r,out,"earnings_interim_estimates.csv");cd_write(bind_rows(deletions),out,"earnings_interim_state_deletions.csv");cd_write(bind_rows(coverage),out,"earnings_interim_coverage.csv")
cd_write(bind_rows(queue),out,"earnings_interim_queue.csv")
cd_write(q %>% count(industry_code,year,qtr,name="county_records"),out,"earnings_interim_quarter_inventory.csv")
print(r %>% filter(specification=="crop_primary",tau>=2025) %>% select(window,n,states,estimate,low,high,percent_at_S03_cut_minus4,percent_low,percent_high))
