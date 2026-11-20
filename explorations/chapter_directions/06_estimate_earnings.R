source("explorations/chapter_directions/helpers.R")
out<-id_output_dir();cd_verify()
cd_verify("explorations/episode_2025/domestic_wage_registry.sha256","explorations/episode_2025")
args<-commandArgs(trailingOnly=TRUE);mode<-if(length(args)>=2)args[2] else "pre"
stopifnot(mode %in% c("pre","lead","post"))
if(mode=="post") {
  cd_verify(file.path(CD,"earnings_amendment.sha256"))
  if(Sys.time()<as.POSIXct("2026-12-02 15:00:00",tz="UTC"))stop("Awaiting 2026 Q2; no partial window")
  meta<-jsonlite::read_json(file.path(out,"release_metadata_used.json"),simplifyVector=TRUE)
  if(meta$status!="published" || !"2026Q2" %in% meta$published_quarters)stop("Unverified full-window release")
  if(file.exists(file.path(out,"earnings_post_completed.json")))stop("Full-window run already completed; explicit dated amendment required for another look")
}
q<-read_parquet(file.path(out,paste0("earnings_",mode,".parquet")))
if(mode=="pre")stopifnot(max(q$year)<=2024)
expo<-read_csv("explorations/episode_2025/domestic_wage_exposure_frozen.csv",col_types=cols(county_fips="c",state_fips="c",cz_id="c",.default="d"))
cd_keys(expo,"county_fips")
review<-jsonlite::read_json(file.path(CD,"sources/ui_classification_review.json"))
ui<-bind_rows(lapply(sort(unique(expo$state_fips)),function(s) {
  entry<-review$states[[s]]
  if(is.null(entry))entry<-list(direct_agriculture="ambiguous",contractor_agriculture="ambiguous",basis=review$default,source_url="https://oui.doleta.gov/unemploy/pdf/uilawcompar/2023/coverage.pdf",secondary_source_url="")
  bind_cols(tibble(state_fips=s),as_tibble(entry))
}))
cd_write(ui,out,"ui_classification.csv")
cd_keys(ui,"state_fips")
res<-list();los<-list();cov<-list();queue<-list()
fit_series<-function(qq,tau,series) {
  d<-cd_window(qq,tau) %>% inner_join(expo,by="county_fips") %>% group_by(state_fips) %>% filter(n()>=2) %>% ungroup() %>%
    mutate(D=S_winsorized*cut_us,D_h2a=S_winsorized*cut_h2a,D_cz=S_cz_winsorized*cut_us) %>% left_join(ui,by="state_fips")
  cov[[length(cov)+1]]<<-tibble(series=series,tau=tau,counties=nrow(d),states=n_distinct(d$state_fips),mean_exposure=mean(d$S_winsorized),prior_employment=sum(d$emp_l),
    excluded_UI_states=n_distinct(d$state_fips[d$direct_agriculture=="excluded"]),ambiguous_UI_states=n_distinct(d$state_fips[d$direct_agriculture=="ambiguous"]))
  configurations<-list(
    list("primary","change ~ D + S_winsorized | state_fips","D",NULL),
    list("employment_weighted","change ~ D + S_winsorized | state_fips","D","emp_l"),
    list("cash_dose","change ~ D_h2a + S_winsorized | state_fips","D_h2a",NULL),
    list("commuting_zone","change ~ D_cz + S_cz_winsorized | state_fips","D_cz",NULL),
    list("no_intensity_control","change ~ D | state_fips","D",NULL))
  if(series %in% c("all_private","animal")) configurations<-configurations[1]
  for(cfg in configurations) {
    m<-cd_fit(d,cfg[[2]],cfg[[3]],paste(series,cfg[[1]],sep="_"),cfg[[4]],track="earnings",tau=tau,tier="county")
    res[[length(res)+1]]<<-m$rows;los[[length(los)+1]]<<-m$loo
  }
  if(series=="crop") for(g in c("covered","excluded","ambiguous")) {
    m<-cd_fit(filter(d,direct_agriculture==g),"change ~ D + S_winsorized | state_fips","D",paste0("crop_UI_",g),track="earnings",tau=tau,tier="county")
    res[[length(res)+1]]<<-m$rows;los[[length(los)+1]]<<-m$loo
  }
}
years<-if(mode=="pre")2016:2024 else if(mode=="post")2026L else integer()
for(tau in years)for(industry in c("111","112","10"))fit_series(filter(q,industry_code==industry),tau,c("111"="crop","112"="animal","10"="all_private")[[industry]])
# The registered prior-window lead is only read alongside the full post window.
if(mode=="post")fit_series(filter(q,industry_code=="111"),2025L,"prior_window_lead")
if(mode=="lead") {
  stopifnot(all(q$year<=2024 | (q$year==2025 & q$qtr %in% 1:2)))
  fit_series(filter(q,industry_code=="111"),2025L,"prior_window_lead")
}
if(mode=="pre") {
  old<-read_parquet(file.path(out,"benchmark/qcew_crop_county_quarter_2014_2024.parquet"))
  audit<-full_join(old %>% select(county_fips,year,qtr,old_wage=avg_wkly_wage,old_emp=emp,old_disclosed=disclosed),
    q %>% filter(industry_code=="111") %>% select(county_fips,year,qtr,new_wage=avg_wkly_wage,new_emp=emp,new_disclosed=disclosed,components),by=c("county_fips","year","qtr")) %>%
    mutate(wage_difference=new_wage-old_wage,employment_difference=new_emp-old_emp)
  cd_write(audit %>% filter(is.na(old_wage)|is.na(new_wage)|abs(wage_difference)>1e-8|abs(employment_difference)>1e-8),out,"earnings_extractor_differences.csv")
  cd_write(audit %>% summarise(rows=n(),joint_disclosed=sum(old_disclosed & new_disclosed,na.rm=TRUE),
    changed_disclosed_wages=sum(old_disclosed & new_disclosed & abs(wage_difference)>1e-8,na.rm=TRUE),
    max_joint_wage_difference=max(abs(wage_difference[old_disclosed & new_disclosed]),na.rm=TRUE),excluded_old_rows=sum(!is.na(old_wage)&is.na(new_wage))),out,"earnings_extractor_audit.csv")
  # Filter inside Arrow: the primary earnings firewall also applies to QWI.
  qw<-open_dataset(path_int("qwi_county_ag_quarterly_earnings.parquet")) %>% filter(year<=2024L,year>=2014L,industry_code=="111") %>%
    select(county_fips,year,qtr,qwi_stable_monthly_earnings,qwi_stable_employment,qwi_stable_monthly_earnings_status) %>% collect() %>%
    transmute(county_fips,year,qtr,avg_wkly_wage=qwi_stable_monthly_earnings,emp=qwi_stable_employment,disclosed=qwi_stable_monthly_earnings_status==1L)
  for(tau in years)fit_series(qw,tau,"QWI_stable_monthly")
} else if(mode=="post") {
  # A separately acquired QWI extract requires a complete six-quarter window.
  qwi_file<-Sys.getenv("CD_QWI_COMPLETE_EXTRACT","")
  if(nzchar(qwi_file)) {
    qwi_meta<-jsonlite::read_json(Sys.getenv("CD_QWI_RELEASE_METADATA"),simplifyVector=TRUE)
    stopifnot(qwi_meta$status=="published", "2026Q2" %in% qwi_meta$published_quarters,
      grepl("^https://([a-z0-9.-]+\\.)?census\\.gov/",qwi_meta$source_url),nzchar(qwi_meta$checked_at),
      digest::digest(file=qwi_file,algo="sha256")==qwi_meta$extract_sha256)
    qw<-read_parquet(qwi_file)
    stopifnot(all(c("county_fips","year","qtr","qwi_stable_monthly_earnings","qwi_stable_employment","qwi_stable_monthly_earnings_status") %in% names(qw)))
    stopifnot(is.character(qw$county_fips),identical(qw$county_fips,county_fips(qw$county_fips)))
    qw<-qw %>% transmute(county_fips,year,qtr,avg_wkly_wage=qwi_stable_monthly_earnings,emp=qwi_stable_employment,disclosed=qwi_stable_monthly_earnings_status==1L)
    fit_series(qw,2026L,"QWI_stable_monthly")
  } else queue[[1]]<-tibble(specification="QWI_secondary",status="awaiting_complete_QWI_window",reason="No full-window QWI extract supplied; do not run a partial series")
}
r<-bind_rows(res) %>% mutate(percent_at_S03_cut_minus4=100*expm1(-1.2*estimate),
  percent_low=100*expm1(-1.2*high),percent_high=100*expm1(-1.2*low),run=mode)
cd_write(r,out,paste0("earnings_",mode,"_estimates.csv"));cd_write(bind_rows(los),out,paste0("earnings_",mode,"_state_deletions.csv"));cd_write(bind_rows(cov),out,paste0("earnings_",mode,"_coverage.csv"))
cd_write(q %>% group_by(industry_code,year,qtr) %>% summarise(counties=n(),disclosed=sum(disclosed),employment=sum(emp[disclosed]),max_rounding_gap=max(abs(avg_wkly_wage-weekly_wage_from_totals),na.rm=TRUE),.groups="drop"),out,paste0("earnings_",mode,"_quarter_audit.csv"))
if(mode=="pre") {
  queue[[1]]<-tibble(specification=c("primary","employment_weighted","cash_dose","commuting_zone","no_intensity_control","all_private_diagnostic","animal_diagnostic","prior_window_lead","UI_splits","QWI_secondary"),status="awaiting_full_window",reason="No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02")
  queue[[1]]$status[queue[[1]]$specification=="prior_window_lead"]<-if(file.exists(file.path(out,"earnings_lead_estimates.csv")))"estimated_pre_policy" else "unavailable_separate_pre_policy_slices"
  queue[[1]]$reason[queue[[1]]$specification=="prior_window_lead"]<-"Needs public 2025Q1 and Q2 crop slices; local DNS unavailable, web CSV unavailable, Cloud tool disallowed by session approval policy, browser connection timed out. Annual archives containing post-policy earnings remain closed."
} else if(mode=="post")jsonlite::write_json(list(completed=as.character(Sys.time()),registry_sha256=digest::digest(file=file.path(CD,"registry.json"),algo="sha256")),file.path(out,"earnings_post_completed.json"),auto_unbox=TRUE,pretty=TRUE)
cd_write(bind_rows(queue),out,paste0("earnings_",mode,"_queue.csv"))
cat("Earnings",mode,"specifications:",nrow(r),"\n")
