source("explorations/chapter_directions/helpers.R")
out<-id_output_dir();cd_verify();filer_dir<-commandArgs(trailingOnly=TRUE)[2]
ids<-read_parquet(file.path(filer_dir,"filer_ids.parquet")) %>% select(fiscal_year,case_number,filer_R5,filer_R7,filer_R8)
cd_keys(ids,c("fiscal_year","case_number"))
states<-read_parquet(path_int("aewr_occupation_skill_rates.parquet"),col_select=c("state_name","state_fips")) %>% distinct() %>%
  mutate(worksite_state=state.abb[match(toupper(state_name),toupper(state.name))]) %>% filter(!is.na(worksite_state)) %>% select(worksite_state,state_fips)
cd_keys(states,"worksite_state")
raw<-open_dataset(path_int("h2a_disclosure_records.parquet")) %>% filter(fiscal_year>=2010L,fiscal_year<=2016L) %>%
  select(fiscal_year,case_number,primary_sub,requested_begin_date,job_order_begin_date,application_begin_date,application_date_basis,
    case_received_date,decision_date,case_status,nbr_workers_certified,worksite_state) %>% collect()
audit<-raw %>% group_by(fiscal_year,primary_sub) %>% summarise(rows=n(),cases=n_distinct(case_number),.groups="drop")
cd_write(audit,out,"admin_master_sub_audit.csv")
# No allocation rows: these are one master application each, not master + sub-employer sums.
r<-raw %>% filter(is.na(primary_sub)|toupper(trimws(primary_sub))!="SUB") %>% distinct() %>%
  group_by(fiscal_year,case_number) %>% summarise(source_rows=n(),
    start=as.Date(cd_unique(application_begin_date)),date_basis=cd_unique(application_date_basis),
    receipt=as.Date(cd_unique(case_received_date)),decision=as.Date(cd_unique(decision_date)),status=toupper(cd_unique(case_status)),
    positions=as.numeric(cd_unique(nbr_workers_certified)),worksite_state=toupper(cd_unique(worksite_state)),.groups="drop") %>%
  left_join(ids,by=c("fiscal_year","case_number"),relationship="one-to-one") %>% left_join(states,by="worksite_state") %>%
  mutate(year=as.integer(format(start,"%Y")),month=as.integer(format(start,"%m")),
    certified=grepl("CERTIFI",status)&!grepl("DENIED|WITHDRAW",status),denied=grepl("DENIED",status),
    known_status=!is.na(status)&grepl("CERTIFI|DENIED|WITHDRAW",status),
    positions=ifelse(known_status & !certified,0,positions),
    plausible=!is.na(receipt)&!is.na(decision)&!is.na(start)&decision>=receipt&abs(as.numeric(start-receipt))<=366&abs(as.numeric(decision-start))<=366,
    requested_basis=date_basis=="dates_of_need")
# The canonical application date must be a requested/job-order start, not certification start.
cd_write(r %>% group_by(fiscal_year,date_basis) %>% summarise(cases=n(),unknown_status=sum(!known_status),invalid_dates=sum(!plausible),duplicate_source_cases=sum(source_rows>1),.groups="drop"),out,"admin_date_audit.csv")
cd_write(r %>% filter(receipt>=as.Date("2013-09-01"),receipt<=as.Date("2013-11-30")) %>% count(receipt,name="case_count") %>% mutate(recoding_date=cd_recoded(receipt)),out,"admin_receipt_dates.csv")
cd_write(r %>% filter(!is.na(decision)) %>% mutate(decision_month=format(decision,"%Y-%m")) %>% group_by(decision_month) %>% summarise(cases=n(),certifications=sum(certified,na.rm=TRUE),.groups="drop"),out,"admin_decision_months.csv")
# Exact repeat case numbers across fiscal files are never counted twice. Conflict is audited and excluded.
cross<-r %>% count(case_number) %>% filter(n>1)
cd_write(cross,out,"admin_cross_fiscal_duplicates.csv")
r<-r %>% anti_join(cross,by="case_number") %>% filter(year %in% 2011:2015,requested_basis,!is.na(state_fips))
panels<-list();bases<-list();exclusions<-list()
for(tier in c("R7","R5","R8")) {
  rr<-r %>% mutate(filer=as.character(.data[[paste0("filer_",tier)]])) %>% filter(!is.na(filer),filer!="") %>%
    mutate(cohort=paste(filer,state_fips,month,sep=":"),season=(month-1L)%/%3L+1L)
  if(tier=="R7") {
    frozen_cases<-rr %>% filter(year %in% 2011:2012) %>%
      select(cohort,filer,state_fips,year,month,case_number,receipt,decision,start,positions) %>%
      mutate(projected_receipt=cd_shift(receipt,2013-year),projected_decision=cd_shift(decision,2013-year),
        projected_start=cd_shift(start,2013-year),case_exposure=cd_overlap(projected_receipt,projected_decision))
    stopifnot(max(frozen_cases$year)<=2012)
    write_parquet(frozen_cases,file.path(out,"admin_exposure_inputs_pre2013.parquet"))
  }
  base<-rr %>% filter(year %in% 2011:2012) %>% group_by(cohort,filer,state_fips,month,season) %>%
    summarise(years=n_distinct(year),cases=n(),complete=all(plausible),
      exposure=mean(cd_overlap(cd_shift(receipt,2013-year),cd_shift(decision,2013-year))),
      predicted_start=as.Date(median(as.numeric(cd_shift(start,2013-year)))),
      baseline_positions=if(all(is.finite(positions)))sum(positions)/2 else NA_real_,
      baseline_processing=mean(as.numeric(decision-receipt)),baseline_lead=mean(as.numeric(start-receipt)),.groups="drop")
  exclusions[[tier]]<-tibble(tier=tier,baseline_cohorts=nrow(base),recurring=sum(base$years==2),complete_recurring=sum(base$years==2 & base$complete),unknown_baseline_positions=sum(!is.finite(base$baseline_positions)))
  base<-base %>% filter(years==2,complete,is.finite(exposure))
  cd_keys(base,"cohort")
  bases[[tier]]<-mutate(base,tier=tier)
  z<-rr %>% inner_join(base %>% select(cohort,predicted_start),by="cohort") %>% mutate(benchmark=cd_shift(predicted_start,year-2013L))
  if(tier=="R7")cd_write(z %>% filter(year==2013L) %>% left_join(base %>% select(cohort,exposure),by="cohort") %>%
    group_by(exposed=exposure>0) %>% summarise(cases=n(),receipt_before_shutdown=sum(receipt<as.Date("2013-10-01"),na.rm=TRUE),
      september_receipts=sum(receipt>=as.Date("2013-09-01") & receipt<as.Date("2013-10-01"),na.rm=TRUE),
      recoded_receipts=sum(cd_recoded(receipt)),certifications=sum(certified),.groups="drop"),out,"admin_event_timing.csv")
  timely<-function(cert,dec,benchmark,known) {
    if(any(cert & !is.na(dec) & dec<=benchmark,na.rm=TRUE)) return(1)
    if(any(!known | (cert & is.na(dec)))) return(NA_real_)
    0
  }
  obs<-z %>% group_by(cohort,year) %>% summarise(observed_filing=1,
    timely=timely(certified,decision,benchmark,known_status),
    catchup7=timely(certified,decision,benchmark+7,known_status),catchup14=timely(certified,decision,benchmark+14,known_status),
    catchup30=timely(certified,decision,benchmark+30,known_status),catchup60=timely(certified,decision,benchmark+60,known_status),
    certified_positions=if(all(is.finite(positions)))sum(positions) else NA_real_,
    timing=if(any(certified & !is.na(decision)))mean(as.numeric(decision-start)[certified],na.rm=TRUE) else NA_real_,
    observed_denial=as.numeric(any(denied,na.rm=TRUE)),unknown_dates_or_status=as.numeric(any(!known_status | is.na(decision))),
    observed_cases=n(),.groups="drop")
  panel<-crossing(base,year=2011:2015) %>% left_join(obs,by=c("cohort","year")) %>%
    mutate(no_record=is.na(observed_filing),observed_filing=replace_na(observed_filing,0),
      across(c(timely,catchup7,catchup14,catchup30,catchup60,certified_positions,observed_denial,unknown_dates_or_status,observed_cases),~ifelse(no_record,0,.x)),
      state_season_year=paste(state_fips,season,year),state_month_year=paste(state_fips,month,year),tier=tier,
      calendar_coverage_known=!year %in% 2014:2015,
      across(c(timely,catchup7,catchup14,catchup30,catchup60,observed_filing,certified_positions,timing,observed_denial),~ifelse(calendar_coverage_known,.x,NA_real_)))
  cd_keys(panel,c("cohort","year"));stopifnot(nrow(panel)==nrow(base)*5)
  panels[[tier]]<-panel
}
write_parquet(bind_rows(panels),file.path(out,"admin_cohort_panel.parquet"));write_parquet(bind_rows(bases),file.path(out,"admin_baseline_cohorts.parquet"))
cd_write(bind_rows(exclusions),out,"admin_exclusions.csv")
cat("Administration cohorts:",nrow(bind_rows(bases)),"\n")
