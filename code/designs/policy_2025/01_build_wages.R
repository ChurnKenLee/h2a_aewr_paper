source("code/designs/policy_2025/helpers.R")
out <- id_output_dir(); cd_verify()
filer_dir <- commandArgs(trailingOnly=TRUE)[2]
ids <- read_parquet(file.path(filer_dir,"filer_ids.parquet")) %>% select(fiscal_year,case_number,filer_R5,filer_R7,filer_R8)
ident <- read_parquet(file.path(filer_dir,"filer_cases.parquet")) %>% select(fiscal_year,case_number,fein,labor_contractor)
cd_keys(ids,c("fiscal_year","case_number")); cd_keys(ident,c("fiscal_year","case_number"))
cp <- read_parquet(path_processed("county_year_panel.parquet"),col_select=c("county_fips","state_fips","aewr_region_id","emp_farm_2011","year")) %>%
  filter(year==2011,!state_fips %in% c("02","15","11"),is.finite(emp_farm_2011),emp_farm_2011>0) %>%
  group_by(aewr_region_id) %>% filter(n_distinct(state_fips)>1) %>% ungroup() %>% select(-year)
frame <- ep_state_frame() %>% semi_join(cp,by="state_fips")
cd_write(frame,out,"wage_state_dose.csv")
disc <- open_dataset(path_int("h2a_disclosure_records.parquet")) %>% filter(fiscal_year>=2023L,fiscal_year<=2026L) %>%
  select(fiscal_year,case_number,decision_date,case_received_date,soc_code) %>% collect() %>% group_by(fiscal_year,case_number) %>%
  summarise(decision=as.Date(cd_first(decision_date)),receipt=as.Date(cd_unique(case_received_date)),case_soc=cd_unique(soc_code),.groups="drop")
nh <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>% filter(fiscal_year>=2012L,fiscal_year<=2025L) %>%
  select(period_begin_date,number_of_hours) %>% collect() %>% filter(!is.na(as.Date(period_begin_date)))
nh_mean <- mean(as.numeric(nh$number_of_hours),na.rm=TRUE); rm(nh)
r <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>% filter(fiscal_year>=2023L,fiscal_year<=2026L) %>%
  select(fiscal_year,case_number,allocation_id,county_fips_list,adjusted_nbr_workers_certified,period_begin_date,period_end_date,number_of_hours,wage_rate,wage_unit,soc_code) %>% collect()
cd_keys(r,"allocation_id")
r <- r %>% mutate(begin=as.Date(period_begin_date),end=as.Date(period_end_date),nh=as.numeric(number_of_hours),nh=ifelse(is.na(nh),nh_mean,nh),
  h=as.numeric(end-begin)/7*nh*as.numeric(adjusted_nbr_workers_certified),pos=as.numeric(adjusted_nbr_workers_certified),wage=as.numeric(wage_rate),
  hourly=toupper(trimws(wage_unit)) %in% c("HOUR","HOURLY","HR") & is.finite(wage) & wage>0 & wage<=100,
  soc=substr(trimws(soc_code),1,7),soc=ifelse(is.na(soc)|!grepl("^[0-9]{2}-[0-9]{4}$",soc),"unknown",soc)) %>%
  left_join(disc,by=c("fiscal_year","case_number"),relationship="many-to-one") %>%
  filter(!is.na(begin),decision>=as.Date(sprintf("%d-10-01",fiscal_year-1L)),decision<=as.Date(sprintf("%d-06-30",fiscal_year))) %>%
  left_join(ids,by=c("fiscal_year","case_number"),relationship="many-to-one") %>%
  left_join(ident,by=c("fiscal_year","case_number"),relationship="many-to-one") %>%
  mutate(county_fips_list=ifelse(is.na(county_fips_list)|county_fips_list=="","00000",county_fips_list),nc=stringr::str_count(county_fips_list,",")+1L)
before <- r %>% select(allocation_id,h,pos)
r <- r %>% separate_rows(county_fips_list,sep=",") %>% mutate(county_fips=harmonize_county_fips_2010(trimws(county_fips_list)),h=h/nc,pos=pos/nc)
check <- r %>% group_by(allocation_id) %>% summarise(h_new=sum(h),pos_new=sum(pos),.groups="drop") %>% left_join(before,by="allocation_id")
stopifnot(max(abs(check$pos_new-check$pos),na.rm=TRUE)<1e-8,max(abs(check$h_new-check$h),na.rm=TRUE)<1e-6)
cd_write(tibble(check_name="allocation_conservation",max_position_error=max(abs(check$pos_new-check$pos),na.rm=TRUE),max_hour_error=max(abs(check$h_new-check$h),na.rm=TRUE)),out,"wage_allocation_check.csv")
r <- r %>% mutate(state_fips=substr(county_fips,1,2)) %>% group_by(fiscal_year,case_number) %>% mutate(multistate=n_distinct(state_fips[state_fips!="00"])>1) %>% ungroup() %>%
  select(-state_fips) %>% inner_join(cp,by="county_fips",relationship="many-to-one") %>% left_join(frame %>% select(state_fips,rate25,us_e1,h2a_e1),by="state_fips")
regional <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>% filter(occupation_scope %in% c("nonrange_reference","fls_common_soc_reference"),worker_category=="covered_workers")
rates <- bind_rows(lapply(2023:2025,function(y) regional %>% filter(effective_start<=as.Date(sprintf("%d-04-01",y)),is.na(effective_end_exclusive)|effective_end_exclusive>as.Date(sprintf("%d-04-01",y))) %>% transmute(state_fips,fiscal_year=y,reference=aewr)))
rates <- bind_rows(rates,frame %>% transmute(state_fips,fiscal_year=2026L,reference=us_e1))
cd_keys(rates,c("state_fips","fiscal_year"))
r <- r %>% left_join(rates,by=c("state_fips","fiscal_year")) %>%
  mutate(common_soc=soc %in% c("45-2041","45-2091","45-2092","45-2093","45-2099"),
    receipt_proxy_new= !is.na(receipt) & receipt>=as.Date("2025-10-02"),
    effective_floor=NA_real_,floor_status="unknown_job_order_date_skill_worker_category_prevailing_wage",
    premium=wage-reference,near_reference=as.numeric(abs(wage-reference)<=.05),
    contractor=case_when(toupper(as.character(labor_contractor)) %in% c("Y","YES","TRUE","1")~1, toupper(as.character(labor_contractor)) %in% c("N","NO","FALSE","0")~0,TRUE~NA_real_))
# Occupation is the case's disclosed job-order SOC. Addendum B allocation rows
# often do not repeat it; joining case metadata is part of this panel's grain.
r<-r %>% mutate(row_soc=soc,case_soc=substr(trimws(case_soc),1,7),
  soc=ifelse(!is.na(case_soc)&grepl("^[0-9]{2}-[0-9]{4}$",case_soc),case_soc,"unknown"),
  common_soc=soc %in% c("45-2041","45-2091","45-2092","45-2093","45-2099"))
candidate<-read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
  filter(soc_code=="FIELD_LIVESTOCK_COMBINED",effective_start==as.Date("2025-10-02")) %>%
  transmute(state_fips,key=paste0(tolower(worker_category),"_",skill_level),rate=as.numeric(hourly_aewr)) %>% distinct() %>% pivot_wider(names_from=key,values_from=rate)
mw<-read_parquet(path_int("state_real_minwages.parquet")) %>% filter(year %in% 2023:2026) %>%
  transmute(state_fips,fiscal_year=as.integer(year),ag_min=prevailing_ag_min_wage,general_min=prevailing_min_wage,ag_floor_rule)
cd_keys(candidate,"state_fips");cd_keys(mw,c("state_fips","fiscal_year"))
r<-r %>% left_join(candidate,by="state_fips") %>% left_join(mw,by=c("state_fips","fiscal_year")) %>%
  mutate(ref_us1=ifelse(fiscal_year==2026 & receipt_proxy_new,us_1,reference),ref_us2=ifelse(fiscal_year==2026 & receipt_proxy_new,us_2,reference),
    ref_h2a1=ifelse(fiscal_year==2026 & receipt_proxy_new,h2a_1,reference),ref_h2a2=ifelse(fiscal_year==2026 & receipt_proxy_new,h2a_2,reference),
    across(starts_with("ref_"),~pmax(.x,ag_min,7.25)),
    reference_distance=pmin(abs(wage-ref_us1),abs(wage-ref_us2),abs(wage-ref_h2a1),abs(wage-ref_h2a2)),
    candidate_near=ifelse(common_soc & !multistate & !is.na(receipt),as.numeric(reference_distance<=.05),NA_real_),
    premium_lower=ifelse(common_soc & !multistate & !is.na(receipt),wage-pmin(ref_us1,ref_us2,ref_h2a1,ref_h2a2),NA_real_))
stopifnot(all(is.na(r$effective_floor)),all(is.na(r$candidate_near[r$multistate])),
  all(is.na(r$candidate_near[is.na(r$receipt)])))
cd_write(r %>% group_by(fiscal_year) %>% summarise(hours=sum(h[hourly & is.finite(h)&h>0]),
  case_SOC_known_share=cd_mean(as.numeric(soc!="unknown"),ifelse(hourly,h,0)),row_SOC_missing_share=cd_mean(as.numeric(row_soc=="unknown"),ifelse(hourly,h,0)),
  candidate_coverage=cd_mean(as.numeric(is.finite(candidate_near)),ifelse(hourly,h,0)),near_any_candidate=cd_mean(candidate_near,ifelse(hourly,h,0)),
  baseline_minimum_exceeds_AEWR=cd_mean(as.numeric(ag_min>reference),ifelse(hourly,h,0)),.groups="drop"),out,"wage_schedule_envelope_audit.csv")
cd_write(r %>% group_by(fiscal_year) %>% summarise(rows=n(),cases=n_distinct(case_number),hourly_hours=sum(h[hourly & is.finite(h) & h>0]),
  missing_R7_rows=sum(is.na(filer_R7)),multistate_hours=cd_mean(as.numeric(multistate),ifelse(hourly,h,0)),
  common_SOC_hours=cd_mean(as.numeric(common_soc),ifelse(hourly,h,0)),receipt_proxy_new_hours=cd_mean(as.numeric(receipt_proxy_new),ifelse(hourly,h,0)),
  missing_receipt_hours=cd_mean(as.numeric(is.na(receipt)),ifelse(hourly,h,0)),resolved_effective_floors=sum(is.finite(effective_floor)),.groups="drop"),out,"wage_floor_audit.csv")
cd_write(r %>% group_by(fiscal_year) %>% summarise(rows=n(),nonhourly_or_invalid_wage=sum(!hourly),invalid_nonpositive_hours=sum(!is.finite(h)|h<=0),unknown_occupation=sum(soc=="unknown"),.groups="drop"),out,"wage_exclusions.csv")
panels<-list(); certified<-list()
for(tier in c("R7","R5","R8","FEIN")) {
  rr<-r %>% mutate(filer=if(tier=="FEIN") fein else as.character(.data[[paste0("filer_",tier)]])) %>% filter(!is.na(filer),filer!="")
  certified[[tier]]<-rr %>% group_by(filer,state_fips,fiscal_year) %>% summarise(positions=sum(pos,na.rm=TRUE),.groups="drop") %>% mutate(tier=tier)
  panels[[tier]]<-rr %>% filter(hourly,is.finite(h),h>0) %>%
    mutate(row_cdf0=as.numeric(premium<=0),row_cdf25=as.numeric(premium<=.25),row_cdf100=as.numeric(premium<=1)) %>%
    group_by(filer,state_fips,aewr_region_id,soc,fiscal_year) %>%
    summarise(wage=cd_mean(wage,h),premium=cd_mean(premium,h),near=cd_mean(near_reference,h),
      cdf0=cd_mean(row_cdf0,h),cdf25=cd_mean(row_cdf25,h),cdf100=cd_mean(row_cdf100,h),
      near_candidate=cd_mean(candidate_near,h),premium_lower=cd_mean(premium_lower,h),
      contractor=cd_mean(contractor,h),common_soc=all(common_soc),single_state=all(!multistate),receipt_new=all(receipt_proxy_new),
      h=sum(h),.groups="drop") %>% mutate(tier=tier)
}
p<-bind_rows(panels); cd_keys(p,c("tier","filer","state_fips","soc","fiscal_year"))
write_parquet(p,file.path(out,"wage_filer_state_occupation.parquet"));write_parquet(bind_rows(certified),file.path(out,"wage_certified_participation.parquet"))
cat("Wage panel written:",nrow(p),"cells\n")
