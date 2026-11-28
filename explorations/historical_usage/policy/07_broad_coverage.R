# Nationwide full-year support for posted hourly wages and filer renewals.
source("code/paths.R")
source(path_code("c00_shared","geography.R"))
library(data.table)
setDTthreads(2L)
out <- commandArgs(trailingOnly=TRUE)[1L]
stopifnot(!is.na(out))
dir.create(out, showWarnings=FALSE, recursive=TRUE)
stopifnot(startsWith(normalizePath(out), "/tmp/"))
emit <- function(x, name) fwrite(x,file.path(out,paste0("broad_",name,".csv")))
read_fields <- function(file, cols) as.data.table(arrow::read_parquet(path_int(file), col_select=tidyselect::all_of(cols)))
keys <- c("fiscal_year","case_number")
raw <- read_fields("h2a_disclosure_records.parquet", c(keys,"primary_sub","wage_rate","wage_unit","soc_code",
                "job_title","case_received_date","decision_date","worksite_state"))
raw <- raw[fiscal_year %between% c(2008L,2025L) & (is.na(primary_sub) | toupper(trimws(primary_sub))!="SUB")]
# Equal nonmissing case fields required across retained master/standalone records.
raw[, `:=`(w=as.numeric(wage_rate),u=toupper(trimws(wage_unit)),soc=substr(trimws(soc_code),1,7),
           title=toupper(trimws(gsub("[[:space:]]+"," ",job_title))),
           receipt=as.IDate(substr(case_received_date,1,10)),decision=as.IDate(substr(decision_date,1,10)))]
raw[, offer := fifelse(u %chin% c("HOUR","HR","HOURLY") & is.finite(w) & w>0 & w<=100,w,NA_real_)]
take_one <- function(v) if (uniqueN(v)==1L) v[1L] else v[NA_integer_]
case <- raw[, lapply(.SD,take_one), by=keys, .SDcols=c("offer","soc","title","receipt","decision","worksite_state")]
audit <- read_fields("h2a_case_allocation_audit.parquet", c(keys,"case_allocation_reconciled","case_certified_target"))
audit <- audit[fiscal_year %between% c(2008L,2025L) & case_allocation_reconciled == TRUE & case_certified_target>0]
setnames(audit,"case_certified_target","positions")
stopifnot(!anyDuplicated(audit,by=keys))
z <- merge(audit,case,by=keys,all.x=TRUE)
sites <- read_fields("h2a_allocated_worksites.parquet", c(keys,"worksite_state","county_fips_list","adjusted_nbr_workers_certified"))
sites <- merge(sites,audit[,..keys],by=keys)
lookup <- c(setNames(state.abb,toupper(state.name)),setNames(state.abb,state.abb))
sites[, st := unname(lookup[toupper(trimws(worksite_state))])]
# Any unassigned or multistate worksite makes the case ineligible for the
# single-state offer/reference comparison, even when some worksites are valid.
state <- sites[, .(st=if (!anyNA(st) && uniqueN(st)==1L) st[1L] else NA_character_),by=keys]
z <- merge(z,state,by=keys,all.x=TRUE)
z[, disclosure_st := unname(lookup[toupper(trimws(worksite_state))])]
z[, one_state := !is.na(st) & !is.na(disclosure_st) & st==disclosure_st]
z[, clocks := !is.na(receipt) & !is.na(decision) & decision>=receipt & decision-receipt<=180 &
               year(decision)+as.integer(month(decision)>=10L)==fiscal_year]
rates <- read_fields("aewr_rate_schedule.parquet", c("state_name","aewr","reference_year","effective_start",
                       "effective_end_exclusive","alternative_effective_start"))
rates[, st := unname(lookup[toupper(state_name)])]
setorder(rates,st,effective_start)
stopifnot(!anyDuplicated(rates,by=c("st","effective_start")))
# Join both dates independently to source half-open intervals; the OES gap has
# no FLS reference. Do not classify disputed December 2024 dates as certain.
lookup_rate <- function(date_col) {
  t <- z[one_state & clocks, c(keys,"st",date_col), with=FALSE]
  setnames(t,date_col,"day")
  m <- rates[t,on=.(st,effective_start<=day),mult="last",
    .(case_number=i.case_number,fiscal_year=i.fiscal_year, day=i.day,start=x.effective_start,
      end=effective_end_exclusive,alt=alternative_effective_start,ref=aewr,refyear=reference_year)]
  m <- m[!is.na(end) & day<end & !(refyear==2025 & !is.na(alt) & day>=start & day<alt)]
  m[,.(fiscal_year,case_number,ref,refyear)]
}
r <- lookup_rate("receipt")
d <- lookup_rate("decision")
z <- merge(z,r,by=keys,all.x=TRUE)
setnames(d,c("ref","refyear"),c("decision_ref","decision_refyear"))
z <- merge(z,d,by=keys,all.x=TRUE)
six <- c("45-2041","45-2091","45-2092","45-2093","45-2099","53-7064")
z[, six_soc := !is.na(soc) & soc %chin% six]
z[, title_ok := !is.na(title) & nzchar(title) & !grepl("HERD(ER|ING)|SHEEP|GOATHERD|RANGE (LIVESTOCK|CATTLE)", title)]
# Conservative scope before 2023: plausibly nonrange, hourly with a nonempty
# title. Use the six common SOCs for all reference-2023-onwards cases, including
# before March 2023. These are wage-schedule diagnostics, not legal assignments.
z[, eligible := is.finite(offer) & !is.na(ref) & title_ok & (refyear<2023L | six_soc)]
z[, same_clock := eligible & is.finite(decision_ref) & decision_ref==ref]
z[, hit := is.finite(offer) & is.finite(ref) & abs(offer-ref)<0.005]
z[, near := (as.integer(abs(offer-ref-0.07)<0.005) + as.integer(abs(offer-ref+0.07)<0.005))/2]
ids <- read_fields("policy_2025/filer/filer_ids.parquet",c(keys,"filer_R7"))
stopifnot(!anyDuplicated(ids,by=keys))
z <- merge(z,ids,by=keys,all.x=TRUE)
stopifnot(nrow(z)==nrow(audit), !anyDuplicated(z,by=keys), !anyNA(z$eligible), all(z$positions>0))
# Connected means condition on an earlier filing in the immediately prior
# reference year. A repeated title or SOC does not establish an identical job.
paired <- function(type) {
  w <- z[eligible & !is.na(filer_R7)]
  w[, job := if(type=="title") title else soc]
  w <- w[!is.na(job) & nzchar(job)]
  groups <- c("filer_R7","st","job","refyear")
  cell <- w[,.(mu=weighted.mean(offer,positions),rate=unique(ref),positions=sum(positions),n=.N),by=groups]
  stopifnot(!anyDuplicated(cell,by=groups))
  prev <- cell[,.(filer_R7,st,job,refyear=refyear+1L,prev_mu=mu,prev_rate=rate)]
  cell <- merge(cell,prev,by=groups)
  cell[, `:=`(dw=mu-prev_mu,dr=rate-prev_rate)]
  out <- merge(w[,c(keys,groups),with=FALSE],cell[,..groups],by=groups)
  cols <- list(matched=out[,..keys], cells=cell)
  cols
}
title_pairs <- paired("title")
soc_pairs <- paired("soc")
z[, `:=`(renewal_title=FALSE,renewal_soc=FALSE)]
z[title_pairs$matched,on=keys,renewal_title:=TRUE]
z[soc_pairs$matched,on=keys,renewal_soc:=TRUE]
steps <- list(national=rep(TRUE,nrow(z)),observed_hourly=is.finite(z$offer),single_state_hourly=z$one_state & is.finite(z$offer),
              schedule_sample=z$eligible, both_clocks=z$same_clock, six_common=z$eligible & z$six_soc,
              renewal_title=z$renewal_title, renewal_soc=z$renewal_soc)
# For each table the national denominator is ALL positive, reconciled FY cases.
total <- z[,.(denom_n=.N,denom_pos=sum(positions)),by=fiscal_year]
summarize <- function(v) {
  v[,.(n=.N,pos=sum(positions),states=uniqueN(st[one_state]),
        mean_offer=if(all(is.finite(offer))) mean(offer) else NA_real_,
        position_mean_offer=if(all(is.finite(offer))) weighted.mean(offer,positions) else NA_real_,
        median_offer=if(all(is.finite(offer))) median(offer) else NA_real_,
        position_mean_ref=if(all(eligible)) weighted.mean(ref,positions) else NA_real_,
        exact_share=if(all(eligible)) mean(hit) else NA_real_,
        exact_position_share=if(all(eligible)) weighted.mean(hit,positions) else NA_real_,
        within5_position_share=if(all(eligible)) weighted.mean(abs(offer-ref)<=0.050001,positions) else NA_real_,
        nearby_position_share=if(all(eligible)) weighted.mean(near,positions) else NA_real_),by=fiscal_year]
}
annual <- rbindlist(lapply(names(steps),function(label) {
  a <- merge(total,summarize(z[steps[[label]]]),by="fiscal_year",all.x=TRUE)
  a[is.na(n),`:=`(n=0L,pos=0,states=0L)]
  a[, `:=`(stage=label,case_coverage=n/denom_n,position_coverage=pos/denom_pos)]
  a
}))
emit(annual,"annual")
all <- annual[,.(cases=sum(n), positions=sum(pos), national_cases=sum(denom_n), national_positions=sum(denom_pos),
                 positive_years=sum(n>0)),by=stage]
all[, `:=`(case_coverage=cases/national_cases, position_coverage=positions/national_positions)]
emit(all,"full_period")
by_state <- z[one_state==TRUE,.(state_assigned_denominator=.N, state_assigned_positions=sum(positions),
                                 cases=sum(eligible), positions=sum(positions[eligible])),
               by=.(fiscal_year,st)]
emit(by_state,"state_year")
# Counties are counted only on individually unambiguous, positive-allocated
# worksite rows. A case can contribute to several counties via different sites.
county <- sites[adjusted_nbr_workers_certified>0 & !is.na(county_fips_list) &
                  grepl("^[0-9]{5}$",county_fips_list),c(keys,"county_fips_list","adjusted_nbr_workers_certified"),with=FALSE]
setnames(county,"county_fips_list","county_fips")
assert_geo_columns(county,"county_fips")
county <- merge(county,z[,.(fiscal_year,case_number,eligible)],by=keys)
county_year <- county[,.(all_counties=uniqueN(county_fips), usable_counties=uniqueN(county_fips[eligible]),
                          all_positions=sum(adjusted_nbr_workers_certified),
                          usable_positions=sum(adjusted_nbr_workers_certified[eligible])),by=fiscal_year]
emit(county_year,"unambiguous_counties")
for (kind in c("title","soc")) {
  ctable <- if(kind=="title") title_pairs$cells else soc_pairs$cells
  sums <- ctable[dr!=0,.(cells=.N,filers=uniqueN(filer_R7), states=uniqueN(st),
              current_cases=sum(n),current_positions=sum(positions),mean_dr=weighted.mean(dr,positions),
              mean_dw=weighted.mean(dw,positions),within5=weighted.mean(abs(dw-dr)<=0.050001,positions),
              same_direction=weighted.mean(sign(dw)==sign(dr),positions)),by=.(refyear,direction=sign(dr))]
  emit(sums,paste0("renewal_",kind))
}
stopifnot(all(annual$n<=annual$denom_n),all(annual$pos<=annual$denom_pos+1e-6),
          all(county_year$usable_counties<=county_year$all_counties))
