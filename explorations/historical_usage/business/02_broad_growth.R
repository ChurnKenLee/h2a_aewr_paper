# Disclosure filing coverage and exact continuing-filer volume accounts.
source("code/paths.R")
library(data.table)
setDTthreads(2L)
out <- commandArgs(trailingOnly=TRUE)[1L]
if (is.na(out)) out <- "/tmp/h2a-broad-growth"
dir.create(out,recursive=TRUE,showWarnings=FALSE)
stopifnot(startsWith(normalizePath(out),"/tmp/"))
fy <- 2008:2025
keys <- c("fiscal_year","case_number")
read_cols <- function(file,cols) {
  as.data.table(arrow::read_parquet(path_int(file),col_select=all_of(cols)))[fiscal_year %in% fy]
}
raw <- read_cols("h2a_disclosure_records.parquet",c(keys,"nbr_workers_certified","case_status"))
raw[, raw_positive := suppressWarnings(as.numeric(nbr_workers_certified))>0]
stopifnot(all(!is.na(raw$case_number) & nzchar(trimws(raw$case_number))))
spine <- raw[,.(any_raw_positive=any(raw_positive,na.rm=TRUE),
                certified_status=all(grepl("CERTIFIED|CERTIFICATION",toupper(case_status)))),
             by=keys]
ids <- read_cols("policy_2025/filer/filer_ids.parquet",c(keys,"filer_R5","filer_R7","filer_R8"))
audit <- read_cols("h2a_case_allocation_audit.parquet",c(keys,"case_allocation_reconciled","case_certified_target"))
stopifnot(!anyDuplicated(ids,by=keys),!anyDuplicated(audit,by=keys),nrow(audit[!spine,on=keys])==0L)
spine <- merge(merge(spine,ids,by=keys,all.x=TRUE),audit,by=keys,all.x=TRUE)
stopifnot(!any(spine[is.na(case_allocation_reconciled),any_raw_positive]))
# Cases absent from the audit have no positive source count. Quarantined
# outcomes remain unknown and those cases are excluded from volume models.
spine[, excluded := !is.na(case_allocation_reconciled) & !case_allocation_reconciled]
spine[, positions := fifelse(is.na(case_allocation_reconciled),0,case_certified_target)]
stopifnot(!anyNA(spine[excluded==FALSE, positions]),all(spine[excluded==FALSE, positions]>=0))
cov <- spine[,.(disclosed_cases=.N, filer_linked=sum(!is.na(filer_R7)),
  missing_allocation_audit=sum(is.na(case_allocation_reconciled)),quarantined=sum(excluded),
  known_positions_in_quarantine=sum(positions[excluded],na.rm=TRUE),
  linked_usable_cases=sum(!excluded & !is.na(filer_R7)),
  all_positive_cases=sum(!excluded & positions>0), positive_linked=sum(!excluded & positions>0 & !is.na(filer_R7)),
  linked_positions=sum(positions[!excluded & !is.na(filer_R7)]),
  total_positions=sum(positions[!excluded]),
  linked_positive_filers=uniqueN(filer_R7[!excluded & positions>0],na.rm=TRUE)),by=fiscal_year]
stopifnot(all(cov$positive_linked==cov$all_positive_cases),all(cov$linked_positions==cov$total_positions))
fwrite(cov[order(fiscal_year)],file.path(out,"coverage.csv"))
fwrite(spine[,.(disclosed_cases=.N,linked_cases=sum(!is.na(filer_R7)),distinct_filers=uniqueN(filer_R7,na.rm=TRUE),
                positive_cases=sum(!excluded & positions>0),
                distinct_positive_filers=uniqueN(filer_R7[!excluded & positions>0],na.rm=TRUE),
                usable_positions=sum(positions[!excluded]))],file.path(out,"national_support.csv"))
model <- spine[!excluded & !is.na(filer_R7)]

# At each adjacent-year comparison, P=n*s for number of cases n and positions
# per case s. Symmetric components are (n1-n0)*(s0+s1)/2 and
# (s1-s0)*(n0+n1)/2, computed within filer before summing. Consecutive-year
# activity can begin or end at any point in the window.
flows <- function(a) {
  setorder(a,region,id,fiscal_year)
  output <- vector("list",length(fy)-1L)
  for (i in seq_along(output)) {
    year <- fy[i+1L]
    p <- a[fiscal_year==year-1L,.(id,region,n0=orders,p0=positions)]
    q <- a[fiscal_year==year,.(id,region,n1=orders,p1=positions)]
    b <- merge(p,q,by=c("id","region"),all=TRUE)
    b[, continuing := !is.na(n0) & !is.na(n1)]
    b[, order_term := 0.0]
    b[, size_term := 0.0]
    b[continuing==TRUE,`:=`(
      order_term=(n1-n0)*(p0/n0+p1/n1)/2,
      size_term=(p1/n1-p0/n0)*(n1+n0)/2)]
    output[[i]] <- b[,.(
      fiscal_year=year,previous=sum(p0,na.rm=TRUE),current=sum(p1,na.rm=TRUE),
      continuing=sum(p1[continuing]-p0[continuing]),order_term=sum(order_term),
      size_term=sum(size_term),appear=sum(p1[is.na(n0)]),disappear=-sum(p0[is.na(n1)])),by=region]
  }
  x <- rbindlist(output)
  stopifnot(max(abs(x$continuing-x$order_term-x$size_term))<1e-6,
            max(abs(x$current-x$previous-x$continuing-x$appear-x$disappear))<1e-6)
  x
}
national <- list()
for (tier in c("filer_R5","filer_R7","filer_R8")) {
  for (sample in c("positive_case","all_linked_cases","positive_confirmed_status")) {
    d <- switch(sample,positive_case=model[positions>0],all_linked_cases=model,
                positive_confirmed_status=model[positions>0 & certified_status==TRUE])
    x <- d[,.(orders=.N,positions=sum(positions)),by=.(id=get(tier),fiscal_year)][,region:="national"]
    national[[length(national)+1L]] <- flows(x)[,`:=`(unit=tier,sample=sample)]
  }
}
all <- rbindlist(national)
# State uses actual row-level worksite states and reconciled allocated positions.
# A multi-state case counts once in each state where it has positive positions;
# national order counts are calculated separately on unique case IDs.
loc <- read_cols("h2a_worksite_exposure_audit.parquet",c(keys,"worksite_state","adjusted_nbr_workers_certified",
                                                         "case_allocation_reconciled"))
loc <- loc[case_allocation_reconciled==TRUE & adjusted_nbr_workers_certified>0]
loc[, region:=toupper(trimws(worksite_state))]
lookup <- c(setNames(state.abb,toupper(state.name)),"DISTRICT OF COLUMBIA"="DC","PUERTO RICO"="PR",
            "VIRGIN ISLANDS"="VI","U.S. VIRGIN ISLANDS"="VI","AMERICAN SAMOA"="AS",
            "GUAM"="GU","NORTHERN MARIANA ISLANDS"="MP")
loc[region %chin% names(lookup),region:=unname(lookup[region])]
loc[is.na(region) | !nzchar(region),region:="unknown"]
site <- loc[,.(positions=sum(adjusted_nbr_workers_certified)),by=.(fiscal_year,case_number,region)]
site <- merge(site,ids[,c(keys,"filer_R7"),with=FALSE],by=keys,all.x=TRUE)
stopifnot(!anyNA(site$filer_R7))
geo_cov <- site[,.(positions=sum(positions),cases=uniqueN(case_number)),by=.(fiscal_year,region)]
geo_cov[, scope:=fcase(region %chin% state.abb,"50_states",region %chin% c("DC","PR","VI","AS","GU","MP"),"DC_territories",
                       default="unresolved_or_other")]
fwrite(geo_cov[order(fiscal_year,region)],file.path(out,"state_coverage.csv"))
test <- merge(cov,geo_cov[,.(allocated_positions=sum(positions)),by=fiscal_year],by="fiscal_year")
stopifnot(max(abs(test$allocated_positions-test$total_positions))<1e-6)
state <- flows(site[region %chin% state.abb,.(orders=.N,positions=sum(positions)),
                    by=.(id=filer_R7,region,fiscal_year)])[, `:=`(unit="filer_R7_in_state",sample="positive_case")]
all <- rbind(all,state)
fwrite(all,file.path(out,"annual_flows.csv"))
windows <- list(c(2008,2025),c(2009,2025),c(2011,2025),c(2008,2019),c(2011,2019),c(2019,2020),c(2020,2025))
period <- rbindlist(lapply(windows,function(w) {
  all[fiscal_year>w[1] & fiscal_year<=w[2],
      .(start=w[1],end=w[2],observed_transitions=.N,change=sum(current-previous),
        continuing=sum(continuing),orders=sum(order_term),size=sum(size_term),
        appearances=sum(appear),disappearances=sum(disappear)),by=.(unit,sample,region)]
}))
period[, `:=`(continuing_share=fifelse(change!=0,continuing/change,NA_real_),
              order_share_of_continuing=fifelse(continuing!=0,orders/continuing,NA_real_))]
fwrite(period[order(unit,sample,region,start,end)],file.path(out,"period_flows.csv"))

# Condition on the filer having positive reconciled positions anywhere in
# both adjacent years. Partition its national change by each worksite state's
# presence in t-1 and t. Territory, foreign and unresolved records stay in the
# identity, with a separately labeled net term outside the 50-state frame.
geo <- site[,.(positions=sum(positions), n_cases=.N),by=.(fiscal_year,region,id=filer_R7)]
geo[, first_region := min(fiscal_year),by=.(id,region)]
nat <- all[unit=="filer_R7" & sample=="positive_case" & region=="national",
           .(fiscal_year,national_continuing=continuing)]
geo_parts <- vector("list",length(fy)-1L)
for (i in seq_along(geo_parts)) {
  year <- fy[i+1L]
  ids_both <- intersect(geo[fiscal_year==year-1L,id],geo[fiscal_year==year,id])
  left <- geo[fiscal_year==year-1L & id %in% ids_both,
              .(id,region,p0=positions,n0=n_cases)]
  right <- geo[fiscal_year==year & id %in% ids_both,
               .(id,region,p1=positions,n1=n_cases,first_region)]
  b <- merge(left,right,by=c("id","region"),all=TRUE)
  b[, account := fcase(region=="unknown","unresolved_state",
                       region %chin% c("DC","PR","VI","AS","GU","MP"),"DC_territories",
                       !region %chin% state.abb,"other_nonstate",
                       !is.na(p0) & !is.na(p1),"continuing_state",
                       is.na(p1),"discontinued_state",
                       first_region==year,"first_seen_state",default="returning_state")]
  b[, delta := fcoalesce(p1,0)-fcoalesce(p0,0)]
  b[, `:=`(orders=0.0,size=0.0)]
  b[account=="continuing_state",`:=`(
    orders=(n1-n0)*(p0/n0+p1/n1)/2,
    size=(p1/n1-p0/n0)*(n1+n0)/2)]
  geo_parts[[i]] <- b[,.(fiscal_year=year,positions=sum(delta),orders=sum(orders),size=sum(size),
                         pairs=.N),by=.(region,account)]
}
changes <- rbindlist(geo_parts)
test <- merge(changes[,.(accounted=sum(positions)),by=fiscal_year],nat,by="fiscal_year")
stopifnot(nrow(test)==17L,max(abs(test$accounted-test$national_continuing))<1e-6)
stopifnot(changes[account=="continuing_state",max(abs(orders+size-positions))]<1e-6)
fwrite(changes[order(fiscal_year,region,account)],file.path(out,"state_flows_national_continuers.csv"))
geoperiod <- rbindlist(lapply(windows,function(w) {
  changes[fiscal_year>w[1] & fiscal_year<=w[2],
          .(start=w[1],end=w[2],positions=sum(positions),orders=sum(orders),
            size=sum(size),annual_pair_observations=sum(pairs)),by=.(region,account)]
}))
geo_total <- geoperiod[,.(positions=sum(positions),orders=sum(orders),size=sum(size)),
                       by=.(start,end,account)]
fwrite(geo_total[order(start,end,account)],file.path(out,"national_continuer_geography.csv"))
# Influence of omitting the state with the largest net contribution. The cohort
# stays nationally defined, so these totals condition on the same filers.
top <- geoperiod[region %chin% state.abb,.(net=sum(positions)),by=.(start,end,region)]
setorder(top,start,end,-net,region)
top[,rank:=seq_len(.N),by=.(start,end)]
sizes <- period[unit=="filer_R7" & sample=="positive_case" & region=="national",
                .(start,end,national_continuing=continuing)]
top <- merge(top,sizes,by=c("start","end"))
top[, `:=`(net_share=net/national_continuing,after_omission=national_continuing-net)]
fwrite(top[order(start,end,rank)],file.path(out,"state_concentration.csv"))
