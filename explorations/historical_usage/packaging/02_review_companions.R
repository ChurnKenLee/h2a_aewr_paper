# Follow-up to the 124-case descriptive test, allowing all filing roles and
# different filers, repeated dates and literal shared worksite addresses.
source("code/paths.R")
library(data.table)
setDTthreads(2L)
out <- commandArgs(trailingOnly=TRUE)[1L]
if (is.na(out)) out <- "/tmp/h2a-history-companions-review"
dir.create(out,recursive=TRUE,showWarnings=FALSE)
stopifnot(startsWith(normalizePath(out),"/tmp/"))
read_cols <- function(name, cols) {
  as.data.table(arrow::read_parquet(path_int(name),col_select=all_of(cols)))
}
fields <- c("fiscal_year","case_number","case_received_date","decision_date",
            "type_of_employer_application","h2a_labor_contractor","nbr_workers_requested","worksite_state")
extra <- c("requested_begin_date","soc_code","case_status")
raw <- read_cols("h2a_disclosure_records.parquet", c(fields,extra))[fiscal_year %between% c(2020L,2025L)]
raw[, received := suppressWarnings(as.IDate(substr(case_received_date,1,10)))]
raw[, decided := suppressWarnings(as.IDate(substr(decision_date,1,10)))]
raw[, workers := suppressWarnings(as.numeric(nbr_workers_requested))]
a <- unique(raw[,.(fiscal_year,case_number,received,decided,workers,worksite_state,
                    type_of_employer_application,h2a_labor_contractor)])
a <- a[nzchar(case_number) & !duplicated(case_number) & !duplicated(case_number,fromLast=TRUE)]
add <- unique(raw[, c("fiscal_year","case_number",extra),with=FALSE])
add <- add[!duplicated(case_number) & !duplicated(case_number,fromLast=TRUE)]
keys <- c("fiscal_year","case_number")
ids <- read_cols("policy_2025/filer/filer_ids.parquet",c(keys,"filer_R7"))
agents <- read_cols("policy_2025/filer/filer_cases.parquet", c(keys,"agent_key"))
stopifnot(!anyDuplicated(ids,by=keys),!anyDuplicated(agents,by=keys))
a <- merge(merge(merge(a,add,by=keys),ids,by=keys,all.x=TRUE),agents,by=keys,all.x=TRUE)
a[, start := suppressWarnings(as.IDate(substr(requested_begin_date,1,10)))]
a[, certified := grepl("CERTIFIED|CERTIFICATION",toupper(case_status))]
a <- a[as.integer(decided-received) %between% c(0L,180L) & workers>0 & workers==floor(workers) &
       !is.na(start) & abs(year(start)-fiscal_year)<=1L]
stopifnot(max(a$decided)<=as.IDate("2025-09-30"),max(a$fiscal_year)==2025L,!anyDuplicated(a$case_number))
a[, soc := gsub("[^0-9]","",soc_code)]
a[nchar(soc)!=6L,soc:=NA_character_] # O*NET extension handled below if present
# Use the six-digit occupation for eight-digit O*NET occupation codes.
a[grepl("^[0-9]{2}-[0-9]{4}\\.[0-9]{2}$",soc_code),soc:=substr(gsub("[^0-9]","",soc_code),1,6)]
a[, era := fcase(received >= "2020-04-01" & received < "2022-04-01","early",
                 received >= "2023-04-01" & received < "2025-04-01","late")]
a[, firm := as.character(filer_R7)]
f <- copy(a[!is.na(era) & type_of_employer_application=="Individual Employer" & h2a_labor_contractor=="Y"])
f[, cap := 50L*ceiling(workers/50)]
f[, pseudo := 25L+50L*ceiling((workers-25L)/50L)]
f[, group := fcase(cap %in% c(150L,200L,250L,300L) & (cap-workers) %between% c(1,5),
                   paste0(era,"_statutory"),
                   pseudo %in% c(125L,175L,225L,275L) & (pseudo-workers) %between% c(1,5),
                   paste0(era,"_nonstep"))]
f <- f[!is.na(group)]
f[endsWith(group,"_nonstep"),cap:=pseudo]
stopifnot(nrow(f[group=="late_statutory"])==124L,nrow(f[group=="early_statutory"])==27L)

# Strict address match on observed street, city, state and ZIP. Ignore
# geocoding coordinates, county imputations, employer addresses and site
# staffing allocations. Include the FY2025 main workbook since the retained
# main-address extractor ends FY2024. No names or addresses leave memory.
fields_site <- c("fiscal_year","case_number","source_dataset","address_id")
links <- read_cols("h2a_worksite_geocode_records.parquet",fields_site)[fiscal_year %between% c(2020L,2025L)]
addresses <- read_cols("h2a_worksite_geocode_addresses.parquet",c("address_id","street","city","state","zip"))
stopifnot(!anyDuplicated(addresses$address_id))
sites <- merge(links,addresses,by="address_id",all.x=TRUE)[,address_id:=NULL]
setnames(sites,c("street","city","state","zip"),c("source_street","source_city","source_state","source_zip"))
if (!nrow(sites[fiscal_year==2025 & source_dataset=="main_disclosure_worksite"])) {
  path <- path_raw("h2a","H-2A_Disclosure_Data_FY2025_Q4.xlsx")
  want <- c("CASE_NUMBER","WORKSITE_ADDRESS","WORKSITE_CITY","WORKSITE_STATE","WORKSITE_POSTAL_CODE")
  header <- names(readxl::read_excel(path,n_max=0))
  stopifnot(all(want %in% header))
  cols <- ifelse(header %in% want,"text","skip")
  r25 <- as.data.table(readxl::read_excel(path,col_types=cols))
  setnames(r25,want,c("case_number","source_street","source_city","source_state","source_zip"))
  r25[, `:=`(fiscal_year=2025L,source_dataset="main_disclosure_worksite")]
  sites <- rbind(sites,r25,use.names=TRUE)
}
clean <- function(z) toupper(trimws(gsub("[[:space:]]+"," ",z)))
for (name in c("source_street","source_city","source_state","source_zip")) set(sites,j=name,value=clean(sites[[name]]))
sites[, source_zip := sub("^([0-9]{5})-[0-9]{4}$","\\1",source_zip)]
sites <- sites[!is.na(source_city) & nzchar(source_city) & nchar(source_city)>1 &
               grepl("^[A-Z]{2}$",source_state) & grepl("^[0-9]{5}$",source_zip) &
               grepl("[0-9]",source_street) & grepl("[A-Z]",source_street) &
               !grepl("P[.]? *O[.]? +BOX|POST OFFICE BOX",source_street)]
sites[, address := .GRP,by=.(source_street,source_city,source_state,source_zip)]
sites <- sites[case_number %chin% a$case_number,.(primary_site=any(source_dataset=="main_disclosure_worksite")),
               by=.(case_number,address)]
# Only retain possible case-to-case pairs sharing at least one observed site.
sf <- sites[case_number %chin% f$case_number,.(case_number,address,primary1=primary_site)]
sp <- sites[,.(peer=case_number,address,primary2=primary_site)]
ss <- merge(sf,sp,by="address",allow.cartesian=TRUE)[case_number!=peer,
              .(matched_site=TRUE,matched_primary=any(primary1 & primary2)),by=.(case_number,peer)]
f[, has_site := case_number %chin% sites$case_number]
f[, has_primary := case_number %chin% sites[primary_site==TRUE,case_number]]
f[, has_agent := !is.na(agent_key) & nzchar(agent_key)]
f[, has_soc := !is.na(soc) & nzchar(soc)]
f[, has_filer := !is.na(firm) & nzchar(firm)]
fwrite(f[,.(focal=.N,observed_site=sum(has_site),observed_primary=sum(has_primary),agent=sum(has_agent),
            occupation=sum(has_soc),filer=sum(has_filer),
            site_agent_soc=sum(has_site & has_agent & has_soc)),
         by=group],file.path(out,"coverage.csv"))

peer <- a[,.(peer=case_number,peer_start=start,peer_day=as.integer(start),peer_size=workers,
             peer_state=worksite_state,peer_agent=agent_key,peer_soc=soc,
             peer_filer=firm,peer_certified=certified,peer_received=received)]
f[, `:=`(lo=as.integer(start)-60L,hi=as.integer(start)+60L)]
p <- peer[f,on=.(peer_day>=lo,peer_day<=hi),nomatch=0L,allow.cartesian=TRUE][peer!=case_number]
p[, distance:=abs(as.integer(peer_start-start))]
p <- merge(p,ss,by=c("case_number","peer"),all.x=TRUE)
p[is.na(matched_site),`:=`(matched_site=FALSE,matched_primary=FALSE)]
p[, state_ok := !is.na(worksite_state) & nzchar(worksite_state) & worksite_state==peer_state]
p[, agent_ok := has_agent & !is.na(peer_agent) & agent_key==peer_agent]
p[, soc_ok := has_soc & !is.na(peer_soc) & soc==peer_soc]
p[, same := has_filer & !is.na(peer_filer) & firm==peer_filer]
p[, filer_match := fcase(same,"same",has_filer & !is.na(peer_filer),"different",default="unknown")]
p[, time_window := fifelse(distance<=30,"within_30","outward_31_60")]
p[, small_cross := peer_size<=10 & peer_size+workers>=cap]
p[, exact_residual := peer_size==cap-workers]
p[, identical := peer_size==workers]
p[, narrow := certified & peer_certified & abs(as.integer(peer_received-received))<=7L]
# SOC alone and an agent alone can each represent many unrelated farms. Compare
# state/agent/SOC matches to progressively tighter exact-address matches.
screens <- list(state_soc=p$state_ok & p$soc_ok,
                state_soc_agent = p$state_ok & p$soc_ok & p$agent_ok,
                address_or_state_agent_soc=p$soc_ok & (p$matched_site | (p$state_ok & p$agent_ok)),
                address_only=p$matched_site,
                address_soc=p$matched_site & p$soc_ok,
                address_soc_agent=p$matched_site & p$soc_ok & p$agent_ok,
                primary_address_soc_agent=p$matched_primary & p$soc_ok & p$agent_ok,
                address_soc_agent_certified_close_receipt=p$matched_site & p$soc_ok & p$agent_ok & p$narrow)
outcomes <- list(any=rep(TRUE,nrow(p)),small_cross=p$small_cross,
                 exact_residual=p$exact_residual,same_size=p$identical)
grid <- CJ(group=unique(f$group),time_window=c("within_30","outward_31_60"),
           filer_match=c("same","different","unknown","any"))
rows <- list()
for (s in names(screens)) for (o in names(outcomes)) {
  yes <- p[which(screens[[s]] & outcomes[[o]])]
  g <- yes[,.(focal_with_match=uniqueN(case_number),pairs=.N,matched_filers=uniqueN(firm,na.rm=TRUE)),
           by=.(group,time_window,filer_match)]
  total <- yes[,.(focal_with_match=uniqueN(case_number),pairs=.N,matched_filers=uniqueN(firm,na.rm=TRUE)),
               by=.(group,time_window)][,filer_match:="any"]
  u <- merge(grid,rbind(g,total),by=c("group","time_window","filer_match"),all.x=TRUE)
  for (name in c("focal_with_match","pairs","matched_filers")) setnafill(u,cols=name,fill=0)
  rows[[length(rows)+1L]] <- u[, `:=`(screen=s,outcome=o)]
}
result <- merge(rbindlist(rows),f[,.(all_focal=.N),by=group],by="group")
stopifnot(all(result$focal_with_match<=result$all_focal))
fwrite(result,file.path(out,"companion_counts.csv"))
