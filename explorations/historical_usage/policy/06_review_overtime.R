# Advertised weekly hours at moving agricultural overtime premium thresholds.
# Primary sources and prior county design:
# https://www.lni.wa.gov/workers-rights/agriculture-policies/agricultural-overtime-faq
# https://www.oregon.gov/boli/employers/Pages/minimum-wage-and-overtime-in-agriculture.aspx
# https://www.dir.ca.gov/dlse/Overtime-for-Agricultural-Workers-FAQ.html
source("code/paths.R")
library(data.table)
setDTthreads(2L)
out <- commandArgs(trailingOnly=TRUE)[1L]
stopifnot(!is.na(out))
dir.create(out, showWarnings=FALSE, recursive=TRUE)
stopifnot(startsWith(normalizePath(out), "/tmp/"))
emit <- function(x, label) fwrite(x, file.path(out, paste0("overtime_", label, ".csv")))
read_fields <- function(file, cols) as.data.table(arrow::read_parquet(path_int(file),
                                      col_select=tidyselect::all_of(cols)))
schedule <- fread(path_root("explorations","identification","sources","state_ag_overtime_schedule.csv"),
                  colClasses=c(state_fips="character"))
s <- schedule[(state_abbrev == "WA" & sector_scope == "all_agriculture_excluding_dairy") |
              (state_abbrev == "OR") | (state_abbrev == "CA")]
s[, effective_start := as.IDate(effective_start)]
s[, effective_end := as.IDate(effective_end)]
s[is.na(effective_end), effective_end := as.IDate("2099-12-31")]
by_year <- rbindlist(lapply(2017:2024, function(yr) {
  d <- as.IDate(sprintf("%d-07-01",yr))
  s[effective_start <= d & effective_end >= d, .(state_abbrev, employer_size_class, year=yr,
                                                 weekly_threshold_hours, daily_threshold_hours)]
}))
stopifnot(nrow(by_year)==32L)
emit(by_year, "schedule")
fields <- c("case_number","fiscal_year","primary_sub","worksite_state","period_begin_date",
            "number_of_hours","soc_code","nbr_workers_certified")
raw <- read_fields("h2a_disclosure_records.parquet", fields)
# Select canonical start-year cohorts, using adjacent FY partitions. Working in
# a three-state frame, exclude mixed/unidentified allocation states (including
# unknowns), and keep a single unique master/standalone case before selecting hours.
raw <- raw[fiscal_year %between% c(2016L,2025L) &
             (is.na(primary_sub) | toupper(trimws(primary_sub))!="SUB")]
raw <- unique(raw[, .(case_number,fiscal_year,worksite_state,start=as.IDate(period_begin_date),
                       h=suppressWarnings(as.numeric(number_of_hours)),soc=substr(soc_code,1,7),
                       positions=suppressWarnings(as.numeric(nbr_workers_certified)))])
raw <- raw[nzchar(trimws(case_number)) & !duplicated(case_number) & !duplicated(case_number,fromLast=TRUE)]
raw <- raw[worksite_state %chin% c("CA","OR","WA") & year(start) %between% c(2017L,2024L) &
           positions > 0 & soc == "45-2092"]  # crop, nursery, and greenhouse farmworkers
sites <- read_fields("h2a_allocated_worksites.parquet", c("case_number","fiscal_year","worksite_state"))
sites <- sites[case_number %chin% raw$case_number]
translate <- c(setNames(state.abb, toupper(state.name)), setNames(state.abb, state.abb))
sites[, st := unname(translate[toupper(trimws(worksite_state))])]
site <- sites[, .(one=!anyNA(st) && uniqueN(st)==1L, st=first(st)), by=.(case_number,fiscal_year)]
x <- merge(raw, site[one==TRUE], by=c("case_number","fiscal_year"))[st==worksite_state]
x[, `:=`(year=year(start), m=month(start), valid=is.finite(h) & h>0 & h<=168)]
stopifnot(nrow(x)>0, !anyDuplicated(x$case_number))
emit(x[, .(cases=.N, observed=sum(valid), below_35=sum(valid & h<35,na.rm=TRUE)), by=.(st,year)],
     "hour_coverage")
x <- x[valid==TRUE]
ids <- read_fields("policy_2025/filer/filer_ids.parquet", c("case_number","fiscal_year","filer_R7"))
stopifnot(!anyDuplicated(ids,by=c("case_number","fiscal_year")))
x <- merge(x,ids,by=c("case_number","fiscal_year"),all.x=TRUE)
full <- copy(x)
profile <- function(z) {
  rbindlist(lapply(c(40,42,45,48,50,52,55), function(k) {
    z[, .(threshold=k, cases=.N, exact=sum(h==k), below=sum(h<k), above=sum(h>k),
          share=mean(h==k), above_share=mean(h>k), mean_posted=mean(h),
          position_weighted_share=weighted.mean(h==k,positions)), by=.(st,year)]
  }))
}
annual <- profile(x)
emit(annual, "annual")
emit(profile(x[m %between% c(4L,9L)]), "april_september")
emit(x[,.(cases=.N, positions=sum(positions)),by=.(st,year,h)], "distribution")
# Compare WA with the adjacent state in the same AEWR region during the same
# transition. OR introduces 55 in 2023, holds it in 2024; it is a different
# regime, not an untreated comparison throughout. Include one-year-early placebo.
region <- unique(as.data.table(arrow::read_parquet(
  path_processed("county_year_panel.parquet"), col_select=c("state_abbrev","aewr_region_id","year","aewr")
))[state_abbrev %chin% c("WA","OR") & year %between% c(2017L,2024L)])
stopifnot(identical(unique(region$aewr_region_id),"1"),
          all(region[,.(n=uniqueN(aewr)),by=year]$n==1L))
events <- data.table(state=c("WA","WA","WA","CA","CA","CA","CA","OR"),
                     yr=c(2022L,2023L,2024L,2019L,2020L,2021L,2022L,2023L),
                     k=c(55,48,40,55,50,45,40,55))
compare <- function(study_state, first, k, timing) {
  z <- full[year %in% c(first-1L,first)]
  z[, hit := as.numeric(h==k)]
  z[, above := as.numeric(h>k)]
  stats <- z[,.(n=.N, share=mean(hit), above_share=mean(above)),by=.(st,year)]
  # Within-filer/state AND start month matched in both years; equal cell weights.
  cells <- z[!is.na(filer_R7),.(rate=mean(hit)),by=.(st,filer_R7,m,year)]
  matched <- cells[, if (.N==2L) .(difference=rate[year==first]-rate[year==first-1L]),
                   by=.(st,filer_R7,m)]
  d <- merge(stats[year==first], stats[year==first-1L], by="st",suffixes=c("_new","_old"))
  d <- merge(d, matched[,.(matched_cells=.N, matched_change=mean(difference)),by=st],by="st")
  d[, `:=`(delta=share_new-share_old, delta_above=above_share_new-above_share_old)]
  comparison <- if(study_state=="WA") "OR" else NA_character_
  dd <- if(study_state=="WA") d[st=="WA",delta] - d[st=="OR",delta] else NA_real_
  dd_fixed <- if(study_state=="WA") d[st=="WA",matched_change] - d[st=="OR",matched_change] else NA_real_
  d[st==study_state, .(st,year=first,k,timing,comparison,n_old,n_new,share_old,share_new,delta,delta_above,
                        matched_cells, matched_change, dd, dd_fixed)]
}
res <- rbindlist(lapply(seq_len(nrow(events)),function(i) {
  rbind(compare(events$state[i],events$yr[i],events$k[i],"phase_in"),
        compare(events$state[i],events$yr[i]-1L,events$k[i],"year_early"))
}))
emit(res,"transitions")
stopifnot(all(res$n_new>0), all(res$matched_cells>0))
