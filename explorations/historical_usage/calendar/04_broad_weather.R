# Summer weather, contemporaneous agricultural outcomes, and next year's H-2A use.
# Run from the repository root. Emits aggregate diagnostics to standard output.
source("code/paths.R")
source(path_code("c00_shared", "geography.R"))
library(data.table)
setDTthreads(2L)
fixest::setFixest_nthreads(2L)
panel <- as.data.table(arrow::read_parquet(path_processed("county_year_panel.parquet"),
  col_select = c("county_fips","state_fips","year","any_cropland_2007",
                 "nbr_workers_certified_start_year","nbr_applications_start_year")))
setnames(panel, c("nbr_workers_certified_start_year","nbr_applications_start_year"), c("positions","applications"))
assert_geo_columns(panel,c("county_fips","state_fips"))
stopifnot(!anyDuplicated(panel, by=c("county_fips","year")), !anyNA(panel$positions),
          !anyNA(panel$applications), all(panel$positions >= 0), all(panel$applications >= 0))
frame <- unique(panel[any_cropland_2007 == TRUE, .(county_fips,state_fips)])
stopifnot(!anyDuplicated(frame$county_fips), nrow(frame)*18L == nrow(panel[any_cropland_2007 == TRUE]))

# Weather aggregation requires each day in the summer. The raw archive stores
# county-area average Celsius and millimeters as strings, with padded invalid dates.
weather <- rbindlist(lapply(1981:2025, function(y) {
  monthly <- rbindlist(lapply(6:8, function(m) {
    x <- as.data.table(arrow::read_parquet(path_raw("epinoaa_nclimgrid", sprintf("%04d%02d.parquet", y,m)),
                              col_select = c("region_type","fips","date","tmax","prcp")))
    x <- x[region_type == "cty" & !is.na(date)]
    stopifnot(!anyDuplicated(x, by=c("fips","date")))
    x[, `:=`(tmax = as.numeric(tmax), prcp = as.numeric(prcp))]
    x[, .(temp = sum(tmax), rain = sum(prcp), days = .N,
          valid = all(is.finite(tmax) & abs(tmax)<75 & is.finite(prcp) & prcp>=0)), by = fips]
  }))
  x <- monthly[, .(temp = sum(temp)/92, rain = sum(rain), days=sum(days), valid=all(valid)), by=fips]
  x[, county_fips := harmonize_county_fips_2010(fips)]
  stopifnot(!anyDuplicated(x$county_fips))
  x[valid == TRUE & days==92L, .(county_fips, year=y, temp, rain)]
}))
assert_geo_columns(weather,"county_fips")
stopifnot(!anyDuplicated(weather, by=c("county_fips","year")))
norm <- weather[year <= 2006L, .(n=.N, mt=mean(temp), st=sd(temp), mr=mean(rain), sr=sd(rain)),by=county_fips]
norm <- norm[n==26L & st>0 & sr>0]
weather <- merge(merge(weather[year>=2007L],norm,by="county_fips"),frame,by="county_fips")
weather[, `:=`(hot=(temp-mt)/st, dry=(mr-rain)/sr)]
stopifnot(all(is.finite(weather$hot)), all(is.finite(weather$dry)))
heads <- merge(panel, weather[,.(county_fips,year=year+1L,hot,dry)],by=c("county_fips","year"))
stopifnot(!anyDuplicated(heads, by=c("county_fips","year")), all(heads[, .N, by=county_fips]$N == 18L))
alloc <- as.data.table(arrow::read_parquet(path_int("h2a_allocated_worksites.parquet"),
  col_select=c("period_begin_date","adjusted_nbr_workers_certified")))
alloc[, year := as.integer(substr(as.character(period_begin_date),1L,4L))]
nat <- alloc[year %in% 2008:2025, .(source_pos=sum(adjusted_nbr_workers_certified,na.rm=TRUE)), by=year]
den <- panel[,.(panel_pos=sum(positions),panel_app=sum(applications)),by=year]
cover <- heads[,.(n=.N,states=uniqueN(state_fips),zero=sum(positions==0),pos=sum(positions),app=sum(applications)),by=year]
cover <- merge(merge(cover,den,by="year"),nat,by="year")
cover[, `:=`(pos_panel=pos/panel_pos,app_panel=app/panel_app,pos_source=pos/source_pos)]
cat("Coverage by start year. Source positions include unresolved geographies with a recorded start date.\n")
print(cover[order(year)])
print(cover[,.(county_years=sum(n),zero=sum(zero),pos=sum(pos),app=sum(app),pos_panel=sum(pos)/sum(panel_pos),
              app_panel=sum(app)/sum(panel_app),pos_source=sum(pos)/sum(source_pos))])
print(weather[year<=2024,.(counties=uniqueN(county_fips),states=uniqueN(state_fips),
  hot_years_above_one_sd=sum(hot>1),dry_years_above_one_sd=sum(dry>1),
  typical_temp_sd=median(st),typical_rain_sd=median(sr),max_hot=max(hot),max_dry=max(dry))])

# Fractions of raw exposure variation left after county and state-by-year effects.
# Also measure concentration across states of each exposure's independent residual sum of squares.
largest <- character()
for (term in c("hot","dry")) {
  f <- as.formula(paste0(term," ~ 1 | county_fips + state_fips^year"))
  f_other <- as.formula(paste0(term," ~ ",if(term=="hot") "dry" else "hot"," | county_fips + state_fips^year"))
  e <- resid(fixest::feols(f,heads,fixef.rm="none",notes=FALSE))
  r <- resid(fixest::feols(f_other,heads,fixef.rm="none",notes=FALSE))
  totals <- data.table(state=heads$state_fips, ss=r^2)[,.(ss=sum(ss)),by=state]
  largest[term] <- totals$state[which.max(totals$ss)]
  print(data.table(term,largest_state=largest[term],sd_after_FE=sd(e),frac_var_after_FE=var(e)/var(heads[[term]]),
    effective_states=sum(totals$ss)^2/sum(totals$ss^2),max_state_share=max(totals$ss)/sum(totals$ss)))
}

# All coefficients are per one historical within-county SD. CIs cluster by state.
fit <- function(data, outcome, label, ppml=FALSE, terms="hot + dry",
                fixed_effects="county_fips + state_fips^year") {
  f <- as.formula(paste(outcome,"~",terms,"|",fixed_effects))
  m <- if (ppml) fixest::fepois(f,data=data,vcov=~state_fips,notes=FALSE) else
    fixest::feols(f,data=data,vcov=~state_fips,notes=FALSE)
  kept <- data[fixest::obs(m)]
  intervals <- confint(m)
  print(data.table(label=label,term=names(coef(m)),b=unname(coef(m)),low=intervals[,1],high=intervals[,2],
    N=nobs(m),counties=uniqueN(kept$county_fips),states=uniqueN(kept$state_fips),
    years=paste(range(kept$year),collapse="-")))
  invisible(m)
}
fit(heads,"log1p(positions)","Prior summer, positions")
fit(heads,"log1p(applications)","Prior summer, applications")
fit(heads,"I(positions > 0)","Prior summer, any positive positions")
fit(heads,"positions","Prior summer, positions PPML",TRUE)
# Compare lag and future weather in identical rows and a single regression.
timing <- merge(heads,weather[,.(county_fips,year=year-1L,hot_future=hot,dry_future=dry)],by=c("county_fips","year"))
timing_model <- fit(timing,"log1p(positions)","Prior and future summer, positions",terms="hot + dry + hot_future + dry_future")
for (term in c("hot","dry")) {
  contrast <- as.numeric(names(coef(timing_model))==term) - as.numeric(names(coef(timing_model))==paste0(term,"_future"))
  diff <- sum(contrast*coef(timing_model))
  margin <- qt(.975, fixest::degrees_freedom(timing_model,"t")) * sqrt(drop(t(contrast) %*% vcov(timing_model) %*% contrast))
  print(data.table(lag_minus_lead=term,b=diff,low=diff-margin,high=diff+margin))
}
fit(timing,"positions","Prior and future summer, positions PPML",TRUE,terms="hot + dry + hot_future + dry_future")
fit(heads[year %in% 2008:2018],"log1p(positions)","Earlier cohort, prior summer")
fit(heads[year %in% 2019:2025],"log1p(positions)","Later cohort, prior summer")

fit(heads[!state_fips %in% largest["hot"]],"log1p(positions)","Drop state with largest independent heat-variation share")
heads[, centered_year := year-2016L]
fit(heads,"log1p(positions)","County-specific linear trends",fixed_effects="county_fips[centered_year] + state_fips^year")

# Independently check whether the same weather variables detect realized crop
# output and covered crop employment in the weather year, before next-year starts.
corn <- arrow::open_dataset(path_int("qs_survey_crops.parquet"))
corn <- dplyr::filter(corn, short_desc=="CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
  agg_level_desc=="COUNTY", freq_desc=="ANNUAL", reference_period_desc=="YEAR",
  domain_desc=="TOTAL", domaincat_desc=="NOT SPECIFIED", year>=2007, year<=2024)
corn <- as.data.table(dplyr::collect(dplyr::select(corn, state_fips, county_code, year, value)))
corn <- corn[!is.na(state_fips) & !is.na(county_code) & nzchar(trimws(county_code))]
corn[, county_fips := harmonize_county_fips_2010(combine_county_fips(state_fips,county_code))]
corn[, corn_yield := suppressWarnings(as.numeric(gsub(",","",value,fixed=TRUE)))]
corn <- merge(weather,corn[is.finite(corn_yield) & corn_yield>0,.(county_fips,year,corn_yield)],by=c("county_fips","year"))
stopifnot(!anyDuplicated(corn,by=c("county_fips","year")))
fit(corn,"log(corn_yield)","Contemporaneous corn grain yield")

qcew <- as.data.table(arrow::read_parquet(path_int("qcew_county_ag_quarterly_employment.parquet")))
qcew <- qcew[industry_code=="111" & reference_month=="october" & qcew_employment_disclosed==TRUE &
  is.finite(qcew_reference_month_emplvl) & year %in% 2007:2024,.(county_fips,year,empl=qcew_reference_month_emplvl)]
assert_geo_columns(qcew,"county_fips")
stopifnot(!anyDuplicated(qcew,by=c("county_fips","year")))
fit(merge(weather,qcew,by=c("county_fips","year")),"log1p(empl)","Contemporaneous October covered crop jobs")

# Was the previous summer finished before the OFLC receipt, where observed?
# Restrict to FY2008-25 primary cases and requested starts in the outcome calendar.
source <- arrow::open_dataset(path_int("h2a_with_fips.parquet"))
source <- dplyr::filter(source, fiscal_year>=2008L, fiscal_year<=2025L)
dates <- as.data.table(dplyr::collect(dplyr::select(source, fiscal_year, case_number, primary_sub,
                                                   case_received_date, requested_begin_date)))
dates <- unique(dates[!primary_sub %in% "SUB" & !is.na(case_number) & nzchar(case_number),
  .(case_number,fiscal_year,receipt=as.IDate(case_received_date,format="%Y-%m-%d"),
    start=as.IDate(requested_begin_date,format="%Y-%m-%d"))])
dates <- dates[!duplicated(case_number) & !duplicated(case_number,fromLast=TRUE)]
dates[, year:=as.integer(format(start,"%Y"))]
dates <- dates[year %in% 2008:2025 & abs(year-fiscal_year)<=1L]
dates[, cutoff:=as.IDate(sprintf("%04d-08-31",year-1L))]
dates[abs(as.integer(format(receipt,"%Y"))-fiscal_year)>1L, receipt:=as.IDate(NA)]
print(dates[,.(cases=.N,dated=sum(!is.na(receipt)),share_receipt_after=mean(receipt>cutoff,na.rm=TRUE)),
            by=.(recent=fiscal_year>=2020L)])
