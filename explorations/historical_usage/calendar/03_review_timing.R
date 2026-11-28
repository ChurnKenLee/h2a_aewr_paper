# Case-receipt timing and frost support on a historically fixed county start calendar.
# Only aggregate diagnostics go to stdout. Run from the repository root.
source("code/paths.R")
source(path_code("c00_shared", "geography.R"))
library(data.table)
setDTthreads(2L)
cols <- c("fiscal_year", "case_number", "primary_sub", "case_received_date", "requested_begin_date",
          "county_fips_list", "nbr_workers_requested")
raw <- arrow::open_dataset(path_int("h2a_with_fips.parquet"))
raw <- dplyr::filter(raw, fiscal_year >= 2008L, fiscal_year <= 2025L)
d <- as.data.table(dplyr::collect(dplyr::select(raw, tidyselect::all_of(cols))))
d <- d[!primary_sub %in% "SUB" & !is.na(case_number) & nzchar(case_number)]
d[, receipt := as.IDate(case_received_date, format = "%Y-%m-%d")]
d[, start := as.IDate(requested_begin_date, format = "%Y-%m-%d")]
d[, requested := suppressWarnings(as.numeric(nbr_workers_requested))]
# Dates outside the source year +/-1 are unsupported. Keep negative lead times visible.
for (v in c("receipt", "start")) {
  valid <- abs(as.integer(format(d[[v]], "%Y")) - d$fiscal_year) <= 1L
  set(d, which(!is.na(valid) & !valid), v, as.IDate(NA))
}
# Disclosure records sometimes repeat a case. Collapse identical parsed records,
# and exclude conflicting versions for the applicable diagnostic.
u <- unique(d[, .(fiscal_year, case_number, receipt, start)])
u <- u[!duplicated(u$case_number) & !duplicated(u$case_number, fromLast = TRUE)]
u[, lead_days := as.integer(start - receipt)]
cat("Administrative receipt to requested start, days, all statuses and unique unambiguous cases\n")
print(u[, .(cases = .N, dated = sum(!is.na(lead_days)), negative = sum(lead_days < 0, na.rm = TRUE),
  median = as.numeric(median(lead_days, na.rm = TRUE)), share_45_75 = mean(between(lead_days,45,75), na.rm = TRUE),
  share_65_or_fewer = mean(lead_days <= 65, na.rm = TRUE),
  share_40_or_more = mean(lead_days >= 40, na.rm = TRUE)), by = fiscal_year][order(fiscal_year)])
eligible <- u[!is.na(lead_days)]
print(eligible[, .(cases = .N, q10 = quantile(lead_days,.1), median = median(lead_days),
  q90 = quantile(lead_days,.9), share_45_75 = mean(between(lead_days,45,75)),
  early_all_before_receipt = mean(lead_days < 65),
  late_all_after_receipt = mean(lead_days > 40)), by = .(recent = fiscal_year >= 2020L)])
geos <- unique(as.data.table(arrow::read_parquet(path_processed("county_year_panel.parquet"),
            col_select = c("county_fips", "state_fips"))))
assert_geo_columns(geos,c("county_fips","state_fips"))
stopifnot(!anyDuplicated(geos$county_fips))
d <- unique(d[, .(fiscal_year, case_number, receipt, start, county_fips = county_fips_list, requested)])
d <- d[!duplicated(d$case_number) & !duplicated(d$case_number, fromLast = TRUE)]
# One known county on the primary record, not a claim about the entire itinerary.
with_place <- merge(d, geos, by = "county_fips")
cat("Recent geographic and start-date coverage\n")
print(d[fiscal_year %in% 2020:2025, .(cases = .N, located = sum(county_fips %in% geos$county_fips),
                                    dated = sum(!is.na(start))), by = fiscal_year][order(fiscal_year)])
with_place[, start_y := as.integer(format(start, "%Y"))]
with_place[, start_m := as.integer(format(start, "%m"))]
with_place[, start_d := as.integer(format(start, "%d"))]
pre <- with_place[fiscal_year %in% 2009:2011 & start_y %in% 2009:2011 & !is.na(receipt) & receipt <= "2011-12-31"]
support <- pre[, .(cases = .N, years = uniqueN(start_y)), by = county_fips][years == 3L & cases >= 6L]
month <- pre[county_fips %in% support$county_fips, .(cases = .N, day = round(median(start_d))), by = .(county_fips, start_m)]
month <- month[, .SD[cases == max(cases)], by = county_fips]
month <- month[!duplicated(county_fips) & !duplicated(county_fips, fromLast = TRUE)]
# A constant nonleap anchor admits all years.
month[start_m == 2L & day == 29L, day := 28L]
stopifnot(nrow(month) > 0L, !anyDuplicated(month$county_fips))
years <- c(1982:2007, 2012:2025)
targets <- month[, .(year = years), by = .(county_fips,start_m,day)]
targets[, anchor := as.IDate(sprintf("%04d-%02d-%02d", year, start_m, day))]
stopifnot(!anyNA(targets$anchor))
wanted <- targets[, .(lag = c(65:110,5:40)), by = .(county_fips,year,anchor)]
wanted[, date := anchor - lag]
wanted[, ym := format(date, "%Y%m")]
wanted[, band := fifelse(lag >= 65L, "early", "late")]
# Raw monthly partitions contain daily Celsius values and padded invalid dates.
wx <- rbindlist(lapply(split(wanted, wanted$ym), function(q) {
  f <- path_raw("epinoaa_nclimgrid", paste0(q$ym[1L], ".parquet"))
  chunk <- as.data.table(arrow::read_parquet(f, col_select = c("region_type","fips","date","tmin")))
  chunk <- chunk[region_type == "cty" & !is.na(date) & fips %in% q$county_fips,
                 .(county_fips = fips, date = as.IDate(date, tz = "UTC"), tmin = suppressWarnings(as.numeric(tmin)))]
  stopifnot(!anyDuplicated(chunk[, .(county_fips,date)]))
  chunk[tmin <= -999, tmin := NA_real_]
  merge(q, chunk, by = c("county_fips","date"), all.x = TRUE)
}))
windows <- wx[, .(complete = !anyNA(tmin), freeze = any(tmin <= 0),
     first_freeze = if (any(tmin <= 0, na.rm = TRUE)) min(date[tmin <= 0 & is.finite(tmin)]) else as.IDate(NA)),
  by = .(county_fips,year,band)]
normals <- windows[year <= 2007 & complete == TRUE, .(obs = .N, climatology = mean(freeze)), by = .(county_fips,band)][obs == 26L]
history_test <- merge(windows[year >= 2012 & complete == TRUE], normals, by = c("county_fips","band"))
history_test <- merge(history_test, geos, by = "county_fips")
cat("Full outcome-free 2012-25 exposure count, independent of reporting-era request support\n")
print(history_test[, .(county_years = .N, counties = uniqueN(county_fips), any_freeze = sum(freeze),
  rare_support = sum(climatology <= .2), rare_events = sum(climatology <= .2 & freeze),
  rare_states = uniqueN(state_fips[climatology <= .2 & freeze]),
  rare_state_years = uniqueN(paste(state_fips, year)[climatology <= .2 & freeze])), by = band])
test <- history_test[year %in% 2021:2024]
cat("Historical start calendar and observed freeze support\n")
cat("Eligible after history screen:", nrow(support), "unique modal start month:", nrow(month), "\n")
print(test[, .(county_years = .N, counties = uniqueN(county_fips), any_freeze = sum(freeze),
  rare_support = sum(climatology <= .2), rare_events = sum(climatology <= .2 & freeze)), by = band])
tp <- dcast(test, county_fips + year ~ band, value.var = c("freeze", "first_freeze", "climatology"))
tp <- merge(tp, targets[, .(county_fips, year, anchor)], by=c("county_fips","year"))
tp <- tp[!is.na(freeze_early) & !is.na(freeze_late)]
tp <- merge(tp, geos, by="county_fips")
# Look up all primary records starting within 30 days of the predetermined anchor,
# including starts in the adjoining calendar year for January/December anchors.
cc <- with_place[start_y %in% 2020:2025, .(county_fips, case_number, start, receipt, requested)]
linked <- merge(cc, tp, by = "county_fips", allow.cartesian = TRUE)
linked <- linked[abs(as.integer(start - anchor)) <= 30L]
stopifnot(!anyDuplicated(linked$case_number))
cat("Case receipt before the first observed freeze in eligible windows with at least one freeze\n")
for (b in c("early","late")) {
  date_col <- paste0("first_freeze_", b)
  print(linked[!is.na(receipt) & !is.na(get(date_col)), .(band = b, cases = .N,
             share_received_before = mean(receipt < get(date_col)),
             share_received_same_day = mean(receipt == get(date_col)))])
}
qty <- linked[, .(cases = .N, req = sum(requested)), by=.(county_fips, year)]
tp <- merge(tp, qty, by = c("county_fips","year"), all.x = TRUE)
tp[is.na(cases), c("cases","req") := list(0L,0)]
stopifnot(!anyDuplicated(tp[,.(county_fips, year)]), !anyNA(tp$req), all(tp$req >= 0))
# Support-only quantity comparison: a freeze in a window with historical probability
# <=20%, controlling for county and state-year. Weather can affect labor directly.
tp[, rare_early := as.integer(climatology_early <= .2 & freeze_early)]
tp[, rare_late := as.integer(climatology_late <= .2 & freeze_late)]
cat("Calendar-window filing counts and requested-position contrasts\n")
print(tp[, .(rows = .N, counties = uniqueN(county_fips), filing_cells = sum(cases > 0),
    cases = sum(cases), requests = sum(req), early_states = uniqueN(state_fips[rare_early == 1]),
    late_states = uniqueN(state_fips[rare_late == 1]), both = sum(rare_early * rare_late))])
for (outcome in c("cases","req")) {
  formula <- as.formula(paste0("log1p(",outcome,") ~ rare_early + rare_late | county_fips + state_fips^year"))
  fit <- fixest::feols(formula, tp, vcov = ~state_fips, notes = FALSE, nthreads=1L)
  cat(outcome, "estimation rows", nobs(fit), "\n")
  print(cbind(estimate = coef(fit), confint(fit)))
}
