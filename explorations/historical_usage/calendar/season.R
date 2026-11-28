# Historical certified capacity over the calendar, using only FY2008--FY2025.
source("code/paths.R")
source(path_code("c00_shared", "geography.R"))
library(data.table)
setDTthreads(2L)
destination <- commandArgs(trailingOnly = TRUE)
stopifnot(length(destination) == 1L)
dir.create(destination, recursive = TRUE, showWarnings = FALSE)
write_result <- function(x, name) {
  stopifnot(nrow(x) > 0L)
  fwrite(x, file.path(destination, paste0(name, ".csv")))
}
years <- 2008:2025
part <- function(yr) {
  cols <- c("fiscal_year", "state_fips", "month", "date_basis", "allocated_certification_position_days")
  x <- as.data.table(arrow::read_parquet(
    path_int("h2a_worksite_months", sprintf("fy_%s.parquet", yr)), col_select = tidyselect::all_of(cols)
  ))
  assert_geo_columns(x, "state_fips", allow_na = "state_fips")
  stopifnot(all(x$fiscal_year == yr), all(is.finite(x$allocated_certification_position_days)),
            all(x$allocated_certification_position_days >= 0), !anyNA(x$month))
  x[, .(pd = sum(allocated_certification_position_days)), by = .(fiscal_year, state_fips, month, date_basis)]
}
months <- rbindlist(lapply(years, part))
coverage <- as.data.table(arrow::read_parquet(path_int("h2a_worksite_exposure_coverage.parquet")))
check <- merge(months[, .(pd = sum(pd), unmatched = sum(pd[is.na(state_fips)])), by = fiscal_year],
               coverage[fiscal_year %in% years], by = "fiscal_year")
stopifnot(nrow(check) == length(years), max(abs(check$pd - check$position_days)) < 0.01,
          max(abs(check$unmatched - check$unmatched_county_position_days)) < 0.01)
months[, `:=`(year = year(month), m = month(month))]
months <- months[year %in% years]
months[, winter := m %in% c(12L, 1L, 2L)]
shape <- function(x) x[, .(position_days = sum(pd), winter_share = sum(pd[winter])/sum(pd),
                            summer_share = sum(pd[m %in% 6:8])/sum(pd)), by = year][order(year)]
annual <- shape(months)
coverage_year <- months[, .(site_share = sum(pd[date_basis == "observed_worksite_requested_dates"])/sum(pd),
                             unmatched_share = sum(pd[is.na(state_fips)])/sum(pd),
                             lag_fy_share = sum(pd[fiscal_year < year])/sum(pd),
                             lead_fy_share = sum(pd[fiscal_year > year])/sum(pd)), by = year]
write_result(merge(annual, coverage_year, by = "year"), "annual_calendar")

# For a national measure per calendar day, divide each month's capacity by its
# own length. Pooling years uses summed days so February and leap years match.
month_grid <- data.table(month = seq(as.Date("2008-01-01"), as.Date("2025-12-01"), by = "month"))
month_grid[, days := as.integer(shift(month, type = "lead", fill = as.Date("2026-01-01")) - month)]
month_grid[, `:=`(year = year(month), m = month(month))]
periods <- list(early = 2010:2012, before_new_forms = 2017:2019, after_new_forms = 2020:2022, late = 2022:2024)
windows <- list()
state_profiles <- list()
for (label in names(periods)) {
  sel <- months[year %in% periods[[label]] & !is.na(state_fips)]
  cal <- month_grid[year %in% periods[[label]], .(days = sum(days)), by = m]
  s <- sel[, .(pd = sum(pd)), by = .(state_fips, m)]
  # Every state with measured capacity contributes all twelve months, including zeros.
  s <- merge(CJ(state_fips = unique(s$state_fips), m = 1:12), s, by = c("state_fips", "m"), all.x = TRUE)
  s[is.na(pd), pd := 0]
  s <- merge(s, cal, by = "m")[order(state_fips, m)]
  s[, capacity := pd / days]
  by_state <- s[, .(pd = sum(pd), winter_share = sum(pd[m %in% c(12L, 1L, 2L)])/sum(pd),
                    local_peak = max(capacity)), by = state_fips][pd > 0]
  by_state[, weight := pd / sum(pd)]
  national <- s[, .(capacity = sum(capacity), pd = sum(pd)), by = m][order(m)]
  windows[[label]] <- national[, .(period = label, winter_share = sum(pd[m %in% c(12L, 1L, 2L)])/sum(pd),
                                    peak_month = m[which.max(capacity)], peak_capacity = max(capacity),
                                    winter_to_summer_daily = (sum(pd[m %in% c(12L, 1L, 2L)]) /
                                      sum(cal$days[cal$m %in% c(12L, 1L, 2L)])) /
                                      (sum(pd[m %in% 6:8]) / sum(cal$days[cal$m %in% 6:8])),
                                    sum_local_peaks = sum(by_state$local_peak),
                                    peak_coincidence = max(capacity)/sum(by_state$local_peak))]
  state_profiles[[label]] <- by_state
}
write_result(rbindlist(windows), "seasonal_windows")

# Symmetric decomposition, with unmatched states excluded and entering/exiting
# states shown separately. The four contributions add to the observed change.
decompose <- function(a, b) {
  x <- merge(state_profiles[[a]], state_profiles[[b]], by = "state_fips",
             all = TRUE, suffixes = c("_a", "_b"))
  x[, `:=`(
    within = fifelse(!is.na(weight_a + weight_b), (weight_a + weight_b)/2 * (winter_share_b - winter_share_a), 0),
    mix = fifelse(!is.na(weight_a + weight_b), (winter_share_a + winter_share_b)/2 * (weight_b - weight_a), 0),
    appearing = fifelse(is.na(weight_a), weight_b * winter_share_b, 0),
    disappearing = fifelse(is.na(weight_b), -weight_a * winter_share_a, 0)
  )]
  delta <- windows[[b]]$winter_share - windows[[a]]$winter_share
  stopifnot(abs(sum(x$within + x$mix + x$appearing + x$disappearing) - delta) < 1e-10)
  x[, .(from = a, to = b, change = delta, within = sum(within), mix = sum(mix),
        appearance = sum(appearing), disappearance = sum(disappearing))]
}
write_result(rbindlist(list(decompose("early", "before_new_forms"), decompose("before_new_forms", "after_new_forms"),
                           decompose("early", "late"))), "winter_decomposition")

# Compare source-selected and uniform case dates on exactly the same allocation
# rows, then check how much mass is lost selecting that common support.
cols <- c("fiscal_year", "adjusted_nbr_workers_certified", "case_allocation_reconciled",
          "valid_case_interval", "valid_site_interval", "exposure_begin", "exposure_end",
          "case_begin", "case_end")
audit <- as.data.table(arrow::read_parquet(path_int("h2a_worksite_exposure_audit.parquet"), col_select = tidyselect::all_of(cols)))
audit <- audit[fiscal_year %in% years & case_allocation_reconciled & adjusted_nbr_workers_certified > 0]
setnames(audit, "adjusted_nbr_workers_certified", "positions")
cohort <- audit[, .(positions = sum(positions),
                    timed_positions = sum(positions[!is.na(exposure_begin)]),
                    pd = sum(positions * as.numeric(exposure_end - exposure_begin + 1), na.rm = TRUE),
                    common_positions = sum(positions[valid_case_interval]),
                    common_case_pd = sum(positions[valid_case_interval] *
                      as.numeric(case_end[valid_case_interval] - case_begin[valid_case_interval] + 1))),
                by = fiscal_year][order(fiscal_year)]
cohort[, `:=`(timed_share = timed_positions/positions, average_days = pd/timed_positions,
             common_case_days = common_case_pd/common_positions)]
stopifnot(max(abs(cohort$pd - check$pd[match(cohort$fiscal_year, check$fiscal_year)])) < 0.01)
write_result(cohort, "filing_cohorts")
calendar_daily <- function(x, first, last) {
  starts <- x[, .(delta = sum(positions)), by = .(day = get(first))]
  ends <- x[, .(delta = -sum(positions)), by = .(day = get(last) + 1L)]
  days <- rbindlist(list(starts, ends))[, .(delta = sum(delta)), by = day]
  days <- merge(data.table(day = seq(min(days$day), max(days$day), by = "day")), days, by = "day", all.x = TRUE)
  days[is.na(delta), delta := 0]
  days[, capacity := cumsum(delta)]
  stopifnot(abs(tail(days$capacity, 1)) < 0.001, min(days$capacity) > -0.001)
  daily <- days[year(day) %in% years,
                .(pd = sum(capacity)), by = .(year = year(day), m = month(day))]
  daily[, winter := m %in% c(12L, 1L, 2L)]
  shape(daily)
}
all_rows <- calendar_daily(audit[!is.na(exposure_begin)], "exposure_begin", "exposure_end")
stopifnot(max(abs(annual$position_days - all_rows$position_days)) < 0.01)
common <- audit[valid_case_interval == TRUE]
selected <- calendar_daily(common, "exposure_begin", "exposure_end")
case_only <- calendar_daily(common, "case_begin", "case_end")
sensitivity <- merge(selected, case_only, by = "year", suffixes = c("_selected", "_case"))
sensitivity[, share_national_pd_covered := position_days_selected / all_rows$position_days[match(year, all_rows$year)]]
write_result(sensitivity, "common_case_sensitivity")
