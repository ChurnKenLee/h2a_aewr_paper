# Descriptive feasibility screen of an ACA contract-duration hypothesis.
# IRS rules: https://www.irs.gov/irb/2014-09_IRB (V.C, VII.C, XV.D.3/6).
# Contracted days and H-2A requests are not hours of service or all-firm employment.
source("code/paths.R")
library(data.table)
library(lubridate)
setDTthreads(2L)
out <- commandArgs(trailingOnly = TRUE)[1L]
stopifnot(!is.na(out))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
stopifnot(startsWith(normalizePath(out), "/tmp/"))
emit <- function(x, label) fwrite(x, file.path(out, paste0("aca_", label, ".csv")))
read_fields <- function(path, columns) as.data.table(arrow::read_parquet(path, col_select = tidyselect::all_of(columns)))
fields <- c("case_number", "fiscal_year", "period_begin_date", "period_end_date", "period_date_basis",
            "primary_sub", "nbr_workers_requested", "nbr_workers_needed", "nbr_workers_certified")
raw <- read_fields(path_int("h2a_disclosure_records.parquet"), fields)[fiscal_year %between% c(2011L, 2016L)]
raw <- raw[nzchar(trimws(case_number))]
raw[, `:=`(request = suppressWarnings(as.numeric(nbr_workers_requested)),
           total_needed = suppressWarnings(as.numeric(nbr_workers_needed)))]
# Canonical decision-FY case totals, avoiding repeated disclosure/subemployer rows.
audit <- read_fields(path_int("h2a_case_allocation_audit.parquet"),
                     c("case_number", "fiscal_year", "case_allocation_reconciled", "case_certified_target"))
audit <- audit[fiscal_year %between% c(2011L, 2016L) & case_allocation_reconciled == TRUE & case_certified_target > 0]
stopifnot(!anyDuplicated(audit, by = c("case_number", "fiscal_year")))
# Require all distinct primary-or-unmarked rows to agree on a valid complete
# interval. A positive request can be summarized only when all retained rows agree.
head <- raw[is.na(primary_sub) | toupper(trimws(primary_sub)) != "SUB"]
head[, `:=`(begin = as.Date(period_begin_date), end = as.Date(period_end_date))]
varies <- function(x) uniqueN(x) != 1L || anyNA(x)
cases <- head[, .(start = if (varies(begin) || varies(end)) as.Date(NA) else begin[1L],
                  last = if (varies(begin) || varies(end)) as.Date(NA) else end[1L],
                  date_basis = if (varies(period_date_basis)) NA_character_ else period_date_basis[1L],
                  request = if (varies(request)) NA_real_ else request[1L],
                  any_needed = any(is.finite(total_needed))),
              by = .(fiscal_year, case_number)]
cases <- merge(audit, cases, by = c("case_number", "fiscal_year"), all.x = TRUE)
cases[, valid := !is.na(start) & !is.na(last) & last >= start &
                   abs(lubridate::year(start)-fiscal_year) <= 1L & abs(lubridate::year(last)-fiscal_year) <= 1L]
cases[, known_request := is.finite(request) & request >= 1 & request == floor(request)]
emit(cases[, .(positive_reconciled_cases=.N, valid_intervals=sum(valid),
               known_requests=sum(known_request), positive_total_need=sum(any_needed == TRUE, na.rm=TRUE)),
           by=fiscal_year][order(fiscal_year)], "coverage")
x <- cases[valid == TRUE]
x[, days := as.integer(last - start) + 1L]
# Calendar convention for this descriptive comparison: an inclusive six-month
# season ends the day before the six-month anniversary (rollback at month end).
x[, cal_end := (start %m+% months(6L)) - 1L]
x[, `:=`(offset_cal = as.integer(last - cal_end), offset_180 = days - 180L)]
emit(x[, .(cases = .N, days_exact_180 = sum(offset_180 == 0), calendar_exact_six = sum(offset_cal == 0),
           within_180 = sum(offset_180 <= 0), within_six = sum(offset_cal <= 0),
           disagree = sum((offset_cal <= 0) != (offset_180 <= 0)), in_170_190 = sum(days %between% c(170,190)),
           end_month_last = sum(lubridate::day(last + 1L) == 1L)),
       by = fiscal_year][order(fiscal_year)], "duration")
emit(x[, .(cases = .N), by = .(fiscal_year, date_basis)], "date_basis")
emit(rbindlist(lapply(c(120L, 150L, 180L, 210L, 240L), function(center) {
  z <- x[abs(days - center) <= 7]
  z[, .(center, near = .N, exactly = sum(days == center), below = sum(days < center), above = sum(days > center)),
    by = fiscal_year]
})), "nearby_day_counts")
ids <- read_fields(path_int("policy_2025", "filer", "filer_ids.parquet"), c("fiscal_year", "case_number", "filer_R7"))
stopifnot(!anyDuplicated(ids, by=c("fiscal_year","case_number")))
linked <- merge(cases, ids, by=c("fiscal_year","case_number"), all.x=TRUE)
stopifnot(nrow(linked) == nrow(cases))
# A pre-rule descriptive support screen: positive certified, known single-case
# requests in FY2011--12. Requests are conditional on certification. Financial
# relationships, other employees, shared employees and hours of service are unobserved.
pre <- linked[fiscal_year %in% 2011:2012 & known_request & !is.na(filer_R7),
              .(max_request = max(request), active_baseline_years = uniqueN(fiscal_year)),
              by = filer_R7]
late_ids <- unique(linked[fiscal_year %in% 2015:2016 & !is.na(filer_R7), filer_R7])
pre[, band := fcase(max_request < 50, "under_50", max_request < 100, "50_to_99",
                    default = "100_plus")]
pre[, observed_later := filer_R7 %in% late_ids]
emit(pre[, .(filers=.N, observed_positive_FY2015_16=sum(observed_later),
             baseline_both_years=sum(active_baseline_years==2L)), by=band], "baseline_support")
# Full microdata needed for the ACA assignment are absent. Report counts, without
# implying an observable treatment or that absence of an H-2A case is nonemployment.
stopifnot(all(x$days >= 1), all(x$offset_180 - x$offset_cal >= 0L),
          all(pre$max_request >= 1), nrow(x) > 0, sum(x$offset_cal == 0) > 0)
