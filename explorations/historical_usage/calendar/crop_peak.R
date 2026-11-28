# Align the H-2A calendar with crop-sector peak employment measured before 2008.
source("code/paths.R")
source(path_code("c00_shared", "geography.R"))
library(data.table)
setDTthreads(2L)
out <- commandArgs(trailingOnly = TRUE)
stopifnot(length(out) == 1L)
dir.create(out, recursive = TRUE, showWarnings = FALSE)

qcew <- as.data.table(arrow::read_parquet(path_int("qcew_county_ag_quarterly_employment.parquet")))
assert_geo_columns(qcew, "county_fips")
qcew <- qcew[industry_code == "111" & year < 2008 & qcew_employment_disclosed == TRUE &
               is.finite(qcew_reference_month_emplvl)]
stopifnot(!anyDuplicated(qcew, by = c("county_fips", "year", "qtr")))
# Restrict to years with all four disclosed quarters, so suppressed or absent
# quarters cannot count as troughs. Pool within-year shares with equal year weights.
qcew <- qcew[, if (.N == 4L && sum(qcew_reference_month_emplvl) > 0) {
  .(qtr, share = qcew_reference_month_emplvl/sum(qcew_reference_month_emplvl),
    level = qcew_reference_month_emplvl)
}, by = .(county_fips, year)]
peak <- function(first, last, min_years) {
  means <- qcew[year %between% c(first, last), .(share = mean(share), emp = mean(level),
                                                 n = .N), by = .(county_fips, qtr)][n >= min_years]
  means[, if (sum(abs(share - max(share)) < 1e-10) == 1L)
    .(peak = qtr[which.max(share)], mean_emp = mean(emp)), by = county_fips]
}
frame <- peak(2003, 2007, 3)
prior <- peak(2000, 2002, 2)
agreement <- merge(frame, prior[, .(county_fips, earlier = peak)], by = "county_fips", all.x = TRUE)
agreement[, stable := !is.na(earlier) & earlier == peak]
agreement[, state_fips := substr(county_fips, 1L, 2L)]
assert_geo_columns(agreement, c("county_fips", "state_fips"))
stopifnot(nrow(agreement) > 0, !anyDuplicated(agreement$county_fips))

get_months <- function(fy) {
  cols <- c("county_fips", "month", "segment_begin", "segment_end",
            "allocated_positions", "allocated_certification_position_days")
  x <- as.data.table(arrow::read_parquet(path_int("h2a_worksite_months", sprintf("fy_%s.parquet", fy)),
                                          col_select = tidyselect::all_of(cols)))
  baseline <- x[year(month) %between% c(2009, 2011) & !is.na(county_fips),
                .(days = sum(allocated_certification_position_days)), by = county_fips]
  x <- x[month(month) %in% c(1L, 4L, 7L, 10L) & year(month) %between% c(2009, 2024)]
  # Keep unmatched counties for coverage accounting.
  ref <- x[, .(days = sum(allocated_certification_position_days),
        midmonth = sum(allocated_positions[segment_begin <= month + 11 & segment_end >= month + 11])),
    by = .(county_fips, month)]
  list(ref = ref, baseline = baseline)
}
parts <- lapply(2008:2025, get_months)
monthly <- rbindlist(lapply(parts, "[[", "ref"))
baseline <- rbindlist(lapply(parts, "[[", "baseline"))[, .(days = sum(days)), by = county_fips]
agreement[, no_early_capacity := !(county_fips %in% baseline[days > 0, county_fips])]
rm(parts)
monthly[, `:=`(year = year(month), qtr = (month(month) - 1L) %/% 3L + 1L)]
monthly[, capacity := days / as.integer(as.Date(format(month + 32, "%Y-%m-01")) - month)]
periods <- list(early = 2010:2012, middle = 2017:2019, late = 2022:2024)
assess <- function(sample, period, profile, measure) {
  counts <- monthly[year %in% periods[[period]], .(amount = sum(get(measure))), by = .(county_fips, qtr)]
  full_total <- sum(counts$amount)
  sample <- copy(sample)
  if (profile == "stable") sample <- sample[stable == TRUE]
  if (profile == "zero_2009_11") sample <- sample[no_early_capacity == TRUE]
  # Matched cells include all eligible counties before outcomes, even zero capacity.
  cells <- merge(sample[, .(qtr = 1:4), by = .(county_fips, state_fips, stable, peak, mean_emp)],
                 counts, by = c("county_fips", "qtr"), all.x = TRUE)
  cells[is.na(amount), amount := 0]
  cells[, total := sum(amount), by = county_fips]
  cells[, class := as.integer(cut(frank(mean_emp, ties.method = "average")/.N,
                                  breaks = c(0, 1/3, 2/3, 1), include.lowest = TRUE)), by = state_fips]
  # Exact expected shares if the historical peak label is permuted in a state or
  # state x pre-period-size tercile. This is a descriptive geographic benchmark.
  cells[, state_freq := mean(peak == qtr), by = .(state_fips, qtr)]
  cells[, size_freq := mean(peak == qtr), by = .(state_fips, class, qtr)]
  positive <- cells[total > 0]
  result <- rbindlist(lapply(c("capacity", "active_county"), function(w) {
    z <- copy(positive)
    z[, mass := if (w == "capacity") amount else amount/total]
    z[, .(period, profile, calendar_measure = measure, weight = w,
          eligible = nrow(sample), active = uniqueN(county_fips), states = uniqueN(state_fips),
          coverage = sum(amount)/full_total, observed = sum(mass*(peak == qtr))/sum(mass),
          state_placebo = sum(mass*state_freq)/sum(mass), state_size_placebo = sum(mass*size_freq)/sum(mass))]
  }))
  vals <- positive[, .(actual = sum(amount*(peak == qtr)), expected = sum(amount*size_freq),
                       mass = sum(amount)), by = state_fips]
  delta_loo <- (sum(vals$actual - vals$expected) - (vals$actual - vals$expected))/(sum(vals$mass) - vals$mass)
  result[weight == "capacity", `:=`(min_state_deletion_gap = min(delta_loo), max_state_deletion_gap = max(delta_loo))]
  result
}
settings <- CJ(period = names(periods), profile = c("all", "stable", "zero_2009_11"), measure = c("capacity", "midmonth"))
settings <- settings[!(profile == "zero_2009_11" & period == "early")]
result <- rbindlist(lapply(seq_len(nrow(settings)), function(i) {
  assess(agreement, settings$period[i], settings$profile[i], settings$measure[i])
}))
stopifnot(nrow(result) == 32L, all(result$active <= result$eligible),
          all(result$coverage <= 1 + 1e-10), all(result$coverage > 0))
fwrite(result, file.path(out, "crop_peak_alignment.csv"))
fwrite(agreement[, .(counties = .N, with_previous = sum(!is.na(earlier)), stable = sum(stable))],
       file.path(out, "crop_peak_support.csv"))
