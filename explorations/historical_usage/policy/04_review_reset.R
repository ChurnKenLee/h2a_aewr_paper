# Receipt and decision timing on the same H-2A cases as wage_resets.R.
# Run from the repository root after configure_r_library.
source("code/paths.R")
library(data.table)
out <- commandArgs(trailingOnly = TRUE)[1L]
stopifnot(!is.na(out))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
stopifnot(startsWith(normalizePath(out), "/tmp/"))
# Reproduce the original sample and cross-check its aggregates before adding
# timing and coverage. The source writes its historical outputs in scratch.
source(path_root("explorations", "historical_usage", "policy", "wage_resets.R"))
events <- copy(joined)
stopifnot(sum(summary$cases) == nrow(events),
          !anyDuplicated(events[, .(fiscal_year, case_number, reference_year)]))
events[, decision_day := as.integer(decision - as.IDate(effective_start))]
events[, direction := fifelse(aewr > prior, "rise", "fall")]
notices <- fread(path_code("a01_sources", "aewr_rate_schedule_manifest.csv"),
                 select = c("year", "publication_date"))
setnames(notices, "year", "reference_year")
stopifnot(!anyDuplicated(notices$reference_year))
events <- merge(events, notices, by = "reference_year", all.x = TRUE)
stopifnot(!anyNA(events$publication_date), all(events$decision_day >= events$day),
          all(events$publication_date <= events$effective_start))
emit <- function(x, name) fwrite(x, file.path(out, paste0("review_reset_", name, ".csv")))
bins <- function(k) {
  fcase(k < -112L, "A -168 through -113", k < -56L, "B -112 through -57",
        k < 0L, "C -56 through -1", default = "D 0 through +55")
}
# All common-frame summaries use the same cases on both clocks. Near-frame
# status requires BOTH dates inside the adjacent 56-day windows.
common_events <- events[decision_day < 56L]
common_events[, receipt_bin := bins(day)]
common_events[, decision_bin := bins(decision_day)]
stopifnot(nrow(common_events) > 0, all(common_events$decision_day >= -168L))
stats <- function(z, by) {
  z[, .(cases = .N, states = uniqueN(st), updates = uniqueN(reference_year),
        state_updates = uniqueN(paste(st, reference_year)),
        new_share = mean(new_hit), old_share = mean(old_hit), near_share = mean(nearby_hit),
        position_weighted_new = weighted.mean(new_hit, positions),
        median_processing_days = as.numeric(median(decision_day - day))), by = by]
}
profile <- rbindlist(lapply(c("receipt_bin", "decision_bin"), function(clock) {
  s <- stats(common_events, clock)
  setnames(s, clock, "window")
  s[, timing := clock]
  s
}))
emit(profile, "common_profiles")
near <- common_events[day >= -56L]
near[, timing := fcase(decision_day < 0L, "1 both before", day < 0L, "2 crosses", default = "3 both after")]
stopifnot(nrow(near) > 0, uniqueN(near$timing) == 3L,
          !any(near$timing == "2 crosses" & (near$day >= 0L | near$decision_day < 0L)))
emit(stats(near, "timing"), "case_status")
emit(stats(near, c("direction", "timing")), "by_direction")
# Compare within exactly the same recurring filer, state, occupation and update
# with cases in ALL three timing groups. Changes can still reflect unobserved jobs.
near_id <- merge(near, ids, by = c("case_number", "fiscal_year"), all.x = TRUE)
stopifnot(nrow(near_id) == nrow(near))
m <- near_id[!is.na(filer_R7), .(n = .N, new = mean(new_hit), old = mean(old_hit)),
             by = .(filer_R7, st, soc, reference_year, direction, timing)]
m <- m[, if (.N == 3L) .SD, by = .(filer_R7, st, soc, reference_year)]
summarize_cells <- function(dt, group) {
  dt[, .(cells = .N, cases = sum(n), filers = uniqueN(filer_R7), updates = uniqueN(reference_year),
         mean_new = mean(new), mean_old = mean(old)), by = group]
}
emit(summarize_cells(m, "timing"), "repeat_cells")
emit(summarize_cells(m, c("direction", "timing")), "repeat_cells_direction")
# Before the official notice is a diagnostic clock only. FLS inputs or forecasts
# can be public before the Federal Register notice; the disclosure stores one wage.
cross <- near[timing == "2 crosses"]
cross[, publication_timing := fifelse(received < publication_date, "pre_notice_receipt", "on_after_notice_receipt")]
emit(stats(cross, c("direction", "publication_timing")), "crossers_notice")

# July annual agricultural minimum benchmarks in both adjacent calendar years.
# This conservative screen omits city rates, prevailing/crop/task rates, union and
# offered-contract wages, legal coverage exceptions and substate effective dates.
states <- unique(rates[, .(st, state_fips)])
stopifnot(!anyDuplicated(states$st))
minimum <- as.data.table(arrow::read_parquet(path_int("state_real_minwages.parquet"),
                 col_select = all_of(c("state_fips", "year", "prevailing_ag_min_wage", "ag_coverage_backcast"))))
stopifnot(!anyDuplicated(minimum, by = c("state_fips", "year")))
minimum <- merge(minimum, states, by = "state_fips")
minimum <- merge(minimum, minimum[, .(st, year = year + 1L, ag_last = prevailing_ag_min_wage)],
                 by = c("st", "year"), all.x = TRUE)
setnames(minimum, "year", "reference_year")
screen <- merge(near, minimum[, .(st, reference_year, ag_last, prevailing_ag_min_wage, ag_coverage_backcast)],
                by = c("st", "reference_year"), all.x = TRUE)
stopifnot(nrow(screen) == nrow(near), all(!screen$ag_coverage_backcast, na.rm = TRUE))
screen[, room := pmin(aewr, prior) - pmax(ag_last, prevailing_ag_min_wage)]
screen[, ag_screen := fcase(is.na(room), "annual_minimum_missing", room <= 0, "at_or_below_observed_minimum",
                          room < 1, "positive_under_one_dollar", default = "one_dollar_or_more")]
emit(stats(screen, c("direction", "timing", "ag_screen")), "minimum_screen")
emit(screen[, .(cases = .N, range_low = min(room), range_high = max(room),
                state_updates = uniqueN(paste(st, reference_year))), by = .(direction, ag_screen)],
     "minimum_support")
# Rerun the original receipt-only post-window percentages under this screen.
post <- merge(events[window == "post"],
              minimum[, .(st, reference_year, ag_last, prevailing_ag_min_wage)],
              by = c("st", "reference_year"), all.x = TRUE)
stopifnot(nrow(post) == sum(events$window == "post"))
post[, room := pmin(aewr, prior) - pmax(ag_last, prevailing_ag_min_wage)]
emit(rbindlist(lapply(c(0, 1, 2), function(dollars) {
  z <- if (dollars == 0) post else post[room >= dollars]
  s <- stats(z, "direction")
  s[, min_gap := if (dollars == 0) "unscreened" else paste0("at_least_", dollars)]
  s
})), "falling_rate_original_frame")
emit(data.table(frame = c("original_receipt_window", "both_clocks_224_days", "both_clocks_112_days"),
                cases = c(nrow(events), nrow(common_events), nrow(near))), "clock_coverage")
