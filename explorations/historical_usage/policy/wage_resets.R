# How quickly do offers switch from the old regional reference to the new one?
source("code/paths.R")
library(data.table)
library(arrow)
setDTthreads(2L)

out <- commandArgs(trailingOnly = TRUE)[1L]
stopifnot(!is.na(out))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
fields <- c("fiscal_year", "case_number", "worksite_state", "case_received_date", "primary_sub",
            "decision_date", "soc_code", "wage_rate", "wage_unit", "nbr_workers_certified")
x <- as.data.table(read_parquet(path_int("h2a_disclosure_records.parquet"), col_select = all_of(fields)))
x <- x[fiscal_year %between% c(2014L, 2025L) & toupper(primary_sub) != "SUB"]
x[, received := suppressWarnings(as.IDate(substr(case_received_date,1,10)))]
x[, decision := suppressWarnings(as.IDate(substr(decision_date,1,10)))]
x[, wage := suppressWarnings(as.numeric(wage_rate))]
x[, positions := suppressWarnings(as.numeric(nbr_workers_certified))]
x[, soc := substr(soc_code, 1, 7)]
# Remove all ambiguous cases before choosing wages, job type or outcome.
x <- unique(x[, .(case_number, fiscal_year, worksite_state, received, decision, wage,
                  wage_unit, soc, positions)])
x <- x[nzchar(case_number) & !duplicated(case_number) & !duplicated(case_number, fromLast = TRUE)]
common_soc <- c("45-2041","45-2091","45-2092","45-2093","45-2099","53-7064")
x <- x[positions > 0 & wage > 0 & wage <= 100 & toupper(wage_unit) %chin% c("HOUR","HR") &
       soc %chin% common_soc & decision >= received & as.integer(decision - received) <= 180]
# Keep only cases where every allocated worksite has the same state abbreviation
# as the disclosure, after translating either full names or two-letter labels.
sites <- as.data.table(read_parquet(path_int("h2a_allocated_worksites.parquet"),
                                     col_select = c("case_number","worksite_state","fiscal_year")))
sites <- sites[fiscal_year %between% c(2014L, 2025L)]
lookup <- c(setNames(state.abb, toupper(state.name)), setNames(state.abb, state.abb))
sites[, st := unname(lookup[toupper(trimws(worksite_state))])]
site <- sites[, .(single = !anyNA(st) && uniqueN(st) == 1L, st = first(st)), by = .(case_number, fiscal_year)]
x <- merge(x, site[single == TRUE], by = c("case_number","fiscal_year"))
x <- x[st == worksite_state]
stopifnot(!anyDuplicated(x$case_number))
rates <- as.data.table(read_parquet(path_int("aewr_rate_schedule.parquet")))
rates[, st := unname(lookup[toupper(state_name)])]
setorder(rates, st, reference_year)
rates[, prior := shift(aewr), by = st]
rates <- rates[reference_year %between% c(2015L, 2024L) & !is.na(st) & aewr != prior]
stopifnot(!anyDuplicated(rates[, .(st,reference_year)]))
# Calendar-day event time is measured from the published effective start.
# +/-7 cents probes nearby wage heaping independently of another region's rate.
joined <- merge(x, rates[, .(st, reference_year, aewr, prior, effective_start)], by = "st",
                allow.cartesian = TRUE)
joined[, day := as.integer(received - as.IDate(effective_start))]
joined <- joined[day >= -168 & day < 56]
joined[, window := fcase(day < -112, "distant_pre", day < -56, "placebo_pre",
                         day < 0, "immediate_pre", default = "post")]
joined[, new_hit := abs(wage-aewr) < 0.005]
joined[, old_hit := abs(wage-prior) < 0.005]
joined[, nearby_hit := (as.integer(abs(wage-aewr-0.07) < 0.005) +
                         as.integer(abs(wage-aewr+0.07) < 0.005)) / 2]
stopifnot(!anyDuplicated(joined[, .(case_number, reference_year)]), nrow(joined) > 0)
cells <- joined[, .(cases = .N, positions = sum(positions), new = sum(new_hit),
                     old = sum(old_hit), nearby = sum(nearby_hit),
                     pos_new = sum(positions * new_hit)), by = .(reference_year, st, window)]
fwrite(cells, file.path(out, "reset_state_windows.csv"))
summary <- cells[, .(cases = sum(cases), new_share = sum(new)/sum(cases),
                     old_share = sum(old)/sum(cases), nearby_share = sum(nearby)/sum(cases),
                     position_weighted_new = sum(pos_new)/sum(positions)), by = window]
fwrite(summary, file.path(out, "reset_windows.csv"))
# Reweight to common state-by-update support and give each update equal weight.
common <- cells[, if (uniqueN(window) == 4L) .SD, by = .(reference_year, st)]
balanced <- common[, .(share = new / cases), by = .(reference_year, st, window)]
annual <- balanced[, .(states = .N, mean_state_share = mean(share)), by = .(reference_year, window)]
fwrite(annual, file.path(out, "reset_balanced.csv"))
print(summary)
print(dcast(annual, reference_year ~ window, value.var = "mean_state_share"))

# Follow-ups: direction of the published revision and a fixed mix of filing
# identities seen in both adjacent windows. This last estimand selects repeat
# filers by observed filing on each side and is descriptive.
jump <- joined[window %chin% c("immediate_pre","post")]
jump[, direction := fifelse(aewr > prior, "rise", "fall")]
signs <- jump[, .(cases=.N, states=uniqueN(st), updates=uniqueN(reference_year),
                   state_updates=uniqueN(.SD,by=c("st","reference_year")),
                   new_share=mean(new_hit), old_share=mean(old_hit)),
               by=.(direction,window), .SDcols=c("st","reference_year","new_hit","old_hit")]
fwrite(signs,file.path(out,"reset_direction.csv"))
ids <- as.data.table(read_parquet(path_int("policy_2025","filer","filer_ids.parquet"),
                   col_select = c("fiscal_year","case_number","filer_R7")))
stopifnot(!anyDuplicated(ids, by=c("fiscal_year","case_number")))
jump <- merge(jump, ids, by=c("fiscal_year","case_number"))
fixed <- jump[!is.na(filer_R7), .(cases=.N, new=mean(new_hit), old=mean(old_hit)),
              by=.(st,reference_year,filer_R7,soc,window)]
fixed <- fixed[, if (.N == 2L) .SD, by=.(st,reference_year,filer_R7,soc)]
stopifnot(nrow(fixed)>0L)
fwrite(fixed[, .(cells=.N, cases=sum(cases), new_share=mean(new), old_share=mean(old)),
             by=window], file.path(out,"reset_repeat_cells.csv"))
