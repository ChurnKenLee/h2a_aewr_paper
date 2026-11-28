# Are H-2ALC applications disproportionately sized just below a bond step?
source("code/paths.R")
library(data.table)
library(arrow)
setDTthreads(2L)

out <- commandArgs(trailingOnly = TRUE)[1L]
stopifnot(!is.na(out))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
fields <- c("fiscal_year", "case_number", "case_received_date", "decision_date",
            "type_of_employer_application", "h2a_labor_contractor",
            "nbr_workers_requested", "worksite_state")
x <- as.data.table(read_parquet(path_int("h2a_disclosure_records.parquet"), col_select = all_of(fields)))
x <- x[fiscal_year %between% c(2020L, 2025L)]
x[, received := suppressWarnings(as.IDate(substr(case_received_date, 1, 10)))]
x[, decision := suppressWarnings(as.IDate(substr(decision_date, 1, 10)))]
x[, workers := suppressWarnings(as.numeric(nbr_workers_requested))]
# Count an application once. Drop conflicting versions rather than picking one.
x <- unique(x[, .(fiscal_year, case_number, received, decision, type_of_employer_application,
                  h2a_labor_contractor, workers, worksite_state)])
x <- x[nzchar(case_number) & !duplicated(case_number) & !duplicated(case_number, fromLast = TRUE)]
# Two full April–March receipt windows, with the same 180 days for decisions.
x[, era := fcase(received >= "2020-04-01" & received < "2022-04-01", "early",
                 received >= "2023-04-01" & received < "2025-04-01", "late")]
x <- x[!is.na(era) & as.integer(decision - received) %between% c(0, 180) &
       type_of_employer_application == "Individual Employer" &
       h2a_labor_contractor %chin% c("Y", "N") & workers > 0 & workers == floor(workers)]
stopifnot(nrow(x) > 0L, !anyDuplicated(x$case_number))
x[, contractor := h2a_labor_contractor == "Y"]
fwrite(x[, .(cases = .N), by = .(era, contractor)], file.path(out, "bond_sample.csv"))

# Local contrasts surround the statutory (or comparison) integer with a
# symmetric count of possible request sizes. Include size = cutoff on high side.
thresholds <- data.table(k = c(25L, 50L, 75L, 100L, 150L, 200L, 250L, 300L,
                                35L, 60L, 85L, 110L, 125L, 175L, 225L, 275L),
                         kind = rep(c("existing step", "new step", "placebo low", "placebo high"),
                                    each = 4))
counts <- list()
for (i in seq_len(nrow(thresholds))) {
  k <- thresholds$k[i]
  for (w in c(1L, 3L, 5L)) {
    local <- x[workers >= k-w & workers < k+w]
    cells <- local[, .(below = sum(workers < k), at = sum(workers == k),
                       total = .N), by = .(era, contractor)]
    cells <- merge(CJ(era = c("early", "late"), contractor = c(FALSE, TRUE)),
                   cells, by = c("era", "contractor"), all.x = TRUE)
    for (n in c("below", "at", "total")) setnafill(cells, cols = n, fill = 0)
    counts[[length(counts)+1L]] <- cells[, `:=`(k = k, width = w, kind = thresholds$kind[i])]
  }
}
tab <- rbindlist(counts)
stopifnot(nrow(tab) == nrow(thresholds)*3L*4L, all(tab$below + tab$at <= tab$total))
fwrite(tab, file.path(out, "bond_size_counts.csv"))
groups <- tab[, .(below = sum(below), at = sum(at), total = sum(total)),
              by = .(kind, width, era, contractor)]
groups[, below_share := below / total]
fwrite(groups, file.path(out, "bond_groups.csv"))
print(groups[width == 5L][order(kind, contractor, era)])

# Follow-up diagnostics after the initial size bins were inspected. Filer R7
# links repeated applications. Its ID can use later identity evidence.
ids <- as.data.table(read_parquet(path_int("policy_2025","filer","filer_ids.parquet"),
                                  col_select = c("fiscal_year","case_number","filer_R7")))
ids <- unique(ids)
stopifnot(!anyDuplicated(ids, by = c("fiscal_year","case_number")))
x <- merge(x, ids, by = c("fiscal_year","case_number"), all.x = TRUE)
x[, id := fifelse(is.na(filer_R7), paste0("unmatched_",case_number), as.character(filer_R7))]
x[, late := as.integer(era == "late")]
x[, alc := as.integer(contractor)]
x[, receipt_year := fcase(received < "2021-04-01", "2020/21",
                          received < "2022-04-01", "2021/22",
                          received < "2024-04-01", "2023/24", default = "2024/25")]
local_bins <- list()
for (i in seq_len(nrow(thresholds))) {
  k <- thresholds$k[i]
  slice <- copy(x[workers >= k-5 & workers < k+5])
  local_bins[[i]] <- slice[, `:=`(cutoff = k, kind = thresholds$kind[i], below = as.integer(workers < k))]
}
local <- rbindlist(local_bins)
fwrite(local[, .(cases = .N, filers = uniqueN(id), below = sum(below),
                 just_below = sum(workers == cutoff-1), at = sum(workers == cutoff)),
             by = .(kind, receipt_year, contractor)], file.path(out,"bond_by_year.csv"))
details <- list()
did <- function(s) {
  contrasts <- s[, .(p = mean(below)), by = .(alc, late)]
  if (nrow(contrasts) != 4L) return(NA_real_)
  contrasts[, sum(p * (2*alc-1) * (2*late-1))]
}
for (group in unique(local$kind)) {
  for (w in c(1L,3L,5L)) {
    s <- local[kind == group & workers >= cutoff-w & workers < cutoff+w]
    fit <- fixest::feols(below ~ alc*late, data = s, vcov = ~id, notes = FALSE)
    ci <- as.numeric(confint(fit)["alc:late",])
    dropped_state <- vapply(unique(s$worksite_state), function(st) did(s[worksite_state != st]), 0.0)
    dropped_filer <- vapply(unique(s$id), function(j) did(s[id != j]), 0.0)
    details[[length(details)+1L]] <- data.table(
      kind=group, width=w, cases=nrow(s), filers=uniqueN(s$id), states=uniqueN(s$worksite_state),
      contrast=did(s), filer_ci_lo=unname(ci[1]), filer_ci_hi=unname(ci[2]),
      state_del_min=min(dropped_state, na.rm=TRUE), state_del_max=max(dropped_state, na.rm=TRUE),
      filer_del_min=min(dropped_filer, na.rm=TRUE), filer_del_max=max(dropped_filer, na.rm=TRUE))
  }
}
fwrite(rbindlist(details), file.path(out, "bond_diagnostics.csv"))
# Predetermined large-filer cohort. Capture total requests from a broader
# all-role frame to keep contractor-role switches and applications away from
# the size bins. Anchor on year 1 only. Include absent filers at zero.
base <- x[receipt_year == "2020/21" & contractor,
          .(peak = max(workers)), by = id][peak >= 100 & peak < 300]
base[, group := fifelse(peak >= 150, "150-299", "100-149")]
all_roles <- unique(as.data.table(read_parquet(path_int("h2a_disclosure_records.parquet"),
                         col_select = c("fiscal_year","case_number","case_received_date",
                                        "decision_date","nbr_workers_requested")))[fiscal_year %between% c(2020L,2025L),
                         .(fiscal_year, case_number, case_received_date, decision_date, nbr_workers_requested)])
all_roles <- all_roles[nzchar(case_number) & !duplicated(case_number) & !duplicated(case_number, fromLast = TRUE)]
all_roles <- merge(all_roles, ids, by = c("fiscal_year","case_number"))
all_roles[, received := suppressWarnings(as.IDate(substr(case_received_date,1,10)))]
all_roles[, lag_days := as.integer(suppressWarnings(as.IDate(substr(decision_date,1,10))) - received)]
all_roles[, workers := suppressWarnings(as.numeric(nbr_workers_requested))]
all_roles <- all_roles[lag_days %between% c(0L,180L) & workers > 0 & workers == floor(workers)]
all_roles[, receipt_year := year(received) - as.integer(month(received) < 4L)]
frame <- merge(CJ(id = base$id, receipt_year = c(2020L,2021L,2023L,2024L)), base, by = "id")
totals <- all_roles[as.character(filer_R7) %chin% base$id,
                    .(cases=.N, requests=sum(workers)), by=.(id=as.character(filer_R7),receipt_year)]
frame <- merge(frame, totals, by=c("id","receipt_year"), all.x=TRUE)
frame[is.na(cases), `:=`(cases=0L,requests=0)]
fwrite(frame[, .(baseline_filers=.N, present=sum(cases>0), cases=sum(cases),
                 requests=sum(requests)), by=.(group,receipt_year)], file.path(out,"bond_filer_quantity.csv"))

# Exact one-worker responses are a follow-up to the initial local shares.
# Broader denominator and split-year leads probe sensitivity to local selection.
large <- x[workers %between% c(100,349)]
exact_summary <- list()
for (design in c("new step", "placebo high", "all large cases")) {
  if (design == "all large cases") {
    s <- copy(large)
    s[, below := as.integer(workers %in% c(149,199,249,299))]
  } else {
    s <- copy(local[kind == design])
    s[, below := as.integer(workers == cutoff-1L)]
  }
  for (period in c("both", "split early window", "split late window")) {
    u <- if (period == "both") copy(s) else if (period == "split early window") copy(s[late == 0L]) else copy(s[late == 1L])
    if (period != "both") u[, late := as.integer(receipt_year %chin% c("2021/22","2024/25"))]
    fit <- fixest::feols(below ~ alc*late, data = u, vcov = ~id, notes = FALSE)
    ci <- as.numeric(confint(fit)["alc:late",])
    state_ci <- as.numeric(confint(fit, vcov = ~worksite_state)["alc:late",])
    exact_summary[[length(exact_summary)+1L]] <- data.table(
      group=design, period=period, cases=nrow(u), effect=did(u),
      filer_lo=ci[1], filer_hi=ci[2], state_lo=state_ci[1], state_hi=state_ci[2])
  }
}
fwrite(rbindlist(exact_summary), file.path(out,"bond_exact_diagnostics.csv"))
# Draw the complete integer distribution. Each line sums to 100 within the
# displayed ten sizes and across the four newly introduced thresholds.
dist <- local[kind == "new step", .(cases = .N), by = .(delta=workers-cutoff, era, contractor)]
dist <- merge(CJ(delta = -5:4, era = c("early","late"), contractor = c(FALSE,TRUE)),
              dist, by=c("delta","era","contractor"), all.x=TRUE)
dist[is.na(cases), cases := 0L]
dist[, pct := 100*cases/sum(cases), by=.(era,contractor)]
fwrite(dist, file.path(out,"bond_distribution.csv"))
png(file.path(out,"bond_distribution.png"), width=1320, height=720, res=130)
op <- par(mfrow=c(1,2), mar=c(5,4,4,1), oma=c(2,1,1,0))
for (contract in c(TRUE, FALSE)) {
  g <- dist[contractor == contract]
  plot(NA, xlim=c(-5,4), ylim=c(0,70), xaxt="n", xlab="Workers requested relative to threshold",
       ylab="Share of nearby applications (%)",
       main=if (contract) "Labor contractors" else "Other individual employers")
  axis(1, at=-5:4)
  abline(v=0, lty=3, col="gray65")
  for (period in c("early","late")) {
    series <- g[era == period][order(delta)]
    lines(series$delta, series$pct, lwd=2, type="b", pch=if (period=="early") 1 else 19,
          col=if (period=="early") "#4B5563" else "#006B78", lty=if (period=="early") 2 else 1)
  }
  legend("topright", c("Apr 2020–Mar 2022","Apr 2023–Mar 2025"), pch=c(1,19),
         lty=c(2,1), col=c("#4B5563","#006B78"), bty="n", cex=0.85)
}
mtext("Thresholds at 150, 200, 250 and 300 requested workers. Zero is the first size in the higher bond tier.",
      outer=TRUE, side=1, line=0.6, cex=0.9)
par(op)
dev.off()

# The same forms separately disclose total workers needed and H-2A workers
# requested. For applications just below a legal step, did the filer report
# a larger total crew on that application?
crew_sizes <- unique(as.data.table(read_parquet(path_int("h2a_disclosure_records.parquet"),
                  col_select = c("fiscal_year","case_number","nbr_workers_needed")))[
                  fiscal_year %between% c(2020L,2025L),
                  .(fiscal_year, case_number, needed = suppressWarnings(as.numeric(nbr_workers_needed)))])
crew_sizes <- crew_sizes[!duplicated(crew_sizes, by=c("fiscal_year","case_number")) &
                 !duplicated(crew_sizes, by=c("fiscal_year","case_number"), fromLast=TRUE)]
crew <- merge(local[kind == "new step"], crew_sizes, by=c("fiscal_year","case_number"), all.x=TRUE)
stopifnot(nrow(crew) == nrow(local[kind == "new step"]), sum(is.finite(crew$needed)) > 0.95*nrow(crew))
fwrite(crew[, .(cases=.N, total_known=sum(is.finite(needed)), smaller=sum(needed < workers, na.rm=TRUE),
                 total_at_or_above_step=sum(needed >= cutoff, na.rm=TRUE),
                 total_equal_step=sum(needed == cutoff, na.rm=TRUE),
                 same_as_request=sum(needed == workers, na.rm=TRUE)),
            by=.(era, contractor, one_below = workers == cutoff-1L)],
       file.path(out,"bond_total_needed.csv"))
higher <- x[, .(workers, contractor, era)]
higher[, k := 50L * floor((workers+25L)/50L)]
higher <- higher[k >= 350L & workers >= k-5L & workers < k+5L]
fwrite(higher[, .(cases=.N, just_below=sum(workers == k-1), below=sum(workers < k), at=sum(workers == k)),
              by=.(era, contractor)], file.path(out,"bond_higher_steps.csv"))

# Aggregate volume per baseline filer, preserving the zero rows. Compare the
# average of two later 12-month periods to the second, non-selection pre-year.
# Stratified delete-one-filer uncertainty conditions on the first-year groups.
members <- dcast(frame, id + group ~ receipt_year, value.var = "requests")
members[, post := (`2023` + `2024`)/2]
net <- list()
for (yrs in list(c("2021","post"), c("2020","2021"))) {
  changes <- members[, {
    before <- get(yrs[1L])
    after <- get(yrs[2L])
    log_change <- log(sum(after)/sum(before))
    drops <- log((sum(after)-after)/(sum(before)-before))
    .(change = log_change, variance = (.N-1)/.N * sum((drops-mean(drops))^2), n = .N)
  }, by=group]
  effect <- changes[group == "150-299",change] - changes[group == "100-149",change]
  se <- sqrt(sum(changes$variance))
  crit <- qt(0.975, min(changes$n)-1)
  net[[length(net)+1L]] <- data.table(base=yrs[1L], comparison=yrs[2L], ratio_change=expm1(effect),
                            lo=expm1(effect-crit*se), hi=expm1(effect+crit*se))
}
fwrite(rbindlist(net), file.path(out,"bond_filer_contrast.csv"))
