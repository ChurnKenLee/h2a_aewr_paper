# Application packaging near the additional 150/200/250/300 position steps.
source("code/paths.R")
library(data.table)
setDTthreads(2L)
out <- project_path("explorations", "historical_usage", "packaging")
fields <- c("fiscal_year", "case_number", "case_received_date", "decision_date",
            "type_of_employer_application", "h2a_labor_contractor",
            "nbr_workers_requested", "worksite_state")
details <- c("requested_begin_date", "requested_end_date", "worksite_city", "soc_code", "job_title", "case_status")
raw <- as.data.table(arrow::read_parquet(path_int("h2a_disclosure_records.parquet"),
                                        col_select = all_of(c(fields, details))))[fiscal_year %between% c(2020,2025)]
x <- unique(raw[, ..fields])
x[, received := suppressWarnings(as.IDate(substr(case_received_date, 1, 10)))]
x[, decision := suppressWarnings(as.IDate(substr(decision_date, 1, 10)))]
x[, workers := suppressWarnings(as.numeric(nbr_workers_requested))]
x <- unique(x[, .(fiscal_year, case_number, received, decision, type_of_employer_application,
                  h2a_labor_contractor, workers, worksite_state)])
x <- x[nzchar(case_number) & !duplicated(case_number) & !duplicated(case_number, fromLast = TRUE)]
x[, era := fcase(received >= "2020-04-01" & received < "2022-04-01", "early",
                 received >= "2023-04-01" & received < "2025-04-01", "late")]
x <- x[!is.na(era) & as.integer(decision - received) %between% c(0, 180) &
       type_of_employer_application == "Individual Employer" &
       h2a_labor_contractor %chin% c("Y", "N") & workers > 0 & workers == floor(workers)]
x[, contractor := h2a_labor_contractor == "Y"]
# Keep the same application sample as bond_thresholds.R before inspecting details.
extra <- unique(raw[, c("fiscal_year", "case_number", details), with = FALSE])
extra <- extra[, if (.N == 1L) .SD else .SD[0L], by = .(fiscal_year, case_number)]
extra[, metadata_unique := TRUE]
ids <- as.data.table(arrow::read_parquet(path_int("policy_2025","filer","filer_ids.parquet"),
                                        col_select = c("fiscal_year", "case_number", "filer_R7")))
stopifnot(!anyDuplicated(ids, by = c("fiscal_year", "case_number")))
x <- merge(x, extra, by = c("fiscal_year", "case_number"), all.x = TRUE)
x <- merge(x, ids, by = c("fiscal_year", "case_number"), all.x = TRUE)
stopifnot(!anyDuplicated(x$case_number))
x[, start := suppressWarnings(as.IDate(substr(requested_begin_date, 1, 10)))]
x[, id := as.character(filer_R7)]
x[, city_key := toupper(trimws(worksite_city))]
x[, certified := grepl("CERTIFIED|CERTIFICATION", toupper(case_status))]
x[, valid := !is.na(metadata_unique) & !is.na(id) & !is.na(start) &
              abs(year(start) - fiscal_year) <= 1L & !is.na(worksite_state) & nzchar(trimws(worksite_state))]
x[, k := 50L * round(workers / 50)]
x[, focal := k %in% c(150L,200L,250L,300L) & workers >= k - 5 & workers < k + 5]
coverage <- x[, .(cases = .N, supported = sum(valid), focal = sum(focal), focal_supported = sum(focal & valid),
                  missing_filer = sum(is.na(id)), metadata_conflict = sum(is.na(metadata_unique)),
                  missing_or_invalid_start = sum(is.na(start) | abs(year(start) - fiscal_year) > 1L)),
              by = .(era, contractor)]
fwrite(coverage[order(era, contractor)], file.path(out, "coverage.csv"))
x <- x[valid == TRUE]
# Same-size applications with distinct case numbers might duplicate or revise a
# filing. Flag exact agreement on requested start/end, occupation, title, city,
# state, filer, number requested and receipt date. Differences in status/decision
# are allowed. Collapsing these pairs is a conservative sensitivity.
key <- c("era","contractor","id","worksite_state","worksite_city","start","requested_end_date",
         "soc_code","job_title","workers","received")
setorder(x, case_number)
x[, duplicate_lookalike := duplicated(x, by = key)]
x[, any_lookalike := .N > 1L, by = key]
fwrite(x[, .(cases = .N, members = sum(any_lookalike), excess = sum(duplicate_lookalike),
              focal_excess = sum(focal & duplicate_lookalike)), by = .(era, contractor)],
       file.path(out, "lookalikes.csv"))

companions <- list()
bundles <- list()
for (sample in c("distinct_cases", "collapsed_lookalikes")) {
  a <- if (sample == "distinct_cases") copy(x) else copy(x[duplicate_lookalike == FALSE])
  # Compare a focal application to all other eligible applications from its
  # filer-state-era, including companions outside the size bins and across flags.
  eligible <- a[, .(id, worksite_state, era, peer_case = case_number, peer_start = start,
                    peer_size = workers, peer_soc = soc_code, peer_received = received,
                    peer_city = city_key, peer_certified = certified)]
  eligible[, peer_day := as.integer(peer_start)]
  targets <- a[focal == TRUE, .(id, worksite_state, era, case_number, contractor, start, received,
                        workers, k, soc_code, city_key, certified, below = workers < k, min_day = as.integer(start)-14L,
                        max_day = as.integer(start)+14L)]
  pairs <- eligible[targets, on = .(id, worksite_state, era, peer_day >= min_day, peer_day <= max_day),
                    nomatch = 0L, allow.cartesian = TRUE]
  pairs <- pairs[peer_case != case_number]
  pairs[, distance := abs(as.integer(start - peer_start))]
  pairs[, same_soc := !is.na(soc_code) & nzchar(trimws(soc_code)) & soc_code == peer_soc]
  pairs[, specific_match := certified & peer_certified & same_soc &
          !is.na(city_key) & nzchar(city_key) & city_key == peer_city &
          abs(as.integer(received - peer_received)) <= 7L]
  for (window in c(0L, 7L, 14L)) {
    s <- pairs[distance <= window, .(n_companions = .N, crossed = sum(peer_size) + workers[1] >= k[1],
                 small = any(peer_size <= 10L), small_cross = any(peer_size <= 10L & peer_size+workers >= k),
                 identical = any(peer_size == workers),
                 specific_small_cross = any(specific_match & peer_size <= 10L & peer_size+workers >= k, na.rm=TRUE),
                 specific_identical = any(specific_match & peer_size == workers, na.rm=TRUE),
                 close_receipt_same_soc = any(same_soc & abs(as.integer(received-peer_received)) <= 7, na.rm=TRUE)),
               by = case_number]
    one <- merge(targets, s, by = "case_number", all.x = TRUE)
    one[is.na(n_companions), `:=`(n_companions = 0L, crossed = FALSE, small = FALSE,
                                 small_cross = FALSE, identical = FALSE, close_receipt_same_soc = FALSE,
                                 specific_small_cross = FALSE, specific_identical = FALSE)]
    for (width in c(1L, 5L)) {
      local <- one[workers >= k-width & workers < k+width]
      counts <- local[, .(cases = .N, filers = uniqueN(id), with_peer = sum(n_companions > 0),
                          with_small = sum(small), with_small_cross = sum(small_cross),
                          with_same_size = sum(identical), with_peer_cross = sum(crossed),
                          same_size_filers = uniqueN(id[identical]), small_cross_filers = uniqueN(id[small_cross]),
                          with_specific_identical = sum(specific_identical), with_specific_small_cross = sum(specific_small_cross),
                          with_close_receipt_same_soc = sum(close_receipt_same_soc)),
                      by = .(era, contractor, below)]
      companions[[length(companions)+1L]] <- counts[, `:=`(sample = sample, days = window, width = width)]
    }
  }
  # Each exact-start group enters once. Summing all cases allows the bundle to
  # move far away from a threshold. Mixed contractor-status bundles are audited.
  for (unit in c("application", "filer_state_start")) {
    bins <- if (unit == "application") a[, .(era, contractor, workers, mix = FALSE)] else
      a[, .(contractor = first(contractor), workers = sum(workers), mix = uniqueN(contractor) > 1L),
        by = .(era, id, worksite_state, start)]
    mixed <- bins[mix == TRUE, .N]
    for (width in c(1L, 5L)) {
      cells <- rbindlist(lapply(c(150L,200L,250L,300L), function(k)
        bins[!mix & workers >= k-width & workers < k+width,
             .(below = sum(workers < k), total=.N), by=.(era,contractor)]))
      totals <- cells[, .(below=sum(below), total=sum(total)), by=.(era,contractor)]
      bundles[[length(bundles)+1L]] <- totals[, `:=`(sample=sample,unit=unit,width=width,omitted_mixed_groups=mixed)]
    }
  }
}
fwrite(rbindlist(companions), file.path(out, "companions.csv"))
fwrite(rbindlist(bundles), file.path(out, "bundle_sizes.csv"))
