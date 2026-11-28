# Full fiscal-year descriptive accounts for certified positions in reconciled cases.
source("code/paths.R")
library(data.table)
setDTthreads(2L)
out <- project_path("explorations", "historical_usage", "business")
years <- 2008:2025
read_columns <- function(file, columns) {
  as.data.table(arrow::read_parquet(path_int(file), col_select = all_of(columns)))
}
rows <- read_columns("h2a_worksite_exposure_audit.parquet", c(
  "allocation_id", "fiscal_year", "case_number", "adjusted_nbr_workers_certified",
  "employer_id_balanced", "employer_is_worksite_business", "case_allocation_reconciled",
  "employer_record_role"))
rows <- rows[fiscal_year %in% years]
stopifnot(!anyDuplicated(rows$allocation_id), !anyNA(rows$case_allocation_reconciled))
audit <- read_columns("h2a_case_allocation_audit.parquet", c(
  "case_number", "fiscal_year", "case_certified_target", "case_allocation_reconciled"))
audit <- audit[fiscal_year %in% years]
stopifnot(!anyDuplicated(audit, by = c("case_number", "fiscal_year")))
setnames(rows, "adjusted_nbr_workers_certified", "positions")
valid <- rows[case_allocation_reconciled == TRUE]
stopifnot(all(is.finite(valid$positions)), all(valid$positions >= 0))
valid <- valid[positions > 0]
status_rows <- read_columns("h2a_case_allocation_source.parquet",
                       c("fiscal_year", "case_number", "case_status"))
status <- status_rows[, .(all_rows_certified = all(grepl("CERTIFIED|CERTIFICATION",
                                                   toupper(case_status)))),
                 by = .(fiscal_year, case_number)]
valid <- merge(valid, status, by = c("fiscal_year", "case_number"), all.x = TRUE)
stopifnot(!anyNA(valid$all_rows_certified))
stopifnot(all(valid[employer_is_worksite_business == TRUE, nzchar(employer_id_balanced) &
                      !is.na(employer_id_balanced)]))
filer <- read_columns("policy_2025/filer/filer_ids.parquet", c(
  "fiscal_year", "case_number", "filer_R5", "filer_R7", "filer_R8"))
stopifnot(!anyDuplicated(filer, by = c("fiscal_year", "case_number")))
valid <- merge(valid, filer, by = c("fiscal_year", "case_number"), all.x = TRUE)
tiers <- read_columns("h2a_employer_record_membership.parquet", c(
  "employer_id_balanced", "employer_id_high_recall", "employer_is_worksite_business",
  "employer_linkage_unit", "employer_linkage_version", "employer_linkage_input_max_year",
  "employer_linkage_review_status"))
stopifnot(all(tiers$employer_linkage_unit == "worksite_business"),
          all(tiers$employer_linkage_version == "v2"),
          all(tiers$employer_linkage_input_max_year == 2026),
          all(tiers$employer_linkage_review_status %in% c("complete", "pending")))
tier_map <- unique(tiers[employer_is_worksite_business == TRUE,
                         .(employer_id_balanced, employer_id_high_recall)])
stopifnot(!anyDuplicated(tier_map$employer_id_balanced))
valid <- merge(valid, tier_map, by = "employer_id_balanced", all.x = TRUE)
named <- valid[employer_is_worksite_business == TRUE]
stopifnot(!anyNA(named$employer_id_high_recall), !anyNA(valid$filer_R7))
coverage <- valid[, .(all_positions = sum(positions),
                      certified_status_positions = sum(positions[all_rows_certified]),
                      named_positions = sum(positions[employer_is_worksite_business]),
                      common_positions = sum(positions[employer_is_worksite_business & !is.na(filer_R7)])),
                  by = fiscal_year]
targets <- audit[, .(reconciled_target = sum(case_certified_target[case_allocation_reconciled]),
                      quarantined_cases = sum(!case_allocation_reconciled),
                      quarantined_known_target = sum(case_certified_target[!case_allocation_reconciled], na.rm = TRUE)),
                  by = fiscal_year]
coverage <- merge(coverage, targets, by = "fiscal_year")
stopifnot(identical(coverage$fiscal_year, years),
          max(abs(coverage$all_positions - coverage$reconciled_target)) < 1e-5)
fwrite(coverage, file.path(out, "coverage.csv"))
# External benchmark audit. The FY2008 workbook marks master records by
# organization flag. All raw certified decisions fall inside the source FY.
raw08 <- as.data.table(readxl::read_excel(path_raw("h2a", "H2A_FY2008.xlsx")))
master08 <- raw08[!is.na(ORGANIZATION_FLAG) & nzchar(trimws(ORGANIZATION_FLAG)) &
                    grepl("Certified", CASE_STATUS, fixed = TRUE)]
stopifnot(all(as.Date(master08$DECISION_DATE) >= as.Date("2007-10-01")),
          all(as.Date(master08$DECISION_DATE) <= as.Date("2008-09-30")))
official <- data.table(fiscal_year = c(2008:2010, 2025L),
                       official_positions = c(82099, 86014, 79011, 398258))
bench <- merge(official, coverage[, .(fiscal_year, all_positions, certified_status_positions)],
               by = "fiscal_year")
bench[fiscal_year == 2008, workbook_certified_master_positions := sum(master08$NBR_WORKERS_CERTIFIED,
                                                                     na.rm = TRUE)]
bench[, difference := all_positions - official_positions]
fwrite(bench, file.path(out, "official_benchmarks.csv"))
roles <- valid[, .(positions = sum(positions)), by = .(fiscal_year, employer_record_role,
                                                      employer_is_worksite_business)]
fwrite(roles[order(fiscal_year, employer_record_role)], file.path(out, "roles.csv"))
# The filer extractor's boolean collapses blanks with "N". Preserve source
# blanks and disagreement for the separate contractor account.
flag_rows <- read_columns("h2a_with_fips.parquet",
                          c("fiscal_year", "case_number", "h2a_labor_contractor"))
flags <- flag_rows[fiscal_year %between% c(2020L, 2025L), {
  observed <- unique(toupper(trimws(h2a_labor_contractor)))
  .(contractor_flag = if (length(observed) == 1L && observed %in% c("Y", "N")) observed else "unknown")
}, by = .(fiscal_year, case_number)]
recent <- merge(valid[fiscal_year >= 2020], flags, by = c("fiscal_year", "case_number"), all.x = TRUE)
stopifnot(!anyNA(recent$contractor_flag))
contractors <- recent[, .(positions = sum(positions),
                          named_positions = sum(positions[employer_is_worksite_business]),
                          cases = uniqueN(case_number)), by = .(fiscal_year, contractor_flag)]
fwrite(contractors[order(fiscal_year, contractor_flag)], file.path(out, "contractors.csv"))

# A first appearance is the first positive certification since 2008.
# Disappearance is one-year absence. Future presence never determines inclusion.
account <- function(data, id, unit) {
  x <- data[!is.na(get(id)), .(positions = sum(positions)),
            by = .(id = get(id), fiscal_year)]
  setorder(x, id, fiscal_year)
  x[, first := min(fiscal_year), by = id]
  yearly <- x[, {
    ranked <- sort(positions, decreasing = TRUE)
    .(units = .N, positions = sum(positions), top_one_percent_share =
        sum(head(ranked, ceiling(.N / 100))) / sum(positions),
      top_100_share = sum(head(ranked, 100L)) / sum(positions))
  }, by = fiscal_year]
  setorder(yearly, fiscal_year)
  flows <- rbindlist(lapply(years[-1], function(t) {
    pair <- merge(x[fiscal_year == t - 1L, .(id, previous = positions)],
                  x[fiscal_year == t, .(id, first, current = positions)], by = "id", all = TRUE)
    pair[, continues := !is.na(current) & !is.na(previous)]
    pair[, .(fiscal_year = t,
             continuing = sum(current[continues] - previous[continues]),
             first_appearance = sum(current[is.na(previous) & first == t]),
             return_after_gap = sum(current[is.na(previous) & first < t]),
             disappearance = -sum(previous[is.na(current)]))]
  }))
  flows[, total_change := continuing + first_appearance + return_after_gap + disappearance]
  stopifnot(max(abs(flows$total_change - diff(yearly$positions))) < 1e-6)
  # Cohorts from 2009 onward. Endpoint is age three and permits intervening gaps.
  cohorts <- rbindlist(lapply(2009:2022, function(t) {
    start <- x[first == t & fiscal_year == t, .(id, initial = positions)]
    later <- x[first == t & fiscal_year == t + 3L, .(id, age_three = positions)]
    cohort <- merge(start, later, by = "id", all.x = TRUE)
    cohort[, .(cohort = t, units = .N, initial_positions = sum(initial),
               observed_at_three = sum(!is.na(age_three)),
               age_three_positions = sum(age_three, na.rm = TRUE),
               retention = mean(!is.na(age_three)),
               position_ratio = sum(age_three, na.rm = TRUE) / sum(initial))]
  }))
  list(yearly = yearly[, unit := unit], flows = flows[, unit := unit],
       cohorts = cohorts[, unit := unit])
}
# Compare both levels on the exact same mass, then retain the broader filer frame.
accounts <- list(
  account(named, "employer_id_balanced", "business_balanced"),
  account(named, "employer_id_high_recall", "business_high_recall"),
  account(named, "filer_R7", "filer_R7_named_positions"),
  account(valid, "filer_R5", "filer_R5"), account(valid, "filer_R7", "filer_R7"),
  account(valid, "filer_R8", "filer_R8"),
  account(valid[all_rows_certified == TRUE], "filer_R7", "filer_R7_certified_status"),
  account(named[all_rows_certified == TRUE], "employer_id_balanced", "business_balanced_certified_status"))
for (table in names(accounts[[1]])) {
  combined <- rbindlist(lapply(accounts, "[[", table))
  fwrite(combined, file.path(out, paste0(table, ".csv")))
}
# At each tier, how much capacity is placed on a newly observed relationship
# between a filer and a business both seen in the previous fiscal year?
edge_flows <- function(id, tier) {
  edges <- named[, .(positions = sum(positions)),
                 by = .(fiscal_year, filer_R7, business = get(id))]
  rbindlist(lapply(years[-1], function(t) {
    old <- edges[fiscal_year == t - 1L]
    cur <- edges[fiscal_year == t]
    cur[, relation := "continuing_relationship"]
    cur[!old, on = .(filer_R7, business), relation := fcase(
      filer_R7 %in% old$filer_R7 & business %in% old$business, "both_previously_active_separately",
      filer_R7 %in% old$filer_R7, "prior_filer_only",
      business %in% old$business, "prior_business_only", default = "neither_previously_active")]
    cur[, .(tier = tier, fiscal_year = t, relationships = .N, positions = sum(positions)), by = relation]
  }))
}
edges <- rbind(edge_flows("employer_id_balanced", "balanced"),
               edge_flows("employer_id_high_recall", "high_recall"))
edge_coverage <- merge(edges[, .(edge_positions = sum(positions)), by = .(tier, fiscal_year)],
                       coverage, by = "fiscal_year")
stopifnot(max(abs(edge_coverage$edge_positions - edge_coverage$common_positions)) < 1e-6)
fwrite(edges[order(tier, fiscal_year, relation)], file.path(out, "relationships.csv"))
