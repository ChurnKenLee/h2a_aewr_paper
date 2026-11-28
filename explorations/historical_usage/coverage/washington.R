# Footprint check. A Washington location or quoted piece rate does not establish eligibility.
source("code/paths.R")
library(data.table)
keys <- c("fiscal_year", "case_number")
fields <- c(keys, "primary_sub", "worksite_state", "piece_rate_offer")
raw <- as.data.table(arrow::read_parquet(path_int("h2a_disclosure_records.parquet"),
                                         col_select = tidyselect::all_of(fields)))
raw <- raw[fiscal_year %between% c(2008L, 2025L) & !is.na(case_number) & nzchar(trimws(case_number))]
raw[, in_wa := toupper(trimws(worksite_state)) %chin% c("WA", "WASHINGTON")]
raw[, master := is.na(primary_sub) | toupper(trimws(primary_sub)) != "SUB"]
cases <- raw[, .(master_wa = any(master & in_wa),
                 master_wa_piece = any(master & in_wa & suppressWarnings(as.numeric(piece_rate_offer)) > 0,
                                       na.rm = TRUE)), by = keys]
fields <- c(keys, "case_certified_target", "case_allocation_reconciled")
audit <- as.data.table(arrow::read_parquet(path_int("h2a_case_allocation_audit.parquet"),
                                          col_select = tidyselect::all_of(fields)))
audit <- audit[case_allocation_reconciled == TRUE & fiscal_year %between% c(2008L,2025L)]
stopifnot(!anyDuplicated(audit, by = keys), nrow(audit[!cases, on = keys]) == 0L)
cases <- merge(cases, audit[, .(fiscal_year, case_number, positions = case_certified_target)],
               by = keys, all.x = TRUE)
fields <- c(keys, "worksite_state", "adjusted_nbr_workers_certified")
sites <- as.data.table(arrow::read_parquet(path_int("h2a_allocated_worksites.parquet"),
                                          col_select = tidyselect::all_of(fields)))
sites <- sites[adjusted_nbr_workers_certified > 0 & toupper(trimws(worksite_state)) %chin% c("WA","WASHINGTON")]
wa <- unique(sites[, ..keys])
cases[, any_wa := master_wa]
cases[wa, on = keys, any_wa := TRUE]
annual <- cases[, .(filings = .N, positions = sum(positions, na.rm = TRUE),
                    principal_wa_filings = sum(master_wa), principal_wa_positions = sum(positions[master_wa], na.rm = TRUE),
                    numeric_piece_filings = sum(master_wa_piece),
                    numeric_piece_positions = sum(positions[master_wa_piece], na.rm = TRUE),
                    any_wa_filings = sum(any_wa), any_wa_case_positions = sum(positions[any_wa], na.rm = TRUE)),
                by = fiscal_year]
stopifnot(nrow(annual) == 18L)
print(annual[fiscal_year %in% 2024:2025])
print(annual[, .(all_filings = sum(filings), national_positions = sum(positions),
                 any_wa_fy24_25_share_filings = sum(any_wa_filings[fiscal_year >= 2024L])/sum(filings),
                 any_wa_fy24_25_share_positions = sum(any_wa_case_positions[fiscal_year >= 2024L])/sum(positions))])
