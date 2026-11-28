# Direct historical use by equipment filers. Run from the repository root.
source("code/paths.R")
source(path_code("c00_shared", "geography.R"))
library(data.table)
setDTthreads(2L)

history <- function(path, cols) {
  z <- dplyr::filter(arrow::open_dataset(path), fiscal_year >= 2021L, fiscal_year <= 2025L)
  as.data.table(dplyr::collect(dplyr::select(z, tidyselect::all_of(cols))))
}
keys <- c("fiscal_year", "case_number")
records <- history(path_int("h2a_disclosure_records.parquet"), c(keys, "primary_sub", "case_received_date",
  "job_title", "soc_code", "worksite_state", "nbr_workers_requested", "nbr_workers_certified",
  "case_status", "wage_rate", "wage_unit"))
ids <- history(path_int("policy_2025", "filer", "filer_ids.parquet"), c(keys, "filer_R7"))
agents <- history(path_int("policy_2025", "filer", "filer_cases.parquet"), c(keys, "agent_key"))
stopifnot(!anyDuplicated(ids[, ..keys]), !anyDuplicated(agents[, ..keys]))
states <- unique(as.data.table(arrow::read_parquet(path_processed("county_year_panel.parquet"),
                             col_select = c("state_abbrev", "state_fips"))))
assert_geo_columns(states, "state_fips")
stopifnot(!anyDuplicated(states$state_abbrev))
r <- records[primary_sub != "SUB"]
r[, receipt := as.IDate(case_received_date, format = "%Y-%m-%d")]
r[, season := as.integer(format(receipt, "%Y"))]
r[, month_day := format(receipt, "%m-%d")]
r[, annual_end := season + (month_day >= "03-01")]
r <- r[annual_end %in% 2022:2025 & month_day != "02-29"]
r <- merge(r, ids, by = keys, all.x = TRUE)
r <- merge(r, agents, by = keys, all.x = TRUE)
r <- merge(r, states, by.x = "worksite_state", by.y = "state_abbrev", all.x = TRUE)
r[, title := trimws(gsub("[^A-Z0-9]+", " ", toupper(job_title)))]
r[, soc := sub("^(\\d{2}-\\d{4})(\\.\\d{2})?$", "\\1", trimws(soc_code), perl = TRUE)]
r[, requested := suppressWarnings(as.numeric(nbr_workers_requested))]
r[, certified := suppressWarnings(as.numeric(nbr_workers_certified))]
r[is.na(certified) & grepl("DENIED|WITHDRAWN", toupper(case_status)), certified := 0]
r[, offer := suppressWarnings(as.numeric(wage_rate))]
r[!tolower(wage_unit) %in% c("hour", "hourly") | !is.finite(offer) | offer <= 0 | offer > 100, offer := NA_real_]
r[, agent := fifelse(!is.na(agent_key) & nzchar(trimws(agent_key)), agent_key, NA_character_)]
cat("Case and linkage coverage\n")
print(r[month_day <= "02-28", .(rows = .N, missing_filer = sum(is.na(filer_R7)), missing_state = sum(is.na(state_fips)),
  missing_requested = sum(is.na(requested)), missing_certified = sum(is.na(certified))), by = season])
r <- unique(r[!is.na(filer_R7) & !is.na(state_fips) & !is.na(case_number) & nzchar(case_number),
  .(case_number, filer = filer_R7, agent, state_fips, title, soc, season, annual_end, month_day, requested, certified, offer)])
cat("Conflicting cases excluded:", r[, .N, by = case_number][N > 1L, .N], "\n")
r <- r[!duplicated(case_number) & !duplicated(case_number, fromLast = TRUE)]
assert_geo_columns(r, "state_fips")
stopifnot(!anyDuplicated(r$case_number), all(is.finite(r$requested) & r$requested >= 0),
          all(r$certified >= 0, na.rm = TRUE))
annual <- r[, .(req = sum(requested)), by = .(filer, state_fips, annual_end)]
r <- r[month_day <= "02-28"]
# Extract wording only for the two preassigned baselines. Outcome files return
# no text, so an outcome-dependent qualification can never enter the cohort.
base_keys <- r[season %in% 2022:2023 & soc == "45-2091", unique(case_number)]
need <- c("CASE_NUMBER", "ADDITIONAL_JOB_REQUIREMENTS")
wording <- rbindlist(lapply(2022:2025, function(fy) {
  file <- path_raw("h2a", sprintf("H-2A_Disclosure_Data_FY%d_Q4.xlsx", fy))
  h <- names(readxl::read_excel(file, n_max = 0, .name_repair = "minimal"))
  stopifnot(all(need %in% h))
  tb <- as.data.table(readxl::read_excel(file, col_types = ifelse(h %in% need, "text", "skip")))
  tb[CASE_NUMBER %in% base_keys]
}))
wording <- unique(wording)
wording <- wording[, if (.N == 1L) .SD, by = CASE_NUMBER]
setnames(wording, need, c("case_number", "text"))
r <- merge(r, wording, by = "case_number", all.x = TRUE)
r[, text_key := trimws(gsub("[[:space:]]+", " ", tolower(text)))]
r[, text_known := !is.na(text_key) & nzchar(text_key)]
pattern <- "\\bcdl\\b|commercial driv|commercial motor|tractor[ -]?trailer|semi[ -]?truck|public road|26,?001"
r[, trucking := text_known & grepl(pattern, text_key)]
# Any one state's removal recomputes both the weights and the full contrast.
# Within a matched stratum, each group receives the baseline treated weight.
contrast <- function(frame, denom, numer, strata) {
  cells <- frame[is.finite(get(numer)) & is.finite(get(denom)),
    .(base = sum(get(denom)), after = sum(get(numer)), units = .N), by = c("state_fips", strata, "truck")]
  cast_vars <- c("state_fips", strata)
  wide <- dcast(cells, as.formula(paste(paste(cast_vars, collapse = " + "), "~ truck")), value.var = c("base", "after", "units"))
  wide <- wide[base_TRUE > 0 & base_FALSE > 0]
  wide[, control_at_treated_scale := after_FALSE * base_TRUE / base_FALSE]
  states <- wide[, .(b = sum(base_TRUE), p = sum(after_TRUE), c = sum(control_at_treated_scale),
                     exposed = sum(units_TRUE), control = sum(units_FALSE)), by = state_fips]
  g <- nrow(states)
  if (g < 3L) return(data.table(states = g, exposed = sum(states$exposed), control = sum(states$control)))
  with(states, {
    dif <- p - c
    est <- sum(dif) / sum(b)
    loo <- (sum(dif) - dif) / (sum(b) - b)
    se <- sqrt((g - 1) / g * sum((loo - mean(loo))^2))
    data.table(states = g, exposed = sum(exposed), control = sum(control), max_state_share = max(b) / sum(b),
      observed_ratio = sum(p) / sum(b), comparison_ratio = sum(c) / sum(b),
      difference = est, lower = est - qt(.975, g - 1L) * se, upper = est + qt(.975, g - 1L) * se,
      min_delete = min(loo), max_delete = max(loo))
  })
}
firm <- c("filer", "state_fips")
all_rows <- list()
for (b in 2022:2023) {
  base <- r[season == b]
  templates <- base[soc == "45-2091" & text_known, .(users = uniqueN(filer)), by = text_key]
  base[, common_text := text_key %in% templates[users >= 3, text_key]]
  base_units <- base[, .(all_equipment = all(soc == "45-2091", na.rm = FALSE), complete_text = all(text_known),
    truck = any(trucking), common_text = any(common_text), fully_named = all(!is.na(title) & nzchar(title)),
    agent = if (uniqueN(agent) == 1L) first(agent) else NA_character_,
    cases0 = .N, req0 = sum(requested), cert0 = sum(certified),
    wage0 = if (any(is.finite(offer))) median(offer, na.rm = TRUE) else NA_real_), by = firm]
  cat("Baseline year, all filer states, all-equipment, complete-text eligible\n")
  print(base_units[, .(baseline = b, total = .N, equipment = sum(all_equipment %in% TRUE),
    eligible = sum(all_equipment %in% TRUE & complete_text & fully_named & req0 > 0))])
  cohort <- base_units[all_equipment == TRUE & complete_text & fully_named & req0 > 0]
  stopifnot(!anyDuplicated(cohort[, ..firm]))
  print(cohort[, .(filers_states = .N, baseline_requests = sum(req0), median_size = median(req0),
                   mean_size = mean(req0), common_text = mean(common_text)), by = truck])
  old_titles <- unique(merge(base, cohort[, ..firm], by = firm)[, .(filer, state_fips, title)])
  for (y in if (b == 2022) 2023 else 2024:2025) {
    year_frame <- merge(cohort[, .(filer, state_fips, truck)], annual[annual_end == b, .(filer, state_fips, req0 = req)], by = firm, all.x = TRUE)
    year_frame <- merge(year_frame, annual[annual_end == y, .(filer, state_fips, req1 = req)], by = firm, all.x = TRUE)
    year_frame[is.na(req1), req1 := 0]
    stopifnot(!anyNA(year_frame$req0), nrow(year_frame) == nrow(cohort))
    a <- contrast(year_frame, "req0", "req1", character())
    all_rows[[length(all_rows) + 1L]] <- cbind(data.table(base = b, follow = y, scope = "full_year", spec = "state", outcome = "req"), a)
    next_cases <- merge(r[season == y], cohort[, ..firm], by = firm)
    for (scope in c("all_titles", "same_titles")) {
      later <- if (scope == "all_titles") next_cases else merge(next_cases, old_titles, by = c(firm, "title"))
      totals <- later[, .(cases1 = .N, req1 = sum(requested), cert1 = sum(certified),
        truck_soc = any(soc == "53-3032"), wage1 = if (any(is.finite(offer))) median(offer, na.rm = TRUE) else NA_real_), by = firm]
      ff <- merge(cohort, totals, by = firm, all.x = TRUE)
      ff[is.na(cases1), c("cases1", "req1", "cert1", "truck_soc") := list(0L, 0, 0, FALSE)]
      ff[, `:=`(return0 = 1, return1 = as.integer(cases1 > 0), truck_soc0 = 1, truck_soc1 = as.integer(truck_soc))]
      stopifnot(nrow(ff) == nrow(cohort), !anyNA(ff$req1))
      if (scope == "all_titles") {
        cat("Outcomes by baseline wording, baseline", b, "outcome season", y, "\n")
        print(ff[, .(filers_states = .N, return_share = mean(cases1 > 0), heavy_truck_share = mean(truck_soc),
          req_before = sum(req0), req_after = sum(req1), wages_paired = sum(is.finite(wage0) & is.finite(wage1)),
          median_offer_change = median(wage1 - wage0, na.rm = TRUE)), by = truck])
      }
      for (spec in c("state", "same_agent", "exclude_shared_text")) {
        if (scope == "same_titles" && spec != "state") next
        fr <- switch(spec, state = ff, same_agent = ff[!is.na(agent)], exclude_shared_text = ff[common_text == FALSE])
        st <- if (spec == "same_agent") "agent" else character()
        outcomes <- if (scope == "all_titles" && spec == "state") c("req", "cert", "cases", "return", "truck_soc") else "req"
        for (out in outcomes) {
          a <- contrast(fr, paste0(out, "0"), paste0(out, "1"), st)
          all_rows[[length(all_rows) + 1L]] <- cbind(data.table(base = b, follow = y, scope, spec, outcome = out), a)
        }
      }
    }
  }
}
cat("State-standardized follow-up / baseline changes and differences\n")
print(rbindlist(all_rows, fill = TRUE), digits = 4)
