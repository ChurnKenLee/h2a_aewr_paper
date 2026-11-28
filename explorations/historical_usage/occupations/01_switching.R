# Descriptive tests of changes in recorded SOC on apparently stable titles.
source("code/paths.R")
source(path_code("c00_shared", "geography.R"))
library(data.table)
setDTthreads(2L)
historical <- function(path, cols) {
  x <- arrow::open_dataset(path)
  x <- dplyr::filter(x, fiscal_year >= 2018L & fiscal_year <= 2025L)
  as.data.table(dplyr::collect(dplyr::select(x, tidyselect::all_of(cols))))
}
keys <- c("fiscal_year", "case_number")
raw <- historical(path_int("h2a_disclosure_records.parquet"), c(keys, "primary_sub", "job_title",
  "soc_code", "worksite_state", "case_received_date", "wage_rate", "wage_unit"))
ids <- historical(path_int("policy_2025", "filer", "filer_ids.parquet"), c(keys, "filer_R7", "filer_R5"))
stopifnot(!anyDuplicated(ids[, ..keys]))
states <- unique(as.data.table(arrow::read_parquet(path_processed("county_year_panel.parquet"),
                   col_select = c("state_abbrev", "state_fips"))))
assert_geo_columns(states, "state_fips")
stopifnot(!anyDuplicated(states$state_abbrev))
raw <- raw[primary_sub != "SUB" & !is.na(case_number) & nzchar(case_number)]
d <- merge(raw, ids, by = keys, all.x = TRUE)
d[, soc := sub("^(\\d{2}-\\d{4})(\\.\\d{2})?$", "\\1", trimws(soc_code), perl = TRUE)]
d[!grepl("^\\d{2}-\\d{4}$", soc, perl = TRUE), soc := NA_character_]
d[, title := trimws(gsub("[^A-Z0-9]+", " ", toupper(job_title)))]
d[, receipt := as.IDate(case_received_date, format = "%Y-%m-%d")]
d[, hourly := tolower(trimws(wage_unit)) %in% c("hour", "hourly")]
d[, offer := suppressWarnings(as.numeric(wage_rate))]
d[!hourly | !is.finite(offer) | offer <= 0 | offer > 100, offer := NA_real_]
d <- merge(d, states, by.x = "worksite_state", by.y = "state_abbrev", all.x = TRUE)
cat("Primary disclosure rows by source FY and available SOC, title, receipt, R7 filer and eligible state\n")
print(d[, .(rows = .N, usable = sum(!is.na(soc) & nzchar(title) & !is.na(receipt) &
  !is.na(filer_R7) & !is.na(state_fips))), by = fiscal_year][order(fiscal_year)])
d <- unique(d[, .(case_number, receipt, soc, title, filer_R7, filer_R5, state_fips, hourly, offer)])
# Drop all conflicting versions rather than choosing one across disclosure years.
cat("Conflicting cases excluded:", d[, .N, by = case_number][N > 1L, .N], "\n")
d <- d[, if (.N == 1L) .SD, by = case_number]
d <- d[!is.na(receipt) & !is.na(filer_R7) & !is.na(soc) & !is.na(title) & nzchar(title) & !is.na(state_fips)]
stopifnot(!anyDuplicated(d$case_number))
assert_geo_columns(d, "state_fips")
big_six <- c("45-2041", "45-2091", "45-2092", "45-2093", "45-2099", "53-7064")
d[, field := soc %in% big_six]
d[, monthday := format(receipt, "%m-%d")]
d[, receipt_year := as.integer(format(receipt, "%Y"))]
windows <- list(
  july_feb = d[monthday >= "07-01" | monthday <= "02-28"][, season := receipt_year - (monthday <= "02-28")][season %in% 2018:2024],
  may_august = d[monthday >= "05-01" & monthday <= "08-15"][, season := receipt_year][season %in% 2018:2025]
)
rates <- as.data.table(arrow::read_parquet(path_int("aewr_oews_occupation_rates.parquet")))
fls <- as.data.table(arrow::read_parquet(path_int("aewr_rate_schedule.parquet")))
first_date <- as.Date("2023-03-30")
rates <- rates[effective_start == first_date, .(state_fips, soc = soc_code, occupation_rate = hourly_aewr)]
fls <- fls[effective_start <= first_date & effective_end_exclusive > first_date, .(state_fips, field_rate = aewr)]
assert_geo_columns(rates, "state_fips")
assert_geo_columns(fls, "state_fips")
stopifnot(!anyDuplicated(rates[, .(state_fips, soc)]), !anyDuplicated(fls$state_fips))
rates <- merge(rates, fls, by = "state_fips")
rates[, schedule_gap := occupation_rate - field_rate]

for (window in names(windows)) for (tier in c("filer_R7", "filer_R5")) {
  cases <- windows[[window]]
  cases <- cases[!is.na(get(tier))]
  cases[, filer := get(tier)]
  # A mixed class (both field and non-field cases under this title) stays visible in support.
  units <- cases[, .(mixed = uniqueN(field) > 1L, field = first(field),
    soc = if (uniqueN(soc) == 1L) first(soc) else NA_character_,
    offer = if (any(is.finite(offer))) median(offer, na.rm = TRUE) else NA_real_,
    cases = .N), by = .(filer, state_fips, title, season)]
  cell_keys <- c("filer", "state_fips", "title", "season")
  stopifnot(!anyDuplicated(units[, ..cell_keys]))
  prev <- copy(units)
  prev[, season := season + 1L]
  pairs <- merge(prev, units, by = cell_keys, suffixes = c("_pre", "_post"))
  recurring <- pairs[, .(recurring = .N, ambiguous = sum(mixed_pre | mixed_post)), by = season]
  pairs <- pairs[!mixed_pre & !mixed_post]
  pairs[, direction := paste0(ifelse(field_pre, "field", "other"), "_", ifelse(field_post, "field", "other"))]
  cat("\nTime window and filer tier:", window, tier, "\n")
  flows <- pairs[, .(other_at_risk = sum(!field_pre), to_field = sum(!field_pre & field_post),
    pct_to_field = 100 * mean(field_post[!field_pre]), field_at_risk = sum(field_pre),
    to_other = sum(field_pre & !field_post), pct_to_other = 100 * mean(!field_post[field_pre]),
    net_to_field = sum(field_post) - sum(field_pre)), by = season]
  print(merge(recurring, flows, by = "season"))
  if (window != "july_feb" || tier != "filer_R7") next

  print(units[, .(units = .N, mixed = sum(mixed), cases = sum(cases)), by = season][order(season)])
  # Use the same cells for the 2021->2022 placebo and 2022->2023 event.
  same_cells <- unique(pairs[season == 2022L, .(filer, state_fips, title)])
  all_three <- merge(same_cells, pairs[season == 2023L, .(filer, state_fips, title)],
                     by = c("filer", "state_fips", "title"))
  cat("Same three-season titles and filers\n")
  stable <- pairs[all_three, on = .(filer, state_fips, title), nomatch = NULL][season %in% 2022:2023]
  print(stable[, .(other_at_risk = sum(!field_pre), to_field = sum(!field_pre & field_post),
                   field_at_risk = sum(field_pre), to_other = sum(field_pre & !field_post)), by = season])
  cat("Most frequent switching titles for July 2022-Feb 2023 to July 2023-Feb 2024\n")
  event <- pairs[season == 2023L]
  print(event[field_pre != field_post, .(cells = .N, filers = uniqueN(filer), states = uniqueN(state_fips)),
        by = .(direction, title)][order(direction, -cells, title)][, head(.SD, 5L), by = direction])
  cat("Within-cell hourly offer changes (nominal dollars per hour) by recorded transition\n")
  print(pairs[season %in% 2021:2024 & is.finite(offer_pre) & is.finite(offer_post), .(cells = .N,
    med_before = median(offer_pre), med_after = median(offer_post),
    median_change = median(offer_post - offer_pre)), by = .(season, direction)][order(season,direction)])
  cat("Recorded SOC transitions and same-state OEWS offer matches among field-to-other cells\n")
  print(event[field_pre != field_post, .(cells = .N), by = .(direction, soc_pre, soc_post)
        ][order(direction, -cells)][, head(.SD, 5L), by = direction])
  # This vintage spans the entire post-event July 2023 to February 2024 receipt window.
  post_rate <- as.data.table(arrow::read_parquet(path_int("aewr_oews_occupation_rates.parquet")))
  post_rate <- post_rate[effective_start == as.Date("2023-07-01"), .(state_fips, soc_post = soc_code, rate = hourly_aewr)]
  stopifnot(!anyDuplicated(post_rate[, .(state_fips, soc_post)]))
  audited <- merge(event[field_post == FALSE], post_rate, by = c("state_fips", "soc_post"), all.x = TRUE)
  print(audited[, .(total = .N, matched = sum(is.finite(rate) & is.finite(offer_post)),
          close = sum(abs(offer_post - rate) <= .050001, na.rm = TRUE)), by = direction])
  # Raw disclosure requirement fields independently describe both periods.
  # A phrase match counts a mention, even if the text qualifies the requirement.
  eq <- "45-2091"
  truck <- "53-3032"
  repeated_jobs <- pairs[season %in% 2022:2024 & (
    (soc_pre == eq & soc_post %in% c(eq, truck)) | (soc_pre == truck & soc_post == truck) |
    (soc_pre == "45-2092" & soc_post == "45-2092")),
    .(filer, state_fips, title, transition_year = season, path = paste(soc_pre, soc_post, sep = " -> "))]
  compare <- rbind(copy(repeated_jobs)[, season := transition_year - 1L],
                   copy(repeated_jobs)[, season := transition_year])
  case_lookup <- merge(cases, compare, by = c("filer", "state_fips", "title", "season"),
                       allow.cartesian = TRUE)
  header <- c("CASE_NUMBER", "DRIVER_REQUIREMENTS", "ADDITIONAL_JOB_REQUIREMENTS")
  raw_req <- rbindlist(lapply(2021:2025, function(fy) {
    suffix <- if (fy == 2021L) "" else "_Q4"
    file <- path_raw("h2a", sprintf("H-2A_Disclosure_Data_FY%d%s.xlsx", fy, suffix))
    cols <- names(readxl::read_excel(file, n_max = 0, .name_repair = "minimal"))
    required <- if (fy == 2021L) header[1:2] else header
    stopifnot(all(required %in% cols))
    tb <- as.data.table(readxl::read_excel(file, col_types = ifelse(cols %in% header, "text", "skip")))
    if (!"ADDITIONAL_JOB_REQUIREMENTS" %in% names(tb)) tb[, ADDITIONAL_JOB_REQUIREMENTS := NA_character_]
    tb[CASE_NUMBER %in% case_lookup$case_number]
  }))
  raw_req <- unique(raw_req)
  # Conflicting versions are unobserved for both indicators.
  raw_req <- raw_req[, if (.N == 1L) .SD, by = CASE_NUMBER]
  setnames(raw_req, header, c("case_number", "driving", "detail"))
  obs <- merge(case_lookup, raw_req, by = "case_number", all.x = TRUE)
  obs[, driver := fifelse(driving %in% c("Y", "N"), as.numeric(driving == "Y"), NA_real_)]
  obs[, detail_known := !is.na(detail) & nzchar(trimws(detail))]
  pattern <- "\\bcdl\\b|commercial driv|commercial motor|tractor[ -]?trailer|semi[ -]?truck|public road|26,?001"
  obs[, mention := fifelse(detail_known, as.numeric(grepl(pattern, tolower(detail))), NA_real_)]
  obs[, timing := ifelse(season == transition_year, "after", "before")]
  facts <- obs[, .(driver = mean(driver, na.rm = TRUE), mention = mean(mention, na.rm = TRUE),
                   usable_text = sum(detail_known)), by = .(transition_year, filer, state_fips, title, path, timing)]
  paired <- dcast(facts, transition_year + filer + state_fips + title + path ~ timing, value.var = c("driver", "mention", "usable_text"))
  cat("Paired requirement fields on equipment and truck transitions, equal cell weights\n")
  for (metric in c("driver", "mention")) {
    pre <- paste0(metric, "_before")
    post <- paste0(metric, "_after")
    cat(metric, "\n")
    print(paired[is.finite(get(pre)) & is.finite(get(post)), .(cells = .N, pre = mean(get(pre)), post = mean(get(post)),
      yes_both = sum(get(pre) == 1 & get(post) == 1)), by = .(transition_year, path)][order(transition_year, path)])
  }
  equipment <- paired[transition_year == 2023 & path == paste(eq, eq, sep = " -> ") &
                      is.finite(mention_before) & is.finite(mention_after)]
  state_changes <- equipment[, .(cells = .N, delta = sum(mention_after - mention_before)), by = state_fips]
  cat("Stable-equipment mention change after omitting each state, one at a time\n")
  print(range((sum(state_changes$delta) - state_changes$delta) /
              (sum(state_changes$cells) - state_changes$cells)))
  wording <- obs[transition_year == 2023 & path == paste(eq, eq, sep = " -> ") & mention == 1]
  wording[, text_key := trimws(gsub("[[:space:]]+", " ", tolower(detail)))]
  templates <- wording[, .(filings = .N, filers = uniqueN(filer)), by = .(timing, text_key)]
  cat("Concentration of exact text among stable equipment cases with trucking-related words\n")
  print(templates[, .(cases = sum(filings), distinct_texts = .N, largest = max(filings),
                      largest_three = sum(head(sort(filings, decreasing = TRUE), 3))), by = timing])
  # A directional falsification. Assign the launch schedule to both placebo and rule transitions.
  other <- pairs[!field_pre & season %in% 2021:2024]
  gap <- merge(other, rates[, .(state_fips, soc_pre = soc, schedule_gap)], by = c("state_fips", "soc_pre"), all.x = TRUE)
  gap[, gap_band := fcase(!is.finite(schedule_gap), "unresolved", schedule_gap <= 0, "zero_or_negative",
                          schedule_gap <= 5, "positive_0_to_5", default = "above_5")]
  cat("Transitions into field by former SOC March 2023 scheduled premium over field rate\n")
  print(gap[, .(cells = .N, switches = sum(field_post), rate = mean(field_post)), by = .(season, gap_band)][order(season,gap_band)])
  print(event[field_pre != field_post, .(cells = .N, filers = uniqueN(filer), states = uniqueN(state_fips),
    biggest_state = max(table(state_fips)), biggest_filer = max(table(filer))), by = direction])
}
