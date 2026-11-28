# Review of possible reporting and filing-agent explanations for the 2023 wording change.
# Run at the repository root with the project R library. Only aggregate diagnostics are printed.
source("code/paths.R")
source(path_code("c00_shared", "geography.R"))
library(data.table)
setDTthreads(2L)

historical <- function(path, columns) {
  x <- dplyr::filter(arrow::open_dataset(path), fiscal_year >= 2022L, fiscal_year <= 2025L)
  as.data.table(dplyr::collect(dplyr::select(x, tidyselect::all_of(columns))))
}
keys <- c("fiscal_year", "case_number")
cases <- historical(path_int("h2a_disclosure_records.parquet"),
  c(keys, "case_received_date", "worksite_state", "job_title", "soc_code", "primary_sub"))
ids <- historical(path_int("policy_2025", "filer", "filer_ids.parquet"), c(keys, "filer_R7"))
agents <- historical(path_int("policy_2025", "filer", "filer_cases.parquet"), c(keys, "agent_key"))
states <- unique(as.data.table(arrow::read_parquet(path_processed("county_year_panel.parquet"),
          col_select = c("state_abbrev", "state_fips"))))
assert_geo_columns(states, "state_fips")
stopifnot(!anyDuplicated(ids[, ..keys]), !anyDuplicated(agents[, ..keys]), !anyDuplicated(states$state_abbrev))
cases <- cases[primary_sub != "SUB" & !is.na(case_number) & nzchar(case_number)]
cases[, received := as.IDate(case_received_date, "%Y-%m-%d")]
cases[, season := fcase(between(received, as.IDate("2022-07-01"), as.IDate("2023-02-28")), "before",
                        between(received, as.IDate("2023-07-01"), as.IDate("2024-02-28")), "after")]
cases <- cases[!is.na(season)]
cases[, title := trimws(gsub("[^A-Z0-9]+", " ", toupper(job_title)))]
cases[, soc := sub("^(\\d{2}-\\d{4})(\\.\\d{2})?$", "\\1", trimws(soc_code), perl = TRUE)]
cases <- cases[grepl("^\\d{2}-\\d{4}$", soc, perl = TRUE) & !is.na(title) & nzchar(title)]
d <- merge(cases, ids, by = keys)
d <- merge(d, agents, by = keys, all.x = TRUE)
d <- merge(d, states, by.x = "worksite_state", by.y = "state_abbrev")
d <- d[!is.na(filer_R7)]
# Keep each case's single unambiguous identity and season across versions.
info <- c("season", "filer_R7", "state_fips", "title", "soc", "agent_key")
d <- unique(d[, c(keys, info), with = FALSE])
duplicate_ids <- d[, .N, by = case_number][N > 1L, case_number]
d <- d[!case_number %in% duplicate_ids]
group <- c("filer_R7", "state_fips", "title")
u <- d[, .(soc = if (uniqueN(soc) == 1L) first(soc) else NA_character_), by = c(group, "season")]
w <- dcast(u, filer_R7 + state_fips + title ~ season, value.var = "soc")
w <- w[before == after & before %in% c("45-2091", "45-2092", "53-3032")]
w[, occupation := fcase(before == "45-2091", "equipment", before == "45-2092", "crop", default = "truck")]
d <- merge(d, w[, c(group, "occupation"), with = FALSE], by = group)
stopifnot(!anyDuplicated(d$case_number))

req <- rbindlist(lapply(2022:2025, function(fy) {
  file <- path_raw("h2a", sprintf("H-2A_Disclosure_Data_FY%d_Q4.xlsx", fy))
  h <- names(readxl::read_excel(file, n_max = 0L, .name_repair = "minimal"))
  cols <- c("CASE_NUMBER", "ADDITIONAL_JOB_REQUIREMENTS")
  stopifnot(all(cols %in% h))
  batch <- as.data.table(readxl::read_excel(file, col_types = ifelse(h %in% cols, "text", "skip")))
  batch <- unique(batch[CASE_NUMBER %in% d$case_number])
  batch <- batch[!duplicated(CASE_NUMBER) & !duplicated(CASE_NUMBER, fromLast = TRUE)]
  batch[, .(case_number = CASE_NUMBER, fiscal_year = fy, detail = ADDITIONAL_JOB_REQUIREMENTS, raw_match = TRUE)]
}))
stopifnot(!anyDuplicated(req[, ..keys]))
d <- merge(d, req, by = keys, all.x = TRUE)
stopifnot(!anyDuplicated(d$case_number))
d[, known := !is.na(detail) & nzchar(trimws(detail))]
d[, chars := fifelse(known, as.numeric(nchar(detail)), NA_real_)]
d[, text := tolower(detail)]
truck_regex <- "\\bcdl\\b|commercial driv|commercial motor|tractor[ -]?trailer|semi[ -]?truck|public road|26,?001"
patterns <- c(trucking = truck_regex, lift = "\\blift(ing|s)?\\b|\\bpounds?\\b",
              experience = "\\bexperience[d]?\\b", drug_alcohol = "\\bdrug(s)?\\b|\\balcohol\\b",
              attachment = "\\baddend(um|a)\\b|\\battachment(s)?\\b")
for (label in names(patterns)) {
  d[, (label) := fifelse(known, as.numeric(grepl(patterns[[label]], text, perl = TRUE)), NA_real_)]
}
first_match <- regexpr(truck_regex, d$text, perl = TRUE)
d[, first_end := as.integer(first_match) + attr(first_match, "match.length") - 1L]
for (n in c(200L, 500L)) d[, (paste0("first", n)) :=
  fifelse(known, as.numeric(trucking == 1 & first_end <= n), NA_real_)]
stopifnot(all(d$first200 <= d$first500 & d$first500 <= d$trucking, na.rm = TRUE))
# Exact person name key on every filing in both seasons. A key is a proxy for the recorded agent.
d[, agent_valid := !is.na(agent_key) & nzchar(trimws(agent_key))]
d[, one_agent := all(agent_valid) && uniqueN(agent_key) == 1L, by = group]
metrics <- c("known", "chars", names(patterns), "first200", "first500")
summarize_cells <- function(records, subset_label) {
  s <- records[, lapply(.SD, function(x) mean(x, na.rm = TRUE)),
               by = c(group, "occupation", "season"), .SDcols = metrics]
  long <- melt(s, id.vars = c(group, "occupation", "season"), variable.name = "metric")
  z <- dcast(long, filer_R7 + state_fips + title + occupation + metric ~ season, value.var = "value")
  out <- z[is.finite(before) & is.finite(after), .(cells = .N, before = mean(before),
               after = mean(after), difference = mean(after - before)), by = .(occupation, metric)]
  out[, scope := subset_label]
  out
}
cat("Raw join, length distribution and nonblank coverage by occupation and season\n")
print(d[, .(cases = .N, linked = sum(raw_match %in% TRUE), nonblank = sum(known),
            p10_chars = quantile(chars, .1, na.rm = TRUE), median_chars = median(chars, na.rm = TRUE),
            p90_chars = quantile(chars, .9, na.rm = TRUE), max_chars = max(chars, na.rm = TRUE)),
        by = .(occupation, season)][order(occupation, season)])
summary <- rbind(summarize_cells(d, "all"), summarize_cells(d[(one_agent)], "same_named_agent"),
                 summarize_cells(d[fiscal_year == 2023], "same_FY2023_workbook"))
cat("Before and after on complete paired cells (proportions or characters)\n")
print(summary[order(scope, occupation, metric)], nrows = Inf)
# A leave-one-agent-out diagnostic checks whether a dominant agent drives the matched result.
equip <- d[occupation == "equipment" & one_agent == TRUE]
avg <- equip[, .(fraction = mean(trucking, na.rm = TRUE)), by = c(group, "agent_key", "season")]
z <- dcast(avg, filer_R7 + state_fips + title + agent_key ~ season, value.var = "fraction")
z <- z[is.finite(before) & is.finite(after)]
a <- z[, .(cells = .N, loss = sum(before - after)), by = agent_key]
cat("Same-agent stable equipment, agent concentration and deletion range\n")
print(a[, .(agents = .N, cells = sum(cells), largest_agent_share = max(cells)/sum(cells),
            largest_loss_share = max(loss)/sum(loss),
            agents_declining = sum(loss > 1e-8), agents_increasing = sum(loss < -1e-8),
            equal_agent_drop = mean(loss/cells),
            smallest_drop = min((sum(loss) - loss)/(sum(cells) - cells)),
            largest_drop = max((sum(loss) - loss)/(sum(cells) - cells)))])
# Whole-field strings can be unique even when filers copy the trucking clause.
# Count recurring short text snippets around the first match without displaying them.
phrases <- d[occupation == "equipment" & trucking == 1]
phrases[, start := as.integer(regexpr(truck_regex, text, perl = TRUE))]
phrases[, fragment := substr(text, pmax(1L, start - 35L), first_end + 35L)]
phrases[, fragment := trimws(gsub("[^a-z]+", " ", fragment))]
forms <- phrases[, .(cases = .N, filers = uniqueN(filer_R7)), by = .(season, fragment)]
cat("Concentration of first trucking mention and surrounding 35 characters on each side\n")
print(forms[, .(cases = sum(cases), phrases = .N, biggest_share = max(cases)/sum(cases),
    top_three_share = sum(head(sort(cases, decreasing = TRUE), 3L))/sum(cases),
    share_shared_by_three_filers = sum(cases[filers >= 3L])/sum(cases)), by = season])
