# Identification search, Phase 0, step 1: shared outcome build.
#
# Builds county x federal FY, employer x FY and county x calendar start-year
# outcome panels for FY2008-FY2026 Q3, plus the case-row table they are
# aggregated from. Exploratory: reads retained intermediates only and writes
# only to the scratch directory given as the first argument (outside the repo).
#
# Usage (repository root):
#   R_LIBS_USER=$(ls -d $PWD/renv/library/*/R-*/* | head -1) \
#     Rscript --vanilla explorations/identification_search/shared/01_outcomes.R "$S/shared"
#
# Construction (declared in explorations/identification_search/shared/declarations.json):
# - Certified positions and hours: b01 fiscal-year logic
#   (code/b01_derived/01_h2a_aggregation_nodupes.R): allocated worksite rows
#   (h2a_allocated_worksites.parquet; denied/withdrawn already zeroed), rows
#   with a period begin date, hours = (end - begin)/7 x weekly hours x
#   positions, missing weekly hours replaced by the pooled mean over all such
#   rows, each row split equally across its county_fips_list.
# - Requested positions and hours: ALL disclosure cases, including denied and
#   withdrawn. Cases in the allocated file use its b01-adjusted requested
#   count; cases absent from it (denied, withdrawn, zero certified) are
#   located with the b01 case-allocation source (pre-FY2020) or Addendum B
#   worksites (FY2020+; requested-weighted, equal if unknown).
# - Requested hours = requested positions x requested weeks x weekly hours.
#   Requested weeks come from worksite/case requested dates and fall back to
#   the period dates when those are missing or invalid (share reported).
# - Windows: fy (disclosure FY, all decisions; b01), dq13 (disclosure FY,
#   decisions 1 Oct - 30 Jun; 04_track1_power.R), rfy (FY of receipt, all
#   decisions), rq13 (received 1 Oct - 30 Jun of the receipt FY, any decision
#   date), rq13d (received and decided 1 Oct - 30 Jun of the receipt FY: the
#   only receipt window complete in FY2026).
# - FY2026 exists only through decision Q3 (30 Jun 2026). fy, rfy and rq13
#   are set to NA in FY2026 because they are incomplete; dq13 and rq13d are
#   complete.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(tidyr); library(readr)})

out_dir <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(out_dir) || !nzchar(out_dir)) stop("Supply a scratch output directory outside the repository")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(out_dir)
if (startsWith(paste0(out_dir, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")

FY_MIN <- 2008L; FY_MAX <- 2026L; LAST_COMPLETE_FY <- 2025L
REQ_MISSING_FY <- c(2008L, 2009L, 2010L, 2013L, 2014L)
AEWR_TOL <- 0.05            # b01 AEWR_OFFER_TOLERANCE
HOURLY_WAGE_MAX <- 100      # b01 HOURLY_WAGE_MAX
MAX_REQ_DAYS <- 400L        # requested-date spans above this fall back to period dates
WINDOWS <- c("fy", "dq13", "rfy", "rq13", "rq13d")
num <- function(x) suppressWarnings(as.numeric(x))
blank_na <- function(x) { x <- as.character(x); x[!is.na(x) & trimws(x) == ""] <- NA_character_; x }
as_date <- function(x) suppressWarnings(as.Date(blank_na(x)))
fed_fy <- function(d) as.integer(format(d, "%Y")) + as.integer(as.integer(format(d, "%m")) >= 10L)
plausible <- function(value, fy) { y <- as.integer(format(value, "%Y")); value[!is.na(y) & abs(y - fy) > 1L] <- as.Date(NA); value }
first_nonempty <- function(x) { x <- x[!is.na(x) & trimws(x) != ""]; if (length(x)) x[[1]] else NA_character_ }
write_csv_out <- function(x, name) { write_csv(x, file.path(out_dir, name)); invisible(x) }

# ---- Case-level dates and status (all disclosure cases) --------------------
disc <- read_parquet(path_int("h2a_disclosure_records.parquet"), col_select = c("case_number", "fiscal_year",
  "case_received_date", "decision_date", "case_status", "nbr_workers_requested", "nbr_workers_needed",
  "nbr_workers_certified", "number_of_hours", "requested_begin_date", "requested_end_date",
  "period_begin_date", "period_end_date", "wage_rate", "wage_unit", "soc_code")) %>%
  filter(fiscal_year >= FY_MIN, fiscal_year <= FY_MAX)
cases <- disc %>% group_by(case_number, fiscal_year) %>% summarise(
  received = first_nonempty(case_received_date), decided = first_nonempty(decision_date),
  case_status = first_nonempty(case_status), .groups = "drop") %>%
  mutate(received = plausible(as_date(received), fiscal_year), decided = plausible(as_date(decided), fiscal_year),
    status = case_when(grepl("PARTIAL", toupper(case_status)) ~ "partial", grepl("DENIED", toupper(case_status)) ~ "denied",
      grepl("WITHDRAWN", toupper(case_status)) ~ "withdrawn", grepl("CERTIF", toupper(case_status)) ~ "certified", TRUE ~ "other"))
stopifnot(!anyDuplicated(cases[c("case_number", "fiscal_year")]))

# nbr_workers_needed coverage: can it stand in for requested positions?
needed_cov <- disc %>% group_by(fiscal_year) %>% summarise(rows = n(),
  requested_nonmissing_share = mean(is.finite(num(nbr_workers_requested))),
  needed_nonmissing_share = mean(is.finite(num(nbr_workers_needed))),
  sum_requested = sum(num(nbr_workers_requested), na.rm = TRUE), sum_needed = sum(num(nbr_workers_needed), na.rm = TRUE),
  needed_equals_requested_share = mean(num(nbr_workers_needed) == num(nbr_workers_requested), na.rm = TRUE),
  needed_ge_requested_share = mean(num(nbr_workers_needed) >= num(nbr_workers_requested), na.rm = TRUE),
  needed_requested_corr = { a <- num(nbr_workers_needed); b <- num(nbr_workers_requested); ok <- is.finite(a) & is.finite(b); if (sum(ok) > 2) cor(a[ok], b[ok]) else NA_real_ },
  weekly_hours_nonmissing_share = mean(is.finite(num(number_of_hours))), .groups = "drop") %>%
  mutate(needed_to_requested_ratio = ifelse(sum_requested > 0, sum_needed / sum_requested, NA_real_))
write_csv_out(needed_cov, "outcome_needed_coverage.csv")

# ---- Allocated (positive-certified) rows: b01 logic ------------------------
aw <- read_parquet(path_int("h2a_allocated_worksites.parquet"), col_select = c("allocation_id", "case_number", "fiscal_year",
  "county_fips_list", "adjusted_nbr_workers_certified", "adjusted_nbr_workers_requested", "period_begin_date",
  "period_end_date", "requested_begin_date", "requested_end_date", "worksite_requested_begin_date",
  "worksite_requested_end_date", "number_of_hours", "wage_rate", "wage_unit", "soc_code", "employer_id_balanced",
  "allocation_source_dataset")) %>%
  filter(fiscal_year >= FY_MIN, fiscal_year <= FY_MAX) %>%
  mutate(begin = as_date(period_begin_date), end = as_date(period_end_date)) %>%
  filter(!is.na(begin))
pooled_nh <- mean(num(aw$number_of_hours), na.rm = TRUE)  # b01 pooled mean over all rows with a begin date
aw <- aw %>% transmute(row_source = "allocated", case_number, fiscal_year, county_fips_list, begin, end,
  rbegin = coalesce(as_date(worksite_requested_begin_date), as_date(requested_begin_date)),
  rend = coalesce(as_date(worksite_requested_end_date), as_date(requested_end_date)),
  nh_obs = num(number_of_hours), cert_pos = num(adjusted_nbr_workers_certified),
  req_pos = num(adjusted_nbr_workers_requested), wage = num(wage_rate), wage_unit, soc_code,
  employer_id = blank_na(employer_id_balanced))
alloc_cases <- distinct(aw, case_number, fiscal_year)

# ---- Cases absent from the allocated file (denied, withdrawn, zero certified)
src <- read_parquet(path_int("h2a_case_allocation_source.parquet")) %>%
  filter(fiscal_year >= FY_MIN, fiscal_year <= FY_MAX) %>%
  anti_join(alloc_cases, by = c("case_number", "fiscal_year"))
case_extra <- disc %>% group_by(case_number, fiscal_year) %>% summarise(nh_case = first(na.omit(num(number_of_hours))),
  wage_case = first(na.omit(num(wage_rate))), wage_unit_case = first_nonempty(wage_unit), soc_case = first_nonempty(soc_code),
  .groups = "drop")
# Pre-FY2020: master total distributed over non-master rows (requested-weighted, else equal).
pre <- src %>% filter(fiscal_year < 2020) %>% group_by(case_number, fiscal_year) %>%
  mutate(n_rows = n(), master = n_rows > 1 & ((fiscal_year < 2016 & !is.na(organization_flag) & organization_flag != "") |
    (fiscal_year >= 2016 & !is.na(primary_sub) & primary_sub == "PRI")),
    case_req = if (n() == 1) nbr_workers_requested else if (any(master)) nbr_workers_requested[master][1] else sum(nbr_workers_requested, na.rm = TRUE),
    keep = if (n() > 1 && any(!master)) !master else TRUE) %>%
  filter(keep) %>%
  mutate(wt = ifelse(is.finite(nbr_workers_requested) & nbr_workers_requested > 0, nbr_workers_requested, 0),
    wt = if (sum(wt) > 0) wt / sum(wt) else rep(1 / n(), n()), req_pos = case_req * wt) %>% ungroup() %>%
  transmute(row_source = "unallocated_main", case_number, fiscal_year, county_fips_list, req_pos,
    begin = as_date(period_begin_date), end = as_date(period_end_date), rbegin = as_date(requested_begin_date), rend = as_date(requested_end_date))
# FY2020+: case total from the main row, located by Addendum B worksites.
post_main <- src %>% filter(fiscal_year >= 2020) %>% group_by(case_number, fiscal_year) %>%
  summarise(case_req = first(nbr_workers_requested), county_main = first(county_fips_list),
    begin = first(as_date(period_begin_date)), end = first(as_date(period_end_date)),
    rbegin = first(as_date(requested_begin_date)), rend = first(as_date(requested_end_date)), .groups = "drop")
addb <- read_parquet(path_int("h2a_addendum_b_with_fips.parquet"), col_select = c("case_number", "fiscal_year",
  "county_fips_list", "total_h2a_workers_requested", "worksite_requested_begin_date", "worksite_requested_end_date")) %>%
  semi_join(post_main, by = c("case_number", "fiscal_year"))
post_add <- addb %>% group_by(case_number, fiscal_year) %>%
  mutate(w = num(total_h2a_workers_requested), w = ifelse(is.finite(w) & w > 0, w, 0),
    wt = if (sum(w) > 0) w / sum(w) else rep(1 / n(), n())) %>% ungroup() %>%
  inner_join(post_main, by = c("case_number", "fiscal_year")) %>%
  transmute(row_source = "unallocated_addendum_b", case_number, fiscal_year, county_fips_list, req_pos = case_req * wt,
    begin, end, rbegin = coalesce(as_date(worksite_requested_begin_date), rbegin),
    rend = coalesce(as_date(worksite_requested_end_date), rend))
post_only <- post_main %>% anti_join(distinct(addb, case_number, fiscal_year), by = c("case_number", "fiscal_year")) %>%
  transmute(row_source = "unallocated_main", case_number, fiscal_year, county_fips_list = county_main, req_pos = case_req,
    begin, end, rbegin, rend)
un <- bind_rows(pre, post_add, post_only) %>%
  left_join(case_extra, by = c("case_number", "fiscal_year")) %>%
  mutate(cert_pos = 0, nh_obs = nh_case, wage = wage_case, wage_unit = wage_unit_case, soc_code = soc_case) %>%
  select(-nh_case, -wage_case, -wage_unit_case, -soc_case)
# Employer ids for unallocated cases: worksite-business members (equal split across ids).
memb <- read_parquet(path_int("h2a_employer_record_membership.parquet"), col_select = c("case_number", "fiscal_year",
  "employer_is_worksite_business", "employer_is_alias_evidence_only", "employer_id_balanced")) %>%
  semi_join(distinct(un, case_number, fiscal_year), by = c("case_number", "fiscal_year")) %>%
  filter(!is.na(employer_id_balanced), employer_id_balanced != "") %>%
  group_by(case_number, fiscal_year) %>%
  filter(if (any(employer_is_worksite_business %in% TRUE)) employer_is_worksite_business %in% TRUE else !(employer_is_alias_evidence_only %in% TRUE)) %>%
  distinct(case_number, fiscal_year, employer_id = employer_id_balanced) %>% ungroup()
un_emp <- memb %>% group_by(case_number, fiscal_year) %>% mutate(emp_share = 1 / n()) %>% ungroup()

# ---- Row table ---------------------------------------------------------------
rows <- bind_rows(aw %>% mutate(emp_share = 1), un) %>%
  left_join(cases %>% select(case_number, fiscal_year, received, decided, status), by = c("case_number", "fiscal_year")) %>%
  mutate(nh = ifelse(is.finite(nh_obs), nh_obs, pooled_nh), nh_imputed = !is.finite(nh_obs),
    weeks = as.numeric(end - begin) / 7,
    req_span_ok = !is.na(rbegin) & !is.na(rend) & rend >= rbegin & as.numeric(rend - rbegin) <= MAX_REQ_DAYS,
    req_weeks = ifelse(req_span_ok, as.numeric(rend - rbegin) / 7, weeks),
    req_dates_fallback = !req_span_ok,
    cert_hours = weeks * nh * cert_pos, cert_posweeks = weeks * cert_pos,
    req_known = is.finite(req_pos) & !(fiscal_year %in% REQ_MISSING_FY),
    req_pos = ifelse(req_known, req_pos, NA_real_), req_hours = req_weeks * nh * req_pos,
    hourly = toupper(trimws(wage_unit)) %in% c("HOUR", "HOURLY", "HR") & is.finite(wage) & wage > 0 & wage <= HOURLY_WAGE_MAX,
    soc7 = substr(gsub("[^0-9-]", "", coalesce(soc_code, "")), 1, 7)) %>%
  group_by(case_number, fiscal_year) %>% mutate(app_w = 1 / n()) %>% ungroup()
# Unallocated rows with an unlinked employer get employer_id NA.
rows <- rows %>% left_join(un_emp %>% rename(emp_u = employer_id, emp_share_u = emp_share),
  by = c("case_number", "fiscal_year"), relationship = "many-to-many") %>%
  mutate(employer_id = coalesce(employer_id, emp_u), emp_share = ifelse(row_source == "allocated", 1, coalesce(emp_share_u, 1))) %>%
  select(-emp_u, -emp_share_u)
stopifnot(abs(sum(rows$app_w * rows$emp_share) - nrow(cases %>% semi_join(rows, by = c("case_number", "fiscal_year")))) < 1e-6)

# ---- AEWR applicable to each row (at-AEWR flag) ----------------------------
ref <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>%
  filter(occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference"), worker_category == "covered_workers") %>%
  transmute(state_fips, aewr = as.numeric(aewr), start = as.Date(effective_start), end = as.Date(effective_end_exclusive),
    alt_start = as.Date(alternative_effective_start)) %>% arrange(state_fips, start) %>%
  group_by(state_fips) %>% mutate(prev_aewr = lag(aewr)) %>% ungroup()
skill <- read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
  transmute(state_fips, sheet = trimws(source_sheet), soc7 = substr(soc_code, 1, 7), aewr = as.numeric(hourly_aewr),
    start = as.Date(effective_start), end = as.Date(effective_end_exclusive))
skill_state <- skill %>% filter(sheet == "Statewide 655.120(b)(1)(i)") %>% distinct(state_fips, aewr, start, end)
skill_soc <- skill %>% filter(sheet == "Statewide 655.120(b)(1)(ii)") %>% distinct(state_fips, soc7, aewr, start, end)
# Candidate rates at a date: the reference rate in effect (plus the prior rate
# before an alternative start, e.g. the 2024 injunction group), and from
# 2025-10-02 the statewide skill rates (US and H2A, skill I and II) and the
# SOC-specific (b)(1)(ii) rates. 2009-01-17 to 2010-03-15 has no reference
# rate (OES period, schedules not acquired): such rows are 'aewr unknown'.
at_aewr_rows <- function(r) {
  # r: row_id, state_fips, soc7, wage, d1, d2, d3 (receipt, decision, period begin)
  long <- bind_rows(lapply(c("d1", "d2", "d3"), function(v) r %>% transmute(row_id, state_fips, soc7, wage, d = .data[[v]]))) %>%
    filter(!is.na(d)) %>% distinct()
  sd <- distinct(long, state_fips, d)
  # Reference rate in effect at (state, date), with the prior rate before an alternative start.
  lk_ref <- sd %>% group_by(state_fips) %>% group_modify(function(g, k) {
    rr <- ref %>% filter(state_fips == k$state_fips) %>% arrange(start)
    i <- if (nrow(rr)) findInterval(as.numeric(g$d), as.numeric(rr$start)) else rep(0L, nrow(g))
    ok <- i > 0; ok[ok] <- is.na(rr$end[i[ok]]) | g$d[ok] < rr$end[i[ok]]
    tibble(d = g$d, r_ref = ifelse(ok, rr$aewr[pmax(i, 1)], NA_real_),
      r_prev = ifelse(ok & !is.na(rr$alt_start[pmax(i, 1)]) & g$d < rr$alt_start[pmax(i, 1)], rr$prev_aewr[pmax(i, 1)], NA_real_))
  }) %>% ungroup()
  lk_state <- sd %>% inner_join(skill_state, by = "state_fips", relationship = "many-to-many") %>%
    filter(d >= start, is.na(end) | d < end) %>% group_by(state_fips, d) %>% summarise(r_state = list(aewr), .groups = "drop")
  lk_soc <- distinct(long, state_fips, soc7, d) %>% filter(d >= min(skill_soc$start)) %>%
    inner_join(skill_soc, by = c("state_fips", "soc7"), relationship = "many-to-many") %>%
    filter(d >= start, is.na(end) | d < end) %>% group_by(state_fips, soc7, d) %>% summarise(r_soc = list(aewr), .groups = "drop")
  long %>% left_join(lk_ref, by = c("state_fips", "d")) %>% left_join(lk_state, by = c("state_fips", "d")) %>%
    left_join(lk_soc, by = c("state_fips", "soc7", "d")) %>%
    mutate(known = !is.na(r_ref) | lengths(r_state) > 0 | lengths(r_soc) > 0,
      hit = (!is.na(r_ref) & abs(wage - r_ref) <= AEWR_TOL) | (!is.na(r_prev) & abs(wage - r_prev) <= AEWR_TOL)) %>%
    { x <- .; j <- which(lengths(x$r_state) + lengths(x$r_soc) > 0)
      x$hit[j] <- x$hit[j] | mapply(function(w, a, b) any(abs(w - c(a, b)) <= AEWR_TOL), x$wage[j], x$r_state[j], x$r_soc[j]); x } %>%
    group_by(row_id) %>% summarise(aewr_known = any(known), at_aewr = any(hit), .groups = "drop")
}

# ---- Split to counties --------------------------------------------------------
rows <- rows %>% mutate(row_id = row_number(),
  county_fips_list = ifelse(is.na(county_fips_list) | trimws(county_fips_list) == "", "00000", county_fips_list),
  n_counties = lengths(strsplit(county_fips_list, ",", fixed = TRUE)))
cr <- rows %>% separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = trimws(county_fips_list), county_fips = ifelse(county_fips %in% c("", "00000"), "00000", harmonize_county_fips_2010(county_fips)),
    share = 1 / n_counties, state_fips = substr(county_fips, 1, 2)) %>% select(-county_fips_list)
aw_flags <- at_aewr_rows(cr %>% filter(hourly, state_fips != "00") %>%
  transmute(row_id = paste(row_id, county_fips), state_fips, soc7, wage, d1 = received, d2 = decided, d3 = begin))
cr <- cr %>% mutate(rc_id = paste(row_id, county_fips)) %>% left_join(aw_flags, by = c("rc_id" = "row_id")) %>%
  mutate(aewr_known = coalesce(aewr_known, FALSE), at_aewr = coalesce(at_aewr, FALSE)) %>% select(-rc_id)

# Windows ---------------------------------------------------------------------
cr <- cr %>% mutate(
  fy_dec = as.integer(fiscal_year), fy_rec = fed_fy(received), start_year = as.integer(format(begin, "%Y")),
  in_dq13 = !is.na(decided) & decided >= as.Date(sprintf("%d-10-01", fy_dec - 1L)) & decided <= as.Date(sprintf("%d-06-30", fy_dec)),
  in_rq13 = !is.na(received) & received <= as.Date(sprintf("%d-06-30", fy_rec)),
  in_rq13d = in_rq13 & !is.na(decided) & decided <= as.Date(sprintf("%d-06-30", fy_rec)),
  w = share * emp_share)
stopifnot(all(cr$share > 0), !anyNA(cr$fy_dec))

case_rows_out <- cr %>% transmute(case_number, fiscal_year = fy_dec, row_source, status, county_fips, state_fips,
  employer_id_balanced = employer_id, alloc_weight = w, received, decided, begin, end, fy_receipt = fy_rec, start_year,
  in_dq13, in_rq13, in_rq13d, cert_pos = cert_pos * w, cert_hours = cert_hours * w, cert_posweeks = cert_posweeks * w,
  req_pos = req_pos * w, req_hours = req_hours * w, req_known, req_dates_fallback, weekly_hours = nh, nh_imputed,
  hourly_wage = ifelse(hourly, wage, NA_real_), soc7, aewr_known, at_aewr, applications = app_w * w)
write_parquet(case_rows_out, file.path(out_dir, "outcomes_case_rows.parquet"))

# Aggregation -------------------------------------------------------------------
sum_req <- function(x, known) if (any(!known)) NA_real_ else sum(x, na.rm = TRUE)
# Every sum is computed before any column is renamed: dplyr evaluates
# summarise() expressions sequentially, so outputs use temporary names.
agg <- function(d, keys) d %>% group_by(across(all_of(keys))) %>% summarise(
  .hourly_hours = sum((cert_hours * w)[hourly], na.rm = TRUE),
  .hourly_aewr_known_hours = sum((cert_hours * w)[hourly & aewr_known], na.rm = TRUE),
  .at_aewr_hours = sum((cert_hours * w)[hourly & aewr_known & at_aewr], na.rm = TRUE),
  .cert_pos = sum(cert_pos * w, na.rm = TRUE), .cert_hours = sum(cert_hours * w, na.rm = TRUE),
  .cert_posweeks = sum(cert_posweeks * w, na.rm = TRUE),
  .req_pos = sum_req(req_pos * w, req_known), .req_hours = sum_req(ifelse(req_known, req_hours * w, 0), req_known),
  .req_unknown_rows = sum(!req_known), .apps = sum(app_w * w), .groups = "drop") %>%
  rename_with(~ sub("^\\.", "", .x), starts_with(".")) %>%
  mutate(share_at_aewr = ifelse(hourly_aewr_known_hours > 0, at_aewr_hours / hourly_aewr_known_hours, NA_real_))
win_frame <- function(d, wname) {
  switch(wname,
    fy = d %>% mutate(year = fy_dec),
    dq13 = d %>% filter(in_dq13) %>% mutate(year = fy_dec),
    rfy = d %>% mutate(year = fy_rec),
    rq13 = d %>% filter(in_rq13) %>% mutate(year = fy_rec),
    rq13d = d %>% filter(in_rq13d) %>% mutate(year = fy_rec)) %>%
    filter(!is.na(year), year >= FY_MIN, year <= FY_MAX)
}
incomplete_2026 <- c("fy", "rfy", "rq13")

# National requested availability by window-year: a year is unavailable when
# >= 1% of its contributing rows (by certified-or-requested weight) lack requested.
req_cov <- bind_rows(lapply(WINDOWS, function(wn) win_frame(cr, wn) %>% group_by(year) %>%
  summarise(window = wn, rows = n(), unknown_row_share = mean(!req_known),
    cert_pos_unknown_req = sum((cert_pos * w)[!req_known]), cert_pos_total = sum(cert_pos * w), .groups = "drop"))) %>%
  mutate(req_available = unknown_row_share < 0.01 & !(window %in% incomplete_2026 & year == FY_MAX))
write_csv_out(req_cov, "outcome_requested_coverage.csv")

# County universe: county_year_panel counties plus any H-2A county.
cyp <- read_parquet(path_processed("county_year_panel.parquet"), col_select = c("county_fips")) %>% distinct()
universe <- tibble(county_fips = sort(unique(c(cyp$county_fips, setdiff(unique(cr$county_fips), "00000"))))) %>%
  mutate(in_county_year_panel = county_fips %in% cyp$county_fips, state_fips = substr(county_fips, 1, 2))
stopifnot(is.character(universe$county_fips), identical(universe$county_fips, harmonize_county_fips_2010(universe$county_fips)))

extensive <- function(p, unit, active_col, prefix) {
  p %>% arrange(.data[[unit]], year) %>% group_by(.data[[unit]]) %>%
    mutate(!!paste0("any_", prefix) := .data[[active_col]] > 0,
      prev = lag(.data[[active_col]] > 0), prev_year = lag(year),
      !!paste0("entry_", prefix) := ifelse(!is.na(prev) & prev_year == year - 1L, .data[[paste0("any_", prefix)]] & !prev, NA),
      !!paste0("exit_", prefix) := ifelse(!is.na(prev) & prev_year == year - 1L, !.data[[paste0("any_", prefix)]] & prev, NA)) %>%
    ungroup() %>% select(-prev, -prev_year)
}

county_wide <- list(); employer_wide <- list(); national <- list()
for (wn in WINDOWS) {
  message("Window ", wn)
  d <- win_frame(cr, wn)
  avail <- req_cov %>% filter(window == wn) %>% select(year, req_available)
  # county x year
  cy <- agg(d %>% filter(county_fips != "00000"), c("county_fips", "year"))
  # employers active (certified) per county-year, and program-wide entrants/exiters
  ea <- d %>% filter(!is.na(employer_id)) %>% group_by(employer_id, year) %>% summarise(cp = sum(cert_pos * w), .groups = "drop")
  act <- ea %>% filter(cp > 0) %>% distinct(employer_id, year)
  new_emp <- act %>% anti_join(act %>% mutate(year = year + 1L), by = c("employer_id", "year")) %>% filter(year > FY_MIN)
  gone_emp <- act %>% mutate(year = year + 1L) %>% anti_join(act, by = c("employer_id", "year")) %>% filter(year <= FY_MAX)
  ec <- d %>% filter(county_fips != "00000", !is.na(employer_id), cert_pos > 0) %>% distinct(county_fips, employer_id, year)
  cnt <- ec %>% group_by(county_fips, year) %>% summarise(n_employers = n(), .groups = "drop") %>%
    full_join(ec %>% semi_join(new_emp, by = c("employer_id", "year")) %>% count(county_fips, year, name = "n_emp_entrants"), by = c("county_fips", "year")) %>%
    full_join(ec %>% mutate(year = year + 1L) %>% semi_join(gone_emp, by = c("employer_id", "year")) %>% count(county_fips, year, name = "n_emp_exiters"), by = c("county_fips", "year"))
  p <- tidyr::crossing(universe %>% select(county_fips), year = FY_MIN:FY_MAX) %>%
    left_join(cy, by = c("county_fips", "year")) %>% left_join(cnt, by = c("county_fips", "year")) %>%
    left_join(avail, by = "year") %>%
    mutate(across(c(cert_pos, cert_hours, cert_posweeks, apps, hourly_hours, hourly_aewr_known_hours, at_aewr_hours,
      n_employers, n_emp_entrants, n_emp_exiters, req_unknown_rows), ~ coalesce(as.numeric(.x), 0)),
      req_pos = ifelse(is.na(req_pos) & req_unknown_rows == 0, 0, req_pos), req_hours = ifelse(is.na(req_hours) & req_unknown_rows == 0, 0, req_hours),
      req_pos = ifelse(req_available %in% TRUE, req_pos, NA_real_), req_hours = ifelse(req_available %in% TRUE, req_hours, NA_real_),
      n_emp_entrants = ifelse(year == FY_MIN, NA_real_, n_emp_entrants), n_emp_exiters = ifelse(year == FY_MIN, NA_real_, n_emp_exiters)) %>%
    select(-req_available, -req_unknown_rows)
  p <- extensive(p, "county_fips", "cert_pos", "cert")
  p <- p %>% mutate(any_req = ifelse(is.na(req_pos), NA, req_pos > 0))
  if (wn %in% incomplete_2026) p <- p %>% mutate(across(-c(county_fips, year), ~ if (is.logical(.x)) ifelse(year == FY_MAX, NA, .x) else ifelse(year == FY_MAX, NA_real_, .x)))
  # entry/exit relative to an NA year are NA
  county_wide[[wn]] <- p %>% rename_with(~ paste0(.x, "_", wn), -c(county_fips, year))
  # employer x year
  ey <- agg(d %>% filter(!is.na(employer_id)), c("employer_id", "year")) %>%
    left_join(d %>% filter(!is.na(employer_id), county_fips != "00000", cert_pos > 0) %>% group_by(employer_id, year) %>%
      summarise(n_counties = n_distinct(county_fips), n_states = n_distinct(state_fips), .groups = "drop"), by = c("employer_id", "year"))
  employer_wide[[wn]] <- ey %>% select(employer_id, year, cert_pos, cert_hours, cert_posweeks, req_pos, req_hours, req_unknown_rows, apps,
    hourly_hours, hourly_aewr_known_hours, at_aewr_hours, n_counties, n_states) %>% mutate(window = wn) %>%
    left_join(avail, by = "year")
  national[[wn]] <- agg(d, "year") %>% mutate(window = wn)
}

county <- Reduce(function(a, b) full_join(a, b, by = c("county_fips", "year")), county_wide) %>%
  left_join(universe, by = "county_fips") %>% relocate(county_fips, state_fips, year, in_county_year_panel)
stopifnot(!anyDuplicated(county[c("county_fips", "year")]), !anyNA(county$county_fips), all(nzchar(county$county_fips)))
write_parquet(county, file.path(out_dir, "outcomes_county_fy.parquet"))

# Employer x FY: balanced over each employer's span? No: balanced over FY2008-2026
# for every employer ever active, so entry/exit and zeros are explicit.
emp_long <- bind_rows(employer_wide)
emp_ids <- sort(unique(emp_long$employer_id))
emp <- NULL
for (wn in WINDOWS) {
  e <- tidyr::crossing(employer_id = emp_ids, year = FY_MIN:FY_MAX) %>%
    left_join(emp_long %>% filter(window == wn) %>% select(-window, -req_available), by = c("employer_id", "year")) %>%
    left_join(req_cov %>% filter(window == wn) %>% select(year, req_available), by = "year") %>%
    mutate(across(c(cert_pos, cert_hours, cert_posweeks, apps, hourly_hours, hourly_aewr_known_hours, at_aewr_hours, n_counties, n_states, req_unknown_rows),
      ~ coalesce(as.numeric(.x), 0)),
      req_pos = ifelse(is.na(req_pos) & req_unknown_rows == 0, 0, req_pos), req_hours = ifelse(is.na(req_hours) & req_unknown_rows == 0, 0, req_hours),
      req_pos = ifelse(req_available %in% TRUE, req_pos, NA_real_), req_hours = ifelse(req_available %in% TRUE, req_hours, NA_real_)) %>%
    select(-req_available, -req_unknown_rows)
  e <- extensive(e, "employer_id", "cert_pos", "cert")
  if (wn %in% incomplete_2026) e <- e %>% mutate(across(-c(employer_id, year), ~ if (is.logical(.x)) ifelse(year == FY_MAX, NA, .x) else ifelse(year == FY_MAX, NA_real_, .x)))
  e <- e %>% rename_with(~ paste0(.x, "_", wn), -c(employer_id, year))
  emp <- if (is.null(emp)) e else full_join(emp, e, by = c("employer_id", "year"))
}
emp_first <- cr %>% filter(!is.na(employer_id), cert_pos > 0) %>% group_by(employer_id) %>%
  summarise(first_fy_cert = min(fy_dec), last_fy_cert = max(fy_dec), .groups = "drop")
emp <- emp %>% left_join(emp_first, by = "employer_id") %>% rename(employer_id_balanced = employer_id)
stopifnot(!anyDuplicated(emp[c("employer_id_balanced", "year")]))
write_parquet(emp, file.path(out_dir, "outcomes_employer_fy.parquet"))

# Calendar start-year county version (b01 start-year: year of period begin).
cal <- agg(cr %>% filter(county_fips != "00000", !is.na(start_year)) %>% mutate(year = start_year), c("county_fips", "year"))
cal_cov <- cr %>% filter(!is.na(start_year)) %>% group_by(year = start_year) %>%
  summarise(unknown_row_share = mean(!req_known), .groups = "drop") %>% mutate(req_available = unknown_row_share < 0.01)
cal_years <- (FY_MIN - 1L):FY_MAX
calendar <- tidyr::crossing(universe %>% select(county_fips), year = cal_years) %>% left_join(cal, by = c("county_fips", "year")) %>%
  left_join(cal_cov %>% select(year, req_available), by = "year") %>%
  mutate(across(c(cert_pos, cert_hours, cert_posweeks, apps, hourly_hours, hourly_aewr_known_hours, at_aewr_hours, req_unknown_rows), ~ coalesce(.x, 0)),
    req_pos = ifelse(is.na(req_pos) & req_unknown_rows == 0, 0, req_pos), req_hours = ifelse(is.na(req_hours) & req_unknown_rows == 0, 0, req_hours),
    req_pos = ifelse(req_available %in% TRUE, req_pos, NA_real_), req_hours = ifelse(req_available %in% TRUE, req_hours, NA_real_),
    year_complete = year >= FY_MIN & year <= LAST_COMPLETE_FY) %>%
  select(-req_available, -req_unknown_rows) %>% left_join(universe, by = "county_fips") %>% relocate(county_fips, state_fips, year)
stopifnot(!anyDuplicated(calendar[c("county_fips", "year")]))
write_parquet(calendar, file.path(out_dir, "outcomes_county_calendar.parquet"))

# ---- Validation against h2a_data_year (owning producer) --------------------
h2y <- read_parquet(path_int("h2a_data_year.parquet")) %>% mutate(year = as.integer(year))
fy_rebuilt <- agg(cr %>% filter(row_source == "allocated"), c("county_fips", "year" = "fy_dec"))
val_fy <- full_join(fy_rebuilt %>% select(county_fips, year, cert_pos, cert_hours, req_pos),
  h2y %>% select(county_fips, year, nbr_workers_certified_fiscal_year, man_hours_certified_fiscal_year, nbr_workers_requested_fiscal_year),
  by = c("county_fips", "year")) %>% filter(year <= LAST_COMPLETE_FY) %>%
  mutate(across(c(cert_pos, cert_hours, nbr_workers_certified_fiscal_year, man_hours_certified_fiscal_year), ~ coalesce(.x, 0))) %>%
  group_by(year) %>% summarise(rebuilt_cert = sum(cert_pos), producer_cert = sum(nbr_workers_certified_fiscal_year),
    rebuilt_hours = sum(cert_hours), producer_hours = sum(man_hours_certified_fiscal_year),
    max_abs_county_diff_cert = max(abs(cert_pos - nbr_workers_certified_fiscal_year)),
    max_rel_county_diff_hours = max(abs(cert_hours - man_hours_certified_fiscal_year) / pmax(1, man_hours_certified_fiscal_year)),
    county_corr_hours = cor(cert_hours, man_hours_certified_fiscal_year),
    rebuilt_req_allocated_only = sum(req_pos, na.rm = TRUE), producer_req = sum(nbr_workers_requested_fiscal_year, na.rm = TRUE),
    .groups = "drop") %>% mutate(rel_diff_hours = rebuilt_hours / producer_hours - 1)
write_csv_out(val_fy, "outcome_validation_fy.csv")
sy_rebuilt <- agg(cr %>% filter(row_source == "allocated", !is.na(start_year)), c("county_fips", "year" = "start_year"))
val_sy <- full_join(sy_rebuilt %>% select(county_fips, year, cert_pos, cert_hours, share_at_aewr),
  h2y %>% select(county_fips, year, nbr_workers_certified_start_year, man_hours_certified_start_year, share_cert_hours_at_aewr_start_year),
  by = c("county_fips", "year")) %>% filter(year >= FY_MIN, year <= LAST_COMPLETE_FY) %>%
  mutate(across(c(cert_pos, cert_hours, nbr_workers_certified_start_year, man_hours_certified_start_year), ~ coalesce(.x, 0))) %>%
  group_by(year) %>% summarise(rebuilt_cert = sum(cert_pos), producer_cert = sum(nbr_workers_certified_start_year),
    rebuilt_hours = sum(cert_hours), producer_hours = sum(man_hours_certified_start_year),
    max_abs_county_diff_cert = max(abs(cert_pos - nbr_workers_certified_start_year)),
    at_aewr_share_corr_vs_b01 = { ok <- is.finite(share_at_aewr) & is.finite(share_cert_hours_at_aewr_start_year); if (sum(ok) > 2) cor(share_at_aewr[ok], share_cert_hours_at_aewr_start_year[ok]) else NA_real_ },
    .groups = "drop")
write_csv_out(val_sy, "outcome_validation_calendar.csv")

# Requested: all-case totals vs case-level disclosure totals (main/master rows).
case_req_total <- disc %>% group_by(fiscal_year) %>% summarise(disclosure_row_sum_requested = sum(num(nbr_workers_requested), na.rm = TRUE), .groups = "drop")
nat <- bind_rows(national) %>% select(window, year, cert_pos, cert_hours, req_pos, req_hours, apps, hourly_hours, hourly_aewr_known_hours, at_aewr_hours, share_at_aewr)
nat_src <- cr %>% group_by(year = fy_dec, row_source) %>% summarise(req = sum(req_pos * w, na.rm = TRUE), cert = sum(cert_pos * w), .groups = "drop") %>%
  pivot_wider(names_from = row_source, values_from = c(req, cert), values_fill = 0) %>%
  left_join(case_req_total, by = c("year" = "fiscal_year"))
write_csv_out(nat, "outcome_national_by_window.csv")
write_csv_out(nat_src, "outcome_requested_by_source.csv")

# Window and data-quality summaries.
qual <- cr %>% group_by(year = fy_dec) %>% summarise(
  cert_pos_total = sum(cert_pos * w), dq13_cert_share = sum((cert_pos * w)[in_dq13]) / sum(cert_pos * w),
  dq13_hours_share = sum((cert_hours * w)[in_dq13], na.rm = TRUE) / sum(cert_hours * w, na.rm = TRUE),
  weekly_hours_imputed_cert_share = sum((cert_pos * w)[nh_imputed]) / sum(cert_pos * w),
  req_dates_fallback_req_share = sum((req_pos * w)[req_dates_fallback], na.rm = TRUE) / sum(req_pos * w, na.rm = TRUE),
  county_unknown_cert_share = sum((cert_pos * w)[county_fips == "00000"]) / sum(cert_pos * w),
  employer_unlinked_req_share = sum((req_pos * w)[is.na(employer_id)], na.rm = TRUE) / sum(req_pos * w, na.rm = TRUE),
  hourly_cert_hours_share = sum((cert_hours * w)[hourly], na.rm = TRUE) / sum(cert_hours * w, na.rm = TRUE),
  aewr_known_hourly_share = sum((cert_hours * w)[hourly & aewr_known], na.rm = TRUE) / sum((cert_hours * w)[hourly], na.rm = TRUE),
  at_aewr_share = sum((cert_hours * w)[hourly & aewr_known & at_aewr], na.rm = TRUE) / sum((cert_hours * w)[hourly & aewr_known], na.rm = TRUE),
  .groups = "drop")
rec_lag <- cr %>% filter(row_source == "allocated") %>% group_by(year = fy_rec) %>%
  summarise(rq13_cert = sum((cert_pos * w)[in_rq13]), rq13d_cert = sum((cert_pos * w)[in_rq13d]), .groups = "drop") %>%
  mutate(rq13d_over_rq13 = rq13d_cert / rq13_cert)
write_csv_out(qual %>% left_join(rec_lag, by = "year"), "outcome_quality_by_fy.csv")

meta <- list(script = "explorations/identification_search/shared/01_outcomes.R", built = as.character(Sys.time()),
  pooled_weekly_hours_mean = pooled_nh, fy_range = c(FY_MIN, FY_MAX), windows = WINDOWS,
  fy2026_na_windows = incomplete_2026, requested_missing_decision_fy = REQ_MISSING_FY,
  files = list(county = "outcomes_county_fy.parquet (county_fips x year; columns <measure>_<window>)",
    employer = "outcomes_employer_fy.parquet (employer_id_balanced x year; balanced FY2008-2026 over every employer ever linked)",
    calendar = "outcomes_county_calendar.parquet (county_fips x calendar start year 2007-2026; year_complete flags 2008-2025)",
    case_rows = "outcomes_case_rows.parquet (case x county x employer allocation rows; measures already multiplied by alloc_weight)"),
  measures = list(cert_pos = "certified positions", cert_hours = "certified contract hours", cert_posweeks = "certified positions x contract weeks (log hours = log pos + log weeks + log weekly hours: weeks = posweeks/pos, weekly hours = hours/posweeks)",
    req_pos = "requested positions, all cases incl. denied/withdrawn; NA where unavailable", req_hours = "requested positions x requested weeks x weekly hours",
    apps = "applications (case = 1, split over rows and counties)", hourly_hours = "certified hours with an hourly offer (0 < wage <= 100)",
    hourly_aewr_known_hours = "hourly certified hours with a known applicable AEWR", at_aewr_hours = "hourly certified hours offered within $0.05 of an applicable AEWR",
    share_at_aewr = "at_aewr_hours / hourly_aewr_known_hours", n_employers = "employers (balanced id) with certified positions in the county",
    n_emp_entrants = "of those, employers with no certified positions anywhere in the previous year", n_emp_exiters = "employers active in the county last year with no certified positions anywhere this year",
    any_cert = "cert_pos > 0", entry_cert = "any_cert and not any_cert last year", exit_cert = "not any_cert and any_cert last year", any_req = "req_pos > 0"))
jsonlite::write_json(meta, file.path(out_dir, "outcomes_manifest.json"), auto_unbox = TRUE, pretty = TRUE)
print(as.data.frame(val_fy), digits = 6); print(as.data.frame(val_sy), digits = 6)
print(as.data.frame(needed_cov), digits = 4); print(as.data.frame(req_cov %>% filter(!req_available)), digits = 4)
print(as.data.frame(qual), digits = 4); print(as.data.frame(nat_src), digits = 6)
