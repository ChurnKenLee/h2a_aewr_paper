# Track 1 offered-wage pass-through, split by employer continuity.
# Declared in track1_employer_decomposition_registry.json.
#
# Arguments: <scratch dir outside the repo> <tau> [<Track 1 scratch dir>]
#   tau = 2026  the declared estimates (FY2025 vs FY2026). Runs only if every
#               hash in track1_employer_decomposition_registry.sha256 matches.
#   tau = 2025  placebo and code-path check (FY2024 vs FY2025 with the FY2026
#               dose). Reads no row after FY2025.
# With the third argument, the county-level total is checked against the
# first-stage estimate that 08_track1_estimate.R wrote there.
#
# Row construction, the Q1-Q3 decision window, the county sample and the dose
# are those of 08_track1_estimate.R. Every estimate is written; nothing is
# selected.
source("explorations/identification/id_helpers.R")
out_dir <- id_output_dir()
args <- commandArgs(trailingOnly = TRUE)
TAU <- as.integer(args[2])
t1_dir <- if (length(args) >= 3) args[3] else NA_character_
stopifnot(TAU %in% c(2025L, 2026L))
Y0 <- TAU - 1L
YEARS <- (TAU - 2L):TAU
FIRST_FY <- min(YEARS)
tag <- paste0("tau", TAU)

if (TAU == 2026L) {
  hash_lines <- readLines("explorations/identification/track1_employer_decomposition_registry.sha256")
  hash_lines <- hash_lines[nzchar(hash_lines)]
  stopifnot(length(hash_lines) >= 3)
  for (line in hash_lines) {
    parts <- strsplit(line, "  ", fixed = TRUE)[[1]]
    actual <- digest::digest(file = file.path("explorations/identification", parts[2]), algo = "sha256")
    if (!identical(actual, parts[1])) stop("Hash mismatch for ", parts[2], ": the registry is not frozen for this code")
  }
}
registry <- jsonlite::read_json("explorations/identification/track1_registry.json", simplifyVector = TRUE)

# Sample counties (identical to 08) --------------------------------------------
cp <- read_parquet(path_processed("county_year_panel.parquet"),
  col_select = c("county_fips", "state_fips", "aewr_region_id", "emp_farm_2011", "year")) %>%
  filter(year == 2011) %>% select(-year)
counties <- cp %>% filter(!state_fips %in% registry$geography$excluded_states,
  is.finite(emp_farm_2011), emp_farm_2011 > 0) %>%
  group_by(aewr_region_id) %>% filter(n_distinct(state_fips) > 1) %>% ungroup() %>%
  mutate(aewr_region_id = as.character(aewr_region_id)) %>%
  arrange(aewr_region_id, state_fips, county_fips)
stopifnot(!anyDuplicated(counties$county_fips), is.character(counties$county_fips))

# Dose: FY2026 statewide U.S. skill-I rate minus the FY2025 regional rate, both
# as in force on April 1 (04_track1_power.R logic) -------------------------------
regional <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>%
  filter(occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference"), worker_category == "covered_workers") %>%
  transmute(state_fips, aewr = as.numeric(aewr), start = as.Date(effective_start), end = as.Date(effective_end_exclusive))
statewide <- read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
  filter(trimws(source_sheet) == registry$policy_event$source_sheet, worker_category == "US", as.character(skill_level) == "1") %>%
  transmute(state_fips, aewr = as.numeric(hourly_aewr), start = as.Date(effective_start), end = as.Date(effective_end_exclusive))
rate_on <- function(s, date) {
  r <- s %>% filter(start <= date, is.na(end) | end > date) %>% select(state_fips, aewr)
  stopifnot(!anyDuplicated(r$state_fips))
  r
}
states <- counties %>% distinct(state_fips, aewr_region_id) %>%
  left_join(rate_on(regional, as.Date("2025-04-01")) %>% rename(rate25 = aewr), by = "state_fips") %>%
  left_join(rate_on(statewide, as.Date("2026-04-01")) %>% rename(rate26 = aewr), by = "state_fips") %>%
  left_join(rate_on(regional, as.Date(sprintf("%d-04-01", Y0))) %>% rename(rate_base = aewr), by = "state_fips") %>%
  mutate(dose = rate26 - rate25)
stopifnot(!anyDuplicated(states$state_fips), all(is.finite(states$dose)), all(is.finite(states$rate_base)))
# Deeper-cut indicator for the trimming bounds: dose below the unweighted mean
# of the state doses in the former region.
states <- states %>% group_by(aewr_region_id) %>% mutate(deep = as.integer(dose < mean(dose))) %>% ungroup()
if (!is.na(t1_dir)) {
  sched <- read_csv(file.path(t1_dir, "track1_treatment_schedule.csv"), show_col_types = FALSE, col_types = cols(state_fips = "c")) %>%
    filter(label == "us_skill_1", year %in% 2025:2026) %>% tidyr::pivot_wider(names_from = year, values_from = aewr, names_prefix = "r")
  chk <- inner_join(states, sched, by = "state_fips")
  stopifnot(nrow(chk) == nrow(states), max(abs(chk$rate25 - chk$r2025)) < 1e-12, max(abs(chk$rate26 - chk$r2026)) < 1e-12)
}
id_write(states, out_dir, paste0("t1emp_state_dose_", tag, ".csv"))

# Rows (08 logic). The pooled weekly-hours mean uses FY2012-FY2025 rows only ----
nh_rows <- arrow::open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, period_begin_date, number_of_hours) %>%
  filter(fiscal_year >= 2012L, fiscal_year <= 2025L) %>% collect() %>%
  filter(!is.na(as.Date(period_begin_date)))
nh_pre_mean <- mean(suppressWarnings(as.numeric(nh_rows$number_of_hours)), na.rm = TRUE)
rm(nh_rows)

plausible <- function(value, fy) { y <- as.integer(format(value, "%Y")); value[!is.na(y) & abs(y - fy) > 1L] <- as.Date(NA); value }
first_date <- function(x) { x <- ifelse(is.na(x) | trimws(x) == "", NA_character_, x); first(na.omit(x)) }
dec <- arrow::open_dataset(path_int("h2a_disclosure_records.parquet")) %>%
  select(fiscal_year, case_number, decision_date) %>%
  filter(fiscal_year >= FIRST_FY, fiscal_year <= TAU) %>% collect() %>%
  group_by(case_number, fiscal_year) %>% summarise(decision_date = first_date(decision_date), .groups = "drop") %>%
  mutate(decision_date = plausible(as.Date(decision_date), fiscal_year))
rows <- arrow::open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, case_number, county_fips_list, adjusted_nbr_workers_certified, period_begin_date, period_end_date,
    number_of_hours, wage_rate, wage_unit, employer_id_balanced) %>%
  filter(fiscal_year >= FIRST_FY, fiscal_year <= TAU) %>% collect()
stopifnot(nrow(rows) > 0, max(rows$fiscal_year) == TAU)
rows <- rows %>%
  mutate(begin = as.Date(period_begin_date), end = as.Date(period_end_date)) %>% filter(!is.na(begin)) %>%
  mutate(nh = suppressWarnings(as.numeric(number_of_hours)), nh = ifelse(is.na(nh), nh_pre_mean, nh),
    workers = as.numeric(adjusted_nbr_workers_certified),
    hours = as.numeric(end - begin) / 7 * nh * workers,
    wage = suppressWarnings(as.numeric(wage_rate)),
    hourly = toupper(trimws(wage_unit)) %in% c("HOUR", "HOURLY", "HR") & is.finite(wage) & wage > 0 & wage <= 100) %>%
  left_join(dec, by = c("case_number", "fiscal_year"), relationship = "many-to-one") %>%
  mutate(q13 = !is.na(decision_date) & decision_date >= as.Date(sprintf("%d-10-01", fiscal_year - 1L)) &
      decision_date <= as.Date(sprintf("%d-06-30", fiscal_year)),
    linked = !is.na(employer_id_balanced) & employer_id_balanced != "",
    id_balanced = ifelse(linked, employer_id_balanced, paste0("unlinked:", fiscal_year, ":", case_number)),
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    n_counties = stringr::str_count(county_fips_list, ",") + 1L) %>%
  tidyr::separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = harmonize_county_fips_2010(trimws(county_fips_list)), share = 1 / n_counties,
    year = as.integer(fiscal_year), h = hours * share, pos = workers * share,
    wage_row = q13 & hourly & is.finite(hours)) %>%
  filter(q13) %>%
  select(year, county_fips, id_balanced, linked, h, pos, wage, wage_row)
stopifnot(all(rows$h[rows$wage_row] >= 0))

# High-recall employer identifiers: every balanced identifier maps to one.
tiers <- read_parquet(path_int("h2a_employer_record_membership.parquet"),
  col_select = c("employer_id_balanced", "employer_id_high_recall")) %>%
  filter(!is.na(employer_id_balanced), !is.na(employer_id_high_recall)) %>% distinct()
stopifnot(!anyDuplicated(tiers$employer_id_balanced))
rows <- rows %>% left_join(tiers, by = c("id_balanced" = "employer_id_balanced"), relationship = "many-to-one") %>%
  mutate(id_high_recall = coalesce(employer_id_high_recall, id_balanced)) %>% select(-employer_id_high_recall) %>%
  inner_join(counties %>% select(county_fips, state_fips, aewr_region_id), by = "county_fips")

link_share <- rows %>% filter(wage_row) %>% group_by(year) %>%
  summarise(hourly_hours = sum(h), unlinked_share_hours = sum(h[!linked]) / sum(h), rows = n(), .groups = "drop")
id_write(link_share, out_dir, paste0("t1emp_link_share_", tag, ".csv"))
print(as.data.frame(link_share))

# Inference helpers ----------------------------------------------------------------
# State leave-one-out jackknife of any statistic stat(d) -> named numeric vector.
jackknife <- function(d, stat) {
  st <- sort(unique(d$state_fips))
  b <- vapply(st, function(s) stat(d[d$state_fips != s, , drop = FALSE]), stat(d))
  b <- matrix(b, ncol = length(st))
  G <- length(st)
  sqrt((G - 1) / G * rowSums((b - rowMeans(b, na.rm = TRUE))^2, na.rm = TRUE))
}
fit <- function(d, y, rhs, w) {
  f <- as.formula(paste(y, "~", rhs, "| aewr_region_id"))
  if (is.null(w)) feols(f, d, vcov = ~state_fips) else feols(f, d, weights = as.formula(paste0("~", w)), vcov = ~state_fips)
}
results <- list()
est <- function(block, spec, tier, d, y, rhs = "dose", w = NULL, unit = "unit") {
  d <- d[is.finite(d[[y]]), , drop = FALSE]
  if (!is.null(w)) d <- d[is.finite(d[[w]]) & d[[w]] > 0, , drop = FALSE]
  m <- fit(d, y, rhs, w)
  terms <- names(coef(m))
  se_j <- jackknife(d, function(x) { b <- coef(fit(x, y, rhs, w)); out <- setNames(rep(NA_real_, length(terms)), terms); out[names(b)] <- b; out })
  G <- n_distinct(d$state_fips)
  wt <- if (is.null(w)) rep(1, nrow(d)) else d[[w]]
  r <- tibble(block = block, spec = spec, id_tier = tier, tau = TAU, outcome = y, term = terms, weights = if (is.null(w)) "equal" else w,
    estimate = unname(coef(m)), se_jackknife = unname(se_j), p_jackknife = 2 * pt(-abs(unname(coef(m)) / unname(se_j)), G - 1),
    se_state_crv1 = unname(se(m)), n = nobs(m), units = n_distinct(d[[unit]]), states = G,
    mean_y = sum(wt * d[[y]]) / sum(wt))
  results[[length(results) + 1]] <<- r
  invisible(r)
}

# Weighted mean that is 0 when the weights are empty (used inside components).
wmean0 <- function(x, w) if (sum(w) > 0) sum(w * x) / sum(w) else 0

lee_region <- function(d) {
  # d: incumbents of one former region with deep, stay, dw, dose.
  g1 <- d[d$deep == 1, , drop = FALSE]; g0 <- d[d$deep == 0, , drop = FALSE]
  if (!nrow(g1) || !nrow(g0)) return(NULL)
  p1 <- mean(g1$stay); p0 <- mean(g0$stay)
  y1 <- g1$dw[g1$stay == 1]; y0 <- g0$dw[g0$stay == 1]
  if (!length(y1) || !length(y0)) return(NULL)
  # Mean after removing a mass q from one end; the boundary observation keeps a fractional weight.
  trim_mean <- function(y, q, side) {
    y <- sort(y, decreasing = side == "top")
    wgt <- pmin(pmax(seq_along(y) - q * length(y), 0), 1)
    sum(wgt * y) / sum(wgt)
  }
  if (p1 >= p0) {
    q <- (p1 - p0) / p1
    hi <- trim_mean(y1, q, "bottom") - mean(y0); lo <- trim_mean(y1, q, "top") - mean(y0)
  } else {
    q <- (p0 - p1) / p0
    hi <- mean(y1) - trim_mean(y0, q, "top"); lo <- mean(y1) - trim_mean(y0, q, "bottom")
  }
  tibble(point = mean(y1) - mean(y0), lo = lo, hi = hi, ddose = mean(g1$dose) - mean(g0$dose),
    dstay = p1 - p0, trim = q, weight = nrow(d) * min(p1, p0))
}
lee_stat <- function(d) {
  r <- bind_rows(lapply(split(d, d$aewr_region_id), lee_region))
  a <- colSums(r[c("point", "lo", "hi", "ddose", "dstay", "trim")] * r$weight) / sum(r$weight)
  per <- a[c("lo", "hi")] / a[["ddose"]]
  c(contrast_point = a[["point"]], contrast_low = a[["lo"]], contrast_high = a[["hi"]], dose_gap = a[["ddose"]],
    stay_gap = a[["dstay"]], trim_share = a[["trim"]], per_dollar_point = a[["point"]] / a[["ddose"]],
    per_dollar_low = min(per), per_dollar_high = max(per), regions = nrow(r))
}

t1_total <- NA_real_
if (!is.na(t1_dir)) {
  t1 <- read_csv(file.path(t1_dir, "track1_post_estimates.csv"), show_col_types = FALSE)
  t1_total <- t1$estimate[t1$spec == if (TAU == 2026L) "B_primary_first_stage" else "placebo_fy2024_fy2025_first_stage"]
  stopifnot(length(t1_total) == 1)
}

groups_out <- list(); lee_out <- list(); means_out <- list(); share_out <- list()
for (tier in c("balanced", "high_recall")) {
  rows$uid <- rows[[paste0("id_", tier)]]

  # P1. Exact county decomposition, employer x county cells -------------------------
  cells <- rows %>% filter(wage_row, year %in% c(Y0, TAU)) %>%
    group_by(county_fips, state_fips, aewr_region_id, uid, year) %>%
    summarise(wb = sum(h * wage), h = sum(h), .groups = "drop") %>% filter(h > 0) %>%
    mutate(w = wb / h, period = ifelse(year == Y0, "0", "1")) %>% select(-wb, -year) %>%
    tidyr::pivot_wider(names_from = period, values_from = c(h, w), values_fill = 0, names_sep = "") %>%
    mutate(group = case_when(h0 > 0 & h1 > 0 ~ "C", h0 > 0 ~ "X", TRUE ~ "N"))
  stopifnot(!anyDuplicated(cells[c("county_fips", "uid")]))
  county <- cells %>% group_by(county_fips, state_fips, aewr_region_id) %>% summarise(
    H0 = sum(h0), H1 = sum(h1), W0 = sum(h0 * w0) / H0, W1 = sum(h1 * w1) / H1,
    HC0 = sum(h0[group == "C"]), HC1 = sum(h1[group == "C"]),
    WC0 = wmean0(w0[group == "C"], h0[group == "C"]), WC1 = wmean0(w1[group == "C"], h1[group == "C"]),
    within_raw = wmean0((w1 - w0)[group == "C"], h0[group == "C"]),
    SN1 = sum(h1[group == "N"]) / H1, WN1 = wmean0(w1[group == "N"], h1[group == "N"]),
    SX0 = sum(h0[group == "X"]) / H0, WX0 = wmean0(w0[group == "X"], h0[group == "X"]),
    cells_C = sum(group == "C"), cells_N = sum(group == "N"), cells_X = sum(group == "X"), .groups = "drop") %>%
    filter(H0 > 0, H1 > 0) %>%
    mutate(total = W1 - W0, has_C = HC0 > 0,
      within = ifelse(has_C, within_raw, 0),
      between = ifelse(has_C, (WC1 - WC0) - within_raw, 0),
      entry = ifelse(has_C, SN1 * (WN1 - WC1), 0),
      exit = ifelse(has_C, SX0 * (WC0 - WX0), 0),
      no_continuer = ifelse(has_C, 0, total)) %>%
    left_join(states %>% select(state_fips, dose, deep), by = "state_fips")
  stopifnot(!anyDuplicated(county$county_fips),
    max(abs(county$within + county$between + county$entry + county$exit + county$no_continuer - county$total)) < 1e-9)
  comps <- c("total", "within", "between", "entry", "exit", "no_continuer")
  for (y in comps) est("P1_county_decomposition", y, tier, county, y, unit = "county_fips")
  p1 <- bind_rows(results) %>% filter(block == "P1_county_decomposition", id_tier == tier)
  stopifnot(abs(sum(p1$estimate[p1$spec != "total"]) - p1$estimate[p1$spec == "total"]) < 1e-9)
  if (!is.na(t1_total)) {
    message("Track 1 first stage ", signif(t1_total, 8), "; rebuilt total ", signif(p1$estimate[p1$spec == "total"], 8))
    stopifnot(abs(p1$estimate[p1$spec == "total"] - t1_total) < 1e-8)
  }
  means_out[[tier]] <- county %>% summarise(id_tier = tier, tau = TAU, counties = n(), counties_without_continuer = sum(!has_C),
    across(all_of(comps), mean, .names = "mean_{.col}"),
    continuer_hours_share_y0 = sum(HC0) / sum(H0), continuer_hours_share_y1 = sum(HC1) / sum(H1),
    cells_C = sum(cells_C), cells_N = sum(cells_N), cells_X = sum(cells_X))
  if (tier == "balanced") id_write(county, out_dir, paste0("t1emp_county_components_", tag, ".csv"))
  # Within share of the total slope, with a state jackknife on the ratio itself.
  share_stat <- function(x) c(within_share = coef(fit(x, "within", "dose", NULL))[["dose"]] / coef(fit(x, "total", "dose", NULL))[["dose"]])
  share_out[[tier]] <- tibble(id_tier = tier, tau = TAU, within_share = unname(share_stat(county)),
    se_jackknife = unname(jackknife(county, share_stat)), states = n_distinct(county$state_fips))

  # Employer x state units (linked rows only) ----------------------------------------
  linked_rows <- rows %>% filter(linked)
  ust <- linked_rows %>% group_by(uid, state_fips, aewr_region_id, year) %>%
    summarise(wb = sum((h * wage)[wage_row]), h = sum(h[wage_row]), pos = sum(pos, na.rm = TRUE), .groups = "drop") %>%
    mutate(w = ifelse(h > 0, wb / h, NA_real_)) %>% select(-wb)
  wide <- ust %>% filter(year %in% YEARS) %>%
    mutate(period = c("m1", "0", "1")[match(year, YEARS)]) %>% select(-year) %>%
    tidyr::pivot_wider(names_from = period, values_from = c(h, w, pos), names_sep = "") %>%
    mutate(across(any_of(c("hm1", "h0", "h1", "posm1", "pos0", "pos1")), ~ coalesce(.x, 0)),
      unit = paste(uid, state_fips)) %>%
    left_join(states %>% select(state_fips, dose, deep, rate_base), by = "state_fips")
  for (v in c("hm1", "h0", "h1", "wm1", "w0", "w1", "posm1", "pos0", "pos1")) if (!v %in% names(wide)) stop("Missing period column ", v)
  stopifnot(!anyDuplicated(wide$unit))
  wide <- wide %>% mutate(dw = w1 - w0, dose_c = dose - mean(states$dose))

  # P2. Within-employer pass-through, continuing units.
  cont <- wide %>% filter(h0 > 0, h1 > 0)
  est("P2_continuer_pass_through", "employer_state_units", tier, cont, "dw")
  est("P2_continuer_pass_through", "employer_state_units", tier, cont, "dw", w = "h0")

  # P3. Selection margins.
  inc <- wide %>% filter(h0 > 0) %>% mutate(exit_hourly = as.numeric(h1 == 0), exit_any = as.numeric(pos1 == 0), stay = 1 - exit_hourly)
  new <- wide %>% filter(h1 > 0) %>% mutate(entrant = as.numeric(h0 == 0))
  for (w in list(NULL, "h0")) {
    est("P3_selection_margins", "exit_of_base_year_incumbents", tier, inc, "exit_hourly", w = w)
    est("P3_selection_margins", "exit_of_base_year_incumbents", tier, inc, "exit_any", w = w)
  }
  est("P3_selection_margins", "entrant_share_of_tau_units", tier, new, "entrant")
  est("P3_selection_margins", "entrant_share_of_tau_units", tier, new, "entrant", w = "h1")
  groups_out[[tier]] <- bind_rows(
    inc %>% group_by(deep) %>% summarise(what = "base-year incumbents", units = n(), exit_rate = mean(exit_hourly),
      exit_rate_any = mean(exit_any), exit_hours_share = sum(h0 * exit_hourly) / sum(h0), .groups = "drop"),
    new %>% group_by(deep) %>% summarise(what = "tau units", units = n(), entrant_rate = mean(entrant),
      entrant_hours_share = sum(h1 * entrant) / sum(h1), .groups = "drop")) %>% mutate(id_tier = tier, tau = TAU)

  if (tier == "balanced") {
    # P4. Selection test: do units that stay differ before the change, and does
    # that difference vary with the dose?
    pre <- inc %>% mutate(prem0 = w0 - rate_base, dw_pre = ifelse(hm1 > 0, w0 - wm1, NA_real_),
      stay_x_dose = stay * dose_c)
    est("P4_selection_test", "base_year_offer_minus_floor", tier, pre, "prem0", rhs = "dose + stay + stay_x_dose")
    est("P4_selection_test", "prior_year_wage_change", tier, pre, "dw_pre", rhs = "dose + stay + stay_x_dose")

    # P5. Trimming bounds on the continuer pass-through.
    lee_d <- inc %>% select(unit, state_fips, aewr_region_id, deep, dose, stay, dw)
    b <- lee_stat(lee_d)
    se_b <- jackknife(lee_d, lee_stat)
    lee_out[[tier]] <- tibble(id_tier = tier, tau = TAU, quantity = names(b), estimate = unname(b), se_jackknife = unname(se_b),
      states = n_distinct(lee_d$state_fips))
  }
}

res <- bind_rows(results)
id_write(res, out_dir, paste0("t1emp_estimates_", tag, ".csv"))
id_write(bind_rows(means_out), out_dir, paste0("t1emp_county_means_", tag, ".csv"))
id_write(bind_rows(groups_out), out_dir, paste0("t1emp_selection_rates_", tag, ".csv"))
id_write(bind_rows(lee_out), out_dir, paste0("t1emp_trimming_bounds_", tag, ".csv"))
id_write(bind_rows(share_out), out_dir, paste0("t1emp_within_share_", tag, ".csv"))
options(width = 220)
print(as.data.frame(res %>% select(block, spec, id_tier, outcome, term, weights, estimate, se_jackknife, p_jackknife, se_state_crv1, n, units, mean_y)), digits = 4)
print(as.data.frame(bind_rows(means_out)), digits = 4)
print(as.data.frame(bind_rows(groups_out)), digits = 4)
print(as.data.frame(bind_rows(lee_out)), digits = 4)
print(as.data.frame(bind_rows(share_out)), digits = 4)
