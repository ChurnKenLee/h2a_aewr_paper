# Track 1 post-policy estimation. The user approved reading FY2026 outcomes,
# offered wages and case counts on 2026-11-02, chose amendment 1
# (track1_registry_amendment1.json: FY2025-vs-FY2026 contrast as primary,
# state jackknife as primary inference) and confirmed the rule's legal status.
# Both registries are hash-checked before any FY2026 row is read. Estimates:
#   primary (B), placebo (FY2024 vs FY2025 with the FY2026 dose),
#   S1 base static FY2012-FY2026 (+ event study), S2 state trends,
#   S3 alternative treatments, S4 cohort-aligned treatment, S5 minimum-wage
#   controls; PPML counts and the offered-wage first stage throughout.
# Every specification run is written; nothing is selected.
source("explorations/identification/id_helpers.R")
out_dir <- id_output_dir()
base_path <- "explorations/identification/track1_registry.json"
amend_path <- "explorations/identification/track1_registry_amendment1.json"
stopifnot(
  digest::digest(file = base_path, algo = "sha256") == "bbffa053953873aa8a36c04a5af71c012fd15460b2cd0ff84bd378f6e65aafd8",
  digest::digest(file = amend_path, algo = "sha256") == "433264ec85845f2ddb97c291073c19585ca9b80e2bb137aa0915fe38e29799e9")
registry <- jsonlite::read_json(base_path, simplifyVector = TRUE)
LAST_FY <- 2026L
B_DRAWS <- 9999L; SEED <- 20261102L

read_through <- function(file, cols) {
  cols <- unique(c("fiscal_year", cols))
  d <- arrow::open_dataset(path_int(file)) %>% select(all_of(cols)) %>%
    filter(fiscal_year >= 2012L, fiscal_year <= LAST_FY) %>% collect()
  stopifnot(nrow(d) > 0, max(d$fiscal_year) == LAST_FY)
  d
}

# Sample (identical to 04) -----------------------------------------------------
cp <- read_parquet(path_processed("county_year_panel.parquet"),
  col_select = c("county_fips", "state_fips", "aewr_region_id", "emp_farm_2011", "year")) %>%
  filter(year == 2011) %>% select(-year)
counties <- cp %>% filter(!state_fips %in% registry$geography$excluded_states,
  is.finite(emp_farm_2011), emp_farm_2011 > 0) %>%
  group_by(aewr_region_id) %>% filter(n_distinct(state_fips) > 1) %>% ungroup() %>%
  arrange(aewr_region_id, state_fips, county_fips)
stopifnot(!anyDuplicated(counties$county_fips), is.character(counties$county_fips))

# Outcomes (04 logic; weekly-hours imputation mean fixed to FY2012-FY2025 rows
# so the pre-period rebuild is identical to the frozen pre panel) ------------
plausible <- function(value, fy) { y <- as.integer(format(value, "%Y")); value[!is.na(y) & abs(y - fy) > 1L] <- as.Date(NA); value }
first_date <- function(x) { x <- ifelse(is.na(x) | trimws(x) == "", NA_character_, x); first(na.omit(x)) }
dec <- read_through("h2a_disclosure_records.parquet", c("case_number", "decision_date", "case_received_date")) %>%
  group_by(case_number, fiscal_year) %>%
  summarise(decision_date = first_date(decision_date), received = first_date(case_received_date), .groups = "drop") %>%
  mutate(decision_date = plausible(as.Date(decision_date), fiscal_year), received = as.Date(substr(received, 1, 10)))
ws <- read_through("h2a_allocated_worksites.parquet", c("case_number", "county_fips_list", "adjusted_nbr_workers_certified",
  "period_begin_date", "period_end_date", "number_of_hours", "wage_rate", "wage_unit")) %>%
  mutate(begin = as.Date(period_begin_date), end = as.Date(period_end_date)) %>% filter(!is.na(begin)) %>%
  mutate(nh = suppressWarnings(as.numeric(number_of_hours)))
nh_pre_mean <- mean(ws$nh[ws$fiscal_year <= 2025L], na.rm = TRUE)
nh_missing <- ws %>% group_by(fiscal_year) %>% summarise(share_missing_weekly_hours = mean(is.na(nh)), .groups = "drop")
ws <- ws %>% mutate(nh = ifelse(is.na(nh), nh_pre_mean, nh),
  workers = as.numeric(adjusted_nbr_workers_certified),
  hours = as.numeric(end - begin) / 7 * nh * workers,
  wage = suppressWarnings(as.numeric(wage_rate)),
  hourly = toupper(trimws(wage_unit)) %in% c("HOUR", "HOURLY", "HR") & is.finite(wage) & wage > 0 & wage <= 100) %>%
  left_join(dec, by = c("case_number", "fiscal_year"), relationship = "many-to-one") %>%
  mutate(q13 = !is.na(decision_date) & decision_date >= as.Date(sprintf("%d-10-01", fiscal_year - 1L)) &
    decision_date <= as.Date(sprintf("%d-06-30", fiscal_year)),
    pre_rule = !is.na(received) & received < as.Date("2025-10-02"),
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    n_counties = stringr::str_count(county_fips_list, ",") + 1L) %>%
  tidyr::separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = harmonize_county_fips_2010(trimws(county_fips_list)), share = 1 / n_counties)
outc <- ws %>% group_by(county_fips, year = as.integer(fiscal_year)) %>% summarise(
  cert_full = sum(workers * share, na.rm = TRUE), cert_q = sum(workers * share * q13, na.rm = TRUE),
  hours_full = sum(hours * share, na.rm = TRUE), hours_q = sum(hours * share * q13, na.rm = TRUE),
  wbill_q = sum(wage * hours * share * (q13 & hourly), na.rm = TRUE),
  hhours_q = sum(hours * share * (q13 & hourly & is.finite(hours)), na.rm = TRUE),
  cert_q_pre_rule = sum(workers * share * q13 * pre_rule, na.rm = TRUE), .groups = "drop")
panel <- tidyr::crossing(counties, year = 2012:LAST_FY) %>%
  left_join(outc, by = c("county_fips", "year")) %>%
  mutate(across(c(cert_full, cert_q, hours_full, hours_q, wbill_q, hhours_q, cert_q_pre_rule), ~ coalesce(.x, 0)),
    y_cert_q = cert_q / emp_farm_2011, y_hours_q = hours_q / emp_farm_2011,
    y_cert_full = cert_full / emp_farm_2011, y_hours_full = hours_full / emp_farm_2011,
    y_wage_q = ifelse(hhours_q > 0, wbill_q / hhours_q, NA_real_))
bite_check_keys(panel)

# The FY2012-FY2025 rebuild must equal the frozen pre panel.
pre <- read_csv(file.path(out_dir, "track1_pre_panel.csv"), show_col_types = FALSE,
  col_types = cols(county_fips = "c", state_fips = "c", aewr_region_id = "c"))
chk <- inner_join(pre, panel %>% filter(year <= 2025), by = c("county_fips", "year"), suffix = c("_pre", "_new"))
stopifnot(nrow(chk) == nrow(pre))
pre_check <- tibble(column = c("y_cert_q", "y_hours_q", "y_cert_full", "y_hours_full", "y_wage_q"),
  max_abs_diff = sapply(column, function(v) max(abs(chk[[paste0(v, "_pre")]] - chk[[paste0(v, "_new")]]), na.rm = TRUE)),
  na_mismatch = sapply(column, function(v) sum(is.na(chk[[paste0(v, "_pre")]]) != is.na(chk[[paste0(v, "_new")]]))))
print(pre_check)
stopifnot(all(pre_check$max_abs_diff < 1e-8), all(pre_check$na_mismatch == 0))

# Descriptives of the FY2026 data -----------------------------------------------
months <- dec %>% filter(fiscal_year %in% 2024:2026, !is.na(decision_date)) %>%
  count(fiscal_year, month = format(decision_date, "%m")) %>% tidyr::pivot_wider(names_from = fiscal_year, values_from = n, names_prefix = "fy")
desc <- panel %>% filter(year %in% 2023:2026) %>% group_by(year) %>% summarise(
  counties = n(), counties_positive_q13 = sum(cert_q > 0), positions_q13 = sum(cert_q), hours_q13 = sum(hours_q),
  mean_y_cert_q = mean(y_cert_q), mean_y_hours_q = mean(y_hours_q),
  counties_with_hourly_offer = sum(!is.na(y_wage_q)), positions_q13_received_pre_rule = sum(cert_q_pre_rule), .groups = "drop") %>%
  left_join(nh_missing, by = c("year" = "fiscal_year"))
id_write(desc, out_dir, "track1_post_descriptives.csv"); id_write(months, out_dir, "track1_post_decision_months.csv")
print(as.data.frame(desc), digits = 4); print(as.data.frame(months))

# Treatments --------------------------------------------------------------------
tr <- read_csv(file.path(out_dir, "track1_treatment_schedule.csv"), show_col_types = FALSE, col_types = cols(state_fips = "c")) %>%
  tidyr::pivot_wider(names_from = label, values_from = aewr, names_prefix = "x_")
panel <- panel %>% left_join(tr, by = c("state_fips", "year"), relationship = "many-to-one") %>%
  group_by(state_fips) %>% mutate(
    dose = x_us_skill_1[year == 2026][1] - x_us_skill_1[year == 2025][1],
    x_placebo = ifelse(year == 2025, x_us_skill_1[year == 2026][1], x_us_skill_1)) %>% ungroup() %>%
  mutate(t = year - 2025L)
stopifnot(all(is.finite(panel$x_us_skill_1)), all(is.finite(panel$dose)))
# Cohort-aligned FY2026 treatment: cases received before 2025-10-02 get the FY2025
# regional rate; county = certified-position-weighted mean; no FY2026 cases -> statutory.
rate25 <- panel %>% filter(year == 2025) %>% distinct(state_fips, r25 = x_us_skill_1)
rate26 <- panel %>% filter(year == 2026) %>% distinct(state_fips, r26 = x_us_skill_1)
coh <- ws %>% filter(fiscal_year == 2026L, q13) %>% inner_join(counties %>% select(county_fips, state_fips), by = "county_fips") %>%
  left_join(rate25, by = "state_fips") %>% left_join(rate26, by = "state_fips") %>%
  mutate(rate = ifelse(pre_rule, r25, r26), w = workers * share) %>% filter(is.finite(w), w > 0) %>%
  group_by(county_fips) %>% summarise(x_cohort = sum(w * rate) / sum(w), .groups = "drop")
panel <- panel %>% left_join(coh, by = "county_fips") %>%
  mutate(x_cohort = ifelse(year == 2026, coalesce(x_cohort, x_us_skill_1), x_us_skill_1))
# Minimum wages (FRED annual value for the calendar year containing April 1 of the FY).
mw <- read_parquet(path_int("fred_state_minwages.parquet")) %>% transmute(state_fips, year = as.integer(year),
  mw_general = coalesce(state_min_wage, federal_min_wage))
mwa <- read_parquet(path_int("state_real_minwages.parquet")) %>% transmute(state_fips, year = as.integer(year),
  mw_ag_current_coding = prevailing_ag_min_wage)
panel <- panel %>% left_join(mw, by = c("state_fips", "year")) %>% left_join(mwa, by = c("state_fips", "year"))
id_write(panel %>% select(county_fips, state_fips, aewr_region_id, year, emp_farm_2011, starts_with("y_"), cert_q, hours_q,
  starts_with("x_"), dose, mw_general, mw_ag_current_coding), out_dir, "track1_panel_fy2012_fy2026.csv")

# Estimation machinery -------------------------------------------------------------
FE <- "county_fips + aewr_region_id^year"
fit <- function(d, y, x, fe = FE, ctrl = NULL, ppml = FALSE) {
  f <- as.formula(paste(y, "~", paste(c(x, ctrl), collapse = " + "), "|", fe))
  if (ppml) fepois(f, d, vcov = ~state_fips) else feols(f, d, vcov = ~state_fips, demeaned = TRUE)
}
jackknife <- function(d, ...) {
  st <- sort(unique(d$state_fips)); args <- list(...)
  b <- vapply(st, function(s) { m <- do.call(fit, c(list(d[d$state_fips != s, , drop = FALSE]), args)); coef(m)[[args$x]] }, numeric(1))
  G <- length(st); sqrt((G - 1) / G * sum((b - mean(b))^2))
}
# Within-former-region permutation of the state-level post treatment (OLS only).
ri <- function(d, m, y, x, post_years, fe = FE) {
  d <- d[obs(m), , drop = FALSE]
  fl <- lapply(strsplit(fe, " \\+ ")[[1]], function(f) if (grepl("\\^", f)) factor(do.call(paste, d[strsplit(f, "\\^")[[1]]])) else factor(d[[f]]))
  yw <- drop(fixest::demean(cbind(d[[y]]), f = fl, nthreads = 1L))
  sx <- d %>% filter(year %in% post_years) %>% distinct(state_fips, aewr_region_id, year, v = .data[[x]])
  stopifnot(!anyDuplicated(sx[c("state_fips", "year")]))
  set.seed(SEED); res <- numeric(0); rem <- B_DRAWS
  while (rem > 0) {
    k <- min(rem, 250L)
    X <- vapply(seq_len(k), function(i) {
      p <- sx %>% group_by(aewr_region_id, year) %>% mutate(v = v[sample.int(n())]) %>% ungroup()
      v <- d[[x]]; hit <- d$year %in% post_years
      v[hit] <- p$v[match(paste(d$state_fips[hit], d$year[hit]), paste(p$state_fips, p$year))]; v
    }, numeric(nrow(d)))
    Xw <- fixest::demean(X, f = fl, nthreads = 1L)
    res <- c(res, colSums(Xw * yw) / colSums(Xw^2)); rem <- rem - k
  }
  (1 + sum(abs(res) >= abs(coef(m)[[x]]) - 1e-12)) / (B_DRAWS + 1)
}
results <- list()
est <- function(spec, d, y, x, fe = FE, ctrl = NULL, ppml = FALSE, post_years = 2026, do_ri = !ppml, do_wild = !ppml && !grepl("\\[", fe)) {
  d <- d %>% bite_complete(c(y, x, ctrl))
  m <- fit(d, y, x, fe, ctrl, ppml)
  dd <- d[obs(m), , drop = FALSE]
  G <- n_distinct(dd$state_fips)
  se_j <- jackknife(d, y = y, x = x, fe = fe, ctrl = ctrl, ppml = ppml)
  mean_y <- mean(dd[[y]]); mean_p <- mean(dd[[x]]); b <- coef(m)[[x]]
  el <- function(v) if (ppml) v * mean_p else id_elasticity(v, mean_p, mean_y)
  cr <- lapply(c("state_fips", "aewr_region_id"), function(cl) { s <- summary(m, vcov = as.formula(paste0("~", cl))); c(se(s)[[x]], n_distinct(dd[[cl]])) })
  tq <- qt(.975, G - 1)
  r <- tibble(spec = spec, outcome = y, treatment = x, estimator = if (ppml) "PPML" else "OLS",
    controls = paste(ctrl, collapse = "+"), fe = fe, n = nobs(m), counties = n_distinct(dd$county_fips), states = G,
    first_year = min(dd$year), last_year = max(dd$year), mean_y = mean_y, mean_price = mean_p,
    estimate = b, se_jackknife = se_j, p_jackknife = 2 * pt(-abs(b / se_j), G - 1),
    se_state_crv1 = cr[[1]][1], p_state_crv1 = 2 * pt(-abs(b / cr[[1]][1]), G - 1),
    se_region_crv1 = cr[[2]][1], p_region_crv1 = 2 * pt(-abs(b / cr[[2]][1]), cr[[2]][2] - 1),
    elasticity = el(b), elasticity_low_jk = el(b - tq * se_j), elasticity_high_jk = el(b + tq * se_j),
    mde_elasticity_jk = abs(el(2.8 * se_j)),
    wild_p_state = if (do_wild) id_wild(m, d, x, "state_fips", B = B_DRAWS, seed = SEED)$wild_p else NA_real_,
    ri_p_within_region = if (do_ri) ri(d, m, y, x, post_years, fe) else NA_real_)
  results[[length(results) + 1]] <<- r
  message(spec, " | ", y, " | ", x, ": b=", signif(b, 4), " se_jk=", signif(se_j, 3), " el=", signif(r$elasticity, 3))
  invisible(r)
}

# Primary (amendment 1, contrast B) and placebo -------------------------------------
dB <- panel %>% filter(year %in% 2025:2026)
dP <- panel %>% filter(year %in% 2024:2025)
for (y in c("y_cert_q", "y_hours_q")) {
  est("B_primary", dB, y, "x_us_skill_1")
  est("placebo_fy2024_fy2025", dP, y, "x_placebo", post_years = 2025)
}
for (y in c("cert_q", "hours_q")) {
  est("B_primary", dB, y, "x_us_skill_1", ppml = TRUE)
  est("placebo_fy2024_fy2025", dP, y, "x_placebo", ppml = TRUE, post_years = 2025)
}
wB <- dB %>% group_by(county_fips) %>% filter(sum(!is.na(y_wage_q)) == 2) %>% ungroup()
wP <- dP %>% group_by(county_fips) %>% filter(sum(!is.na(y_wage_q)) == 2) %>% ungroup()
est("B_primary_first_stage", wB, "y_wage_q", "x_us_skill_1")
est("placebo_fy2024_fy2025_first_stage", wP, "y_wage_q", "x_placebo", post_years = 2025)

# S3 alternative treatments, S4 cohort-aligned, S5 minimum-wage controls ----------------
for (y in c("y_cert_q", "y_hours_q")) {
  # x_cohort varies by county within state in FY2026, so state-level permutation does not apply.
  for (x in c("x_h2a_skill_1", "x_us_skill_2", "x_cohort")) est(paste0(ifelse(x == "x_cohort", "S4_", "S3_"), x), dB, y, x, do_ri = x != "x_cohort")
  est("S5_control_mw_general", dB, y, "x_us_skill_1", ctrl = "mw_general")
  est("S5_control_mw_ag_current_coding", dB, y, "x_us_skill_1", ctrl = "mw_ag_current_coding")
}
for (x in c("x_h2a_skill_1", "x_us_skill_2")) est(paste0("S3_first_stage_", x), wB, "y_wage_q", x)

# S1 base static model FY2012-FY2026 and S2 state trends ---------------------------------
for (y in c("y_cert_q", "y_hours_q")) {
  for (x in c("x_us_skill_1", "x_h2a_skill_1", "x_us_skill_2")) est(paste0("S1_base_static_", x), panel, y, x)
  est("S2_state_trends", panel, y, "x_us_skill_1", fe = paste(FE, "+ state_fips[t]"), do_ri = FALSE)
}
for (y in c("cert_q", "hours_q")) est("S1_base_static_x_us_skill_1", panel, y, "x_us_skill_1", ppml = TRUE)
wS1 <- panel %>% group_by(county_fips) %>% filter(sum(!is.na(y_wage_q)) == n_distinct(panel$year)) %>% ungroup()
est("S1_base_static_first_stage_balanced", wS1, "y_wage_q", "x_us_skill_1")
# S1 full-FY pre-period outcome check is pre-only (04); no full-FY FY2026 outcome exists.

res <- bind_rows(results)
id_write(res, out_dir, "track1_post_estimates.csv")

# Event study FY2012-FY2026 (base registry), reference FY2025 ------------------------------
ev <- list()
for (y in c("y_cert_q", "y_hours_q")) {
  m <- feols(as.formula(paste(y, "~ i(year, dose, ref = 2025) |", FE)), panel, vcov = ~state_fips)
  ct <- coeftable(m)
  ev[[y]] <- tibble(outcome = y, year = as.integer(sub("year::([0-9]+):dose", "\\1", rownames(ct))),
    estimate = ct[, 1], se_state_crv1 = ct[, 2], p = ct[, 4])
}
ev <- bind_rows(ev); id_write(ev, out_dir, "track1_post_event_study.csv")
print(as.data.frame(ev %>% filter(year >= 2022)), digits = 3)
print(as.data.frame(res %>% select(spec, outcome, treatment, estimator, estimate, se_jackknife, p_jackknife, p_state_crv1,
  wild_p_state, ri_p_within_region, elasticity, elasticity_low_jk, elasticity_high_jk, mde_elasticity_jk)), digits = 3)
