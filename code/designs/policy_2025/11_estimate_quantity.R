# Purpose: Estimate the complete FY2026 Q4 Track 1 county H-2A quantity response
# and lower-bound elasticity matrix (FY2025 vs FY2026 primary contrast, Q1-Q3
# benchmark, symmetric pre-2026-08-03 and pre-2026-08-26 cutoffs, FY2024 vs
# FY2025 placebo, declared S1-S5 sensitivities, and post-hoc epsilon_cost
# conversion linked to policy_2025 offered-wage pass-through).
# Inputs:  already hash-pinned in sources/input_vintage.json + wage_estimates.csv
# Outputs: written to $POLICY2025_WORK (published by 12_publish.R after validation)

source("code/designs/policy_2025/helpers.R")
out <- id_output_dir()
cd_verify()

suppressPackageStartupMessages({
  library(tibble)
})

LAST_FY <- 2026L
B_RI_DRAWS <- 9999L
SEED_RI <- 20261102L
COST_RATIO_CENTRAL <- 1.2401
COST_RATIO_GRID <- c(central = 1.2401, low = 1.20, high = 1.36)

# ------------------------------------------------------------------------------
# 1. Exact Track 1 Multi-State County Frame (2,921 counties, 46 states, 15 regs)
# ------------------------------------------------------------------------------
cp <- read_parquet(
  path_processed("county_year_panel.parquet"),
  col_select = c("county_fips", "state_fips", "aewr_region_id", "emp_farm_2011", "year")
) %>%
  filter(year == 2011L) %>%
  select(-year) %>%
  mutate(
    county_fips = harmonize_county_fips_2010(county_fips),
    state_fips = state_fips(state_fips),
    aewr_region_id = aewr_region_id(aewr_region_id)
  )

assert_geo_columns(cp, c("county_fips", "state_fips", "aewr_region_id"))

counties <- cp %>%
  filter(
    !state_fips %in% c("02", "15", "11"),
    is.finite(emp_farm_2011),
    emp_farm_2011 > 0
  ) %>%
  group_by(aewr_region_id) %>%
  filter(n_distinct(state_fips) > 1L) %>%
  ungroup() %>%
  arrange(aewr_region_id, state_fips, county_fips)

cd_keys(counties, "county_fips")
stopifnot(
  nrow(counties) == 2921L,
  n_distinct(counties$state_fips) == 46L,
  n_distinct(counties$aewr_region_id) == 15L
)

# ------------------------------------------------------------------------------
# 2. Rebuild FY2012-FY2026 County Outcomes with Symmetric Calendar Cutoffs
# ------------------------------------------------------------------------------
plausible_date <- function(value, fy) {
  y <- as.integer(format(value, "%Y"))
  value[!is.na(y) & abs(y - fy) > 1L] <- as.Date(NA)
  value
}

first_char <- function(x) {
  x <- as.character(x)
  x <- x[!is.na(x) & nzchar(trimws(x))]
  if (length(x) > 0L) x[[1L]] else NA_character_
}

dec <- open_dataset(path_int("h2a_disclosure_records.parquet")) %>%
  select(fiscal_year, case_number, decision_date, case_received_date) %>%
  filter(fiscal_year >= 2012L, fiscal_year <= LAST_FY) %>%
  collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year)) %>%
  group_by(case_number, fiscal_year) %>%
  summarise(
    decision_raw = first_char(decision_date),
    received_raw = first_char(case_received_date),
    .groups = "drop"
  ) %>%
  mutate(
    decision_date = plausible_date(as.Date(substr(as.character(decision_raw), 1L, 10L)), fiscal_year),
    received = as.Date(substr(as.character(received_raw), 1L, 10L))
  ) %>%
  select(case_number, fiscal_year, decision_date, received)

cd_keys(dec, c("case_number", "fiscal_year"))

ws_raw <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(
    fiscal_year, case_number, county_fips_list, adjusted_nbr_workers_certified,
    period_begin_date, period_end_date, number_of_hours, wage_rate, wage_unit
  ) %>%
  filter(fiscal_year >= 2012L, fiscal_year <= LAST_FY) %>%
  collect() %>%
  mutate(
    fiscal_year = as.integer(fiscal_year),
    begin = as.Date(period_begin_date),
    end = as.Date(period_end_date)
  ) %>%
  filter(!is.na(begin)) %>%
  mutate(nh_raw = suppressWarnings(as.numeric(number_of_hours)))

# Weekly-hours imputation mean fixed strictly to FY2012-FY2025 (matching 01_build_wages.R
# and original Track 1 08_track1_estimate.R so pre-period hours never drift).
nh_pre_mean <- mean(ws_raw$nh_raw[ws_raw$fiscal_year <= 2025L], na.rm = TRUE)
stopifnot(is.finite(nh_pre_mean), nh_pre_mean > 0)

ws <- ws_raw %>%
  mutate(
    nh = ifelse(is.na(nh_raw), nh_pre_mean, nh_raw),
    workers = as.numeric(adjusted_nbr_workers_certified),
    hours = as.numeric(end - begin) / 7 * nh * workers,
    wage = suppressWarnings(as.numeric(wage_rate)),
    hourly = toupper(trimws(wage_unit)) %in% c("HOUR", "HOURLY", "HR") &
      is.finite(wage) & wage > 0 & wage <= 100
  ) %>%
  left_join(dec, by = c("case_number", "fiscal_year"), relationship = "many-to-one") %>%
  mutate(
    fy_start = as.Date(sprintf("%d-10-01", fiscal_year - 1L)),
    fy_end = as.Date(sprintf("%d-09-30", fiscal_year)),
    w_full_fy = !is.na(decision_date) & decision_date >= fy_start & decision_date <= fy_end,
    w_q1_q3 = !is.na(decision_date) & decision_date >= fy_start &
      decision_date <= as.Date(sprintf("%d-06-30", fiscal_year)),
    w_pre_aug03_decision = w_full_fy & decision_date <= as.Date(sprintf("%d-08-02", fiscal_year)),
    w_pre_aug03_receipt = w_full_fy & !is.na(received) &
      received < as.Date(sprintf("%d-08-03", fiscal_year)),
    w_pre_aug26_decision = w_full_fy & decision_date <= as.Date(sprintf("%d-08-25", fiscal_year)),
    w_pre_aug26_receipt = w_full_fy & !is.na(received) &
      received < as.Date(sprintf("%d-08-26", fiscal_year)),
    w_ge_sep02_decision = w_full_fy & decision_date >= as.Date(sprintf("%d-09-02", fiscal_year)),
    w_ge_sep02_receipt = w_full_fy & !is.na(received) &
      received >= as.Date(sprintf("%d-09-02", fiscal_year)),
    pre_rule_2025 = !is.na(received) & received < EP_EVENT1,
    second_sched_2026 = !is.na(received) & received >= EP_EVENT2,
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    n_counties = stringr::str_count(county_fips_list, ",") + 1L
  ) %>%
  tidyr::separate_rows(county_fips_list, sep = ",") %>%
  mutate(
    county_fips = harmonize_county_fips_2010(trimws(county_fips_list)),
    share = 1 / n_counties
  )

rm(ws_raw)

# Window & legal-cutoff audit (FY2024-FY2026) on the 2,921-county multi-state frame
ws_frame <- ws %>% semi_join(counties, by = "county_fips")
cutoff_audit <- ws_frame %>%
  filter(fiscal_year %in% 2024L:2026L) %>%
  group_by(fiscal_year) %>%
  summarise(
    cases_full_fy = n_distinct(case_number[w_full_fy]),
    positions_all_fy = sum(workers * share, na.rm = TRUE),
    hours_all_fy = sum(hours * share, na.rm = TRUE),
    positions_full_fy = sum(workers * share * w_full_fy, na.rm = TRUE),
    hours_full_fy = sum(hours * share * w_full_fy, na.rm = TRUE),
    positions_q1_q3 = sum(workers * share * w_q1_q3, na.rm = TRUE),
    hours_q1_q3 = sum(hours * share * w_q1_q3, na.rm = TRUE),
    q4_positions_share = 1 - positions_q1_q3 / positions_full_fy,
    q4_hours_share = 1 - hours_q1_q3 / hours_full_fy,
    pre_2025_10_02_hours_share = if (first(fiscal_year) == 2026L) {
      sum(hours * share * (w_full_fy & pre_rule_2025), na.rm = TRUE) / hours_full_fy
    } else {
      NA_real_
    },
    ge_aug03_decision_hours_share = 1 - sum(hours * share * w_pre_aug03_decision, na.rm = TRUE) / hours_full_fy,
    ge_aug03_receipt_hours_share = 1 - sum(hours * share * w_pre_aug03_receipt, na.rm = TRUE) / hours_full_fy,
    ge_aug26_decision_hours_share = 1 - sum(hours * share * w_pre_aug26_decision, na.rm = TRUE) / hours_full_fy,
    ge_aug26_receipt_hours_share = 1 - sum(hours * share * w_pre_aug26_receipt, na.rm = TRUE) / hours_full_fy,
    ge_sep02_decision_hours_share = sum(hours * share * w_ge_sep02_decision, na.rm = TRUE) / hours_full_fy,
    ge_sep02_receipt_hours_share = sum(hours * share * w_ge_sep02_receipt, na.rm = TRUE) / hours_full_fy,
    missing_weekly_hours_share = mean(is.na(nh_raw)),
    .groups = "drop"
  ) %>%
  arrange(fiscal_year)

cd_write(cutoff_audit, out, "quantity_cutoff_audit.csv")

# Aggregate county x fiscal_year across all 6 symmetric windows + all_fy parity check
outc <- ws %>%
  group_by(county_fips, year = fiscal_year) %>%
  summarise(
    cert_all_fy = sum(workers * share, na.rm = TRUE),
    hours_all_fy = sum(hours * share, na.rm = TRUE),
    cert_full_fy = sum(workers * share * w_full_fy, na.rm = TRUE),
    hours_full_fy = sum(hours * share * w_full_fy, na.rm = TRUE),
    wbill_full_fy = sum(wage * hours * share * (w_full_fy & hourly), na.rm = TRUE),
    hhours_full_fy = sum(hours * share * (w_full_fy & hourly & is.finite(hours)), na.rm = TRUE),
    cert_q1_q3 = sum(workers * share * w_q1_q3, na.rm = TRUE),
    hours_q1_q3 = sum(hours * share * w_q1_q3, na.rm = TRUE),
    wbill_q1_q3 = sum(wage * hours * share * (w_q1_q3 & hourly), na.rm = TRUE),
    hhours_q1_q3 = sum(hours * share * (w_q1_q3 & hourly & is.finite(hours)), na.rm = TRUE),
    cert_pre_aug03_decision = sum(workers * share * w_pre_aug03_decision, na.rm = TRUE),
    hours_pre_aug03_decision = sum(hours * share * w_pre_aug03_decision, na.rm = TRUE),
    wbill_pre_aug03_decision = sum(wage * hours * share * (w_pre_aug03_decision & hourly), na.rm = TRUE),
    hhours_pre_aug03_decision = sum(hours * share * (w_pre_aug03_decision & hourly & is.finite(hours)), na.rm = TRUE),
    cert_pre_aug03_receipt = sum(workers * share * w_pre_aug03_receipt, na.rm = TRUE),
    hours_pre_aug03_receipt = sum(hours * share * w_pre_aug03_receipt, na.rm = TRUE),
    wbill_pre_aug03_receipt = sum(wage * hours * share * (w_pre_aug03_receipt & hourly), na.rm = TRUE),
    hhours_pre_aug03_receipt = sum(hours * share * (w_pre_aug03_receipt & hourly & is.finite(hours)), na.rm = TRUE),
    cert_pre_aug26_decision = sum(workers * share * w_pre_aug26_decision, na.rm = TRUE),
    hours_pre_aug26_decision = sum(hours * share * w_pre_aug26_decision, na.rm = TRUE),
    wbill_pre_aug26_decision = sum(wage * hours * share * (w_pre_aug26_decision & hourly), na.rm = TRUE),
    hhours_pre_aug26_decision = sum(hours * share * (w_pre_aug26_decision & hourly & is.finite(hours)), na.rm = TRUE),
    cert_pre_aug26_receipt = sum(workers * share * w_pre_aug26_receipt, na.rm = TRUE),
    hours_pre_aug26_receipt = sum(hours * share * w_pre_aug26_receipt, na.rm = TRUE),
    wbill_pre_aug26_receipt = sum(wage * hours * share * (w_pre_aug26_receipt & hourly), na.rm = TRUE),
    hhours_pre_aug26_receipt = sum(hours * share * (w_pre_aug26_receipt & hourly & is.finite(hours)), na.rm = TRUE),
    .groups = "drop"
  )

# ------------------------------------------------------------------------------
# 3. Treatment Schedules (April 1 of each FY) & Unified State Minimum Wages
# ------------------------------------------------------------------------------
regional_sched <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>%
  filter(
    occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference"),
    worker_category == "covered_workers"
  ) %>%
  transmute(
    state_fips = state_fips(state_fips),
    aewr = as.numeric(aewr),
    start = as.Date(effective_start),
    end = as.Date(effective_end_exclusive)
  )

statewide_sched <- read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
  filter(
    soc_code == "FIELD_LIVESTOCK_COMBINED",
    trimws(source_sheet) == "Statewide 655.120(b)(1)(i)"
  ) %>%
  transmute(
    state_fips = state_fips(state_fips),
    worker_category = toupper(trimws(as.character(worker_category))),
    skill_level = as.character(skill_level),
    aewr = as.numeric(hourly_aewr),
    start = as.Date(effective_start),
    end = as.Date(effective_end_exclusive)
  ) %>%
  distinct()

cd_keys(statewide_sched, c("state_fips", "start", "worker_category", "skill_level"))

rate_on <- function(sched, date) {
  r <- sched %>%
    filter(start <= date, is.na(end) | end > date) %>%
    select(state_fips, aewr)
  cd_keys(r, "state_fips")
  r
}

variants <- tibble(
  label = c("us_skill_1", "h2a_skill_1", "us_skill_2"),
  worker_category = c("US", "H2A", "US"),
  skill_level = c("1", "1", "2")
)

frame_states <- sort(unique(counties$state_fips))
treat_long <- bind_rows(lapply(seq_len(nrow(variants)), function(i) {
  v <- variants[i, ]
  sw_v <- statewide_sched %>%
    filter(worker_category == v$worker_category, skill_level == v$skill_level)
  bind_rows(lapply(2012L:LAST_FY, function(fy) {
    d_apr1 <- as.Date(sprintf("%d-04-01", fy))
    r <- if (fy >= 2026L) rate_on(sw_v, d_apr1) else rate_on(regional_sched, d_apr1)
    tibble(state_fips = frame_states, year = fy, label = v$label) %>%
      left_join(r, by = "state_fips", relationship = "one-to-one")
  }))
}))

stopifnot(all(is.finite(treat_long$aewr)))
treat_wide <- treat_long %>%
  tidyr::pivot_wider(names_from = label, values_from = aewr, names_prefix = "x_")

# Second statewide schedule (effective 2026-08-03) for dual-schedule cohort check
rate26_e2 <- statewide_sched %>%
  filter(worker_category == "US", skill_level == "1", start == EP_EVENT2) %>%
  transmute(state_fips, r26_e2 = aewr)
cd_keys(rate26_e2, "state_fips")

rate25 <- treat_wide %>% filter(year == 2025L) %>% select(state_fips, r25 = x_us_skill_1)
rate26_e1 <- treat_wide %>% filter(year == 2026L) %>% select(state_fips, r26_e1 = x_us_skill_1)

coh_fy26 <- ws %>%
  filter(fiscal_year == 2026L, w_full_fy) %>%
  inner_join(counties %>% select(county_fips, state_fips), by = "county_fips") %>%
  left_join(rate25, by = "state_fips") %>%
  left_join(rate26_e1, by = "state_fips") %>%
  left_join(rate26_e2, by = "state_fips") %>%
  mutate(
    rate_coh = ifelse(pre_rule_2025, r25, r26_e1),
    rate_dual = case_when(
      pre_rule_2025 ~ r25,
      second_sched_2026 ~ r26_e2,
      TRUE ~ r26_e1
    ),
    w = workers * share
  ) %>%
  filter(is.finite(w), w > 0) %>%
  group_by(county_fips) %>%
  summarise(
    x_cohort_2026 = sum(w * rate_coh) / sum(w),
    x_cohort_dual_2026 = sum(w * rate_dual) / sum(w),
    .groups = "drop"
  )

# Unified minimum wages (NEVER fred_state_minwages.parquet!)
mw <- read_parquet(path_int("state_real_minwages.parquet")) %>%
  filter(year >= 2012L, year <= LAST_FY) %>%
  transmute(
    state_fips = state_fips(state_fips),
    year = as.integer(year),
    mw_general = as.numeric(prevailing_min_wage),
    mw_ag = as.numeric(prevailing_ag_min_wage)
  )
cd_keys(mw, c("state_fips", "year"))

panel <- tidyr::crossing(counties, year = 2012L:LAST_FY) %>%
  left_join(outc, by = c("county_fips", "year"), relationship = "one-to-one") %>%
  mutate(across(starts_with(c("cert_", "hours_", "wbill_", "hhours_")), ~ coalesce(.x, 0))) %>%
  left_join(treat_wide, by = c("state_fips", "year"), relationship = "many-to-one") %>%
  left_join(coh_fy26, by = "county_fips", relationship = "many-to-one") %>%
  left_join(mw, by = c("state_fips", "year"), relationship = "many-to-one") %>%
  group_by(state_fips) %>%
  mutate(
    dose = x_us_skill_1[year == 2026L][1L] - x_us_skill_1[year == 2025L][1L],
    x_placebo = ifelse(year == 2025L, x_us_skill_1[year == 2026L][1L], x_us_skill_1)
  ) %>%
  ungroup() %>%
  mutate(
    t = year - 2025L,
    x_cohort = ifelse(year == 2026L, coalesce(x_cohort_2026, x_us_skill_1), x_us_skill_1),
    x_cohort_dual = ifelse(year == 2026L, coalesce(x_cohort_dual_2026, x_us_skill_1), x_us_skill_1),
    ln_x_us_skill_1 = log(x_us_skill_1),
    ln_x_placebo = log(x_placebo),
    y_cert_full_fy = cert_full_fy / emp_farm_2011,
    y_hours_full_fy = hours_full_fy / emp_farm_2011,
    y_wage_full_fy = ifelse(hhours_full_fy > 0, wbill_full_fy / hhours_full_fy, NA_real_),
    y_cert_q1_q3 = cert_q1_q3 / emp_farm_2011,
    y_hours_q1_q3 = hours_q1_q3 / emp_farm_2011,
    y_wage_q1_q3 = ifelse(hhours_q1_q3 > 0, wbill_q1_q3 / hhours_q1_q3, NA_real_),
    y_cert_pre_aug03_decision = cert_pre_aug03_decision / emp_farm_2011,
    y_hours_pre_aug03_decision = hours_pre_aug03_decision / emp_farm_2011,
    y_wage_pre_aug03_decision = ifelse(hhours_pre_aug03_decision > 0, wbill_pre_aug03_decision / hhours_pre_aug03_decision, NA_real_),
    y_cert_pre_aug03_receipt = cert_pre_aug03_receipt / emp_farm_2011,
    y_hours_pre_aug03_receipt = hours_pre_aug03_receipt / emp_farm_2011,
    y_wage_pre_aug03_receipt = ifelse(hhours_pre_aug03_receipt > 0, wbill_pre_aug03_receipt / hhours_pre_aug03_receipt, NA_real_),
    y_cert_pre_aug26_decision = cert_pre_aug26_decision / emp_farm_2011,
    y_hours_pre_aug26_decision = hours_pre_aug26_decision / emp_farm_2011,
    y_wage_pre_aug26_decision = ifelse(hhours_pre_aug26_decision > 0, wbill_pre_aug26_decision / hhours_pre_aug26_decision, NA_real_),
    y_cert_pre_aug26_receipt = cert_pre_aug26_receipt / emp_farm_2011,
    y_hours_pre_aug26_receipt = hours_pre_aug26_receipt / emp_farm_2011,
    y_wage_pre_aug26_receipt = ifelse(hhours_pre_aug26_receipt > 0, wbill_pre_aug26_receipt / hhours_pre_aug26_receipt, NA_real_)
  ) %>%
  select(-x_cohort_2026, -x_cohort_dual_2026)

assert_geo_columns(panel, c("county_fips", "state_fips", "aewr_region_id"))
cd_keys(panel, c("county_fips", "year"))
stopifnot(
  nrow(panel) == 2921L * 15L,
  all(is.finite(panel$x_us_skill_1)),
  all(is.finite(panel$dose)),
  all(is.finite(panel$mw_general)),
  all(is.finite(panel$mw_ag))
)

write_parquet(panel, file.path(out, "quantity_county_fy.parquet"))

# ------------------------------------------------------------------------------
# 4. Estimation Machinery (Exact 46-State CV3J Jackknife + Within-Region RI)
# ------------------------------------------------------------------------------
FE_BASE <- "county_fips + aewr_region_id^year"

complete_df <- function(d, cols) {
  d <- as.data.frame(d)
  for (v in cols) {
    keep <- if (is.numeric(d[[v]])) is.finite(d[[v]]) else (!is.na(d[[v]]) & nzchar(as.character(d[[v]])))
    d <- d[keep, , drop = FALSE]
  }
  d
}

fit_model <- function(d, y, x, fe = FE_BASE, ctrl = NULL, ppml = FALSE) {
  rhs <- paste(c(x, ctrl), collapse = " + ")
  fml <- stats::as.formula(paste0(y, " ~ ", rhs, " | ", fe))
  if (ppml) {
    fixest::fepois(fml, data = d, vcov = ~state_fips, notes = FALSE)
  } else {
    fixest::feols(fml, data = d, vcov = ~state_fips, notes = FALSE)
  }
}

within_region_ri <- function(d, m, y, x, post_years, fe = FE_BASE) {
  dd <- d[fixest::obs(m), , drop = FALSE]
  fl <- lapply(strsplit(fe, " \\+ ")[[1L]], function(part) {
    if (grepl("\\^", part)) {
      factor(do.call(paste, c(dd[strsplit(part, "\\^")[[1L]]], list(sep = "_"))))
    } else {
      factor(dd[[part]])
    }
  })
  yw <- drop(fixest::demean(cbind(dd[[y]]), f = fl, nthreads = 1L))
  sx <- dd %>%
    filter(year %in% post_years) %>%
    distinct(state_fips, aewr_region_id, year, v = .data[[x]])
  stopifnot(!anyDuplicated(sx[c("state_fips", "year")]))
  set.seed(SEED_RI)
  res <- numeric(0)
  rem <- B_RI_DRAWS
  hit <- dd$year %in% post_years
  hit_keys <- paste(dd$state_fips[hit], dd$year[hit], sep = "_")
  while (rem > 0L) {
    k <- min(rem, 250L)
    X <- vapply(seq_len(k), function(i) {
      p <- sx %>%
        group_by(aewr_region_id, year) %>%
        mutate(v = v[sample.int(n())]) %>%
        ungroup()
      v <- dd[[x]]
      v[hit] <- p$v[match(hit_keys, paste(p$state_fips, p$year, sep = "_"))]
      v
    }, numeric(nrow(dd)))
    Xw <- fixest::demean(X, f = fl, nthreads = 1L)
    res <- c(res, colSums(Xw * yw) / colSums(Xw^2))
    rem <- rem - k
  }
  (1 + sum(abs(res) >= abs(stats::coef(m)[[x]]) - 1e-12)) / (B_RI_DRAWS + 1)
}

results_list <- list()
loo_list <- list()

run_est <- function(
  window,
  contrast,
  specification,
  outcome_label,
  d,
  y_col,
  x_col,
  fe = FE_BASE,
  ctrl = NULL,
  ppml = FALSE,
  log_rate = FALSE,
  is_first_stage = FALSE,
  post_years = 2026L,
  do_ri = FALSE
) {
  needed <- unique(c(y_col, x_col, ctrl, "county_fips", "state_fips", "aewr_region_id", "year"))
  if (grepl("\\[t\\]", fe)) needed <- unique(c(needed, "t"))
  cc <- complete_df(d, needed)
  m <- fit_model(cc, y_col, x_col, fe = fe, ctrl = ctrl, ppml = ppml)
  dd <- cc[fixest::obs(m), , drop = FALSE]
  states <- sort(unique(dd$state_fips))
  G <- length(states)
  R_n <- n_distinct(dd$aewr_region_id)
  b <- unname(stats::coef(m)[[x_col]])

  loo_vals <- vapply(states, function(s) {
    mm <- tryCatch(
      fit_model(cc[cc$state_fips != s, , drop = FALSE], y_col, x_col, fe = fe, ctrl = ctrl, ppml = ppml),
      error = function(e) NULL
    )
    if (is.null(mm) || !x_col %in% names(stats::coef(mm))) NA_real_ else unname(stats::coef(mm)[[x_col]])
  }, numeric(1))

  ok_jk <- is.finite(b) && length(loo_vals) == G && all(is.finite(loo_vals))
  se_jk <- if (ok_jk) sqrt((G - 1) / G * sum((loo_vals - mean(loo_vals))^2)) else NA_real_
  tq <- stats::qt(0.975, df = G - 1L)
  p_jk <- if (ok_jk && se_jk > 0) 2 * stats::pt(-abs(b / se_jk), df = G - 1L) else NA_real_

  s_st <- summary(m, vcov = ~state_fips)
  se_st <- unname(fixest::se(s_st)[[x_col]])
  p_st <- 2 * stats::pt(-abs(b / se_st), df = G - 1L)

  s_reg <- summary(m, vcov = ~aewr_region_id)
  se_reg <- unname(fixest::se(s_reg)[[x_col]])
  p_reg <- 2 * stats::pt(-abs(b / se_reg), df = R_n - 1L)

  mean_y <- mean(dd[[y_col]])
  mean_p <- mean(dd[[x_col]])

  if (is_first_stage) {
    el_b <- NA_real_
    el_se <- NA_real_
    el_lo <- NA_real_
    el_hi <- NA_real_
    el_mde <- NA_real_
  } else if (log_rate) {
    el_b <- b
    el_se <- se_jk
    el_lo <- if (ok_jk) b - tq * se_jk else NA_real_
    el_hi <- if (ok_jk) b + tq * se_jk else NA_real_
    el_mde <- if (ok_jk) 2.8 * se_jk else NA_real_
  } else if (ppml) {
    el_b <- b * mean_p
    el_se <- se_jk * mean_p
    el_lo <- if (ok_jk) (b - tq * se_jk) * mean_p else NA_real_
    el_hi <- if (ok_jk) (b + tq * se_jk) * mean_p else NA_real_
    el_mde <- if (ok_jk) 2.8 * se_jk * mean_p else NA_real_
  } else {
    scale_el <- mean_p / mean_y
    el_b <- b * scale_el
    el_se <- se_jk * scale_el
    el_lo <- if (ok_jk) (b - tq * se_jk) * scale_el else NA_real_
    el_hi <- if (ok_jk) (b + tq * se_jk) * scale_el else NA_real_
    el_mde <- if (ok_jk) 2.8 * se_jk * scale_el else NA_real_
  }

  ri_p <- if (isTRUE(do_ri) && !ppml) within_region_ri(cc, m, y_col, x_col, post_years, fe = fe) else NA_real_

  row <- tibble(
    track = "quantity",
    window = window,
    contrast = contrast,
    specification = specification,
    outcome = outcome_label,
    estimator = if (ppml) "PPML" else "OLS",
    treatment = x_col,
    controls = if (is.null(ctrl)) "none" else paste(ctrl, collapse = "+"),
    fe = fe,
    status = if (ok_jk) "estimated" else "jackknife_not_identified",
    n = as.integer(stats::nobs(m)),
    counties = as.integer(n_distinct(dd$county_fips)),
    states = as.integer(G),
    regions = as.integer(R_n),
    first_year = as.integer(min(dd$year)),
    last_year = as.integer(max(dd$year)),
    mean_y = mean_y,
    mean_price = mean_p,
    estimate = b,
    se_jackknife = se_jk,
    ci95_low_jk = if (ok_jk) b - tq * se_jk else NA_real_,
    ci95_high_jk = if (ok_jk) b + tq * se_jk else NA_real_,
    p_jackknife = p_jk,
    se_state_crv1 = se_st,
    p_state_crv1 = p_st,
    se_region_crv1 = se_reg,
    p_region_crv1 = p_reg,
    ri_p_within_region = ri_p,
    elasticity = el_b,
    elasticity_se_jk = el_se,
    elasticity_low_jk = el_lo,
    elasticity_high_jk = el_hi,
    mde_elasticity_jk = el_mde
  )

  loo_tbl <- tibble(
    window = window,
    contrast = contrast,
    specification = specification,
    outcome = outcome_label,
    estimator = if (ppml) "PPML" else "OLS",
    treatment = x_col,
    omitted_state = states,
    estimate = loo_vals
  )

  results_list[[length(results_list) + 1L]] <<- row
  loo_list[[length(loo_list) + 1L]] <<- loo_tbl
  invisible(list(row = row, loo = loo_tbl))
}

# Helper for continuing-county hourly-offer first stage (2 years)
balanced_wage_df <- function(d, y_wage_col) {
  n_yrs <- n_distinct(d$year)
  d %>%
    group_by(county_fips) %>%
    filter(sum(is.finite(.data[[y_wage_col]])) == n_yrs) %>%
    ungroup()
}

dB <- panel %>% filter(year %in% 2025L:2026L)
dP <- panel %>% filter(year %in% 2024L:2025L)

# ------------------------------------------------------------------------------
# 5. Estimate Headline (full_fy), Matched Q1-Q3, Cutoffs, Placebos & S1-S5
# ------------------------------------------------------------------------------

# 5a. Headline complete FY2026 Q4 (full_fy, FY2025 vs FY2026) — 7 rows
run_est("full_fy", "fy2025_2026", "B_primary", "hours", dB, "hours_full_fy", "x_us_skill_1", ppml = TRUE)
run_est("full_fy", "fy2025_2026", "B_primary", "cert", dB, "cert_full_fy", "x_us_skill_1", ppml = TRUE)
run_est("full_fy", "fy2025_2026", "B_primary", "y_hours", dB, "y_hours_full_fy", "x_us_skill_1", do_ri = TRUE)
run_est("full_fy", "fy2025_2026", "B_primary", "y_cert", dB, "y_cert_full_fy", "x_us_skill_1", do_ri = TRUE)
run_est("full_fy", "fy2025_2026", "B_primary_log_rate", "hours", dB, "hours_full_fy", "ln_x_us_skill_1", ppml = TRUE, log_rate = TRUE)
run_est("full_fy", "fy2025_2026", "B_primary_log_rate", "cert", dB, "cert_full_fy", "ln_x_us_skill_1", ppml = TRUE, log_rate = TRUE)
fs_full_res <- run_est(
  "full_fy", "fy2025_2026", "B_primary_first_stage", "y_wage",
  balanced_wage_df(dB, "y_wage_full_fy"), "y_wage_full_fy", "x_us_skill_1",
  is_first_stage = TRUE, do_ri = TRUE
)

# 5b. Full-FY FY2024 vs FY2025 Placebo — 7 rows
run_est("full_fy", "placebo_fy2024_2025", "placebo_fy2024_fy2025", "hours", dP, "hours_full_fy", "x_placebo", ppml = TRUE, post_years = 2025L)
run_est("full_fy", "placebo_fy2024_2025", "placebo_fy2024_fy2025", "cert", dP, "cert_full_fy", "x_placebo", ppml = TRUE, post_years = 2025L)
run_est("full_fy", "placebo_fy2024_2025", "placebo_fy2024_fy2025", "y_hours", dP, "y_hours_full_fy", "x_placebo", post_years = 2025L, do_ri = TRUE)
run_est("full_fy", "placebo_fy2024_2025", "placebo_fy2024_fy2025", "y_cert", dP, "y_cert_full_fy", "x_placebo", post_years = 2025L, do_ri = TRUE)
run_est("full_fy", "placebo_fy2024_2025", "placebo_fy2024_fy2025_log_rate", "hours", dP, "hours_full_fy", "ln_x_placebo", ppml = TRUE, log_rate = TRUE, post_years = 2025L)
run_est("full_fy", "placebo_fy2024_2025", "placebo_fy2024_fy2025_log_rate", "cert", dP, "cert_full_fy", "ln_x_placebo", ppml = TRUE, log_rate = TRUE, post_years = 2025L)
run_est(
  "full_fy", "placebo_fy2024_2025", "placebo_fy2024_fy2025_first_stage", "y_wage",
  balanced_wage_df(dP, "y_wage_full_fy"), "y_wage_full_fy", "x_placebo",
  is_first_stage = TRUE, post_years = 2025L, do_ri = TRUE
)

# 5c. Matched Q1-Q3 Benchmark (FY2025 vs FY2026 & FY2024 vs FY2025 placebo) — 12 rows
run_est("q1_q3", "fy2025_2026", "B_primary", "hours", dB, "hours_q1_q3", "x_us_skill_1", ppml = TRUE)
run_est("q1_q3", "fy2025_2026", "B_primary", "cert", dB, "cert_q1_q3", "x_us_skill_1", ppml = TRUE)
run_est("q1_q3", "fy2025_2026", "B_primary", "y_hours", dB, "y_hours_q1_q3", "x_us_skill_1", do_ri = TRUE)
run_est("q1_q3", "fy2025_2026", "B_primary", "y_cert", dB, "y_cert_q1_q3", "x_us_skill_1", do_ri = TRUE)
run_est("q1_q3", "fy2025_2026", "B_primary_log_rate", "hours", dB, "hours_q1_q3", "ln_x_us_skill_1", ppml = TRUE, log_rate = TRUE)
run_est("q1_q3", "fy2025_2026", "B_primary_log_rate", "cert", dB, "cert_q1_q3", "ln_x_us_skill_1", ppml = TRUE, log_rate = TRUE)
fs_q13_res <- run_est(
  "q1_q3", "fy2025_2026", "B_primary_first_stage", "y_wage",
  balanced_wage_df(dB, "y_wage_q1_q3"), "y_wage_q1_q3", "x_us_skill_1",
  is_first_stage = TRUE, do_ri = TRUE
)

run_est("q1_q3", "placebo_fy2024_2025", "placebo_fy2024_fy2025", "hours", dP, "hours_q1_q3", "x_placebo", ppml = TRUE, post_years = 2025L)
run_est("q1_q3", "placebo_fy2024_2025", "placebo_fy2024_fy2025", "cert", dP, "cert_q1_q3", "x_placebo", ppml = TRUE, post_years = 2025L)
run_est("q1_q3", "placebo_fy2024_2025", "placebo_fy2024_fy2025", "y_hours", dP, "y_hours_q1_q3", "x_placebo", post_years = 2025L, do_ri = TRUE)
run_est("q1_q3", "placebo_fy2024_2025", "placebo_fy2024_fy2025", "y_cert", dP, "y_cert_q1_q3", "x_placebo", post_years = 2025L, do_ri = TRUE)
run_est(
  "q1_q3", "placebo_fy2024_2025", "placebo_fy2024_fy2025_first_stage", "y_wage",
  balanced_wage_df(dP, "y_wage_q1_q3"), "y_wage_q1_q3", "x_placebo",
  is_first_stage = TRUE, post_years = 2025L
)

# 5d. Symmetric Pre-2026-08-03 & Pre-2026-08-26 Cutoff Sensitivities — 20 rows
for (win in c("pre_aug03_decision", "pre_aug03_receipt", "pre_aug26_decision", "pre_aug26_receipt")) {
  run_est(win, "fy2025_2026", "B_primary", "hours", dB, paste0("hours_", win), "x_us_skill_1", ppml = TRUE)
  run_est(win, "fy2025_2026", "B_primary", "cert", dB, paste0("cert_", win), "x_us_skill_1", ppml = TRUE)
  run_est(win, "fy2025_2026", "B_primary", "y_hours", dB, paste0("y_hours_", win), "x_us_skill_1")
  run_est(win, "fy2025_2026", "B_primary", "y_cert", dB, paste0("y_cert_", win), "x_us_skill_1")
  run_est(
    win, "fy2025_2026", "B_primary_first_stage", "y_wage",
    balanced_wage_df(dB, paste0("y_wage_", win)), paste0("y_wage_", win), "x_us_skill_1",
    is_first_stage = TRUE
  )
}

# 5e. Declared Track 1 Sensitivities on Complete FY2026 Q4 (full_fy) — 33 rows
# S3: alternative statutory rates (10 rows)
for (x_alt in c("x_h2a_skill_1", "x_us_skill_2")) {
  sp <- paste0("S3_", sub("^x_", "", x_alt))
  run_est("full_fy", "fy2025_2026", sp, "hours", dB, "hours_full_fy", x_alt, ppml = TRUE)
  run_est("full_fy", "fy2025_2026", sp, "cert", dB, "cert_full_fy", x_alt, ppml = TRUE)
  run_est("full_fy", "fy2025_2026", sp, "y_hours", dB, "y_hours_full_fy", x_alt)
  run_est("full_fy", "fy2025_2026", sp, "y_cert", dB, "y_cert_full_fy", x_alt)
  run_est(
    "full_fy", "fy2025_2026", paste0("S3_first_stage_", sub("^x_", "", x_alt)), "y_wage",
    balanced_wage_df(dB, "y_wage_full_fy"), "y_wage_full_fy", x_alt,
    is_first_stage = TRUE
  )
}

# S4: cohort-aligned & dual-schedule cohort treatments (8 rows)
for (x_coh in c("x_cohort", "x_cohort_dual")) {
  sp <- paste0("S4_", x_coh)
  run_est("full_fy", "fy2025_2026", sp, "hours", dB, "hours_full_fy", x_coh, ppml = TRUE)
  run_est("full_fy", "fy2025_2026", sp, "cert", dB, "cert_full_fy", x_coh, ppml = TRUE)
  run_est("full_fy", "fy2025_2026", sp, "y_hours", dB, "y_hours_full_fy", x_coh)
  run_est("full_fy", "fy2025_2026", sp, "y_cert", dB, "y_cert_full_fy", x_coh)
}

# S5: unified 46-state minimum-wage controls (8 rows)
for (mw_col in c("mw_general", "mw_ag")) {
  sp <- paste0("S5_control_", mw_col)
  run_est("full_fy", "fy2025_2026", sp, "hours", dB, "hours_full_fy", "x_us_skill_1", ctrl = mw_col, ppml = TRUE)
  run_est("full_fy", "fy2025_2026", sp, "cert", dB, "cert_full_fy", "x_us_skill_1", ctrl = mw_col, ppml = TRUE)
  run_est("full_fy", "fy2025_2026", sp, "y_hours", dB, "y_hours_full_fy", "x_us_skill_1", ctrl = mw_col)
  run_est("full_fy", "fy2025_2026", sp, "y_cert", dB, "y_cert_full_fy", "x_us_skill_1", ctrl = mw_col)
}

# S1 base static FY2012-FY2026 & S2 state linear trends (7 rows)
run_est("full_fy", "static_fy2012_2026", "S1_base_static", "hours", panel, "hours_full_fy", "x_us_skill_1", ppml = TRUE)
run_est("full_fy", "static_fy2012_2026", "S1_base_static", "cert", panel, "cert_full_fy", "x_us_skill_1", ppml = TRUE)
run_est("full_fy", "static_fy2012_2026", "S1_base_static", "y_hours", panel, "y_hours_full_fy", "x_us_skill_1")
run_est("full_fy", "static_fy2012_2026", "S1_base_static", "y_cert", panel, "y_cert_full_fy", "x_us_skill_1")
run_est(
  "full_fy", "static_fy2012_2026", "S1_base_static_first_stage", "y_wage",
  balanced_wage_df(panel, "y_wage_full_fy"), "y_wage_full_fy", "x_us_skill_1",
  is_first_stage = TRUE
)
run_est("full_fy", "static_fy2012_2026", "S2_state_trends", "y_hours", panel, "y_hours_full_fy", "x_us_skill_1", fe = paste(FE_BASE, "+ state_fips[t]"))
run_est("full_fy", "static_fy2012_2026", "S2_state_trends", "y_cert", panel, "y_cert_full_fy", "x_us_skill_1", fe = paste(FE_BASE, "+ state_fips[t]"))

qty_estimates <- bind_rows(results_list)
qty_deletions <- bind_rows(loo_list)

stopifnot(nrow(qty_estimates) == 79L)
cd_keys(qty_estimates, c("track", "window", "contrast", "specification", "outcome", "estimator", "treatment"))
cd_write(qty_estimates, out, "quantity_estimates.csv")
cd_write(qty_deletions, out, "quantity_state_deletions.csv")

# ------------------------------------------------------------------------------
# 6. Full-FY FY2012-FY2026 Event Study (ref = FY2025, state CRV1 + CV3J)
# ------------------------------------------------------------------------------
es_years_nonref <- setdiff(2012L:LAST_FY, 2025L)
es_terms <- paste0("year::", es_years_nonref, ":dose")
es_rows_list <- list()
for (es_spec in list(
  list(out = "hours", col = "hours_full_fy", ppml = TRUE),
  list(out = "cert", col = "cert_full_fy", ppml = TRUE),
  list(out = "y_hours", col = "y_hours_full_fy", ppml = FALSE),
  list(out = "y_cert", col = "y_cert_full_fy", ppml = FALSE)
)) {
  fml_es <- stats::as.formula(paste0(es_spec$col, " ~ i(year, dose, ref = 2025) | ", FE_BASE))
  m_es <- if (es_spec$ppml) fixest::fepois(fml_es, data = panel, vcov = ~state_fips, notes = FALSE) else fixest::feols(fml_es, data = panel, vcov = ~state_fips, notes = FALSE)
  dd_es <- panel[fixest::obs(m_es), , drop = FALSE]
  st_es <- sort(unique(dd_es$state_fips))
  G_es <- length(st_es)
  loo_mat <- matrix(NA_real_, nrow = G_es, ncol = length(es_terms), dimnames = list(st_es, es_terms))
  for (i in seq_len(G_es)) {
    sub_d <- panel[panel$state_fips != st_es[[i]], , drop = FALSE]
    mm <- tryCatch(
      if (es_spec$ppml) fixest::fepois(fml_es, data = sub_d, vcov = ~state_fips, notes = FALSE) else fixest::feols(fml_es, data = sub_d, vcov = ~state_fips, notes = FALSE),
      error = function(e) NULL
    )
    if (!is.null(mm)) {
      bb <- stats::coef(mm)
      hit <- intersect(es_terms, names(bb))
      if (length(hit) > 0L) loo_mat[i, hit] <- unname(bb[hit])
    }
  }
  b_es <- unname(stats::coef(m_es)[es_terms])
  se_crv1_es <- unname(fixest::se(m_es)[es_terms])
  se_jk_es <- apply(loo_mat, 2L, function(v) if (all(is.finite(v))) sqrt((G_es - 1) / G_es * sum((v - mean(v))^2)) else NA_real_)
  tq_es <- stats::qt(0.975, df = G_es - 1L)
  es_rows_list[[es_spec$out]] <- bind_rows(
    tibble(
      window = "full_fy",
      outcome = es_spec$out,
      estimator = if (es_spec$ppml) "PPML" else "OLS",
      year = es_years_nonref,
      estimate = b_es,
      se_jackknife = unname(se_jk_es),
      ci95_low_jk = b_es - tq_es * unname(se_jk_es),
      ci95_high_jk = b_es + tq_es * unname(se_jk_es),
      p_jackknife = 2 * stats::pt(-abs(b_es / unname(se_jk_es)), df = G_es - 1L),
      se_state_crv1 = se_crv1_es,
      p_state_crv1 = 2 * stats::pt(-abs(b_es / se_crv1_es), df = G_es - 1L),
      states = as.integer(G_es)
    ),
    tibble(
      window = "full_fy",
      outcome = es_spec$out,
      estimator = if (es_spec$ppml) "PPML" else "OLS",
      year = 2025L,
      estimate = 0,
      se_jackknife = 0,
      ci95_low_jk = 0,
      ci95_high_jk = 0,
      p_jackknife = NA_real_,
      se_state_crv1 = 0,
      p_state_crv1 = NA_real_,
      states = as.integer(G_es)
    )
  ) %>% arrange(year)
}
cd_write(bind_rows(es_rows_list), out, "quantity_event_study.csv")

# ------------------------------------------------------------------------------
# 7. Post-Hoc epsilon_cost Conversion (Dynamic policy_2025 R7 + County Paired JK)
# ------------------------------------------------------------------------------
wage_est_path <- file.path(out, "wage_estimates.csv")
stopifnot(file.exists(wage_est_path))
wage_primary <- read_csv(wage_est_path, show_col_types = FALSE) %>%
  filter(tier == "R7", tau == 2026L, specification == "occupation_dw", weights == "h0")
stopifnot(nrow(wage_primary) == 1L, is.finite(wage_primary$estimate))

pi_policy2025 <- wage_primary$estimate[[1]]
pi_county_full <- fs_full_res$row$estimate[[1]]
pi_county_q13 <- fs_q13_res$row$estimate[[1]]
w_bar_full_hw <- sum(dB$wbill_full_fy, na.rm = TRUE) / sum(dB$hhours_full_fy, na.rm = TRUE)
w_bar_q13_hw <- sum(dB$wbill_q1_q3, na.rm = TRUE) / sum(dB$hhours_q1_q3, na.rm = TRUE)

convert_targets <- qty_estimates %>%
  filter(
    (window == "full_fy" & specification == "B_primary" & outcome %in% c("hours", "cert", "y_hours", "y_cert")) |
      (window == "q1_q3" & specification == "B_primary" & outcome == "hours")
  )

conv_rows <- list()
for (i in seq_len(nrow(convert_targets))) {
  r <- convert_targets[i, ]
  w_bar <- if (r$window == "full_fy") w_bar_full_hw else w_bar_q13_hw
  fs_obj <- if (r$window == "full_fy") fs_full_res else fs_q13_res
  pi_cty <- fs_obj$row$estimate[[1]]
  semi_b <- if (r$estimator == "PPML") r$estimate else r$estimate / r$mean_y
  semi_se <- if (r$estimator == "PPML") r$se_jackknife else r$se_jackknife / r$mean_y
  tq <- stats::qt(0.975, df = r$states - 1L)

  loo_q <- qty_deletions %>%
    filter(
      window == r$window, contrast == r$contrast, specification == r$specification,
      outcome == r$outcome, estimator == r$estimator, treatment == r$treatment
    ) %>%
    select(omitted_state, est_q = estimate)
  loo_pi <- fs_obj$loo %>% select(omitted_state, est_pi = estimate)
  paired <- inner_join(loo_q, loo_pi, by = "omitted_state")
  stopifnot(nrow(paired) == r$states)

  for (r_name in names(COST_RATIO_GRID)) {
    rc <- COST_RATIO_GRID[[r_name]]
    # (1) Fixed-pi conversion using policy_2025 full-FY R7 continuing-filer pass-through
    fac_p25 <- rc * w_bar / pi_policy2025
    conv_rows[[length(conv_rows) + 1L]] <- tibble(
      window = r$window,
      outcome = r$outcome,
      estimator = r$estimator,
      pass_through_source = "policy_2025_R7_occupation_dw_h0_fixed_pi",
      cost_ratio_case = r_name,
      cost_ratio = rc,
      w_bar_offered = w_bar,
      pi_pass_through = pi_policy2025,
      semi_per_stat_dollar = semi_b,
      eps_statutory = r$elasticity,
      eps_statutory_low_jk = r$elasticity_low_jk,
      eps_statutory_high_jk = r$elasticity_high_jk,
      eps_cost = semi_b * fac_p25,
      eps_cost_se_jk = semi_se * fac_p25,
      eps_cost_low_jk = (semi_b - tq * semi_se) * fac_p25,
      eps_cost_high_jk = (semi_b + tq * semi_se) * fac_p25,
      mde_eps_cost_jk = 2.8 * semi_se * fac_p25,
      conversion_status = "post_hoc_secondary_fixed_pi"
    )
    # (2) Fixed-pi conversion using Track 1 county first stage
    fac_cty <- rc * w_bar / pi_cty
    conv_rows[[length(conv_rows) + 1L]] <- tibble(
      window = r$window,
      outcome = r$outcome,
      estimator = r$estimator,
      pass_through_source = "track1_county_first_stage_fixed_pi",
      cost_ratio_case = r_name,
      cost_ratio = rc,
      w_bar_offered = w_bar,
      pi_pass_through = pi_cty,
      semi_per_stat_dollar = semi_b,
      eps_statutory = r$elasticity,
      eps_statutory_low_jk = r$elasticity_low_jk,
      eps_statutory_high_jk = r$elasticity_high_jk,
      eps_cost = semi_b * fac_cty,
      eps_cost_se_jk = semi_se * fac_cty,
      eps_cost_low_jk = (semi_b - tq * semi_se) * fac_cty,
      eps_cost_high_jk = (semi_b + tq * semi_se) * fac_cty,
      mde_eps_cost_jk = 2.8 * semi_se * fac_cty,
      conversion_status = "post_hoc_secondary_fixed_pi"
    )
    # (3) Paired 46-state jackknife ratio accounting for uncertainty in county pi
    semi_loo <- if (r$estimator == "PPML") paired$est_q else paired$est_q / r$mean_y
    eps_cost_loo <- semi_loo * rc * w_bar / paired$est_pi
    G_p <- nrow(paired)
    se_paired <- sqrt((G_p - 1) / G_p * sum((eps_cost_loo - mean(eps_cost_loo))^2))
    eps_c_pt <- semi_b * fac_cty
    conv_rows[[length(conv_rows) + 1L]] <- tibble(
      window = r$window,
      outcome = r$outcome,
      estimator = r$estimator,
      pass_through_source = "track1_county_first_stage_paired_state_jk",
      cost_ratio_case = r_name,
      cost_ratio = rc,
      w_bar_offered = w_bar,
      pi_pass_through = pi_cty,
      semi_per_stat_dollar = semi_b,
      eps_statutory = r$elasticity,
      eps_statutory_low_jk = r$elasticity_low_jk,
      eps_statutory_high_jk = r$elasticity_high_jk,
      eps_cost = eps_c_pt,
      eps_cost_se_jk = se_paired,
      eps_cost_low_jk = eps_c_pt - tq * se_paired,
      eps_cost_high_jk = eps_c_pt + tq * se_paired,
      mde_eps_cost_jk = 2.8 * se_paired,
      conversion_status = "post_hoc_secondary_paired_jk"
    )
  }
}

cost_conv <- bind_rows(conv_rows)
cd_write(cost_conv, out, "quantity_cost_conversion.csv")

# ------------------------------------------------------------------------------
# 8. Manuscript Booktabs LaTeX Table: table_policy_2025_quantity.tex
# ------------------------------------------------------------------------------
fmt_n <- function(x, d = 3L) if (is.finite(x)) sprintf(paste0("%.", d, "f"), x) else "---"
fmt_i <- function(x) if (is.finite(x)) formatC(as.integer(round(x)), format = "d", big.mark = ",") else "---"
stars <- function(p) if (!is.finite(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""

pick_q <- function(win, spec, out_nm, est_nm = NULL) {
  r <- qty_estimates %>% filter(window == !!win, specification == !!spec, outcome == !!out_nm)
  if (!is.null(est_nm)) r <- r %>% filter(estimator == !!est_nm)
  stopifnot(nrow(r) == 1L)
  r[1L, ]
}

fmt_qrow <- function(r, lbl, show_el = TRUE) {
  el_txt <- if (show_el) {
    paste0(fmt_n(r$elasticity, 2L), " [", fmt_n(r$elasticity_low_jk, 2L), ", ", fmt_n(r$elasticity_high_jk, 2L), "]")
  } else {
    paste0("Pass-through [", fmt_n(r$ci95_low_jk, 2L), ", ", fmt_n(r$ci95_high_jk, 2L), "]")
  }
  mde_txt <- if (show_el) fmt_n(r$mde_elasticity_jk, 2L) else "---"
  paste0(
    lbl, " & ",
    fmt_n(r$estimate, 4L), stars(r$p_jackknife), " & ",
    "[", fmt_n(r$se_jackknife, 4L), "] & ",
    "(", fmt_n(r$se_state_crv1, 4L), ") & ",
    el_txt, " & ",
    mde_txt, " & ",
    fmt_i(r$states), " \\\\"
  )
}

tex_lines <- c(
  "\\begin{table}[!htbp]",
  "\\centering",
  "\\caption{2025 Statewide AEWR Rule: County H-2A Quantity Response and Lower-Bound Elasticities (Track 1, FY2025--FY2026)}",
  "\\label{tab:policy_2025_quantity}",
  "\\small",
  "\\begin{tabular}{lcccccc}",
  "\\toprule",
  "Specification / Outcome & Slope / $\\hat{\\beta}$ & JK SE & CRV1 SE & Statutory Elasticity [95\\% JK CI] & MDE & States \\\\",
  "\\midrule",
  "\\multicolumn{7}{l}{\\textit{Panel A: Headline Complete FY2026 Q4 (Full Fiscal Year Oct 1--Sep 30, FY2025 vs.\\ FY2026)}} \\\\",
  fmt_qrow(pick_q("full_fy", "B_primary", "hours", "PPML"), "PPML certified hours (welfare anchor)"),
  fmt_qrow(pick_q("full_fy", "B_primary_log_rate", "hours", "PPML"), "PPML certified hours ($\\ln$ statutory rate)"),
  fmt_qrow(pick_q("full_fy", "B_primary", "cert", "PPML"), "PPML certified positions"),
  fmt_qrow(pick_q("full_fy", "B_primary", "y_hours", "OLS"), "OLS hours per 2011 farm worker"),
  fmt_qrow(pick_q("full_fy", "B_primary", "y_cert", "OLS"), "OLS positions per 2011 farm worker"),
  fmt_qrow(pick_q("full_fy", "B_primary_first_stage", "y_wage", "OLS"), "County offered-wage first stage (\\$/hr)", show_el = FALSE),
  "\\midrule",
  "\\multicolumn{7}{l}{\\textit{Panel B: Window Cutoffs, Benchmark Reproduction, and Pre-Policy Placebo (PPML Certified Hours)}} \\\\",
  fmt_qrow(pick_q("q1_q3", "B_primary", "hours", "PPML"), "Matched Q1--Q3 benchmark (Oct 1--Jun 30)"),
  fmt_qrow(pick_q("pre_aug03_decision", "B_primary", "hours", "PPML"), "Symmetric pre-Aug 3 decision cutoff"),
  fmt_qrow(pick_q("pre_aug03_receipt", "B_primary", "hours", "PPML"), "Symmetric pre-Aug 3 receipt cutoff"),
  fmt_qrow(pick_q("pre_aug26_decision", "B_primary", "hours", "PPML"), "Symmetric pre-Aug 26 decision cutoff (\\textit{UFW})"),
  fmt_qrow(pick_q("pre_aug26_receipt", "B_primary", "hours", "PPML"), "Symmetric pre-Aug 26 receipt cutoff (\\textit{UFW})"),
  fmt_qrow(pick_q("full_fy", "placebo_fy2024_fy2025", "hours", "PPML"), "Pre-policy placebo (FY2024 vs.\\ FY2025, full FY)"),
  fmt_qrow(pick_q("q1_q3", "placebo_fy2024_fy2025", "hours", "PPML"), "Pre-policy placebo (FY2024 vs.\\ FY2025, Q1--Q3)"),
  "\\midrule",
  "\\multicolumn{7}{l}{\\textit{Panel C: Treatment \\& Control Sensitivities on Complete FY2026 Q4 (PPML Certified Hours)}} \\\\",
  fmt_qrow(pick_q("full_fy", "S3_h2a_skill_1", "hours", "PPML"), "S3: H-2A skill-I cash rate"),
  fmt_qrow(pick_q("full_fy", "S3_us_skill_2", "hours", "PPML"), "S3: U.S.\\ skill-II rate"),
  fmt_qrow(pick_q("full_fy", "S4_x_cohort", "hours", "PPML"), "S4: Pre-Oct 2 receipt cohort-aligned rate"),
  fmt_qrow(pick_q("full_fy", "S4_x_cohort_dual", "hours", "PPML"), "S4: Dual-schedule (Oct 2 + Aug 3) cohort rate"),
  fmt_qrow(pick_q("full_fy", "S5_control_mw_general", "hours", "PPML"), "S5: + General state min.\\ wage (all 46 states)"),
  fmt_qrow(pick_q("full_fy", "S5_control_mw_ag", "hours", "PPML"), "S5: + Applicable ag.\\ min.\\ wage (all 46 states)"),
  fmt_qrow(pick_q("full_fy", "S1_base_static", "hours", "PPML"), "S1: Static FY2012--FY2026 panel"),
  "\\bottomrule",
  "\\end{tabular}",
  "\\begin{minipage}{0.96\\linewidth}",
  "\\vspace{0.4em}\\footnotesize",
  paste0(
    "\\textit{Notes:} Sample spans all 2,921 counties with positive fixed-2011 BEA farm employment in the 15 multi-state former ",
    "AEWR regions (46 states; CA, FL, AK, HI, and DC excluded). All models include County and Former-Region $\\times$ FY fixed effects. ",
    "Primary inference is exact delete-one-state jackknife (CV3J, $t(45)$); state-clustered CRV1 SEs appear in parentheses. ",
    "Statutory elasticities scale linear-dollar slopes by estimation-sample mean statutory rate ($\\hat{\\beta}\\bar{p}$ for PPML; ",
    "$\\hat{\\beta}\\bar{p}/\\bar{y}$ for OLS ratios); MDE $= 2.8 \\times \\text{JK elasticity SE}$. ",
    "Stars denote jackknife p-values ($^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$)."
  ),
  "\\end{minipage}",
  "\\end{table}"
)

writeLines(tex_lines, file.path(out, "table_policy_2025_quantity.tex"))

hp <- pick_q("full_fy", "B_primary", "hours", "PPML")
q13p <- pick_q("q1_q3", "B_primary", "hours", "PPML")
cat(
  sprintf(
    "Estimated Track 1 quantity module (%d rows): full_fy PPML hours el=%.3f [%.3f, %.3f] (MDE=%.2f, G=%d); Q1-Q3 PPML hours el=%.3f [%.3f, %.3f]\n",
    nrow(qty_estimates),
    hp$elasticity, hp$elasticity_low_jk, hp$elasticity_high_jk, hp$mde_elasticity_jk, hp$states,
    q13p$elasticity, q13p$elasticity_low_jk, q13p$elasticity_high_jk
  )
)
