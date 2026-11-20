# Track 1 design-stage power: 2025 statewide AEWR rule, all counties in
# multi-state former AEWR regions, county + former-region x FY FE, state
# clusters. Declared in track1_registry.json. Reads ONLY pre-FY2026 outcomes
# (FY2012-FY2025, via id_read_* which filter inside the scan) and the actual
# statewide rate schedules. Errors are simulated on the actual geography and
# FY2026/FY2027 rates and scaled by pre-period FE-only residuals.
source("explorations/identification/id_helpers.R")
source(path_code("designs", "border_discontinuity", "power.R"))  # ar draws, power grid, Wilson
out_dir <- id_output_dir()
reg_path <- "explorations/identification/track1_registry.json"
registry <- jsonlite::read_json(reg_path, simplifyVector = TRUE)
registry_sha256 <- digest::digest(file = reg_path, algo = "sha256")
v6 <- jsonlite::read_json(path_code("designs", "border_discontinuity", "design.json"), simplifyVector = TRUE)
settings <- v6$redesign$power; assumptions <- v6$redesign$precision_assumptions
stopifnot(settings$nthreads == 1L)

# Sample ---------------------------------------------------------------------
cp <- read_parquet(path_processed("county_year_panel.parquet"),
  col_select = c("county_fips", "state_fips", "aewr_region_id", "emp_farm_2011", "year")) %>%
  filter(year == 2011) %>% select(-year)
counties <- cp %>% filter(!state_fips %in% registry$geography$excluded_states,
  is.finite(emp_farm_2011), emp_farm_2011 > 0) %>%
  group_by(aewr_region_id) %>% filter(n_distinct(state_fips) > 1) %>% ungroup() %>%
  arrange(aewr_region_id, state_fips, county_fips)
stopifnot(!anyDuplicated(counties$county_fips), is.character(counties$county_fips))

# Pre-period outcomes (b01 fiscal-year logic, Q1-Q3 decisions) ---------------
plausible <- function(value, fy) { y <- as.integer(format(value, "%Y")); value[!is.na(y) & abs(y - fy) > 1L] <- as.Date(NA); value }
dec <- id_read_disclosures(c("case_number", "decision_date")) %>% filter(fiscal_year >= 2012) %>%
  mutate(decision_date = ifelse(is.na(decision_date) | trimws(decision_date) == "", NA_character_, decision_date)) %>%
  group_by(case_number, fiscal_year) %>% summarise(decision_date = first(na.omit(decision_date)), .groups = "drop") %>%
  mutate(decision_date = plausible(as.Date(decision_date), fiscal_year))
ws <- id_read_worksites(c("case_number", "county_fips_list", "adjusted_nbr_workers_certified", "period_begin_date",
  "period_end_date", "number_of_hours", "wage_rate", "wage_unit")) %>%
  filter(fiscal_year >= 2012) %>%
  mutate(begin = as.Date(period_begin_date), end = as.Date(period_end_date)) %>% filter(!is.na(begin))
ws <- ws %>% mutate(nh = suppressWarnings(as.numeric(number_of_hours)),
  nh = ifelse(is.na(nh), mean(nh, na.rm = TRUE), nh),
  workers = as.numeric(adjusted_nbr_workers_certified),
  hours = as.numeric(end - begin) / 7 * nh * workers,
  wage = suppressWarnings(as.numeric(wage_rate)),
  hourly = toupper(trimws(wage_unit)) %in% c("HOUR", "HOURLY", "HR") & is.finite(wage) & wage > 0 & wage <= 100) %>%
  left_join(dec, by = c("case_number", "fiscal_year"), relationship = "many-to-one") %>%
  mutate(q13 = !is.na(decision_date) & decision_date >= as.Date(sprintf("%d-10-01", fiscal_year - 1L)) &
    decision_date <= as.Date(sprintf("%d-06-30", fiscal_year)),
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    n_counties = stringr::str_count(county_fips_list, ",") + 1L) %>%
  tidyr::separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = harmonize_county_fips_2010(trimws(county_fips_list)), share = 1 / n_counties)
id_assert_pre(ws$fiscal_year)
outc <- ws %>% group_by(county_fips, year = as.integer(fiscal_year)) %>% summarise(
  cert_full = sum(workers * share, na.rm = TRUE), cert_q = sum(workers * share * q13, na.rm = TRUE),
  hours_full = sum(hours * share, na.rm = TRUE), hours_q = sum(hours * share * q13, na.rm = TRUE),
  wbill_q = sum(wage * hours * share * (q13 & hourly), na.rm = TRUE),
  hhours_q = sum(hours * share * (q13 & hourly & is.finite(hours)), na.rm = TRUE), .groups = "drop")

# Validation against the owning producer's full-FY counts.
h2y <- read_parquet(path_int("h2a_data_year.parquet"), col_select = c("county_fips", "year", "nbr_workers_certified_fiscal_year",
  "man_hours_certified_fiscal_year")) %>% mutate(year = as.integer(year)) %>% filter(year >= 2012, year <= 2025)
val <- full_join(outc %>% select(county_fips, year, cert_full, hours_full), h2y, by = c("county_fips", "year")) %>%
  mutate(across(where(is.numeric), ~ coalesce(.x, 0)))
validation <- val %>% group_by(year) %>% summarise(rebuilt_cert = sum(cert_full), producer_cert = sum(nbr_workers_certified_fiscal_year),
  rebuilt_hours = sum(hours_full), producer_hours = sum(man_hours_certified_fiscal_year),
  county_corr_cert = cor(cert_full, nbr_workers_certified_fiscal_year),
  max_abs_county_diff_cert = max(abs(cert_full - nbr_workers_certified_fiscal_year)), .groups = "drop")
q_share <- outc %>% group_by(year) %>% summarise(q13_share_cert = sum(cert_q) / sum(cert_full),
  q13_share_hours = sum(hours_q) / sum(hours_full), .groups = "drop")
id_write(validation, out_dir, "track1_outcome_validation.csv")
id_write(q_share, out_dir, "track1_q13_share.csv")

panel_pre <- tidyr::crossing(counties, year = 2012:2025) %>%
  left_join(outc, by = c("county_fips", "year")) %>%
  mutate(across(c(cert_full, cert_q, hours_full, hours_q, wbill_q, hhours_q), ~ coalesce(.x, 0)),
    y_cert_q = cert_q / emp_farm_2011, y_hours_q = hours_q / emp_farm_2011,
    y_cert_full = cert_full / emp_farm_2011, y_hours_full = hours_full / emp_farm_2011,
    y_wage_q = ifelse(hhours_q > 0, wbill_q / hhours_q, NA_real_))
id_write(panel_pre %>% select(county_fips, state_fips, aewr_region_id, year, emp_farm_2011, starts_with("y_")),
  out_dir, "track1_pre_panel.csv")

# Treatment: rate in effect on April 1 of each FY --------------------------
regional <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>%
  filter(occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference"), worker_category == "covered_workers") %>%
  transmute(state_fips, aewr = as.numeric(aewr), start = as.Date(effective_start), end = as.Date(effective_end_exclusive))
statewide <- read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
  filter(trimws(source_sheet) == registry$policy_event$source_sheet) %>%
  transmute(state_fips, worker_category, skill_level = as.character(skill_level), aewr = as.numeric(hourly_aewr),
    start = as.Date(effective_start), end = as.Date(effective_end_exclusive))
rate_on <- function(s, date) { r <- s %>% filter(start <= date, is.na(end) | end > date) %>% select(state_fips, aewr)
  stopifnot(!anyDuplicated(r$state_fips)); r }
variants <- bind_rows(as_tibble(registry$treatment$primary), as_tibble(registry$treatment$sensitivities) %>% select(-any_of("note")))
treat <- bind_rows(lapply(seq_len(nrow(variants)), function(i) {
  v <- variants[i, ]
  bind_rows(lapply(2012:2027, function(fy) {
    date <- as.Date(sprintf("%d-04-01", fy))
    r <- if (fy >= 2026) rate_on(statewide %>% filter(worker_category == v$worker_category, skill_level == v$skill_level), date) else rate_on(regional, date)
    tibble(state_fips = unique(counties$state_fips), year = fy) %>% left_join(r, by = "state_fips") %>% mutate(label = v$label)
  }))
}))
stopifnot(all(is.finite(treat$aewr)))
id_write(treat, out_dir, "track1_treatment_schedule.csv")

# State contiguity graph (from county adjacency) ---------------------------
adj <- read_parquet(path_int("county_adjacency2010.parquet")) %>%
  transmute(a = substr(harmonize_county_fips_2010(county_fips), 1, 2), b = substr(harmonize_county_fips_2010(neighbor_county_fips), 1, 2)) %>%
  filter(a != b, a %in% counties$state_fips, b %in% counties$state_fips) %>% distinct()
state_graph <- function(states, coefficient) {
  A <- matrix(0, length(states), length(states), dimnames = list(states, states))
  A[cbind(adj$a, adj$b)] <- 1; A <- pmax(A, t(A))
  deg <- rowSums(A); sc <- ifelse(deg > 0, 1 / sqrt(pmax(deg, 1)), 0)
  L <- solve(diag(length(states)) - coefficient * A * outer(sc, sc))
  L / sqrt(rowSums(L^2))
}

# Simulation core ------------------------------------------------------------
prep <- function(frame, x = NULL) {
  frame <- frame %>% arrange(county_fips, year)
  fe <- list(factor(frame$county_fips), factor(paste(frame$aewr_region_id, frame$year)))
  out <- list(data = frame, fe = fe, years = sort(unique(frame$year)), states = sort(unique(frame$state_fips)),
    regions = sort(unique(frame$aewr_region_id)), cnty = sort(unique(frame$county_fips)))
  if (!is.null(x)) {
    xw <- drop(fixest::demean(cbind(x = frame[[x]]), f = fe, tol = 1e-10, nthreads = 1L))
    out$xw <- xw; out$sxx <- sum(xw^2)
    out$state_index <- match(frame$state_fips, out$states); out$region_index <- match(frame$aewr_region_id, out$regions)
    # Exact leave-one-state-out slope operators: on retained rows the re-demeaned
    # treatment is orthogonal to the FE, so beta_(-s) = L_s' e for raw errors e.
    out$loo <- t(vapply(out$states, function(s) {
      keep <- frame$state_fips != s
      fk <- lapply(fe, function(f) droplevels(f[keep]))
      xk <- drop(fixest::demean(cbind(x = frame[[x]][keep]), f = fk, tol = 1e-10, nthreads = 1L))
      op <- numeric(nrow(frame)); op[keep] <- xk / sum(xk^2); op
    }, numeric(nrow(frame))))
  }
  out
}
unit_errors <- function(pp, R, process, loading) {
  d <- pp$data; ny <- length(pp$years); ti <- match(d$year, pp$years) - 1L
  raw <- matrix(0, nrow(d), R)
  if (process$county > 0) {
    ci <- match(d$county_fips, pp$cnty); e <- border_power_ar_draws(length(pp$cnty), ny, R, settings$ar1)
    raw <- raw + sqrt(process$county) * e[ti * length(pp$cnty) + ci, , drop = FALSE]
  }
  if (process$region > 0) {  # border "region" = state node
    si <- match(d$state_fips, pp$states); e <- border_power_ar_draws(length(pp$states), ny, R, settings$ar1)
    if (isTRUE(process$spatial)) for (t in seq_len(ny)) { rr <- (t - 1L) * length(pp$states) + seq_along(pp$states)
      e[rr, ] <- loading[pp$states, pp$states] %*% e[rr, , drop = FALSE] }
    raw <- raw + sqrt(process$region) * e[ti * length(pp$states) + si, , drop = FALSE]
  }
  if (process$pair > 0) {  # border "pair" = former-region shock (absorbed by region x FY FE)
    ri <- match(d$aewr_region_id, pp$regions); e <- border_power_ar_draws(length(pp$regions), ny, R, settings$ar1)
    raw <- raw + sqrt(process$pair) * e[ti * length(pp$regions) + ri, , drop = FALSE]
  }
  raw
}
demean_m <- function(E, pp) fixest::demean(E, f = pp$fe, tol = 1e-10, nthreads = 1L)
draw_stats <- function(pp, E) {
  yw <- demean_m(E, pp); b <- drop(crossprod(pp$xw, yw)) / pp$sxx
  u <- yw - outer(pp$xw, b); sc <- pp$xw * u
  n <- nrow(yw); corr <- (n - 1) / (n - 2)
  Gs <- length(pp$states); Gr <- length(pp$regions)
  vs <- corr * Gs / (Gs - 1) * colSums(rowsum(sc, pp$state_index)^2) / pp$sxx^2
  vr <- corr * Gr / (Gr - 1) * colSums(rowsum(sc, pp$region_index)^2) / pp$sxx^2
  bl <- pp$loo %*% E; bl <- sweep(bl, 2, colMeans(bl)); vj <- (Gs - 1) / Gs * colSums(bl^2)
  list(b = b, var = rbind(state_CRV1 = vs, state_CV3J = vj, former_region_CRV1 = vr),
    df = c(state_CRV1 = Gs - 1, state_CV3J = Gs - 1, former_region_CRV1 = Gr - 1))
}
calibrate <- function(pre, y, process, loading) {
  obs <- sum(demean_m(cbind(pre$data[[y]]), pre)^2)
  rss <- unlist(lapply(split(seq_len(settings$calibration_repetitions),
    ceiling(seq_len(settings$calibration_repetitions) / settings$chunk_size)), function(ix)
      colSums(demean_m(unit_errors(pre, length(ix), process, loading), pre)^2)))
  list(sigma = sqrt(obs / mean(rss)), observed_rss = obs, expected_rss = mean(rss))
}

# Combinations ---------------------------------------------------------------
cal <- list(primary = registry$calibration$primary_years, sensitivity = registry$calibration$sensitivity_years)
scen <- as_tibble(registry$power$scenarios)
outcomes <- c(cert = "y_cert_q", hours = "y_hours_q", wage = "y_wage_q")
combos <- bind_rows(
  tidyr::crossing(outcome = "cert", label = variants$label, scenario_id = scen$scenario_id, calibration = names(cal)),
  tidyr::crossing(outcome = "hours", label = "us_skill_1", scenario_id = scen$scenario_id, calibration = names(cal)),
  tidyr::crossing(outcome = "wage", label = "us_skill_1", scenario_id = scen$scenario_id, calibration = "primary"))
loading <- state_graph(sort(unique(counties$state_fips)), settings$spatial_graph_coefficient)
power_rows <- mde_rows <- cal_rows <- sup_rows <- list()
for (k in seq_len(nrow(combos))) {
  cb <- combos[k, ]; message("Track 1 power: ", paste(cb, collapse = " / "))
  yv <- outcomes[[cb$outcome]]; cy <- seq(cal[[cb$calibration]][1], cal[[cb$calibration]][2])
  pyr <- seq(scen$fiscal_years[[which(scen$scenario_id == cb$scenario_id)]][1], scen$fiscal_years[[which(scen$scenario_id == cb$scenario_id)]][2])
  keep <- counties$county_fips
  if (cb$outcome == "wage") {  # counties with hourly offers in every calibration year; regions keep >1 state
    ok <- panel_pre %>% filter(year %in% cy) %>% group_by(county_fips) %>% summarise(ok = all(is.finite(y_wage_q)), .groups = "drop")
    keep <- ok$county_fips[ok$ok]
  }
  base <- counties %>% filter(county_fips %in% keep) %>% group_by(aewr_region_id) %>% filter(n_distinct(state_fips) > 1) %>% ungroup()
  pre <- prep(panel_pre %>% semi_join(base, by = "county_fips") %>% filter(year %in% cy))
  pol_frame <- tidyr::crossing(base, year = pyr) %>%
    left_join(treat %>% filter(label == cb$label) %>% select(state_fips, year, x = aewr), by = c("state_fips", "year"))
  pol <- prep(pol_frame, "x")
  mean_y <- mean(pre$data[[yv]]); mean_price <- mean(pol_frame$x)
  beta_e1 <- mean_y / mean_price  # beta giving an elasticity of 1 at these means
  thr <- if (cb$outcome == "cert") registry$power$thresholds$cert_positions_q1q3$registry_beta else if (cb$outcome == "wage") registry$power$thresholds$offered_wage_q1q3$registry_beta else NA_real_
  egrid <- round(seq(0.01, 10, by = 0.01), 2)
  grid <- if (cb$outcome == "wage") round(seq(0.005, 2, by = 0.005), 3) else egrid * beta_e1
  effects <- sort(unique(c(0, grid, -grid, if (is.finite(thr)) c(thr, -thr), beta_e1, -beta_e1)))
  post_dev <- pol_frame %>% filter(year >= 2026) %>% group_by(year, aewr_region_id) %>% mutate(dev = x - mean(x)) %>% ungroup()
  sup_rows[[k]] <- bind_cols(cb, tibble(counties = length(pol$cnty), states = length(pol$states), regions = length(pol$regions),
    policy_years = paste(range(pyr), collapse = "-"), calibration_years = paste(range(cy), collapse = "-"),
    mean_y_pre = mean_y, sd_y_pre = sd(pre$data[[yv]]), mean_price = mean_price, beta_elasticity_1 = beta_e1,
    post_within_region_sd_county = sd(post_dev$dev), partial_treatment_ss = pol$sxx,
    max_state_leverage_share = max(rowsum(pol$xw^2, pol$state_index)) / pol$sxx,
    effective_state_count = pol$sxx^2 / sum(rowsum(pol$xw^2, pol$state_index)^2)))
  for (dgp in unlist(settings$dgp_ids)) {
    set.seed(as.integer(settings$seed))
    fr <- settings$variance_fractions[[dgp]]
    process <- list(region = fr$region, pair = fr$pair, county = fr$county, spatial = dgp == "spatial_region_serial")
    sc <- calibrate(pre, yv, process, loading)
    draws <- lapply(split(seq_len(settings$repetitions), ceiling(seq_len(settings$repetitions) / settings$chunk_size)),
      function(ix) draw_stats(pol, unit_errors(pol, length(ix), process, loading)))
    errs <- unlist(lapply(draws, `[[`, "b")); vars <- do.call(cbind, lapply(draws, `[[`, "var")); dfs <- draws[[1]]$df
    cal_rows[[length(cal_rows) + 1]] <- bind_cols(cb, tibble(dgp_id = dgp, sigma = sc$sigma, observed_rss = sc$observed_rss,
      expected_unit_rss = sc$expected_rss, sim_coef_error_sd = sd(errs) * sc$sigma))
    for (meth in rownames(vars)) {
      curve <- border_power_grid(errs, vars[meth, ], dfs[[meth]], effects, sc$sigma, assumptions$alpha)
      null <- curve[curve$effect == 0, ]
      gate <- null$invalid_repetitions == 0L && null$mc_confidence_high <= assumptions$size_maximum
      keep_eff <- c(0, if (is.finite(thr)) c(thr, -thr), beta_e1, -beta_e1)
      power_rows[[length(power_rows) + 1]] <- bind_cols(cb, tibble(dgp_id = dgp, inference_method = meth, size_gate_pass = gate),
        curve[curve$effect %in% keep_eff, ] %>% mutate(effect_label = case_when(effect == 0 ~ "null",
          abs(abs(effect) - beta_e1) < 1e-12 ~ "elasticity_1", TRUE ~ "registry_threshold")))
      for (sg in c(-1, 1)) {
        dir <- curve[match(sg * grid, curve$effect), ]
        pos <- which(dir$rejection_rate >= assumptions$target_power)
        mde <- if (length(pos)) grid[pos[1]] else NA_real_
        mde_rows[[length(mde_rows) + 1]] <- bind_cols(cb, tibble(dgp_id = dgp, inference_method = meth, effect_sign = sg,
          grid_mde = mde, size_valid_mde = if (gate) mde else NA_real_, size_gate_pass = gate,
          null_rejection_rate = null$rejection_rate, null_mc_high = null$mc_confidence_high,
          mde_elasticity = if (cb$outcome == "wage") NA_real_ else mde / beta_e1, mean_y_pre = mean_y, mean_price = mean_price))
      }
    }
  }
}
support <- bind_rows(sup_rows); calib <- bind_rows(cal_rows); power <- bind_rows(power_rows); mde <- bind_rows(mde_rows)
shared <- setdiff(unlist(settings$dgp_ids), "county_serial")
keys <- c("outcome", "label", "scenario_id", "calibration", "inference_method")
decision <- power %>% filter(dgp_id %in% shared, effect_label != "null") %>%
  group_by(across(all_of(c(keys, "effect_label")))) %>% summarise(min_power_both_signs = min(rejection_rate), .groups = "drop") %>%
  left_join(mde %>% filter(dgp_id %in% shared) %>% group_by(across(all_of(keys))) %>%
    summarise(all_shared_size_gates = all(size_gate_pass), max_grid_mde = max(grid_mde),
      max_size_valid_mde = if (all(is.finite(size_valid_mde))) max(size_valid_mde) else NA_real_,
      max_mde_elasticity = max(mde_elasticity), max_null_rejection = max(null_rejection_rate), .groups = "drop"), by = keys) %>%
  left_join(support %>% select(outcome, label, scenario_id, calibration, beta_elasticity_1), by = c("outcome", "label", "scenario_id", "calibration")) %>%
  mutate(threshold = case_when(effect_label == "elasticity_1" ~ beta_elasticity_1,
    outcome == "cert" ~ registry$power$thresholds$cert_positions_q1q3$registry_beta,
    outcome == "wage" ~ registry$power$thresholds$offered_wage_q1q3$registry_beta),
    pass = all_shared_size_gates & min_power_both_signs >= assumptions$target_power &
      is.finite(max_size_valid_mde) & max_size_valid_mde <= threshold + 1e-12)
id_write(support, out_dir, "track1_power_support.csv"); id_write(calib, out_dir, "track1_power_calibration.csv")
id_write(power, out_dir, "track1_power.csv"); id_write(mde, out_dir, "track1_power_mde.csv")
id_write(decision, out_dir, "track1_power_decision.csv")
writeLines(c(paste("registry", reg_path), paste("sha256", registry_sha256)), file.path(out_dir, "track1_registry_sha256.txt"))
print(as.data.frame(support), digits = 3); print(as.data.frame(decision), digits = 3)
print(as.data.frame(validation), digits = 4); print(as.data.frame(q_share), digits = 3)
