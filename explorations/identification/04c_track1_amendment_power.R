# Track 1 design-stage power for two candidate registry amendments, prompted
# by the pre-period drift found in 04b (run before any FY2026 outcome is read):
#   A. state-specific linear trends added to county + former-region x FY FE;
#   B. a two-period contrast: FY2025 and FY2026 only (county + region x FY FE).
# Same DGPs, seed, calibration window, thresholds and gate as
# 04_track1_power.R; the simulation core below mirrors that script, with the
# trend projection added. Reads track1_pre_panel.csv and
# track1_treatment_schedule.csv from the scratch dir (written by 04).
source("explorations/identification/id_helpers.R")
source(path_code("designs", "border_discontinuity", "power.R"))
out_dir <- id_output_dir()
registry <- jsonlite::read_json("explorations/identification/track1_registry.json", simplifyVector = TRUE)
v6 <- jsonlite::read_json(path_code("designs", "border_discontinuity", "design.json"), simplifyVector = TRUE)
settings <- v6$redesign$power; assumptions <- v6$redesign$precision_assumptions
pp_all <- read_csv(file.path(out_dir, "track1_pre_panel.csv"), show_col_types = FALSE,
  col_types = cols(county_fips = "c", state_fips = "c", aewr_region_id = "c"))
id_assert_pre(pp_all$year)
treat <- read_csv(file.path(out_dir, "track1_treatment_schedule.csv"), show_col_types = FALSE, col_types = cols(state_fips = "c"))
counties <- pp_all %>% distinct(county_fips, state_fips, aewr_region_id, emp_farm_2011)

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
# prep with optional state linear trends (partialled after the FE within-transform).
prep <- function(frame, x = NULL, trends = FALSE) {
  frame <- frame %>% arrange(county_fips, year)
  fe <- list(factor(frame$county_fips), factor(paste(frame$aewr_region_id, frame$year)))
  out <- list(data = frame, fe = fe, years = sort(unique(frame$year)), states = sort(unique(frame$state_fips)),
    regions = sort(unique(frame$aewr_region_id)), cnty = sort(unique(frame$county_fips)), Tq = NULL)
  if (trends) {
    S <- outer(frame$state_fips, out$states, "==") * (frame$year - 2025)
    Tt <- fixest::demean(S, f = fe, tol = 1e-10, nthreads = 1L)
    Tt <- Tt[, colSums(Tt^2) > 1e-10, drop = FALSE]
    out$Tq <- qr.Q(qr(Tt)); out$Tq <- out$Tq[, seq_len(qr(Tt)$rank), drop = FALSE]
  }
  if (!is.null(x)) {
    xw <- drop(fixest::demean(cbind(x = frame[[x]]), f = fe, tol = 1e-10, nthreads = 1L))
    if (trends) xw <- xw - drop(out$Tq %*% crossprod(out$Tq, xw))
    out$xw <- xw; out$sxx <- sum(xw^2)
    out$state_index <- match(frame$state_fips, out$states); out$region_index <- match(frame$aewr_region_id, out$regions)
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
  if (process$region > 0) {
    si <- match(d$state_fips, pp$states); e <- border_power_ar_draws(length(pp$states), ny, R, settings$ar1)
    if (isTRUE(process$spatial)) for (t in seq_len(ny)) { rr <- (t - 1L) * length(pp$states) + seq_along(pp$states)
      e[rr, ] <- loading[pp$states, pp$states] %*% e[rr, , drop = FALSE] }
    raw <- raw + sqrt(process$region) * e[ti * length(pp$states) + si, , drop = FALSE]
  }
  if (process$pair > 0) {
    ri <- match(d$aewr_region_id, pp$regions); e <- border_power_ar_draws(length(pp$regions), ny, R, settings$ar1)
    raw <- raw + sqrt(process$pair) * e[ti * length(pp$regions) + ri, , drop = FALSE]
  }
  raw
}
within_m <- function(E, pp) {
  W <- fixest::demean(E, f = pp$fe, tol = 1e-10, nthreads = 1L)
  if (!is.null(pp$Tq)) W <- W - pp$Tq %*% crossprod(pp$Tq, W)
  W
}
draw_stats <- function(pp, E) {
  yw <- within_m(E, pp); b <- drop(crossprod(pp$xw, yw)) / pp$sxx
  u <- yw - outer(pp$xw, b); sc <- pp$xw * u
  n <- nrow(yw); k <- 1 + if (is.null(pp$Tq)) 0 else ncol(pp$Tq); corr <- (n - 1) / (n - k - 1)
  Gs <- length(pp$states); Gr <- length(pp$regions)
  vs <- corr * Gs / (Gs - 1) * colSums(rowsum(sc, pp$state_index)^2) / pp$sxx^2
  vr <- corr * Gr / (Gr - 1) * colSums(rowsum(sc, pp$region_index)^2) / pp$sxx^2
  list(b = b, var = rbind(state_CRV1 = vs, former_region_CRV1 = vr), df = c(state_CRV1 = Gs - 1, former_region_CRV1 = Gr - 1))
}
calibrate <- function(pre, y, process, loading) {
  obs <- sum(within_m(cbind(pre$data[[y]]), pre)^2)
  rss <- unlist(lapply(split(seq_len(settings$calibration_repetitions),
    ceiling(seq_len(settings$calibration_repetitions) / settings$chunk_size)), function(ix)
      colSums(within_m(unit_errors(pre, length(ix), process, loading), pre)^2)))
  list(sigma = sqrt(obs / mean(rss)), observed_rss = obs, expected_rss = mean(rss))
}

variants <- tribble(~variant, ~years, ~trends, ~cal_years,
  "A_state_trends_fy2012_fy2026", 2012:2026, TRUE, 2012:2025,
  "A_state_trends_fy2012_fy2027", 2012:2027, TRUE, 2012:2025,
  "B_fy2025_fy2026_only", 2025:2026, FALSE, 2020:2025)
loading <- state_graph(sort(unique(counties$state_fips)), settings$spatial_graph_coefficient)
Y <- c(cert = "y_cert_q", hours = "y_hours_q")
mde_rows <- power_rows <- sup <- list()
for (k in seq_len(nrow(variants))) for (nm in names(Y)) {
  v <- variants[k, ]; message(v$variant, " / ", nm)
  # Trend calibration needs a long pre-period, so variant A calibrates on FY2012-FY2025 with trends.
  pre <- prep(pp_all %>% filter(year %in% v$cal_years[[1]]), trends = v$trends)
  pf <- tidyr::crossing(counties, year = v$years[[1]]) %>%
    left_join(treat %>% filter(label == "us_skill_1") %>% select(state_fips, year, x = aewr), by = c("state_fips", "year"))
  pol <- prep(pf, "x", trends = v$trends)
  mean_y <- mean(pp_all[[Y[[nm]]]][pp_all$year %in% registry$calibration$primary_years[1]:registry$calibration$primary_years[2]])
  mean_price <- mean(pf$x); beta_e1 <- mean_y / mean_price
  grid <- round(seq(0.01, 10, by = 0.01), 2) * beta_e1
  thr <- if (nm == "cert") 0.01 else NA_real_
  effects <- sort(unique(c(0, grid, -grid, if (is.finite(thr)) c(thr, -thr))))
  sup[[length(sup) + 1]] <- tibble(variant = v$variant, outcome = nm, counties = length(pol$cnty), states = length(pol$states),
    trend_columns = if (is.null(pol$Tq)) 0L else ncol(pol$Tq), partial_treatment_ss = pol$sxx, mean_y = mean_y,
    mean_price = mean_price, beta_elasticity_1 = beta_e1)
  for (dgp in unlist(settings$dgp_ids)) {
    set.seed(as.integer(settings$seed))
    fr <- settings$variance_fractions[[dgp]]
    process <- list(region = fr$region, pair = fr$pair, county = fr$county, spatial = dgp == "spatial_region_serial")
    sc <- calibrate(pre, Y[[nm]], process, loading)
    draws <- lapply(split(seq_len(settings$repetitions), ceiling(seq_len(settings$repetitions) / settings$chunk_size)),
      function(ix) draw_stats(pol, unit_errors(pol, length(ix), process, loading)))
    errs <- unlist(lapply(draws, `[[`, "b")); vars <- do.call(cbind, lapply(draws, `[[`, "var")); dfs <- draws[[1]]$df
    for (meth in rownames(vars)) {
      curve <- border_power_grid(errs, vars[meth, ], dfs[[meth]], effects, sc$sigma, assumptions$alpha)
      null <- curve[curve$effect == 0, ]
      gate <- null$invalid_repetitions == 0L && null$mc_confidence_high <= assumptions$size_maximum
      if (is.finite(thr)) power_rows[[length(power_rows) + 1]] <- tibble(variant = v$variant, outcome = nm, dgp_id = dgp,
        inference_method = meth, size_gate_pass = gate, curve[abs(abs(curve$effect) - thr) < 1e-12, ])
      for (sg in c(-1, 1)) {
        dir <- curve[match(sg * grid, curve$effect), ]; pos <- which(dir$rejection_rate >= assumptions$target_power)
        mde <- if (length(pos)) grid[pos[1]] else NA_real_
        mde_rows[[length(mde_rows) + 1]] <- tibble(variant = v$variant, outcome = nm, dgp_id = dgp, inference_method = meth,
          effect_sign = sg, sigma = sc$sigma, grid_mde = mde, size_valid_mde = if (gate) mde else NA_real_, size_gate_pass = gate,
          null_rejection_rate = null$rejection_rate, null_mc_high = null$mc_confidence_high, mde_elasticity = mde / beta_e1)
      }
    }
  }
}
mde <- bind_rows(mde_rows)
decision <- mde %>% filter(dgp_id != "county_serial") %>% group_by(variant, outcome, inference_method) %>%
  summarise(all_shared_size_gates = all(size_gate_pass), max_grid_mde = max(grid_mde),
    max_mde_elasticity = max(mde_elasticity), max_null_rejection = max(null_rejection_rate), .groups = "drop") %>%
  mutate(pass_elasticity_1 = all_shared_size_gates & is.finite(max_mde_elasticity) & max_mde_elasticity <= 1,
    pass_registry_0.01 = ifelse(outcome == "cert", all_shared_size_gates & is.finite(max_grid_mde) & max_grid_mde <= 0.01, NA))
id_write(bind_rows(sup), out_dir, "track1_amend_support.csv"); id_write(mde, out_dir, "track1_amend_mde.csv")
id_write(bind_rows(power_rows), out_dir, "track1_amend_power.csv"); id_write(decision, out_dir, "track1_amend_decision.csv")
print(as.data.frame(bind_rows(sup)), digits = 3); print(as.data.frame(decision), digits = 3)
