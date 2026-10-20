# County entropy recovery. R owns moment construction, normalization, the
# penalized dual objective, Newton iterations and all convergence guards.
# Calibration is deterministic; no simulated survey weights are produced.
IV_SOURCE_YEARS <- 2010:2021
IV_QUARTERS <- c("january", "april", "july", "october")
IV_PRIOR_SPEC <- "census_hired_workers_qcew_annual_updated_v2"
IV_UPDATE_SPEC <- "qcew_annual_qwi_bea_two_sided_state_raked_v2"
IV_WEIGHT_SPEC <- "fls_pseudo_county_entropy_v2"
IV_WAGE_SPEC <- "fls_county_wage_only_soft_rho010_v2"
IV_PRIMARY_SPEC <- "fls_county_wage_seasonal_composition_soft_rho010_v2"
IV_RHO <- 0.10
IV_SCALE_TOLERANCE <- 1e-12
IV_WEIGHT_TOLERANCE <- 1e-10
IV_GRADIENT_TOLERANCE <- 1e-10
IV_MAX_ITERATIONS <- 100L

iv_require_unique <- function(data, keys, label) {
  if (!all(keys %in% names(data)) || !nrow(data) || anyNA(data[keys]) ||
      anyDuplicated(data[keys])) stop(label, ": empty, missing or duplicate keys")
  invisible(data)
}

iv_sort <- function(data, keys) {
  result <- data[do.call(order, c(unname(data[keys]), list(na.last = FALSE, method = "radix"))), , drop = FALSE]
  if (!inherits(result, "tbl_df")) rownames(result) <- NULL
  result
}

iv_key <- function(...) paste(..., sep = "|")
iv_dot <- function(x, y) as.numeric(crossprod(x, y))
iv_helmert <- function(n) {
  if (n < 2L) stop("At least two survey weeks are required")
  basis <- matrix(0, n, n - 1L)
  for (j in seq_len(n - 1L)) {
    basis[seq_len(j), j] <- 1 / sqrt(j * (j + 1))
    basis[j + 1L, j] <- -j / sqrt(j * (j + 1))
  }
  basis
}

iv_standardize <- function(values, target, prior) {
  if (length(values) != length(prior) || any(!is.finite(c(values, target, prior))) ||
      any(prior < 0) || sum(prior) <= 0) stop("Invalid moment or prior")
  prior <- prior / sum(prior)
  center <- iv_dot(prior, values)
  scale <- sqrt(iv_dot(prior, (values - center)^2))
  active <- scale > IV_SCALE_TOLERANCE * max(1, abs(values), abs(center), abs(target))
  list(prior_center = center, prior_scale = scale, active = active,
    status = if (active) "active" else "inactive_zero_prior_variation",
    standardized_values = if (active) (values - center) / scale else NULL,
    standardized_target = if (active) (target - center) / scale else NA_real_)
}

iv_dual_state <- function(log_prior, design, target, rho, multipliers) {
  eta <- log_prior + as.numeric(design %*% multipliers)
  maximum <- max(eta)
  exponential <- exp(eta - maximum)
  normalizer <- sum(exponential)
  weights <- exponential / normalizer
  moments <- as.numeric(crossprod(weights, design))
  list(weights = weights, moments = moments,
    objective = maximum + log(normalizer) - sum(multipliers * target) + sum(multipliers^2) / (2 * rho),
    gradient = moments - target + multipliers / rho)
}

iv_solve_entropy <- function(priors, design, target, rho = IV_RHO) {
  if (is.null(dim(priors))) priors <- matrix(priors, nrow = 1L)
  if (ncol(priors) != nrow(design) || length(target) != ncol(design) ||
      any(!is.finite(priors) | priors < 0) || any(rowSums(priors) <= 0)) stop("Invalid entropy dimensions/prior")
  priors <- priors / rowSums(priors)
  if (any(!is.finite(c(design, target)))) stop("Nonfinite active moment")
  n <- nrow(priors)
  p <- ncol(design)
  result <- list(weights = priors, success = rep(TRUE, n),
    status = rep("calibrated_prior_no_active_moments", n), iterations = integer(n),
    input_imbalance_norm = numeric(n), calibrated_imbalance_norm = numeric(n),
    maximum_absolute_imbalance = numeric(n), kl_divergence = numeric(n),
    effective_county_count = 1 / rowSums(priors^2), maximum_county_weight = apply(priors, 1L, max))
  if (!p) return(result)
  for (d in seq_len(n)) {
    prior <- priors[d, ]
    log_prior <- log(pmax(prior, .Machine$double.xmin * .Machine$double.eps))
    multipliers <- numeric(p)
    failed <- FALSE
    for (iteration in seq_len(IV_MAX_ITERATIONS)) {
      state <- iv_dual_state(log_prior, design, target, rho, multipliers)
      if (max(abs(state$gradient)) <= IV_GRADIENT_TOLERANCE) break
      hessian <- crossprod(design, design * state$weights) - tcrossprod(state$moments) + diag(p) / rho
      direction <- tryCatch(solve(hessian, state$gradient), error = function(e) NULL)
      if (is.null(direction)) { failed <- TRUE; break }
      derivative <- iv_dot(state$gradient, direction)
      step <- 1
      accepted <- FALSE
      while (step >= 2^-20) {
        proposal <- multipliers - step * direction
        next_state <- iv_dual_state(log_prior, design, target, rho, proposal)
        tolerance <- 1e-12 * (1 + abs(state$objective))
        if (is.finite(next_state$objective) &&
            (next_state$objective <= state$objective - 1e-4 * step * derivative ||
             next_state$objective <= state$objective + tolerance)) {
          multipliers <- proposal
          result$iterations[d] <- iteration
          accepted <- TRUE
          break
        }
        step <- step * 0.5
      }
      if (!accepted) { failed <- TRUE; break }
    }
    state <- iv_dual_state(log_prior, design, target, rho, multipliers)
    weights <- state$weights / sum(state$weights)
    converged <- max(abs(state$gradient)) <= IV_GRADIENT_TOLERANCE
    input <- as.numeric(crossprod(prior, design)) - target
    imbalance <- state$moments - target
    valid <- abs(sum(weights) - 1) <= IV_WEIGHT_TOLERANCE && all(is.finite(weights) & weights >= 0)
    input_norm <- sqrt(sum(input^2))
    calibrated_norm <- sqrt(sum(imbalance^2))
    success <- converged && !failed && valid && calibrated_norm <= input_norm + 1e-9
    status <- if (success) "calibrated_soft" else "optimizer_failed"
    if (failed) status <- "line_search_or_hessian_failed"
    if (!failed && !converged) status <- "maximum_iterations_reached"
    if (converged && !valid) status <- "invalid_weight_solution"
    result$weights[d, ] <- weights
    result$success[d] <- success
    result$status[d] <- status
    result$input_imbalance_norm[d] <- input_norm
    result$calibrated_imbalance_norm[d] <- calibrated_norm
    result$maximum_absolute_imbalance[d] <- max(abs(imbalance))
    result$kl_divergence[d] <- sum(weights * (log(pmax(weights, .Machine$double.xmin * .Machine$double.eps)) - log_prior))
    result$effective_county_count[d] <- 1 / sum(weights^2)
    result$maximum_county_weight[d] <- max(weights)
  }
  result
}

iv_spec_design <- function(cell, families) {
  selected <- Filter(function(m) m$moment_family %in% families, cell$moments)
  active <- Filter(function(m) m$active, selected)
  design <- if (length(active)) do.call(cbind, lapply(active, `[[`, "standardized_values")) else matrix(numeric(), length(cell$prior), 0)
  target <- vapply(active, `[[`, numeric(1), "standardized_target")
  list(design = design, target = target, moments = selected)
}

iv_annual_targets <- function(source, years) {
  parts <- list()
  for (vintage in c("preliminary", "revised")) {
    parts[[vintage]] <- data.frame(aewr_region_id = source$aewr_region_id,
      source_year = as.integer(source[[paste0(vintage, "_year")]]),
      fls_annual_field_livestock_hourly_wage = as.numeric(source[[paste0("field_livestock_", vintage)]]),
      fls_annual_release_year = as.integer(source$estimate_year),
      fls_annual_wage_vintage = vintage, fls_annual_source_zip = source$source_zip,
      fls_annual_source_csv = source$source_csv, priority = match(vintage, c("preliminary", "revised")))
  }
  targets <- dplyr::bind_rows(parts)
  targets <- targets[targets$source_year %in% years & is.finite(targets$fls_annual_field_livestock_hourly_wage) &
    targets$fls_annual_field_livestock_hourly_wage > 0, ]
  targets <- iv_sort(targets, c("aewr_region_id", "source_year", "priority", "fls_annual_release_year"))
  targets <- targets[!duplicated(targets[c("aewr_region_id", "source_year")], fromLast = TRUE), ]
  targets$priority <- NULL
  iv_require_unique(targets, c("aewr_region_id", "source_year"), "Annual FLS targets")
  targets
}

iv_paired_targets <- function(workers, wages, years) {
  keys <- c("aewr_region_id", "year", "quarter")
  iv_require_unique(workers, keys, "FLS workers")
  iv_require_unique(wages, keys, "FLS wages")
  shared <- c("source_zip", "release_year", "release_month", "release_day")
  workers <- workers[c(keys, "fls_hired_workers", "fls_hired_workers_150_days_or_more",
    "fls_hired_workers_149_days_or_less", "fls_gross_hours_worked", shared,
    "fls_pair_values_available", "fls_pair_value_status")]
  wages <- wages[c(keys, "fls_field_hourly_wage", "fls_livestock_hourly_wage",
    "fls_field_livestock_hourly_wage", "fls_all_hired_hourly_wage", shared,
    "wage_source_csv", "wage_table_title")]
  names(workers)[match(shared, names(workers))] <- paste0("worker_", shared)
  names(wages)[match(shared, names(wages))] <- paste0("wage_", shared)
  paired <- dplyr::inner_join(workers, wages, by = keys, relationship = "one-to-one")
  names(paired)[names(paired) == "year"] <- "source_year"
  paired <- paired[paired$source_year %in% years, ]
  for (name in shared) {
    if (any(paired[[paste0("worker_", name)]] != paired[[paste0("wage_", name)]], na.rm = TRUE))
      stop("Quarterly worker/wage releases differ")
  }
  if (nrow(paired) != length(years) * 17L * 4L) stop("Incomplete paired FLS grid")
  available <- paired[paired$fls_pair_values_available, ]
  fields <- c("fls_hired_workers", "fls_field_hourly_wage", "fls_livestock_hourly_wage",
    "fls_field_livestock_hourly_wage", "fls_all_hired_hourly_wage")
  if (any(!is.finite(as.matrix(available[fields]))) || any(available$fls_hired_workers < 0))
    stop("Invalid available FLS pair")
  missing <- paired[!paired$fls_pair_values_available, ]
  if (nrow(missing) != (if (2011L %in% years) 17L else 0L) || any(missing$source_year != 2011L | missing$quarter != "april" |
      missing$fls_pair_value_status != "survey_not_conducted")) stop("Unexpected FLS survey gap")
  if (any(!paired$quarter %in% IV_QUARTERS)) stop("Invalid FLS quarter")
  iv_sort(paired, c("aewr_region_id", "source_year", "quarter"))
}

iv_impute_under_prior <- function(values, observed, prior) {
  was_vector <- is.null(dim(values))
  values <- as.matrix(values)
  mass <- sum(prior[observed])
  if (mass <= 0) stop("A moment has no observed positive prior mass")
  average <- colSums(values[observed, , drop = FALSE] * prior[observed]) / mass
  values[!observed, ] <- rep(average, each = sum(!observed))
  list(values = if (was_vector) as.numeric(values) else values, mass = mass)
}

iv_build_features <- function(frame, county_panel, qcew, annual, quarterly, years, regions = NULL) {
  prior <- dplyr::filter(frame, source_year %in% years, weight_spec == IV_PRIOR_SPEC,
    annual_update_spec == IV_UPDATE_SPEC)
  if (!is.null(regions)) prior <- prior[prior$aewr_region_id %in% regions, ]
  prior <- prior[c("county_fips", "state_fips", "aewr_region_id", "source_year", "frame_employment_mass",
    "annual_update_source", "qcew_strict_complete", "qwi_annual_fallback_used", "bea_annual_fallback_used", "quality_flags")]
  prior <- iv_sort(prior, c("aewr_region_id", "source_year", "county_fips"))
  iv_require_unique(prior, c("aewr_region_id", "source_year", "county_fips"), "County prior")
  assert_geo_columns(prior, c("county_fips", "state_fips", "aewr_region_id"))
  panel_wages <- county_panel[c("county_fips", "year", "oews_big_six_mean_hourly_wage", "oews_wage_observed")]
  names(panel_wages)[2] <- "source_year"
  iv_require_unique(panel_wages, c("county_fips", "source_year"), "County OEWS wages")
  prior <- dplyr::left_join(prior, panel_wages, by = c("county_fips", "source_year"), relationship = "one-to-one")
  iv_require_unique(qcew, c("county_fips", "year", "qtr", "industry_code"), "Quarterly QCEW")
  qcew <- dplyr::filter(qcew, year %in% years, industry_code %in% c("111", "112"),
    qcew_employment_disclosed, is.finite(qcew_reference_month_emplvl), qcew_reference_month_emplvl >= 0)
  q_lookup <- setNames(qcew$qcew_reference_month_emplvl, iv_key(qcew$county_fips, qcew$year, qcew$qtr, qcew$industry_code))
  annual_keys <- iv_key(annual$aewr_region_id, annual$source_year)
  pair_keys <- iv_key(quarterly$aewr_region_id, quarterly$source_year, quarterly$quarter)
  group_keys <- iv_key(prior$aewr_region_id, prior$source_year)
  features <- list()
  cells <- list()
  for (key in unique(group_keys)) {
    group <- prior[group_keys == key, ]
    region <- group$aewr_region_id[1]
    year <- group$source_year[1]
    counties <- group$county_fips
    n <- length(counties)
    mass <- group$frame_employment_mass
    if (any(!is.finite(mass) | mass < 0) || sum(mass) <= 0) stop("Invalid regional county mass")
    q <- mass / sum(mass)
    employment <- array(NA_real_, c(n, 4L, 2L))
    for (quarter in 1:4) for (industry in 1:2)
      employment[, quarter, industry] <- q_lookup[iv_key(counties, year, quarter, c("111", "112")[industry])]
    pairs <- quarterly[match(iv_key(region, year, IV_QUARTERS), pair_keys), ]
    counts <- pairs$fls_hired_workers
    available <- which(is.finite(counts) & counts >= 0)
    if (length(available) < 2 || sum(counts[available]) <= 0) stop("Insufficient seasonal targets")
    seasonal_employment <- employment[, available, , drop = FALSE]
    totals <- apply(seasonal_employment, 1L, sum)
    observed <- apply(is.finite(seasonal_employment), 1L, all) & totals > 0
    raw <- matrix(NA_real_, n, length(available))
    for (j in seq_along(available)) raw[observed, j] <-
      rowSums(employment[observed, available[j], , drop = FALSE], dims = 1L) / totals[observed]
    seasonal <- iv_impute_under_prior(raw, observed, q)
    target <- counts[available] / sum(counts[available])
    basis <- iv_helmert(length(available))
    moments <- list()
    for (j in 1:3) {
      active <- j < length(available)
      moments[[length(moments) + 1L]] <- list(moment_id = paste0("seasonal_helmert_", j),
        moment_family = "seasonal", raw_values = if (active) as.numeric(seasonal$values %*% basis[, j]) else numeric(n),
        raw_target = if (active) iv_dot(target, basis[, j]) else 0,
        observed = if (active) observed else rep(FALSE, n), observed_prior_mass = if (active) seasonal$mass else 0,
        feature_source = "qcew_111_112_quarterly_share",
        target_source = if (active) "fls_quarterly_worker_share" else "fls_survey_not_conducted",
        quarter = NA_character_, forced_status = if (!active) "inactive_fls_survey_not_conducted" else NULL)
    }
    for (j in 1:4) {
      observed <- apply(is.finite(employment[, j, , drop = FALSE]), 1L, all)
      wage <- as.numeric(pairs[j, c("fls_field_hourly_wage", "fls_livestock_hourly_wage", "fls_field_livestock_hourly_wage")])
      wage_available <- all(is.finite(wage))
      if (wage_available) {
        raw <- (wage[1] - wage[3]) * employment[, j, 1] + (wage[2] - wage[3]) * employment[, j, 2]
        filled <- iv_impute_under_prior(raw, observed, q)
      } else {
        observed <- rep(FALSE, n)
        filled <- list(values = numeric(n), mass = 0)
      }
      moments[[length(moments) + 1L]] <- list(moment_id = paste0("composition_", IV_QUARTERS[j]),
        moment_family = "composition", raw_values = filled$values, raw_target = 0,
        observed = observed, observed_prior_mass = filled$mass,
        feature_source = "qcew_111_112_undivided_residual", target_source = "fls_quarterly_field_livestock_wages",
        quarter = IV_QUARTERS[j], forced_status = if (!wage_available) "inactive_fls_survey_not_conducted" else NULL)
    }
    a <- annual[match(iv_key(region, year), annual_keys), ]
    if (nrow(a) != 1L || is.na(a$source_year)) stop("Missing annual FLS target")
    wage <- group$oews_big_six_mean_hourly_wage
    observed <- is.finite(wage) & wage > 0
    filled <- iv_impute_under_prior(wage, observed, q)
    moments[[8L]] <- list(moment_id = "annual_fls_oews_hourly_wage", moment_family = "annual_wage",
      raw_values = filled$values, raw_target = a$fls_annual_field_livestock_hourly_wage,
      observed = observed, observed_prior_mass = filled$mass,
      feature_source = "county_mapped_oews_big_six_hourly_wage",
      target_source = "fls_annual_field_livestock_hourly_wage", quarter = NA_character_)
    for (j in seq_along(moments)) {
      m <- moments[[j]]
      if (is.null(m$forced_status)) {
        standard <- iv_standardize(m$raw_values, m$raw_target, q)
      } else {
        standard <- list(prior_center = iv_dot(q, m$raw_values), prior_scale = 0,
          active = FALSE, status = m$forced_status, standardized_values = NULL, standardized_target = NA_real_)
      }
      m <- c(m, standard)
      moments[[j]] <- m
      features[[length(features) + 1L]] <- data.frame(aewr_region_id = region, source_year = year,
        county_fips = counties, moment_id = m$moment_id, moment_family = m$moment_family, quarter = m$quarter,
        frame_prior_weight = q, raw_feature_value = m$raw_values, feature_observed = m$observed,
        feature_imputed = !m$observed, observed_prior_mass = m$observed_prior_mass, raw_target = m$raw_target,
        prior_center = m$prior_center, prior_scale = m$prior_scale,
        standardized_feature_value = if (m$active) m$standardized_values else NA_real_,
        standardized_target = m$standardized_target, moment_active = m$active, moment_status = m$status,
        feature_source = m$feature_source, target_source = m$target_source,
        baseline_weight_spec = IV_PRIOR_SPEC, annual_update_spec = IV_UPDATE_SPEC, weight_spec = IV_WEIGHT_SPEC,
        fls_worker_reference_week_total_diagnostic = sum(counts, na.rm = TRUE),
        fls_annual_wage_vintage = a$fls_annual_wage_vintage, fls_annual_release_year = a$fls_annual_release_year)
    }
    cells[[key]] <- list(region = region, year = year, county_codes = counties, prior = q,
      moments = moments, annual_update_sources = group$annual_update_source)
  }
  features <- iv_sort(dplyr::bind_rows(features), c("aewr_region_id", "source_year", "county_fips", "moment_id"))
  iv_require_unique(features, c("aewr_region_id", "source_year", "county_fips", "moment_id"), "County features")
  list(features = features, cells = cells)
}

iv_recover_cells <- function(cells) {
  summaries <- diagnostics <- moments <- list()
  for (i in seq_along(cells)) {
    cell <- cells[[i]]; region <- cell$region; year <- cell$year
    if (i == 1L || i %% 25L == 0L) message("County recovery ", i, "/", length(cells))
    prior <- cell$prior
    supported <- prior > 0
    normalized <- prior[supported] / sum(prior[supported])
    for (primary in c(FALSE, TRUE)) {
      specification <- if (primary) IV_PRIMARY_SPEC else IV_WAGE_SPEC
      moment_spec <- if (primary) "fls_oews_wage_plus_qcew_seasonal_and_composition" else "fls_field_livestock_wage_only"
      selected <- iv_spec_design(cell, if (primary) c("annual_wage", "seasonal", "composition") else "annual_wage")
      center <- iv_solve_entropy(normalized, selected$design[supported, , drop = FALSE], selected$target)
      if (!center$success[1]) stop("County center failed: ", region, "/", year, "/", specification)
      active <- sum(vapply(selected$moments, `[[`, logical(1), "active"))
      inactive <- length(selected$moments) - active
      center_all <- numeric(length(prior)); center_all[supported] <- center$weights[1, ]
      diagnostics[[length(diagnostics) + 1L]] <- data.frame(aewr_region_id = region, source_year = year,
        specification = specification, moment_spec = moment_spec, is_primary = primary,
        optimizer_success = center$success[1], optimizer_status = center$status[1],
        optimizer_iterations = center$iterations[1], input_standardized_residual_norm = center$input_imbalance_norm[1],
        calibrated_standardized_residual_norm = center$calibrated_imbalance_norm[1],
        maximum_absolute_standardized_residual = center$maximum_absolute_imbalance[1], kl_divergence = center$kl_divergence[1],
        frame_effective_county_count = 1 / sum(normalized^2),
        calibrated_effective_county_count = center$effective_county_count[1],
        maximum_calibrated_county_weight = center$maximum_county_weight[1],
        active_moment_count = active, inactive_moment_count = inactive, weight_spec = IV_WEIGHT_SPEC,
        baseline_weight_spec = IV_PRIOR_SPEC, rho = IV_RHO)
      summaries[[length(summaries) + 1L]] <- data.frame(aewr_region_id = region, source_year = year,
        county_fips = cell$county_codes, frame_prior_weight = prior, calibrated_center_weight = center_all,
        center_solver_status = center$status[1], active_moment_count = active, inactive_moment_count = inactive,
        frame_effective_county_count = 1 / sum(normalized^2),
        calibrated_effective_county_count = center$effective_county_count[1],
        maximum_calibrated_county_weight = center$maximum_county_weight[1], specification = specification,
        moment_spec = moment_spec, weight_spec = IV_WEIGHT_SPEC, baseline_weight_spec = IV_PRIOR_SPEC,
        rho = IV_RHO, is_primary = primary)
      for (m in selected$moments) {
        prior_raw <- iv_dot(prior, m$raw_values); calibrated_raw <- iv_dot(center_all, m$raw_values)
        prior_standard <- if (m$active) iv_dot(normalized, m$standardized_values[supported]) else NA_real_
        calibrated_standard <- if (m$active) iv_dot(center_all[supported], m$standardized_values[supported]) else NA_real_
        moments[[length(moments) + 1L]] <- data.frame(aewr_region_id = region, source_year = year,
          specification = specification, moment_spec = moment_spec, is_primary = primary,
          moment_id = m$moment_id, moment_family = m$moment_family, quarter = m$quarter,
          moment_active = m$active, moment_status = m$status, observed_prior_mass = m$observed_prior_mass,
          raw_target = m$raw_target, prior_raw_moment = prior_raw, calibrated_raw_moment = calibrated_raw,
          prior_raw_residual = prior_raw - m$raw_target, calibrated_raw_residual = calibrated_raw - m$raw_target,
          standardized_target = m$standardized_target, prior_standardized_moment = prior_standard,
          calibrated_standardized_moment = calibrated_standard,
          calibrated_standardized_residual = calibrated_standard - m$standardized_target,
          prior_center = m$prior_center, prior_scale = m$prior_scale, weight_spec = IV_WEIGHT_SPEC, rho = IV_RHO)
      }
    }
  }
  list(weights = dplyr::bind_rows(summaries), calibration_diagnostics = dplyr::bind_rows(diagnostics),
    moment_diagnostics = dplyr::bind_rows(moments))
}

