# Conditional size and power under declared border-error processes.
# Pure branch helpers: no persistent reads or writes; the entry point owns I/O.

border_power_wilson <- function(successes, repetitions, level = 0.95) {
  z <- stats::qnorm(1 - (1 - level) / 2)
  p <- successes / repetitions
  denominator <- 1 + z^2 / repetitions
  center <- (p + z^2 / (2 * repetitions)) / denominator
  half <- z * sqrt(p * (1 - p) / repetitions +
    z^2 / (4 * repetitions^2)) / denominator
  cbind(low = pmin(p, pmax(0, center - half)),
        high = pmax(p, pmin(1, center + half)))
}

border_power_projection <- function(data, x, y, fe_only = FALSE) {
  required <- c(
    "county_fips", "region_pair_id", "aewr_region_id", "stratum_id",
    "unit_fe_id", "stratum_year_id", "year", "estimation_weight"
  )
  if (!all(required %in% names(data))) {
    stop("Power data are missing declared identifiers or weights.")
  }
  for (name in setdiff(required, c("year", "estimation_weight"))) {
    if (!(is.character(data[[name]]) || is.factor(data[[name]]))) {
      stop("Power identifiers must be strings: ", name)
    }
    data[[name]] <- as.character(data[[name]])
    if (anyNA(data[[name]]) || any(!nzchar(data[[name]]))) {
      stop("Power identifiers must be nonempty: ", name)
    }
  }
  # The baseline wage field contains a documented nonstatutory 2009 measure.
  # FE-only calibration must not inspect or use the preperiod x argument.
  x <- if (fe_only) matrix(numeric(), nrow(data), 0L) else as.matrix(x)
  y <- as.numeric(y)
  if ((!fe_only && (is.null(colnames(x)) || ncol(x) < 1L)) || nrow(data) != nrow(x) ||
      length(y) != nrow(data) || any(!is.finite(c(x, y))) ||
      any(!is.finite(data$estimation_weight)) ||
      any(data$estimation_weight <= 0) || anyNA(data$year)) {
    stop("Power matrices must be finite, aligned, named, and positively weighted.")
  }
  years <- sort(unique(data$year))
  units <- sort(unique(data$unit_fe_id))
  if (length(years) < 3L || any(diff(years) != 1L) ||
      anyDuplicated(data[c("unit_fe_id", "year")]) ||
      nrow(data) != length(units) * length(years)) {
    stop("Power simulation requires a balanced consecutive unit-year panel.")
  }
  unit_rows <- split(seq_len(nrow(data)), data$unit_fe_id)
  for (rows in unit_rows) {
    if (any(vapply(
      data[rows, c("county_fips", "region_pair_id", "aewr_region_id",
                   "stratum_id", "estimation_weight"), drop = FALSE],
      function(z) length(unique(z)) != 1L, logical(1)
    ))) {
      stop("Power weights and geographic membership must be fixed within unit.")
    }
  }
  if (any(vapply(split(data$region_pair_id, data$stratum_id),
                 function(z) length(unique(z)) != 1L, logical(1))) ||
      any(vapply(split(data$year, data$stratum_year_id),
                 function(z) length(unique(z)) != 1L, logical(1))) ||
      any(vapply(split(data$stratum_id, data$stratum_year_id),
                 function(z) length(unique(z)) != 1L, logical(1)))) {
    stop("Power fixed effects must be nested in a border pair and year.")
  }
  pair_regions <- lapply(split(data$aewr_region_id, data$region_pair_id),
                         function(z) sort(unique(z)))
  if (any(lengths(pair_regions) != 2L)) {
    stop("Every power border must retain both AEWR regions.")
  }
  weights <- data$estimation_weight / mean(data$estimation_weight)
  fe <- list(factor(data$unit_fe_id), factor(data$stratum_year_id))
  within <- fixest::demean(
    cbind(y, x), f = fe, weights = weights, tol = 1e-10, nthreads = 1L
  )
  weighted <- within * sqrt(weights)
  xw <- weighted[, -1L, drop = FALSE]
  yw <- weighted[, 1L]
  sxx <- crossprod(xw)
  if (!fe_only && qr(sxx, tol = 1e-10)$rank != ncol(sxx)) {
    stop("Power treatment/controls are not identified after fixed effects.")
  }
  bread <- if (fe_only) matrix(numeric(), 0L, 0L) else solve(sxx)
  fit <- drop(bread %*% crossprod(xw, yw))
  residual <- yw - drop(xw %*% fit)
  pairs <- names(pair_regions)
  regions <- sort(unique(data$aewr_region_id))
  incidence <- matrix(0, length(regions), length(pairs),
                      dimnames = list(regions, pairs))
  for (pair in pairs) incidence[pair_regions[[pair]], pair] <- 1
  pair_index <- match(data$region_pair_id, pairs)
  region_index <- match(data$aewr_region_id, regions)
  intersections <- interaction(data$region_pair_id, data$aewr_region_id,
                               drop = TRUE, lex.order = TRUE)
  partial_influence <- if (fe_only) rep(0, nrow(data)) else drop(xw %*% bread[, 1L])
  # Because every FE is nested in pair, deleting one entire pair preserves the
  # within transform of all retained rows. These operators exactly refit the
  # slope and every nuisance coefficient; no one-step jackknife is substituted.
  loo_operator <- matrix(NA_real_, length(pairs), nrow(data))
  loo_identified <- TRUE
  for (g in if (fe_only) integer() else seq_along(pairs)) {
    rows <- which(pair_index == g)
    retained_sxx <- sxx - crossprod(xw[rows, , drop = FALSE])
    if (qr(retained_sxx, tol = 1e-10)$rank < ncol(sxx)) {
      loo_identified <- FALSE
    } else {
      loo_operator[g, ] <- drop(xw %*% solve(retained_sxx)[, 1L])
      loo_operator[g, rows] <- 0
    }
  }
  list(
    data = data, xw = xw, yw = yw, bread = bread, fit = fit,
    residual = residual, residual_ss = sum(residual^2),
    weights = weights, fe = fe, years = years, units = units,
    pairs = pairs, regions = regions, pair_regions = pair_regions,
    pair_index = pair_index, region_index = region_index,
    intersection_index = as.integer(intersections), incidence = incidence,
    partial_influence = partial_influence, loo_operator = loo_operator,
    loo_identified = loo_identified
  )
}

border_power_within <- function(raw, prepared) {
  fixest::demean(
    as.matrix(raw), f = prepared$fe, weights = prepared$weights,
    tol = 1e-10, nthreads = 1L
  ) * sqrt(prepared$weights)
}

border_power_ar_draws <- function(n_units, n_years, repetitions, ar1) {
  paths <- array(stats::rnorm(n_units * n_years * repetitions),
                 dim = c(n_units, n_years, repetitions))
  innovation_scale <- sqrt(1 - ar1^2)
  for (year in seq.int(2L, n_years)) {
    paths[, year, ] <- ar1 * paths[, year - 1L, ] +
      innovation_scale * paths[, year, ]
  }
  matrix(paths, nrow = n_units * n_years, ncol = repetitions)
}

border_power_region_graph <- function(simulation_inputs, coefficient) {
  regions <- sort(unique(unlist(lapply(simulation_inputs, function(model) {
    as.character(model$policy$data$aewr_region_id)
  }))))
  adjacency <- matrix(0, length(regions), length(regions),
                      dimnames = list(regions, regions))
  for (model in simulation_inputs) {
    edges <- lapply(split(as.character(model$policy$data$aewr_region_id),
                          as.character(model$policy$data$region_pair_id)), unique)
    for (edge in edges) {
      if (length(edge) != 2L) stop("Spatial graph has an incomplete border.")
      adjacency[edge[1L], edge[2L]] <- 1
      adjacency[edge[2L], edge[1L]] <- 1
    }
  }
  degree <- rowSums(adjacency)
  scale <- ifelse(degree > 0, 1 / sqrt(pmax(degree, 1)), 0)
  normalized <- adjacency * outer(scale, scale)
  loading <- solve(diag(length(regions)) - coefficient * normalized)
  loading <- loading / sqrt(rowSums(loading^2))
  list(regions = regions, loading = loading)
}

border_power_unit_errors <- function(prepared, repetitions, scenario, ar1, graph) {
  data <- prepared$data
  n <- nrow(data)
  years <- prepared$years
  time_index <- match(data$year, years) - 1L
  counties <- sort(unique(data$county_fips))
  county_index <- match(data$county_fips, counties)
  raw <- matrix(0, n, repetitions)
  if (scenario$county > 0) {
    county <- border_power_ar_draws(length(counties), length(years),
                                    repetitions, ar1)
    raw <- raw + sqrt(scenario$county) *
      county[time_index * length(counties) + county_index, , drop = FALSE]
  }
  if (scenario$region > 0) {
    regions <- graph$regions
    region <- border_power_ar_draws(length(regions), length(years),
                                    repetitions, ar1)
    if (isTRUE(scenario$spatial)) {
      for (time in seq_along(years)) {
        rows <- (time - 1L) * length(regions) + seq_along(regions)
        region[rows, ] <- graph$loading %*% region[rows, , drop = FALSE]
      }
    }
    raw <- raw + sqrt(scenario$region) *
      region[time_index * length(regions) +
               match(data$aewr_region_id, regions), , drop = FALSE]
  }
  if (scenario$pair > 0) {
    pair <- border_power_ar_draws(length(prepared$pairs), length(years),
                                  repetitions, ar1)
    left <- vapply(prepared$pair_regions, `[`, character(1), 1L)
    side <- ifelse(data$aewr_region_id == left[data$region_pair_id], 1, -1)
    raw <- raw + sqrt(scenario$pair) * side *
      pair[time_index * length(prepared$pairs) + prepared$pair_index, , drop = FALSE]
  }
  raw
}

border_power_draw_statistics <- function(prepared, raw_errors) {
  yw <- border_power_within(raw_errors, prepared)
  beta <- prepared$bread %*% crossprod(prepared$xw, yw)
  residual <- yw - prepared$xw %*% beta
  observation_scores <- prepared$partial_influence * residual
  pair_score <- rowsum(observation_scores, prepared$pair_index, reorder = TRUE)
  region_score <- rowsum(observation_scores, prepared$region_index, reorder = TRUE)
  intersection_score <- rowsum(observation_scores,
                               prepared$intersection_index, reorder = TRUE)
  n <- nrow(prepared$xw)
  k <- ncol(prepared$xw)
  g <- length(prepared$pairs)
  r <- length(prepared$regions)
  if (g < 3L || r < 3L || n <= k) stop("Too little policy support for power.")
  # fixest's default K.fixef="nonnested" retains one common FE intercept even
  # when both FE dimensions are nested in the clustering dimension. Counting
  # only the explicitly estimated slopes would understate its CRV1 adjustment.
  correction <- (n - 1) / (n - k - 1L)
  pair_var <- correction * g / (g - 1) * colSums(pair_score^2)
  two_way_var <- correction * min(g, r) / (min(g, r) - 1) *
    (colSums(pair_score^2) + colSums(region_score^2) -
       colSums(intersection_score^2))
  # Whole-pair scores cover either endpoint after within-pair-year absorption.
  # This covariance is retained raw: negative values are unavailable, not zero.
  # The observed dyadic comparison uses only the declared node-count
  # correction, not fixest's regression-df multiplier. Match it exactly.
  dyadic_var <- r / (r - 1) *
    (colSums((prepared$incidence %*% pair_score)^2) - colSums(pair_score^2))
  if (prepared$loo_identified) {
    loo <- prepared$loo_operator %*% yw
    loo <- sweep(loo, 2L, colMeans(loo), "-")
    jackknife_var <- (g - 1) / g * colSums(loo^2)
  } else {
    jackknife_var <- rep(NA_real_, ncol(yw))
  }
  list(
    coefficient_error = beta[1L, ],
    variance = rbind(
      pair_CRV1 = pair_var, pair_region_CRV1 = two_way_var,
      pair_CV3J = jackknife_var, shared_endpoint_dyadic = dyadic_var
    ),
    df = c(pair_CRV1 = g - 1L, pair_region_CRV1 = min(g, r) - 1L,
           pair_CV3J = g - 1L, shared_endpoint_dyadic = r - 1L),
    pair_null_scores = if (ncol(prepared$xw) == 1L) {
      rowsum(prepared$xw[, 1L] * yw, prepared$pair_index, reorder = TRUE)
    } else NULL,
    residual_ss = colSums(residual^2)
  )
}

border_power_calibrate <- function(prepared, scenario, settings, graph) {
  unit_rss <- numeric(settings$calibration_repetitions)
  chunks <- split(seq_along(unit_rss),
                  ceiling(seq_along(unit_rss) / settings$chunk_size))
  for (indices in chunks) {
    errors <- border_power_unit_errors(prepared, length(indices), scenario,
                                       settings$ar1, graph)
    weighted <- border_power_within(errors, prepared)
    residual <- weighted - prepared$xw %*%
      (prepared$bread %*% crossprod(prepared$xw, weighted))
    unit_rss[indices] <- colSums(residual^2)
  }
  expected <- mean(unit_rss)
  if (!is.finite(expected) || expected <= 0 || prepared$residual_ss <= 0) {
    stop("Power calibration has no positive residual variation.")
  }
  list(
    sigma = sqrt(prepared$residual_ss / expected),
    observed_residual_ss = prepared$residual_ss,
    expected_unit_residual_ss = expected,
    expected_unit_residual_ss_mc_se = stats::sd(unit_rss) / sqrt(length(unit_rss))
  )
}

border_power_grid <- function(errors, variances, df, effects, sigma, alpha) {
  valid <- is.finite(variances) & variances > 0 & is.finite(errors)
  critical <- stats::qt(1 - alpha / 2, df)
  repetitions <- length(errors)
  result <- lapply(effects, function(effect) {
    reject <- rep(FALSE, repetitions)
    reject[valid] <- abs(effect + sigma * errors[valid]) >
      critical * sigma * sqrt(variances[valid])
    successes <- sum(reject)
    interval <- border_power_wilson(successes, repetitions)
    data.frame(
      effect = effect, repetitions = repetitions,
      valid_repetitions = sum(valid), invalid_repetitions = sum(!valid),
      rejections = successes, rejection_rate = successes / repetitions,
      rejection_rate_valid_only = if (any(valid)) successes / sum(valid) else NA_real_,
      rejection_rate_mc_se = sqrt(successes / repetitions *
        (1 - successes / repetitions) / repetitions),
      mc_confidence_low = interval[1L, "low"],
      mc_confidence_high = interval[1L, "high"], reference_df = df
    )
  })
  do.call(rbind, result)
}

run_border_power <- function(simulation_inputs, config) {
  settings <- config$redesign$power
  assumptions <- config$redesign$precision_assumptions
  required <- c("seed", "repetitions", "calibration_repetitions", "chunk_size",
                 "ar1", "nthreads", "spatial_graph_coefficient", "dgp_ids",
                 "variance_fractions", "effect_grid")
  if (!all(required %in% names(settings)) || settings$nthreads != 1L ||
      settings$repetitions < 99L || settings$calibration_repetitions < 99L ||
      settings$chunk_size < 1L || abs(settings$ar1) >= 1 ||
      settings$spatial_graph_coefficient < 0 ||
      settings$spatial_graph_coefficient >= 1) {
    stop("Power registry must explicitly declare valid calibration and DGP settings.")
  }
  if (!all(c("alpha", "target_power", "size_maximum") %in% names(assumptions)) ||
      assumptions$alpha <= 0 || assumptions$alpha >= 1 ||
      assumptions$target_power <= assumptions$alpha || assumptions$target_power >= 1) {
    stop("Power registry must declare size and power criteria.")
  }
  # Decimal registry values and seq() can represent the same declared effect
  # with different final binary bits. Canonicalize before unique(), matching,
  # and CSV serialization so each declared model/DGP/method/effect key is unique.
  effects <- unique(round(as.numeric(unlist(settings$effect_grid)), 12L))
  if (!0 %in% effects || any(!is.finite(effects)) ||
      !all(-effects %in% effects)) stop("Power effect grid must include zero and both signs.")
  mde_grid <- if (!is.null(settings$mde_grid)) {
    sort(unique(as.numeric(unlist(settings$mde_grid))))
  } else if (!is.null(settings$mde_grid_step) && !is.null(settings$mde_grid_max)) {
    if (!is.finite(settings$mde_grid_step) || settings$mde_grid_step <= 0 ||
        !is.finite(settings$mde_grid_max) || settings$mde_grid_max < settings$mde_grid_step) {
      stop("Invalid declared MDE grid step or maximum.")
    }
    seq(settings$mde_grid_step, settings$mde_grid_max, by = settings$mde_grid_step)
  } else sort(unique(abs(effects[effects != 0])))
  if (!length(mde_grid) || any(!is.finite(mde_grid)) || any(mde_grid <= 0)) {
    stop("Power MDE grid must be finite and positive.")
  }
  mde_grid <- sort(unique(round(mde_grid, 12L)))
  ids <- vapply(simulation_inputs, `[[`, character(1), "model_id")
  if (!length(ids) || anyDuplicated(ids)) stop("Power model IDs must be unique.")
  graph <- border_power_region_graph(simulation_inputs,
                                     settings$spatial_graph_coefficient)
  power_rows <- list()
  calibration_rows <- list()
  mde_rows <- list()
  diagnostic_rows <- list()
  bootstrap_inputs <- list()
  set.seed(as.integer(settings$seed))
  for (model in simulation_inputs) {
    message("Power: ", model$model_id)
    pre <- border_power_projection(model$pre$data, NULL, model$pre$y, fe_only = TRUE)
    policy <- border_power_projection(model$policy$data, model$policy$x, model$policy$y)
    if (!identical(pre$units, policy$units)) {
      stop("Preperiod calibration and policy samples must have identical units: ",
           model$model_id)
    }
    for (dgp in unlist(settings$dgp_ids)) {
      fractions <- settings$variance_fractions[[dgp]]
      if (is.null(fractions) ||
          !all(c("region", "pair", "county") %in% names(fractions))) {
        stop("Missing declared DGP variance fractions: ", dgp)
      }
      fractions <- as.list(fractions)
      values <- as.numeric(unlist(fractions[c("region", "pair", "county")]))
      if (any(!is.finite(values)) || any(values < 0) || abs(sum(values) - 1) > 1e-12) {
        stop("DGP variance fractions must be nonnegative and sum to one.")
      }
      scenario <- c(fractions[c("region", "pair", "county")],
                    list(spatial = identical(dgp, "spatial_region_serial")))
      scales <- list(
        preperiod_only = border_power_calibrate(pre, scenario, settings, graph),
        policy_residual_sensitivity = border_power_calibrate(policy, scenario, settings, graph)
      )
      draws <- list()
      chunks <- split(seq_len(settings$repetitions),
                      ceiling(seq_len(settings$repetitions) / settings$chunk_size))
      for (indices in chunks) {
        errors <- border_power_unit_errors(policy, length(indices), scenario,
                                           settings$ar1, graph)
        draws[[length(draws) + 1L]] <- border_power_draw_statistics(policy, errors)
      }
      errors <- unlist(lapply(draws, `[[`, "coefficient_error"))
      variances <- do.call(cbind, lapply(draws, `[[`, "variance"))
      reference_df <- draws[[1L]]$df
      if (model$model_id %in% unlist(settings$bootstrap_model_ids)) {
        bootstrap_count <- as.integer(settings$bootstrap_simulation_repetitions)
        if (ncol(policy$xw) != 1L || !is.finite(bootstrap_count) ||
            bootstrap_count < 99L || bootstrap_count > settings$repetitions) {
          stop("Scalar bootstrap simulation requires valid declared model/count.")
        }
        pair_null_scores <- do.call(cbind, lapply(draws, `[[`, "pair_null_scores"))
        bootstrap_inputs[[length(bootstrap_inputs) + 1L]] <- list(
          model_id = model$model_id, dgp_id = dgp,
          pair_null_scores = pair_null_scores[, seq_len(bootstrap_count), drop = FALSE],
          pair_treatment_ss = drop(rowsum(policy$xw[, 1L]^2,
                                           policy$pair_index, reorder = TRUE)),
          pair_ids = policy$pairs, scales = scales,
          n_obs = nrow(policy$data), n_pairs = length(policy$pairs),
          n_regions = length(policy$regions), n_regressors = ncol(policy$xw),
          fixed_effects_nested_in_pair = TRUE
        )
      }
      for (scale_name in names(scales)) {
        scale <- scales[[scale_name]]
        calibration_rows[[length(calibration_rows) + 1L]] <- data.frame(
          model_id = model$model_id, dgp_id = dgp, scale_information = scale_name,
          sigma = scale$sigma, observed_residual_ss = scale$observed_residual_ss,
          expected_unit_residual_ss = scale$expected_unit_residual_ss,
          expected_unit_residual_ss_mc_se = scale$expected_unit_residual_ss_mc_se,
          calibration_repetitions = settings$calibration_repetitions,
          calibration_projection = if (scale_name == "preperiod_only") {
            "unit_and_stratum_year_FE_only_no_wage_regressor"
          } else "unit_and_stratum_year_FE_plus_full_policy_regressors",
          ar1 = settings$ar1, region_variance_fraction = scenario$region,
          pair_variance_fraction = scenario$pair, county_variance_fraction = scenario$county,
          spatial_graph_coefficient = if (scenario$spatial) settings$spatial_graph_coefficient else 0,
          n_preperiod_years = length(pre$years), n_policy_years = length(policy$years),
          n_counties = length(unique(policy$data$county_fips)),
          n_pairs = length(policy$pairs), n_regions = length(policy$regions),
          seed = settings$seed
        )
        for (method in rownames(variances)) {
          curve <- border_power_grid(
            errors, variances[method, ], reference_df[[method]],
            unique(c(effects, mde_grid, -mde_grid)), scale$sigma, assumptions$alpha
          )
          null <- curve[curve$effect == 0, , drop = FALSE]
          size_gate <- null$invalid_repetitions == 0L &&
            null$mc_confidence_high <= assumptions$size_maximum
          curve$model_id <- model$model_id
          curve$dgp_id <- dgp
          curve$scale_information <- scale_name
          curve$inference_method <- method
          curve$size_gate_pass <- size_gate
          curve$alpha <- assumptions$alpha
          curve$seed <- settings$seed
          curve$declared_meaningful_effect <- curve$effect %in% effects
          power_rows[[length(power_rows) + 1L]] <- curve
          for (sign in c(-1, 1)) {
            directional <- curve[match(sign * mde_grid, curve$effect), , drop = FALSE]
            crossing <- function(values) {
              positions <- which(values >= assumptions$target_power)
              if (length(positions)) mde_grid[positions[1L]] else NA_real_
            }
            descriptive <- crossing(directional$rejection_rate)
            mde_rows[[length(mde_rows) + 1L]] <- data.frame(
              model_id = model$model_id, dgp_id = dgp,
              scale_information = scale_name, inference_method = method,
              effect_sign = sign, target_power = assumptions$target_power,
              grid_mde = descriptive,
              credible_grid_mde = if (size_gate) descriptive else NA_real_,
              mc_mde_lower = crossing(directional$mc_confidence_high),
              mc_mde_upper = crossing(directional$mc_confidence_low),
              mde_above_grid = is.na(descriptive), maximum_grid_effect = max(mde_grid),
              size_gate_pass = size_gate, null_rejection_rate = null$rejection_rate,
              null_mc_confidence_low = null$mc_confidence_low,
              null_mc_confidence_high = null$mc_confidence_high,
              power_curve_nonmonotone = any(diff(directional$rejection_rate) < 0),
              invalid_repetitions = null$invalid_repetitions,
              repetitions = settings$repetitions, seed = settings$seed
            )
          }
        }
      }
      diagnostic_rows[[length(diagnostic_rows) + 1L]] <- data.frame(
        model_id = model$model_id, dgp_id = dgp,
        pair_jackknife_identified = policy$loo_identified,
        negative_two_way_variances = sum(variances["pair_region_CRV1", ] < 0, na.rm = TRUE),
        negative_dyadic_variances = sum(variances["shared_endpoint_dyadic", ] < 0, na.rm = TRUE),
        simulated_coefficient_error_mean = mean(errors),
        simulated_coefficient_error_sd = stats::sd(errors),
        bootstrap_simulation_status = if (model$model_id %in%
                                          unlist(settings$bootstrap_model_ids)) {
          "sufficient_statistics_exported_for_separately_tested_WCR11_supplement"
        } else "not_in_declared_scalar_bootstrap_supplement",
        repetitions = settings$repetitions
      )
    }
  }
  list(
    power = do.call(rbind, power_rows), calibration = do.call(rbind, calibration_rows),
    mde = do.call(rbind, mde_rows),
    simulation_diagnostics = do.call(rbind, diagnostic_rows),
    bootstrap_inputs = bootstrap_inputs
  )
}
