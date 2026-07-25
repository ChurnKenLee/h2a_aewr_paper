# Pure numerical routines shared by the IV entropy-calibration scripts.
# Callers are responsible for loading dplyr/tibble before sourcing this file.

wage_entropy_weights <- function(lambda, prior_weight, wage) {
  log_weight <- log(prior_weight) + lambda * wage
  log_weight <- log_weight - max(log_weight)
  weight <- exp(log_weight)
  weight / sum(weight)
}

wage_entropy_mean <- function(lambda, prior_weight, wage) {
  weight <- wage_entropy_weights(lambda, prior_weight, wage)
  sum(weight * wage)
}

calibrate_wage_cell <- function(
  data,
  entropy_target_wage,
  tolerance = 1e-10
) {
  prior_weight <- data$oews_area_prior_weight
  wage <- data$oews_area_mean_hourly_wage
  prior_mean <- sum(prior_weight * wage)
  minimum_wage <- min(wage)
  maximum_wage <- max(wage)

  data$oews_prior_weighted_wage <- prior_mean
  data$oews_minimum_wage <- minimum_wage
  data$oews_maximum_wage <- maximum_wage
  data$entropy_lambda <- NA_real_
  data$oews_area_weight_wage_calibrated <- NA_real_
  data$calibration_status <- "outside_support"

  if (
    entropy_target_wage < minimum_wage - tolerance ||
      entropy_target_wage > maximum_wage + tolerance
  ) {
    return(data)
  }

  if (abs(entropy_target_wage - prior_mean) <= tolerance) {
    data$entropy_lambda <- 0
    data$oews_area_weight_wage_calibrated <- prior_weight
    data$calibration_status <- "calibrated"
    return(data)
  }

  if (abs(maximum_wage - minimum_wage) <= tolerance) {
    data$calibration_status <- "no_wage_variation"
    return(data)
  }

  if (abs(entropy_target_wage - minimum_wage) <= tolerance) {
    keep <- abs(wage - minimum_wage) <= tolerance
    weight <- if_else(keep, prior_weight, 0)
    data$entropy_lambda <- -Inf
    data$oews_area_weight_wage_calibrated <- weight / sum(weight)
    data$calibration_status <- "calibrated_boundary"
    return(data)
  }

  if (abs(entropy_target_wage - maximum_wage) <= tolerance) {
    keep <- abs(wage - maximum_wage) <= tolerance
    weight <- if_else(keep, prior_weight, 0)
    data$entropy_lambda <- Inf
    data$oews_area_weight_wage_calibrated <- weight / sum(weight)
    data$calibration_status <- "calibrated_boundary"
    return(data)
  }

  lower <- -1
  while (wage_entropy_mean(lower, prior_weight, wage) > entropy_target_wage) {
    lower <- lower * 2
  }

  upper <- 1
  while (wage_entropy_mean(upper, prior_weight, wage) < entropy_target_wage) {
    upper <- upper * 2
  }

  lambda <- uniroot(
    function(value) {
      wage_entropy_mean(value, prior_weight, wage) - entropy_target_wage
    },
    interval = c(lower, upper),
    tol = tolerance
  )$root

  data$entropy_lambda <- lambda
  data$oews_area_weight_wage_calibrated <- wage_entropy_weights(
    lambda,
    prior_weight,
    wage
  )
  data$calibration_status <- "calibrated"
  data
}

entropy_weights <- function(lambda, prior_weight, design) {
  log_weight <- log(prior_weight) + as.vector(design %*% lambda)
  log_weight <- log_weight - max(log_weight)
  weight <- exp(log_weight)
  weight / sum(weight)
}

entropy_offset_wage_weights <- function(lambda, log_weight_offset, wage) {
  log_weight <- log_weight_offset + lambda * wage
  log_weight <- log_weight - max(log_weight)
  weight <- exp(log_weight)
  weight / sum(weight)
}

entropy_wage_weights <- function(lambda, prior_weight, wage) {
  entropy_offset_wage_weights(lambda, log(prior_weight), wage)
}

entropy_offset_wage_mean <- function(
  lambda,
  log_weight_offset,
  wage
) {
  weight <- entropy_offset_wage_weights(
    lambda,
    log_weight_offset,
    wage
  )
  sum(weight * wage)
}

entropy_wage_mean <- function(lambda, prior_weight, wage) {
  entropy_offset_wage_mean(lambda, log(prior_weight), wage)
}

find_offset_wage_lambda <- function(
  log_weight_offset,
  wage,
  target_wage,
  tolerance = 1e-12
) {
  prior_mean <- entropy_offset_wage_mean(0, log_weight_offset, wage)
  if (abs(target_wage - prior_mean) <= tolerance) {
    return(0)
  }

  lower <- -1
  while (
    entropy_offset_wage_mean(lower, log_weight_offset, wage) > target_wage
  ) {
    lower <- lower * 2
  }

  upper <- 1
  while (
    entropy_offset_wage_mean(upper, log_weight_offset, wage) < target_wage
  ) {
    upper <- upper * 2
  }

  uniroot(
    function(value) {
      entropy_offset_wage_mean(value, log_weight_offset, wage) - target_wage
    },
    interval = c(lower, upper),
    tol = tolerance
  )$root
}

find_wage_lambda <- function(
  prior_weight,
  wage,
  target_wage,
  tolerance = 1e-12
) {
  find_offset_wage_lambda(
    log(prior_weight),
    wage,
    target_wage,
    tolerance
  )
}

solve_entropy_dual <- function(
  prior_weight,
  design,
  target,
  inverse_penalty,
  initial_lambda
) {
  objective <- function(lambda) {
    linear_predictor <- as.vector(design %*% lambda)
    maximum <- max(log(prior_weight) + linear_predictor)
    log_normalizer <- maximum + log(sum(exp(
      log(prior_weight) + linear_predictor - maximum
    )))
    log_normalizer - sum(lambda * target) +
      0.5 * sum(inverse_penalty * lambda^2)
  }

  gradient <- function(lambda) {
    weight <- entropy_weights(lambda, prior_weight, design)
    as.vector(crossprod(design, weight)) - target +
      inverse_penalty * lambda
  }

  optimization <- optim(
    par = initial_lambda,
    fn = objective,
    gr = gradient,
    method = "BFGS",
    control = list(maxit = 5000, reltol = 1e-14)
  )

  list(
    lambda = optimization$par,
    weight = entropy_weights(
      optimization$par,
      prior_weight,
      design
    ),
    convergence = optimization$convergence,
    objective = optimization$value
  )
}

solve_profiled_entropy_dual <- function(
  prior_weight,
  wage,
  target_wage,
  soft_design,
  soft_target,
  soft_penalty,
  initial_soft_lambda
) {
  profiled_state <- function(soft_lambda) {
    log_weight_offset <- log(prior_weight) +
      as.vector(soft_design %*% soft_lambda)
    wage_lambda <- find_offset_wage_lambda(
      log_weight_offset,
      wage,
      target_wage
    )
    log_weight <- log_weight_offset + wage_lambda * wage
    maximum <- max(log_weight)
    log_normalizer <- maximum + log(sum(exp(log_weight - maximum)))
    weight <- exp(log_weight - maximum)
    weight <- weight / sum(weight)
    list(
      wage_lambda = wage_lambda,
      weight = weight,
      log_normalizer = log_normalizer
    )
  }

  objective <- function(soft_lambda) {
    state <- profiled_state(soft_lambda)
    state$log_normalizer - state$wage_lambda * target_wage -
      sum(soft_lambda * soft_target) +
      sum(soft_lambda^2) / (2 * soft_penalty)
  }

  gradient <- function(soft_lambda) {
    state <- profiled_state(soft_lambda)
    as.vector(crossprod(soft_design, state$weight)) - soft_target +
      soft_lambda / soft_penalty
  }

  optimization <- optim(
    par = initial_soft_lambda,
    fn = objective,
    gr = gradient,
    method = "BFGS",
    control = list(maxit = 1000, reltol = 1e-12)
  )
  final_state <- profiled_state(optimization$par)

  list(
    wage_lambda = final_state$wage_lambda,
    soft_lambda = optimization$par,
    weight = final_state$weight,
    convergence = optimization$convergence,
    objective = optimization$value
  )
}

initialize_calibration_columns <- function(data) {
  data$oews_area_weight_soft_calibrated <- NA_real_
  data$calibration_status <- "not_attempted"
  data$requested_moment_spec <- NA_character_
  data$resolved_moment_spec <- NA_character_
  data$optimizer_convergence <- NA_integer_
  data$optimizer_dual_objective <- NA_real_
  data$soft_moment_count <- 0L
  data$entropy_lambda_wage <- NA_real_

  output_columns <- unique(c(
    unname(feature_output_names),
    unname(lambda_output_names),
    unname(center_output_names),
    unname(scale_output_names),
    unname(observed_mass_output_names)
  ))
  for (column in output_columns) {
    data[[column]] <- NA_real_
  }
  data
}

calibrate_soft_cell <- function(
  data,
  entropy_target_wage,
  include_wage_target,
  moment_spec,
  soft_penalty,
  tolerance = 1e-10,
  minimum_scale = 1e-8
) {
  data <- initialize_calibration_columns(data)
  requested_moment_spec <- paste(moment_spec, collapse = ", ")
  if (!nzchar(requested_moment_spec)) {
    requested_moment_spec <- "<empty>"
  }
  supported_moment_specs <- c(
    "wage_only",
    "duration",
    "duration_seasonal",
    "wage_duration",
    "wage_duration_seasonal"
  )
  if (
    length(moment_spec) != 1L ||
      is.na(moment_spec) ||
      !moment_spec %in% supported_moment_specs
  ) {
    warning(
      paste0(
        "Unknown `moment_spec` (",
        requested_moment_spec,
        "); defaulting to `wage_only`."
      ),
      call. = FALSE
    )
    moment_spec <- "wage_only"
    include_wage_target <- TRUE
  }
  data$requested_moment_spec <- requested_moment_spec
  data$resolved_moment_spec <- moment_spec

  prior_weight <- data$oews_area_prior_weight
  prior_weight <- prior_weight / sum(prior_weight)
  wage <- data$oews_area_mean_hourly_wage
  prior_wage <- sum(prior_weight * wage)
  minimum_wage <- min(wage)
  maximum_wage <- max(wage)

  data$oews_prior_weighted_wage <- prior_wage
  data$oews_minimum_wage <- minimum_wage
  data$oews_maximum_wage <- maximum_wage

  if (include_wage_target) {
    if (is.na(entropy_target_wage)) {
      data$calibration_status <- "missing_wage_target"
      return(data)
    }
    if (
      entropy_target_wage < minimum_wage - tolerance ||
        entropy_target_wage > maximum_wage + tolerance
    ) {
      data$calibration_status <- "outside_wage_support"
      return(data)
    }
  }

  intended_features <- character()
  intended_targets <- character()
  if (moment_spec %in% c(
    "duration",
    "duration_seasonal",
    "wage_duration",
    "wage_duration_seasonal"
  )) {
    intended_features <- c(intended_features, duration_feature)
    intended_targets <- c(intended_targets, duration_target)
  }
  if (moment_spec %in% c(
    "duration_seasonal",
    "wage_duration_seasonal"
  )) {
    intended_features <- c(intended_features, seasonal_features)
    intended_targets <- c(intended_targets, seasonal_targets)
  }

  standardized_features <- list()
  standardized_targets <- numeric()
  missing_moments <- character()

  for (index in seq_along(intended_features)) {
    feature_name <- intended_features[[index]]
    target_name <- intended_targets[[index]]
    feature <- data[[feature_name]]
    target_value <- data[[target_name]][[1]]
    observed <- !is.na(feature)
    observed_prior_mass <- sum(prior_weight[observed])

    data[[observed_mass_output_names[[feature_name]]]] <- observed_prior_mass

    if (is.na(target_value) || observed_prior_mass <= tolerance) {
      missing_moments <- c(missing_moments, feature_name)
      next
    }

    center <- sum(prior_weight[observed] * feature[observed]) /
      observed_prior_mass
    imputed_feature <- if_else(observed, feature, center)
    scale <- sqrt(sum(prior_weight * (imputed_feature - center)^2))

    data[[center_output_names[[feature_name]]]] <- center
    data[[scale_output_names[[feature_name]]]] <- scale
    data[[feature_output_names[[feature_name]]]] <- imputed_feature

    if (!is.finite(scale) || scale <= minimum_scale) {
      missing_moments <- c(missing_moments, feature_name)
      next
    }

    standardized_features[[feature_name]] <-
      (imputed_feature - center) / scale
    standardized_targets[[feature_name]] <-
      (target_value - center) / scale
  }

  if (length(missing_moments) > 0) {
    data$calibration_status <- "missing_or_constant_soft_moment"
    return(data)
  }

  data$soft_moment_count <- length(standardized_features)

  # With no active targets, retain the prior weights. This branch also makes
  # the no-wage behavior explicit if a prior-only specification is added.
  if (length(standardized_features) == 0) {
    if (!include_wage_target) {
      data$oews_area_weight_soft_calibrated <- prior_weight
      data$optimizer_convergence <- 0L
      data$calibration_status <- "calibrated_prior"
      return(data)
    }

    # Preserve the existing wage-only path exactly. Boundary targets have no
    # finite wage multiplier.
    if (abs(entropy_target_wage - minimum_wage) <= tolerance) {
      keep <- abs(wage - minimum_wage) <= tolerance
      weight <- if_else(keep, prior_weight, 0)
      data$oews_area_weight_soft_calibrated <- weight / sum(weight)
      data$entropy_lambda_wage <- -Inf
      data$calibration_status <- "calibrated_boundary"
      return(data)
    }
    if (abs(entropy_target_wage - maximum_wage) <= tolerance) {
      keep <- abs(wage - maximum_wage) <= tolerance
      weight <- if_else(keep, prior_weight, 0)
      data$oews_area_weight_soft_calibrated <- weight / sum(weight)
      data$entropy_lambda_wage <- Inf
      data$calibration_status <- "calibrated_boundary"
      return(data)
    }
    if (abs(maximum_wage - minimum_wage) <= tolerance) {
      data$calibration_status <- "no_wage_variation"
      return(data)
    }

    wage_lambda <- find_wage_lambda(
      prior_weight,
      wage,
      entropy_target_wage
    )
    data$oews_area_weight_soft_calibrated <- entropy_wage_weights(
      wage_lambda,
      prior_weight,
      wage
    )
    data$entropy_lambda_wage <- wage_lambda
    data$optimizer_convergence <- 0L
    data$calibration_status <- "calibrated"
    return(data)
  }

  soft_design <- do.call(cbind, standardized_features)
  soft_target <- unname(standardized_targets)

  if (!include_wage_target) {
    solution <- solve_entropy_dual(
      prior_weight = prior_weight,
      design = soft_design,
      target = soft_target,
      inverse_penalty = rep(1 / soft_penalty, length(soft_target)),
      initial_lambda = rep(0, length(soft_target))
    )
    calibrated_weight <- solution$weight
    wage_lambda <- NA_real_
    soft_lambda <- solution$lambda
    calibration_status <- "calibrated"
  } else {
    boundary_minimum <- abs(entropy_target_wage - minimum_wage) <= tolerance
    boundary_maximum <- abs(entropy_target_wage - maximum_wage) <= tolerance

    if (boundary_minimum || boundary_maximum) {
      keep <- if (boundary_minimum) {
        abs(wage - minimum_wage) <= tolerance
      } else {
        abs(wage - maximum_wage) <= tolerance
      }
      boundary_prior <- prior_weight[keep]
      boundary_prior <- boundary_prior / sum(boundary_prior)
      solution <- solve_entropy_dual(
        prior_weight = boundary_prior,
        design = soft_design[keep, , drop = FALSE],
        target = soft_target,
        inverse_penalty = rep(1 / soft_penalty, length(soft_target)),
        initial_lambda = rep(0, length(soft_target))
      )
      calibrated_weight <- rep(0, nrow(data))
      calibrated_weight[keep] <- solution$weight
      wage_lambda <- if (boundary_minimum) -Inf else Inf
      soft_lambda <- solution$lambda
      calibration_status <- "calibrated_boundary"
    } else {
      if (abs(maximum_wage - minimum_wage) <= tolerance) {
        data$calibration_status <- "no_wage_variation"
        return(data)
      }
      design <- cbind(wage = wage, soft_design)
      joint_solution <- solve_entropy_dual(
        prior_weight = prior_weight,
        design = design,
        target = c(entropy_target_wage, soft_target),
        inverse_penalty = c(0, rep(1 / soft_penalty, length(soft_target))),
        initial_lambda = c(
          find_wage_lambda(prior_weight, wage, entropy_target_wage),
          rep(0, length(soft_target))
        )
      )
      # Re-optimize the soft multipliers with the wage multiplier profiled out.
      # Every objective and gradient evaluation therefore satisfies the original
      # wage constraint up to the one-dimensional root tolerance.
      solution <- solve_profiled_entropy_dual(
        prior_weight = prior_weight,
        wage = wage,
        target_wage = entropy_target_wage,
        soft_design = soft_design,
        soft_target = soft_target,
        soft_penalty = soft_penalty,
        initial_soft_lambda = joint_solution$lambda[-1]
      )
      calibrated_weight <- solution$weight
      wage_lambda <- solution$wage_lambda
      soft_lambda <- solution$soft_lambda
      calibration_status <- "calibrated"
    }
  }

  data$optimizer_convergence <- solution$convergence
  data$optimizer_dual_objective <- solution$objective
  if (
    solution$convergence != 0 ||
      any(!is.finite(calibrated_weight)) ||
      (
        include_wage_target &&
          abs(sum(calibrated_weight * wage) - entropy_target_wage) > 1e-7
      )
  ) {
    data$calibration_status <- "optimizer_failed"
    return(data)
  }

  data$oews_area_weight_soft_calibrated <- calibrated_weight
  data$entropy_lambda_wage <- wage_lambda
  for (index in seq_along(intended_features)) {
    feature_name <- intended_features[[index]]
    data[[lambda_output_names[[feature_name]]]] <- soft_lambda[[index]]
  }
  data$calibration_status <- calibration_status
  data
}

weighted_sum_if_observed <- function(weight, value) {
  if (all(is.na(weight)) || all(is.na(value))) {
    return(NA_real_)
  }
  sum(weight * value, na.rm = FALSE)
}

max_if_observed <- function(value) {
  if (all(is.na(value))) {
    return(NA_real_)
  }
  max(value, na.rm = TRUE)
}

# Generic exact/interval/soft calibration -----------------------------------

supported_entropy_moment_specs <- function() {
  c(
    "wage_only_exact",
    "wage_seasonal_exact",
    "wage_seasonal_qwi_duration_exact",
    "wage_seasonal_census_duration_exact",
    "wage_seasonal_interval",
    paste0(
      "wage_seasonal_soft_rho",
      c("001", "003", "010", "030", "100")
    )
  )
}

resolve_entropy_moment_spec <- function(moment_spec) {
  requested <- paste(moment_spec, collapse = ", ")
  if (!nzchar(requested)) {
    requested <- "<empty>"
  }
  valid <- length(moment_spec) == 1L &&
    !is.na(moment_spec) &&
    moment_spec %in% supported_entropy_moment_specs()
  if (!valid) {
    warning(
      paste0(
        "Unknown `moment_spec` (",
        requested,
        "); defaulting to `wage_only_exact`."
      ),
      call. = FALSE
    )
    return(list(
      requested = requested,
      resolved = "wage_only_exact",
      defaulted = TRUE
    ))
  }
  list(
    requested = requested,
    resolved = moment_spec,
    defaulted = FALSE
  )
}

normalize_entropy_prior <- function(prior_weight) {
  if (
    length(prior_weight) == 0L ||
      any(!is.finite(prior_weight)) ||
      any(prior_weight <= 0) ||
      sum(prior_weight) <= 0
  ) {
    stop("Entropy priors must be finite and strictly positive.", call. = FALSE)
  }
  prior_weight / sum(prior_weight)
}

calibration_lp <- function(
  observation_count,
  exact_design = NULL,
  exact_target = numeric(),
  interval_design = NULL,
  interval_lower = numeric(),
  interval_upper = numeric(),
  objective = rep(0, observation_count),
  direction = "min"
) {
  if (!requireNamespace("lpSolve", quietly = TRUE)) {
    stop(
      paste0(
        "Package `lpSolve` is required for the feasibility check. ",
        "Restore the project renv before running calibration."
      ),
      call. = FALSE
    )
  }

  constraint_matrix <- matrix(
    1,
    nrow = 1L,
    ncol = observation_count
  )
  constraint_direction <- "="
  constraint_rhs <- 1

  if (!is.null(exact_design) && ncol(exact_design) > 0L) {
    constraint_matrix <- rbind(
      constraint_matrix,
      t(exact_design)
    )
    constraint_direction <- c(
      constraint_direction,
      rep("=", ncol(exact_design))
    )
    constraint_rhs <- c(constraint_rhs, exact_target)
  }

  if (!is.null(interval_design) && ncol(interval_design) > 0L) {
    constraint_matrix <- rbind(
      constraint_matrix,
      t(interval_design),
      t(interval_design)
    )
    constraint_direction <- c(
      constraint_direction,
      rep(">=", ncol(interval_design)),
      rep("<=", ncol(interval_design))
    )
    constraint_rhs <- c(
      constraint_rhs,
      interval_lower,
      interval_upper
    )
  }

  lpSolve::lp(
    direction = direction,
    objective.in = objective,
    const.mat = constraint_matrix,
    const.dir = constraint_direction,
    const.rhs = constraint_rhs,
    all.int = FALSE,
    all.bin = FALSE,
    compute.sens = 0
  )
}

independent_calibration_columns <- function(design, tolerance = 1e-10) {
  if (is.null(design) || ncol(design) == 0L) {
    return(integer())
  }
  augmented <- cbind(intercept = 1, design)
  decomposition <- qr(augmented, tol = tolerance, LAPACK = FALSE)
  selected <- decomposition$pivot[seq_len(decomposition$rank)]
  selected <- selected[selected != 1L] - 1L
  sort(selected)
}

exact_feasible_support <- function(
  design,
  target,
  tolerance = 1e-10
) {
  observation_count <- nrow(design)
  maximum_feasible_weight <- numeric(observation_count)
  for (index in seq_len(observation_count)) {
    objective <- rep(0, observation_count)
    objective[[index]] <- 1
    solution <- calibration_lp(
      observation_count = observation_count,
      exact_design = design,
      exact_target = target,
      objective = objective,
      direction = "max"
    )
    if (solution$status == 0L) {
      maximum_feasible_weight[[index]] <- solution$objval
    }
  }
  maximum_feasible_weight > tolerance
}

validate_entropy_weight <- function(weight, tolerance = 1e-8) {
  length(weight) > 0L &&
    all(is.finite(weight)) &&
    all(weight >= -tolerance) &&
    abs(sum(weight) - 1) <= tolerance
}

entropy_kl_divergence <- function(weight, prior_weight) {
  positive <- weight > 0
  sum(weight[positive] * log(weight[positive] / prior_weight[positive]))
}

refine_exact_entropy_dual <- function(
  lambda,
  prior_weight,
  design,
  target,
  tolerance,
  maximum_iterations = 100L
) {
  dual_objective <- function(current_lambda) {
    linear_predictor <- as.vector(design %*% current_lambda)
    maximum <- max(log(prior_weight) + linear_predictor)
    log_normalizer <- maximum + log(sum(exp(
      log(prior_weight) + linear_predictor - maximum
    )))
    log_normalizer - sum(current_lambda * target)
  }

  for (iteration in seq_len(maximum_iterations)) {
    weight <- entropy_weights(lambda, prior_weight, design)
    moment <- as.vector(crossprod(design, weight))
    gradient <- moment - target
    if (max(abs(gradient)) <= tolerance) {
      break
    }
    centered_design <- sweep(design, 2, moment, "-")
    hessian <- crossprod(
      centered_design,
      centered_design * weight
    )
    newton_step <- tryCatch(
      solve(
        hessian + diag(1e-14, ncol(hessian)),
        gradient
      ),
      error = function(error) rep(NA_real_, length(lambda))
    )
    if (any(!is.finite(newton_step))) {
      break
    }
    current_objective <- dual_objective(lambda)
    current_gradient_norm <- max(abs(gradient))
    step_size <- 1
    accepted <- FALSE
    while (step_size >= 2^-20) {
      candidate <- lambda - step_size * newton_step
      candidate_objective <- dual_objective(candidate)
      candidate_weight <- entropy_weights(
        candidate,
        prior_weight,
        design
      )
      candidate_gradient_norm <- max(abs(
        crossprod(design, candidate_weight) - target
      ))
      if (
        is.finite(candidate_objective) &&
          (
            candidate_gradient_norm < current_gradient_norm ||
              candidate_objective < current_objective
          )
      ) {
        lambda <- candidate
        accepted <- TRUE
        break
      }
      step_size <- step_size / 2
    }
    if (!accepted) {
      break
    }
  }
  list(
    lambda = lambda,
    weight = entropy_weights(lambda, prior_weight, design),
    objective = dual_objective(lambda)
  )
}

solve_exact_entropy <- function(
  prior_weight,
  design,
  target,
  tolerance = 1e-8
) {
  prior_weight <- normalize_entropy_prior(prior_weight)
  design <- as.matrix(design)
  target <- as.numeric(target)
  feasibility <- calibration_lp(
    observation_count = length(prior_weight),
    exact_design = design,
    exact_target = target
  )
  if (feasibility$status != 0L) {
    return(list(
      status = "exact_infeasible",
      weight = rep(NA_real_, length(prior_weight)),
      lambda = rep(NA_real_, ncol(design)),
      lp_status = feasibility$status,
      convergence = NA_integer_,
      objective = NA_real_,
      max_residual = NA_real_
    ))
  }

  retained_columns <- independent_calibration_columns(design)
  reduced_design <- design[, retained_columns, drop = FALSE]
  reduced_target <- target[retained_columns]
  if (ncol(reduced_design) == 0L) {
    weight <- prior_weight
    residual <- if (ncol(design) == 0L) {
      0
    } else {
      max(abs(crossprod(design, weight) - target))
    }
    return(list(
      status = "calibrated_exact",
      weight = weight,
      lambda = rep(0, ncol(design)),
      lp_status = feasibility$status,
      convergence = 0L,
      objective = entropy_kl_divergence(weight, prior_weight),
      max_residual = residual
    ))
  }

  fit_dual <- function(active) {
    active_prior <- normalize_entropy_prior(prior_weight[active])
    solution <- solve_entropy_dual(
      prior_weight = active_prior,
      design = reduced_design[active, , drop = FALSE],
      target = reduced_target,
      inverse_penalty = rep(0, length(reduced_target)),
      initial_lambda = rep(0, length(reduced_target))
    )
    refined <- refine_exact_entropy_dual(
      lambda = solution$lambda,
      prior_weight = active_prior,
      design = reduced_design[active, , drop = FALSE],
      target = reduced_target,
      tolerance = tolerance
    )
    solution$lambda <- refined$lambda
    solution$weight <- refined$weight
    solution$objective <- refined$objective
    weight <- rep(0, length(prior_weight))
    weight[active] <- solution$weight
    residual <- max(abs(crossprod(design, weight) - target))
    list(
      solution = solution,
      weight = weight,
      residual = residual
    )
  }

  active <- rep(TRUE, length(prior_weight))
  fit <- fit_dual(active)
  valid <- validate_entropy_weight(fit$weight, tolerance) &&
    is.finite(fit$residual) &&
    fit$residual <= tolerance

  # A target on the boundary of the convex hull has zero weight outside its
  # feasible face and no finite full-support dual solution. Identify that face
  # when the ordinary dual fit fails or approaches zero support; this is still
  # the exact KL projection and is not a fallback to a softer specification.
  if (!valid || any(fit$weight <= tolerance)) {
    active <- exact_feasible_support(design, target, tolerance / 10)
    if (any(active) && !all(active)) {
      fit <- fit_dual(active)
      valid <- validate_entropy_weight(fit$weight, tolerance) &&
        is.finite(fit$residual) &&
        fit$residual <= tolerance
    }
  }

  lambda <- rep(0, ncol(design))
  lambda[retained_columns] <- fit$solution$lambda
  if (!valid) {
    return(list(
      status = "optimizer_failed_exact",
      weight = rep(NA_real_, length(prior_weight)),
      lambda = lambda,
      lp_status = feasibility$status,
      convergence = fit$solution$convergence,
      objective = fit$solution$objective,
      max_residual = fit$residual
    ))
  }

  list(
    status = if (all(active)) {
      "calibrated_exact"
    } else {
      "calibrated_exact_boundary"
    },
    weight = fit$weight,
    lambda = lambda,
    lp_status = feasibility$status,
    convergence = fit$solution$convergence,
    objective = entropy_kl_divergence(fit$weight, prior_weight),
    max_residual = fit$residual
  )
}

solve_interval_entropy_dual <- function(
  prior_weight,
  exact_design,
  exact_target,
  interval_design,
  interval_lower,
  interval_upper
) {
  exact_count <- ncol(exact_design)
  interval_count <- ncol(interval_design)

  if (exact_count == 1L) {
    profiled_state <- function(parameter) {
      upper_multiplier <- parameter[seq_len(interval_count)]
      lower_multiplier <- parameter[
        interval_count + seq_len(interval_count)
      ]
      log_weight_offset <- log(prior_weight) +
        as.vector(interval_design %*% (
          lower_multiplier - upper_multiplier
        ))
      exact_lambda <- find_offset_wage_lambda(
        log_weight_offset,
        exact_design[, 1],
        exact_target[[1]]
      )
      log_weight <- log_weight_offset +
        exact_lambda * exact_design[, 1]
      maximum <- max(log_weight)
      log_normalizer <- maximum + log(sum(exp(
        log_weight - maximum
      )))
      weight <- exp(log_weight - maximum)
      weight <- weight / sum(weight)
      list(
        exact_lambda = exact_lambda,
        upper_multiplier = upper_multiplier,
        lower_multiplier = lower_multiplier,
        log_normalizer = log_normalizer,
        weight = weight
      )
    }

    objective <- function(parameter) {
      current <- profiled_state(parameter)
      current$log_normalizer -
        current$exact_lambda * exact_target[[1]] +
        sum(current$upper_multiplier * interval_upper) -
        sum(current$lower_multiplier * interval_lower)
    }

    gradient <- function(parameter) {
      current <- profiled_state(parameter)
      interval_moment <- as.vector(crossprod(
        interval_design,
        current$weight
      ))
      c(
        interval_upper - interval_moment,
        interval_moment - interval_lower
      )
    }

    optimization <- optim(
      par = rep(0, 2L * interval_count),
      fn = objective,
      gr = gradient,
      method = "L-BFGS-B",
      lower = rep(0, 2L * interval_count),
      control = list(maxit = 5000, factr = 10, pgtol = 1e-12)
    )
    final <- profiled_state(optimization$par)
    return(list(
      weight = final$weight,
      exact_lambda = final$exact_lambda,
      interval_lambda = final$lower_multiplier -
        final$upper_multiplier,
      upper_multiplier = final$upper_multiplier,
      lower_multiplier = final$lower_multiplier,
      convergence = optimization$convergence,
      objective = optimization$value
    ))
  }

  state <- function(parameter) {
    exact_lambda <- parameter[seq_len(exact_count)]
    upper_multiplier <- parameter[
      exact_count + seq_len(interval_count)
    ]
    lower_multiplier <- parameter[
      exact_count + interval_count + seq_len(interval_count)
    ]
    linear_predictor <- as.vector(exact_design %*% exact_lambda) +
      as.vector(interval_design %*% (
        lower_multiplier - upper_multiplier
      ))
    maximum <- max(log(prior_weight) + linear_predictor)
    log_normalizer <- maximum + log(sum(exp(
      log(prior_weight) + linear_predictor - maximum
    )))
    weight <- exp(log(prior_weight) + linear_predictor - maximum)
    weight <- weight / sum(weight)
    list(
      exact_lambda = exact_lambda,
      upper_multiplier = upper_multiplier,
      lower_multiplier = lower_multiplier,
      log_normalizer = log_normalizer,
      weight = weight
    )
  }

  objective <- function(parameter) {
    current <- state(parameter)
    current$log_normalizer -
      sum(current$exact_lambda * exact_target) +
      sum(current$upper_multiplier * interval_upper) -
      sum(current$lower_multiplier * interval_lower)
  }

  gradient <- function(parameter) {
    current <- state(parameter)
    exact_moment <- as.vector(crossprod(
      exact_design,
      current$weight
    ))
    interval_moment <- as.vector(crossprod(
      interval_design,
      current$weight
    ))
    c(
      exact_moment - exact_target,
      interval_upper - interval_moment,
      interval_moment - interval_lower
    )
  }

  initial <- rep(0, exact_count + 2L * interval_count)
  lower_bound <- c(
    rep(-Inf, exact_count),
    rep(0, 2L * interval_count)
  )
  optimization <- optim(
    par = initial,
    fn = objective,
    gr = gradient,
    method = "L-BFGS-B",
    lower = lower_bound,
    control = list(maxit = 5000, factr = 10, pgtol = 1e-12)
  )
  final <- state(optimization$par)
  list(
    weight = final$weight,
    exact_lambda = final$exact_lambda,
    interval_lambda = final$lower_multiplier -
      final$upper_multiplier,
    upper_multiplier = final$upper_multiplier,
    lower_multiplier = final$lower_multiplier,
    convergence = optimization$convergence,
    objective = optimization$value
  )
}

solve_interval_entropy <- function(
  prior_weight,
  exact_design,
  exact_target,
  interval_design,
  interval_lower,
  interval_upper,
  tolerance = 1e-8
) {
  prior_weight <- normalize_entropy_prior(prior_weight)
  exact_design <- as.matrix(exact_design)
  interval_design <- as.matrix(interval_design)
  feasibility <- calibration_lp(
    observation_count = length(prior_weight),
    exact_design = exact_design,
    exact_target = exact_target,
    interval_design = interval_design,
    interval_lower = interval_lower,
    interval_upper = interval_upper
  )
  if (feasibility$status != 0L) {
    return(list(
      status = "interval_infeasible",
      weight = rep(NA_real_, length(prior_weight)),
      exact_lambda = rep(NA_real_, ncol(exact_design)),
      interval_lambda = rep(NA_real_, ncol(interval_design)),
      upper_multiplier = rep(NA_real_, ncol(interval_design)),
      lower_multiplier = rep(NA_real_, ncol(interval_design)),
      lp_status = feasibility$status,
      convergence = NA_integer_,
      objective = NA_real_,
      exact_max_residual = NA_real_,
      interval_max_violation = NA_real_
    ))
  }

  if (ncol(exact_design) == 1L) {
    exact_value <- exact_design[, 1]
    boundary_minimum <- abs(
      exact_target[[1]] - min(exact_value)
    ) <= tolerance
    boundary_maximum <- abs(
      exact_target[[1]] - max(exact_value)
    ) <= tolerance
    if (boundary_minimum || boundary_maximum) {
      keep <- if (boundary_minimum) {
        abs(exact_value - min(exact_value)) <= tolerance
      } else {
        abs(exact_value - max(exact_value)) <= tolerance
      }
      boundary_solution <- solve_interval_entropy(
        prior_weight = prior_weight[keep],
        exact_design = matrix(
          numeric(),
          nrow = sum(keep),
          ncol = 0L
        ),
        exact_target = numeric(),
        interval_design = interval_design[keep, , drop = FALSE],
        interval_lower = interval_lower,
        interval_upper = interval_upper,
        tolerance = tolerance
      )
      weight <- rep(NA_real_, length(prior_weight))
      if (startsWith(
        boundary_solution$status,
        "calibrated"
      )) {
        weight[] <- 0
        weight[keep] <- boundary_solution$weight
      }
      return(list(
        status = if (startsWith(
          boundary_solution$status,
          "calibrated"
        )) {
          "calibrated_interval_boundary"
        } else {
          boundary_solution$status
        },
        weight = weight,
        exact_lambda = if (boundary_minimum) -Inf else Inf,
        interval_lambda = boundary_solution$interval_lambda,
        upper_multiplier = boundary_solution$upper_multiplier,
        lower_multiplier = boundary_solution$lower_multiplier,
        lp_status = feasibility$status,
        convergence = boundary_solution$convergence,
        objective = boundary_solution$objective,
        exact_max_residual = if (all(is.na(weight))) {
          NA_real_
        } else {
          max(abs(
            crossprod(exact_design, weight) - exact_target
          ))
        },
        interval_max_violation =
          boundary_solution$interval_max_violation
      ))
    }
  }

  solution <- solve_interval_entropy_dual(
    prior_weight = prior_weight,
    exact_design = exact_design,
    exact_target = exact_target,
    interval_design = interval_design,
    interval_lower = interval_lower,
    interval_upper = interval_upper
  )
  exact_residual <- if (ncol(exact_design) == 0L) {
    0
  } else {
    max(abs(
      crossprod(exact_design, solution$weight) - exact_target
    ))
  }
  interval_moment <- as.vector(crossprod(
    interval_design,
    solution$weight
  ))
  interval_violation <- max(
    0,
    interval_lower - interval_moment,
    interval_moment - interval_upper
  )

  # L-BFGS-B can stop a few machine-precision units outside a binding band.
  # Once the active side is unambiguous, solve the corresponding equality
  # projection and verify every inactive band before accepting it.
  active_lower <- solution$lower_multiplier >
    solution$upper_multiplier &
    solution$lower_multiplier > 1e-10
  active_upper <- solution$upper_multiplier >
    solution$lower_multiplier &
    solution$upper_multiplier > 1e-10
  active_interval <- active_lower | active_upper
  if (
    any(active_interval) &&
      (
        solution$convergence != 0L ||
          interval_violation > tolerance / 2
      )
  ) {
    active_target <- ifelse(
      active_lower[active_interval],
      interval_lower[active_interval],
      interval_upper[active_interval]
    )
    refined <- solve_exact_entropy(
      prior_weight = prior_weight,
      design = cbind(
        exact_design,
        interval_design[, active_interval, drop = FALSE]
      ),
      target = c(exact_target, active_target),
      tolerance = tolerance / 10
    )
    if (startsWith(refined$status, "calibrated")) {
      refined_interval_moment <- as.vector(crossprod(
        interval_design,
        refined$weight
      ))
      refined_interval_violation <- max(
        0,
        interval_lower - refined_interval_moment,
        refined_interval_moment - interval_upper
      )
      refined_exact_residual <- if (ncol(exact_design) == 0L) {
        0
      } else {
        max(abs(
          crossprod(exact_design, refined$weight) - exact_target
        ))
      }
      if (
        refined_exact_residual <= tolerance &&
          refined_interval_violation <= tolerance
      ) {
        solution$weight <- refined$weight
        solution$exact_lambda <- head(
          refined$lambda,
          ncol(exact_design)
        )
        solution$interval_lambda[] <- 0
        solution$interval_lambda[active_interval] <- tail(
          refined$lambda,
          sum(active_interval)
        )
        solution$convergence <- refined$convergence
        solution$objective <- entropy_kl_divergence(
          refined$weight,
          prior_weight
        )
        exact_residual <- refined_exact_residual
        interval_moment <- refined_interval_moment
        interval_violation <- refined_interval_violation
      }
    }
  }

  valid <- validate_entropy_weight(solution$weight, tolerance) &&
    exact_residual <= tolerance &&
    interval_violation <= tolerance

  list(
    status = if (valid) {
      "calibrated_interval"
    } else {
      "optimizer_failed_interval"
    },
    weight = if (valid) {
      solution$weight
    } else {
      rep(NA_real_, length(prior_weight))
    },
    exact_lambda = solution$exact_lambda,
    interval_lambda = solution$interval_lambda,
    upper_multiplier = solution$upper_multiplier,
    lower_multiplier = solution$lower_multiplier,
    lp_status = feasibility$status,
    convergence = solution$convergence,
    objective = solution$objective,
    exact_max_residual = exact_residual,
    interval_max_violation = interval_violation
  )
}

solve_soft_entropy <- function(
  prior_weight,
  exact_design,
  exact_target,
  soft_design,
  soft_target,
  soft_penalty,
  tolerance = 1e-8
) {
  prior_weight <- normalize_entropy_prior(prior_weight)
  exact_design <- as.matrix(exact_design)
  soft_design <- as.matrix(soft_design)
  feasibility <- calibration_lp(
    observation_count = length(prior_weight),
    exact_design = exact_design,
    exact_target = exact_target
  )
  if (feasibility$status != 0L) {
    return(list(
      status = "exact_infeasible",
      weight = rep(NA_real_, length(prior_weight)),
      exact_lambda = rep(NA_real_, ncol(exact_design)),
      soft_lambda = rep(NA_real_, ncol(soft_design)),
      lp_status = feasibility$status,
      convergence = NA_integer_,
      objective = NA_real_,
      exact_max_residual = NA_real_
    ))
  }
  if (
    length(soft_penalty) != 1L ||
      !is.finite(soft_penalty) ||
      soft_penalty <= 0
  ) {
    stop("`soft_penalty` (rho) must be finite and positive.", call. = FALSE)
  }

  if (ncol(exact_design) == 0L) {
    solution <- solve_entropy_dual(
      prior_weight = prior_weight,
      design = soft_design,
      target = soft_target,
      inverse_penalty = rep(1 / soft_penalty, length(soft_target)),
      initial_lambda = rep(0, length(soft_target))
    )
    valid <- validate_entropy_weight(solution$weight, tolerance)
    return(list(
      status = if (valid) {
        "calibrated_soft"
      } else {
        "optimizer_failed_soft"
      },
      weight = if (valid) {
        solution$weight
      } else {
        rep(NA_real_, length(prior_weight))
      },
      exact_lambda = numeric(),
      soft_lambda = solution$lambda,
      lp_status = feasibility$status,
      convergence = solution$convergence,
      objective = solution$objective,
      exact_max_residual = 0
    ))
  }
  if (ncol(exact_design) != 1L) {
    stop(
      "Soft entropy calibration supports zero or one exact moment.",
      call. = FALSE
    )
  }

  exact_value <- exact_design[, 1]
  boundary_minimum <- abs(
    exact_target[[1]] - min(exact_value)
  ) <= tolerance
  boundary_maximum <- abs(
    exact_target[[1]] - max(exact_value)
  ) <= tolerance
  if (boundary_minimum || boundary_maximum) {
    keep <- if (boundary_minimum) {
      abs(exact_value - min(exact_value)) <= tolerance
    } else {
      abs(exact_value - max(exact_value)) <= tolerance
    }
    boundary_prior <- normalize_entropy_prior(prior_weight[keep])
    boundary_solution <- solve_entropy_dual(
      prior_weight = boundary_prior,
      design = soft_design[keep, , drop = FALSE],
      target = soft_target,
      inverse_penalty = rep(
        1 / soft_penalty,
        length(soft_target)
      ),
      initial_lambda = rep(0, length(soft_target))
    )
    weight <- rep(0, length(prior_weight))
    weight[keep] <- boundary_solution$weight
    exact_residual <- max(abs(
      crossprod(exact_design, weight) - exact_target
    ))
    valid <- validate_entropy_weight(weight, tolerance) &&
      exact_residual <= tolerance
    return(list(
      status = if (valid) {
        "calibrated_soft_boundary"
      } else {
        "optimizer_failed_soft"
      },
      weight = if (valid) {
        weight
      } else {
        rep(NA_real_, length(prior_weight))
      },
      exact_lambda = if (boundary_minimum) -Inf else Inf,
      soft_lambda = boundary_solution$lambda,
      lp_status = feasibility$status,
      convergence = boundary_solution$convergence,
      objective = boundary_solution$objective,
      exact_max_residual = exact_residual
    ))
  }

  # The implemented specifications have one exact wage moment. Profiling its
  # multiplier makes that moment exact at every soft-dual iteration.
  solution <- solve_profiled_entropy_dual(
    prior_weight = prior_weight,
    wage = exact_design[, 1],
    target_wage = exact_target[[1]],
    soft_design = soft_design,
    soft_target = soft_target,
    soft_penalty = soft_penalty,
    initial_soft_lambda = rep(0, ncol(soft_design))
  )
  exact_residual <- max(abs(
    crossprod(exact_design, solution$weight) - exact_target
  ))
  valid <- validate_entropy_weight(solution$weight, tolerance) &&
    exact_residual <= tolerance
  list(
    status = if (valid) {
      "calibrated_soft"
    } else {
      "optimizer_failed_soft"
    },
    weight = if (valid) {
      solution$weight
    } else {
      rep(NA_real_, length(prior_weight))
    },
    exact_lambda = solution$wage_lambda,
    soft_lambda = solution$soft_lambda,
    lp_status = feasibility$status,
    convergence = solution$convergence,
    objective = solution$objective,
    exact_max_residual = exact_residual
  )
}

initialize_generic_calibration_columns <- function(data) {
  data$oews_area_weight_entropy_calibrated <- NA_real_
  # Retain the historical column name so existing consumers can migrate
  # without treating an exact solution as a different weight object.
  data$oews_area_weight_soft_calibrated <- NA_real_
  data$calibration_status <- "not_attempted"
  data$requested_moment_spec <- NA_character_
  data$resolved_moment_spec <- NA_character_
  data$optimizer_convergence <- NA_integer_
  data$optimizer_dual_objective <- NA_real_
  data$lp_feasibility_status <- NA_integer_
  data$exact_max_abs_residual <- NA_real_
  data$interval_max_violation <- NA_real_
  data$minimum_active_observed_prior_mass <- NA_real_
  data$entropy_lambda_wage <- NA_real_
  common_labels <- c(
    "duration",
    "seasonal_january",
    "seasonal_april",
    "seasonal_july"
  )
  for (label in common_labels) {
    data[[paste0("calibration_feature_", label)]] <- NA_real_
    data[[paste0(label, "_feature_prior_mean")]] <- NA_real_
    data[[paste0(label, "_feature_scale")]] <- NA_real_
    data[[paste0(label, "_feature_observed_prior_mass")]] <- NA_real_
    data[[paste0(label, "_target")]] <- NA_real_
    data[[paste0("entropy_lambda_", label)]] <- NA_real_
    data[[paste0("calibrated_", label, "_moment")]] <- NA_real_
    data[[paste0(label, "_standardized_imbalance")]] <- NA_real_
    data[[paste0(label, "_interval_half_width")]] <- NA_real_
    data[[paste0(label, "_interval_slack")]] <- NA_real_
  }
  data
}

calibrate_entropy_cell <- function(
  data,
  entropy_target_wage,
  moment_spec,
  calibration_mode,
  feature_names = character(),
  target_names = character(),
  feature_labels = character(),
  feature_observed_share_names = rep(NA_character_, length(feature_names)),
  soft_penalty = NA_real_,
  interval_half_widths = numeric(),
  minimum_observed_prior_mass = 0.90,
  prior_column = "oews_area_prior_weight",
  wage_column = "oews_area_mean_hourly_wage",
  tolerance = 1e-8,
  minimum_scale = 1e-8
) {
  data <- initialize_generic_calibration_columns(data)
  specification <- resolve_entropy_moment_spec(moment_spec)
  data$requested_moment_spec <- specification$requested
  data$resolved_moment_spec <- specification$resolved

  if (specification$defaulted) {
    calibration_mode <- "exact"
    feature_names <- character()
    target_names <- character()
    feature_labels <- character()
    feature_observed_share_names <- character()
    interval_half_widths <- numeric()
  }

  if (!calibration_mode %in% c("exact", "interval", "soft")) {
    stop("Unknown `calibration_mode`: ", calibration_mode, call. = FALSE)
  }
  if (
    length(feature_names) != length(target_names) ||
      length(feature_names) != length(feature_labels)
  ) {
    stop("Feature, target, and label vectors must have equal length.", call. = FALSE)
  }
  if (length(feature_observed_share_names) != length(feature_names)) {
    stop("One observed-share column is required per feature.", call. = FALSE)
  }

  prior_weight <- normalize_entropy_prior(data[[prior_column]])
  wage <- as.numeric(data[[wage_column]])
  if (
    length(entropy_target_wage) != 1L ||
      !is.finite(entropy_target_wage)
  ) {
    data$calibration_status <- "missing_wage_target"
    return(data)
  }
  if (
    entropy_target_wage < min(wage) - tolerance ||
      entropy_target_wage > max(wage) + tolerance
  ) {
    data$calibration_status <- "outside_wage_support"
    return(data)
  }

  wage_center <- sum(prior_weight * wage)
  wage_scale <- sqrt(sum(prior_weight * (wage - wage_center)^2))
  if (wage_scale <= minimum_scale) {
    if (abs(entropy_target_wage - wage_center) <= tolerance) {
      wage_design <- matrix(0, nrow = nrow(data), ncol = 0L)
      wage_target <- numeric()
    } else {
      data$calibration_status <- "no_wage_variation"
      return(data)
    }
  } else {
    wage_design <- matrix(
      (wage - wage_center) / wage_scale,
      ncol = 1L,
      dimnames = list(NULL, "wage")
    )
    wage_target <- (entropy_target_wage - wage_center) / wage_scale
  }

  standardized_features <- list()
  standardized_targets <- numeric()
  observed_masses <- numeric()
  feature_scales <- numeric()
  for (index in seq_along(feature_names)) {
    feature_name <- feature_names[[index]]
    target_name <- target_names[[index]]
    label <- feature_labels[[index]]
    feature <- as.numeric(data[[feature_name]])
    target_value <- as.numeric(data[[target_name]][[1]])
    observed <- is.finite(feature)

    observed_share_name <- feature_observed_share_names[[index]]
    if (
      is.na(observed_share_name) ||
        !nzchar(observed_share_name) ||
        !observed_share_name %in% names(data)
    ) {
      observed_share <- as.numeric(observed)
    } else {
      observed_share <- pmin(
        1,
        pmax(0, as.numeric(data[[observed_share_name]]))
      )
      observed_share[!is.finite(observed_share) | !observed] <- 0
    }
    observed_mass <- sum(prior_weight * observed_share)
    observed_masses[[label]] <- observed_mass
    data[[paste0(label, "_feature_observed_prior_mass")]] <-
      observed_mass

    if (
      !is.finite(target_value) ||
        observed_mass < minimum_observed_prior_mass ||
        !any(observed)
    ) {
      data$minimum_active_observed_prior_mass <- if (
        length(observed_masses) > 0L
      ) {
        min(observed_masses)
      } else {
        NA_real_
      }
      data$calibration_status <- "insufficient_auxiliary_coverage"
      return(data)
    }

    center_weight <- prior_weight * observed_share
    center_weight <- center_weight / sum(center_weight)
    center <- sum(center_weight[observed] * feature[observed])
    imputed_feature <- feature
    imputed_feature[!observed] <- center
    scale <- sqrt(sum(prior_weight * (imputed_feature - center)^2))

    data[[paste0("calibration_feature_", label)]] <- imputed_feature
    data[[paste0(label, "_feature_prior_mean")]] <- center
    data[[paste0(label, "_feature_scale")]] <- scale
    data[[paste0(label, "_target")]] <- target_value

    if (!is.finite(scale) || scale <= minimum_scale) {
      data$minimum_active_observed_prior_mass <- min(observed_masses)
      data$calibration_status <- "constant_auxiliary_moment"
      return(data)
    }
    standardized_features[[label]] <- (imputed_feature - center) / scale
    standardized_targets[[label]] <- (target_value - center) / scale
    feature_scales[[label]] <- scale
  }
  data$minimum_active_observed_prior_mass <- if (
    length(observed_masses) > 0L
  ) {
    min(observed_masses)
  } else {
    1
  }

  feature_design <- if (length(standardized_features) == 0L) {
    matrix(numeric(), nrow = nrow(data), ncol = 0L)
  } else {
    do.call(cbind, standardized_features)
  }
  feature_target <- unname(standardized_targets)
  raw_moment_scales <- c(
    if (ncol(wage_design) == 1L) wage_scale else numeric(),
    unname(feature_scales)
  )
  solver_tolerance <- tolerance / max(c(1, raw_moment_scales))

  if (calibration_mode == "exact") {
    exact_design <- cbind(wage_design, feature_design)
    exact_target <- c(wage_target, feature_target)
    solution <- solve_exact_entropy(
      prior_weight = prior_weight,
      design = exact_design,
      target = exact_target,
      tolerance = solver_tolerance
    )
    moment_lambda <- if (length(feature_names) > 0L) {
      tail(solution$lambda, length(feature_names))
    } else {
      numeric()
    }
    wage_lambda <- if (ncol(wage_design) == 1L) {
      solution$lambda[[1]]
    } else {
      0
    }
    exact_residual <- solution$max_residual
    interval_violation <- NA_real_
  } else if (calibration_mode == "interval") {
    if (length(interval_half_widths) != length(feature_names)) {
      stop("Interval calibration needs one band per feature.", call. = FALSE)
    }
    if (
      any(!is.finite(interval_half_widths)) ||
        any(interval_half_widths < 0)
    ) {
      data$calibration_status <- "missing_interval_band"
      return(data)
    }
    solution <- solve_interval_entropy(
      prior_weight = prior_weight,
      exact_design = wage_design,
      exact_target = wage_target,
      interval_design = feature_design,
      interval_lower = feature_target - interval_half_widths,
      interval_upper = feature_target + interval_half_widths,
      tolerance = solver_tolerance
    )
    moment_lambda <- solution$interval_lambda
    wage_lambda <- if (ncol(wage_design) == 1L) {
      solution$exact_lambda[[1]]
    } else {
      0
    }
    exact_residual <- solution$exact_max_residual
    interval_violation <- solution$interval_max_violation
  } else {
    solution <- solve_soft_entropy(
      prior_weight = prior_weight,
      exact_design = wage_design,
      exact_target = wage_target,
      soft_design = feature_design,
      soft_target = feature_target,
      soft_penalty = soft_penalty,
      tolerance = solver_tolerance
    )
    moment_lambda <- solution$soft_lambda
    wage_lambda <- if (ncol(wage_design) == 1L) {
      solution$exact_lambda[[1]]
    } else {
      0
    }
    exact_residual <- solution$exact_max_residual
    interval_violation <- NA_real_
  }

  data$calibration_status <- solution$status
  data$optimizer_convergence <- solution$convergence
  data$optimizer_dual_objective <- solution$objective
  data$lp_feasibility_status <- solution$lp_status
  data$exact_max_abs_residual <- exact_residual
  data$interval_max_violation <- interval_violation
  data$entropy_lambda_wage <- wage_lambda
  if (startsWith(solution$status, "calibrated")) {
    data$oews_area_weight_entropy_calibrated <- solution$weight
    data$oews_area_weight_soft_calibrated <- solution$weight
  }

  for (index in seq_along(feature_labels)) {
    label <- feature_labels[[index]]
    data[[paste0("entropy_lambda_", label)]] <- moment_lambda[[index]]
    if (!all(is.na(solution$weight))) {
      calibrated_moment <- sum(
        solution$weight * data[[paste0("calibration_feature_", label)]]
      )
      target_value <- data[[paste0(label, "_target")]][[1]]
      scale <- data[[paste0(label, "_feature_scale")]][[1]]
      standardized_imbalance <- (
        calibrated_moment - target_value
      ) / scale
    } else {
      calibrated_moment <- NA_real_
      standardized_imbalance <- NA_real_
    }
    data[[paste0("calibrated_", label, "_moment")]] <-
      calibrated_moment
    data[[paste0(label, "_standardized_imbalance")]] <-
      standardized_imbalance
    if (calibration_mode == "interval") {
      data[[paste0(label, "_interval_half_width")]] <-
        interval_half_widths[[index]]
      data[[paste0(label, "_interval_slack")]] <-
        interval_half_widths[[index]] - abs(standardized_imbalance)
    }
  }
  data
}
