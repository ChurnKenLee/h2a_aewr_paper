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
    control = list(maxit = 1000, reltol = 1e-12)
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

