#!/usr/bin/env Rscript

here::i_am("code/paths.R")

# Standalone numerical checks; run from the repository root after renv restore.
stopifnot(requireNamespace("lpSolve", quietly = TRUE))
source(here::here("code", "c00_shared", "entropy_calibration.R"))

assert_close <- function(actual, expected, tolerance = 1e-8) {
  stopifnot(
    length(actual) == length(expected),
    max(abs(actual - expected)) <= tolerance
  )
}

assert_valid_weight <- function(weight) {
  stopifnot(
    all(is.finite(weight)),
    all(weight >= 0),
    abs(sum(weight) - 1) <= 1e-8
  )
}

# Interior exact solution.
prior <- c(0.2, 0.3, 0.1, 0.4)
exact_design <- cbind(
  wage = c(-1, -0.2, 0.4, 1),
  january = c(0.1, 0.7, 0.2, 0.4)
)
exact_target <- c(0.15, 0.35)
exact_fit <- solve_exact_entropy(
  prior,
  exact_design,
  exact_target
)
stopifnot(exact_fit$status == "calibrated_exact")
assert_valid_weight(exact_fit$weight)
assert_close(
  as.vector(crossprod(exact_design, exact_fit$weight)),
  exact_target
)

# Infeasible exact cells have no weights and are not softened.
infeasible_fit <- solve_exact_entropy(
  prior,
  exact_design,
  c(3, 0.35)
)
stopifnot(
  infeasible_fit$status == "exact_infeasible",
  all(is.na(infeasible_fit$weight))
)

# Boundary targets retain the KL projection on the feasible face.
boundary_design <- matrix(c(0, 0, 1), ncol = 1)
boundary_fit <- solve_exact_entropy(
  c(0.2, 0.3, 0.5),
  boundary_design,
  0
)
stopifnot(boundary_fit$status == "calibrated_exact_boundary")
assert_close(boundary_fit$weight, c(0.4, 0.6, 0))

# Interval solution keeps wage exact and the auxiliary moment inside its band.
interval_fit <- solve_interval_entropy(
  prior_weight = prior,
  exact_design = exact_design[, "wage", drop = FALSE],
  exact_target = 0.15,
  interval_design = exact_design[, "january", drop = FALSE],
  interval_lower = 0.30,
  interval_upper = 0.40
)
stopifnot(interval_fit$status == "calibrated_interval")
assert_valid_weight(interval_fit$weight)
assert_close(
  crossprod(
    exact_design[, "wage", drop = FALSE],
    interval_fit$weight
  ),
  0.15
)
interval_moment <- as.numeric(crossprod(
  exact_design[, "january", drop = FALSE],
  interval_fit$weight
))
stopifnot(interval_moment >= 0.30 - 1e-8)
stopifnot(interval_moment <= 0.40 + 1e-8)

# Interval calibration also retains an exact target on a convex-hull face.
boundary_interval_fit <- solve_interval_entropy(
  prior_weight = c(0.2, 0.3, 0.5),
  exact_design = boundary_design,
  exact_target = 0,
  interval_design = matrix(c(0.2, 0.4, 0.8), ncol = 1),
  interval_lower = 0.25,
  interval_upper = 0.35
)
stopifnot(
  boundary_interval_fit$status == "calibrated_interval_boundary"
)
assert_valid_weight(boundary_interval_fit$weight)
assert_close(
  crossprod(boundary_design, boundary_interval_fit$weight),
  0
)
boundary_interval_moment <- as.numeric(crossprod(
  matrix(c(0.2, 0.4, 0.8), ncol = 1),
  boundary_interval_fit$weight
))
stopifnot(boundary_interval_moment >= 0.25 - 1e-8)
stopifnot(boundary_interval_moment <= 0.35 + 1e-8)

infeasible_interval_fit <- solve_interval_entropy(
  prior_weight = prior,
  exact_design = exact_design[, "wage", drop = FALSE],
  exact_target = 0.15,
  interval_design = exact_design[, "january", drop = FALSE],
  interval_lower = 2,
  interval_upper = 3
)
stopifnot(
  infeasible_interval_fit$status == "interval_infeasible",
  all(is.na(infeasible_interval_fit$weight))
)

# Low rho is a sensitivity, while the wage moment remains exact.
soft_fit <- solve_soft_entropy(
  prior_weight = prior,
  exact_design = exact_design[, "wage", drop = FALSE],
  exact_target = 0.15,
  soft_design = exact_design[, "january", drop = FALSE],
  soft_target = 0.35,
  soft_penalty = 0.01
)
stopifnot(soft_fit$status == "calibrated_soft")
assert_valid_weight(soft_fit$weight)
assert_close(
  crossprod(
    exact_design[, "wage", drop = FALSE],
    soft_fit$weight
  ),
  0.15
)
wage_only_fit <- solve_exact_entropy(
  prior,
  exact_design[, "wage", drop = FALSE],
  0.15
)
rho_one_fit <- solve_soft_entropy(
  prior_weight = prior,
  exact_design = exact_design[, "wage", drop = FALSE],
  exact_target = 0.15,
  soft_design = exact_design[, "january", drop = FALSE],
  soft_target = 0.35,
  soft_penalty = 1
)
stopifnot(
  sum(abs(soft_fit$weight - wage_only_fit$weight)) <
    sum(abs(rho_one_fit$weight - wage_only_fit$weight))
)

# A constant wage that already equals its target is a redundant exact moment.
constant_wage_cell <- data.frame(
  oews_area_prior_weight = c(0.2, 0.3, 0.5),
  oews_area_mean_hourly_wage = rep(15, 3),
  seasonal_feature = c(0.1, 0.4, 0.8),
  seasonal_target = 0.45
)
constant_soft_fit <- calibrate_entropy_cell(
  data = constant_wage_cell,
  entropy_target_wage = 15,
  moment_spec = "wage_seasonal_soft_rho010",
  calibration_mode = "soft",
  feature_names = "seasonal_feature",
  target_names = "seasonal_target",
  feature_labels = "seasonal_january",
  soft_penalty = 0.1
)
stopifnot(
  unique(constant_soft_fit$calibration_status) == "calibrated_soft"
)
assert_valid_weight(
  constant_soft_fit$oews_area_weight_entropy_calibrated
)

# Unknown specifications resolve to the wage-only exact estimator.
cell <- data.frame(
  oews_area_prior_weight = prior,
  oews_area_mean_hourly_wage = c(10, 14, 18, 25),
  seasonal_feature = c(0.1, 0.3, 0.6, 0.8),
  seasonal_target = 0.4
)
unknown_fit <- suppressWarnings(calibrate_entropy_cell(
  data = cell,
  entropy_target_wage = 17,
  moment_spec = "not_a_spec",
  calibration_mode = "soft",
  feature_names = "seasonal_feature",
  target_names = "seasonal_target",
  feature_labels = "seasonal_january",
  soft_penalty = 1
))
stopifnot(
  unique(unknown_fit$requested_moment_spec) == "not_a_spec",
  unique(unknown_fit$resolved_moment_spec) == "wage_only_exact",
  unique(unknown_fit$calibration_status) == "calibrated_exact"
)
assert_valid_weight(unknown_fit$oews_area_weight_entropy_calibrated)
assert_close(
  sum(
    unknown_fit$oews_area_weight_entropy_calibrated *
      unknown_fit$oews_area_mean_hourly_wage
  ),
  17
)
legacy_wage_only <- calibrate_wage_cell(cell, 17)
assert_close(
  unknown_fit$oews_area_weight_entropy_calibrated,
  legacy_wage_only$oews_area_weight_wage_calibrated
)

# The 90-percent coverage rule is enforced before calibration.
cell$observed_share <- c(1, 1, 0.5, 0)
coverage_fit <- calibrate_entropy_cell(
  data = cell,
  entropy_target_wage = 17,
  moment_spec = "wage_seasonal_exact",
  calibration_mode = "exact",
  feature_names = "seasonal_feature",
  target_names = "seasonal_target",
  feature_labels = "seasonal_january",
  feature_observed_share_names = "observed_share",
  minimum_observed_prior_mass = 0.90
)
stopifnot(
  unique(coverage_fit$calibration_status) ==
    "insufficient_auxiliary_coverage",
  all(is.na(coverage_fit$oews_area_weight_entropy_calibrated))
)

cat("entropy calibration tests passed\n")
