# Plain helpers for the Kitagawa instrument-decomposition diagnostic.
#
# For a target cluster g and source year t with balanced donor weights p_ct
# (summing to one) and donor wages w_ct, fix reference values p_bar_c (the
# time mean of p_ct) and w_bar_c (the predetermined base wage). Then exactly
#   Z_gt = sum p_bar w_bar + sum (p - p_bar) w_bar
#        + sum p_bar (w - w_bar) + sum (p - p_bar)(w - w_bar).
# The weight term does not depend on p_bar apart from a cluster constant.
# Callers are responsible for loading dplyr.

IV_DECOMPOSITION_BASE_WAGE_YEARS <- 2008:2010
IV_DECOMPOSITION_VERSION <- "panel_iv_kitagawa_decomposition_v1"
IV_DECOMPOSITION_IDENTITY_TOLERANCE <- 1e-9

iv_kitagawa_components <- function(data, group_columns) {
  required <- c(
    group_columns,
    "county_fips",
    "source_year",
    "decomposition_weight",
    "donor_real_hourly_wage",
    "base_real_hourly_wage"
  )
  missing <- setdiff(required, names(data))
  if (length(missing)) {
    stop("Decomposition input missing: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  values <- data[c("decomposition_weight", "donor_real_hourly_wage", "base_real_hourly_wage")]
  if (!all(vapply(values, function(value) all(is.finite(value)), logical(1)))) {
    stop("Decomposition weights and wages must be finite.", call. = FALSE)
  }
  balance <- data %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(group_columns))) %>%
    dplyr::summarise(
      cells = dplyr::n(),
      counties = dplyr::n_distinct(.data$county_fips),
      years = dplyr::n_distinct(.data$source_year),
      .groups = "drop"
    )
  if (any(balance$cells != balance$counties * balance$years)) {
    stop("Decomposition donor sets must be balanced across source years.", call. = FALSE)
  }
  weight_sums <- data %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(c(group_columns, "source_year")))) %>%
    dplyr::summarise(weight_sum = sum(.data$decomposition_weight), .groups = "drop")
  if (any(abs(weight_sums$weight_sum - 1) > IV_DECOMPOSITION_IDENTITY_TOLERANCE)) {
    stop("Decomposition weights must sum to one within each group-year.", call. = FALSE)
  }

  components <- data %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(c(group_columns, "county_fips")))) %>%
    dplyr::mutate(reference_weight = mean(.data$decomposition_weight)) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(
      weight_deviation = .data$decomposition_weight - .data$reference_weight,
      wage_deviation = .data$donor_real_hourly_wage - .data$base_real_hourly_wage
    ) %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(c(group_columns, "source_year")))) %>%
    dplyr::summarise(
      z_balanced = sum(.data$decomposition_weight * .data$donor_real_hourly_wage),
      z_constant = sum(.data$reference_weight * .data$base_real_hourly_wage),
      z_weight = sum(.data$weight_deviation * .data$base_real_hourly_wage),
      z_wage = sum(.data$reference_weight * .data$wage_deviation),
      z_interaction = sum(.data$weight_deviation * .data$wage_deviation),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      z_weight_shapley = .data$z_weight + .data$z_interaction / 2,
      z_wage_shapley = .data$z_wage + .data$z_interaction / 2,
      identity_residual = .data$z_balanced -
        (.data$z_constant + .data$z_weight + .data$z_wage + .data$z_interaction)
    )
  if (any(abs(components$identity_residual) >
    IV_DECOMPOSITION_IDENTITY_TOLERANCE * pmax(1, abs(components$z_balanced)))) {
    stop("Kitagawa components do not add up to the balanced instrument.", call. = FALSE)
  }
  components
}

# Anderson-Rubin confidence set for a just-identified model. The AR
# statistic for beta0 is (pi_y - beta0 pi_x)^2 / Var(pi_y - beta0 pi_x), where
# pi_y and pi_x are reduced-form and first-stage coefficients on the instrument
# and the variance is quadratic in beta0.
iv_ar_confidence_set <- function(pi_y, pi_x, var_y, var_x, cov_yx, critical_value) {
  a <- pi_x^2 - critical_value * var_x
  b <- pi_y * pi_x - critical_value * cov_yx
  c <- pi_y^2 - critical_value * var_y
  discriminant <- b^2 - a * c
  if (!all(is.finite(c(a, b, c)))) {
    return(list(lower = NA_real_, upper = NA_real_, type = "undefined"))
  }
  if (a > 0) {
    root <- sqrt(max(discriminant, 0))
    return(list(lower = (b - root) / a, upper = (b + root) / a, type = "bounded"))
  }
  if (a < 0 && discriminant > 0) {
    root <- sqrt(discriminant)
    roots <- sort(c((b - root) / a, (b + root) / a))
    return(list(lower = roots[[1]], upper = roots[[2]], type = "complement_of_interval"))
  }
  list(lower = -Inf, upper = Inf, type = "unbounded")
}

# Cluster-robust two-step GMM and Hansen J on data already partialled of
# fixed effects and exogenous controls. Returns the first-step 2SLS estimate,
# which equals the fixest 2SLS coefficient on the same partialled data.
iv_cluster_hansen_j <- function(y, x, z, cluster) {
  z <- as.matrix(z)
  x <- as.matrix(x)
  if (nrow(z) != length(y) || nrow(x) != length(y) || length(cluster) != length(y)) {
    stop("Hansen J inputs must have matching rows.", call. = FALSE)
  }
  overidentification <- ncol(z) - ncol(x)
  if (overidentification < 1L) {
    stop("Hansen J requires more instruments than endogenous regressors.", call. = FALSE)
  }
  zx <- crossprod(z, x)
  zy <- crossprod(z, y)
  zz_inverse <- solve(crossprod(z))
  beta_2sls <- solve(t(zx) %*% zz_inverse %*% zx, t(zx) %*% zz_inverse %*% zy)
  residual_2sls <- as.vector(y - x %*% beta_2sls)
  scores <- rowsum(z * residual_2sls, cluster, reorder = FALSE)
  weight <- solve(crossprod(scores))
  beta_gmm <- solve(t(zx) %*% weight %*% zx, t(zx) %*% weight %*% zy)
  moment <- crossprod(z, as.vector(y - x %*% beta_gmm))
  statistic <- as.numeric(t(moment) %*% weight %*% moment)
  list(
    beta_2sls = as.numeric(beta_2sls),
    beta_gmm = as.numeric(beta_gmm),
    statistic = statistic,
    df = overidentification,
    p_value = stats::pchisq(statistic, overidentification, lower.tail = FALSE)
  )
}
