# Numerical, construction, formula, and postestimation helpers for the
# multilevel Mundlak-Chamberlain-Wooldridge branch.  Callers load packages.

mc_finite_mean <- function(value) {
  keep <- is.finite(value)
  if (!any(keep)) {
    return(NA_real_)
  }
  mean(value[keep])
}

mc_finite_sd <- function(value) {
  keep <- is.finite(value)
  if (sum(keep) < 2L) {
    return(NA_real_)
  }
  stats::sd(value[keep])
}

mc_linear_slope <- function(value, year) {
  keep <- is.finite(value) & is.finite(year)
  if (sum(keep) < 2L) {
    return(NA_real_)
  }
  centered_year <- year[keep] - mean(year[keep])
  denominator <- sum(centered_year^2)
  if (!is.finite(denominator) || denominator <= 0) {
    return(NA_real_)
  }
  sum(centered_year * value[keep]) / denominator
}

mc_safe_standardize <- function(value) {
  center <- mc_finite_mean(value)
  scale <- mc_finite_sd(value)
  if (!is.finite(scale) || scale <= sqrt(.Machine$double.eps)) {
    return(list(
      value = rep(0, length(value)),
      center = center,
      scale = NA_real_
    ))
  }
  list(
    value = (value - center) / scale,
    center = center,
    scale = scale
  )
}

mc_transform_baseline <- function(value, variable_name, emp_farm) {
  transformed <- switch(
    variable_name,
    h2a_cert_intensity = asinh(value),
    h2a_application_intensity = asinh(value),
    aewr_bite = value,
    log_population = value,
    farm_employment_share = value,
    employment_population_ratio = value,
    crop_income_share = value,
    animal_income_share = value,
    hired_labor_cost_share = value,
    low_wage = ifelse(value > 0, log(value), NA_real_),
    cropland = log1p(pmax(value, 0) / pmax(emp_farm, 1)),
    predicted_h2a_intensity = asinh(value),
    stop("Unknown baseline transform: ", variable_name, call. = FALSE)
  )
  ifelse(is.finite(transformed), transformed, NA_real_)
}

mc_impute_by_region <- function(data, column) {
  value <- data[[column]]
  missing <- !is.finite(value)
  region_median <- ave(
    value,
    data$aewr_region_id,
    FUN = function(x) {
      keep <- is.finite(x)
      if (!any(keep)) NA_real_ else stats::median(x[keep])
    }
  )
  national_median <- if (any(is.finite(value))) {
    stats::median(value[is.finite(value)])
  } else {
    0
  }
  replacement <- ifelse(
    is.finite(region_median),
    region_median,
    national_median
  )
  data[[column]] <- ifelse(missing, replacement, value)
  data[[paste0(column, "_missing")]] <- as.integer(missing)
  data
}

mc_hierarchical_components <- function(
  data,
  source_column,
  component_stub
) {
  value <- data[[source_column]]
  market_mean <- ave(value, data$mc_market_id, FUN = mc_finite_mean)
  state_mean <- ave(value, data$state_fips, FUN = mc_finite_mean)
  region_mean <- ave(value, data$aewr_region_id, FUN = mc_finite_mean)
  national_mean <- mc_finite_mean(value)

  raw_components <- list(
    county = value - market_mean,
    market = market_mean - state_mean,
    state = state_mean - region_mean,
    region = region_mean - national_mean
  )

  scaling_rows <- list()
  for (level in names(raw_components)) {
    column <- paste0("mc_", substr(level, 1, 1), "_", component_stub)
    standardized <- mc_safe_standardize(raw_components[[level]])
    data[[column]] <- standardized$value
    scaling_rows[[length(scaling_rows) + 1L]] <- data.frame(
      constructed_column = column,
      source_column = source_column,
      hierarchy_level = level,
      center = standardized$center,
      scale = standardized$scale,
      stringsAsFactors = FALSE
    )
  }

  list(
    data = data,
    scaling = do.call(rbind, scaling_rows),
    columns = vapply(
      names(raw_components),
      function(level) {
        paste0("mc_", substr(level, 1, 1), "_", component_stub)
      },
      character(1)
    )
  )
}

mc_formula_collapse <- function(terms) {
  terms <- unique(terms[nzchar(terms)])
  if (length(terms) == 0L) "1" else paste(terms, collapse = " + ")
}

mc_year_interactions <- function(terms, reference_year) {
  if (length(terms) == 0L) {
    return(character())
  }
  paste0(
    "i(year, ",
    terms,
    ", ref = ",
    as.integer(reference_year),
    ")"
  )
}

mc_treatment_interactions <- function(treatment, terms) {
  if (length(terms) == 0L) {
    return(character())
  }
  paste0(treatment, ":", terms)
}

mc_polynomial_basis_name <- function(treatment, degree) {
  if (degree == 1L) {
    return(treatment)
  }
  if (degree == 2L) {
    return(paste0(treatment, "_sq"))
  }
  if (degree == 3L) {
    return(paste0(treatment, "_cu"))
  }
  stop("The treatment basis supports powers one through three.", call. = FALSE)
}

mc_explicit_year_term <- function(year, ...) {
  variables <- c(...)
  paste0(
    "I((year == ",
    as.integer(year),
    ") * ",
    paste(variables, collapse = " * "),
    ")"
  )
}

mc_master_causal_terms <- function(
  metadata,
  include_lead = FALSE,
  polynomial_degrees = MC_MASTER_POLYNOMIAL_DEGREES
) {
  horizons <- unname(MC_DYNAMIC_HORIZONS)
  horizon_years <- rep(
    list(metadata$analysis_years),
    length(horizons)
  )
  if (include_lead) {
    horizons <- c(horizons, "mc_dose_lead1")
    horizon_years <- c(
      horizon_years,
      list(
        metadata$analysis_years[
          metadata$analysis_years < max(metadata$analysis_years)
        ]
      )
    )
  }

  terms <- character()
  for (index in seq_along(horizons)) {
    treatment <- horizons[[index]]
    for (degree in polynomial_degrees) {
      basis <- mc_polynomial_basis_name(treatment, degree)
      for (year in horizon_years[[index]]) {
        terms <- c(
          terms,
          mc_explicit_year_term(year, basis),
          mc_explicit_year_term(year, basis, MC_Z_COLUMN)
        )
      }
    }
  }
  terms
}

mc_master_history_terms <- function(
  metadata,
  include_lead = FALSE
) {
  history_map <- metadata$region_treatment_history_map
  terms <- character()
  for (
    outcome_year in setdiff(
      metadata$analysis_years,
      metadata$reference_year
    )
  ) {
    excluded_history_years <- outcome_year - MC_LAG_ORDERS
    if (include_lead) {
      excluded_history_years <- c(
        excluded_history_years,
        outcome_year + 1L
      )
    }
    allowed <- history_map[
      !history_map$history_year %in% excluded_history_years,
      ,
      drop = FALSE
    ]

    # Keep one degree of assignment-path rank in reserve for the CCV reference
    # states.  In the terminal outcome year, the reference-year (2013) history
    # coordinate is a nuisance projection term, not a focal current/lag dose.
    # Retaining it makes the observed assignment just full rank but makes every
    # nontrivial balanced path reassignment singular.  Omitting this single
    # nuisance coordinate gives all 17 finite-design states the same identified
    # coefficient basis, which the random-denominator Lean theorem requires.
    terminal_estimable_year <- max(metadata$analysis_years) -
      as.integer(include_lead)
    if (outcome_year == terminal_estimable_year) {
      allowed <- allowed[
        allowed$history_year != metadata$reference_year,
        ,
        drop = FALSE
      ]
    }
    for (history_column in allowed$constructed_column) {
      terms <- c(
        terms,
        mc_explicit_year_term(
          outcome_year,
          history_column
        ),
        mc_explicit_year_term(
          outcome_year,
          history_column,
          MC_Z_COLUMN
        )
      )
    }
  }
  terms
}

mc_master_baseline_trend_terms <- function(metadata) {
  terms <- character()
  trend_years <- setdiff(
    metadata$analysis_years,
    metadata$reference_year
  )
  for (baseline_term in metadata$master_baseline_trend_terms) {
    for (year in trend_years) {
      terms <- c(
        terms,
        mc_explicit_year_term(year, baseline_term),
        mc_explicit_year_term(
          year,
          baseline_term,
          MC_Z_COLUMN
        )
      )
    }
  }
  c(
    vapply(
      trend_years,
      mc_explicit_year_term,
      character(1),
      MC_Z_COLUMN
    ),
    terms
  )
}

mc_build_formula <- function(
  outcome,
  model_id,
  metadata,
  reference_year
) {
  if (identical(model_id, "twfe_benchmark")) {
    return(stats::as.formula(paste(
      outcome,
      "~ mc_dose_current + mc_dose_lag1 + mc_dose_lag2",
      "| county_fips + year"
    )))
  }

  base_terms <- c(
    "factor(year)",
    "factor(aewr_region_id)",
    "mc_dose_current",
    "mc_dose_lag1",
    "mc_dose_lag2"
  )

  if (identical(model_id, "mundlak_multilevel")) {
    # Region-level moderators are constant within the treatment-assignment
    # unit.  Interacting them with a region-level dose leaves the coefficient
    # basis dependent on which AEWR path is assigned to which region, so the
    # random-denominator CCV states do not share an identified parameterization.
    # The benchmark therefore uses county/market/state slope moderators; the
    # rich primary model retains its separately identified Z heterogeneity.
    mundlak_slope_terms <- metadata$mundlak_slope_terms[
      !grepl("^mc_r_", metadata$mundlak_slope_terms)
    ]
    dynamic_slope_terms <- metadata$dynamic_slope_terms[
      !grepl("^mc_r_", metadata$dynamic_slope_terms)
    ]
    terms <- c(
      base_terms,
      metadata$mundlak_intercept_terms,
      mc_year_interactions(
        metadata$mundlak_trend_terms,
        reference_year
      ),
      mc_treatment_interactions(
        "mc_dose_current",
        mundlak_slope_terms
      ),
      mc_treatment_interactions(
        "mc_dose_lag1",
        dynamic_slope_terms
      ),
      mc_treatment_interactions(
        "mc_dose_lag2",
        dynamic_slope_terms
      )
    )
  } else {
    include_lead <- identical(model_id, "chamberlain_lead_test")
    terms <- c(
      "factor(year)",
      "factor(aewr_region_id)",
      MC_Z_COLUMN,
      metadata$chamberlain_intercept_terms,
      mc_master_causal_terms(
        metadata,
        include_lead = include_lead
      ),
      mc_master_history_terms(
        metadata,
        include_lead = include_lead
      ),
      mc_master_baseline_trend_terms(metadata)
    )
  }

  stats::as.formula(paste(outcome, "~", mc_formula_collapse(terms)))
}

mc_refresh_treatment_basis <- function(data) {
  data$mc_dose_current_sq <- data$mc_dose_current^2 / 25
  data$mc_dose_current_cu <- data$mc_dose_current^3 / 125
  data$mc_dose_lag1_sq <- data$mc_dose_lag1^2 / 25
  data$mc_dose_lag1_cu <- data$mc_dose_lag1^3 / 125
  data$mc_dose_lag2_sq <- data$mc_dose_lag2^2 / 25
  data$mc_dose_lag2_cu <- data$mc_dose_lag2^3 / 125
  if ("mc_dose_lead1" %in% names(data)) {
    data$mc_dose_lead1_sq <- data$mc_dose_lead1^2 / 25
    data$mc_dose_lead1_cu <- data$mc_dose_lead1^3 / 125
  }
  data$mc_dose_current_x_lag1 <-
    data$mc_dose_current * data$mc_dose_lag1 / 25
  data$mc_dose_current_x_lag2 <-
    data$mc_dose_current * data$mc_dose_lag2 / 25
  data$mc_dose_lag1_x_lag2 <-
    data$mc_dose_lag1 * data$mc_dose_lag2 / 25
  data$mc_dose_current_x_lag1_x_lag2 <-
    data$mc_dose_current *
      data$mc_dose_lag1 *
      data$mc_dose_lag2 / 125
  data
}

# ---------------------------------------------------------------------------
# Continuous-treatment design-covariance CCV
# ---------------------------------------------------------------------------
#
# This block implements the random-denominator covariance-kernel result in
# `ccv_symlink.lean`, not the archival scalar convex combination of HC and
# cluster-robust variances.  The relevant Lean objects are:
#
#   * `FiniteDesign.normalizedTreatment`, which absorbs the state-specific OLS
#     denominator into the score regressor;
#   * `FiniteDesign.variance_betaHatError_eq_designVariance_normalizedTreatment`,
#     which takes the covariance over those normalized score regressors; and
#   * `DesignCovariance.dcCCV`, whose feasible quadratic form replaces the
#     fixed structural residual and design kernel by estimated inputs.
#
# In this application the 17 complete AEWR-region policy paths are the observed
# support of a finite reference design.  State s cyclically assigns path r+s to
# recipient region r.  Across the 17 equally likely states, every observed path
# is assigned to every recipient region exactly once.  County covariates,
# outcomes, region labels, and the fitted residual vector remain fixed.  This is
# a transparent conditional reference law; it is not a claim that the
# historical Department of Labor rule literally randomized the paths.

mc_ccv_assignment_columns <- function(data, metadata) {
  columns <- unique(c(
    unname(MC_DYNAMIC_HORIZONS),
    "mc_dose_lead1",
    metadata$region_treatment_history_map$constructed_column
  ))
  absent <- setdiff(columns, names(data))
  if (length(absent) > 0L) {
    stop(
      "CCV assignment columns are absent: ",
      paste(absent, collapse = ", "),
      call. = FALSE
    )
  }
  columns
}

mc_ccv_reference_state <- function(
  data,
  metadata,
  state_index
) {
  regions <- sort(unique(data$aewr_region_id))
  region_count <- length(regions)
  if (
    region_count != MC_CCV_REFERENCE_STATES ||
      !state_index %in% 0:(region_count - 1L)
  ) {
    stop(
      "CCV requires one cyclic state for each of the 17 AEWR regions.",
      call. = FALSE
    )
  }

  assignment_columns <- mc_ccv_assignment_columns(data, metadata)
  lookup_columns <- c(
    "aewr_region_id",
    "year",
    assignment_columns
  )
  lookup <- unique(data[, lookup_columns, drop = FALSE])
  lookup_key <- paste(
    lookup$aewr_region_id,
    lookup$year,
    sep = ":"
  )
  if (anyDuplicated(lookup_key) > 0L) {
    stop(
      "AEWR assignment variables are not unique within region-year cells.",
      call. = FALSE
    )
  }

  recipient_position <- match(data$aewr_region_id, regions)
  donor_position <- (
    recipient_position - 1L + as.integer(state_index)
  ) %% region_count + 1L
  donor_region <- regions[donor_position]
  donor_key <- paste(donor_region, data$year, sep = ":")
  donor_row <- match(donor_key, lookup_key)
  if (anyNA(donor_row)) {
    stop(
      "A CCV reference state lacks a donor region-year treatment cell.",
      call. = FALSE
    )
  }

  state_data <- data
  state_data[, assignment_columns] <-
    lookup[donor_row, assignment_columns, drop = FALSE]

  # Linear, quadratic, cubic, and cross-horizon columns must all describe the
  # reassigned path.  Recomputing them here prevents a hybrid state in which a
  # linear dose comes from the donor but a polynomial term comes from the
  # observed recipient path.
  mc_refresh_treatment_basis(state_data)
}

mc_ccv_residual_formula <- function(formula) {
  ccv_formula <- formula
  ccv_formula[[2L]] <- as.name(".mc_ccv_residual")
  ccv_formula
}

mc_design_covariance_ccv <- function(
  model,
  data,
  formula,
  metadata
) {
  if (!inherits(model, "fixest") || !identical(model$method, "feols")) {
    stop(
      paste(
        "The current CCV implementation is the linear OLS design",
        "formalized in the Lean file; nonlinear estimators require a",
        "separate score/Hessian argument."
      ),
      call. = FALSE
    )
  }

  coefficient_names <- names(stats::coef(model))
  residual <- stats::residuals(model)
  if (
    length(residual) != nrow(data) ||
      any(!is.finite(residual))
  ) {
    stop("CCV requires one finite fitted residual per estimation row.", call. = FALSE)
  }

  # `uhat` is held fixed across assignment states exactly as in the finite
  # design theorem.  For each reassigned design matrix X_s we solve
  #
  #   b_error(s) = (X_s' X_s)^(-1) X_s' uhat.
  #
  # Re-solving, instead of holding the observed bread fixed, incorporates the
  # random denominator/Gram matrix.  It is the vector-OLS counterpart of using
  # Dtilde/(Dtilde'Dtilde) in the scalar Lean theorem.
  ccv_formula <- mc_ccv_residual_formula(formula)
  state_count <- metadata$ccv_reference_states
  state_errors <- matrix(
    NA_real_,
    nrow = state_count,
    ncol = length(coefficient_names),
    dimnames = list(
      paste0("state_", 0:(state_count - 1L)),
      coefficient_names
    )
  )

  for (state_index in 0:(state_count - 1L)) {
    if (state_index == 0L) {
      # State zero is the observed design.  The fitted OLS residual satisfies
      # X_0' uhat = 0 by the normal equations, hence its coefficient-error
      # vector is exactly zero.  Setting that identity directly avoids feeding
      # roundoff from a second solve of this very ill-conditioned rich design
      # into the finite-state covariance.
      state_errors[1L, ] <- 0
      next
    }
    state_data <- mc_ccv_reference_state(
      data = data,
      metadata = metadata,
      state_index = state_index
    )
    state_data$.mc_ccv_residual <- residual
    state_coefficient <- fixest::feols(
      fml = ccv_formula,
      data = state_data,
      warn = FALSE,
      notes = FALSE,
      only.coef = TRUE
    )
    missing_coefficient <- setdiff(
      coefficient_names,
      names(state_coefficient)
    )
    if (
      length(missing_coefficient) > 0L ||
        any(!is.finite(state_coefficient[coefficient_names]))
    ) {
      stop(
        "CCV state ",
        state_index,
        " does not retain the observed model's coefficient basis.",
        call. = FALSE
      )
    }
    state_errors[state_index + 1L, ] <-
      state_coefficient[coefficient_names]
  }

  # The finite design has p_s = 1/17.  Thus this is a probability-weighted
  # covariance (division by 17), not the sample covariance division by 16:
  #
  #   V_dcCCV = sum_s p_s (b_error(s)-E_p[b_error])
  #                         (b_error(s)-E_p[b_error])'.
  #
  # Writing it as crossprod(centered_errors)/17 makes positive
  # semidefiniteness immediate, matching `DesignCovariance.dcCCV_nonneg`.
  centered_errors <- scale(
    state_errors,
    center = TRUE,
    scale = FALSE
  )
  covariance <- crossprod(centered_errors) / state_count
  covariance <- (covariance + t(covariance)) / 2

  if (
    any(!is.finite(covariance)) ||
      any(diag(covariance) < -1e-10)
  ) {
    stop("The constructed CCV covariance is not finite and PSD.", call. = FALSE)
  }

  # The nonzero eigenvalues of A'A equal those of AA'.  Inspecting the small
  # 17 x 17 matrix avoids an unnecessary eigendecomposition of a roughly
  # 900 x 900 coefficient covariance.
  small_kernel <- tcrossprod(centered_errors) / state_count
  kernel_eigenvalues <- eigen(
    (small_kernel + t(small_kernel)) / 2,
    symmetric = TRUE,
    only.values = TRUE
  )$values
  positive_tolerance <- max(abs(kernel_eigenvalues), 1) * 1e-10

  list(
    covariance = covariance,
    diagnostics = list(
      method = metadata$ccv_method,
      reference_design = metadata$ccv_reference_design,
      reference_states = state_count,
      design_df = state_count - 1L,
      covariance_rank = sum(kernel_eigenvalues > positive_tolerance),
      minimum_kernel_eigenvalue = min(kernel_eigenvalues),
      minimum_variance = min(diag(covariance)),
      maximum_observed_state_error = max(abs(state_errors[1L, ])),
      mean_state_error_norm = sqrt(sum(colMeans(state_errors)^2))
    )
  )
}

mc_model_matrix <- function(model, newdata) {
  matrix <- stats::model.matrix(model, data = newdata)
  coefficient_names <- names(stats::coef(model))
  absent <- setdiff(coefficient_names, colnames(matrix))
  if (length(absent) > 0L) {
    stop(
      "Counterfactual model matrix lacks estimated columns: ",
      paste(absent, collapse = ", "),
      call. = FALSE
    )
  }
  matrix[, coefficient_names, drop = FALSE]
}

mc_inverse_link <- function(eta, family) {
  if (identical(family, "gaussian")) {
    return(eta)
  }
  if (identical(family, "poisson")) {
    return(exp(pmin(eta, 700)))
  }
  if (identical(family, "binomial")) {
    return(stats::plogis(eta))
  }
  stop("Unsupported postestimation family: ", family, call. = FALSE)
}

mc_inverse_link_derivative <- function(mu, family) {
  if (identical(family, "gaussian")) {
    return(rep(1, length(mu)))
  }
  if (identical(family, "poisson")) {
    return(mu)
  }
  if (identical(family, "binomial")) {
    return(mu * (1 - mu))
  }
  stop("Unsupported postestimation family: ", family, call. = FALSE)
}

mc_inverse_link_second_derivative <- function(mu, family) {
  if (identical(family, "gaussian")) {
    return(rep(0, length(mu)))
  }
  if (identical(family, "poisson")) {
    return(mu)
  }
  if (identical(family, "binomial")) {
    return(mu * (1 - mu) * (1 - 2 * mu))
  }
  stop("Unsupported postestimation family: ", family, call. = FALSE)
}

mc_inference_metadata <- function(model, fallback_df = 16L) {
  list(
    variance_method = if (is.null(model$covariance_method)) {
      "unspecified"
    } else {
      model$covariance_method
    },
    reference_design = if (
      is.null(model$ccv_diagnostics$reference_design)
    ) {
      NA_character_
    } else {
      model$ccv_diagnostics$reference_design
    },
    inference_df = if (is.null(model$ccv_diagnostics$design_df)) {
      fallback_df
    } else {
      model$ccv_diagnostics$design_df
    }
  )
}

mc_counterfactual_data <- function(
  data,
  treatment_column,
  dose_change
) {
  counterfactual <- data
  counterfactual[[treatment_column]] <-
    counterfactual[[treatment_column]] + dose_change
  mc_refresh_treatment_basis(counterfactual)
}

mc_delta_components <- function(
  model,
  data,
  treatment_column,
  dose_change,
  family,
  offset_column = NA_character_
) {
  counterfactual <- mc_counterfactual_data(
    data,
    treatment_column,
    dose_change
  )

  x0 <- mc_model_matrix(model, data)
  x1 <- mc_model_matrix(model, counterfactual)
  beta <- stats::coef(model)

  offset <- if (
    length(offset_column) == 1L &&
      !is.na(offset_column) &&
      nzchar(offset_column)
  ) {
    data[[offset_column]]
  } else {
    rep(0, nrow(data))
  }

  eta0 <- drop(x0 %*% beta) + offset
  eta1 <- drop(x1 %*% beta) + offset
  mu0 <- mc_inverse_link(eta0, family)
  mu1 <- mc_inverse_link(eta1, family)
  derivative0 <- mc_inverse_link_derivative(mu0, family)
  derivative1 <- mc_inverse_link_derivative(mu1, family)

  gradient_mu0 <- x0 * derivative0
  gradient_mu1 <- x1 * derivative1
  difference <- mu1 - mu0
  gradient_difference <- gradient_mu1 - gradient_mu0

  list(
    difference = difference,
    gradient_difference = gradient_difference,
    mu0 = mu0,
    gradient_mu0 = gradient_mu0,
    observations = nrow(data)
  )
}

mc_derivative_components <- function(
  model,
  data,
  treatment_column,
  family,
  offset_column = NA_character_,
  epsilon = 1e-5
) {
  if (!is.finite(epsilon) || epsilon <= 0) {
    stop("Derivative epsilon must be positive.", call. = FALSE)
  }
  upper <- mc_counterfactual_data(
    data,
    treatment_column,
    epsilon
  )
  lower <- mc_counterfactual_data(
    data,
    treatment_column,
    -epsilon
  )

  x0 <- mc_model_matrix(model, data)
  x_upper <- mc_model_matrix(model, upper)
  x_lower <- mc_model_matrix(model, lower)
  x_derivative <- (x_upper - x_lower) / (2 * epsilon)
  beta <- stats::coef(model)

  offset <- if (
    length(offset_column) == 1L &&
      !is.na(offset_column) &&
      nzchar(offset_column)
  ) {
    data[[offset_column]]
  } else {
    rep(0, nrow(data))
  }

  eta0 <- drop(x0 %*% beta) + offset
  mu0 <- mc_inverse_link(eta0, family)
  first_link_derivative <- mc_inverse_link_derivative(mu0, family)
  second_link_derivative <- mc_inverse_link_second_derivative(
    mu0,
    family
  )
  index_derivative <- drop(x_derivative %*% beta)
  response_derivative <- first_link_derivative * index_derivative

  gradient_response_derivative <-
    x_derivative * first_link_derivative +
    x0 * (second_link_derivative * index_derivative)
  gradient_mu0 <- x0 * first_link_derivative

  list(
    difference = response_derivative,
    gradient_difference = gradient_response_derivative,
    mu0 = mu0,
    gradient_mu0 = gradient_mu0,
    observations = nrow(data)
  )
}

mc_aggregate_delta <- function(
  model,
  components,
  data,
  aggregation = c("average", "total", "rate_per_1000", "percent"),
  cluster_df = 16L,
  subset = NULL,
  weights = NULL
) {
  aggregation <- match.arg(aggregation)
  observations <- length(components$difference)
  if (nrow(data) != observations) {
    stop("Delta components and data have different row counts.", call. = FALSE)
  }

  keep <- if (is.null(subset)) {
    rep(TRUE, observations)
  } else if (is.logical(subset) && length(subset) == observations) {
    subset & !is.na(subset)
  } else if (is.numeric(subset)) {
    seq_len(observations) %in% subset
  } else {
    stop("subset must be NULL, logical, or numeric.", call. = FALSE)
  }
  if (!any(keep)) {
    stop("Delta aggregation subset is empty.", call. = FALSE)
  }

  supplied_weights <- if (is.null(weights)) {
    rep(1, observations)
  } else {
    if (length(weights) != observations) {
      stop("Delta weights have the wrong length.", call. = FALSE)
    }
    weights
  }
  if (
    any(!is.finite(supplied_weights[keep])) ||
      any(supplied_weights[keep] < 0) ||
      sum(supplied_weights[keep]) <= 0
  ) {
    stop(
      "Delta weights must be finite, nonnegative, and nonzero.",
      call. = FALSE
    )
  }
  aggregation_weights <- ifelse(keep, supplied_weights, 0)
  weight_sum <- sum(aggregation_weights)
  weighted_gradient_difference <- drop(crossprod(
    aggregation_weights,
    components$gradient_difference
  ))

  if (identical(aggregation, "average")) {
    estimate <- sum(
      aggregation_weights * components$difference
    ) / weight_sum
    gradient <- weighted_gradient_difference / weight_sum
  } else if (identical(aggregation, "total")) {
    estimate <- sum(aggregation_weights * components$difference)
    gradient <- weighted_gradient_difference
  } else if (identical(aggregation, "rate_per_1000")) {
    if (!"emp_farm_2011" %in% names(data)) {
      stop("Rate aggregation requires emp_farm_2011.", call. = FALSE)
    }
    denominator <- sum(
      aggregation_weights * data$emp_farm_2011
    )
    estimate <- 1000 * sum(
      aggregation_weights * components$difference
    ) / denominator
    gradient <- 1000 * weighted_gradient_difference / denominator
  } else {
    numerator <- sum(
      aggregation_weights * components$difference
    )
    denominator <- sum(aggregation_weights * components$mu0)
    gradient_numerator <- weighted_gradient_difference
    gradient_denominator <- drop(crossprod(
      aggregation_weights,
      components$gradient_mu0
    ))
    estimate <- 100 * numerator / denominator
    gradient <- 100 * (
      gradient_numerator * denominator -
        numerator * gradient_denominator
    ) / denominator^2
  }

  coefficient_names <- names(stats::coef(model))
  covariance <- stats::vcov(model)
  covariance <- covariance[
    coefficient_names,
    coefficient_names,
    drop = FALSE
  ]
  variance <- drop(gradient %*% covariance %*% gradient)
  if (!is.finite(variance)) {
    variance <- NA_real_
  } else if (variance < 0 && abs(variance) < 1e-10) {
    variance <- 0
  }
  standard_error <- if (is.finite(variance) && variance >= 0) {
    sqrt(variance)
  } else {
    NA_real_
  }
  critical <- stats::qt(0.975, df = cluster_df)
  inference_metadata <- mc_inference_metadata(model, cluster_df)

  result <- data.frame(
    aggregation = aggregation,
    estimate = estimate,
    standard_error = standard_error,
    conf_low = estimate - critical * standard_error,
    conf_high = estimate + critical * standard_error,
    baseline_prediction = if (
      identical(aggregation, "percent")
    ) {
      100
    } else {
      sum(aggregation_weights * components$mu0) / weight_sum
    },
    observations = sum(keep),
    weight_sum = weight_sum,
    variance_method = inference_metadata$variance_method,
    reference_design = inference_metadata$reference_design,
    inference_df = inference_metadata$inference_df,
    stringsAsFactors = FALSE
  )
  attr(result, "gradient") <- gradient
  result
}

mc_delta_contrast <- function(
  model,
  data,
  treatment_column,
  dose_change,
  family,
  offset_column = NA_character_,
  aggregation = c("average", "total", "rate_per_1000", "percent"),
  cluster_df = 16L,
  subset = NULL,
  weights = NULL
) {
  aggregation <- match.arg(aggregation)
  components <- mc_delta_components(
    model = model,
    data = data,
    treatment_column = treatment_column,
    dose_change = dose_change,
    family = family,
    offset_column = offset_column
  )
  mc_aggregate_delta(
    model = model,
    components = components,
    data = data,
    aggregation = aggregation,
    cluster_df = cluster_df,
    subset = subset,
    weights = weights
  )
}

mc_derivative_contrast <- function(
  model,
  data,
  treatment_column,
  family,
  offset_column = NA_character_,
  aggregation = c("average", "total", "rate_per_1000", "percent"),
  cluster_df = 16L,
  subset = NULL,
  weights = NULL,
  epsilon = 1e-5
) {
  aggregation <- match.arg(aggregation)
  components <- mc_derivative_components(
    model = model,
    data = data,
    treatment_column = treatment_column,
    family = family,
    offset_column = offset_column,
    epsilon = epsilon
  )
  mc_aggregate_delta(
    model = model,
    components = components,
    data = data,
    aggregation = aggregation,
    cluster_df = cluster_df,
    subset = subset,
    weights = weights
  )
}

mc_master_index_ame <- function(
  model,
  treatment_column,
  year,
  dose,
  z,
  polynomial_degrees = MC_MASTER_POLYNOMIAL_DEGREES,
  cluster_df = 16L
) {
  beta <- stats::coef(model)
  gradient <- stats::setNames(rep(0, length(beta)), names(beta))

  for (degree in polynomial_degrees) {
    basis <- mc_polynomial_basis_name(
      treatment_column,
      degree
    )
    derivative_scale <- switch(
      as.character(degree),
      "1" = 1,
      "2" = 2 * dose / 25,
      "3" = 3 * dose^2 / 125,
      stop("Unsupported master polynomial degree.", call. = FALSE)
    )
    base_term <- mc_explicit_year_term(year, basis)
    z_term <- mc_explicit_year_term(year, basis, MC_Z_COLUMN)
    absent <- setdiff(c(base_term, z_term), names(beta))
    if (length(absent) > 0L) {
      stop(
        "Master AME requires unidentified or absent terms: ",
        paste(absent, collapse = ", "),
        call. = FALSE
      )
    }
    gradient[[base_term]] <- derivative_scale
    gradient[[z_term]] <- derivative_scale * z
  }

  estimate <- sum(gradient * beta)
  covariance <- stats::vcov(model)
  active <- which(gradient != 0)
  covariance <- covariance[
    names(beta)[active],
    names(beta)[active],
    drop = FALSE
  ]
  active_gradient <- gradient[active]
  variance <- drop(
    active_gradient %*% covariance %*% active_gradient
  )
  if (!is.finite(variance)) {
    variance <- NA_real_
  } else if (variance < 0 && abs(variance) < 1e-10) {
    variance <- 0
  }
  standard_error <- if (is.finite(variance) && variance >= 0) {
    sqrt(variance)
  } else {
    NA_real_
  }
  critical <- stats::qt(0.975, df = cluster_df)
  inference_metadata <- mc_inference_metadata(model, cluster_df)

  data.frame(
    year = as.integer(year),
    treatment_column = treatment_column,
    dose = dose,
    z = z,
    estimate = estimate,
    standard_error = standard_error,
    conf_low = estimate - critical * standard_error,
    conf_high = estimate + critical * standard_error,
    variance_method = inference_metadata$variance_method,
    reference_design = inference_metadata$reference_design,
    inference_df = inference_metadata$inference_df,
    stringsAsFactors = FALSE
  )
}

mc_master_sample_effect <- function(
  model,
  data,
  treatment_column,
  dose_change = NULL,
  derivative = FALSE,
  weights = NULL,
  subset = NULL,
  polynomial_degrees = MC_MASTER_POLYNOMIAL_DEGREES,
  normalize = TRUE,
  cluster_df = 16L
) {
  if (
    (isTRUE(derivative) && !is.null(dose_change)) ||
      (!isTRUE(derivative) && is.null(dose_change))
  ) {
    stop(
      "Choose exactly one of derivative = TRUE or dose_change.",
      call. = FALSE
    )
  }
  observations <- nrow(data)
  keep <- if (is.null(subset)) {
    rep(TRUE, observations)
  } else if (is.logical(subset) && length(subset) == observations) {
    subset & !is.na(subset)
  } else if (is.numeric(subset)) {
    seq_len(observations) %in% subset
  } else {
    stop("subset must be NULL, logical, or numeric.", call. = FALSE)
  }
  supplied_weights <- if (is.null(weights)) {
    rep(1, observations)
  } else {
    if (length(weights) != observations) {
      stop("Master-effect weights have the wrong length.", call. = FALSE)
    }
    weights
  }
  if (
    !any(keep) ||
      any(!is.finite(supplied_weights[keep])) ||
      any(supplied_weights[keep] < 0) ||
      sum(supplied_weights[keep]) <= 0
  ) {
    stop("Invalid master-effect sample or weights.", call. = FALSE)
  }
  aggregation_weights <- ifelse(keep, supplied_weights, 0)
  normalization <- if (normalize) {
    sum(aggregation_weights)
  } else {
    1
  }

  beta <- stats::coef(model)
  gradient <- stats::setNames(rep(0, length(beta)), names(beta))
  treatment <- data[[treatment_column]]

  for (year in sort(unique(data$year[keep]))) {
    year_weight <- aggregation_weights * as.integer(data$year == year)
    if (sum(year_weight) <= 0) {
      next
    }
    for (degree in polynomial_degrees) {
      basis <- mc_polynomial_basis_name(
        treatment_column,
        degree
      )
      basis_scale <- switch(
        as.character(degree),
        "1" = 1,
        "2" = 25,
        "3" = 125,
        stop("Unsupported master polynomial degree.", call. = FALSE)
      )
      row_effect <- if (derivative) {
        degree * treatment^(degree - 1L) / basis_scale
      } else {
        (
          (treatment + dose_change)^degree -
            treatment^degree
        ) / basis_scale
      }
      base_term <- mc_explicit_year_term(year, basis)
      z_term <- mc_explicit_year_term(year, basis, MC_Z_COLUMN)
      absent <- setdiff(c(base_term, z_term), names(beta))
      if (length(absent) > 0L) {
        stop(
          "Master effect requires unidentified or absent terms: ",
          paste(absent, collapse = ", "),
          call. = FALSE
        )
      }
      gradient[[base_term]] <- sum(
        year_weight * row_effect
      ) / normalization
      gradient[[z_term]] <- sum(
        year_weight * row_effect * data[[MC_Z_COLUMN]]
      ) / normalization
    }
  }

  estimate <- sum(gradient * beta)
  covariance <- stats::vcov(model)
  active <- which(gradient != 0)
  covariance <- covariance[
    names(beta)[active],
    names(beta)[active],
    drop = FALSE
  ]
  active_gradient <- gradient[active]
  variance <- drop(
    active_gradient %*% covariance %*% active_gradient
  )
  if (!is.finite(variance)) {
    variance <- NA_real_
  } else if (variance < 0 && abs(variance) < 1e-10) {
    variance <- 0
  }
  standard_error <- if (is.finite(variance) && variance >= 0) {
    sqrt(variance)
  } else {
    NA_real_
  }
  critical <- stats::qt(0.975, df = cluster_df)
  inference_metadata <- mc_inference_metadata(model, cluster_df)

  result <- data.frame(
    estimand = if (derivative) {
      "average_marginal_effect"
    } else {
      "finite_dose_change"
    },
    dose_change = if (derivative) NA_real_ else dose_change,
    estimate = estimate,
    standard_error = standard_error,
    conf_low = estimate - critical * standard_error,
    conf_high = estimate + critical * standard_error,
    observations = sum(keep),
    weight_sum = sum(aggregation_weights),
    normalized = normalize,
    variance_method = inference_metadata$variance_method,
    reference_design = inference_metadata$reference_design,
    inference_df = inference_metadata$inference_df,
    stringsAsFactors = FALSE
  )
  attr(result, "gradient") <- gradient
  result
}

mc_format_number <- function(value, digits = 3L) {
  ifelse(
    is.na(value),
    "",
    formatC(
      value,
      format = "f",
      digits = digits,
      big.mark = ","
    )
  )
}

mc_latex_escape <- function(value) {
  value <- gsub("\\\\", "\\\\textbackslash{}", value)
  value <- gsub("([&_#%$])", "\\\\\\1", value)
  value
}

# Compact storage methods.  Full fixest objects retain N x K score and design
# structures that are unnecessary once coefficient estimates, their CCV and
# comparison clustered covariances, the formula, and the estimation rows have
# been frozen.
coef.mc_compact_model <- function(object, ...) {
  object$coefficients
}

vcov.mc_compact_model <- function(object, ...) {
  object$covariance
}

nobs.mc_compact_model <- function(object, ...) {
  object$observations
}
