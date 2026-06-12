# Wooldridge-style continuous-treatment DiD candidates for AEWR bite.
#
# This script leaves the original TWFE/event-study script intact and adds:
# 1. Alternative bite definitions, including log and positive-log "binding" bite.
# 2. Baseline-dose and pre-period mean-dose event studies.
# 3. Nonparametric pre-period bite quartile event studies.
# 4. Conditional Wooldridge-style event studies with baseline covariate trends.
# 5. Source-of-variation diagnostics for AEWR versus local wage components.
# 6. Continuous-dose CCV standard errors for selected fixest coefficients.

if (!exists("path_processed", mode = "function")) {
  local({
    split_current_file <- function() {
      frames <- sys.frames()
      for (idx in rev(seq_along(frames))) {
        ofile <- frames[[idx]]$ofile
        if (!is.null(ofile)) {
          return(normalizePath(ofile, winslash = "/", mustWork = FALSE))
        }
      }

      file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
      if (length(file_arg) > 0) {
        return(normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = FALSE))
      }

      normalizePath(getwd(), winslash = "/", mustWork = FALSE)
    }

    source(file.path(dirname(split_current_file()), "c00_setup.R"))
  })
}

split_load_analysis_inputs(
  include_county_df = TRUE,
  include_samples = TRUE
)

if (!requireNamespace("marginaleffects", quietly = TRUE)) {
  stop("The marginaleffects package is required for CCV delta-method inference.")
}

OUTCOME <- "h2a_cert_share_farm_workers_2011_start_year"
BASE_YEAR <- 2011L
PRE_YEARS <- 2008:2011
POST_YEARS <- 2012:2022
CLUSTER_VAR <- "cz_aewr_region_fe"
CCV_Q <- 1
CCV_Q_GRID <- c(0, 0.25, 0.5, 0.75, 1)
CCV_CONF_LEVEL <- 0.95

mean_or_na <- function(x) {
  if (all(is.na(x))) {
    return(NA_real_)
  }
  mean(x, na.rm = TRUE)
}

first_or_na <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) {
    return(NA_real_)
  }
  x[[1]]
}

standardize <- function(x) {
  sx <- sd(x, na.rm = TRUE)
  if (is.na(sx) || sx == 0) {
    return(x * NA_real_)
  }
  (x - mean(x, na.rm = TRUE)) / sx
}

complete_for_model <- function(data, vars) {
  data %>%
    filter(if_all(all_of(vars), ~ !is.na(.x) & is.finite(.x)))
}

## Bite definitions ------------------------------------------------------------

make_wooldridge_did_sample <- function(data) {
  data <- data %>%
    mutate(
      aewr_bite_log_p25_l1 = log(aewr_ppi_l1) - log(wage_p25_l1),
      aewr_bite_log_excess_p25_l1 = pmax(aewr_bite_log_p25_l1, 0),
      aewr_bite_pct_p25_l1 = aewr_ppi_l1 / wage_p25_l1 - 1
    )

  county_baseline <- data %>%
    group_by(countyfips) %>%
    summarise(
      bite_level_2011 = first_or_na(aewr_cz_p25_l1[year == BASE_YEAR]),
      bite_level_pre_mean = mean_or_na(aewr_cz_p25_l1[year %in% PRE_YEARS]),
      bite_log_2011 = first_or_na(aewr_bite_log_p25_l1[year == BASE_YEAR]),
      bite_log_pre_mean = mean_or_na(aewr_bite_log_p25_l1[year %in% PRE_YEARS]),
      bite_log_excess_2011 = first_or_na(aewr_bite_log_excess_p25_l1[year == BASE_YEAR]),
      bite_log_excess_pre_mean = mean_or_na(aewr_bite_log_excess_p25_l1[year %in% PRE_YEARS]),
      bite_pct_2011 = first_or_na(aewr_bite_pct_p25_l1[year == BASE_YEAR]),
      bite_pct_pre_mean = mean_or_na(aewr_bite_pct_p25_l1[year %in% PRE_YEARS]),
      base_ln_pop_2011 = first_or_na(ln_pop_census[year == BASE_YEAR]),
      base_emp_pop_ratio_2011 = first_or_na(emp_pop_ratio[year == BASE_YEAR]),
      base_ln_emp_farm_2011 = first_or_na(log1p(emp_farm_2011[year == BASE_YEAR])),
      base_ln_cropland_2007 = first_or_na(log1p(cropland_acr_2007[year == BASE_YEAR])),
      base_h2a_cert_share_2011 = first_or_na(.data[[OUTCOME]][year == BASE_YEAR]),
      base_pred_h2a_share_2011 = first_or_na(h2a_predicted_share_2011[year == BASE_YEAR]),
      .groups = "drop"
    ) %>%
    mutate(
      bite_level_2011_z = standardize(bite_level_2011),
      bite_level_pre_mean_z = standardize(bite_level_pre_mean),
      bite_log_2011_z = standardize(bite_log_2011),
      bite_log_pre_mean_z = standardize(bite_log_pre_mean),
      bite_log_excess_2011_z = standardize(bite_log_excess_2011),
      bite_log_excess_pre_mean_z = standardize(bite_log_excess_pre_mean),
      bite_pct_2011_z = standardize(bite_pct_2011),
      bite_pct_pre_mean_z = standardize(bite_pct_pre_mean),
      base_ln_pop_2011_z = standardize(base_ln_pop_2011),
      base_emp_pop_ratio_2011_z = standardize(base_emp_pop_ratio_2011),
      base_ln_emp_farm_2011_z = standardize(base_ln_emp_farm_2011),
      base_ln_cropland_2007_z = standardize(base_ln_cropland_2007),
      base_h2a_cert_share_2011_z = standardize(base_h2a_cert_share_2011),
      base_pred_h2a_share_2011_z = standardize(base_pred_h2a_share_2011),
      bite_level_pre_mean_quartile = factor(
        ntile(bite_level_pre_mean, 4),
        levels = 1:4,
        labels = paste0("Q", 1:4)
      ),
      bite_log_excess_pre_mean_quartile = factor(
        ntile(bite_log_excess_pre_mean, 4),
        levels = 1:4,
        labels = paste0("Q", 1:4)
      )
    )

  data %>%
    left_join(county_baseline, by = "countyfips")
}

wool_base <- make_wooldridge_did_sample(samp_base)
wool_no_border <- make_wooldridge_did_sample(samp_no_border)

## Continuous-dose CCV ---------------------------------------------------------

partial_out_column <- function(x, j) {
  x <- as.matrix(x)
  target <- x[, j]
  others <- x[, -j, drop = FALSE]

  if (ncol(others) == 0) {
    return(as.numeric(target))
  }

  fit <- lm.fit(x = others, y = target)
  as.numeric(fit$residuals)
}

backtick_term <- function(term) {
  paste0("`", gsub("`", "\\\\`", term), "`")
}

ccv_delta_method_test <- function(model, term, variance, conf_level = CCV_CONF_LEVEL) {
  ccv_vcov <- vcov(model)
  coef_names <- names(coef(model))

  ccv_vcov <- ccv_vcov[coef_names, coef_names, drop = FALSE]
  ccv_vcov[term, term] <- variance

  out <- marginaleffects::hypotheses(
    model,
    hypothesis = paste(backtick_term(term), "= 0"),
    vcov = ccv_vcov,
    conf_level = conf_level,
    df = Inf
  )

  as.data.frame(out)[1, , drop = FALSE]
}

ccv_components_for_feols <- function(model, data, cluster_var = CLUSTER_VAR) {
  if (is.null(model$X_demeaned)) {
    stop("Estimate the fixest model with demeaned = TRUE before computing CCV components.")
  }

  x <- as.matrix(model$X_demeaned)
  coef_names <- colnames(x)
  active_coef_names <- names(coef(model))

  if (all(active_coef_names %in% coef_names)) {
    x <- x[, active_coef_names, drop = FALSE]
    coef_names <- colnames(x)
  }

  uhat <- as.numeric(resid(model))
  cluster <- data[[cluster_var]]
  xtx_inv <- solve(crossprod(x))
  loading_matrix <- x %*% xtx_inv
  colnames(loading_matrix) <- coef_names

  score_matrix <- loading_matrix * uhat
  robust_vcov <- crossprod(score_matrix)
  cluster_score_matrix <- rowsum(score_matrix, group = cluster, reorder = FALSE, na.rm = TRUE)
  cluster_vcov <- crossprod(cluster_score_matrix)

  dimnames(robust_vcov) <- list(coef_names, coef_names)
  dimnames(cluster_vcov) <- list(coef_names, coef_names)

  list(
    coef_names = coef_names,
    coef = coef(model)[coef_names],
    loading_matrix = loading_matrix,
    robust_vcov = robust_vcov,
    cluster_vcov = cluster_vcov,
    cluster = cluster
  )
}

ccv_lambda_from_loading <- function(loading, cluster, q = CCV_Q, omega = c("cluster_mean", "unit_sum")) {
  omega <- match.arg(omega)
  omega_g <- tapply(
    loading^2,
    cluster,
    if (omega == "cluster_mean") {
      function(z) mean(z, na.rm = TRUE)
    } else {
      function(z) sum(z, na.rm = TRUE)
    }
  )
  omega_g <- as.numeric(omega_g)
  omega_g <- omega_g[is.finite(omega_g)]
  omega_ratio <- mean(omega_g)^2 / mean(omega_g^2)
  min(max(1 - q * omega_ratio, 0), 1)
}

ccv_lambda_from_loading_matrix <- function(loadings, cluster, q = CCV_Q) {
  omega_g <- tapply(
    rowSums(loadings^2),
    cluster,
    function(z) mean(z, na.rm = TRUE)
  )
  omega_g <- as.numeric(omega_g)
  omega_g <- omega_g[is.finite(omega_g)]
  omega_ratio <- mean(omega_g)^2 / mean(omega_g^2)
  min(max(1 - q * omega_ratio, 0), 1)
}

ccv_vcov_for_weights <- function(components, weights, q = CCV_Q) {
  weights <- weights[components$coef_names]
  weights[is.na(weights)] <- 0
  combo_loading <- as.numeric(components$loading_matrix %*% weights)
  lambda_d <- ccv_lambda_from_loading(combo_loading, components$cluster, q = q)
  v_ccv <- lambda_d * components$cluster_vcov +
    (1 - lambda_d) * components$robust_vcov
  list(vcov = v_ccv, lambda = lambda_d)
}

ccv_linear_combination <- function(model, data, weights, label, q = CCV_Q) {
  components <- ccv_components_for_feols(model, data)
  weights_full <- setNames(rep(0, length(components$coef_names)), components$coef_names)
  weights_full[names(weights)] <- weights
  ccv_v <- ccv_vcov_for_weights(components, weights_full, q = q)

  hypothesis_matrix <- matrix(
    weights_full,
    ncol = 1,
    dimnames = list(components$coef_names, label)
  )

  out <- marginaleffects::hypotheses(
    model,
    hypothesis = hypothesis_matrix,
    vcov = ccv_v$vcov,
    conf_level = CCV_CONF_LEVEL,
    df = Inf
  )

  as_tibble(as.data.frame(out)) %>%
    transmute(
      contrast = label,
      estimate,
      ccv_se = std.error,
      ccv_statistic = statistic,
      ccv_p = p.value,
      ccv_conf_low = conf.low,
      ccv_conf_high = conf.high,
      ccv_lambda = ccv_v$lambda,
      ccv_q = q
    )
}

ccv_joint_wald <- function(model, data, weights_matrix, label, q = CCV_Q) {
  components <- ccv_components_for_feols(model, data)
  weights_matrix <- weights_matrix[components$coef_names, , drop = FALSE]
  weights_matrix[is.na(weights_matrix)] <- 0

  restriction_loadings <- components$loading_matrix %*% weights_matrix
  lambda_d <- ccv_lambda_from_loading_matrix(
    restriction_loadings,
    components$cluster,
    q = q
  )

  v_ccv <- lambda_d * components$cluster_vcov +
    (1 - lambda_d) * components$robust_vcov
  restriction_estimates <- as.numeric(t(weights_matrix) %*% components$coef)
  restriction_vcov <- t(weights_matrix) %*% v_ccv %*% weights_matrix
  wald_stat <- as.numeric(
    t(restriction_estimates) %*%
      qr.solve(restriction_vcov, restriction_estimates)
  )
  df <- ncol(weights_matrix)

  tibble(
    test = label,
    statistic = wald_stat,
    df = df,
    p_value = pchisq(wald_stat, df = df, lower.tail = FALSE),
    ccv_lambda = lambda_d,
    ccv_q = q
  )
}

event_terms_for_years <- function(model, years, dose_pattern) {
  coef_names <- names(coef(model))
  year_pattern <- paste(years, collapse = "|")
  grep(
    paste0("year::(", year_pattern, "):.*", dose_pattern),
    coef_names,
    value = TRUE
  )
}

average_weights <- function(terms) {
  setNames(rep(1 / length(terms), length(terms)), terms)
}

identity_weights_matrix <- function(coef_names, terms) {
  weights <- matrix(
    0,
    nrow = length(coef_names),
    ncol = length(terms),
    dimnames = list(coef_names, terms)
  )
  for (term in terms) {
    weights[term, term] <- 1
  }
  weights
}

ccv_for_feols <- function(
    model,
    data,
    cluster_var = CLUSTER_VAR,
    term_pattern = NULL,
    q = 1,
    omega = c("cluster_mean", "unit_sum")) {
  omega <- match.arg(omega)

  if (is.null(model$X_demeaned)) {
    stop("Estimate the fixest model with demeaned = TRUE before calling ccv_for_feols().")
  }

  x <- as.matrix(model$X_demeaned)
  coef_names <- colnames(x)
  if (is.null(coef_names)) {
    coef_names <- names(coef(model))
    colnames(x) <- coef_names
  }

  active_coef_names <- names(coef(model))
  if (all(active_coef_names %in% coef_names)) {
    x <- x[, active_coef_names, drop = FALSE]
    coef_names <- colnames(x)
  }

  keep_terms <- coef_names
  if (!is.null(term_pattern)) {
    keep_terms <- grep(term_pattern, coef_names, value = TRUE)
  }

  if (length(keep_terms) == 0) {
    return(tibble())
  }

  uhat <- as.numeric(resid(model))
  if (nrow(data) != length(uhat)) {
    stop(
      "CCV data/model row mismatch. Pass the complete-case data used by feols; ",
      "this script prefilters model samples to avoid fixest row drops."
    )
  }

  cluster <- data[[cluster_var]]
  if (is.null(cluster)) {
    stop("Cluster variable not found in data: ", cluster_var)
  }

  keep_idx <- match(keep_terms, coef_names)
  xtx_inv <- solve(crossprod(x))
  influence_loadings <- x %*% xtx_inv[, keep_idx, drop = FALSE]
  colnames(influence_loadings) <- keep_terms

  map_dfr(seq_along(keep_terms), function(k) {
    term <- keep_terms[[k]]
    j <- keep_idx[[k]]
    loading <- as.numeric(influence_loadings[, k])
    score <- loading * uhat
    denom <- 1 / xtx_inv[j, j]

    cluster_score <- tapply(score, cluster, sum, na.rm = TRUE)
    omega_g <- tapply(
      loading^2,
      cluster,
      if (omega == "cluster_mean") {
        function(z) mean(z, na.rm = TRUE)
      } else {
        function(z) sum(z, na.rm = TRUE)
      }
    )

    omega_g <- as.numeric(omega_g)
    omega_g <- omega_g[is.finite(omega_g)]
    omega_ratio <- mean(omega_g)^2 / mean(omega_g^2)
    lambda_d <- 1 - q * omega_ratio
    lambda_d <- min(max(lambda_d, 0), 1)

    v_robust <- sum(score^2)
    v_cluster <- sum(cluster_score^2, na.rm = TRUE)
    v_ccv <- lambda_d * v_cluster + (1 - lambda_d) * v_robust
    ccv_test <- ccv_delta_method_test(model, term, v_ccv)

    tibble(
      term = term,
      estimate = unname(coef(model)[term]),
      fixest_cluster_se = model$coeftable[term, "Std. Error"],
      fixest_cluster_p = model$coeftable[term, "Pr(>|t|)"],
      ccv_robust_se = sqrt(v_robust),
      ccv_cluster_se = sqrt(v_cluster),
      ccv_se = ccv_test$std.error,
      ccv_statistic = ccv_test$statistic,
      ccv_p = ccv_test$p.value,
      ccv_conf_low = ccv_test$conf.low,
      ccv_conf_high = ccv_test$conf.high,
      ccv_lambda = lambda_d,
      ccv_q = q,
      ccv_omega = omega,
      ccv_clusters = length(unique(cluster)),
      ccv_score_denom = denom
    )
  })
}

delta_log_ratio_var <- function(
    a,
    w,
    var_a = NULL,
    var_w = NULL,
    cov_aw = NULL) {
  if (is.null(var_a) || is.null(var_w)) {
    return(rep(NA_real_, length(a)))
  }
  if (is.null(cov_aw)) {
    cov_aw <- rep(0, length(a))
  }
  var_a / a^2 + var_w / w^2 - 2 * cov_aw / (a * w)
}

delta_positive_part_var <- function(x, var_x) {
  ifelse(x > 0, var_x, 0)
}

## The current data do not include sampling variances for AEWR or CZ wage
## quantiles. The delta helpers above are included for future generated-dose
## uncertainty; the CCV estimates below use realized residualized dose moments.

## Model builders --------------------------------------------------------------

conditional_trend_terms <- paste(
  "i(year, base_ln_emp_farm_2011_z, ref = 2011)",
  "i(year, base_ln_cropland_2007_z, ref = 2011)",
  "i(year, base_h2a_cert_share_2011_z, ref = 2011)",
  "i(year, base_pred_h2a_share_2011_z, ref = 2011)",
  "i(year, base_ln_pop_2011_z, ref = 2011)",
  sep = " + "
)

estimate_feols_clean <- function(fml, data, vars, vcov = as.formula(paste0("~", CLUSTER_VAR))) {
  model_data <- complete_for_model(data, unique(c(vars, OUTCOME, CLUSTER_VAR)))
  model <- feols(
    fml,
    data = model_data,
    vcov = vcov,
    demeaned = TRUE,
    notes = FALSE
  )
  list(model = model, data = model_data)
}

estimate_event_study <- function(
    data,
    dose_var,
    controls = FALSE,
    conditional_trends = FALSE,
    region_year_fe = FALSE) {
  rhs <- paste0("i(year, ", dose_var, ", ref = ", BASE_YEAR, ")")
  vars <- c("year", dose_var, "county_fe")

  if (controls) {
    rhs <- paste(rhs, "ln_pop_census", "emp_pop_ratio", sep = " + ")
    vars <- c(vars, "ln_pop_census", "emp_pop_ratio")
  }

  if (conditional_trends) {
    rhs <- paste(rhs, conditional_trend_terms, sep = " + ")
    vars <- c(
      vars,
      "base_ln_emp_farm_2011_z",
      "base_ln_cropland_2007_z",
      "base_h2a_cert_share_2011_z",
      "base_pred_h2a_share_2011_z",
      "base_ln_pop_2011_z"
    )
  }

  fe <- if (region_year_fe) {
    vars <- c(vars, "aewrregtime_fe")
    "county_fe + aewrregtime_fe"
  } else {
    vars <- c(vars, "year_fe")
    "county_fe + year_fe"
  }

  estimate_feols_clean(
    as.formula(paste(OUTCOME, "~", rhs, "|", fe)),
    data = data,
    vars = vars
  )
}

estimate_quartile_event_study <- function(data, group_var, controls = FALSE) {
  rhs <- paste0(
    "i(year, ", group_var, ", ref = ", BASE_YEAR, ', ref2 = "Q1")'
  )
  vars <- c("year", group_var, "county_fe", "year_fe")
  if (controls) {
    rhs <- paste(rhs, "ln_pop_census", "emp_pop_ratio", sep = " + ")
    vars <- c(vars, "ln_pop_census", "emp_pop_ratio")
  }
  estimate_feols_clean(
    as.formula(paste(OUTCOME, "~", rhs, "| county_fe + year_fe")),
    data = data,
    vars = vars
  )
}

estimate_current_twfe <- function(data, bite_var, controls = FALSE, region_year_fe = FALSE) {
  rhs <- paste0(bite_var, " * postdummy")
  vars <- c(bite_var, "postdummy", "county_fe")
  if (controls) {
    rhs <- paste(rhs, "ln_pop_census", "emp_pop_ratio", sep = " + ")
    vars <- c(vars, "ln_pop_census", "emp_pop_ratio")
  }
  fe <- if (region_year_fe) {
    vars <- c(vars, "aewrregtime_fe")
    "county_fe + aewrregtime_fe"
  } else {
    vars <- c(vars, "year_fe")
    "county_fe + year_fe"
  }
  estimate_feols_clean(
    as.formula(paste(OUTCOME, "~", rhs, "|", fe)),
    data = data,
    vars = vars
  )
}

## Candidate specifications ----------------------------------------------------

wool_models <- list(
  current_level = estimate_current_twfe(wool_base, "aewr_cz_p25_l1"),
  current_level_controls = estimate_current_twfe(wool_base, "aewr_cz_p25_l1", controls = TRUE),
  current_log_excess = estimate_current_twfe(wool_base, "aewr_bite_log_excess_p25_l1"),
  current_level_region_year_fe = estimate_current_twfe(
    wool_base,
    "aewr_cz_p25_l1",
    region_year_fe = TRUE
  ),
  es_timevarying_level = estimate_event_study(wool_base, "aewr_cz_p25_l1"),
  es_baseline_2011_level = estimate_event_study(wool_base, "bite_level_2011_z"),
  es_premean_level = estimate_event_study(wool_base, "bite_level_pre_mean_z"),
  es_premean_level_controls = estimate_event_study(
    wool_base,
    "bite_level_pre_mean_z",
    controls = TRUE
  ),
  es_premean_level_conditional = estimate_event_study(
    wool_base,
    "bite_level_pre_mean_z",
    conditional_trends = TRUE
  ),
  es_premean_log_excess = estimate_event_study(
    wool_base,
    "bite_log_excess_pre_mean_z"
  ),
  es_premean_level_region_year_fe = estimate_event_study(
    wool_base,
    "bite_level_pre_mean_z",
    region_year_fe = TRUE
  ),
  es_premean_level_no_border = estimate_event_study(
    wool_no_border,
    "bite_level_pre_mean_z"
  ),
  es_quartile_premean_level = estimate_quartile_event_study(
    wool_base,
    "bite_level_pre_mean_quartile"
  ),
  es_quartile_log_excess = estimate_quartile_event_study(
    wool_base,
    "bite_log_excess_pre_mean_quartile"
  )
)

## Tables ----------------------------------------------------------------------

etable(
  wool_models$current_level$model,
  wool_models$current_level_controls$model,
  wool_models$current_log_excess$model,
  wool_models$current_level_region_year_fe$model,
  tex = TRUE,
  title = "Continuous-Dose DiD: Current TWFE and Bite Redefinitions",
  headers = c(
    "Level Bite",
    "Level Bite + Controls",
    "Positive Log Bite",
    "Level Bite + AEWR Region-Year FE"
  ),
  dict = c(
    "aewr_cz_p25_l1" = "Lagged AEWR - CZ p25 wage",
    "aewr_bite_log_excess_p25_l1" = "Positive log AEWR/CZ p25 wage",
    "postdummy" = "Post-2011",
    "ln_pop_census" = "Log population",
    "emp_pop_ratio" = "Employment-to-pop ratio"
  ),
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
  file = path_tables("table_3_wooldridge_current_bite_redefinitions.tex"),
  replace = TRUE
)

etable(
  wool_models$es_timevarying_level$model,
  wool_models$es_baseline_2011_level$model,
  wool_models$es_premean_level$model,
  wool_models$es_premean_level_conditional$model,
  wool_models$es_premean_log_excess$model,
  tex = TRUE,
  title = "Wooldridge-Style Continuous-Dose Event Studies",
  headers = c(
    "Time-Varying Level",
    "2011 Level Dose",
    "Pre-Mean Level Dose",
    "Pre-Mean + Baseline Trends",
    "Pre-Mean Positive Log Dose"
  ),
  keep = "%year::",
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
  file = path_tables("table_4_wooldridge_continuous_event_studies.tex"),
  replace = TRUE
)

etable(
  wool_models$es_quartile_premean_level$model,
  wool_models$es_quartile_log_excess$model,
  tex = TRUE,
  title = "Wooldridge-Style Dose-Group Event Studies",
  headers = c("Pre-Mean Level Quartiles", "Pre-Mean Positive Log Quartiles"),
  keep = "%year::",
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
  file = path_tables("table_5_wooldridge_quartile_event_studies.tex"),
  replace = TRUE
)

## Coefficient extraction and plots -------------------------------------------

extract_fixest_event_study <- function(model, spec, term_pattern = "year::") {
  ct <- as.data.frame(model$coeftable)
  ct$term <- rownames(ct)
  names(ct) <- make.names(names(ct))

  ct %>%
    filter(str_detect(term, term_pattern)) %>%
    mutate(
      spec = spec,
      year = as.integer(str_match(term, "year::([0-9]{4})")[, 2]),
      lower_ci = Estimate - 1.96 * Std..Error,
      upper_ci = Estimate + 1.96 * Std..Error
    ) %>%
    select(spec, year, term, estimate = Estimate, se = Std..Error, lower_ci, upper_ci)
}

event_study_coefficients <- imap_dfr(
  wool_models[c(
    "es_timevarying_level",
    "es_baseline_2011_level",
    "es_premean_level",
    "es_premean_level_conditional",
    "es_premean_log_excess",
    "es_premean_level_no_border",
    "es_premean_level_region_year_fe",
    "es_quartile_premean_level",
    "es_quartile_log_excess"
  )],
  ~ extract_fixest_event_study(.x$model, .y)
)

write_csv(
  event_study_coefficients,
  path_tables("wooldridge_event_study_coefficients.csv")
)

plot_continuous_event_study <- function(coefs, spec, filename, y_label) {
  plot_df <- coefs %>%
    filter(.data$spec == !!spec) %>%
    bind_rows(tibble(
      spec = spec,
      year = BASE_YEAR,
      term = "base year",
      estimate = 0,
      se = 0,
      lower_ci = 0,
      upper_ci = 0
    )) %>%
    arrange(year)

  plot <- ggplot(plot_df, aes(x = year, y = estimate)) +
    geom_hline(yintercept = 0, color = "grey45") +
    geom_vline(xintercept = BASE_YEAR, linetype = "dashed", color = "grey45") +
    geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci), alpha = 0.2, fill = "steelblue") +
    geom_line(color = "steelblue", linewidth = 1) +
    geom_point(color = "steelblue", size = 2) +
    labs(x = "Year", y = y_label) +
    theme_clean()

  ggsave(
    plot = plot,
    filename = path_figures(filename),
    width = 8,
    height = 5,
    device = "png"
  )
}

plot_continuous_event_study(
  event_study_coefficients,
  "es_premean_level",
  "coefplot_wooldridge_premean_level_bite.png",
  "Coefficient per SD of pre-period AEWR bite"
)

plot_continuous_event_study(
  event_study_coefficients,
  "es_premean_level_conditional",
  "coefplot_wooldridge_premean_level_bite_conditional.png",
  "Coefficient per SD of pre-period AEWR bite"
)

plot_continuous_event_study(
  event_study_coefficients,
  "es_premean_log_excess",
  "coefplot_wooldridge_premean_positive_log_bite.png",
  "Coefficient per SD of positive log AEWR bite"
)

quartile_plot_df <- event_study_coefficients %>%
  filter(spec == "es_quartile_premean_level") %>%
  mutate(
    quartile = str_match(term, "bite_level_pre_mean_quartile::(Q[2-4])")[, 2]
  ) %>%
  filter(!is.na(quartile)) %>%
  bind_rows(tibble(
    spec = "es_quartile_premean_level",
    year = rep(BASE_YEAR, 3),
    term = "base year",
    estimate = 0,
    se = 0,
    lower_ci = 0,
    upper_ci = 0,
    quartile = paste0("Q", 2:4)
  ))

quartile_plot <- ggplot(
  quartile_plot_df,
  aes(x = year, y = estimate, color = quartile, fill = quartile)
) +
  geom_hline(yintercept = 0, color = "grey45") +
  geom_vline(xintercept = BASE_YEAR, linetype = "dashed", color = "grey45") +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  labs(
    x = "Year",
    y = "Difference relative to pre-period bite Q1",
    color = "Bite group",
    fill = "Bite group"
  ) +
  theme_clean()

ggsave(
  plot = quartile_plot,
  filename = path_figures("coefplot_wooldridge_premean_bite_quartiles.png"),
  width = 8,
  height = 5,
  device = "png"
)

## CCV summaries ---------------------------------------------------------------

term_patterns <- tribble(
  ~spec, ~pattern,
  "current_level", "aewr_cz_p25_l1:postdummy|postdummy:aewr_cz_p25_l1",
  "current_level_controls", "aewr_cz_p25_l1:postdummy|postdummy:aewr_cz_p25_l1",
  "current_log_excess", "aewr_bite_log_excess_p25_l1:postdummy|postdummy:aewr_bite_log_excess_p25_l1",
  "es_timevarying_level", "year::.*aewr_cz_p25_l1|aewr_cz_p25_l1.*year::",
  "es_baseline_2011_level", "year::.*bite_level_2011_z|bite_level_2011_z.*year::",
  "es_premean_level", "year::.*bite_level_pre_mean_z|bite_level_pre_mean_z.*year::",
  "es_premean_level_conditional", "year::.*bite_level_pre_mean_z|bite_level_pre_mean_z.*year::",
  "es_premean_log_excess", "year::.*bite_log_excess_pre_mean_z|bite_log_excess_pre_mean_z.*year::",
  "es_quartile_premean_level", "year::.*bite_level_pre_mean_quartile|bite_level_pre_mean_quartile.*year::"
)

ccv_summary <- pmap_dfr(term_patterns, function(spec, pattern) {
  fit <- wool_models[[spec]]
  ccv_for_feols(
    model = fit$model,
    data = fit$data,
    term_pattern = pattern,
    q = CCV_Q,
    omega = "cluster_mean"
  ) %>%
    mutate(spec = spec, .before = term)
})

write_csv(
  ccv_summary,
  path_tables("wooldridge_ccv_standard_errors.csv")
)

ccv_q_sensitivity <- expand_grid(
  term_patterns %>%
    filter(str_starts(spec, "current_")),
  q = CCV_Q_GRID
) %>%
  pmap_dfr(function(spec, pattern, q) {
    fit <- wool_models[[spec]]
    ccv_for_feols(
      model = fit$model,
      data = fit$data,
      term_pattern = pattern,
      q = q,
      omega = "cluster_mean"
    ) %>%
      mutate(spec = spec, .before = term)
  })

write_csv(
  ccv_q_sensitivity,
  path_tables("wooldridge_ccv_q_sensitivity_current_specs.csv")
)

linear_combo_specs <- tribble(
  ~spec, ~dose_pattern,
  "es_timevarying_level", "aewr_cz_p25_l1",
  "es_baseline_2011_level", "bite_level_2011_z",
  "es_premean_level", "bite_level_pre_mean_z",
  "es_premean_level_conditional", "bite_level_pre_mean_z",
  "es_premean_log_excess", "bite_log_excess_pre_mean_z"
)

ccv_linear_combinations <- pmap_dfr(linear_combo_specs, function(spec, dose_pattern) {
  fit <- wool_models[[spec]]
  pre_terms <- event_terms_for_years(fit$model, 2008:2010, dose_pattern)
  post_terms <- event_terms_for_years(fit$model, POST_YEARS, dose_pattern)

  bind_rows(
    if (length(pre_terms) > 0) {
      ccv_linear_combination(
        fit$model,
        fit$data,
        average_weights(pre_terms),
        "average_pre_event_coefficient"
      )
    },
    if (length(post_terms) > 0) {
      ccv_linear_combination(
        fit$model,
        fit$data,
        average_weights(post_terms),
        "average_post_event_coefficient"
      )
    }
  ) %>%
    mutate(spec = spec, .before = contrast)
})

write_csv(
  ccv_linear_combinations,
  path_tables("wooldridge_ccv_linear_combinations.csv")
)

ccv_wald_tests <- pmap_dfr(linear_combo_specs, function(spec, dose_pattern) {
  fit <- wool_models[[spec]]
  coef_names <- names(coef(fit$model))
  pre_terms <- event_terms_for_years(fit$model, 2008:2010, dose_pattern)
  post_terms <- event_terms_for_years(fit$model, POST_YEARS, dose_pattern)

  bind_rows(
    if (length(pre_terms) > 0) {
      ccv_joint_wald(
        fit$model,
        fit$data,
        identity_weights_matrix(coef_names, pre_terms),
        "joint_pre_event_coefficients_zero"
      )
    },
    if (length(post_terms) > 0) {
      ccv_joint_wald(
        fit$model,
        fit$data,
        identity_weights_matrix(coef_names, post_terms),
        "joint_post_event_coefficients_zero"
      )
    }
  ) %>%
    mutate(spec = spec, .before = test)
})

write_csv(
  ccv_wald_tests,
  path_tables("wooldridge_ccv_wald_tests.csv")
)

## Source-of-variation diagnostics --------------------------------------------

variation_model_data <- complete_for_model(
  wool_base,
  c(
    OUTCOME,
    "aewr_cz_p25_l1",
    "aewr_ppi_l1",
    "wage_p25_l1",
    "postdummy",
    "county_fe",
    "year_fe",
    CLUSTER_VAR,
    "aewr_region_num",
    "year"
  )
)

residualize_on_county_year <- function(var, data) {
  fit <- feols(
    as.formula(paste(var, "~ 1 | county_fe + year_fe")),
    data = data,
    notes = FALSE
  )
  as.numeric(resid(fit))
}

variation_diag <- variation_model_data %>%
  mutate(
    bite_resid = residualize_on_county_year("aewr_cz_p25_l1", variation_model_data),
    aewr_resid = residualize_on_county_year("aewr_ppi_l1", variation_model_data),
    wage_p25_resid = residualize_on_county_year("wage_p25_l1", variation_model_data)
  )

source_of_variation <- tibble(
  object = c("bite", "aewr_component", "wage_p25_component"),
  raw_variance = c(
    var(variation_diag$aewr_cz_p25_l1),
    var(variation_diag$aewr_ppi_l1),
    var(variation_diag$wage_p25_l1)
  ),
  twfe_residual_variance = c(
    var(variation_diag$bite_resid),
    var(variation_diag$aewr_resid),
    var(variation_diag$wage_p25_resid)
  )
) %>%
  bind_rows(tibble(
    object = c("corr_resid_bite_aewr", "corr_resid_bite_wage_p25"),
    raw_variance = NA_real_,
    twfe_residual_variance = c(
      cor(variation_diag$bite_resid, variation_diag$aewr_resid),
      cor(variation_diag$bite_resid, variation_diag$wage_p25_resid)
    )
  ))

write_csv(
  source_of_variation,
  path_tables("wooldridge_bite_source_of_variation.csv")
)

current_diag_model <- estimate_current_twfe(wool_base, "aewr_cz_p25_l1")
current_diag_ccv <- ccv_for_feols(
  current_diag_model$model,
  current_diag_model$data,
  term_pattern = "aewr_cz_p25_l1:postdummy|postdummy:aewr_cz_p25_l1",
  q = CCV_Q
)

current_term <- current_diag_ccv$term[[1]]
current_x <- as.matrix(current_diag_model$model$X_demeaned)
current_active_coef_names <- names(coef(current_diag_model$model))
if (all(current_active_coef_names %in% colnames(current_x))) {
  current_x <- current_x[, current_active_coef_names, drop = FALSE]
}
current_dtilde <- partial_out_column(current_x, match(current_term, colnames(current_x)))
current_y_demeaned <- as.numeric(current_diag_model$model$y_demeaned)
current_score <- current_dtilde * current_y_demeaned
current_denom <- sum(current_dtilde^2)
current_numer <- sum(current_score)

leverage_contributions <- current_diag_model$data %>%
  mutate(
    dtilde = current_dtilde,
    score = current_score,
    leverage = dtilde^2 / current_denom,
    numerator_contribution = score / current_numer
  ) %>%
  group_by(year, aewr_region_num) %>%
  summarise(
    leverage = sum(leverage, na.rm = TRUE),
    numerator_contribution = sum(numerator_contribution, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

write_csv(
  leverage_contributions,
  path_tables("wooldridge_current_twfe_leverage_contributions.csv")
)

cat("\nWooldridge-style continuous DiD models estimated.\n")
cat("Wrote tables and diagnostics to:\n")
cat("  ", path_tables("table_3_wooldridge_current_bite_redefinitions.tex"), "\n")
cat("  ", path_tables("table_4_wooldridge_continuous_event_studies.tex"), "\n")
cat("  ", path_tables("table_5_wooldridge_quartile_event_studies.tex"), "\n")
cat("  ", path_tables("wooldridge_ccv_standard_errors.csv"), "\n")
cat("  ", path_tables("wooldridge_ccv_q_sensitivity_current_specs.csv"), "\n")
cat("  ", path_tables("wooldridge_ccv_linear_combinations.csv"), "\n")
cat("  ", path_tables("wooldridge_ccv_wald_tests.csv"), "\n")
cat("  ", path_tables("wooldridge_bite_source_of_variation.csv"), "\n")
