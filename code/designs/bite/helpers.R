# Shared helpers for the standalone full-panel PPML bite design.
# Plain imperative functions only; no cross-design imports.

suppressPackageStartupMessages({
  library(dplyr)
  library(fixest)
  library(tibble)
})

setFixest_nthreads(1L)
setFixest_notes(FALSE)
setFixest_ssc(ssc(K.adj = TRUE, K.fixef = "nonnested", G.adj = TRUE))

# docs-ground:start bite-estimation-contract
BITE_BASELINE_YEARS <- c(2008L, 2010L)
BITE_COVARIATE_YEARS <- 2008L:2010L
BITE_ESTIMATION_START_YEAR <- 2011L
BITE_ESTIMATION_END_YEAR <- 2025L
BITE_EVENT_START_YEAR <- 2008L
BITE_EVENT_END_YEAR <- 2025L
BITE_EVENT_REF_YEAR <- 2011L

# Midpoint trapezoid weights across the five ACS CZ real wage quantiles.
BITE_SHORTFALL_WEIGHTS <- c(
  wage_p10 = 0.175,
  wage_p25 = 0.200,
  wage_p50 = 0.250,
  wage_p75 = 0.200,
  wage_p90 = 0.175
)

bite_safe_mean <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) {
    return(NA_real_)
  }
  mean(x)
}

bite_std_within_region <- function(x, region) {
  if (length(x) != length(region)) {
    stop("bite_std_within_region: length mismatch between x and region.", call. = FALSE)
  }
  out <- rep(NA_real_, length(x))
  regions <- unique(region[!is.na(region)])
  for (reg in regions) {
    idx <- which(region == reg & is.finite(x))
    if (length(idx) < 2L) {
      stop(
        "bite_std_within_region: fewer than 2 finite values in AEWR region ",
        reg,
        ".",
        call. = FALSE
      )
    }
    mu <- mean(x[idx])
    sd_val <- stats::sd(x[idx])
    if (!is.finite(sd_val) || sd_val <= 0) {
      stop(
        "bite_std_within_region: non-positive SD in AEWR region ",
        reg,
        ".",
        call. = FALSE
      )
    }
    out[idx] <- (x[idx] - mu) / sd_val
  }
  out
}

bite_compute_shortfall_log <- function(df) {
  req_cols <- c("aewr_ppi", "prevailing_ag_min_wage_ppi", names(BITE_SHORTFALL_WEIGHTS))
  missing_cols <- setdiff(req_cols, names(df))
  if (length(missing_cols) > 0L) {
    stop(
      "bite_compute_shortfall_log: missing columns: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }

  valid <- is.finite(df$aewr_ppi) & df$aewr_ppi > 0 &
    is.finite(df$prevailing_ag_min_wage_ppi) & df$prevailing_ag_min_wage_ppi > 0
  for (q_col in names(BITE_SHORTFALL_WEIGHTS)) {
    valid <- valid & is.finite(df[[q_col]]) & df[[q_col]] > 0
  }

  out <- rep(NA_real_, nrow(df))
  if (!any(valid)) {
    return(out)
  }

  ln_aewr_real <- log(df$aewr_ppi[valid])
  ag_floor_real <- df$prevailing_ag_min_wage_ppi[valid]
  sf <- rep(0, sum(valid))
  for (q_col in names(BITE_SHORTFALL_WEIGHTS)) {
    w_q <- BITE_SHORTFALL_WEIGHTS[[q_col]]
    eff_wage <- pmax(df[[q_col]][valid], ag_floor_real)
    sf <- sf + w_q * pmax(0, ln_aewr_real - log(eff_wage))
  }
  out[valid] <- sf
  out
}

bite_make_quartiles <- function(x) {
  breaks <- stats::quantile(x, probs = c(0, 0.25, 0.50, 0.75, 1), na.rm = TRUE, names = FALSE)
  if (any(!is.finite(breaks)) || any(diff(breaks) <= 0)) {
    stop("bite_make_quartiles: non-strictly-increasing quartile breaks.", call. = FALSE)
  }
  as.character(cut(
    x,
    breaks = breaks,
    include.lowest = TRUE,
    labels = c("Q1", "Q2", "Q3", "Q4")
  ))
}

bite_complete_cases <- function(data, cols) {
  keep <- rep(TRUE, nrow(data))
  for (col in cols) {
    v <- data[[col]]
    if (is.numeric(v)) {
      keep <- keep & is.finite(v)
    } else if (is.character(v)) {
      keep <- keep & !is.na(v) & nzchar(v)
    } else {
      keep <- keep & !is.na(v)
    }
  }
  data[keep, , drop = FALSE]
}

bite_used_rows <- function(fit, data) {
  idx <- fixest::obs(fit)
  if (length(idx) != stats::nobs(fit)) {
    stop(
      "bite_used_rows: fixest::obs(fit) length (",
      length(idx),
      ") does not match nobs(fit) (",
      stats::nobs(fit),
      ").",
      call. = FALSE
    )
  }
  idx
}

bite_fit_single <- function(formula, data, estimator, cluster_col) {
  vcov_fml <- stats::as.formula(paste0("~", cluster_col))
  if (estimator == "fepois") {
    fixest::fepois(formula, data = data, vcov = vcov_fml, notes = FALSE)
  } else if (estimator == "feols") {
    fixest::feols(formula, data = data, vcov = vcov_fml, notes = FALSE)
  } else {
    stop("Unsupported estimator: ", estimator, call. = FALSE)
  }
}

# Exact delete-one-cluster jackknife (MacKinnon-Nielsen-Webb CV3J):
# V_CV3J = ((G - 1) / G) * sum_g (beta_(-g) - beta_bar)(beta_(-g) - beta_bar)'
bite_fit_cv3j <- function(
  data,
  formula,
  terms = NULL,
  estimator = c("fepois", "feols"),
  cluster_col = "aewr_region_id",
  spec_id,
  spec_group,
  label,
  outcome,
  exposure_name,
  fe_label,
  sample_label,
  notes = ""
) {
  estimator <- match.arg(estimator)
  fml <- if (inherits(formula, "formula")) formula else stats::as.formula(formula)
  needed_cols <- unique(c(all.vars(fml), cluster_col, "county_fips", "year"))
  cc <- bite_complete_cases(as.data.frame(data), needed_cols)
  n_complete <- nrow(cc)

  full_fit <- bite_fit_single(fml, cc, estimator, cluster_col)
  used_idx <- bite_used_rows(full_fit, cc)
  used <- cc[used_idx, , drop = FALSE]

  coef_full <- stats::coef(full_fit)
  if (is.null(terms)) {
    terms <- names(coef_full)
  }
  missing_terms <- setdiff(terms, names(coef_full))
  if (length(missing_terms) > 0L) {
    stop(
      "bite_fit_cv3j (",
      spec_id,
      "): requested term(s) absent from fit: ",
      paste(missing_terms, collapse = ", "),
      call. = FALSE
    )
  }

  clusters <- sort(unique(as.character(used[[cluster_col]])))
  G <- length(clusters)
  if (G < 2L) {
    stop("bite_fit_cv3j (", spec_id, "): fewer than 2 clusters.", call. = FALSE)
  }

  k <- length(terms)
  loo_mat <- matrix(NA_real_, nrow = G, ncol = k, dimnames = list(clusters, terms))
  for (i in seq_len(G)) {
    g <- clusters[[i]]
    sub_cc <- cc[as.character(cc[[cluster_col]]) != g, , drop = FALSE]
    loo_fit <- tryCatch(
      bite_fit_single(fml, sub_cc, estimator, cluster_col),
      error = function(e) NULL
    )
    if (!is.null(loo_fit)) {
      b_loo <- stats::coef(loo_fit)
      matched <- intersect(terms, names(b_loo))
      if (length(matched) > 0L) {
        loo_mat[i, matched] <- unname(b_loo[matched])
      }
    }
  }

  vcov_crv1 <- stats::vcov(full_fit)[terms, terms, drop = FALSE]
  se_crv1_vec <- unname(fixest::se(full_fit)[terms])
  p_crv1_vec <- unname(fixest::pvalue(full_fit)[terms])

  if (all(is.finite(loo_mat))) {
    loo_means <- colMeans(loo_mat)
    centered <- sweep(loo_mat, 2L, loo_means, FUN = "-")
    vcov_cv3j <- ((G - 1) / G) * crossprod(centered)
    dimnames(vcov_cv3j) <- list(terms, terms)
  } else {
    vcov_cv3j <- matrix(NA_real_, nrow = k, ncol = k, dimnames = list(terms, terms))
  }

  t_crit <- stats::qt(0.975, df = G - 1L)
  n_obs_val <- as.integer(stats::nobs(full_fit))
  n_counties_val <- as.integer(dplyr::n_distinct(used$county_fips))
  yr_min <- as.integer(min(used$year))
  yr_max <- as.integer(max(used$year))

  summary_rows <- lapply(seq_len(k), function(j) {
    term_j <- terms[[j]]
    b_j <- unname(coef_full[[term_j]])
    loo_j <- loo_mat[, j]
    ok_j <- is.finite(b_j) && all(is.finite(loo_j))
    se_j <- if (ok_j) sqrt(((G - 1) / G) * sum((loo_j - mean(loo_j))^2)) else NA_real_
    p_j <- if (ok_j && se_j > 0) 2 * stats::pt(-abs(b_j / se_j), df = G - 1L) else NA_real_
    tibble(
      spec_id = spec_id,
      spec_group = spec_group,
      label = label,
      estimator = estimator,
      outcome = outcome,
      exposure = exposure_name,
      fe = fe_label,
      sample = sample_label,
      cluster_var = cluster_col,
      term = term_j,
      status = if (ok_j) "estimated" else "jackknife_not_identified",
      estimate = b_j,
      se_crv1 = se_crv1_vec[[j]],
      p_crv1 = p_crv1_vec[[j]],
      se_cv3j = se_j,
      ci95_low_cv3j = if (ok_j) b_j - t_crit * se_j else NA_real_,
      ci95_high_cv3j = if (ok_j) b_j + t_crit * se_j else NA_real_,
      p_cv3j = p_j,
      mde_cv3j = if (ok_j) 2.8 * se_j else NA_real_,
      loo_min = if (ok_j) min(loo_j) else NA_real_,
      loo_max = if (ok_j) max(loo_j) else NA_real_,
      n_obs = n_obs_val,
      n_complete_rows = as.integer(n_complete),
      n_counties = n_counties_val,
      n_clusters = as.integer(G),
      year_min = yr_min,
      year_max = yr_max,
      notes = notes
    )
  })

  loo_rows <- lapply(seq_len(k), function(j) {
    tibble(
      spec_id = spec_id,
      spec_group = spec_group,
      outcome = outcome,
      cluster_var = cluster_col,
      omitted_cluster = clusters,
      term = terms[[j]],
      estimate = unname(loo_mat[, j])
    )
  })

  list(
    summary = dplyr::bind_rows(summary_rows),
    loo = dplyr::bind_rows(loo_rows),
    vcov_cv3j = vcov_cv3j,
    vcov_crv1 = vcov_crv1,
    fit = full_fit,
    used = used
  )
}

# Exact paired cluster-jackknife & score-based CRV1 inference for
# implied hours/position = theta_hours - theta_workers.
bite_implied_difference <- function(
  res_hours,
  res_workers,
  term,
  spec_id,
  spec_group,
  label,
  outcome = "implied_hours_per_position",
  exposure_name,
  fe_label,
  sample_label,
  notes = ""
) {
  row_h <- res_hours$summary %>% dplyr::filter(term == !!term)
  row_n <- res_workers$summary %>% dplyr::filter(term == !!term)
  if (nrow(row_h) != 1L || nrow(row_n) != 1L) {
    stop("bite_implied_difference: expected exactly 1 row per model for term ", term, call. = FALSE)
  }

  loo_h <- res_hours$loo %>%
    dplyr::filter(term == !!term) %>%
    dplyr::select(omitted_cluster, est_h = estimate)
  loo_n <- res_workers$loo %>%
    dplyr::filter(term == !!term) %>%
    dplyr::select(omitted_cluster, est_n = estimate)
  paired <- dplyr::inner_join(loo_h, loo_n, by = "omitted_cluster")
  if (nrow(paired) != nrow(loo_h) || nrow(paired) != nrow(loo_n)) {
    stop("bite_implied_difference: cluster sets differ between hours and workers fits.", call. = FALSE)
  }

  G <- nrow(paired)
  est_diff <- row_h$estimate[[1]] - row_n$estimate[[1]]
  loo_diff <- paired$est_h - paired$est_n
  se_cv3j_diff <- sqrt(((G - 1) / G) * sum((loo_diff - mean(loo_diff))^2))
  t_crit <- stats::qt(0.975, df = G - 1L)
  p_cv3j_diff <- 2 * stats::pt(-abs(est_diff / se_cv3j_diff), df = G - 1L)

  # Exact paired CRV1 SE from cluster-aggregated observation scores scaled to
  # match fixest's marginal CRV1 SEs (exact for scalar parameter k = 1).
  se_crv1_diff <- tryCatch({
    cl_col <- row_h$cluster_var[[1]]
    sc_h <- as.matrix(fixest::scores(res_hours$fit))[, term]
    sc_n <- as.matrix(fixest::scores(res_workers$fit))[, term]
    g_h <- tapply(sc_h, as.character(res_hours$used[[cl_col]]), sum)
    g_n <- tapply(sc_n, as.character(res_workers$used[[cl_col]]), sum)
    cl_names <- sort(intersect(names(g_h), names(g_n)))
    stopifnot(length(cl_names) == G)
    psi_h <- row_h$se_crv1[[1]] * unname(g_h[cl_names]) / sqrt(sum(g_h[cl_names]^2))
    psi_n <- row_n$se_crv1[[1]] * unname(g_n[cl_names]) / sqrt(sum(g_n[cl_names]^2))
    sqrt(sum((psi_h - psi_n)^2))
  }, error = function(e) NA_real_)
  p_crv1_diff <- if (is.finite(se_crv1_diff) && se_crv1_diff > 0) {
    2 * stats::pt(-abs(est_diff / se_crv1_diff), df = G - 1L)
  } else {
    NA_real_
  }

  summary_tbl <- tibble(
    spec_id = spec_id,
    spec_group = spec_group,
    label = label,
    estimator = "fepois_diff",
    outcome = outcome,
    exposure = exposure_name,
    fe = fe_label,
    sample = sample_label,
    cluster_var = row_h$cluster_var[[1]],
    term = term,
    status = "estimated",
    estimate = est_diff,
    se_crv1 = se_crv1_diff,
    p_crv1 = p_crv1_diff,
    se_cv3j = se_cv3j_diff,
    ci95_low_cv3j = est_diff - t_crit * se_cv3j_diff,
    ci95_high_cv3j = est_diff + t_crit * se_cv3j_diff,
    p_cv3j = p_cv3j_diff,
    mde_cv3j = 2.8 * se_cv3j_diff,
    loo_min = min(loo_diff),
    loo_max = max(loo_diff),
    n_obs = row_h$n_obs[[1]],
    n_complete_rows = row_h$n_complete_rows[[1]],
    n_counties = row_h$n_counties[[1]],
    n_clusters = as.integer(G),
    year_min = row_h$year_min[[1]],
    year_max = row_h$year_max[[1]],
    notes = notes
  )

  loo_tbl <- tibble(
    spec_id = spec_id,
    spec_group = spec_group,
    outcome = outcome,
    cluster_var = row_h$cluster_var[[1]],
    omitted_cluster = paired$omitted_cluster,
    term = term,
    estimate = loo_diff
  )

  list(summary = summary_tbl, loo = loo_tbl)
}

bite_joint_wald <- function(beta_vec, V_mat, df2) {
  q <- length(beta_vec)
  if (q == 0L || any(!is.finite(beta_vec)) || any(!is.finite(V_mat))) {
    return(list(wald = NA_real_, f_stat = NA_real_, df1 = q, df2 = df2, p_value = NA_real_))
  }
  w_val <- as.numeric(t(beta_vec) %*% solve(V_mat, beta_vec))
  f_val <- w_val / q
  p_val <- stats::pf(f_val, df1 = q, df2 = df2, lower.tail = FALSE)
  list(wald = w_val, f_stat = f_val, df1 = as.integer(q), df2 = as.integer(df2), p_value = p_val)
}

# Effective cluster count G_eff (Fisher-information Herfindahl) and cluster
# leverage shares for a scalar FE-PPML regression.
bite_cluster_leverage <- function(
  fit,
  used,
  regressor_col,
  fe_rhs = "county_fips + aewr_region_id^year",
  cluster_col = "aewr_region_id",
  outcome_col
) {
  work <- as.data.frame(used)
  work$.ppml_mu <- as.numeric(stats::fitted(fit))
  stopifnot(length(work$.ppml_mu) == nrow(work), all(is.finite(work$.ppml_mu)), all(work$.ppml_mu >= 0))

  proj_fml <- stats::as.formula(paste0(regressor_col, " ~ 1 | ", fe_rhs))
  proj_fit <- fixest::feols(proj_fml, data = work, weights = ~.ppml_mu, notes = FALSE)
  x_tilde <- as.numeric(stats::resid(proj_fit))
  work$.fisher_info <- work$.ppml_mu * (x_tilde^2)

  by_cluster <- work %>%
    dplyr::group_by(cluster_id = as.character(.data[[cluster_col]])) %>%
    dplyr::summarise(
      n_obs = dplyr::n(),
      n_counties = dplyr::n_distinct(county_fips),
      outcome_total = sum(.data[[outcome_col]], na.rm = TRUE),
      fisher_info = sum(.fisher_info, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      fisher_info_share = fisher_info / sum(fisher_info),
      outcome_share = outcome_total / sum(outcome_total),
      county_share = n_counties / sum(n_counties)
    ) %>%
    dplyr::arrange(as.integer(cluster_id))

  g_eff_fisher <- 1 / sum(by_cluster$fisher_info_share^2)
  g_eff_outcome <- 1 / sum(by_cluster$outcome_share^2)
  g_eff_county <- 1 / sum(by_cluster$county_share^2)
  ca_share <- by_cluster$fisher_info_share[by_cluster$cluster_id == "17"]
  ca_share <- if (length(ca_share) == 1L) ca_share else NA_real_

  coef_idx <- match(regressor_col, names(stats::coef(fit)))
  hess_info <- 1 / as.numeric(fit$cov.unscaled[coef_idx, coef_idx])
  rel_err <- abs(sum(by_cluster$fisher_info) - hess_info) / hess_info

  list(
    by_cluster = by_cluster,
    g_eff_fisher = g_eff_fisher,
    g_eff_outcome = g_eff_outcome,
    g_eff_county = g_eff_county,
    ca_fisher_share = ca_share,
    fisher_info_total = sum(by_cluster$fisher_info),
    hessian_rel_err = rel_err
  )
}

# Exact residualized variance split of regressor under C+RY vs C+RY+E*Y vs C+SY.
bite_variance_decomposition <- function(
  data,
  regressor_col = "E_p25_x_ln_aewr",
  exposure_col = "E_p25",
  weights_vec = NULL
) {
  work <- as.data.frame(data)
  x <- work[[regressor_col]]
  if (is.null(weights_vec)) {
    w <- rep(1, nrow(work))
    mu_x <- mean(x)
    var_tot <- mean((x - mu_x)^2)
    f_cry <- fixest::feols(
      stats::as.formula(paste0(regressor_col, " ~ 1 | county_fips + aewr_region_id^year")),
      data = work,
      notes = FALSE
    )
    f_cry_ey <- fixest::feols(
      stats::as.formula(paste0(
        regressor_col, " ~ i(year, ", exposure_col, ", ref = ", BITE_EVENT_REF_YEAR, ") | county_fips + aewr_region_id^year"
      )),
      data = work,
      notes = FALSE
    )
    f_csy <- fixest::feols(
      stats::as.formula(paste0(regressor_col, " ~ 1 | county_fips + state_fips^year")),
      data = work,
      notes = FALSE
    )
  } else {
    w <- as.numeric(weights_vec)
    work$.w <- w
    mu_x <- stats::weighted.mean(x, w)
    var_tot <- sum(w * (x - mu_x)^2) / sum(w)
    f_cry <- fixest::feols(
      stats::as.formula(paste0(regressor_col, " ~ 1 | county_fips + aewr_region_id^year")),
      data = work,
      weights = ~.w,
      notes = FALSE
    )
    f_cry_ey <- fixest::feols(
      stats::as.formula(paste0(
        regressor_col, " ~ i(year, ", exposure_col, ", ref = ", BITE_EVENT_REF_YEAR, ") | county_fips + aewr_region_id^year"
      )),
      data = work,
      weights = ~.w,
      notes = FALSE
    )
    f_csy <- fixest::feols(
      stats::as.formula(paste0(regressor_col, " ~ 1 | county_fips + state_fips^year")),
      data = work,
      weights = ~.w,
      notes = FALSE
    )
  }

  r_cry <- as.numeric(stats::resid(f_cry))
  r_cry_ey <- as.numeric(stats::resid(f_cry_ey))
  r_csy <- as.numeric(stats::resid(f_csy))

  var_cry <- sum(w * r_cry^2) / sum(w)
  var_cry_ey <- sum(w * r_cry_ey^2) / sum(w)
  var_csy <- sum(w * r_csy^2) / sum(w)

  tibble(
    weighting = if (is.null(weights_vec)) "unweighted" else "ppml_fitted_hours",
    var_total = var_tot,
    sd_total = sqrt(var_tot),
    var_resid_C_RY = var_cry,
    sd_resid_C_RY = sqrt(var_cry),
    r2_C_RY = 1 - var_cry / var_tot,
    var_resid_C_RY_EY = var_cry_ey,
    sd_resid_C_RY_EY = sqrt(var_cry_ey),
    r2_C_RY_EY = 1 - var_cry_ey / var_tot,
    share_1a_of_1b = var_cry_ey / var_cry,
    var_resid_C_SY = var_csy,
    sd_resid_C_SY = sqrt(var_csy),
    r2_C_SY = 1 - var_csy / var_tot,
    share_within_SY_of_1b = var_csy / var_cry
  )
}

# State-year residual variance ratio vs pure county noise on primary FE-PPML
# deviance residuals (nested within Region x Year).
bite_state_year_noise_ratio <- function(fit, used, outcome_col) {
  y <- as.numeric(used[[outcome_col]])
  mu <- pmax(as.numeric(stats::fitted(fit)), 1e-12)
  dev_sq <- 2 * (ifelse(y > 0, y * log(y / mu), 0) - (y - mu))
  dev_resid <- sign(y - mu) * sqrt(pmax(dev_sq, 0))

  work <- data.frame(
    dev_resid = dev_resid,
    aewr_region_id = as.character(used$aewr_region_id),
    state_fips = as.character(used$state_fips),
    year = as.integer(used$year),
    stringsAsFactors = FALSE
  )

  fit_ry <- fixest::feols(dev_resid ~ 1 | aewr_region_id^year, data = work, notes = FALSE)
  fit_sy <- fixest::feols(dev_resid ~ 1 | state_fips^year, data = work, notes = FALSE)

  ssr_ry <- sum(stats::resid(fit_ry)^2)
  ssr_sy <- sum(stats::resid(fit_sy)^2)
  ss_sy <- max(ssr_ry - ssr_sy, 0)

  n_ry <- dplyr::n_distinct(paste(work$aewr_region_id, work$year, sep = "_"))
  n_sy <- dplyr::n_distinct(paste(work$state_fips, work$year, sep = "_"))
  df_sy <- as.integer(n_sy - n_ry)
  df_county <- as.integer(nrow(work) - n_sy)

  ms_sy <- ss_sy / df_sy
  ms_county <- ssr_sy / df_county
  f_stat <- ms_sy / ms_county
  p_val <- stats::pf(f_stat, df1 = df_sy, df2 = df_county, lower.tail = FALSE)
  r2_sy_within_ry <- ss_sy / ssr_ry
  avg_counties_per_sy <- nrow(work) / n_sy
  re_var_ratio <- max(ms_sy - ms_county, 0) / (avg_counties_per_sy * ms_county)

  tibble(
    outcome = outcome_col,
    n_obs = as.integer(nrow(work)),
    df_state_year_extra = df_sy,
    df_county_noise = df_county,
    r2_state_year_within_ry = r2_sy_within_ry,
    ms_ratio_sy_to_county = f_stat,
    re_var_ratio_sy_to_county = re_var_ratio,
    f_p_value = p_val
  )
}
# docs-ground:end bite-estimation-contract

# --- Formatting and LaTeX table helpers ---

bite_stars <- function(p) {
  if (!is.finite(p)) {
    return("")
  }
  if (p < 0.01) {
    return("***")
  }
  if (p < 0.05) {
    return("**")
  }
  if (p < 0.10) {
    return("*")
  }
  ""
}

bite_fmt <- function(x, digits = 3L) {
  if (!is.finite(x)) {
    return("---")
  }
  sprintf(paste0("%.", digits, "f"), x)
}

bite_fmt_int <- function(x) {
  if (!is.finite(x)) {
    return("---")
  }
  formatC(as.integer(round(x)), format = "d", big.mark = ",")
}

bite_md_table <- function(df, digits = 4L) {
  d <- as.data.frame(df)
  if (nrow(d) == 0L) {
    return("None.")
  }
  for (nm in names(d)) {
    if (is.numeric(d[[nm]])) {
      d[[nm]] <- ifelse(
        is.na(d[[nm]]),
        "—",
        formatC(d[[nm]], digits = digits, format = "fg")
      )
    }
    d[[nm]] <- gsub("|", "\\|", ifelse(is.na(d[[nm]]), "—", as.character(d[[nm]])), fixed = TRUE)
  }
  header <- paste0("| ", paste(names(d), collapse = " | "), " |")
  sep <- paste0("| ", paste(rep("---", ncol(d)), collapse = " | "), " |")
  rows <- apply(d, 1L, function(r) paste0("| ", paste(r, collapse = " | "), " |"))
  paste(c(header, sep, rows), collapse = "\n")
}

bite_write_main_tex <- function(main_df, var_decomp_df, path) {
  get_row <- function(id) {
    r <- main_df %>% dplyr::filter(spec_id == !!id)
    stopifnot(nrow(r) == 1L)
    r[1L, ]
  }

  cols <- c(
    "main_p25_hours",
    "state_year_p25_hours",
    "xpre_p25_hours",
    "oews_hours",
    "sf_log_hours",
    "spec1a_p25_hours"
  )
  cols_w <- c(
    "main_p25_workers",
    "state_year_p25_workers",
    "xpre_p25_workers",
    "oews_workers",
    "sf_log_workers",
    "spec1a_p25_workers"
  )

  fmt_panel_rows <- function(ids, row_label) {
    rs <- lapply(ids, get_row)
    est_line <- paste0(
      row_label, " & ",
      paste(vapply(rs, function(r) paste0(bite_fmt(r$estimate, 3L), bite_stars(r$p_cv3j)), character(1)), collapse = " & "),
      " \\\\"
    )
    crv1_line <- paste0(
      "\\quad (Region CRV1 SE) & ",
      paste(vapply(rs, function(r) paste0("(", bite_fmt(r$se_crv1, 3L), ")"), character(1)), collapse = " & "),
      " \\\\"
    )
    cv3j_line <- paste0(
      "\\quad [Region CV3J SE] & ",
      paste(vapply(rs, function(r) paste0("[", bite_fmt(r$se_cv3j, 3L), "]"), character(1)), collapse = " & "),
      " \\\\"
    )
    mde_line <- paste0(
      "\\quad MDE ($2.8 \\times \\text{CV3J}$) & ",
      paste(vapply(rs, function(r) bite_fmt(r$mde_cv3j, 3L), character(1)), collapse = " & "),
      " \\\\"
    )
    loo_line <- paste0(
      "\\quad Region LOO [min, max] & ",
      paste(vapply(rs, function(r) paste0("[", bite_fmt(r$loo_min, 2L), ", ", bite_fmt(r$loo_max, 2L), "]"), character(1)), collapse = " & "),
      " \\\\"
    )
    obs_line <- paste0(
      "\\quad Observations (Counties) & ",
      paste(vapply(rs, function(r) paste0(bite_fmt_int(r$n_obs), " (", bite_fmt_int(r$n_counties), ")"), character(1)), collapse = " & "),
      " \\\\"
    )
    c(est_line, crv1_line, cv3j_line, mde_line, loo_line, obs_line)
  }

  r_ge5 <- get_row("main_p25_any_ge5")
  r_dur <- get_row("main_p25_hours_per_pos")
  unw_share <- var_decomp_df$share_1a_of_1b[var_decomp_df$weighting == "unweighted"][[1]]
  wt_share <- var_decomp_df$share_1a_of_1b[var_decomp_df$weighting == "ppml_fitted_hours"][[1]]

  lines <- c(
    "\\begin{table}[!htbp]",
    "\\centering",
    "\\caption{Full-Panel FE-PPML Bite Estimates of AEWR Responsiveness (2011--2025)}",
    "\\label{tab:bite_main}",
    "\\small",
    "\\begin{tabular}{lcccccc}",
    "\\toprule",
    " & (1) & (2) & (3) & (4) & (5) & (6) \\\\",
    " & Primary & State $\\times$ Yr & $+X_{\\text{pre}} \\times$ Yr & OEWS & Shortfall & 1a-PPML \\\\",
    " & $E_{p25}$ & $E_{p25}$ & $E_{p25}$ & $E_{\\text{oews}}$ & $E_{\\text{sf}}$ & $+E_{p25} \\times$ Yr \\\\",
    "\\midrule",
    "\\multicolumn{7}{l}{\\textit{Panel A: Certified H-2A Hours (FE-PPML)}} \\\\",
    fmt_panel_rows(cols, "$E_c \\times \\ln(\\text{AEWR}_{rt})$"),
    "\\midrule",
    "\\multicolumn{7}{l}{\\textit{Panel B: Certified H-2A Positions (FE-PPML)}} \\\\",
    fmt_panel_rows(cols_w, "$E_c \\times \\ln(\\text{AEWR}_{rt})$"),
    "\\midrule",
    "\\multicolumn{7}{l}{\\textit{Panel C: Primary Extensive Margin and Implied Duration (Col. 1 FE)}} \\\\",
    paste0(
      "$\\mathbf{1}[\\text{Positions} \\ge 5]$ (LPM) & ",
      bite_fmt(r_ge5$estimate, 3L), bite_stars(r_ge5$p_cv3j),
      " & \\multicolumn{5}{l}{CRV1 (", bite_fmt(r_ge5$se_crv1, 3L), "), CV3J [", bite_fmt(r_ge5$se_cv3j, 3L),
      "], MDE ", bite_fmt(r_ge5$mde_cv3j, 3L), ", LOO [", bite_fmt(r_ge5$loo_min, 2L), ", ", bite_fmt(r_ge5$loo_max, 2L),
      "], $N=$ ", bite_fmt_int(r_ge5$n_obs), " (", bite_fmt_int(r_ge5$n_counties), ")} \\\\"
    ),
    paste0(
      "Implied $\\ln(\\text{Hours/Pos.})$ ($\\hat{\\theta}_H - \\hat{\\theta}_N$) & ",
      bite_fmt(r_dur$estimate, 3L), bite_stars(r_dur$p_cv3j),
      " & \\multicolumn{5}{l}{CRV1 (", bite_fmt(r_dur$se_crv1, 3L), "), CV3J [", bite_fmt(r_dur$se_cv3j, 3L),
      "], MDE ", bite_fmt(r_dur$mde_cv3j, 3L), ", LOO [", bite_fmt(r_dur$loo_min, 2L), ", ", bite_fmt(r_dur$loo_max, 2L), "]} \\\\"
    ),
    "\\midrule",
    "County FE & Yes & Yes & Yes & Yes & Yes & Yes \\\\",
    "Region $\\times$ Year FE & Yes & --- & Yes & Yes & Yes & Yes \\\\",
    "State $\\times$ Year FE & No & Yes & No & No & No & No \\\\",
    "Predetermined $X_{\\text{pre}} \\times$ Year & No & No & Yes & No & No & No \\\\",
    "$E_{p25} \\times$ Year FE & No & No & No & No & No & Yes \\\\",
    "AEWR Region Clusters ($G$) & 17 & 17 & 17 & 17 & 17 & 17 \\\\",
    "\\bottomrule",
    "\\end{tabular}",
    "\\begin{minipage}{0.96\\linewidth}",
    "\\vspace{0.4em}\\footnotesize",
    paste0(
      "\\textit{Notes:} Sample covers 2011--2025 contiguous U.S.\\ counties (2009 excluded because statutory AEWR is missing; ",
      "2025 start-year H-2A counts are provisional pending FY2027 decisions). Exposures $E_c$ are frozen 2008/2010 averages ",
      "standardized to mean 0, SD 1 within each of the 17 AEWR regions. Coefficients report the differential log-point (semi-elasticity) ",
      "response per $+1$ within-region SD of baseline exposure per unit increase in $\\ln(\\text{AEWR}_{rt})$ — not an aggregate elasticity. ",
      "Parentheses show region-clustered CRV1 SEs; brackets show exact delete-one-region jackknife (CV3J) SEs with stars from CV3J $t(G-1)$ ",
      "p-values ($^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$). Col.\\ 3 interacts predetermined 2008--2010 covariates ",
      "($\\ln\\text{Pop}_{2010}$, farm employment share, $\\ln(1+\\text{EmpFarm})$) with year indicators, preserving 2023--2025. ",
      "Col.\\ 6 absorbs $E_{p25} \\times$ Year FE, retaining ", bite_fmt(100 * unw_share, 1L), "\\% unweighted (",
      bite_fmt(100 * wt_share, 1L), "\\% PPML-hours-weighted) of the Col.\\ 1 residualized regressor variance."
    ),
    "\\end{minipage}",
    "\\end{table}"
  )

  writeLines(lines, path)
}

bite_write_diag_tex <- function(diag_df, path) {
  get_row <- function(id, term_val = NULL) {
    r <- diag_df %>% dplyr::filter(spec_id == !!id)
    if (!is.null(term_val)) {
      r <- r %>% dplyr::filter(term == !!term_val)
    }
    stopifnot(nrow(r) == 1L)
    r[1L, ]
  }

  fmt_row <- function(r, label_tex) {
    paste0(
      label_tex, " & ",
      bite_fmt(r$estimate, 3L), bite_stars(r$p_cv3j), " & ",
      "(", bite_fmt(r$se_crv1, 3L), ") & ",
      "[", bite_fmt(r$se_cv3j, 3L), "] & ",
      bite_fmt(r$mde_cv3j, 3L), " & ",
      bite_fmt_int(r$n_obs), " & ",
      bite_fmt_int(r$n_clusters), " \\\\"
    )
  }

  r_q2 <- get_row("spt_quartile_hours", "E_p25_quartile::Q2:ln_aewr")
  r_q3 <- get_row("spt_quartile_hours", "E_p25_quartile::Q3:ln_aewr")
  r_q4 <- get_row("spt_quartile_hours", "E_p25_quartile::Q4:ln_aewr")
  r_ppi_aewr <- get_row("ppi_decouple_hours", "E_p25_x_ln_aewr")
  r_ppi_wpu <- get_row("ppi_decouple_hours", "E_p25_x_ln_ppi_2012")
  r_gt0_full <- get_row("addendum_b_gt0_2011_2025")
  r_gt0_pre <- get_row("addendum_b_gt0_2011_2019")
  r_ge5_full <- get_row("addendum_b_ge5_2011_2025")
  r_ge5_pre <- get_row("addendum_b_ge5_2011_2019")
  r_emp_pre <- get_row("employers_balanced_2011_2019")
  r_emp_post <- get_row("employers_balanced_2020_2025")
  r_notop1_h <- get_row("drop_top1pct_hours")
  r_notop1_w <- get_row("drop_top1pct_workers")
  r_fisher <- get_row("farm_fisher_price")
  r_lshare <- get_row("farm_labor_share")
  r_femp_ppml <- get_row("farm_emp_ppml")
  r_femp_ols <- get_row("farm_emp_ols")
  r_dom_h <- get_row("domestic_floor_hours")
  r_dom_w <- get_row("domestic_floor_workers")

  lines <- c(
    "\\begin{table}[!htbp]",
    "\\centering",
    "\\caption{Bite Design Diagnostics, Farm Outcomes, and Standalone Domestic-Floor Margin}",
    "\\label{tab:bite_diagnostics}",
    "\\small",
    "\\begin{tabular}{lcccccc}",
    "\\toprule",
    "Specification / Outcome & Estimate & CRV1 SE & CV3J SE & MDE & Obs. & Clusters \\\\",
    "\\midrule",
    "\\multicolumn{7}{l}{\\textit{Panel A: Quartile SPT Monotonicity (Certified Hours, vs.\\ Q1, 2011--2025)}} \\\\",
    fmt_row(r_q2, "Q2($E_{p25}$) $\\times \\ln(\\text{AEWR})$"),
    fmt_row(r_q3, "Q3($E_{p25}$) $\\times \\ln(\\text{AEWR})$"),
    fmt_row(r_q4, "Q4($E_{p25}$) $\\times \\ln(\\text{AEWR})$"),
    "\\midrule",
    "\\multicolumn{7}{l}{\\textit{Panel B: WPU01 Farm-Price Decoupling \\& Concentration (2011--2025)}} \\\\",
    fmt_row(r_ppi_aewr, "Hours: $E_{p25} \\times \\ln(\\text{AEWR})$ (with PPI control)"),
    fmt_row(r_ppi_wpu, "Hours: $E_{p25} \\times \\ln(\\text{WPU01 PPI})$"),
    fmt_row(r_notop1_h, "Hours: drop top 1\\% H-2A counties"),
    fmt_row(r_notop1_w, "Positions: drop top 1\\% H-2A counties"),
    "\\midrule",
    "\\multicolumn{7}{l}{\\textit{Panel C: Addendum-B Extensive-Margin \\& Employer-Linkage Checks}} \\\\",
    fmt_row(r_gt0_full, "$\\mathbf{1}[\\text{Positions} > 0]$ (LPM, 2011--2025)"),
    fmt_row(r_gt0_pre, "$\\mathbf{1}[\\text{Positions} > 0]$ (LPM, 2011--2019)"),
    fmt_row(r_ge5_full, "$\\mathbf{1}[\\text{Positions} \\ge 5]$ (LPM, 2011--2025)"),
    fmt_row(r_ge5_pre, "$\\mathbf{1}[\\text{Positions} \\ge 5]$ (LPM, 2011--2019)"),
    fmt_row(r_emp_pre, "Balanced worksite employers (PPML, 2011--2019)"),
    fmt_row(r_emp_post, "Balanced worksite employers (PPML, 2020--2025)"),
    "\\midrule",
    "\\multicolumn{7}{l}{\\textit{Panel D: Agricultural Outcomes ($E_{p25} \\times \\ln(\\text{AEWR})$, County + Region $\\times$ Year FE)}} \\\\",
    fmt_row(r_fisher, "Fisher output price index (real, OLS, 2011--2024)"),
    fmt_row(r_lshare, "Farm labor share of prod.\\ exp.\\ (OLS, 2011--2022)"),
    fmt_row(r_femp_ppml, "BEA farm employment (PPML, 2011--2022)"),
    fmt_row(r_femp_ols, "BEA farm employment (OLS, 2011--2022)"),
    "\\midrule",
    "\\multicolumn{7}{l}{\\textit{Panel E: Standalone Domestic-Floor Margin ($\\ln(\\text{Prevailing Ag Min Wage})$, State Clusters, 2011--2025)}} \\\\",
    fmt_row(r_dom_h, "Certified H-2A hours (PPML, all 48 states)"),
    fmt_row(r_dom_w, "Certified H-2A positions (PPML, all 48 states)"),
    "\\bottomrule",
    "\\end{tabular}",
    "\\begin{minipage}{0.96\\linewidth}",
    "\\vspace{0.4em}\\footnotesize",
    paste0(
      "\\textit{Notes:} All models include County and Region $\\times$ Year fixed effects. Panels A--D cluster at the AEWR region level ",
      "($G=17$); Panel E clusters at the state level ($G=48$ states retained in the panel; 46 multi-state-region states provide within-region ",
      "variation because CA and FL are single-state AEWR regions) and reports the standalone reduced-form response to the statutory domestic ",
      "agricultural minimum-wage floor (never translated into an AEWR elasticity). Stars reflect CV3J $t(G-1)$ p-values ",
      "($^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$); MDE $= 2.8 \\times \\text{CV3J SE}$."
    ),
    "\\end{minipage}",
    "\\end{table}"
  )

  writeLines(lines, path)
}
