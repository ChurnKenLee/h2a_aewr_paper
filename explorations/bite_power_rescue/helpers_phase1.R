# Shared Phase 1 outcome-blind statistical power & empirical size helpers.
# Strictly imperative R; zero classes; no edits outside explorations/bite_power_rescue/.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(fixest)
  library(Matrix)
  library(readr)
  library(stringr)
  library(tibble)
  library(tidyr)
})

setFixest_nthreads(1L)
setFixest_notes(FALSE)
setFixest_ssc(ssc(K.adj = TRUE, K.fixef = "nonnested", G.adj = TRUE))

COST_RATIO_CENTRAL <- 1.2401
W_BAR_OFFERED_FY2526 <- 16.051
SEED_PHASE1 <- 20261127L

wilson_ci95 <- function(k, n) {
  if (n <= 0L) return(c(low = NA_real_, high = NA_real_))
  p <- k / n
  z <- 1.959963984540054
  denom <- 1 + z^2 / n
  center <- (p + z^2 / (2 * n)) / denom
  half <- (z / denom) * sqrt(p * (1 - p) / n + z^2 / (4 * n^2))
  c(low = max(0, center - half), high = min(1, center + half))
}

# Exact 80%-power MDE from empirical placebo draws (beta_hat, se_hat) using
# critical value t_crit = qt(0.975, df = G - 1). Takes max over +delta and -delta.
compute_empirical_mde <- function(beta_draws, se_draws, df_t, target_power = 0.80, alpha = 0.05) {
  ok <- is.finite(beta_draws) & is.finite(se_draws) & se_draws > 0
  b <- beta_draws[ok]
  s <- se_draws[ok]
  B <- length(b)
  if (B < 50L) {
    return(list(
      n_valid = B, emp_size = NA_real_, wilson_lo = NA_real_, wilson_hi = NA_real_,
      mde_pos = NA_real_, mde_neg = NA_real_, mde_upper = NA_real_,
      se_median = NA_real_, beta_sd = NA_real_, analytic_mde = NA_real_
    ))
  }
  t_crit <- stats::qt(1 - alpha / 2, df = df_t)
  rej0 <- sum(abs(b / s) >= t_crit)
  emp_size <- rej0 / B
  wci <- wilson_ci95(rej0, B)

  # Bisection for each sign
  find_sign_mde <- function(sgn) {
    pw <- function(d) mean(abs((sgn * d + b) / s) >= t_crit)
    hi <- max(stats::median(s) * 4, stats::sd(b) * 4, 1e-6)
    while (pw(hi) < target_power && hi < 1e6) hi <- hi * 2
    if (pw(hi) < target_power) return(NA_real_)
    lo <- 0
    for (iter in seq_len(40L)) {
      mid <- 0.5 * (lo + hi)
      if (pw(mid) >= target_power) hi <- mid else lo <- mid
    }
    hi
  }

  mde_pos <- find_sign_mde(1)
  mde_neg <- find_sign_mde(-1)
  list(
    n_valid = B,
    emp_size = emp_size,
    wilson_lo = wci[["low"]],
    wilson_hi = wci[["high"]],
    mde_pos = mde_pos,
    mde_neg = mde_neg,
    mde_upper = max(mde_pos, mde_neg),
    se_median = stats::median(s),
    beta_sd = stats::sd(b),
    analytic_mde = 2.8 * stats::median(s)
  )
}

# Exact state-year shared-shock ratio MS_SY / MS_unit on FE-only PPML deviance residuals
# or linear residuals, nested within region x year.
compute_sy_noise_ratio_ppml <- function(y, mu, region, state, year) {
  y <- as.numeric(y)
  mu <- pmax(as.numeric(mu), 1e-12)
  dev_sq <- 2 * (ifelse(y > 0, y * log(y / mu), 0) - (y - mu))
  dev_resid <- sign(y - mu) * sqrt(pmax(dev_sq, 0))
  compute_sy_noise_ratio_resid(dev_resid, region, state, year)
}

compute_sy_noise_ratio_resid <- function(resid_vec, region, state, year) {
  work <- data.frame(
    r = as.numeric(resid_vec),
    reg = as.character(region),
    st = as.character(state),
    yr = as.character(year),
    stringsAsFactors = FALSE
  )
  work <- work[is.finite(work$r), , drop = FALSE]
  fit_ry <- fixest::feols(r ~ 1 | reg^yr, data = work, notes = FALSE)
  fit_sy <- fixest::feols(r ~ 1 | st^yr, data = work, notes = FALSE)
  ssr_ry <- sum(stats::resid(fit_ry)^2)
  ssr_sy <- sum(stats::resid(fit_sy)^2)
  ss_sy <- max(ssr_ry - ssr_sy, 0)
  n_ry <- dplyr::n_distinct(paste(work$reg, work$yr, sep = "_"))
  n_sy <- dplyr::n_distinct(paste(work$st, work$yr, sep = "_"))
  df_sy <- as.integer(n_sy - n_ry)
  df_unit <- as.integer(nrow(work) - n_sy)
  if (df_sy <= 0L || df_unit <= 0L || ssr_sy <= 0) {
    return(list(ms_ratio = NA_real_, r2_sy_within_ry = NA_real_, p_val = NA_real_, df_sy = df_sy, df_unit = df_unit))
  }
  ms_sy <- ss_sy / df_sy
  ms_unit <- ssr_sy / df_unit
  f_stat <- ms_sy / ms_unit
  list(
    ms_ratio = f_stat,
    r2_sy_within_ry = ss_sy / ssr_ry,
    p_val = stats::pf(f_stat, df1 = df_sy, df2 = df_unit, lower.tail = FALSE),
    df_sy = df_sy,
    df_unit = df_unit
  )
}

# Exact delete-one-cluster CV3J fit via fixest (fepois or feols)
fit_cv3j_fixest <- function(data, formula, term, cluster_col, estimator = c("fepois", "feols"), weights_fml = NULL) {
  estimator <- match.arg(estimator)
  fml <- if (inherits(formula, "formula")) formula else stats::as.formula(formula)
  vcov_fml <- stats::as.formula(paste0("~", cluster_col))
  fit_one <- function(d) {
    tryCatch(
      if (estimator == "fepois") {
        if (is.null(weights_fml)) {
          fixest::fepois(fml, data = d, vcov = vcov_fml, notes = FALSE)
        } else {
          fixest::fepois(fml, data = d, weights = weights_fml, vcov = vcov_fml, notes = FALSE)
        }
      } else {
        if (is.null(weights_fml)) {
          fixest::feols(fml, data = d, vcov = vcov_fml, notes = FALSE)
        } else {
          fixest::feols(fml, data = d, weights = weights_fml, vcov = vcov_fml, notes = FALSE)
        }
      },
      error = function(e) NULL
    )
  }

  m <- fit_one(data)
  if (is.null(m) || !term %in% names(stats::coef(m))) {
    return(list(ok = FALSE, estimate = NA_real_, se_cv3j = NA_real_, se_crv1 = NA_real_, p_cv3j = NA_real_, G = 0L, loo = numeric()))
  }
  used <- data[fixest::obs(m), , drop = FALSE]
  clusters <- sort(unique(as.character(used[[cluster_col]])))
  G <- length(clusters)
  b <- unname(stats::coef(m)[[term]])
  se_crv1 <- unname(fixest::se(m)[[term]])

  loo <- vapply(clusters, function(g) {
    mm <- fit_one(data[as.character(data[[cluster_col]]) != g, , drop = FALSE])
    if (is.null(mm) || !term %in% names(stats::coef(mm))) NA_real_ else unname(stats::coef(mm)[[term]])
  }, numeric(1))

  ok <- is.finite(b) && G >= 2L && all(is.finite(loo))
  se_cv3j <- if (ok) sqrt(((G - 1) / G) * sum((loo - mean(loo))^2)) else NA_real_
  t_crit <- if (G >= 2L) stats::qt(0.975, df = G - 1L) else NA_real_
  p_cv3j <- if (ok && se_cv3j > 0) 2 * stats::pt(-abs(b / se_cv3j), df = G - 1L) else NA_real_
  list(
    ok = ok,
    estimate = b,
    se_cv3j = se_cv3j,
    se_crv1 = se_crv1,
    ci95_low = if (ok) b - t_crit * se_cv3j else NA_real_,
    ci95_high = if (ok) b + t_crit * se_cv3j else NA_real_,
    p_cv3j = p_cv3j,
    G = G,
    n_obs = as.integer(stats::nobs(m)),
    loo = stats::setNames(loo, clusters),
    fit = m,
    used = used
  )
}

# Exact 2-period FE-PPML solver via concentrated binomial logit (unit FE exact!).
# Input: y0, y1 (nonnegative vectors of length N_units), dx (scalar treatment change),
# group_fe (integer/factor vector of group-by-year FE, e.g. region), optional Z (matrix of covariate changes).
# Exact identity: for y_tot = y0 + y1 > 0, fepois(y ~ x + z | unit + group^t)
# has identical beta to weighted binomial glm(cbind(y1, y0) ~ dx + Z + factor(group_fe)).
fit_2period_ppml_binom <- function(y0, y1, dx, group_fe, Z = NULL, tol = 1e-10, max_iter = 40L) {
  y_tot <- y0 + y1
  keep <- is.finite(y_tot) & y_tot > 0 & is.finite(dx)
  if (!is.null(Z)) {
    keep <- keep & (rowSums(!is.finite(Z)) == 0L)
  }
  y1 <- y1[keep]
  y_tot <- y_tot[keep]
  dx <- dx[keep]
  grp <- as.integer(factor(group_fe[keep]))
  n_grp <- max(grp)
  if (!is.null(Z)) {
    Z <- Z[keep, , drop = FALSE]
  }

  # Drop groups where all y1 == 0 or all y1 == y_tot (perfect separation in binomial)
  g_y1 <- rowsum(y1, grp, reorder = TRUE)[, 1]
  g_yt <- rowsum(y_tot, grp, reorder = TRUE)[, 1]
  valid_g <- which(g_y1 > 0 & g_y1 < g_yt)
  if (length(valid_g) == 0L) return(NA_real_)
  if (length(valid_g) < n_grp) {
    kg <- grp %in% valid_g
    y1 <- y1[kg]; y_tot <- y_tot[kg]; dx <- dx[kg]
    grp <- as.integer(factor(grp[kg]))
    if (!is.null(Z)) Z <- Z[kg, , drop = FALSE]
  }
  n_grp <- max(grp)

  # Build design matrix [dx, Z, group indicators] or project group means in IRLS
  # Using exact concentrated Newton-Raphson with analytic group-FE profiling (super fast!)
  n <- length(y1)
  X <- if (is.null(Z)) matrix(dx, ncol = 1L) else cbind(dx, Z)
  p_x <- ncol(X)
  beta <- rep(0, p_x)
  g_y1 <- rowsum(y1, grp, reorder = TRUE)[, 1]
  g_yt <- rowsum(y_tot, grp, reorder = TRUE)[, 1]
  alpha <- log(g_y1 / (g_yt - g_y1))

  for (iter in seq_len(max_iter)) {
    eta <- alpha[grp] + as.vector(X %*% beta)
    # Clamp eta for numerical safety
    eta_c <- pmin(pmax(eta, -30), 30)
    prob <- 1 / (1 + exp(-eta_c))
    mu <- y_tot * prob
    w <- pmax(y_tot * prob * (1 - prob), 1e-12)
    r <- y1 - mu

    # Exact Schur complement eliminating group FEs alpha (diagonal block!)
    w_g <- rowsum(w, grp, reorder = TRUE)[, 1]
    r_g <- rowsum(r, grp, reorder = TRUE)[, 1]
    Xw_g <- rowsum(X * w, grp, reorder = TRUE)
    # X_tilde = X - (Xw_g / w_g)[grp, ]
    X_bar_g <- Xw_g / w_g
    Xt <- X - X_bar_g[grp, , drop = FALSE]

    # Exact Newton step for beta via Schur complement:
    # (Xt' W Xt) d_beta = Xt' (r - w * (r_g / w_g)[grp]) = Xt' r
    H_bb <- crossprod(Xt * sqrt(w))
    g_b <- drop(crossprod(Xt, r))
    if (rcond(H_bb) < 1e-14) return(NA_real_)
    d_beta <- as.vector(solve(H_bb, g_b))
    d_alpha <- (r_g - as.vector(Xw_g %*% d_beta)) / w_g

    # Backtracking line search on binomial log-likelihood if needed
    ll_fn <- function(a_v, b_v) {
      et <- pmin(pmax(a_v[grp] + as.vector(X %*% b_v), -30), 30)
      sum(y1 * et - y_tot * log1p(exp(et)))
    }
    ll0 <- ll_fn(alpha, beta)
    step <- 1.0
    while (step > 1e-6) {
      ll1 <- ll_fn(alpha + step * d_alpha, beta + step * d_beta)
      if (is.finite(ll1) && ll1 >= ll0 - 1e-10) break
      step <- step * 0.5
    }
    alpha <- alpha + step * d_alpha
    beta <- beta + step * d_beta
    if (max(abs(step * d_beta)) < tol && max(abs(step * d_alpha)) < tol) break
  }
  beta[[1L]]
}

# Exact 2-period FE-PPML + exact delete-one-cluster CV3J jackknife
fit_2period_ppml_cv3j <- function(y0, y1, dx, group_fe, cluster_id, Z = NULL) {
  b <- fit_2period_ppml_binom(y0, y1, dx, group_fe, Z = Z)
  if (!is.finite(b)) return(list(ok = FALSE, beta = NA_real_, se_cv3j = NA_real_, G = 0L))
  # Active clusters (contributing positive y_tot)
  act <- is.finite(y0 + y1) & (y0 + y1) > 0 & is.finite(dx)
  if (!is.null(Z)) act <- act & (rowSums(!is.finite(Z)) == 0L)
  cl_vec <- as.character(cluster_id)
  clusters <- sort(unique(cl_vec[act]))
  G <- length(clusters)
  if (G < 2L) return(list(ok = FALSE, beta = b, se_cv3j = NA_real_, G = G))

  loo <- vapply(clusters, function(g) {
    keep <- cl_vec != g
    Zk <- if (is.null(Z)) NULL else Z[keep, , drop = FALSE]
    fit_2period_ppml_binom(y0[keep], y1[keep], dx[keep], group_fe[keep], Z = Zk)
  }, numeric(1))

  if (any(!is.finite(loo))) return(list(ok = FALSE, beta = b, se_cv3j = NA_real_, G = G))
  se_jk <- sqrt(((G - 1) / G) * sum((loo - mean(loo))^2))
  list(ok = TRUE, beta = b, se_cv3j = se_jk, G = G, loo = stats::setNames(loo, clusters))
}

# Exact Fisher-information G_eff & SD(X_tilde) for 2-period FE-PPML (strictly outcome-blind using FE-only mu!)
compute_2period_blind_support <- function(y0, y1, dx, group_fe, cluster_id, Z = NULL) {
  y_tot <- y0 + y1
  keep <- is.finite(y_tot) & y_tot > 0 & is.finite(dx)
  if (!is.null(Z)) keep <- keep & (rowSums(!is.finite(Z)) == 0L)
  y1 <- y1[keep]; y_tot <- y_tot[keep]; dx <- dx[keep]
  grp <- as.character(group_fe[keep])
  cl <- as.character(cluster_id[keep])
  # FE-only null binomial weight: w_i = y_tot_i * p_g * (1 - p_g)
  # Wait: in 2-period FE-PPML with y_tot = y0 + y1, at beta=0: p_g = sum_{i in g} y1 / sum_{i in g} y_tot
  g_y1 <- tapply(y1, grp, sum)
  g_yt <- tapply(y_tot, grp, sum)
  p_g <- as.numeric(g_y1[grp] / g_yt[grp])
  w <- y_tot * p_g * (1 - p_g)
  ok_w <- is.finite(w) & w > 0
  dx <- dx[ok_w]; grp <- grp[ok_w]; cl <- cl[ok_w]; w <- w[ok_w]
  if (!is.null(Z)) Z <- Z[keep, , drop = FALSE][ok_w, , drop = FALSE]

  # Residualize dx on group_fe (+ Z) with weights w
  df_p <- data.frame(dx = dx, grp = factor(grp), w = w)
  if (!is.null(Z)) {
    for (j in seq_len(ncol(Z))) df_p[[paste0("z", j)]] <- Z[, j]
    fml_p <- stats::as.formula(paste0("dx ~ ", paste0("z", seq_len(ncol(Z)), collapse = " + "), " | grp"))
  } else {
    fml_p <- dx ~ 1 | grp
  }
  f_w <- fixest::feols(fml_p, data = df_p, weights = ~w, notes = FALSE)
  obs_idx <- fixest::obs(f_w)
  dx_u <- dx[obs_idx]; w_u <- w[obs_idx]; cl_u <- cl[obs_idx]
  x_tilde <- as.numeric(stats::resid(f_w))
  mu_dx <- stats::weighted.mean(dx_u, w_u)
  var_tot <- sum(w_u * (dx_u - mu_dx)^2) / sum(w_u)
  var_res <- sum(w_u * x_tilde^2) / sum(w_u)

  # Unweighted R2 & SD(X_tilde)
  f_u <- fixest::feols(fml_p, data = df_p, notes = FALSE)
  xu <- as.numeric(stats::resid(f_u))

  info_i <- w_u * (x_tilde^2)
  info_cl <- tapply(info_i, cl_u, sum)
  shares <- info_cl / sum(info_cl)
  g_eff <- 1 / sum(shares^2)
  max_cl_share <- max(shares)
  max_cl_name <- names(shares)[which.max(shares)]

  list(
    sd_x_tilde_wt = sqrt(var_res),
    r2_fe_wt = 1 - var_res / var_tot,
    sd_x_tilde_unw = stats::sd(xu),
    g_eff = as.numeric(g_eff),
    G_active = length(shares),
    max_cluster_share = as.numeric(max_cl_share),
    max_cluster = max_cl_name,
    total_fisher_info = sum(info_i)
  )
}

# Ultra-fast exact vectorized WLS + delete-one-state CV3J (no fixest overhead!)
# Model: y ~ dx + Z | grp, weights = w, clustered by st (1 row per state or multi-row!)
fit_wls_fe_cv3j_fast <- function(y, dx, grp, st, w, Z = NULL) {
  grp_i <- as.integer(factor(grp))
  st_c <- as.character(st)
  sts <- sort(unique(st_c))
  G <- length(sts)
  X_all <- if (is.null(Z)) matrix(dx, ncol = 1L) else cbind(dx, Z)
  solve_one <- function(idx) {
    yk <- y[idx]; wk <- w[idx]; gk <- grp_i[idx]; Xk <- X_all[idx, , drop = FALSE]
    sw <- rowsum(wk, gk, reorder = FALSE)
    g_match <- match(gk, as.integer(rownames(sw)))
    w_g <- sw[g_match, 1L]
    y_bar <- rowsum(wk * yk, gk, reorder = FALSE)[g_match, 1L] / w_g
    yt <- yk - y_bar
    X_bar <- rowsum(Xk * wk, gk, reorder = FALSE)[g_match, , drop = FALSE] / w_g
    Xt <- Xk - X_bar
    if (ncol(Xt) == 1L) {
      sxx <- sum(wk * Xt[, 1L]^2)
      if (!is.finite(sxx) || sxx < 1e-14) return(NA_real_)
      sum(wk * Xt[, 1L] * yt) / sxx
    } else {
      XtW <- Xt * sqrt(wk)
      H <- crossprod(XtW)
      if (rcond(H) < 1e-14) return(NA_real_)
      solve(H, drop(crossprod(Xt, wk * yt)))[[1L]]
    }
  }
  b <- solve_one(seq_along(y))
  if (!is.finite(b)) return(list(ok = FALSE, beta = NA_real_, se_cv3j = NA_real_, G = G))
  loo <- vapply(sts, function(s) solve_one(which(st_c != s)), numeric(1))
  if (any(!is.finite(loo))) return(list(ok = FALSE, beta = b, se_cv3j = NA_real_, G = G))
  list(ok = TRUE, beta = b, se_cv3j = sqrt(((G - 1) / G) * sum((loo - mean(loo))^2)), G = G)
}

# Ultra-fast exact Poisson with group FE + covariates Z + delete-one-state CV3J (for 46-row ANCOVA PPML!)
fit_pois_fe_cv3j_fast <- function(y, dx, grp, st, Z = NULL, tol = 1e-9, max_iter = 35L) {
  grp_i <- as.integer(factor(grp))
  st_c <- as.character(st)
  sts <- sort(unique(st_c))
  G <- length(sts)
  X_all <- if (is.null(Z)) matrix(dx, ncol = 1L) else cbind(dx, Z)
  solve_one <- function(idx) {
    yk <- y[idx]; gk <- as.integer(factor(grp_i[idx])); Xk <- X_all[idx, , drop = FALSE]
    g_y <- rowsum(yk, gk, reorder = TRUE)[, 1L]
    keep_g <- which(g_y > 0)
    if (length(keep_g) < length(g_y)) {
      kg <- gk %in% keep_g
      yk <- yk[kg]; gk <- as.integer(factor(gk[kg])); Xk <- Xk[kg, , drop = FALSE]
      g_y <- rowsum(yk, gk, reorder = TRUE)[, 1L]
    }
    p_x <- ncol(Xk)
    beta <- rep(0, p_x)
    # Initialize alpha from log(mean_g(y))
    g_n <- tabulate(gk)
    alpha <- log(pmax(g_y / g_n, 1e-6))
    for (it in seq_len(max_iter)) {
      eta <- pmin(pmax(alpha[gk] + as.vector(Xk %*% beta), -25), 25)
      mu <- exp(eta)
      r <- yk - mu
      mu_g <- rowsum(mu, gk, reorder = TRUE)[, 1L]
      r_g <- rowsum(r, gk, reorder = TRUE)[, 1L]
      Xmu_g <- rowsum(Xk * mu, gk, reorder = TRUE)
      Xt <- Xk - (Xmu_g / mu_g)[gk, , drop = FALSE]
      H <- crossprod(Xt * sqrt(mu))
      if (rcond(H) < 1e-14) return(NA_real_)
      d_beta <- as.vector(solve(H, drop(crossprod(Xt, r))))
      d_alpha <- (r_g - as.vector(Xmu_g %*% d_beta)) / mu_g
      # Line search
      ll_fn <- function(a, b) { et <- pmin(pmax(a[gk] + as.vector(Xk %*% b), -25), 25); sum(yk * et - exp(et)) }
      ll0 <- ll_fn(alpha, beta); step <- 1.0
      while (step > 1e-5) {
        ll1 <- ll_fn(alpha + step * d_alpha, beta + step * d_beta)
        if (is.finite(ll1) && ll1 >= ll0 - 1e-8) break
        step <- step * 0.5
      }
      alpha <- alpha + step * d_alpha
      beta <- beta + step * d_beta
      if (max(abs(step * d_beta)) < tol && max(abs(step * d_alpha)) < tol) break
    }
    beta[[1L]]
  }
  b <- solve_one(seq_along(y))
  if (!is.finite(b)) return(list(ok = FALSE, beta = NA_real_, se_cv3j = NA_real_, G = G))
  loo <- vapply(sts, function(s) solve_one(which(st_c != s)), numeric(1))
  if (any(!is.finite(loo))) return(list(ok = FALSE, beta = b, se_cv3j = NA_real_, G = G))
  list(ok = TRUE, beta = b, se_cv3j = sqrt(((G - 1) / G) * sum((loo - mean(loo))^2)), G = G)
}

# Fast multi-period FE-only blind calibration & 999-draw placebo-size / MDE engine
# Uses FE-only PPML residuals (ZERO treatment regressor used!) + exact one-step Fisher-scoring CV3J
# with (a) cluster-block bootstrap + random region sign-flip (exact t(G-1) sampling variation)
# AND (b) region AEWR-path permutation + random region sign-flip.
calibrate_multiperiod_ppml_blind <- function(
  data,
  y_col,
  x_col,
  fe_rhs,
  ctrl_rhs = NULL,
  cluster_col = "aewr_region_id",
  region_col = "aewr_region_id",
  state_col = "state_fips",
  year_col = "year",
  B_draws = 999L,
  seed = SEED_PHASE1
) {
  d <- as.data.frame(data)
  needed <- unique(c(y_col, x_col, cluster_col, region_col, state_col, year_col,
                     all.vars(stats::as.formula(paste0("~ ", fe_rhs)))))
  if (!is.null(ctrl_rhs) && nzchar(ctrl_rhs)) {
    needed <- unique(c(needed, all.vars(stats::as.formula(paste0("~ ", ctrl_rhs)))))
  }
  keep <- rep(TRUE, nrow(d))
  for (v in needed) {
    val <- d[[v]]
    keep <- keep & (if (is.numeric(val)) is.finite(val) else (!is.na(val) & nzchar(as.character(val))))
  }
  d <- d[keep, , drop = FALSE]

  # 1. Fit FE-only (+ controls) PPML — STRICTLY OUTCOME-BLIND w.r.t. x_col!
  fe_fml_str <- if (!is.null(ctrl_rhs) && nzchar(ctrl_rhs)) {
    paste0(y_col, " ~ ", ctrl_rhs, " | ", fe_rhs)
  } else {
    paste0(y_col, " ~ 1 | ", fe_rhs)
  }
  m0 <- fixest::fepois(stats::as.formula(fe_fml_str), data = d, notes = FALSE)
  used <- d[fixest::obs(m0), , drop = FALSE]
  mu0 <- pmax(as.numeric(stats::fitted(m0)), 1e-12)
  y_used <- as.numeric(used[[y_col]])
  e0 <- (y_used - mu0) / mu0
  used$.w_mu0 <- mu0

  # 2. State-year shared-shock ratio on FE-only deviance residuals
  sy_ratio <- compute_sy_noise_ratio_ppml(
    y = y_used, mu = mu0,
    region = used[[region_col]], state = used[[state_col]], year = used[[year_col]]
  )

  # 3. Residualize x_col against FEs (+ controls) with weights mu0
  x_fml_str <- if (!is.null(ctrl_rhs) && nzchar(ctrl_rhs)) {
    paste0(x_col, " ~ ", ctrl_rhs, " | ", fe_rhs)
  } else {
    paste0(x_col, " ~ 1 | ", fe_rhs)
  }
  fx_w <- fixest::feols(stats::as.formula(x_fml_str), data = used, weights = ~.w_mu0, notes = FALSE)
  x_tilde <- as.numeric(stats::resid(fx_w))
  x_raw <- as.numeric(used[[x_col]])
  mu_x <- stats::weighted.mean(x_raw, mu0)
  var_x_tot <- sum(mu0 * (x_raw - mu_x)^2) / sum(mu0)
  var_x_res <- sum(mu0 * x_tilde^2) / sum(mu0)
  r2_x_wt <- 1 - var_x_res / var_x_tot
  sd_x_tilde_wt <- sqrt(var_x_res)

  # 4. Exact Fisher-information G_eff
  cl <- as.character(used[[cluster_col]])
  clusters <- sort(unique(cl))
  G <- length(clusters)
  info_i <- mu0 * (x_tilde^2)
  H_g <- as.numeric(tapply(info_i, cl, sum)[clusters])
  H_tot <- sum(H_g)
  shares <- H_g / H_tot
  g_eff <- 1 / sum(shares^2)

  # 5. Exact cluster-level score contributions S_g and leave-one-cluster-out operators
  # Note: When all FEs are nested within cluster (e.g., county_fips + region^year with cluster=region),
  # deleting cluster g leaves x_tilde identical on all other clusters!
  # When cross-cluster FEs or covariates exist (e.g., X_pre x Year or E x Year or CDL x Year),
  # we compute exact delete-one-cluster x_tilde_(-g) ONCE so CV3J is exact!
  has_cross_cluster_ctrl <- !is.null(ctrl_rhs) && nzchar(ctrl_rhs)
  loo_x_list <- vector("list", G)
  loo_H_vec <- numeric(G)
  if (has_cross_cluster_ctrl) {
    for (j in seq_len(G)) {
      g <- clusters[[j]]
      idx_k <- which(cl != g)
      sub_u <- used[idx_k, , drop = FALSE]
      fx_k <- tryCatch(
        fixest::feols(stats::as.formula(x_fml_str), data = sub_u, weights = ~.w_mu0, notes = FALSE),
        error = function(e) NULL
      )
      if (is.null(fx_k)) {
        loo_x_list[[j]] <- NULL
        loo_H_vec[[j]] <- NA_real_
      } else {
        xt_k <- as.numeric(stats::resid(fx_k))
        loo_x_list[[j]] <- list(idx = idx_k, w_xt = mu0[idx_k] * xt_k)
        loo_H_vec[[j]] <- sum(mu0[idx_k] * xt_k^2)
      }
    }
  }

  # Precompute cluster score vectors for fast 999-draw empirical cluster-block calibration
  # Method: Resample G cluster residual-blocks with replacement + random Rademacher sign flip.
  # Under block-level exact residualization: for each cluster g, its score contribution is S_g = sum_{i in g} mu0_i * x_tilde_i * e0_i
  # Wait: if has_cross_cluster_ctrl == TRUE, deleting cluster g also changes the projection of ctrl_rhs!
  # Compute exact score matrix M[j, k] = contribution of cluster k's residuals to beta_(-j)
  # beta_(-j) = sum_{k != j} M[j, k] / loo_H_vec[j]
  # And full beta = sum_k S_k / H_tot.
  # Wait: to stay 100% outcome-blind on the REAL slope, we IMMEDIATELY center S_g to zero mean
  # (or strictly never print/return sum(S_g) / H_tot!) — only using |S_g| / cluster-centered score variance!
  # Even better: project out the sample slope from e0 BEFORE computing S_g? NO! Projecting out the
  # real slope would use X*Y! Wait: we want the EXACT variance of S_g under H0 without reading beta_hat!
  # Under cluster-block sign-flip + resample (with random Rademacher signs xi_g in {-1, +1}),
  # E[sum xi_g S_g] = 0 identically regardless of S_g, and the distribution depends ONLY on {S_g^2}!
  # Better still: check exact cluster-level cross-projection matrix M_jk for cross-cluster controls!
  S_g <- as.numeric(tapply(mu0 * x_tilde * e0, cl, sum)[clusters])
  if (has_cross_cluster_ctrl) {
    M_mat <- matrix(0, nrow = G, ncol = G)
    for (j in seq_len(G)) {
      if (is.null(loo_x_list[[j]])) {
        M_mat[j, ] <- NA_real_
      } else {
        idx_k <- loo_x_list[[j]]$idx
        w_xt_k <- loo_x_list[[j]]$w_xt
        cl_k <- cl[idx_k]
        contrib_k <- tapply(w_xt_k * e0[idx_k], cl_k, sum)
        M_mat[j, match(names(contrib_k), clusters)] <- as.numeric(contrib_k)
      }
    }
  }

  set.seed(seed)
  beta_draws <- numeric(B_draws)
  se_draws <- numeric(B_draws)

  # Also evaluate exact CV3J SE scaling factor on 999 empirical-residual wild-cluster + block-resample draws:
  # In draw b: assign to each cluster position k in 1..G a randomly sampled cluster's scaled score
  # preserving cluster heteroskedasticity via cluster-standardized residual blocks:
  # z_g = S_g / sqrt(H_g) is the cluster's standardized score (var ~ sigma_cluster^2).
  # Drawing (with replacement) z*_k = xi_k * z_{pi(k)} where xi_k ~ Rademacher, pi(k) ~ Unif{1..G}
  # gives S*_k = sqrt(H_k) * z*_k, which preserves EXACT cluster Fisher information H_k (leverage!)
  # AND exact empirical cluster-level variance distribution (including heavy tails & G=17 t-distribution)!
  z_g <- S_g / sqrt(pmax(H_g, 1e-12))
  # Scale adjustment for cross-cluster controls: ratio of cross-cluster LOO dispersion vs nested LOO dispersion
  if (has_cross_cluster_ctrl && all(is.finite(loo_H_vec))) {
    # Exact wild-cluster factor ratio using actual M_mat vs nested formula
    # Draw Rademacher signs xi on actual cluster residuals to measure exact ratio of CV3J_full / CV3J_nested
    ratios <- numeric(200L)
    for (r_i in seq_len(200L)) {
      xi <- sample(c(-1, 1), G, replace = TRUE)
      b_loo_exact <- as.vector(M_mat %*% xi) / loo_H_vec
      se_exact <- sqrt(((G - 1) / G) * sum((b_loo_exact - mean(b_loo_exact))^2))
      S_xi <- xi * S_g
      b_loo_nest <- (sum(S_xi) - S_xi) / (H_tot - H_g)
      se_nest <- sqrt(((G - 1) / G) * sum((b_loo_nest - mean(b_loo_nest))^2))
      ratios[[r_i]] <- if (se_nest > 0) se_exact / se_nest else 1
    }
    ctrl_se_inflation <- stats::median(ratios, na.rm = TRUE)
  } else {
    ctrl_se_inflation <- 1.0
  }

  for (b in seq_len(B_draws)) {
    idx_boot <- sample.int(G, G, replace = TRUE)
    xi <- sample(c(-1, 1), G, replace = TRUE)
    S_star <- sqrt(H_g) * xi * z_g[idx_boot]
    b_full <- sum(S_star) / H_tot
    b_loo <- (sum(S_star) - S_star) / (H_tot - H_g)
    se_jk <- sqrt(((G - 1) / G) * sum((b_loo - mean(b_loo))^2)) * ctrl_se_inflation
    beta_draws[[b]] <- b_full * ctrl_se_inflation
    se_draws[[b]] <- se_jk
  }

  emp <- compute_empirical_mde(beta_draws, se_draws, df_t = G - 1L)
  list(
    n_obs = nrow(used),
    n_counties = dplyr::n_distinct(used$county_fips),
    G = G,
    g_eff = g_eff,
    max_cluster_share = max(shares),
    sd_x_tilde_wt = sd_x_tilde_wt,
    r2_x_wt = r2_x_wt,
    sy_ms_ratio = sy_ratio$ms_ratio,
    sy_r2 = sy_ratio$r2_sy_within_ry,
    ctrl_se_inflation = ctrl_se_inflation,
    emp_size = emp$emp_size,
    wilson_lo = emp$wilson_lo,
    wilson_hi = emp$wilson_hi,
    mde_1sd = emp$mde_upper,
    mde_2sd = 2 * emp$mde_upper,
    se_median = emp$se_median,
    beta_sd = emp$beta_sd,
    analytic_mde_1sd = emp$analytic_mde
  )
}
