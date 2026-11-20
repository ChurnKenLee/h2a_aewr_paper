# Identification search, Phase 0, step 3: shared design-stage power harness.
#
# Generalizes explorations/identification/04_track1_power.R to any
# unit x period dose matrix, fixed-effect list and cluster structure. Pure
# functions: no reads or writes. Source it from a strategy script:
#   source("explorations/identification_search/shared/power_harness.R")
#
# Error processes, seed, repetitions, AR(1) coefficient, spatial graph
# coefficient, variance fractions, alpha, target power and the size gate come
# from the border branch (code/designs/border_discontinuity/design.json,
# redesign.power and redesign.precision_assumptions): seed 201103, 999
# repetitions, 999 calibration repetitions, AR(1) 0.5, size gate = upper 95%
# Wilson bound of the null rejection rate <= 0.075, alpha 0.05, power 0.80.
#
# Main entry point: ish_power(). See explorations/identification_search/README.md
# for the API and how to apply it to PPML and hours outcomes.
suppressPackageStartupMessages({library(dplyr); library(fixest)})
if (!exists("border_power_ar_draws")) source(file.path("code", "designs", "border_discontinuity", "power.R"))

ish_power_settings <- function(path = file.path("code", "designs", "border_discontinuity", "design.json")) {
  v6 <- jsonlite::read_json(path, simplifyVector = TRUE)
  s <- v6$redesign$power; a <- v6$redesign$precision_assumptions
  stopifnot(s$seed == 201103L, s$repetitions == 999L, s$ar1 == 0.5, a$size_maximum == 0.075, s$nthreads == 1L)
  list(seed = as.integer(s$seed), repetitions = as.integer(s$repetitions),
    calibration_repetitions = as.integer(s$calibration_repetitions), chunk_size = as.integer(s$chunk_size),
    ar1 = s$ar1, spatial_graph_coefficient = s$spatial_graph_coefficient, dgp_ids = unlist(s$dgp_ids),
    variance_fractions = s$variance_fractions, alpha = a$alpha, target_power = a$target_power,
    size_maximum = a$size_maximum, ri_seed = 20261102L, ri_permutations = 199L)
}

# Row-normalized spatial loading for shock nodes from an edge list (a, b),
# as in 04_track1_power.R state_graph(): (I - c * D^-1/2 A D^-1/2)^-1.
ish_spatial_loading <- function(nodes, edges, coefficient) {
  nodes <- sort(unique(nodes))
  A <- matrix(0, length(nodes), length(nodes), dimnames = list(nodes, nodes))
  e <- edges[edges$a %in% nodes & edges$b %in% nodes & edges$a != edges$b, , drop = FALSE]
  A[cbind(e$a, e$b)] <- 1; A <- pmax(A, t(A))
  deg <- rowSums(A); sc <- ifelse(deg > 0, 1 / sqrt(pmax(deg, 1)), 0)
  L <- solve(diag(length(nodes)) - coefficient * A * outer(sc, sc))
  L / sqrt(rowSums(L^2))
}

# Internal: prepared frame. Rows sorted by unit, period; FE are factors.
ish_prep <- function(frame, unit, period, fe, node, group, weights, x = NULL, clusters = NULL, jk_cluster = NULL) {
  frame <- frame[order(frame[[unit]], frame[[period]]), , drop = FALSE]
  fel <- lapply(fe, function(f) factor(frame[[f]]))
  w <- if (is.null(weights)) rep(1, nrow(frame)) else frame[[weights]] / mean(frame[[weights]])
  stopifnot(all(is.finite(w)), all(w > 0))
  out <- list(data = frame, fe = fel, w = w, sw = sqrt(w), unit = unit, period = period,
    periods = sort(unique(frame[[period]])), units = sort(unique(frame[[unit]])),
    nodes = sort(unique(frame[[node]])), groups = if (!is.null(group)) sort(unique(frame[[group]])) else NULL,
    node = node, group = group)
  stopifnot(all(diff(out$periods) == 1))
  if (!is.null(x)) {
    xw <- drop(fixest::demean(cbind(x = frame[[x]]), f = fel, weights = w, tol = 1e-10, nthreads = 1L)) * out$sw
    out$xw <- xw; out$sxx <- sum(xw^2)
    if (out$sxx <= 1e-12) stop("Dose has no variation after fixed effects")
    out$cl_index <- lapply(setNames(clusters, clusters), function(cl) match(frame[[cl]], sort(unique(frame[[cl]]))))
    out$cl_G <- vapply(out$cl_index, function(i) length(unique(i)), integer(1))
    jk <- sort(unique(frame[[jk_cluster]])); out$jk <- jk; out$jk_cluster <- jk_cluster
    # Exact leave-one-cluster-out operators (04_track1_power.R): on retained
    # rows the re-demeaned dose is orthogonal to the FE, so beta_(-g) = L_g' e
    # for raw errors e (weighted: L_g = w * x~ / sum(w * x~^2)).
    out$loo <- t(vapply(jk, function(g) {
      keep <- frame[[jk_cluster]] != g
      fk <- lapply(fel, function(f) droplevels(f[keep]))
      xk <- drop(fixest::demean(cbind(x = frame[[x]][keep]), f = fk, weights = w[keep], tol = 1e-10, nthreads = 1L))
      op <- numeric(nrow(frame)); s <- sum(w[keep] * xk^2)
      if (s > 1e-12) op[keep] <- w[keep] * xk / s else op[] <- NA_real_
      op
    }, numeric(nrow(frame))))
  }
  out
}

# Internal: raw (untransformed) error draws, n x R. Same call order as 04:
# unit AR(1), then node AR(1) (spatially loaded if requested), then group AR(1).
ish_errors <- function(pp, R, process, loading, ar1) {
  d <- pp$data; ny <- length(pp$periods); ti <- match(d[[pp$period]], pp$periods) - 1L
  raw <- matrix(0, nrow(d), R)
  if (process$county > 0) {
    ci <- match(d[[pp$unit]], pp$units); e <- border_power_ar_draws(length(pp$units), ny, R, ar1)
    raw <- raw + sqrt(process$county) * e[ti * length(pp$units) + ci, , drop = FALSE]
  }
  if (process$region > 0) {
    si <- match(d[[pp$node]], pp$nodes); e <- border_power_ar_draws(length(pp$nodes), ny, R, ar1)
    if (isTRUE(process$spatial)) {
      if (is.null(loading)) stop("Spatial DGP requires a loading matrix")
      for (t in seq_len(ny)) { rr <- (t - 1L) * length(pp$nodes) + seq_along(pp$nodes)
        e[rr, ] <- loading[pp$nodes, pp$nodes] %*% e[rr, , drop = FALSE] }
    }
    raw <- raw + sqrt(process$region) * e[ti * length(pp$nodes) + si, , drop = FALSE]
  }
  if (process$pair > 0) {
    ri <- match(d[[pp$group]], pp$groups); e <- border_power_ar_draws(length(pp$groups), ny, R, ar1)
    raw <- raw + sqrt(process$pair) * e[ti * length(pp$groups) + ri, , drop = FALSE]
  }
  raw
}
ish_within <- function(E, pp) fixest::demean(E, f = pp$fe, weights = pp$w, tol = 1e-10, nthreads = 1L) * pp$sw

ish_draw_stats <- function(pp, E, ri = NULL) {
  yw <- ish_within(E, pp); b <- drop(crossprod(pp$xw, yw)) / pp$sxx
  u <- yw - outer(pp$xw, b); sc <- pp$xw * u
  n <- nrow(yw); corr <- (n - 1) / (n - 2)
  v <- lapply(names(pp$cl_index), function(cl) { G <- pp$cl_G[[cl]]
    corr * G / (G - 1) * colSums(rowsum(sc, pp$cl_index[[cl]])^2) / pp$sxx^2 })
  names(v) <- paste0(names(pp$cl_index), "_CRV1")
  bl <- pp$loo %*% (E); bl <- sweep(bl, 2, colMeans(bl)); Gj <- length(pp$jk)
  v[[paste0(pp$jk_cluster, "_CV3J")]] <- (Gj - 1) / Gj * colSums(bl^2)
  df <- c(setNames(pp$cl_G[names(pp$cl_index)] - 1, paste0(names(pp$cl_index), "_CRV1")), setNames(Gj - 1, paste0(pp$jk_cluster, "_CV3J")))
  out <- list(b = b, var = do.call(rbind, v), df = df)
  if (!is.null(ri)) out$bperm <- crossprod(ri$xw_perm, yw) / ri$sxx_perm  # P x R
  out
}

# Randomization inference: permute assignment-unit dose paths within strata.
# Requires the dose to be constant within assignment unit x period.
ish_ri_prepare <- function(pp, x, assign, strata, P, seed) {
  d <- pp$data
  chk <- tapply(d[[x]], paste(d[[assign]], d[[pp$period]]), function(z) diff(range(z)))
  if (any(chk > 1e-10)) return(NULL)
  au <- distinct(data.frame(a = d[[assign]], s = if (is.null(strata)) "all" else d[[strata]]))
  if (anyDuplicated(au$a)) stop("Assignment units must nest in RI strata")
  path <- tapply(d[[x]], list(d[[assign]], d[[pp$period]]), mean)
  old <- if (exists(".Random.seed", envir = globalenv())) get(".Random.seed", envir = globalenv()) else NULL
  set.seed(seed)
  perms <- replicate(P, { m <- setNames(au$a, au$a)
    for (s in unique(au$s)) { a <- au$a[au$s == s]; m[a] <- if (length(a) > 1) sample(a) else a }; m })
  if (!is.null(old)) assign(".Random.seed", old, envir = globalenv())
  ti <- as.character(d[[pp$period]])
  Xp <- vapply(seq_len(P), function(p) path[cbind(perms[d[[assign]], p], ti)], numeric(nrow(d)))
  xw_perm <- fixest::demean(Xp, f = pp$fe, weights = pp$w, tol = 1e-10, nthreads = 1L) * pp$sw
  list(xw_perm = xw_perm, sxx_perm = colSums(xw_perm^2), C = drop(crossprod(xw_perm, pp$xw)) / colSums(xw_perm^2), P = P)
}

ish_ri_power <- function(effect, b, bperm, C, sigma, alpha) {
  obs <- abs(effect + sigma * b)
  perm <- abs(effect * C + sigma * bperm)
  p <- (1 + colSums(sweep(perm, 2, obs, `>=`))) / (nrow(bperm) + 1)
  mean(p <= alpha)
}

# ---- Main entry point ---------------------------------------------------------
# policy:   data.frame, one row per unit x period of the design (balanced,
#           consecutive integer periods), with the dose column `x`, the FE
#           columns, cluster columns and (optionally) weights.
# pre:      data.frame for scale calibration with the outcome column `y`, the
#           same unit/period/FE/node/group columns (and weights). Only FE-only
#           residuals of `y` are used (no dose regressor).
# fe:       character vector of FE columns (build interactions beforehand).
# clusters: CRV1 cluster columns; jk_cluster: the leave-one-out jackknife
#           cluster (default clusters[1]); node: shared-shock node (the border
#           "region" shock; default jk_cluster); group: optional second shock
#           node (the border "pair" shock); if NULL its variance share is
#           added to the node shock.
# beta_e1:  coefficient corresponding to an elasticity of 1 at the design's
#           means (mean_y / mean_price for levels; 1 / mean_price for the
#           PPML/semi-elasticity linearization).
# ri:       NULL or list(assign = col, strata = col or NULL, P = 199).
# Returns list(support, calibration, mde, null, summary).
ish_power <- function(policy, pre, y, x, unit, period, fe, clusters, jk_cluster = clusters[1], node = jk_cluster,
  group = NULL, weights = NULL, beta_e1, loading = NULL, dgps = NULL, settings = ish_power_settings(),
  egrid = round(seq(0.01, 10, by = 0.01), 2), ri = NULL, label = list()) {
  dgps <- if (is.null(dgps)) settings$dgp_ids else dgps
  pre_pp <- ish_prep(pre, unit, period, fe, node, group, weights)
  pol <- ish_prep(policy, unit, period, fe, node, group, weights, x = x, clusters = clusters, jk_cluster = jk_cluster)
  ri_obj <- if (!is.null(ri)) ish_ri_prepare(pol, x, ri$assign, ri$strata, if (is.null(ri$P)) settings$ri_permutations else ri$P, settings$ri_seed) else NULL
  if (!is.null(ri) && is.null(ri_obj)) message("RI skipped: dose varies within assignment unit x period")
  ylev <- function(pp, col) { v <- pp$data[[col]]; if (is.null(v)) NA_real_ else sum(pp$w * v) / sum(pp$w) }
  lev <- rowsum(pol$xw^2, match(pol$data[[jk_cluster]], pol$jk))[, 1]
  support <- tibble(!!!label, units = length(pol$units), periods = length(pol$periods), rows = nrow(pol$data),
    jk_clusters = length(pol$jk), clusters_with_dose = sum(lev > 1e-12 * pol$sxx),
    dose_sd_after_fe = sqrt(pol$sxx / sum(pol$w)), partial_dose_ss = pol$sxx,
    effective_clusters = pol$sxx^2 / sum(lev^2), max_leverage_share = max(lev) / pol$sxx,
    max_leverage_cluster = pol$jk[as.integer(names(lev))[which.max(lev)]], beta_elasticity_1 = beta_e1,
    mean_y_pre = ylev(pre_pp, y), sd_y_pre = sd(pre_pp$data[[y]]))
  grid <- egrid * beta_e1
  effects <- sort(unique(c(0, grid, -grid)))
  cal_rows <- mde_rows <- null_rows <- list()
  for (dgp in dgps) {
    fr <- settings$variance_fractions[[dgp]]
    process <- list(region = fr$region + if (is.null(group)) fr$pair else 0, pair = if (is.null(group)) 0 else fr$pair,
      county = fr$county, spatial = dgp == "spatial_region_serial")
    if (process$spatial && is.null(loading)) next
    set.seed(settings$seed)
    obs <- sum(ish_within(cbind(pre_pp$data[[y]]), pre_pp)^2)
    chunks <- split(seq_len(settings$calibration_repetitions), ceiling(seq_len(settings$calibration_repetitions) / settings$chunk_size))
    rss <- unlist(lapply(chunks, function(ix) colSums(ish_within(ish_errors(pre_pp, length(ix), process, loading, settings$ar1), pre_pp)^2)))
    sigma <- sqrt(obs / mean(rss))
    pchunks <- split(seq_len(settings$repetitions), ceiling(seq_len(settings$repetitions) / settings$chunk_size))
    draws <- lapply(pchunks, function(ix) ish_draw_stats(pol, ish_errors(pol, length(ix), process, loading, settings$ar1), ri_obj))
    errs <- unlist(lapply(draws, `[[`, "b")); vars <- do.call(cbind, lapply(draws, `[[`, "var")); dfs <- draws[[1]]$df
    cal_rows[[dgp]] <- tibble(!!!label, dgp_id = dgp, sigma = sigma, observed_rss = obs, expected_unit_rss = mean(rss),
      sim_coef_error_sd = sd(errs) * sigma)
    methods <- rownames(vars)
    res <- lapply(methods, function(meth) {
      curve <- border_power_grid(errs, vars[meth, ], dfs[[meth]], effects, sigma, settings$alpha)
      list(meth = meth, curve = curve)
    })
    if (!is.null(ri_obj)) {
      bperm <- do.call(cbind, lapply(draws, `[[`, "bperm"))
      pw <- function(e) ish_ri_power(e, errs, bperm, ri_obj$C, sigma, settings$alpha)
      p0 <- pw(0); k0 <- round(p0 * length(errs)); ci <- border_power_wilson(k0, length(errs))
      ri_curve <- data.frame(effect = 0, repetitions = length(errs), valid_repetitions = length(errs), invalid_repetitions = 0L,
        rejections = k0, rejection_rate = p0, mc_confidence_high = ci[1, "high"])
      res[[length(res) + 1]] <- list(meth = paste0(ri$assign, "_RI"), curve = ri_curve, pw = pw)
    }
    for (r in res) {
      null <- r$curve[r$curve$effect == 0, ]
      gate <- null$invalid_repetitions == 0L && null$mc_confidence_high <= settings$size_maximum
      null_rows[[length(null_rows) + 1]] <- tibble(!!!label, dgp_id = dgp, inference_method = r$meth,
        null_rejection_rate = null$rejection_rate, null_mc_high = null$mc_confidence_high, size_gate_pass = gate)
      for (sg in c(-1, 1)) {
        if (is.null(r$pw)) {
          dir <- r$curve[match(sg * grid, r$curve$effect), ]
          pos <- which(dir$rejection_rate >= settings$target_power)
          k <- if (length(pos)) pos[1] else NA_integer_
        } else {  # RI: bisection on the grid (power assumed monotone in |effect|)
          lo <- 1L; hi <- length(grid)
          if (r$pw(sg * grid[hi]) < settings$target_power) k <- NA_integer_ else {
            while (lo < hi) { mid <- (lo + hi) %/% 2L; if (r$pw(sg * grid[mid]) >= settings$target_power) hi <- mid else lo <- mid + 1L }
            k <- lo }
        }
        mde_rows[[length(mde_rows) + 1]] <- tibble(!!!label, dgp_id = dgp, inference_method = r$meth, effect_sign = sg,
          grid_mde_beta = if (is.na(k)) NA_real_ else grid[k], mde_elasticity = if (is.na(k)) NA_real_ else egrid[k],
          size_gate_pass = gate, size_valid_mde_elasticity = if (gate && !is.na(k)) egrid[k] else NA_real_,
          beyond_grid = is.na(k))
      }
    }
  }
  mde <- bind_rows(mde_rows); nulls <- bind_rows(null_rows)
  # Size-valid MDE per DGP: the first gate-passing method in the order
  # primary-cluster CRV1, jackknife CV3J, RI, other CRV1; upper value across signs.
  order_m <- c(paste0(clusters[1], "_CRV1"), paste0(jk_cluster, "_CV3J"), if (!is.null(ri)) paste0(ri$assign, "_RI"), paste0(clusters[-1], "_CRV1"))
  summ <- mde %>% group_by(dgp_id, inference_method) %>%
    summarise(size_gate_pass = all(size_gate_pass), mde_elasticity_upper = max(mde_elasticity),
      beyond_grid = any(beyond_grid), .groups = "drop") %>%
    mutate(rank = match(inference_method, order_m)) %>% filter(size_gate_pass) %>% arrange(dgp_id, rank) %>%
    group_by(dgp_id) %>% slice(1) %>% ungroup() %>% select(dgp_id, preferred_method = inference_method,
      size_valid_mde_elasticity = mde_elasticity_upper, beyond_grid)
  summ <- tibble(dgp_id = unique(mde$dgp_id)) %>% left_join(summ, by = "dgp_id") %>% mutate(!!!label)
  list(support = support, calibration = bind_rows(cal_rows), mde = mde, null = nulls, summary = summ)
}

# Share of FE-only residual variance at node x period cells, and the share
# expected under pure unit noise (04b_track1_preperiod.R logic): evidence for
# choosing among the declared error processes. Report, do not select on it alone.
ish_node_period_share <- function(pre, y, unit, period, fe, node, weights = NULL) {
  pp <- ish_prep(pre, unit, period, fe, node, NULL, weights)
  r <- drop(ish_within(cbind(pp$data[[y]]), pp))
  cell <- paste(pp$data[[node]], pp$data[[period]])
  cm <- ave(r, cell, FUN = mean)
  observed <- sum(cm^2) / sum(r^2)
  set.seed(201103L)
  E <- matrix(rnorm(nrow(pp$data) * 199), ncol = 199); Ew <- ish_within(E, pp)
  sim <- mean(apply(Ew, 2, function(z) { m <- ave(z, cell, FUN = mean); sum(m^2) / sum(z^2) }))
  tibble(observed_node_period_share = observed, pure_unit_noise_share = sim)
}

# PPML / size-weighted linearization. Near the null, the PPML score with unit
# FE is sum (y - mu) x~ with information sum mu x~^2, i.e. WLS of
# y / mu_i - 1 on the dose with weights mu_i. Use mu_i = the unit's mean count
# in the calibration years (units with mu_i = 0 contribute nothing and are
# dropped). The coefficient is then a semi-elasticity; beta_e1 = 1 / mean_price
# where mean_price is the mu-weighted mean price.
ish_ppml_linearize <- function(pre, policy, y, unit) {
  mu <- tapply(pre[[y]], pre[[unit]], mean)
  mu <- mu[is.finite(mu) & mu > 0]
  add <- function(d) { d <- d[d[[unit]] %in% names(mu), , drop = FALSE]
    d$ppml_weight <- as.numeric(mu[d[[unit]]]); d }
  pre <- add(pre); policy <- add(policy)
  pre$ppml_y <- pre[[y]] / pre$ppml_weight - 1
  list(pre = pre, policy = policy, y = "ppml_y", weights = "ppml_weight")
}
